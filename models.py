import re
import sys
from pathlib import Path
from typing import Literal

import huggingface_hub as hf
import numpy as np
import torch
import torchaudio
import torchaudio.functional as tf
import transformers
from llama_cpp import Llama
from omegaconf import OmegaConf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

transformers.logging.set_verbosity_error()

sys.path.append(str(Path(__file__).parent / "spark_tts"))
from sparktts.models.bicodec import BiCodec


class Bicodec:
    def __init__(
        self,
        bicodec: Path | str = "annuvin/bicodec",
        wav2vec2: Path | str = "annuvin/wav2vec2-st",
        device: str = "cuda",
        dtype: str = "float16",
        flash_attn: bool = True,
    ) -> None:
        if not (bicodec := Path(bicodec)).is_dir():
            bicodec = Path(hf.snapshot_download(bicodec.as_posix()))

        if not (wav2vec2 := Path(wav2vec2)).is_dir():
            wav2vec2 = Path(hf.snapshot_download(wav2vec2.as_posix()))

        self.device = device
        self.dtype = getattr(torch, dtype)
        self.model = BiCodec.load_from_checkpoint(bicodec).to(device, self.dtype)

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec2)
        self.extractor = Wav2Vec2Model.from_pretrained(
            pretrained_model_name_or_path=wav2vec2,
            attn_implementation=(
                "flash_attention_2"
                if flash_attn and dtype in ["bfloat16", "float16"]
                else "sdpa"
            ),
            torch_dtype=self.dtype,
        )
        self.extractor.config.output_hidden_states = True
        self.extractor.to(device)

        self.config = OmegaConf.load(Path(bicodec) / "config.yaml")
        self.hop_len = self.config.audio_tokenizer.mel_params.hop_length
        self.sample_rate = self.config.audio_tokenizer.mel_params.sample_rate
        self.pattern = re.compile(r"<\|bicodec_semantic_(\d+)\|>")

    def load(self, path: Path | str, max_len: int | None = None) -> torch.Tensor:
        audio, sample_rate = torchaudio.load(path)
        audio = audio.to(self.device, self.dtype)

        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        if sample_rate != self.sample_rate:
            audio = tf.resample(audio, sample_rate, self.sample_rate)

        return audio[:, :max_len]

    def process(self, audio: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
        if max_len and max_len > audio.shape[1]:
            audio = torch.tile(audio, (1, max_len // audio.shape[1] + 1))

        return audio[:, :max_len]

    def extract(self, audio: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(
            raw_speech=audio.squeeze(),
            output_hidden_states=True,
            padding=True,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
        ).input_values.to(self.device, self.dtype)

        features = self.extractor(inputs).hidden_states
        return (features[11] + features[14] + features[16]) / 3

    def encode(
        self, path: str, wav_len: int = 30, ref_len: int = 6
    ) -> tuple[torch.Tensor, str]:
        wav_len = self.sample_rate * wav_len // self.hop_len * self.hop_len
        ref_len = self.sample_rate * ref_len // self.hop_len * self.hop_len

        wav = self.load(path, wav_len)
        ref_wav = self.process(wav, ref_len)
        feat = self.extract(wav)

        _, tokens = self.model.tokenize({"wav": wav, "ref_wav": ref_wav, "feat": feat})
        codes = "".join([f"<|bicodec_global_{t}|>" for t in tokens.squeeze()])
        return tokens, codes

    def decode(self, tokens: torch.Tensor, codes: str) -> np.ndarray:
        ids = [int(c) for c in re.findall(self.pattern, codes)]
        data = torch.tensor([ids], dtype=torch.long, device=self.device)
        audio = self.model.detokenize(data, tokens)
        return audio.squeeze().float().cpu().numpy()


class Spark:
    def __init__(
        self,
        path: Path | str = "annuvin/spark-gguf",
        file: str = "model.q8_0.gguf",
        context: int = 4096,
        flash_attn: bool = True,
    ) -> None:
        if not (path := Path(path)).is_file():
            path = Path(hf.hf_hub_download(path.as_posix(), file))

        self.model = Llama(
            model_path=str(path),
            n_gpu_layers=-1,
            n_ctx=context,
            n_batch=context,
            n_ubatch=context,
            flash_attn=flash_attn,
            verbose=False,
        )

    def encode(self, text: str, bos: bool = False, special: bool = False) -> list[int]:
        return self.model.tokenize(text.encode(), bos, special)

    def decode(self, tokens: list[int], special: bool = False) -> str:
        return self.model.detokenize(tokens, special=special).decode()

    def generate(
        self,
        text: str,
        codes: str,
        top_k: int = 50,
        top_p: float = 0.95,
        min_p: float = 0.0,
        typical_p: float = 1.0,
        temp: float = 0.8,
        repeat_penalty: float = 1.0,
    ) -> str:
        inputs = (
            "<|task_tts|>"
            "<|start_content|>"
            f"{text}"
            "<|end_content|>"
            "<|start_global_token|>"
            f"{codes}"
            "<|end_global_token|>"
            "<|start_semantic_token|>"
        )

        inputs = self.encode(inputs, special=True)
        max_tokens = max(0, self.model.n_ctx() - len(inputs))
        outputs = []

        for token in self.model.generate(
            tokens=inputs,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            temp=temp,
            repeat_penalty=repeat_penalty,
        ):
            if token == self.model.token_eos() or len(outputs) >= max_tokens:
                break

            outputs.append(token)

        return self.decode(outputs, special=True)

    def unload(self) -> None:
        if self.model._sampler:
            self.model._sampler.close()

        self.model.close()
        self.model = None
        torch.cuda.empty_cache()


class Whisper:
    def __init__(
        self,
        model: str = "openai/whisper-large-v3-turbo",
        device: str = "cuda",
        dtype: str = "float16",
        flash_attn: bool = True,
        language: str = "en",
        task: Literal["transcribe", "translate"] = "transcribe",
        beams: int = 5,
    ) -> None:
        self.model = transformers.pipeline(
            task="automatic-speech-recognition",
            model=model,
            device=device,
            torch_dtype=getattr(torch, dtype),
            model_kwargs={
                "attn_implementation": (
                    "flash_attention_2"
                    if flash_attn and dtype in ["bfloat16", "float16"]
                    else "sdpa"
                )
            },
        )

        self.language = language
        self.task = task
        self.beams = beams

    def transcribe(self, audio: np.ndarray) -> str:
        return self.model(
            inputs=audio,
            generate_kwargs={
                "language": self.language,
                "task": self.task,
                "num_beams": self.beams,
            },
        )["text"].strip()

    def unload(self) -> None:
        self.model = None
        torch.cuda.empty_cache()


class FasterWhisper:
    def __init__(
        self,
        model: str = "turbo",
        device: str = "cuda",
        dtype: str = "float16",
        language: str = "en",
        task: Literal["transcribe", "translate"] = "transcribe",
        beams: int = 5,
    ) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device, compute_type=dtype)
        self.language = language
        self.task = task
        self.beams = beams

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio=audio,
            language=self.language,
            task=self.task,
            beam_size=self.beams,
        )

        return " ".join([s.text.strip() for s in segments])

    def unload(self) -> None:
        self.model = None
        torch.cuda.empty_cache()
