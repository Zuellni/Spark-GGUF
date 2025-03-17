import contextlib
import re
import sys
import warnings
from pathlib import Path
from typing import Literal

warnings.simplefilter("ignore")

import huggingface_hub as hf
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import transformers
from faster_whisper import WhisperModel
from llama_cpp import Llama
from omegaconf import OmegaConf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

transformers.logging.set_verbosity_error()

sys.path.append(str(Path(__file__).parent / "spark_tts"))
from sparktts.models.bicodec import BiCodec


class Codec:
    def __init__(
        self,
        model: Path = "annuvin/bicodec",
        wav2vec2: Path = "annuvin/wav2vec2",
        device: str = "cuda",
        dtype: Literal["float16", "float32"] = "float16",
        flash_attn: bool = True,
        ref_len: int = 6,
    ) -> None:
        if not Path(model).is_dir():
            model = hf.snapshot_download(model)

        if not Path(wav2vec2).is_dir():
            wav2vec2 = hf.snapshot_download(wav2vec2)

        self.device = device
        self.dtype = getattr(torch, dtype)

        with contextlib.redirect_stdout(None):
            self.model = BiCodec.load_from_checkpoint(model).to(self.device, self.dtype)

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec2)
        self.extractor = Wav2Vec2Model.from_pretrained(
            pretrained_model_name_or_path=wav2vec2,
            attn_implementation=(
                "flash_attention_2"
                if flash_attn and self.dtype == torch.float16
                else "sdpa"
            ),
            torch_dtype=self.dtype,
        ).to(self.device)
        self.extractor.config.output_hidden_states = True

        self.config = OmegaConf.load(Path(model) / "config.yaml")
        self.hop_len = self.config.audio_tokenizer.mel_params.hop_length
        self.sample_rate = self.config.audio_tokenizer.mel_params.sample_rate
        self.ref_len = int(self.sample_rate * ref_len) // self.hop_len * self.hop_len
        self.pattern = re.compile(r"<\|bicodec_semantic_(\d+)\|>")

    def _load(self, path: str) -> torch.Tensor:
        wav, sample_rate = torchaudio.load(path)
        wav = wav.to(self.device, self.dtype)

        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        if sample_rate != self.sample_rate:
            wav = F.resample(wav, sample_rate, self.sample_rate)

        return wav

    def _process(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.shape[1] < self.ref_len:
            wav = torch.tile(wav, (1, self.ref_len // wav.shape[1] + 1))

        return wav[:, : self.ref_len]

    def encode(self, path: str) -> tuple[torch.Tensor, str]:
        wav = self._load(path)
        ref = self._process(wav)

        inputs = self.processor(
            raw_speech=wav.squeeze(),
            output_hidden_states=True,
            padding=True,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
        ).input_values.to(self.device, self.dtype)

        feat = self.extractor(inputs).hidden_states
        feat = (feat[11] + feat[14] + feat[16]) / 3.0

        _, tokens = self.model.tokenize({"wav": wav, "ref_wav": ref, "feat": feat})
        tokens_str = "".join([f"<|bicodec_global_{t}|>" for t in tokens.squeeze()])
        return tokens, tokens_str

    def decode(self, tokens: torch.Tensor, tokens_str: str) -> np.ndarray:
        wav = [int(t) for t in re.findall(self.pattern, tokens_str)]
        wav = torch.tensor([wav], device=self.device)
        wav = self.model.detokenize(wav, tokens)
        return wav.squeeze().float().cpu().numpy()

    def warmup(self, tokens: torch.Tensor) -> None:
        self.model.detokenize(tokens.squeeze(0), tokens)


class Spark:
    def __init__(
        self,
        model: str = "annuvin/spark-gguf",
        dtype: str = "f16",
        context: int = 2048,
        flash_attn: bool = True,
    ) -> None:
        if not Path(model).is_file():
            model = hf.hf_hub_download(model, f"model.{dtype}.gguf")

        with contextlib.redirect_stderr(None), contextlib.redirect_stdout(None):
            self.model = Llama(
                model_path=model,
                n_gpu_layers=-1,
                n_ctx=context,
                flash_attn=flash_attn,
                verbose=False,
            )

    def encode(self, text: str, bos: bool = False, special: bool = False) -> list[int]:
        return self.model.tokenize(text.encode(), bos, special)

    def decode(self, tokens: list[int], special: bool = False) -> str:
        return self.model.detokenize(tokens, special=special).decode()

    def unload(self) -> None:
        if self.model._sampler:
            self.model._sampler.close()

        self.model.close()

    def __call__(self, text: str, tokens_str: str) -> str:
        text = (
            "<|task_tts|>"
            "<|start_content|>"
            f"{text}"
            "<|end_content|>"
            "<|start_global_token|>"
            f"{tokens_str}"
            "<|end_global_token|>"
            "<|start_semantic_token|>"
        )

        tokens = self.encode(text, special=True)
        tokens_list = []

        for token in self.model.generate(tokens):
            if token == self.model.token_eos():
                break

            tokens_list.append(token)

        return self.decode(tokens_list, special=True)


class Whisper:
    def __init__(
        self,
        model: str = "turbo",
        device: str = "cuda",
        dtype: str = "float16",
        language: str = "en",
        task: Literal["transcribe", "translate"] = "transcribe",
    ) -> None:
        self.model = WhisperModel(model, device, compute_type=dtype)
        self.language = language
        self.task = task

    def __call__(self, wav: np.ndarray) -> str:
        segments, _ = self.model.transcribe(wav, self.language, self.task)
        return " ".join([s.text.strip() for s in segments if s.text.strip()])
