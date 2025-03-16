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
from faster_whisper import WhisperModel
from llama_cpp import Llama
from omegaconf import OmegaConf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

sys.path.append(str(Path(__file__).parent / "spark_tts"))
from sparktts.models.bicodec import BiCodec


class Codec:
    def __init__(
        self,
        model: str = "annuvin/bicodec",
        wav2vec2: str = "annuvin/wav2vec2",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        ref_len: int = 6,
    ) -> None:
        if not Path(model).is_dir():
            model = hf.snapshot_download(model)

        if not Path(wav2vec2).is_dir():
            wav2vec2 = hf.snapshot_download(wav2vec2)

        model = Path(model)
        wav2vec2 = Path(wav2vec2)
        config = OmegaConf.load(model / "config.yaml")
        hop_length = config["audio_tokenizer"]["mel_params"]["hop_length"]
        self.sample_rate = config["audio_tokenizer"]["mel_params"]["sample_rate"]

        with contextlib.redirect_stdout(None):
            self.model = BiCodec.load_from_checkpoint(model).to(device, dtype)

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec2)
        self.extractor = Wav2Vec2Model.from_pretrained(wav2vec2, torch_dtype=dtype)
        self.extractor.config.output_hidden_states = True
        self.extractor.to(device)

        self.ref_len = int(self.sample_rate * ref_len) // hop_length * hop_length
        self.pattern = re.compile(r"<\|bicodec_semantic_(\d+)\|>")
        self.device = device
        self.dtype = dtype

    def _load(self, path: str) -> torch.Tensor:
        wav, sample_rate = torchaudio.load(path)
        wav = wav.to(self.device, self.dtype)

        if len(wav) > 1:
            wav = torch.mean(wav, dim=0)

        if sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate)

        return wav.squeeze()

    def _process(self, wav: torch.Tensor) -> torch.Tensor:
        if len(wav) < self.ref_len:
            wav = torch.tile(wav, (0, self.ref_len // len(wav) + 1))

        return wav[: self.ref_len]

    def encode(self, path: str) -> tuple[torch.Tensor, str]:
        wav = self._load(path)
        ref = self._process(wav)

        inputs = self.processor(
            raw_speech=wav,
            output_hidden_states=True,
            padding=True,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
        ).input_values.to(self.device, self.dtype)

        feat = self.extractor(inputs).hidden_states
        feat = (feat[11] + feat[14] + feat[16]) / 3.0
        wav = wav.unsqueeze(0)
        ref = ref.unsqueeze(0)

        _, tensor = self.model.tokenize({"wav": wav, "ref_wav": ref, "feat": feat})
        token_str = "".join([f"<|bicodec_global_{t}|>" for t in tensor.squeeze()])
        return tensor, token_str

    def decode(self, tensor: torch.Tensor, token_str: str) -> np.ndarray:
        tokens = [int(t) for t in re.findall(self.pattern, token_str)]
        tokens = torch.tensor([tokens], device=self.device)
        tokens = self.model.detokenize(tokens, tensor)
        return tokens.squeeze().float().cpu().numpy()


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

    def encode(
        self, text: str, add_bos: bool = False, special: bool = False
    ) -> list[int]:
        return self.model.tokenize(text.encode(), add_bos, special)

    def decode(self, tokens: list[int], special: bool = False) -> str:
        return self.model.detokenize(tokens, special=special).decode()

    def unload(self) -> None:
        if self.model._sampler:
            self.model._sampler.close()

        self.model.close()

    def __call__(self, text: str, token_str: str) -> str:
        text = (
            "<|task_tts|>"
            "<|start_content|>"
            f"{text}"
            "<|end_content|>"
            "<|start_global_token|>"
            f"{token_str}"
            "<|end_global_token|>"
        )

        tokens = self.encode(text, special=True)
        token_list = []

        for token in self.model.generate(tokens):
            if token == self.model.token_eos():
                break

            token_list.append(token)

        return self.decode(token_list, special=True)


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
