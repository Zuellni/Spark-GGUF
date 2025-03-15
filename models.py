import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Literal
from warnings import simplefilter

simplefilter("ignore")

import numpy as np
import torch
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from llama_cpp import Llama

sys.path.append(str(Path(__file__).parent / "spark_tts"))
from spark_tts.sparktts.models.audio_tokenizer import BiCodecTokenizer


class Codec:
    def __init__(
        self, model: str = "sparkaudio/spark-tts-0.5b", device: str = "cuda"
    ) -> None:
        with redirect_stdout(None):
            if not Path(model).is_dir():
                model = snapshot_download(model)

            self.model = BiCodecTokenizer(model, device)

        self.device = device
        self.pattern = re.compile(r"<\|bicodec_semantic_(\d+)\|>")

    def encode(self, audio: str) -> tuple[torch.Tensor, str]:
        audio, _ = self.model.tokenize(audio)
        audio = audio.to(self.device).squeeze(0)
        tokens = [f"<|bicodec_global_{a}|>" for a in audio.squeeze()]
        return audio, "".join(tokens)

    def decode(self, audio: torch.Tensor, tokens: str) -> np.ndarray:
        tokens = [int(t) for t in re.findall(self.pattern, tokens)]
        tokens = torch.tensor([tokens], dtype=torch.long, device=self.device)
        return self.model.detokenize(audio, tokens)

    def warmup(self, audio: torch.Tensor) -> None:
        self.model.detokenize(audio, audio)


class Spark:
    def __init__(self, model: str, context: int = 2048) -> None:
        with redirect_stderr(None), redirect_stdout(None):
            self.model = Llama(
                model_path=model,
                n_gpu_layers=-1,
                n_ctx=context,
                flash_attn=True,
                verbose=False,
            )

    def encode(self, text: str, bos: bool = False, special: bool = False) -> list[int]:
        return self.model.tokenize(text.encode(), add_bos=bos, special=special)

    def decode(self, tokens: list[int], special: bool = False) -> str:
        return self.model.detokenize(tokens, special=special).decode()

    def unload(self) -> None:
        self.model._sampler.close()
        self.model.close()

    def __call__(self, text: str, audio: str) -> str:
        text = (
            "<|task_tts|>"
            "<|start_content|>"
            f"{text}"
            "<|end_content|>"
            "<|start_global_token|>"
            f"{audio}"
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

    def __call__(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, self.language, self.task)
        return " ".join([s.text.strip() for s in segments if s.text.strip()])
