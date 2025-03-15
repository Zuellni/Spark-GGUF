import re
import sys
from argparse import ArgumentParser
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from queue import Queue
from time import time
from typing import Literal, Self
from warnings import simplefilter

simplefilter("ignore")

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from llama_cpp import Llama
from rich import print

sys.path.append(str(Path(__file__).parent / "sparktts"))
from sparktts.sparktts.models.audio_tokenizer import BiCodecTokenizer


class Timer:
    def __init__(self) -> None:
        self.start = 0.0
        self.total = 0.0

    def __enter__(self) -> Self:
        self.start = time()
        return self

    def __exit__(self, *args) -> None:
        self.total = time() - self.start

    def __call__(self, text: str) -> None:
        print(f"{text} in {self.total:.2f} seconds")


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
        speaker, _ = self.model.tokenize(audio)
        speaker = speaker.to(self.device).squeeze(0)
        tokens = "".join([f"<|bicodec_global_{s}|>" for s in speaker.squeeze()])
        return speaker, tokens

    def decode(self, speaker: torch.Tensor, tokens: str) -> np.ndarray:
        tokens = [int(t) for t in re.findall(self.pattern, tokens)]
        tokens = torch.tensor([tokens], dtype=torch.long, device=self.device)
        return self.model.detokenize(speaker, tokens)

    def warmup(self, speaker: torch.Tensor) -> None:
        self.model.detokenize(speaker, speaker)


class Model:
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

    def __call__(self, text: str) -> str:
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

    def __call__(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, self.language, self.task)
        return " ".join([s.text.strip() for s in segments if s.text.strip()])


class App:
    def __init__(
        self,
        input: int,
        output: int,
        codec: Codec,
        model: Model,
        whisper: Whisper,
        speaker: torch.Tensor,
        tokens: str,
        sample_rate: int = 16000,
        block_size: int = 30,
        detection_threshold: float = 0.01,
        silence_threshold: int = 5,
        queue_threshold: int = 10,
    ) -> None:
        self.input = input
        self.output = output

        self.codec = codec
        self.model = model
        self.whisper = whisper

        self.speaker = speaker
        self.tokens = tokens

        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_size / 1000)
        self.detection_threshold = detection_threshold

        self.silence_threshold = silence_threshold
        self.silence_counter = 0

        self.queue_threshold = queue_threshold
        self.queue = Queue()

    def callback(self, data: np.ndarray, *args) -> None:
        if np.sqrt(np.mean(data**2)) >= self.detection_threshold:
            self.queue.put(data.copy())
            self.silence_counter = 0

    def __call__(self) -> None:
        with sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=self.input,
            channels=1,
            callback=self.callback,
        ):
            while True:
                sd.sleep(self.queue_threshold)

                if self.silence_counter < self.silence_threshold:
                    self.silence_counter += 1
                    continue

                if self.queue.qsize() < self.queue_threshold:
                    continue

                self.silence_counter = 0

                with Timer() as timer:
                    audio = [self.queue.get() for _ in range(self.queue.qsize())]
                    audio = np.concatenate(audio).squeeze()
                    text = self.whisper(audio)

                timer(f'[cyan]STT[/cyan]: Transcribed "{text}"')

                if not text:
                    continue

                prompt = (
                    "<|task_tts|>"
                    "<|start_content|>"
                    f"{text}"
                    "<|end_content|>"
                    "<|start_global_token|>"
                    f"{self.tokens}"
                    "<|end_global_token|>"
                )

                with Timer() as timer:
                    tokens = self.model(prompt)
                    audio = self.codec.decode(self.speaker, tokens)

                timer(
                    "[blue]TTS[/blue]: Generated "
                    f"{audio.shape[0] / self.sample_rate:.2f} "
                    "seconds of audio"
                )

                sd.play(audio, self.sample_rate, blocking=True, device=self.output)

    @staticmethod
    def devices(kind: Literal["input", "output"]) -> tuple[list[int], str]:
        device_ids = []
        device_str = []

        for device in sd.query_devices():
            if device[f"max_{kind}_channels"] and not device["hostapi"]:
                device_str.append(f"{device['index']}: {device['name']}")
                device_ids.append(device["index"])

        return device_ids, ", ".join(device_str)


if __name__ == "__main__":
    ic, ih = App.devices("input")
    oc, oh = App.devices("output")

    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=int, choices=ic, required=True, help=ih)
    parser.add_argument("-o", "--output", type=int, choices=oc, required=True, help=oh)
    parser.add_argument("-c", "--codec", default="sparkaudio/spark-tts-0.5b")
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-w", "--whisper", default="turbo")
    parser.add_argument("-s", "--speaker", required=True)
    args = parser.parse_args()

    with Timer() as timer:
        codec = Codec(args.codec)
    timer(f"[yellow]APP[/yellow]: Loaded codec")

    with Timer() as timer:
        model = Model(args.model)
    timer(f"[yellow]APP[/yellow]: Loaded model")

    with Timer() as timer:
        whisper = Whisper(args.whisper)
    timer(f"[yellow]APP[/yellow]: Loaded whisper")

    speaker, tokens = codec.encode(args.speaker)
    codec.warmup(speaker)

    app = App(
        args.input,
        args.output,
        codec,
        model=model,
        whisper=whisper,
        speaker=speaker,
        tokens=tokens,
    )

    try:
        print("[green]APP[/green]: Listening")
        app()
    except KeyboardInterrupt:
        print("[red]APP[/red]: Quitting")
        model.unload()
