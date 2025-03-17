from argparse import ArgumentParser
from queue import Queue
from typing import Literal

import numpy as np
import sounddevice as sd
import torch

from models import Codec, Spark, Whisper
from utils import Logger, Timer


class Application:
    @staticmethod
    def devices(kind: Literal["input", "output"]) -> tuple[list[int], str]:
        device_list = []
        device_ids = []

        for device in sd.query_devices():
            if device[f"max_{kind}_channels"] and device["hostapi"] == 0:
                device_list.append(f"{device['index']}: {device['name']}")
                device_ids.append(device["index"])

        return device_ids, ", ".join(device_list)

    def __init__(
        self,
        input: int,
        output: int,
        codec: Codec,
        spark: Spark,
        whisper: Whisper,
        tokens: torch.Tensor,
        tokens_str: str,
        block_duration: int = 30,
        detection_threshold: float = 0.01,
        silence_threshold: int = 8,
        queue_threshold: int = 16,
    ) -> None:
        self.input = input
        self.output = output

        self.codec = codec
        self.spark = spark
        self.whisper = whisper

        self.tokens = tokens
        self.tokens_str = tokens_str

        self.block_duration = block_duration
        self.block_size = int(codec.sample_rate * block_duration / 1000)
        self.detection_threshold = detection_threshold
        self.silence_threshold = silence_threshold
        self.queue_threshold = queue_threshold

        self.silence_counter = 0
        self.queue = Queue()

    def callback(
        self, data: np.ndarray, frames: int, _, status: sd.CallbackFlags
    ) -> None:
        if frames != self.block_size or status:
            Logger.error("Recording error")
            return

        if np.sqrt(np.mean(data**2)) >= self.detection_threshold:
            self.queue.put(data.copy())
            self.silence_counter = 0

    def __call__(self) -> None:
        with sd.InputStream(
            samplerate=self.codec.sample_rate,
            blocksize=self.block_size,
            device=self.input,
            channels=1,
            callback=self.callback,
        ):
            while True:
                sd.sleep(self.block_duration)

                if self.silence_counter < self.silence_threshold:
                    self.silence_counter += 1
                    continue

                if self.queue.qsize() < self.queue_threshold:
                    continue

                self.silence_counter = 0

                with Timer() as timer:
                    try:
                        data = [self.queue.get() for _ in range(self.queue.qsize())]
                        data = np.concatenate(data).squeeze()
                        text = whisper(data)
                    except:
                        Logger.error("Transcription error")
                        continue

                timer(
                    f"Transcribed {len(data) / self.codec.sample_rate:.2f}"
                    " seconds of audio"
                )

                if text:
                    Logger.info(f'Transcribed "{text}"')
                else:
                    Logger.warn("Empty transcript")
                    continue

                with Timer() as timer:
                    try:
                        tokens_str = spark(text, self.tokens_str)
                    except:
                        Logger.error("Generation error")
                        continue

                    try:
                        data = codec.decode(self.tokens, tokens_str)
                    except:
                        Logger.error("Decoding error")
                        continue

                timer(
                    f"Generated {len(data) / self.codec.sample_rate:.2f}"
                    " seconds of audio"
                )

                try:
                    Logger.info("Playing")
                    sd.play(data, self.codec.sample_rate, device=self.output)
                except:
                    Logger.error("Playback error")


if __name__ == "__main__":
    ic, ih = Application.devices("input")
    oc, oh = Application.devices("output")

    parser = ArgumentParser()
    parser.add_argument("-a", "--audio", required=True, help="audio file")
    parser.add_argument("-i", "--input", type=int, choices=ic, required=True, help=ih)
    parser.add_argument("-o", "--output", type=int, choices=oc, required=True, help=oh)
    parser.add_argument("-c", "--codec", default="annuvin/bicodec")
    parser.add_argument("-s", "--spark", default="annuvin/spark-gguf")
    parser.add_argument("-v", "--wav2vec2", default="annuvin/wav2vec2")
    parser.add_argument("-w", "--whisper", default="openai/whisper-large-v3-turbo")
    args = parser.parse_args()

    with Timer("Loaded codec"):
        codec = Codec(args.codec, args.wav2vec2)

    with Timer("Loaded spark"):
        spark = Spark(args.spark)

    with Timer("Loaded whisper"):
        whisper = Whisper(args.whisper)

    with Timer("Encoded audio"):
        tokens, tokens_str = codec.encode(args.audio)

    application = Application(
        input=args.input,
        output=args.output,
        codec=codec,
        spark=spark,
        whisper=whisper,
        tokens=tokens,
        tokens_str=tokens_str,
    )

    try:
        Logger.info("Listening")
        application()
    except KeyboardInterrupt:
        Logger.warn("Quitting")
        spark.unload()
