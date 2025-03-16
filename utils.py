from time import time
from typing import Self

from rich import print


class Logger:
    @staticmethod
    def info(text: str) -> None:
        print(f"[green]INFO[/]: {text}")

    @staticmethod
    def warn(text: str) -> None:
        print(f"[yellow]WARN[/]: {text}")

    @staticmethod
    def error(text: str) -> None:
        print(f"[red]ERRR[/]: {text}")


class Timer:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.start = 0.0
        self.stop = 0.0
        self.total = 0.0

    def __enter__(self) -> Self:
        self.start = time()
        return self

    def __exit__(self, *args) -> None:
        self.stop = time()
        self.total = self.stop - self.start
        self.text and self(self.text)

    def __call__(self, text: str) -> None:
        Logger.info(f"{text} in {self.total:.2f} seconds")
