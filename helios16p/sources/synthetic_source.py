"""A Source wrapper around `helios16p.synthetic`, for development and demos
with no sensor and no recording."""
from __future__ import annotations

from typing import Iterator

from ..synthetic import synthetic_sequence
from ..types import Frame
from .base import Source


class SyntheticSource(Source):
    name = "synthetic"

    def __init__(self, n_frames: int = 240, fps: float = 10.0, n_people: int = 2, loop: bool = False):
        self.n_frames = n_frames
        self.fps = fps
        self.n_people = n_people
        self.loop = loop

    def __iter__(self) -> Iterator[Frame]:
        while True:
            yield from synthetic_sequence(self.n_frames, self.fps, self.n_people)
            if not self.loop:
                break
