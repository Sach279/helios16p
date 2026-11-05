"""Source abstraction: anything that yields `Frame`s.

A Source is an iterable of Frame. Concrete sources (mcap, udp, ros, synthetic)
must not depend on each other. The rest of the pipeline only ever sees Frames,
which is what makes "with ROS / without ROS" a one-line swap.
"""
from __future__ import annotations

import abc
from typing import Iterator

from ..types import Frame


class Source(abc.ABC):
    """Base class for all frame sources."""

    name: str = "source"

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Frame]:
        """Yield Frames until the stream ends (offline) or forever (live)."""
        raise NotImplementedError

    def close(self) -> None:
        """Release sockets / file handles. Safe to call multiple times."""

    def __enter__(self) -> "Source":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False
