"""McapSource: read ROS 2 MCAP bags without a ROS install.

Uses the `rosbags` library (v0.11.x) to deserialise sensor_msgs/PointCloud2
messages from a bag directory or a single .mcap file, converting each message
to a `helios16p.types.Frame` via the shared pc2_to_frame helper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..types import Frame
from .base import Source
from ._pointcloud2 import pc2_to_frame

TOPIC = "/rslidar_points"


class McapSource(Source):
    """Yield :class:`Frame` objects from a ROS 2 MCAP bag.

    Parameters
    ----------
    path:
        Path to a bag *directory* (containing ``metadata.yaml``) or to a
        single ``.mcap`` file.
    loop:
        If ``True``, restart from the beginning when the bag is exhausted.
    max_frames:
        Stop after this many frames (``None`` = unlimited).
    """

    name = "mcap"

    def __init__(
        self,
        path: str | Path,
        loop: bool = False,
        max_frames: int | None = None,
    ) -> None:
        self._path = Path(path)
        self._loop = loop
        self._max_frames = max_frames

    # ------------------------------------------------------------------
    # Source protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Frame]:
        from rosbags.highlevel import AnyReader  # local import — avoids hard dep at module load

        frame_id = 0

        while True:
            with AnyReader([self._path]) as reader:
                conns = [c for c in reader.connections if c.topic == TOPIC]
                if not conns:
                    raise RuntimeError(
                        f"No topic {TOPIC!r} found in bag {self._path}. "
                        f"Available topics: {[c.topic for c in reader.connections]}"
                    )

                for conn, _ts, raw in reader.messages(connections=conns):
                    msg = reader.deserialize(raw, conn.msgtype)
                    frame = pc2_to_frame(
                        width=msg.width,
                        height=msg.height,
                        point_step=msg.point_step,
                        fields=msg.fields,
                        data=bytes(msg.data),
                        is_bigendian=bool(msg.is_bigendian),
                        timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                        frame_id=frame_id,
                    )
                    yield frame
                    frame_id += 1
                    if self._max_frames is not None and frame_id >= self._max_frames:
                        return

            if not self._loop:
                break

    def close(self) -> None:
        """No-op: AnyReader is used as a context manager per iteration."""
