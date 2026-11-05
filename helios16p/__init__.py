"""helios16p: the RoboSense Helios 16P LiDAR in pure Python.

    source (Frame) -> ProcessingPipeline (FrameAnnotations) -> viewer

Sources: PacketFileSource (recordings), UdpSource (live), McapSource (ROS 2
bags without ROS), RosSource (a sourced ROS 2), SyntheticSource.  The data
contracts live in ``helios16p.types``; the sensor's angle calibration in
``helios16p.calibration``.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

__version__ = "0.2.0"


def bundled_clip() -> Path:
    """The 4 s indoor recording shipped with the package (a person walking past the sensor)."""
    return Path(str(resources.files("helios16p.data").joinpath("walk_4s.npz")))


def __getattr__(name: str):  # lazy, so `import helios16p` stays light
    if name == "PacketFileSource":
        from .sources.packet_file_source import PacketFileSource
        return PacketFileSource
    if name == "UdpSource":
        from .sources.udp_source import UdpSource
        return UdpSource
    if name == "Calibration":
        from .calibration import Calibration
        return Calibration
    raise AttributeError(name)


__all__ = ["bundled_clip", "PacketFileSource", "UdpSource", "Calibration", "__version__"]
