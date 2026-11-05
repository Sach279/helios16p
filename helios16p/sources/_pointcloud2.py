"""Shared helper: decode a ROS `sensor_msgs/PointCloud2` payload into a `Frame`.

Used by both `mcap_source` and `ros_source` so the PointCloud2 byte-unpacking
lives in exactly one place. Works on the raw fields, so it does not import ROS
or rosbags itself.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import Frame

# sensor_msgs/PointField datatype enum -> numpy base type
_PF_TO_NP = {
    1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8",
}


def pc2_to_frame(
    *,
    width: int,
    height: int,
    point_step: int,
    fields: Sequence,          # objects with .name, .offset, .datatype
    data: bytes,
    is_bigendian: bool,
    timestamp: float,
    frame_id: int,
    drop_nan: bool = True,
) -> Frame:
    """Build a Frame from PointCloud2 components.

    `fields` items must expose `.name` (str), `.offset` (int), `.datatype` (int).
    """
    bo = ">" if is_bigendian else "<"
    names, formats, offsets = [], [], []
    for f in fields:
        if f.datatype not in _PF_TO_NP:
            continue
        names.append(f.name)
        formats.append(bo + _PF_TO_NP[f.datatype])
        offsets.append(int(f.offset))

    dt = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": int(point_step)})
    n = int(width) * int(height)
    arr = np.frombuffer(data, dtype=dt, count=n)

    xyz = np.stack(
        [arr["x"].astype(np.float32), arr["y"].astype(np.float32), arr["z"].astype(np.float32)],
        axis=1,
    )
    intensity = (
        arr["intensity"].astype(np.float32) if "intensity" in names else np.zeros(n, np.float32)
    )
    ring = arr["ring"].astype(np.int16) if "ring" in names else None

    if drop_nan:
        m = np.isfinite(xyz).all(axis=1)
        xyz = xyz[m]
        intensity = intensity[m]
        if ring is not None:
            ring = ring[m]

    return Frame(timestamp=float(timestamp), points=np.ascontiguousarray(xyz),
                 intensity=np.ascontiguousarray(intensity),
                 ring=None if ring is None else np.ascontiguousarray(ring),
                 frame_id=frame_id)
