"""Binary PLY writer with intensity and ring as vertex properties (numpy only).

CloudCompare, MeshLab and Open3D read these files; the extra scalar fields
survive as proper attributes, not only as baked-in colours.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ply(path: str | Path, points: np.ndarray, intensity: np.ndarray | None = None,
              ring: np.ndarray | None = None, colors: np.ndarray | None = None) -> Path:
    """Write ``points`` (N, 3) as binary little-endian PLY.

    ``intensity`` (N,) is stored as float32, ``ring`` (N,) as uchar, ``colors``
    (N, 3) in 0..1 as uchar red/green/blue.  Raises OSError when the file cannot
    be written, so callers never report a success that did not happen.
    """
    pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    n = len(pts)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    header = ["ply", "format binary_little_endian 1.0", f"comment written by helios16p", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if intensity is not None:
        fields.append(("intensity", "<f4")); header.append("property float intensity")
    if ring is not None:
        fields.append(("ring", "u1")); header.append("property uchar ring")
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    rec = np.empty(n, dtype=np.dtype(fields))
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if intensity is not None:
        rec["intensity"] = np.asarray(intensity, dtype=np.float32).reshape(-1)
    if ring is not None:
        rec["ring"] = np.clip(np.asarray(ring).reshape(-1), 0, 255).astype(np.uint8)
    if colors is not None:
        c = np.clip(np.asarray(colors, dtype=np.float64).reshape(-1, 3) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        rec["red"], rec["green"], rec["blue"] = c[:, 0], c[:, 1], c[:, 2]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(rec.tobytes())
    return p


def read_ply_xyz(path: str | Path) -> np.ndarray:
    """Read the x, y, z of a binary or ascii PLY written by this module or by Open3D (float or double)."""
    import open3d as o3d  # the general reader; our own files are a subset of what it reads
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)
