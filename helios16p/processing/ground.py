"""Ground segmentation for the Helios 16P pipeline.

Approach
--------
1. Pre-filter: keep only points with z < cfg.ground_max_z as candidates.
2. Fit a dominant plane via RANSAC (open3d PointCloud.segment_plane, which
   uses the open3d RANSAC implementation under the hood — fast and robust).
3. A point is marked ground if:
   - its signed distance to the fitted plane is within cfg.ground_distance_threshold, AND
   - the plane normal is near-horizontal, i.e. |n_z| > cos(15°) ≈ 0.966.
4. Degenerate cases (< 3 candidate points, RANSAC failure) return all-False.
"""
from __future__ import annotations

import numpy as np

from ..config import DemoConfig

# Maximum tilt for the ground plane normal (degrees from vertical).
_MAX_TILT_DEG = 15.0
_MIN_NZ = np.cos(np.radians(_MAX_TILT_DEG))   # ≈ 0.966


def _fit_plane_numpy(pts: np.ndarray, cfg: DemoConfig, n_iter: int = 200) -> np.ndarray | None:
    """Minimal RANSAC plane fit.  Returns (4,) [a, b, c, d] (unit normal) or None."""
    rng = np.random.default_rng(42)
    best_n = 0
    best_plane = None
    thr = cfg.ground_distance_threshold

    for _ in range(n_iter):
        idx = rng.choice(len(pts), size=3, replace=False)
        p0, p1, p2 = pts[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -float(n @ p0)
        dist = np.abs(pts @ n + d)
        inliers = int((dist < thr).sum())
        if inliers > best_n:
            best_n = inliers
            best_plane = np.array([n[0], n[1], n[2], d], dtype=np.float64)

    return best_plane


def segment_ground(points: np.ndarray, cfg: DemoConfig) -> np.ndarray:
    """Segment ground points.

    Parameters
    ----------
    points : (N, 3) float32
        Full point cloud in sensor frame, z up, NaN-free.
    cfg : DemoConfig

    Returns
    -------
    (N,) bool
        True where the point is classified as ground.
    """
    N = points.shape[0]
    ground_mask = np.zeros(N, dtype=bool)

    if N < 3:
        return ground_mask

    # --- Step 1: candidate points below max_z ---
    cand_idx = np.where(points[:, 2] < cfg.ground_max_z)[0]
    if len(cand_idx) < 3:
        return ground_mask

    cand = points[cand_idx].astype(np.float64)

    # --- Step 2: fit plane ---
    plane = None
    try:
        import open3d as o3d  # optional fast path
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cand)
        o3d.utility.random.seed(int(getattr(cfg, "seed", 0)))  # repeatable results run to run
        plane_model, _ = pcd.segment_plane(
            distance_threshold=cfg.ground_distance_threshold,
            ransac_n=3,
            num_iterations=200,
        )
        plane = np.array(plane_model, dtype=np.float64)  # [a, b, c, d]
    except Exception:
        plane = _fit_plane_numpy(cand, cfg)

    if plane is None:
        return ground_mask

    a, b, c, d = plane
    # Normalise normal vector (open3d already returns unit normal, numpy RANSAC too)
    n_len = np.sqrt(a * a + b * b + c * c)
    if n_len < 1e-9:
        return ground_mask
    a, b, c = a / n_len, b / n_len, c / n_len
    d = d / n_len

    # --- Step 3: near-horizontal check ---
    if abs(c) < _MIN_NZ:
        return ground_mask

    # --- Step 4: apply to all points ---
    pts64 = points.astype(np.float64)
    dist = np.abs(pts64[:, 0] * a + pts64[:, 1] * b + pts64[:, 2] * c + d)
    ground_mask = dist < cfg.ground_distance_threshold

    return ground_mask
