"""Object clustering for the Helios 16P pipeline.

Approach
--------
1. Select non-ground points (indices where ground_mask is False).
2. Optionally voxel-downsample the clustering input (cfg.cluster_voxel > 0):
   - Quantise non-ground points to a voxel grid; cluster one representative
     point per voxel.  Each original point inherits its voxel's cluster label,
     so Detection geometry and point_indices are always full-resolution.
3. Run DBSCAN via the selected backend (cfg.cluster_backend):
   - "sklearn": sklearn.cluster.DBSCAN  (pure Python / Cython, reference)
   - "open3d":  open3d PointCloud.cluster_dbscan  (C++ / OpenMP, faster)
4. For each cluster label != -1:
   a. Compute AABB (bbox_min, bbox_max) and centroid on full-res points.
   b. Reject clusters whose vertical extent is outside
      [cfg.cluster_min_height, cfg.cluster_max_height].
   c. Build a Detection using the *original* frame.points indices.

Notes
-----
- point_indices are always indices into the ORIGINAL frame.points array.
- DBSCAN is applied in 3-D; the XY clustering criterion naturally groups
  vertical objects while the height filter removes short flat blobs (debris)
  and impossibly tall structures (walls).
"""
from __future__ import annotations

import numpy as np

from ..config import DemoConfig
from ..types import Detection, Frame


def _dbscan_sklearn(pts: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Run sklearn DBSCAN; returns label array (M,)."""
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
    return db.fit_predict(pts)


def _dbscan_open3d(pts: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """Run open3d cluster_dbscan (C++/OpenMP); returns label array (M,)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    labels = pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    return np.asarray(labels, dtype=np.int32)


def _voxel_downsample(
    pts: np.ndarray, voxel: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantise pts to a voxel grid; return (rep_pts, inv, first_occ).

    Parameters
    ----------
    pts : (M, 3) float32
        Points to downsample.
    voxel : float
        Voxel edge length in metres (> 0).

    Returns
    -------
    rep_pts : (V, 3) float32
        One representative point per occupied voxel (the first point that
        falls in it, in the order pts arrives).
    inv : (M,) int
        For each of the M original points, the voxel index (0..V-1) it belongs
        to.  Equivalently, ``rep_pts[inv[i]]`` is the representative of
        ``pts[i]``.
    first_occ : (V,) int
        Index into pts of the representative chosen for each voxel.
    """
    keys = np.floor(pts / voxel).astype(np.int32)  # (M, 3)
    # np.unique sorts lexicographically and returns the first occurrence
    _, first_occ, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    rep_pts = pts[first_occ]
    return rep_pts, inv, first_occ


def cluster_objects(
    frame: Frame,
    ground_mask: np.ndarray,
    cfg: DemoConfig,
) -> list[Detection]:
    """Cluster non-ground points into object candidates.

    Parameters
    ----------
    frame : Frame
        The current LiDAR frame.
    ground_mask : (N,) bool
        True where frame.points[i] is ground (from segment_ground).
    cfg : DemoConfig
        Clustering knobs: cluster_backend, cluster_voxel, cluster_eps,
        cluster_min_points, cluster_min_height, cluster_max_height.

    Returns
    -------
    list[Detection]
        One Detection per accepted cluster.  track_id is None; point_indices
        index the ORIGINAL frame.points array.
    """
    points = frame.points  # (N, 3) float32

    # Indices of non-ground points in the original array
    fg_idx = np.where(~ground_mask)[0]  # (M,)

    if len(fg_idx) < cfg.cluster_min_points:
        return []

    fg_pts = points[fg_idx]  # (M, 3)

    # ------------------------------------------------------------------
    # Optional voxel downsampling of the CLUSTERING INPUT
    # ------------------------------------------------------------------
    if cfg.cluster_voxel > 0.0:
        rep_pts, inv, _ = _voxel_downsample(fg_pts, cfg.cluster_voxel)
        cluster_pts = rep_pts          # (V, 3) — what DBSCAN sees
    else:
        cluster_pts = fg_pts           # (M, 3) — full resolution
        inv = None                     # not needed when no downsampling

    if len(cluster_pts) < cfg.cluster_min_points:
        return []

    # ------------------------------------------------------------------
    # DBSCAN — backend selection
    # ------------------------------------------------------------------
    backend = cfg.cluster_backend.lower()
    if backend == "open3d":
        voxel_labels = _dbscan_open3d(cluster_pts, cfg.cluster_eps, cfg.cluster_min_points)
    elif backend == "sklearn":
        voxel_labels = _dbscan_sklearn(cluster_pts, cfg.cluster_eps, cfg.cluster_min_points)
    else:
        raise ValueError(
            f"Unknown cluster_backend {cfg.cluster_backend!r}; expected 'open3d' or 'sklearn'."
        )

    # ------------------------------------------------------------------
    # Propagate voxel labels back to every original non-ground point
    # ------------------------------------------------------------------
    if inv is not None:
        # Each original fg point inherits the label of its voxel representative
        labels = voxel_labels[inv]     # (M,)
    else:
        labels = voxel_labels          # (M,)

    # ------------------------------------------------------------------
    # Build Detections from full-resolution points
    # ------------------------------------------------------------------
    detections: list[Detection] = []

    for lbl in np.unique(labels):
        if lbl == -1:
            continue  # noise

        local_mask = labels == lbl     # (M,) bool, selects fg points in cluster
        local_pts = fg_pts[local_mask]  # full-resolution cluster points

        bbox_min = local_pts.min(axis=0)  # (3,)
        bbox_max = local_pts.max(axis=0)  # (3,)
        vertical_extent = float(bbox_max[2] - bbox_min[2])

        # Height filter
        if (
            vertical_extent < cfg.cluster_min_height
            or vertical_extent > cfg.cluster_max_height
        ):
            continue

        # Footprint filter: reject clusters too wide horizontally (walls,
        # furniture, merged structures) so only person-/obstacle-sized objects
        # survive. A standing person is ~0.3-0.8 m across.
        footprint = float(max(bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1]))
        if footprint > cfg.cluster_max_footprint:
            continue

        centroid = local_pts.mean(axis=0)  # (3,)

        # Map back to original frame.points indices
        orig_indices = fg_idx[local_mask]  # (k,) int

        detections.append(
            Detection(
                centroid=centroid.astype(np.float32),
                bbox_min=bbox_min.astype(np.float32),
                bbox_max=bbox_max.astype(np.float32),
                num_points=int(local_mask.sum()),
                point_indices=orig_indices,
                track_id=None,
            )
        )

    return detections
