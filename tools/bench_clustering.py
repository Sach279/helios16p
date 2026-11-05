"""Benchmark clustering backends and voxel settings on a real recording.

Usage
-----
    python tools/bench_clustering.py [--rec PATH] [--frames N] [--reps K]

Replays up to --frames frames (default 20) through PacketFileSource, and for
each combination of backend in {sklearn, open3d} and voxel in {0.0, 0.05, 0.10}
times cluster_objects (best of --reps trials per frame, default 3).

Reports a table of mean latency (ms) and mean cluster count per combination,
then validates that Detection.point_indices correctly index frame.points
(i.e. frame.points[det.point_indices] lies within det.bbox_min / det.bbox_max).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path when run as a script.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from helios16p.config import DemoConfig
from helios16p.processing.clustering import cluster_objects
from helios16p.processing.ground import segment_ground
from helios16p.sources.packet_file_source import PacketFileSource

from importlib import resources as _res
DEFAULT_REC = str(_res.files("helios16p.data").joinpath("walk_4s.npz"))  # the bundled clip; pass --rec for a longer one

BACKENDS = ["sklearn", "open3d"]
VOXELS = [0.0, 0.05, 0.10]


def _time_cluster(frame, ground_mask, cfg: DemoConfig, reps: int) -> tuple[float, int]:
    """Return (best_ms, cluster_count) for the given cfg on one frame."""
    best = float("inf")
    n_clusters = 0
    for _ in range(reps):
        t0 = time.perf_counter()
        dets = cluster_objects(frame, ground_mask, cfg)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
            n_clusters = len(dets)
    return best * 1e3, n_clusters


def _validate_point_indices(frames, ground_masks, cfg: DemoConfig) -> tuple[int, int]:
    """Check that frame.points[det.point_indices] lies within det.bbox.

    Returns (n_checked, n_violated).
    """
    checked = 0
    violated = 0
    tol = 1e-4  # float32 rounding tolerance (metres)
    for frame, gm in zip(frames, ground_masks):
        dets = cluster_objects(frame, gm, cfg)
        for det in dets:
            pts = frame.points[det.point_indices]  # (k, 3)
            # All points must be >= bbox_min - tol and <= bbox_max + tol
            ok = np.all(pts >= det.bbox_min - tol) and np.all(pts <= det.bbox_max + tol)
            checked += len(det.point_indices)
            if not ok:
                violated += int((~(
                    (pts >= det.bbox_min - tol).all(axis=1) &
                    (pts <= det.bbox_max + tol).all(axis=1)
                )).sum())
    return checked, violated


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark clustering backends")
    parser.add_argument("--rec", default=DEFAULT_REC, help="Path to .npz recording")
    parser.add_argument("--frames", type=int, default=20, help="Frames to replay (default 20)")
    parser.add_argument("--reps", type=int, default=3, help="Repeats per frame (best-of, default 3)")
    args = parser.parse_args()

    rec_path = args.rec
    if not Path(rec_path).exists():
        sys.exit(f"ERROR: recording not found: {rec_path}")

    print(f"Loading up to {args.frames} frames from:\n  {rec_path}\n")

    # Load frames + ground masks once (amortise I/O from timing)
    base_cfg = DemoConfig()
    src = PacketFileSource(rec_path, max_frames=args.frames)
    frames = []
    ground_masks = []
    for frame in src:
        gm = segment_ground(frame.points, base_cfg)
        frames.append(frame)
        ground_masks.append(gm)

    n_frames = len(frames)
    print(f"Loaded {n_frames} frames.\n")

    if n_frames == 0:
        sys.exit("ERROR: no frames decoded from recording.")

    # ------------------------------------------------------------------
    # Benchmark grid: backend x voxel
    # ------------------------------------------------------------------
    results: dict[tuple[str, float], tuple[float, float]] = {}  # (backend, voxel) -> (mean_ms, mean_n)

    col_w = 12
    header = (
        f"{'Backend':<10} {'Voxel(m)':<10} {'Mean ms':>{col_w}} {'Mean clusters':>{col_w}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for backend in BACKENDS:
        for voxel in VOXELS:
            cfg = DemoConfig()
            cfg.cluster_backend = backend
            cfg.cluster_voxel = voxel

            ms_list: list[float] = []
            n_list: list[int] = []

            for frame, gm in zip(frames, ground_masks):
                ms, n = _time_cluster(frame, gm, cfg, args.reps)
                ms_list.append(ms)
                n_list.append(n)

            mean_ms = float(np.mean(ms_list))
            mean_n = float(np.mean(n_list))
            results[(backend, voxel)] = (mean_ms, mean_n)

            print(
                f"{backend:<10} {voxel:<10.2f} {mean_ms:>{col_w}.2f} {mean_n:>{col_w}.1f}"
            )

    print(sep)

    # ------------------------------------------------------------------
    # Speedup summary
    # ------------------------------------------------------------------
    print("\nSpeedup open3d vs sklearn (same voxel setting):")
    for voxel in VOXELS:
        ms_sk = results[("sklearn", voxel)][0]
        ms_o3d = results[("open3d", voxel)][0]
        speedup = ms_sk / ms_o3d if ms_o3d > 0 else float("inf")
        tag = "FASTER" if speedup > 1.0 else "SLOWER"
        print(f"  voxel={voxel:.2f}m : sklearn {ms_sk:.2f} ms  open3d {ms_o3d:.2f} ms  -> {speedup:.2f}x {tag}")

    # ------------------------------------------------------------------
    # Point-index integrity check (open3d, no voxel — most points)
    # ------------------------------------------------------------------
    print("\nValidating Detection.point_indices integrity (open3d, voxel=0.0) ...")
    val_cfg = DemoConfig()
    val_cfg.cluster_backend = "open3d"
    val_cfg.cluster_voxel = 0.0
    n_checked, n_violated = _validate_point_indices(frames, ground_masks, val_cfg)
    if n_violated == 0:
        print(f"  OK — all {n_checked} indexed points lie within their detection bbox.")
    else:
        print(f"  FAIL — {n_violated}/{n_checked} points lie OUTSIDE their detection bbox!")
        sys.exit(1)

    # Also validate with voxel downsampling
    print("Validating Detection.point_indices integrity (open3d, voxel=0.05) ...")
    val_cfg2 = DemoConfig()
    val_cfg2.cluster_backend = "open3d"
    val_cfg2.cluster_voxel = 0.05
    n_checked2, n_violated2 = _validate_point_indices(frames, ground_masks, val_cfg2)
    if n_violated2 == 0:
        print(f"  OK — all {n_checked2} indexed points lie within their detection bbox.")
    else:
        print(f"  FAIL — {n_violated2}/{n_checked2} points lie OUTSIDE their detection bbox!")
        sys.exit(1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
