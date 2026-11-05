"""Tests for ground segmentation and object clustering.

Run with:
    python tests/test_processing.py

Uses synthetic frames (no sensor required).

Sensitivity note
----------------
The default DemoConfig() is designed for real Helios-16P data (~1 m sensor height,
outdoor/corridor scenes with many thousands of points).  For the synthetic scene
(ground at z≈0, walls, two people per frame) the defaults work well because:
  - ground_max_z=0.30 m catches the Gaussian ground at z~N(0,0.01)
  - ground_distance_threshold=0.10 m is generous for an almost-flat synthetic floor
  - cluster_eps=0.5 m finds the 0.44 m-diameter person cylinders
  - cluster_min_points=15 is easily met by the 350-point person blobs
  - height filter [0.3, 2.5] m brackets the 1.75 m tall cylinders

If the synthetic geometry changes (e.g. person height shrinks below 0.3 m) the
height filter would need relaxing, but with the current synthetic.py those default
values work.
"""

import sys
import os

# Ensure project root is on sys.path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import helios16p.synthetic as syn
from helios16p.config import DemoConfig
from helios16p.processing.ground import segment_ground
from helios16p.processing.clustering import cluster_objects


def test_ground_segmentation(frames, cfg):
    print("\n--- Ground Segmentation ---")
    for frame in frames:
        N = frame.num_points
        mask = segment_ground(frame.points, cfg)

        assert mask.shape == (N,), f"mask shape mismatch: {mask.shape} vs ({N},)"
        assert mask.dtype == bool, f"mask dtype should be bool, got {mask.dtype}"

        frac = mask.sum() / N
        print(f"  frame {frame.frame_id}: {mask.sum()} / {N} ground points ({frac:.1%})")

        # The synthetic ground has 4000 points and walls+people ~7×1500+700 ≈ 11200.
        # Ground should be the largest single surface; we expect > 20 % of total cloud.
        assert frac > 0.20, (
            f"frame {frame.frame_id}: ground fraction too low ({frac:.1%}); "
            "expected > 20%"
        )

        # All flagged-ground points should be near z = 0 (within threshold + small slack)
        z_limit = cfg.ground_distance_threshold + 0.05
        ground_z = frame.points[mask, 2]
        max_abs_z = float(np.abs(ground_z).max()) if ground_z.size > 0 else 0.0
        print(f"    max |z| of ground points: {max_abs_z:.4f} m")
        assert max_abs_z <= z_limit, (
            f"frame {frame.frame_id}: a ground point has |z|={max_abs_z:.4f} m > {z_limit:.3f} m"
        )

    print("  ground segmentation PASS")


def test_clustering(frames, cfg):
    print("\n--- Object Clustering ---")
    # Track per-frame detection counts to assess average
    det_counts = []

    for frame in frames:
        mask = segment_ground(frame.points, cfg)
        dets = cluster_objects(frame, mask, cfg)
        n_det = len(dets)
        det_counts.append(n_det)
        print(f"  frame {frame.frame_id}: {n_det} detections")

        for i, det in enumerate(dets):
            # point_indices must index into frame.points correctly
            assert det.point_indices.ndim == 1, "point_indices must be 1-D"
            assert det.point_indices.dtype in (
                np.int32, np.int64, np.intp, int
            ), f"point_indices dtype unexpected: {det.point_indices.dtype}"

            # All indices in valid range
            assert det.point_indices.min() >= 0
            assert det.point_indices.max() < frame.num_points, (
                f"det {i}: index {det.point_indices.max()} out of range "
                f"(frame has {frame.num_points} points)"
            )

            # Points via indices should lie within the detection bounding box
            indexed_pts = frame.points[det.point_indices]
            for ax in range(3):
                assert np.all(indexed_pts[:, ax] >= det.bbox_min[ax] - 1e-4), (
                    f"det {i} axis {ax}: indexed point below bbox_min"
                )
                assert np.all(indexed_pts[:, ax] <= det.bbox_max[ax] + 1e-4), (
                    f"det {i} axis {ax}: indexed point above bbox_max"
                )

            # Centroid must be inside (or very close to) the bbox
            for ax in range(3):
                assert det.centroid[ax] >= det.bbox_min[ax] - 1e-3, (
                    f"det {i} centroid[{ax}] below bbox_min"
                )
                assert det.centroid[ax] <= det.bbox_max[ax] + 1e-3, (
                    f"det {i} centroid[{ax}] above bbox_max"
                )

            # Height filter was applied
            height = float(det.bbox_max[2] - det.bbox_min[2])
            assert cfg.cluster_min_height <= height <= cfg.cluster_max_height, (
                f"det {i}: height {height:.3f} m outside filter "
                f"[{cfg.cluster_min_height}, {cfg.cluster_max_height}]"
            )

            # Person-height cluster: centroid z should be around 0..1.75 / 2 ≈ 0.875
            assert 0.0 <= det.centroid[2] <= 2.0, (
                f"det {i}: centroid z={det.centroid[2]:.3f} implausible"
            )

            # track_id must be None (not set by clustering)
            assert det.track_id is None, f"det {i}: track_id should be None from clustering"

            print(
                f"    det {i}: centroid=({det.centroid[0]:.2f},{det.centroid[1]:.2f},"
                f"{det.centroid[2]:.2f})  height={height:.2f} m  pts={det.num_points}"
            )

    # Across the sequence we expect roughly 2–3 detections per frame:
    # the 2 person cylinders plus occasionally the wall assembly (all 4 walls
    # merge into one blob in the synthetic room; their vertical extent is ~2.5 m
    # which just passes the cfg.cluster_max_height=2.5 filter).  That wall blob
    # is a known synthetic artefact — in real data walls are further from the
    # sensor and their z span exceeds the threshold.  Allow 1..4 per frame.
    avg_dets = np.mean(det_counts)
    print(f"\n  Average detections per frame: {avg_dets:.2f}")
    assert 1.0 <= avg_dets <= 4.0, (
        f"Average detections {avg_dets:.2f} not in expected range [1, 4]. "
        "Check cluster_eps / height filter."
    )

    print("  clustering PASS")


def test_edge_cases(cfg):
    """Degenerate inputs must not crash and must return correct shapes."""
    print("\n--- Edge Cases ---")

    # Empty point cloud
    empty = np.zeros((0, 3), dtype=np.float32)
    mask = segment_ground(empty, cfg)
    assert mask.shape == (0,) and mask.dtype == bool, "empty input failed"
    print("  empty cloud: PASS")

    # Single point
    one = np.array([[0.0, 0.0, 0.05]], dtype=np.float32)
    mask = segment_ground(one, cfg)
    assert mask.shape == (1,) and mask.dtype == bool
    print("  single-point cloud: PASS")

    # Two points
    two = np.array([[0.0, 0.0, 0.05], [1.0, 0.0, 0.05]], dtype=np.float32)
    mask = segment_ground(two, cfg)
    assert mask.shape == (2,) and mask.dtype == bool
    print("  two-point cloud: PASS")

    # All points very high (no ground candidates)
    high = np.random.default_rng(0).uniform(5, 10, size=(100, 3)).astype(np.float32)
    mask = segment_ground(high, cfg)
    assert mask.shape == (100,) and mask.dtype == bool and not mask.any()
    print("  all-high cloud: PASS")

    print("  edge cases PASS")


def main():
    cfg = DemoConfig()
    print(f"DemoConfig ground_max_z={cfg.ground_max_z}  "
          f"dist_threshold={cfg.ground_distance_threshold}  "
          f"cluster_eps={cfg.cluster_eps}  "
          f"min_pts={cfg.cluster_min_points}  "
          f"height=[{cfg.cluster_min_height},{cfg.cluster_max_height}]")

    frames = list(syn.synthetic_sequence(n_frames=5, fps=10, n_people=2))
    print(f"Generated {len(frames)} frames, "
          f"first frame has {frames[0].num_points} points.")

    test_ground_segmentation(frames, cfg)
    test_clustering(frames, cfg)
    test_edge_cases(cfg)

    print("\n========== ALL TESTS PASS ==========")


if __name__ == "__main__":
    main()
