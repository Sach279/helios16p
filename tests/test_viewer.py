"""Tests for helios16p.viz.open3d_viewer pure rendering helpers.

Run with:
    python tests/test_viewer.py

The pure-function checks do NOT require a display. The windowed smoke test is
attempted inside a try/except and skipped gracefully when there is no display.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import open3d as o3d

# Make sure the package root is on sys.path when run as a script.
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helios16p.synthetic import synthetic_frame
from helios16p.types import Detection, Frame, FrameAnnotations, Track
from helios16p.viz.open3d_viewer import (
    make_bbox_lineset,
    make_control_ray,
    make_ground_grid,
    make_marker_sphere,
    make_point_cloud,
    make_trail_lineset,
    velocity_arrow,
)


# ---------------------------------------------------------------------------
# Build a synthetic FrameAnnotations
# ---------------------------------------------------------------------------

def _build_annotation() -> FrameAnnotations:
    """Construct a FrameAnnotations with 2 detections and 2 tracks by hand."""
    frame = synthetic_frame(0, 0.0, people_xy=[[2, 1], [-3, 2]])

    N = frame.num_points
    ground_mask = np.zeros(N, dtype=bool)
    # Mark roughly the first 4000 points (ground layer) as ground
    ground_mask[:min(4000, N)] = True

    # Detection 0 — points around [2, 1, 0.9]
    det0 = Detection(
        centroid=np.array([2.0, 1.0, 0.9], dtype=np.float32),
        bbox_min=np.array([1.8, 0.8, 0.0], dtype=np.float32),
        bbox_max=np.array([2.2, 1.2, 1.8], dtype=np.float32),
        num_points=50,
        point_indices=np.arange(50, dtype=int),
        track_id=1,
    )

    # Detection 1 — points around [-3, 2, 0.9]
    det1 = Detection(
        centroid=np.array([-3.0, 2.0, 0.9], dtype=np.float32),
        bbox_min=np.array([-3.2, 1.8, 0.0], dtype=np.float32),
        bbox_max=np.array([-2.8, 2.2, 1.8], dtype=np.float32),
        num_points=45,
        point_indices=np.arange(50, 95, dtype=int),
        track_id=2,
    )

    # Track 0 — a few history points, non-zero velocity
    history_0 = [
        np.array([1.7, 0.9, 0.9], dtype=np.float32),
        np.array([1.85, 0.95, 0.9], dtype=np.float32),
        np.array([2.0, 1.0, 0.9], dtype=np.float32),
    ]
    track0 = Track(
        track_id=1,
        position=np.array([2.0, 1.0, 0.9], dtype=np.float32),
        velocity=np.array([0.5, 0.1, 0.0], dtype=np.float32),
        covariance=np.eye(6, dtype=np.float32) * 0.01,
        age=5,
        hits=4,
        time_since_update=0,
        confirmed=True,
        history=history_0,
    )

    # Track 1 — also a few history points
    history_1 = [
        np.array([-3.3, 1.8, 0.9], dtype=np.float32),
        np.array([-3.15, 1.9, 0.9], dtype=np.float32),
        np.array([-3.0, 2.0, 0.9], dtype=np.float32),
    ]
    track1 = Track(
        track_id=2,
        position=np.array([-3.0, 2.0, 0.9], dtype=np.float32),
        velocity=np.array([-0.3, 0.2, 0.0], dtype=np.float32),
        covariance=np.eye(6, dtype=np.float32) * 0.02,
        age=4,
        hits=3,
        time_since_update=0,
        confirmed=True,
        history=history_1,
    )

    control = {
        "target_id": 1,
        "gimbal_azimuth_deg": 27.0,
        "azimuth_cmd_deg": 27.0,
        "error_deg": 0.5,
        "aim_point": None,
    }

    stats = {
        "frame_id": 0,
        "n_points": N,
        "n_ground": int(ground_mask.sum()),
        "n_clusters": 2,
        "n_tracks": 2,
        "proc_ms": 12.3,
        "proc_fps": 81.3,
    }

    return FrameAnnotations(
        frame=frame,
        ground_mask=ground_mask,
        detections=[det0, det1],
        tracks=[track0, track1],
        control=control,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------

def test_make_point_cloud_modes():
    ann = _build_annotation()
    frame = ann.frame
    N = frame.num_points

    for mode in ("height", "intensity", "ring"):
        pcd = make_point_cloud(frame, ann, mode=mode, show_ground=True)
        assert isinstance(pcd, o3d.geometry.PointCloud), f"Expected PointCloud for mode {mode!r}"
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors)
        assert pts.shape[0] == N, f"mode={mode}: expected {N} points, got {pts.shape[0]}"
        assert cols.shape == (N, 3), f"mode={mode}: colour shape wrong: {cols.shape}"
        assert cols.min() >= 0.0 and cols.max() <= 1.0, f"mode={mode}: colours out of [0,1]"
    print("  [OK] make_point_cloud — all 3 colour modes, show_ground=True")


def test_make_point_cloud_hide_ground():
    ann = _build_annotation()
    frame = ann.frame
    n_ground = int(ann.ground_mask.sum())
    N = frame.num_points

    pcd = make_point_cloud(frame, ann, mode="height", show_ground=False)
    pts = np.asarray(pcd.points)
    expected = N - n_ground
    assert pts.shape[0] == expected, (
        f"show_ground=False: expected {expected} non-ground points, got {pts.shape[0]}"
    )
    print(f"  [OK] make_point_cloud — show_ground=False, removed {n_ground} ground points")


def test_make_bbox_lineset():
    ann = _build_annotation()
    for i, det in enumerate(ann.detections):
        ls = make_bbox_lineset(det)
        assert isinstance(ls, o3d.geometry.LineSet), f"Detection {i}: expected LineSet"
        lines = np.asarray(ls.lines)
        assert lines.shape == (12, 2), (
            f"Detection {i}: AABB should have 12 edges, got {lines.shape}"
        )
        pts = np.asarray(ls.points)
        assert pts.shape == (8, 3), f"Detection {i}: AABB should have 8 corners"
    print("  [OK] make_bbox_lineset — 12 edges, 8 corners for each detection")


def test_make_trail_lineset():
    ann = _build_annotation()
    for track in ann.tracks:
        ls = make_trail_lineset(track)
        assert isinstance(ls, o3d.geometry.LineSet), "Expected LineSet for trail"
        pts = np.asarray(ls.points)
        lines = np.asarray(ls.lines)
        # history has 3 points + current position = 4 total, 3 segments
        expected_pts = len(track.history) + 1
        expected_lines = expected_pts - 1
        assert pts.shape[0] == expected_pts, (
            f"Trail: expected {expected_pts} pts, got {pts.shape[0]}"
        )
        assert lines.shape == (expected_lines, 2), (
            f"Trail: expected {expected_lines} lines, got {lines.shape}"
        )
    print("  [OK] make_trail_lineset — correct segment counts for both tracks")


def test_velocity_arrow():
    ann = _build_annotation()
    for track in ann.tracks:
        arr = velocity_arrow(track, scale=0.5)
        assert isinstance(arr, o3d.geometry.LineSet), "Expected LineSet for arrow"
        pts = np.asarray(arr.points)
        lines = np.asarray(arr.lines)
        assert pts.shape == (2, 3), f"Arrow: expected 2 points, got {pts.shape}"
        assert lines.shape == (1, 2), f"Arrow: expected 1 line, got {lines.shape}"
        # Verify tip is origin + velocity * 0.5
        expected_tip = track.position.astype(np.float64) + track.velocity.astype(np.float64) * 0.5
        np.testing.assert_allclose(pts[1], expected_tip, atol=1e-6)
    print("  [OK] velocity_arrow — correct tip position for both tracks")


def test_make_trail_lineset_short_history():
    """Track with only 1 history point (< 2 total including position) gives empty trail."""
    track = Track(
        track_id=99,
        position=np.array([1.0, 0.0, 0.5], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        covariance=np.eye(6, dtype=np.float32),
        age=1,
        hits=1,
        time_since_update=0,
        confirmed=False,
        history=[],  # no history at all -> pts = [position] -> len < 2
    )
    ls = make_trail_lineset(track)
    assert isinstance(ls, o3d.geometry.LineSet)
    assert len(ls.lines) == 0, "Empty history: trail should have 0 lines"
    print("  [OK] make_trail_lineset — empty history produces empty trail")


def test_make_ground_grid():
    grid = make_ground_grid(size=6.0, step=1.0)
    assert isinstance(grid, o3d.geometry.LineSet)
    pts = np.asarray(grid.points)
    lines = np.asarray(grid.lines)
    assert pts.shape[1] == 3
    assert lines.shape[1] == 2
    assert len(lines) > 0
    # All z coords should be 0
    np.testing.assert_allclose(pts[:, 2], 0.0, atol=1e-9)
    print(f"  [OK] make_ground_grid — {len(lines)} grid lines, all z=0")


def test_make_control_ray_azimuth():
    control = {"gimbal_azimuth_deg": 45.0}
    ls = make_control_ray(control)
    assert ls is not None, "Control ray should be non-None for gimbal_azimuth_deg"
    assert isinstance(ls, o3d.geometry.LineSet)
    pts = np.asarray(ls.points)
    assert pts.shape == (2, 3)
    print("  [OK] make_control_ray — azimuth ray has 2 points")


def test_make_control_ray_aim_point():
    control = {"aim_point": [3.0, 2.0, 0.5]}
    ls = make_control_ray(control)
    assert ls is not None
    pts = np.asarray(ls.points)
    np.testing.assert_allclose(pts[1], [3.0, 2.0, 0.5], atol=1e-6)
    print("  [OK] make_control_ray — aim_point tip matches control dict")


def test_make_control_ray_no_keys():
    ls = make_control_ray({"target_id": 1})
    assert ls is None, "No azimuth / aim_point key should return None"
    print("  [OK] make_control_ray — returns None when no usable key")


def test_point_cloud_no_ring():
    """make_point_cloud with ring=None in ring mode should not crash."""
    frame = synthetic_frame(0, 0.0, people_xy=[[1, 1]])
    # Remove ring info
    frame_no_ring = Frame(
        timestamp=frame.timestamp,
        points=frame.points,
        intensity=frame.intensity,
        ring=None,
        frame_id=frame.frame_id,
    )
    ann = FrameAnnotations(frame=frame_no_ring)
    pcd = make_point_cloud(frame_no_ring, ann, mode="ring", show_ground=True)
    assert isinstance(pcd, o3d.geometry.PointCloud)
    assert len(pcd.points) > 0
    print("  [OK] make_point_cloud — ring=None handled in 'ring' mode")


# ---------------------------------------------------------------------------
# Windowed smoke test (skipped gracefully when headless / no GPU)
# ---------------------------------------------------------------------------

_SMOKE_SCRIPT = """
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__ if '__file__' in dir() else sys.argv[0]))))

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

app = gui.Application.instance
app.initialize()
w = app.create_window("smoke-test", 640, 400)

sc_widget = gui.SceneWidget()
sc_widget.scene = rendering.Open3DScene(w.renderer)
w.add_child(sc_widget)

def _close():
    return True
w.set_on_close(_close)

def _quit():
    w.close()
app.post_to_main_thread(w, _quit)
app.run()
print("SMOKE_OK")
"""


def smoke_test_window():
    """Attempt to open a window in a subprocess; treat segfault or exception as 'headless: skipped'."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", _SMOKE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=15,
    )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode == 0 and "SMOKE_OK" in stdout:
        print("  [OK] windowed smoke test — window opened and closed cleanly")
    else:
        # segfault returns -11, other failures return non-zero
        msg_parts = []
        if result.returncode < 0:
            import signal
            try:
                sig_name = signal.Signals(-result.returncode).name
            except Exception:
                sig_name = str(-result.returncode)
            msg_parts.append(f"signal {sig_name}")
        else:
            msg_parts.append(f"exit {result.returncode}")
        if stderr:
            # show first line of stderr only
            msg_parts.append(stderr.split("\n")[0])
        print(f"headless: window smoke skipped ({'; '.join(msg_parts)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== test_viewer.py ===")
    print()

    pure_tests = [
        test_make_point_cloud_modes,
        test_make_point_cloud_hide_ground,
        test_make_bbox_lineset,
        test_make_trail_lineset,
        test_velocity_arrow,
        test_make_trail_lineset_short_history,
        test_make_ground_grid,
        test_make_control_ray_azimuth,
        test_make_control_ray_aim_point,
        test_make_control_ray_no_keys,
        test_point_cloud_no_ring,
    ]

    failures = []
    for fn in pure_tests:
        try:
            fn()
        except Exception:
            failures.append(fn.__name__)
            traceback.print_exc()

    print()
    print("--- windowed smoke test ---")
    try:
        smoke_test_window()
    except Exception as e:
        print(f"headless: window smoke skipped (unexpected exception: {e})")

    print()
    if failures:
        print(f"FAIL — {len(failures)} pure-function test(s) failed: {failures}")
        sys.exit(1)
    else:
        print("PASS")


if __name__ == "__main__":
    main()
