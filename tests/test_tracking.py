"""Smoke tests for MultiObjectTracker and VirtualGimbal.

Run directly:  python tests/test_tracking.py
(No external test framework required.)

Test scenario
-------------
Object A: starts at (3, 1, 0.9) m, constant velocity (0.5, 0.2, 0.0) m/s.
Object B: starts at (-4, 2, 1.1) m, constant velocity (-0.3, 0.1, 0.0) m/s.
Both tracked for 40 steps at dt = 0.1 s with Gaussian centroid noise (σ=0.05 m).

Assertions
----------
1. After enough hits a CONFIRMED track exists with a STABLE track_id for Object A.
2. Estimated velocity converges to within 0.2 m/s of the injected velocity.
3. detection.track_id is stamped in-place on associated detections.
4. VirtualGimbal error_deg decreases (converges) over the first ~10 steps.
5. target_id stays sticky while the target is alive.
"""

import sys
import os

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np

from helios16p.config import DemoConfig
from helios16p.types import Detection, Track
from helios16p.processing.tracking import MultiObjectTracker
from helios16p.control.gimbal import VirtualGimbal

RNG = np.random.default_rng(42)
NOISE_STD = 0.05   # m


def make_detection(centroid: np.ndarray, add_noise: bool = True) -> Detection:
    """Build a Detection around a known centroid (±0.3 m half-box)."""
    if add_noise:
        c = centroid + RNG.normal(0, NOISE_STD, size=3).astype(np.float32)
    else:
        c = centroid.copy()
    half = np.array([0.3, 0.3, 0.5], dtype=np.float32)
    return Detection(
        centroid=c.astype(np.float32),
        bbox_min=(c - half).astype(np.float32),
        bbox_max=(c + half).astype(np.float32),
        num_points=30,
        point_indices=np.arange(30, dtype=int),
        track_id=None,
    )


def test_tracker_and_gimbal() -> None:
    cfg = DemoConfig()
    tracker = MultiObjectTracker(cfg)
    gimbal = VirtualGimbal(cfg)

    dt = 0.1   # seconds per step
    n_steps = 40

    # Ground-truth kinematics
    pos_A0 = np.array([3.0, 1.0, 0.9], dtype=np.float32)
    vel_A = np.array([0.5, 0.2, 0.0], dtype=np.float32)

    pos_B0 = np.array([-4.0, 2.0, 1.1], dtype=np.float32)
    vel_B = np.array([-0.3, 0.1, 0.0], dtype=np.float32)

    # ── book-keeping ─────────────────────────────────────────────────────────
    stable_id_A: int | None = None       # track_id we expect to stay constant
    confirmed_ever = False
    last_confirmed_tracks: list[Track] = []

    gimbal_errors: list[float] = []
    target_ids_seen: list[int | None] = []

    for step in range(n_steps):
        t = step * dt
        true_A = pos_A0 + vel_A * t
        true_B = pos_B0 + vel_B * t

        det_A = make_detection(true_A)
        det_B = make_detection(true_B)
        detections = [det_A, det_B]

        tracks = tracker.update(detections, dt)

        # ── assertion 3 (rolling): stamped track_id ───────────────────────
        # After the tracker has had time to create tracks, associated
        # detections must have their track_id set.
        if step >= cfg.track_min_hits:
            for det in detections:
                assert det.track_id is not None, (
                    f"Step {step}: detection.track_id not stamped (got None)"
                )

        # ── track for Object A (the one near pos_A0 at step 0) ───────────
        confirmed_A: Track | None = None
        for trk in tracks:
            if trk.confirmed:
                # Identify the track that's closest to true_A's ground truth
                dist = np.linalg.norm(trk.position - true_A)
                # Object B is far away (>6 m initially), so <2 m is unambiguous
                if dist < 2.0:
                    confirmed_A = trk
                    break

        if confirmed_A is not None:
            confirmed_ever = True
            if stable_id_A is None:
                stable_id_A = confirmed_A.track_id
            else:
                assert confirmed_A.track_id == stable_id_A, (
                    f"Step {step}: track_id changed from {stable_id_A} "
                    f"to {confirmed_A.track_id} — not stable!"
                )
            last_confirmed_tracks = tracks  # remember for velocity check

        # ── gimbal ───────────────────────────────────────────────────────
        ctrl = gimbal.update(tracks)
        if ctrl is not None:
            gimbal_errors.append(abs(ctrl["error_deg"]))
            target_ids_seen.append(ctrl["target_id"])

    # ── assertion 1: a confirmed track was found ─────────────────────────────
    assert confirmed_ever, "No confirmed track for Object A was ever found"
    print(f"  Confirmed track_id for Object A: {stable_id_A}  ✓")

    # ── assertion 2: velocity convergence ────────────────────────────────────
    assert last_confirmed_tracks, "No confirmed tracks available for velocity check"
    vel_estimated: np.ndarray | None = None
    for trk in last_confirmed_tracks:
        if trk.track_id == stable_id_A:
            vel_estimated = trk.velocity
            break
    assert vel_estimated is not None, "Could not find final velocity for Object A"
    vel_error = float(np.linalg.norm(vel_estimated - vel_A))
    print(f"  Injected velocity A:   {vel_A}")
    print(f"  Estimated velocity A:  {vel_estimated}")
    print(f"  Velocity error (m/s):  {vel_error:.4f}")
    assert vel_error < 0.2, (
        f"Velocity error {vel_error:.4f} m/s exceeds 0.2 m/s threshold"
    )
    print("  Velocity convergence  ✓")

    # ── assertion 4: gimbal converges (errors decrease in early steps) ────────
    assert len(gimbal_errors) >= 10, (
        f"Not enough gimbal steps to check convergence ({len(gimbal_errors)})"
    )
    # Compare mean error in the first 5 steps vs next 5 steps
    early_mean = float(np.mean(gimbal_errors[:5]))
    later_mean = float(np.mean(gimbal_errors[5:10]))
    print(f"  Gimbal error (first 5 steps mean): {early_mean:.3f} deg")
    print(f"  Gimbal error (steps 5–10 mean):    {later_mean:.3f} deg")
    assert later_mean <= early_mean, (
        f"Gimbal not converging: early mean {early_mean:.3f} < later mean {later_mean:.3f}"
    )
    print("  Gimbal convergence    ✓")

    # ── assertion 5: target_id sticky ────────────────────────────────────────
    # Once a confirmed target is picked, the same id should persist for many steps.
    non_none = [tid for tid in target_ids_seen if tid is not None]
    assert non_none, "Gimbal never acquired a target"
    # Find the most-common target id — it should dominate
    from collections import Counter
    counts = Counter(non_none)
    dominant_id, dominant_count = counts.most_common(1)[0]
    dominance_ratio = dominant_count / len(non_none)
    print(f"  Target sticky: id={dominant_id}, held {dominant_count}/{len(non_none)} steps "
          f"({dominance_ratio:.0%})")
    assert dominance_ratio >= 0.8, (
        f"Target not sticky enough: dominant id held only {dominance_ratio:.0%} of steps"
    )
    print("  Target sticky         ✓")


if __name__ == "__main__":
    print("Running tracking + gimbal tests …")
    test_tracker_and_gimbal()
    print("\nPASS")
