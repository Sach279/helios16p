"""Virtual gimbal controller.

Tracks the "most interesting" confirmed target and slews a simulated gimbal
azimuth toward it with a proportional gain and a hard per-step slew limit so
convergence is visually observable.

Coordinate convention (from types.py):  x forward, y left, z up.
  → azimuth = atan2(y, x),  0° = forward, +CCW when viewed from above.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

from ..config import DemoConfig
from ..types import Track

# Gimbal dynamics
_SLEW_LIMIT_DEG = 5.0        # maximum degrees per update step
_P_GAIN = 0.3                # proportional gain (fraction of error applied per step)
_AIM_RADIUS_M = 5.0          # radius of the aim-point sphere (metres)


def _angle_diff(a: float, b: float) -> float:
    """Signed shortest-path difference  a − b  in degrees, in (−180, 180]."""
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


class VirtualGimbal:
    """Simulated single-axis (azimuth only) gimbal controller.

    Parameters
    ----------
    cfg : DemoConfig
        Configuration (track_* fields used upstream; here we only need the
        existence of cfg to match the pipeline contract).
    """

    def __init__(self, cfg: DemoConfig) -> None:
        self.cfg = cfg
        self._gimbal_az: float = 0.0    # current simulated gimbal azimuth (deg)
        self._target_id: int | None = None

    # ── main entry point ──────────────────────────────────────────────────────
    def update(self, tracks: List[Track]) -> dict | None:
        """Select a target, slew the gimbal, return the control dict."""
        if not tracks:
            self._target_id = None
            return None

        # 1. Pick target: prefer a MOVING confirmed track (a walking person),
        #    keep it sticky while it stays a good choice, and fall back to the
        #    longest-lived confirmed track when nothing is moving.
        live = {t.track_id: t for t in tracks}
        thr = getattr(self.cfg, "moving_speed_threshold", 0.3)
        moving = [t for t in tracks if t.confirmed and t.speed >= thr]

        cur = live.get(self._target_id)
        if cur is not None and (not moving or cur in moving):
            target = cur                                   # keep sticky target
        elif moving:
            target = max(moving, key=lambda t: t.speed)    # fastest mover
        else:
            confirmed = [t for t in tracks if t.confirmed] or tracks
            target = max(confirmed, key=lambda t: (t.confirmed, t.age))
        self._target_id = target.track_id

        # 2. Compute target azimuth from position
        pos = target.position.astype(np.float64)
        target_az = math.degrees(math.atan2(float(pos[1]), float(pos[0])))

        # 3. Slew gimbal toward target with P-gain + slew limit
        error = _angle_diff(target_az, self._gimbal_az)
        raw_cmd = _P_GAIN * error
        # clamp to slew limit
        slew = float(np.clip(raw_cmd, -_SLEW_LIMIT_DEG, _SLEW_LIMIT_DEG))
        self._gimbal_az = (self._gimbal_az + slew) % 360.0

        # 4. Compute aim-point: a point on a sphere of fixed radius along
        #    the current gimbal azimuth, at the same z as the target
        az_rad = math.radians(self._gimbal_az)
        aim_point = np.array([
            _AIM_RADIUS_M * math.cos(az_rad),
            _AIM_RADIUS_M * math.sin(az_rad),
            float(pos[2]),
        ], dtype=np.float32)

        # Remaining error after slew
        remaining_error = _angle_diff(target_az, self._gimbal_az)

        return {
            "target_id": self._target_id,
            "target_azimuth_deg": float(target_az),
            "gimbal_azimuth_deg": float(self._gimbal_az),
            "error_deg": float(remaining_error),
            "target_position": pos.astype(np.float32),
            "aim_point": aim_point,
            "target_speed": float(target.speed),
            "target_moving": bool(target.speed >= thr),
        }
