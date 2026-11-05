"""Multi-object tracker with constant-velocity Kalman filters.

State vector: [x, y, z, vx, vy, vz]  (6-DOF, metric, sensor frame)
Measurement:  [x, y, z]  (centroid only)

Association uses scipy.optimize.linear_sum_assignment (Hungarian) with a
hard distance gate of cfg.track_gate_distance.  Unmatched detections spawn
tentative tracks; tracks are confirmed after cfg.track_min_hits hits; tracks
older than cfg.track_max_age frames without a detection are deleted.
"""
from __future__ import annotations

import itertools
from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import DemoConfig
from ..types import Detection, Track

# ── Kalman matrices (constant-velocity model) ─────────────────────────────────

def _build_F(dt: float) -> np.ndarray:
    """6×6 state-transition matrix for Δt = dt."""
    F = np.eye(6, dtype=np.float64)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F

# Measurement matrix H: extract [x, y, z] from state
_H = np.zeros((3, 6), dtype=np.float64)
_H[0, 0] = 1.0
_H[1, 1] = 1.0
_H[2, 2] = 1.0

def _build_Q(dt: float, sigma_a: float = 1.0) -> np.ndarray:
    """Process noise covariance (constant-acceleration disturbance model).

    sigma_a: acceleration noise standard deviation (m/s²).
    """
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    q = sigma_a ** 2
    Q = np.zeros((6, 6), dtype=np.float64)
    # position–position block
    Q[0, 0] = Q[1, 1] = Q[2, 2] = dt4 / 4.0 * q
    # velocity–velocity block
    Q[3, 3] = Q[4, 4] = Q[5, 5] = dt2 * q
    # cross terms
    Q[0, 3] = Q[3, 0] = dt3 / 2.0 * q
    Q[1, 4] = Q[4, 1] = dt3 / 2.0 * q
    Q[2, 5] = Q[5, 2] = dt3 / 2.0 * q
    return Q

# Measurement noise covariance (centroid uncertainty ~0.1 m std)
_R = np.eye(3, dtype=np.float64) * (0.1 ** 2)

# Initial state covariance
_P0_diag = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0], dtype=np.float64)


# ── Internal track representation ────────────────────────────────────────────

class _KFTrack:
    """One Kalman-filter track instance."""

    _id_counter = itertools.count(1)  # monotonically increasing from 1

    def __init__(self, detection: Detection, track_id: int) -> None:
        self.track_id = track_id
        centroid = detection.centroid.astype(np.float64)
        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6, dtype=np.float64)
        self.x[:3] = centroid
        self.P = np.diag(_P0_diag.copy())
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.confirmed = False
        self.history: List[np.ndarray] = [centroid[:3].copy()]

    # -- Kalman predict -------------------------------------------------------
    def predict(self, dt: float) -> None:
        F = _build_F(dt)
        Q = _build_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age += 1
        self.time_since_update += 1

    # -- Kalman update --------------------------------------------------------
    def update(self, detection: Detection) -> None:
        z = detection.centroid.astype(np.float64)
        y = z - _H @ self.x                          # innovation
        S = _H @ self.P @ _H.T + _R                  # innovation covariance
        K = self.P @ _H.T @ np.linalg.inv(S)         # Kalman gain
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ _H) @ self.P
        self.hits += 1
        self.time_since_update = 0
        pos = self.x[:3].copy()
        self.history.append(pos)
        if len(self.history) > 50:
            self.history = self.history[-50:]

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].astype(np.float32)

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].astype(np.float32)

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            position=self.position,
            velocity=self.velocity,
            covariance=self.P.astype(np.float32),
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            confirmed=self.confirmed,
            history=[p.astype(np.float32) for p in self.history],
        )


# ── Public class ──────────────────────────────────────────────────────────────

class MultiObjectTracker:
    """Frame-to-frame multi-object tracker.

    Parameters
    ----------
    cfg : DemoConfig
        Uses track_max_age, track_min_hits, track_gate_distance.
    """

    def __init__(self, cfg: DemoConfig) -> None:
        self.cfg = cfg
        self._tracks: List[_KFTrack] = []
        self._next_id: int = 1

    # ── main entry point ─────────────────────────────────────────────────────
    def update(self, detections: List[Detection], dt: float) -> List[Track]:
        """Predict existing tracks, associate detections, return live tracks."""
        dt = max(dt, 1e-6)

        # 1. Predict all tracks forward by dt
        for trk in self._tracks:
            trk.predict(dt)

        # 2. Build cost matrix (Euclidean centroid distance)
        n_trk = len(self._tracks)
        n_det = len(detections)

        matched_trk_ids: set[int] = set()
        matched_det_ids: set[int] = set()

        if n_trk > 0 and n_det > 0:
            cost = np.full((n_trk, n_det), fill_value=1e9, dtype=np.float64)
            for i, trk in enumerate(self._tracks):
                for j, det in enumerate(detections):
                    d = np.linalg.norm(trk.x[:3] - det.centroid.astype(np.float64))
                    cost[i, j] = d

            row_ind, col_ind = linear_sum_assignment(cost)

            gate = self.cfg.track_gate_distance
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= gate:
                    self._tracks[r].update(detections[c])
                    detections[c].track_id = self._tracks[r].track_id  # stamp in place
                    matched_trk_ids.add(r)
                    matched_det_ids.add(c)

        # 3. Promote confirmed tracks
        for trk in self._tracks:
            if trk.hits >= self.cfg.track_min_hits:
                trk.confirmed = True

        # 4. Spawn new (tentative) tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_det_ids:
                new_trk = _KFTrack(det, self._next_id)
                self._next_id += 1
                if new_trk.hits >= self.cfg.track_min_hits:
                    new_trk.confirmed = True
                self._tracks.append(new_trk)

        # 5. Delete stale tracks
        self._tracks = [
            t for t in self._tracks
            if t.time_since_update <= self.cfg.track_max_age
        ]

        # 6. Return public Track objects for all live tracks
        return [t.to_track() for t in self._tracks]
