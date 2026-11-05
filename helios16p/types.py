"""Core data contracts shared across sources, processing and visualization.

THIS FILE IS THE INTEGRATION CONTRACT. Do not change field names/shapes without
coordinating every module that builds against these definitions.

Conventions
-----------
- All geometry is in the LiDAR sensor frame, metres, right-handed.
  x forward, y left, z up (RoboSense convention).
- Point arrays are float32, shape (N, 3). Per-point attributes are length-N.
- Angles in this package's *Python* code are radians unless suffixed _deg.
  (The raw wire protocol uses centi-degrees; that stays inside udp_source.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Frame:
    """One full LiDAR revolution.

    Attributes
    ----------
    timestamp : float
        Seconds (POSIX) of the frame (first point).
    points : np.ndarray
        (N, 3) float32, metres, sensor frame. NaN points already removed.
    intensity : np.ndarray
        (N,) float32, reflectivity 0..255.
    ring : np.ndarray | None
        (N,) int16, beam index by elevation, 0 = lowest beam (None if unavailable).
    frame_id : int
        Monotonic sequence index assigned by the source.
    """

    timestamp: float
    points: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray | None = None
    frame_id: int = 0

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])


@dataclass
class Detection:
    """A clustered object (candidate person/obstacle) in one frame."""

    centroid: np.ndarray            # (3,) float32
    bbox_min: np.ndarray            # (3,) float32, axis-aligned box corner
    bbox_max: np.ndarray            # (3,) float32
    num_points: int
    point_indices: np.ndarray       # (k,) int, indices into Frame.points
    track_id: int | None = None     # filled in by the tracker


@dataclass
class Track:
    """A persistent target estimated across frames (Kalman filter output)."""

    track_id: int
    position: np.ndarray            # (3,) float32, estimated position
    velocity: np.ndarray            # (3,) float32, estimated velocity (m/s)
    covariance: np.ndarray          # (6, 6) float32, state covariance [pos, vel]
    age: int                        # frames since the track was created
    hits: int                       # number of associated detections
    time_since_update: int          # frames since last associated detection
    confirmed: bool = False         # promoted past the tentative stage
    history: list[np.ndarray] = field(default_factory=list)  # past positions, for the trail

    @property
    def speed(self) -> float:
        """Horizontal speed in m/s; the vertical component of a cluster centroid is mostly noise."""
        return float(np.hypot(float(self.velocity[0]), float(self.velocity[1])))


@dataclass
class FrameAnnotations:
    """Everything the viewer needs for one frame: the cloud + perception results."""

    frame: Frame
    ground_mask: np.ndarray | None = None     # (N,) bool, True = ground point
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    control: dict[str, Any] | None = None     # e.g. {"target_id", "azimuth_cmd_deg", "error_deg"}
    stats: dict[str, Any] = field(default_factory=dict)  # fps, n_points, n_clusters, ...
