"""The processing pipeline: turns a raw `Frame` into `FrameAnnotations`.

This is the integration point. It wires together the modules that the
perception/control modules implement, against these exact signatures:

    processing.ground.segment_ground(points: (N,3) f32, cfg) -> (N,) bool   # True = ground
    processing.clustering.cluster_objects(frame, ground_mask, cfg) -> list[Detection]
    processing.tracking.MultiObjectTracker(cfg).update(detections, dt) -> list[Track]
        (also stamps detection.track_id in place)
    control.gimbal.VirtualGimbal(cfg).update(tracks) -> dict | None

If you change a signature, change it here and in the module together.
"""
from __future__ import annotations

import time

from .config import DemoConfig
from .types import Frame, FrameAnnotations


class ProcessingPipeline:
    def __init__(self, cfg: DemoConfig, enable_control: bool = True) -> None:
        # Imported lazily so a partially-built tree still lets you import the package.
        from .processing.tracking import MultiObjectTracker

        self.cfg = cfg
        self.tracker = MultiObjectTracker(cfg)
        self._enable_control = enable_control
        self._gimbal = None
        if enable_control:
            from .control.gimbal import VirtualGimbal

            self._gimbal = VirtualGimbal(cfg)
        self._last_ts: float | None = None
        self._ema_fps: float | None = None

    def warmup(self) -> float:
        """Import the heavy libraries (Open3D takes seconds) and run one synthetic frame; returns the seconds taken.

        Windows call this on the main thread before starting the worker: importing
        Open3D from a background thread while a Qt event loop runs can stall for good."""
        from .synthetic import synthetic_frame
        t = time.perf_counter()
        try:
            import open3d  # noqa: F401
        except ImportError:
            pass
        state = (self._last_ts, self._ema_fps)
        self.process(synthetic_frame(0))
        self._last_ts, self._ema_fps = state
        self.tracker = type(self.tracker)(self.cfg)  # forget the synthetic tracks
        if self._gimbal is not None:
            self._gimbal = type(self._gimbal)(self.cfg)
        return time.perf_counter() - t

    def process(self, frame: Frame) -> FrameAnnotations:
        from .processing.clustering import cluster_objects
        from .processing.ground import segment_ground

        wall = time.perf_counter()
        dt = self.cfg.frame_rate and (1.0 / self.cfg.frame_rate)
        if self._last_ts is not None and frame.timestamp > self._last_ts:
            dt = frame.timestamp - self._last_ts
        self._last_ts = frame.timestamp

        ground_mask = segment_ground(frame.points, self.cfg)
        detections = cluster_objects(frame, ground_mask, self.cfg)
        tracks = self.tracker.update(detections, dt)
        control = self._gimbal.update(tracks) if self._gimbal else None

        elapsed = time.perf_counter() - wall
        inst_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self._ema_fps = inst_fps if self._ema_fps is None else 0.9 * self._ema_fps + 0.1 * inst_fps

        stats = {
            "frame_id": frame.frame_id,
            "n_points": frame.num_points,
            "n_ground": int(ground_mask.sum()) if ground_mask is not None else 0,
            "n_clusters": len(detections),
            "n_tracks": len(tracks),
            "proc_ms": elapsed * 1e3,
            "proc_fps": self._ema_fps,
        }
        return FrameAnnotations(
            frame=frame,
            ground_mask=ground_mask,
            detections=detections,
            tracks=tracks,
            control=control,
            stats=stats,
        )
