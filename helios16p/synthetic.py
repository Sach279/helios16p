"""Synthetic Helios-16P-like scans, so every module can be developed and tested
without the sensor or a recording.

Produces a flat ground plane, four room walls, and a configurable number of
"people" (vertical point blobs) that walk along straight paths over time.
Geometry roughly matches a 16-ring, 360deg, ~10 Hz mechanical LiDAR in a room.

Used by tests and by `demo.py --source synthetic`.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from .types import Frame

_ROOM = 6.0   # half-extent of the room (metres), walls at +/- _ROOM
_RNG_GROUND = 4000
_RNG_WALL = 1500


def _ground(rng: np.random.Generator) -> np.ndarray:
    xy = rng.uniform(-_ROOM, _ROOM, size=(_RNG_GROUND, 2))
    z = rng.normal(0.0, 0.01, size=(_RNG_GROUND, 1))
    return np.hstack([xy, z]).astype(np.float32)


def _walls(rng: np.random.Generator) -> np.ndarray:
    pts = []
    for axis in (0, 1):
        for sign in (-1.0, 1.0):
            n = _RNG_WALL
            along = rng.uniform(-_ROOM, _ROOM, size=n)
            height = rng.uniform(0.0, 2.6, size=n)
            block = np.zeros((n, 3), dtype=np.float32)
            block[:, axis] = sign * _ROOM
            block[:, 1 - axis] = along
            block[:, 2] = height
            pts.append(block)
    return np.vstack(pts).astype(np.float32)


def _person(center_xy: np.ndarray, rng: np.random.Generator, n: int = 350) -> np.ndarray:
    # vertical-ish cylinder ~0.4 m wide, ~1.75 m tall
    theta = rng.uniform(0, 2 * np.pi, size=n)
    r = rng.uniform(0.0, 0.22, size=n)
    z = rng.uniform(0.0, 1.75, size=n)
    x = center_xy[0] + r * np.cos(theta)
    y = center_xy[1] + r * np.sin(theta)
    return np.column_stack([x, y, z]).astype(np.float32)


def synthetic_frame(
    frame_id: int = 0,
    t: float = 0.0,
    people_xy: list[np.ndarray] | None = None,
    seed: int | None = None,
) -> Frame:
    """Build one Frame. `people_xy` is a list of (x, y) ground positions."""
    rng = np.random.default_rng(seed if seed is not None else frame_id)
    parts = [_ground(rng), _walls(rng)]
    for c in (people_xy or []):
        parts.append(_person(np.asarray(c, dtype=np.float32), rng))
    pts = np.vstack(parts).astype(np.float32)

    # crude ring assignment from elevation angle, 16 rings over [-15, 15] deg
    rng_xy = np.linalg.norm(pts[:, :2], axis=1) + 1e-6
    elev = np.degrees(np.arctan2(pts[:, 2] - 1.0, rng_xy))  # sensor ~1 m up
    ring = np.clip(((elev + 15.0) / 30.0 * 15.0).round(), 0, 15).astype(np.int16)
    intensity = rng.uniform(20, 200, size=pts.shape[0]).astype(np.float32)

    return Frame(timestamp=t, points=pts, intensity=intensity, ring=ring, frame_id=frame_id)


def synthetic_sequence(
    n_frames: int = 120,
    fps: float = 10.0,
    n_people: int = 2,
    seed: int = 0,
) -> Iterator[Frame]:
    """Yield a walking-people sequence. People move on straight lines, bouncing
    off the walls, so the tracker has consistent, predictable motion to follow."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-_ROOM + 1, _ROOM - 1, size=(n_people, 2))
    vel = rng.uniform(-1.0, 1.0, size=(n_people, 2))  # m/s
    dt = 1.0 / fps
    for i in range(n_frames):
        pos = pos + vel * dt
        # bounce
        for k in range(n_people):
            for ax in (0, 1):
                if abs(pos[k, ax]) > _ROOM - 0.5:
                    pos[k, ax] = np.clip(pos[k, ax], -(_ROOM - 0.5), _ROOM - 0.5)
                    vel[k, ax] *= -1.0
        yield synthetic_frame(
            frame_id=i, t=i * dt, people_xy=[pos[k].copy() for k in range(n_people)]
        )
