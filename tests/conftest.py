"""Shared fixtures: synthetic frames and the default demo configuration."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helios16p import synthetic as syn  # noqa: E402
from helios16p.config import DemoConfig  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture
def cfg() -> DemoConfig:
    return DemoConfig()


@pytest.fixture
def frames():
    return list(syn.synthetic_sequence(n_frames=6, fps=10.0, n_people=2))


@pytest.fixture
def difop_packet() -> bytes:
    return (DATA / "difop_sample.bin").read_bytes()


@pytest.fixture
def mini_recording() -> Path:
    return DATA / "mini_walk.npz"
