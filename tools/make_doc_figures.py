#!/usr/bin/env python3
"""Regenerate the documentation figures with ``helios16p-snapshot`` (listed in docs/README.md).

    python tools/make_doc_figures.py [--rec-dir recordings]

The bundled clip is always available; the dense room and the corridor station
are rendered only when their recordings are found in --rec-dir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helios16p.apps.snapshot import main as snapshot  # noqa: E402
from helios16p.cli import bundled_clip  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "docs" / "figures"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rec-dir", default="recordings")
    args = ap.parse_args()
    rd = Path(args.rec_dir)
    clip = str(bundled_clip())
    snapshot(["--rec", clip, "--out", str(FIG / "walk_frame.png"), "--hud"])
    snapshot(["--rec", clip, "--out", str(FIG / "walk_rings.png"), "--color", "ring"])
    room = sorted(rd.glob("room_static_*.npz"))
    if room:
        snapshot(["--rec", str(room[0]), "--merge", "60", "--out", str(FIG / "room_dense.png"), "--point-size", "2"])
    corridor = sorted(rd.glob("corridor1_*.npz"))
    if corridor:
        snapshot(["--rec", str(corridor[0]), "--merge", "100", "--out", str(FIG / "corridor_dense.png"), "--point-size", "1.2"])
        snapshot(["--rec", str(corridor[0]), "--merge", "30", "--compare-calibration", "--view", "top",
                  "--out", str(FIG / "calibration_before_after.png"), "--size", "2000x1000", "--point-size", "2"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
