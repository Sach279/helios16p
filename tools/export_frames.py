"""export_frames.py — export MCAP bag frames to a portable .npz archive.

Usage
-----
    python tools/export_frames.py --bag /path/to/bag_dir --out frames.npz
    python tools/export_frames.py --bag /path/to/bag_dir --out frames.npz --max-frames 50

The resulting .npz contains:
    timestamps          : (N,)       float64, seconds
    frame_{i}_points    : (Ni, 3)    float32, metres
    frame_{i}_intensity : (Ni,)      float32, reflectivity 0..255
    frame_{i}_ring      : (Ni,)      int16   (omitted if ring is None)

Load example::

    import numpy as np
    npz = np.load("frames.npz")
    timestamps = npz["timestamps"]
    pts = npz["frame_0_points"]   # (N, 3)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helios16p.sources.mcap_source import McapSource


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export MCAP bag PointCloud2 frames to a .npz archive.",
    )
    p.add_argument("--bag", required=True, metavar="PATH", help="Bag directory or .mcap file")
    p.add_argument("--out", required=True, metavar="FILE", help="Output .npz path")
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of frames to export (default: all)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    bag_path = Path(args.bag)
    out_path = Path(args.out)
    if not bag_path.exists():
        sys.exit(f"Error: bag path does not exist: {bag_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading bag : {bag_path}")
    print(f"Output      : {out_path}")
    if args.max_frames:
        print(f"Max frames  : {args.max_frames}")

    source = McapSource(bag_path, loop=False, max_frames=args.max_frames)

    arrays: dict[str, np.ndarray] = {}
    timestamps: list[float] = []
    total_points = 0
    t0 = time.monotonic()

    for frame in source:
        i = frame.frame_id
        arrays[f"frame_{i}_points"] = frame.points
        arrays[f"frame_{i}_intensity"] = frame.intensity
        if frame.ring is not None:
            arrays[f"frame_{i}_ring"] = frame.ring
        timestamps.append(frame.timestamp)
        total_points += frame.num_points
        print(f"  frame {i:4d}: {frame.num_points:6d} pts  ts={frame.timestamp:.3f}")

    n_frames = len(timestamps)
    arrays["timestamps"] = np.array(timestamps, dtype=np.float64)

    np.savez_compressed(out_path, **arrays)
    elapsed = time.monotonic() - t0

    print()
    print("Summary")
    print(f"  Frames exported : {n_frames}")
    print(f"  Total points    : {total_points:,}")
    print(
        f"  Avg pts/frame   : {total_points // n_frames if n_frames else 0:,}"
    )
    print(f"  Duration (bag)  : {timestamps[-1] - timestamps[0]:.3f} s" if n_frames > 1 else "")
    print(f"  Wall time       : {elapsed:.1f} s")
    print(f"  Saved to        : {out_path.resolve()}")


if __name__ == "__main__":
    main()
