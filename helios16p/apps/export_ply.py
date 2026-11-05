"""Export point clouds from a Helios 16P recording to PLY (CloudCompare, MeshLab, Open3D).

    # one dense cloud from all frames of a static scene
    helios16p-export --rec room_static.npz --merge

    # every 5th frame as its own PLY, coloured by intensity
    helios16p-export --rec walk_one.npz --every 5 --color intensity

PLYs carry x y z, intensity and ring as vertex properties plus a viridis colour
(height, intensity or ring).  Without ``--out`` the files land in the current directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..sources.packet_file_source import PacketFileSource


def _viridis(t: np.ndarray) -> np.ndarray:
    try:
        from matplotlib import cm
        return cm.viridis(t)[:, :3]
    except ImportError:  # a small built-in ramp so the export never needs matplotlib
        stops = np.array([[0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530],
                          [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
                          [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
                          [0.741, 0.873, 0.150], [0.993, 0.906, 0.144]])
        x = np.linspace(0, 1, len(stops))
        return np.stack([np.interp(t, x, stops[:, k]) for k in range(3)], axis=1)


def colorize(pts: np.ndarray, mode: str, intensity: np.ndarray | None, ring: np.ndarray | None) -> np.ndarray:
    if mode == "intensity" and intensity is not None:
        t = np.clip(intensity / 255.0, 0.0, 1.0)
    elif mode == "ring" and ring is not None:
        t = ring.astype(np.float64) / 15.0
    else:
        z = pts[:, 2]
        t = (z - z.min()) / (np.ptp(z) + 1e-9)
    return _viridis(t)


def save_ply(pts: np.ndarray, col: np.ndarray, path: Path, intensity=None, ring=None) -> None:
    from ..ply import write_ply
    write_ply(path, pts, intensity, ring, col)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", default=None, help="recording .npz (default: the bundled 4 s clip)")
    ap.add_argument("--out", default=None, help="output file (--merge) or directory; default: ./<recording name>_merged.ply or ./<name>_ply/")
    ap.add_argument("--merge", action="store_true", help="accumulate selected frames into one dense cloud")
    ap.add_argument("--every", type=int, default=1, help="export every Kth frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--frame", type=int, default=None, help="export only this frame index")
    ap.add_argument("--color", choices=["height", "intensity", "ring"], default="height")
    ap.add_argument("--calib", default=None, help="calibration .json to use instead of the recording's DIFOP")
    args = ap.parse_args(argv)

    from ..cli import resolve_recording
    rec = resolve_recording(args.rec)
    calib = None
    if args.calib:
        from ..calibration import Calibration
        calib = Calibration.load(args.calib)
    src = PacketFileSource(str(rec), max_frames=args.max_frames, calibration=calib)
    print(f"calibration: {src.calibration.source}")
    frames = list(src)
    if args.frame is not None:
        frames = [frames[args.frame]]
    else:
        frames = frames[:: args.every]
    if not frames:
        raise SystemExit("no frames to export")

    if args.merge:
        pts = np.vstack([f.points for f in frames])
        inten = np.concatenate([f.intensity for f in frames])
        ring = np.concatenate([f.ring for f in frames]) if frames[0].ring is not None else None
        outp = Path(args.out) if args.out else Path(f"{rec.stem}_merged.ply")
        save_ply(pts, colorize(pts, args.color, inten, ring), outp, inten, ring)
        print(f"merged {len(frames)} frames -> {outp}  ({len(pts)} points)")
    else:
        outdir = Path(args.out) if args.out else Path(f"{rec.stem}_ply")
        outdir.mkdir(parents=True, exist_ok=True)
        for f in frames:
            save_ply(f.points, colorize(f.points, args.color, f.intensity, f.ring),
                     outdir / f"frame_{f.frame_id:04d}.ply", f.intensity, f.ring)
        print(f"wrote {len(frames)} PLY file(s) -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
