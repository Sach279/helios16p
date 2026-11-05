"""Describe a recording: packets, duration, frames, and the calibration it carries.

    helios16p-info --rec walk_one.npz
    helios16p-info --rec walk_one.npz --save-calib my_sensor.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..calibration import difop_return_mode, difop_rpm, parse_difop
from ..sources.packet_file_source import PacketFileSource


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", default=None, help="recording .npz (default: the bundled 4 s clip)")
    ap.add_argument("--save-calib", default=None, help="write the recording's DIFOP calibration to this .json")
    ap.add_argument("--frames", action="store_true", help="also decode everything and count frames and points")
    args = ap.parse_args(argv)

    from ..cli import resolve_recording
    p = resolve_recording(args.rec)
    z = np.load(p)
    msop = z["msop"]; ts = z["msop_ts"]
    difop = z["difop"] if "difop" in z.files else np.zeros((0, 1248), np.uint8)
    dur = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    print(f"{p.name}: {msop.shape[0]} MSOP packets, {difop.shape[0]} DIFOP packets, {dur:.1f} s"
          + (f", {msop.shape[0] / dur:.0f} packets/s" if dur else ""))

    cal = None
    for row in difop:
        cal = parse_difop(row.tobytes())
        if cal is not None:
            pkt = row.tobytes()
            print(f"sensor: {difop_rpm(pkt)} rpm, {difop_return_mode(pkt)} return")
            break
    if cal is None:
        print("calibration: none in the recording (replay uses the bundled file)")
    else:
        print("calibration: from DIFOP")
        print("  laser :", " ".join(f"{i:6d}" for i in range(16)))
        print("  vert  :", " ".join(f"{v:6.2f}" for v in cal.vert_deg))
        print("  horiz :", " ".join(f"{v:6.2f}" for v in cal.horiz_deg))
        print("  ring  :", " ".join(f"{int(r):6d}" for r in cal.ring_of_laser))
        if args.save_calib:
            cal.save(args.save_calib)
            print(f"  saved -> {args.save_calib}")

    if args.frames:
        src = PacketFileSource(str(p))
        n = 0; pts = 0
        for f in src:
            n += 1; pts += f.num_points
        print(f"frames: {n} ({n / dur:.1f} Hz), points: {pts} ({pts // max(n, 1)} per frame)" if dur else f"frames: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
