"""Cut a recording down to a time window, keeping its DIFOP packets, so it can be shared.

    helios16p-trim --rec walk_one.npz --start 20 --seconds 4 --out walk_4s.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="seconds from the start of the recording")
    ap.add_argument("--seconds", type=float, required=True, help="length of the window")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    z = np.load(Path(args.rec))
    ts = z["msop_ts"]; t0 = ts[0] + args.start
    keep = (ts >= t0) & (ts < t0 + args.seconds)
    if not keep.any():
        raise SystemExit("the window holds no packets")
    difop = z["difop"] if "difop" in z.files else np.zeros((0, 1248), np.uint8)
    difop_ts = z["difop_ts"] if "difop_ts" in z.files else np.zeros((0,), np.float64)
    np.savez_compressed(args.out, msop=z["msop"][keep], msop_ts=ts[keep], difop=difop[:2], difop_ts=difop_ts[:2])
    print(f"{int(keep.sum())} packets, {ts[keep][-1] - ts[keep][0]:.2f} s, {min(2, len(difop))} DIFOP -> {args.out}"
          f" ({Path(args.out).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
