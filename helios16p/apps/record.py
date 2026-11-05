"""Record the raw UDP packets of a Helios 16P to one ``.npz`` file.

Pure Python, no ROS, no sudo, no GUI.  Binds the MSOP (6699) and DIFOP (7788)
sockets and stores every 1248-byte packet with its host timestamp, so the
recording can be replayed and re-decoded offline with full fidelity, including
the sensor's own calibration (the DIFOP packets).

    helios16p-record --name walk1 --seconds 30
    helios16p-record --name freewalk              # until Ctrl+C
"""
from __future__ import annotations

import argparse
import signal
import socket
import time
from pathlib import Path

import numpy as np

from ..config import DIFOP_PORT, MSOP_PORT

DEFAULT_DIR = "recordings"
PKT_LEN = 1248


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="rec", help="recording name prefix")
    ap.add_argument("--seconds", type=float, default=0.0, help="duration; 0 = until Ctrl+C")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="output directory (default: ./recordings)")
    ap.add_argument("--msop-port", type=int, default=MSOP_PORT)
    ap.add_argument("--difop-port", type=int, default=DIFOP_PORT)
    args = ap.parse_args(argv)

    outdir = Path(args.dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}.npz"

    m = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    m.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    m.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 24)  # 16 MB to avoid drops
    m.bind(("0.0.0.0", args.msop_port))
    m.settimeout(0.5)

    d = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    d.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    d.bind(("0.0.0.0", args.difop_port))
    d.setblocking(False)

    msop, msop_ts, difop, difop_ts = [], [], [], []
    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda *a: stop.__setitem__("v", True))

    dur = f" for {args.seconds:g}s" if args.seconds else " (Ctrl+C to stop)"
    print(f"Recording{dur} -> {outpath}")
    print("waiting for packets on UDP", args.msop_port, "(MSOP) and", args.difop_port, "(DIFOP) ...")
    t0 = time.time()
    last = t0
    try:
        while not stop["v"]:
            if args.seconds and time.time() - t0 >= args.seconds:
                break
            try:
                pkt = m.recv(2048)
                if len(pkt) == PKT_LEN:
                    msop.append(pkt)
                    msop_ts.append(time.time())
            except socket.timeout:
                pass
            try:
                while True:
                    dp = d.recv(2048)
                    if len(dp) == PKT_LEN:
                        difop.append(dp)
                        difop_ts.append(time.time())
            except (BlockingIOError, OSError):
                pass
            now = time.time()
            if now - last >= 1.0:
                mb = len(msop) * PKT_LEN / 1e6
                print(f"\r  MSOP {len(msop):6d} pkt ({mb:5.1f} MB) | "
                      f"DIFOP {len(difop):3d} | {now - t0:4.0f}s", end="", flush=True)
                last = now
    finally:
        m.close()
        d.close()

    if not msop:
        print("\nno MSOP packets received: check the NIC address (host 192.168.1.102/24 by default) and the ports")
        return 1
    if not difop:
        print("\nwarning: no DIFOP packet received; the recording will replay with the bundled calibration")

    print("\nsaving...")

    def pack(lst):
        if not lst:
            return np.zeros((0, PKT_LEN), np.uint8)
        return np.frombuffer(b"".join(lst), dtype=np.uint8).reshape(-1, PKT_LEN)

    np.savez_compressed(
        outpath,
        msop=pack(msop), msop_ts=np.asarray(msop_ts, np.float64),
        difop=pack(difop), difop_ts=np.asarray(difop_ts, np.float64),
    )
    sz = outpath.stat().st_size / 1e6
    secs = (msop_ts[-1] - msop_ts[0]) if len(msop_ts) > 1 else 0.0
    print(f"saved {len(msop)} MSOP + {len(difop)} DIFOP packets "
          f"({secs:.1f}s) -> {outpath} ({sz:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
