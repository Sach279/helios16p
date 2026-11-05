"""Convert a packet capture (tcpdump, Wireshark, RSView; libpcap format) into a recording.

    sudo tcpdump -i <NIC> udp port 6699 or udp port 7788 -w lab.pcap
    helios16p-import-pcap --pcap lab.pcap --out recordings/lab.npz
    helios16p-demo --rec recordings/lab.npz

Capture both ports: 7788 carries the calibration. A capture of MSOP only replays
with the bundled calibration (and says so).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import DIFOP_PORT, MSOP_PORT
from ..pcap import pcap_to_recording


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--out", required=True, help="recording .npz to write")
    ap.add_argument("--msop-port", type=int, default=MSOP_PORT)
    ap.add_argument("--difop-port", type=int, default=DIFOP_PORT)
    args = ap.parse_args(argv)
    if not Path(args.pcap).exists():
        raise SystemExit(f"no such file: {args.pcap}")
    n = pcap_to_recording(args.pcap, args.out, args.msop_port, args.difop_port)
    print(f"{n['msop']} MSOP + {n['difop']} DIFOP packets ({n['seconds']:.1f} s) -> {args.out}")
    if n["msop"] == 0:
        print("no MSOP packets found: was the capture filtered to the right UDP port?")
        return 1
    if n["difop"] == 0:
        print("no DIFOP packets: the recording will replay with the bundled calibration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
