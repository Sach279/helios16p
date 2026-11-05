"""Standalone pcap replayer for Helios 16P MSOP packets.

Reads a standard libpcap file (little-endian or big-endian global header)
WITHOUT any external dependency (no scapy, no dpkt, no pypcap).

Ethernet (14 B) + IPv4 (variable IHL) + UDP (8 B) framing is stripped;
the UDP payload is fed to decode_msop_packet for any dst-port == 6699 packet.

Frames are assembled on azimuth wrap (current block 0 azimuth < previous
block 0 azimuth from the preceding packet).

Usage
-----
    python tools/pcap_replay.py --pcap capture.pcap [--out frames.npz] [--max-frames N]

Output
------
Without --out: print per-frame statistics to stdout.
With    --out: save an .npz with keys frames_xyz (list saved as object array)
               plus frame_count, total_points.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

# Make sure the project root is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from helios16p.calibration import Calibration, parse_difop
from helios16p.calibration import default as default_calibration
from helios16p.config import (
    BLOCKS_PER_PKT,
    DIFOP_PORT,
    HEADER_LEN,
    MSOP_LEN,
    MSOP_PORT,
)
from helios16p.sources.udp_source import decode_msop_packet

from helios16p.pcap import extract_udp_payload as _extract_udp_payload, iter_pcap_records as _iter_pcap_records  # noqa: E402


# ---------------------------------------------------------------------------
# Frame assembler
# ---------------------------------------------------------------------------

def _first_block_az(pkt: bytes) -> int:
    """Return the azimuth (centi-degrees) of the first block in an MSOP packet."""
    return int.from_bytes(pkt[HEADER_LEN + 2: HEADER_LEN + 4], "big")


def replay_pcap(
    pcap_path: Path,
    calibration: Calibration | None = None,
    max_frames: int | None = None,
) -> list[dict]:
    """Read a pcap file and return a list of decoded frames.

    Each frame is a dict with keys: points, intensity, ring, azimuth_deg,
    plus frame_id and timestamp.  The calibration comes from the first valid
    DIFOP packet (UDP 7788) anywhere in the capture, else from ``calibration``
    or the bundled file.
    """
    data = pcap_path.read_bytes()
    if calibration is None:
        calibration = default_calibration()
        for _ts, record in _iter_pcap_records(data):
            result = _extract_udp_payload(record)
            if result is None:
                continue
            dst_port, payload = result
            if dst_port == DIFOP_PORT:
                cal = parse_difop(payload)
                if cal is not None:
                    calibration = cal
                    break  # DIFOP packets repeat about once a second; the first valid one is enough
        print(f"calibration: {calibration.source}")

    frames: list[dict] = []
    # Accumulation
    acc_pts:       list[np.ndarray] = []
    acc_intensity: list[np.ndarray] = []
    acc_ring:      list[np.ndarray] = []
    acc_az:        list[np.ndarray] = []
    frame_id = 0
    prev_az: int | None = None
    frame_ts: float = 0.0

    for ts, record in _iter_pcap_records(data):
        result = _extract_udp_payload(record)
        if result is None:
            continue
        dst_port, payload = result

        if dst_port != MSOP_PORT:
            continue
        if len(payload) != MSOP_LEN:
            continue

        from helios16p.config import MSOP_ID
        if payload[:4] != MSOP_ID:
            continue

        try:
            decoded = decode_msop_packet(payload, calibration=calibration)
        except ValueError:
            continue

        first_az = _first_block_az(payload)

        wrap = (prev_az is not None) and (first_az < prev_az)
        prev_az = _first_block_az(payload)  # track per-packet first block az

        if wrap:
            if acc_pts:
                frame = {
                    "frame_id":   frame_id,
                    "timestamp":  frame_ts,
                    "points":     np.concatenate(acc_pts, axis=0).astype(np.float32),
                    "intensity":  np.concatenate(acc_intensity).astype(np.float32),
                    "ring":       np.concatenate(acc_ring).astype(np.int16),
                    "azimuth_deg": np.concatenate(acc_az).astype(np.float32),
                }
                frames.append(frame)
                frame_id += 1
                print(f"Frame {frame_id-1:4d}: {frame['points'].shape[0]:6d} points  ts={frame_ts:.6f}")

            if max_frames is not None and len(frames) >= max_frames:
                break

            acc_pts, acc_intensity, acc_ring, acc_az = [], [], [], []
            frame_ts = ts

        if decoded["points"].shape[0] > 0:
            acc_pts.append(decoded["points"])
            acc_intensity.append(decoded["intensity"])
            acc_ring.append(decoded["ring"])
            acc_az.append(decoded["azimuth_deg"])

    if acc_pts and (max_frames is None or len(frames) < max_frames):  # trailing partial revolution
        frames.append({
            "frame_id":   frame_id,
            "timestamp":  frame_ts,
            "points":     np.concatenate(acc_pts, axis=0).astype(np.float32),
            "intensity":  np.concatenate(acc_intensity).astype(np.float32),
            "ring":       np.concatenate(acc_ring).astype(np.int16),
            "azimuth_deg": np.concatenate(acc_az).astype(np.float32),
        })

    return frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode a pcap of Helios 16P MSOP packets (no external deps)"
    )
    parser.add_argument("--pcap", required=True, help="Input .pcap file")
    parser.add_argument("--out", default=None, help="Output .npz file (optional)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after this many frames")
    args = parser.parse_args()

    pcap_path = Path(args.pcap)
    if not pcap_path.exists():
        print(f"ERROR: {pcap_path} not found", file=sys.stderr)
        sys.exit(1)

    frames = replay_pcap(pcap_path, max_frames=args.max_frames)
    total_pts = sum(f["points"].shape[0] for f in frames)
    print(f"\nTotal frames: {len(frames)},  total points: {total_pts}")

    if args.out:
        out_path = Path(args.out)
        # numpy can't save a ragged list of arrays as a single 2D array;
        # save each frame's xyz separately under keys "frame_000", etc.
        save_dict: dict = {"frame_count": len(frames), "total_points": total_pts}
        for i, fr in enumerate(frames):
            save_dict[f"frame_{i:04d}_xyz"]       = fr["points"]
            save_dict[f"frame_{i:04d}_intensity"]  = fr["intensity"]
            save_dict[f"frame_{i:04d}_ring"]       = fr["ring"]
        np.savez_compressed(str(out_path), **save_dict)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
