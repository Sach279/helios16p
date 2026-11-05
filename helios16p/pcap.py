"""Read libpcap captures (tcpdump, Wireshark, RSView) without external dependencies.

Ethernet (14 B) + IPv4 (variable IHL) + UDP (8 B) framing is stripped and the
UDP payload returned with its destination port.  ``pcap_to_recording`` turns a
capture into the ``.npz`` recording every ``helios16p-*`` command reads.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import DIFOP_PORT, MSOP_PORT

_PCAP_MAGIC_LE = 0xA1B2C3D4     # little-endian, microseconds
_PCAP_MAGIC_BE = 0xD4C3B2A1     # big-endian, microseconds
_PCAP_MAGIC_LE_NS = 0xA1B23C4D  # little-endian, nanoseconds
_PCAP_MAGIC_BE_NS = 0x4D3CB2A1  # big-endian, nanoseconds
GLOBAL_HDR_LEN = 24
RECORD_HDR_LEN = 16
ETH_HDR_LEN = 14
ETH_TYPE_IPv4_BE = bytes([0x08, 0x00])
IP_PROTO_UDP = 17
UDP_HDR_LEN = 8
PKT_LEN = 1248


def iter_pcap_records(data: bytes) -> Iterator[tuple[float, bytes]]:
    """Yield (timestamp_s, frame_bytes) for every record of a libpcap file (either endianness)."""
    if len(data) < GLOBAL_HDR_LEN:
        raise ValueError("File too short for a pcap global header")
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic in (_PCAP_MAGIC_LE, _PCAP_MAGIC_LE_NS):
        endian = "<"
    elif magic in (_PCAP_MAGIC_BE, _PCAP_MAGIC_BE_NS):
        endian = ">"
    else:
        raise ValueError(f"Not a valid libpcap file (magic={magic:#010x}); pcapng is not supported, convert with editcap -F pcap")
    ns_ts = magic in (_PCAP_MAGIC_LE_NS, _PCAP_MAGIC_BE_NS)
    pos = GLOBAL_HDR_LEN
    while pos + RECORD_HDR_LEN <= len(data):
        ts_sec, ts_frac, incl_len, _orig_len = struct.unpack_from(f"{endian}IIII", data, pos)
        pos += RECORD_HDR_LEN
        if pos + incl_len > len(data):
            break  # truncated file
        rec = data[pos: pos + incl_len]
        pos += incl_len
        yield ts_sec + ts_frac * (1e-9 if ns_ts else 1e-6), rec


def extract_udp_payload(frame: bytes) -> tuple[int, bytes] | None:
    """Strip Ethernet + IPv4 + UDP and return (dst_port, payload), or None for anything else."""
    if len(frame) < ETH_HDR_LEN or frame[12:14] != ETH_TYPE_IPv4_BE:
        return None
    ip_start = ETH_HDR_LEN
    if len(frame) < ip_start + 20:
        return None
    ihl = (frame[ip_start] & 0x0F) * 4
    if frame[ip_start + 9] != IP_PROTO_UDP:
        return None
    udp_start = ip_start + ihl
    if len(frame) < udp_start + UDP_HDR_LEN:
        return None
    _src, dst_port, _length, _checksum = struct.unpack_from(">HHHH", frame, udp_start)
    return dst_port, frame[udp_start + UDP_HDR_LEN:]


def pcap_to_recording(pcap_path: str | Path, out_path: str | Path,
                      msop_port: int = MSOP_PORT, difop_port: int = DIFOP_PORT) -> dict:
    """Write the MSOP and DIFOP packets of a capture to a ``.npz`` recording; return counts."""
    data = Path(pcap_path).read_bytes()
    msop, msop_ts, difop, difop_ts = [], [], [], []
    for ts, rec in iter_pcap_records(data):
        r = extract_udp_payload(rec)
        if r is None:
            continue
        port, payload = r
        if len(payload) != PKT_LEN:
            continue
        if port == msop_port:
            msop.append(payload); msop_ts.append(ts)
        elif port == difop_port:
            difop.append(payload); difop_ts.append(ts)

    def pack(lst):
        return np.frombuffer(b"".join(lst), dtype=np.uint8).reshape(-1, PKT_LEN) if lst else np.zeros((0, PKT_LEN), np.uint8)

    np.savez_compressed(out_path, msop=pack(msop), msop_ts=np.asarray(msop_ts, np.float64),
                        difop=pack(difop), difop_ts=np.asarray(difop_ts, np.float64))
    return {"msop": len(msop), "difop": len(difop),
            "seconds": (msop_ts[-1] - msop_ts[0]) if len(msop_ts) > 1 else 0.0}
