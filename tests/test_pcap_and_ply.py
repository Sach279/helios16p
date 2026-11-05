"""The pcap importer and the PLY writer, on real packets."""
from __future__ import annotations

import struct

import numpy as np

from helios16p.config import DIFOP_PORT, MSOP_PORT
from helios16p.pcap import pcap_to_recording
from helios16p.ply import write_ply
from helios16p.sources.packet_file_source import PacketFileSource


def _pcap(records: list[tuple[int, bytes]]) -> bytes:
    """A minimal little-endian libpcap file with Ethernet/IPv4/UDP frames."""
    out = bytearray(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
    for i, (port, payload) in enumerate(records):
        udp = struct.pack(">HHHH", 50000, port, 8 + len(payload), 0) + payload
        ip = bytes([0x45, 0]) + struct.pack(">H", 20 + len(udp)) + b"\x00\x00\x40\x00\x40\x11\x00\x00" + bytes([192, 168, 1, 200]) + bytes([192, 168, 1, 102]) + udp
        frame = b"\xff" * 6 + b"\x00" * 6 + b"\x08\x00" + ip
        out += struct.pack("<IIII", 1700000000 + i, 1000 * i, len(frame), len(frame)) + frame
    return bytes(out)


def test_pcap_to_recording_roundtrip(tmp_path, mini_recording, difop_packet):
    z = np.load(mini_recording)
    msop = [row.tobytes() for row in z["msop"][:40]]
    records = [(MSOP_PORT, m) for m in msop[:20]] + [(DIFOP_PORT, difop_packet)] + [(MSOP_PORT, m) for m in msop[20:]]
    pcap = tmp_path / "x.pcap"
    pcap.write_bytes(_pcap(records))
    out = tmp_path / "x.npz"
    n = pcap_to_recording(pcap, out)
    assert n == {"msop": 40, "difop": 1, "seconds": n["seconds"]}
    back = np.load(out)
    assert back["msop"].shape == (40, 1248) and back["difop"].shape == (1, 1248)
    assert back["msop"][5].tobytes() == msop[5]
    src = PacketFileSource(str(out))
    assert src.calibration.source == "difop"  # the DIFOP in the middle of the capture was found


def test_write_ply_with_attributes(tmp_path, mini_recording):
    f = list(PacketFileSource(str(mini_recording)))[1]
    p = write_ply(tmp_path / "f.ply", f.points, f.intensity, f.ring, np.zeros((f.num_points, 3)))
    data = p.read_bytes()
    head, _, body = data.partition(b"end_header\n")
    assert b"property float intensity" in head and b"property uchar ring" in head
    assert len(body) == f.num_points * (3 * 4 + 4 + 1 + 3)
    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(p))
    assert len(pc.points) == f.num_points
    np.testing.assert_allclose(np.asarray(pc.points)[:5], f.points[:5], atol=1e-6)
