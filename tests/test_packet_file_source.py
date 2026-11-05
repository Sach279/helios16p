"""Replay of a short real recording (160 packets with one DIFOP packet)."""
from __future__ import annotations

import numpy as np

from helios16p.calibration import parse_difop
from helios16p.sources.packet_file_source import PacketFileSource
from helios16p.sources.udp_source import decode_msop_packet


def test_replay_uses_the_recording_calibration(mini_recording):
    src = PacketFileSource(str(mini_recording))
    assert src.calibration.source == "difop"
    frames = list(src)
    assert len(frames) >= 2, "160 packets at 750 packets/s should hold at least two revolutions"
    f = frames[1]  # the first full frame
    assert f.points.shape[1] == 3 and f.points.dtype == np.float32
    assert f.num_points > 10_000
    assert f.ring.min() >= 0 and f.ring.max() <= 15
    assert len(set(f.ring.tolist())) == 16, "all sixteen beams present"
    d = np.linalg.norm(f.points, axis=1)
    assert d.min() >= 0.1 and d.max() <= 180.0


def test_ring_is_elevation_order(mini_recording):
    """ring 0 must be the lowest beam and ring 15 the highest (rs_driver convention)."""
    f = list(PacketFileSource(str(mini_recording)))[1]
    el = np.degrees(np.arctan2(f.points[:, 2], np.hypot(f.points[:, 0], f.points[:, 1])))
    med = [np.median(el[f.ring == r]) for r in range(16)]
    assert all(med[i] < med[i + 1] for i in range(15)), med
    assert -16 < med[0] < -14 and 14 < med[15] < 16


def test_horizontal_offsets_change_the_cloud(mini_recording, difop_packet):
    """The horizontal table moves points by several degrees of azimuth."""
    z = np.load(mini_recording)
    pkt = z["msop"][0].tobytes()
    cal = parse_difop(difop_packet)
    with_h = decode_msop_packet(pkt, calibration=cal)
    without_h = decode_msop_packet(pkt, cal.vert_deg, None)
    assert np.array_equal(with_h["laser"], without_h["laser"])
    daz = (with_h["azimuth_deg"] - without_h["azimuth_deg"] + 180) % 360 - 180
    for laser in range(16):
        m = with_h["laser"] == laser
        if m.any():
            assert abs(np.median(daz[m]) - cal.horiz_deg[laser]) < 1e-3  # measured error is about 1e-5 deg; 1e-3 catches a one-centidegree slip


def test_max_frames_and_loop(mini_recording):
    assert len(list(PacketFileSource(str(mini_recording), max_frames=1))) == 1
    looped = PacketFileSource(str(mini_recording), loop=True, max_frames=5)
    assert len(list(looped)) == 5
