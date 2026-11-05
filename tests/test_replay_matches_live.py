"""Replay and live decode of the same packets must agree frame by frame.

The recorded packets are streamed over loopback into ``UdpSource``; the frames
it emits are compared with ``PacketFileSource`` on the same recording.
"""
from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

from helios16p.config import SensorConfig
from helios16p.sources.packet_file_source import PacketFileSource
from helios16p.sources.udp_source import UdpSource


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.timeout(60)
def test_replay_frames_equal_live_frames(mini_recording, difop_packet):
    z = np.load(mini_recording)
    msop = [row.tobytes() for row in z["msop"]]
    cfg = SensorConfig(msop_port=_free_port(), difop_port=_free_port())
    live = UdpSource(cfg)

    def send():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for _ in range(3):  # make sure the calibration lands before the data
            s.sendto(difop_packet, ("127.0.0.1", cfg.difop_port))
            time.sleep(0.05)
        for pkt in msop:
            s.sendto(pkt, ("127.0.0.1", cfg.msop_port))
            time.sleep(0.0005)
        s.close()

    t = threading.Thread(target=send, daemon=True)
    t.start()
    expected = list(PacketFileSource(str(mini_recording)))
    got = []
    deadline = time.time() + 20
    it = iter(live)
    while len(got) < len(expected) and time.time() < deadline:
        try:
            got.append(next(it))
        except StopIteration:
            break
    live.close()
    t.join(timeout=5)
    assert live.calibration.source == "difop"
    assert len(got) == len(expected), (len(got), len(expected))
    for a, b in zip(got, expected):
        assert a.num_points == b.num_points
        np.testing.assert_array_equal(a.points, b.points)
        np.testing.assert_array_equal(a.ring, b.ring)
