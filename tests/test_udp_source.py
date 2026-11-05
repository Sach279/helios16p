"""Tests for helios16p/sources/udp_source.py and tools/pcap_replay.py.

Run as:
    python tests/test_udp_source.py

No hardware required.  No live socket is opened.
All assertions use plain assert + informative messages; prints PASS on success.
"""
from __future__ import annotations

import math
import os
import struct
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import helios16p.config as config
from helios16p.sources.udp_source import decode_msop_packet

# ---------------------------------------------------------------------------
# Helpers: synthetic packet builders
# ---------------------------------------------------------------------------

def build_msop_packet(
    block_azimuths_centideg: list[int],
    channel_data: list[tuple[int, int]] | None = None,
    block_offset: int = 0,  # which block to put channel_data in
) -> bytes:
    """Build a syntactically valid 1248-byte MSOP packet.

    Parameters
    ----------
    block_azimuths_centideg:
        List of 12 azimuth values (0..35999 centi-degrees), one per block.
    channel_data:
        List of (distance_raw: int, intensity: int) for the first N channels
        of ``block_offset``.  All other channel slots are zero-filled
        (distance=0, intensity=0) and will be filtered out as invalid.
    block_offset:
        Which block (0..11) receives the non-zero channel_data entries.
    """
    assert len(block_azimuths_centideg) == config.BLOCKS_PER_PKT, \
        f"Need exactly {config.BLOCKS_PER_PKT} block azimuths"

    # ---- Header (42 bytes) ------------------------------------------------
    # id[4] + protocol_version[2] + reserved1[14] + timestamp[10]
    # + lidar_type[1] + reserved2[7] + temp[2] + reserved3[2]
    header = bytearray(config.HEADER_LEN)
    header[0:4] = config.MSOP_ID          # 55 AA 05 5A
    # Everything else can be zeros (protocol_version, timestamps, temp, …)

    # ---- Blocks (12 × 100 bytes) ------------------------------------------
    blocks_buf = bytearray(config.BLOCKS_PER_PKT * config.BLOCK_LEN)

    for blk_i, az in enumerate(block_azimuths_centideg):
        off = blk_i * config.BLOCK_LEN
        # Block id: FF EE
        blocks_buf[off + 0] = 0xFF
        blocks_buf[off + 1] = 0xEE
        # Azimuth: uint16 big-endian
        struct.pack_into(">H", blocks_buf, off + 2, az & 0xFFFF)
        # Channels: 32 × 3 bytes (dist u16 BE + intensity u8)
        # Default: all zeros (→ distance=0 → filtered out in decode)
        ch_start = off + 4
        if blk_i == block_offset and channel_data:
            for ch_i, (dist_raw, intens) in enumerate(channel_data):
                ch_off = ch_start + ch_i * 3
                struct.pack_into(">H", blocks_buf, ch_off, dist_raw & 0xFFFF)
                blocks_buf[ch_off + 2] = intens & 0xFF

    # ---- Tail (6 bytes): index(4) + tail(2) --------------------------------
    tail = bytearray(6)
    # index and tail can be anything

    pkt = bytes(header) + bytes(blocks_buf) + bytes(tail)
    assert len(pkt) == config.MSOP_LEN, f"built packet length {len(pkt)}, expected {config.MSOP_LEN}"
    return pkt


# ---------------------------------------------------------------------------
# Test 1: packet layout constants
# ---------------------------------------------------------------------------

def test_packet_layout():
    """Verify that the packet size equation holds with the constants in config."""
    expected = config.HEADER_LEN + config.BLOCKS_PER_PKT * config.BLOCK_LEN + 6
    assert expected == config.MSOP_LEN, (
        f"Layout: {config.HEADER_LEN} + {config.BLOCKS_PER_PKT}×{config.BLOCK_LEN} + 6 = "
        f"{expected} != MSOP_LEN={config.MSOP_LEN}"
    )
    assert config.HEADER_LEN == 42,              "HEADER_LEN must be 42"
    assert config.BLOCK_LEN == 100,              "BLOCK_LEN must be 100"
    assert config.BLOCKS_PER_PKT == 12,          "BLOCKS_PER_PKT must be 12"
    assert config.CHANNELS_PER_BLOCK == 32,      "CHANNELS_PER_BLOCK must be 32"
    assert config.LASER_NUM == 16,               "LASER_NUM must be 16"
    assert config.MSOP_LEN == 1248,              "MSOP_LEN must be 1248"
    assert config.DISTANCE_RES == pytest_approx(0.0025), "DISTANCE_RES must be 0.0025"
    print("  test_packet_layout PASS")


def pytest_approx(v, rel=1e-6):
    """Tiny stand-in so we can use approximate equality without pytest."""
    class _Approx:
        def __eq__(self, other):
            return abs(other - v) <= abs(v) * rel + 1e-12
        def __repr__(self):
            return f"≈{v}"
    return _Approx()


# ---------------------------------------------------------------------------
# Test 2: distance recovery
# ---------------------------------------------------------------------------

def test_distance_recovery():
    """A channel with raw distance D should appear at D * DISTANCE_RES metres."""
    DIST_RAW  = 4000   # 4000 × 0.0025 = 10.0 m
    INTENSITY = 200

    # Put the channel in block 0, channel index 0 (laser 0, vert ~ -15°)
    pkt = build_msop_packet(
        block_azimuths_centideg=[0] * 12,
        channel_data=[(DIST_RAW, INTENSITY)],
        block_offset=0,
    )

    result = decode_msop_packet(pkt, config.DEFAULT_VERT_ANGLES_DEG)

    assert result["points"].ndim == 2 and result["points"].shape[1] == 3, \
        "points must be (M, 3)"
    assert result["intensity"].ndim == 1, "intensity must be 1-D"
    assert result["ring"].ndim == 1, "ring must be 1-D"
    assert result["azimuth_deg"].ndim == 1, "azimuth_deg must be 1-D"

    # There should be at least 1 valid point (the one we set)
    assert result["points"].shape[0] >= 1, "Expected at least one decoded point"

    # The first decoded point came from channel 0 of block 0.
    # Laser 0 has vert ~ -15°.  Az ~ 0°.
    # x = d*cos(vert)*cos(az) + RX*cos(az_raw)
    # y = -d*cos(vert)*sin(az) - RX*sin(az_raw)
    # z = d*sin(vert) + RZ
    d = DIST_RAW * config.DISTANCE_RES  # 10.0 m
    vert_rad = math.radians(config.DEFAULT_VERT_ANGLES_DEG[0])  # laser 0, ~-15°
    az_rad   = 0.0  # azimuth ≈ 0

    expected_x = d * math.cos(vert_rad) * math.cos(az_rad) + config.RX * math.cos(0.0)
    expected_y = -d * math.cos(vert_rad) * math.sin(az_rad) - config.RX * math.sin(0.0)
    expected_z = d * math.sin(vert_rad) + config.RZ

    # Find the point that matches laser 0
    pts = result["points"]
    laser0_mask = result["laser"] == 0
    assert np.any(laser0_mask), "No point from wire laser 0 found"
    p = pts[laser0_mask][0]

    tol = 0.01  # 1 cm
    assert abs(p[0] - expected_x) < tol, f"x mismatch: got {p[0]:.4f}, expected {expected_x:.4f}"
    assert abs(p[1] - expected_y) < tol, f"y mismatch: got {p[1]:.4f}, expected {expected_y:.4f}"
    assert abs(p[2] - expected_z) < tol, f"z mismatch: got {p[2]:.4f}, expected {expected_z:.4f}"

    # Intensity check
    intens = result["intensity"]
    assert float(intens[laser0_mask][0]) == float(INTENSITY), \
        f"Intensity mismatch: {intens[laser0_mask][0]} != {INTENSITY}"

    print("  test_distance_recovery PASS")


# ---------------------------------------------------------------------------
# Test 3: near-equatorial azimuth=0 channel
# ---------------------------------------------------------------------------

def test_equatorial_axis():
    """A channel at az=0 with vert~0 should land near the +x axis (y≈0, z≈0)."""
    # Wire laser 6 in DEFAULT_VERT_ANGLES_DEG (firing order) has vert = +1.0°, the closest to 0
    LASER_IDX = 6   # vert = 1.0°
    CHAN_IDX   = LASER_IDX  # first firing: chan == laser for chan < 16

    DIST_RAW = 2000  # 5.0 m

    pkt = build_msop_packet(
        block_azimuths_centideg=[0] * 12,
        channel_data=[(DIST_RAW if i == CHAN_IDX else 0, 100 if i == CHAN_IDX else 0)
                      for i in range(config.CHANNELS_PER_BLOCK)],
        block_offset=0,
    )

    result = decode_msop_packet(pkt, config.DEFAULT_VERT_ANGLES_DEG)
    pts   = result["points"]

    mask = result["laser"] == LASER_IDX
    assert np.any(mask), f"No point for laser {LASER_IDX}"
    p = pts[mask][0]

    d = DIST_RAW * config.DISTANCE_RES
    vert_deg = config.DEFAULT_VERT_ANGLES_DEG[LASER_IDX]  # 1.0°

    # At az≈0 and vert≈0: x≈d, y≈0, z≈small
    # We just check y and z are small relative to x
    assert abs(p[1]) < 0.1, f"y should be near 0 for az=0, got {p[1]:.4f}"
    expected_z = d * math.sin(math.radians(vert_deg))
    assert abs(p[2] - expected_z) < 0.05, f"z mismatch: got {p[2]:.4f}, expected {expected_z:.4f}"

    print("  test_equatorial_axis PASS")


# ---------------------------------------------------------------------------
# Test 4: bad packet rejected
# ---------------------------------------------------------------------------

def test_bad_packet_rejected():
    """decode_msop_packet must raise ValueError on wrong length or id."""
    import traceback

    # Wrong length
    try:
        decode_msop_packet(b"\x00" * 100, config.DEFAULT_VERT_ANGLES_DEG)
        assert False, "Should have raised ValueError for short packet"
    except ValueError:
        pass

    # Wrong id
    pkt = build_msop_packet(block_azimuths_centideg=[0] * 12)
    bad = bytearray(pkt)
    bad[0] = 0x00  # corrupt the id
    try:
        decode_msop_packet(bytes(bad), config.DEFAULT_VERT_ANGLES_DEG)
        assert False, "Should have raised ValueError for bad MSOP id"
    except ValueError:
        pass

    print("  test_bad_packet_rejected PASS")


# ---------------------------------------------------------------------------
# Test 5: pcap round-trip
# ---------------------------------------------------------------------------

def _build_pcap(msop_payload: bytes) -> bytes:
    """Wrap one MSOP payload in a minimal libpcap file (Ethernet+IPv4+UDP)."""
    # Global header (24 B): magic LE, version 2.4, thiszone=0, sigfigs=0,
    #   snaplen=65535, network=1 (Ethernet)
    global_hdr = struct.pack("<IHHiIII",
        0xA1B2C3D4,  # magic LE, microsecond timestamps
        2, 4,        # major, minor version
        0,           # thiszone
        0,           # sigfigs
        65535,       # snaplen
        1,           # link type: Ethernet
    )

    # Ethernet frame
    dst_mac  = bytes(6)
    src_mac  = bytes(6)
    eth_type = bytes([0x08, 0x00])  # IPv4
    eth_hdr  = dst_mac + src_mac + eth_type

    # IPv4 header (20 B, no options)
    udp_len     = 8 + len(msop_payload)
    total_len   = 20 + udp_len
    ip_hdr = struct.pack("!BBHHHBBH4s4s",
        0x45,         # version=4, IHL=5 (20 B)
        0,            # DSCP/ECN
        total_len,
        0,            # identification
        0,            # flags + fragment offset
        64,           # TTL
        17,           # protocol = UDP
        0,            # checksum (zero = unchecked)
        bytes([192, 168, 1, 200]),  # src = sensor
        bytes([192, 168, 1, 102]),  # dst = host
    )

    # UDP header (8 B)
    udp_hdr = struct.pack("!HHHH",
        2000,                 # src port
        config.MSOP_PORT,     # dst port (6699)
        udp_len,
        0,                    # checksum (zero)
    )

    frame = eth_hdr + ip_hdr + udp_hdr + msop_payload

    # Record header (16 B)
    ts_sec  = 1700000000
    ts_usec = 0
    incl_len = len(frame)
    orig_len = incl_len
    rec_hdr = struct.pack("<IIII", ts_sec, ts_usec, incl_len, orig_len)

    return global_hdr + rec_hdr + frame


def test_pcap_roundtrip():
    """Build a tiny in-memory pcap, write it, replay it, check we get the payload back."""
    import importlib, sys

    # Build the pcap with our synthetic MSOP packet
    DIST_RAW  = 3000   # 7.5 m
    INTENSITY = 150
    pkt = build_msop_packet(
        block_azimuths_centideg=[0] * 12,
        channel_data=[(DIST_RAW, INTENSITY)],
        block_offset=0,
    )
    pcap_bytes = _build_pcap(pkt)

    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        f.write(pcap_bytes)
        tmp_path = Path(f.name)

    try:
        # Import replay module
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pcap_replay",
            Path(__file__).resolve().parent.parent / "tools" / "pcap_replay.py"
        )
        pcap_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pcap_mod)

        # Test the low-level parser: should find exactly one record
        data = tmp_path.read_bytes()
        records = list(pcap_mod._iter_pcap_records(data))
        assert len(records) == 1, f"Expected 1 pcap record, got {len(records)}"

        # The record should contain our UDP payload
        ts, frame_bytes = records[0]
        result = pcap_mod._extract_udp_payload(frame_bytes)
        assert result is not None, "_extract_udp_payload returned None"
        dst_port, payload = result
        assert dst_port == config.MSOP_PORT, f"dst_port={dst_port}, expected {config.MSOP_PORT}"
        assert payload == pkt, "Payload mismatch after pcap round-trip"

        # Decode the extracted payload
        decoded = decode_msop_packet(payload, config.DEFAULT_VERT_ANGLES_DEG)
        assert decoded["points"].shape[0] >= 1, "No points decoded from pcap payload"

        # Check at least one point has the right intensity
        intens = decoded["intensity"]
        mask = decoded["laser"] == 0   # channel 0 → wire laser 0
        assert np.any(mask), "No laser-0 point found"
        assert float(intens[mask][0]) == float(INTENSITY), \
            f"Intensity mismatch: {intens[mask][0]} != {INTENSITY}"

        print("  test_pcap_roundtrip PASS")

    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 6: UdpSource can be instantiated (no live socket test)
# ---------------------------------------------------------------------------

def test_udpsource_import():
    """UdpSource can be imported and its name attribute is correct."""
    from helios16p.sources.udp_source import UdpSource
    assert UdpSource.name == "udp", f"Expected name='udp', got {UdpSource.name!r}"
    print("  test_udpsource_import PASS")


# ---------------------------------------------------------------------------
# Test 7: clean shutdown — no "Bad file descriptor" flood
# ---------------------------------------------------------------------------

def test_udpsource_clean_shutdown():
    """
    Construct a UdpSource on ephemeral ports (no hardware), iterate in a
    background thread for a moment, call close(), and assert:
      - close() returns in < 3 s
      - the reader thread is joined (not alive)
      - stdout/stderr contain no repeated MSOP recv-error lines
    """
    import io
    import threading
    import time
    import socket

    from helios16p.sources.udp_source import UdpSource
    from helios16p.config import SensorConfig

    # Pick free ephemeral ports so the test works without root / conflicting
    # with a live sensor.  We bind two temporary sockets to grab free ports,
    # then immediately release them before UdpSource binds its own.
    def _free_port():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    msop_port  = _free_port()
    difop_port = _free_port()

    cfg = SensorConfig(msop_port=msop_port, difop_port=difop_port)

    # Capture log output to check for error spam.
    import logging
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.WARNING)
    root_logger = logging.getLogger("helios16p")
    root_logger.addHandler(handler)
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)

    captured_stderr = io.StringIO()
    original_stderr = sys.stderr
    sys.stderr = captured_stderr

    try:
        src = UdpSource(cfg)

        # Spin up an iteration thread; it will block waiting for frames.
        frames_received = []
        iter_done = threading.Event()

        def _iterate():
            try:
                for frame in src:
                    frames_received.append(frame)
            except Exception:
                pass
            finally:
                iter_done.set()

        iter_thread = threading.Thread(target=_iterate, daemon=True)
        iter_thread.start()

        # Let the source sit for a short moment with no hardware data.
        time.sleep(0.3)

        # Now close — must return promptly.
        t0 = time.time()
        src.close()
        elapsed = time.time() - t0

        assert elapsed < 3.0, f"close() took too long: {elapsed:.2f} s"

        # Reader thread must be joined (not alive) shortly after close().
        src._reader_thread.join(timeout=2.0)
        assert not src._reader_thread.is_alive(), "Reader thread is still alive after close()"

        # Iterator thread should also exit because stop flag is set.
        iter_done.wait(timeout=2.0)
        assert iter_done.is_set(), "Iterator thread did not exit after close()"

        # Check for repeated recv-error messages (the flood).
        log_text = log_capture.getvalue()
        stderr_text = captured_stderr.getvalue()
        combined = log_text + stderr_text

        bad_fd_lines = [
            line for line in combined.splitlines()
            if "Bad file descriptor" in line or "Errno 9" in line
        ]
        assert len(bad_fd_lines) <= 1, (
            f"Expected at most 1 'Bad file descriptor' line, got {len(bad_fd_lines)}:\n"
            + "\n".join(bad_fd_lines[:10])
        )

    finally:
        sys.stderr = original_stderr
        root_logger.removeHandler(handler)
        root_logger.setLevel(original_level)

    print("  test_udpsource_clean_shutdown PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running Helios 16P UDP source tests (no hardware required)...")
    print()

    tests = [
        test_packet_layout,
        test_distance_recovery,
        test_equatorial_axis,
        test_bad_packet_rejected,
        test_pcap_roundtrip,
        test_udpsource_import,
        test_udpsource_clean_shutdown,
    ]

    failed = 0
    for t in tests:
        try:
            print(f"[{t.__name__}]")
            t()
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1

    print()
    if failed == 0:
        print(f"All {len(tests)} tests PASS")
    else:
        print(f"{failed}/{len(tests)} tests FAILED")
        sys.exit(1)
