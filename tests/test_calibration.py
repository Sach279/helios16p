"""The DIFOP calibration parser against a packet recorded from the sensor.

The expected angles were cross-checked against the vendor driver (rs_driver
1.5.18) on 4 September 2026: decoding with these tables matches the vendor
cloud to a fraction of a millimetre (see tests/test_vendor_parity.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from helios16p import calibration as C

EXPECTED_VERT = [12.99, 14.98, 8.99, 10.99, 5.0, 6.99, 1.0, 3.0, -3.0, -1.0, -6.99, -4.99, -10.99, -8.99, -14.98, -12.99]
EXPECTED_HORIZ = [3.15, -7.64, 3.12, -7.55, 3.1, -7.5, 3.1, -7.47, 3.1, -7.46, 3.11, -7.48, 3.13, -7.52, 3.16, -7.59]
# ring (0 = lowest beam) of each wire laser, as the vendor driver reports it
EXPECTED_RING = [14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1]


def test_parse_difop_reads_both_tables(difop_packet):
    cal = C.parse_difop(difop_packet)
    assert cal is not None
    assert cal.source == "difop"
    np.testing.assert_allclose(cal.vert_deg, EXPECTED_VERT, atol=1e-9)
    np.testing.assert_allclose(cal.horiz_deg, EXPECTED_HORIZ, atol=1e-9)
    assert cal.ring_of_laser.tolist() == EXPECTED_RING


def test_difop_header_fields(difop_packet):
    assert difop_packet[:8] == C.DIFOP_ID
    assert C.difop_rpm(difop_packet) == 600
    assert C.difop_return_mode(difop_packet) == "single"


def test_old_offsets_hold_no_calibration(difop_packet):
    """The 303/399 offsets used before November 2026 do not point at the tables."""
    assert C._read_angle_table(difop_packet, 303) is None
    horiz_at_399 = C._read_angle_table(difop_packet, 399)
    assert horiz_at_399 is None or not np.any(horiz_at_399)


def test_parse_difop_rejects_garbage():
    assert C.parse_difop(b"\x00" * 1248) is None
    assert C.parse_difop(C.DIFOP_ID + b"\x00" * 100) is None
    zeros = bytearray(C.DIFOP_ID + b"\x00" * 1240)
    assert C.parse_difop(bytes(zeros)) is None, "sixteen beams at the same elevation is not a calibration"
    good = bytearray(C.DIFOP_ID + b"\x00" * 1240)
    for i, v in enumerate(range(-1500, 1700, 200)):  # distinct elevations in 0.01 deg
        good[C.DIFOP_VERT_OFF + 3 * i] = 0 if v >= 0 else 1
        good[C.DIFOP_VERT_OFF + 3 * i + 1: C.DIFOP_VERT_OFF + 3 * i + 3] = abs(v).to_bytes(2, "big")
    assert C.parse_difop(bytes(good)) is not None
    good[C.DIFOP_VERT_OFF] = 0xFF  # one uncalibrated entry spoils the table
    assert C.parse_difop(bytes(good)) is None


def test_bundled_matches_our_unit(difop_packet):
    b = C.bundled()
    cal = C.parse_difop(difop_packet)
    np.testing.assert_allclose(b.vert_deg, cal.vert_deg)
    np.testing.assert_allclose(b.horiz_deg, cal.horiz_deg)
    assert b.source == "bundled"


def test_nominal_is_the_firing_order():
    n = C.nominal()
    assert n.vert_deg.tolist() == [13, 15, 9, 11, 5, 7, 1, 3, -3, -1, -7, -5, -11, -9, -15, -13]
    assert n.ring_of_laser.tolist() == EXPECTED_RING
    assert not np.any(n.horiz_deg)


def test_save_and_load_roundtrip(tmp_path, difop_packet):
    cal = C.parse_difop(difop_packet)
    p = tmp_path / "c.json"
    cal.save(p)
    back = C.Calibration.load(p)
    np.testing.assert_allclose(back.vert_deg, cal.vert_deg)
    np.testing.assert_allclose(back.horiz_deg, cal.horiz_deg)
    assert back.source.startswith("file:")


def test_calibration_shape_is_checked():
    with pytest.raises(ValueError):
        C.Calibration(np.zeros(32), np.zeros(16))
