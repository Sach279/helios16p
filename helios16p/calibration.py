"""Per-laser angle calibration of the Helios 16P and the DIFOP parser that reads it.

The sensor publishes its factory calibration in every DIFOP packet (UDP 7788,
1248 bytes): two tables of 32 ``RSCalibrationAngle`` entries (sign byte + big-endian
uint16 in 0.01 degrees), vertical first, horizontal second.  In the Helios DIFOP
layout of rs_driver (``RSHELIOSDifopPkt``) they start at byte 468 (vertical) and
byte 564 (horizontal); the Helios 16P fills the first 16 entries and marks the
rest 0xFF (uncalibrated).

Both tables matter.  The vertical table carries the firing order of the beams,
which is interleaved (+13, +15, +9, +11, ...), not sorted; the horizontal table
carries per-beam azimuth offsets of several degrees (about +3.1 and -7.5 on our
unit).  Decoding without either places points far from where the vendor driver
puts them: median distance to the nearest vendor point 112 mm with an evenly
spaced guess, 53 mm with the vertical table alone, float32 rounding with both
(``helios16p-validate``, docs/07-calibration.md).

Until a DIFOP packet is seen, sources fall back to the calibration bundled in
``helios16p/data/helios16p_calibration.json``, read from our own unit.  The
vertical angles are the nominal ones of every Helios 16P; the horizontal offsets
are unit-specific, so a live sensor (or a recording that kept its DIFOP packets)
always overrides the bundled file.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

LASER_NUM = 16

DIFOP_ID = bytes([0xA5, 0xFF, 0x00, 0x5A, 0x11, 0x11, 0x55, 0x55])
DIFOP_LEN = 1248
DIFOP_RPM_OFF = 8            # uint16 big-endian, 600 or 1200
DIFOP_RETURN_MODE_OFF = 300  # 0x00 dual, 0x04/0x05/0x06 single (rs_driver decodeDifopCommon)
DIFOP_VERT_OFF = 468         # RSCalibrationAngle vert_angle_cali[32]
DIFOP_HORIZ_OFF = 564        # RSCalibrationAngle horiz_angle_cali[32]
_ANGLE_ENTRY = 3             # sign u8 + value u16 big-endian (0.01 deg)

# Nominal Helios 16P vertical angles in wire (firing) order, laser 0..15.
NOMINAL_VERT_DEG = np.array(
    [13.0, 15.0, 9.0, 11.0, 5.0, 7.0, 1.0, 3.0, -3.0, -1.0, -7.0, -5.0, -11.0, -9.0, -15.0, -13.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Calibration:
    """Per-laser angles in degrees, indexed by wire laser id (channel % 16)."""

    vert_deg: np.ndarray
    horiz_deg: np.ndarray
    source: str = "unknown"

    def __post_init__(self) -> None:
        v = np.asarray(self.vert_deg, dtype=np.float64).reshape(-1)
        h = np.asarray(self.horiz_deg, dtype=np.float64).reshape(-1)
        if v.shape != (LASER_NUM,) or h.shape != (LASER_NUM,):
            raise ValueError(f"calibration needs {LASER_NUM} vertical and horizontal angles")
        object.__setattr__(self, "vert_deg", v)
        object.__setattr__(self, "horiz_deg", h)

    @property
    def ring_of_laser(self) -> np.ndarray:
        """Map wire laser id -> ring index, 0 = lowest beam (rs_driver's ``ring``)."""
        order = np.argsort(self.vert_deg, kind="stable")
        ring = np.empty(LASER_NUM, dtype=np.int16)
        ring[order] = np.arange(LASER_NUM, dtype=np.int16)
        return ring

    def to_dict(self) -> dict:
        return {"vert_deg": self.vert_deg.tolist(), "horiz_deg": self.horiz_deg.tolist(), "source": self.source}

    @classmethod
    def from_dict(cls, d: dict, source: str | None = None) -> "Calibration":
        return cls(np.asarray(d["vert_deg"]), np.asarray(d["horiz_deg"]), source or d.get("source", "json"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text()), source=f"file:{p.name}")


def _read_angle_table(pkt: bytes, off: int, n: int = LASER_NUM) -> np.ndarray | None:
    """Read ``n`` RSCalibrationAngle entries at ``off``; None if any is missing or absurd."""
    if len(pkt) < off + n * _ANGLE_ENTRY:
        return None
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        sign = pkt[off + i * _ANGLE_ENTRY]
        val = struct.unpack_from(">H", pkt, off + i * _ANGLE_ENTRY + 1)[0]
        if sign == 0xFF:
            return None  # uncalibrated entry
        v = val if sign == 0 else -val
        if not (-9000 <= v < 18000):  # rs_driver angleCheck, centi-degrees
            return None
        out[i] = v * 0.01
    return out


def parse_difop(pkt: bytes) -> Calibration | None:
    """Return the calibration carried by a DIFOP packet, or None if it holds none."""
    if len(pkt) < DIFOP_LEN or pkt[:8] != DIFOP_ID:
        return None
    vert = _read_angle_table(pkt, DIFOP_VERT_OFF)
    horiz = _read_angle_table(pkt, DIFOP_HORIZ_OFF)
    if vert is None or horiz is None:
        return None
    if np.unique(vert).size < LASER_NUM:
        return None  # sixteen beams at distinct elevations, or it is not a calibration
    return Calibration(vert, horiz, source="difop")


def difop_rpm(pkt: bytes) -> int | None:
    return struct.unpack_from(">H", pkt, DIFOP_RPM_OFF)[0] if len(pkt) >= DIFOP_LEN else None


def difop_return_mode(pkt: bytes) -> str | None:
    if len(pkt) < DIFOP_LEN:
        return None
    b = pkt[DIFOP_RETURN_MODE_OFF]
    return "dual" if b == 0x00 else "single" if b in (0x04, 0x05, 0x06) else f"unknown(0x{b:02x})"


def bundled() -> Calibration:
    """The calibration shipped with the package (read from our unit's DIFOP)."""
    with resources.files("helios16p.data").joinpath("helios16p_calibration.json").open() as f:
        d = json.load(f)
    return Calibration(np.asarray(d["vert_deg"]), np.asarray(d["horiz_deg"]), source="bundled")


def nominal() -> Calibration:
    """Nominal vertical angles in firing order and zero horizontal offsets (last resort)."""
    return Calibration(NOMINAL_VERT_DEG.copy(), np.zeros(LASER_NUM), source="nominal")


def default() -> Calibration:
    """Best calibration available without a sensor: the bundled file, else nominal."""
    try:
        return bundled()
    except (FileNotFoundError, KeyError, ValueError, ModuleNotFoundError):
        return nominal()
