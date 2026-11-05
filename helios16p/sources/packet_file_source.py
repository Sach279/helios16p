"""Replay a raw-packet recording (from ``helios16p-record``) as a Frame source.

Decodes the recorded MSOP packets with the same numpy decoder, the same range
window and the same frame split as the live ``UdpSource``, so a recording
replays as the sensor was seen live, with no hardware.  The recording's own
DIFOP packets provide the sensor's calibration; a recording without them falls
back to the calibration bundled with the package.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

from ..calibration import Calibration, difop_return_mode, difop_rpm, parse_difop
from ..calibration import default as default_calibration
from ..config import HEADER_LEN, SensorConfig
from ..types import Frame
from .base import Source
from .udp_source import decode_msop_packet

log = logging.getLogger(__name__)


class PacketFileSource(Source):
    """Frames from a ``.npz`` recording with keys ``msop``, ``msop_ts`` and, ideally, ``difop``.

    Parameters
    ----------
    path:
        The recording.
    loop:
        Restart from the beginning when the recording is exhausted.
    max_frames:
        Stop after this many frames (``None`` = all).
    calibration:
        Override the per-laser angles (default: the recording's DIFOP, else bundled).
    min_distance, max_distance:
        Range window in metres (default: ``SensorConfig``, the vendor's 0.1 to 180 m).

    Like the live source, a frame is one revolution: the first packet's partial
    revolution is emitted as frame 0, and the trailing partial revolution at the
    end of the recording is not emitted.
    """

    name = "file"

    def __init__(
        self,
        path: str,
        loop: bool = False,
        max_frames: int | None = None,
        calibration: Calibration | None = None,
        min_distance: float | None = None,
        max_distance: float | None = None,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no such recording: {self.path}")
        data = np.load(self.path)
        self._msop = data["msop"]            # (N, 1248) uint8
        self._msop_ts = data["msop_ts"]      # (N,) float64
        self.loop = loop
        self.max_frames = max_frames
        defaults = SensorConfig()
        self.min_distance = float(defaults.min_distance if min_distance is None else min_distance)
        self.max_distance = float(defaults.max_distance if max_distance is None else max_distance)

        self.rpm: int | None = None
        self.return_mode: str | None = None
        difop = data["difop"] if "difop" in data.files else None
        if difop is not None and len(difop) > 0:
            for row in difop:
                pkt = row.tobytes()
                cal = parse_difop(pkt)
                if cal is not None:
                    self.rpm, self.return_mode = difop_rpm(pkt), difop_return_mode(pkt)
                    if calibration is None:
                        calibration = cal
                    break
        if calibration is None:
            calibration = default_calibration()
            log.warning("%s carries no usable DIFOP packet: decoding with the %s calibration",
                        self.path.name, calibration.source)
        self.calibration = calibration
        if self.return_mode not in (None, "single") or self.rpm not in (None, 600):
            log.warning("%s: sensor reports %s rpm, %s return; the decoder is validated for 600 rpm single return",
                        self.path.name, self.rpm, self.return_mode)

    @property
    def num_packets(self) -> int:
        return int(self._msop.shape[0])

    @property
    def duration(self) -> float:
        return float(self._msop_ts[-1] - self._msop_ts[0]) if len(self._msop_ts) > 1 else 0.0

    def __iter__(self) -> Iterator[Frame]:
        emitted = 0
        while True:
            prev_az = None
            buf_p, buf_i, buf_r = [], [], []
            frame_ts = float(self._msop_ts[0]) if len(self._msop_ts) else 0.0
            fid = 0
            for row, ts in zip(self._msop, self._msop_ts):
                pkt = row.tobytes()
                try:
                    r = decode_msop_packet(pkt, calibration=self.calibration)
                except ValueError:
                    continue
                # frame split on the raw first-block azimuth, exactly as UdpSource
                first_az = int.from_bytes(pkt[HEADER_LEN + 2: HEADER_LEN + 4], "big")
                if prev_az is not None and first_az < prev_az and buf_p:
                    yield self._make_frame(buf_p, buf_i, buf_r, frame_ts, fid)
                    emitted += 1
                    fid += 1
                    if self.max_frames and emitted >= self.max_frames:
                        return
                    buf_p, buf_i, buf_r = [], [], []
                    frame_ts = float(ts)
                prev_az = first_az
                if len(r["points"]):
                    dist = np.linalg.norm(r["points"], axis=1)
                    keep = (dist >= self.min_distance) & (dist <= self.max_distance)
                    buf_p.append(r["points"][keep])
                    buf_i.append(r["intensity"][keep])
                    buf_r.append(r["ring"][keep])
            if not self.loop:
                break

    @staticmethod
    def _make_frame(buf_p, buf_i, buf_r, ts, fid) -> Frame:
        return Frame(
            timestamp=ts,
            points=np.vstack(buf_p).astype(np.float32),
            intensity=np.concatenate(buf_i).astype(np.float32),
            ring=np.concatenate(buf_r).astype(np.int16),
            frame_id=fid,
        )
