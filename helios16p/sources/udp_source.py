"""Pure-Python UDP source for the RoboSense Helios 16P.

No ROS, no vendor SDK at runtime: only stdlib + numpy.

Wire-format reference
---------------------
MSOP (1248 B):
  offset  0 ..  3   MSOP id: 55 AA 05 5A
  offset  4 .. 41   header (42 B total)
  offset 42 ..1241  12 blocks × 100 B each
     block layout:
       [0:2]   id FF EE
       [2:4]   azimuth uint16 big-endian (centi-degrees, 0..35999)
       [4:100] 32 channels × 3 B:
                 [0:2] distance uint16 big-endian
                 [2]   intensity uint8
  offset 1242..1245  index uint32
  offset 1246..1247  tail uint16

DIFOP (1248 B):
  RSCalibrationAngle: sign(1) + value(2, big-endian) in 0.01 deg
  vert_angle_cali[32]  starts at byte 468 (0-indexed in the 1248-B DIFOP)
  horiz_angle_cali[32] starts at byte 564
  (parsed in helios16p.calibration; the 16P fills 16 entries, the rest are 0xFF)

Azimuth interpolation (single-return Rs16SingleReturnBlockIterator):
  CHAN_AZIS[chan] = firing_tss[chan] / (blk_ts * 2)   # fraction of 2-block window
  block_az_diff  = az[blk+1] - az[blk]  (centi-deg, wraps at 36000)
  per-channel az (centi-deg) = block_az + block_az_diff * CHAN_AZIS[chan]
"""
from __future__ import annotations

import errno
import logging
import math
import queue
import socket
import struct
import threading
import time
from typing import Iterator

import numpy as np

from ..config import (
    BLOCK_ID,
    BLOCK_LEN,
    BLOCKS_PER_PKT,
    CHANNELS_PER_BLOCK,
    DEFAULT_VERT_ANGLES_DEG,
    DIFOP_ID,
    DISTANCE_RES,
    FIRING_TSS_US,
    BLOCK_DURATION_US,
    HEADER_LEN,
    LASER_NUM,
    MSOP_ID,
    MSOP_LEN,
    RX,
    RZ,
    SensorConfig,
)
from ..calibration import Calibration, difop_return_mode, difop_rpm, parse_difop
from ..calibration import default as default_calibration
from ..types import Frame
from .base import Source

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-computed channel azimuth fractions (single-return Rs16 iterator)
# ---------------------------------------------------------------------------
# From Rs16SingleReturnBlockIterator::calcChannel in block_iterator.hpp:
#   az_percents[chan] = firing_tss[chan] / (blk_ts * 2)
# Because one 16P block contains TWO firings of 16 lasers, the az fraction
# is normalised over TWO block durations (i.e. 2 × 55.56 µs = 111.12 µs).
_BLK_TS_US = BLOCK_DURATION_US  # 55.56 µs
_CHAN_AZIS: np.ndarray = (FIRING_TSS_US / (_BLK_TS_US * 2.0)).astype(np.float32)

# Structured dtype for one 100-byte block.
# Header bytes 0-1: id (FF EE), bytes 2-3: azimuth uint16 BE
# Followed by 32 channels each 3 B: dist uint16 BE + intensity uint8
_BLOCK_DTYPE = np.dtype([
    ("id",       "u1", (2,)),
    ("azimuth",  ">u2"),
    ("channels", np.dtype([("distance", ">u2"), ("intensity", "u1")]), (32,)),
])
assert _BLOCK_DTYPE.itemsize == BLOCK_LEN, f"block dtype size mismatch: {_BLOCK_DTYPE.itemsize}"

# DIFOP calibration offsets live in helios16p.calibration (vert 468, horiz 564,
# return_mode 300).  The offsets used before November 2026 (303 / 399) pointed
# into the serial-number area of the packet, so no calibration was ever applied
# and every cloud was built on an evenly spaced guess of the beam angles.

# Frame queue capacity: if the consumer falls behind, drop oldest frames so
# the reader thread keeps draining the socket buffer.
_QUEUE_MAXSIZE = 4

# MSOP reader-thread socket timeout (seconds): small enough to react to the
# stop flag without accumulating recv-error spam.
_READER_SOCK_TIMEOUT = 0.5


# ---------------------------------------------------------------------------
# Public decode function
# ---------------------------------------------------------------------------

def decode_msop_packet(
    pkt: bytes,
    vert_angles_deg: np.ndarray | None = None,
    az_corrections_deg: np.ndarray | None = None,
    *,
    calibration: Calibration | None = None,
) -> dict:
    """Decode one 1248-byte MSOP packet into numpy arrays.

    Parameters
    ----------
    pkt:
        Raw UDP payload, must be exactly 1248 bytes with correct MSOP id.
    calibration:
        Per-laser vertical and horizontal angles (``helios16p.calibration``).
        This is the normal way to call the decoder; when given it overrides
        the two positional arrays below.
    vert_angles_deg:
        Per-laser vertical angles in degrees, shape (16,), indexed by wire
        laser id (channel % 16).  Kept for callers that predate ``calibration``.
    az_corrections_deg:
        Per-laser horizontal angle corrections in degrees, shape (16,) or (32,).
        *None* applies no correction.
        If neither this nor ``calibration`` is given, the bundled calibration
        is used.

    Returns
    -------
    dict with keys:
        points      (M, 3) float32   x/y/z in metres (sensor frame)
        intensity   (M,)   float32   reflectivity 0..255
        ring        (M,)   int16     beam index 0..15, 0 = lowest beam (rs_driver's ``ring``)
        laser       (M,)   int16     wire laser id 0..15 (firing order)
        azimuth_deg (M,)   float32   per-point horizontal angle (degrees)
    """
    if calibration is None and vert_angles_deg is None:
        calibration = default_calibration()
    if calibration is not None:
        vert_angles_deg = calibration.vert_deg
        az_corrections_deg = calibration.horiz_deg
        ring_of_laser = calibration.ring_of_laser
    else:
        vert_angles_deg = np.asarray(vert_angles_deg, dtype=np.float64).reshape(-1)[:LASER_NUM]
        order = np.argsort(vert_angles_deg, kind="stable")
        ring_of_laser = np.empty(LASER_NUM, dtype=np.int16)
        ring_of_laser[order] = np.arange(LASER_NUM, dtype=np.int16)

    # --- validation ---------------------------------------------------------
    if len(pkt) != MSOP_LEN:
        raise ValueError(f"MSOP packet length {len(pkt)}, expected {MSOP_LEN}")
    if pkt[:4] != MSOP_ID:
        raise ValueError(f"Bad MSOP id: {pkt[:4].hex()}")

    # --- parse all 12 blocks at once ----------------------------------------
    raw = np.frombuffer(pkt, dtype="u1")
    blk_start = HEADER_LEN  # 42
    blocks_raw = raw[blk_start: blk_start + BLOCKS_PER_PKT * BLOCK_LEN]
    blocks = np.frombuffer(blocks_raw.tobytes(), dtype=_BLOCK_DTYPE)
    # blocks.shape == (12,)

    # Verify block ids (FF EE)
    block_ids = blocks["id"]  # (12, 2)
    if not np.all((block_ids[:, 0] == 0xFF) & (block_ids[:, 1] == 0xEE)):
        raise ValueError("Block id mismatch in MSOP packet")

    # Block azimuths in centi-degrees (integer, 0..35999)
    blk_az = blocks["azimuth"].astype(np.int32)  # (12,)

    # --- compute per-block az_diff (single-return Rs16 iterator) ------------
    # az_diff[blk] = az[blk+1] - az[blk], wrapping at 36000
    # Last block uses a default step.  FOV blind zone: if diff > 100 centi-deg
    # use default (20 centi-deg * 2 for 16P two-firing layout).
    DEFAULT_AZ_DIFF_CENTIDEG = 20 * 2  # matches block_az_diff * 2 in Rs16SingleReturn

    az_diff = np.empty(BLOCKS_PER_PKT, dtype=np.int32)
    raw_diff = (blk_az[1:] - blk_az[:-1]) % 36000  # (11,)
    # Clamp FOV blind zone (large gap means we were in the blind zone)
    az_diff[:-1] = np.where(raw_diff > 100, DEFAULT_AZ_DIFF_CENTIDEG, raw_diff)
    az_diff[-1] = DEFAULT_AZ_DIFF_CENTIDEG  # last block gets default

    # --- per-channel azimuth ------------------------------------------------
    # shape: (12, 32)
    # angle_horiz[blk, chan] = blk_az[blk] + az_diff[blk] * CHAN_AZIS[chan]
    # This mirrors: angle_horiz = block_az + (int32_t)((float)block_az_diff * CHAN_AZIS[chan])
    blk_az_2d  = blk_az[:, np.newaxis]                    # (12, 1)
    az_diff_2d = az_diff[:, np.newaxis].astype(np.float32) # (12, 1)
    # CHAN_AZIS shape (32,); broadcast → (12, 32)
    chan_az_centideg = (blk_az_2d + az_diff_2d * _CHAN_AZIS[np.newaxis, :]).astype(np.int32)

    # The vendor's lens-offset term (RX) uses the per-channel interpolated azimuth
    # BEFORE the horizontal correction (angle_horiz in decoder_RSHELIOS_16P.hpp).
    chan_az_uncorr_centideg = chan_az_centideg % 36000                # (12, 32)

    # Apply horizontal correction from DIFOP (in centi-degrees), laser = chan%16
    if az_corrections_deg is not None:
        corr = np.asarray(az_corrections_deg, dtype=np.float64).reshape(-1)[:LASER_NUM]
        horiz_corr_centi = np.rint(corr * 100.0).astype(np.int32)       # (16,)
        laser_idx = np.arange(CHANNELS_PER_BLOCK) % LASER_NUM            # (32,)
        chan_az_centideg = chan_az_centideg + horiz_corr_centi[laser_idx][np.newaxis, :]

    # Wrap to [0, 36000)
    chan_az_centideg = chan_az_centideg % 36000  # (12, 32)

    # --- distance and intensity ---------------------------------------------
    dist_raw   = blocks["channels"]["distance"]   # (12, 32) uint16 BE (already native after numpy parse)
    intensity  = blocks["channels"]["intensity"]  # (12, 32) uint8

    dist_m = dist_raw.astype(np.float32) * DISTANCE_RES  # (12, 32)

    # --- geometry -----------------------------------------------------------
    # vert angle: laser = chan % 16, shape (32,)
    laser_idx = np.arange(CHANNELS_PER_BLOCK, dtype=np.int16) % LASER_NUM
    vert_rad = np.deg2rad(vert_angles_deg)[laser_idx].astype(np.float32)  # (32,)

    # az in radians, using the per-channel interpolated azimuth
    chan_az_deg = chan_az_centideg * 0.01                          # (12, 32) float
    chan_az_rad = np.deg2rad(chan_az_deg).astype(np.float32)       # (12, 32)

    # Per-channel azimuth without the DIFOP correction, for the RX lens-offset term
    az_raw_rad = np.deg2rad(chan_az_uncorr_centideg * 0.01).astype(np.float32)  # (12, 32)

    cos_vert = np.cos(vert_rad)   # (32,)
    sin_vert = np.sin(vert_rad)   # (32,)
    cos_az   = np.cos(chan_az_rad)  # (12, 32)
    sin_az   = np.sin(chan_az_rad)  # (12, 32)
    cos_az_raw = np.cos(az_raw_rad)      # (12, 32)
    sin_az_raw = np.sin(az_raw_rad)      # (12, 32)

    # Geometry from rs_driver (decoder_RSHELIOS_16P.hpp lines 245-247):
    #   x =  d * cos(vert) * cos(az_final) + RX * cos(az_raw)
    #   y = -d * cos(vert) * sin(az_final) - RX * sin(az_raw)
    #   z =  d * sin(vert) + RZ
    x = dist_m * cos_vert * cos_az + RX * cos_az_raw   # (12, 32)
    y = -dist_m * cos_vert * sin_az - RX * sin_az_raw  # (12, 32)
    z = dist_m * sin_vert + RZ                          # (12, 32)

    # Flatten → (384,)
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    dist_flat  = dist_m.ravel()
    intens_flat = intensity.ravel().astype(np.float32)
    laser_flat = np.tile(laser_idx, BLOCKS_PER_PKT)       # (384,) int16, wire id
    ring_flat  = ring_of_laser[laser_flat]                # (384,) int16, 0 = lowest beam
    az_deg_flat = chan_az_deg.ravel().astype(np.float32)  # (384,)

    # --- distance filter ----------------------------------------------------
    # Valid: distance > 0 and within sensor range (not yet using SensorConfig
    # limits here; the caller/UdpSource applies those).
    valid = (dist_flat > 0.0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    pts = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float32)

    return {
        "points":      pts,
        "intensity":   intens_flat[valid],
        "ring":        ring_flat[valid].astype(np.int16),
        "laser":       laser_flat[valid].astype(np.int16),
        "azimuth_deg": az_deg_flat[valid],
    }


# ---------------------------------------------------------------------------
# DIFOP calibration parser (compatibility wrapper; see helios16p.calibration)
# ---------------------------------------------------------------------------

def _parse_difop_vert_angles(difop_pkt: bytes) -> np.ndarray | None:
    """Vertical angles in degrees (shape 16) from a DIFOP packet, or None."""
    cal = parse_difop(difop_pkt)
    return None if cal is None else cal.vert_deg.copy()


# ---------------------------------------------------------------------------
# UdpSource
# ---------------------------------------------------------------------------

class UdpSource(Source):
    """Live UDP source for the RoboSense Helios 16P.

    Binds two sockets (MSOP + DIFOP) and yields one ``Frame`` per full
    360° rotation.  A new frame is started whenever a packet's first-block
    azimuth is *smaller* than the previous packet's first-block azimuth
    (i.e. the scanner wrapped around 0° somewhere in between).

    A dedicated reader thread continuously drains the MSOP socket and
    assembles frames, pushing completed Frames onto an internal queue.
    The queue has a fixed capacity; if the consumer is slow the oldest
    frame is dropped so the reader never blocks and the OS socket buffer
    does not overflow.

    Shutdown is coordinated via a threading.Event stop flag.  close() sets
    the flag, closes the sockets, and joins the reader thread.  Any socket
    errors that occur after the stop flag is set are silently swallowed —
    no "Bad file descriptor" flood.
    """

    name = "udp"

    def __init__(self, sensor: SensorConfig, calibration: Calibration | None = None) -> None:
        self._sensor = sensor
        # Until the sensor's own DIFOP packet arrives (about once per second),
        # decode with the calibration given here, else the bundled one.
        self._calib: Calibration = calibration or default_calibration()
        self._difop_seen = False
        self._frame_id: int = 0
        self._closed: bool = False

        # Stop flag: set by close() to signal the reader thread to exit.
        self._stop = threading.Event()

        # Frame queue: reader pushes, __iter__ pops.
        self._frame_queue: queue.Queue[Frame] = queue.Queue(maxsize=_QUEUE_MAXSIZE)

        # Always bind 0.0.0.0: receive on whatever NIC carries the LiDAR subnet.
        bind_addr = "0.0.0.0"

        # MSOP socket — short timeout so the reader thread checks the stop flag
        # frequently without burning CPU in a tight loop.
        self._msop_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._msop_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Large receive buffer to absorb bursts while the pipeline processes.
        self._msop_sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 24  # 16 MiB
        )
        self._msop_sock.bind((bind_addr, sensor.msop_port))
        self._msop_sock.settimeout(_READER_SOCK_TIMEOUT)

        # DIFOP socket (non-blocking)
        try:
            self._difop_sock: socket.socket | None = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM
            )
            self._difop_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._difop_sock.bind((bind_addr, sensor.difop_port))
            self._difop_sock.setblocking(False)
        except OSError as exc:
            log.warning("Could not bind DIFOP socket: %s", exc)
            self._difop_sock = None

        # Lock protecting _vert_angles and _frame_id (written by reader,
        # potentially read by the caller after iteration).
        self._state_lock = threading.Lock()

        # Start the dedicated reader thread immediately so it can drain the
        # socket even before __iter__ is called.
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="udp-reader",
            daemon=True,
        )
        self._reader_thread.start()

    @property
    def calibration(self) -> Calibration:
        """The calibration currently used to decode (bundled until the sensor's DIFOP arrives)."""
        with self._state_lock:
            return self._calib

    # ------------------------------------------------------------------
    # Internal: reader thread
    # ------------------------------------------------------------------

    def _poll_difop(self) -> None:
        """Try to read one DIFOP packet without blocking (called from reader thread)."""
        if self._difop_sock is None:
            return
        try:
            data = self._difop_sock.recv(2048)
            if data[:8] == DIFOP_ID:
                cal = parse_difop(data)
                if cal is not None:
                    with self._state_lock:
                        changed = not np.array_equal(cal.vert_deg, self._calib.vert_deg) or \
                                  not np.array_equal(cal.horiz_deg, self._calib.horiz_deg)
                        self._calib = cal
                    if not self._difop_seen or changed:
                        if not self._difop_seen:
                            rpm, mode = difop_rpm(data), difop_return_mode(data)
                            if mode != "single" or rpm != 600:
                                log.warning("sensor reports %s rpm, %s return; the decoder is validated for 600 rpm single return", rpm, mode)
                        self._difop_seen = True
                        log.info("DIFOP calibration received (vert %s..%s deg, horiz %s..%s deg)",
                                 cal.vert_deg.min(), cal.vert_deg.max(), cal.horiz_deg.min(), cal.horiz_deg.max())
        except BlockingIOError:
            pass
        except OSError as exc:
            # After stop is set, socket errors are expected — swallow silently.
            if not self._stop.is_set():
                log.debug("DIFOP recv error: %s", exc)

    def _reader_loop(self) -> None:
        """Dedicated thread: drain MSOP socket, assemble frames, push to queue."""
        acc_pts:       list[np.ndarray] = []
        acc_intensity: list[np.ndarray] = []
        acc_ring:      list[np.ndarray] = []
        frame_ts: float = time.time()
        prev_az: int | None = None
        t_start = time.time()
        got_any = False
        warned_silent = False

        min_dist = self._sensor.min_distance
        max_dist = self._sensor.max_distance
        want_ip = self._sensor.device_ip

        while not self._stop.is_set():
            # Poll DIFOP opportunistically (non-blocking, fast)
            self._poll_difop()

            # Receive next MSOP packet (times out every _READER_SOCK_TIMEOUT s)
            try:
                data, (src_ip, _src_port) = self._msop_sock.recvfrom(MSOP_LEN + 64)
            except TimeoutError:
                if not got_any and not warned_silent and time.time() - t_start > 3.0:
                    warned_silent = True
                    log.warning("no MSOP packets on UDP %d after 3 s. Check: sensor powered and linked; this host has "
                                "an address on the sensor's subnet (192.168.1.102/24 by default); the firewall allows "
                                "UDP %d and %d in; no other program holds the port.",
                                self._sensor.msop_port, self._sensor.msop_port, self._sensor.difop_port)
                continue
            except OSError as exc:
                # After stop is set (or socket closed), errors are expected.
                if self._stop.is_set() or exc.errno == errno.EBADF:
                    break
                log.warning("MSOP recv error: %s", exc)
                continue

            if want_ip and src_ip != want_ip:
                continue  # another sender on the same ports
            if not got_any:
                got_any = True
                log.info("MSOP stream from %s on UDP %d", src_ip, self._sensor.msop_port)
            if len(data) != MSOP_LEN:
                log.debug("Non-MSOP length packet %d, skipping", len(data))
                continue
            if data[:4] != MSOP_ID:
                log.debug("Non-MSOP id, skipping")
                continue

            # --- decode the packet ------------------------------------------
            with self._state_lock:
                calib = self._calib
            try:
                decoded = decode_msop_packet(data, calibration=calib)
            except ValueError as exc:
                log.warning("decode_msop_packet: %s", exc)
                continue

            # --- distance filter --------------------------------------------
            if decoded["points"].shape[0] > 0:
                dist = np.linalg.norm(decoded["points"], axis=1)
                keep = (dist >= min_dist) & (dist <= max_dist)
                pts_filt  = decoded["points"][keep]
                int_filt  = decoded["intensity"][keep]
                ring_filt = decoded["ring"][keep]
            else:
                pts_filt  = decoded["points"]
                int_filt  = decoded["intensity"]
                ring_filt = decoded["ring"]

            # --- frame split detection (azimuth wrap) -----------------------
            # Compare FIRST-block azimuths of consecutive packets (same logic
            # as packet_file_source / pcap_replay).  Tracking the last block
            # instead misses every wrap when the 0° crossing falls inside a
            # packet: 36000 is an exact multiple of the per-packet azimuth
            # step, so the crossing stays phase-locked inside a packet for the
            # whole session and no frame is ever emitted.
            first_blk_az = int.from_bytes(data[HEADER_LEN + 2: HEADER_LEN + 4], "big")

            wrap = prev_az is not None and first_blk_az < prev_az
            prev_az = first_blk_az

            if wrap and acc_pts:
                # Assemble completed frame
                with self._state_lock:
                    fid = self._frame_id
                    self._frame_id += 1
                frame = Frame(
                    timestamp=frame_ts,
                    points=np.concatenate(acc_pts, axis=0).astype(np.float32),
                    intensity=np.concatenate(acc_intensity).astype(np.float32),
                    ring=np.concatenate(acc_ring).astype(np.int16),
                    frame_id=fid,
                )

                # Push to queue; if full, drop the OLDEST frame to keep latest.
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()  # discard oldest
                    except queue.Empty:
                        pass
                    try:
                        self._frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass  # extremely unlikely; just drop this frame

                # Reset accumulation for the next revolution
                acc_pts       = []
                acc_intensity = []
                acc_ring      = []
                frame_ts = time.time()

            # Accumulate this packet's points
            if pts_filt.shape[0] > 0:
                acc_pts.append(pts_filt)
                acc_intensity.append(int_filt)
                acc_ring.append(ring_filt)

        log.debug("UDP reader thread exiting.")

    # ------------------------------------------------------------------
    # Source interface
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Signal shutdown, close sockets, join the reader thread.

        Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True

        # 1. Signal the reader to stop.
        self._stop.set()

        # 2. Close sockets so any blocking recv() unblocks immediately.
        for sock in (self._msop_sock, self._difop_sock):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._difop_sock = None

        # 3. Join the reader thread (give it 2 s, then give up).
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)

    def __iter__(self) -> Iterator[Frame]:
        """Yield Frame objects assembled by the reader thread."""
        while not self._stop.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.2)
                yield frame
            except queue.Empty:
                # No frame yet — loop and check stop flag.
                continue
