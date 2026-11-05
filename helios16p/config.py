"""Sensor + demo configuration for the RoboSense Helios 16P.

Wire-protocol constants extracted from rs_driver's decoder_RSHELIOS.hpp /
decoder_RSHELIOS_16P.hpp (BSD-licensed). See docs/PROTOCOL_NOTES.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# Network (defaults match the unit; override on the CLI if your setup differs)
# ----------------------------------------------------------------------------
DEVICE_IP = "192.168.1.200"   # the LiDAR's factory address (the host only needs an address on its subnet)
HOST_IP = "192.168.1.102"     # the address the sensor is configured to send to (documentation only; sockets bind 0.0.0.0)
MSOP_PORT = 6699              # point data
DIFOP_PORT = 7788             # device info / calibration

# ----------------------------------------------------------------------------
# MSOP packet layout (Helios / Helios 16P), bytes
# ----------------------------------------------------------------------------
MSOP_LEN = 1248
MSOP_ID = bytes([0x55, 0xAA, 0x05, 0x5A])
DIFOP_ID = bytes([0xA5, 0xFF, 0x00, 0x5A, 0x11, 0x11, 0x55, 0x55])
BLOCK_ID = bytes([0xFF, 0xEE])
HEADER_LEN = 42               # RSHELIOSMsopHeader
BLOCK_LEN = 100               # id[2] + azimuth[2] + 32 * (dist u16 + intensity u8)
BLOCKS_PER_PKT = 12
CHANNELS_PER_BLOCK = 32       # 2 firings of the 16 lasers; laser = chan % 16
LASER_NUM = 16
DISTANCE_RES = 0.0025         # metres per LSB
TEMPERATURE_RES = 0.0625

# Lens optical-centre offsets (metres), used in the spherical->cartesian transform
RX = 0.03498
RY = -0.015
RZ = 0.0

# Per-channel firing time offsets within a block (microseconds), 32 entries.
FIRING_TSS_US = np.array([
    0.00, 1.73, 3.46, 5.19, 6.92, 8.65, 10.38, 12.11,
    13.84, 15.57, 17.30, 19.03, 20.76, 22.49, 24.22, 25.95,
    27.68, 29.41, 31.14, 32.87, 34.60, 36.33, 38.06, 39.79,
    41.52, 43.25, 44.98, 46.71, 48.44, 50.17, 51.90, 53.63,
], dtype=np.float64)
BLOCK_DURATION_US = 55.56

# Nominal Helios 16P vertical angles (degrees) in WIRE order, laser 0..15
# (laser = channel % 16).  The beams fire interleaved, not sorted: laser 0 is
# the +13 deg beam and laser 14 the -15 deg one.  Sources prefer the sensor's own
# DIFOP calibration (helios16p.calibration), which also carries the per-laser
# horizontal offsets of several degrees that these defaults cannot provide.
DEFAULT_VERT_ANGLES_DEG = np.array(
    [13.0, 15.0, 9.0, 11.0, 5.0, 7.0, 1.0, 3.0,
     -3.0, -1.0, -7.0, -5.0, -11.0, -9.0, -15.0, -13.0],
    dtype=np.float64,
)


@dataclass
class SensorConfig:
    device_ip: str | None = None  # accept packets from this sender only (two sensors on one switch); None = any
    msop_port: int = MSOP_PORT
    difop_port: int = DIFOP_PORT
    min_distance: float = 0.1     # metres, the vendor driver's window; live and replay share it
    max_distance: float = 180.0
    dense_points: bool = True     # drop NaN/out-of-range points


@dataclass
class DemoConfig:
    """End-to-end demo knobs. Tunable from the CLI."""
    sensor: SensorConfig = field(default_factory=SensorConfig)

    # ground segmentation
    ground_distance_threshold: float = 0.10   # m, RANSAC inlier band
    ground_max_z: float = 0.3                  # only fit ground below this z

    # clustering
    cluster_eps: float = 0.5                   # m, DBSCAN neighbourhood
    cluster_min_points: int = 15
    cluster_min_height: float = 0.3            # m, reject flat blobs
    cluster_max_height: float = 2.5
    cluster_backend: str = "open3d"            # "open3d" (C++, fast) | "sklearn" (Python)
    cluster_voxel: float = 0.05                # m, voxel downsample of clustering input; 0 = off
    cluster_max_footprint: float = 1.2         # m, reject clusters wider than this (walls/furniture)
    moving_speed_threshold: float = 0.3        # m/s, a track faster than this counts as "moving"

    # tracking
    track_max_age: int = 10                    # frames to keep an unmatched track
    track_min_hits: int = 3                    # hits before a track is confirmed
    track_gate_distance: float = 1.5           # m, association gate

    # display
    point_size: float = 2.0
    frame_rate: float = 10.0                   # Hz, replay pacing for offline sources

    seed: int = 0                              # RANSAC seed, so the same input gives the same output
