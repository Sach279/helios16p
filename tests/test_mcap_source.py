"""tests/test_mcap_source.py

Integration test for McapSource against the real Helios 16P bag.

Run with:
    python tests/test_mcap_source.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BAG_PATH = Path(os.environ.get("HELIOS16P_BAG", "recordings/helios16p_points"))
MAX_FRAMES = 3

# ------------------------------------------------------------------
# Helper: dump the raw PointField layout from the first message
# ------------------------------------------------------------------

def _dump_fields(bag_path: Path) -> None:
    from rosbags.highlevel import AnyReader

    TOPIC = "/rslidar_points"
    print(f"\nProbing PointCloud2 field layout from {bag_path} ...")
    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.topic == TOPIC]
        assert conns, f"Topic {TOPIC!r} not found in bag"
        for conn, _ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            print(f"  width={msg.width}, height={msg.height}, point_step={msg.point_step}, "
                  f"is_bigendian={msg.is_bigendian}")
            # PF datatype -> human label
            _DT = {1:"INT8",2:"UINT8",3:"INT16",4:"UINT16",5:"INT32",6:"UINT32",7:"FLOAT32",8:"FLOAT64"}
            print("  Fields:")
            for f in msg.fields:
                print(f"    name={f.name!r:12s}  offset={f.offset:3d}  "
                      f"datatype={f.datatype} ({_DT.get(f.datatype,'?')})  count={f.count}")
            break


def test_mcap_source() -> None:
    if not BAG_PATH.exists():
        import pytest
        pytest.skip(f"no bag at {BAG_PATH} (set HELIOS16P_BAG)")
    from helios16p.sources.mcap_source import McapSource

    # ----------------------------------------------------------------
    # 0. Dump the raw PointField layout
    # ----------------------------------------------------------------
    _dump_fields(BAG_PATH)

    # ----------------------------------------------------------------
    # 1. Read 3 frames
    # ----------------------------------------------------------------
    print(f"\nReading {MAX_FRAMES} frames via McapSource ...")
    source = McapSource(BAG_PATH, loop=False, max_frames=MAX_FRAMES)
    frames = list(source)

    assert len(frames) == MAX_FRAMES, (
        f"Expected {MAX_FRAMES} frames, got {len(frames)}"
    )

    # ----------------------------------------------------------------
    # 2. Per-frame diagnostics + assertions
    # ----------------------------------------------------------------
    for frame in frames:
        pts = frame.points       # (N, 3) float32
        inten = frame.intensity  # (N,)   float32
        N = frame.num_points

        print(f"\n  frame_id={frame.frame_id}  timestamp={frame.timestamp:.6f}")
        print(f"    num_points = {N:,}")
        print(f"    x in [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}] m")
        print(f"    y in [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] m")
        print(f"    z in [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}] m")
        print(f"    intensity in [{inten.min():.1f}, {inten.max():.1f}]")
        if frame.ring is not None:
            print(f"    ring in [{frame.ring.min()}, {frame.ring.max()}]")
        else:
            print( "    ring = None (not present in this bag)")

        # -- assertions --
        assert N > 1000, f"frame {frame.frame_id}: expected >1000 points, got {N}"

        assert pts.dtype == np.float32, (
            f"frame {frame.frame_id}: points dtype should be float32, got {pts.dtype}"
        )
        assert np.isfinite(pts).all(), (
            f"frame {frame.frame_id}: points contain non-finite values"
        )

        # x and y extents within a few hundred metres (indoor/campus bag)
        x_extent = float(pts[:,0].max() - pts[:,0].min())
        y_extent = float(pts[:,1].max() - pts[:,1].min())
        assert x_extent < 500.0, (
            f"frame {frame.frame_id}: x extent {x_extent:.1f} m looks unreasonable"
        )
        assert y_extent < 500.0, (
            f"frame {frame.frame_id}: y extent {y_extent:.1f} m looks unreasonable"
        )

        # timestamp is a plausible POSIX time (after year 2000)
        assert frame.timestamp > 946684800.0, (
            f"frame {frame.frame_id}: timestamp {frame.timestamp} looks wrong"
        )

    # ----------------------------------------------------------------
    # 3. Check frame_id sequence
    # ----------------------------------------------------------------
    assert [f.frame_id for f in frames] == list(range(MAX_FRAMES)), (
        "frame_id values are not monotonically incremented from 0"
    )

    # ----------------------------------------------------------------
    # 4. Check close() is a no-op (does not raise)
    # ----------------------------------------------------------------
    source.close()

    print("\nPASS")


if __name__ == "__main__":
    test_mcap_source()
