"""Replay, live view or process Helios 16P frames with one command.

    helios16p-demo                                # replay the bundled 4 s clip
    helios16p-demo --source udp                   # live sensor
    helios16p-demo --source file --rec my.npz     # your own recording
    helios16p-demo --source mcap --bag bag_dir    # a ROS 2 bag, without ROS
    helios16p-demo --source ros                   # a ROS 2 topic (sourced ROS 2)
    helios16p-demo --source synthetic             # no data at all

The source is the only thing that changes; the processing pipeline and the
viewer are identical in every mode.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import bundled_clip
from .config import DemoConfig, SensorConfig
from .sources.base import Source

SOURCES = ["file", "udp", "mcap", "ros", "synthetic"]


def resolve_recording(rec: str | None) -> Path:
    """The recording to use: ``rec`` if given (one clear error if missing), else the bundled clip."""
    if not rec:
        return bundled_clip()
    p = Path(rec)
    if not p.exists():
        raise SystemExit(f"no such recording: {p}  (helios16p-record makes one; no --rec replays the bundled clip)")
    return p


def make_source(args, cfg: DemoConfig) -> Source:
    """Import only the source that is asked for, so a missing optional dependency never blocks the others."""
    calib = None
    if getattr(args, "calib", None):
        from .calibration import Calibration
        calib = Calibration.load(args.calib)
    if args.source == "synthetic":
        from .sources.synthetic_source import SyntheticSource
        return SyntheticSource(n_frames=args.max_frames or 240, fps=cfg.frame_rate, loop=args.loop)
    if args.source == "mcap":
        from .sources.mcap_source import McapSource
        if not args.bag:
            raise SystemExit("--source mcap requires --bag <bag_dir_or_file>")
        return McapSource(args.bag, loop=args.loop, max_frames=args.max_frames)
    if args.source == "udp":
        from .sources.udp_source import UdpSource
        return UdpSource(cfg.sensor, calibration=calib)
    if args.source == "ros":
        from .sources.ros_source import RosSource
        return RosSource(topic=args.topic)
    if args.source == "file":
        from .sources.packet_file_source import PacketFileSource
        rec = resolve_recording(args.rec)
        if not args.rec:
            print(f"no --rec given: replaying the bundled clip {rec.name}")
        return PacketFileSource(str(rec), loop=args.loop, max_frames=args.max_frames, calibration=calib)
    raise SystemExit(f"unknown source: {args.source}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="helios16p-demo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="file", choices=SOURCES, help="where frames come from (default: file)")
    p.add_argument("--rec", help="raw-packet recording .npz (default: the bundled 4 s clip)")
    p.add_argument("--bag", help="mcap bag dir/file (for --source mcap)")
    p.add_argument("--topic", default="/rslidar_points", help="ROS topic (for --source ros)")
    p.add_argument("--calib", help="calibration .json (helios16p-info --save-calib) instead of DIFOP/bundled")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--loop", action="store_true", help="loop offline sources (on by default for the bundled clip unless --max-frames is given)")
    p.add_argument("--no-control", action="store_true", help="disable the virtual gimbal stage")
    p.add_argument("--fps", type=float, default=10.0, help="replay pacing for offline sources")
    p.add_argument("--msop-port", type=int, default=None, help="UDP port of the point data (default 6699)")
    p.add_argument("--difop-port", type=int, default=None, help="UDP port of the calibration packets (default 7788)")
    p.add_argument("--device-ip", default=None, help="accept packets from this sensor address only (two sensors on one switch)")
    p.add_argument("--viewer", default="pyvista", choices=["pyvista", "open3d", "none"],
                   help="pyvista (default, works on X11 and XWayland), open3d, or none (print stats only)")
    p.add_argument("--color", default="height", choices=["height", "intensity", "ring"], help="initial colouring")
    p.add_argument("--only-moving", action="store_true", help="start in moving-objects-only mode")
    p.add_argument("--screenshot", default=None, help="save the window to this PNG once (see --screenshot-after)")
    p.add_argument("--screenshot-after", type=float, default=3.0, help="seconds after the first frame (default 3)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def demo(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sensor = SensorConfig()
    if args.msop_port:
        sensor.msop_port = args.msop_port
    if args.difop_port:
        sensor.difop_port = args.difop_port
    if args.device_ip:
        sensor.device_ip = args.device_ip
    cfg = DemoConfig(sensor=sensor)
    cfg.frame_rate = args.fps
    if args.source == "file" and not args.rec and args.max_frames is None:
        args.loop = True  # a 4 s clip is more useful looping

    from .pipeline import ProcessingPipeline
    source = make_source(args, cfg)
    pipeline = ProcessingPipeline(cfg, enable_control=not args.no_control)

    if args.viewer == "none":
        n = 0
        with source:
            for frame in source:
                ann = pipeline.process(frame)
                n += 1
                print(f"frame {frame.frame_id:5d}: {frame.num_points:6d} pts, "
                      f"{len(ann.detections):2d} objects, {len(ann.tracks):2d} tracks")
                if args.max_frames and n >= args.max_frames:
                    break
        return 0
    if args.viewer == "open3d":
        from .viz.open3d_viewer import Open3DViewer as Viewer
    else:
        from .viz.pyvista_viewer import PyVistaViewer as Viewer
    if args.viewer == "open3d":
        viewer = Viewer(cfg)
    else:
        viewer = Viewer(cfg, color_mode=args.color, only_moving=args.only_moving,
                        screenshot=args.screenshot, screenshot_after=args.screenshot_after)
    with source:
        viewer.run(source, pipeline)
    return 0


def record(argv: list[str] | None = None) -> int:
    from .apps.record import main
    return main(argv)


def export_ply(argv: list[str] | None = None) -> int:
    from .apps.export_ply import main
    return main(argv)


def info(argv: list[str] | None = None) -> int:
    from .apps.info import main
    return main(argv)


def trim(argv: list[str] | None = None) -> int:
    from .apps.trim import main
    return main(argv)


def validate(argv: list[str] | None = None) -> int:
    from .apps.validate_vendor import main
    return main(argv)


def snapshot(argv: list[str] | None = None) -> int:
    from .apps.snapshot import main
    return main(argv)


def import_pcap(argv: list[str] | None = None) -> int:
    from .apps.import_pcap import main
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(demo())
