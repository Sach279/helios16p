"""Render a recording to PNG without opening a window (documentation figures, CI, remote machines).

    helios16p-snapshot --rec walk.npz --out walk.png                 # one annotated frame (boxes, tracks, HUD)
    helios16p-snapshot --rec room.npz --merge 60 --out room.png      # 60 revolutions merged, height-coloured
    helios16p-snapshot --rec corridor.npz --merge 100 --compare-calibration --out before_after.png

Uses the same PyVista scene as the live viewer, rendered off screen.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..calibration import Calibration
from ..config import DemoConfig
from ..pipeline import ProcessingPipeline
from ..sources.packet_file_source import PacketFileSource
from ..types import Frame

OLD_GRID = np.array([-15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
BG = "#14141c"


def _plotter(size, shape=(1, 1)):
    import pyvista as pv
    p = pv.Plotter(off_screen=True, window_size=list(size), shape=shape, border=False)
    p.set_background(BG)
    return p


def _bounds(pts: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> list[float]:
    """Framing box from percentiles, so a few stray far returns do not shrink the scene."""
    a = np.percentile(pts, lo, axis=0); b = np.percentile(pts, hi, axis=0)
    return [a[0], b[0], a[1], b[1], a[2], b[2]]


def _camera(p, view: str, zoom: float = 1.0, pts: np.ndarray | None = None) -> None:
    if view == "top":
        p.view_xy()
    elif view == "front":
        p.view_yz(negative=True)  # looking along +x, the sensor's forward axis
    else:
        p.view_isometric()
    if pts is not None and len(pts) > 100:
        p.reset_camera(bounds=_bounds(pts))
    else:
        p.reset_camera()
    p.camera.zoom(zoom)


def annotated_frame(rec: Path, index: int | None, calib: Calibration | None, cfg: DemoConfig):
    src = PacketFileSource(str(rec), calibration=calib)
    pipe = ProcessingPipeline(cfg, enable_control=True)
    frames = list(src)
    if not frames:
        raise SystemExit("empty recording")
    if index is None:
        index = min(len(frames) - 1, max(1, len(frames) // 2))
    ann = None
    for f in frames[: index + 1]:  # run the tracker up to the chosen frame so tracks have history
        ann = pipe.process(f)
    return ann, src.calibration


def merged_points(rec: Path, n: int, calib: Calibration | None):
    src = PacketFileSource(str(rec), max_frames=n, calibration=calib)
    frames = [f for f in src][1:] or list(src)  # skip the partial first revolution
    pts = np.vstack([f.points for f in frames]).astype(np.float64)
    inten = np.concatenate([f.intensity for f in frames])
    ring = np.concatenate([f.ring for f in frames])
    return pts, inten, ring, src.calibration


def add_cloud(p, pts, scalars, cmap="viridis", point_size=2.0, clim=None):
    import pyvista as pv
    cloud = pv.PolyData(pts)
    cloud["s"] = scalars
    p.add_mesh(cloud, scalars="s", cmap=cmap, style="points", point_size=point_size,
               render_points_as_spheres=False, show_scalar_bar=False, clim=clim)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", default=None, help="recording .npz (default: the bundled 4 s clip)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, default=None, help="frame index for the annotated render (default: middle)")
    ap.add_argument("--merge", type=int, default=0, help="merge this many revolutions into one dense cloud (no annotations)")
    ap.add_argument("--color", choices=["height", "intensity", "ring"], default="height")
    ap.add_argument("--view", choices=["iso", "top", "front"], default="iso")
    ap.add_argument("--size", default="1600x1000")
    ap.add_argument("--point-size", type=float, default=None)
    ap.add_argument("--calib", default=None, help="calibration .json instead of the recording's DIFOP")
    ap.add_argument("--compare-calibration", action="store_true",
                    help="two panels: the pre-fix evenly spaced guess (left) against the sensor's calibration (right)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--hud", action="store_true", help="overlay the viewer's statistics on the annotated render")
    args = ap.parse_args(argv)

    import pyvista as pv
    size = tuple(int(v) for v in args.size.lower().split("x"))
    from ..cli import resolve_recording
    rec = resolve_recording(args.rec)
    calib = Calibration.load(args.calib) if args.calib else None
    cfg = DemoConfig()
    psize = args.point_size or (2.0 if args.merge else cfg.point_size)

    if args.compare_calibration:
        n = args.merge or 30
        old = Calibration(OLD_GRID, np.zeros(16), source="pre-fix grid")
        p = _plotter(size, shape=(1, 2))
        panels = []
        for cal in (old, calib):
            pts, inten, ring, used = merged_points(rec, n, cal)
            sc = pts[:, 2] if args.color == "height" else inten if args.color == "intensity" else ring.astype(float)
            panels.append((pts, sc, used))
        ref = panels[1][1]
        clim = (float(np.percentile(ref, 2)), float(np.percentile(ref, 98))) if args.color != "ring" else (0.0, 15.0)
        for col, ((pts, sc, used), label) in enumerate(zip(panels, ["before: evenly spaced guess, no horizontal offsets",
                                                                     "after: the sensor's own DIFOP calibration"])):
            p.subplot(0, col)
            p.set_background(BG)
            add_cloud(p, pts, sc, point_size=psize, clim=clim)  # one colour scale for both panels
            p.add_text(label if col == 0 else f"{label} ({used.source})", position="upper_left", font_size=10, color="white")
            _camera(p, args.view, 1.0, pts)
        p.link_views()
        p.screenshot(args.out)
        print(f"wrote {args.out}")
        return 0

    p = _plotter(size)
    if args.merge:
        pts, inten, ring, used = merged_points(rec, args.merge, calib)
        sc = pts[:, 2] if args.color == "height" else inten if args.color == "intensity" else ring.astype(float)
        clim = (np.percentile(sc, 2), np.percentile(sc, 98)) if args.color != "ring" else (0, 15)
        add_cloud(p, pts, sc, point_size=psize, clim=clim)
        title = args.title or f"{rec.name}: {args.merge} revolutions merged, {len(pts):,} points, coloured by {args.color}"
        p.add_text(title, position="upper_left", font_size=10, color="white")
    else:
        from ..viz.pyvista_viewer import PyVistaViewer, Scene
        ann, used = annotated_frame(rec, args.frame, calib, cfg)
        scene = Scene(cfg, args.color)
        scene.rebuild(p, ann)
        if args.hud:
            PyVistaViewer(cfg, color_mode=args.color)._update_hud(p, ann)
        title = args.title or (f"{rec.name}: frame {ann.frame.frame_id}, {ann.frame.num_points:,} points, "
                               f"{len(ann.detections)} objects, {len([t for t in ann.tracks if t.confirmed])} tracks")
        p.add_text(title, position="lower_left", font_size=10, color="#cccccc")
        pts = ann.frame.points
    _camera(p, args.view, 1.0 if args.merge else 1.6, pts)
    p.screenshot(args.out)
    print(f"wrote {args.out} (calibration: {used.source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
