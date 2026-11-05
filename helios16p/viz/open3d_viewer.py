"""Interactive Open3D GUI viewer for the Helios-16P LiDAR demo.

Architecture
------------
- Open3DViewer.run() opens a gui.Application window with a SceneWidget (3-D) and
  a side control panel (gui.Vert).
- A background worker thread iterates the source, calls pipeline.process(), and
  posts each FrameAnnotations to the GUI main thread via
  gui.Application.instance.post_to_main_thread().
- All geometry construction is pure-function so it can be tested headlessly.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from ..config import DemoConfig
from ..types import Detection, Frame, FrameAnnotations, Track

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

# A stable categorical palette for track IDs (up to 16 distinct colours).
_TRACK_PALETTE: list[tuple[float, float, float]] = [
    (0.902, 0.098, 0.294),
    (0.235, 0.706, 0.294),
    (1.000, 0.882, 0.098),
    (0.263, 0.388, 0.847),
    (0.961, 0.510, 0.192),
    (0.569, 0.118, 0.706),
    (0.275, 0.941, 0.941),
    (0.941, 0.196, 0.902),
    (0.737, 0.965, 0.047),
    (0.980, 0.745, 0.745),
    (0.000, 0.502, 0.502),
    (0.902, 0.745, 1.000),
    (0.604, 0.388, 0.141),
    (1.000, 0.980, 0.784),
    (0.502, 0.000, 0.000),
    (0.667, 1.000, 0.765),
]


def _track_colour(track_id: int) -> tuple[float, float, float]:
    return _TRACK_PALETTE[track_id % len(_TRACK_PALETTE)]


def _viridis(t: np.ndarray) -> np.ndarray:
    """Approximate viridis colormap, t in [0, 1]. Returns (N, 3) float64."""
    t = np.clip(t, 0.0, 1.0).reshape(-1, 1)
    r = np.clip(0.267 + 0.004 * t + 2.329 * t**2 - 3.020 * t**3 + 1.416 * t**4, 0, 1)
    g = np.clip(0.005 + 1.614 * t - 0.536 * t**2 - 0.527 * t**3 + 0.441 * t**4, 0, 1)
    b = np.clip(0.329 + 1.501 * t - 2.937 * t**2 + 1.990 * t**3 - 0.694 * t**4, 0, 1)
    return np.hstack([r, g, b]).astype(np.float64)


def _categorical_colours(ids: np.ndarray, n_cats: int = 16) -> np.ndarray:
    """Map integer IDs 0..n_cats-1 to colours, returns (N, 3) float64."""
    palette = np.array(_TRACK_PALETTE[:n_cats], dtype=np.float64)
    return palette[ids.astype(int) % len(palette)]


# ---------------------------------------------------------------------------
# Pure geometry helpers (testable without a display)
# ---------------------------------------------------------------------------


def make_point_cloud(
    frame: Frame,
    ann: FrameAnnotations,
    mode: str = "height",
    show_ground: bool = True,
) -> o3d.geometry.PointCloud:
    """Build a coloured PointCloud from *frame* and *ann*.

    Parameters
    ----------
    frame   : the raw frame (points, intensity, ring).
    ann     : FrameAnnotations for this frame (ground_mask).
    mode    : one of "height", "intensity", "ring".
    show_ground : if False, ground points are removed from the cloud.

    Returns
    -------
    o3d.geometry.PointCloud  (may be empty if all points are ground and
    show_ground is False).
    """
    pts = frame.points.astype(np.float64)  # (N, 3)
    N = pts.shape[0]

    ground = ann.ground_mask  # (N,) bool or None
    if ground is None:
        ground = np.zeros(N, dtype=bool)

    if not show_ground:
        mask = ~ground
        pts = pts[mask]
        intensity = frame.intensity[mask].astype(np.float64)
        ring = frame.ring[mask].astype(np.int16) if frame.ring is not None else None
        is_ground = ground[mask]  # all False
    else:
        intensity = frame.intensity.astype(np.float64)
        ring = frame.ring.astype(np.int16) if frame.ring is not None else None
        is_ground = ground

    n = pts.shape[0]
    if n == 0:
        pcd = o3d.geometry.PointCloud()
        return pcd

    # Compute colour by mode
    if mode == "height":
        z = pts[:, 2]
        z_min, z_max = float(z.min()), float(z.max())
        span = z_max - z_min if z_max > z_min else 1.0
        t = (z - z_min) / span
        colours = _viridis(t)
    elif mode == "intensity":
        t = np.clip(intensity / 255.0, 0.0, 1.0)
        colours = _viridis(t)
    elif mode == "ring":
        if ring is not None:
            colours = _categorical_colours(ring.astype(np.int32))
        else:
            colours = np.tile([0.6, 0.6, 0.6], (n, 1)).astype(np.float64)
    else:
        raise ValueError(f"Unknown colour mode: {mode!r}")

    # Override ground points with dim grey
    if show_ground and is_ground.any():
        colours[is_ground] = [0.35, 0.35, 0.35]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colours)
    return pcd


def make_bbox_lineset(det: Detection, colour: tuple[float, float, float] | None = None) -> o3d.geometry.LineSet:
    """Build a LineSet for one Detection's axis-aligned bounding box.

    12 edges, 8 corners.
    """
    if colour is None:
        if det.track_id is not None:
            colour = _track_colour(det.track_id)
        else:
            colour = (0.8, 0.8, 0.1)

    bb = o3d.geometry.AxisAlignedBoundingBox(
        det.bbox_min.astype(np.float64),
        det.bbox_max.astype(np.float64),
    )
    ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bb)
    c = np.array([colour] * len(ls.lines), dtype=np.float64)
    ls.colors = o3d.utility.Vector3dVector(c)
    return ls


def make_trail_lineset(track: Track) -> o3d.geometry.LineSet:
    """Build a LineSet through track.history + current position (the trail).

    Returns an empty LineSet if there are fewer than 2 points in the trail.
    """
    colour = _track_colour(track.track_id)
    pts = list(track.history) + [track.position]
    if len(pts) < 2:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        ls.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        return ls

    points = np.array([p.astype(np.float64) for p in pts])
    n = len(points)
    lines = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int32)
    colours = np.array([colour] * len(lines), dtype=np.float64)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colours)
    return ls


def velocity_arrow(
    track: Track, scale: float = 0.5
) -> o3d.geometry.LineSet:
    """Build a simple 2-point LineSet arrow: position -> position + velocity*scale.

    Returns a 1-line LineSet. If speed is near zero the arrow is still returned
    but will be degenerate (zero length).
    """
    colour = _track_colour(track.track_id)
    origin = track.position.astype(np.float64)
    tip = origin + track.velocity.astype(np.float64) * scale

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array([origin, tip]))
    ls.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.array([colour], dtype=np.float64))
    return ls


def make_marker_sphere(
    position: np.ndarray, radius: float = 0.15, colour: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> o3d.geometry.TriangleMesh:
    """A small sphere marking a track's current position."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
    sphere.translate(position.astype(np.float64))
    sphere.paint_uniform_color(colour)
    sphere.compute_vertex_normals()
    return sphere


def make_ground_grid(size: float = 6.0, step: float = 1.0) -> o3d.geometry.LineSet:
    """A faint XY grid on z=0 for spatial reference."""
    pts = []
    lines = []
    idx = 0
    vals = np.arange(-size, size + step * 0.5, step)
    for v in vals:
        # lines parallel to X
        pts.append([v, -size, 0.0])
        pts.append([v, size, 0.0])
        lines.append([idx, idx + 1])
        idx += 2
        # lines parallel to Y
        pts.append([-size, v, 0.0])
        pts.append([size, v, 0.0])
        lines.append([idx, idx + 1])
        idx += 2

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    n = len(lines)
    ls.colors = o3d.utility.Vector3dVector(
        np.tile([0.25, 0.25, 0.25], (n, 1)).astype(np.float64)
    )
    return ls


def make_coordinate_frame(size: float = 0.5) -> o3d.geometry.TriangleMesh:
    """Small coordinate frame (axes) at the origin."""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.0, 0.0, 0.0])


def make_control_ray(control: dict[str, Any]) -> o3d.geometry.LineSet | None:
    """Build a LineSet visualising the gimbal aim direction.

    Reads control["gimbal_azimuth_deg"] (and/or "aim_point") to draw a ray/line
    from the origin. Returns None if no usable key is present.
    """
    origin = np.array([0.0, 0.0, 1.0])  # sensor is ~1 m up

    aim_point = control.get("aim_point")
    if aim_point is not None:
        tip = np.asarray(aim_point, dtype=np.float64)
    else:
        az_deg = control.get("gimbal_azimuth_deg") or control.get("azimuth_cmd_deg")
        if az_deg is None:
            return None
        az_rad = math.radians(float(az_deg))
        tip = origin + np.array([math.cos(az_rad), math.sin(az_rad), 0.0]) * 4.0

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array([origin, tip]))
    ls.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.4, 0.0]], dtype=np.float64))
    return ls


# ---------------------------------------------------------------------------
# Material helper
# ---------------------------------------------------------------------------


def _unlit_mat(point_size: float = 2.0) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = 2.0
    mat.point_size = point_size
    return mat


def _points_mat(point_size: float = 2.0) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size
    return mat


def _mesh_mat() -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    return mat


# ---------------------------------------------------------------------------
# Open3DViewer
# ---------------------------------------------------------------------------

_FONT_SIZE = 14
_PANEL_WIDTH = 240


class Open3DViewer:
    """Interactive viewer built on Open3D's legacy GLFW Visualizer.

    The newer ``open3d.visualization.gui`` (Filament) backend segfaults on some
    Intel/Mesa/Wayland stacks (e.g. Lunar Lake), so we use the rock-solid legacy
    Visualizer: full mouse orbit/zoom/pan, keyboard toggles, and a live status
    line printed to the terminal (the legacy backend has no in-window text panel).

    Usage::

        Open3DViewer(cfg).run(source, pipeline)
    """

    _KEYMAP_HELP = (
        "[keys] 1/2/3 colour=height/intensity/ring | G ground | B boxes | "
        "K tracks | L trails | M control | Q quit"
    )

    def __init__(self, cfg: DemoConfig) -> None:
        self.cfg = cfg
        self._color_mode: str = "height"
        self._show_ground: bool = True
        self._show_clusters: bool = True
        self._show_tracks: bool = True
        self._show_trails: bool = True
        self._show_control: bool = True

        self._latest_ann: FrameAnnotations | None = None
        self._selected_track_id: int | None = None
        self._lock = threading.Lock()
        self._has_new = False        # worker produced a new annotation
        self._dirty = False          # a toggle changed; rebuild from latest
        self._running = False
        self._dyn: list = []         # dynamic geometry handles currently in the scene

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, source, pipeline) -> None:  # noqa: ANN001
        """Open the window and render until it is closed (blocks)."""
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("Helios-16P LiDAR Demo", 1280, 720)

        # Static reference geometry; reset the camera once to frame the room.
        vis.add_geometry(make_ground_grid(), reset_bounding_box=True)
        vis.add_geometry(make_coordinate_frame(), reset_bounding_box=False)

        # One poll initialises the renderer so get_render_option() is non-None.
        vis.poll_events()
        vis.update_renderer()
        opt = vis.get_render_option()
        if opt is not None:
            opt.background_color = np.asarray([0.08, 0.08, 0.10])
            opt.point_size = float(self.cfg.point_size)

        self._register_keys(vis)
        print(self._KEYMAP_HELP)

        self._running = True
        worker = threading.Thread(target=self._worker, args=(source, pipeline),
                                  daemon=True, name="lidar-worker")
        worker.start()

        first = True
        last_status = 0.0
        try:
            while self._running:
                ann = None
                with self._lock:
                    if self._has_new or self._dirty:
                        ann = self._latest_ann
                        self._has_new = False
                        self._dirty = False
                if ann is not None:
                    self._rebuild(vis, ann, reset_camera=first)
                    first = False
                    now = time.perf_counter()
                    if now - last_status > 0.4:
                        self._print_status(ann)
                        last_status = now
                if not vis.poll_events():
                    break
                vis.update_renderer()
                time.sleep(0.003)
        finally:
            self._running = False
            worker.join(timeout=3.0)
            vis.destroy_window()
            print()  # newline after the live status line

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _register_keys(self, vis) -> None:  # noqa: ANN001
        def toggle(attr):
            def cb(_vis):  # noqa: ANN001
                setattr(self, attr, not getattr(self, attr))
                self._dirty = True
                return False
            return cb

        def set_mode(mode):
            def cb(_vis):  # noqa: ANN001
                self._color_mode = mode
                self._dirty = True
                return False
            return cb

        def quit_cb(_vis):  # noqa: ANN001
            self._running = False
            return False

        vis.register_key_callback(ord("1"), set_mode("height"))
        vis.register_key_callback(ord("2"), set_mode("intensity"))
        vis.register_key_callback(ord("3"), set_mode("ring"))
        vis.register_key_callback(ord("G"), toggle("_show_ground"))
        vis.register_key_callback(ord("B"), toggle("_show_clusters"))
        vis.register_key_callback(ord("K"), toggle("_show_tracks"))
        vis.register_key_callback(ord("L"), toggle("_show_trails"))
        vis.register_key_callback(ord("M"), toggle("_show_control"))
        vis.register_key_callback(ord("Q"), quit_cb)

    def _worker(self, source, pipeline) -> None:  # noqa: ANN001
        """Background: pull frames and run the pipeline. All Visualizer calls
        stay on the main thread; here we only produce FrameAnnotations."""
        interval = 1.0 / self.cfg.frame_rate if self.cfg.frame_rate > 0 else 0.0
        try:
            for frame in source:
                if not self._running:
                    break
                t0 = time.perf_counter()
                ann = pipeline.process(frame)
                with self._lock:
                    self._latest_ann = ann
                    self._has_new = True
                if interval > 0:
                    rest = interval - (time.perf_counter() - t0)
                    if rest > 0:
                        time.sleep(rest)
        except Exception as exc:  # surface worker errors instead of dying silently
            print(f"\n[viewer] source/pipeline stopped: {exc!r}")

    def _rebuild(self, vis, ann: FrameAnnotations, reset_camera: bool) -> None:  # noqa: ANN001
        """Replace all dynamic geometry from a fresh annotation (main thread)."""
        for g in self._dyn:
            vis.remove_geometry(g, reset_bounding_box=False)
        self._dyn.clear()

        def add(geo):
            if geo is None:
                return
            vis.add_geometry(geo, reset_bounding_box=False)
            self._dyn.append(geo)

        frame = ann.frame
        pcd = make_point_cloud(frame, ann, self._color_mode, self._show_ground)
        if len(pcd.points) > 0:
            add(pcd)

        if self._show_clusters:
            for det in ann.detections:
                col = _track_colour(det.track_id) if det.track_id is not None else (0.8, 0.8, 0.1)
                add(make_bbox_lineset(det, col))

        for track in ann.tracks:
            if self._show_tracks:
                add(make_marker_sphere(track.position, colour=_track_colour(track.track_id)))
                add(velocity_arrow(track))
            if self._show_trails:
                trail = make_trail_lineset(track)
                if len(trail.lines) > 0:
                    add(trail)

        if self._show_control and ann.control is not None:
            add(make_control_ray(ann.control))

        if reset_camera:
            vis.reset_view_point(True)

    def _print_status(self, ann: FrameAnnotations) -> None:
        s = ann.stats
        sel = self._selected_track_id
        if sel is None and ann.tracks:
            sel = ann.tracks[0].track_id
        tgt = ""
        if sel is not None:
            m = [t for t in ann.tracks if t.track_id == sel]
            if m:
                tgt = f" | target#{sel} {np.linalg.norm(m[0].position):.1f}m {m[0].speed:.1f}m/s"
        ctrl = ""
        if ann.control:
            ctrl = f" | aim err {ann.control.get('error_deg', 0.0):+.0f}deg"
        line = (f"\rFPS {s.get('proc_fps', 0):4.1f} | pts {s.get('n_points', 0):6d} | "
                f"clusters {s.get('n_clusters', 0):2d} | tracks {s.get('n_tracks', 0):2d}"
                f"{tgt}{ctrl}  [{self._color_mode}]   ")
        print(line, end="", flush=True)

