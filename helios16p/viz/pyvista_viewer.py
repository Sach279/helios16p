"""The 3D scene (shared by the plain window, the Qt window and the off-screen snapshot) and a plain PyVista window.

``Scene`` turns one ``FrameAnnotations`` into actors on any PyVista plotter:
the point cloud with a colour legend, a grid on the fitted ground, wireframe
boxes around objects, spheres with velocity arrows and labels on confirmed
tracks, trails, and the gimbal's aim ray.  Its flags are the view state the
windows and the keyboard toggle.

``PyVistaViewer`` is the dependency-free window (keyboard only, a small HUD);
``helios16p.viz.qt_viewer.QtViewer`` wraps the same scene in a Qt panel with
buttons, metrics and tooltips when PySide6 is installed.

Why VTK and not Open3D for the live window: Open3D's bundled GLFW is Wayland
only and its GL init fails on some Wayland sessions; VTK renders through
X11/GLX (XWayland included).  Open3D stays available as ``--viewer open3d``.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pyvista as pv

from ..config import DemoConfig
from ..types import FrameAnnotations

BG = "#14141c"
GRID_COLOR = "#2e2e40"
GHOST_COLOR = "#50506a"
RAY_COLOR = "#ff7000"

# stable 16-colour palette for track ids
_PALETTE = np.array([
    [0.90, 0.10, 0.10], [0.10, 0.60, 0.90], [0.20, 0.80, 0.20], [0.95, 0.65, 0.10],
    [0.70, 0.30, 0.90], [0.10, 0.85, 0.80], [0.95, 0.40, 0.70], [0.60, 0.80, 0.20],
    [0.40, 0.40, 0.95], [0.85, 0.55, 0.35], [0.30, 0.90, 0.55], [0.95, 0.85, 0.20],
    [0.60, 0.60, 0.65], [0.20, 0.45, 0.70], [0.80, 0.20, 0.40], [0.50, 0.75, 0.95],
])

COLOR_MODES = ("height", "intensity", "ring")
COLOR_TITLES = {"height": "height  m", "intensity": "intensity", "ring": "beam (0 = lowest)"}


def _track_colour(tid: int | None):
    if tid is None:
        return (0.85, 0.85, 0.15)
    return tuple(_PALETTE[tid % len(_PALETTE)])


def _ground_grid(size: float = 8.0, step: float = 1.0, z: float = 0.0) -> pv.PolyData:
    pts, lines = [], []
    i = 0
    vals = np.arange(-size, size + step * 0.5, step)
    for v in vals:
        pts += [[v, -size, z], [v, size, z], [-size, v, z], [size, v, z]]
        lines += [2, i, i + 1, 2, i + 2, i + 3]
        i += 4
    grid = pv.PolyData(np.asarray(pts, float))
    grid.lines = np.asarray(lines)
    return grid


def _bounds(pts: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> list[float]:
    a = np.percentile(pts, lo, axis=0); b = np.percentile(pts, hi, axis=0)
    return [a[0], b[0], a[1], b[1], a[2], b[2]]


def fit_camera(p, pts: np.ndarray | None, view: str = "iso", zoom: float = 1.6) -> None:
    """Frame the scene on the 1st to 99th percentile of the points (stray far returns do not shrink it)."""
    if view == "top":
        p.view_xy()
    elif view == "front":
        p.view_yz(negative=True)
    elif view == "side":
        p.view_xz()
    else:
        p.view_isometric()
    if pts is not None and len(pts) > 100:
        p.reset_camera(bounds=_bounds(pts))
    else:
        p.reset_camera()
    if view != "top":
        p.camera.zoom(zoom)
    else:
        p.camera.zoom(zoom * 0.8)


def _merge_boxes(boxes: list, rgb: list):
    if not boxes:
        return None
    out = pv.merge(boxes) if len(boxes) > 1 else boxes[0].copy()
    cells = np.concatenate([np.repeat([c], b.n_cells, axis=0) for b, c in zip(boxes, rgb)])
    out.cell_data["rgb"] = (np.asarray(cells, float) * 255).astype(np.uint8)
    return out


def _points_mesh(pts: list, rgb: list):
    if not pts:
        return None
    m = pv.PolyData(np.asarray(pts, float))
    m.point_data["rgb"] = (np.asarray(rgb, float) * 255).astype(np.uint8)
    return m


def _lines_mesh(polylines: list, rgb: list):
    """Several polylines in one PolyData, one colour per polyline (cell data)."""
    if not polylines:
        return None
    pts, cells, colours = [], [], []
    off = 0
    for line, c in zip(polylines, rgb):
        n = len(line)
        pts.append(line); cells += [n, *range(off, off + n)]; colours.append(c); off += n
    m = pv.PolyData(np.vstack(pts).astype(float), lines=np.asarray(cells))
    m.cell_data["rgb"] = (np.asarray(colours, float) * 255).astype(np.uint8)
    return m


class Scene:
    """View state plus the code that draws one frame on a PyVista plotter."""

    def __init__(self, cfg: DemoConfig, color_mode: str = "height", only_moving: bool = False) -> None:
        self.cfg = cfg
        self.color_mode = color_mode
        self.show_ground = True
        self.show_clusters = True
        self.show_tracks = True
        self.show_trails = True
        self.show_control = True
        self.show_labels = True
        self.show_grid = True
        self.only_moving = only_moving
        self.ghost = True            # keep a grey shadow of the static scene in only-moving mode
        self._dyn: list[str] = []
        self.ground_z: float | None = None

    def toggle(self, attr: str) -> bool:
        setattr(self, attr, not getattr(self, attr))
        return getattr(self, attr)

    def rebuild(self, p, ann: FrameAnnotations) -> None:  # noqa: ANN001
        """Draw one frame.  The cloud, the ghost and the grid are updated in place; the rest is re-created."""
        for name in self._dyn:
            p.remove_actor(name, render=False)
        self._dyn.clear()

        def add(mesh, name, **kw):
            p.add_mesh(mesh, name=name, render=False, **kw)
            self._dyn.append(name)

        frame = ann.frame
        pts = frame.points.astype(np.float64)
        gmask = ann.ground_mask if ann.ground_mask is not None else np.zeros(len(pts), bool)
        if gmask.any():
            self.ground_z = float(np.median(pts[gmask, 2]))

        # grid: re-create only when toggled or when the ground height moved by more than 2 cm
        gz = self.ground_z if self.ground_z is not None else 0.0
        state = getattr(self, "_grid_state", None)
        want = (self.show_grid, round(gz / 0.02))
        if state != want:
            if state is not None and state[0]:
                p.remove_actor("_grid", render=False)
            if self.show_grid:
                p.add_mesh(_ground_grid(z=gz), name="_grid", color=GRID_COLOR, line_width=1, render=False)
            self._grid_state = want

        thr = self.cfg.moving_speed_threshold
        moving_ids = {t.track_id for t in ann.tracks if t.speed >= thr}

        keep = np.ones(len(pts), bool)
        if not self.show_ground:
            keep &= ~gmask
        ghost_pts = None
        if self.only_moving:
            sel = np.zeros(len(pts), bool)
            for det in ann.detections:
                if det.track_id in moving_ids:
                    sel[det.point_indices] = True
            if self.ghost:
                ghost_pts = pts[keep & ~sel]
            keep &= sel

        # ghost cloud, updated in place
        if ghost_pts is not None and len(ghost_pts):
            if getattr(self, "_ghost_actor", None) is None:
                self._ghost_actor = p.add_mesh(pv.PolyData(ghost_pts), name="_ghost", color=GHOST_COLOR, style="points",
                                               point_size=max(1.0, self.cfg.point_size - 1), render_points_as_spheres=False, render=False)
            else:
                self._ghost_actor.mapper.dataset = pv.PolyData(ghost_pts)
                self._ghost_actor.SetVisibility(True)
        elif getattr(self, "_ghost_actor", None) is not None:
            self._ghost_actor.SetVisibility(False)

        # coloured cloud, updated in place; actor re-created only when the colour mode or point size changes
        if self.color_mode == "intensity":
            scal = np.clip(frame.intensity[keep], 0, 255).astype(np.float64); clim = (0.0, 255.0)
        elif self.color_mode == "ring" and frame.ring is not None:
            scal = frame.ring[keep].astype(np.float64); clim = (0.0, 15.0)
        else:
            scal = pts[keep, 2]
            clim = (float(np.percentile(scal, 1)), float(np.percentile(scal, 99))) if len(scal) > 10 else (0.0, 1.0)
        cloud_pts = pts[keep] if keep.any() else np.zeros((1, 3))
        if not keep.any():
            scal = np.zeros(1)
        key = (self.color_mode, self.cfg.point_size)
        mesh = pv.PolyData(cloud_pts); mesh["s"] = scal
        if getattr(self, "_cloud_key", None) != key or getattr(self, "_cloud_actor", None) is None:
            if getattr(self, "_cloud_actor", None) is not None:
                p.remove_actor("_cloud", render=False)
                for title in list(getattr(p, "scalar_bars", {}).keys()):
                    try:
                        p.remove_scalar_bar(title, render=False)
                    except Exception:
                        pass
            self._cloud_actor = p.add_mesh(mesh, name="_cloud", scalars="s", cmap="viridis", style="points",
                       point_size=self.cfg.point_size, clim=clim, render_points_as_spheres=False, show_scalar_bar=True,
                       scalar_bar_args=dict(title=COLOR_TITLES[self.color_mode], vertical=True, position_x=0.92, position_y=0.25,
                                            width=0.04, height=0.5, title_font_size=12, label_font_size=11, color="#cfcfd8",
                                            n_labels=4 if self.color_mode != "ring" else 6, fmt="%.1f"), render=False)
            self._cloud_key = key
            self._clim = clim
        else:
            self._cloud_actor.mapper.dataset = mesh
            if self.color_mode == "height" and (abs(clim[0] - self._clim[0]) > 0.05 or abs(clim[1] - self._clim[1]) > 0.05):
                try:
                    p.update_scalar_bar_range(clim, name=COLOR_TITLES[self.color_mode])
                except Exception:
                    pass
                self._clim = clim

        # objects, tracks, arrows, trails and the ray: one persistent actor per kind, updated in place
        boxes, box_rgb = [], []
        if self.show_clusters:
            for det in ann.detections:
                if self.only_moving and det.track_id not in moving_ids:
                    continue
                lo, hi = det.bbox_min.astype(float), det.bbox_max.astype(float)
                boxes.append(pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])))
                box_rgb.append(_track_colour(det.track_id))
        marks, mark_rgb, arrows, arrow_rgb, trails, trail_rgb, label_pts, label_txt = [], [], [], [], [], [], [], []
        for t in ann.tracks:
            if not t.confirmed or (self.only_moving and t.speed < thr):
                continue
            col = _track_colour(t.track_id)
            pos = t.position.astype(float)
            if self.show_tracks:
                marks.append(pos); mark_rgb.append(col)
                if t.speed > 0.05:
                    v = t.velocity.astype(float).copy(); v[2] = 0.0
                    n = np.linalg.norm(v)
                    if n > 1e-6:
                        arrows.append((pos, pos + v / n * min(2.0, max(0.4, t.speed)))); arrow_rgb.append(col)
                if self.show_labels:
                    label_pts.append(pos + [0, 0, 0.35]); label_txt.append(f"#{t.track_id}  {t.speed:.1f} m/s")
            if self.show_trails and len(t.history) >= 2:
                trails.append(np.array([h.astype(float) for h in (list(t.history) + [t.position])])); trail_rgb.append(col)
        ray = None
        if self.show_control and ann.control is not None and ann.control.get("target_moving"):
            aim = ann.control.get("aim_point")
            if aim is not None:
                ray = np.asarray(aim, float)

        self._set_actor(p, "_boxes", _merge_boxes(boxes, box_rgb), style="wireframe", line_width=3)
        self._set_actor(p, "_marks", _points_mesh(marks, mark_rgb), style="points", point_size=18, render_points_as_spheres=True)
        self._set_actor(p, "_arrows", _lines_mesh([np.array(a) for a in arrows], arrow_rgb), line_width=6)
        self._set_actor(p, "_trails", _lines_mesh(trails, trail_rgb), line_width=3)
        self._set_actor(p, "_ray", _lines_mesh([np.array([[0, 0, 0], ray])], [(1.0, 0.44, 0.0)]) if ray is not None else None, line_width=4)
        if label_pts:
            p.add_point_labels(np.asarray(label_pts), label_txt, name="_labels", font_size=12, text_color="white",
                               point_size=0, shape_opacity=0.35, shape_color="black", always_visible=True, render=False)
            self._dyn.append("_labels")

    def _set_actor(self, p, name: str, mesh, **kw) -> None:  # noqa: ANN001
        """Keep one actor per kind; swap its dataset each frame, hide it when there is nothing to draw."""
        actors = self.__dict__.setdefault("_actors", {})
        if mesh is None:
            if name in actors:
                actors[name].SetVisibility(False)
            return
        if name not in actors:
            actors[name] = p.add_mesh(mesh, name=name, scalars="rgb", rgb=True, render=False, **kw)
        else:
            actors[name].mapper.dataset = mesh
            actors[name].SetVisibility(True)

    @staticmethod
    def metrics(ann: FrameAnnotations, cfg: DemoConfig) -> dict:
        """The numbers a panel shows, computed once per frame."""
        s = ann.stats
        n = max(1, s.get("n_points", 0))
        confirmed = [t for t in ann.tracks if t.confirmed]
        moving = [t for t in confirmed if t.speed >= cfg.moving_speed_threshold]
        out = {
            "frame": s.get("frame_id", 0),
            "points": s.get("n_points", 0),
            "ground_pct": 100.0 * s.get("n_ground", 0) / n,
            "objects": s.get("n_clusters", 0),
            "tracks": len(confirmed),
            "moving": len(moving),
            "proc_ms": s.get("proc_ms", 0.0),
            "proc_hz": s.get("proc_fps", 0.0),
            "target": None,
        }
        c = ann.control
        if c and c.get("target_moving"):
            out["target"] = (c.get("target_id"), c.get("target_speed", 0.0), c.get("error_deg", 0.0))
        return out


class _Runner:
    """Runs source -> pipeline in a thread and keeps the latest annotation."""

    def __init__(self, cfg: DemoConfig) -> None:
        self.cfg = cfg
        self.latest: FrameAnnotations | None = None
        self.has_new = False
        self.lock = threading.Lock()
        self.running = False
        self.paused = False
        self.finished = False
        self.error: str | None = None

    def start(self, source, pipeline) -> threading.Thread:  # noqa: ANN001
        self.running = True
        t = threading.Thread(target=self._loop, args=(source, pipeline), daemon=True, name="lidar-worker")
        t.start()
        return t

    def _loop(self, source, pipeline) -> None:  # noqa: ANN001
        interval = 1.0 / self.cfg.frame_rate if self.cfg.frame_rate > 0 else 0.0
        try:
            for frame in source:
                while self.paused and self.running:
                    time.sleep(0.05)
                if not self.running:
                    break
                t0 = time.perf_counter()
                ann = pipeline.process(frame)
                with self.lock:
                    self.latest = ann
                    self.has_new = True
                if interval > 0:
                    rest = interval - (time.perf_counter() - t0)
                    if rest > 0:
                        time.sleep(rest)
        except Exception as exc:  # noqa: BLE001
            self.error = repr(exc)
        self.finished = True

    def take(self) -> FrameAnnotations | None:
        with self.lock:
            if self.has_new:
                self.has_new = False
                return self.latest
        return None


def save_frame_ply(ann: FrameAnnotations, directory: str | Path = "recordings/ply") -> Path:
    from ..apps.export_ply import colorize
    from ..ply import write_ply
    fr = ann.frame
    f = Path(directory) / f"frame_{time.strftime('%Y%m%d_%H%M%S')}.ply"
    return write_ply(f, fr.points, fr.intensity, fr.ring, colorize(fr.points, "height", fr.intensity, fr.ring))


class PyVistaViewer:
    """The plain window: keyboard shortcuts and a small HUD, no Qt needed."""

    KEYS = ("1/2/3 colour   G ground   B boxes   K tracks   L trails   M ray   N moving only   "
            "T labels   C backend   V voxel   R refit view   P ply   S screenshot   Q quit")

    def __init__(self, cfg: DemoConfig, color_mode: str = "height", only_moving: bool = False,
                 screenshot: str | None = None, screenshot_after: float = 3.0, source_label: str = "") -> None:
        self.cfg = cfg
        self.scene = Scene(cfg, color_mode, only_moving)
        self.runner = _Runner(cfg)
        self.source_label = source_label
        self._screenshot = screenshot
        self._screenshot_after = screenshot_after
        self._dirty = False
        self._running = False
        self._last: FrameAnnotations | None = None

    # ------------------------------------------------------------------
    def run(self, source, pipeline) -> None:  # noqa: ANN001
        p = pv.Plotter(title="Helios 16P", window_size=(1400, 860))
        p.set_background(BG)
        p.add_text(self.KEYS, position="lower_left", font_size=8, color="#8a8a99", name="_help")
        p.add_text("waiting for the first frame...", position="upper_left", font_size=11, color="white", name="_hud")
        self._register_keys(p)
        self._running = True
        try:
            p.iren.add_observer("ExitEvent", lambda *a: setattr(self, "_running", False))
        except Exception:
            pass
        worker = self.runner.start(source, pipeline)
        p.show(interactive_update=True, auto_close=False)
        first = True
        shot_done = False
        t_open = time.perf_counter()
        try:
            while self._running and not (self.runner.finished and self.runner.latest is None):
                ann = self.runner.take()
                if ann is None and self._dirty and self._last is not None:
                    ann = self._last
                if ann is not None:
                    self._dirty = False
                    self._last = ann
                    self.scene.rebuild(p, ann)
                    self._update_hud(p, ann)
                    if first:
                        fit_camera(p, ann.frame.points)
                        first = False
                if self._screenshot and not shot_done and not first and time.perf_counter() - t_open >= self._screenshot_after:
                    self._shot(p, self._screenshot)
                    shot_done = True
                if self.runner.finished and not self.runner.has_new:
                    if self.runner.error:
                        print(f"\n[viewer] source stopped: {self.runner.error}")
                    else:
                        print("\n[viewer] end of source, closing")
                    break
                try:
                    p.update()
                except Exception:
                    break
                time.sleep(0.01)
        finally:
            self._running = False
            self.runner.running = False
            worker.join(timeout=3.0)
            try:
                p.close()
            except Exception:
                pass
            print()

    # ------------------------------------------------------------------
    def _register_keys(self, p) -> None:  # noqa: ANN001
        for k in ("v", "b", "r", "s", "t", "q"):  # PyVista's own bindings for these keys
            try:
                p.clear_events_for_key(k)
            except Exception:
                pass

        def toggle(attr):
            def cb():
                self.scene.toggle(attr); self._dirty = True
            return cb

        def mode(m):
            def cb():
                self.scene.color_mode = m; self._dirty = True
            return cb

        p.add_key_event("1", mode("height")); p.add_key_event("2", mode("intensity")); p.add_key_event("3", mode("ring"))
        p.add_key_event("g", toggle("show_ground")); p.add_key_event("b", toggle("show_clusters"))
        p.add_key_event("k", toggle("show_tracks")); p.add_key_event("l", toggle("show_trails"))
        p.add_key_event("m", toggle("show_control")); p.add_key_event("n", toggle("only_moving"))
        p.add_key_event("t", toggle("show_labels"))
        p.add_key_event("c", self._cycle_backend); p.add_key_event("v", self._cycle_voxel)
        p.add_key_event("r", lambda: fit_camera(p, self._last.frame.points) if self._last else None)
        p.add_key_event("p", self._save_ply); p.add_key_event("s", lambda: self._shot(p))
        p.add_key_event("q", lambda: setattr(self, "_running", False))

    def _save_ply(self) -> None:
        if self._last is None:
            print("\n[ply] no frame yet"); return
        try:
            f = save_frame_ply(self._last)
            print(f"\n[ply] saved {self._last.frame.num_points} points -> {f}")
        except OSError as exc:
            print(f"\n[ply] could not write: {exc}")

    def _shot(self, p, path: str | None = None) -> None:  # noqa: ANN001
        f = Path(path) if path else Path("recordings/shots") / f"shot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        f.parent.mkdir(parents=True, exist_ok=True)
        try:
            p.screenshot(str(f)); print(f"\n[shot] saved {f}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[shot] failed: {exc}")

    def _cycle_backend(self) -> None:
        self.cfg.cluster_backend = "sklearn" if self.cfg.cluster_backend == "open3d" else "open3d"
        self._dirty = True

    def _cycle_voxel(self) -> None:
        seq = [0.0, 0.05, 0.10]
        try:
            i = seq.index(round(self.cfg.cluster_voxel, 2))
        except ValueError:
            i = 0
        self.cfg.cluster_voxel = seq[(i + 1) % len(seq)]
        self._dirty = True

    def _update_hud(self, p, ann: FrameAnnotations) -> None:  # noqa: ANN001
        m = Scene.metrics(ann, self.cfg)
        lines = [f"{m['points']:,} pts   {m['objects']} objects   {m['tracks']} tracks ({m['moving']} moving)",
                 f"{m['proc_ms']:.0f} ms/frame ({m['proc_hz']:.1f} Hz)   {self.cfg.cluster_backend}, voxel {self.cfg.cluster_voxel * 100:.0f} cm   colour: {self.scene.color_mode}"]
        if m["target"]:
            tid, spd, err = m["target"]
            lines.append(f"gimbal on #{tid}  {spd:.1f} m/s  error {err:+.0f} deg")
        if self.scene.only_moving:
            lines.append("moving objects only")
        if self.source_label:
            lines.append(self.source_label)
        p.add_text("\n".join(lines), position="upper_left", font_size=10, color="white", name="_hud")
