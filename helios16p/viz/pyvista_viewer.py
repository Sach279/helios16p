"""Interactive viewer built on PyVista / VTK.

Why not Open3D for the live window? Open3D's bundled GLFW is Wayland-only and
its GL init fails on some Wayland sessions, and its newer Filament GUI can crash
on Intel/Mesa stacks. VTK renders through X11/GLX (XWayland included), which
works on those machines too. The Open3D viewer stays available as an option.

Interface:  PyVistaViewer(cfg).run(source, pipeline)
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pyvista as pv

from ..config import DemoConfig
from ..types import FrameAnnotations, Track

# stable 16-colour palette for track ids
_PALETTE = np.array([
    [0.90, 0.10, 0.10], [0.10, 0.60, 0.90], [0.20, 0.80, 0.20], [0.95, 0.65, 0.10],
    [0.70, 0.30, 0.90], [0.10, 0.85, 0.80], [0.95, 0.40, 0.70], [0.60, 0.80, 0.20],
    [0.40, 0.40, 0.95], [0.85, 0.55, 0.35], [0.30, 0.90, 0.55], [0.95, 0.85, 0.20],
    [0.60, 0.60, 0.65], [0.20, 0.45, 0.70], [0.80, 0.20, 0.40], [0.50, 0.75, 0.95],
])


def _track_colour(tid: int | None):
    if tid is None:
        return (0.85, 0.85, 0.15)
    return tuple(_PALETTE[tid % len(_PALETTE)])


def _ground_grid(size: float = 6.0, step: float = 1.0) -> pv.PolyData:
    pts, lines = [], []
    i = 0
    vals = np.arange(-size, size + step * 0.5, step)
    for v in vals:
        pts += [[v, -size, 0.0], [v, size, 0.0], [-size, v, 0.0], [size, v, 0.0]]
        lines += [[2, i, i + 1], [2, i + 2, i + 3]]
        i += 4
    poly = pv.PolyData(np.asarray(pts, float))
    poly.lines = np.asarray(lines).ravel()
    return poly


class PyVistaViewer:
    _HELP = ("1/2/3 colour  G ground  B boxes  K tracks  L trails  M control  "
             "N moving-only  C backend  V voxel  P save-ply  Q quit")

    def __init__(self, cfg: DemoConfig) -> None:
        self.cfg = cfg
        self._color_mode = "height"
        self._show_ground = True
        self._show_clusters = True
        self._show_tracks = True
        self._show_trails = True
        self._show_control = True
        self._only_moving = False

        self._latest_ann: FrameAnnotations | None = None
        self._lock = threading.Lock()
        self._has_new = False
        self._dirty = False
        self._running = False
        self._dyn: list[str] = []   # names of dynamic actors currently in the scene

    # ------------------------------------------------------------------
    def run(self, source, pipeline) -> None:  # noqa: ANN001
        p = pv.Plotter(title="Helios-16P LiDAR Demo", window_size=(1280, 800))
        p.set_background("#14141c")
        p.add_mesh(_ground_grid(), color="#333344", line_width=1, name="_grid")
        p.add_axes()
        p.add_text(self._HELP, position="lower_left", font_size=8, color="#aaaaaa", name="_help")
        p.add_text("starting...", position="upper_left", font_size=11, color="white", name="_hud")

        self._register_keys(p)
        self._running = True
        # detect window close (X button / VTK exit)
        try:
            p.iren.add_observer("ExitEvent", lambda *a: setattr(self, "_running", False))
        except Exception:
            pass

        worker = threading.Thread(target=self._worker, args=(source, pipeline),
                                  daemon=True, name="lidar-worker")
        worker.start()

        p.show(interactive_update=True, auto_close=False)
        p.view_isometric()
        self._add_widgets(p)
        # keep the legend pinned to the right edge as the window is resized
        try:
            p.iren.add_observer("ConfigureEvent", lambda *a: self._reposition_widgets(p))
        except Exception:
            pass
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
                    self._rebuild(p, ann)
                    self._update_hud(p, ann)
                    if first:
                        p.reset_camera()
                        p.view_isometric()
                        first = False
                    now = time.perf_counter()
                    if now - last_status > 0.5:
                        self._print_status(ann)
                        last_status = now
                try:
                    p.update()
                except Exception:
                    break
                time.sleep(0.01)
        finally:
            self._running = False
            worker.join(timeout=3.0)
            try:
                p.close()
            except Exception:
                pass
            print()

    # ------------------------------------------------------------------
    def _register_keys(self, p) -> None:  # noqa: ANN001
        def toggle(attr):
            def cb():
                setattr(self, attr, not getattr(self, attr))
                self._dirty = True
            return cb

        def mode(m):
            def cb():
                self._color_mode = m
                self._dirty = True
            return cb

        for k in ("v", "b"):  # PyVista binds v (isometric view) and b (point picking) itself
            try:
                p.clear_events_for_key(k)
            except Exception:
                pass
        p.add_key_event("1", mode("height"))
        p.add_key_event("2", mode("intensity"))
        p.add_key_event("3", mode("ring"))
        p.add_key_event("g", toggle("_show_ground"))
        p.add_key_event("b", toggle("_show_clusters"))
        p.add_key_event("k", toggle("_show_tracks"))
        p.add_key_event("l", toggle("_show_trails"))
        p.add_key_event("m", toggle("_show_control"))
        p.add_key_event("n", toggle("_only_moving"))
        p.add_key_event("c", self._cycle_backend)
        p.add_key_event("v", self._cycle_voxel)
        p.add_key_event("p", self._save_ply)
        p.add_key_event("q", lambda: setattr(self, "_running", False))

    def _save_ply(self) -> None:
        """Dump the current frame's point cloud to a height-coloured PLY for
        CloudCompare / external viewers."""
        ann = self._latest_ann
        if ann is None:
            print("\n[ply] no frame yet")
            return
        import time
        from pathlib import Path

        from ..apps.export_ply import colorize
        from ..ply import write_ply

        fr = ann.frame
        f = Path("recordings/ply") / f"frame_{time.strftime('%Y%m%d_%H%M%S')}.ply"
        try:
            write_ply(f, fr.points, fr.intensity, fr.ring, colorize(fr.points, "height", fr.intensity, fr.ring))
        except OSError as exc:
            print(f"\n[ply] could not write {f}: {exc}")
            return
        print(f"\n[ply] saved {fr.num_points} points -> {f} (x y z, intensity, ring, colour)")

    def _cycle_backend(self) -> None:
        """Toggle the clustering backend live (Python sklearn <-> C++ Open3D)."""
        self.cfg.cluster_backend = "sklearn" if self.cfg.cluster_backend == "open3d" else "open3d"
        self._dirty = True

    def _cycle_voxel(self) -> None:
        """Cycle the clustering voxel size live: off -> 5 cm -> 10 cm -> off."""
        seq = [0.0, 0.05, 0.10]
        try:
            i = seq.index(round(self.cfg.cluster_voxel, 2))
        except ValueError:
            i = 0
        self.cfg.cluster_voxel = seq[(i + 1) % len(seq)]
        self._dirty = True

    def _add_widgets(self, p) -> None:  # noqa: ANN001
        """On-screen clickable controls (mouse), in parallel with the keyboard
        shortcuts (shown in parentheses on each label).

        Button/label pairs are kept in ``self._widgets`` so ``_reposition_widgets``
        can re-anchor them to the right edge on window resize (see the
        ConfigureEvent hook in ``run``); PyVista places them at absolute pixel
        coordinates once, they don't track the window edge on their own."""
        specs = [
            ("ground (G)", "_show_ground", (0.27, 0.53, 1.0)),
            ("clusters (B)", "_show_clusters", (1.0, 0.67, 0.0)),
            ("tracks (K)", "_show_tracks", (0.0, 0.80, 0.40)),
            ("trails (L)", "_show_trails", (0.80, 0.40, 1.0)),
            ("control (M)", "_show_control", (1.0, 0.44, 0.0)),
            ("moving-only (N)", "_only_moving", (1.0, 0.20, 0.40)),
        ]
        self._widgets: list[tuple] = []  # (button_widget, label_actor) in row order
        for label, attr, color in specs:
            def make(a):
                def cb(state):
                    setattr(self, a, state)
                    self._dirty = True
                return cb
            btn = p.add_checkbox_button_widget(make(attr), value=getattr(self, attr),
                                               position=(0, 0), size=24, border_size=2,
                                               color_on=color, color_off=(0.3, 0.3, 0.3))
            lbl = p.add_text(label, position=(0, 0), font_size=9, color="white", name=f"_w_{attr}")
            self._widgets.append((btn, lbl))

        def _backend(state):
            self.cfg.cluster_backend = "open3d" if state else "sklearn"
            self._dirty = True
        btn = p.add_checkbox_button_widget(_backend, value=(self.cfg.cluster_backend == "open3d"),
                                           position=(0, 0), size=24, border_size=2,
                                           color_on=(0.13, 0.80, 0.53), color_off=(0.5, 0.5, 0.5))
        lbl = p.add_text("C++ clustering (C)", position=(0, 0), font_size=9, color="white", name="_w_backend")
        self._widgets.append((btn, lbl))
        self._reposition_widgets(p)

        def _voxel(val):
            self.cfg.cluster_voxel = float(val)
            self._dirty = True
        p.add_slider_widget(_voxel, rng=[0.0, 0.15], value=self.cfg.cluster_voxel,
                            title="voxel m (V)", pointa=(0.72, 0.09), pointb=(0.97, 0.09),
                            style="modern", fmt="%.2f")

    def _reposition_widgets(self, p) -> None:  # noqa: ANN001
        """Re-anchor the legend checkboxes/labels to the current window's right edge."""
        w, h = p.window_size
        btn_x = w - 44       # checkbox near the right edge
        lbl_x = w - 200      # label sits to the left of the checkbox
        y = h - 50
        for btn, lbl in self._widgets:
            btn.GetRepresentation().PlaceWidget([btn_x, btn_x + 24, y, y + 24, 0.0, 0.0])
            lbl.SetPosition(lbl_x, y + 3)
            y -= 32

    def _worker(self, source, pipeline) -> None:  # noqa: ANN001
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
        except Exception as exc:
            print(f"\n[viewer] source/pipeline stopped: {exc!r}")
        else:
            if self._running:
                print("\n[viewer] end of source, closing")
        self._running = False

    # ------------------------------------------------------------------
    def _rebuild(self, p, ann: FrameAnnotations) -> None:  # noqa: ANN001
        for name in self._dyn:
            p.remove_actor(name, render=False)
        self._dyn.clear()

        def add(mesh, name, **kw):
            p.add_mesh(mesh, name=name, render=False, **kw)
            self._dyn.append(name)

        frame = ann.frame
        pts = frame.points.astype(np.float64)
        gmask = ann.ground_mask if ann.ground_mask is not None else np.zeros(len(pts), bool)
        if not self._show_ground:
            keep = ~gmask
            pts, gmask = pts[keep], gmask[keep]
            intensity = frame.intensity[keep]
            ring = frame.ring[keep] if frame.ring is not None else None
        else:
            intensity = frame.intensity
            ring = frame.ring

        if len(pts) > 0:
            cloud = pv.PolyData(pts)
            if self._color_mode == "intensity":
                cloud["s"] = np.clip(intensity, 0, 255)
            elif self._color_mode == "ring" and ring is not None:
                cloud["s"] = ring.astype(float)
            else:  # height
                cloud["s"] = pts[:, 2]
            add(cloud, "_cloud", scalars="s", cmap="viridis", style="points",
                point_size=self.cfg.point_size, show_scalar_bar=False)

        thr = self.cfg.moving_speed_threshold
        moving_ids = {t.track_id for t in ann.tracks if t.speed >= thr}

        if self._show_clusters:
            for i, det in enumerate(ann.detections):
                if self._only_moving and det.track_id not in moving_ids:
                    continue
                lo, hi = det.bbox_min.astype(float), det.bbox_max.astype(float)
                box = pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
                col = _track_colour(det.track_id)
                add(box, f"_box{i}", style="wireframe", line_width=3, color=col)

        for t in ann.tracks:
            if self._only_moving and t.speed < thr:
                continue
            col = _track_colour(t.track_id)
            if self._show_tracks:
                add(pv.Sphere(radius=0.16, center=t.position.astype(float)),
                    f"_mk{t.track_id}", color=col)
                if t.speed > 0.05:
                    v = t.velocity.astype(float).copy(); v[2] = 0.0  # draw the horizontal velocity
                    add(pv.Arrow(start=t.position.astype(float), direction=v, scale=min(2.0, max(0.4, t.speed))),
                        f"_vel{t.track_id}", color=col)
            if self._show_trails and len(t.history) >= 2:
                pts_t = np.array([h.astype(float) for h in (list(t.history) + [t.position])])
                add(pv.lines_from_points(pts_t), f"_tr{t.track_id}", color=col, line_width=3)

        if self._show_control and ann.control is not None:
            aim = ann.control.get("aim_point")
            if aim is not None:
                line = pv.Line([0, 0, 1.0], np.asarray(aim, float))
                add(line, "_ctrl", color="#ff7000", line_width=4)

    def _update_hud(self, p, ann: FrameAnnotations) -> None:  # noqa: ANN001
        s = ann.stats
        vox = self.cfg.cluster_voxel
        lines = [
            f"proc {s.get('proc_fps', 0):.1f} Hz   pts {s.get('n_points', 0)}",
            f"clusters {s.get('n_clusters', 0)}   tracks {s.get('n_tracks', 0)}",
            f"colour: {self._color_mode}",
            f"cluster: {self.cfg.cluster_backend}  voxel {vox*100:.0f}cm  {s.get('proc_ms', 0):.0f}ms",
        ]
        if ann.control:
            tid = ann.control.get("target_id")
            err = ann.control.get("error_deg", 0.0)
            spd = ann.control.get("target_speed", 0.0)
            mv = "moving" if ann.control.get("target_moving") else "static"
            lines.append(f"gimbal -> target #{tid} ({mv} {spd:.1f} m/s)  err {err:+.0f} deg")
        if self._only_moving:
            lines.append("[only-moving ON]")
        p.add_text("\n".join(lines), position="upper_left", font_size=11,
                   color="white", name="_hud")

    def _print_status(self, ann: FrameAnnotations) -> None:
        s = ann.stats
        print(f"\rproc {s.get('proc_fps', 0):4.1f} Hz | pts {s.get('n_points', 0):6d} | "
              f"clusters {s.get('n_clusters', 0):2d} | tracks {s.get('n_tracks', 0):2d} "
              f" [{self._color_mode}]   ", end="", flush=True)
