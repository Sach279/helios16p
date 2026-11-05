"""Qt window: the 3D scene with a side panel of metrics, buttons and tooltips.

Needs PySide6 and pyvistaqt (installed with the package).  The scene itself is
``helios16p.viz.pyvista_viewer.Scene``; this file is only the panel around it.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import DemoConfig
from ..types import FrameAnnotations
from .pyvista_viewer import BG, Scene, _Runner, fit_camera, save_frame_ply

PANEL_QSS = """
QWidget { background: #1b1b24; color: #d8d8e0; font-size: 12px; }
QGroupBox { border: 1px solid #2e2e40; border-radius: 6px; margin-top: 14px; padding: 8px 6px 6px 6px; font-weight: 600; color: #9a9ab0; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QPushButton { background: #262636; border: 1px solid #3a3a50; border-radius: 5px; padding: 5px 8px; }
QPushButton:hover { background: #30304a; border-color: #5a5a80; }
QPushButton:checked { background: #1f5f6b; border-color: #2d8a9b; color: white; }
QCheckBox { spacing: 6px; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #5a5a80; border-radius: 3px; background: #262636; }
QCheckBox::indicator:checked { background: #2d8a9b; border-color: #2d8a9b; }
QLabel#value { font-size: 15px; font-weight: 600; color: #ffffff; }
QLabel#key { color: #8a8aa0; }
QLabel#title { font-size: 16px; font-weight: 700; color: #ffffff; }
QLabel#sub { color: #8a8aa0; }
QLabel#help { color: #8a8aa0; font-size: 11px; }
QSlider::groove:horizontal { height: 4px; background: #2e2e40; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -6px 0; background: #2d8a9b; border-radius: 7px; }
QComboBox { background: #262636; border: 1px solid #3a3a50; border-radius: 5px; padding: 3px 6px; }
QStatusBar { background: #15151d; color: #9a9ab0; }
"""

HELP = ("Drag to orbit, wheel to zoom, right-drag to pan.\n"
        "1 2 3 colour   R refit   G ground   B boxes   K tracks\n"
        "L trails   T labels   M ray   N moving only   Space pause\n"
        "C backend   P save PLY   S screenshot   Q quit")


class QtViewer:
    """Same call as the plain window: ``QtViewer(cfg).run(source, pipeline)``."""

    def __init__(self, cfg: DemoConfig, color_mode: str = "height", only_moving: bool = False,
                 screenshot: str | None = None, screenshot_after: float = 3.0, source_label: str = "") -> None:
        self.cfg = cfg
        self.scene = Scene(cfg, color_mode, only_moving)
        self.runner = _Runner(cfg)
        self.source_label = source_label
        self._screenshot = screenshot
        self._screenshot_after = screenshot_after
        self._last: FrameAnnotations | None = None
        self._dirty = False
        self._first = True

    # ------------------------------------------------------------------
    def run(self, source, pipeline) -> None:  # noqa: ANN001
        from PySide6 import QtCore, QtGui, QtWidgets
        from pyvistaqt import QtInteractor

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = QtWidgets.QMainWindow()
        win.setWindowTitle("Helios 16P")
        win.resize(1500, 900)
        central = QtWidgets.QWidget(); win.setCentralWidget(central)
        lay = QtWidgets.QHBoxLayout(central); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)

        self.plotter = p = QtInteractor(central)
        p.set_background(BG)
        lay.addWidget(p, stretch=1)

        panel = QtWidgets.QWidget(); panel.setFixedWidth(340); panel.setStyleSheet(PANEL_QSS)
        pl = QtWidgets.QVBoxLayout(panel); pl.setContentsMargins(12, 10, 12, 8); pl.setSpacing(8)
        lay.addWidget(panel)
        status = QtWidgets.QStatusBar(); win.setStatusBar(status)
        status.setStyleSheet(PANEL_QSS)
        self._status = status

        hints = {}

        class _Hover(QtCore.QObject):
            """Show the short hint in the status bar while the mouse is over a control."""
            def eventFilter(self, obj, ev):
                if ev.type() == QtCore.QEvent.Enter and obj in hints:
                    status.showMessage(hints[obj])
                elif ev.type() == QtCore.QEvent.Leave and obj in hints:
                    status.clearMessage()
                return False
        self._hover = _Hover(win)

        def tip(w, short: str, long: str):
            w.setToolTip(long); hints[w] = short; w.installEventFilter(self._hover)

        # header
        title = QtWidgets.QLabel("Helios 16P"); title.setObjectName("title"); pl.addWidget(title)
        sub = QtWidgets.QLabel(self.source_label or ""); sub.setObjectName("sub"); sub.setWordWrap(True); pl.addWidget(sub)
        self._sub = sub

        # metrics
        g = QtWidgets.QGroupBox("Metrics"); grid = QtWidgets.QGridLayout(g); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(4)
        self._vals: dict[str, QtWidgets.QLabel] = {}
        rows = [("points", "Points", "Returns in this revolution after the range window (28,800 is the sensor's maximum at 600 rpm)."),
                ("ground", "Ground", "Share of points on the RANSAC ground plane. Press G to hide them."),
                ("objects", "Objects", "Clusters above the ground this frame (DBSCAN on a voxel-downsampled cloud)."),
                ("tracks", "Tracks", "Confirmed Kalman tracks; in brackets, those moving faster than the moving threshold."),
                ("proc", "Processing", "Pipeline time per frame and the resulting rate; the sensor spins at 10 Hz."),
                ("target", "Gimbal", "The track the virtual gimbal follows, its speed and the remaining pointing error.")]
        for r, (key, label, long) in enumerate(rows):
            k = QtWidgets.QLabel(label); k.setObjectName("key"); v = QtWidgets.QLabel("–"); v.setObjectName("value")
            tip(k, long, long); tip(v, long, long)
            grid.addWidget(k, r, 0); grid.addWidget(v, r, 1, alignment=QtCore.Qt.AlignRight)
            self._vals[key] = v
        grid.setColumnStretch(1, 1); grid.setColumnMinimumWidth(1, 190)
        pl.addWidget(g)

        # view
        g = QtWidgets.QGroupBox("View"); vl = QtWidgets.QVBoxLayout(g); vl.setSpacing(6)
        row = QtWidgets.QHBoxLayout(); self._color_btns = {}
        for mode, label, long in [("height", "Height", "Colour by z, metres above the sensor (key 1)."),
                                  ("intensity", "Intensity", "Colour by return intensity, 0 to 255 (key 2)."),
                                  ("ring", "Beam", "Colour by beam index, 0 is the lowest of the sixteen (key 3).")]:
            b = QtWidgets.QPushButton(label); b.setCheckable(True); b.setChecked(mode == self.scene.color_mode)
            tip(b, long, long); b.clicked.connect(lambda _c=False, m=mode: self._set_color(m)); row.addWidget(b)
            self._color_btns[mode] = b
        vl.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        for view, label, long in [("iso", "Iso", "Three-quarter view, framed on the cloud (key R)."), ("top", "Top", "Plan view from above."),
                                  ("front", "Front", "Looking along the sensor's forward axis."), ("side", "Side", "Looking across the sensor.")]:
            b = QtWidgets.QPushButton(label); tip(b, long, long)
            b.clicked.connect(lambda _c=False, v=view: self._refit(v)); row.addWidget(b)
        vl.addLayout(row)
        row = QtWidgets.QHBoxLayout(); row.addWidget(QtWidgets.QLabel("Point size"))
        sl = QtWidgets.QSlider(QtCore.Qt.Horizontal); sl.setRange(1, 6); sl.setValue(int(self.cfg.point_size))
        tip(sl, "Screen size of each point.", "Screen size of each point in pixels; larger reads better from afar, smaller shows structure.")
        sl.valueChanged.connect(self._point_size); row.addWidget(sl); vl.addLayout(row)
        pl.addWidget(g)

        # objects
        g = QtWidgets.QGroupBox("Show"); ol = QtWidgets.QGridLayout(g); ol.setVerticalSpacing(4)
        self._checks: dict[str, QtWidgets.QCheckBox] = {}
        checks = [("show_ground", "Ground points", "Points on the fitted ground plane (G)."),
                  ("show_grid", "Grid", "A 1 m grid drawn at the fitted ground height."),
                  ("show_clusters", "Object boxes", "Axis-aligned boxes around each cluster, coloured by track (B)."),
                  ("show_tracks", "Tracks", "A sphere at each confirmed track with its horizontal velocity arrow (K)."),
                  ("show_labels", "Labels", "Track id and speed next to each confirmed track (T)."),
                  ("show_trails", "Trails", "The last positions of each track (L)."),
                  ("show_control", "Gimbal ray", "The virtual gimbal's aim ray, shown while it follows a moving track (M)."),
                  ("only_moving", "Moving only", "Hide everything that is not moving; the static scene stays as a grey shadow (N).")]
        for i, (attr, label, long) in enumerate(checks):
            c = QtWidgets.QCheckBox(label); c.setChecked(getattr(self.scene, attr)); tip(c, long, long)
            c.toggled.connect(lambda state, a=attr: self._set_flag(a, state)); ol.addWidget(c, i // 2, i % 2)
            self._checks[attr] = c
        pl.addWidget(g)

        # processing
        g = QtWidgets.QGroupBox("Processing"); prl = QtWidgets.QVBoxLayout(g); prl.setSpacing(6)
        row = QtWidgets.QHBoxLayout(); row.addWidget(QtWidgets.QLabel("Clustering"))
        cb = QtWidgets.QComboBox(); cb.addItems(["Open3D (C++)", "scikit-learn"]); cb.setCurrentIndex(0 if self.cfg.cluster_backend == "open3d" else 1)
        tip(cb, "DBSCAN implementation.", "Which DBSCAN runs: Open3D's C++ is faster on dense clouds, scikit-learn is the pure-Python reference (key C).")
        cb.currentIndexChanged.connect(self._backend); row.addWidget(cb); prl.addLayout(row); self._backend_box = cb
        row = QtWidgets.QHBoxLayout(); lab = QtWidgets.QLabel(f"Voxel {self.cfg.cluster_voxel * 100:.0f} cm"); row.addWidget(lab)
        vs = QtWidgets.QSlider(QtCore.Qt.Horizontal); vs.setRange(0, 15); vs.setValue(int(round(self.cfg.cluster_voxel * 100)))
        tip(vs, "Voxel size before clustering; 0 is off.", "The cloud is downsampled to this voxel size before DBSCAN; 5 cm keeps people and halves the time, 0 clusters every point.")
        vs.valueChanged.connect(lambda v: (self._voxel(v), lab.setText(f"Voxel {v} cm"))); row.addWidget(vs); prl.addLayout(row)
        row = QtWidgets.QHBoxLayout(); lab2 = QtWidgets.QLabel(f"Moving > {self.cfg.moving_speed_threshold:.1f} m/s"); row.addWidget(lab2)
        ms = QtWidgets.QSlider(QtCore.Qt.Horizontal); ms.setRange(0, 20); ms.setValue(int(round(self.cfg.moving_speed_threshold * 10)))
        tip(ms, "Speed above which a track counts as moving.", "Tracks faster than this are 'moving': they get the gimbal's attention and survive the Moving-only filter.")
        ms.valueChanged.connect(lambda v: (self._moving_thr(v), lab2.setText(f"Moving > {v / 10:.1f} m/s"))); row.addWidget(ms); prl.addLayout(row)
        pl.addWidget(g)

        # actions
        row = QtWidgets.QHBoxLayout()
        self._pause_btn = b = QtWidgets.QPushButton("Pause"); b.setCheckable(True)
        tip(b, "Pause or resume the source (Space).", "Stops pulling frames from the source; the last frame stays on screen and can still be explored, saved or screenshotted.")
        b.toggled.connect(self._pause); row.addWidget(b)
        b = QtWidgets.QPushButton("Save PLY"); tip(b, "Write this frame to ./recordings/ply (P).", "Writes the current frame as a binary PLY with x y z, intensity, ring and colour, readable by CloudCompare and MeshLab.")
        b.clicked.connect(self._save_ply); row.addWidget(b)
        b = QtWidgets.QPushButton("Screenshot"); tip(b, "Save the window to ./recordings/shots (S).", "Saves a PNG of the whole window, panel included.")
        b.clicked.connect(lambda: self._shot(None)); row.addWidget(b)
        pl.addLayout(row)

        help_lbl = QtWidgets.QLabel(HELP); help_lbl.setObjectName("help"); help_lbl.setWordWrap(True); pl.addWidget(help_lbl)
        pl.addStretch(1)

        # shortcuts
        def sc(key, fn):
            QtGui.QShortcut(QtGui.QKeySequence(key), win, activated=fn)
        sc("1", lambda: self._set_color("height")); sc("2", lambda: self._set_color("intensity")); sc("3", lambda: self._set_color("ring"))
        for key, attr in [("G", "show_ground"), ("B", "show_clusters"), ("K", "show_tracks"), ("L", "show_trails"),
                          ("M", "show_control"), ("N", "only_moving"), ("T", "show_labels")]:
            sc(key, lambda a=attr: self._checks[a].toggle())
        sc("R", lambda: self._refit("iso")); sc("C", lambda: self._backend_box.setCurrentIndex(1 - self._backend_box.currentIndex()))
        sc("P", self._save_ply); sc("S", lambda: self._shot(None)); sc("Space", lambda: self._pause_btn.toggle())
        sc("Q", win.close); sc("Ctrl+Q", win.close)

        # run
        self._win = win
        self._frame_lbl = QtWidgets.QLabel(""); status.addPermanentWidget(self._frame_lbl)
        timer = QtCore.QTimer(); timer.setInterval(25)
        t_open = time.perf_counter(); shot = {"done": False}

        def tick():
            ann = self.runner.take()
            if ann is None and self._dirty and self._last is not None:
                ann = self._last
            if ann is not None:
                self._dirty = False
                self._last = ann
                self.scene.rebuild(p, ann)
                self._update_metrics(ann)
                if self._first:
                    fit_camera(p, ann.frame.points); self._first = False
                p.render()
            if self._screenshot and not shot["done"] and not self._first and time.perf_counter() - t_open >= self._screenshot_after:
                shot["done"] = True; self._shot(self._screenshot)
            if self.runner.finished and not self.runner.has_new:
                status.showMessage("end of source" if not self.runner.error else f"source stopped: {self.runner.error}")
                if self._screenshot and not shot["done"]:
                    shot["done"] = True; self._shot(self._screenshot)
                timer.stop()
                if self._screenshot or self.cfg.frame_rate <= 0:
                    win.close()
        timer.timeout.connect(tick); timer.start()
        status.showMessage("starting the pipeline (the first frame takes a few seconds: Open3D loads)...")

        win.show()
        app.processEvents()
        secs = pipeline.warmup()  # on the main thread, window already visible
        status.showMessage(f"pipeline ready in {secs:.1f} s; hover a control for what it does", 5000)
        worker = self.runner.start(source, pipeline)
        try:
            app.exec()
        finally:
            self.runner.running = False
            worker.join(timeout=3.0)
            try:
                p.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _set_color(self, mode: str) -> None:
        self.scene.color_mode = mode; self._dirty = True
        for m, b in self._color_btns.items():
            b.setChecked(m == mode)

    def _set_flag(self, attr: str, state: bool) -> None:
        setattr(self.scene, attr, bool(state)); self._dirty = True

    def _refit(self, view: str) -> None:
        if self._last is not None:
            fit_camera(self.plotter, self._last.frame.points, view); self.plotter.render()

    def _point_size(self, v: int) -> None:
        self.cfg.point_size = float(v); self._dirty = True

    def _backend(self, idx: int) -> None:
        self.cfg.cluster_backend = "open3d" if idx == 0 else "sklearn"; self._dirty = True

    def _voxel(self, v: int) -> None:
        self.cfg.cluster_voxel = v / 100.0; self._dirty = True

    def _moving_thr(self, v: int) -> None:
        self.cfg.moving_speed_threshold = v / 10.0; self._dirty = True

    def _pause(self, state: bool) -> None:
        self.runner.paused = bool(state)
        self._pause_btn.setText("Resume" if state else "Pause")
        self._status.showMessage("paused: the last frame stays, explore, save or screenshot it" if state else "running", 3000)

    def _save_ply(self) -> None:
        if self._last is None:
            self._status.showMessage("no frame yet"); return
        try:
            f = save_frame_ply(self._last); self._status.showMessage(f"saved {self._last.frame.num_points} points to {f}")
        except OSError as exc:
            self._status.showMessage(f"could not write PLY: {exc}")

    def _shot(self, path: str | None) -> None:
        f = Path(path) if path else Path("recordings/shots") / f"shot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        f.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._win.grab().save(str(f)); self._status.showMessage(f"screenshot saved to {f}")
        except Exception as exc:  # noqa: BLE001
            self._status.showMessage(f"screenshot failed: {exc}")

    def _update_metrics(self, ann: FrameAnnotations) -> None:
        m = Scene.metrics(ann, self.cfg)
        self._vals["points"].setText(f"{m['points']:,}")
        self._vals["ground"].setText(f"{m['ground_pct']:.0f} %")
        self._vals["objects"].setText(str(m["objects"]))
        self._vals["tracks"].setText(f"{m['tracks']}  ({m['moving']} moving)")
        self._vals["proc"].setText(f"{m['proc_ms']:.0f} ms, {m['proc_hz']:.1f} Hz")
        if m["target"]:
            tid, spd, err = m["target"]
            self._vals["target"].setText(f"#{tid}  {spd:.1f} m/s  err {err:+.0f}°")
        else:
            self._vals["target"].setText("no moving target")
        self._frame_lbl.setText(f"frame {m['frame']}   {m['proc_hz']:.1f} frames/s" + ("   paused" if self.runner.paused else "") + "  ")

