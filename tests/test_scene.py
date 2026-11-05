"""The shared scene renders real frames off screen in every mode without error and fast enough."""
from __future__ import annotations

import time

import numpy as np
import pytest

from helios16p.config import DemoConfig
from helios16p.pipeline import ProcessingPipeline
from helios16p.sources.packet_file_source import PacketFileSource


@pytest.fixture(scope="module")
def annotations():
    from helios16p import bundled_clip
    cfg = DemoConfig()
    pipe = ProcessingPipeline(cfg)
    return cfg, [pipe.process(f) for f in PacketFileSource(str(bundled_clip()), max_frames=10)]


def _plotter():
    import pyvista as pv
    try:
        return pv.Plotter(off_screen=True, window_size=(640, 400))
    except Exception as exc:  # no GL at all
        pytest.skip(f"no off-screen rendering here: {exc}")


def test_scene_every_mode(annotations):
    from helios16p.viz.pyvista_viewer import Scene, fit_camera
    cfg, anns = annotations
    p = _plotter()
    sc = Scene(cfg)
    for a in anns[:3]:
        sc.rebuild(p, a)
    fit_camera(p, anns[2].frame.points)
    for mode in ("height", "intensity", "ring"):
        sc.color_mode = mode; sc.rebuild(p, anns[3]); p.render()
    for attr in ("show_ground", "show_grid", "show_clusters", "show_tracks", "show_labels", "show_trails", "show_control", "only_moving"):
        sc.toggle(attr); sc.rebuild(p, anns[4]); p.render()
        sc.toggle(attr)
    img = p.screenshot(return_img=True)
    assert img.shape[:2] == (400, 640) and img.std() > 5, "the render is not blank"
    m = Scene.metrics(anns[4], cfg)
    assert m["points"] == anns[4].frame.num_points and 0 <= m["ground_pct"] <= 100


def test_scene_rebuild_is_cheap(annotations):
    from helios16p.viz.pyvista_viewer import Scene
    cfg, anns = annotations
    p = _plotter()
    sc = Scene(cfg)
    sc.rebuild(p, anns[2]); p.render()
    t = time.perf_counter()
    for a in anns[3:]:
        sc.rebuild(p, a); p.render()
    per = (time.perf_counter() - t) / len(anns[3:])
    assert per < 0.08, f"{per * 1e3:.0f} ms per frame; the scene used to cost 80 ms when every object was its own actor"
