"""End-to-end integration smoke test (headless, no viewer window).

Runs the full ProcessingPipeline over (a) the synthetic scene and (b) the real
recorded mcap, exercising every module together through the real wiring
in helios16p.pipeline. The viewer's GUI window is NOT tested here (needs a
display); that is verified interactively on the workstation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helios16p.config import DemoConfig
from helios16p.pipeline import ProcessingPipeline
from helios16p.synthetic import synthetic_sequence
from helios16p.types import FrameAnnotations

REAL_BAG = os.environ.get("HELIOS16P_BAG", "recordings/helios16p_points")


def test_synthetic_pipeline() -> None:
    print("== synthetic ==")
    cfg = DemoConfig()
    pipe = ProcessingPipeline(cfg, enable_control=True)
    confirmed_ids = set()
    last = None
    for frame in synthetic_sequence(n_frames=40, fps=10.0, n_people=2):
        ann = pipe.process(frame)
        assert isinstance(ann, FrameAnnotations)
        for t in ann.tracks:
            if t.confirmed:
                confirmed_ids.add(t.track_id)
        last = ann
    print(f"   last frame: {last.stats}")
    if last.control:
        print(f"   control: target={last.control['target_id']} "
              f"err={last.control['error_deg']:.1f} deg")
    print(f"   distinct confirmed track ids over run: {sorted(confirmed_ids)}")
    assert confirmed_ids, "expected at least one confirmed track on the synthetic scene"
    print("   synthetic OK")


def test_real_bag_pipeline() -> None:
    if not Path(REAL_BAG).exists():
        import pytest
        pytest.skip(f"no bag at {REAL_BAG} (set HELIOS16P_BAG)")
    print("== real mcap ==")
    from helios16p.sources.mcap_source import McapSource

    cfg = DemoConfig()
    pipe = ProcessingPipeline(cfg, enable_control=True)
    n = 0
    for frame in McapSource(REAL_BAG, max_frames=15):
        ann = pipe.process(frame)
        assert isinstance(ann, FrameAnnotations)
        if n < 3 or n % 5 == 0:
            print(f"   f{ann.stats['frame_id']:>3}: pts={ann.stats['n_points']:>6} "
                  f"ground={ann.stats['n_ground']:>6} clusters={ann.stats['n_clusters']} "
                  f"tracks={ann.stats['n_tracks']} proc={ann.stats['proc_ms']:.1f}ms")
        n += 1
    assert n > 0, "no frames read from the real bag"
    print(f"   processed {n} real frames OK")


if __name__ == "__main__":
    test_synthetic_pipeline()
    test_real_bag_pipeline()
    print("PASS")
