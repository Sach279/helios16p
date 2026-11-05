"""Parity with the vendor driver on a full station recording (slow; needs local data).

Set HELIOS16P_PARITY_REC (raw .npz) and HELIOS16P_PARITY_PLY (rs_driver cloud of
the same packets) to run it elsewhere; it is skipped when the files are absent.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

REC = Path(os.environ.get("HELIOS16P_PARITY_REC", "recordings/parity/corridor1_20260703_170202.npz"))
PLY = Path(os.environ.get("HELIOS16P_PARITY_PLY", "recordings/parity/corridor1_20260703_170202_vendor.ply"))

FIXTURE = Path(__file__).resolve().parent / "data" / "parity_corridor1_150pkts.npz"


def test_fixture_matches_vendor_to_float32_rounding():
    """150 real packets against the rs_driver points nearest to them (shipped fixture, runs everywhere)."""
    import numpy as np
    from scipy.spatial import cKDTree
    from helios16p.calibration import Calibration, parse_difop
    from helios16p.apps.validate_vendor import OLD_GRID, decode_all
    z = np.load(FIXTURE)
    cal = parse_difop(z["difop"][0].tobytes())
    assert cal is not None
    vendor = z["vendor_xyz"].astype(np.float64)
    tree = cKDTree(vendor)
    good = decode_all(z["msop"], cal)
    d, _ = tree.query(good.astype(np.float64), k=1)
    assert np.median(d) < 1e-6 and np.mean(d < 1e-3) > 0.999, (np.median(d), np.mean(d < 1e-3))
    old = decode_all(z["msop"], Calibration(OLD_GRID, np.zeros(16), source="old"))
    d_old, _ = tree.query(old.astype(np.float64), k=1)
    assert np.median(d_old) > 0.05, np.median(d_old)


@pytest.mark.slow
@pytest.mark.skipif(not (REC.exists() and PLY.exists()), reason="vendor parity data not on this machine")
def test_decoder_matches_vendor_within_a_millimetre():
    from helios16p.apps.validate_vendor import run
    out = run(REC, PLY, samples=50_000, modes=("difop", "default-grid"))
    good = out["results"]["difop"]
    old = out["results"]["default-grid"]
    assert good["median_mm"] < 0.5, good
    assert good["within_1mm"] > 0.99, good
    assert old["median_mm"] > 50, old  # the pre-fix decoder: 112 mm to the nearest vendor point, 0.56 m point to point
