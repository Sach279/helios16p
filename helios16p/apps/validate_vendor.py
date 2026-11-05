"""Compare this decoder with the vendor driver on the same packets.

Takes a raw recording of a static scene and the point cloud the vendor driver
(rs_driver) produced from the same packets, decodes the recording three ways and
reports, for a random sample of decoded points, the distance to the nearest
vendor point:

    bundled/difop   vertical and horizontal angles from the DIFOP (or the bundled file)
    vertical-only   DIFOP vertical angles, no horizontal offsets
    default-grid    the evenly spaced -15..+15 degree guess used before November 2026

    helios16p-validate --rec corridor1.npz --vendor-ply corridor1_vendor.ply

A correct decoder agrees with the vendor to a fraction of a millimetre; the
other two rows show what each missing table costs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..calibration import Calibration, parse_difop
from ..calibration import default as default_calibration
from ..sources.udp_source import decode_msop_packet

OLD_GRID = np.array([-15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])


def decode_all(msop: np.ndarray, calibration: Calibration, min_d: float = 0.1, max_d: float = 180.0) -> np.ndarray:
    chunks = []
    for row in msop:
        try:
            r = decode_msop_packet(row.tobytes(), calibration=calibration)
        except ValueError:
            continue
        d = np.linalg.norm(r["points"], axis=1)
        chunks.append(r["points"][(d >= min_d) & (d <= max_d)])
    return np.vstack(chunks) if chunks else np.zeros((0, 3), np.float32)


def nn_stats(pts: np.ndarray, vendor: np.ndarray, samples: int, seed: int = 0) -> dict:
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=min(samples, len(pts)), replace=False)
    d, _ = cKDTree(vendor).query(pts[idx].astype(np.float64), k=1, workers=-1)
    return {"n": int(len(d)), "median_mm": float(np.median(d) * 1e3), "p90_mm": float(np.percentile(d, 90) * 1e3),
            "p95_mm": float(np.percentile(d, 95) * 1e3), "max_mm": float(d.max() * 1e3),
            "within_1mm": float(np.mean(d < 1e-3)), "within_1cm": float(np.mean(d < 1e-2))}


def load_ply_xyz(path: Path) -> np.ndarray:
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def run(rec: Path, vendor_ply: Path, samples: int = 200_000, modes: tuple[str, ...] = ("difop", "vertical-only", "default-grid")) -> dict:
    z = np.load(rec)
    msop = z["msop"]
    cal = None
    for row in (z["difop"] if "difop" in z.files else []):
        cal = parse_difop(row.tobytes())
        if cal is not None:
            break
    cal = cal or default_calibration()
    variants = {
        "difop": cal,
        "vertical-only": Calibration(cal.vert_deg, np.zeros(16), source="vertical-only"),
        "default-grid": Calibration(OLD_GRID, np.zeros(16), source="default-grid"),
    }
    vendor = load_ply_xyz(vendor_ply)
    out = {"recording": rec.name, "vendor_ply": vendor_ply.name, "vendor_points": int(len(vendor)),
           "calibration_source": cal.source, "samples": samples, "results": {}}
    for m in modes:
        t = time.time()
        pts = decode_all(msop, variants[m])
        st = nn_stats(pts, vendor, samples)
        st["decoded_points"] = int(len(pts)); st["seconds"] = round(time.time() - t, 1)
        out["results"][m] = st
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", required=True, help="raw recording .npz of a static scene")
    ap.add_argument("--vendor-ply", required=True, help="vendor (rs_driver) cloud of the same packets")
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--json", default=None, help="also write the numbers here")
    args = ap.parse_args(argv)
    out = run(Path(args.rec), Path(args.vendor_ply), args.samples)
    print(f"{out['recording']} against {out['vendor_ply']} ({out['vendor_points']} vendor points), "
          f"calibration from {out['calibration_source']}, {out['samples']} sampled points")
    print(f"{'decode':15s} {'median':>9s} {'p90':>9s} {'p95':>9s} {'max':>9s} {'<1 mm':>7s} {'<1 cm':>7s}")
    for m, s in out["results"].items():
        print(f"{m:15s} {s['median_mm']:8.3f}mm {s['p90_mm']:8.3f}mm {s['p95_mm']:8.3f}mm {s['max_mm']:8.1f}mm "
              f"{s['within_1mm'] * 100:6.1f}% {s['within_1cm'] * 100:6.1f}%")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
