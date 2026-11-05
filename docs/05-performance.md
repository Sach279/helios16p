# 05 — Performance

All numbers in this document are from measurements on real `walk_one` capture data (~24k points per revolution) unless otherwise noted. Timings are wall-clock, best-of-3, single-threaded processing on the development machine (Intel Lunar Lake).

---

## Pipeline stages and their costs

The `ProcessingPipeline.process(Frame) -> FrameAnnotations` call chains four stages:

| Stage | Implementation | Typical cost |
|-------|----------------|-------------|
| Ground segmentation | RANSAC plane fit (open3d fast path, 200 iterations on z-filtered candidates) | ~10 ms/frame |
| Clustering | DBSCAN with voxel downsample — see table below | 17–209 ms depending on config |
| Tracking | Hungarian assignment + Kalman predict/update (scipy) | < 1 ms (negligible) |
| Gimbal control | Proportional slew, pure Python | < 0.1 ms (negligible) |

Rendering is separate from processing (worker thread + main VTK thread):

| Rendering step | Cost |
|----------------|------|
| `pv.PolyData` build from numpy array | ~2 ms |
| VTK render call (PyVista plotter update) | tens of ms (display-rate bound, not pipeline-bound) |

**Rendering is not the bottleneck.** The pipeline runs in a background thread. The VTK render loop polls for new `FrameAnnotations` at ~10 ms intervals and rebuilds actors only when a new frame arrives. At real-time processing rates the display updates comfortably faster than the sensor's 10 Hz spin rate.

---

## Clustering benchmark

Measured on a real ~24k-point frame (`walk_one`), best of 3, using `tools/bench_clustering.py`:

| backend | voxel (m) | ms/frame |
|---------|-----------|----------|
| sklearn | 0.00 (off) | 186 |
| open3d  | 0.00 (off) | 209 |
| sklearn | 0.05 | 38 |
| open3d  | 0.05 | **30** |
| sklearn | 0.10 | 22 |
| open3d  | 0.10 | **17** |

---

## The real-time story

**Naive baseline (about 5 Hz).** Running DBSCAN at full resolution on ~25k non-ground points is DBSCAN-bound. At ~186–209 ms per frame, the pipeline achieves about 5 Hz — below the sensor's 10 Hz spin rate.

**With voxel 0.05 m and open3d backend (~20–30 Hz processing).** A 5 cm voxel reduces the clustering input from ~24k non-ground points to roughly 4–5k voxel representatives, cutting the DBSCAN step from ~200 ms to ~30 ms. Combined with ~10 ms ground RANSAC, total per-frame cost is approximately 40–45 ms, giving ~20–30 Hz sustainable processing throughput — comfortably ahead of the sensor's 10 Hz delivery rate and leaving headroom for tracking and display.

**The full-resolution cloud is always displayed.** The voxel downsample applies only to the DBSCAN input. Every non-ground point inherits its voxel's cluster label via the `inv` index array, so:
- `Detection.point_indices` indexes the original `frame.points` array at full resolution.
- Bounding boxes and centroids are computed on full-resolution cluster members.
- The rendered point cloud in the viewer shows all points from the frame, coloured by height, intensity, or ring.

This means the user sees a full-density scan while benefiting from faster clustering — there is no visual quality penalty from the voxel downsample.

---

## Configuration defaults

Default values (from `DemoConfig` in `helios16p/config.py`) that directly affect performance:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `cluster_backend` | `"open3d"` | C++/OpenMP DBSCAN (fastest at voxel > 0) |
| `cluster_voxel` | `0.05` m | 5 cm voxel — the real-time enabler |
| `ground_distance_threshold` | `0.10` m | RANSAC inlier band |
| `ground_max_z` | `0.3` m | Candidate pre-filter height (keeps RANSAC fast) |
| `cluster_eps` | `0.5` m | DBSCAN neighbourhood radius |
| `cluster_min_points` | `15` | Minimum cluster size |
| `track_max_age` | `10` frames | Track persistence |
| `track_min_hits` | `3` | Hits before track is confirmed |
| `track_gate_distance` | `1.5` m | Hungarian assignment gate |

---

## Notes on the open3d vs. sklearn finding

At full resolution (voxel off), open3d is *slower* than sklearn (209 ms vs. 186 ms). The overhead comes from constructing an `o3d.geometry.PointCloud`, copying ~24k points as float64 into the C++ object, running DBSCAN, and receiving the label array back as a Python list. When the input is large, this marshalling cost exceeds the gain from C++ parallel processing.

At reduced resolution (voxel 0.05–0.10 m), the marshalling cost is proportionally smaller and the C++/OpenMP advantage becomes visible: open3d is 1.25–1.3× faster than sklearn at voxel 0.05 m and 1.3× faster at voxel 0.10 m. This is why the default is `cluster_backend="open3d"` combined with `cluster_voxel=0.05`: the two settings work together.
