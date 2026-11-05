# 04 — Design Decisions

Eight decisions shaped the architecture of `helios16p`. Each entry gives the context (what problem needed solving) and the rationale (why this choice and not the obvious alternative).

---

## D1 — From-scratch numpy MSOP decoder (no vendor SDK at runtime)

**Context.** RoboSense ships the open-source `rs_driver` C++ library (BSD-licensed) and an `rslidar_sdk` ROS 2 wrapper around it. The vendor path works but requires: a sourced ROS 2 workspace, the rslidar_sdk package, and the rs_driver submodule — roughly 300 MB of build artefacts that must be compiled against a specific ROS distro.

**Decision.** We re-implemented the MSOP wire format from `rs_driver`'s `decoder_RSHELIOS.hpp` / `decoder_RSHELIOS_16P.hpp` / `decoder_mech.hpp` and implemented a pure-numpy decoder (`helios16p/sources/udp_source.py`). The decoder parses the 42-byte header, 12 blocks × 100 bytes each (block ID `FF EE`, azimuth u16 big-endian in centi-degrees, 32 channels × 3 bytes: distance u16 BE × 0.0025 m + intensity u8), applies the per-channel vertical angles and per-channel firing-time offsets, and converts spherical coordinates to Cartesian with the lens optical-centre offsets (Rx = 0.03498 m, Ry = −0.015 m).

**Rationale.** The decoder itself needs only numpy and the standard library (the package installs Open3D and VTK for processing and display). It is portable, auditable, and testable without the sensor or a ROS installation. The sensor's DIFOP calibration packets (port 7788) are applied as soon as one arrives, with a bundled fallback before that; see `07-calibration.md`.

---

## D2 — PyVista/VTK live viewer (not Open3D GUI)

**Context.** The development machine ran a Wayland session on Intel/Mesa graphics. Open3D offers two GUI paths:

- **Legacy GLFW viewer** (`open3d.visualization.draw_geometries`): Open3D bundles its own libGLFW compiled Wayland-only, without X11/XWayland fallback. The GLEW initialisation fails on this machine's native Wayland session.
- **New Filament-based GUI** (`open3d.visualization.gui`): crashes with a segfault on the Intel/Mesa Filament path.

Neither recovers gracefully; both fail silently or crash before rendering a frame.

**Decision.** The interactive viewer (`helios16p/viz/pyvista_viewer.py`) is built on PyVista/VTK. VTK renders via X11/GLX through XWayland — exactly the same path the project's Qt-based tools use — and works reliably.

**Rationale.** Open3D is still used where it works: clustering (its C++ DBSCAN), RANSAC, the optional second viewer and reading PLY files. Off-screen renders (`helios16p-snapshot`) use PyVista, and PLY export writes the file with numpy. Keeping Open3D for these roles retains its library strengths (compressed PLY writing, EGL headless) without exposing the broken GUI path.

The PyVista viewer runs the pipeline in a background worker thread and polls for new `FrameAnnotations` from the main VTK render loop, so the pipeline never blocks the UI and the UI never stalls the pipeline.

---

## D3 — Selectable clustering backend (sklearn / open3d) + voxel downsample

**Context.** DBSCAN on a raw Helios 16P frame (~24,000–33,000 non-ground points) is the dominant processing cost. Two obvious knobs exist: swap the DBSCAN implementation (Python/Cython sklearn vs. C++/OpenMP open3d), and reduce the input point count via voxel downsampling.

**Decision.** Both knobs are exposed as live toggles (keyboard `C` and `V`, or the on-screen widgets), backed by `DemoConfig.cluster_backend` and `DemoConfig.cluster_voxel`.

**Benchmark — real ~24k-point frame (`walk_one`), best of 3, via `tools/bench_clustering.py`:**

| backend | voxel (m) | ms/frame |
|---------|-----------|----------|
| sklearn | 0.00 (off) | 186 |
| open3d  | 0.00 (off) | 209 |
| sklearn | 0.05 | 38 |
| open3d  | 0.05 | **30** |
| sklearn | 0.10 | 22 |
| open3d  | 0.10 | **17** |

**Honest finding.** At full resolution, open3d is *not* faster than sklearn — it is ~12% slower. The Python↔C++ marshalling overhead (constructing a `o3d.geometry.PointCloud`, copying points as float64, receiving results as a Python list) erases the C++/OpenMP advantage when the point cloud is large. The real performance lever is the voxel downsample: a 5 cm voxel reduces the clustering input by roughly 5–6× and cuts latency from ~186 ms (sklearn, full res) to ~30–38 ms — a 5× improvement. With voxel 0.10 m, the combined latency drops to 17–22 ms.

**Implementation note.** Downsampling applies only to the DBSCAN *input*. Each original non-ground point inherits its voxel representative's cluster label via an `inv` index array, so `Detection.point_indices` always indexes the original full-resolution `frame.points` array. Bounding boxes, centroids, and the rendered point cloud are all full-resolution.

---

## D4 — Footprint filter (cluster_max_footprint = 1.2 m)

**Context.** Indoor environments contain walls, benches, and furniture that produce large merged clusters after DBSCAN. A height filter alone (0.3–2.5 m vertical extent) is insufficient: a 10 m wall segment has a sensible height but is clearly not a person.

**Decision.** After computing the axis-aligned bounding box of each cluster, the maximum horizontal extent — `max(bbox_max.x − bbox_min.x, bbox_max.y − bbox_min.y)` — is compared against `cfg.cluster_max_footprint` (default 1.2 m). Clusters wider than this are discarded.

**Rationale.** A standing or walking person is approximately 0.3–0.8 m across; 1.2 m provides a comfortable margin for clothing, bags, and centroid jitter without admitting furniture. Validated on `walk_one`: raw DBSCAN produced 7 clusters / 9 tracks; with the footprint filter applied, this reduced to ~3 clusters / 4 tracks, matching the number of people actually present.

---

## D5 — Motion-based targeting + moving-only filter

**Context.** A static indoor scene contains many confirmed tracks: chairs, pillars, equipment. A gimbal control loop that merely targets the "nearest" or "oldest" track would lock onto furniture.

**Decision.** `VirtualGimbal.update` (in `helios16p/control/gimbal.py`) selects its primary target from confirmed tracks with `speed >= moving_speed_threshold` (default 0.3 m/s). Among moving targets, it picks the fastest. The target is sticky — the same `track_id` is retained as long as it remains in the moving set — and falls back to the longest-lived confirmed track only when nothing is moving. The viewer's moving-only mode (key `N`) independently filters the rendered detections and tracks to the same dynamic subset.

**Rationale.** This closes the sensing → estimation → control story on the object that matters: a walking person. The virtual gimbal (orange ray in the 3D view) locks on, slews with proportional gain (P = 0.3) clamped to 5°/step, and reports the residual error in the HUD. In the `walk_frame.png` screenshot the walking person is isolated and the error reads ~0°.

---

## D6 — ROS-optional with lazy rclpy import

**Context.** The package should work in a plain Python venv without ROS, for replay, offline analysis, and deployment on machines without a ROS installation.

**Decision.** `import helios16p` never touches rclpy. The `RosSource` class in `helios16p/sources/ros_source.py` imports rclpy inside its `__init__` (validated immediately on construction) and inside `__iter__` (for the message type). The demo entrypoint (`scripts/demo.py`) uses `--source` to select the source and imports only the selected source module — so the rclpy import never executes unless `--source ros` is passed.

**Rationale.** A missing optional dependency (rclpy, sensor_msgs) raises a clear `ImportError` with an actionable message ("source /opt/ros/jazzy/setup.bash") rather than a cryptic import-time traceback on startup. All other sources — synthetic, mcap, udp, file — work in any Python 3.12 venv with the package installed.

---

## D7 — Raw-packet recording format (.npz of 1248-byte MSOP packets)

**Context.** The alternative recording path (ROS) uses `ros2 bag record` to write an mcap containing decoded `sensor_msgs/PointCloud2` messages. This is convenient for RViz but: (1) requires the vendor rslidar_sdk ROS node running during capture, (2) commits to the decoded representation (floating-point XYZ), and (3) mcap reading requires the `mcap` Python library and familiarity with ROS message schemas.

**Decision.** `helios16p-record` binds the MSOP port (6699) and DIFOP port (7788) directly via UDP sockets, collects the raw 1248-byte packets with host timestamps, and saves them to a compressed `.npz` (arrays: `msop` uint8 (N×1248), `msop_ts` float64, `difop` uint8 (M×1248), `difop_ts` float64). Replay is via `PacketFileSource` (`--source file --rec recording.npz`), which feeds the packets through the exact same numpy decoder as the live path.

**Rationale.** Full fidelity: the recording preserves the raw wire bytes, so future decoder improvements (e.g. absorbing DIFOP vertical-angle calibration) can be applied to old recordings without re-capturing. No runtime ROS dependency at record or replay time. Format is self-describing numpy, inspectable with a single `np.load`. A 30-second recording at ~750 packets/s produces ~750 × 30 × 1248 ≈ 28 MB uncompressed; compressed .npz is about 70 % of that.

---

## D8 — PLY export (`helios16p-export`, viewer key P)

**Context.** Analysis and ground-truth validation workflows (CloudCompare, MeshLab, comparison against photogrammetry reconstructions) require a standard point-cloud interchange format.

**Decision.** Two export paths:

- **Viewer shortcut** (`P`): exports the current frame as a height-coloured PLY using Open3D's file writer.
- **Batch tool** (`helios16p-export`): accepts a `.npz` recording and exports frames as individual PLY files or merges them into a single dense cloud. Colouring is selectable (`--color height` or `--color intensity`).

**Rationale.** Height-coloured PLY files open directly in CloudCompare and other standard tools. The merge mode (`--merge`) accumulates multiple revolutions into one dense cloud: the `room_dense.png` figure was produced from 60 merged revolutions (~1.4 M points). The export tool uses Open3D's `io.write_point_cloud` (binary PLY with normal stripping suppressed) for compact, widely-compatible output.
