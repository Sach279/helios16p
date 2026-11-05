# helios16p — System Internals

This document describes each stage of the stack in detail, including the design
rationale for every major decision. These are standard
techniques, competently integrated. The value is the self-contained, ROS-free
pipeline, not algorithmic novelty.

---

## 1. The from-scratch numpy MSOP decoder

**File:** `helios16p/sources/udp_source.py` — `decode_msop_packet()`

**Why not the vendor SDK?**
The vendor `rslidar_sdk` is a ROS 2 package that wraps `rs_driver`, a C++ library.
Pulling it in at runtime would mean depending on a full ROS installation, a compiled
shared library, and a platform-specific build. The BSD-licensed `rs_driver` source
(RoboSense's open-source `rs_driver`, github.com/RoboSense-LiDAR/rs_driver) documents the
wire protocol completely. Re-implementing it in Python with numpy costs nothing at
runtime and produces a decoder that is portable, testable without the sensor, and
transparent to inspection.

**Packet layout.** Each MSOP UDP datagram is exactly 1248 bytes:

```
Offset      Length    Contents
------      ------    --------
  0–3          4      MSOP magic: 55 AA 05 5A
  4–41        38      Header (id[4] already counted; protocol_version[2],
                       reserved[14], RSTimestampUTC[10], lidar_type[1],
                       reserved[7], RSTemperature[2], reserved[2])
42–1241      1200     12 data blocks × 100 bytes each
1242–1245      4      Index (uint32)
1246–1247      2      Tail (uint16)
```

Each 100-byte block has the sub-structure:

```
Block offset   Length   Contents
------------   ------   --------
  0–1            2      Block magic: FF EE
  2–3            2      Azimuth, uint16 big-endian, centi-degrees (0..35999)
  4–99          96      32 channels × 3 bytes:
                          [0:2] Distance, uint16 big-endian → × 0.0025 m
                          [2]   Intensity, uint8, 0..255
```

**16P block specifics.** The Helios 16P has 16 lasers but packs two firings into
each block (`CHANNELS_PER_BLOCK = 32`). Laser index = `chan % 16`. Azimuth
interpolation within a block uses pre-computed per-channel fractions
(`CHAN_AZIS[chan] = FIRING_TSS_US[chan] / (BLOCK_DURATION_US × 2)`) following
`Rs16SingleReturnBlockIterator` in `rs_driver`.

**Decoding path.** The decoder is fully vectorised with numpy structured dtypes:

1. Cast the 1200-byte block region as a structured array of 12 block records; each
   record contains the azimuth and a (32,) array of `{distance u16, intensity u8}`
   channel records. One `np.frombuffer()` call, no Python loops over blocks.
2. Compute per-channel interpolated azimuths: `blk_az + az_diff × CHAN_AZIS`.
   `az_diff` accounts for the FOV blind zone (if the gap between consecutive block
   azimuths exceeds 100 centi-degrees the default step of 40 centi-degrees is used).
3. Add the per-laser horizontal angle corrections of the calibration
   (`horiz_angle_cali`, centi-degrees); without them a beam sits several degrees off.
4. Convert distances: `dist_m = raw_u16 × 0.0025`.
5. Apply the spherical → Cartesian geometry from `decoder_RSHELIOS_16P.hpp`:
   ```
   x =  d · cos(vert) · cos(az) + RX · cos(az_raw)
   y = −d · cos(vert) · sin(az) − RX · sin(az_raw)
   z =  d · sin(vert)
   ```
   with lens offset `RX = 0.03498 m`, `RZ = 0.0`. All six trigonometric arrays
   broadcast over the (12, 32) block/channel grid.
6. Flatten to (384,) vectors, apply distance validity filter, return dict of
   `points (M,3)`, `intensity (M,)`, `ring (M,)`, `azimuth_deg (M,)`.

**Angle calibration.** Until the first DIFOP packet arrives, the decoder uses
the calibration bundled with the package (`helios16p.calibration.default()`,
read from our unit). `helios16p.calibration.parse_difop()` reads the sixteen
valid entries of `vert_angle_cali` (byte 468) and `horiz_angle_cali` (byte 564)
of the DIFOP payload (sign byte + uint16 big-endian value in 0.01°); an entry
with sign byte 0xFF (uncalibrated) or a value out of range (−90°..+180°)
invalidates the packet. Both tables are applied: the vertical one gives each
wire laser its elevation (the beams fire interleaved, +13, +15, +9, +11, ...),
the horizontal one its azimuth offset (about +3.1° and −7.5° alternating on our
unit). See `docs/07-calibration.md` for the validation against the vendor driver.

**Frame assembly.** `UdpSource` runs a dedicated reader thread that calls
`decode_msop_packet()` for every incoming packet and accumulates points in Python
lists. A new frame is triggered when the first-block azimuth of a packet is
smaller than the first-block azimuth of the previous packet, i.e. the scanner has
wrapped back through 0° (the replay source uses the same rule). Completed frames are pushed onto a four-slot
`queue.Queue`; if the consumer lags, the oldest frame is dropped so the socket
buffer never fills. The MSOP socket is set with `SO_RCVBUF = 16 MiB` to absorb
burst arrivals.

---

## 2. Ground segmentation — RANSAC plane fit

**File:** `helios16p/processing/ground.py` — `segment_ground()`

**Config knobs:**

| Parameter | Default | Meaning |
|---|---|---|
| `ground_max_z` | 0.3 m | Pre-filter: only points below this z are RANSAC candidates |
| `ground_distance_threshold` | 0.10 m | RANSAC inlier band (point-to-plane distance) |

**Why RANSAC?** The ground plane in a typical indoor LiDAR scan is not perfectly
horizontal (sensor tilt, uneven floors) and is mixed with near-floor clutter. A
robust estimator that tolerates outliers is the natural choice. The implementation
is 200-iteration RANSAC, using `open3d.geometry.PointCloud.segment_plane()` when
Open3D is available (fast C++ RANSAC), with a pure-numpy fallback (`_fit_plane_numpy`)
for environments without Open3D.

**Additional validity check.** After fitting, the plane is accepted only if its
normal vector points near-vertically: `|n_z| > cos(15°) ≈ 0.966`. This rejects
spurious planes fitted to walls or furniture.

**Output.** A boolean mask `(N,)` over `Frame.points`; True = ground point. This
mask is threaded through all downstream stages: the viewer can colour or hide
ground points; the clustering stage uses it to select the non-ground subset.

**Measured cost:** approximately 10 ms per frame on the test hardware.

---

## 3. Object clustering — DBSCAN with voxel downsample

**File:** `helios16p/processing/clustering.py` — `cluster_objects()`

**Config knobs:**

| Parameter | Default | Meaning |
|---|---|---|
| `cluster_backend` | `"open3d"` | DBSCAN implementation: `"open3d"` (C++/OpenMP) or `"sklearn"` |
| `cluster_voxel` | 0.05 m | Voxel edge length for downsampling clustering input; 0 = off |
| `cluster_eps` | 0.5 m | DBSCAN neighbourhood radius |
| `cluster_min_points` | 15 | DBSCAN minimum cluster size |
| `cluster_min_height` | 0.3 m | Reject clusters shorter than this |
| `cluster_max_height` | 2.5 m | Reject clusters taller than this |
| `cluster_max_footprint` | 1.2 m | Reject clusters wider than this in XY |

**Why DBSCAN?** In an unstructured point cloud of a room interior, objects are not
arranged on a grid and their count is unknown. DBSCAN finds arbitrarily-shaped
dense regions without requiring a cluster count, and explicitly labels noise. It is
the standard approach for this problem domain.

**The real-time story.** With the default configuration (open3d, 0.05 m voxel):

| Backend | Voxel (m) | ms/frame (best-of-3, ~24k pts) |
|---|---|---|
| sklearn | 0.00 | 186 |
| open3d | 0.00 | 209 |
| sklearn | 0.05 | 38 |
| open3d | 0.05 | 30 |
| sklearn | 0.10 | 22 |
| open3d | 0.10 | 17 |

(Measured on `walk_one` recording, approximately 24k non-NaN points per frame,
via `tools/bench_clustering.py`.)

The key observation: at full resolution, `open3d` is actually *slower* than
`sklearn` because Python↔C++ marshalling of a large array dominates. The decisive
gain comes from the **voxel downsample**, not from the C++ backend. With a 5 cm
voxel, both backends run in 30–38 ms; adding the C++ backend brings it to 30 ms,
the extra 8 ms is the backend's contribution after marshalling is already cheap.

Without any downsample, the naive pipeline ran at about 5 Hz on 25k real points
(DBSCAN-bound). With open3d at 0.05 m the pipeline processes at ~20–30 Hz; the
display always shows the full-resolution cloud (only the clustering input is
downsampled).

**Voxel label propagation.** The voxel step produces a representative point per
occupied voxel, runs DBSCAN on those representatives, then maps each representative's
label back to every original non-ground point in its voxel (`inv` array from
`np.unique`). Detection geometry (centroid, AABB, `point_indices`) is always
computed from the full-resolution points, not the representatives. This ensures the
bounding boxes and per-point indices are correct.

**Footprint filter.** After height filtering, clusters whose maximum XY extent
exceeds `cluster_max_footprint` (1.2 m) are rejected. A standing person is
approximately 0.3–0.8 m across; walls and large furniture pieces that survive
the height filter are eliminated here. On the `walk_one` recording this reduced
the cluster count from approximately 7 clusters and 9 tracks to approximately 3
clusters and 4 tracks — the actual people plus at most one nearby obstacle.

---

## 4. Multi-object tracking — constant-velocity Kalman + Hungarian

**File:** `helios16p/processing/tracking.py` — `MultiObjectTracker`

**Config knobs:**

| Parameter | Default | Meaning |
|---|---|---|
| `track_gate_distance` | 1.5 m | Hard association gate (Euclidean distance) |
| `track_min_hits` | 3 | Detections required before a track is confirmed |
| `track_max_age` | 10 | Frames a track survives without a detection |

**State vector.** Each track maintains a 6-DOF state `[x, y, z, vx, vy, vz]` in
the sensor frame, with a 6×6 covariance matrix. The measurement is `[x, y, z]`
(cluster centroid only). This is the standard constant-velocity model; it is
appropriate because persons typically walk at near-constant speed over the time
scales of a few tenths of a second, and the centroid is a poor velocity estimator
per frame (the Kalman smoother provides the actual estimate).

**Per-frame update sequence:**

1. **Predict.** Propagate all existing tracks forward by `dt` using the
   state-transition matrix `F` and process noise covariance `Q` (constant-
   acceleration disturbance model, `σ_a = 1.0 m/s²`).
2. **Associate.** Build an `(n_tracks × n_detections)` cost matrix of Euclidean
   centroid distances. Solve the global assignment with
   `scipy.optimize.linear_sum_assignment` (Hungarian algorithm). Accept pairs only
   if the cost is within `track_gate_distance`.
3. **Update.** For each matched pair, apply the Kalman measurement update
   (`K = P H^T S^{-1}`), stamp `detection.track_id` in place (so the viewer knows
   which box belongs to which track), and append the new position to the track's
   history (capped at the last 50 positions for trail rendering).
4. **Promote.** Mark a track as `confirmed = True` when `hits >= track_min_hits`.
   Only confirmed tracks are eligible for gimbal targeting.
5. **Spawn.** Each unmatched detection creates a new tentative track with a fresh
   monotonically increasing `track_id`.
6. **Delete.** Tracks with `time_since_update > track_max_age` are pruned.

**Measurement noise.** `R = 0.01 I₃` (centroid uncertainty ≈ 0.1 m standard
deviation), which is appropriate for a cluster centroid from a 30 cm voxel cloud.
Initial position covariance is 1 m² per axis; initial velocity covariance is
5 (m/s)² per axis.

**Track lifecycle.** A new detection spawns a tentative track immediately. After
`track_min_hits` successful associations (default 3 frames ≈ 0.3 s at 10 Hz),
the track is promoted to confirmed. Confirmed tracks are displayed in the viewer
and are eligible for gimbal targeting. When a person leaves the scene or
occluded by furniture, the predicted track persists for up to `track_max_age`
frames (default 10 ≈ 1 s) before deletion.

---

## 5. Virtual gimbal — motion-based targeting

**File:** `helios16p/control/gimbal.py` — `VirtualGimbal`

**Config knobs:**

| Parameter | Default | Meaning |
|---|---|---|
| `moving_speed_threshold` | 0.3 m/s | A track faster than this is classified as "moving" |

**Internal constants** (not in `DemoConfig`):

| Constant | Value | Meaning |
|---|---|---|
| `_SLEW_LIMIT_DEG` | 5.0 | Maximum azimuth slew per update step |
| `_P_GAIN` | 0.3 | Proportional gain (fraction of error applied per step) |
| `_AIM_RADIUS_M` | 5.0 | Length of the visualised aim ray |

**Why motion-based targeting?** A typical indoor scene has many confirmed tracks:
chairs, tables, shelving, the fixed sensor mount itself. Without a motion
discriminator, the gimbal would lock onto the largest or nearest confirmed track,
which is usually a piece of furniture. The `moving_speed_threshold` filter
restricts eligible targets to tracks whose Kalman-estimated velocity magnitude
exceeds 0.3 m/s. This reliably selects a walking person and ignores the background.

**Target selection (sticky, with fallback):**

1. If the current target track is still live and still moving, keep it (sticky
   tracking avoids chattering when two people are present).
2. If moving tracks exist but the current target is no longer moving (or has been
   deleted), switch to the fastest confirmed mover.
3. If no confirmed moving tracks exist, fall back to the longest-lived confirmed
   track (or any track if none are confirmed).

**Control law.** The gimbal maintains a single state variable `_gimbal_az` (degrees).
Each update:
1. Compute `target_az = atan2(pos.y, pos.x)` in degrees.
2. Compute the signed shortest-path error: `error = target_az − gimbal_az`,
   normalised to (−180°, +180°].
3. Apply P-control: `raw_cmd = 0.3 × error`, clamped to ±5°.
4. Update: `gimbal_az += slew`.
5. Compute the aim point at radius 5 m along the current gimbal azimuth at the
   target's z height.

The slew limit ensures the gimbal motion is visually observable (it does not
teleport) and models a realistic actuator constraint.

**Output dict.** `VirtualGimbal.update()` returns a dict stored in
`FrameAnnotations.control`:

```python
{
    "target_id": int,
    "target_azimuth_deg": float,
    "gimbal_azimuth_deg": float,
    "error_deg": float,           # remaining error after this step's slew
    "target_position": (3,) f32,
    "aim_point": (3,) f32,        # rendered as the orange ray endpoint
    "target_speed": float,
    "target_moving": bool,
}
```

The HUD displays `target_id`, `target_moving`, `target_speed`, and `error_deg`
so the user can observe the control loop in real time.

---

## 6. Interactive PyVista viewer

**File:** `helios16p/viz/pyvista_viewer.py` — `PyVistaViewer`

**Why PyVista/VTK, not Open3D?**
On some Wayland sessions (seen on an Intel/Mesa laptop), Open3D's bundled GLFW
is compiled for Wayland only and its GLEW initialisation fails when creating an
OpenGL context, and the newer `visualization.gui` API (Filament renderer) can
crash on the same stack. VTK renders through X11/GLX (XWayland), which works
there. Open3D stays available as `--viewer open3d`; off-screen renders use
PyVista (`helios16p-snapshot`).

**Architecture.** The viewer runs a background worker thread that pulls frames
from the source and calls `pipeline.process()`, storing the latest
`FrameAnnotations` in a shared variable protected by a threading lock. The main
thread runs the VTK event loop (`p.update()` at ~100 Hz) and rebuilds the scene
actors whenever a new annotation is available. This decouples the pipeline rate
from the render rate and prevents the VTK event loop from blocking on slow frames.

**Scene elements:**

| Element | Condition |
|---|---|
| Point cloud | Always; coloured by height / intensity / ring |
| Reference grid | Static; 6 m radius, 1 m step |
| Bounding boxes | Per detection (wireframe, track colour) |
| Track spheres + velocity arrows | Per confirmed track; arrow scaled by speed |
| History trails | Per track; last 50 positions |
| Gimbal ray | Orange line from 1 m above the sensor to the aim point |

**Keyboard controls:**

| Key | Action |
|---|---|
| `1` | Colour by height (viridis, default) |
| `2` | Colour by intensity |
| `3` | Colour by ring (beam by elevation, 0 lowest to 15 highest) |
| `G` | Toggle ground points |
| `B` | Toggle bounding boxes |
| `K` | Toggle track markers |
| `L` | Toggle history trails |
| `M` | Toggle gimbal ray |
| `N` | Toggle moving-only mode (hide static tracks and their boxes) |
| `C` | Cycle clustering backend (open3d ↔ sklearn), live |
| `V` | Cycle voxel size (off → 5 cm → 10 cm → off), live |
| `P` | Save current frame to height-coloured PLY |
| `Q` | Quit |

**On-screen widgets.** Six toggle checkboxes (ground, clusters, tracks, trails,
control, moving-only), one backend checkbox (open3d / sklearn), and a voxel-size
slider are placed along the right edge of the window. These parallel the keyboard
shortcuts and are labelled with their key binding in parentheses. The slider uses
`add_slider_widget` with range [0.0, 0.15 m].

**HUD.** A text overlay (upper-left) shows:
- Frame rate (EMA), point count
- Cluster count, track count
- Current colour mode
- Clustering backend, voxel size, pipeline latency in ms
- (if control active) Gimbal target ID, moving/static, speed, error in degrees

**PLY export.** Pressing `P` saves the current frame's full-resolution points to a
height-coloured PLY file (viridis colormap via matplotlib, written with Open3D
offscreen). Files land at
`recordings/ply/frame_<timestamp>.ply`
and can be opened in CloudCompare or MeshLab directly.

**PyVista PolyData build cost:** approximately 2 ms per frame. VTK render time is
tens of ms and depends on point count and window size. Rendering is not the
pipeline bottleneck; DBSCAN at full resolution is.
