# 06 — Usage

---

## Installation

Requires Python 3.10 to 3.12 (Open3D has no 3.13 wheel yet). A virtual environment is strongly recommended.

```bash
cd helios16p

python3.12 -m venv .venv
source .venv/bin/activate

pip install -e .
```

This installs the `helios16p` package in editable mode along with its mandatory dependencies (numpy, scipy, scikit-learn, pyvista, open3d, mcap, matplotlib). ROS-related dependencies (rclpy, sensor_msgs) are optional and must come from a sourced ROS 2 installation — they are not in `pyproject.toml`.

---

## The commands

Installing the package puts seven commands on the PATH; the `scripts/` and
`tools/` entry points of the checkout call the same code.

| command | checkout equivalent | purpose |
|---|---|---|
| `helios16p-demo` | `helios16p-demo` | the viewer; no arguments replays the bundled 4 s clip |
| `helios16p-record` | `helios16p-record` | raw MSOP + DIFOP packets to `./recordings/<name>_<date>.npz` |
| `helios16p-info` | | describe a recording, print or save its calibration |
| `helios16p-export` | `helios16p-export` | PLY export, merged or per frame |
| `helios16p-snapshot` | | off-screen PNG of a recording (annotated frame, merge, calibration comparison) |
| `helios16p-trim` | | cut a recording to a time window |
| `helios16p-validate` | | compare the decoder with a vendor (rs_driver) cloud of the same packets |

Every command answers `--help`.

```bash
helios16p-demo                                   # bundled clip, looping
helios16p-demo --rec recordings/lab_*.npz        # your recording
helios16p-demo --viewer none --max-frames 20     # no window: per-frame statistics
helios16p-info --rec recordings/lab_*.npz --frames --save-calib my_sensor.json
helios16p-snapshot --rec recordings/lab_*.npz --out lab.png --hud
helios16p-export --rec recordings/room_*.npz --merge --color intensity
```

---

## Live network setup (UDP source)

The Helios 16P sends data over 100Base-T1 Ethernet. Its default device IP is `192.168.1.200`. The host NIC on the LiDAR subnet must be set to `192.168.1.102/24`.

**Configure the NIC with nmcli (persistent profile):**

```bash
# Replace <IFACE> with your NIC name, e.g. enp3s0 or enx00...
nmcli con add type ethernet ifname <IFACE> con-name helios-lidar \
    ipv4.method manual ipv4.addresses 192.168.1.102/24 \
    ipv4.gateway "" connection.autoconnect yes

nmcli con up helios-lidar
```

Verify connectivity:

```bash
ping 192.168.1.200
```

The sensor does not reply to ICMP by default, but packet capture on port 6699 should show incoming MSOP traffic:

```bash
sudo tcpdump -i <IFACE> udp port 6699 -c 5
```

No elevated privileges are required to run the demo itself — Python can bind to ports > 1024 without root.

---

## Running the demo (`helios16p-demo`)

`helios16p-demo` is the single entrypoint (`scripts/demo.py` in a checkout). The `--source` flag selects the data path; all other pipeline and viewer settings come from `DemoConfig` defaults (tunable via the viewer's live controls).

### Synthetic source (no hardware needed)

```bash
helios16p-demo --source synthetic
```

Generates a moving-people scene in software. Useful for verifying the installation and exploring the viewer controls without a sensor or recording.

### Replay from an mcap bag (ROS bag, no ROS runtime needed)

```bash
helios16p-demo --source mcap --bag /path/to/bag_dir
```

Reads `sensor_msgs/PointCloud2` frames from a ROS 2 mcap bag using the `rosbags` Python library. No ROS installation required. The bag directory produced by `ros2 bag record` works directly.

Optional flags: `--loop` to repeat the bag, `--max-frames N` to stop after N frames, `--fps 10.0` to control replay pacing.

### Live sensor over UDP

```bash
helios16p-demo --source udp
```

Binds MSOP port 6699 and DIFOP port 7788, receives raw packets from the sensor, decodes them with the numpy decoder, and feeds frames into the pipeline. The host NIC must be on `192.168.1.102/24` (see network setup above).

### Live ROS 2 topic

```bash
source /opt/ros/jazzy/setup.bash
helios16p-demo --source ros --topic /rslidar_points
```

Subscribes to a live `sensor_msgs/PointCloud2` topic via rclpy. Requires a sourced ROS 2 workspace and the vendor `rslidar_sdk` node running. The default topic is `/rslidar_points`; override with `--topic`.

### Replay a raw-packet recording

```bash
helios16p-demo --source file --rec /path/to/recording.npz
```

Replays a `.npz` recording produced by `helios16p-record` through the exact same numpy decoder as the live path. Supports `--loop` and `--max-frames`.

---

## Recording (`helios16p-record`)

Records raw MSOP and DIFOP packets directly from the sensor to a compressed `.npz`.

```bash
# Record for 30 seconds
helios16p-record --name walk1 --seconds 30

# Record until Ctrl+C
helios16p-record --name freewalk

# Custom output directory
helios16p-record --name bench --seconds 60 --dir /tmp/helios_recs
```

Default output directory is `./recordings`. The output filename is `<name>_<YYYYMMDD_HHMMSS>.npz`. A 30-second capture at ~750 packets/s produces approximately 28 MB before compression.

Progress is printed to stdout every second:

```
MSOP   22500 pkt ( 28.1 MB) | DIFOP   3 |   30s
```

---

## Exporting to PLY (tools/export_ply.py)

Converts a `.npz` recording to height- or intensity-coloured PLY files for CloudCompare, Meshroom, or other point-cloud tools.

```bash
# Single dense cloud from all frames (great for CloudCompare inspection)
helios16p-export --rec room_static_20240501_143200.npz --merge

# Every 5th frame as individual PLY files, coloured by intensity
helios16p-export --rec walk_one_20240501_150000.npz --every 5 --color intensity

# Specific frame only
helios16p-export --rec walk_one_20240501_150000.npz --frame 42

# Custom output path
helios16p-export --rec recording.npz --merge --out /tmp/room_dense.ply
```

The PLY files are binary and carry x y z, intensity and ring as vertex properties plus a viridis colour; CloudCompare and MeshLab open them directly. `room_dense.png` in the docs was produced from 60 merged revolutions.

The viewer key `P` exports the current frame to PLY directly from the live window (see below).

---

## Viewer controls

The PyVista viewer (`helios16p/viz/pyvista_viewer.py`) supports both keyboard shortcuts and on-screen clickable widgets. The keyboard shortcuts are listed in the status bar at the bottom-left of the window.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `1` | Colour cloud by height (z) |
| `2` | Colour cloud by return intensity |
| `3` | Colour cloud by beam ring index |
| `G` | Toggle ground points on/off |
| `B` | Toggle cluster bounding boxes on/off |
| `K` | Toggle track markers and velocity arrows on/off |
| `L` | Toggle track history trails on/off |
| `M` | Toggle gimbal control ray on/off |
| `N` | Toggle moving-only mode (hides static detections and tracks) |
| `C` | Cycle clustering backend: open3d ↔ sklearn (live, takes effect next frame) |
| `V` | Cycle voxel size: off → 5 cm → 10 cm → off (live) |
| `P` | Save the current frame to `./recordings/ply/` as PLY (x y z, intensity, ring, colour) |
| `S` | Save a screenshot of the window to `./recordings/shots/` |
| `T` | Toggle the id and speed labels on confirmed tracks |
| `R` | Refit the camera on the cloud |
| `Space` | Pause and resume the source (Qt window) |
| `Q` | Quit |

### On-screen clickable widgets (right side of window)

Six checkbox buttons appear on the right edge of the window, each mirroring a keyboard toggle:

| Widget label | Keyboard | Controls |
|--------------|----------|----------|
| ground (G) | `G` | Show/hide ground points |
| clusters (B) | `B` | Show/hide cluster bounding boxes |
| tracks (K) | `K` | Show/hide track spheres and velocity arrows |
| trails (L) | `L` | Show/hide track history trails |
| control (M) | `M` | Show/hide gimbal control ray |
| moving-only (N) | `N` | Filter to dynamic objects only |

One additional checkbox:

| Widget label | Keyboard | Controls |
|--------------|----------|----------|
| C++ clustering (C) | `C` | Toggle open3d (on) / sklearn (off) backend |

One slider:

| Widget label | Keyboard | Range | Controls |
|--------------|----------|-------|----------|
| voxel m (V) | `V` | 0.00–0.15 m | Clustering voxel size (continuous; drag to set) |

### HUD (upper-left overlay)

The upper-left text block displays live pipeline statistics updated every frame:

```
FPS 24.3   pts 28440
clusters 3   tracks 4
colour: height
cluster: open3d  voxel 5cm  41ms
gimbal -> target #2 (moving 1.3 m/s)  err +2 deg
```

When moving-only mode is active, `[only-moving ON]` appears as an additional line.

---

## Disabling the gimbal stage

```bash
helios16p-demo --source synthetic --no-control
```

`--no-control` skips the `VirtualGimbal.update` call. The gimbal ray (`M`) will have nothing to display. Useful when profiling pure perception throughput.
