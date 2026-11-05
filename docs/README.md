# helios16p — Documentation Index

`helios16p` is a pure-Python, real-time perception and visualisation stack for the **RoboSense Helios 16P** spinning LiDAR. It decodes the sensor's raw UDP packets using a numpy implementation of the MSOP wire protocol (no vendor SDK, no ROS at runtime), runs a four-stage processing pipeline — ground removal (RANSAC), object clustering (DBSCAN with voxel downsample), multi-object Kalman tracking (Hungarian assignment), and a virtual gimbal control loop — and renders the results in an interactive 3D viewer built on PyVista/VTK. Five swappable sources (synthetic, mcap bag, UDP live, ROS 2 topic, raw-packet file replay) share a single pipeline and viewer, so the same stack works with or without a physical sensor and with or without a ROS installation.

---

## Table of contents

| Document | Description |
|----------|-------------|
| [00-overview.md](00-overview.md) | Project goals, sensor facts, scope |
| [01-architecture.md](01-architecture.md) | Data contracts (`types.py`), pipeline wiring, source ABC |
| [02-our-system.md](02-our-system.md) | Our UDP/numpy acquisition path end-to-end |
| [03-ros-and-vendor.md](03-ros-and-vendor.md) | ROS 2 path (rslidar_sdk node, mcap bags, RViz figures) |
| [PROTOCOL_NOTES.md](PROTOCOL_NOTES.md) | MSOP/DIFOP wire-format reference (packet layout, field offsets) |
| [04-design-decisions.md](04-design-decisions.md) | Eight design decisions with context and rationale |
| [05-performance.md](05-performance.md) | Per-stage timing, clustering benchmark, the real-time story |
| [06-usage.md](06-usage.md) | Install, network setup, all five sources, recording, PLY export, viewer controls |
| [07-calibration.md](07-calibration.md) | The DIFOP angle tables, how they are applied, validation against the vendor driver, the 0.1 bug |

---

## Figures

Figures are under `docs/figures/`:

- `walk_frame.png` — one frame of the bundled clip through the pipeline, with boxes, tracks, the gimbal ray and the HUD (`helios16p-snapshot --hud`)
- `walk_rings.png` — the same frame coloured by beam, seen from the front
- `room_dense.png` — a static room, sixty revolutions merged
- `calibration_before_after.png` — thirty revolutions of a corridor station from above, pre-0.2 decode against the DIFOP calibration
- regenerate with `python tools/make_doc_figures.py --rec-dir recordings`

- `helios16p_product.webp` — the sensor (image: generationrobots.com / RoboSense; attributed)
- `figures/ros/rviz_helios_interact.png` — the same sensor in RViz via the vendor rslidar_sdk ROS 2 driver
