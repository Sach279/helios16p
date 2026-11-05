# A ten-minute tour

Everything below runs on the recording shipped with the package. With a sensor
plugged in, replace `helios16p-demo` with `helios16p-demo --source udp`.

```bash
helios16p-demo                       # the room, the walking person boxed and tracked
```
Orbit with the mouse, wheel to zoom. Keys: `1 2 3` colour by height, intensity,
beam; `G` hide the ground; `B` boxes; `K` tracks; `L` trails; `M` the gimbal ray;
`N` moving objects only; `C` switch the clustering backend (C++ Open3D or
scikit-learn, watch the HUD time); `V` cycle the clustering voxel; `P` save the
frame as PLY to `./recordings/ply/`; `Q` quit.

```bash
helios16p-info --frames              # 3000 packets, 4 s, 600 rpm single return, the calibration tables, 42 frames
helios16p-demo --viewer none --max-frames 10        # the pipeline without a window
helios16p-snapshot --out frame.png --hud            # the same scene rendered off screen
helios16p-export --merge --color intensity          # walk_4s_merged.ply, open it in CloudCompare
```

With a sensor:

```bash
helios16p-record --name tour --seconds 20           # raw packets, calibration included
helios16p-info --rec recordings/tour_<date>.npz --frames
helios16p-demo --rec recordings/tour_<date>.npz     # replays as it was seen live
helios16p-demo --source udp --viewer none            # if nothing arrives, a warning after 3 s says what to check
```

What to say while it runs, in one sentence each:

- *Decoding:* 75 packets of 1248 bytes per revolution, decoded in numpy from the
  documented wire format; no vendor code at runtime.
- *Calibration:* the sensor sends its own per-beam angles (DIFOP); applying them is
  the difference between the two panels of `docs/figures/calibration_before_after.png`.
- *Perception:* RANSAC ground plane, DBSCAN clusters, Kalman tracks with Hungarian
  assignment, a virtual gimbal that keeps the chosen track centred.
- *Same code with or without ROS:* the source is the only thing that changes
  (`--source udp | file | mcap | ros | synthetic`).
