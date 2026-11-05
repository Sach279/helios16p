# Changelog

## 0.2.0 (4 November 2026)

The release for sharing: the decoder is now right, and the package installs
with its own commands and a bundled recording.

### Fixed
- **DIFOP calibration was never applied.** The parser read the two angle tables
  at bytes 303 and 399 of the DIFOP packet, which hold no calibration (the
  tables start at 468 and 564), and it also required all 32 entries to be
  valid when the 16P only fills 16. Every decode therefore fell back to an
  evenly spaced guess of the beam angles, with the beams in the wrong order
  and without the per-beam horizontal offsets of about +3.1 and -7.5 degrees.
  Against the vendor driver on the same packets the old decoder sat 112 mm
  from the nearest vendor point (median; 0.56 m point to point); the fixed one
  sits 0.0001 mm, with 99.99 % of sampled points within 1 mm
  (`docs/validation_corridor1_2026-11-04.json`, `helios16p-validate`).
- The lens-offset term (RX) now uses the per-channel interpolated azimuth, as
  the vendor driver does, not the block azimuth (worth up to 0.12 mm).
- The pcap replayer finds the DIFOP packet anywhere in the capture (it used to
  give up at the first MSOP packet) and keeps the trailing partial revolution.
- Replay now matches live decode frame for frame: the same range window
  (0.1 to 180 m), the same frame split on the raw first-block azimuth; a test
  streams the bundled packets over loopback into `UdpSource` to check it.
- The RANSAC ground fit is seeded, so a run repeats.
- `helios16p-export` writes to the current directory by default (it used to
  write next to the recording, inside the installed package for the bundled
  clip) and fails loudly when it cannot write.
- The viewer closes when an offline source ends; `V` and `B` no longer also
  trigger PyVista's own bindings.
- `PacketFileSource` and the pcap replayer apply the horizontal offsets too,
  and the replay range window matches the vendor (0.1 to 180 m).

### Changed
- `Frame.ring` is now the beam index by elevation (0 = lowest beam), as the
  vendor driver reports it; the wire laser id is still available as `laser` in
  the per-packet decode. Colour-by-ring in the viewer therefore shows ordered bands.
- `config.DEFAULT_VERT_ANGLES_DEG` lists the nominal angles in firing order.
- `decode_msop_packet(pkt, calibration=...)` is the normal call; the old
  positional arrays still work.

### Added
- A Qt window (`--viewer qt`, the default when PySide6 is installed): side
  panel with live metrics, colour and camera buttons, toggles, clustering and
  moving-threshold sliders, pause, PLY export and screenshot; every control has
  a tooltip and a status-bar hint. The plain PyVista window keeps the keys.
- The scene is shared by both windows and the snapshot tool: camera framed on
  the cloud, grid on the fitted ground, colour legend, id and speed labels on
  tracks, a grey shadow of the static scene in moving-only mode, horizontal
  velocity arrows; persistent actors bring a frame from 80 ms to about 15 ms.
- `--color`, `--only-moving`, `--screenshot` on the demo; `S` saves a screenshot.
- `helios16p.calibration`: `Calibration`, `parse_difop`, bundled fallback
  (`helios16p/data/helios16p_calibration.json`, from our unit), save/load JSON.
- Console commands: `helios16p-demo`, `helios16p-record`, `helios16p-export`,
  `helios16p-info`, `helios16p-trim`, `helios16p-snapshot`, `helios16p-import-pcap`,
  `helios16p-validate`; every one works on the bundled clip without `--rec`.
- PLY export carries intensity and ring as vertex properties (numpy writer).
- `UdpSource` warns after 3 s without packets, filters by `device_ip` when
  asked, and warns when the DIFOP reports other than 600 rpm single return.
- `--msop-port`, `--difop-port`, `--device-ip` on the demo.
- A 150-packet vendor-parity fixture runs with the normal tests.
- A 4 s real recording ships in the package; `helios16p-demo` with no
  arguments replays it.
- `--viewer none` (stats only), `--viewer open3d`, `--calib file.json`.
- Tests for the calibration parser and the file replay on real packets, a
  slow vendor-parity test, `tests/conftest.py` (the processing tests had no
  fixtures and could not run), a GitHub Actions workflow.
- MIT licence, citation file, this changelog.

### Removed
- Hard-coded paths to the author's machine in tools and docs. Recordings go
  to `./recordings` by default.

## 0.1.0 (5 June 2026)

First version: UDP decoder, mcap and ROS sources, processing pipeline,
Open3D and PyVista viewers, recorder, PLY export, docs.
