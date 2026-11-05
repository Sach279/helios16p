# RoboSense Helios 16P — wire protocol notes

Extracted from the official BSD-licensed `rs_driver`
(`decoder_RSHELIOS.hpp`, `decoder_RSHELIOS_16P.hpp`, `decoder_mech.hpp`).
These are the facts the pure-Python `udp_source` decoder must reproduce.

## Network
- LiDAR ships data over UDP. Two streams:
  - **MSOP** (point data) — UDP port **6699**
  - **DIFOP** (device info + calibration) — UDP port **7788**
- Default device IP `192.168.1.200`, host NIC `192.168.1.102`.

## MSOP packet — 1248 bytes total
Layout (`#pragma pack(1)`):

| region | bytes | notes |
|---|---|---|
| header | 42 | see below |
| blocks[12] | 1200 | 12 × 100-byte blocks |
| index | 4 | `unsigned int` |
| tail | 2 | |

**Header (42 B):** `id[4]` = `55 AA 05 5A`, `protocol_version`(2), `reserved1`(14),
`timestamp` RSTimestampUTC(10), `lidar_type`(1), `reserved2`(7), `temp` RSTemperature(2),
`reserved3`(2).

**Block (100 B):** `id[2]` = `FF EE`, `azimuth` uint16 **big-endian** (ntohs),
then `channels[32]`. Azimuth is in centi-degrees (0..35999).

**Channel (3 B):** `distance` uint16 **big-endian**, `intensity` uint8.
- `distance_metres = raw_u16 * 0.0025`  (DISTANCE_RES = 0.0025)
- valid range used in the recording: 0.2 .. 200 m.

### ASCII byte-layout diagram

```
Byte offset  Length   Field
───────────  ──────   ──────────────────────────────────────────────────
      0         4     MSOP magic: 55 AA 05 5A
      4         2     protocol_version
      6        14     reserved1
     20        10     RSTimestampUTC  (YY MM DD HH mm ss sub[4])
     30         1     lidar_type
     31         7     reserved2
     38         2     RSTemperature
     40         2     reserved3
───────────────────── end of 42-byte header ──────────────────────────
     42       100     Block 0   ┐
    142       100     Block 1   │
    242       100     Block 2   │  12 × 100-byte data blocks
    342       100     Block 3   │
    ...                         │
   1142       100     Block 11  ┘
───────────────────── end of 1200-byte block region ──────────────────
   1242         4     Index  (uint32, little-endian)
   1246         2     Tail   (uint16)
───────────────────── total: 1248 bytes ──────────────────────────────

Each 100-byte block:
  Block byte  Length   Field
  ──────────  ──────   ──────────────────────────────────────────────
       0        2      Block magic: FF EE
       2        2      Azimuth, uint16 big-endian, centi-degrees 0..35999
       4–99    96      32 channels × 3 bytes:
                         [+0 +1]  Distance, uint16 big-endian
                                  → metres = raw × 0.0025 (DISTANCE_RES)
                         [+2]     Intensity, uint8, 0..255

16P channel-to-laser mapping:
  chan 0..31 → laser = chan % 16
  Channels 0..15: first firing of the 16 lasers
  Channels 16..31: second firing of the 16 lasers (same block, ~27.7 µs later)

Per-channel firing time offsets (µs) within one block:
  chan  0: 0.00   chan  1: 1.73   chan  2: 3.46   chan  3: 5.19
  chan  4: 6.92   chan  5: 8.65   chan  6:10.38   chan  7:12.11
  chan  8:13.84   chan  9:15.57   chan 10:17.30   chan 11:19.03
  chan 12:20.76   chan 13:22.49   chan 14:24.22   chan 15:25.95
  chan 16:27.68   chan 17:29.41   chan 18:31.14   chan 19:32.87
  chan 20:34.60   chan 21:36.33   chan 22:38.06   chan 23:39.79
  chan 24:41.52   chan 25:43.25   chan 26:44.98   chan 27:46.71
  chan 28:48.44   chan 29:50.17   chan 30:51.90   chan 31:53.63
  Block duration: 55.56 µs
```

## 16P specifics
- `LASER_NUM = 16`, but `CHANNELS_PER_BLOCK = 32`: each block holds **two firings**
  of the 16 lasers. Map `laser = chan % 16`.
- Per-channel firing time offsets within a block (µs), 32 entries, in `config.FIRING_TSS_US`.
  `BLOCK_DURATION = 55.56 µs`.
- Single vs dual return is read from DIFOP `return_mode`:
  `0x00` = dual; `0x04/0x05/0x06` = single. Azimuth interpolation across the
  two half-blocks differs between the two (`Rs16SingleReturnBlockIterator` vs
  `Rs16DualReturnBlockIterator`). For a first cut, support single return; the
  recorded data tells you which mode the unit is in.

## Geometry (spherical → cartesian)
Per point, with horizontal angle `az` and per-laser vertical angle `vert`
(both centi-degrees → radians), distance `d`:

```
x =  d * cos(vert) * cos(az) + RX * cos(az_raw)
y = -d * cos(vert) * sin(az) - RX * sin(az_raw)
z =  d * sin(vert) + RZ
```
with lens offsets `RX = 0.03498`, `RZ = 0.0` (RY unused in the transform).
`az` includes a per-channel azimuth correction `block_az + block_az_diff * CHAN_AZIS[chan]`.

## Vertical / horizontal angles
- Authoritative source is the **DIFOP** packet: `vert_angle_cali[32]` at byte
  468 and `horiz_angle_cali[32]` at byte 564 (`RSCalibrationAngle`, sign u8 +
  value u16 big-endian in 0.01°; the 16P fills 16 entries, the rest are 0xFF).
  Also: rpm at byte 8 (u16 BE), return mode at byte 300.
- The beams fire interleaved: wire laser 0..15 = +13, +15, +9, +11, +5, +7, +1,
  +3, −3, −1, −7, −5, −11, −9, −15, −13 degrees (nominal). `ring` = index by
  elevation, 0 = lowest beam.
- Until a DIFOP packet arrives, `helios16p.calibration.default()` (the bundled
  file from our unit) is used. See `07-calibration.md` for the measured effect.

## Validation strategy
The recorded `.mcap` holds already-decoded `PointCloud2` (topic `/rslidar_points`),
not raw packets — so it validates the *processing/viz*, not the UDP decoder directly.
To validate the decoder: capture live UDP to a `.pcap` (`tcpdump -i <nic> udp port 6699 -w x.pcap`),
decode it with both this Python decoder and rs_driver's pcap mode, and compare point clouds.
