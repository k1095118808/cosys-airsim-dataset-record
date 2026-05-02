# Cosys-AirSim Multi-Camera RGBD Collector

Tools for collecting synchronized RGB + depth + state datasets from a
[Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim) environment,
plus a ground-truth voxel map of the world.

## Files

| File                   | Purpose                                                                       |
| ---------------------- | ----------------------------------------------------------------------------- |
| `settings.json`        | AirSim config: a `Drone1` multirotor with 4 RGBD cameras (front/left/right/down) |
| `collect_dataset.py`   | Time-driven collector: capture every *N* seconds while the drone moves        |
| `collect_random.py`    | Pose-driven collector: teleport to random collision-free poses and capture    |
| `generate_voxel_map.py`| Build a ground-truth `.binvox` occupancy grid (and optional point cloud)      |
| `random_ds/`           | Output of a `collect_random.py` run                                           |

## Prerequisites

- A Cosys-AirSim Unreal binary (`.exe` on Windows or Linux build)
- Python 3.8+
- `pip install cosysairsim numpy opencv-python rpc-msgpack`
  (Or install `cosysairsim` from the `PythonClient/` folder of the
  Cosys-AirSim repo for the latest API surface.)
- Optional, for inspecting voxel maps: `viewvox` and `binvox2bt` from
  <https://www.patrickmin.com/binvox/>.

## Setup

1. **Install `settings.json`** so AirSim picks it up at startup. Either copy:
   ```bash
   cp settings.json ~/Documents/AirSim/settings.json        # Linux/WSL
   # Windows: copy to %USERPROFILE%\Documents\AirSim\settings.json
   ```
   Or pass it on the command line when launching the binary:
   ```
   AirSim.exe -settings="C:\path\to\settings.json"
   ./Blocks.sh -settings="$(pwd)/settings.json"
   ```
2. **Launch the AirSim binary.** You should see the drone with three small
   subwindows (front RGB, down RGB, front depth-vis).
3. **Run a collector.** See [Usage](#usage).

## Camera layout

All cameras: 640×480 @ 90° FOV, capturing `Scene` (RGB) + `DepthPlanar`
(metric depth). Poses are body-frame NED meters (+X forward, +Y right,
+Z **down**):

| Camera | X    | Y     | Z     | Yaw | Pitch | Notes                          |
| ------ | ---- | ----- | ----- | --- | ----- | ------------------------------ |
| front  | 0.35 |  0.00 | -0.05 |   0 |    0  | nose-mounted                   |
| left   | 0.00 | -0.35 | -0.10 | -90 |    0  | outboard, above prop plane     |
| right  | 0.00 |  0.35 | -0.10 | +90 |    0  | mirror of left                 |
| down   | 0.00 |  0.00 |  0.10 |   0 |  -90  | belly-cam looking straight down|

Tweak inside `settings.json` → `Vehicles.Drone1.Cameras.<name>` and
restart the binary (settings are read once at startup).

## Usage

### Time-driven collection — `collect_dataset.py`

Capture every `--interval` seconds while the drone moves. Use this for
trajectory-style data.

```bash
# Manual flight (keyboard/RC), capture at 5 Hz
python3 collect_dataset.py --out dataset --num 500 --interval 0.2

# Auto-fly the WAYPOINTS list defined at the top of the script
python3 collect_dataset.py --out dataset --num 500 --interval 0.2 --auto-fly
```

Useful flags:
- `--no-pause` — skip pause/unpause around capture (faster, less synced)
- `--no-swap-rb` — disable RGB↔BGR swap; use only if your build returns BGR
- `--auto-fly` — run the in-script `WAYPOINTS` list in a side thread

### Random-pose collection — `collect_random.py`

Teleport to random collision-free poses inside a bounding box. Use this
for IID-like training samples.

Two safety strategies:

- **With a voxel map** *(recommended)*: build one once with
  `generate_voxel_map.py`, then pass `--map`. Validation becomes a numpy
  cube lookup — essentially free.
- **Without a map**: each candidate is teleported, physics runs for a
  fraction of a second, then `simGetCollisionInfo` decides. Works with
  no pre-build but slower per candidate.

```bash
# Voxel-map path (recommended)
python3 collect_random.py --out random_ds --num 300 \
    --bounds "-50,50, -50,50, -20,-1" \
    --map /mnt/c/temp/env.binvox --margin 1.0

# Fallback path (no map)
python3 collect_random.py --out random_ds --num 300 \
    --bounds "-50,50, -50,50, -20,-1"
```

Useful flags:
- `--bounds x_min,x_max,y_min,y_max,z_min,z_max` — NED meters; remember
  +Z is **down**, so `z=-20,-1` is 1–20 m above the spawn altitude.
- `--max-pitch 30 --max-roll 15` — random rotation ranges (deg)
- `--margin 1.0` — required clearance from occupied voxels (m)
- `--seed 42` — RNG seed for reproducibility
- `--no-swap-rb` — same as above

### Voxel map — `generate_voxel_map.py`

Queries Unreal collision over a cubic region and writes a `.binvox` file
on the AirSim host's filesystem.

```bash
# Whole-area map + point cloud export
python3 generate_voxel_map.py --out /mnt/c/temp/env.binvox \
    --cx 0 --cy 0 --cz 0 \
    --size_x 200 --size_y 200 --size_z 30 --res 0.5 \
    --to-points /mnt/c/temp/env_points.npy

# Small fine-grained local map
python3 generate_voxel_map.py --out /mnt/c/temp/local.binvox \
    --cx 10 --cy 5 --cz -5 \
    --size_x 30 --size_y 30 --size_z 10 --res 0.1
```

Cost scales as `(sx * sy * sz) / res³`. 200×200×30 m at 0.5 m is
~9.6 M cells (fine, ~minute). Same volume at 0.1 m is 1.2 B cells (will
hang).

Inspect:
```bash
viewvox    /mnt/c/temp/env.binvox     # 3D viewer
binvox2bt  /mnt/c/temp/env.binvox     # → OctoMap .bt
```

## Output format

Both collectors produce the same layout:

```
<out>/
  cameras.json            # per-camera FOV + extrinsics (saved once)
  metadata.jsonl          # one JSON line per frame
  rgb/
    front/000000.png      # 640×480 BGR PNG
    left/000000.png
    right/000000.png
    down/000000.png
    ...
  depth/
    front/000000.npy      # 640×480 float32, planar depth in meters
    left/000000.npy
    right/000000.npy
    down/000000.npy
    ...
```

A `metadata.jsonl` line:
```json
{"frame": 12,
 "timestamp_ns": 1234567890000,
 "position_ned": [10.0, -3.4, -5.2],
 "orientation_wxyz": [0.92, 0.0, 0.0, 0.39],
 "linear_velocity": [...],
 "collision": false}
```

Pinhole intrinsics from `cameras.json`:
```
fx = fy = (W/2) / tan(FOV/2)
cx = W/2,  cy = H/2
```

## Coordinate system

AirSim uses **NED** in SI units throughout the API:
- +X = forward (north)
- +Y = right (east)
- +Z = **down** — altitude above the spawn point is *negative* z
- The drone always spawns at `(0, 0, 0)` regardless of where Player Start
  sits in Unreal

Angles are degrees in `settings.json`, radians in the API. Pitch = −90°
points straight down.

## Gotchas

Things that came up while building this; capturing here so they don't
re-bite.

- **WSL ↔ Windows paths.** If Python runs in WSL but AirSim runs on
  Windows, files AirSim writes (`simCreateVoxelGrid`) land on the
  Windows filesystem. Use paths under `/mnt/c/...` so both sides see the
  same file. `generate_voxel_map.py` translates via `wslpath -w`, but
  only `/mnt/c/...` translates to a clean `C:\...` path; pure WSL paths
  translate to `\\wsl.localhost\...` UNC paths that some Unreal builds
  reject silently.

- **`simCreateVoxelGrid` "bad cast" error.** The Python doc says the size
  args are floats; the C++ side wants `int`. `generate_voxel_map.py`
  handles this; calling the API directly: `client.simCreateVoxelGrid(
  center, int(sx), int(sy), int(sz), float(res), path)`.

- **`module 'cosysairsim' has no attribute 'to_quaternion'`.**
  cosysairsim doesn't export the helper. Use the inline
  `euler_to_quaternion` in `collect_random.py`.

- **RGB looks wrong (red/blue swapped).** Cosys-AirSim against UE5
  returns the Scene buffer as RGB, but `cv2.imwrite` expects BGR. Both
  collectors swap channels by default. If colors look wrong *after* the
  swap, your build returns BGR — pass `--no-swap-rb`.

- **Side cameras see drone parts.** Means the lens is inside the
  propeller swept area. Increase `Y` offset (further outboard) and/or
  decrease `Z` (above the prop plane) in `settings.json`. The current
  ±0.6 m / −0.1 m values clear the default FlyingPawn.

- **Threading + msgpackrpc.** Don't share one client across threads. Each
  background thread needs its *own* client *and* its own asyncio loop:
  `asyncio.set_event_loop(asyncio.new_event_loop())` before any RPC call.

- **`bad cast` from any other RPC.** Same root cause class — Python sent
  a float where C++ wanted int (or vice versa). Check the C++ signature
  in the Cosys-AirSim source.