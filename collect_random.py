#!/usr/bin/env python3
"""
collect_random.py
-----------------
Teleport the drone to random collision-free poses inside a user-defined
volume and capture synchronized RGBD + state at each one.

Two safety strategies are supported (pick whichever fits your scene):

  1) --map  <map.binvox>  (RECOMMENDED)
       Use a pre-generated occupancy voxel grid (from generate_voxel_map.py)
       to reject any pose closer than --margin meters to occupied geometry.
       This is fast and doesn't risk embedding the drone in walls.

  2) No map (fallback):
       Teleport the drone with ignore_collision=True, run physics for
       --settle seconds with simContinueForTime(), then read
       simGetCollisionInfo. If it has_collided, retry.

The simulation is held PAUSED outside of those brief settle windows, so
gravity won't make the drone drift before you capture.

Usage:
    # Generate a map first (one-time):
    python generate_voxel_map.py --out env.binvox \
        --cx 0 --cy 0 --cz 0 \
        --size_x 200 --size_y 200 --size_z 30 --res 0.5

    # Then collect:
    python collect_random.py --out random_ds --num 300 \
        --bounds " -50,50, -50,50, -20,-1" \
        --map env.binvox --margin 1.0
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import cosysairsim as airsim


CAMERA_NAMES = ["front", "left", "right", "down"]
VEHICLE_NAME = "Drone1"


# ---- shared image plumbing (same as collect_dataset.py) ---------------------

def request_tags():
    return [(c, k) for c in CAMERA_NAMES for k in ("rgb", "depth")]


def build_requests():
    reqs = []
    for cam in CAMERA_NAMES:
        reqs.append(airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False))
        reqs.append(airsim.ImageRequest(cam, airsim.ImageType.DepthPlanar, True, False))
    return reqs


def decode_rgb(r, swap_rb=True):
    """AirSim's Scene buffer is uint8 in either RGB or BGR depending on
    the Unreal pixel format your build uses. UE5 defaults to RGBA, so by
    default we swap R and B before handing to cv2 (which writes BGR).
    Set swap_rb=False if your build is already BGR and colors look wrong
    after this swap."""
    buf = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
    n3 = r.height * r.width * 3
    n4 = r.height * r.width * 4
    if buf.size == n3:
        img = buf.reshape(r.height, r.width, 3)
    elif buf.size == n4:
        img = buf.reshape(r.height, r.width, 4)[:, :, :3]
    else:
        raise RuntimeError(
            f"Unexpected RGB buffer size {buf.size}, "
            f"expected {n3} or {n4} for a {r.width}x{r.height} image"
        )
    if swap_rb:
        img = img[:, :, ::-1].copy()
    return img


def decode_depth(r):
    return airsim.list_to_2d_float_array(
        r.image_data_float, r.width, r.height
    ).astype(np.float32)


def make_dirs(out_dir: Path):
    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    for cam in CAMERA_NAMES:
        (out_dir / "rgb" / cam).mkdir(exist_ok=True)
        (out_dir / "depth" / cam).mkdir(exist_ok=True)


def save_camera_info(client, vehicle, out_path: Path):
    info_dict = {}
    for cam in CAMERA_NAMES:
        info = client.simGetCameraInfo(cam, vehicle)
        p, q = info.pose.position, info.pose.orientation
        info_dict[cam] = {
            "fov_deg": info.fov,
            "pose_in_vehicle": {
                "position_xyz": [p.x_val, p.y_val, p.z_val],
                "orientation_wxyz": [q.w_val, q.x_val, q.y_val, q.z_val],
            },
        }
    with open(out_path, "w") as f:
        json.dump(info_dict, f, indent=2)


# ---- voxel map ingestion ----------------------------------------------------

def read_binvox(path):
    """Read AirSim's binvox output. Returns
        (grid_xyz_bool[nx,ny,nz], world_min_xyz, cell_size_m)."""
    with open(path, "rb") as f:
        if not f.readline().startswith(b"#binvox"):
            raise ValueError(f"Not a binvox file: {path}")
        dims = translate = scale = None
        while True:
            line = f.readline()
            if line.startswith(b"data"):
                break
            tok = line.split()
            if tok[0] == b"dim":
                dims = tuple(int(x) for x in tok[1:4])
            elif tok[0] == b"translate":
                translate = tuple(float(x) for x in tok[1:4])
            elif tok[0] == b"scale":
                scale = float(tok[1])
        raw = f.read()

    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size % 2:
        arr = arr[:-1]
    flat = np.repeat(arr[0::2], arr[1::2]).astype(bool)

    dx, dy, dz = dims
    flat = flat[: dx * dy * dz]
    # AirSim writer order: idx = i + nx*(k + nz*j)  =>  reshape (y, z, x)
    grid = flat.reshape(dy, dz, dx).transpose(2, 0, 1)  # -> (x, y, z)

    cell = scale / max(dims)
    world_min = np.array(translate, dtype=np.float64)
    return grid, world_min, cell


def cell_index(world_xyz, world_min, cell):
    return tuple(int(math.floor((c - m) / cell))
                 for c, m in zip(world_xyz, world_min))


def is_pose_safe_voxel(grid, world_min, cell, position, margin):
    """True iff a cube of half-side `margin` centered on `position`
    contains no occupied voxels and lies fully inside the map."""
    nx, ny, nz = grid.shape
    ix, iy, iz = cell_index(position, world_min, cell)
    r = max(1, int(math.ceil(margin / cell)))
    if (ix - r < 0 or iy - r < 0 or iz - r < 0
            or ix + r >= nx or iy + r >= ny or iz + r >= nz):
        return False
    return not grid[ix - r: ix + r + 1,
                    iy - r: iy + r + 1,
                    iz - r: iz + r + 1].any()


# ---- random sampling --------------------------------------------------------

def parse_bounds(s):
    vals = [float(x) for x in s.replace(" ", "").split(",")]
    if len(vals) != 6:
        raise argparse.ArgumentTypeError(
            "bounds must be x_min,x_max,y_min,y_max,z_min,z_max")
    return (
        (vals[0], vals[1]),
        (vals[2], vals[3]),
        (vals[4], vals[5]),
    )


def sample_pose(bounds, max_pitch_deg, max_roll_deg):
    x = random.uniform(*bounds[0])
    y = random.uniform(*bounds[1])
    z = random.uniform(*bounds[2])
    yaw_deg   = random.uniform(-180.0, 180.0)
    pitch_deg = random.uniform(-max_pitch_deg, max_pitch_deg)
    roll_deg  = random.uniform(-max_roll_deg,  max_roll_deg)
    return x, y, z, pitch_deg, roll_deg, yaw_deg


def euler_to_quaternion(pitch, roll, yaw):
    """Pitch, roll, yaw in RADIANS -> airsim.Quaternionr.
    Matches AirSim's NED Euler->quaternion convention exactly so
    simSetVehiclePose / simGetGroundTruthKinematics interpret it the
    same way. (cosysairsim does not export `to_quaternion` at module
    level, so we do the math ourselves.)"""
    cy, sy = math.cos(yaw * 0.5),   math.sin(yaw * 0.5)
    cr, sr = math.cos(roll * 0.5),  math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    q = airsim.Quaternionr()
    q.w_val = cy * cr * cp + sy * sr * sp
    q.x_val = cy * sr * cp - sy * cr * sp
    q.y_val = cy * cr * sp + sy * sr * cp
    q.z_val = sy * cr * cp - cy * sr * sp
    return q


def make_airsim_pose(x, y, z, pitch_deg, roll_deg, yaw_deg):
    q = euler_to_quaternion(
        math.radians(pitch_deg),
        math.radians(roll_deg),
        math.radians(yaw_deg),
    )
    return airsim.Pose(airsim.Vector3r(x, y, z), q)


# ---- safety check via brief settle (no-map fallback) ------------------------

def is_pose_safe_settle(client, vehicle, pose, settle_s):
    client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=vehicle)
    # run physics briefly so the engine registers any wall intersection
    client.simContinueForTime(settle_s)   # auto-pauses when done
    info = client.simGetCollisionInfo(vehicle)
    return not info.has_collided


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="random_dataset")
    ap.add_argument("--num", type=int, default=200,
                    help="Target number of valid frames")
    ap.add_argument("--bounds", type=parse_bounds,
                    default=parse_bounds("-50,50,-50,50,-20,-1"),
                    help="x_min,x_max,y_min,y_max,z_min,z_max NED meters")
    ap.add_argument("--vehicle", default=VEHICLE_NAME)
    ap.add_argument("--map", default=None,
                    help="Pre-generated binvox for safety check (recommended)")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="Required clearance from occupied voxels (meters)")
    ap.add_argument("--max-tries", type=int, default=200,
                    help="Per-frame retry budget")
    ap.add_argument("--settle", type=float, default=0.05,
                    help="Settle time for no-map collision check (seconds)")
    ap.add_argument("--max-pitch", type=float, default=30.0,
                    help="Max abs pitch (deg). Random in [-max, +max]. Default 30.")
    ap.add_argument("--max-roll", type=float, default=15.0,
                    help="Max abs roll (deg). Random in [-max, +max]. Default 15.")
    ap.add_argument("--no-swap-rb", dest="swap_rb", action="store_false",
                    help="Disable RGB<->BGR swap. Use only if your build "
                         "returns BGR already and colors look wrong WITH the swap.")
    ap.set_defaults(swap_rb=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    out_dir = Path(args.out).resolve()
    make_dirs(out_dir)

    # connect
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, args.vehicle)
    client.armDisarm(False, args.vehicle)   # rotors off; we'll teleport
    save_camera_info(client, args.vehicle, out_dir / "cameras.json")

    # voxel map
    grid = world_min = cell = None
    if args.map:
        grid, world_min, cell = read_binvox(args.map)
        print(f"[map] loaded {grid.shape} grid, cell={cell:.3f} m, "
              f"world_min={world_min}")

    requests = build_requests()
    tags = request_tags()
    meta_f = open(out_dir / "metadata.jsonl", "w")

    # keep sim paused throughout to avoid gravity drift
    client.simPause(True)

    saved = 0
    attempts = 0
    rejected = 0
    print(f"[collect] target {args.num} frames -> {out_dir}")
    print(f"[collect] bounds={args.bounds}  "
          f"strategy={'voxel' if grid is not None else 'settle-collision'}")

    try:
        while saved < args.num:
            # -- find a safe pose --
            chosen = None
            for _ in range(args.max_tries):
                attempts += 1
                x, y, z, pitch, roll, yaw = sample_pose(
                    args.bounds, args.max_pitch, args.max_roll
                )
                pos = (x, y, z)

                if grid is not None:
                    if is_pose_safe_voxel(grid, world_min, cell, pos, args.margin):
                        chosen = (x, y, z, pitch, roll, yaw)
                        break
                else:
                    pose = make_airsim_pose(x, y, z, pitch, roll, yaw)
                    if is_pose_safe_settle(client, args.vehicle, pose, args.settle):
                        chosen = (x, y, z, pitch, roll, yaw)
                        break
                rejected += 1

            if chosen is None:
                print(f"[abort] no safe pose found in {args.max_tries} tries. "
                      f"Try widening bounds, lowering --margin, "
                      f"or regenerating the voxel map.")
                break

            x, y, z, pitch, roll, yaw = chosen
            pose = make_airsim_pose(x, y, z, pitch, roll, yaw)
            # final placement (re-set in case the settle path nudged it)
            client.simSetVehiclePose(pose, ignore_collision=True,
                                     vehicle_name=args.vehicle)
            client.simPause(True)  # ensure paused for the capture

            # -- capture: all 8 images + state from same paused tick --
            responses = client.simGetImages(requests, args.vehicle)
            gt = client.simGetGroundTruthKinematics(args.vehicle)
            ms_ts = client.getMultirotorState(args.vehicle).timestamp

            if len(responses) != len(tags):
                print(f"[warn] frame {saved}: bad response count, skipping")
                continue

            # -- save images --
            for r, (cam, kind) in zip(responses, tags):
                if kind == "rgb":
                    cv2.imwrite(
                        str(out_dir / "rgb" / cam / f"{saved:06d}.png"),
                        decode_rgb(r, swap_rb=args.swap_rb),
                    )
                else:
                    np.save(
                        str(out_dir / "depth" / cam / f"{saved:06d}.npy"),
                        decode_depth(r),
                    )

            # -- save state --
            p, q = gt.position, gt.orientation
            meta = {
                "frame": saved,
                "timestamp_ns": ms_ts,
                "requested_pose": {
                    "position_ned": [x, y, z],
                    "pitch_deg": pitch,
                    "roll_deg":  roll,
                    "yaw_deg":   yaw,
                },
                "actual_pose": {
                    "position_ned": [p.x_val, p.y_val, p.z_val],
                    "orientation_wxyz": [q.w_val, q.x_val, q.y_val, q.z_val],
                },
            }
            meta_f.write(json.dumps(meta) + "\n")
            meta_f.flush()

            saved += 1
            if saved % 10 == 0:
                rate = saved / max(1, attempts)
                print(f"[{saved:5d}/{args.num}]  "
                      f"attempts={attempts}  rejected={rejected}  "
                      f"acceptance={rate:.1%}")

    finally:
        meta_f.close()
        try:
            client.simPause(False)
        except Exception:
            pass
        client.armDisarm(False, args.vehicle)
        client.enableApiControl(False, args.vehicle)
        print(f"[done] saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()