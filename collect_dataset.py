#!/usr/bin/env python3
"""
collect_dataset.py
------------------
Collect synchronized multi-camera RGBD + vehicle-state frames from a
running Cosys-AirSim binary configured with 4 cameras: front, left,
right, down (see companion settings.json).

For every frame it pauses the sim, grabs ground-truth kinematics, then
issues ONE simGetImages call that returns all 8 images (4 cams x
{Scene, DepthPlanar}) captured at the same simulation tick, then
unpauses. This guarantees images and state are perfectly aligned.

Output layout:
    <out>/
        cameras.json                   # FOV + extrinsics (saved once)
        metadata.jsonl                 # one JSON line per frame
        rgb/<cam>/000000.png ...
        depth/<cam>/000000.npy ...     # float32, meters, planar depth

Usage:
    python collect_dataset.py --out dataset --num 500 --interval 0.2

If you want to drive the drone yourself (keyboard / RC) just leave the
script running and fly around. If you want it to fly itself, set
--auto-fly and edit the WAYPOINTS list near the bottom of the file.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import cosysairsim as airsim


# ---- Configuration constants ------------------------------------------------

CAMERA_NAMES = ["front", "left", "right", "down"]
VEHICLE_NAME = "Drone1"

# Maps an ImageRequest into a (camera, kind) tag so we can route responses.
# Order MUST match the order requests are appended in build_requests().
def request_tags():
    tags = []
    for cam in CAMERA_NAMES:
        tags.append((cam, "rgb"))
        tags.append((cam, "depth"))
    return tags


def build_requests():
    """One ImageRequest per (camera, image_type)."""
    reqs = []
    for cam in CAMERA_NAMES:
        # Scene / RGB:  pixels_as_float=False, compress=False -> raw uint8 buffer
        reqs.append(airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False))
        # DepthPlanar:  pixels_as_float=True -> floats in meters, plane-parallel
        reqs.append(airsim.ImageRequest(cam, airsim.ImageType.DepthPlanar, True, False))
    return reqs


# ---- Decoding ---------------------------------------------------------------

def decode_rgb(response, swap_rb=True):
    """Cosys-AirSim's Scene buffer is uint8 in either RGB or BGR depending
    on the Unreal pixel format your build uses. UE5 defaults to RGBA, so
    by default we swap R and B before handing to cv2 (which writes BGR).
    Set swap_rb=False if your build is BGR and colors look wrong AFTER
    this swap."""
    buf = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    n3 = response.height * response.width * 3
    n4 = response.height * response.width * 4
    if buf.size == n3:
        img = buf.reshape(response.height, response.width, 3)
    elif buf.size == n4:
        img = buf.reshape(response.height, response.width, 4)[:, :, :3]
    else:
        raise RuntimeError(
            f"RGB buffer size mismatch: got {buf.size}, "
            f"expected {n3} or {n4}. Camera may not have rendered yet -- "
            f"try increasing --warmup."
        )
    if swap_rb:
        img = img[:, :, ::-1].copy()
    return img


def decode_depth(response):
    """DepthPlanar with pixels_as_float=True -> float32 meters."""
    if not response.pixels_as_float:
        raise RuntimeError("Expected float depth response.")
    depth = airsim.list_to_2d_float_array(
        response.image_data_float, response.width, response.height
    )
    return depth.astype(np.float32)


# ---- Setup ------------------------------------------------------------------

def make_dirs(out_dir: Path):
    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    for cam in CAMERA_NAMES:
        (out_dir / "rgb" / cam).mkdir(exist_ok=True)
        (out_dir / "depth" / cam).mkdir(exist_ok=True)


def save_camera_info(client, vehicle, out_path: Path):
    """Snapshot per-camera FOV + extrinsics (relative to vehicle body)."""
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
    print(f"[setup] camera info -> {out_path}")


# ---- Optional auto-flight ---------------------------------------------------

WAYPOINTS = [
    # (x, y, z, velocity)  -- NED meters
    ( 10,   0, -5, 3),
    ( 10,  10, -5, 3),
    (  0,  10, -5, 3),
    (  0,   0, -5, 3),
]


def fly_waypoints(vehicle):
    """Run in a side thread. msgpackrpc needs an asyncio event loop in
    every thread that calls it, and the RPC client itself isn't safe to
    share across threads -- so we make our own of each here."""
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, vehicle)
    client.armDisarm(True, vehicle)

    print("[flight] takeoff")
    client.takeoffAsync(vehicle_name=vehicle).join()
    for (x, y, z, v) in WAYPOINTS:
        print(f"[flight] -> ({x},{y},{z}) at {v} m/s")
        client.moveToPositionAsync(x, y, z, v, vehicle_name=vehicle).join()
    print("[flight] waypoints complete (drone will hover)")


# ---- Main loop --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset", help="Output directory")
    ap.add_argument("--num", type=int, default=200, help="Number of frames to collect")
    ap.add_argument("--interval", type=float, default=0.2,
                    help="Wall-clock seconds between captures")
    ap.add_argument("--vehicle", default=VEHICLE_NAME)
    ap.add_argument("--no-pause", action="store_true",
                    help="Don't pause/unpause around capture (faster, less synced)")
    ap.add_argument("--auto-fly", action="store_true",
                    help="Run the WAYPOINTS list in a background thread")
    ap.add_argument("--warmup", type=float, default=1.0,
                    help="Seconds to wait after connect for cameras to render")
    ap.add_argument("--no-swap-rb", dest="swap_rb", action="store_false",
                    help="Disable RGB<->BGR swap. Use only if your build "
                         "returns BGR already and colors look wrong WITH the swap.")
    ap.set_defaults(swap_rb=True)
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    make_dirs(out_dir)

    # --- connect ---
    client = airsim.MultirotorClient()
    client.confirmConnection()
    if not args.auto_fly:
        # In auto-fly mode the flight thread arms; doing it twice is harmless
        # but cleaner to skip. We still want API control on the main client
        # so that simPause / simGetImages / simGetGroundTruthKinematics work.
        client.enableApiControl(True, args.vehicle)
        client.armDisarm(True, args.vehicle)
    time.sleep(args.warmup)

    save_camera_info(client, args.vehicle, out_dir / "cameras.json")

    # --- optional auto-flight in a side thread (creates its own client) ---
    if args.auto_fly:
        import threading
        threading.Thread(target=fly_waypoints,
                         args=(args.vehicle,), daemon=True).start()

    requests = build_requests()
    tags = request_tags()
    meta_path = out_dir / "metadata.jsonl"
    meta_f = open(meta_path, "w")

    print(f"[collect] {args.num} frames, every {args.interval}s, -> {out_dir}")
    t0 = time.time()

    try:
        for i in range(args.num):
            if not args.no_pause:
                client.simPause(True)

            # Ground-truth state -- guaranteed from same paused tick
            gt = client.simGetGroundTruthKinematics(args.vehicle)
            ms = client.getMultirotorState(args.vehicle)

            # All 8 images from this same tick in one RPC
            responses = client.simGetImages(requests, args.vehicle)

            if not args.no_pause:
                client.simPause(False)

            if len(responses) != len(tags):
                print(f"[warn] frame {i}: got {len(responses)} responses, "
                      f"expected {len(tags)} -- skipping")
                continue

            # Save images
            for r, (cam, kind) in zip(responses, tags):
                if kind == "rgb":
                    img = decode_rgb(r, swap_rb=args.swap_rb)
                    cv2.imwrite(str(out_dir / "rgb" / cam / f"{i:06d}.png"), img)
                else:
                    depth = decode_depth(r)
                    np.save(str(out_dir / "depth" / cam / f"{i:06d}.npy"), depth)

            # Save state
            p, q = gt.position, gt.orientation
            v, w = gt.linear_velocity, gt.angular_velocity
            la, aa = gt.linear_acceleration, gt.angular_acceleration
            meta = {
                "frame": i,
                "timestamp_ns": ms.timestamp,
                "position_ned":      [p.x_val, p.y_val, p.z_val],
                "orientation_wxyz":  [q.w_val, q.x_val, q.y_val, q.z_val],
                "linear_velocity":   [v.x_val, v.y_val, v.z_val],
                "angular_velocity":  [w.x_val, w.y_val, w.z_val],
                "linear_acc":        [la.x_val, la.y_val, la.z_val],
                "angular_acc":       [aa.x_val, aa.y_val, aa.z_val],
                "collision":         client.simGetCollisionInfo(args.vehicle).has_collided,
            }
            meta_f.write(json.dumps(meta) + "\n")
            meta_f.flush()

            if (i + 1) % 20 == 0:
                fps = (i + 1) / (time.time() - t0)
                print(f"[{i+1:5d}/{args.num}]  effective {fps:5.2f} fps")

            time.sleep(args.interval)

    finally:
        meta_f.close()
        try:
            client.simPause(False)
        except Exception:
            pass
        client.armDisarm(False, args.vehicle)
        client.enableApiControl(False, args.vehicle)
        print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()