#!/usr/bin/env python3
"""
generate_voxel_map.py
---------------------
Build a ground-truth occupancy voxel grid of the Cosys-AirSim world
around a chosen center point, save as .binvox, and (optionally) export
the occupied voxel centers as a .npy point cloud.

Cosys-AirSim implements `simCreateVoxelGrid(position, x, y, z, res, of)`
where x/y/z are the cube-region sizes in meters and `res` is the cell
edge in meters. See:
  https://cosys-lab.github.io/Cosys-AirSim/voxel_grid/

Notes:
- This is GROUND TRUTH from Unreal collision queries, not a sensor.
- Cost grows with (size_x * size_y * size_z) / res^3, so prefer either:
    * a moderate res (>= 0.5 m) for a whole map, OR
    * a small region (e.g. 30 m cube) at fine res (0.1 m).
- The output .binvox can be visualized with `viewvox map.binvox`
  and converted to OctoMap with `binvox2bt map.binvox`.

Usage:
    python generate_voxel_map.py --out map.binvox \
        --cx 0 --cy 0 --cz 0 \
        --size_x 100 --size_y 100 --size_z 30 --res 0.5 \
        --to-points map_points.npy
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import cosysairsim as airsim


# ---- Voxel grid generation --------------------------------------------------

def _is_wsl():
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


def _to_windows_path(linux_path):
    """Use `wslpath -w` to convert a Linux path to a Windows path."""
    try:
        r = subprocess.run(["wslpath", "-w", linux_path],
                           capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def make_voxel_grid(out_path: str, center_xyz, size_xyz, res):
    client = airsim.VehicleClient()
    client.confirmConnection()

    cx, cy, cz = center_xyz
    sx, sy, sz = size_xyz

    # IMPORTANT: AirSim's simCreateVoxelGrid declares the size arguments
    # as int (meters) on the C++ side, even though the Python docstring
    # says "float". Passing floats triggers a server-side `bad cast`.
    sx_i, sy_i, sz_i = int(round(sx)), int(round(sy)), int(round(sz))
    if (sx_i, sy_i, sz_i) != (sx, sy, sz):
        print(f"[voxel] note: non-integer sizes rounded to "
              f"({sx_i},{sy_i},{sz_i}) m")

    nc = int(sx_i / res) * int(sy_i / res) * int(sz_i / res)
    print(f"[voxel] center=({cx},{cy},{cz})  size=({sx_i},{sy_i},{sz_i}) m  "
          f"res={res} m  ~{nc:,} cells")

    out_abs = os.path.abspath(out_path)

    # AirSim writes the file from its own filesystem, NOT the client's.
    # If we're in WSL, the AirSim binary almost certainly runs on the
    # Windows host and won't understand a Linux-style path.
    in_wsl = _is_wsl()
    server_path = out_abs
    if in_wsl:
        win = _to_windows_path(out_abs)
        if win is None:
            print("[warn] WSL detected but `wslpath` is unavailable. "
                  "If AirSim runs on Windows the write will likely fail "
                  "silently.")
        else:
            print(f"[voxel] WSL detected; sending Windows path to AirSim:")
            print(f"        {win}")
            server_path = win

    center = airsim.Vector3r(float(cx), float(cy), float(cz))
    client.simCreateVoxelGrid(center, sx_i, sy_i, sz_i, float(res), server_path)

    # Verify the file actually exists on our side. simCreateVoxelGrid
    # returns happily even if the path was nonsense to AirSim.
    if os.path.exists(out_abs):
        n_bytes = os.path.getsize(out_abs)
        print(f"[voxel] wrote {out_abs} ({n_bytes:,} bytes)")
        return out_abs

    # Failure: file isn't where we expected it to be.
    print(f"[error] RPC succeeded but no file at {out_abs}")
    if in_wsl:
        suggested = f"/mnt/c/temp/{os.path.basename(out_abs)}"
        print( "        AirSim on Windows cannot reliably write to a")
        print( "        WSL-only path even with `wslpath` translation")
        print( "        (UNC paths via \\\\wsl.localhost\\... can fail in")
        print( "        some build configurations).")
        print( "        Use a path on a Windows drive that WSL can also see:")
        print(f"          mkdir -p /mnt/c/temp")
        print(f"          python3 generate_voxel_map.py --out '{suggested}' ...")
    else:
        print( "        Check AirSim's current working directory; the write")
        print( "        may have landed there instead.")
    sys.exit(1)


# ---- Minimal binvox reader --------------------------------------------------
# Cosys-AirSim writes binvox following Patrick Min's spec. The ordering used
# in the writer (see Cosys-AirSim docs):
#     idx = i + ncells_x * (k + ncells_z * j)
# i.e. linear order is (j-slow, k-mid, i-fast) == (y, z, x).
# After reshape to (ny, nz, nx), transposing axes (2,0,1) gives a clean
# [x, y, z]-indexed bool array.

def read_binvox(path):
    with open(path, "rb") as f:
        line = f.readline()
        if not line.startswith(b"#binvox"):
            raise ValueError(f"Not a binvox file: {path}")

        dims = translate = scale = None
        while True:
            line = f.readline()
            if line.startswith(b"data"):
                break
            tok = line.split()
            key = tok[0]
            if key == b"dim":
                dims = tuple(int(x) for x in tok[1:4])
            elif key == b"translate":
                translate = tuple(float(x) for x in tok[1:4])
            elif key == b"scale":
                scale = float(tok[1])
        if dims is None or translate is None or scale is None:
            raise ValueError("Malformed binvox header")

        raw = f.read()

    # RLE decode: alternating (value, count) bytes
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size % 2 != 0:
        arr = arr[:-1]
    values = arr[0::2]
    counts = arr[1::2]
    flat = np.repeat(values, counts).astype(bool)

    # binvox dims are (dx, dy, dz) but write order in AirSim is y-slow,
    # z-mid, x-fast.  Reshape accordingly then transpose to (x, y, z).
    dx, dy, dz = dims
    if flat.size != dx * dy * dz:
        # Fall back to total
        flat = flat[: dx * dy * dz]
    grid_yzx = flat.reshape(dy, dz, dx)
    grid_xyz = grid_yzx.transpose(2, 0, 1)  # (x, y, z)

    return grid_xyz, scale, np.array(translate, dtype=np.float64), dims


def binvox_to_points(binvox_path: str, out_npy: str):
    grid, scale, translate, dims = read_binvox(binvox_path)
    nx, ny, nz = grid.shape
    cell = scale / max(dims)  # binvox normalizes longest axis to `scale`

    occ = np.argwhere(grid)  # Nx3 integer indices in (x, y, z)
    # voxel CENTERS in the world frame used by simCreateVoxelGrid
    pts = (occ.astype(np.float64) + 0.5) * cell + translate
    pts = pts.astype(np.float32)

    np.save(out_npy, pts)
    print(f"[points] {len(pts):,} occupied voxels  "
          f"(cell={cell:.3f} m)  -> {out_npy}")
    return pts


# ---- CLI --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="map.binvox", help="Output .binvox path")
    ap.add_argument("--cx", type=float, default=0.0, help="Center X (NED, m)")
    ap.add_argument("--cy", type=float, default=0.0, help="Center Y (NED, m)")
    ap.add_argument("--cz", type=float, default=0.0, help="Center Z (NED, m)")
    ap.add_argument("--size_x", type=float, default=100.0, help="Cube X size (m)")
    ap.add_argument("--size_y", type=float, default=100.0, help="Cube Y size (m)")
    ap.add_argument("--size_z", type=float, default=30.0,  help="Cube Z size (m)")
    ap.add_argument("--res",    type=float, default=0.5,   help="Cell size (m)")
    ap.add_argument("--to-points", default=None,
                    help="If given, also export occupied voxel centers to this .npy")
    args = ap.parse_args()

    binvox_path = make_voxel_grid(
        args.out,
        (args.cx, args.cy, args.cz),
        (args.size_x, args.size_y, args.size_z),
        args.res,
    )

    if args.to_points:
        binvox_to_points(binvox_path, args.to_points)

    print("\nNext steps:")
    print(f"  viewvox {binvox_path}              # 3D viewer")
    print(f"  binvox2bt {binvox_path}            # -> OctoMap .bt")


if __name__ == "__main__":
    main()