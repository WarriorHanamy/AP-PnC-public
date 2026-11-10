#!/usr/bin/env python3
"""Download SUPER's high-speed PCD and write a lidar_sim xyz PLY under .artifacts.

The upstream map is `random_map_24_6635.pcd` from hku-mars/SUPER
(`mars_uav_sim/perfect_drone_sim/pcd/`). It is not committed here. The PCD and
the converted PLY stay under `$AP_PNC_DIR/.artifacts/`.

Upstream config used by the bringup (not copied into this repository):
`mars_uav_sim/perfect_drone_sim/config/high_speed.yaml`
start (0, -50, 1.5), yaw 1.5741, waypoints (0, 50, 1.5) and (0, -50, 1.5).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

SUPER_PCD_URL = (
    "https://raw.githubusercontent.com/hku-mars/SUPER/master/"
    "mars_uav_sim/perfect_drone_sim/pcd/random_map_24_6635.pcd"
)
DEFAULT_PCD = ".artifacts/dev-sim/scenes/random_map_24_6635.pcd"
DEFAULT_PLY = ".artifacts/dev-sim/scenes/random_map_24_6635.ply"


def _import_env():
    try:
        from tooling.env import artifact_path, project_path
    except ImportError:
        root = Path(__file__).resolve().parents[4]
        sys.path.insert(0, str(root))
        from tooling.env import artifact_path, project_path
    return artifact_path, project_path


def _header(handle):
    fields, sizes, types, counts = [], [], [], []
    points = None
    data = None
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("PCD header truncated")
        parts = line.decode("ascii", "replace").strip().split()
        if not parts:
            continue
        key = parts[0].upper()
        if key == "FIELDS":
            fields = parts[1:]
        elif key == "SIZE":
            sizes = [int(item) for item in parts[1:]]
        elif key == "TYPE":
            types = parts[1:]
        elif key == "COUNT":
            counts = [int(item) for item in parts[1:]]
        elif key == "POINTS":
            points = int(parts[1])
        elif key == "DATA":
            if len(parts) != 2:
                raise ValueError("PCD DATA line is missing its format")
            data = parts[1].lower()
            break
    if not fields or points is None or data is None:
        raise ValueError("PCD header is missing FIELDS, POINTS, or DATA")
    if not counts:
        counts = [1] * len(fields)
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("PCD FIELDS, SIZE, TYPE, and COUNT disagree")
    if any(count != 1 for count in counts):
        raise ValueError("PCD COUNT other than 1 is not supported")
    return fields, sizes, types, points, data


def read_pcd_xyz(path: Path) -> np.ndarray:
    """Read x, y, z from an uncompressed PCD. binary_compressed is rejected."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("PCD path must be absolute")
    with path.open("rb") as handle:
        fields, sizes, types, count, data = _header(handle)
        if data == "binary_compressed":
            raise ValueError("compressed PCD is not supported; export an uncompressed PCD")
        index = {name: position for position, name in enumerate(fields)}
        missing = {"x", "y", "z"} - set(index)
        if missing:
            raise ValueError(f"PCD is missing {sorted(missing)}")
        if data == "ascii":
            rows = []
            for _ in range(count):
                row = handle.readline().split()
                if len(row) < len(fields):
                    raise ValueError("PCD ASCII row is shorter than FIELDS")
                rows.append([float(row[index[axis]]) for axis in "xyz"])
            points = np.asarray(rows, dtype=np.float32)
        elif data == "binary":
            for axis in "xyz":
                slot = index[axis]
                if types[slot] != "F" or sizes[slot] != 4:
                    raise ValueError("binary PCD x, y, z must be float32")
            point_step = sum(sizes)
            raw = handle.read(count * point_step)
            if len(raw) != count * point_step:
                raise ValueError("PCD binary payload is truncated")
            offsets = np.cumsum([0, *sizes[:-1]])
            parsed = np.frombuffer(raw, dtype=np.dtype({
                "names": ["x", "y", "z"],
                "formats": ["<f4", "<f4", "<f4"],
                "offsets": [int(offsets[index[axis]]) for axis in "xyz"],
                "itemsize": point_step,
            }), count=count)
            points = np.column_stack((parsed["x"], parsed["y"], parsed["z"])).astype(np.float32, copy=False)
        else:
            raise ValueError(f"unsupported PCD DATA format {data}")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("PCD produced no xyz points")
    finite = np.isfinite(points).all(axis=1)
    dropped = int((~finite).sum())
    points = np.ascontiguousarray(points[finite])
    if len(points) == 0:
        raise ValueError("PCD produced no finite xyz points")
    if dropped:
        print(f"dropped {dropped} nonfinite PCD points", file=sys.stderr)
    return points


def write_xyz_ply(path: Path, points: np.ndarray) -> None:
    points = np.ascontiguousarray(points, dtype=np.float32)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode())
        handle.write(points.tobytes())


def convert_pcd_to_ply(pcd_path: Path, ply_path: str | Path) -> Path:
    """Convert `pcd_path` and write the PLY under `.artifacts/`."""
    artifact_path, _project_path = _import_env()
    source = Path(pcd_path)
    if not source.is_absolute():
        raise ValueError("PCD path must be absolute")
    output = artifact_path(ply_path)
    write_xyz_ply(output, read_pcd_xyz(source))
    return output


def download_pcd(destination: str | Path = DEFAULT_PCD, url: str = SUPER_PCD_URL) -> Path:
    """Download the upstream PCD into `.artifacts/` and return that absolute path."""
    artifact_path, _project_path = _import_env()
    output = artifact_path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output)
    return output


def prepare(pcd: Path | None = None, output: str | Path = DEFAULT_PLY) -> Path:
    artifact_path, _project_path = _import_env()
    source = pcd if pcd is not None else download_pcd()
    ply = convert_pcd_to_ply(source, output)
    print(artifact_path(ply))
    return ply


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, default=None,
                        help="Absolute uncompressed PCD. Omit to download the SUPER high-speed map.")
    parser.add_argument("--output", default=DEFAULT_PLY,
                        help="PLY path under $AP_PNC_DIR/.artifacts (absolute or project-relative)")
    args = parser.parse_args(argv)
    if args.pcd is not None and not args.pcd.is_absolute():
        parser.error("--pcd must be absolute")
    prepare(args.pcd, args.output)


if __name__ == "__main__":
    main()
