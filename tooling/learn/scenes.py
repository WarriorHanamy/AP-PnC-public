"""Post-CIRI scene directory.

This repository does not contain CIRI reconstructions. A scene is a directory
the caller prepares:

```
scene.yaml          source: post-ciri
occupancy.ply       xyz metres, ENU, from the caller's CIRI output
```

``source`` must be the string ``post-ciri``. Other labels, including synthetic
rooms used by the smoke check, are rejected here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from tooling.learn.occupancy import Occupancy

MISSING_SCENE = (
    "Post-CIRI scene not found. This checkout has no CIRI assets. "
    "CityGaussian/MatrixCity demos and the pillar lidar scene are different inputs. "
    "Place scene.yaml with source: post-ciri and an occupancy PLY from your CIRI "
    "reconstruction in an absolute directory, then pass that directory as --scene."
)


@dataclass(frozen=True)
class PostCiriScene:
    name: str
    directory: Path
    occupancy: Occupancy
    start: np.ndarray
    goal: np.ndarray

    @property
    def points(self) -> np.ndarray:
        return self.occupancy.points


def _vector3(value: object, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be three finite numbers")
    return array


def _load_xyz_ply(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if not raw.startswith(b"ply"):
        raise ValueError(f"{path} is not a PLY file")
    header, _, body = raw.partition(b"end_header\n")
    if not body and b"end_header\r\n" in raw:
        header, _, body = raw.partition(b"end_header\r\n")
    lines = header.splitlines()
    ascii_ply = any(line.startswith(b"format ascii") for line in lines)
    count = None
    names: list[str] = []
    for line in lines:
        parts = line.split()
        if parts[:2] == [b"element", b"vertex"]:
            count = int(parts[2])
        elif parts[:1] == [b"property"]:
            names.append(parts[-1].decode())
    if count is None or "x" not in names or "y" not in names or "z" not in names:
        raise ValueError(f"{path} has no xyz vertex properties")
    if ascii_ply:
        rows = [line.split() for line in body.splitlines() if line.strip()][:count]
        if len(rows) != count:
            raise ValueError(f"{path} ended before {count} vertices")
        index = {name: position for position, name in enumerate(names)}
        points = np.array(
            [[float(row[index[axis]]) for axis in ("x", "y", "z")] for row in rows],
            dtype=float,
        )
    else:
        dtype = np.dtype([(name, "<f4") for name in names])
        vertices = np.frombuffer(body, dtype=dtype, count=count)
        if len(vertices) != count:
            raise ValueError(f"{path} ended before {count} vertices")
        points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(float)
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError(f"{path} has no finite points")
    return points


def load_post_ciri_scene(directory: Path) -> PostCiriScene:
    if not directory.is_absolute():
        raise ValueError("post-CIRI scene directory must be absolute")
    scene_yaml = directory / "scene.yaml"
    if not scene_yaml.is_file():
        raise FileNotFoundError(MISSING_SCENE + f" Missing {scene_yaml}.")
    document = yaml.safe_load(scene_yaml.read_text())
    if not isinstance(document, dict):
        raise ValueError("scene.yaml must be a mapping")
    if document.get("source") != "post-ciri":
        raise ValueError(
            "train and eval accept only source: post-ciri. "
            "A synthetic smoke room is not a CIRI scene."
        )
    name = str(document.get("name", directory.name))
    if not name or any(character in name for character in "/\\."):
        raise ValueError("scene name must be a single path segment")
    relative = Path(str(document.get("occupancy", "occupancy.ply")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("occupancy path must stay inside the scene directory")
    occupancy_path = (directory / relative).resolve()
    if not occupancy_path.is_relative_to(directory.resolve()):
        raise ValueError("occupancy path must stay inside the scene directory")
    if not occupancy_path.is_file():
        raise FileNotFoundError(MISSING_SCENE + f" Missing {occupancy_path}.")
    voxel_m = float(document.get("voxel_m", 0.4))
    ground_z = float(document.get("ground_z", 0.0))
    occupancy = Occupancy(_load_xyz_ply(occupancy_path), voxel_m, ground_z)
    start = _vector3(document["start"], "start")
    goal = _vector3(document["goal"], "goal")
    if occupancy.clearance(start[None, :])[0] < 0.3:
        raise ValueError("start is inside the post-CIRI occupancy")
    if np.linalg.norm(goal - start) < 0.5:
        raise ValueError("goal must be at least 0.5 m from the start")
    return PostCiriScene(name, directory.resolve(), occupancy, start, goal)
