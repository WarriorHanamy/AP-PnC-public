"""Synthetic radar recordings for the two learning pipelines.

Each view accumulates radar sweeps in the world frame. From that same cloud:

- RL stores a SUPER CIRI h-polytope (configuration-space corridor).
- YOPO stores the depth image from erosion followed by projection.

The obstacle field that casts the rays is synthetic. The manifest says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tooling.env import artifact_path
from tooling.learn.ciri import MAX_PLANES, corridor_from_points, pad_planes
from tooling.learn.occupancy import Occupancy
from tooling.learn.quadrotor import rotation_world_from_body
from tooling.learn.radar_depth import (
    RadarDepthConfig,
    accumulate,
    depth_from_accumulated,
    radar_points,
)

SCHEMA = "ap-pnc/radar-accumulated/v1"
MAX_CLOUD_POINTS = 256
WARMUP_SWEEPS = 2
SEED_LENGTH_M = 1.2


@dataclass(frozen=True)
class RadarSample:
    depth: np.ndarray
    valid: np.ndarray
    accumulated_points: np.ndarray
    planes: np.ndarray
    plane_count: int
    ciri_success: bool
    position: np.ndarray
    rpy: np.ndarray
    env_index: int
    velocity_body: np.ndarray
    acceleration_body: np.ndarray
    goal_direction_body: np.ndarray


def dataset_root(name: str) -> Path:
    if not name or any(character in name for character in "/\\."):
        raise ValueError("dataset name must be a single path segment")
    directory = artifact_path(f".artifacts/learn/datasets/{name}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_dataset(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("dataset directory must be absolute")
    resolved = path.resolve()
    base = artifact_path(".artifacts/learn/datasets")
    if not resolved.is_relative_to(base):
        raise ValueError(f"dataset must live under {base}")
    manifest = resolved / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"No radar dataset at {resolved}. Run `ap-pnc learn collect` first."
        )
    document = json.loads(manifest.read_text())
    if document.get("schema") != SCHEMA:
        raise ValueError(f"{manifest} is not a {SCHEMA} dataset")
    return resolved


def _environment(env_index: int) -> Occupancy:
    """Synthetic walls used only to cast radar rays."""
    shift = 0.15 * env_index
    xs = np.linspace(0.0, 6.0, 25)
    zs = np.linspace(0.4, 1.6, 4)
    columns = []
    for z in zs:
        columns.append(np.column_stack((xs, np.full_like(xs, 1.5 + shift), np.full_like(xs, z))))
        columns.append(np.column_stack((xs, np.full_like(xs, -1.5), np.full_like(xs, z))))
    return Occupancy(np.vstack(columns), voxel_m=0.4, ground_z=0.0)


def _pad_cloud(clouds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array([min(len(cloud), MAX_CLOUD_POINTS) for cloud in clouds], dtype=np.int32)
    padded = np.zeros((len(clouds), MAX_CLOUD_POINTS, 3), dtype=np.float32)
    for index, cloud in enumerate(clouds):
        kept = cloud[:MAX_CLOUD_POINTS]
        if len(kept):
            padded[index, : len(kept)] = kept
    return counts, padded


def collect_dataset(
    name: str,
    envs: int,
    samples_per_env: int,
    seed: int,
    config: RadarDepthConfig | None = None,
) -> Path:
    """Accumulate synthetic radar sweeps and store CIRI planes plus eroded depth."""
    if envs < 1 or samples_per_env < 1:
        raise ValueError("envs and samples_per_env must be positive")
    sensor = config or RadarDepthConfig(range_noise_m=0.0, keep_probability=1.0)
    root = dataset_root(name)
    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    for env_index in range(envs):
        occupancy = _environment(env_index)
        folder = raw / f"{env_index:03d}"
        folder.mkdir(exist_ok=True)
        np.save(folder / "environment.npy", occupancy.points.astype(np.float32))
        depths, valids, positions, rpys = [], [], [], []
        planes, counts, successes, clouds = [], [], [], []
        world_sweeps: list[np.ndarray] = []
        total = samples_per_env + WARMUP_SWEEPS
        for sweep in range(total):
            position = np.array([0.4 + 0.45 * sweep, 0.0, 1.0])
            yaw = 0.0
            rotation = rotation_world_from_body(0.0, 0.0, yaw)
            body = radar_points(occupancy, position, rotation, sensor, rng)
            world = body @ rotation.T + position if len(body) else np.zeros((0, 3))
            world_sweeps.append(world)
            if sweep < WARMUP_SWEEPS:
                continue
            accumulated = accumulate(world_sweeps)
            if len(accumulated) > MAX_CLOUD_POINTS:
                accumulated = accumulated[-MAX_CLOUD_POINTS:]
            body_cloud = (accumulated - position) @ rotation if len(accumulated) else accumulated
            depth, valid = depth_from_accumulated(body_cloud, sensor)
            seed_end = position + rotation @ np.array([SEED_LENGTH_M, 0.0, 0.0])
            polytope = corridor_from_points(accumulated, position, seed_end)
            block, plane_count = pad_planes(polytope)
            depths.append(depth)
            valids.append(valid)
            positions.append(position)
            rpys.append(np.array([0.0, 0.0, yaw]))
            planes.append(block)
            counts.append(plane_count)
            successes.append(polytope.success)
            clouds.append(accumulated.astype(np.float32))
        cloud_counts, cloud_pad = _pad_cloud(clouds)
        np.savez(
            folder / "label.npz",
            depth=np.stack(depths).astype(np.float32),
            valid=np.stack(valids),
            positions=np.stack(positions).astype(np.float32),
            rpy=np.stack(rpys).astype(np.float32),
            planes=np.stack(planes),
            plane_count=np.asarray(counts, dtype=np.int32),
            ciri_success=np.asarray(successes, dtype=bool),
            cloud_count=cloud_counts,
            accumulated=cloud_pad,
        )
    manifest = {
        "schema": SCHEMA,
        "synthetic": True,
        "label": "synthetic radar sweeps accumulated in a corridor; not a field recording",
        "pipelines": {
            "rl": ["accumulate", "ciri"],
            "yopo": ["accumulate", "erosion", "projection"],
        },
        "ciri": {
            "method": "SUPER CIRI (configuration-space iterative regional inflation)",
            "source": "hku-mars/SUPER",
            "planes": "n·x + d <= 0",
            "max_planes": MAX_PLANES,
            "robot_r_m": 0.2,
            "iterations": 2,
        },
        "envs": envs,
        "samples_per_env": samples_per_env,
        "warmup_sweeps": WARMUP_SWEEPS,
        "seed": seed,
        "depth": {
            "height": sensor.height,
            "width": sensor.width,
            "azimuth_fov_deg": sensor.azimuth_fov_deg,
            "elevation_fov_deg": sensor.elevation_fov_deg,
            "max_range_m": sensor.max_range_m,
            "erosion_voxel_m": sensor.erosion_voxel_m,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return root


def pack_dataset(directory: Path, seed: int = 0, val_fraction: float = 0.2) -> Path:
    """Shuffle recorded views into train and val, and keep each environment map."""
    root = resolve_dataset(directory)
    folders = sorted(path for path in (root / "raw").iterdir() if path.is_dir())
    if not folders:
        raise FileNotFoundError(f"{root / 'raw'} has no environment folders")
    arrays: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "depth",
            "valid",
            "positions",
            "rpy",
            "planes",
            "plane_count",
            "ciri_success",
            "cloud_count",
            "accumulated",
        )
    }
    env_ids: list[np.ndarray] = []
    environments = root / "environments"
    environments.mkdir(exist_ok=True)
    for env_index, folder in enumerate(folders):
        label = np.load(folder / "label.npz")
        for key in arrays:
            arrays[key].append(label[key])
        env_ids.append(np.full(len(label["depth"]), env_index, dtype=np.int32))
        np.save(environments / f"{env_index:03d}.npy", np.load(folder / "environment.npy"))
    stacked = {key: np.concatenate(value) for key, value in arrays.items()}
    stacked["env_index"] = np.concatenate(env_ids)
    count = len(stacked["depth"])
    if count < 2:
        raise ValueError("need at least two views to pack a train/val split")
    order = np.random.default_rng(seed).permutation(count)
    val_count = max(1, int(round(val_fraction * count)))
    if val_count >= count:
        val_count = 1
    packed = root / "packed"
    packed.mkdir(exist_ok=True)
    splits = {"train": order[val_count:], "val": order[:val_count]}
    for split, indices in splits.items():
        np.savez(packed / f"{split}.npz", **{key: value[indices] for key, value in stacked.items()})
    (packed / "split.json").write_text(
        json.dumps(
            {"train": int(len(splits["train"])), "val": int(len(splits["val"])), "seed": seed},
            indent=2,
        )
    )
    return packed


def augment_state(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random body velocity, acceleration, and forward goal, as in YOPO's loader."""
    forward = float(np.clip(rng.lognormal(mean=0.4, sigma=0.35), 0.3, 6.0))
    velocity = np.array([forward, float(rng.normal(0.0, 0.4)), float(rng.normal(0.0, 0.2))])
    acceleration = rng.normal(0.0, 0.3, size=3)
    goal = rng.normal(0.0, 0.3, size=3) + np.array([1.0, 0.0, 0.0])
    return velocity, acceleration, goal / np.linalg.norm(goal)


class RadarDepthDataset:
    """One packed split. Depth and CIRI planes are the stored values."""

    def __init__(self, directory: Path, split: str):
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        root = resolve_dataset(directory)
        path = root / "packed" / f"{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; run `ap-pnc learn pack` first")
        stored = np.load(path)
        self.depth = stored["depth"]
        self.valid = stored["valid"]
        self.positions = stored["positions"]
        self.rpy = stored["rpy"]
        self.planes = stored["planes"]
        self.plane_count = stored["plane_count"]
        self.ciri_success = stored["ciri_success"]
        self.cloud_count = stored["cloud_count"]
        self.accumulated = stored["accumulated"]
        self.env_index = stored["env_index"]
        self.split = split
        self.root = root
        manifest = json.loads((root / "manifest.json").read_text())
        depth = manifest["depth"]
        self.config = RadarDepthConfig(
            height=int(depth["height"]),
            width=int(depth["width"]),
            azimuth_fov_deg=float(depth["azimuth_fov_deg"]),
            elevation_fov_deg=float(depth["elevation_fov_deg"]),
            max_range_m=float(depth["max_range_m"]),
            erosion_voxel_m=float(depth["erosion_voxel_m"]),
            range_noise_m=0.0,
            keep_probability=1.0,
        )
        self._environments: dict[int, Occupancy] = {}

    def __len__(self) -> int:
        return len(self.depth)

    def environment(self, index: int) -> Occupancy:
        if index not in self._environments:
            points = np.load(self.root / "environments" / f"{index:03d}.npy")
            self._environments[index] = Occupancy(points, voxel_m=0.4, ground_z=0.0)
        return self._environments[index]

    def sample(self, index: int, augment_seed: int) -> RadarSample:
        count = int(self.cloud_count[index])
        velocity, acceleration, goal = augment_state(np.random.default_rng(augment_seed))
        return RadarSample(
            self.depth[index].astype(float),
            self.valid[index],
            self.accumulated[index, :count].astype(float),
            self.planes[index].astype(float),
            int(self.plane_count[index]),
            bool(self.ciri_success[index]),
            self.positions[index].astype(float),
            self.rpy[index].astype(float),
            int(self.env_index[index]),
            velocity,
            acceleration,
            goal,
        )
