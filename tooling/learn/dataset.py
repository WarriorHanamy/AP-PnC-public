"""Recorded radar-to-depth sets, packed like a YOPO training corpus.

This is not a CIRI dataset. CIRI's representation is unspecified here, so these
files store radar returns and the depth image rasterized from those returns.
A synthetic obstacle field is only the simulator that casts the radar rays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tooling.env import artifact_path
from tooling.learn.occupancy import Occupancy
from tooling.learn.quadrotor import rotation_world_from_body
from tooling.learn.radar_depth import RadarDepthConfig, depth_from_radar, radar_points

SCHEMA = "ap-pnc/radar-depth/v1"
MAX_RADAR_POINTS = 64


@dataclass(frozen=True)
class DepthSample:
    depth: np.ndarray
    valid: np.ndarray
    radar_points: np.ndarray
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
            f"No radar-depth dataset at {resolved}. Run `ap-pnc learn collect` first."
        )
    document = json.loads(manifest.read_text())
    if document.get("schema") != SCHEMA:
        raise ValueError(f"{manifest} is not a {SCHEMA} dataset")
    return resolved


def _environment(env_index: int) -> Occupancy:
    """Obstacle field used only to simulate radar. It is not CIRI."""
    shift = 0.35 * env_index
    xs = np.linspace(0.0, 6.0, 13)
    left = np.column_stack((xs, np.full_like(xs, 1.2), np.ones_like(xs)))
    right = np.column_stack((xs, np.full_like(xs, -1.2), np.ones_like(xs)))
    gate = np.column_stack((np.full(5, 3.0 + shift), np.linspace(-0.8, 0.8, 5), np.ones(5)))
    return Occupancy(np.vstack((left, right, gate)), voxel_m=0.4, ground_z=0.0)


def _sample_pose(occupancy: Occupancy, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    for _ in range(40):
        position = np.array([rng.uniform(0.4, 4.8), rng.uniform(-0.5, 0.5), 1.0], dtype=float)
        if occupancy.clearance(position[None, :])[0] > 0.45:
            return position, float(rng.uniform(-np.pi, np.pi))
    return np.array([0.4, 0.0, 1.0]), 0.0


def _pad_radar(clouds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array([len(cloud) for cloud in clouds], dtype=np.int32)
    padded = np.zeros((len(clouds), MAX_RADAR_POINTS, 3), dtype=np.float32)
    for index, cloud in enumerate(clouds):
        kept = cloud[:MAX_RADAR_POINTS]
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
    """Save radar clouds and their depth rasters, one folder per environment."""
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
        depths, valids, positions, rpys, clouds = [], [], [], [], []
        for _ in range(samples_per_env):
            position, yaw = _sample_pose(occupancy, rng)
            rotation = rotation_world_from_body(0.0, 0.0, yaw)
            points = radar_points(occupancy, position, rotation, sensor, rng)
            depth, valid = depth_from_radar(points, sensor)
            depths.append(depth)
            valids.append(valid)
            positions.append(position)
            rpys.append(np.array([0.0, 0.0, yaw]))
            clouds.append(points.astype(np.float32))
        counts, padded = _pad_radar(clouds)
        np.savez(
            folder / "label.npz",
            depth=np.stack(depths).astype(np.float32),
            valid=np.stack(valids),
            positions=np.stack(positions).astype(np.float32),
            rpy=np.stack(rpys).astype(np.float32),
            radar_counts=counts,
            radar_points=padded,
        )
    manifest = {
        "schema": SCHEMA,
        "generated": True,
        "label": "synthetic radar point clouds rasterized to depth; not CIRI",
        "ciri": "unspecified",
        "pipeline": ["synthetic-environment", "radar-point-cloud", "depth"],
        "envs": envs,
        "samples_per_env": samples_per_env,
        "seed": seed,
        "depth": {
            "height": sensor.height,
            "width": sensor.width,
            "azimuth_fov_deg": sensor.azimuth_fov_deg,
            "elevation_fov_deg": sensor.elevation_fov_deg,
            "max_range_m": sensor.max_range_m,
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
    depths, valids, positions, rpys = [], [], [], []
    counts, clouds, env_ids = [], [], []
    environments = root / "environments"
    environments.mkdir(exist_ok=True)
    for env_index, folder in enumerate(folders):
        label = np.load(folder / "label.npz")
        count = len(label["depth"])
        depths.append(label["depth"])
        valids.append(label["valid"])
        positions.append(label["positions"])
        rpys.append(label["rpy"])
        counts.append(label["radar_counts"])
        clouds.append(label["radar_points"])
        env_ids.append(np.full(count, env_index, dtype=np.int32))
        np.save(environments / f"{env_index:03d}.npy", np.load(folder / "environment.npy"))
    depth = np.concatenate(depths)
    order = np.random.default_rng(seed).permutation(len(depth))
    if len(depth) < 2:
        raise ValueError("need at least two views to pack a train/val split")
    val_count = max(1, int(round(val_fraction * len(depth))))
    if val_count >= len(depth):
        val_count = 1
    packed = root / "packed"
    packed.mkdir(exist_ok=True)
    arrays = {
        "depth": depth,
        "valid": np.concatenate(valids),
        "positions": np.concatenate(positions),
        "rpy": np.concatenate(rpys),
        "radar_counts": np.concatenate(counts),
        "radar_points": np.concatenate(clouds),
        "env_index": np.concatenate(env_ids),
    }
    splits = {"train": order[val_count:], "val": order[:val_count]}
    for split, indices in splits.items():
        np.savez(packed / f"{split}.npz", **{key: value[indices] for key, value in arrays.items()})
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
    """One packed split. Depth is the stored radar raster, not a live render."""

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
        self.radar_counts = stored["radar_counts"]
        self.radar_points = stored["radar_points"]
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

    def sample(self, index: int, augment_seed: int) -> DepthSample:
        count = int(self.radar_counts[index])
        velocity, acceleration, goal = augment_state(np.random.default_rng(augment_seed))
        return DepthSample(
            self.depth[index].astype(float),
            self.valid[index],
            self.radar_points[index, :count].astype(float),
            self.positions[index].astype(float),
            self.rpy[index].astype(float),
            int(self.env_index[index]),
            velocity,
            acceleration,
            goal,
        )
