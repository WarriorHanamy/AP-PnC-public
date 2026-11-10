"""Radar sweeps, then depth by accumulation, erosion, and projection.

A single sparse scan is not the depth image. Sweeps are accumulated in one
frame, isolated occupied voxels are eroded away, and only the surviving points
are projected into the forward depth raster YOPO reads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tooling.learn.occupancy import Occupancy


@dataclass(frozen=True)
class RadarDepthConfig:
    width: int = 32
    height: int = 24
    azimuth_fov_deg: float = 90.0
    elevation_fov_deg: float = 40.0
    max_range_m: float = 12.0
    range_noise_m: float = 0.05
    keep_probability: float = 0.7
    azimuth_beams: int = 16
    elevation_beams: int = 4
    min_range_m: float = 0.3
    erosion_voxel_m: float = 0.35

    def __post_init__(self) -> None:
        if self.width % 4 or self.height % 4:
            raise ValueError("depth size must be divisible by 4")
        if not 0.0 < self.keep_probability <= 1.0:
            raise ValueError("keep_probability must be in (0, 1]")
        if not 0.0 <= self.range_noise_m:
            raise ValueError("range noise must be non-negative")
        if not 0.0 < self.min_range_m < self.max_range_m:
            raise ValueError("range bounds are inconsistent")
        if self.erosion_voxel_m <= 0.0:
            raise ValueError("erosion voxel must be positive")


def _beam_directions(config: RadarDepthConfig) -> np.ndarray:
    azimuth = np.linspace(
        -np.radians(config.azimuth_fov_deg) / 2,
        np.radians(config.azimuth_fov_deg) / 2,
        config.azimuth_beams,
    )
    elevation = np.linspace(
        -np.radians(config.elevation_fov_deg) / 2,
        np.radians(config.elevation_fov_deg) / 2,
        config.elevation_beams,
    )
    elevation_grid, azimuth_grid = np.meshgrid(elevation, azimuth, indexing="ij")
    forward = np.cos(elevation_grid) * np.cos(azimuth_grid)
    left = np.cos(elevation_grid) * np.sin(azimuth_grid)
    up = np.sin(elevation_grid)
    return np.column_stack((forward.ravel(), left.ravel(), up.ravel()))


def radar_points(
    occupancy: Occupancy,
    position: np.ndarray,
    rotation_world_from_body: np.ndarray,
    config: RadarDepthConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """One radar sweep in the body frame. This cloud is not yet a depth image."""
    directions_body = _beam_directions(config)
    directions_world = directions_body @ rotation_world_from_body.T
    ranges = occupancy.raycast(position, directions_world, config.max_range_m)
    hit = np.isfinite(ranges) & (ranges >= config.min_range_m)
    if not hit.any():
        return np.zeros((0, 3))
    measured = ranges[hit] + rng.normal(0.0, config.range_noise_m, int(hit.sum()))
    keep = (measured >= config.min_range_m) & (measured <= config.max_range_m)
    keep &= rng.random(len(measured)) < config.keep_probability
    return directions_body[hit][keep] * measured[keep, None]


def accumulate(clouds: list[np.ndarray]) -> np.ndarray:
    """Stack sweeps that already share one coordinate frame."""
    usable = [cloud.reshape(-1, 3) for cloud in clouds if len(cloud)]
    if not usable:
        return np.zeros((0, 3))
    return np.vstack(usable)


def erode_occupied(points: np.ndarray, voxel_m: float) -> np.ndarray:
    """Remove occupied voxels that have no face-adjacent occupied neighbor.

    Accumulation fills a surface. A lone radar return does not, and erosion
    drops it before projection.
    """
    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(cloud) == 0:
        return np.zeros((0, 3))
    keys = np.floor(cloud / voxel_m).astype(np.int64)
    occupied = {tuple(int(value) for value in key) for key in keys}
    faces = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    kept = [
        key
        for key in occupied
        if any((key[0] + dx, key[1] + dy, key[2] + dz) in occupied for dx, dy, dz in faces)
    ]
    if not kept:
        return np.zeros((0, 3))
    return (np.asarray(kept, dtype=float) + 0.5) * voxel_m


def project_depth(points: np.ndarray, config: RadarDepthConfig) -> tuple[np.ndarray, np.ndarray]:
    """Project points to a forward-range image. Nearest forward range wins.

    Image right is body -Y and image down is body -Z. Empty pixels stay at the
    maximum range and are marked invalid.
    """
    depth = np.full((config.height, config.width), config.max_range_m)
    valid = np.zeros(depth.shape, dtype=bool)
    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(cloud) == 0:
        return depth, valid
    forward = cloud[:, 0]
    visible = forward > config.min_range_m
    if not visible.any():
        return depth, valid
    half_azimuth = np.tan(np.radians(config.azimuth_fov_deg) / 2)
    half_elevation = np.tan(np.radians(config.elevation_fov_deg) / 2)
    focal_x = (config.width / 2) / half_azimuth
    focal_y = (config.height / 2) / half_elevation
    center_x = (config.width - 1) / 2
    center_y = (config.height - 1) / 2
    columns = np.rint(center_x - focal_x * cloud[visible, 1] / forward[visible]).astype(int)
    rows = np.rint(center_y - focal_y * cloud[visible, 2] / forward[visible]).astype(int)
    inside = (columns >= 0) & (columns < config.width) & (rows >= 0) & (rows < config.height)
    if not inside.any():
        return depth, valid
    selected_rows = rows[inside]
    selected_columns = columns[inside]
    selected_depth = forward[visible][inside]
    order = np.argsort(selected_depth)
    for row, column, value in zip(
        selected_rows[order], selected_columns[order], selected_depth[order], strict=True
    ):
        if not valid[row, column]:
            depth[row, column] = value
            valid[row, column] = True
    return depth, valid


def depth_from_accumulated(
    points: np.ndarray, config: RadarDepthConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Erode an accumulated cloud, then project it. Do not skip the erosion."""
    return project_depth(erode_occupied(points, config.erosion_voxel_m), config)
