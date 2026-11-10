"""Sparse forward radar returns, rasterized into the depth image YOPO reads.

The depth image is not a depth-camera measurement. Each pixel is the nearest
forward range of a radar return that projects into it. Pixels with no return
stay at the maximum range and are marked invalid.
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

    def __post_init__(self) -> None:
        if self.width % 4 or self.height % 4:
            raise ValueError("depth size must be divisible by 4")
        if not 0.0 < self.keep_probability <= 1.0:
            raise ValueError("keep_probability must be in (0, 1]")
        if not 0.0 <= self.range_noise_m:
            raise ValueError("range noise must be non-negative")
        if not 0.0 < self.min_range_m < self.max_range_m:
            raise ValueError("range bounds are inconsistent")


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
    """Radar returns in the body frame. Body +X is the optical axis."""
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


def depth_from_radar(points: np.ndarray, config: RadarDepthConfig) -> tuple[np.ndarray, np.ndarray]:
    """Project body-frame radar points to a forward-range image.

    Image right is body -Y and image down is body -Z. Depth is the forward
    coordinate x, in metres.
    """
    depth = np.full((config.height, config.width), config.max_range_m)
    valid = np.zeros(depth.shape, dtype=bool)
    if len(points) == 0:
        return depth, valid
    forward = points[:, 0]
    visible = forward > config.min_range_m
    if not visible.any():
        return depth, valid
    half_azimuth = np.tan(np.radians(config.azimuth_fov_deg) / 2)
    half_elevation = np.tan(np.radians(config.elevation_fov_deg) / 2)
    focal_x = (config.width / 2) / half_azimuth
    focal_y = (config.height / 2) / half_elevation
    center_x = (config.width - 1) / 2
    center_y = (config.height - 1) / 2
    columns = np.rint(center_x - focal_x * points[visible, 1] / forward[visible]).astype(int)
    rows = np.rint(center_y - focal_y * points[visible, 2] / forward[visible]).astype(int)
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
