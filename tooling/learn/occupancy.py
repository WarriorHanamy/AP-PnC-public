"""Voxel occupancy built from a metric point cloud."""

from __future__ import annotations

import numpy as np


class Occupancy:
    """Occupied voxels plus brute-force clearance against the stored points."""

    def __init__(self, points: np.ndarray, voxel_m: float, ground_z: float):
        cloud = np.asarray(points, dtype=float)
        if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) == 0:
            raise ValueError("occupancy requires a nonempty (N, 3) point cloud")
        if not np.isfinite(cloud).all():
            raise ValueError("occupancy points must be finite")
        if not np.isfinite(voxel_m) or voxel_m <= 0.0 or not np.isfinite(ground_z):
            raise ValueError("voxel size and ground height must be finite")
        self.voxel_m = float(voxel_m)
        self.ground_z = float(ground_z)
        self.points = cloud
        self.origin = cloud.min(axis=0) - self.voxel_m
        indices = np.floor((cloud - self.origin) / self.voxel_m).astype(np.int64)
        self._occupied = {tuple(row) for row in indices}

    def clearance(self, samples: np.ndarray) -> np.ndarray:
        query = np.asarray(samples, dtype=float)
        if query.ndim != 2 or query.shape[1] != 3:
            raise ValueError("clearance samples must have shape (N, 3)")
        distances = np.linalg.norm(query[:, None, :] - self.points[None, :, :], axis=2).min(axis=1)
        above_ground = query[:, 2] - self.ground_z
        return np.minimum(distances, above_ground)

    def raycast(self, origin: np.ndarray, directions: np.ndarray, max_range_m: float) -> np.ndarray:
        start = np.asarray(origin, dtype=float)
        rays = np.asarray(directions, dtype=float)
        if start.shape != (3,) or rays.ndim != 2 or rays.shape[1] != 3:
            raise ValueError("raycast expects an origin (3,) and directions (N, 3)")
        if not np.isfinite(max_range_m) or max_range_m <= 0.0:
            raise ValueError("max range must be positive")
        step = self.voxel_m
        count = int(np.ceil(max_range_m / step))
        ranges = np.full(len(rays), np.inf)
        pending = np.ones(len(rays), dtype=bool)
        for index in range(1, count + 1):
            if not pending.any():
                break
            distance = min(index * step, max_range_m)
            selected = np.flatnonzero(pending)
            samples = start + rays[selected] * distance
            voxels = np.floor((samples - self.origin) / self.voxel_m).astype(np.int64)
            hit = np.fromiter(
                (tuple(row) in self._occupied for row in voxels), dtype=bool, count=len(voxels)
            )
            ranges[selected[hit]] = distance
            pending[selected[hit]] = False
        return ranges
