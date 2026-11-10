"""SUPER CIRI: configuration-space iterative regional inflation.

This follows the corridor interface in hku-mars/SUPER (`CIRI::comvexDecomposition`
and `CorridorGenerator::GeneratePolytopeFromLine`). Obstacle points and a seed
segment go in. The output is an h-polytope, one row `[nx, ny, nz, d]` per
plane, with the free set `n·x + d <= 0`. A positive `robot_r` inflates each
obstacle to a sphere, so the polytope is a safe flight corridor in
configuration space.

The halfspaces are the same representation SUPER stores on `Polytope::planes`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_PLANES = 16
EPSILON = 1e-6


class CiriFailed(RuntimeError):
    """The seed segment is not a feasible corridor under the robot radius."""


@dataclass
class Ellipsoid:
    rotation: np.ndarray
    radii: np.ndarray
    center: np.ndarray

    @property
    def shape(self) -> np.ndarray:
        return self.rotation @ np.diag(self.radii) @ self.rotation.T

    @property
    def inverse_shape(self) -> np.ndarray:
        inverse_radii = 1.0 / np.maximum(self.radii, 1e-6)
        return self.rotation @ np.diag(inverse_radii) @ self.rotation.T

    def points_to_frame(self, points: np.ndarray) -> np.ndarray:
        return (points - self.center) @ self.inverse_shape.T

    def plane_to_frame(self, planes: np.ndarray) -> np.ndarray:
        framed = np.empty_like(planes)
        framed[:, :3] = planes[:, :3] @ self.shape
        framed[:, 3] = planes[:, 3] + planes[:, :3] @ self.center
        return framed

    def plane_to_world(self, plane: np.ndarray) -> np.ndarray:
        normal = self.inverse_shape.T @ plane[:3]
        return np.array([normal[0], normal[1], normal[2], plane[3] - np.dot(normal, self.center)])

    def contained(self, points: np.ndarray) -> tuple[np.ndarray, int] | None:
        if len(points) == 0:
            return None
        norms = np.linalg.norm(self.points_to_frame(points), axis=1)
        inside = norms <= 1.0
        if not inside.any():
            return None
        subset = points[inside]
        return subset, int(np.argmin(norms[inside]))


@dataclass
class CiriPolytope:
    """One SUPER safe-flight polytope."""

    planes: np.ndarray
    seed_a: np.ndarray
    seed_b: np.ndarray
    robot_r: float
    success: bool

    def contains(self, point: np.ndarray, margin: float = 0.01) -> bool:
        if len(self.planes) == 0:
            return False
        values = self.planes[:, :3] @ np.asarray(point, dtype=float) + self.planes[:, 3]
        return bool(np.max(values) <= margin)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_from_two_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation taking `source` onto `target`."""
    a = source / np.linalg.norm(source)
    b = target / np.linalg.norm(target)
    cross = np.cross(a, b)
    cosine = float(np.dot(a, b))
    if cosine < -0.999999:
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, axis)
        axis = axis / np.linalg.norm(axis)
        skew = _skew(axis)
        return np.eye(3) + 2.0 * skew @ skew
    skew = _skew(cross)
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (np.dot(cross, cross) + 1e-16))


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def boundary_planes(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """Axis-aligned box as six halfspaces, the `bd` argument of SUPER CIRI."""
    low = np.asarray(box_min, dtype=float)
    high = np.asarray(box_max, dtype=float)
    planes = np.zeros((6, 4))
    planes[0] = [1.0, 0.0, 0.0, -high[0]]
    planes[1] = [-1.0, 0.0, 0.0, low[0]]
    planes[2] = [0.0, 1.0, 0.0, -high[1]]
    planes[3] = [0.0, -1.0, 0.0, low[1]]
    planes[4] = [0.0, 0.0, 1.0, -high[2]]
    planes[5] = [0.0, 0.0, -1.0, low[2]]
    return planes


def _point_inside_planes(planes: np.ndarray, point: np.ndarray) -> bool:
    return bool(np.max(planes[:, :3] @ point + planes[:, 3]) <= EPSILON)


def distance_point_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    span = end - start
    length2 = float(np.dot(span, span))
    if length2 < 1e-12:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, span) / length2, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * span)))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise CiriFailed("CIRI produced a zero plane normal")
    return vector / norm


def tangent_plane_of_sphere(
    center: np.ndarray,
    radius: float,
    pass_point: np.ndarray,
    seed: np.ndarray,
) -> np.ndarray:
    """Plane through `pass_point` tangent to the sphere, keeping `seed` inside.

    This is the correction SUPER applies when a separating plane would cut the
    seed segment.
    """
    nudged = seed.astype(float).copy()
    radial = pass_point - center
    if np.linalg.norm(radial[:2]) > 1e-3:
        horizontal = radial / np.linalg.norm(radial)
        horizontal[2] = 0.0
        horizontal = _unit(horizontal)
        nudged = seed + 0.01 * np.cross(horizontal, np.array([0.0, 0.0, 1.0]))
    else:
        axis = np.cross(radial, np.array([1.0, 0.0, 0.0]))
        nudged = seed + 0.01 * _unit(axis)
    normal_axis = _unit(np.cross(radial, nudged - center))
    rotation = rotation_from_two_vectors(normal_axis, np.array([0.0, 0.0, 1.0]))
    planar = rotation @ radial
    seed_planar = rotation @ (nudged - center)
    planar_norm = float(planar[0] ** 2 + planar[1] ** 2)
    if planar_norm <= radius * radius + 1e-8:
        direction = _unit(seed - center) if np.linalg.norm(seed - center) > 1e-6 else _unit(radial)
        contact = center + radius * direction
        normal = _unit(center - contact)
        return np.array([*normal, -float(np.dot(normal, contact))])
    gap = np.sqrt(planar_norm - radius * radius)
    scale = radius / planar_norm
    first = np.array(
        [
            scale * (planar[0] * radius - planar[1] * gap),
            scale * (planar[1] * radius + planar[0] * gap),
            0.0,
        ]
    )
    second = np.array(
        [
            scale * (planar[0] * radius + planar[1] * gap),
            scale * (planar[1] * radius - planar[0] * gap),
            0.0,
        ]
    )
    local = second if first[0] * seed_planar[0] + first[1] * seed_planar[1] < 0.0 else first
    normal = rotation.T @ local
    contact = normal + center
    plane = np.array([normal[0], normal[1], normal[2], -float(np.dot(contact, normal))])
    if float(np.dot(plane[:3], nudged) + plane[3]) > EPSILON:
        plane = -plane
    plane[:3] = _unit(plane[:3])
    plane[3] = -float(np.dot(plane[:3], contact))
    if float(np.dot(plane[:3], seed) + plane[3]) > EPSILON:
        plane = -plane
    return plane


def _sphere_plane(
    obstacle: np.ndarray,
    robot_r: float,
    ellipsoid: Ellipsoid,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    toward = ellipsoid.center - obstacle
    if np.linalg.norm(toward) < 1e-8:
        toward = start - obstacle
    direction = _unit(toward)
    contact = obstacle + robot_r * direction
    normal = _unit(obstacle - contact)
    plane = np.array([*normal, -float(np.dot(normal, contact))])
    if float(np.dot(plane[:3], start) + plane[3]) > -EPSILON:
        plane = tangent_plane_of_sphere(obstacle, robot_r, start, ellipsoid.center)
    elif float(np.dot(plane[:3], end) + plane[3]) > -EPSILON:
        plane = tangent_plane_of_sphere(obstacle, robot_r, end, ellipsoid.center)
    return plane


def find_ellipsoid(
    points: np.ndarray, start: np.ndarray, end: np.ndarray, robot_r: float
) -> Ellipsoid:
    """Line-seed ellipsoid used before the first inflation pass."""
    length = float(np.linalg.norm(end - start))
    focal = length / 2.0
    radii = np.array([focal + robot_r, focal, focal])
    if radii[0] > 0.0:
        radii *= radii[1] / radii[0]
    center = (start + end) / 2.0
    rotation = rotation_from_two_vectors(np.array([1.0, 0.0, 0.0]), (end - start) / length)
    ellipsoid = Ellipsoid(rotation, radii.copy(), center)
    contained = ellipsoid.contained(points)
    if contained is None:
        return ellipsoid
    subset, index = contained
    point = subset[index]
    for _ in range(100):
        local = rotation.T @ (point - center)
        rotation = rotation @ _rotation_x(float(np.arctan2(local[2], local[1])))
        local = rotation.T @ (point - center)
        if abs(local[0]) < radii[0]:
            radii[1] = abs(local[1]) / np.sqrt(max(1.0 - (local[0] / radii[0]) ** 2, 1e-12))
        if not np.isfinite(radii[1]) or radii[1] < 1e-3:
            break
        ellipsoid = Ellipsoid(rotation, radii.copy(), center)
        contained = ellipsoid.contained(subset)
        if contained is None:
            break
        subset, index = contained
        point = subset[index]
    contained = ellipsoid.contained(points)
    if contained is None:
        return ellipsoid
    subset, index = contained
    point = subset[index]
    for _ in range(100):
        local = rotation.T @ (point - center)
        residual = 1.0 - (local[0] / radii[0]) ** 2 - (local[1] / max(radii[1], 1e-6)) ** 2
        if residual > EPSILON:
            radii[2] = abs(local[2]) / np.sqrt(residual)
        if not np.isfinite(radii[2]) or radii[2] < 1e-3:
            break
        ellipsoid = Ellipsoid(rotation, radii.copy(), center)
        contained = ellipsoid.contained(subset)
        if contained is None:
            return ellipsoid
        subset, index = contained
        point = subset[index]
    return ellipsoid


def _inscribed_sphere(planes: np.ndarray, hint: np.ndarray) -> Ellipsoid:
    """Largest sphere at a feasible center. Used between CIRI iterations."""
    center = hint.astype(float).copy()
    for _ in range(40):
        signed = planes[:, :3] @ center + planes[:, 3]
        worst = int(np.argmax(signed))
        normal = planes[worst, :3]
        norm = float(np.linalg.norm(normal)) + 1e-12
        center = center - signed[worst] * normal / norm**2
    distances = -(planes[:, :3] @ center + planes[:, 3]) / (
        np.linalg.norm(planes[:, :3], axis=1) + 1e-12
    )
    radius = max(float(np.min(distances)), 0.05)
    return Ellipsoid(np.eye(3), np.full(3, radius), center)


def convex_decomposition(
    boundary: np.ndarray,
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    robot_r: float,
    iterations: int,
) -> CiriPolytope:
    """Inflate one seed segment. Matches SUPER's `comvexDecomposition` arguments."""
    if iterations < 1:
        raise ValueError("CIRI iterations must be positive")
    seed_a = np.asarray(start, dtype=float)
    seed_b = np.asarray(end, dtype=float)
    obstacles = np.asarray(points, dtype=float).reshape(-1, 3)
    if not _point_inside_planes(boundary, seed_a) or not _point_inside_planes(boundary, seed_b):
        raise CiriFailed("seed segment is outside the CIRI boundary")
    if len(obstacles) == 0:
        return CiriPolytope(boundary.copy(), seed_a, seed_b, robot_r, True)
    if float(np.linalg.norm(seed_a - seed_b)) > 0.1:
        ellipsoid = find_ellipsoid(obstacles, seed_a, seed_b, robot_r)
    else:
        ellipsoid = Ellipsoid(np.eye(3), np.ones(3), (seed_a + seed_b) / 2.0)
    count = len(obstacles)
    planes = boundary
    for loop in range(iterations):
        framed_boundary = ellipsoid.plane_to_frame(boundary)
        boundary_distance = np.abs(framed_boundary[:, 3]) / (
            np.linalg.norm(framed_boundary[:, :3], axis=1) + 1e-12
        )
        obstacle_distance = np.linalg.norm(ellipsoid.points_to_frame(obstacles), axis=1)
        boundary_open = np.ones(len(boundary), dtype=bool)
        obstacle_open = np.ones(count, dtype=bool)
        chosen: list[np.ndarray] = []
        for _ in range(len(boundary) + count):
            open_boundary = np.where(boundary_open)[0]
            open_obstacle = np.where(obstacle_open)[0]
            min_boundary = (
                float(boundary_distance[open_boundary].min()) if len(open_boundary) else np.inf
            )
            min_obstacle = (
                float(obstacle_distance[open_obstacle].min()) if len(open_obstacle) else np.inf
            )
            if not np.isfinite(min_boundary) and not np.isfinite(min_obstacle):
                break
            if min_boundary < min_obstacle:
                index = int(open_boundary[np.argmin(boundary_distance[open_boundary])])
                plane = ellipsoid.plane_to_world(framed_boundary[index])
                boundary_open[index] = False
            else:
                index = int(open_obstacle[np.argmin(obstacle_distance[open_obstacle])])
                obstacle = obstacles[index]
                if distance_point_to_segment(obstacle, seed_a, seed_b) < robot_r - 1e-2:
                    raise CiriFailed("obstacle intersects the robot along the seed")
                plane = _sphere_plane(obstacle, robot_r, ellipsoid, seed_a, seed_b)
                obstacle_open[index] = False
                signed = obstacles @ plane[:3] + plane[3]
                obstacle_open &= signed <= robot_r - EPSILON
                obstacle_open[index] = False
            chosen.append(plane)
        if not chosen:
            break
        planes = np.vstack(chosen)
        if loop == iterations - 1:
            break
        if not np.isfinite(planes).all():
            raise CiriFailed("CIRI planes are not finite")
        ellipsoid = _inscribed_sphere(planes, (seed_a + seed_b) / 2.0)
    if not _point_inside_planes(planes, seed_a) or not _point_inside_planes(planes, seed_b):
        raise CiriFailed("CIRI polytope excluded its seed")
    return CiriPolytope(planes, seed_a, seed_b, robot_r, True)


def corridor_from_points(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    robot_r: float = 0.2,
    bound_dis: float = 2.5,
    iterations: int = 2,
    ground_z: float = 0.0,
    ceil_z: float = 3.0,
) -> CiriPolytope:
    """One corridor, the `GeneratePolytopeFromLine` step around a seed segment."""
    seed_a = np.asarray(start, dtype=float)
    seed_b = np.asarray(end, dtype=float)
    box_min = np.minimum(seed_a, seed_b) - bound_dis
    box_max = np.maximum(seed_a, seed_b) + bound_dis
    box_min[2] = max(box_min[2], ground_z + robot_r)
    box_max[2] = min(box_max[2], ceil_z - robot_r)
    box_min = np.minimum(box_min, np.minimum(seed_a, seed_b) - 0.05)
    box_max = np.maximum(box_max, np.maximum(seed_a, seed_b) + 0.05)
    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(cloud):
        inside = np.all((cloud >= box_min) & (cloud <= box_max), axis=1)
        cloud = cloud[inside]
    try:
        return convex_decomposition(
            boundary_planes(box_min, box_max), cloud, seed_a, seed_b, robot_r, iterations
        )
    except CiriFailed:
        return CiriPolytope(np.zeros((0, 4)), seed_a, seed_b, robot_r, False)


def pad_planes(polytope: CiriPolytope, limit: int = MAX_PLANES) -> tuple[np.ndarray, int]:
    """Fixed-width plane block for the RL observation and the packed dataset."""
    block = np.zeros((limit, 4), dtype=np.float32)
    count = min(len(polytope.planes), limit)
    if count:
        block[:count] = polytope.planes[:count]
    return block, count


def ciri_features(
    planes: np.ndarray,
    count: int,
    position: np.ndarray,
    rotation_world_from_body: np.ndarray,
    goal_direction_body: np.ndarray,
) -> np.ndarray:
    """Body-frame corridor planes plus the goal direction. This is the RL observation."""
    body = np.zeros((MAX_PLANES, 4))
    used = min(int(count), MAX_PLANES)
    if used:
        normals = planes[:used, :3]
        body_normals = normals @ rotation_world_from_body
        offsets = planes[:used, 3] + normals @ position
        scale = np.maximum(np.linalg.norm(body_normals, axis=1, keepdims=True), 1e-9)
        body[:used, :3] = body_normals / scale
        body[:used, 3:] = offsets[:, None] / scale
    return np.concatenate([body.ravel(), goal_direction_body])
