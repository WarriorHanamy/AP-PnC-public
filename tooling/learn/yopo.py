"""Radar-depth YOPO: one forward pass over primitive anchors.

The published YOPO planner reads a depth image and predicts, for each motion
primitive, an endpoint offset and a score. Here the depth image is the radar
point-cloud raster from ``radar_depth``. Pretrained depth-camera weights are
not used and are not shipped.

Each primitive emits ten values: endpoint position, velocity and acceleration
offsets, then a score. The highest score is integrated as a quintic and flown
by the quadrotor. Training is guidance learning: the score head is pulled
toward the lowest-cost primitive, and the selected offsets receive the
numerical gradient of that cost.
"""

from __future__ import annotations

import numpy as np

from tooling.learn.net import ReluMlp
from tooling.learn.occupancy import Occupancy
from tooling.learn.quadrotor import QuadrotorState, rotation_world_from_body
from tooling.learn.radar_depth import RadarDepthConfig

PRIMITIVE_ELEVATION = 3
PRIMITIVE_AZIMUTH = 5
PRIMITIVE_COUNT = PRIMITIVE_ELEVATION * PRIMITIVE_AZIMUTH
OUTPUTS_PER_PRIMITIVE = 10
HORIZON_S = 1.0
HORIZON_M = 3.0
OBSERVATION_DIM = 54
YOPO_SIZES = (OBSERVATION_DIM, 64, PRIMITIVE_COUNT * OUTPUTS_PER_PRIMITIVE)


def observation(
    depth: np.ndarray, state: QuadrotorState, goal: np.ndarray, config: RadarDepthConfig
) -> np.ndarray:
    if depth.shape != (config.height, config.width):
        raise ValueError("depth image does not match the radar raster")
    pooled = depth.reshape(config.height // 4, 4, config.width // 4, 4).mean(axis=(1, 3))
    rotation = rotation_world_from_body(state.roll, state.pitch, state.yaw)
    velocity_body = rotation.T @ state.velocity
    delta = goal - state.position
    norm = float(np.linalg.norm(delta))
    direction = delta / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    return np.concatenate(
        [pooled.ravel() / config.max_range_m, velocity_body, rotation.T @ direction]
    )


def anchors(config: RadarDepthConfig) -> np.ndarray:
    elevation = np.linspace(
        -np.radians(config.elevation_fov_deg) / 2,
        np.radians(config.elevation_fov_deg) / 2,
        PRIMITIVE_ELEVATION,
    )
    azimuth = np.linspace(
        -np.radians(config.azimuth_fov_deg) / 2,
        np.radians(config.azimuth_fov_deg) / 2,
        PRIMITIVE_AZIMUTH,
    )
    elevation_grid, azimuth_grid = np.meshgrid(elevation, azimuth, indexing="ij")
    forward = np.cos(elevation_grid) * np.cos(azimuth_grid)
    left = np.cos(elevation_grid) * np.sin(azimuth_grid)
    up = np.sin(elevation_grid)
    return HORIZON_M * np.column_stack((forward.ravel(), left.ravel(), up.ravel()))


def quintic_coefficients(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    position_end: np.ndarray,
    velocity_end: np.ndarray,
    acceleration_end: np.ndarray,
    horizon_s: float = HORIZON_S,
) -> np.ndarray:
    """Return (6, 3) coefficients of a quintic, lowest degree first."""
    duration = horizon_s
    constant = position
    linear = velocity
    quadratic = acceleration / 2.0
    system = np.array(
        [
            [duration**3, duration**4, duration**5],
            [3 * duration**2, 4 * duration**3, 5 * duration**4],
            [6 * duration, 12 * duration**2, 20 * duration**3],
        ]
    )
    remainder = np.stack(
        [
            position_end - constant - linear * duration - quadratic * duration**2,
            velocity_end - linear - 2 * quadratic * duration,
            acceleration_end - 2 * quadratic,
        ],
        axis=0,
    )
    higher = np.linalg.solve(system, remainder)
    return np.stack([constant, linear, quadratic, higher[0], higher[1], higher[2]])


def quintic_samples(coefficients: np.ndarray, times: np.ndarray) -> np.ndarray:
    powers = np.stack([times**power for power in range(6)], axis=1)
    return powers @ coefficients


def _boundary(
    state: QuadrotorState, raw: np.ndarray, anchor: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = rotation_world_from_body(state.roll, state.pitch, state.yaw)
    position_end = state.position + rotation @ (anchor + np.tanh(raw[:3]))
    velocity_end = rotation @ (anchor / HORIZON_S + np.tanh(raw[3:6]))
    acceleration_end = rotation @ (2.0 * np.tanh(raw[6:9]))
    return position_end, velocity_end, acceleration_end


def trajectory_cost(
    samples: np.ndarray, goal: np.ndarray, occupancy: Occupancy, margin_m: float = 0.45
) -> float:
    clearance = occupancy.clearance(samples)
    collision = np.maximum(margin_m - clearance, 0.0).sum()
    goal_cost = float(np.linalg.norm(samples[-1] - goal))
    length = float(np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())
    return float(10.0 * collision + goal_cost + 0.05 * length)


def _primitive_cost(
    state: QuadrotorState,
    raw: np.ndarray,
    anchor: np.ndarray,
    goal: np.ndarray,
    occupancy: Occupancy,
) -> float:
    position_end, velocity_end, acceleration_end = _boundary(state, raw, anchor)
    coefficients = quintic_coefficients(
        state.position,
        state.velocity,
        np.zeros(3),
        position_end,
        velocity_end,
        acceleration_end,
    )
    samples = quintic_samples(coefficients, np.linspace(0.0, HORIZON_S, 8))
    return trajectory_cost(samples, goal, occupancy)


def guidance_step(
    net: ReluMlp,
    features: np.ndarray,
    state: QuadrotorState,
    goal: np.ndarray,
    occupancy: Occupancy,
    config: RadarDepthConfig,
    learning_rate: float = 0.05,
) -> float:
    """One guidance update. Returns the cost of the selected primitive."""
    prediction, cache = net.forward(features)
    blocks = prediction.reshape(PRIMITIVE_COUNT, OUTPUTS_PER_PRIMITIVE)
    primitive_anchors = anchors(config)
    costs = np.array(
        [
            _primitive_cost(state, blocks[index, :9], primitive_anchors[index], goal, occupancy)
            for index in range(PRIMITIVE_COUNT)
        ]
    )
    selected = int(np.argmin(costs))
    logits = blocks[:, 9]
    shifted = logits - logits.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    score_gradient = probabilities.copy()
    score_gradient[selected] -= 1.0
    base = blocks[selected, :9].copy()
    epsilon = 1e-2

    def cost_at(offset: np.ndarray) -> float:
        return _primitive_cost(state, base + offset, primitive_anchors[selected], goal, occupancy)

    parameter_gradient = np.zeros(9)
    for index in range(9):
        step = np.zeros(9)
        step[index] = epsilon
        parameter_gradient[index] = (cost_at(step) - cost_at(-step)) / (2 * epsilon)
    output_gradient = np.zeros_like(prediction)
    output_gradient.reshape(PRIMITIVE_COUNT, OUTPUTS_PER_PRIMITIVE)[selected, :9] = (
        parameter_gradient
    )
    output_gradient.reshape(PRIMITIVE_COUNT, OUTPUTS_PER_PRIMITIVE)[:, 9] = score_gradient
    net.sgd(net.backward(output_gradient, cache), learning_rate)
    return float(costs[selected])


def plan_acceleration(
    net: ReluMlp,
    features: np.ndarray,
    state: QuadrotorState,
    config: RadarDepthConfig,
    dt: float,
) -> np.ndarray:
    """World acceleration command from the highest-scoring primitive."""
    prediction, _ = net.forward(features)
    blocks = prediction.reshape(PRIMITIVE_COUNT, OUTPUTS_PER_PRIMITIVE)
    selected = int(np.argmax(blocks[:, 9]))
    position_end, velocity_end, acceleration_end = _boundary(
        state, blocks[selected, :9], anchors(config)[selected]
    )
    coefficients = quintic_coefficients(
        state.position,
        state.velocity,
        np.zeros(3),
        position_end,
        velocity_end,
        acceleration_end,
    )
    powers = np.array([0.0, 1.0, 2 * dt, 3 * dt**2, 4 * dt**3, 5 * dt**4])
    planned_velocity = powers @ coefficients
    return (planned_velocity - state.velocity) / dt
