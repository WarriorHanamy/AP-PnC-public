"""Reduced quadrotor used by learning rollouts.

Specific thrust is along body +Z. This is not the acados NMPC model; that
bundle is regenerated separately with ``ap-pnc gen-nmpc-lib``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81


@dataclass
class QuadrotorState:
    position: np.ndarray
    velocity: np.ndarray
    roll: float
    pitch: float
    yaw: float

    def copy(self) -> QuadrotorState:
        return QuadrotorState(
            self.position.copy(),
            self.velocity.copy(),
            self.roll,
            self.pitch,
            self.yaw,
        )


def rotation_world_from_body(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX rotation. Columns are the body axes expressed in the world."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def step(state: QuadrotorState, thrust: float, body_rates: np.ndarray, dt: float) -> QuadrotorState:
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")
    rates = np.asarray(body_rates, dtype=float)
    if rates.shape != (3,) or not np.isfinite(rates).all() or not np.isfinite(thrust):
        raise ValueError("thrust and body rates must be finite")
    pitch = float(np.clip(state.pitch, -1.2, 1.2))
    roll = float(state.roll)
    yaw = float(state.yaw)
    wx, wy, wz = rates
    roll_dot = wx + np.sin(roll) * np.tan(pitch) * wy + np.cos(roll) * np.tan(pitch) * wz
    pitch_dot = np.cos(roll) * wy - np.sin(roll) * wz
    yaw_dot = (np.sin(roll) * wy + np.cos(roll) * wz) / max(np.cos(pitch), 0.2)
    rotation = rotation_world_from_body(roll, pitch, yaw)
    acceleration = rotation @ np.array([0.0, 0.0, thrust]) + np.array([0.0, 0.0, -GRAVITY])
    return QuadrotorState(
        position=state.position + state.velocity * dt,
        velocity=state.velocity + acceleration * dt,
        roll=roll + float(roll_dot) * dt,
        pitch=float(np.clip(pitch + float(pitch_dot) * dt, -1.2, 1.2)),
        yaw=yaw + float(yaw_dot) * dt,
    )


def command_from_acceleration(
    state: QuadrotorState, acceleration: np.ndarray, attitude_gain: float = 4.0
) -> np.ndarray:
    """Map a world acceleration into collective thrust and body-rate commands."""
    specific = np.asarray(acceleration, dtype=float) + np.array([0.0, 0.0, GRAVITY])
    norm = float(np.linalg.norm(specific))
    thrust = float(np.clip(norm, 0.0, 20.0))
    if norm < 1e-6:
        return np.array([GRAVITY, 0.0, 0.0, 0.0])
    desired_axis = specific / norm
    rotation = rotation_world_from_body(state.roll, state.pitch, state.yaw)
    body_axis = rotation[:, 2]
    tilt = rotation.T @ np.cross(body_axis, desired_axis)
    rates = attitude_gain * tilt
    return np.array([thrust, rates[0], rates[1], rates[2]])
