# BSD 3-Clause License
# Copyright (c) 2025 Sun Yat-sen University. All rights reserved.
# Authors: Erchao Rong: rongerch@outlook.com
# Authors: Zihao Liu
# Authors: Junning Liang

"""ENU/FLU quadrotor NMPC with ideal inner-loop tracking and command integrators.

x = [position_W, velocity_W, body_rate_command, quaternion_WB_wxyz,
     specific_thrust_command] (14).

u = d[specific_thrust_command, body_rate_command]/dt (4).
Collective thrust is along body +Z. There is no wing force, angle-of-attack
state, or aerodynamic parameter. Motor dynamics, torque and rate-loop lag
belong to the plant, not this model. Ground speed only: zero wind is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as cs

if TYPE_CHECKING:
    from acados_template import AcadosModel


def quaternion_kinematics_matrix(body_rate: cs.SX) -> cs.SX:
    """Hamilton right-product matrix Ω(ω), with q̇ = ½ Ω(ω) q (wxyz)."""
    wx, wy, wz = body_rate[0], body_rate[1], body_rate[2]
    return cs.vertcat(
        cs.horzcat(0, -wx, -wy, -wz),
        cs.horzcat(wx, 0, wz, -wy),
        cs.horzcat(wy, -wz, 0, wx),
        cs.horzcat(wz, wy, -wx, 0),
    )


def rotation_matrix_from_quaternion(quaternion_wxyz: cs.SX) -> cs.SX:
    """R_WB ∈ SO(3): rotate FLU body vectors into ENU world coordinates."""
    qw, qx, qy, qz = (quaternion_wxyz[i] for i in range(4))
    return cs.vertcat(
        cs.horzcat(1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
        cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)),
        cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)),
    )


def create_quadrotor_model() -> AcadosModel:
    """Construct the quadrotor command-rate model (no actuator lag, no wing force)."""
    from acados_template import AcadosModel

    position_world = cs.SX.sym("position_world", 3)
    velocity_world = cs.SX.sym("velocity_world", 3)
    body_rate_command = cs.SX.sym("body_rate_command", 3)
    attitude_wxyz = cs.SX.sym("attitude_wxyz", 4)
    specific_thrust_command = cs.SX.sym("specific_thrust_command")
    state = cs.vertcat(
        position_world,
        velocity_world,
        body_rate_command,
        attitude_wxyz,
        specific_thrust_command,
    )
    command_derivative = cs.SX.sym("command_derivative", 4)  # N/kg/s, rad/s²
    rotation_world_from_body = rotation_matrix_from_quaternion(attitude_wxyz)
    acceleration_world = rotation_world_from_body @ cs.vertcat(
        0, 0, specific_thrust_command
    ) + cs.vertcat(0, 0, -9.81)
    state_derivative = cs.vertcat(
        velocity_world,
        acceleration_world,
        command_derivative[1:4],
        0.5 * quaternion_kinematics_matrix(body_rate_command) @ attitude_wxyz,
        command_derivative[0],
    )
    model = AcadosModel()
    model.name = "quadrotor_flu"  # generated C symbol prefix, not mathematical notation
    model.x = state
    model.xdot = cs.SX.sym("state_derivative", state.numel())
    model.u = command_derivative
    model.f_expl_expr = state_derivative
    model.f_impl_expr = model.xdot - state_derivative
    return model
