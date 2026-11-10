"""Train and compare quadrotor RL against radar-depth YOPO."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tooling.learn.dataset import RadarDepthDataset, collect_dataset, pack_dataset
from tooling.learn.net import ReluMlp
from tooling.learn.occupancy import Occupancy
from tooling.learn.quadrotor import (
    QuadrotorState,
    command_from_acceleration,
    rotation_world_from_body,
    step,
)
from tooling.learn.radar_depth import RadarDepthConfig
from tooling.learn.rl import action_to_command, initial_state, sample_action
from tooling.learn.yopo import (
    YOPO_SIZES,
    guidance_step,
    observation,
    plan_acceleration,
    selected_primitive_cost,
)

DT = 0.1
CLEARANCE_M = 0.35
GOAL_RADIUS_M = 0.6


@dataclass
class EpisodeStats:
    success: bool
    collision: bool
    steps: int
    path_length_m: float
    min_clearance_m: float


def _sense(
    occupancy: Occupancy,
    state: QuadrotorState,
    goal: np.ndarray,
    config: RadarDepthConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    from tooling.learn.radar_depth import depth_from_radar, radar_points

    rotation = rotation_world_from_body(state.roll, state.pitch, state.yaw)
    points = radar_points(occupancy, state.position, rotation, config, rng)
    depth, _valid = depth_from_radar(points, config)
    return observation(depth, state, goal, config)


def rollout(
    policy: str,
    net: ReluMlp,
    occupancy: Occupancy,
    start: np.ndarray,
    goal: np.ndarray,
    rng: np.random.Generator,
    max_steps: int,
    config: RadarDepthConfig | None = None,
) -> tuple[EpisodeStats, list[np.ndarray], list[np.ndarray], list[float]]:
    if policy not in {"rl", "yopo"}:
        raise ValueError("policy must be rl or yopo")
    sensor = config or RadarDepthConfig(range_noise_m=0.0, keep_probability=1.0)
    state = initial_state(start, goal)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    previous = float(np.linalg.norm(state.position - goal))
    length = 0.0
    min_clearance = float(occupancy.clearance(state.position[None, :])[0])
    success = False
    collision = False
    steps = 0
    for _ in range(max_steps):
        features = _sense(occupancy, state, goal, sensor, rng)
        if policy == "rl":
            action, _mean = sample_action(net, features, rng)
            thrust, rates = action_to_command(action)
        else:
            action = np.zeros(4)
            acceleration = plan_acceleration(net, features, state, sensor, DT)
            command = command_from_acceleration(state, acceleration)
            thrust, rates = float(command[0]), command[1:]
        nxt = step(state, thrust, rates, DT)
        clearance = float(occupancy.clearance(nxt.position[None, :])[0])
        distance = float(np.linalg.norm(nxt.position - goal))
        reward = previous - distance - 0.01 * float(np.linalg.norm(rates))
        previous = distance
        length += float(np.linalg.norm(nxt.position - state.position))
        min_clearance = min(min_clearance, clearance)
        state = nxt
        steps += 1
        collision = clearance < CLEARANCE_M
        success = distance < GOAL_RADIUS_M
        if collision:
            reward -= 5.0
        elif success:
            reward += 5.0
        observations.append(features)
        actions.append(action)
        rewards.append(reward)
        if collision or success:
            break
    stats = EpisodeStats(success, collision, steps, length, min_clearance)
    return stats, observations, actions, rewards


def _save_net(path: Path, net: ReluMlp) -> None:
    payload = {}
    for index, (weight, bias) in enumerate(zip(net.weights, net.biases, strict=True)):
        payload[f"W{index}"] = weight
        payload[f"b{index}"] = bias
    np.savez(path, **payload)


def _load_net(path: Path, sizes: tuple[int, ...]) -> ReluMlp:
    if not path.is_file():
        raise FileNotFoundError(f"missing weights {path}; train this policy first")
    stored = np.load(path)
    net = ReluMlp(sizes, np.random.default_rng(0))
    for index in range(len(sizes) - 1):
        net.weights[index] = stored[f"W{index}"]
        net.biases[index] = stored[f"b{index}"]
    return net


def train_rl(dataset: Path, updates: int, seed: int) -> Path:
    """RL is not trained here. Its observation was specified as CIRI, which is undefined."""
    del dataset, updates, seed
    from tooling.learn.ciri import require_ciri

    require_ciri()
    raise AssertionError("unreachable")


def _state_from_sample(sample) -> tuple[QuadrotorState, np.ndarray]:
    roll, pitch, yaw = (float(value) for value in sample.rpy)
    rotation = rotation_world_from_body(roll, pitch, yaw)
    state = QuadrotorState(
        sample.position.copy(),
        rotation @ sample.velocity_body,
        roll,
        pitch,
        yaw,
    )
    goal = sample.position + rotation @ (4.0 * sample.goal_direction_body)
    return state, goal


def train_yopo(dataset: Path, updates: int, seed: int) -> Path:
    """Guidance updates on packed radar-depth views. The depth is not recomputed."""
    if updates < 1:
        raise ValueError("updates must be positive")
    split = RadarDepthDataset(dataset, "train")
    rng = np.random.default_rng(seed)
    net = ReluMlp(YOPO_SIZES, rng)
    for step_index in range(updates):
        sample = split.sample(step_index % len(split), augment_seed=seed + step_index)
        state, goal = _state_from_sample(sample)
        features = observation(sample.depth, state, goal, split.config)
        guidance_step(
            net,
            features,
            state,
            goal,
            split.environment(sample.env_index),
            split.config,
        )
    path = split.root / "yopo.npz"
    _save_net(path, net)
    return path


def evaluate(dataset: Path, seed: int) -> Path:
    """Score YOPO on the packed validation views. RL is reported as unfinished."""
    split = RadarDepthDataset(dataset, "val")
    net = _load_net(split.root / "yopo.npz", YOPO_SIZES)
    costs = []
    for index in range(len(split)):
        sample = split.sample(index, augment_seed=seed + index)
        state, goal = _state_from_sample(sample)
        features = observation(sample.depth, state, goal, split.config)
        costs.append(
            selected_primitive_cost(
                net,
                features,
                state,
                goal,
                split.environment(sample.env_index),
                split.config,
            )
        )
    report = {
        "schema": "ap-pnc/radar-depth-eval/v1",
        "ciri": "unfinished",
        "dataset": str(split.root),
        "yopo_input": "stored depth rasterized from the recorded radar point cloud",
        "val_views": len(split),
        "yopo_mean_primitive_cost": float(np.mean(costs)),
        "rl": "not run; CIRI observation interface is unspecified",
    }
    return _write_report(split.root, report)


def _write_report(directory: Path, report: dict) -> Path:
    (directory / "comparison.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# Radar-depth YOPO",
        "",
        "CIRI is unfinished. This comparison does not use a CIRI representation.",
        "",
        f"validation views: {report['val_views']}",
        f"YOPO mean primitive cost: {report['yopo_mean_primitive_cost']:.4f}",
        f"RL: {report['rl']}",
        "",
    ]
    (directory / "comparison.md").write_text("\n".join(lines))
    return directory


def smoke(seed: int = 0) -> Path:
    """Collect a tiny synthetic radar-depth set, pack it, and train YOPO for one step."""
    name = f"smoke-{seed}"
    root = collect_dataset(name, envs=1, samples_per_env=4, seed=seed)
    pack_dataset(root, seed=seed)
    train_yopo(root, updates=1, seed=seed)
    return evaluate(root, seed=seed)
