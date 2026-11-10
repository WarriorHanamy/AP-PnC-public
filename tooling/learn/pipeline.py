"""Train and compare quadrotor RL against radar-depth YOPO."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tooling.env import artifact_path
from tooling.learn.net import ReluMlp
from tooling.learn.occupancy import Occupancy
from tooling.learn.quadrotor import (
    QuadrotorState,
    command_from_acceleration,
    step,
)
from tooling.learn.radar_depth import RadarDepthConfig, depth_from_radar, radar_points
from tooling.learn.rl import (
    RL_SIZES,
    action_to_command,
    initial_state,
    policy_update,
    sample_action,
)
from tooling.learn.scenes import PostCiriScene
from tooling.learn.yopo import YOPO_SIZES, guidance_step, observation, plan_acceleration

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
    from tooling.learn.quadrotor import rotation_world_from_body

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


def _run_directory(name: str) -> Path:
    directory = artifact_path(f".artifacts/learn/{name}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def train_rl(scene: PostCiriScene, updates: int, seed: int) -> Path:
    if updates < 1:
        raise ValueError("updates must be positive")
    directory = _run_directory(scene.name)
    rng = np.random.default_rng(seed)
    net = ReluMlp(RL_SIZES, rng)
    config = RadarDepthConfig()
    for _ in range(updates):
        _stats, observations, actions, rewards = rollout(
            "rl", net, scene.occupancy, scene.start, scene.goal, rng, max_steps=30, config=config
        )
        policy_update(net, observations, actions, rewards)
    path = directory / "rl.npz"
    _save_net(path, net)
    return path


def train_yopo(scene: PostCiriScene, updates: int, seed: int) -> Path:
    if updates < 1:
        raise ValueError("updates must be positive")
    directory = _run_directory(scene.name)
    rng = np.random.default_rng(seed)
    net = ReluMlp(YOPO_SIZES, rng)
    config = RadarDepthConfig()
    state = initial_state(scene.start, scene.goal)
    for _ in range(updates):
        features = _sense(scene.occupancy, state, scene.goal, config, rng)
        guidance_step(net, features, state, scene.goal, scene.occupancy, config)
    path = directory / "yopo.npz"
    _save_net(path, net)
    return path


def _summarize(episodes: list[EpisodeStats]) -> dict[str, float]:
    count = len(episodes)
    return {
        "episodes": count,
        "success_rate": sum(item.success for item in episodes) / count,
        "collision_rate": sum(item.collision for item in episodes) / count,
        "mean_steps": sum(item.steps for item in episodes) / count,
        "mean_path_length_m": sum(item.path_length_m for item in episodes) / count,
        "mean_min_clearance_m": sum(item.min_clearance_m for item in episodes) / count,
    }


def evaluate(scene: PostCiriScene, episodes: int, seed: int) -> Path:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    directory = _run_directory(scene.name)
    rl_net = _load_net(directory / "rl.npz", RL_SIZES)
    yopo_net = _load_net(directory / "yopo.npz", YOPO_SIZES)
    config = RadarDepthConfig(range_noise_m=0.0, keep_probability=1.0)
    results = {}
    for name, net in (("rl", rl_net), ("yopo_radar_depth", yopo_net)):
        policy = "rl" if name == "rl" else "yopo"
        stats = [
            rollout(
                policy,
                net,
                scene.occupancy,
                scene.start,
                scene.goal,
                np.random.default_rng(seed + index),
                max_steps=40,
                config=config,
            )[0]
            for index in range(episodes)
        ]
        results[name] = _summarize(stats)
    report = {
        "scene_source": "post-ciri",
        "not_ciri": False,
        "scene": scene.name,
        "policies": results,
        "yopo_input": "depth rasterized from a sparse forward radar point cloud",
    }
    return _write_report(directory, report)


def _write_report(directory: Path, report: dict) -> Path:
    (directory / "comparison.json").write_text(json.dumps(report, indent=2))
    lines = [
        f"# {report['scene']}",
        "",
        f"scene_source: {report['scene_source']}",
        "",
        "| policy | success | collision | mean steps | path length m | min clearance m |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in report["policies"].items():
        lines.append(
            f"| {name} | {stats['success_rate']:.2f} | {stats['collision_rate']:.2f} | "
            f"{stats['mean_steps']:.1f} | {stats['mean_path_length_m']:.2f} | "
            f"{stats['mean_min_clearance_m']:.2f} |"
        )
    if report["not_ciri"]:
        lines.extend(["", "This run is not a post-CIRI result."])
    (directory / "comparison.md").write_text("\n".join(lines) + "\n")
    return directory


def synthetic_room() -> tuple[Occupancy, np.ndarray, np.ndarray]:
    """A box corridor used only to exercise the code. It is not a CIRI scene."""
    xs = np.linspace(0.0, 6.0, 13)
    left = np.column_stack((xs, np.full_like(xs, 1.2), np.full_like(xs, 1.0)))
    right = np.column_stack((xs, np.full_like(xs, -1.2), np.full_like(xs, 1.0)))
    gate = np.column_stack((np.full(5, 3.0), np.linspace(-0.8, 0.8, 5), np.full(5, 1.0)))
    points = np.vstack((left, right, gate))
    start = np.array([0.4, 0.0, 1.0])
    goal = np.array([5.5, 0.0, 1.0])
    return Occupancy(points, voxel_m=0.4, ground_z=0.0), start, goal


def smoke(seed: int = 0) -> Path:
    """Run a few RL and YOPO updates on the synthetic room and write a report."""
    occupancy, start, goal = synthetic_room()
    rng = np.random.default_rng(seed)
    config = RadarDepthConfig(range_noise_m=0.0, keep_probability=1.0)
    rl_net = ReluMlp(RL_SIZES, rng)
    yopo_net = ReluMlp(YOPO_SIZES, np.random.default_rng(seed + 1))
    state = initial_state(start, goal)
    for _ in range(2):
        stats, observations, actions, rewards = rollout(
            "rl", rl_net, occupancy, start, goal, rng, max_steps=8, config=config
        )
        policy_update(rl_net, observations, actions, rewards)
        del stats
        features = _sense(occupancy, state, goal, config, rng)
        guidance_step(yopo_net, features, state, goal, occupancy, config)
    results = {}
    for name, net, policy in (
        ("rl", rl_net, "rl"),
        ("yopo_radar_depth", yopo_net, "yopo"),
    ):
        episode = rollout(
            policy, net, occupancy, start, goal, np.random.default_rng(seed + 3), 8, config
        )[0]
        results[name] = _summarize([episode])
    directory = _run_directory(f"smoke-{seed}")
    report = {
        "scene_source": "synthetic-smoke",
        "not_ciri": True,
        "scene": "synthetic-room",
        "policies": results,
        "yopo_input": "depth rasterized from a sparse forward radar point cloud",
    }
    return _write_report(directory, report)
