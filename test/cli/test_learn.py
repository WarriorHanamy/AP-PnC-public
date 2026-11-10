"""Radar-depth YOPO, quadrotor RL, and the post-CIRI scene gate."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from tooling.cli import app
from tooling.env import get_git_root
from tooling.learn.occupancy import Occupancy
from tooling.learn.pipeline import evaluate, smoke, train_rl, train_yopo
from tooling.learn.quadrotor import QuadrotorState
from tooling.learn.radar_depth import RadarDepthConfig, depth_from_radar
from tooling.learn.scenes import load_post_ciri_scene
from tooling.learn.yopo import quintic_coefficients, quintic_samples

runner = CliRunner()


def _ply(path: Path, points: np.ndarray) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    lines.extend(" ".join(f"{value:.4f}" for value in row) for row in points)
    path.write_text("\n".join(lines) + "\n")


def _scene_dir(root: Path, source: str = "post-ciri") -> Path:
    root.mkdir()
    xs = np.linspace(0.0, 6.0, 8)
    points = np.column_stack((xs, np.full_like(xs, 2.0), np.ones_like(xs)))
    _ply(root / "occupancy.ply", points)
    (root / "scene.yaml").write_text(
        yaml.safe_dump(
            {
                "source": source,
                "name": "fixture-room",
                "occupancy": "occupancy.ply",
                "voxel_m": 0.5,
                "ground_z": 0.0,
                "start": [0.0, 0.0, 1.0],
                "goal": [5.0, 0.0, 1.0],
            }
        )
    )
    return root


def test_boresight_radar_return_lands_on_the_depth_center():
    config = RadarDepthConfig()
    depth, valid = depth_from_radar(np.array([[5.0, 0.0, 0.0]]), config)
    center = (config.height // 2, config.width // 2)
    assert valid[center]
    assert depth[center] == pytest.approx(5.0)
    assert depth[0, 0] == config.max_range_m
    assert not valid[0, 0]


def test_empty_radar_cloud_is_all_invalid_max_range():
    config = RadarDepthConfig()
    depth, valid = depth_from_radar(np.zeros((0, 3)), config)
    assert not valid.any()
    assert np.all(depth == config.max_range_m)


def test_quintic_meets_the_requested_endpoint():
    coefficients = quintic_coefficients(
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.1, 0.0, 0.0]),
        np.zeros(3),
    )
    end = quintic_samples(coefficients, np.array([1.0]))[0]
    assert end == pytest.approx([1.0, 2.0, 3.0])


def test_post_ciri_loader_rejects_missing_and_non_ciri_sources(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no CIRI assets"):
        load_post_ciri_scene(tmp_path / "absent")
    scene = _scene_dir(tmp_path / "synthetic", source="synthetic-smoke")
    with pytest.raises(ValueError, match="post-ciri"):
        load_post_ciri_scene(scene)
    with pytest.raises(ValueError, match="absolute"):
        load_post_ciri_scene(Path("relative/scene"))


def test_post_ciri_train_and_eval_round_trip(tmp_path: Path):
    scene = load_post_ciri_scene(_scene_dir(tmp_path / "fixture-room"))
    assert scene.name == "fixture-room"
    train_rl(scene, updates=1, seed=0)
    train_yopo(scene, updates=1, seed=0)
    directory = evaluate(scene, episodes=1, seed=1)
    report = json.loads((directory / "comparison.json").read_text())
    assert report["scene_source"] == "post-ciri"
    assert report["not_ciri"] is False
    assert set(report["policies"]) == {"rl", "yopo_radar_depth"}
    assert "radar" in report["yopo_input"]


def test_eval_without_weights_names_the_missing_file(tmp_path: Path):
    scene = load_post_ciri_scene(_scene_dir(tmp_path / "untrained"))
    scene = type(scene)("untrained-eval", scene.directory, scene.occupancy, scene.start, scene.goal)
    with pytest.raises(FileNotFoundError, match="rl.npz"):
        evaluate(scene, episodes=1, seed=0)


def test_smoke_report_is_explicitly_not_ciri():
    directory = smoke(seed=2)
    report = json.loads((directory / "comparison.json").read_text())
    assert report["not_ciri"] is True
    assert report["scene_source"] == "synthetic-smoke"
    assert directory.is_relative_to(get_git_root() / ".artifacts")
    text = (directory / "comparison.md").read_text()
    assert "not a post-CIRI result" in text


def test_learn_smoke_cli():
    result = runner.invoke(app, ["learn", "smoke", "--seed", "4"])
    assert result.exit_code == 0, result.output
    assert "comparison.md" in result.output


def test_relative_scene_is_rejected_by_the_cli(tmp_path: Path):
    result = runner.invoke(app, ["learn", "train-rl", "--scene", "scenes/ciri"])
    assert result.exit_code == 2
    assert "absolute" in result.output


def test_occupancy_ray_stops_on_the_first_voxel():
    points = np.array([[2.0, 0.0, 0.0]])
    occupancy = Occupancy(points, voxel_m=0.5, ground_z=-1.0)
    ranges = occupancy.raycast(np.zeros(3), np.array([[1.0, 0.0, 0.0]]), 5.0)
    assert np.isfinite(ranges[0])
    assert ranges[0] <= 2.5
    state = QuadrotorState(np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0)
    assert state.copy().position is not state.position
