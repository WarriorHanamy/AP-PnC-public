"""SUPER CIRI corridors and accumulated, eroded, projected radar depth."""

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from tooling.cli import app
from tooling.env import get_git_root
from tooling.learn.ciri import corridor_from_points
from tooling.learn.dataset import RadarDepthDataset, collect_dataset, pack_dataset
from tooling.learn.occupancy import Occupancy
from tooling.learn.pipeline import evaluate, smoke, train_rl, train_yopo
from tooling.learn.quadrotor import QuadrotorState, rotation_world_from_body
from tooling.learn.radar_depth import (
    RadarDepthConfig,
    depth_from_accumulated,
    erode_occupied,
    project_depth,
)
from tooling.learn.yopo import quintic_coefficients, quintic_samples

runner = CliRunner()


def test_erosion_drops_an_isolated_return_and_keeps_a_surface():
    voxel = 0.35
    isolated = np.array([[2.0, 0.0, 0.0]])
    surface = np.array([[4.0, 0.0, 0.0], [4.2, 0.0, 0.0], [4.4, 0.0, 0.0]])
    eroded = erode_occupied(np.vstack((isolated, surface)), voxel)
    assert len(eroded) >= 1
    assert np.all(eroded[:, 0] > 3.0)
    config = RadarDepthConfig(erosion_voxel_m=voxel)
    depth, valid = depth_from_accumulated(np.vstack((isolated, surface)), config)
    direct, direct_valid = project_depth(isolated, config)
    assert valid.any()
    assert not np.array_equal(valid, direct_valid)
    assert depth[valid].min() > 3.0
    empty, empty_valid = depth_from_accumulated(isolated, config)
    assert not empty_valid.any()
    assert np.all(empty == config.max_range_m)


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


def test_ciri_corridor_contains_the_seed_and_excludes_the_obstacle():
    start = np.array([0.0, 0.0, 1.0])
    end = np.array([1.2, 0.0, 1.0])
    obstacle = np.array([[0.6, 1.4, 1.0], [0.6, 1.6, 1.0], [0.8, 1.4, 1.0]])
    polytope = corridor_from_points(obstacle, start, end, robot_r=0.2, bound_dis=2.0)
    assert polytope.success
    assert polytope.planes.shape[1] == 4
    assert polytope.contains(start)
    assert polytope.contains(end)
    assert not polytope.contains(obstacle[0], margin=0.0)
    values = polytope.planes[:, :3] @ obstacle[0] + polytope.planes[:, 3]
    assert float(np.max(values)) > 0.0


def test_collect_pack_trains_ciri_rl_and_eroded_depth_yopo():
    root = collect_dataset("unit-radar", envs=1, samples_per_env=4, seed=1)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema"] == "ap-pnc/radar-accumulated/v1"
    assert manifest["synthetic"] is True
    assert manifest["pipelines"]["rl"] == ["accumulate", "ciri"]
    assert manifest["pipelines"]["yopo"] == ["accumulate", "erosion", "projection"]
    pack_dataset(root, seed=1)
    train = RadarDepthDataset(root, "train")
    val = RadarDepthDataset(root, "val")
    assert len(train) + len(val) == 4
    sample = train.sample(0, augment_seed=3)
    assert sample.depth.shape == (24, 32)
    assert sample.ciri_success
    assert sample.plane_count >= 6
    rotation = rotation_world_from_body(*sample.rpy)
    body = (sample.accumulated_points - sample.position) @ rotation
    recomputed, recomputed_valid = depth_from_accumulated(body, train.config)
    assert np.array_equal(recomputed_valid, sample.valid)
    assert recomputed == pytest.approx(sample.depth)
    rl_weights = train_rl(root, updates=1, seed=1)
    yopo_weights = train_yopo(root, updates=1, seed=1)
    assert rl_weights.is_file()
    assert yopo_weights.is_file()
    report = json.loads((evaluate(root, seed=1) / "comparison.json").read_text())
    assert report["synthetic"] is True
    assert "CIRI" in report["rl_input"]
    assert "erosion" in report["yopo_input"]
    assert np.isfinite(report["yopo_mean_primitive_cost"])
    assert np.isfinite(report["rl_mean_reward"])


def test_smoke_labels_the_run_synthetic():
    directory = smoke(seed=2)
    report = json.loads((directory / "comparison.json").read_text())
    assert report["synthetic"] is True
    assert directory.is_relative_to(get_git_root() / ".artifacts")
    text = (directory / "comparison.md").read_text()
    assert "synthetic" in text


def test_learn_smoke_cli():
    result = runner.invoke(app, ["learn", "smoke", "--seed", "4"])
    assert result.exit_code == 0, result.output
    assert "comparison.md" in result.output


def test_relative_dataset_is_rejected():
    result = runner.invoke(app, ["learn", "train-yopo", "--dataset", "datasets/radar"])
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
