"""Radar-to-depth dataset for YOPO. CIRI is an unfinished hook."""

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from tooling.cli import app
from tooling.env import get_git_root
from tooling.learn.ciri import CiriUnavailable, require_ciri
from tooling.learn.dataset import RadarDepthDataset, collect_dataset, pack_dataset
from tooling.learn.occupancy import Occupancy
from tooling.learn.pipeline import evaluate, smoke, train_rl, train_yopo
from tooling.learn.quadrotor import QuadrotorState
from tooling.learn.radar_depth import RadarDepthConfig, depth_from_radar
from tooling.learn.yopo import quintic_coefficients, quintic_samples

runner = CliRunner()


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


def test_ciri_hook_does_not_invent_a_representation():
    with pytest.raises(CiriUnavailable, match="unspecified|unfinished"):
        require_ciri()


def test_collect_pack_and_train_use_stored_radar_depth():
    root = collect_dataset("unit-radar", envs=1, samples_per_env=4, seed=1)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema"] == "ap-pnc/radar-depth/v1"
    assert manifest["ciri"] == "unspecified"
    assert manifest["generated"] is True
    pack_dataset(root, seed=1)
    train = RadarDepthDataset(root, "train")
    val = RadarDepthDataset(root, "val")
    assert len(train) + len(val) == 4
    assert len(val) >= 1
    sample = train.sample(0, augment_seed=3)
    assert sample.depth.shape == (24, 32)
    recomputed, recomputed_valid = depth_from_radar(sample.radar_points, train.config)
    assert recomputed == pytest.approx(sample.depth)
    assert np.array_equal(recomputed_valid, sample.valid)
    again = train.sample(0, augment_seed=3)
    assert again.velocity_body == pytest.approx(sample.velocity_body)
    assert again.goal_direction_body == pytest.approx(sample.goal_direction_body)
    weights = train_yopo(root, updates=1, seed=1)
    assert weights.is_file()
    report_dir = evaluate(root, seed=1)
    report = json.loads((report_dir / "comparison.json").read_text())
    assert report["ciri"] == "unfinished"
    assert "radar" in report["yopo_input"]
    assert np.isfinite(report["yopo_mean_primitive_cost"])
    with pytest.raises(CiriUnavailable):
        train_rl(root, updates=1, seed=1)


def test_smoke_does_not_claim_ciri():
    directory = smoke(seed=2)
    report = json.loads((directory / "comparison.json").read_text())
    assert report["ciri"] == "unfinished"
    assert directory.is_relative_to(get_git_root() / ".artifacts")
    text = (directory / "comparison.md").read_text()
    assert "CIRI is unfinished" in text


def test_learn_smoke_cli():
    result = runner.invoke(app, ["learn", "smoke", "--seed", "4"])
    assert result.exit_code == 0, result.output
    assert "comparison.md" in result.output


def test_relative_dataset_and_rl_are_rejected():
    result = runner.invoke(app, ["learn", "train-yopo", "--dataset", "datasets/radar"])
    assert result.exit_code == 2
    assert "absolute" in result.output
    absolute = get_git_root() / ".artifacts" / "learn" / "datasets" / "unused"
    refused = runner.invoke(app, ["learn", "train-rl", "--dataset", str(absolute)])
    assert refused.exit_code == 2
    assert "CIRI" in refused.output


def test_occupancy_ray_stops_on_the_first_voxel():
    points = np.array([[2.0, 0.0, 0.0]])
    occupancy = Occupancy(points, voxel_m=0.5, ground_z=-1.0)
    ranges = occupancy.raycast(np.zeros(3), np.array([[1.0, 0.0, 0.0]]), 5.0)
    assert np.isfinite(ranges[0])
    assert ranges[0] <= 2.5
    state = QuadrotorState(np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0)
    assert state.copy().position is not state.position
