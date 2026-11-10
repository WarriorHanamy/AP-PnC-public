"""SUPER high-speed scene staging stays out of the source tree."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from tooling.env import get_git_root

ROOT = get_git_root()
SCRIPT = ROOT / "infra/sim_infra/lidar_sim/scripts/prepare_super_high_speed_scene.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_super_high_speed_scene", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pcd(path: Path, points, fmt="ascii"):
    header = [
        "# .PCD v0.7",
        "VERSION 0.7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        f"WIDTH {len(points)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(points)}",
        f"DATA {fmt}",
    ]
    if fmt == "ascii":
        body = "\n".join(f"{x} {y} {z}" for x, y, z in points) + "\n"
        path.write_text("\n".join(header) + "\n" + body)
        return
    payload = np.asarray(points, dtype="<f4").tobytes()
    path.write_bytes(("\n".join(header) + "\n").encode() + payload)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("AP_PNC_DIR", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("fmt", ["ascii", "binary"])
def test_pcd_xyz_becomes_a_lidar_sim_ply_under_artifacts(project, fmt):
    sys.path.insert(0, str(ROOT / "infra/sim_infra/lidar_sim"))
    from lidar_sim.geometry import load_xyz_ply

    module = _load_script()
    points = np.array([[0.0, -50.0, 1.5], [0.0, 50.0, 1.5]], dtype=np.float32)
    pcd = project / "local.pcd"
    _write_pcd(pcd, points, fmt)
    ply = module.convert_pcd_to_ply(pcd, ".artifacts/dev-sim/scenes/tiny.ply")
    assert ply == project / ".artifacts/dev-sim/scenes/tiny.ply"
    np.testing.assert_allclose(load_xyz_ply(ply), points)


def test_ply_output_cannot_leave_artifacts(project):
    module = _load_script()
    pcd = project / "local.pcd"
    _write_pcd(pcd, [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="artifacts"):
        module.convert_pcd_to_ply(pcd, project / "scene.ply")


def test_nonfinite_pcd_rows_are_dropped(project, capsys):
    module = _load_script()
    pcd = project / "sparse.pcd"
    _write_pcd(pcd, [[0.0, -50.0, 1.5], [float("nan"), float("nan"), float("nan")]])
    points = module.read_pcd_xyz(pcd)
    assert len(points) == 1
    assert "dropped 1 nonfinite" in capsys.readouterr().err


def test_relative_pcd_is_rejected(project):
    module = _load_script()
    with pytest.raises(ValueError, match="absolute"):
        module.read_pcd_xyz(Path("relative.pcd"))


def test_super_sources_and_maps_are_not_tracked():
    tracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-files"], text=True)
    assert ".pcd" not in tracked
    assert "random_map" not in tracked
    assert "rog_map/" not in tracked
    assert "super_core/ciri.h" not in tracked


def test_high_speed_sensor_matches_the_published_benchmark():
    lidar = yaml.safe_load(
        (ROOT / "core/bringup/config/lidar_sim_super_high_speed.yaml").read_text()
    )["lidar_sim"]
    sensor = lidar["sensor"]
    assert sensor["channels"] == 128
    assert sensor["columns"] == 1200
    assert sensor["vertical_fov_deg"] == 178.0
    assert sensor["rate_hz"] == 10.0
    assert sensor["min_range_m"] == 0.1
    assert sensor["max_range_m"] == 70.0
    assert lidar["scene"]["type"] == "point_cloud"
    assert lidar["scene"]["path"].endswith("random_map_24_6635.ply")
    mission = yaml.safe_load(
        (ROOT / "core/bringup/config/planning_super_high_speed.yaml").read_text()
    )["problem_formulation"]
    assert mission["waypoints"] == [[0.0, -50.0, 1.5], [0.0, 50.0, 1.5], [0.0, -50.0, 1.5]]
    assert mission["basic_traj"]["type"] == 4
    assert mission["cost"]["v_max"] == 8.0
    assert mission["map_backend"] == "voxel"
    assert mission["corridor_backend"] == "firi"
    default = yaml.safe_load((ROOT / "core/bringup/config/planning.yaml").read_text())
    assert default["problem_formulation"]["corridor_source"] == "box_sfc"
    assert "waypoints" not in default["problem_formulation"]
    assert (ROOT / "core/bringup/launch/super_high_speed.launch.py").is_file()
    assert (ROOT / "infra/SUPER_SCENE.md").is_file()
