# SUPER high-speed scene, phase 1

This brings SUPER's high-speed benchmark geometry into the existing tailsitter
stack. It does not vendor SUPER, ROG-Map, or CIRI, and it does not replace
GCOPTER/MINCO or the tailsitter flatness map.

The default `planning.yaml` mission is unchanged (`corridor_source: box_sfc`,
circle `basic_traj`, voxel map + FIRI). Nothing in the default build downloads
or compiles third-party SUPER sources.

## License

AP-PnC is BSD-3-Clause. SUPER source headers are commonly LGPL-3.0 and the
SUPER repository has no root `LICENSE`. The standalone ROG-Map repository's
`LICENSE` is GPL-3.0, while its ROS 2 `package.xml` says BSD. Those statements
disagree. Until HKU MaRS confirms an SPDX identifier in writing, this
repository does not vendor, subtree, or submodule SUPER, ROG-Map, or CIRI, and
it does not commit their map binaries.

Optional libraries are linked from an external prefix that stays outside the
git tree (a gitignored checkout under `.artifacts/` is fine). See
`core/ros_packages/planner/cmake/optional_rog_ciri.cmake`.

## Scene assets

Upstream map and mission, from
[hku-mars/SUPER](https://github.com/hku-mars/SUPER) `master`:

| Item | Upstream |
|---|---|
| Point cloud | `mars_uav_sim/perfect_drone_sim/pcd/random_map_24_6635.pcd` |
| Sensor / start | `mars_uav_sim/perfect_drone_sim/config/high_speed.yaml` |
| Waypoints | `mission_planner/data/benchmark.txt`: `(0, 50, 1.5)` then `(0, -50, 1.5)` |
| Start pose | `(0, -50, 1.5)`, yaw `1.5741` rad |

Generate the PLY (network, or pass a PCD you already have). The upstream file
contains a few `nan` rows; the script drops those and writes the finite points.
Output stays under `.artifacts/` and is not committed:

```bash
python3 "$AP_PNC_DIR/infra/sim_infra/lidar_sim/scripts/prepare_super_high_speed_scene.py"
# or: --pcd /absolute/random_map_24_6635.pcd
```

`lidar_sim` then ray-marches that PLY. The sensor block in
`core/bringup/config/lidar_sim_super_high_speed.yaml` follows the upstream
file: 360°, 178° vertical field, 0.3° azimuth (1200 columns), 10 Hz, 0.1–70 m.
`lidar_sim` accepts at most 128 channels, so the vertical channel count is 128
rather than `178 / 0.3`. Range noise is zero in that upstream file and here.

## Bringup

Pillar, garden, and city scenes are unchanged. This scene is an extra dev-sim
choice. It still needs the local practical actuator profile.

```bash
uv run --project "$AP_PNC_DIR" ap-pnc dev-sim prepare \
  --scene super-high-speed --v-max 8 --corridor-source box_sfc
# ground-truth voxel corridor instead of the fixed waypoint boxes:
#   --corridor-source map_sfc
uv run --project "$AP_PNC_DIR" ap-pnc dev-sim run "$JOB"
```

`prepare` copies `planning_super_high_speed.yaml` onto the job's
`planning.yaml` and the sensor file onto `lidar_sim.yaml`, and mounts the PLY
at `/workspace/scene/random_map_24_6635.ply`. `simple_sim` reads that
`planning.yaml`, so the in-process `plan()` starts at `(0, -50, 1.5)` and uses
the round-trip waypoints. It does not use the default circle.

`core/bringup/launch/super_high_speed.launch.py` launches the same plant and
`lidar_sim` nodes. Use it with the prepared config directory. The plant reads
`planning.yaml`, not the scene file's source name.

`--v-max` overwrites the scene file. SUPER's quadrotor config uses 25 m/s
(`super_planner/config/static_high_speed.yaml`, `max_vel`). Do not pass 25.
The scene file recommends 8 m/s; the dev-sim CLI default of 4 m/s is also
inside the tailsitter band. `simple_sim` still has no contact model: the cloud
is what the lidar sees and, with `map_sfc`, what the ground-truth corridor
uses. It does not push on the vehicle.

The cruise tangent is +Y, matching yaw ≈ π/2 (upstream yaw is 1.5741 rad).
The low-speed tailsitter hover heading is the existing flatness policy.

`/lidar_sim/points` stays in the `lidar` frame. `planner_node` transforms it
with the odometry pose and the mount in `lidar_sim.yaml`, then publishes
`/planner/cloud_world` in `world` (ENU). A lidar-frame cloud is not forwarded.
`/lidar_sim/map` is already in `world` and remains the `map_sfc` source.

## Optional ROG-Map and CIRI

Default configure does not look for these libraries.

```bash
cmake -DAP_PNC_WITH_ROG_MAP=ON -DAP_PNC_ROG_MAP_PREFIX=/absolute/rog-map-prefix
cmake -DAP_PNC_WITH_CIRI=ON -DAP_PNC_CIRI_PREFIX=/absolute/ciri-prefix
```

Set `problem_formulation.map_backend: rog` only in that ROG-Map build, and set
`rog_map_config` to an absolute ROG-Map yaml. The adapter loads that yaml,
forces `ros_callback.enable` off, and calls `updateMap(cloud, pose)` with the
world-frame cloud and the body pose. Their `init()` opens log files under the
`ROOT_DIR` baked into that external library; that checkout has to be writable
there. The ground-truth voxel map remains the `map_sfc` corridor source in
this phase.

`corridor_backend: ciri` calls an external ROS-free header, not SUPER's
`super_core/ciri.h` (that header includes a ROS interface). The prefix must
provide `include/ciri/corridor.hpp` and a library named `ciri_core` or `ciri`:

```cpp
namespace ciri {
bool decompose_segment(const Eigen::MatrixX4d &bounds,
                       const Eigen::Matrix3Xd &obstacles,
                       const Eigen::Vector3d &segment_start,
                       const Eigen::Vector3d &segment_end,
                       double robot_radius_m, int iterations,
                       Eigen::MatrixX4d &polytope);
}
```

`polytope` rows are `ax + by + cz + d <= 0`, the same `MatrixX4d` GCOPTER
already consumes. A one-file adapter with that signature can live in the
external prefix. It is not added to this tree.

Python bindings always compile the stub and do not define the optional macros.

## Not in this phase

- Online scans do not replace the ground-truth `/lidar_sim/map` voxel map.
- The dense SUPER benchmark is not wired.
- There is no second trajectory, backup trajectory, or 15 Hz SUPER FSM.
- Quadrotor flatness and `perfect_drone_sim` are not ported.
- `simple_sim` still does not collide with the cloud.
