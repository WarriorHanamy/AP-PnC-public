/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2025 - <YEARS>, Sun Yat-sen University.
 * All rights reserved.
 *
 * Authors:
 * Erchao Rong: rongerch@outlook.com
 * Zihao Liu: liuzh297@gmail.com
 * Junning Liang: gordonliang27@foxmail.com
 *
 * Paper:
 * Aerodynamic Prior-free Trajectory Generation and Tracking Control for a
 * Tail-sitter UAV.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <planner_core/ciri.hpp>
#include <planner_core/corridor_adapter.hpp>
#include <planner_core/map_query.hpp>
#include <planner_core/planner.hpp>
#include <planner_core/waypoint_route.hpp>
#include <planner_node/lidar_cloud_world.hpp>
#include <stdexcept>

namespace
{

  bool inside(const Eigen::MatrixX4d &poly, const Eigen::Vector3d &point)
  {
    return (poly.leftCols<3>() * point + poly.col(3)).maxCoeff() <= 1e-6;
  }

}  // namespace

TEST(MapBackends, DefaultBuildKeepsVoxelAndFiri)
{
  planner::core::ProblemConfig config;
  EXPECT_EQ(config.problem.map_backend, "voxel");
  EXPECT_EQ(config.problem.corridor_backend, "firi");
  EXPECT_TRUE(config.problem.waypoints.empty());
  EXPECT_EQ(config.problem.corridor_source, "box_sfc");
  EXPECT_TRUE(planner::core::map_backend_available("voxel"));
  EXPECT_FALSE(planner::core::map_backend_available("rog"));
  EXPECT_TRUE(planner::core::corridor_backend_available("firi"));
  EXPECT_TRUE(planner::core::corridor_backend_available("ciri"));
  EXPECT_FALSE(planner::core::make_rog_map_adapter("")->available());
}

TEST(MapBackends, VoxelAdapterMatchesTheMapAndFiriBoundsAnEmptyScene)
{
  voxel_map::VoxelMap            map(Eigen::Vector3i(20, 20, 10),
                                     Eigen::Vector3d(-5.0, -5.0, -1.0), 0.5);
  planner::core::VoxelMapAdapter adapter(map);
  const Eigen::Vector3d          inside_point(0.2, 0.2, 0.2);
  const Eigen::Vector3d          outside(40.0, 0.0, 0.0);
  EXPECT_EQ(adapter.occupied(inside_point), map.query(inside_point));
  EXPECT_FALSE(adapter.occupied(inside_point));
  EXPECT_TRUE(adapter.occupied(outside));
  EXPECT_EQ(adapter.backend(), "voxel");

  std::vector<Eigen::MatrixX4d> polys;
  std::vector<Eigen::Vector3d>  route;
  std::string                   error;
  const auto                    corridor = planner::core::make_corridor_adapter(
      "firi", {}, [&error](const std::string &msg) { error = msg; });
  ASSERT_TRUE(corridor->build(adapter, Eigen::Vector3d(-1.0, 0.0, 0.0),
                              Eigen::Vector3d(1.0, 0.0, 0.0), polys, route))
      << error;
  ASSERT_FALSE(polys.empty());
  EXPECT_TRUE(inside(polys.front(), Eigen::Vector3d::Zero()));
  for (const Eigen::MatrixX4d &poly : polys)
  {
    EXPECT_EQ(poly.cols(), 4);
  }
}

TEST(MapBackends, CiriHalfspacesSeparateASideObstacle)
{
  Eigen::Matrix<double, 6, 4> bounds = Eigen::Matrix<double, 6, 4>::Zero();
  bounds(0, 0)                       = 1.0;
  bounds(0, 3)                       = -3.0;
  bounds(1, 0)                       = -1.0;
  bounds(1, 3)                       = -3.0;
  bounds(2, 1)                       = 1.0;
  bounds(2, 3)                       = -3.0;
  bounds(3, 1)                       = -1.0;
  bounds(3, 3)                       = -3.0;
  bounds(4, 2)                       = 1.0;
  bounds(4, 3)                       = -3.0;
  bounds(5, 2)                       = -1.0;
  bounds(5, 3)                       = -3.0;
  const std::vector<Eigen::Vector3d> obstacles{{1.2, 0.0, 0.0}};
  Eigen::MatrixX4d                   polytope;
  ASSERT_TRUE(planner::core::ciri_decompose(
      bounds, obstacles, Eigen::Vector3d(0.0, -1.0, 0.0),
      Eigen::Vector3d(0.0, 1.0, 0.0), 0.2, 2, polytope));
  EXPECT_TRUE(inside(polytope, Eigen::Vector3d::Zero()));
  EXPECT_TRUE(inside(polytope, Eigen::Vector3d(0.0, -1.0, 0.0)));
  EXPECT_TRUE(inside(polytope, Eigen::Vector3d(0.0, 1.0, 0.0)));
  EXPECT_FALSE(inside(polytope, obstacles.front()));
}

TEST(MapBackends, WaypointRouteReplacesTheDefaultCircleSeed)
{
  planner::core::ProblemConfig config;
  config.problem.piece_n            = 8;
  config.problem.waypoint_speed_m_s = 8.0;
  config.problem.waypoints          = {Eigen::Vector3d(0.0, -50.0, 1.5),
                                       Eigen::Vector3d(0.0, 50.0, 1.5),
                                       Eigen::Vector3d(0.0, -50.0, 1.5)};
  planner::core::Planner planner(config);
  const auto             route = planner.mission_route();
  ASSERT_EQ(route.size(), 9U);
  EXPECT_NEAR(route.front()(0, 0), 0.0, 1e-9);
  EXPECT_NEAR(route.front()(1, 0), -50.0, 1e-9);
  EXPECT_NEAR(route.front()(2, 0), 1.5, 1e-9);
  EXPECT_NEAR(route[4](1, 0), 50.0, 1e-6);
  EXPECT_NEAR(route.back()(1, 0), -50.0, 1e-6);
  EXPECT_GT(route.front().col(1).y(), 0.0);
  EXPECT_NEAR(route.front().col(1).norm(), 8.0, 1e-9);

  planner::core::ProblemConfig plain;
  plain.problem.piece_n = 8;
  planner::core::Planner circle(plain);
  const auto             nominal = circle.mission_route();
  EXPECT_GT((nominal.front().col(0) - route.front().col(0)).norm(), 1.0);
}

TEST(LidarCloudWorld, LidarFramePointBecomesWorldEnu)
{
  planner::node::BodyPose body;
  body.position_world = Eigen::Vector3d(1.0, 2.0, 3.0);
  planner::node::LidarMount          mount;
  const std::vector<Eigen::Vector3d> lidar = {Eigen::Vector3d(4.0, 5.0, 6.0)};
  const auto                         translated =
      planner::node::lidar_points_to_world(lidar, body, mount);
  ASSERT_EQ(translated.size(), 1U);
  EXPECT_NEAR(translated.front().x(), 5.0, 1e-12);
  EXPECT_NEAR(translated.front().y(), 7.0, 1e-12);
  EXPECT_NEAR(translated.front().z(), 9.0, 1e-12);

  body.position_world    = Eigen::Vector3d(0.0, -50.0, 1.5);
  body.orientation_world = Eigen::Quaterniond(
      Eigen::AngleAxisd(std::acos(-1.0) / 2.0, Eigen::Vector3d::UnitZ()));
  mount.position_body = Eigen::Vector3d(0.0, 0.0, 0.12);
  const auto yawed    = planner::node::lidar_points_to_world(
      {Eigen::Vector3d(1.0, 0.0, 0.0)}, body, mount);
  ASSERT_EQ(yawed.size(), 1U);
  EXPECT_NEAR(yawed.front().x(), 0.0, 1e-9);
  EXPECT_NEAR(yawed.front().y(), -49.0, 1e-9);
  EXPECT_NEAR(yawed.front().z(), 1.62, 1e-9);
}
