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

#pragma once

#include <Eigen/Dense>
#include <basic_trajectories/trajectory.hpp>
#include <vector>

namespace planner::core
{

  /// Arc length of a polyline [m]. Consecutive points must be distinct.
  double waypoint_route_length(const std::vector<Eigen::Vector3d> &waypoints);

  /**
   * @brief Sample a waypoint polyline the same way `basic_traj` is sampled.
   *
   * Returns `piece_n + 1` points. Column 0 is position, column 1 is the
   * outgoing tangent times `speed_m_s`, and acceleration/jerk columns are zero.
   * This is the mission route for a configured waypoint task. It does not add
   * a second optimizer and it does not replace the tailsitter flatness map.
   */
  std::vector<basic_trajectories::Mat3x4> sample_waypoint_route(
      const std::vector<Eigen::Vector3d> &waypoints, int piece_n,
      double speed_m_s);

}  // namespace planner::core
