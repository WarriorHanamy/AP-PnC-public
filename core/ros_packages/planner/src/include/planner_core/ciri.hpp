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
#include <vector>

namespace planner::core
{

  /**
   * @brief One CIRI polytope around a seed segment.
   *
   * Convex iterative regional inflation: grow an ellipsoid around the segment
   * until obstacle points stop it, then take the supporting halfspaces and
   * pull them in by the robot radius. This is original code. It does not
   * include SUPER's CIRI sources.
   *
   * `bounds` and the result use rows `ax + by + cz + d <= 0`.
   * Returns false when the seed cannot be separated from the obstacles.
   */
  bool ciri_decompose(const Eigen::MatrixX4d             &bounds,
                      const std::vector<Eigen::Vector3d> &obstacles,
                      const Eigen::Vector3d              &segment_start,
                      const Eigen::Vector3d &segment_end, double robot_radius_m,
                      int iterations, Eigen::MatrixX4d &polytope);

}  // namespace planner::core
