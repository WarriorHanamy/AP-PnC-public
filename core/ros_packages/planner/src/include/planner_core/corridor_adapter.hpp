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
#include <functional>
#include <memory>
#include <planner_core/map_query.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace planner::core
{

  using CorridorLog = std::function<void(const std::string &)>;

  /// Halfspace corridor parameters. CIRI fields are unused by the FIRI backend.
  struct CorridorSettings
  {
    double progress_m      = 6.0;
    double range_m         = 3.0;
    double robot_radius_m  = 0.2;
    int    ciri_iterations = 2;
  };

  /**
   * @brief Builds the `std::vector<Eigen::MatrixX4d>` corridor GCOPTER already
   * consumes. Each row is a plane `ax + by + cz + d <= 0`.
   */
  class CorridorAdapter
  {
   public:
    virtual ~CorridorAdapter() = default;

    virtual bool build(const MapQuery &map, const Eigen::Vector3d &start,
                       const Eigen::Vector3d         &goal,
                       std::vector<Eigen::MatrixX4d> &hPolys,
                       std::vector<Eigen::Vector3d>  &route) const = 0;

    virtual std::string_view backend() const = 0;
  };

  std::unique_ptr<CorridorAdapter> make_corridor_adapter(
      const std::string &backend, const CorridorSettings &settings,
      CorridorLog log);

  /// `firi` is always available. `ciri` needs an external ROS-free library.
  bool corridor_backend_available(const std::string &backend);

}  // namespace planner::core
