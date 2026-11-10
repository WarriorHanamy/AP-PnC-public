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

#include <algorithm>
#include <cmath>
#include <planner_core/waypoint_route.hpp>
#include <stdexcept>

namespace planner::core
{

  double waypoint_route_length(const std::vector<Eigen::Vector3d> &waypoints)
  {
    if (waypoints.size() < 2)
    {
      throw std::invalid_argument("a waypoint route needs at least two points");
    }
    double length = 0.0;
    for (std::size_t i = 1; i < waypoints.size(); ++i)
    {
      if (!waypoints[i - 1].allFinite() || !waypoints[i].allFinite())
      {
        throw std::invalid_argument("waypoint coordinates must be finite");
      }
      const double segment = (waypoints[i] - waypoints[i - 1]).norm();
      if (!(segment > 1e-6))
      {
        throw std::invalid_argument("consecutive waypoints must be distinct");
      }
      length += segment;
    }
    return length;
  }

  std::vector<basic_trajectories::Mat3x4> sample_waypoint_route(
      const std::vector<Eigen::Vector3d> &waypoints, int piece_n,
      double speed_m_s)
  {
    if (piece_n < 1)
    {
      throw std::invalid_argument("piece_n must be positive");
    }
    if (!std::isfinite(speed_m_s) || speed_m_s <= 0.0)
    {
      throw std::invalid_argument("waypoint speed must be finite and positive");
    }
    std::vector<double> cumulative(waypoints.size(), 0.0);
    for (std::size_t i = 1; i < waypoints.size(); ++i)
    {
      cumulative[i] =
          cumulative[i - 1] + (waypoints[i] - waypoints[i - 1]).norm();
    }
    const double length = waypoint_route_length(waypoints);

    std::vector<basic_trajectories::Mat3x4> samples;
    samples.reserve(static_cast<std::size_t>(piece_n) + 1);
    for (int i = 0; i <= piece_n; ++i)
    {
      const double distance =
          length * static_cast<double>(i) / static_cast<double>(piece_n);
      std::size_t segment = 1;
      while (segment + 1 < cumulative.size() && cumulative[segment] < distance)
      {
        ++segment;
      }
      const double span = cumulative[segment] - cumulative[segment - 1];
      const double alpha =
          std::clamp((distance - cumulative[segment - 1]) / span, 0.0, 1.0);
      const Eigen::Vector3d tangent =
          (waypoints[segment] - waypoints[segment - 1]) / span;
      basic_trajectories::Mat3x4 point = basic_trajectories::Mat3x4::Zero();
      point.col(0) =
          (1.0 - alpha) * waypoints[segment - 1] + alpha * waypoints[segment];
      point.col(1) = tangent * speed_m_s;
      samples.push_back(point);
    }
    return samples;
  }

}  // namespace planner::core
