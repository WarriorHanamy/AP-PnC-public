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

#include <Eigen/Geometry>
#include <stdexcept>
#include <string>
#include <vector>

namespace planner::node
{

  /**
   * @brief Body-from-lidar mount, matching lidar_sim's sensor pose.
   *
   * `orientation_body` rotates lidar-frame vectors into the body frame.
   */
  struct LidarMount
  {
    Eigen::Vector3d    position_body    = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation_body = Eigen::Quaterniond::Identity();
  };

  /// World-from-body pose. `orientation_world` is body FLU expressed in ENU.
  struct BodyPose
  {
    Eigen::Vector3d    position_world    = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation_world = Eigen::Quaterniond::Identity();
  };

  inline void require_rotation(const Eigen::Quaterniond &rotation,
                               const char               *what)
  {
    if (!rotation.coeffs().allFinite() ||
        std::abs(rotation.norm() - 1.0) > 1e-5)
    {
      throw std::invalid_argument(std::string(what) +
                                  " quaternion must be finite and unit length");
    }
  }

  /**
   * @brief Transform lidar-frame points into world ENU.
   *
   * `p_body = R_body_lidar * p_lidar + t_body_lidar`, then
   * `p_world = R_world_body * p_body + t_world_body`. Callers must not pass
   * the lidar-frame cloud to a map that ignores TF.
   */
  inline std::vector<Eigen::Vector3d> lidar_points_to_world(
      const std::vector<Eigen::Vector3d> &points_lidar, const BodyPose &body,
      const LidarMount &mount)
  {
    if (!body.position_world.allFinite() || !mount.position_body.allFinite())
    {
      throw std::invalid_argument(
          "lidar transform translations must be finite");
    }
    require_rotation(body.orientation_world, "body");
    require_rotation(mount.orientation_body, "lidar mount");
    const Eigen::Quaterniond world_from_body =
        body.orientation_world.normalized();
    const Eigen::Quaterniond body_from_lidar =
        mount.orientation_body.normalized();
    std::vector<Eigen::Vector3d> world;
    world.reserve(points_lidar.size());
    for (const Eigen::Vector3d &point : points_lidar)
    {
      if (!point.allFinite())
      {
        throw std::invalid_argument("lidar point must be finite");
      }
      const Eigen::Vector3d body_point =
          body_from_lidar * point + mount.position_body;
      world.push_back(world_from_body * body_point + body.position_world);
    }
    return world;
  }

}  // namespace planner::node
