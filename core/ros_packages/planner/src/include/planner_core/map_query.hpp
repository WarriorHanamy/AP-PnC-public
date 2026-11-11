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
#include <gcopter/voxel_map.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace planner::core
{

  /**
   * @brief ROS-free occupancy query used by corridor construction.
   *
   * Corridor consumers see halfspace polytopes (`MatrixX4d`, `ax+by+cz+d <=
   * 0`). They do not see a third-party polytope type. The default
   * implementation is the existing voxel map. An external ROG-Map core is
   * optional and is not part of this source tree.
   */
  class MapQuery
  {
   public:
    virtual ~MapQuery() = default;

    /// True when `position` is occupied or outside the map, world ENU [m].
    virtual bool occupied(const Eigen::Vector3d &position) const = 0;

    virtual Eigen::Vector3d origin() const = 0;
    virtual Eigen::Vector3d corner() const = 0;

    /// Occupied surface samples in world ENU. Cleared, then filled.
    virtual void surface(std::vector<Eigen::Vector3d> &points) const = 0;

    virtual std::string_view backend() const = 0;
  };

  /// Adapts the existing `voxel_map::VoxelMap` (map_sfc / FIRI) as a MapQuery.
  class VoxelMapAdapter final : public MapQuery
  {
   public:
    explicit VoxelMapAdapter(const voxel_map::VoxelMap &map) : map_(map) {}

    bool occupied(const Eigen::Vector3d &position) const override
    {
      return map_.query(position);
    }

    Eigen::Vector3d origin() const override { return map_.getOrigin(); }

    Eigen::Vector3d corner() const override { return map_.getCorner(); }

    void surface(std::vector<Eigen::Vector3d> &points) const override
    {
      points.clear();
      map_.getSurf(points);
    }

    std::string_view backend() const override { return "voxel"; }

    const voxel_map::VoxelMap &map() const { return map_; }

   private:
    const voxel_map::VoxelMap &map_;
  };

  /**
   * @brief Optional external map that can ingest an already world-frame cloud.
   *
   * The default build returns an adapter whose `available()` is false. The
   * linked implementation calls ROG-Map's `updateMap` only when an external
   * prefix was provided at configure time, and it forces `ros_callback` off.
   * No SUPER / ROG-Map source is compiled into the default build.
   */
  class MapUpdater
  {
   public:
    virtual ~MapUpdater() = default;

    virtual bool available() const = 0;

    virtual void update(const std::vector<Eigen::Vector3d> &points_world,
                        const Eigen::Vector3d              &position_world,
                        const Eigen::Quaterniond &orientation_world) = 0;

    virtual bool occupied(const Eigen::Vector3d &position_world) const = 0;
  };

  std::unique_ptr<MapUpdater> make_rog_map_adapter(
      const std::string &config_yaml);

  /// `voxel` is always available. `rog` is available only in an opt-in build.
  bool map_backend_available(const std::string &backend);

}  // namespace planner::core
