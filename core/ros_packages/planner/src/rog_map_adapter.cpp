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

#include <chrono>
#include <planner_core/map_query.hpp>
#include <stdexcept>
#include <utility>

#if defined(AP_PNC_HAS_ROG_MAP)
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rog_map/rog_map.h>
#endif

namespace planner::core
{

  bool map_backend_available(const std::string &backend)
  {
    if (backend == "voxel")
    {
      return true;
    }
#if defined(AP_PNC_HAS_ROG_MAP)
    if (backend == "rog")
    {
      return true;
    }
#endif
    return false;
  }

#if defined(AP_PNC_HAS_ROG_MAP)

  namespace
  {

    class LinkedRogMap final : public rog_map::ROGMap, public MapUpdater
    {
     public:
      explicit LinkedRogMap(const std::string &config_yaml)
      {
        if (config_yaml.empty())
        {
          throw std::invalid_argument(
              "ROG-Map config path is required when map_backend is rog");
        }
        cfg_                 = rog_map::Config(config_yaml);
        cfg_.ros_callback_en = false;
        init();
        // updateMap() refuses the call while the ROS subscription is enabled.
        cfg_.ros_callback_en = false;
      }

      const double getSystemWalltimeNow() override
      {
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration<double>(now).count();
      }

      bool available() const override { return true; }

      bool occupied(const Eigen::Vector3d &position_world) const override
      {
        return isOccupied(position_world);
      }

      void update(const std::vector<Eigen::Vector3d> &points_world,
                  const Eigen::Vector3d              &position_world,
                  const Eigen::Quaterniond &orientation_world) override
      {
        cfg_.ros_callback_en = false;
        rog_map::PointCloud cloud;
        cloud.points.reserve(points_world.size());
        for (const Eigen::Vector3d &point : points_world)
        {
          pcl::PointXYZI sample;
          sample.x         = static_cast<float>(point.x());
          sample.y         = static_cast<float>(point.y());
          sample.z         = static_cast<float>(point.z());
          sample.intensity = 0.F;
          cloud.points.push_back(sample);
        }
        cloud.width    = static_cast<std::uint32_t>(cloud.points.size());
        cloud.height   = 1;
        cloud.is_dense = true;
        super_utils::Pose pose;
        pose.first  = position_world;
        pose.second = orientation_world;
        updateMap(cloud, pose);
      }
    };

  }  // namespace

  std::unique_ptr<MapUpdater> make_rog_map_adapter(
      const std::string &config_yaml)
  {
    return std::make_unique<LinkedRogMap>(config_yaml);
  }

#else

  namespace
  {

    class UnavailableRogMap final : public MapUpdater
    {
     public:
      bool available() const override { return false; }

      bool occupied(const Eigen::Vector3d &) const override { return true; }

      void update(const std::vector<Eigen::Vector3d> &, const Eigen::Vector3d &,
                  const Eigen::Quaterniond &) override
      {
        throw std::logic_error(
            "ROG-Map is not linked; build with -DAP_PNC_WITH_ROG_MAP=ON and "
            "an external prefix");
      }
    };

  }  // namespace

  std::unique_ptr<MapUpdater> make_rog_map_adapter(const std::string &)
  {
    return std::make_unique<UnavailableRogMap>();
  }

#endif

}  // namespace planner::core
