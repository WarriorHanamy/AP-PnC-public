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
#include <gcopter/sfc_gen.hpp>
#include <planner_core/corridor_adapter.hpp>
#include <planner_core/grid_astar.hpp>
#include <stdexcept>
#include <utility>

#if defined(AP_PNC_HAS_CIRI)
#include <ciri/corridor.hpp>
#endif

namespace planner::core
{

  namespace
  {

    class FiriCorridor final : public CorridorAdapter
    {
     public:
      FiriCorridor(CorridorSettings settings, CorridorLog log)
          : settings_(settings), log_(std::move(log))
      {
      }

      std::string_view backend() const override { return "firi"; }

      bool build(const MapQuery &map, const Eigen::Vector3d &start,
                 const Eigen::Vector3d         &goal,
                 std::vector<Eigen::MatrixX4d> &hPolys,
                 std::vector<Eigen::Vector3d>  &route) const override
      {
        const auto *voxels = dynamic_cast<const VoxelMapAdapter *>(&map);
        if (voxels == nullptr)
        {
          if (log_)
          {
            log_("FIRI corridor requires the voxel map adapter");
          }
          return false;
        }
        if (!gridAstar(start, goal, voxels->map(), route))
        {
          if (log_)
          {
            log_("Corridor planning: no collision-free route found");
          }
          return false;
        }

        std::vector<Eigen::Vector3d> surf;
        map.surface(surf);
        if (surf.empty())
        {
          // Nothing occupied anywhere: skip FIRI (which indexes surf[0]
          // unconditionally) and use one corridor spanning the whole map.
          Eigen::MatrixX4d box(6, 4);
          box.setZero();
          box(0, 0) = 1.0;
          box(0, 3) = -map.corner().x();
          box(1, 0) = -1.0;
          box(1, 3) = map.origin().x();
          box(2, 1) = 1.0;
          box(2, 3) = -map.corner().y();
          box(3, 1) = -1.0;
          box(3, 3) = map.origin().y();
          box(4, 2) = 1.0;
          box(4, 3) = -map.corner().z();
          box(5, 2) = -1.0;
          box(5, 3) = map.origin().z();
          hPolys    = {box};
          return true;
        }

        sfc_gen::convexCover(route, surf, map.origin(), map.corner(),
                             settings_.progress_m, settings_.range_m, hPolys);
        sfc_gen::shortCut(hPolys);
        if (hPolys.empty())
        {
          if (log_)
          {
            log_("Corridor planning: FIRI produced no polytopes");
          }
          return false;
        }
        return true;
      }

     private:
      CorridorSettings settings_;
      CorridorLog      log_;
    };

#if defined(AP_PNC_HAS_CIRI)
    class CiriCorridor final : public CorridorAdapter
    {
     public:
      CiriCorridor(CorridorSettings settings, CorridorLog log)
          : settings_(settings), log_(std::move(log))
      {
      }

      std::string_view backend() const override { return "ciri"; }

      bool build(const MapQuery &map, const Eigen::Vector3d &start,
                 const Eigen::Vector3d         &goal,
                 std::vector<Eigen::MatrixX4d> &hPolys,
                 std::vector<Eigen::Vector3d>  &route) const override
      {
        const auto *voxels = dynamic_cast<const VoxelMapAdapter *>(&map);
        if (voxels == nullptr)
        {
          if (log_)
          {
            log_("CIRI corridor requires the voxel map adapter in this phase");
          }
          return false;
        }
        if (!gridAstar(start, goal, voxels->map(), route))
        {
          if (log_)
          {
            log_("Corridor planning: no collision-free route found");
          }
          return false;
        }
        std::vector<Eigen::Vector3d> surf;
        map.surface(surf);
        Eigen::Matrix3Xd obstacles(3, static_cast<Eigen::Index>(surf.size()));
        for (Eigen::Index column = 0; column < obstacles.cols(); ++column)
        {
          obstacles.col(column) = surf[static_cast<std::size_t>(column)];
        }
        hPolys.clear();
        hPolys.reserve(route.size());
        for (std::size_t i = 1; i < route.size(); ++i)
        {
          const Eigen::Vector3d      &a = route[i - 1];
          const Eigen::Vector3d      &b = route[i];
          Eigen::Matrix<double, 6, 4> bounds =
              Eigen::Matrix<double, 6, 4>::Zero();
          bounds(0, 0)       = 1.0;
          bounds(1, 0)       = -1.0;
          bounds(2, 1)       = 1.0;
          bounds(3, 1)       = -1.0;
          bounds(4, 2)       = 1.0;
          bounds(5, 2)       = -1.0;
          const double range = settings_.range_m;
          bounds(0, 3) =
              -std::min(std::max(a.x(), b.x()) + range, map.corner().x());
          bounds(1, 3) =
              std::max(std::min(a.x(), b.x()) - range, map.origin().x());
          bounds(2, 3) =
              -std::min(std::max(a.y(), b.y()) + range, map.corner().y());
          bounds(3, 3) =
              std::max(std::min(a.y(), b.y()) - range, map.origin().y());
          bounds(4, 3) =
              -std::min(std::max(a.z(), b.z()) + range, map.corner().z());
          bounds(5, 3) =
              std::max(std::min(a.z(), b.z()) - range, map.origin().z());
          Eigen::MatrixX4d polytope;
          if (!ciri::decompose_segment(bounds, obstacles, a, b,
                                       settings_.robot_radius_m,
                                       settings_.ciri_iterations, polytope) ||
              polytope.cols() != 4 || polytope.rows() < 4)
          {
            if (log_)
            {
              log_("CIRI produced no polytope");
            }
            return false;
          }
          hPolys.push_back(std::move(polytope));
        }
        return !hPolys.empty();
      }

     private:
      CorridorSettings settings_;
      CorridorLog      log_;
    };
#endif

  }  // namespace

  std::unique_ptr<CorridorAdapter> make_corridor_adapter(
      const std::string &backend, const CorridorSettings &settings,
      CorridorLog log)
  {
    if (backend == "firi")
    {
      return std::make_unique<FiriCorridor>(settings, std::move(log));
    }
#if defined(AP_PNC_HAS_CIRI)
    if (backend == "ciri")
    {
      return std::make_unique<CiriCorridor>(settings, std::move(log));
    }
#endif
    throw std::invalid_argument(
        "corridor backend '" + backend +
        "' is not linked in this build; the default backend is firi");
  }

  bool corridor_backend_available(const std::string &backend)
  {
    if (backend == "firi")
    {
      return true;
    }
#if defined(AP_PNC_HAS_CIRI)
    if (backend == "ciri")
    {
      return true;
    }
#endif
    return false;
  }

}  // namespace planner::core
