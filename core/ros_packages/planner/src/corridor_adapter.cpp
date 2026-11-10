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
#include <planner_core/ciri.hpp>
#include <planner_core/corridor_adapter.hpp>
#include <planner_core/grid_astar.hpp>
#include <stdexcept>
#include <utility>

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
            log_("CIRI corridor requires the voxel map adapter");
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

        std::vector<Eigen::Vector3d> seeds{route.front()};
        for (std::size_t index = 1; index < route.size(); ++index)
        {
          const Eigen::Vector3d from = route[index - 1];
          const Eigen::Vector3d edge = route[index] - from;
          const double          span = edge.norm();
          if (!(span > 1e-3))
          {
            continue;
          }
          const int pieces = std::max(
              1, static_cast<int>(std::ceil(span / settings_.progress_m)));
          for (int piece = 1; piece <= pieces; ++piece)
          {
            seeds.push_back(from +
                            edge * (static_cast<double>(piece) / pieces));
          }
        }

        hPolys.clear();
        for (std::size_t index = 1; index < seeds.size(); ++index)
        {
          const Eigen::Vector3d      &a = seeds[index - 1];
          const Eigen::Vector3d      &b = seeds[index];
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
          std::vector<Eigen::Vector3d> nearby;
          nearby.reserve(surf.size());
          for (const Eigen::Vector3d &point : surf)
          {
            const Eigen::Vector3d chord = b - a;
            const double          denom = chord.squaredNorm();
            const double          time =
                denom < 1e-12
                             ? 0.0
                             : std::clamp((point - a).dot(chord) / denom, 0.0, 1.0);
            if ((point - (a + time * chord)).norm() <= range)
            {
              nearby.push_back(point);
            }
          }
          Eigen::MatrixX4d polytope;
          if (!ciri_decompose(bounds, nearby, a, b, settings_.robot_radius_m,
                              settings_.ciri_iterations, polytope))
          {
            const Eigen::Vector3d mid = 0.5 * (a + b);
            Eigen::MatrixX4d      left;
            Eigen::MatrixX4d      right;
            if ((b - a).norm() > 0.8 &&
                ciri_decompose(bounds, nearby, a, mid, settings_.robot_radius_m,
                               settings_.ciri_iterations, left) &&
                ciri_decompose(bounds, nearby, mid, b, settings_.robot_radius_m,
                               settings_.ciri_iterations, right))
            {
              hPolys.push_back(std::move(left));
              hPolys.push_back(std::move(right));
              continue;
            }
            if (log_)
            {
              log_("CIRI produced no polytope");
            }
            return false;
          }
          hPolys.push_back(std::move(polytope));
        }
        sfc_gen::shortCut(hPolys);
        if (hPolys.empty())
        {
          if (log_)
          {
            log_("CIRI produced no polytopes");
          }
          return false;
        }
        return true;
      }

     private:
      CorridorSettings settings_;
      CorridorLog      log_;
    };

  }  // namespace

  std::unique_ptr<CorridorAdapter> make_corridor_adapter(
      const std::string &backend, const CorridorSettings &settings,
      CorridorLog log)
  {
    if (backend == "firi")
    {
      return std::make_unique<FiriCorridor>(settings, std::move(log));
    }
    if (backend == "ciri")
    {
      return std::make_unique<CiriCorridor>(settings, std::move(log));
    }
    throw std::invalid_argument("corridor backend '" + backend +
                                "' is unknown; choose firi or ciri");
  }

  bool corridor_backend_available(const std::string &backend)
  {
    return backend == "firi" || backend == "ciri";
  }

}  // namespace planner::core
