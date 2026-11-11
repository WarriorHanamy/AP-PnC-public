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
#include <limits>
#include <planner_core/ciri.hpp>

namespace planner::core
{
  namespace
  {

    struct SeedFrame
    {
      Eigen::Vector3d center   = Eigen::Vector3d::Zero();
      Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
      double          length   = 0.0;
    };

    bool frame_on_segment(const Eigen::Vector3d &start,
                          const Eigen::Vector3d &end, SeedFrame &frame)
    {
      const Eigen::Vector3d chord = end - start;
      frame.length                = chord.norm();
      if (!(frame.length > 1e-3) || !start.allFinite() || !end.allFinite())
      {
        return false;
      }
      frame.center          = 0.5 * (start + end);
      frame.rotation.col(0) = chord / frame.length;
      frame.rotation.col(1) = frame.rotation.col(0).unitOrthogonal();
      frame.rotation.col(2) =
          frame.rotation.col(0).cross(frame.rotation.col(1));
      return frame.rotation.allFinite();
    }

    double mahalanobis2(const SeedFrame &frame, const Eigen::Vector3d &point,
                        double longitudinal, double transverse)
    {
      const Eigen::Vector3d local =
          frame.rotation.transpose() * (point - frame.center);
      const double x = local.x() / longitudinal;
      const double y = local.y() / transverse;
      const double z = local.z() / transverse;
      return x * x + y * y + z * z;
    }

    bool ellipsoid_clear(const SeedFrame                    &frame,
                         const std::vector<Eigen::Vector3d> &obstacles,
                         double longitudinal, double transverse)
    {
      for (const Eigen::Vector3d &point : obstacles)
      {
        if (mahalanobis2(frame, point, longitudinal, transverse) < 1.0)
        {
          return false;
        }
      }
      return true;
    }

    bool halfspace_contains(const Eigen::MatrixX4d &planes,
                            const Eigen::Vector3d &point, double slack)
    {
      if (planes.rows() == 0)
      {
        return false;
      }
      const Eigen::Vector4d homogeneous(point.x(), point.y(), point.z(), 1.0);
      return (planes * homogeneous).maxCoeff() <= slack;
    }

    Eigen::MatrixX4d supporting_planes(
        const SeedFrame &frame, const std::vector<Eigen::Vector3d> &obstacles,
        const Eigen::MatrixX4d &bounds, double longitudinal, double transverse,
        double robot_radius_m)
    {
      const Eigen::Vector3d start =
          frame.center - frame.rotation.col(0) * (0.5 * frame.length);
      const Eigen::Vector3d end =
          frame.center + frame.rotation.col(0) * (0.5 * frame.length);
      std::vector<Eigen::RowVector4d> planes;
      planes.reserve(static_cast<std::size_t>(bounds.rows()) +
                     obstacles.size());
      for (Eigen::Index row = 0; row < bounds.rows(); ++row)
      {
        Eigen::RowVector4d plane = bounds.row(row);
        const double       norm  = plane.head<3>().norm();
        if (norm > 1e-9)
        {
          plane /= norm;
        }
        planes.push_back(plane);
      }

      const Eigen::Vector3d axes(longitudinal, transverse, transverse);
      for (const Eigen::Vector3d &point : obstacles)
      {
        const double metric =
            mahalanobis2(frame, point, longitudinal, transverse);
        if (!(metric > 1.0) || metric > 16.0)
        {
          continue;
        }
        const Eigen::Vector3d local =
            frame.rotation.transpose() * (point - frame.center);
        const Eigen::Vector3d grad =
            frame.rotation * (local.array() / axes.array().square()).matrix();
        const double grad_norm = grad.norm();
        if (!(grad_norm > 1e-9))
        {
          continue;
        }
        const Eigen::Vector3d normal = grad / grad_norm;
        Eigen::Matrix3d       shape  = frame.rotation *
                                axes.array().square().matrix().asDiagonal() *
                                frame.rotation.transpose();
        const double support = std::sqrt(normal.transpose() * shape * normal);
        Eigen::RowVector4d plane;
        plane << normal.transpose(), -normal.dot(frame.center) - support;
        plane(3) += robot_radius_m;
        const auto contains = [&plane](const Eigen::Vector3d &point)
        { return plane.head<3>().dot(point) + plane(3) <= 1e-5; };
        if (contains(start) && contains(end) && contains(frame.center))
        {
          planes.push_back(plane);
        }
      }

      std::vector<Eigen::RowVector4d> kept;
      kept.reserve(planes.size());
      for (const Eigen::RowVector4d &plane : planes)
      {
        bool merged = false;
        for (Eigen::RowVector4d &prior : kept)
        {
          if (prior.head<3>().dot(plane.head<3>()) > 0.98)
          {
            if (plane(3) > prior(3))
            {
              prior = plane;
            }
            merged = true;
            break;
          }
        }
        if (!merged)
        {
          kept.push_back(plane);
        }
      }
      if (kept.size() > 30)
      {
        const Eigen::Vector3d probe = frame.center;
        std::sort(kept.begin(), kept.end(),
                  [&probe](const Eigen::RowVector4d &lhs,
                           const Eigen::RowVector4d &rhs)
                  {
                    return lhs.head<3>().dot(probe) + lhs(3) >
                           rhs.head<3>().dot(probe) + rhs(3);
                  });
        kept.resize(30);
      }
      Eigen::MatrixX4d polytope(static_cast<Eigen::Index>(kept.size()), 4);
      for (Eigen::Index row = 0; row < polytope.rows(); ++row)
      {
        polytope.row(row) = kept[static_cast<std::size_t>(row)];
      }
      return polytope;
    }

  }  // namespace

  bool ciri_decompose(const Eigen::MatrixX4d             &bounds,
                      const std::vector<Eigen::Vector3d> &obstacles,
                      const Eigen::Vector3d              &segment_start,
                      const Eigen::Vector3d &segment_end, double robot_radius_m,
                      int iterations, Eigen::MatrixX4d &polytope)
  {
    SeedFrame frame;
    if (!frame_on_segment(segment_start, segment_end, frame) ||
        bounds.rows() < 4 || bounds.cols() != 4 || iterations < 1 ||
        !(robot_radius_m >= 0.0))
    {
      return false;
    }
    std::vector<Eigen::Vector3d> local;
    local.reserve(obstacles.size());
    for (const Eigen::Vector3d &point : obstacles)
    {
      if (point.allFinite() && halfspace_contains(bounds, point, 1e-4))
      {
        local.push_back(point);
      }
    }
    const double longitudinal = 0.5 * frame.length + 1e-2;
    // Start thinner than the robot. A* chords can pass close to the dilated
    // surface; the inflation search finds the largest ellipsoid that still
    // excludes those points, and the plane offset is limited to that room.
    double transverse = 0.02;
    if (!ellipsoid_clear(frame, local, longitudinal, transverse))
    {
      return false;
    }
    double high = std::max(transverse, 8.0);
    if (ellipsoid_clear(frame, local, longitudinal, high))
    {
      transverse = high;
    }
    else
    {
      double low = transverse;
      for (int step = 0; step < 24; ++step)
      {
        const double mid = 0.5 * (low + high);
        if (ellipsoid_clear(frame, local, longitudinal, mid))
        {
          low = mid;
        }
        else
        {
          high = mid;
        }
      }
      transverse = low;
    }
    const double shrink = std::min(robot_radius_m, 0.5 * transverse);

    Eigen::MatrixX4d planes(0, 4);
    const int        rounds = std::max(1, iterations);
    for (int round = 0; round < rounds; ++round)
    {
      planes = supporting_planes(frame, local, bounds, longitudinal, transverse,
                                 shrink);
      if (!halfspace_contains(planes, segment_start, 1e-5) ||
          !halfspace_contains(planes, segment_end, 1e-5))
      {
        return false;
      }
      // A later round uses a slightly tighter ellipsoid so an outlier plane
      // from the first tangent set can be replaced by the next-nearest points.
      transverse =
          std::max(0.05, transverse * (round + 1 == rounds ? 1.0 : 0.85));
    }
    if (planes.rows() < 4 || !planes.allFinite())
    {
      return false;
    }
    polytope = std::move(planes);
    return true;
  }

}  // namespace planner::core
