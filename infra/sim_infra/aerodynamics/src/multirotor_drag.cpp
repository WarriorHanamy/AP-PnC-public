/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2025 - <YEARS>, Sun Yat-sen University.
 * All rights reserved.
 */

#include <yaml-cpp/yaml.h>

#include <aerodynamics/multirotor_drag.hpp>
#include <aerodynamics/project_paths.hpp>
#include <cmath>
#include <stdexcept>

namespace aerodynamics
{

  MultirotorDrag::MultirotorDrag(double drag_kg_per_m)
      : drag_kg_per_m_(drag_kg_per_m)
  {
    if (!std::isfinite(drag_kg_per_m_) || drag_kg_per_m_ < 0.0)
      throw std::invalid_argument(
          "multirotor drag coefficient must be finite and non-negative");
  }

  bool MultirotorDrag::getAeroWrench(const Eigen::Vector3d &body_speed,
                                     Eigen::Vector3d       &aero_force_b,
                                     Eigen::Vector3d       &aero_moment_b,
                                     double &alpha, double &beta)
  {
    std::tie(alpha, beta) = air_angles(body_speed);
    const double speed    = body_speed.norm();
    aero_force_b          = -drag_kg_per_m_ * speed * body_speed;
    aero_moment_b         = Eigen::Vector3d::Zero();
    return true;
  }

  bool MultirotorDragCreator::createAerodynamics()
  {
    const YAML::Node config =
        YAML::LoadFile(paths::aero_config("multirotor").string());
    if (config["body_frame"].as<std::string>() != "FLU")
      throw std::runtime_error("multirotor parameters require body_frame: FLU");
    const double drag = config["drag_kg_per_m"].as<double>();
    _instance         = std::make_unique<MultirotorDrag>(drag);
    return true;
  }

}  // namespace aerodynamics
