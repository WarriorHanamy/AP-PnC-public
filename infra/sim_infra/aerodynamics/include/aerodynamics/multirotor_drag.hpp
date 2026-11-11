/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2025 - <YEARS>, Sun Yat-sen University.
 * All rights reserved.
 */

#pragma once
#include <aerodynamics/aero_interface.hpp>

namespace aerodynamics
{

  /**
   * @brief Isotropic quadratic drag for a multirotor, through the centre of
   * mass.
   *
   * Body FLU. Force is -k |v| v. There is no wing, no lift, and no moment.
   * This is not the tailsitter lyu or phi model.
   */
  class MultirotorDrag : public AerodynamicsInterface
  {
   public:
    explicit MultirotorDrag(double drag_kg_per_m);
    bool getAeroWrench(const Eigen::Vector3d &body_speed,
                       Eigen::Vector3d       &aero_force_b,
                       Eigen::Vector3d &aero_moment_b, double &alpha,
                       double &beta) override;

   private:
    double drag_kg_per_m_;
  };

  class MultirotorDragCreator : public AerodynamicsCreator
  {
   public:
    bool createAerodynamics() override;
  };

}  // namespace aerodynamics
