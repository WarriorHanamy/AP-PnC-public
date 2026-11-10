#pragma once

#include <functional>

#include "simple_sim/config.hpp"
#include "simple_sim/types.hpp"

namespace simple_sim
{

  // Evaluated at every RK4 stage. Must be deterministic and side-effect free.
  using ForceModel      = std::function<Wrench(const State &)>;
  using Actuation       = std::function<Wrench(const State &, const Command &)>;
  using MotorDerivative = std::function<Eigen::Vector4d(const State &)>;
  // d(collective specific force)/dt for the ideal inner loop.
  using ThrustDerivative =
      std::function<double(const State &, const Command &)>;

  class Plant
  {
   public:
    // `external` is an optional body-FLU wrench. The quadrotor plant passes
    // none: rotor force and moment are the only applied loads.
    explicit Plant(PlantConfig config, ForceModel external = {},
                   MotorDerivative motors = {}, ThrustDerivative thrust = {});
    State    advance(const State &state, const Command &command, double dt,
                     const Actuation &actuation) const;
    Feedback feedback(const State &state, const Command &command,
                      const Actuation &actuation) const;
    const PlantConfig &config() const { return config_; }

   private:
    PlantConfig      config_;
    ForceModel       external_;
    MotorDerivative  motors_;
    ThrustDerivative thrust_;
    Wrench           external_wrench(const State &state) const;
  };

}  // namespace simple_sim
