#pragma once

#include <array>
#include <memory>
#include <vector>

#include "nmpc_solver.h"

namespace nmpc
{
  struct TrackingConfig
  {
    double horizon_s               = 1.0;
    double traj_res_s              = 0.1;
    double ctrl_frq                = 50.0;
    double initial_specific_thrust = 9.81;  // initialization only
    void   validate() const;
  };

  using CostReference = std::array<double, NMPC_NY>;  // NMPC objective layout,
                                                      // not a trajectory type
  CostReference cost_reference(const std::array<double, 3> &position,
                               const std::array<double, 3> &velocity,
                               const std::array<double, 3> &yb);

  struct TrackingResult
  {
    std::array<double, 3> rates{};
    double                specific_force = 0.0;
    int                   status         = 0;
  };

  struct TrackingTrace
  {
    std::vector<CostReference>   references;
    std::array<double, NMPC_NYN> terminal{};
    std::array<double, NMPC_NX>  state{};
    int                          status    = -1;
    bool                         submitted = false;
    std::array<double, NMPC_NU>  command_derivative{};
    std::array<double, NMPC_NU>  integrated_command{};
    bool                         accepted = false;
  };

  // ROS-free quadrotor controller shared by the flight node and simple_sim.
  class TrackingController
  {
   public:
    explicit TrackingController(TrackingConfig config);
    void
    reset();  // destroys/recreates the solver, including all warm-start state
    void           reset(const std::array<double, NMPC_NU> &initial_command);
    TrackingResult compute(const std::array<double, 13>     &state,
                           const std::vector<CostReference> &horizon);
    const TrackingConfig &config() const { return config_; }
    const TrackingTrace  &trace() const { return trace_; }

   private:
    TrackingConfig                                                 config_;
    std::unique_ptr<nmpc_solver_t, decltype(&nmpc_solver_destroy)> solver_{
        nullptr, nmpc_solver_destroy};
    TrackingTrace               trace_;
    std::array<double, NMPC_NU> previous_command_{};
  };

}  // namespace nmpc
