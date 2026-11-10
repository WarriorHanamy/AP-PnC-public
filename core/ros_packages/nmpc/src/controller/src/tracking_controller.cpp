#include "nmpc_controller/tracking_controller.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nmpc
{
  void TrackingConfig::validate() const
  {
    if (!std::isfinite(horizon_s) || horizon_s <= 0.0 ||
        !std::isfinite(traj_res_s) || traj_res_s <= 0.0 ||
        !std::isfinite(ctrl_frq) || ctrl_frq <= 0.0 ||
        1.0 / ctrl_frq > traj_res_s + 1e-9 ||
        !std::isfinite(initial_specific_thrust) ||
        initial_specific_thrust <= 0.0 ||
        std::abs(horizon_s / traj_res_s - std::round(horizon_s / traj_res_s)) >
            1e-9)
      throw std::invalid_argument("invalid NMPC horizon/control configuration");
  }
  CostReference cost_reference(const std::array<double, 3> &p,
                               const std::array<double, 3> &v,
                               const std::array<double, 3> &yb)
  {
    CostReference y{};
    std::copy(p.begin(), p.end(), y.begin());
    std::copy(v.begin(), v.end(), y.begin() + 3);
    std::copy(yb.begin(), yb.end(), y.begin() + 6);
    // y[9:13] = 0: command-derivative regularization, no absolute command
    // target.
    return y;
  }
  TrackingController::TrackingController(TrackingConfig config)
      : config_(config)
  {
    config_.validate();
    if (nmpc_solver_abi_version() != NMPC_SOLVER_ABI)
      throw std::runtime_error("stale NMPC bundle ABI; regenerate and relink");
    reset();
  }
  void TrackingController::reset()
  {
    reset({config_.initial_specific_thrust, 0.0, 0.0, 0.0});
  }
  void TrackingController::reset(
      const std::array<double, NMPC_NU> &initial_command)
  {
    for (double value : initial_command)
      if (!std::isfinite(value))
        throw std::invalid_argument("invalid initial NMPC command");
    solver_.reset(nmpc_solver_create());
    if (!solver_)
      throw std::runtime_error("cannot create NMPC solver");
    const int n = nmpc_solver_horizon(solver_.get());
    if (std::abs(n * config_.traj_res_s - config_.horizon_s) > 1e-9)
      throw std::runtime_error(
          "NMPC bundle horizon differs from nmpc.yaml; regenerate bundle");
    for (int i = 0; i < n; ++i)
      if (std::abs(nmpc_solver_stage_dt(solver_.get(), i) -
                   config_.traj_res_s) > 1e-9)
        throw std::runtime_error(
            "NMPC bundle grid differs from nmpc.yaml; regenerate bundle");
    previous_command_ = initial_command;
    trace_            = {};
  }
  TrackingResult TrackingController::compute(
      const std::array<double, 13>     &state,
      const std::vector<CostReference> &horizon)
  {
    trace_      = {};
    const int n = nmpc_solver_horizon(solver_.get());
    if (horizon.size() != static_cast<std::size_t>(n + 1))
      throw std::invalid_argument("NMPC requires exactly N+1 reference points");
    for (double value : state)
      if (!std::isfinite(value))
        throw std::runtime_error("non-finite NMPC state");
    for (const auto &y : horizon)
    {
      for (double value : y)
        if (!std::isfinite(value))
          throw std::runtime_error("non-finite NMPC reference");
      for (int i = 9; i < NMPC_NY; ++i)
        if (y[i] != 0.0)
          throw std::invalid_argument(
              "command derivative reference must be zero");
    }
    const double qw = state[9], qx = state[10], qy = state[11], qz = state[12];
    if (std::abs(qw * qw + qx * qx + qy * qy + qz * qz - 1.0) > 1e-6)
      throw std::runtime_error("NMPC requires a unit quaternion");
    std::array<double, NMPC_NX> x{};
    std::copy(state.begin(), state.end(), x.begin());
    // Augmented states are COMMANDS, not measured angular rates. Never
    // overwrite their memory with the plant's inner-loop response.
    std::copy_n(previous_command_.begin() + 1, 3, x.begin() + 6);
    x[13] = previous_command_[0];
    nmpc_solver_set_state(solver_.get(), x.data());
    for (int i = 0; i < n; ++i)
      nmpc_solver_set_ref_stage(solver_.get(), i, horizon[i].data());
    std::array<double, NMPC_NYN> terminal{};
    std::copy_n(horizon.back().begin(), NMPC_NYN, terminal.begin());
    nmpc_solver_set_ref_terminal(solver_.get(), terminal.data());
    trace_.references = horizon;
    trace_.terminal   = terminal;
    trace_.state      = x;
    trace_.submitted  = true;
    std::array<double, NMPC_NU> command_derivative{};
    std::array<double, NMPC_NX> predicted{};
    const int                   status = nmpc_solver_solve(
        solver_.get(), command_derivative.data(), predicted.data());
    trace_.status             = status;
    trace_.command_derivative = command_derivative;
    if (status != 0 && status != 2)
      throw std::runtime_error("NMPC solve failed, status=" +
                               std::to_string(status));
    for (double value : predicted)
      if (!std::isfinite(value))
        throw std::runtime_error("non-finite NMPC prediction");
    std::array<double, NMPC_NU> next_command{};
    for (int i = 0; i < NMPC_NU; ++i)
    {
      next_command[i] =
          previous_command_[i] + command_derivative[i] / config_.ctrl_frq;
      if (!std::isfinite(command_derivative[i]) ||
          !std::isfinite(next_command[i]))
        throw std::runtime_error("non-finite NMPC command integration");
    }
    // Commit only an accepted finite solve. Integrate at the actual control dt,
    // not the OCP's 100 ms x_1, and do not publish du/dt as an absolute
    // command.
    previous_command_         = next_command;
    trace_.integrated_command = next_command;
    trace_.accepted           = true;
    return {{next_command[1], next_command[2], next_command[3]},
            next_command[0],
            status};
  }
}  // namespace nmpc
