#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "nmpc_controller/tracking_controller.hpp"

// Contract tests use an explicit fake C solver, not a pretend acados
// integration.
struct nmpc_solver
{
  int unused = 0;
};
namespace
{
  int                         creates = 0, destroys = 0, status = 0;
  double                      compiled_dt = 0.1;
  std::array<double, NMPC_NX> last_state{};
  std::array<double, NMPC_NY> first_reference{};
  bool                        emit_nan = false;
  int                         abi      = NMPC_SOLVER_ABI;
  std::array<double, NMPC_NU> emitted_derivative{9.81, 0.1, 0.2, 0.3};
  void                        check(bool c, const char *message)
  {
    if (!c)
      throw std::runtime_error(message);
  }
  template <class F>
  void throws(F f)
  {
    bool caught = false;
    try
    {
      f();
    }
    catch (const std::exception &)
    {
      caught = true;
    }
    check(caught, "expected controller failure");
  }
}  // namespace
extern "C"
{
  int nmpc_solver_abi_version()
  {
    return abi;
  }
  nmpc_solver_t *nmpc_solver_create()
  {
    ++creates;
    return new nmpc_solver;
  }
  void nmpc_solver_destroy(nmpc_solver_t *s)
  {
    if (s)
    {
      ++destroys;
      delete s;
    }
  }
  int nmpc_solver_horizon(const nmpc_solver_t *)
  {
    return 10;
  }
  double nmpc_solver_stage_dt(const nmpc_solver_t *, int)
  {
    return compiled_dt;
  }
  void nmpc_solver_set_state(nmpc_solver_t *, const double x[NMPC_NX])
  {
    std::copy_n(x, NMPC_NX, last_state.begin());
  }
  void nmpc_solver_set_ref_stage(nmpc_solver_t *, int stage,
                                 const double y[NMPC_NY])
  {
    if (stage == 0)
      std::copy_n(y, NMPC_NY, first_reference.begin());
  }
  void nmpc_solver_set_ref_terminal(nmpc_solver_t *, const double[NMPC_NYN]) {}
  int  nmpc_solver_solve(nmpc_solver_t *, double du[NMPC_NU], double x[NMPC_NX])
  {
    std::copy(emitted_derivative.begin(), emitted_derivative.end(), du);
    std::fill_n(x, NMPC_NX, 0.0);
    x[6]  = 1.1;
    x[7]  = 1.2;
    x[8]  = 1.3;  // must NOT be published as command
    x[13] = 8.0;
    if (emit_nan)
      x[0] = std::numeric_limits<double>::quiet_NaN();
    return status;
  }
}
int main()
{
  try
  {
    std::array<double, 13> state{};
    state[9] = 1.0;
    const auto reference =
        nmpc::cost_reference({1, 2, 3}, {4, 5, 6}, {0, 1, 0});
    std::vector<nmpc::CostReference> horizon(11, reference);
    {
      nmpc::TrackingController controller({});
      const auto               result = controller.compute(state, horizon);
      check(std::abs(result.specific_force - (9.81 + 0.02 * 9.81)) < 1e-12,
            "collective derivative not integrated at the actual 20 ms period");
      check(std::abs(result.rates[0] - 0.002) < 1e-12 &&
                std::abs(result.rates[2] - 0.006) < 1e-12,
            "rate derivative was published directly or used the 100 ms "
            "prediction");
      check(last_state[13] == 9.81 && last_state.size() == 14,
            "quadrotor command state layout");
      check(first_reference[0] == 1 && first_reference[3] == 4 &&
                first_reference[7] == 1 &&
                std::all_of(first_reference.begin() + 9, first_reference.end(),
                            [](double v) { return v == 0.0; }),
            "cost must be p/v/yb plus zero command derivatives, never hover");
      state[6]          = -0.8;  // sensor rate must not replace command memory
      const auto second = controller.compute(state, horizon);
      check(std::abs(last_state[6] - result.rates[0]) < 1e-12 &&
                std::abs(last_state[13] - result.specific_force) < 1e-12 &&
                std::abs(second.rates[0] - 0.004) < 1e-12,
            "command integrator was reinitialized from measured rates");
      check(controller.trace().submitted &&
                controller.trace().references == horizon &&
                controller.trace().references[0] == first_reference,
            "actual solver reference trace changed");
      status = 2;
      check(controller.compute(state, horizon).status == 2,
            "accepted status 2 not recorded");
      status = 4;
      throws([&] { controller.compute(state, horizon); });
      check(controller.trace().status == 4 && controller.trace().submitted,
            "failed solver reference trace lost");
      const auto failed_state = last_state;
      status                  = 0;
      controller.compute(state, horizon);
      check(last_state[13] == failed_state[13] &&
                last_state[6] == failed_state[6],
            "failed solve advanced command memory");
      emit_nan = true;
      throws([&] { controller.compute(state, horizon); });
      emit_nan = false;
      horizon.pop_back();
      throws([&] { controller.compute(state, horizon); });
      horizon.push_back(reference);
      emitted_derivative.fill(0.0);
      controller.reset({12.0, 0.3, -0.4, 0.5});
      const auto held = controller.compute(state, horizon);
      check(held.specific_force == 12.0 && held.rates[0] == 0.3 &&
                held.rates[1] == -0.4,
            "zero derivative must preserve non-hover/nonzero commands");
      emitted_derivative[0] = std::numeric_limits<double>::quiet_NaN();
      throws([&] { controller.compute(state, horizon); });
      emitted_derivative.fill(0.0);
      check(controller.compute(state, horizon).specific_force == 12.0,
            "invalid derivative advanced command memory");
      horizon[0][9] = 1.0;
      throws([&] { controller.compute(state, horizon); });
      horizon[0][9]         = 0.0;
      emitted_derivative    = {9.81, 0.1, 0.2, 0.3};
      const int old_creates = creates, old_destroys = destroys;
      controller.reset();
      check(creates == old_creates + 1 && destroys == old_destroys + 1,
            "solver reset was incomplete");
      std::cout << "PASS NMPC units/layout/status/failure/reset contract (fake "
                   "solver)\n";
    }
    {
      nmpc::TrackingConfig config;
      config.ctrl_frq = 100.0;
      nmpc::TrackingController controller(config);
      const auto               output = controller.compute(state, horizon);
      check(std::abs(output.specific_force - (9.81 + 0.01 * 9.81)) < 1e-12 &&
                std::abs(output.rates[0] - 0.001) < 1e-12,
            "command integration ignored configured control period");
    }
    nmpc::TrackingConfig invalid;
    invalid.initial_specific_thrust = -0.1;
    throws([&] { nmpc::TrackingController controller(invalid); });
    invalid          = {};
    invalid.ctrl_frq = 5.0;
    throws([&] { nmpc::TrackingController controller(invalid); });
    abi = 2;
    throws([] { nmpc::TrackingController controller({}); });
    abi         = NMPC_SOLVER_ABI;
    compiled_dt = 0.2;
    throws([] { nmpc::TrackingController controller({}); });
    check(creates == destroys, "solver lifetime leak");
    std::cout << "PASS stale NMPC bundle grid rejection (fake solver)\n";
    return 0;
  }
  catch (const std::exception &e)
  {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
