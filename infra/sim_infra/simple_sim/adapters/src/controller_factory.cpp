#include <yaml-cpp/yaml.h>

#include <stdexcept>

#include "simple_sim/adapters/controllers.hpp"

#ifdef SIMPLE_SIM_WITH_NMPC
#include "simple_sim/adapters/nmpc_controller.hpp"
#endif

namespace simple_sim
{

  std::unique_ptr<TrackingController> make_controller(
      const ControllerSelection &selection)
  {
    if (!selection.trajectory || selection.trajectory->empty())
      throw std::invalid_argument("controller requires a trajectory");
    if (selection.backend == "se3")
    {
      return std::make_unique<Se3Controller>(
          selection.trajectory, selection.gravity, selection.se3,
          selection.aero_model, selection.mass, selection.zero_thrust_axis_aero,
          selection.aero);
    }
    if (selection.backend == "nmpc")
    {
#ifndef SIMPLE_SIM_WITH_NMPC
      throw std::runtime_error(
          "NMPC support disabled in this build; rebuild on Linux with solver "
          "bundle");
#else
      if (!selection.nmpc_yaml.is_absolute())
        throw std::invalid_argument("NMPC yaml path must be absolute");
      const YAML::Node nmpc =
          YAML::LoadFile(selection.nmpc_yaml.string())["nmpc"];
      nmpc::TrackingConfig tracking;
      tracking.horizon_s  = nmpc["horizon_s"].as<double>();
      tracking.traj_res_s = nmpc["traj_res_s"].as<double>();
      tracking.ctrl_frq   = nmpc["ctrl_frq"].as<double>();
      tracking.initial_specific_thrust =
          nmpc["initial_specific_thrust"].as<double>();
      tracking.cx_alpha_slope      = nmpc["cx_alpha_slope"].as<double>();
      tracking.cx_slope_estimation = nmpc["cx_slope_estimation"].as<bool>();
      tracking.cx_slope_forgetting = nmpc["cx_slope_forgetting"].as<double>();
      tracking.cx_slope_limit      = nmpc["cx_slope_limit"].as<double>();
      tracking.cx_slope_covariance_limit =
          nmpc["cx_slope_covariance_limit"].as<double>();
      tracking.cx_slope_min_excitation =
          nmpc["cx_slope_min_excitation"].as<double>();
      tracking.cx_slope_lag_s = nmpc["cx_slope_lag_s"].as<double>();
      return std::make_unique<NmpcController>(selection.trajectory, tracking,
                                              selection.trace_file);
#endif
    }
    throw std::invalid_argument("controller must be se3 or nmpc, got '" +
                                selection.backend + "'");
  }

}  // namespace simple_sim
