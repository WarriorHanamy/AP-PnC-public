#pragma once

#include <aerodynamics/aero_interface.hpp>
#include <filesystem>
#include <memory>
#include <optional>
#include <planner_core/trajectory.hpp>

#include "simple_sim/adapters/aero_model.hpp"
#include "simple_sim/controller.hpp"

namespace simple_sim
{

  /**
   * @brief Tracking backend. SE3 and NMPC are implementations, not the API.
   *
   * Runner only needs Controller. A flight that replaces the reference calls
   * set_reference on this interface and does not name a backend.
   */
  class TrackingController : public Controller
  {
   public:
    virtual void set_reference(
        std::shared_ptr<const planner::core::ReferenceTrajectory> trajectory,
        double                                                    now_s) = 0;
    /// Generated-solver ABI when this backend links one. The reactive law has
    /// none, so the flight record omits the field.
    virtual std::optional<int> solver_abi() const { return std::nullopt; }
  };

  struct Se3Gains
  {
    Eigen::Vector3d position = Eigen::Vector3d(4.0, 4.0, 4.0);
    Eigen::Vector3d velocity = Eigen::Vector3d(3.0, 3.0, 3.0);
    Eigen::Vector3d attitude = Eigen::Vector3d(6.0, 6.0, 6.0);
    // Flatness feedforward on the attitude: solve the angle of attack from the
    // true aerodynamic force balance numerically, rebuild the thrust axis from
    // it, and keep the planned yb. Off by default.
    bool aero_flatness_feedforward = false;
    // Body-rate output limit [rad/s, FLU], per axis.
    Eigen::Vector3d max_rate = Eigen::Vector3d(1.5, 2.5, 1.5);
    // Zero the thrust-axis component of the aerodynamic force in the
    // feedforward model, mirroring the plant's own injection rule for the ideal
    // profile. The two must agree or the feedforward fights a force that is not
    // there.
    bool aero_zero_thrust_axis = false;
  };

  class Se3Controller final : public TrackingController
  {
   public:
    Se3Controller(
        std::shared_ptr<const planner::core::ReferenceTrajectory> trajectory,
        double gravity, Se3Gains gains,
        std::shared_ptr<aerodynamics::AerodynamicsInterface> aero_model =
            nullptr,
        double mass = 0.0, bool zero_thrust_axis_aero = false,
        AeroConfig aero_config = {});
    void          reset(const State &initial) override;
    ControlResult compute(const StepContext &, const State &,
                          const Feedback &) override;
    void          set_reference(
                 std::shared_ptr<const planner::core::ReferenceTrajectory> trajectory,
                 double now_s) override;

   private:
    std::shared_ptr<const planner::core::ReferenceTrajectory> trajectory_;
    double                                                    gravity_;
    Se3Gains                                                  gains_;
    std::shared_ptr<aerodynamics::AerodynamicsInterface>      aero_model_;
    double                                                    mass_;
    double                                                    alpha_prev_;
    bool       zero_thrust_axis_aero_ = false;
    bool       alpha_initialized_     = false;
    AeroConfig aero_config_;
  };

  State initial_state(const planner::core::ReferencePoint &reference,
                      double                               gravity);

  struct ControllerSelection
  {
    std::string                                               backend = "se3";
    std::shared_ptr<const planner::core::ReferenceTrajectory> trajectory;
    double                                                    gravity = 9.81;
    double                                                    mass    = 2.0;
    Se3Gains                                                  se3;
    bool       zero_thrust_axis_aero = false;
    AeroConfig aero;
    std::shared_ptr<aerodynamics::AerodynamicsInterface> aero_model;
    std::filesystem::path                                nmpc_yaml;
    std::filesystem::path                                trace_file;
  };

  /// `se3` or `nmpc`. Unknown names are rejected. The result is only the
  /// tracking interface.
  std::unique_ptr<TrackingController> make_controller(
      const ControllerSelection &selection);

}  // namespace simple_sim
