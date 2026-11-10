#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>

namespace simple_sim
{

  // All actuator/PID state lives in the transactional State, not in a mutable
  // callback. Feedback and RK4 force evaluations must never advance it.
  struct MotorState
  {
    Eigen::Vector4d rpm    = Eigen::Vector4d::Zero();
    Eigen::Vector4d rpm_sp = Eigen::Vector4d::Zero();
    Eigen::Vector4d duty   = Eigen::Vector4d::Constant(-1.0);
    // Inner rate-loop integral and error are body-FLU vectors.
    Eigen::Vector3d integral       = Eigen::Vector3d::Zero();
    Eigen::Vector3d previous_error = Eigen::Vector3d::Zero();
    std::int64_t    physics_ticks  = 0;
    bool            limited        = false;
    // Ideal inner loop only: the filtered collective specific force [m/s^2]
    // driven by the command through a first-order lag. Zero for the practical
    // plant, whose thrust comes out of the motor model instead.
    double specific_force = 0.0;
  };

  // World ENU; body FLU; attitude rotates body vectors into world.
  struct State
  {
    Eigen::Vector3d    position         = Eigen::Vector3d::Zero();
    Eigen::Vector3d    velocity         = Eigen::Vector3d::Zero();
    Eigen::Quaterniond attitude         = Eigen::Quaterniond::Identity();
    Eigen::Vector3d    angular_velocity = Eigen::Vector3d::Zero();
    MotorState         motors;
  };

  struct AeroObservation
  {
    // Unused on the quadrotor plant. Kept so a logged wrench can record that
    // no external flow observation was produced.
    Eigen::Vector3d velocity_body = Eigen::Vector3d::Zero();
    double          alpha_rad = 0.0, beta_rad = 0.0;
    bool            valid = false;
  };

  struct Wrench
  {
    Eigen::Vector3d force  = Eigen::Vector3d::Zero();  // body FLU, N
    Eigen::Vector3d moment = Eigen::Vector3d::Zero();  // body FLU, N m
    AeroObservation observation{};
  };

  struct Command
  {
    double          specific_force = 0.0;             // along body +Z, m/s^2
    Eigen::Vector3d rates = Eigen::Vector3d::Zero();  // body FLU, rad/s
  };

  struct Feedback
  {
    Eigen::Vector3d specific_force =
        Eigen::Vector3d::Zero();  // ideal IMU, no gravity
    // Optional external specific force (body FLU, m/s^2). The quadrotor plant
    // leaves this at zero; rotor loads are `applied`.
    Eigen::Vector3d external_specific_force = Eigen::Vector3d::Zero();
    Wrench          external;
    Wrench          applied;
  };

  struct StepContext
  {
    std::int64_t index         = 0;
    std::int64_t time_ns       = 0;
    std::int64_t control_dt_ns = 0;
    double       time_s() const { return static_cast<double>(time_ns) * 1e-9; }
  };

  struct ControlResult
  {
    Command command;
    int     solver_status = 0;
  };

  struct StepRecord
  {
    StepContext   context;
    State         before;
    State         after;
    Feedback      feedback_before;
    Feedback      feedback_after;
    ControlResult control;
    double        solve_time_ms = 0.0;  // diagnostic wall time only
  };

  void validate_state(const State &state);
  void validate_command(const Command &command);

}  // namespace simple_sim
