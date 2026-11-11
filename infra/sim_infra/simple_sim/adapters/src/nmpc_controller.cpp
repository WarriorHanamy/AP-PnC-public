#include "simple_sim/adapters/nmpc_controller.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <utility>

namespace simple_sim
{
  NmpcController::NmpcController(
      std::shared_ptr<const planner::core::ReferenceTrajectory> trajectory,
      nmpc::TrackingConfig config, std::filesystem::path trace_file)
      : trajectory_(std::move(trajectory)), controller_(config)
  {
    if (!trajectory_ || trajectory_->empty())
      throw std::invalid_argument("empty NMPC trajectory");
    if (!trace_file.empty())
    {
      if (!trace_file.is_absolute())
        throw std::invalid_argument("absolute trace path required");
      trace_file_.exceptions(std::ios::failbit | std::ios::badbit);
      trace_file_.open(trace_file);
      trace_file_ << "tick,control_time_ns,reference_time_ns,stage,ny,solver_"
                     "status,sample_time_ns";
      for (int i = 0; i < NMPC_NY; ++i)
        trace_file_ << ",yref_" << i;
      for (const auto *name : {"p", "v", "a", "omega", "yb"})
        for (const auto *axis : {"x", "y", "z"})
          trace_file_ << ",planner_" << name << '_' << axis;
      trace_file_ << ",planner_specific_thrust,fallback";
      for (int i = 0; i < NMPC_NP; ++i)
        trace_file_ << ",parameter_" << i;
      for (int i = 0; i < NMPC_NX; ++i)
        trace_file_ << ",initial_state_" << i;
      for (const auto *prefix : {"u0_derivative_", "published_command_"})
        for (int i = 0; i < NMPC_NU; ++i)
          trace_file_ << ',' << prefix << i;
      trace_file_ << ",solution_accepted\n" << std::setprecision(17);
    }
  }
  void NmpcController::set_reference(
      std::shared_ptr<const planner::core::ReferenceTrajectory> trajectory,
      double                                                    now_s)
  {
    if (!trajectory || trajectory->empty())
      throw std::invalid_argument("empty NMPC trajectory update");
    if (trajectory_ &&
        trajectory_->duration() < now_s + controller_.config().horizon_s)
      throw std::logic_error(
          "reference append would alter the active NMPC horizon");
    trajectory_ = std::move(trajectory);
  }
  void NmpcController::reset(const State &initial)
  {
    validate_state(initial);
    shown_ready_  = false;
    cruise_ready_ = false;
    controller_.reset({controller_.config().initial_specific_thrust,
                       initial.angular_velocity.x(),
                       initial.angular_velocity.y(),
                       initial.angular_velocity.z()});
  }
  Eigen::Vector3d NmpcController::shape_body_y(const Eigen::Vector3d &flat_yb,
                                               double speed, double dt,
                                               Eigen::Vector3d &shown,
                                               Eigen::Vector3d &cruise,
                                               bool            &shown_ready,
                                               bool &cruise_ready) const
  {
    const auto     &cfg = controller_.config();
    Eigen::Vector3d target =
        flat_yb.norm() > 1e-6 ? flat_yb.normalized() : Eigen::Vector3d::UnitY();
    if (cfg.yb_hold_speed_m_s > 0.0 && speed >= cfg.yb_hold_speed_m_s)
    {
      cruise       = target;
      cruise_ready = true;
    }
    else if (cfg.yb_hold_speed_m_s > 0.0 && cruise_ready)
    {
      target = cruise;
    }
    if (!shown_ready)
    {
      shown       = target;
      shown_ready = true;
      return shown;
    }
    shown.normalize();
    if (!(cfg.yb_slew_rad_s > 0.0))
    {
      shown = target;
      return shown;
    }
    if (!(dt > 0.0))
      return shown;
    const double angle = std::acos(std::clamp(shown.dot(target), -1.0, 1.0));
    const double step  = cfg.yb_slew_rad_s * dt;
    if (angle <= step || angle < 1e-8)
    {
      shown = target;
      return shown;
    }
    Eigen::Vector3d axis = shown.cross(target);
    if (axis.norm() < 1e-8)
    {
      axis = shown.cross(Eigen::Vector3d::UnitZ());
      if (axis.norm() < 1e-8)
        axis = shown.cross(Eigen::Vector3d::UnitX());
    }
    shown = Eigen::AngleAxisd(step, axis.normalized()) * shown;
    shown.normalize();
    return shown;
  }
  ControlResult NmpcController::compute(const StepContext &ctx, const State &s,
                                        const Feedback &feedback)
  {
    const auto &cfg = controller_.config();
    if (std::abs(static_cast<double>(ctx.control_dt_ns) * 1e-9 -
                 1.0 / cfg.ctrl_frq) > 1e-9)
      throw std::runtime_error("runner control period differs from nmpc.yaml");
    const auto n =
        static_cast<int>(std::llround(cfg.horizon_s / cfg.traj_res_s));
    std::vector<nmpc::CostReference>           horizon;
    std::vector<planner::core::ReferencePoint> planned;
    horizon.reserve(n + 1);
    Eigen::Vector3d shown        = shown_yb_;
    Eigen::Vector3d cruise       = cruise_yb_;
    bool            shown_ready  = shown_ready_;
    bool            cruise_ready = cruise_ready_;
    for (int i = 0; i <= n; ++i)
    {
      const double t   = ctx.time_s() + i * cfg.traj_res_s;
      const auto   ref = trajectory_->sample(t);
      planned.push_back(ref);
      const double dt = i == 0 ? 0.0 : cfg.traj_res_s;
      const auto   yb = shape_body_y(ref.yb, ref.v.norm(), dt, shown, cruise,
                                     shown_ready, cruise_ready);
      horizon.push_back(nmpc::cost_reference({ref.p.x(), ref.p.y(), ref.p.z()},
                                             {ref.v.x(), ref.v.y(), ref.v.z()},
                                             {yb.x(), yb.y(), yb.z()}));
    }
    const auto now = trajectory_->sample(ctx.time_s());
    shape_body_y(now.yb, now.v.norm(), 1.0 / cfg.ctrl_frq, shown_yb_,
                 cruise_yb_, shown_ready_, cruise_ready_);
    const std::array<double, 13> x = {
        s.position.x(),         s.position.y(),         s.position.z(),
        s.velocity.x(),         s.velocity.y(),         s.velocity.z(),
        s.angular_velocity.x(), s.angular_velocity.y(), s.angular_velocity.z(),
        s.attitude.w(),         s.attitude.x(),         s.attitude.y(),
        s.attitude.z()};
    nmpc::TrackingResult result;
    try
    {
      result = controller_.compute(x, feedback.specific_force.x(), horizon);
    }
    catch (...)
    {
      write_trace(ctx, planned);
      throw;
    }
    write_trace(ctx, planned);
    return {
        {result.specific_force,
         Eigen::Vector3d(result.rates[0], result.rates[1], result.rates[2])},
        result.status};
  }
  void NmpcController::write_trace(
      const StepContext                                &ctx,
      const std::vector<planner::core::ReferencePoint> &planned)
  {
    const auto &trace = controller_.trace();
    if (!trace_file_.is_open() || !trace.submitted)
      return;
    const auto dt = static_cast<std::int64_t>(
        std::llround(controller_.config().traj_res_s * 1e9));
    for (std::size_t i = 0; i < trace.references.size(); ++i)
    {
      const bool terminal = i + 1 == trace.references.size();
      trace_file_ << ctx.index << ',' << ctx.time_ns << ','
                  << ctx.time_ns + static_cast<std::int64_t>(i) * dt << ',' << i
                  << ',' << (terminal ? NMPC_NYN : NMPC_NY) << ','
                  << trace.status << ','
                  << static_cast<std::int64_t>(
                         std::llround(planned[i].t * 1e9));
      for (int field = 0; field < NMPC_NY; ++field)
        trace_file_ << ','
                    << (terminal
                            ? (field < NMPC_NYN ? trace.terminal[field] : 0.0)
                            : trace.references[i][field]);
      const auto &r = planned[i];
      for (const auto &v : {r.p, r.v, r.a, r.omega, r.yb})
        for (int a = 0; a < 3; ++a)
          trace_file_ << ',' << v[a];
      trace_file_ << ',' << r.thrust << ',' << r.flatness_fallback;
      for (double p : trace.parameters)
        trace_file_ << ',' << p;
      for (double x : trace.state)
        trace_file_ << ',' << x;
      for (const auto &values :
           {trace.command_derivative, trace.integrated_command})
        for (double value : values)
          trace_file_ << ',' << value;
      trace_file_ << ',' << trace.accepted << '\n';
    }
    trace_file_.flush();
  }
}  // namespace simple_sim
