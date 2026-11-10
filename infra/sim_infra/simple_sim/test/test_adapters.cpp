#include <iostream>
#include <limits>
#include <quadrotor_df/flu_flatness.hpp>
#include <quadrotor_df/flu_reference.hpp>
#include <stdexcept>

#include "simple_sim/adapters/controllers.hpp"
#include "simple_sim/rate_loop.hpp"
#include "simple_sim/runner.hpp"

namespace
{
  void check(bool c, const char *message)
  {
    if (!c)
      throw std::runtime_error(message);
  }
  void flu_flatness_jacobian()
  {
    // The analytic Jacobian must agree with central differences of the same
    // value map, otherwise the MINCO cost gradient is wrong.
    quadrotor_df::FluFlatnessParams params;
    const auto                      value = [&params](const Eigen::Vector3d &v,
                                 const Eigen::Vector3d &a,
                                 const Eigen::Vector3d &j)
    {
      const auto r = quadrotor_df::flu_flatness(v, a, j, params);
      Eigen::Matrix<double, 4, 1> out;
      out << r.specific_thrust, r.omega;
      return out;
    };
    const std::array<Eigen::Vector3d, 4> velocities{
        Eigen::Vector3d(10.0, 0.0, 0.0), Eigen::Vector3d(6.0, -3.0, 1.0),
        Eigen::Vector3d(0.0, 10.0, 0.0), Eigen::Vector3d(5.0, 5.0, 2.0)};
    const Eigen::Vector3d acc(0.0, -5.0, 0.0);
    const Eigen::Vector3d jerk(1.0, 0.5, -0.5);
    double                worst = 0.0;
    for (const auto &v : velocities)
    {
      const auto flatness = quadrotor_df::flu_flatness(v, acc, jerk, params);
      check(flatness.nominal, "expected the nominal branch");
      const std::array<Eigen::Vector3d, 3> inputs{v, acc, jerk};
      const double                         step = 1e-6;
      for (int block = 0; block < 3; ++block)
        for (int axis = 0; axis < 3; ++axis)
        {
          Eigen::Vector3d plus = inputs[block], minus = inputs[block];
          plus[axis] += step;
          minus[axis] -= step;
          Eigen::Vector3d p[3] = {inputs[0], inputs[1], inputs[2]};
          Eigen::Vector3d m[3] = {inputs[0], inputs[1], inputs[2]};
          p[block]             = plus;
          m[block]             = minus;
          const Eigen::Matrix<double, 4, 1> central =
              (value(p[0], p[1], p[2]) - value(m[0], m[1], m[2])) / (2 * step);
          const double error =
              (central - flatness.jacobian.col(block * 3 + axis))
                  .cwiseAbs()
                  .maxCoeff();
          worst = std::max(worst, error);
          check(error < 1e-6,
                "FLU flatness Jacobian disagrees with finite differences");
        }
    }
    // Quadrotor flatness aligns thrust with specific force, so the wing
    // residual is zero.
    const Eigen::Vector3d v(10.0, 0.0, 0.0);
    const Eigen::Vector3d circle_acc(0.0, -5.0, 0.0);
    const auto            r = quadrotor_df::flu_flatness(v, circle_acc,
                                                         Eigen::Vector3d::Zero(), params);
    const Eigen::Vector3d eta =
        circle_acc + params.gravity * Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d residual =
        r.rotation.transpose() * (eta - r.specific_thrust * r.rotation.col(2));
    check(residual.norm() < 1e-9,
          "quadrotor flatness must leave no wing-force residual");
    std::cout << "FLU flatness: worst Jacobian/FD error = " << worst << "\n";
  }
  void se3_level_flight()
  {
    const double                gravity = 9.81;
    Eigen::Matrix<double, 3, 6> polynomial =
        Eigen::Matrix<double, 3, 6>::Zero();
    polynomial(0, 4) = 4.0;
    polynomial(2, 5) = 2.0;
    TrajectoryRepresentation<5> representation;
    representation.emplace_back(2.0, polynomial);
    auto reference = std::make_shared<const planner::core::ReferenceTrajectory>(
        std::move(representation));
    simple_sim::Se3Controller controller(reference, gravity, {});
    simple_sim::State         state =
        simple_sim::initial_state(reference->sample(0.0), gravity);
    controller.reset(state);
    const auto command =
        controller.compute({0, 0, 20000000}, state, {}).command;
    check(command.rates.norm() < 1e-8,
          "matched quadrotor attitude must not demand a rate");
    check(std::abs(command.specific_force - gravity) < 1e-8,
          "level flight thrust must balance weight");
  }
  void tracking()
  {
    // Use planner-owned continuous trajectory; no sim-owned reference
    // representation.
    Eigen::Matrix<double, 3, 6> polynomial =
        Eigen::Matrix<double, 3, 6>::Zero();
    polynomial(0, 4) = 1.0;  // p_x = t
    polynomial(2, 5) = 2.0;
    TrajectoryRepresentation<5> representation;
    representation.emplace_back(2.0, polynomial);
    auto reference = std::make_shared<const planner::core::ReferenceTrajectory>(
        std::move(representation));
    const auto endpoint = reference->sample(reference->duration());
    for (double t : {reference->duration(), reference->duration() + 0.1,
                     reference->duration() + 100.0})
    {
      const auto held = reference->sample(t);
      check(held.t == endpoint.t && held.p == endpoint.p &&
                held.v == endpoint.v && held.a == endpoint.a &&
                held.yb == endpoint.yb && held.omega == endpoint.omega &&
                held.rotation == endpoint.rotation &&
                held.thrust == endpoint.thrust &&
                held.flatness_fallback == endpoint.flatness_fallback,
            "terminal reference was extrapolated or modified");
    }
    for (double t : {-1.0, std::numeric_limits<double>::quiet_NaN()})
    {
      bool caught = false;
      try
      {
        reference->sample(t);
      }
      catch (const std::invalid_argument &)
      {
        caught = true;
      }
      check(caught, "invalid reference time accepted");
    }
    simple_sim::PlantConfig   plant;
    simple_sim::Se3Controller controller(reference, plant.gravity, {});
    simple_sim::Runner runner({1000000, 20000000, 50}, simple_sim::Plant(plant),
                              simple_sim::RateLoop(plant, {}), controller);
    const auto         first = reference->sample(0.0);
    const auto initial       = simple_sim::initial_state(first, plant.gravity);
    runner.reset(initial, {plant.gravity, Eigen::Vector3d::Zero()});
    for (int i = 0; i < 50; ++i)
    {
      const auto r   = runner.step();
      const auto ref = reference->sample(r.context.time_s() + 0.02);
      check((r.after.position - ref.p).norm() < 1e-6,
            "line tracking regression");
    }
    check(runner.status() == simple_sim::RunStatus::Finished,
          "tracking did not finish");
  }
}  // namespace
int main()
{
  try
  {
    const auto hover = quadrotor_df::flu_reference(Eigen::Vector3d::Zero(),
                                                   Eigen::Vector3d::Zero(),
                                                   Eigen::Vector3d::Zero());
    check(hover.rotation.isApprox(Eigen::Matrix3d::Identity()) &&
              std::abs(hover.specific_thrust - 9.81) < 1e-12,
          "+Z FLU hover reference");
    for (double speed : {0.0, 0.499999, 0.500001, 1.0, 6.0})
    {
      const auto r =
          quadrotor_df::flu_reference({speed, 0, 0}, {0, 0, 0}, {0, 0, 0});
      check(r.rotation.col(1).isApprox(Eigen::Vector3d::UnitY()),
            "yb flipped across speed guard");
      check(std::abs(r.rotation.determinant() - 1) < 1e-12,
            "reference reflection");
    }
    flu_flatness_jacobian();
    std::cout << "PASS quadrotor FLU flatness Jacobian\n";
    se3_level_flight();
    std::cout << "PASS quadrotor SE3 level flight\n";
    tracking();
    std::cout << "PASS planner reference + SE3 line tracking\n";
    return 0;
  }
  catch (const std::exception &e)
  {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
