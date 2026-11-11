#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <nmpc_controller/tracking_controller.hpp>
#include <planner_core/trajectory.hpp>

#include "simple_sim/adapters/controllers.hpp"

namespace simple_sim
{

  class NmpcController final : public TrackingController
  {
   public:
    NmpcController(
        std::shared_ptr<const planner::core::ReferenceTrajectory> trajectory,
        nmpc::TrackingConfig config, std::filesystem::path trace_file = {});
    // Reference-only update: retain solver warm start and command integration.
    void set_reference(
        std::shared_ptr<const planner::core::ReferenceTrajectory> trajectory,
        double now_s) override;
    std::optional<int> solver_abi() const override { return NMPC_SOLVER_ABI; }
    void               reset(const State &initial) override;
    ControlResult      compute(const StepContext &, const State &,
                               const Feedback &) override;

   private:
    std::shared_ptr<const planner::core::ReferenceTrajectory> trajectory_;
    nmpc::TrackingController                                  controller_;
    std::ofstream                                             trace_file_;
    // Body-y actually shown to the solver, and the last cruise value held
    // through a low-speed reversal. Reset with the plant; a reference append
    // keeps them so the heading does not jump.
    Eigen::Vector3d shown_yb_     = Eigen::Vector3d::UnitY();
    Eigen::Vector3d cruise_yb_    = Eigen::Vector3d::UnitY();
    bool            shown_ready_  = false;
    bool            cruise_ready_ = false;
    Eigen::Vector3d shape_body_y(const Eigen::Vector3d &flat_yb, double speed,
                                 double dt, Eigen::Vector3d &shown,
                                 Eigen::Vector3d &cruise, bool &shown_ready,
                                 bool &cruise_ready) const;
    void            write_trace(const StepContext &,
                                const std::vector<planner::core::ReferencePoint> &);
  };

}  // namespace simple_sim
