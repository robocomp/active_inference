#pragma once

// ── THE GROUND-TRUTH CHANNEL, LIFTED OUT OF SpecificWorker ───────────────────────────────────────
// Grades the localiser against the simulator's supervisor: one row per corrected publish to
// tmp/sdf_localizer/gt_error.csv, with the pose, the estimate, the fit and the motion inputs beside
// each other so an error can be attributed rather than just observed.
//
// ★ IT IS SELF-CHECKING ABOUT THE HEADING CONVENTION, which is the only subtle thing in here.
// robot_concept publishes gt heading in a convention that differencing est - gt turns into garbage —
// it reported heading "errors" of +201/+224/-119 deg before that was found. So BOTH conventions are
// scored every cycle by how CONSTANT the resulting offset is, and gt_convention_report() names the
// winner. If robot_concept is ever fixed the winner flips and the log says so, instead of a silent
// negation inverting a correct signal.

#include <atomic>
#include <cstdint>
#include <fstream>

#include "room_concept.h"

namespace DSR { class DSRGraph; }

namespace rc
{
    class GroundTruthLog
    {
    public:
        GroundTruthLog(std::shared_ptr<DSR::DSRGraph> graph, RoomConcept& room,
                       const std::atomic<bool>& shutting_down)
            : G(std::move(graph)), room_concept_(room), shutting_down_(shutting_down) {}

        /// One row per accepted CORRECTED publish. Wired to PosePublisher's hook — this is a
        /// diagnostic ABOUT a published pose, not part of publishing one.
        void log_ground_truth(const RoomConcept::UpdateResult& res);

    private:
        void gt_convention_report(float est_th, float gt_th);

        std::shared_ptr<DSR::DSRGraph> G;
        RoomConcept& room_concept_;
        const std::atomic<bool>& shutting_down_;

        double gt_sum_diff_c_ = 0, gt_sum_diff_s_ = 0;   ///< circular accumulators for est - gt
        double gt_sum_sum_c_  = 0, gt_sum_sum_s_  = 0;   ///< and for est + gt
        long   gt_n_ = 0;
        long   gt_report_at_ = 200;
        std::ofstream gt_csv_;
        bool          gt_csv_open_attempted_ = false;
    };
}   // namespace rc
