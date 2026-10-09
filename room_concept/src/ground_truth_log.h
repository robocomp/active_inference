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
        /// One row per localiser result into tmp/heading/heading_<start>.csv -- real robot AND
        /// simulation (gt columns are NaN without ground truth). Everything needed to judge the
        /// wheel x gyro heading fusion offline: raw channel rotations, weights, densities, ZUPT time,
        /// all calibration values/sigmas, and the pose before and after the optimiser.
        void log_heading(const RoomConcept::UpdateResult& res);
        /// The room frame is the layout recentred on its bbox (RecenterRoomPolygon): room = world - offset.
        /// The supervisor pose is WORLD, so gt_x/gt_y are written minus this offset (the raw values go to
        /// gt_x_world/gt_y_world). Set once from initialize_room_model_from_svg(); zero = frames coincide.
        void set_world_offset(const Eigen::Vector2f& off) { off_x_.store(off.x()); off_y_.store(off.y()); }

    private:
        void gt_convention_report(float est_th, float gt_th);
        /// The world->room offset is right only if the scenario's SVG is drawn in Webots world
        /// coordinates (verified for apartamento only). Accumulates est - gt_room and, at 200/2000/20000
        /// samples, logs its mean against the error the localiser can explain: its own scatter PLUS its
        /// claimed sigma (scatter alone fails when parked: 1 mm scatter, 9 mm constant bias). A wrong frame
        /// is metres; a localiser bias is centimetres, so the 3x margin is generous to the latter.
        /// Diagnostic only -- it changes nothing.
        void gt_frame_report(float dx, float dy, float claimed_var_xy);

        std::shared_ptr<DSR::DSRGraph> G;
        RoomConcept& room_concept_;
        const std::atomic<bool>& shutting_down_;

        double gt_sum_diff_c_ = 0, gt_sum_diff_s_ = 0;   ///< circular accumulators for est - gt
        double gt_sum_sum_c_  = 0, gt_sum_sum_s_  = 0;   ///< and for est + gt
        long   gt_n_ = 0;
        double fr_sx_ = 0, fr_sy_ = 0, fr_sxx_ = 0, fr_syy_ = 0;   ///< est - gt_room moments
        double fr_var_ = 0;   ///< sum of the published posterior's var_x + var_y
        long   fr_n_ = 0, fr_report_at_ = 200;
        std::atomic<float> off_x_{0.f}, off_y_{0.f};   ///< world -> room translation (set on main, read on localiser)
        long   gt_report_at_ = 200;
        std::ofstream gt_csv_;
        bool          gt_csv_open_attempted_ = false;
        std::ofstream hd_csv_;
        bool          hd_csv_open_attempted_ = false;
    };
}   // namespace rc
