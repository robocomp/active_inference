/*
 * bottle_evaluator.h
 *
 * Validation harness for bottle_concept (NOT part of the perception pipeline):
 *  - Webots ground-truth acquisition (DEF bottle → room frame).
 *  - One-shot scene placement (Scene.PlaceBottleOnStart).
 *  - Moving-bottle sweep and static-restart drivers (Eval.MoveExperiment / StaticPoseTest).
 *  - Per-cycle fit-vs-GT CSV logging (pos/size error + NEES).
 *
 * All no-ops unless the corresponding Eval/Scene flag is set, so a shipped concept
 * agent pays nothing for it. Constructed by SpecificWorker with the deps it needs; the
 * table-top lookup is injected as a callback to stay decoupled from the scene-graph layer.
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <functional>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <genericworker.h>                    // RoboCompWebots2Robocomp proxy + ObjectPose
#include <dsr/api/dsr_inner_eigen_api.h>

#include "bottle_config.h"
#include "bottle_instance.h"

namespace rc {

class BottleEvaluator
{
public:
    // table_top(bx,by) → table-top z in the room frame when the bottle stands on a table (else nullopt).
    using TableTopFn = std::function<std::optional<float>(float, float)>;

    BottleEvaluator(BottleConfig& cfg,
                    RoboCompWebots2Robocomp::Webots2RobocompPrxPtr proxy,
                    DSR::InnerEigenAPI* inner_eigen,
                    TableTopFn table_top);

    // Scene.PlaceBottleOnStart: setObjectPose the bottle once to the configured world x,y.
    void place_bottle_on_start();
    bool place_done() const { return place_bottle_done_; }

    // Eval.StaticPoseTest: move the bottle to grid pose static_pose_index once. Returns true once issued.
    bool place_static_test_pose();

    // Eval.MoveExperiment: drive the bottle through the grid, re-seeding each pose. No-op unless enabled.
    void step_move_experiment(std::unordered_map<std::uint64_t, BottleInstance>& instances);

    // Append one fit-vs-GT row (pos/size error + NEES) to the eval CSV. No-op unless Eval.Enabled.
    void log_eval(const BottleInstance& inst, float free_energy);

private:
    // One-shot: query the bottle's Webots pose and express its CENTRE in the room frame → cfg_.gt_*.
    bool acquire_webots_gt();

    BottleConfig&                                   cfg_;
    RoboCompWebots2Robocomp::Webots2RobocompPrxPtr proxy_;
    DSR::InnerEigenAPI*                            inner_eigen_ = nullptr;
    TableTopFn                                     table_top_;

    std::ofstream eval_log_;            // open when cfg_.eval_enabled; header on first row
    bool gt_acquired_       = false;
    bool place_bottle_done_ = false;    // one-shot guard for cfg_.place_on_start

    // Moving-bottle experiment runtime state.
    bool                                move_started_ = false;
    RoboCompWebots2Robocomp::ObjectPose move_home_{};       // bottle's world pose at experiment start
    std::vector<std::pair<float,float>> move_offsets_;      // grid in world mm (absolute or home-relative)
    std::size_t                         move_idx_ = 0;
    int                                 move_settle_ = 0;    // cycles held at the current pose
    int                                 move_reseed_in_ = 0; // countdown to forcing a fresh cold-start
};

}  // namespace rc
