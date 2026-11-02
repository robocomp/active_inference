// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// Motion-calibration service (the camera mount pooled solve was removed 2026-10-29).

#include "room_concept.h"
#include "../../common/status_stream/status_sink.h"
#include <pthread.h>   // pthread_setname_np: name the worker so a per-thread CPU sample attributes itself
#include "pointcloud_center_estimator.h"
#include <numeric>
#include <set>
#include "room_gn_solver.h"
#include "room_obs_weights.h"
#include "room_gn_solver.h"
#include "image_edge_accumulate.h"
#include "image_edge_ops.h"
#include <algorithm>
#include <ranges>
#include <fstream>
#include <filesystem>
#include <limits>
#include <locale>
#include <ctime>
#include <print>
#include <sstream>
#include <sys/stat.h>
#include <QDebug>

namespace rc
{

    /// Calibration bookkeeping, run every cycle: apply a pending reset, restore the window once,
    /// drain any closed pivots into the estimator, and keep the window persisted.
    ///
    /// ★ UNGATED, and that matters. All of this used to live inside adapt_motion_model(), which was
    /// gated by LearnMotionModel — a flag switched off in 2026-08 to disable a DIFFERENT learner (the
    /// EMA slip-k path, A/B'd and rejected, now deleted). One switch was therefore turning off two
    /// unrelated things, and the batch estimator's whole bookkeeping went with it: closures reached
    /// nothing, the state file was never written, and nothing said so.
    void RoomConcept::service_calibration()
    {
        // ── Reset, if the calibration window asked for one ────────────────────────────────────────
        if (calib_reset_pending_.exchange(false))
        {
            motion_calib_.reset_state(params.calib_state_file);
            qInfo() << "[calib] RESET: window emptied and" 
                    << QString::fromStdString(params.calib_state_file)
                    << "deleted. Every parameter is back at its prior and reports NOT informed, which"
                       " is the honest state of a robot that has just been told to un-learn.";
        }

        // ── Restore the window once, then keep it saved ───────────────────────────────────────────
        // Evidence, not parameters: the solve after a restart is the solve it would have been had the
        // run never stopped, with the prior still at zero and nothing ratcheting.
        if (not calib_state_loaded_)
        {
            calib_state_loaded_ = true;
            const std::size_t n = motion_calib_.load_state(params.calib_state_file);
            if (n > 0)
                qInfo().nospace() << "[calib] restored " << n << " measurements from "
                                  << QString::fromStdString(params.calib_state_file)
                                  << " (" << motion_calib_.closures() << " closed pivot(s)) — the "
                                     "window resumes, the priors do not move";
            // Localiser thread: rc::status::event only pushes into the stream's queue.
            rc::status::event("loaded", rc::status::Obj{}
                                            .s("what", "motion_calib")
                                            .s("path", params.calib_state_file)
                                            .s("detail", std::format("{} measurements, {} closed pivot(s){}", n,
                                                                     motion_calib_.closures(),
                                                                     n > 0 ? "" : " (nothing on disk: priors)")));
            if (const std::size_t d = motion_calib_.legacy_dropped(); d > 0)
                qWarning().nospace() << "[calib] DROPPED " << d << " episode rows from "
                                     << QString::fromStdString(params.calib_state_file)
                                     << " — written before the heading became two factors (one shared "
                                        "heading covariate, bias of opposite sign); no exact split exists, "
                                        "so the window re-earns them";
        }

        // Drain any closed pivots first, on THIS thread, before the episode path runs. A closure is
        // the strongest measurement of the rotation model this robot can make — no map, no survey,
        // no localiser in the number — and until 2026-08-26 it reached nothing: the pivot measured
        // the scale to +/-0.18%, logged it, and the estimator that PRICES the manoeuvre never heard.
        // So the offer's value never fell and the robot pivoted for ever, unable to extinguish its
        // own worth. Feeding it back is what closes that loop.
        {
            std::vector<ClosureObs> pending;
            {
                std::scoped_lock lk(pending_closures_mutex_);
                pending.swap(pending_closures_);
            }
            for (const auto& c : pending)
            {
                motion_calib_.observe_closure(c.truth_rad, c.turned_rad, c.rate_rad_s, c.sigma_s);
                // Say whether the closure actually TAUGHT k_omega, not just that it arrived. A
                // parameter's value cannot distinguish "measured" from "left where the prior put
                // it", and the accessors now apply it only once it is taught — so this is the line
                // that says whether the pivot changed the robot's behaviour or only its logs.
                qInfo().nospace() << "[calib] CLOSURE -> batch estimator | truth "
                                  << QString::number(c.truth_rad * 180.0 / M_PI, 'f', 1) << " deg, odometry "
                                  << QString::number(c.turned_rad * 180.0 / M_PI, 'f', 1) << " deg, rate "
                                  << QString::number(c.rate_rad_s, 'f', 3) << " rad/s, sigma "
                                  << QString::number(c.sigma_s * 100.0, 'f', 3) << "% | k_omega now "
                                  << QString::number((motion_calib_.omega_scale() - 1.f) * 100.f, 'f', 4)
                                  << "% +/- " << QString::number(motion_calib_.k_w_sigma() * 100.f, 'f', 4)
                                  << "% over " << motion_calib_.closures() << " closure(s), "
                                  << (motion_calib_.taught(rc::calib::P_K_OMEGA)
                                          ? "TAUGHT — the scale is now applied"
                                          : "still UNTAUGHT — nominal is applied, the value is only reported");
            }
            // A closure is minutes of robot time; persist immediately rather than waiting for the
            // periodic save, so a crash between here and then cannot throw it away.
            if (not pending.empty())
                motion_calib_.save_state(params.calib_state_file);
        }

        // Periodic save of the window. Every 30 s: often enough that a crash costs little, rare
        // enough to be invisible next to a solve.
        {
            const std::int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            if (calib_state_last_save_ms_ == 0) calib_state_last_save_ms_ = now_ms;
            else if (now_ms - calib_state_last_save_ms_ >= 30000)
            {
                motion_calib_.save_state(params.calib_state_file);
                calib_state_last_save_ms_ = now_ms;
            }
        }

    }

} // namespace rc
