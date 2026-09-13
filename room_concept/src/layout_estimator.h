#pragma once

// ── THE LAYOUT ESTIMATOR ─────────────────────────────────────────────────────────────────────────
// One of the two halves of room_concept. The other is the LOCALISER (RoomConcept proper: the sliding
// window, the SDF, the solve, the covariance, the early exit). This half answers "what shape is this
// room?"; that half answers "where am I in it?".
//
//   SEARCHING   the layout is unknown. The walls ARE the estimate: landmarks in the solver, no SDF on
//               the polygon derived from them (scoring a pose against it would score the walls against
//               themselves), and a gauge, because the map frame is otherwise unobservable.
//   LOCALIZING  the layout is known and frozen. Nothing about it is optimised — no births, deaths,
//               merges, re-derivations, wall updates or theta0 motion — and the robot localises
//               against it exactly as against a layout loaded from file.
//
// The transition is one-way and fires on the first polygon that closes, re-anchors and passes the
// publish bar with every wall agreeing with its own Manhattan class. A file-given layout starts in
// LOCALIZING and never enters SEARCHING.
//
// ★ THE DEPENDENCY RUNS ONE WAY, AND THAT IS THE POINT OF THE SPLIT. This class never reaches into
//   the localiser. It takes the scan and the pose, and it RETURNS what it found — associations for
//   the window, a polygon to publish, and events (re-anchor this frame, the layout just froze) for
//   the caller to apply. No callbacks, no back-pointer: a seam made of data cannot be half-wired,
//   which is the failure mode that cost this codebase an evening when a hand-over was declared and
//   silently never called.
//
// ⚠ TODO, deliberately not built: a slow refinement channel for LOCALIZING — a low-rate thread that
//   accumulates residuals against the frozen layout and proposes small LOCAL amendments (a column
//   never seen from the exploring trajectory, an alcove behind furniture, an extent cut short by
//   occlusion) without reopening the global estimate. The state machine is what makes that safe to
//   contemplate: a local amendment is evidence about ONE edge, while re-derivation replaces the whole
//   cycle, and it was re-derivation storms that destroyed a correct map on 2026-09-12.

#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <vector>

#include "wall_map.h"

namespace rc
{
    class LayoutEstimator
    {
    public:
        enum class State { Searching, Localizing };

        /// What one observe() produced, for the caller to fold into its own window.
        struct Observation
        {
            const wallmap::FrameResult* frame = nullptr;   ///< associations, births, deaths
        };

        /// What one after_solve() produced. Every field is something the LOCALISER must act on; the
        /// estimator itself has already done everything that is its own business.
        struct Update
        {
            wallmap::Polygon raw;              ///< the derived polygon — in-loop consumers want THIS
            wallmap::Polygon published;        ///< the Manhattan-projected one that leaves the agent
            bool reanchor_requested = false;   ///< the frame must move; the localiser owns that
            Eigen::Vector2f reanchor_c{0.f, 0.f};
            float           reanchor_rot = 0.f;
            bool froze_this_frame = false;     ///< LOCALIZING was entered on this call
        };


        // ── PHASE 1 OF 2, AND THE HEADER SAYS SO RATHER THAN PRETENDING OTHERWISE ────────────────
        // What exists today is the BOUNDARY: the two states named as one type, and the shape of the
        // data that must cross the seam. RoomConcept currently still owns the WallMap and the two
        // bodies (wall_slam_observe, ~226 lines; wall_slam_after_solve, ~275) and uses this State.
        // Phase 2 moves those in. It is deliberately not done in the same commit, because the move
        // has ONE ordering subtlety that cannot be checked by compiling:
        //
        //   after_solve() decides the frame must be re-anchored, but the re-anchor belongs to the
        //   LOCALISER — it moves the window poses, the model's pose tensors, the covariance and the
        //   object landmarks as well as the wall map. So the call becomes two phase:
        //   after_solve() returns `reanchor_requested`, the localiser applies it, and the estimator
        //   then recomputes its polygon IN THE NEW FRAME. Today that recompute sits mid-function,
        //   three lines after the re-anchor call, where the ordering is implicit.
        //
        // Get that sequence wrong and the published layout is one frame stale in the old frame —
        // metres out, briefly, and only while re-anchoring, which is exactly the kind of defect this
        // session has repeatedly shown compiles cleanly and shows up as something else entirely.
        // It wants a validation run beside it, not a late-night commit.
    };
}   // namespace rc
