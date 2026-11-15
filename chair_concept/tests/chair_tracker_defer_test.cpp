// chair_tracker_defer_keeps_candidate — Phase 0 review focus 4: a candidate must SURVIVE the WAIT.
//
// While the room has no usable outline, chair_concept holds births (spec: WAIT, never guess). The tracker's pending
// candidate must keep its evidence during the hold, so that it is born the cycle the outline arrives — at the
// tracked position — instead of being lost and re-accumulated from zero. Run: ./chair_tracker_defer_test
#include <cstdio>
#include <vector>

#include <Eigen/Core>

#include "instance_tracker/instance_tracker.h"

static int fails = 0;
#define CHECK(cond, msg) do { const bool ok_ = (cond); std::printf("  %-60s %s\n", msg, ok_ ? "ok" : "FAIL"); if (not ok_) ++fails; } while (0)

int main()
{
    rc::TrackerParams tp;
    tp.birth_frames = 8.0f;          // chair's Tracker.BirthFrames default
    rc::InstanceTracker tracker;
    tracker.set_params(tp);

    rc::DetectionView dv;
    dv.xy = Eigen::Vector2f(1.2f, -0.7f);
    dv.slice_index = 0;
    const std::vector<rc::TrackView> no_tracks;
    const std::vector<rc::DetectionView> dets{dv};

    std::printf("chair_tracker_defer_keeps_candidate\n");
    int births_while_held = 0;
    const int held_frames = static_cast<int>(tp.birth_frames) + 2;
    for (int f = 0; f < held_frames; ++f)
        births_while_held += static_cast<int>(tracker.update(no_tracks, dets, /*hold_births=*/true).births.size());
    CHECK(births_while_held == 0, "no birth while births are held");

    const auto res = tracker.update(no_tracks, dets, /*hold_births=*/false);
    CHECK(res.births.size() == 1, "birth on the NEXT frame once births are allowed");
    CHECK(not res.births.empty() and res.births.front() == 0, "the birth is the tracked detection");
    CHECK(res.birth_cand_ids.size() == 1 and res.birth_cand_ids.front() != 0, "it carries the held candidate's id");

    // Classic path unchanged: without a hold, the same detection is born at birth_frames, not before.
    rc::InstanceTracker fresh;
    fresh.set_params(tp);
    int first_birth = -1;
    for (int f = 0; f < held_frames and first_birth < 0; ++f)
        if (not fresh.update(no_tracks, dets).births.empty()) first_birth = f + 1;
    CHECK(first_birth == static_cast<int>(tp.birth_frames), "without a hold: born at exactly birth_frames");

    std::printf(fails == 0 ? "chair_tracker_defer_test: PASS\n" : "chair_tracker_defer_test: %d FAIL\n", fails);
    return fails == 0 ? 0 : 1;
}
