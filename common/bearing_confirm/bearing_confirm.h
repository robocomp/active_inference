/*
 * common/bearing_confirm/bearing_confirm.h
 *
 * Part C (confirm half) of the RGB-360 peripheral-detection pipeline — see
 * active_inference/RICOH_360_PERIPHERAL_DETECTION.md.
 *
 * A ricoh 360 detection is a BEARING only (no depth): the retina publishes it into the shared "masks"
 * node as a no-depth slice tagged with a room-frame azimuth (Part B). This header matches those bearings
 * to the agent's LIVE instances by predicted azimuth from the robot position — the peripheral-vision "it's
 * still there" signal. It does NOT fit anything (there is nothing 3D to fit) and it does NOT birth anything
 * (a single bearing can't place an object at an unknown range — that is Part C-birth / Part D).
 *
 * Object-AGNOSTIC and header-only (Eigen only): the agent supplies its live tracks (room-frame centres,
 * reusing rc::TrackView from instance_tracker.h), the bearing slices, and the robot's room-frame position.
 * The angular gate mirrors the tracker's position gate style (a soft innovation gate, not a new hard
 * cutoff): a bearing confirms the nearest-in-angle track within `angular_gate_rad`, greedily 1-to-1.
 *
 * The agent acts on the result by HOLDING the confirmed track's death-miss this cycle (set the TrackView's
 * expected_visible=false before InstanceTracker::update — a 360 glance is evidence the object was NOT
 * removed, exactly like being out of the zed frustum), and by logging the match so the (provisional)
 * azimuth convention can be verified live before any harder consumer relies on it.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

#include "../instance_tracker/instance_tracker.h"   // rc::TrackView

namespace rc {

// One RGB-360 bearing detection (a no-depth mask slice of the agent's label). slice_index maps back to
// the packet; azimuth_room_rad is the absolute room-frame direction robot→object published in Part B.
struct BearingDetectionView
{
    float azimuth_room_rad = 0.0f;
    int   slice_index      = -1;
};

// A bearing that lined up with a live instance (peripheral confirmation of existence).
struct BearingConfirm
{
    std::uint64_t track_id       = 0;
    int           track_index    = -1;   // index into the `tracks` vector passed in
    int           slice_index    = -1;   // the confirming bearing's packet slice
    float         innovation_rad = 0.0f; // |observed − predicted| azimuth after wrap
};

// Wrap an angle to (-π, π].
inline float wrap_angle(float a) { return std::atan2(std::sin(a), std::cos(a)); }

// Match bearings to tracks by predicted azimuth from the robot position. Greedy lowest-innovation-first,
// each track and bearing used at most once. Returns the confirmed (track, bearing) pairs.
inline std::vector<BearingConfirm> confirm_tracks_by_bearing(
    const std::vector<TrackView>&            tracks,
    const std::vector<BearingDetectionView>& bearings,
    const Eigen::Vector2f&                   robot_xy,
    float                                    angular_gate_rad)
{
    struct Pair { float innov; int t; int b; };
    std::vector<Pair> pairs;
    for (int t = 0; t < static_cast<int>(tracks.size()); ++t)
    {
        const Eigen::Vector2f d = tracks[t].xy - robot_xy;
        if (d.squaredNorm() < 1e-8f)          // robot on top of the track → bearing undefined
            continue;
        const float predicted = std::atan2(d.y(), d.x());
        for (int b = 0; b < static_cast<int>(bearings.size()); ++b)
        {
            const float e = std::abs(wrap_angle(bearings[b].azimuth_room_rad - predicted));
            if (e <= angular_gate_rad)
                pairs.push_back({e, t, b});
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b){ return a.innov < b.innov; });

    std::vector<char> t_used(tracks.size(), 0), b_used(bearings.size(), 0);
    std::vector<BearingConfirm> out;
    for (const auto& p : pairs)
    {
        if (t_used[p.t] or b_used[p.b])
            continue;
        t_used[p.t] = 1;
        b_used[p.b] = 1;
        out.push_back({tracks[p.t].id, p.t, bearings[p.b].slice_index, p.innov});
    }
    return out;
}

// ─── Bearing-only NEW-object hypothesis staging (Part C-birth) ────────────────────────────────────────
// A 360 bearing that matches NO live track is a candidate for something new. A single frame isn't enough
// (clutter, a spurious detection), so stage it: a bearing starts/advances a streak; a streak that survives
// `birth_frames` promotes → the agent births a broad-Σ hypothesis (see <obj>_belief::seed_bearing). Streaks
// are GAP-TOLERANT (a bearing may drop out for up to `max_miss` cycles without resetting) because peripheral
// 360 detection is intermittent — especially near the panorama seam. Object-agnostic, stateful (held by the
// worker across cycles).
class BearingHypothesisStager
{
public:
    void set_params(int birth_frames, float match_rad, int max_miss)
    { birth_frames_ = std::max(1, birth_frames); match_rad_ = match_rad; max_miss_ = std::max(0, max_miss); }

    // Feed the UNMATCHED bearing azimuths (those confirm_tracks_by_bearing did NOT consume) once per cycle.
    // Returns the azimuths that JUST promoted this cycle (birth them, then they're cleared from staging).
    std::vector<float> update(const std::vector<float>& unmatched_azimuths)
    {
        for (auto& c : cands_) c.seen = false;
        for (const float az : unmatched_azimuths)
        {
            int best = -1; float best_d = match_rad_;
            for (int i = 0; i < static_cast<int>(cands_.size()); ++i)
            {
                if (cands_[i].seen) continue;
                const float d = std::abs(wrap_angle(az - cands_[i].azimuth));
                if (d <= best_d) { best_d = d; best = i; }
            }
            if (best >= 0) { cands_[best].azimuth = az; ++cands_[best].streak; cands_[best].miss = 0; cands_[best].seen = true; }
            else           cands_.push_back({az, 1, 0, true});
        }
        std::vector<float> promoted;
        std::vector<Cand> survivors;
        for (auto& c : cands_)
        {
            if (not c.seen and ++c.miss > max_miss_) continue;       // dropped: gone too long
            if (c.streak >= birth_frames_) { promoted.push_back(c.azimuth); continue; }   // promoted: leaves staging
            survivors.push_back(c);
        }
        cands_.swap(survivors);
        return promoted;
    }

    void reset() { cands_.clear(); }

private:
    struct Cand { float azimuth; int streak; int miss; bool seen; };
    std::vector<Cand> cands_;
    int   birth_frames_ = 8;
    float match_rad_    = 0.15f;
    int   max_miss_     = 3;
};

}  // namespace rc
