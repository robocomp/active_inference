#pragma once
// entry_reloc.h — when the ENTRY RELOCALISATION (RoomSceneGraph::dsr_update_entry_relocalisation) hands afford_room back to the
// room's planners. Pure, tested in tools/entry_reloc_test.cpp.
//
// After a crossing, afford_room is the free standpoint nearest the new room's centroid: relocalise away from the walls before
// anything else. It ended ONLY when the robot came within half its diagonal of that point. explore_spike1 (2026-11-05): the
// controller handed the centre back ("no route … the body cannot stand there") and drove to repaired standpoints 0.65 m away,
// so the relocalisation never ended, the layout explorer stayed locked out for 750 s, its `finished` never came and the
// crossing-born room was never promoted. The consumer's own answer is the other way out: once it has COMPLETED the claim —
// satisfied, or handed back as refused / infeasible / unreachable / outside the room — it has done what it can with the centre,
// and a better standpoint is the planners' job. A protocol event, not a threshold.
//
// ★ WHAT THE CENTRE IS WORTH (clean47, 2026-11-11). The centre target was offered at a FIXED gain of 1.0 and held the layout
// explorer out until it was reached. In room_1 the chair / table looks (1.2-5 nats) won every selection for 5.5 min, afford_room
// was chosen 0 times, the layout explorer never ran, and the room was never promoted although its outline had been right for
// minutes — while the pose sat at sigma 3 mm, as good as it ever gets. A trip to the centre is worth the POSE INFORMATION it can
// still add: half the log-ratio of the pose covariance now to what this localiser TYPICALLY attains inside a room (the median
// log-det of the previous room's steady state — measured, not tuned; the median because sigma_theta swings ~5x cycle to cycle).
// Zero when the pose is already as good as typical: the relocalisation is then over and the room's planners take afford_room.
// While it lasts the target is offered at that gain, in nats, the currency of every other look.
#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace rc::entry
{
[[nodiscard]] constexpr bool still_relocalising(bool at_centre, bool claim_completed, float gain_nats = 1.f)
{
    return not at_centre and not claim_completed and gain_nats > 0.f;
}

/// The localiser's typical in-room pose log-det: samples of the room being left, its median the reference for the next.
struct PoseReference
{
    std::vector<float> samples;          ///< ln det(pose covariance), steady state of the current room
    std::optional<float> reference;      ///< median of the previous room's samples

    void add(float logdet) { if (std::isfinite(logdet)) samples.push_back(logdet); }
    /// At a crossing: the room left becomes the reference (kept if it gave no samples), and the new room starts empty.
    void close_room()
    {
        if (not samples.empty())
        {
            auto s = samples;
            const auto mid = s.begin() + static_cast<std::ptrdiff_t>(s.size() / 2);
            std::ranges::nth_element(s, mid);
            reference = *mid;
        }
        samples.clear();
    }
    /// Expected pose information of a well-placed view, nats: 0 with no reference (no evidence the pose is degraded).
    [[nodiscard]] float gain(float logdet_now) const
    {
        if (not reference.has_value() or not std::isfinite(logdet_now)) return 0.f;
        return std::max(0.f, 0.5f * (logdet_now - *reference));
    }
};
}  // namespace rc::entry
