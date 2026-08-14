/*
 * birth_fragment.h — retain the PROBATION BURST of a pending birth (shared, header-only)
 *
 * The evidence counterpart of common/instance_tracker/birth_evidence.h. `InstanceTracker` decides WHEN a
 * candidate is promoted and `rc::birth::evidence` decides what each frame CONTRIBUTES to that decision;
 * this keeps the OBSERVATIONS THEMSELVES while the candidate matures, so the instance that is finally
 * created can be built from — and judged on — everything that was seen, not just the last frame.
 *
 * WHY. A candidate matures over birth_frames observations, and until now the tracker carried only a streak
 * counter and the latest xy. Every mask cloud examined on the way was discarded, so an agent paid the full
 * latency of a probation window and then seeded the new instance from ONE frame's centroid, falling back on
 * hardcoded default geometry for everything else. Two things were lost:
 *
 *   1. GEOMETRY. The burst is a multi-viewpoint cloud of the object; its union spans extents no single
 *      partial view does. Seeding from it starts the belief at the object instead of at a constant.
 *   2. THE RIGHT TO REFUSE. With the burst in hand, a birth can be REJECTED by the same plausibility the
 *      agent already applies to a fitted instance — before anything reaches the graph. Without it, a bad
 *      hypothesis must be created first and retracted later by the existence channel.
 *
 * This is the "object fragment" of Khronos (Schmid et al., RSS 2024, arXiv:2402.13817) reduced to the one
 * place this codebase lacked it. There, measurements Z are grouped into fragments Y that are kept intact
 * until extraction, and the object's representation — and whether it is an object at all — is decided then,
 * with all of Z̄_k known. Everything else that formulation buys (recursive belief, covariance-gated
 * association, evidence-of-absence removal) this codebase already has, and better.
 *
 * LOCAL CONSISTENCY. A burst is only meaningful if it is one view of one object: Khronos admits observations
 * into a fragment while consecutive ones are < δ apart, on the argument that state-estimation error and scene
 * change are both small over a short interval. span_ok() is that δ. A candidate whose observations are spread
 * over too long a window (a blob that flickered in and out across a room traverse) is not locally consistent
 * and its burst must not be treated as a single object's surface.
 *
 * Points are stored in whatever frame the caller banks them in — for a room-frame agent that is directly
 * fusable across frames with no transform chaining. Dedup is by quantised cell key, so re-observing the
 * same surface from a similar viewpoint costs nothing and a burst stays bounded regardless of frame rate.
 *
 * Pure: Eigen only, no DSR, no mask type, no agent config — unit-testable in isolation.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

namespace rc {

// FNV-1a hash of a point quantised to a `quantization_m` grid — O(1) dedup key for accumulated clouds.
// Shared with the per-instance support banks so a burst handed to a fitter dedups identically to the bank
// it seeds (rc::support_bank::key delegates here).
inline std::uint64_t cell_key(const Eigen::Vector3f& point, float quantization_m)
{
    const float q = std::max(1e-4f, quantization_m);
    const int ix = static_cast<int>(std::floor(point.x() / q));
    const int iy = static_cast<int>(std::floor(point.y() / q));
    const int iz = static_cast<int>(std::floor(point.z() / q));

    std::uint64_t h = 1469598103934665603ULL;  // FNV-1a offset basis
    auto mix = [&](std::uint64_t v) { h ^= v; h *= 1099511628211ULL; };
    mix(static_cast<std::uint64_t>(ix));
    mix(static_cast<std::uint64_t>(iy));
    mix(static_cast<std::uint64_t>(iz));
    return h;
}

// Everything one pending candidate has been seen to be, accumulated across the frames it matured over.
struct Burst
{
    std::vector<Eigen::Vector3f>      pts;              // de-duplicated union of every observation's cloud
    std::unordered_set<std::uint64_t> keys;             // cell keys present in `pts`
    int                               n_obs   = 0;      // observations banked (frames that contributed)
    std::uint64_t                     t_first = 0;      // capture stamp of the first (ms)
    std::uint64_t                     t_last  = 0;      // capture stamp of the last (ms)
    bool                              capped  = false;  // hit max_pts: `pts` is a prefix, not the full union

    bool empty() const { return pts.empty(); }
    // Wall-clock span of the burst (ms). 0 for a single observation.
    std::uint64_t span_ms() const { return t_last >= t_first ? t_last - t_first : 0; }
};

// Per-candidate burst store. Keyed by InstanceTracker's candidate id (TrackerResult::cand_of_det).
class BirthFragment
{
public:
    // Bank one observation's cloud against a maturing candidate. De-duplicated by cell key and hard-capped
    // at max_pts, so a candidate that lingers cannot grow without bound. Stamps are the mask packet's
    // capture time (NOT the compute cycle) so span_ms() measures the observations, not our scheduling.
    void accumulate(std::uint64_t cand_id, std::span<const Eigen::Vector3f> pts,
                    std::uint64_t stamp_ms, float cell_m, std::size_t max_pts)
    {
        if (cand_id == 0) return;

        auto& b = bursts_[cand_id];
        if (b.n_obs == 0) b.t_first = stamp_ms;
        b.t_last = stamp_ms;
        ++b.n_obs;

        const std::size_t cap = std::max<std::size_t>(1, max_pts);
        for (const auto& p : pts)
        {
            if (not p.allFinite()) continue;            // a NaN would poison every fit seeded from this burst
            if (b.pts.size() >= cap) { b.capped = true; break; }
            if (b.keys.insert(cell_key(p, cell_m)).second)
                b.pts.push_back(p);
        }
    }

    // Free the bursts of candidates the tracker dropped (blob gone). Called every cycle with
    // TrackerResult::expired_candidates — without it the store grows for the life of the process.
    void expire(std::span<const std::uint64_t> ids)
    {
        for (const auto id : ids) bursts_.erase(id);
    }

    // Hand over a promoted candidate's burst and remove it from the store. nullopt if nothing was banked
    // (e.g. every observation was a bearing-only slice with no 3-D support).
    std::optional<Burst> take(std::uint64_t cand_id)
    {
        const auto it = bursts_.find(cand_id);
        if (it == bursts_.end()) return std::nullopt;
        Burst out = std::move(it->second);
        bursts_.erase(it);
        if (out.empty()) return std::nullopt;
        return out;
    }

    // Local consistency (Khronos δ): are these observations close enough together in time to be one view of
    // one object? delta_ms == 0 disables the test. A burst that fails is still a valid birth trigger — the
    // streak matured — but its cloud must not be fused as a single surface.
    static bool span_ok(const Burst& b, std::uint64_t delta_ms)
    {
        return delta_ms == 0 or b.span_ms() <= delta_ms;
    }

    std::size_t size() const { return bursts_.size(); }
    void clear() { bursts_.clear(); }

    // Self-test: dedup, capping, expiry, take-erases, δ. Returns true on success.
    static bool self_test();

private:
    std::unordered_map<std::uint64_t, Burst> bursts_;
};

inline bool BirthFragment::self_test()
{
    bool ok = true;
    auto check = [&](bool c, const char* m) { if (!c) { ok = false; std::printf("  FAIL: %s\n", m); } };

    // Dedup: the same surface re-observed twice occupies the bank once, but counts as two observations.
    {
        BirthFragment f;
        const std::vector<Eigen::Vector3f> cloud{{0.00f, 0.0f, 0.0f}, {0.01f, 0.0f, 0.0f}, {1.00f, 0.0f, 0.0f}};
        f.accumulate(7, cloud, 1000, 0.05f, 1000);
        f.accumulate(7, cloud, 1100, 0.05f, 1000);
        const auto b = f.take(7);
        check(b.has_value(), "dedup: burst present");
        check(b and b->pts.size() == 2, "dedup: 0.00/0.01 share a 5 cm cell, 1.00 is separate");
        check(b and b->n_obs == 2, "dedup: both observations counted");
        check(b and b->span_ms() == 100, "dedup: span spans first→last stamp");
        check(f.size() == 0, "take() erases the entry");
        check(not f.take(7).has_value(), "take() is idempotent-empty");
    }

    // Non-finite points are refused before they can reach a fit.
    {
        BirthFragment f;
        const std::vector<Eigen::Vector3f> cloud{{std::nanf(""), 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}};
        f.accumulate(1, cloud, 0, 0.05f, 1000);
        const auto b = f.take(1);
        check(b and b->pts.size() == 1, "NaN point rejected, finite one kept");
    }

    // Cap: the burst stops growing and says so.
    {
        BirthFragment f;
        std::vector<Eigen::Vector3f> cloud;
        for (int i = 0; i < 50; ++i) cloud.emplace_back(static_cast<float>(i), 0.0f, 0.0f);
        f.accumulate(2, cloud, 0, 0.05f, 10);
        const auto b = f.take(2);
        check(b and b->pts.size() == 10, "cap: honoured");
        check(b and b->capped, "cap: flagged");
    }

    // Expiry frees; an unbanked or zero id is a no-op.
    {
        BirthFragment f;
        const std::vector<Eigen::Vector3f> cloud{{0.0f, 0.0f, 0.0f}};
        f.accumulate(3, cloud, 0, 0.05f, 100);
        f.accumulate(4, cloud, 0, 0.05f, 100);
        f.accumulate(0, cloud, 0, 0.05f, 100);          // id 0 = "no candidate": must not be stored
        check(f.size() == 2, "expiry: id 0 ignored");
        const std::uint64_t gone[] = {3, 99};
        f.expire(gone);
        check(f.size() == 1, "expiry: known id freed, unknown id harmless");
    }

    // δ: a burst spread across a long window is not locally consistent.
    {
        Burst b; b.t_first = 1000; b.t_last = 1400;
        check(BirthFragment::span_ok(b, 0),    "delta: 0 disables the test");
        check(BirthFragment::span_ok(b, 500),  "delta: 400 ms burst inside a 500 ms window");
        check(not BirthFragment::span_ok(b, 300), "delta: 400 ms burst outside a 300 ms window");
    }

    std::printf("BirthFragment::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
