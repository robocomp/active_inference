// place_table.h — Task 7 of the door-existence model: the PLACE TABLE. One row per door place per room. The row IS the
// belief's memory: L without the structural prior (L_evidence), the view ledger, last seen, visit. Rows are NEVER deleted:
// a removed phantom's row (removed_wall_ms set) means "no door here" and is the prior for any new candidate at that place.
// It replaced the RAM registry of removed doors and the existence half of the identity table (deleted in Task 10).
// Header-only, DSR-free. Persisted through door_identity_table.h's row (columns ledger, visit, removed_ms; all optional).
//
// match():   minimum SQUARED Mahalanobis distance m2 with the SUM of both covariances (row + candidate), m2 <= gate. `gate` is the tracker's gate_mahalanobis, which is ALWAYS m2 (9.0 = 3 sigma).
// Hazard:    how fast a remembered place changes. Gamma prior, shape 1, mean 1/600 s^-1 (rate parameter 600 s). Posterior mean
//            lambda = (1 + changes) / (600 + sum elapsed_s); a revisit that does NOT find the door counts as a change.
//            ★1/600 is only the PRIOR'S MEAN (the old fixed restore hazard, now gone); it is not a constant in use.
//            It is learned from revisits (note_revisit).
// aged_L():  L_evidence * exp(-lambda * dt): the remembered belief relaxes toward 0 (ignorance). The structural prior is added
//            by the CALLER. Every row ages alike, a removed one included (controller ruling R15, spec 3.7): a deletion is evidence, not a fact.
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include "view_ledger.h"
#include "place_belief.h"
#include "../door_identity_table.h"

namespace rc::door::core
{
struct PlaceRow
{
    std::string room, name;
    Vec2f xy = Vec2f::Zero(); Eigen::Matrix2f cov = Eigen::Matrix2f::Identity(); float yaw = 0.f, width = 0.f, height = 0.f;
    float depth = 0.f;                    // the aperture's wall depth, set by the caller from the fit; 0 = unknown
    float L_evidence = 0.f;               // ledger sum + crossing evidence, WITHOUT the structural prior
    ViewLedger ledger; int visit = 0; std::int64_t last_seen_wall_ms = 0;
    std::optional<std::int64_t> removed_wall_ms;   // a deleted phantom: "no door here", kept for ever
};

class PlaceTable
{
public:
    explicit PlaceTable(bool geometry_may_refute) : geometry_may_refute_(geometry_may_refute) {}

    [[nodiscard]] std::span<const PlaceRow> rows() const { return rows_; }

    PlaceRow* match(std::string_view room, const Vec2f& xy, const Eigen::Matrix2f& cov, float gate)
    {
        PlaceRow* best = nullptr; float dbest = gate;   // squared
        for (auto& r : rows_)
        {
            if (r.room != room) continue;
            const Eigen::Matrix2f S = r.cov + cov;
            if (not (S.determinant() > 0.f) or not S.allFinite()) continue;
            const Vec2f d = xy - r.xy;
            const float m2 = d.dot(S.inverse() * d);
            if (not (m2 >= 0.f)) continue;
            if (m2 <= dbest) { dbest = m2; best = &r; }
        }
        return best;
    }

    /// Insert, or replace the row of the same (room, name). Never deletes.
    PlaceRow& upsert(PlaceRow r)
    {
        const auto it = std::ranges::find_if(rows_, [&](const PlaceRow& o) { return o.room == r.room and o.name == r.name; });
        if (it != rows_.end()) { *it = std::move(r); return *it; }
        rows_.push_back(std::move(r));
        return rows_.back();
    }

    /// One revisit of a remembered place after `elapsed_s`: found=false is a change. Updates the shared change hazard.
    /// The hazard is table-wide, so no row is needed.
    void note_revisit(bool found, double elapsed_s)
    {
        if (not std::isfinite(elapsed_s) or elapsed_s < 0.0) return;
        if (not found) changes_ += 1.0;
        sum_elapsed_s_ += elapsed_s;
    }

    [[nodiscard]] double hazard_per_s() const { return (kPriorShape + changes_) / (kPriorRateS + sum_elapsed_s_); }

    [[nodiscard]] float aged_L(const PlaceRow& r, std::int64_t now_wall_ms) const
    {
        const double dt = std::max<double>(0.0, double(now_wall_ms - r.last_seen_wall_ms) * 1e-3);
        return float(double(r.L_evidence) * std::exp(-hazard_per_s() * dt));
    }

    /// Load rows from a parsed identity table. Rows without a pose cannot be places (no size) and are skipped.
    /// Old rows (no ledger column) get an empty ledger and exist_logodds as L. `fallback_sd`: the place's sigma (m) for a
    /// row that does not carry place_sd (caller supplies the tracker's seed sigma).
    void load(const ParsedIdentityTable& t, float fallback_sd)
    {
        for (const auto& i : t.rows) upsert(row_of(i, fallback_sd, geometry_may_refute_));
    }

    /// One identity row as a place row (the restore path reads the same thing: the row IS the belief's memory).
    [[nodiscard]] static PlaceRow row_of(const IdentityRow& i, float fallback_sd, bool geometry_may_refute)
    {
        PlaceRow r; r.room = i.room; r.name = i.name; r.xy = i.xy;
        const float sd = i.place_sd.value_or(fallback_sd);
        r.cov = Eigen::Matrix2f::Identity() * sd * sd;
        if (i.pose) { r.yaw = i.pose->yaw; r.width = i.pose->width; r.height = i.pose->height; r.depth = i.pose->depth; }
        r.L_evidence = i.exist_logodds.value_or(0.f);
        r.visit = i.visit.value_or(0);
        r.last_seen_wall_ms = i.seen_wall_ms.value_or(0);
        r.removed_wall_ms = i.removed_ms;
        if (i.ledger)
            if (auto L = ViewLedger::parse(*i.ledger, geometry_may_refute)) r.ledger = std::move(*L);   // corrupt cell: empty ledger
        return r;
    }

    /// A restore (Task 9): the belief comes back from the row. Its ledger is the row's; the evidence term is the row's aged L_evidence
    /// (hazard Gamma prior, learned from revisits), less the ledger sum that L() adds back. The structural prior re-enters from the room.
    [[nodiscard]] PlaceBelief restored_belief(const PlaceRow& r, std::int64_t now_wall_ms) const
    {
        PlaceBelief b;
        b.ledger = r.ledger;
        b.carried = aged_L(r, now_wall_ms) - r.ledger.sum();
        return b;
    }

    [[nodiscard]] std::vector<IdentityRow> to_identity_rows() const
    {
        std::vector<IdentityRow> out;
        for (const auto& r : rows_)
        {
            IdentityRow i; i.room = r.room; i.name = r.name; i.xy = r.xy;
            if (r.width > 0.f and r.height > 0.f and r.depth > 0.f) i.pose = RestorePose{r.yaw, r.width, r.height, r.depth};   // never fabricate geometry
            i.exist_logodds = r.L_evidence;
            if (r.last_seen_wall_ms > 0) i.seen_wall_ms = r.last_seen_wall_ms;
            i.place_sd = std::sqrt(std::max(1e-12f, 0.5f * (r.cov(0, 0) + r.cov(1, 1))));
            if (not r.ledger.views().empty()) i.ledger = r.ledger.serialise();
            i.visit = r.visit; i.removed_ms = r.removed_wall_ms;
            out.push_back(std::move(i));
        }
        return out;
    }

private:
    // Gamma(shape 1, rate 600 s) prior on the change hazard: mean 1/600 s^-1 (see the header comment).
    static constexpr double kPriorShape = 1.0, kPriorRateS = 600.0;
    bool geometry_may_refute_;
    std::vector<PlaceRow> rows_;
    double changes_ = 0.0, sum_elapsed_s_ = 0.0;
};

/// Task 9: the belief of a door CARRIED across a room switch (the `exit` fact). The new room's ledger is empty (views are per room).
///  - with the source face's row: prior = 0 (the new room's prior is entered by the first update, which REPLACES it), carried = the
///    face's L_evidence, crossing = crossing_llr;
///  - without one (the face was never known): prior = the new room's place prior (`new_room_prior`, default none yet), carried = 0.
[[nodiscard]] inline PlaceBelief carried_belief(const PlaceRow* src, float crossing_llr, const PlacePrior& new_room_prior = {})
{
    PlaceBelief b;
    if (src) b.carried = src->L_evidence; else b.prior = new_room_prior;
    b.crossing = crossing_llr;
    return b;
}
}  // namespace rc::door::core
