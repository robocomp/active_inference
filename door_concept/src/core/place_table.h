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
#include <set>
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
    float height_sd = 0.f;                // the fit's posterior sd of `height` (m); 0 = unknown (a row loaded from the CSV, which has no column for it)
    float L_evidence = 0.f;             // PlaceBelief::L_evidence(): everything but the structural prior
    float physical = 0.f;                 // PlaceBelief::Sp(): the physical part of L_evidence (what a memory keeps outside vision's saturation, place_belief.h)
    bool crossed = false;                 // proof, for ever: a crossed place has no delete line (place_belief.h)
    ViewLedger ledger; int visit = 0; std::int64_t last_seen_wall_ms = 0;
    std::int64_t sighted_wall_ms = 0;     // the last SIGHTING (commands move last_seen, not this): the hazard's exposure counts from here
    std::optional<std::int64_t> removed_wall_ms;   // a deleted phantom: "no door here", kept for ever
};

class PlaceTable
{
public:
    explicit PlaceTable(bool geometry_may_refute, float lookalike = kLookalikeProbDefault) : geometry_may_refute_(geometry_may_refute), lookalike_(lookalike) {}

    [[nodiscard]] std::span<const PlaceRow> rows() const { return rows_; }
    [[nodiscard]] bool geometry_may_refute() const { return geometry_may_refute_; }
    [[nodiscard]] float lookalike() const { return lookalike_; }

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
    /// The hazard is table-wide, so no row is needed. Callers go through sample_revisit (one sample per place per visit).
    void note_revisit(bool found, double elapsed_s)
    {
        if (not std::isfinite(elapsed_s) or elapsed_s < 0.0) return;
        if (not found) changes_ += 1.0;
        sum_elapsed_s_ += elapsed_s;
    }

    /// ONE SAMPLE PER PLACE PER VISIT (the hazard is learned per visit of a place, never from a re-detection inside one visit):
    /// found = the place was matched again in that visit; not found = it was in view and removed in that visit. A place that has
    /// already given its sample this visit gives none (a removal followed by a re-detection is one visit, one sample). The elapsed
    /// exposure runs from the row's last SIGHTING to now_wall_ms. Returns whether a sample was taken.
    /// ★A REVISIT IS OF A PLACE WRITTEN IN AN EARLIER VISIT (re-review I2, ruling): the row's visit must be < `visit` at the moment of
    ///  sampling. A row written THIS visit (a door removed a moment ago, under its own name) is no revisit: its re-birth is a flicker,
    ///  not a change. Visit ids are session-scoped (DoorCore starts counting after the loaded table's largest visit, max_visit()), so
    ///  a row from an earlier session is always earlier.
    bool sample_revisit(const PlaceRow& row, int visit, bool found, std::int64_t now_wall_ms)
    {
        if (not (row.visit < visit)) return false;
        if (row.sighted_wall_ms <= 0 or now_wall_ms <= row.sighted_wall_ms) return false;
        if (not sampled_.emplace(row.room + "/" + row.name, visit).second) return false;
        note_revisit(found, double(now_wall_ms - row.sighted_wall_ms) * 1e-3);
        return true;
    }
    /// The largest visit id in the table (rows and their ledger views): a new session counts its visits after it.
    [[nodiscard]] int max_visit() const
    {
        int m = 0;
        for (const auto& r : rows_)
        {
            m = std::max(m, r.visit);
            for (const auto& v : r.ledger.views()) m = std::max(m, static_cast<int>(v.key.visit));
        }
        return m;
    }
    [[nodiscard]] PlaceRow* find(std::string_view room, std::string_view name)
    {
        const auto it = std::ranges::find_if(rows_, [&](const PlaceRow& r) { return r.room == room and r.name == name; });
        return it == rows_.end() ? nullptr : &*it;
    }
    /// A crossing at `xy` in `room` lifts every removal of that hole (rows within `same_hole_m`): "no door here" was wrong there. Returns
    /// how many rows were lifted. The rows stay (names, ledgers, revisit history); only their removal mark goes.
    int lift_refutations(std::string_view room, const Vec2f& xy, float same_hole_m)
    {
        int n = 0;
        for (auto& r : rows_)
            if (r.room == room and r.removed_wall_ms.has_value() and r.xy.allFinite() and (r.xy - xy).norm() < same_hole_m)
            { r.removed_wall_ms.reset(); ++n; }
        return n;
    }
    /// The learned hazard's state (changes, summed exposure in s): persisted with the identity table (`#hazard` line).
    [[nodiscard]] ParsedIdentityTable::Hazard hazard_state() const { return {changes_, sum_elapsed_s_}; }
    void set_hazard(const ParsedIdentityTable::Hazard& h) { changes_ = h.changes; sum_elapsed_s_ = h.sum_elapsed_s; }

    [[nodiscard]] double hazard_per_s() const { return (kPriorShape + changes_) / (kPriorRateS + sum_elapsed_s_); }

    [[nodiscard]] double age_factor(const PlaceRow& r, std::int64_t now_wall_ms) const
    {
        const double dt = std::max<double>(0.0, double(now_wall_ms - r.last_seen_wall_ms) * 1e-3);
        return std::exp(-hazard_per_s() * dt);
    }
    [[nodiscard]] float aged_L(const PlaceRow& r, std::int64_t now_wall_ms) const { return float(double(r.L_evidence) * age_factor(r, now_wall_ms)); }

    /// ★THE HEIGHT A PLACE WAS MEASURED AT IS PART OF ITS MEMORY (2026-11-13, clean60 door_7). A pending candidate measures its blob's
    ///  WIDTH but no height (door_existence.cpp prepare_candidates: its height is the prior's, so the height half of the size term is 0),
    ///  while a born door's fit measures both. A door refuted by its HEIGHT — clean60's GlassBay pane, fitted 1.00 x 2.62 m (σ_h 0.024 m)
    ///  under a 2.10 ± 0.08 m prior: size −19.7 against vision +1.5 — is removed, and the row keeps only L_evidence, which EXCLUDES the size
    ///  term (it is part of the structural prior). So the removed row said "door-shaped" (+), the next candidate at the place — still unable
    ///  to see the height — was born again, refuted again: 18 births and removals in a row, each one a node, an affordance and a target the
    ///  controller chased. The measured height is evidence about the OBJECT at the place, so the memory supplies it as the candidate's height
    ///  measurement: {height, sd}, with the precision aged by the same change hazard that ages the row's L (sd / √age_factor), so a stale
    ///  memory fades to no information instead of refuting for ever. `unknown_sd`: the sd for a row that carries none (a loaded row).
    ///  nullopt: no height remembered, or a memory aged to nothing (sd → ∞: the term is 0 — the limit, not a cutoff).
    struct RememberedHeight { float height = 0.f, sd = 0.f; };
    [[nodiscard]] std::optional<RememberedHeight> remembered_height(const PlaceRow& r, std::int64_t now_wall_ms, float unknown_sd) const
    {
        if (not (std::isfinite(r.height) and r.height > 0.f)) return std::nullopt;
        const double g = age_factor(r, now_wall_ms);
        if (not (g > 0.0)) return std::nullopt;
        const float sd = r.height_sd > 0.f ? r.height_sd : unknown_sd;
        return RememberedHeight{r.height, float(double(sd) / std::sqrt(g))};
    }

    /// Load rows from a parsed identity table. Rows without a pose cannot be places (no size) and are skipped.
    /// Old rows (no ledger column) get an empty ledger and exist_logodds as L. `fallback_sd`: the place's sigma (m) for a
    /// row that does not carry place_sd (caller supplies the tracker's seed sigma).
    void load(const ParsedIdentityTable& t, float fallback_sd)
    {
        for (const auto& i : t.rows) upsert(row_of(i, fallback_sd, geometry_may_refute_, lookalike_));
        if (t.hazard.has_value()) set_hazard(*t.hazard);   // SET, not added: loading twice is the same table
    }

    /// One identity row as a place row (the restore path reads the same thing: the row IS the belief's memory).
    /// ★A row without `exist_physical` (written before the look-alike model, 2026-11-08): its exist_logodds is the OLD unbounded sum, which is on the Sc
    /// scale, not on the saturated one: it is saturated once here (L_evidence = its ledger's physical part + saturated rest). A phantom remembered at +89 comes back near the bound.
    [[nodiscard]] static PlaceRow row_of(const IdentityRow& i, float fallback_sd, bool geometry_may_refute, float lookalike = kLookalikeProbDefault)
    {
        PlaceRow r; r.room = i.room; r.name = i.name; r.xy = i.xy;
        const float sd = i.place_sd.value_or(fallback_sd);
        r.cov = Eigen::Matrix2f::Identity() * sd * sd;
        if (i.pose) { r.yaw = i.pose->yaw; r.width = i.pose->width; r.height = i.pose->height; r.depth = i.pose->depth; }
        r.L_evidence = i.exist_logodds.value_or(0.f);
        r.visit = i.visit.value_or(0);
        r.last_seen_wall_ms = i.seen_wall_ms.value_or(0);
        r.removed_wall_ms = i.removed_ms;
        r.crossed = i.crossed.value_or(false);
        r.sighted_wall_ms = i.sighted_ms.value_or(i.seen_wall_ms.value_or(0));
        if (i.ledger)
            if (auto L = ViewLedger::parse(*i.ledger, geometry_may_refute)) r.ledger = std::move(*L);   // corrupt cell: empty ledger
        if (i.exist_physical.has_value())
            r.physical = *i.exist_physical;
        else
        {
            r.physical = r.ledger.physical_sum();
            if (i.exist_logodds.has_value())
                r.L_evidence = r.physical + float(saturated_vision(double(*i.exist_logodds) - r.physical, lookalike_lambda(lookalike, 0.0)));   // (the room's base rate is not known here)
        }
        return r;
    }

    /// A restore (Task 9): the belief comes back from the row. Its ledger is the row's; the evidence term is the row's aged L_evidence
    /// (hazard Gamma prior, learned from revisits), split into its physical and perceptual parts and entered so that L_evidence() equals it (PlaceBelief::remember:
    /// the ledger's views stay inside the saturation, so a revisit's new views cannot lift it above the bound). The structural prior re-enters from the room.
    void restore_into(PlaceBelief& b, const PlaceRow& r, std::int64_t now_wall_ms) const
    {
        const double g = age_factor(r, now_wall_ms);
        b.ledger = r.ledger;
        b.lookalike = lookalike_;
        b.crossed = r.crossed;
        b.remember(float(double(r.L_evidence) * g), float(double(r.physical) * g));
    }
    [[nodiscard]] PlaceBelief restored_belief(const PlaceRow& r, std::int64_t now_wall_ms) const
    {
        PlaceBelief b;
        restore_into(b, r, now_wall_ms);
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
            i.exist_physical = r.physical;
            if (r.crossed) i.crossed = true;
            if (r.last_seen_wall_ms > 0) i.seen_wall_ms = r.last_seen_wall_ms;
            i.place_sd = std::sqrt(std::max(1e-12f, 0.5f * (r.cov(0, 0) + r.cov(1, 1))));
            if (not r.ledger.views().empty()) i.ledger = r.ledger.serialise();
            i.visit = r.visit; i.removed_ms = r.removed_wall_ms;
            if (r.sighted_wall_ms > 0) i.sighted_ms = r.sighted_wall_ms;
            out.push_back(std::move(i));
        }
        return out;
    }

private:
    // Gamma(shape 1, rate 600 s) prior on the change hazard: mean 1/600 s^-1 (see the header comment).
    static constexpr double kPriorShape = 1.0, kPriorRateS = 600.0;
    bool geometry_may_refute_;
    float lookalike_;
    std::vector<PlaceRow> rows_;
    double changes_ = 0.0, sum_elapsed_s_ = 0.0;
    std::set<std::pair<std::string, int>> sampled_;   // (room/name, visit) pairs that already gave their revisit sample (this process)
};

/// Task 9: the belief of a door CARRIED across a room switch (the `exit` fact). The new room's ledger is empty (views are per room).
///  - with the source face's row: prior = 0 (the new room's prior is entered by the first update, which REPLACES it), carried = the
///    face's L_evidence, crossing = crossing_llr;
///  - without one (the face was never known): prior = the new room's place prior (`new_room_prior`, default none yet), carried = 0.
[[nodiscard]] inline PlaceBelief carried_belief(const PlaceRow* src, float crossing_llr, const PlacePrior& new_room_prior = {},
                                                float lookalike = kLookalikeProbDefault)
{
    PlaceBelief b;
    b.lookalike = lookalike;
    if (src) b.remember(src->L_evidence, src->physical); else b.prior = new_room_prior;
    b.crossed = src != nullptr and src->crossed;     // a proof already standing is carried; this crossing's waits for the confirmation (place_belief.h)
    if (crossing_llr > 0.f) b.note_crossing();
    b.crossing = crossing_llr;
    return b;
}
}  // namespace rc::door::core
