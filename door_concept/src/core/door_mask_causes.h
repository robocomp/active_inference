// door_mask_causes.h — every door-class mask is explained by exactly one of FOUR causes (one-model, spec §3.2):
//   Aperture  door i's frame / shut leaf (the fit's own mixture at its believed state),
//   Leaf      door i's leaf marginalised over φ × hinge × BOTH swing sides (DoorBelief::leaf_mixture, the four LeafChannel filters),
//   NewDoor   the birth density (a door nobody has yet),
//   Clutter   κ (a spurious door-class mask).
// The responsibilities ρ(m → cause) are the normalised products prior × likelihood (bernoulli_filter.h responsibilities). One
// computation replaces the swept-arc veto, the claimed-slice veto, the greedy pairing and the hard "is this an existing door"
// gates: a mask a known door's leaf explains gives "new door" little responsibility, continuously.
//
// CURRENCY. A mask is ONE observation: its residual across the surface it is attributed to, scored by the mean over its points of
// the fit's own unnormalised point mixture u (door_belief.cpp mixture_unnorm: u_surface = (1 − ε) exp(−d²/2R), 1 = dead on the
// surface). Each cause is a DENSITY in that residual (units 1/m), so causes are comparable without a hand-set scale:
//   door i :  r_i · p_D,i · exp(mean log u) · (R/R_eff)^½ / √(2π R_eff) · exp(−½ δ²/σ²_along),  R_eff = R + σ²_across (the door's
//             uncertainty across its wall convolves the kernel) and δ the mask's offset along the wall against the door's σ_along —
//             what the old code spelt as one Mahalanobis radius;
//   new    :  b · p_D · (1 − ε) / Λ        b = the wall's base-rate probability, Λ = the length of the wall the mask lies on;
//   clutter:  κ · (1 − ε) / Λ              κ per view, spread over the SAME measure (uniform along the wall, spec §3.4).
// New and clutter both fit the mask perfectly (a spurious mask is as door-shaped as a real new one): only their PRIORS, b against
// κ, tell them apart, and the geometry decides which EXISTING door, if any, owns the mask. A door's prior mass r·p_D is split
// between its two appearances by P(leaf open) (the φ posteriors): shut, the leaf IS the aperture; open, the leaf is where the
// mask is — no door is counted twice. The resolution-cell factor 1/√(2πR) is what makes a mask exactly on a known door
// overwhelmingly that door's, which a per-place likelihood ratio (hit/FA) alone cannot say.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <Eigen/Dense>

#include "bernoulli_filter.h"
#include "../door_belief.h"

namespace rc::door::core
{

enum class CauseKind : std::uint8_t { Aperture, Leaf, NewDoor, Clutter };

struct Cause
{
    CauseKind kind;
    std::uint64_t door = 0;
    double log_w = 0;   // log g + log(r · p_D) or log b or log κ
};

struct MaskCauses
{
    int slice = -1;
    std::vector<Cause> causes;
    std::vector<double> rho;   // normalised responsibilities, parallel to `causes`
    int new_leaf_hyp = -1;     // the new-door cause's best LEAF-mode hypothesis (index into MaskCauseModel::leaf_log_w) when it outweighs the aperture mode, else -1
    double new_leaf_share = 0; // P(leaf mode | new door): the leaf hypotheses' share of the new-door weight

    /// Σ ρ over a door's causes (aperture + leaf): how much of this mask door `id` explains.
    [[nodiscard]] double rho_door(std::uint64_t id) const
    {
        double s = 0.0;
        for (std::size_t i = 0; i < causes.size(); ++i)
            if ((causes[i].kind == CauseKind::Aperture or causes[i].kind == CauseKind::Leaf) and causes[i].door == id)
                s += rho[i];
        return s;
    }
    /// The share of a mask that is a DOOR's or a new door's, clutter set aside: ρ renormalised over the non-clutter causes. Clutter is NOT a
    /// competitor for a place's hit weight — its rate enters the place once, through the ledger's false-alarm rate (hit_llr = log h/f).
    [[nodiscard]] double non_clutter() const
    {
        const double c = rho_kind(CauseKind::Clutter);
        return c < 1.0 ? 1.0 - c : 0.0;
    }
    [[nodiscard]] double hit_door(std::uint64_t id) const { const double n = non_clutter(); return n > 0.0 ? rho_door(id) / n : 0.0; }
    [[nodiscard]] double hit_new() const { const double n = non_clutter(); return n > 0.0 ? rho_kind(CauseKind::NewDoor) / n : 0.0; }
    [[nodiscard]] double rho_kind(CauseKind k) const
    {
        double s = 0.0;
        for (std::size_t i = 0; i < causes.size(); ++i)
            if (causes[i].kind == k)
                s += rho[i];
        return s;
    }
};

/// One door (or restored door) as a cause of masks.
struct DoorCauseInput
{
    std::uint64_t id = 0;
    const rc::DoorBelief* belief = nullptr;            // the fit's belief; null = no geometry to score (no cause)
    const rc::door::LeafChannel* leaf = nullptr;       // the four hinge × swing φ filters; null/uninitialised = aperture only
    double log_r = 0.0;                                // log existence probability (last cycle's p)
    double log_pd = 0.0;                               // log P(this sensor reports it | it exists, the mask is there)
    float place_var = 0.0f;                            // the door's variance ACROSS its wall (recess + pose chain): convolves the SDF kernel
    float along_offset_m = 0.0f;                       // the mask centroid's offset from the aperture centre ALONG the wall
    float along_var = 0.0f;                            // the door's variance along its wall (belief σ_s² + pose chain) + the detection's: 0 = no term
    float phi_max_rad = 2.0944f;                       // the φ grid of the leaf filters
};

/// The priors and scales of the two non-door causes.
struct MaskCauseModel
{
    double log_birth = -4.0;        // log b: the probability a door is born at this place (place_prior; the per-wall base rate charges a second one)
    double log_kappa = -1.6;        // log κ: the false-alarm rate of door-class masks (the fitted detector table's false_alarm)
    double log_pd_new = 0.0;        // log P(the sensor reports a new door | it exists) (the detector's hit rate)
    double along_len_m = 5.0;       // the length of the wall the mask lies on: births AND spurious masks are uniform along it (spec §3.4)
    // The new-door cause has two modes (spec §3.6): the mask is the aperture's own face (a shut door in its wall: the weight above), or the
    // mask is a LEAF seen from outside its frame wall (a side-on leaf, door_leaf_birth.h). Both are in the same currency, the density of a
    // door born with this mask, so the cause's weight is their sum.
    double log_aperture_factor = 0.0;   // log of the aperture mode's angular likelihood (door_leaf_birth.h aperture_angle_likelihood); 0 = no information
    std::vector<double> leaf_log_w;     // one per leaf-mode hypothesis (−inf = none): log b_hyp − log Λ_hyp + log P(open) + log(1 − aperture angular) + the LiDAR opening;
                                        // causes_of_mask adds log p_D + log(1 − ε), exactly as for the aperture mode
};

/// One mask's causes: every door within likelihood reach (log density not below log κ − 20 nats, a NUMERICAL floor on a weight
/// that would otherwise underflow, not a decision), the new-door term, clutter.
/// `pts`/`R`: the mask's points and the fit's R plus the slice's depth variance.
[[nodiscard]] inline MaskCauses causes_of_mask(int slice, const std::vector<Eigen::Vector3f>& pts, float R,
                                               std::span<const DoorCauseInput> doors, const MaskCauseModel& model,
                                               const rc::DoorBeliefParams& bp)
{
    MaskCauses out;
    out.slice = slice;
    constexpr double kTwoPi = 6.283185307179586;
    const double eps = std::clamp(static_cast<double>(bp.clutter_frac), 1e-6, 0.99);   // clamp: the mixture's own domain (door_belief.cpp)
    const double Rr = std::max(1e-9, static_cast<double>(R));
    const double log_clutter = model.log_kappa + std::log1p(-eps) - std::log(std::max(1e-9, model.along_len_m));
    const double reach_floor = log_clutter - 20.0;
    for (const auto& d : doors)
    {
        if (d.belief == nullptr or pts.empty())
            continue;
        const float Reff = R + std::max(0.0f, d.place_var);
        const double norm = 0.5 * std::log(Rr / std::max(1e-9, static_cast<double>(Reff))) - 0.5 * std::log(kTwoPi * Reff);
        double p_open = 0.0;
        double log_leaf = -std::numeric_limits<double>::infinity();
        if (d.leaf != nullptr)
        {
            const auto m = d.belief->leaf_mixture(pts, Reff, *d.leaf, d.phi_max_rad);
            if (m.ok)
            {
                p_open = m.p_open;
                log_leaf = m.mean_log_u + norm;
            }
        }
        const double log_ap = d.belief->aperture_loglik(pts, Reff) + norm;
        // where the mask sits along the wall against where the door may be along it: what the old Mahalanobis radius said, as a likelihood
        const double along = d.along_var > 0.0f ? -0.5 * double(d.along_offset_m) * d.along_offset_m / double(d.along_var) : 0.0;
        const double base = d.log_r + d.log_pd + along;
        if (p_open < 1.0 and log_ap >= reach_floor)
            out.causes.push_back({CauseKind::Aperture, d.id, base + std::log1p(-p_open) + log_ap});
        if (p_open > 0.0 and log_leaf >= reach_floor)
            out.causes.push_back({CauseKind::Leaf, d.id, base + std::log(p_open) + log_leaf});
    }
    const double log_new_ap = model.log_birth + model.log_pd_new + std::log1p(-eps) - std::log(std::max(1e-9, model.along_len_m)) + model.log_aperture_factor;
    double log_new = log_new_ap;
    if (not model.leaf_log_w.empty())
    {
        const double fixed = model.log_pd_new + std::log1p(-eps);
        double mx = log_new_ap;
        for (const double v : model.leaf_log_w) mx = std::max(mx, v + fixed);
        double z = std::isfinite(log_new_ap) ? std::exp(log_new_ap - mx) : 0.0, zl = 0.0, best = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < model.leaf_log_w.size(); ++i)
            if (const double v = model.leaf_log_w[i] + fixed; std::isfinite(v))
            {
                zl += std::exp(v - mx);
                if (v > best) { best = v; out.new_leaf_hyp = static_cast<int>(i); }
            }
        if (zl > 0.0)
        {
            log_new = mx + std::log(z + zl);
            out.new_leaf_share = zl / (z + zl);
            if (not (best > log_new_ap))
                out.new_leaf_hyp = -1;   // the aperture mode is the better single hypothesis: the door is born at the mask
        }
    }
    out.causes.push_back({CauseKind::NewDoor, 0, log_new});
    out.causes.push_back({CauseKind::Clutter, 0, log_clutter});
    std::vector<double> lw(out.causes.size());
    for (std::size_t i = 0; i < lw.size(); ++i)
        lw[i] = out.causes[i].log_w;
    out.rho = bf::responsibilities(lw);
    return out;
}

}   // namespace rc::door::core
