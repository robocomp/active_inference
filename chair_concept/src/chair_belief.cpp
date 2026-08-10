/*
 * chair_belief.cpp  —  AI2 chair belief (pose-only [cx,cy,yaw]; size is a fixed template).
 */

#include "chair_belief.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

namespace rc
{

namespace
{
// Exact box SDF from per-axis face distances (|local| − half_extent).
float box_sdf(float dx, float dy, float dz)
{
    const float ox = std::max(dx, 0.0f), oy = std::max(dy, 0.0f), oz = std::max(dz, 0.0f);
    const float outside = std::sqrt(ox * ox + oy * oy + oz * oz);
    const float inside  = std::min(std::max(dx, std::max(dy, dz)), 0.0f);
    return outside + inside;
}
}  // namespace

Eigen::Vector2f ChairBelief::leg_center_local(int k) const
{
    static const std::array<std::array<float, 2>, 4> sg = {{ {1, 1}, {-1, 1}, {-1, -1}, {1, -1} }};
    const float ox = 0.5f * params_.tpl_seat_w - params_.leg_half;
    const float oy = 0.5f * params_.tpl_seat_d - params_.leg_half;
    return {sg[k][0] * ox, sg[k][1] * oy};
}

// ─── SDF primitives (fixed template dims, placed by the pose (cx,cy,yaw); cz pinned to floor) ─────

float ChairBelief::sdf_seat(const Eigen::Vector3f& p, const ChairBeliefState& s) const
{
    const float c = std::cos(-s.yaw), sn = std::sin(-s.yaw);
    const float px = p.x() - s.cx, py = p.y() - s.cy;
    const float lx = px * c - py * sn, ly = px * sn + py * c, lz = p.z() - params_.floor_z;
    const float half_t = 0.5f * params_.seat_thickness;
    const float seat_cz = params_.tpl_seat_h - half_t;
    return box_sdf(std::abs(lx) - 0.5f * params_.tpl_seat_w,
                   std::abs(ly) - 0.5f * params_.tpl_seat_d,
                   std::abs(lz - seat_cz) - half_t);
}

float ChairBelief::sdf_back(const Eigen::Vector3f& p, const ChairBeliefState& s) const
{
    const float c = std::cos(-s.yaw), sn = std::sin(-s.yaw);
    const float px = p.x() - s.cx, py = p.y() - s.cy;
    const float lx = px * c - py * sn, ly = px * sn + py * c, lz = p.z() - params_.floor_z;
    const float half_t = 0.5f * params_.seat_thickness;
    const float back_cy = -0.5f * params_.tpl_seat_d + half_t;   // backrest on the −y seat edge (fixes yaw)
    const float back_cz = params_.tpl_seat_h + 0.5f * params_.tpl_back_h;
    return box_sdf(std::abs(lx) - 0.5f * params_.tpl_seat_w,
                   std::abs(ly - back_cy) - half_t,
                   std::abs(lz - back_cz) - 0.5f * params_.tpl_back_h);
}

float ChairBelief::sdf_leg(const Eigen::Vector3f& p, const ChairBeliefState& s, int k) const
{
    const float c = std::cos(-s.yaw), sn = std::sin(-s.yaw);
    const float px = p.x() - s.cx, py = p.y() - s.cy;
    const float lx = px * c - py * sn, ly = px * sn + py * c, lz = p.z() - params_.floor_z;
    const auto  ctr = leg_center_local(k);
    const float hh  = leg_half_height();
    return box_sdf(std::abs(lx - ctr.x()) - params_.leg_half, std::abs(ly - ctr.y()) - params_.leg_half,
                   std::abs(lz - hh) - hh);
}

float ChairBelief::sdf_prim(const Eigen::Vector3f& p, const ChairBeliefState& s, int prim) const
{
    if (prim == 0) return sdf_seat(p, s);
    if (prim == 1) return sdf_back(p, s);
    return sdf_leg(p, s, prim - 2);
}

float ChairBelief::sdf_compound(const Eigen::Vector3f& p, const ChairBeliefState& s) const
{
    float m = std::min(sdf_seat(p, s), sdf_back(p, s));
    for (int k = 0; k < 4; ++k) m = std::min(m, sdf_leg(p, s, k));
    return m;
}

// ─── Mixture responsibilities (z-band part attribution) ──────────────────────────

std::array<float, 7> ChairBelief::mixture_unnorm(const Eigen::Vector3f& p, const ChairBeliefState& s, float R) const
{
    const float eps     = std::clamp(params_.clutter_frac, 0.0f, 0.99f);
    const float pi_surf = (1.0f - eps) / 6.0f;
    const float inv2R   = 0.5f / std::max(1e-9f, R);

    // Parts occupy disjoint vertical bands (legs below the seat, backrest above). Gate each component's
    // weight by a smooth z-membership about the two join planes so a seat-edge point near a corner leg is
    // attributed to the right part. Physical, not a threshold. Bands are fixed (seat_h is a constant now).
    const float lz     = p.z() - params_.floor_z;
    const float z_low  = params_.tpl_seat_h - params_.seat_thickness;   // leg top == seat bottom
    const float z_high = params_.tpl_seat_h;                            // seat top == backrest bottom
    const float band   = std::max(0.03f, 0.5f * params_.seat_thickness);
    const float leg_g  = 1.0f / (1.0f + std::exp((lz - z_low) / band));
    const float back_g = 1.0f / (1.0f + std::exp((z_high - lz) / band));
    const float seat_g = (1.0f - leg_g) * (1.0f - back_g);

    std::array<float, 7> u{};
    u[0] = pi_surf * seat_g * std::exp(-sdf_seat(p, s) * sdf_seat(p, s) * inv2R);
    u[1] = pi_surf * back_g * std::exp(-sdf_back(p, s) * sdf_back(p, s) * inv2R);
    for (int k = 0; k < 4; ++k)
    {
        const float d = sdf_leg(p, s, k);
        u[2 + k] = pi_surf * leg_g * std::exp(-d * d * inv2R);
    }
    const float cs = params_.clutter_scale_m;
    u[6] = eps * std::exp(-cs * cs * inv2R);
    return u;   // UNNORMALIZED
}

std::array<float, 7> ChairBelief::responsibilities(const Eigen::Vector3f& p, const ChairBeliefState& s, float R) const
{
    std::array<float, 7> u = mixture_unnorm(p, s, R);
    float sum = 0.0f;
    for (float v : u) sum += v;
    if (sum <= 0.0f) { u = {}; u[6] = 1.0f; return u; }
    for (float& v : u) v /= sum;
    return u;
}

float ChairBelief::mean_energy(const std::vector<Eigen::Vector3f>& pts, const ChairBeliefState& s, float R) const
{
    if (pts.empty()) return 0.0f;
    double e = 0.0;
    for (const auto& p : pts)
    {
        const auto r = responsibilities(p, s, R);
        for (int prim = 0; prim < 6; ++prim)
        {
            const float d = sdf_prim(p, s, prim);
            e += 0.5 * r[prim] * d * d / R;
        }
    }
    return static_cast<float>(e / static_cast<double>(pts.size()));
}

// Mean per-point mixture NLL (includes clutter) — the association evidence. A far / wrong-instance slice
// (all points → flat clutter) has a tiny mixture likelihood → HIGH nll → not claimed (unlike mean_energy,
// which scores a fully-clutter'd far slice ~0 = a perfect match). The common Gaussian normaliser cancels in
// the argmin, so the raw unnormalised sum suffices.
float ChairBelief::mixture_nll(const std::vector<Eigen::Vector3f>& pts, const ChairBeliefState& s, float R) const
{
    if (pts.empty()) return 0.0f;
    double nll = 0.0;
    for (const auto& p : pts)
    {
        const auto u = mixture_unnorm(p, s, R);
        double sum = 0.0;
        for (const float v : u) sum += v;
        nll += -std::log(std::max(1e-30, sum));
    }
    return static_cast<float>(nll / static_cast<double>(pts.size()));
}

float ChairBelief::association_nll(const std::vector<Eigen::Vector3f>& pts, float R) const
{
    return mixture_nll(pts, state_, R);
}

float ChairBelief::clutter_fraction(const std::vector<Eigen::Vector3f>& pts, float R) const
{
    if (pts.empty()) return 0.0f;
    double c = 0.0;
    for (const auto& p : pts) c += responsibilities(p, state_, R)[6];   // index 6 = clutter
    return static_cast<float>(c / static_cast<double>(pts.size()));
}

// ─── Yaw disambiguation: 4-way sequential Bayesian model comparison ──────────────

bool ChairBelief::resolve_orientation(const std::vector<Eigen::Vector3f>& pts, float R, float evidence_weight,
                                      float view_azimuth)
{
    if (pts.empty()) return false;
    constexpr float kHalfPi = 0.5f * static_cast<float>(M_PI);
    // THRESHOLD (flagged, physical): clamp each accumulated mode-evidence to ±kFlipClamp so a long run of
    // confident frames can't drive flip_acc_ hundreds of nats deep and make the belief unable to RECANT when
    // the chair is physically rotated. kFlipClamp≈6 nats ≈ a 400:1 mode odds — decisive but recoverable.
    // Mirrors TableBelief's ±6 clamp on flip_evidence_. (Now a class constant: the rig prior's cap is
    // derived from it — see kRigModeShare.)
    const float w = std::clamp(evidence_weight, 0.0f, 1.0f);
    const auto wrap = [](float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); };

    // Compare orientation modes by the PROPER mixture NLL (includes clutter), NOT mean_energy: mean_energy
    // omits the clutter term, so an orientation that swings a primitive onto EMPTY space (its points fall to
    // clutter → 0 energy) scores LOWER than the true pose that explains the real backrest — which flipped
    // every chair's backrest 180° onto empty space. The NLL penalises unexplained points, so the orientation
    // that explains the real backrest wins. Same fix already applied to association (mean_energy→nll).
    const float e0 = mixture_nll(pts, state_, R);

    // ★OBSERVABILITY of the yaw mode in THIS frame. Only the backrest breaks the 4-fold symmetry —
    // the square seat, the symmetric legs and the clutter term are all invariant under a 90° rotation,
    // so they cancel exactly in the differences below. The honest weight is therefore the backrest
    // point mass actually observed, saturating: past `mode_sat_back_pts` more of the same surface is
    // redundant (it shares the frame's common-mode error), exactly as in the continuous channel.
    float frame_w = w;
    if (params_.mode_obs_weighting)
    {
        float n_back = 0.0f;
        for (const auto& p : pts)
            n_back += responsibilities(p, state_, R)[1];   // [seat, BACK, leg0..3, clutter]
        const float obs_w = n_back / (n_back + std::max(1e-3f, params_.mode_sat_back_pts));

        // ★NOVELTY as a SATURATING PER-VIEWPOINT BUDGET (was 1/(1+visits) until 2026-07-30).
        // The intent is unchanged: dwell must not manufacture certainty, because 2000 identical frames share
        // one common-mode error and are nowhere near 2000 independent observations. But dividing EVERY frame
        // by the visit count made a viewpoint's TOTAL contribution grow only as log(n), which denies the
        // other half of the requirement — a DELIBERATE FIXATION is an intentional epistemic action and its
        // evidence must be able to settle the pose. Under log(n) the robot could park and stare for three
        // minutes at a chair whose model was visibly ~45° off and still not resolve it.
        // So give each bearing bin a BUDGET: the total weight it may ever contribute saturates at
        // view_budget, approached asymptotically. Each frame is scaled by what is left of its bin's budget,
        //     novelty = remaining / (remaining + view_budget),   remaining = max(0, budget − spent)
        // ⇒ a sustained fixation converges to that viewpoint's full information worth (decisive), staring on
        // past it adds ~nothing (dwell still cannot inflate confidence), and orbiting to a NEW bearing opens
        // a fresh budget (genuinely new evidence still accumulates). Both halves of the rule.
        // ★AN UNKNOWN VIEWPOINT FAILS CLOSED. This was `novelty = 1.0f` with the NaN case falling through at
        // FULL weight — the one branch where we cannot tell a fresh view from the same view repeated, given
        // the discovery rate. view_azimuth is NaN until the first extrinsic resolves, i.e. exactly at
        // startup, when the accumulator is EMPTY and a full-weight vote is at its most decisive. Charge it as
        // a REPEAT instead: attribute the look to the bin already spent the most, which also makes the branch
        // self-limiting (successive unknown looks deplete that bin, novelty decays to ~0). Same fix as
        // RefrigeratorBelief::resolve_front, whose budget was modelled on this one.
        constexpr int   B    = ChairBeliefParams::kViewBins;
        const float     span = 2.0f * static_cast<float>(M_PI) / static_cast<float>(B);
        int bin = 0;
        if (std::isfinite(view_azimuth))
            bin = std::clamp(static_cast<int>(std::floor((wrap(view_azimuth) + static_cast<float>(M_PI)) / span)),
                             0, B - 1);
        else
            bin = static_cast<int>(std::distance(view_spent_.begin(),
                                   std::max_element(view_spent_.begin(), view_spent_.end())));
        const float budget    = std::max(1e-3f, params_.view_budget);
        const float remaining = std::max(0.0f, budget - view_spent_[bin]);
        const float novelty   = remaining / (remaining + budget);
        view_spent_[bin] += w * obs_w * novelty;   // charge the bin what this frame actually contributed
        ++view_visits_[bin];                       // retained for diagnostics only
        frame_w = w * obs_w * novelty;
    }

    for (int k = 1; k < 4; ++k)
    {
        ChairBeliefState rot = state_;
        rot.yaw = wrap(state_.yaw + static_cast<float>(k) * kHalfPi);
        // Weighted log-odds accumulation. With mode_obs_weighting the weight is a genuine measure of
        // the frame's independent information (ego-motion × backrest observability × viewpoint
        // novelty); without it, this is the legacy dwell-time sum.
        flip_acc_[k] += frame_w * (mixture_nll(pts, rot, R) - e0);   // >0 current better, <0 mode k better
        flip_acc_[k] = std::clamp(flip_acc_[k], -kFlipClamp, kFlipClamp);
    }
    // ★Pick the mode on accumulated DATA evidence + the rig's structural prior, evaluated HERE and
    // not folded into flip_acc_. flip_acc_ accumulates over frames; a prior added to it each frame
    // would be counted once per frame and manufacture certainty within seconds. Adding it only at
    // comparison time means the data keeps accumulating while the prior contributes its honest
    // single-observation weight — and the moment the rig retracts (κ→0) the decision reverts to the
    // data alone, with no residue left behind in the accumulator.
    int kbest = 0;
    for (int k = 1; k < 4; ++k)
        if (flip_acc_[k] + rig_mode_cost(k) < flip_acc_[kbest] + rig_mode_cost(kbest)) kbest = k;
    if (kbest == 0)
        return false;

    state_.yaw = wrap(state_.yaw + static_cast<float>(kbest) * kHalfPi);   // pure yaw change (size is fixed)
    const std::array<float, 4> old = flip_acc_;
    for (int m = 0; m < 4; ++m)
        flip_acc_[m] = std::clamp(old[(m + kbest) % 4] - old[kbest], -kFlipClamp, kFlipClamp);   // re-baseline
    // The viewpoint history counted evidence for the OLD mode labelling; after adopting a new mode the
    // next frames are genuinely informative again about the new hypothesis, so let them count.
    view_visits_.fill(0);
    return true;
}

// ─── Rig arrangement prior on the discrete yaw mode ──────────────────────────────────────────────
// von-Mises log-density of mode k, up to a constant: −κ·cos(yaw_k − ψ_rig). Returned on the same
// LOWER-IS-BETTER scale as flip_acc_, so the two simply add. κ = 0 ⇒ identically 0 for every mode,
// which leaves both the mode pick and mode_posterior() bit-for-bit as they were without a rig.
float ChairBelief::rig_mode_cost(int k) const
{
    if (rig_yaw_kappa_ <= 0.0f)
        return 0.0f;
    constexpr float kHalfPi = 0.5f * static_cast<float>(M_PI);

    // ★The prior may BREAK A TIE; it may never OUTVOTE saturated data. See kRigKappaMax for the
    // derivation — κ = kFlipClamp is the largest value that still loses to fully-saturated evidence
    // (which spans 2·kFlipClamp), while being enough to overturn the partially-accumulated wrong mode
    // that a backrest-poor view produces. The consumer's configured κ is bounded here, not trusted.
    const float kappa = std::min(rig_yaw_kappa_, kRigKappaMax);

    const float yaw_k = state_.yaw + static_cast<float>(k) * kHalfPi;
    return -kappa * std::cos(yaw_k - rig_yaw_prior_);
}

// ─── Reported orientation-mode uncertainty (fold the discrete-mode entropy into σ_yaw) ───────────
// flip_acc_[k] is the accumulated (E_k − E_0), so LOWER = better; the mode posterior is
// softmax(−(flip_acc_ + rig prior)). ★The rig term belongs here as well as in the mode pick: it is
// part of the belief over modes, so a rig that RESOLVES an otherwise-tied ambiguity must also
// collapse the reported σ_yaw — otherwise the chair keeps advertising an ambiguity it no longer has,
// and the NBV planner keeps spending looks on a question that is already answered.
std::array<float, 4> ChairBelief::mode_posterior() const
{
    std::array<float, 4> p{};
    std::array<float, 4> score{};
    for (int k = 0; k < 4; ++k) score[k] = flip_acc_[k] + rig_mode_cost(k);
    const float mn = *std::min_element(score.begin(), score.end());   // subtract min for stability
    float sum = 0.0f;
    for (int k = 0; k < 4; ++k) { p[k] = std::exp(-(score[k] - mn)); sum += p[k]; }
    if (sum <= 0.0f) { p = {0.25f, 0.25f, 0.25f, 0.25f}; return p; }
    for (float& v : p) v /= sum;
    return p;
}


// REPORTED yaw variance: the within-mode Σ(2,2) plus the variance of the discrete-mode yaw offset Δ_k = k·π/2
// under the mode posterior. All modes tied → ~Var of {0,±π/2,π} ≈ (large, honest ambiguity); resolved → →0.
float ChairBelief::yaw_marginal_var() const
{
    constexpr float kHalfPi = 0.5f * static_cast<float>(M_PI);
    const auto wrap = [](float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); };
    const auto p = mode_posterior();
    float mean = 0.0f, msq = 0.0f;
    for (int k = 0; k < 4; ++k)
    {
        const float d = wrap(static_cast<float>(k) * kHalfPi);   // ∈ (−π,π]
        mean += p[k] * d; msq += p[k] * d * d;
    }
    return Sigma_(2, 2) + std::max(0.0f, msq - mean * mean);
}

Eigen::Matrix<float, 3, 3> ChairBelief::covariance_reported() const
{
    Eigen::Matrix<float, 3, 3> S = Sigma_;
    S(2, 2) = yaw_marginal_var();
    return S;
}

// ─── Jacobian (central finite difference over the 3 pose DOFs) ───────────────────

Eigen::Matrix<float, 3, 1> ChairBelief::sdf_jacobian(const Eigen::Vector3f& p, const ChairBeliefState& s, int prim) const
{
    Eigen::Matrix<float, 3, 1> J;
    const Eigen::Matrix<float, 3, 1> base = s.vec();
    const float e = params_.fd_eps;
    for (int j = 0; j < 3; ++j)
    {
        Eigen::Matrix<float, 3, 1> vp = base, vm = base;
        vp(j) += e; vm(j) -= e;
        J(j) = (sdf_prim(p, ChairBeliefState::from_vec(vp), prim) -
                sdf_prim(p, ChairBeliefState::from_vec(vm), prim)) / (2.0f * e);
    }
    return J;
}

// ─── Constraints (yaw wrap only; position free, size/cz fixed) ───────────────────

void ChairBelief::apply_constraints(ChairBeliefState& s) const
{
    s.yaw = std::remainder(s.yaw, 2.0f * static_cast<float>(M_PI));
}

// ─── Engine hooks: prior cov, process noise, common-mode ─────────────────────────

Eigen::Matrix<float, 3, 1> ChairBelief::prior_cov_diag() const
{
    const float pp = params_.prior_pos_std * params_.prior_pos_std;
    const float py = params_.prior_yaw_std * params_.prior_yaw_std;
    return (Eigen::Matrix<float, 3, 1>() << pp, pp, py).finished();
}

Eigen::Matrix<float, 3, 1> ChairBelief::process_noise_diag() const
{
    const float qm = params_.process_std_m * params_.process_std_m;
    const float qy = params_.process_std_yaw * params_.process_std_yaw;
    return (Eigen::Matrix<float, 3, 1>() << qm, qm, qy).finished();
}

Eigen::Matrix<float, 3, 1> ChairBelief::common_mode_inv_diag(const ChairFrame& frame) const
{
    const float p2 = params_.common_mode_pos_std * params_.common_mode_pos_std;
    const float y2 = params_.common_mode_yaw_std * params_.common_mode_yaw_std;
    return (Eigen::Matrix<float, 3, 1>() <<
            1.0f / std::max(1e-9f, p2 + frame.chain_cov_xx),
            1.0f / std::max(1e-9f, p2 + frame.chain_cov_yy),
            1.0f / std::max(1e-9f, y2 + frame.chain_cov_yaw)).finished();
}

// ─── Self-test ───────────────────────────────────────────────────────────────────

bool ChairBelief::self_test()
{
    std::mt19937 rng(2026);
    std::normal_distribution<float> noise(0.0f, 0.008f);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f), U01(0.0f, 1.0f);

    ChairBeliefParams P;
    const ChairBeliefState gt{0.20f, -0.30f, 0.40f};   // pose ground truth (size = template)

    const float c = std::cos(gt.yaw), sn = std::sin(gt.yaw);
    const auto to_world = [&](float lx, float ly, float lz) -> Eigen::Vector3f
    { return {gt.cx + c * lx - sn * ly, gt.cy + sn * lx + c * ly, P.floor_z + lz}; };

    std::vector<Eigen::Vector3f> pts;
    for (int i = 0; i < 800; ++i)   // seat top
        pts.push_back(to_world(U(rng) * 0.5f * P.tpl_seat_w, U(rng) * 0.5f * P.tpl_seat_d, P.tpl_seat_h + noise(rng)));
    for (int i = 0; i < 500; ++i)   // backrest (−y edge, vertical)
        pts.push_back(to_world(U(rng) * 0.5f * P.tpl_seat_w, -0.5f * P.tpl_seat_d + noise(rng),
                               P.tpl_seat_h + U01(rng) * P.tpl_back_h));
    for (int k = 0; k < 4; ++k)     // legs
    {
        const float sx = (k == 0 || k == 3) ? 1.f : -1.f, sy = (k < 2) ? 1.f : -1.f;
        const float ox = sx * (0.5f * P.tpl_seat_w - P.leg_half), oy = sy * (0.5f * P.tpl_seat_d - P.leg_half);
        for (int i = 0; i < 150; ++i)
            pts.push_back(to_world(ox + P.leg_half * U(rng), oy + P.leg_half * U(rng),
                                   U01(rng) * (P.tpl_seat_h - P.seat_thickness)));
    }
    std::vector<Eigen::Vector3f> clutter;
    for (int i = 0; i < 120; ++i)
        clutter.push_back({gt.cx + U(rng) * 1.2f, gt.cy + U(rng) * 1.2f, P.floor_z + U01(rng) * 0.04f});
    std::vector<Eigen::Vector3f> all = pts;
    all.insert(all.end(), clutter.begin(), clutter.end());

    bool ok = true;
    auto check = [&](bool cond, const char* msg) { if (!cond) { ok = false; std::printf("  FAIL: %s\n", msg); } };

    // (a) SDF sanity
    {
        ChairBelief b(gt, P);
        check(b.sdf_compound(to_world(0, 0, P.tpl_seat_h - 0.5f * P.seat_thickness), gt) < -1e-3f, "seat centre SDF should be inside");
        check(std::abs(b.sdf_seat(to_world(0.2f * P.tpl_seat_w, 0.1f * P.tpl_seat_d, P.tpl_seat_h), gt)) < 1e-2f, "on-seat SDF ~0");
        check(b.sdf_compound({gt.cx + 3.0f, gt.cy, 0.5f}, gt) > 1.0f, "far point SDF large");
    }

    // (b) Pose recovery from a realistic seed: the fitter seeds cx,cy at the cloud centroid + a coarse yaw
    // (a far seed sees every point as clutter → no gradient), then the belief refines. Mirror that.
    Eigen::Vector3f cen = Eigen::Vector3f::Zero();
    for (const auto& q : all) cen += q;
    cen /= static_cast<float>(all.size());
    ChairBelief belief(ChairBeliefState{cen.x(), cen.y(), 0.20f}, P);
    ChairFrame frame; frame.points = all;
    float e = 0.0f;
    for (int it = 0; it < 40; ++it) e = belief.update(frame);
    const auto& s = belief.state();
    std::printf("  recovered: cx=%.3f cy=%.3f yaw=%.3f  (E=%.4f)\n", s.cx, s.cy, s.yaw, e);
    std::printf("  truth:     cx=%.3f cy=%.3f yaw=%.3f\n", gt.cx, gt.cy, gt.yaw);
    check(std::abs(s.cx - gt.cx) < 0.03f, "cx not recovered");
    check(std::abs(s.cy - gt.cy) < 0.03f, "cy not recovered");
    check(std::abs(std::remainder(s.yaw - gt.yaw, 2.0f * static_cast<float>(M_PI))) < 0.10f, "yaw not recovered");

    // (c) association_nll: a far slice must score HIGH (unclaimable); own slice low.
    {
        ChairBelief b(gt, P);
        const float R = P.sigma_base_m * P.sigma_base_m;
        const float e_own = b.association_nll(pts, R);
        std::vector<Eigen::Vector3f> far;
        for (const auto& q : pts) far.push_back({q.x() + 1.0f, q.y(), q.z()});   // shift 1 m
        const float e_far = b.association_nll(far, R);
        std::printf("  association_nll: own=%.2f far(1m)=%.2f (far must be >> own)\n", e_own, e_far);
        check(e_far > e_own + 3.0f, "association_nll does not reject a far slice");
    }

    // (d) 4-way orientation: a correct chair must NOT flip; a 180°-wrong one must restore.
    {
        const float R = P.sigma_base_m * P.sigma_base_m;
        ChairBelief good(gt, P); int gflips = 0;
        for (int i = 0; i < 40; ++i) if (good.resolve_orientation(all, R)) ++gflips;
        ChairBelief wrong(ChairBeliefState{gt.cx, gt.cy, gt.yaw + static_cast<float>(M_PI)}, P);
        int wfirst = -1;
        for (int i = 0; i < 40; ++i) if (wrong.resolve_orientation(all, R) and wfirst < 0) wfirst = i;
        const float dyaw = std::abs(std::remainder(wrong.state().yaw - gt.yaw, 2.0f * static_cast<float>(M_PI)));
        std::printf("  seq-flip: correct→%d flips (want 0)  180°-wrong→restored Δyaw=%.0f° @frame %d\n",
                    gflips, dyaw * 57.2958f, wfirst);
        check(gflips == 0, "resolve_orientation flipped a correct chair (oscillation)");
        check(dyaw < 0.3f, "resolve_orientation did not restore a 180°-wrong chair");
    }

    // (d2) ★RIG YAW PRIOR — the standing-in test for the live 2026-07-26 failure: a chair converged to
    // the right heading, snapped 90°, then held the WRONG mode for 570 cycles and escaped only on a
    // process restart. Reproduced deterministically here by feeding the mode test a BACKREST-BLIND
    // cloud (seat + legs only), which is what a side-on or occluded view leaves: the four modes are
    // then genuinely tied and the chair cannot recover on its own. The rig prior must break the tie —
    // and must do so WITHOUT compounding across frames, and without touching a chair whose own
    // evidence is good.
    {
        const float R = P.sigma_base_m * P.sigma_base_m;

        // Backrest-blind: seat + legs, no backrest points.
        std::vector<Eigen::Vector3f> blind;
        for (int i = 0; i < 800; ++i)
            blind.push_back(to_world(U(rng) * 0.5f * P.tpl_seat_w, U(rng) * 0.5f * P.tpl_seat_d,
                                     P.tpl_seat_h + noise(rng)));

        // (i) Without the rig it stays stuck 90° off — the failure we are fixing.
        ChairBelief stuck(ChairBeliefState{gt.cx, gt.cy, gt.yaw + 0.5f * static_cast<float>(M_PI)}, P);
        for (int i = 0; i < 60; ++i) stuck.resolve_orientation(blind, R);
        const float d_stuck = std::abs(std::remainder(stuck.state().yaw - gt.yaw, 2.0f * static_cast<float>(M_PI)));

        // (ii) With the rig pointing at the true heading, the same chair recovers.
        ChairBelief helped(ChairBeliefState{gt.cx, gt.cy, gt.yaw + 0.5f * static_cast<float>(M_PI)}, P);
        helped.set_rig_yaw_prior(gt.yaw, 8.0f);
        for (int i = 0; i < 60; ++i) helped.resolve_orientation(blind, R);
        const float d_helped = std::abs(std::remainder(helped.state().yaw - gt.yaw, 2.0f * static_cast<float>(M_PI)));

        std::printf("  rig-prior: backrest-blind stuck Δyaw=%.0f°  →  with rig Δyaw=%.0f°\n",
                    d_stuck * 57.2958f, d_helped * 57.2958f);
        check(d_stuck > 1.0f,  "backrest-blind 90°-wrong chair was expected to stay stuck without the rig");
        check(d_helped < 0.3f, "rig yaw prior did not recover a backrest-blind 90°-wrong chair");

        // (iii) ★The prior must NOT out-shout good data. A chair whose backrest IS visible and correct
        // must ignore a rig prior that points 90° the wrong way — the arrangement is a tendency, and
        // a genuinely turned-away chair keeps its own heading.
        ChairBelief defiant(gt, P);
        defiant.set_rig_yaw_prior(gt.yaw + 0.5f * static_cast<float>(M_PI), 8.0f);
        for (int i = 0; i < 60; ++i) defiant.resolve_orientation(all, R);
        const float d_defiant = std::abs(std::remainder(defiant.state().yaw - gt.yaw, 2.0f * static_cast<float>(M_PI)));
        std::printf("  rig-prior: wrong prior vs good backrest evidence → Δyaw=%.0f° (want 0)\n",
                    d_defiant * 57.2958f);
        check(d_defiant < 0.3f, "a WRONG rig prior overrode good backrest evidence");

        // (iii-b) ★THE 180° HOLE. rig_mode_cost = −κ·cos(Δ) gives a differential of κ at 90° but 2κ at
        // 180°, so the 180° case — NOT the 90° one above — is what actually bounds κ. This was untested
        // when the cap was first derived, and the shipped κ=5.4 failed it: a prior pointing 180° away
        // rotated a chair whose backrest was fully seen and correctly explained.
        ChairBelief defiant180(gt, P);
        defiant180.set_rig_yaw_prior(gt.yaw + static_cast<float>(M_PI), 8.0f);   // ask for the cap
        for (int i = 0; i < 60; ++i) defiant180.resolve_orientation(all, R, 1.0f, 0.0f);
        const float d_def180 = std::abs(std::remainder(defiant180.state().yaw - gt.yaw, 2.0f * static_cast<float>(M_PI)));
        std::printf("  rig-prior: 180° prior vs good backrest evidence → Δyaw=%.0f° (want 0)\n",
                    d_def180 * 57.2958f);
        check(d_def180 < 0.3f, "a 180° rig prior overrode good backrest evidence (the 2*kappa hole)");

        // (iv) ★The prior must not COMPOUND across frames. With backrest-blind data flip_acc_ stays
        // ~tied no matter how long we run, so the mode posterior is set by the prior ALONE and must be
        // the SAME after 10 frames as after 400. If the prior were folded into flip_acc_ it would be
        // re-added every frame and the posterior would keep sharpening toward certainty — the exact
        // defect that pinned the rig's own log-odds before it was made a low-pass.
        // Controlled: the same blind cloud run with and without the prior. flip_acc_ drifts on its own
        // (tied data still has tiny sampling asymmetries that accumulate toward the clamp), so the
        // question is not whether p(mode0) moves — it is whether the PRIOR'S CONTRIBUTION grows. The
        // prior is aligned with the current mode, so neither belief ever adopts a different mode and
        // the labels stay comparable.
        ChairBelief ref(gt, P), pri(gt, P);
        pri.set_rig_yaw_prior(gt.yaw, 4.0f);
        for (int i = 0; i < 10; ++i) { ref.resolve_orientation(blind, R); pri.resolve_orientation(blind, R); }
        const float effect_short = pri.mode_posterior()[0] - ref.mode_posterior()[0];
        for (int i = 0; i < 390; ++i) { ref.resolve_orientation(blind, R); pri.resolve_orientation(blind, R); }
        const float effect_long = pri.mode_posterior()[0] - ref.mode_posterior()[0];
        std::printf("  rig-prior: contribution to p(mode0) after 10 frames=%+.4f, after 400=%+.4f (must not grow)\n",
                    effect_short, effect_long);
        check(effect_long <= effect_short + 0.02f,
              "rig prior COMPOUNDS across frames (it leaked into flip_acc_)");
    }

    // (d3) ★THE LIVE FAILURE, 2026-07-27: a chair resolved from a RICH near view must not be rotated
    // by STARVED far frames. Live, chair_3 held the correct −47° on ~1700 points at 3.5 m; the robot
    // drove away and 21- then 29-point frames at 6.5 m flipped it 90° TWICE, after which flip_acc_
    // saturated and it locked. The cause is that mixture_nll is a MEAN per-point NLL, so a 21-point
    // frame's vote is the same size as a 1700-point one. Reproduced here by subsampling the same
    // cloud down to ~20 points, which is what range does to a mask.
    {
        const float R = P.sigma_base_m * P.sigma_base_m;

        // A ~20-point fragment that actively favours a WRONG mode — the whole cloud rotated 90° about
        // the chair centre, then decimated. That is the shape of the live failure: not merely sparse,
        // but sparse AND biased (a far, partly-occluded, clutter-heavy fragment whose handful of points
        // happen to sit where a rotated chair would put them). A uniformly-thinned cloud is NOT a
        // reproduction — it keeps the backrest proportion and still votes correctly.
        std::vector<Eigen::Vector3f> starved;
        for (std::size_t i = 0; i < all.size(); i += all.size() / 20 + 1)
        {
            const Eigen::Vector3f& q = all[i];
            const float dx = q.x() - gt.cx, dy = q.y() - gt.cy;
            starved.emplace_back(gt.cx - dy, gt.cy + dx, q.z());   // rotate +90° about the centre
        }

        // With observability weighting: resolve on the rich view, then stare with the starved one.
        // Viewpoints: the rich views come from one bearing, the starved ones from another (the robot
        // drove away). Passing them is what enables the novelty discount.
        constexpr float kAzRich = 0.0f, kAzFar = 2.0f;
        ChairBelief rich(gt, P);
        for (int i = 0; i < 40; ++i) rich.resolve_orientation(all, R, 1.0f, kAzRich);
        const float yaw_resolved = rich.state().yaw;
        for (int i = 0; i < 2000; ++i) rich.resolve_orientation(starved, R, 1.0f, kAzFar);
        const float drift = std::abs(std::remainder(rich.state().yaw - yaw_resolved, 2.0f * static_cast<float>(M_PI)));

        // Same experiment with the LEGACY running sum, to prove the test discriminates.
        ChairBeliefParams L = P; L.mode_obs_weighting = false;
        ChairBelief legacy(gt, L);
        for (int i = 0; i < 40; ++i) legacy.resolve_orientation(all, R, 1.0f, kAzRich);
        const float legacy_resolved = legacy.state().yaw;
        for (int i = 0; i < 2000; ++i) legacy.resolve_orientation(starved, R, 1.0f, kAzFar);
        const float legacy_drift = std::abs(std::remainder(legacy.state().yaw - legacy_resolved, 2.0f * static_cast<float>(M_PI)));

        std::printf("  starved-view: %zu pts vs %zu — obs-weighted drift=%.0f°  legacy drift=%.0f°\n",
                    starved.size(), all.size(), drift * 57.2958f, legacy_drift * 57.2958f);
        check(drift < 0.3f, "starved far frames rotated a chair resolved from a rich near view");
    }

    // (e) reported covariance: a fresh (mode-tied) belief must advertise a LARGER σ_yaw than the within-mode
    // Σ(2,2); after many frames of consistent backrest evidence the mode resolves and it collapses back to Σ.
    {
        const float R = P.sigma_base_m * P.sigma_base_m;
        ChairBelief fresh(gt, P);
        const float within0 = fresh.covariance()(2, 2);
        const float report0 = fresh.covariance_reported()(2, 2);
        check(report0 > within0 + 0.1f, "covariance_reported does not inflate σ_yaw while the mode is ambiguous");
        for (int i = 0; i < 60; ++i) fresh.resolve_orientation(all, R);
        const float report1 = fresh.covariance_reported()(2, 2);
        std::printf("  yaw_var: within=%.4f ambiguous=%.4f resolved=%.4f\n", within0, report0, report1);
        check(report1 < report0, "covariance_reported did not collapse σ_yaw once the mode resolved");
    }

    // (f) flip accumulator stays bounded (recoverable): 500 confident frames must NOT drive it out of ±6, and
    // a 180°-wrong chair must still restore afterwards (an unbounded accumulator can never recant).
    {
        const float R = P.sigma_base_m * P.sigma_base_m;
        ChairBelief wrong(ChairBeliefState{gt.cx, gt.cy, gt.yaw + static_cast<float>(M_PI)}, P);
        bool restored = false;
        for (int i = 0; i < 500; ++i) if (wrong.resolve_orientation(all, R)) restored = true;
        const float dyaw = std::abs(std::remainder(wrong.state().yaw - gt.yaw, 2.0f * static_cast<float>(M_PI)));
        check(restored and dyaw < 0.3f, "flip accumulator did not restore after a long run (unbounded/latched)");
    }

    const auto& S = belief.covariance();
    std::printf("  Σ diag (std): cx=%.3f cy=%.3f yaw=%.3f\n",
                std::sqrt(std::max(0.f, S(0, 0))), std::sqrt(std::max(0.f, S(1, 1))), std::sqrt(std::max(0.f, S(2, 2))));
    std::printf("ChairBelief::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
