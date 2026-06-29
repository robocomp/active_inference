/*
 * table_belief.cpp  —  AI2 table belief (see TABLE_FIT_AI2.md)
 */

#include "table_belief.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

namespace rc
{

// ─── SDF primitives (scalar, room→local handled by caller) ──────────────────────

namespace
{
// Box SDF given the point's distances-to-face along each axis (|local| − half_extent).
float box_sdf(float dx, float dy, float dz)
{
    const float ox = std::max(dx, 0.0f), oy = std::max(dy, 0.0f), oz = std::max(dz, 0.0f);
    const float outside = std::sqrt(ox * ox + oy * oy + oz * oz);
    const float inside  = std::min(std::max(dx, std::max(dy, dz)), 0.0f);
    return outside + inside;
}

// Vertical cylinder SDF: dl = (local point − cylinder centre), radius r, half-height hh.
float cylinder_sdf(float dlx, float dly, float dlz, float r, float hh)
{
    const float d_radial = std::sqrt(dlx * dlx + dly * dly) - r;
    const float d_vert   = std::abs(dlz) - hh;
    const float outside  = std::sqrt(std::max(d_radial, 0.0f) * std::max(d_radial, 0.0f) +
                                     std::max(d_vert, 0.0f)   * std::max(d_vert, 0.0f));
    const float inside   = std::min(std::max(d_radial, d_vert), 0.0f);
    return outside + inside;
}
}  // namespace

Eigen::Vector2f TableBelief::leg_center_local(const TableBeliefState& s, int k) const
{
    static const std::array<std::array<float, 2>, 4> sg = {{ {1, 1}, {-1, 1}, {-1, -1}, {1, -1} }};
    const float lx = sg[k][0] * (0.5f * s.w - params_.leg_radius);
    const float ly = sg[k][1] * (0.5f * s.h - params_.leg_radius);
    return {lx, ly};
}

float TableBelief::sdf_top(const Eigen::Vector3f& p, const TableBeliefState& s) const
{
    const float c = std::cos(-s.yaw), sn = std::sin(-s.yaw);
    const float px = p.x() - s.cx, py = p.y() - s.cy;
    const float lx = px * c - py * sn;
    const float ly = px * sn + py * c;
    const float half_t = 0.5f * params_.top_thickness;
    const float top_cz = s.H - half_t;
    return box_sdf(std::abs(lx) - 0.5f * s.w, std::abs(ly) - 0.5f * s.h, std::abs(p.z() - top_cz) - half_t);
}

float TableBelief::sdf_leg(const Eigen::Vector3f& p, const TableBeliefState& s, int k) const
{
    const float c = std::cos(-s.yaw), sn = std::sin(-s.yaw);
    const float px = p.x() - s.cx, py = p.y() - s.cy;
    const float lx = px * c - py * sn;
    const float ly = px * sn + py * c;
    const auto  ctr = leg_center_local(s, k);
    const float hh  = leg_half_height(s);
    return cylinder_sdf(lx - ctr.x(), ly - ctr.y(), p.z() - hh, params_.leg_radius, hh);
}

float TableBelief::sdf_prim(const Eigen::Vector3f& p, const TableBeliefState& s, int prim) const
{
    return prim == 0 ? sdf_top(p, s) : sdf_leg(p, s, prim - 1);
}

float TableBelief::sdf_compound(const Eigen::Vector3f& p, const TableBeliefState& s) const
{
    float m = sdf_top(p, s);
    for (int k = 0; k < 4; ++k) m = std::min(m, sdf_leg(p, s, k));
    return m;
}

// ─── Mixture responsibilities ───────────────────────────────────────────────────

std::array<float, 6> TableBelief::responsibilities(const Eigen::Vector3f& p, const TableBeliefState& s, float R) const
{
    const float eps    = std::clamp(params_.clutter_frac, 0.0f, 0.99f);
    const float pi_surf = (1.0f - eps) / 5.0f;
    const float inv2R  = 0.5f / std::max(1e-9f, R);

    // Height-based attribution. The leg cylinders' lateral surface exists only over z∈[0, H−top_thickness];
    // the top slab only over z∈[H−top_thickness, H]. The two meet at the join plane, so a tabletop point near
    // a corner (z≈H, laterally close to an inset corner leg) is otherwise mis-attributed to that leg, and GN
    // shrinks w,h to slide the legs under it → systematic under-size. Gate each component's mixing weight by a
    // smooth vertical compatibility about the join plane (band = slab half-thickness — physical, not tuned).
    // EM holds responsibilities fixed within a GN iteration, so this needs no Jacobian change.
    const float z_join = s.H - params_.top_thickness;       // leg top == slab bottom
    const float band   = std::max(0.5f * params_.top_thickness, 1e-3f);
    const float leg_z  = 1.0f / (1.0f + std::exp((p.z() - z_join) / band));  // →1 below join, →0 above (tabletop)
    const float top_z  = 1.0f - leg_z;                                       // →1 above join (tabletop), →0 below

    std::array<float, 6> u{};
    u[0] = pi_surf * top_z * std::exp(-sdf_top(p, s) * sdf_top(p, s) * inv2R);
    for (int k = 0; k < 4; ++k)
    {
        const float d = sdf_leg(p, s, k);
        u[1 + k] = pi_surf * leg_z * std::exp(-d * d * inv2R);
    }
    // Clutter = uniform; modelled as a Gaussian floor at clutter_scale so a point further than that from
    // EVERY surface is explained by clutter rather than dragging the model.
    const float cs = params_.clutter_scale_m;
    u[5] = eps * std::exp(-cs * cs * inv2R);

    float sum = 0.0f;
    for (float v : u) sum += v;
    if (sum <= 0.0f) { u = {}; u[5] = 1.0f; return u; }
    for (float& v : u) v /= sum;
    return u;
}

// ─── Jacobian (central finite difference) ──────────────────────────────────────

Eigen::Matrix<float, 6, 1> TableBelief::sdf_jacobian(const Eigen::Vector3f& p, const TableBeliefState& s, int prim) const
{
    Eigen::Matrix<float, 6, 1> J;
    Eigen::Matrix<float, 6, 1> base = s.vec();
    const float e = params_.fd_eps;
    for (int j = 0; j < 6; ++j)
    {
        Eigen::Matrix<float, 6, 1> vp = base, vm = base;
        vp(j) += e; vm(j) -= e;
        J(j) = (sdf_prim(p, TableBeliefState::from_vec(vp), prim) -
                sdf_prim(p, TableBeliefState::from_vec(vm), prim)) / (2.0f * e);
    }
    return J;
}

// ─── Constraints / canonical form ───────────────────────────────────────────────

void TableBelief::apply_constraints(TableBeliefState& s) const
{
    s.w = std::max(s.w, 0.10f);
    s.h = std::max(s.h, 0.10f);
    s.H = std::max(s.H, params_.top_thickness + 0.05f);
    s.yaw = std::remainder(s.yaw, 2.0f * static_cast<float>(M_PI));   // wrap to (−π, π]
}

// ─── Prior covariance ───────────────────────────────────────────────────────────

void TableBelief::init_prior_cov()
{
    Sigma_.setZero();
    const float ss = params_.prior_size_std * params_.prior_size_std;
    Sigma_(0, 0) = 0.30f * 0.30f;   // cx
    Sigma_(1, 1) = 0.30f * 0.30f;   // cy
    Sigma_(2, 2) = ss;              // H
    Sigma_(3, 3) = ss;              // w
    Sigma_(4, 4) = ss;              // h
    Sigma_(5, 5) = 0.60f * 0.60f;   // yaw
}

void TableBelief::predict()
{
    if (Sigma_.isZero()) init_prior_cov();
    const float qm = params_.process_std_m * params_.process_std_m;
    const float qy = params_.process_std_yaw * params_.process_std_yaw;
    Eigen::Matrix<float, 6, 1> q;
    q << qm, qm, qm, qm, qm, qy;   // rigid + static ⇒ small process noise; length DOFs share qm, yaw qy
    Sigma_ += q.asDiagonal();
    prior_mean_ = state_.vec();
    have_prior_ = true;
}

// ─── Recursive update (predict + EM Gauss-Newton) ──────────────────────────────

float TableBelief::update(const TableFrame& frame)
{
    predict();
    if (frame.points.empty())
        return 0.0f;

    const float sigma2 = params_.sigma_base_m * params_.sigma_base_m;
    const auto  R_of   = [&](std::size_t i) -> float
    { return (i < frame.R.size() and frame.R[i] > 0.0f) ? frame.R[i] : sigma2; };

    const Eigen::Matrix<float, 6, 6> P0 = Sigma_.inverse();   // transition-prior precision (Σ_pred⁻¹)
    Eigen::Matrix<float, 6, 1> theta = state_.vec();

    // Inverse of the per-frame common-mode covariance Σ_c (diagonal): position (cx,cy) = config floor +
    // pose-chain variance; size (H,w,h) and yaw = config measurement stds. Marginalising this shared
    // error makes the frame's information SATURATE at Σ_c regardless of point count (so N correlated
    // points cannot collapse σ). Σ_c⁻¹ is used via Woodbury so no Σ_c or data-info inverse is needed.
    const float p2 = params_.common_mode_pos_std  * params_.common_mode_pos_std;
    const float s2 = params_.common_mode_size_std * params_.common_mode_size_std;
    const float y2 = params_.common_mode_yaw_std  * params_.common_mode_yaw_std;
    Eigen::Matrix<float, 6, 1> sc_inv;
    sc_inv << 1.0f / std::max(1e-9f, p2 + frame.chain_cov_xx),
              1.0f / std::max(1e-9f, p2 + frame.chain_cov_yy),
              1.0f / s2, 1.0f / s2, 1.0f / s2, 1.0f / y2;
    const Eigen::Matrix<float, 6, 6> Sc_inv = sc_inv.asDiagonal();

    // Accumulate the DATA-ONLY information Id and gradient bd (no prior) at th.
    Eigen::Matrix<float, 6, 6> Id;
    Eigen::Matrix<float, 6, 1> bd;
    const auto accumulate_data = [&](const Eigen::Matrix<float, 6, 1>& th)
    {
        const TableBeliefState s = TableBeliefState::from_vec(th);
        Id.setZero();
        bd.setZero();
        for (std::size_t i = 0; i < frame.points.size(); ++i)
        {
            const float R = R_of(i);
            const auto  r = responsibilities(frame.points[i], s, R);
            for (int prim = 0; prim < 5; ++prim)
            {
                const float w = r[prim] / R;
                if (w < 1e-6f) continue;
                const float d = sdf_prim(frame.points[i], s, prim);
                const Eigen::Matrix<float, 6, 1> J = sdf_jacobian(frame.points[i], s, prim);
                Id.noalias() += w * (J * J.transpose());
                bd.noalias() += -w * J * d;
            }
        }
    };

    // Saturate the data evidence by the common-mode (exact marginalisation, Woodbury):
    //   I_eff = Id − Id (Id + Σc⁻¹)⁻¹ Id ,   b_eff = bd − Id (Id + Σc⁻¹)⁻¹ bd
    // I_eff → Σc⁻¹ as Id → ∞, so a frame contributes at most Σc⁻¹ information.
    Eigen::Matrix<float, 6, 6> Ieff;
    Eigen::Matrix<float, 6, 1> beff;
    const auto saturate = [&]()
    {
        const Eigen::Matrix<float, 6, 6> IdMinv = Id * (Id + Sc_inv).inverse();
        Ieff = Id - IdMinv * Id;
        beff = bd - IdMinv * bd;
    };

    // MEAN (MAP): use the FULL data information so the point estimate converges fast/accurately.
    // The common-mode saturation is applied ONLY to the posterior covariance below (calibration) —
    // it must not slow the point estimate.
    for (int it = 0; it < params_.gn_iters; ++it)
    {
        accumulate_data(theta);
        const Eigen::Matrix<float, 6, 6> A = P0 + Id;
        const Eigen::Matrix<float, 6, 1> b = P0 * (prior_mean_ - theta) + bd;
        const Eigen::Matrix<float, 6, 1> dtheta = A.ldlt().solve(b);
        if (!dtheta.allFinite()) break;
        theta += dtheta;
        TableBeliefState s = TableBeliefState::from_vec(theta);
        apply_constraints(s);
        theta = s.vec();
    }

    TableBeliefState s = TableBeliefState::from_vec(theta);
    apply_constraints(s);
    // Canonical w ≥ h: fold the box's π/2 symmetry into the representation (swap w↔h, yaw += π/2, and the
    // matching covariance rows/cols) instead of letting a yaw flip silently swap the extents.
    if (s.w < s.h)
    {
        std::swap(s.w, s.h);
        s.yaw = std::remainder(s.yaw + 0.5f * static_cast<float>(M_PI), 2.0f * static_cast<float>(M_PI));
    }
    state_ = s;

    accumulate_data(state_.vec());            // data information at the converged state
    saturate();                                // cap it by the common-mode (calibrated posterior)
    Sigma_ = (P0 + Ieff).inverse();            // posterior covariance

    // Mean per-point data energy (diagnostic, ~negative-log-likelihood proxy).
    double energy = 0.0;
    for (std::size_t i = 0; i < frame.points.size(); ++i)
    {
        const float R = R_of(i);
        const auto  r = responsibilities(frame.points[i], state_, R);
        for (int prim = 0; prim < 5; ++prim)
        {
            const float d = sdf_prim(frame.points[i], state_, prim);
            energy += 0.5 * r[prim] * d * d / R;
        }
    }
    return static_cast<float>(energy / static_cast<double>(frame.points.size()));
}

// ─── Self-test ───────────────────────────────────────────────────────────────────

bool TableBelief::self_test()
{
    std::mt19937 rng(12345);
    std::normal_distribution<float> noise(0.0f, 0.01f);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f), U01(0.0f, 1.0f);

    TableBeliefParams P;
    const TableBeliefState gt{0.20f, -0.30f, 0.74f, 1.50f, 1.00f, 0.30f};   // ground truth (w>h, canonical)

    const float c = std::cos(gt.yaw), sn = std::sin(gt.yaw);
    const auto to_world = [&](float lx, float ly, float lz) -> Eigen::Vector3f
    { return {gt.cx + c * lx - sn * ly, gt.cy + sn * lx + c * ly, lz}; };

    std::vector<Eigen::Vector3f> pts;
    // Top surface
    for (int i = 0; i < 1200; ++i)
    {
        const float lx = U(rng) * 0.5f * gt.w, ly = U(rng) * 0.5f * gt.h;
        pts.push_back(to_world(lx, ly, gt.H + noise(rng)));
    }
    // Legs (4 cylinders)
    for (int k = 0; k < 4; ++k)
    {
        const float sx = (k == 0 || k == 3) ? 1.f : -1.f, sy = (k < 2) ? 1.f : -1.f;
        const float cxk = sx * (0.5f * gt.w - P.leg_radius), cyk = sy * (0.5f * gt.h - P.leg_radius);
        for (int i = 0; i < 200; ++i)
        {
            const float ang = U(rng) * static_cast<float>(M_PI);
            const float z = U01(rng) * (gt.H - P.top_thickness);
            pts.push_back(to_world(cxk + P.leg_radius * std::cos(ang) + noise(rng),
                                   cyk + P.leg_radius * std::sin(ang) + noise(rng), z));
        }
    }
    // Clutter (floor / off-table)
    std::vector<Eigen::Vector3f> clutter;
    for (int i = 0; i < 150; ++i)
        clutter.push_back({gt.cx + U(rng) * 1.5f, gt.cy + U(rng) * 1.5f, U01(rng) * 0.05f});
    std::vector<Eigen::Vector3f> all = pts;
    all.insert(all.end(), clutter.begin(), clutter.end());

    bool ok = true;
    auto check = [&](bool cond, const char* msg) { if (!cond) { ok = false; std::printf("  FAIL: %s\n", msg); } };

    // (a) SDF correctness at known points
    {
        TableBelief b(gt, P);
        const float s_center = b.sdf_compound(to_world(0, 0, gt.H - 0.5f * P.top_thickness), gt);
        check(s_center < -1e-3f, "top-slab centre SDF should be negative (inside)");
        const float s_surf = b.sdf_top(to_world(0.2f * gt.w, 0.1f * gt.h, gt.H), gt);
        check(std::abs(s_surf) < 1e-2f, "on-surface SDF should be ~0");
        const float s_far = b.sdf_compound({gt.cx + 3.0f, gt.cy, 1.5f}, gt);
        check(s_far > 1.0f, "far point SDF should be large positive");
    }

    // (b) Jacobian: coarse FD (used in solver) vs fine FD — must agree
    {
        TableBelief b(gt, P);
        const Eigen::Vector3f q = to_world(0.45f * gt.w, 0.0f, gt.H);   // near a top edge → nonzero gradient
        const auto Jc = b.sdf_jacobian(q, gt, 0);
        TableBeliefParams Pf = P; Pf.fd_eps = 1e-4f;
        TableBelief bf(gt, Pf);
        const auto Jf = bf.sdf_jacobian(q, gt, 0);
        check((Jc - Jf).norm() < 1e-2f, "SDF Jacobian (coarse vs fine FD) disagree");
    }

    // (c) Fit recovery from a perturbed init
    TableBelief belief(TableBeliefState{0.0f, 0.0f, 0.75f, 1.0f, 0.6f, 0.0f}, P);
    TableFrame frame; frame.points = all;
    float e = 0.0f;
    for (int it = 0; it < 8; ++it) e = belief.update(frame);
    const auto& s = belief.state();
    std::printf("  recovered: cx=%.3f cy=%.3f H=%.3f w=%.3f h=%.3f yaw=%.3f  (energy=%.4f)\n",
                s.cx, s.cy, s.H, s.w, s.h, s.yaw, e);
    std::printf("  truth:     cx=%.3f cy=%.3f H=%.3f w=%.3f h=%.3f yaw=%.3f\n",
                gt.cx, gt.cy, gt.H, gt.w, gt.h, gt.yaw);
    check(std::abs(s.w - gt.w)   < 0.06f, "w not recovered");
    check(std::abs(s.h - gt.h)   < 0.06f, "h not recovered");
    check(std::abs(s.H - gt.H)   < 0.03f, "H not recovered");
    check(std::abs(s.cx - gt.cx) < 0.03f, "cx not recovered");
    check(std::abs(s.cy - gt.cy) < 0.03f, "cy not recovered");
    check(std::abs(std::remainder(s.yaw - gt.yaw, static_cast<float>(M_PI))) < 0.06f, "yaw not recovered");

    // (d) Clutter responsibility: a far floor point is mostly clutter
    {
        const auto r = belief.responsibilities({gt.cx + 1.2f, gt.cy + 1.2f, 0.02f}, belief.state(),
                                               P.sigma_base_m * P.sigma_base_m);
        std::printf("  clutter-point responsibilities: top=%.2f legs=%.2f clutter=%.2f\n",
                    r[0], r[1] + r[2] + r[3] + r[4], r[5]);
        check(r[5] > 0.5f, "far floor point should be mostly clutter");
    }

    // (e) Σ finite & SPD
    {
        const Eigen::Matrix<float, 6, 6>& S = belief.covariance();
        check(S.allFinite(), "Σ not finite");
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> es(S);
        check(es.eigenvalues().minCoeff() > 0.0f, "Σ not SPD");
        std::printf("  Σ diag (std, m/rad): cx=%.3f cy=%.3f H=%.3f w=%.3f h=%.3f yaw=%.3f\n",
                    std::sqrt(S(0,0)), std::sqrt(S(1,1)), std::sqrt(S(2,2)),
                    std::sqrt(S(3,3)), std::sqrt(S(4,4)), std::sqrt(S(5,5)));
    }

    std::printf("TableBelief::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
