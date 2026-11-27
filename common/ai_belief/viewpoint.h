/*
 * common/ai_belief/viewpoint.h — (moved from countable_concept/src/countable_viewpoint.{h,cpp}; header-only) how much a sensor's SHARED error has decorrelated since the channel last updated the
 * instance (spec 2026-11-15 lidar-measures §7c.2). Pure Eigen, no OpenCV/DSR/Qt.
 *
 * A sensor's systematic error (a bias in where the camera sees the edge, a ring-gap aliasing on the LiDAR) is
 * the same on consecutive frames and only decorrelates when the VIEWPOINT changes. rho in [0,1] is the
 * remaining correlation: 1 = parked (the shared error is unchanged, so re-measuring adds nothing), 0 = fully
 * decorrelated. It is a continuous function of the odometry path relative to the object — no gate.
 * Rotation in place moves neither the sensor's bearing to the object nor its range, so it leaves rho = 1.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>

#include <Eigen/Dense>

#include "../place_memory/place_map.h"   // place::se2_inverse_jacobian

namespace rc::ai::view {

// Accumulated since the channel's last update of the instance; the caller resets it after an update.
struct ViewpointPath
{
    float dphi_rad = 0.f;   // |bearing change| as seen from the object, summed (tangential path / range)
    float dz_m     = 0.f;   // |range change| to the object, summed (radial path)
};

// Adds |step·t̂|/z to dphi and |step·r̂| to dz, r̂ = unit(sensor − object), t̂ ⊥ r̂, z = |sensor − object|.
inline void accumulate_path(ViewpointPath& p, const Eigen::Vector2f& sensor_xy_room,
                     const Eigen::Vector2f& step_xy_room, const Eigen::Vector2f& object_xy)
{
    const Eigen::Vector2f r = sensor_xy_room - object_xy;
    const float z = r.norm();
    // Degenerate geometry (the sensor sits at the object centre: r̂ undefined), not a tuning threshold.
    if (z < 1e-3f)
        return;
    const Eigen::Vector2f rhat = r / z;
    const Eigen::Vector2f that(-rhat.y(), rhat.x());
    p.dphi_rad += std::abs(step_xy_room.dot(that)) / z;
    p.dz_m     += std::abs(step_xy_room.dot(rhat));
}

// Room-frame displacement of the robot over dt. Robot frame: +x right, +y FORWARD (controller/FRAMES.md,
// ROBOT_GEOMETRY.md), CCW-positive yaw; v_adv is along +y, v_side along +x; rotated into the room by yaw_room.
inline Eigen::Vector2f odometry_step_room(float v_adv, float v_side, float yaw_room, float dt_s)
{
    const float c = std::cos(yaw_room), s = std::sin(yaw_room);
    const Eigen::Vector2f robot(v_side * dt_s, v_adv * dt_s);   // x right, y forward
    return {c * robot.x() - s * robot.y(), s * robot.x() + c * robot.y()};
}

namespace detail {
// |Δφ| / half_angle, with the 0/0 limit defined as 0: no bearing change decorrelates nothing, whatever the object's
// width (degenerate geometry w⊥ = 0, not a tuning guard). Δφ > 0 with half_angle = 0 stays +inf ⇒ ρ = 0.
inline float bearing_term(float dphi, float half_angle)
{
    if (dphi == 0.f) return 0.f;
    return std::abs(dphi) / half_angle;
}
}  // namespace detail

// exp(−|Δφ|/atan(w⊥/2z) − |Δz|/z): e-fold when the camera has moved by the object's angular half-width.
inline float rho_camera(const ViewpointPath& p, float z_m, float w_perp_m)
{
    const float half_angle = std::atan(w_perp_m / (2.f * z_m));
    return std::exp(-detail::bearing_term(p.dphi_rad, half_angle) - std::abs(p.dz_m) / z_m);
}
// exp(−|Δφ|/g), g = ring gap (rad). g ≤ 0 (SensorModel absent) ⇒ 1.
inline float rho_lidar_lateral(const ViewpointPath& p, float g)
{
    if (g <= 0.f) return 1.f;
    return std::exp(-std::abs(p.dphi_rad) / g);
}
// exp(−|Δz||tan ε|/(g z)), ε = elevation of the beam. g ≤ 0 ⇒ 1.
inline float rho_lidar_height(const ViewpointPath& p, float z_m, float eps_rad, float g)
{
    if (g <= 0.f) return 1.f;
    return std::exp(-std::abs(p.dz_m) * std::abs(std::tan(eps_rad)) / (g * z_m));
}
namespace detail {
inline float wrap_pi(float a)
{
    a = std::fmod(a + static_cast<float>(M_PI), 2.f * static_cast<float>(M_PI));
    if (a <= 0.f) a += 2.f * static_cast<float>(M_PI);
    return a - static_cast<float>(M_PI);
}
}  // namespace detail

// ── F1 (task-F1, D2's T1): ρ from the SEPARATION of viewpoints, not from the arc length of the odometry ───────────────────
// The kernels above are exp(−d(x, x′)/ℓ) in viewpoint space x = (φ, z) — φ the sensor's bearing seen from the object, z its
// range. Their argument used to be the SUM of |odometry steps| since the channel's last update (ViewpointPath). Parked, zero-
// mean odometry jitter sums to ~0.04 rad / 0.1 m per dwell against < 1 mm of true motion, so b_L was re-drawn about twice per
// dwell from motion that did not happen. The argument must be how far the views are APART, which does not grow under jitter.
//
// Viewpoint: odometric coordinates of a sensor about the object. A robot displacement dr that brought the sensor to its room
// position s (from s − dr) adds the EXACT bearing and range change of that move as seen from the object c:
// φ += wrap(∠(s − c) − ∠(s − dr − c)), z += |s − c| − |s − dr − c|. (accumulate_path's first-order |dr·t̂|/z, |dr·r̂| summed with
// signs would leave a one-signed second-order residue |dr⊥|²/2z per step in z, which a long jitter integrates into drift; the
// exact increments telescope when the sensor returns.) The odometry, not the localised pose, moves it — as before, so a
// localiser correction is ρ_chain's business, not ρ_cam's / ρ_L's, and a rotation in place moves neither coordinate. Only
// DIFFERENCES of viewpoints are read; the origin is arbitrary.
// (Measured on g1x_aba_bed_1's parked dwells: the localised camera position drifts 6–38 mm per dwell, the signed odometry
// viewpoint 2–4 mrad / 5–6 mm — against the old |step| sums of 34–38 mrad / 97–105 mm.)
struct Viewpoint { float phi = 0.f, z = 0.f; };
inline Viewpoint advance_viewpoint(Viewpoint x, const Eigen::Vector2f& sensor_xy_room, const Eigen::Vector2f& dr_room,
                                   const Eigen::Vector2f& object_xy)
{
    const Eigen::Vector2f r1 = sensor_xy_room - object_xy, r0 = r1 - dr_room;
    const float z1 = r1.norm(), z0 = r0.norm();
    if (z1 < 1e-3f or z0 < 1e-3f) return x;   // degenerate geometry (bearing undefined), as accumulate_path — not a tuning threshold
    x.phi += detail::wrap_pi(std::atan2(r1.y(), r1.x()) - std::atan2(r0.y(), r0.x()));
    x.z   += z1 - z0;
    return x;
}

// A latent's anchor: the viewpoint a at which its current value was established, and the half-width m (per coordinate) of the
// views it has ALREADY been decorrelated over — its paid frontier.
//
// The kernel is an L1 exponential, i.e. a PRODUCT of one-dimensional exponential (Ornstein–Uhlenbeck) kernels, one per
// coordinate, and an OU process is Markov in its own coordinate: given the value at the nearest view already drawn, a view
// beyond it at distance Δ has correlation exp(−Δ/ℓ) with it, exactly, whatever came before. So, per coordinate, the new view's
// excess separation beyond the frontier, Δ = max(0, |x − a| − m), is the argument the unchanged kernel reads:
//     ρ_k = k(x_k, a) / k(x*, a),  x* = the farthest view already paid for (|x* − a| = m).
// On a monotone drive the frontier is the previous update's view, so this is D2's ratio k(x_k, x_a)/k(x_{k−1}, x_a) and the per-
// update ρ equals the old arc-length ρ (the arc length of a monotone path is its separation; same kernel).
//
// ★ THE BOUND (controller ruling, task-F1 brief; flagged, not tuned). A view INSIDE the frontier makes that ratio exceed 1. A
//   lag-1 (Markov) latent predicted b′ = ρb + √(1−ρ²)ε cannot RAISE its correlation — ρ > 1 makes the innovation variance
//   negative — so ρ = 1 there (the max(0, ·) below) is the FEASIBILITY BOUND of the Markov latent, not a threshold. It errs
//   conservative: frames that return toward views already drawn are treated as sharing the latent, so they add less
//   information than the exact model would. The honest form is a per-view latent bank (a non-Markov latent keyed by viewpoint,
//   D2 Concern 3), NOT built.
//   The bound is applied against the FRONTIER, not against the previous view x_{k−1} (D2's literal ratio): with x_{k−1} the
//   bound re-pays every outward half-swing of a jitter that keeps returning, so Σ(1 − ρ²) still grows linearly with the dwell
//   (measured on g1x_aba_bed_1's dwells: Σ −ln ρ_L = 0.42–0.53 per dwell with x_{k−1}, 0.05–0.11 with the frontier, 0.87–0.96
//   for the old arc length). With the frontier the bound is the same statement — a Markov latent cannot be re-correlated —
//   applied once per view instead of once per swing; the kernel ratio and the anchor rule are unchanged.
// The anchor then follows the view at the latent's own rate (D2's rule), a ← a + (1 − ρ_k)(x_k − a), and the frontier is re-
// referenced so its LEADING edge stays put (m ← max(m, |x_k − a|) − |shift|): on a straight drive the next update pays exactly
// its own increment again. The trailing edge is pulled in by twice the shift, so the latent forgets the views it leaves behind
// at its own rate (on a long drive the anchor lags ~ℓ and views up to ~2ℓ behind still count as drawn — the conservative side).
struct ViewAnchor
{
    bool set = false;   // false until the latent's first update (born at its prior: ρ irrelevant, the engine's new source)
    Viewpoint a;        // anchor
    float m_phi = 0.f;  // paid frontier half-widths (rad, m)
    float m_z = 0.f;
};

// The separation of view x beyond the anchor's paid frontier, per coordinate, as the ViewpointPath the kernels read. Unset ⇒ 0.
inline ViewpointPath anchor_separation(const ViewAnchor& A, const Viewpoint& x)
{
    if (not A.set) return {};
    ViewpointPath p;
    p.dphi_rad = std::max(0.f, std::abs(detail::wrap_pi(x.phi - A.a.phi)) - A.m_phi);   // ★ the feasibility bound, see above
    p.dz_m     = std::max(0.f, std::abs(x.z - A.a.z) - A.m_z);                         // ★ the same bound
    return p;
}

// After the latent's update with view x and the ρ it used: the first update sets the anchor at x; later ones move it.
inline ViewAnchor anchor_commit(ViewAnchor A, const Viewpoint& x, float rho)
{
    if (not A.set) return {true, x, 0.f, 0.f};
    const float k = 1.f - rho;
    const float dphi = detail::wrap_pi(x.phi - A.a.phi), dz = x.z - A.a.z;
    A.m_phi = std::max(A.m_phi, std::abs(dphi)) - k * std::abs(dphi);   // ≥ 0: max(m, s) ≥ s ≥ (1 − ρ)s
    A.m_z   = std::max(A.m_z, std::abs(dz)) - k * std::abs(dz);
    A.a.phi += k * dphi;
    A.a.z   += k * dz;
    return A;
}

// One instance's viewpoint bookkeeping, carried across updates (fitter) and staged on each frame (what the update used).
// cam / L: the camera's / LiDAR's viewpoint; cam_xy / L_xy: their room positions there (the geometry of the next increment
// and of the offer). Anchors: b_cam (one ρ for every DOF), b_L lateral (the ring gap, reads φ), b_L height (reads z).
struct ViewState
{
    bool on = false;                        // staged by an F1 fitter (absent in older dumps / other callers)
    Viewpoint cam, L;
    std::optional<Eigen::Vector2f> cam_xy, L_xy;
    ViewAnchor a_cam, a_lat, a_H;
};
// The view of THIS frame: the last accepted update's viewpoints advanced by the odometry dr since then, at the sensors' current
// positions (nullopt ⇒ that sensor's viewpoint is not advanced: no position, nothing integrated — fails closed, ρ → 1).
inline ViewState view_stage(ViewState v, const std::optional<Eigen::Vector2f>& cam_xy,
                            const std::optional<Eigen::Vector2f>& L_xy, const Eigen::Vector2f& dr, const Eigen::Vector2f& c)
{
    v.on = true;
    if (cam_xy.has_value()) { v.cam = advance_viewpoint(v.cam, *cam_xy, dr, c); v.cam_xy = cam_xy; }
    if (L_xy.has_value())   { v.L   = advance_viewpoint(v.L,   *L_xy,   dr, c); v.L_xy = L_xy; }
    return v;
}
// After an ACCEPTED update with that frame: the measuring channels' anchors move (ρ_cam, ρ_L lateral, ρ_L height as used).
inline ViewState view_commit(ViewState v, bool camera, bool lidar, float rho_cam, float rho_lat, float rho_H)
{
    if (camera) v.a_cam = anchor_commit(v.a_cam, v.cam, rho_cam);
    if (lidar)
    {
        v.a_lat = anchor_commit(v.a_lat, v.L, rho_lat);
        v.a_H   = anchor_commit(v.a_H,   v.L, rho_H);
    }
    return v;
}

// ── The localiser's correction for ρ_chain (§7c.2), as testable bookkeeping. ─────────────────────────────────────────
// c = (pose at the last odometry-consistent sample − ref) − odometry integrated since ref. It must FAIL CLOSED like the
// camera/LiDAR paths: a stretch whose odometry cannot be integrated (no speed, a stale stamp, an unknown interval) is
// NOT a localiser correction, so the reference absorbs that stretch's pose increment (ref += Δpose) — the corrections
// accumulated before it are kept. ref and odo restart at the last consistent sample when the chain latent is updated.
struct ChainTracker
{
    Eigen::Vector3f ref  = Eigen::Vector3f::Zero();   // (x, y, psi) the correction is measured from
    Eigen::Vector3f prev = Eigen::Vector3f::Zero();   // pose at the last observed (odometry-consistent) sample
    Eigen::Vector3f odo  = Eigen::Vector3f::Zero();   // odometry (dx, dy, dpsi) integrated since ref
    bool have = false;
};
// SE(2) odometry step: (x, y) as odometry_step_room, psi = omega·dt.
inline Eigen::Vector3f odometry_step_room_se2(float v_adv, float v_side, float omega, float yaw_room, float dt_s)
{
    const Eigen::Vector2f d = odometry_step_room(v_adv, v_side, yaw_room, dt_s);
    return {d.x(), d.y(), omega * dt_s};
}
// One sample: pose (x, y, psi) now, and the odometry step over the interval since the previous sample (nullopt ⇔ it
// cannot be integrated ⇒ the stretch is absorbed into ref). The first sample initialises the tracker.
inline void chain_observe(ChainTracker& t, const Eigen::Vector3f& pose, std::optional<Eigen::Vector3f> odo_step)
{
    if (not t.have)
    {
        t.ref = t.prev = pose;
        t.odo.setZero();
        t.have = true;
        return;
    }
    if (odo_step.has_value()) t.odo += *odo_step;
    else                      t.ref += pose - t.prev;   // unknown odometry: not a correction (fail closed)
    t.prev = pose;
}
// c, the localiser's correction since ref (zero before the first sample); psi wrapped to (−pi, pi].
inline Eigen::Vector3f chain_correction(const ChainTracker& t)
{
    if (not t.have) return Eigen::Vector3f::Zero();
    Eigen::Vector3f c = (t.prev - t.ref) - t.odo;
    c.z() = detail::wrap_pi(c.z());
    return c;
}
// After the chain latent's update: measure from the last consistent sample.
inline void chain_rebase(ChainTracker& t)
{
    t.ref = t.prev;
    t.odo.setZero();
}

// exp(−½ cᵀΣ⁻¹c) on the non-zero subspace of Σ (pseudo-inverse). A zero-variance direction with c ≠ 0 ⇒ 0; c = 0 ⇒ 1.
inline double rho_chain(const Eigen::Vector3d& c, const Eigen::Matrix3d& S)
{
    // A coordinate whose Σ diagonal is exactly zero is declared by the data to have no uncertainty (PSD ⇒ its row and
    // column are zero): any non-zero c there cannot be explained at all ⇒ 0. No tolerance anywhere. The remaining
    // principal sub-block (symmetrised) is decomposed; eigen-directions with λ ≤ 0 (sign test) contribute nothing.
    std::array<int, 3> keep{};
    int n = 0;
    for (int i = 0; i < 3; ++i)
    {
        if (S(i, i) == 0.0) { if (c(i) != 0.0) return 0.0; }
        else keep[static_cast<std::size_t>(n++)] = i;
    }
    if (n == 0) return 1.0;
    Eigen::MatrixXd Sub(n, n);
    Eigen::VectorXd cs(n);
    for (int i = 0; i < n; ++i)
    {
        cs(i) = c(keep[static_cast<std::size_t>(i)]);
        for (int j = 0; j < n; ++j)
            Sub(i, j) = 0.5 * (S(keep[static_cast<std::size_t>(i)], keep[static_cast<std::size_t>(j)])
                             + S(keep[static_cast<std::size_t>(j)], keep[static_cast<std::size_t>(i)]));
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sub);
    double maha = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double lam = es.eigenvalues()(i);
        if (lam > 0.0)
        {
            const double p = es.eigenvectors().col(i).dot(cs);
            maha += p * p / lam;
        }
    }
    return std::exp(-0.5 * maha);
}

// ── The robot's pose error in the room frame (spec 7c.7 b) ──────────────────────────────────────────────────────────
// Σ_pose (x, y, psi) in the room frame from the robot→room edge's 6×6 row-major rt_covariance block (one 36-slot block,
// ordered [x y z rx ry rz], in the edge's parent frame): slots {0, 1, 5}, pushed through rc::se2_inverse_jacobian
// evaluated at the pose of the edge (the pose being inverted). Empty span ⇒ zero.
inline Eigen::Matrix3d pose_cov_in_room(std::span<const float> rt_cov36, const Eigen::Vector3f& edge_pose)
{
    if (rt_cov36.size() < 36) return Eigen::Matrix3d::Zero();
    constexpr int idx[3] = {0, 1, 5};
    Eigen::Matrix3d S;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            S(i, j) = rt_cov36[static_cast<std::size_t>(idx[i] * 6 + idx[j])];
    const Eigen::Matrix3d J = place::se2_inverse_jacobian(edge_pose).cast<double>();
    Eigen::Matrix3d R = J * S * J.transpose();   // double: float J Σ Jᵀ would put ~1e-7 noise on a null direction
    R = 0.5 * (R + R.transpose());
    return R;
}

// F_t = [[1,0,−Δr_y],[0,1,Δr_x],[0,0,1]]: how the pose error e = (δx, δy, δψ) is carried by a robot translation Δr.
inline Eigen::Matrix3d chain_transport(const Eigen::Vector2f& dr)
{
    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    F(0, 2) = -dr.y();
    F(1, 2) =  dr.x();
    return F;
}

// The lever ℓ (object centre − robot position, room frame) through which a robot yaw error moves the object.
struct ChainLever { Eigen::Vector2f l = Eigen::Vector2f::Zero(); bool set = false; };
inline void lever_start(ChainLever& L, const Eigen::Vector2f& seed_c, const Eigen::Vector2f& robot_r)
{
    L.l = seed_c - robot_r;
    L.set = true;
}
// ℓ ← ρ(ℓ − Δr) + (1 − ρ)(c − r).
inline void lever_step(ChainLever& L, const Eigen::Vector2f& dr, float rho, const Eigen::Vector2f& c, const Eigen::Vector2f& r)
{
    L.l = rho * (L.l - dr) + (1.f - rho) * (c - r);
}
// Θ = (cx, cy, H, w, h, yaw): rows cx = [1,0,−ℓy], cy = [0,1,ℓx], yaw = [0,0,1]; H, w, h zero.
inline Eigen::Matrix<double, 6, 3> chain_B(const Eigen::Vector2f& l)
{
    Eigen::Matrix<double, 6, 3> B = Eigen::Matrix<double, 6, 3>::Zero();
    B(0, 0) = 1.0; B(0, 2) = -l.y();
    B(1, 1) = 1.0; B(1, 2) =  l.x();
    B(5, 2) = 1.0;
    return B;
}

}  // namespace rc::ai::view
