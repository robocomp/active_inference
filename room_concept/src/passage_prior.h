#pragma once
// ── THE CROSSED DOOR AS A PRIOR ON THE POSE (2026-10-21, Task R1b) ─────────────────────────────────────────
// A hand-over through a door says where the robot is, independently of the tracker that may be failing at that very
// moment (the doorway view is the one the room explains worst): in the aperture, its centre just past the door line,
// heading through it. Two things are built from that one fact, both pure functions here so the selftest runs them:
//
//  1. THE DOORWAY TERM — P(the robot is still in the passage, i.e. the room it entered does not yet owe it the view
//     seen back through the door). The same conjunction as the scene graph's pre-birth doorway term, seen from the
//     NEW room's side: p_open · P(within the span) · Φ((r_body − d)/σ_s), d = depth of the centre into the new room.
//     It expires by geometry, as the body leaves the mouth — not by a timer, and not by a hazard-set belief seed
//     (reviewer I-2 on 8ce48e5e: a MISMATCH seed suppressed recovery from a WRONG recall for ~18 s driving).
//
//  2. THE DOOR ANCHOR — a Gaussian on SE(2) at the hand-over: mean = the centre's crossing point and the heading
//     through the door; covariance = the crossing's own σ along the normal ⊕ the aperture's free play along it ⊕ the
//     yaw the body can have and still fit through. It is then carried by ODOMETRY alone (never by the tracker), so a
//     search fired while the tracker diverges in the doorway still knows which of two symmetric twins came through
//     the door (live rtD1g2: twins at w 0.50/0.50 — the continuity prior was centred on a pose already 4 m off).
//
// No constant is introduced: the free play and the yaw come from the door width and the body's own dimensions.
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace rc::passage
{
    struct Passage
    {
        Eigen::Vector2f centre{0.f, 0.f};    ///< aperture centre, INTERNAL frame
        Eigen::Vector2f n_in{0.f, 1.f};      ///< unit normal INTO the room just entered
        Eigen::Vector2f tangent{1.f, 0.f};   ///< unit along the aperture
        float width   = 1.f;                 ///< aperture width (m)
        float p_open  = 1.f;                 ///< door_open_prob at the crossing (-1 unmeasured ⇒ ½)
        float r_body  = 0.25f;               ///< half the larger body dimension (m)
        float body_w  = 0.46f;               ///< body width (m) — what has to fit through the aperture
        float body_l  = 0.48f;               ///< body length (m)
        float sig_s   = 0.03f;               ///< pose σ along the normal at the crossing (m)
        float sig_t   = 0.03f;               ///< pose σ along the aperture at the crossing (m)
        float s       = 0.f;                 ///< centre depth past the line at the crossing (m)
        float u       = 0.f;                 ///< centre offset along the aperture at the crossing (m)
        std::uint64_t door_id = 0;           ///< the door node crossed (0 unknown)
        /// The localiser's (pre-clamp) heading at the crossing and its σ. NaN = unknown: forward and reversed equally.
        float heading = std::numeric_limits<float>::quiet_NaN();
        float sig_heading = 0.f;
    };

    inline float phi(float z) { return 0.5f * std::erfc(-z / std::sqrt(2.0f)); }

    /// P(still in the passage) from the new room's side, under the pose's position covariance.
    inline float doorway_term(const Passage& p, const Eigen::Vector2f& xy, const Eigen::Matrix2f& cov_xy)
    {
        const Eigen::Vector2f r = xy - p.centre;
        const float d = r.dot(p.n_in), u = r.dot(p.tangent);
        const float sig_s = std::sqrt(p.n_in.dot(cov_xy * p.n_in) + 1e-8f);       // 1e-8: numerical, see door_crossing.h
        const float sig_t = std::sqrt(p.tangent.dot(cov_xy * p.tangent) + 1e-8f);
        const float half = 0.5f * p.width;
        const float span = phi((half - u) / sig_t) * phi((half + u) / sig_t);
        const float open = p.p_open < 0.f ? 0.5f : std::clamp(p.p_open, 0.f, 1.f);
        return open * span * phi((p.r_body - d) / sig_s);
    }

    /// The heading whose body +y axis (forward, rc convention) points along `dir`.
    inline float heading_along(const Eigen::Vector2f& dir) { return std::atan2(-dir.x(), dir.y()); }

    struct Anchor { Eigen::Vector3f pose = Eigen::Vector3f::Zero(); Eigen::Matrix3f cov = Eigen::Matrix3f::Identity(); float w = 1.f; };

    /// The anchor at the hand-over: TWO components, the body driving through the door FORWARD (heading along n_in) or
    /// BACKING through it (heading along −n_in). ★ Reviewer on 2c353ae3: one component at heading_along(n_in) put a
    /// reversed crossing (a doorway backoff re-crossing into the room left — a recall) ~8σ off in yaw, every mode on the
    /// kidnap floor, and a wrong incumbent held ~1000× more stubbornly. The weight of each is the probability, under the
    /// localiser's own heading at the crossing, that the body faced that way: P(|θ − θ_in| < π/2), Gaussian in θ.
    inline std::vector<Anchor> anchor_at_handover(const Passage& p)
    {
        Anchor a;
        const float half = 0.5f * p.width;
        const float u = std::clamp(p.u, -half, half);
        const Eigen::Vector2f xy = p.centre + p.tangent * u + p.n_in * std::max(p.s, 0.f);
        const float th_in = heading_along(p.n_in);
        a.pose = {xy.x(), xy.y(), th_in};
        // Free play of the centre along the aperture: uniform over the part of the span the body fits in (variance
        // L²/12 of a uniform); the crossing's own σ adds to it. Along the normal: the crossing's σ.
        const float play = std::max(0.f, p.width - p.body_w);
        const float var_t = play * play / 12.f + p.sig_t * p.sig_t;
        const float var_s = p.sig_s * p.sig_s;
        // Yaw: the largest yaw at which the body still fits through the aperture (|ψ| ≤ atan(play / length)), taken
        // as the ± 2σ extent of the heading through the door.
        const float psi = std::atan2(play, std::max(p.body_l, 1e-3f));
        const float var_th = 0.25f * psi * psi + 1e-6f;   // 1e-6: numerical (a door the body exactly fills)
        Eigen::Matrix2f R; R.col(0) = p.tangent; R.col(1) = p.n_in;
        const Eigen::Matrix2f S = R * Eigen::Vector2f(var_t, var_s).asDiagonal() * R.transpose();
        a.cov.setZero();
        a.cov.topLeftCorner<2, 2>() = S;
        a.cov(2, 2) = var_th;
        float p_fwd = 0.5f;
        if (std::isfinite(p.heading))
        {
            const float mu = std::remainder(p.heading - th_in, 2.f * static_cast<float>(M_PI));
            const float sg = std::max(p.sig_heading, 1e-3f);   // 1e-3: numerical (a zero σ is a legal input)
            const float h = 0.5f * static_cast<float>(M_PI);
            p_fwd = std::clamp(phi((h - mu) / sg) - phi((-h - mu) / sg), 0.f, 1.f);
        }
        Anchor b = a;
        b.pose.z() = std::remainder(th_in + static_cast<float>(M_PI), 2.f * static_cast<float>(M_PI));
        a.w = p_fwd; b.w = 1.f - p_fwd;
        return {a, b};
    }

    /// Carry the anchor by one odometry increment `u` = (dx, dy, dθ) in the ROBOT frame, with its covariance `Su`
    /// (robot frame). First-order SE(2) propagation: a' = a ∘ u, Σ' = Ja Σ Jaᵀ + Ju Su Juᵀ.
    inline void propagate(Anchor& a, const Eigen::Vector3f& u, const Eigen::Matrix3f& Su)
    {
        const float c = std::cos(a.pose.z()), s = std::sin(a.pose.z());
        Eigen::Matrix3f Ja = Eigen::Matrix3f::Identity();
        Ja(0, 2) = -s * u.x() - c * u.y();
        Ja(1, 2) =  c * u.x() - s * u.y();
        Eigen::Matrix3f Ju = Eigen::Matrix3f::Identity();
        Ju(0, 0) = c; Ju(0, 1) = -s; Ju(1, 0) = s; Ju(1, 1) = c;
        a.pose.x() += c * u.x() - s * u.y();
        a.pose.y() += s * u.x() + c * u.y();
        a.pose.z()  = std::remainder(a.pose.z() + u.z(), 2.f * static_cast<float>(M_PI));
        a.cov = Ja * a.cov * Ja.transpose() + Ju * Su * Ju.transpose();
    }
}   // namespace rc::passage
