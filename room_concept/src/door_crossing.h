#pragma once

// ── HAS THE ROBOT GONE THROUGH A DOOR? ──────────────────────────────────────────────────────────
// The first half of the proto-room: evidence that the robot is in space the current room does not
// explain, BECAUSE it went through one of that room's open doorways. Losing the pose and crossing a
// door look identical from the SDF misfit alone (both read "the map no longer fits"), so the misfit is
// NOT used here at all. This asks a purely geometric question about the belief we already hold:
//
//   p_geom  = P(body centre past the aperture line AND the ODOMETRIC path carried it across between the jambs)
//             (Task 16, 2026-10-29: PassageMemory) — with no memory, P(past)·P(current centre within the span)
//   p_cross = P(passed through | path, door prior): three classes (through / through a wall / not crossed), one
//             contact constant ε for leaf and wall alike — see evaluate_core() (Task C 2026-10-23, Task 16)
//
//   · p_open — door_concept's own marginalised belief that the leaf leaves a passable gap. A robot the
//     pose says is "past" a SHUT door is a robot whose pose is wrong, and p_open is what says so.
//   ★ HISTORY: until 2026-10-13 the two were FUSED in log-odds (fuse(), kept below), because door_open_prob never
//     rose above 0.14 then and a product would have capped the birth. That let a confident open door outvote a
//     confident "not past" (try35: a room born 3.5 m from its door); p_open now reaches ≈1 for an open door, so the
//     honest conjunction is used (evaluate()).
//   · past the line — s is the signed distance of the robot centre beyond the aperture, measured AWAY
//     from the room interior; the robot is in the next room once its centre is past (s > 0; R1b). σ_s is the pose covariance
//     projected on the aperture normal, so an uncertain pose gives a soft answer, not a confident one.
//   · within the span — the lateral offset along the aperture must lie inside it, under the pose
//     covariance projected on the aperture tangent. Without this a robot beside the wall, outside the
//     polygon for any other reason (a bad fit), would be read as having used this door.
//
// No threshold lives here. The one decision taken on p_cross (birth of the proto-room) is at its call
// site and flagged there.

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <locale>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "door_apertures.h"   // rc::DoorAperture

namespace rc::crossing
{
    /// ε = P(body passes a SHUT leaf without contact). ⚠ A model constant, not measured: a shut door blocks the
    /// 0.5 m-wide body; the only way "past" happens is the pose being wrong, which the pose σ already prices. 1e-3
    /// says: a pose reading must be ~1000:1 for "past" before it overrides a door believed shut (the controller's
    /// contact model stops the base at a shut leaf, so a genuine passage has no other route). Why no model term
    /// replaces it: ε is the only link between "leaf shut" and "body past"; with ε = 0 a shut door could never
    /// be the explanation of ANY crossing, even when the shut belief is itself wrong (door_concept P(open) is a belief).
    inline constexpr double kShutPassageEps = 1e-3;

    /// Standard normal CDF.
    inline float phi(float z) { return 0.5f * std::erfc(-z / std::sqrt(2.0f)); }

    /// Two independent probabilities of the same event, combined in log-odds (uniform prior).
    /// ⚠ The clamp to [1e-6, 1 - 1e-6] is where float stops representing a probability (Φ of a 25-sigma
    ///   offset is exactly 1.0f, whose logit is +inf). It is not neutral: it bounds how far a very confident
    ///   pose can outvote a door believed shut (|logit| <= 13.8 per term). Flagged — the honest fix is a
    ///   pose-error model with a heavy tail (the pose covariance is known to collapse when parked), not a
    ///   bigger or smaller number here.
    inline float fuse(float p_a, float p_b)
    {
        constexpr double lo = 1e-6, hi = 1.0 - 1e-6;
        const double a = std::clamp(static_cast<double>(p_a), lo, hi);
        const double b = std::clamp(static_cast<double>(p_b), lo, hi);
        const double l = std::log(a / (1.0 - a)) + std::log(b / (1.0 - b));
        return static_cast<float>(1.0 / (1.0 + std::exp(-l)));
    }

    /// Outward unit normal of the aperture: EXACTLY perpendicular to a→b, pointing AWAY from the room interior.
    /// ★ THE AXIS COMES FROM THE APERTURE, THE POLYGON ONLY CHOOSES THE SIGN (2026-10-22, Task S). The normal used
    ///   to be the perpendicular of the polygon edge NEAREST the aperture centre. On a recalled many-vertex outline
    ///   that edge was a jamb/notch edge running ACROSS the door (live crossing/gold1: y = (-1, 0.0115) against
    ///   x = a→b, det 0.019), so the proto frame was near-singular, s/u/σ_s were measured along the wrong axis and
    ///   the robot was placed km away in the proto. Now n = perp(a→b), orthonormal by construction.
    /// The sign is the outward side of the wall edge the door lies IN: of all edges, the one minimising
    /// d² / cos²(angle to a→b) — a continuous preference for near AND parallel-to-the-aperture edges, so a jamb
    /// (cos ≈ 0) can never be chosen; the 1e-6 terms are numerical guards, not model floors. The interior side
    /// comes from the polygon's signed area. Returns false when the polygon or the aperture is degenerate.
    inline bool outward_normal(const DoorAperture& ap, const std::vector<Eigen::Vector2f>& polygon,
                               Eigen::Vector2f& n_out)
    {
        const std::size_t n = polygon.size();
        if (n < 3) return false;
        const Eigen::Vector2f ab = ap.b - ap.a;
        if (ab.squaredNorm() <= 0.f) return false;
        const Eigen::Vector2f t = ab.normalized();
        float area2 = 0.f;
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto& p = polygon[i];
            const auto& q = polygon[(i + 1) % n];
            area2 += p.x() * q.y() - q.x() * p.y();
        }
        if (area2 == 0.f) return false;

        const Eigen::Vector2f c = 0.5f * (ap.a + ap.b);
        std::size_t best = n;
        float best_score = std::numeric_limits<float>::max();
        for (std::size_t i = 0; i < n; ++i)
        {
            const Eigen::Vector2f p = polygon[i];
            const Eigen::Vector2f e = polygon[(i + 1) % n] - p;
            const float L2 = e.squaredNorm();
            if (L2 <= 0.f) continue;
            const float u = std::clamp((c - p).dot(e) / L2, 0.f, 1.f);
            const float d2 = (c - (p + u * e)).squaredNorm();
            const float cos2 = (e.dot(t) * e.dot(t)) / L2;
            if (const float score = (d2 + 1e-6f) / (cos2 + 1e-6f); score < best_score)
            {
                best_score = score;
                best = i;
            }
        }
        if (best == n) return false;
        Eigen::Vector2f e = polygon[(best + 1) % n] - polygon[best];
        e.normalize();
        // Counter-clockwise polygon (area2 > 0): interior on the LEFT of each edge, so outward = right.
        const Eigen::Vector2f right(e.y(), -e.x());
        const Eigen::Vector2f out = area2 > 0.f ? right : Eigen::Vector2f(-right);
        const Eigen::Vector2f perp(-t.y(), t.x());
        n_out = perp.dot(out) >= 0.f ? perp : Eigen::Vector2f(-perp);
        return true;
    }

    struct Crossing
    {
        float p_cross  = 0.f;   ///< P(passed through this aperture | path, door prior) — see evaluate()
        float p_geom   = 0.f;   ///< = p_through, the path's own answer (no door prior)
        float p_through = 0.f;  ///< P(centre past AND the odometric path carried it across between the jambs) (Task 16)
        float p_open_post = 0.f; ///< P(leaf open | path): the posterior the passage implies about the door
        float p_reread = 0.f;    ///< P(past only by a re-read of the door or a shift of the estimate) — "not crossed"
        float p_past   = 0.f;   ///< P(body centre past the aperture line), no p_open, no span
        float p_span   = 0.f;   ///< P(CURRENT centre within the aperture span) — the doorway term's, not the passage's
        float s        = 0.f;   ///< signed distance of the robot centre past the line (m)
        float u        = 0.f;   ///< lateral offset from the aperture centre along a→b (m)
        float sig_s    = 0.f;   ///< pose sigma projected on the aperture NORMAL (m)
        float sig_t    = 0.f;   ///< pose sigma projected on the aperture TANGENT (m)
        float span_w   = 0.f;   ///< aperture width |a→b| (m)
        Eigen::Vector2f centre{0.f, 0.f};
        Eigen::Vector2f tangent{1.f, 0.f};   ///< unit a→b
        Eigen::Vector2f n_out{0.f, 1.f};     ///< unit, away from the room interior
        bool valid = false;
    };

    /// The proto-room frame: origin at the aperture centre, x along a→b, y = n_out (the aperture's perpendicular,
    /// so orthonormal by construction). Right-handed: if the aperture's a/b order made it left-handed, x flips.
    inline Eigen::Affine2f proto_frame(const Crossing& c)
    {
        Eigen::Vector2f x_axis = c.tangent;
        if (x_axis.x() * c.n_out.y() - x_axis.y() * c.n_out.x() < 0.f)
            x_axis = -x_axis;
        Eigen::Affine2f T = Eigen::Affine2f::Identity();
        T.linear().col(0) = x_axis;
        T.linear().col(1) = c.n_out;
        T.translation()   = c.centre;
        return T;
    }

    /// ★ THE MOUTH POSE (2026-10-23, Task M): the proto frame's pose in the OLD room's frame, as the numbers the DSR
    /// carries on the proto room node (rt_translation [x y 0], rt_rotation_euler_xyz [0 0 yaw], rt_covariance 6x6
    /// row-major over [x y z rx ry rz]; a NODE-level overload of the RT attribute names) so ltsm's seam
    /// T(new<-old) = inverse(mouth) needs no door node and no recess constant. The pose IS proto_frame(c).
    /// Covariance, in the old room's frame, with the two sources named:
    ///   · ALONG the aperture (tangent): `sigma_tangent` = door_concept's position σ of the door (ap.sigma_along), the
    ///     uncertainty of where the opening lies along the wall; the caller passes the layout bar when door_concept
    ///     stated none.
    ///   · ALONG THE NORMAL: `sigma_normal` = the uncertainty of the WALL the mouth sits on (seat_in_room projects the
    ///     door onto the room outline's edge, so the mouth's normal coordinate IS the outline's). It is NOT the door's σ:
    ///     the mouth lies ~0.49 m off the door node along the normal (gold2). The caller passes the layout's
    ///     publish_corner_sigma, the σ every published corner met (an upper bound on the edge's position σ).
    ///   · yaw: the two ends of a→b, w apart, each carry σ_normal across the wall: σ_yaw² = 2 σ_normal² / w².
    /// Always stated when the aperture has a width: a precision is derived from the outline, never silently dropped.
    struct Mouth
    {
        float x = 0.f, y = 0.f, yaw = 0.f;
        std::array<float, 36> cov{};
        bool has_cov = false;
    };

    inline Mouth mouth_of(const Crossing& c, float sigma_tangent, float sigma_normal)
    {
        const Eigen::Affine2f T = proto_frame(c);
        Mouth m;
        m.x   = T.translation().x();
        m.y   = T.translation().y();
        m.yaw = std::atan2(T.linear()(1, 0), T.linear()(0, 0));
        if (sigma_tangent > 0.f and sigma_normal > 0.f and c.span_w > 0.f)
        {
            m.has_cov = true;
            const Eigen::Vector2f t = T.linear().col(0), n = T.linear().col(1);
            const Eigen::Matrix2f C = sigma_tangent * sigma_tangent * t * t.transpose()
                                    + sigma_normal * sigma_normal * n * n.transpose();
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    m.cov[i * 6 + j] = C(i, j);
            m.cov[5 * 6 + 5] = 2.f * sigma_normal * sigma_normal / (c.span_w * c.span_w);
        }
        return m;
    }

    // ── DIAGNOSTIC CSV: tmp/crossing.csv ────────────────────────────────────────────────────────
    // EVERY aperture evaluated, EVERY cycle — not just the winner. A proto-room that is not born is
    // diagnosed by which FACTOR fell short (p_open vs p_past vs p_span), and the call site sees only
    // the argmax, which cannot show that. The row carries each one separately: p_span is the CURRENT
    // centre's, p_through (= p_geom) the path's, p_cross the posterior, p_open_post what the path says of the leaf.
    // Truncated once per process run, like tmp/door_filter.csv.
    // ★WRITTEN THROUGH THE CLASSIC LOCALE. These machines run LANG=es_ES.UTF-8 and Qt calls
    // setlocale(LC_ALL,"") at startup; an un-imbued stream can emit a COMMA decimal separator into a
    // comma-separated file, which corrupts the file silently. Read it back with std::from_chars.
    inline void log_row(const std::string& name, float p_open, const struct Crossing& c,
                        float r_body, const Eigen::Vector2f& xy);

    /// One position of the robot, ROOM frame, with its 2×2 covariance.
    struct PoseSample
    {
        Eigen::Vector2f xy{0.f, 0.f};
        Eigen::Matrix2f cov = Eigen::Matrix2f::Zero();
    };

    /// ⚠ P(the localiser's correction this frame is a frame SHIFT, not a continuous reading) a priori — the prior of the
    ///   "teleport" hypothesis in p_continuous(). A model constant, not measured: relocalisation commits are rare events
    ///   (a handful per tour of ~10⁴ frames). Its effect is logarithmic: the innovation at which a correction stops
    ///   counting as continuous sits where N(ν; 0, S) = π_J / ((1 − π_J)·A), i.e. d² ≈ 2 ln((1 − π_J)·A /
    ///   (2π√|S| π_J)) — ≈ 5.8σ for A = 30 m², σ = 1 cm; a factor 10 on π_J moves that by ~0.4σ. Why no model term
    ///   replaces it: the localiser publishes no jump rate.
    inline constexpr double kPoseJumpPrior = 1e-3;

    /// Double-precision Φ: the classes below are computed DIRECTLY (Φ(−s/σ) for "not past"), never as 1 − Φ, so a 10-σ
    /// answer is not cancelled to exactly 0 or 1 by float.
    inline double phi_d(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }

    /// P(the estimate's correction is CONTINUOUS | innovation ν = est − pred, its covariance S, room area A).
    ///   C  continuous:  ν ~ N(0, S)
    ///   J  jump:        the estimate landed anywhere in the room, density 1/A
    /// p = (1 − π_J)·N / ((1 − π_J)·N + π_J/A), π_J = kPoseJumpPrior (flagged). `jump_committed` = the localiser itself
    /// committed a discrete jump this frame (its reloc_epoch advanced): then the answer is 0 by its own declaration.
    inline double p_continuous(const Eigen::Vector2f& nu, const Eigen::Matrix2f& S_in, double area, bool jump_committed)
    {
        if (jump_committed)
            return 0.0;
        // 1e-8 m² is numerical (a zero covariance is legal from a frozen/synthetic pose), not a floor.
        const Eigen::Matrix2d S = S_in.cast<double>() + Eigen::Matrix2d::Identity() * 1e-8;
        const Eigen::Vector2d v = nu.cast<double>();
        const double d2 = v.dot(S.ldlt().solve(v));
        const double N = std::exp(-0.5 * d2) / (2.0 * M_PI * std::sqrt(S.determinant()));
        const double J = kPoseJumpPrior / std::max(area, 1e-6);   // 1e-6 m²: a degenerate polygon, numerical
        const double C = (1.0 - kPoseJumpPrior) * N;
        return C + J > 0.0 ? C / (C + J) : 0.0;
    }

    /// Unsigned area of a polygon (m²).
    inline double polygon_area(const std::vector<Eigen::Vector2f>& poly)
    {
        double a2 = 0.0;
        for (std::size_t i = 0; i < poly.size(); ++i)
        {
            const auto& p = poly[i];
            const auto& q = poly[(i + 1) % poly.size()];
            a2 += double(p.x()) * q.y() - double(q.x()) * p.y();
        }
        return 0.5 * std::abs(a2);
    }

    /// The aperture as one read of it: where the line is, which way is out, and how well both are known.
    struct ApertureGeo
    {
        Eigen::Vector2f centre{0.f, 0.f}, tangent{1.f, 0.f}, n_out{0.f, 1.f};
        double half = 0.0, sig_normal = 0.0, sig_along = 0.0;
    };

    /// One position read against one aperture.
    struct Reading { double s = 0, u = 0, sig_s = 0, sig_t = 0, past = 0, span = 0; };

    /// `half_body` = half the body's WIDTH (the dimension that must clear the jambs). See evaluate_core() for why the
    /// span is the body's, not the centre's.
    inline Reading read(const ApertureGeo& g, const Eigen::Vector2f& p, const Eigen::Matrix2f& cv, double half_body)
    {
        const Eigen::Vector2f d = p - g.centre;
        Reading r;
        r.s = d.dot(g.n_out);
        r.u = d.dot(g.tangent);
        // The additive 1e-8 is numerical (a zero covariance is a legal input from a frozen or synthetic pose), not a
        // model floor: at 0.1 mm it cannot change any answer.
        r.sig_s = std::sqrt(static_cast<double>(g.n_out.dot(cv * g.n_out)) + g.sig_normal * g.sig_normal + 1e-8);
        r.sig_t = std::sqrt(static_cast<double>(g.tangent.dot(cv * g.tangent)) + g.sig_along * g.sig_along + 1e-8);
        // ★ WHICH SIDE OF THE DOOR LINE IS THE BODY CENTRE ON (2026-10-21, Task R1b). Rooms partition the floor at the
        //   door line, so "which room is the robot in" is the side its CENTRE is on, under the pose's own covariance —
        //   not "is the whole footprint past" (live rtD1g2: a robot paused straddling the doorway was left in the room
        //   behind it, the tracker dragged 2–4 m back). r_body stays the doorway term's.
        r.past = phi_d(r.s / r.sig_s);
        const double free = g.half - half_body;   // may be negative: an aperture narrower than the body
        r.span = phi_d((free - r.u) / r.sig_t) * phi_d((free + r.u) / r.sig_t);
        return r;
    }

    /// Per-door passage state (round 5): the ANCHOR read the robot's motion is judged against, the passage mass A, the
    /// jumped mass J, and the estimate / cumulative odometric displacement / continuity at this door's last update.
    struct DoorPassage
    {
        ApertureGeo geo;                       ///< G*: the anchor read of the line
        double thru = 0.0;                     ///< A: P(past G* AND the body got there by odometric motion between its jambs)
        double jumped = 0.0;                   ///< J: P(past G* only because the estimate was shifted — no motion)
        PoseSample last;                       ///< the estimate at this door's last update
        Eigen::Vector2f d_last{0.f, 0.f};      ///< the cumulative odometric displacement then
        double w_since = 1.0;                  ///< P(every estimate correction since then was continuous)
    };

    inline Crossing evaluate_core(const DoorAperture& ap, const std::vector<Eigen::Vector2f>& polygon,
                                  const Eigen::Vector2f& xy, const Eigen::Matrix2f& cov_xy, float r_body,
                                  float sigma_normal, double half_body, DoorPassage* state, const Eigen::Vector2f& d_odo);

    /// Evaluate one aperture from the CURRENT pose alone (no memory): the pre-Task-16 reading, with the centre's span
    /// (half_body = 0). `xy`/`cov_xy` are the robot position and its 2×2 covariance, ROOM frame.
    inline Crossing evaluate(const DoorAperture& ap, const std::vector<Eigen::Vector2f>& polygon,
                             const Eigen::Vector2f& xy, const Eigen::Matrix2f& cov_xy, float r_body,
                             float sigma_normal = 0.f);

    /// ★ THE PASSAGE MEMORY (Task 16, round 5 — 2026-10-29). One per room (the caller clears it on a room change and
    ///   moves it with the frame on a re-anchor). The robot's position IS the localiser's estimate (after a relocalisation
    ///   the relocaliser is trusted: room_concept owns the pose — controller ruling). What the memory adds is the SPLIT of
    ///   each frame's displacement: the ODOMETRIC step (pred − previous estimate: motion of the body, both hypotheses
    ///   below agree on it) and the estimate's CORRECTION (est − pred), which with probability w = p_continuous(est − pred,
    ///   P_pred + Σ) is a continuous re-seed (the labels of the past mass are kept) and with 1 − w a frame shift (what it
    ///   pushes past the line carries no passage: it is "jumped", priced as not crossed; 0 on a committed relocalisation).
    ///   Per door, an ANCHOR read of the line and the two masses A / J (evaluate_core).
    ///   Cycle: begin_frame(est, pred, area, jump_committed, pred_cov) → evaluate(...) per door.
    class PassageMemory
    {
        std::unordered_map<std::uint64_t, DoorPassage> doors_;
        std::optional<PoseSample> prev_;         ///< the previous frame's estimate (pred − prev_ = the odometric step)
        Eigen::Vector2f d_{0.f, 0.f};            ///< cumulative odometric displacement since the memory started
        double half_body_ = 0.0;
        double w_last_ = 1.0;
        static std::uint64_t key(const DoorAperture& ap)
        { return ap.id != 0 ? ap.id : std::hash<std::string>{}(ap.name); }
    public:
        explicit PassageMemory(float half_body_width = 0.f) : half_body_(half_body_width) {}
        void set_half_body_width(float h) { half_body_ = h; }
        void clear() { doors_.clear(); prev_.reset(); d_.setZero(); w_last_ = 1.0; }
        std::size_t doors() const { return doors_.size(); }
        double last_continuity() const { return w_last_; }   ///< w of the last frame's correction

        /// `pred` = the localiser's odometry prediction of this frame (previous estimate + this frame's odometric step),
        /// `pred_cov` its covariance P_pred = F P F' + Q (the innovation est − pred has covariance ≤ P_pred; P_pred + Σ_est
        /// is the generous side, so a doorway snap the prediction's own noise explains is never read as a jump — fix
        /// round 2). A zero / non-finite pred_cov (a caller without one) falls back to Σ_now + Σ_prev.
        void begin_frame(const Eigen::Vector2f& xy, const Eigen::Matrix2f& cov, const Eigen::Vector2f& pred,
                         double area, bool jump_committed,
                         const Eigen::Matrix2f& pred_cov = Eigen::Matrix2f::Zero())
        {
            if (prev_.has_value())
            {
                d_ += pred - prev_->xy;
                const bool have_pred_cov = pred_cov.allFinite() and pred_cov.trace() > 0.f;
                const Eigen::Matrix2f S = have_pred_cov ? Eigen::Matrix2f(pred_cov + cov) : Eigen::Matrix2f(cov + prev_->cov);
                w_last_ = p_continuous(xy - pred, S, area, jump_committed);
                for (auto& [k, d] : doors_) d.w_since *= w_last_;
            }
            prev_ = PoseSample{xy, cov};
        }

        Crossing evaluate(const DoorAperture& ap, const std::vector<Eigen::Vector2f>& polygon,
                          const Eigen::Vector2f& xy, const Eigen::Matrix2f& cov_xy, float r_body, float sigma_normal);

        /// The frame moved: p_new = T · p_old (the estimator's re-anchor).
        void apply(const Eigen::Affine2f& T)
        {
            const Eigen::Matrix2f R = T.linear();
            const auto mv = [&](PoseSample& p) { p.xy = T * p.xy; p.cov = R * p.cov * R.transpose(); };
            if (prev_.has_value()) mv(*prev_);
            d_ = R * d_;
            for (auto& [k, d] : doors_)
            {
                mv(d.last);
                d.d_last      = R * d.d_last;
                d.geo.centre  = T * d.geo.centre;
                d.geo.tangent = (R * d.geo.tangent).normalized();
                d.geo.n_out   = (R * d.geo.n_out).normalized();
            }
        }
    };

    inline Crossing evaluate_core(const DoorAperture& ap, const std::vector<Eigen::Vector2f>& polygon,
                                  const Eigen::Vector2f& xy, const Eigen::Matrix2f& cov_xy, float r_body,
                                  float sigma_normal, double half_body, DoorPassage* state, const Eigen::Vector2f& d_odo)
    {
        Crossing c;
        const Eigen::Vector2f ab = ap.b - ap.a;
        const float w = ab.norm();
        if (w <= 0.f or not outward_normal(ap, polygon, c.n_out))
            return c;
        c.tangent = ab / w;
        c.centre  = 0.5f * (ap.a + ap.b);
        // ★ THE APERTURE IS A BELIEF TOO (fix round 1, Task C review). "Past the line" is measured against an ESTIMATED line:
        //   its position error adds to the pose's. ACROSS the line (normal) the error is the wall's: `sigma_normal`, the
        //   caller's layout σ (publish_corner_sigma, the same σ Task M gives the mouth; door node and mouth differed by
        //   ~0.49 m along the normal on gold2). ALONG it, door_concept's own position σ (ap.sigma_along), or the same
        //   layout σ when door_concept stated none. Never 0 in the caller; without them (default 0, tests/legacy) the
        //   pose alone decides, which is the over-confident reading this exists to correct.
        ApertureGeo g;
        g.centre = c.centre; g.tangent = c.tangent; g.n_out = c.n_out;
        g.half = 0.5 * w;
        g.sig_normal = sigma_normal;
        g.sig_along  = ap.sigma_along > 0.f ? ap.sigma_along : sigma_normal;

        // ★ THE BODY MUST CLEAR THE JAMBS, NOT ONLY ITS CENTRE (Task 16 fix round 1). A body of half-width h passing an
        //   aperture of half-width W proves |u| ≤ W − h at the passing frame: span = Φ((W−h−u)/σ_t)·Φ((W−h+u)/σ_t). A
        //   0.3 m aperture can no longer pass a 0.46 m robot (W − h < 0 ⇒ span ≈ 0), and a passage that clipped a jamb
        //   lands in the "through a wall" class, which the contact model prices at ε like a shut leaf. h is the body's
        //   half-WIDTH (the dimension across the doorway), not r_body (half the larger dimension, the doorway term's).
        //   The memory-less evaluate() keeps h = 0 (the centre test it always was).
        const Reading now = read(g, xy, cov_xy, half_body);   // the estimate against the CURRENT read

        // ★ DID THE BODY GO THROUGH THE APERTURE — A QUESTION ABOUT THE ODOMETRIC PATH, NOT THE POSE (Task 16, round 5).
        //   recog_proto5 leg 5: centre 2.73 m past door_39, then 1.4 m sideways along the new room ⇒ p_span(now) 2e-11,
        //   no birth: the span belongs to the moment of PASSAGE. And a passage is MOTION of the body: the estimate
        //   jumping (relocalisation, try58: 4.32 m in a doorway), jittering, or the door's read moving across a parked
        //   robot moves no body across anything. So, per door, against an anchor read G*:
        //   1. THE ODOMETRIC STEP since this door's last update (the estimate then + the odometric displacement since) is
        //      the only thing that CREATES label: Δ = ΔP*(past) > 0 enters A × span at the point the step meets the line;
        //      Δ < 0 leaves A and J in proportion. A parked robot (zero odometric step) never adds passage, whatever the
        //      estimate, the read or the jitter do; A is 0 until the odometry has carried the body across.
        //   2. THE ESTIMATE'S CORRECTION (from there to the estimate now) is a mixture: with w = P(continuous since the last
        //      update) a re-seed that KEEPS the labels (A, J rescale with P*(past): the fraction is the label); with 1 − w
        //      a frame SHIFT that carries no passage — what it pushes past the line is J ("jumped"), what it pulls back
        //      leaves in proportion. J is never converted by motion onward: a jump that lands the body past the door is
        //      not a passage; driving back and crossing again is.
        //   3. THE ANCHOR follows the current read with κ = sides · min(1, |odometric step| / |read − anchor|), sides =
        //      (1 − |P(past)_read − P(past)_G*|)·|2P(past)_read − 1|·|2P(past)_G* − 1| (the estimate against each line):
        //      1 when the body is clearly on the SAME side of both (re-anchoring changes nothing it is judged by), → 0 near
        //      either line or on different sides. The pace factor makes κ a weight on MOTION, not on time (a per-frame
        //      blend integrates: 200 frames at 1.4 % walked the anchor through a parked robot): a parked robot's anchor is
        //      frozen whatever the read does, a driving robot re-adopts a corrected read at the pace of its own approach.
        //      κ is a constant-free SPEED LIMIT on the anchor (shaping), not a generative term: it states no likelihood,
        //      it only bounds how fast the line the motion is judged against may be redrawn. Its move rescales A and J.
        //      ⚠ By design: a line that moved INTO the room while the robot was clearly in front of both is adopted as it
        //      drives on, and later motion across it IS a passage under the belief. A read whose σ is over-stated keeps
        //      the body "near the line" longer (sides small), so door_concept must state its σ honestly.
        //   LABELS of the current past mass p_now = P(estimate past the CURRENT read): R = max(0, p_now − P*(past)) (past
        //   under the read but not under the anchor: a re-read) + J's share — both the physical state "not crossed",
        //   priced like N; the rest carries A, and W = the remainder (through a wall/jamb).
        double qA = 0.0, R = 0.0;
        if (state == nullptr)
            qA = now.past * now.span;   // no memory: the instantaneous reading
        else
        {
            // FIRST SIGHT of this door: no path relative to it yet. Whatever of the robot is already past its line got there
            // BEFORE the door was known, so none of it is evidence of a passage through it: the past mass starts in J ("not
            // crossed — past only because the read appeared there"), and only motion seen from now on can move mass into A.
            // ★It used to start in A as past·span (the instantaneous reading): recog_proto15's door_46, first seen with the
            //  robot standing in ANOTHER doorway past its line (p_span 0.08), got A 0.08 against W ~0.9 priced at ε — p_cross
            //  0.954, a false crossing and a duplicate room. Cost: a door first seen with the body already partly past its line
            //  can never credit that part (p_cross ≤ 1 − past at first sight) — crossing_passage_test (n).
            if (state->geo.half <= 0.0)
            {
                state->geo = g; state->thru = 0.0; state->jumped = now.past;
                state->last = PoseSample{xy, cov_xy}; state->d_last = d_odo; state->w_since = 1.0;
            }
            const ApertureGeo& G = state->geo;
            double A = state->thru, J = state->jumped;
            // 1. the odometric step
            const Reading a = read(G, state->last.xy, state->last.cov, half_body);
            const Reading b = read(G, state->last.xy + (d_odo - state->d_last), cov_xy, half_body);
            if (b.past > a.past)
            {
                const double ds = b.s - a.s;
                const double f  = ds != 0.0 ? std::clamp(-a.s / ds, 0.0, 1.0) : 1.0;   // step ∩ line (geometry clamp)
                const double uc = a.u + f * (b.u - a.u), st = a.sig_t + f * (b.sig_t - a.sig_t);
                const double free = G.half - half_body;
                A += (b.past - a.past) * phi_d((free - uc) / st) * phi_d((free + uc) / st);
            }
            else if (a.past > 0.0)
            {
                A *= b.past / a.past; J *= b.past / a.past;
            }
            // 2. the estimate's correction: continuous re-seed (labels kept) vs frame shift (pushes J)
            const Reading e = read(G, xy, cov_xy, half_body);
            const double wc = state->w_since;
            const double r  = b.past > 0.0 ? std::min(e.past / b.past, 1.0) : 0.0;   // ≤ 1: the shrink both share
            const double Ac = b.past > 0.0 ? A * e.past / b.past : 0.0, Jc = b.past > 0.0 ? J * e.past / b.past : 0.0;
            const double Aj = A * r, Jj = J * r + std::max(0.0, e.past - b.past);
            A = wc * Ac + (1.0 - wc) * Aj; J = wc * Jc + (1.0 - wc) * Jj;
            // 3. the anchor follows the read at the pace of the body
            const double sides = (1.0 - std::abs(now.past - e.past)) * std::abs(2.0 * now.past - 1.0) * std::abs(2.0 * e.past - 1.0);
            const float gh = static_cast<float>(g.half), Gh = static_cast<float>(G.half);
            const float dist = std::max((g.centre - gh * g.tangent - (G.centre - Gh * G.tangent)).norm(),
                                        (g.centre + gh * g.tangent - (G.centre + Gh * G.tangent)).norm());
            const double moved = (d_odo - state->d_last).norm();
            const double kappa = dist > 0.f ? sides * std::min(1.0, moved / dist) : 0.0;
            const float k = static_cast<float>(kappa);
            ApertureGeo Gn = G;
            Gn.centre  = G.centre + k * (g.centre - G.centre);
            Gn.tangent = (G.tangent + k * (g.tangent - G.tangent)).normalized();
            Gn.n_out   = (G.n_out + k * (g.n_out - G.n_out)).normalized();
            Gn.half       = G.half + kappa * (g.half - G.half);
            Gn.sig_normal = G.sig_normal + kappa * (g.sig_normal - G.sig_normal);
            Gn.sig_along  = G.sig_along + kappa * (g.sig_along - G.sig_along);
            if (not Gn.tangent.allFinite() or not Gn.n_out.allFinite()) Gn = g;   // antiparallel reads: numerical
            const Reading on = read(Gn, xy, cov_xy, half_body);
            const double ra = e.past > 0.0 ? on.past / e.past : 0.0;
            A = std::min(A * ra, on.past); J = std::min(J * ra, on.past - A);   // A + J ≤ P*(past): numerical
            state->geo = Gn; state->thru = A; state->jumped = J;
            state->last = PoseSample{xy, cov_xy}; state->d_last = d_odo; state->w_since = 1.0;
            // 4. the labels of the current past mass
            const double share = on.past > 0.0 ? std::min(1.0, now.past / on.past) : 0.0;
            qA = A * share;
            R  = std::max(0.0, now.past - on.past) + J * share;
        }

        c.s       = static_cast<float>(now.s);
        c.u       = static_cast<float>(now.u);
        c.sig_s   = static_cast<float>(now.sig_s);
        c.sig_t   = static_cast<float>(now.sig_t);
        c.p_past  = static_cast<float>(now.past);
        c.p_span  = static_cast<float>(now.span);
        c.p_through = static_cast<float>(qA);
        c.p_reread  = static_cast<float>(R);
        c.p_geom  = c.p_through;

        // ★ THE PASSAGE IS EVIDENCE ABOUT THE LEAF — three classes, not two (Task C 2026-10-23; Task 16 2026-10-29).
        //   The path posterior splits the robot's whereabouts into
        //     A = past, entered between the jambs (q_A),
        //     W = past through a wall/jamb (q_W = p_now − q_A − R),
        //     N = not crossed: not past (Φ(−s/σ_s), computed directly) PLUS R — "past" only because the door's read or
        //         the pose estimate moved, the body itself did not.
        //   The door supplies the physical prior over them, with ONE contact constant ε (a body does not pass a solid
        //   barrier, leaf or wall, without contact):
        //     open:  A (1−ε)/2   N (1−ε)/2   W ε          (A vs N: the same flat prior the pose reading assumes)
        //     shut:  A ε         N 1−2ε      W ε
        //   p_cross = P(A | path) ∝ P(A)·q_A over the three. Task C's two-class form lumped W into "not crossed" with
        //   the flat ½. Now the passage overrides a wrong P(open) prior exactly as the contact model says it should:
        //   p_open 1e-4 ⇒ prior odds of A ≈ 1e-3, and a path through the jambs with q_N ≈ 0 (the body is metres past)
        //   gives p_cross → 1; a pose jitter of a few σ past a SHUT door (q_N ~ 1e-3) cannot.
        //   P(open | path) is reported too (p_open_post). ε = kShutPassageEps (flagged there); the 0.95 decision
        //   boundary at the call site acts on this posterior. Unmeasured door (p_open < 0): ½.
        {
            const double e  = kShutPassageEps;
            const double po = ap.p_open < 0.f ? 0.5 : std::clamp(static_cast<double>(ap.p_open), 0.0, 1.0);
            const double qW = std::max(0.0, now.past - qA - R);
            const double qN = phi_d(-now.s / now.sig_s) + R;
            const double open_A = 0.5 * (1.0 - e), open_N = 0.5 * (1.0 - e), open_W = e;
            const double shut_A = e, shut_N = 1.0 - 2.0 * e, shut_W = e;
            const double jA = (po * open_A + (1.0 - po) * shut_A) * qA;
            const double jN = (po * open_N + (1.0 - po) * shut_N) * qN;
            const double jW = (po * open_W + (1.0 - po) * shut_W) * qW;
            const double den = jA + jN + jW;
            c.p_cross = den > 0.0 ? static_cast<float>(jA / den) : 0.f;
            const double j_open = po * (open_A * qA + open_N * qN + open_W * qW);
            c.p_open_post = den > 0.0 ? static_cast<float>(j_open / den) : static_cast<float>(po);
        }
        c.span_w  = w;
        c.valid = true;
        log_row(ap.name, ap.p_open, c, r_body, xy);
        return c;
    }

    inline Crossing evaluate(const DoorAperture& ap, const std::vector<Eigen::Vector2f>& polygon,
                             const Eigen::Vector2f& xy, const Eigen::Matrix2f& cov_xy, float r_body,
                             float sigma_normal)
    {
        return evaluate_core(ap, polygon, xy, cov_xy, r_body, sigma_normal, 0.0, nullptr, Eigen::Vector2f::Zero());
    }

    inline Crossing PassageMemory::evaluate(const DoorAperture& ap, const std::vector<Eigen::Vector2f>& polygon,
                                            const Eigen::Vector2f& xy, const Eigen::Matrix2f& cov_xy, float r_body,
                                            float sigma_normal)
    {
        return evaluate_core(ap, polygon, xy, cov_xy, r_body, sigma_normal, half_body_, &doors_[key(ap)], d_);
    }

    inline void log_row(const std::string& name, float p_open, const Crossing& c,
                        float r_body, const Eigen::Vector2f& xy)
    {
        static std::ofstream f = []
        {
            std::ofstream o("tmp/crossing.csv", std::ios::out | std::ios::trunc);
            o.imbue(std::locale::classic());
            o << "ts_ms,door,p_open,p_past,p_span,p_geom,p_cross,s,u,r_body,"
                 "sig_s,sig_t,span_w,x,y,p_through,p_open_post,p_reread\n";
            return o;
        }();
        if (not f)
            return;
        const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
        f << ts << ',' << (name.empty() ? std::string{"?"} : name) << ',' << p_open << ','
          << c.p_past << ',' << c.p_span << ',' << c.p_geom << ',' << c.p_cross << ','
          << c.s << ',' << c.u << ',' << r_body << ',' << c.sig_s << ',' << c.sig_t << ','
          << c.span_w << ',' << xy.x() << ',' << xy.y() << ',' << c.p_through << ',' << c.p_open_post << ',' << c.p_reread << '\n';
        f.flush();
    }
}   // namespace rc::crossing
