#pragma once

// ── HAS THE ROBOT GONE THROUGH A DOOR? ──────────────────────────────────────────────────────────
// The first half of the proto-room: evidence that the robot is in space the current room does not
// explain, BECAUSE it went through one of that room's open doorways. Losing the pose and crossing a
// door look identical from the SDF misfit alone (both read "the map no longer fits"), so the misfit is
// NOT used here at all. This asks a purely geometric question about the belief we already hold:
//
//   p_geom  = P(body centre past the aperture line AND it crossed the line between the jambs), over the stored
//             PATH (Task 16, 2026-10-29) — with no path, P(past)·P(current centre within the span)
//   p_cross = P(passed through | path, door prior): three classes (through / through a wall / not past), one
//             contact constant ε for leaf and wall alike — see evaluate() (Task C 2026-10-23, Task 16 2026-10-29)
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
#include <span>
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
        float p_through = 0.f;  ///< P(centre past AND it entered between the jambs), over the stored path (Task 16)
        float p_open_post = 0.f; ///< P(leaf open | path): the posterior the passage implies about the door
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

    /// One earlier pose of the robot, ROOM frame: where it was and how sure the localiser was of it.
    struct PoseSample
    {
        Eigen::Vector2f xy{0.f, 0.f};
        Eigen::Matrix2f cov = Eigen::Matrix2f::Zero();
    };

    /// ⚠ MEMORY BOUND on the stored passage trajectory (samples, oldest dropped). Not a model constant: the caller
    ///   stores one sample per pose-σ of motion (a parked robot adds nothing), so 4096 samples is ~40–80 m of path at
    ///   the 1–2 cm σ the localiser runs at — far longer than any approach-and-cross. When the oldest sample is
    ///   dropped the recursion simply starts later, from the instantaneous answer at that sample.
    inline constexpr std::size_t kMaxPassageSamples = 4096;

    /// The robot's path in the current room's frame, as the crossing evaluation reads it (Task 16). Owned by the caller
    /// (RoomSceneGraph), one per room: cleared when the room changes, moved with the frame on a re-anchor.
    /// A new sample is stored only once the pose has moved by more than its own σ (√(½·tr Σ)) since the last stored one:
    /// two poses closer than the belief's resolution say nothing new about which side of a line the robot is on, and a
    /// parked robot then adds nothing. That is the belief's resolution, not a tuned step.
    class PassagePath
    {
        std::vector<PoseSample> s_;
    public:
        void clear() { s_.clear(); }
        std::span<const PoseSample> samples() const { return s_; }
        void observe(const Eigen::Vector2f& xy, const Eigen::Matrix2f& cov)
        {
            const float sig = std::sqrt(std::max(0.f, 0.5f * cov.trace()));
            if (not s_.empty() and not ((xy - s_.back().xy).norm() > sig))
                return;
            if (s_.size() >= kMaxPassageSamples)
                s_.erase(s_.begin());
            s_.push_back({xy, cov});
        }
        /// The frame moved: p_new = T · p_old (the estimator's re-anchor, absorbed by the caller).
        void apply(const Eigen::Affine2f& T)
        {
            const Eigen::Matrix2f R = T.linear();
            for (auto& p : s_) { p.xy = T * p.xy; p.cov = R * p.cov * R.transpose(); }
        }
    };

    /// Double-precision Φ for the posterior: the three classes below are computed DIRECTLY (Φ(−s/σ) for "not past"),
    /// never as 1 − Φ, so a 10-σ answer is not cancelled to exactly 0 or 1 by float.
    inline double phi_d(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }

    /// Evaluate one aperture. `xy`/`cov_xy` are the robot position and its 2×2 covariance, ROOM frame.
    /// `history` = the robot's EARLIER poses in the same frame, oldest first, not including the current one (empty:
    /// the passage is judged from the current pose alone, the pre-Task-16 reading).
    inline Crossing evaluate(const DoorAperture& ap, const std::vector<Eigen::Vector2f>& polygon,
                             const Eigen::Vector2f& xy, const Eigen::Matrix2f& cov_xy, float r_body,
                             float sigma_normal = 0.f, std::span<const PoseSample> history = {})
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
        // The additive 1e-8 is numerical (a zero covariance is a legal input from a frozen or synthetic pose), not a
        // model floor: at 0.1 mm it cannot change any answer.
        const double sig_along = ap.sigma_along > 0.f ? ap.sigma_along : sigma_normal;
        const double half = 0.5 * w;
        struct Reading { double s, u, sig_s, sig_t, past, span; };
        const auto read = [&](const Eigen::Vector2f& p, const Eigen::Matrix2f& cv) -> Reading
        {
            const Eigen::Vector2f d = p - c.centre;
            Reading r{};
            r.s = d.dot(c.n_out);
            r.u = d.dot(c.tangent);
            r.sig_s = std::sqrt(static_cast<double>(c.n_out.dot(cv * c.n_out)) + double(sigma_normal) * sigma_normal + 1e-8);
            r.sig_t = std::sqrt(static_cast<double>(c.tangent.dot(cv * c.tangent)) + sig_along * sig_along + 1e-8);
            // ★ WHICH SIDE OF THE DOOR LINE IS THE BODY CENTRE ON (2026-10-21, Task R1b). Rooms partition the floor at
            //   the door line, so "which room is the robot in" is the side its CENTRE is on, under the pose's own
            //   covariance — not "is the whole footprint past" (live rtD1g2: a robot paused straddling the doorway was
            //   left in the room behind it, the tracker dragged 2–4 m back). r_body stays the doorway term's.
            r.past = phi_d(r.s / r.sig_s);
            r.span = phi_d((half - r.u) / r.sig_t) * phi_d((half + r.u) / r.sig_t);
            return r;
        };
        const Reading now = read(xy, cov_xy);

        // ★ DID THE BODY GO THROUGH THE APERTURE — A QUESTION ABOUT THE PATH, NOT THE POSE (2026-10-29, Task 16).
        //   Until now "within the span" was asked of the CURRENT pose: P(past)·P(u now inside the jambs). Once through,
        //   the robot walks wherever the new room lets it, and the question answers the wrong thing — recog_proto5 leg 5:
        //   centre 2.73 m past door_39, then 1.4 m sideways along the new room ⇒ p_span 2e-11, p_cross 1e-12, no birth,
        //   the robot stood in room_2 believed to be in room_3. The span belongs to the moment of PASSAGE.
        //   Mass flow over the stored path, every sample re-read against the CURRENT aperture (so a re-estimated door —
        //   door_39 moved 2 m mid-leg — is judged against the path, not against where the robot stood when it moved):
        //     thru = P(centre past the line AND it got there between the jambs)
        //     step k: Δ = P(past)_k − P(past)_{k−1}. Δ ≥ 0 is mass flowing front→past across the line in this step; it
        //     enters through the jambs with P(span) at the point the segment meets the line (the segment point nearest
        //     the line when it does not reach it). Δ < 0 is mass flowing back; it leaves both kinds of "past"
        //     proportionally (thru ← thru · P_k / P_{k−1}). By construction 0 ≤ thru ≤ P(past).
        //   The sign of Δ is the direction of the flow, not a gate; the first sample starts from its instantaneous
        //   answer P(past)·P(span), so an empty history reproduces the old reading exactly.
        double thru = 0.0;
        {
            bool first = true;
            Reading prev{};
            const auto step = [&](const Reading& r)
            {
                if (first) { thru = r.past * r.span; first = false; prev = r; return; }
                const double delta = r.past - prev.past;
                if (delta >= 0.0)
                {
                    const double ds = r.s - prev.s;
                    const double f  = ds != 0.0 ? std::clamp(-prev.s / ds, 0.0, 1.0) : 1.0;   // segment ∩ line (geometry clamp)
                    const double uc = prev.u + f * (r.u - prev.u);
                    const double st = prev.sig_t + f * (r.sig_t - prev.sig_t);
                    thru += delta * phi_d((half - uc) / st) * phi_d((half + uc) / st);
                }
                else if (prev.past > 0.0)
                    thru *= r.past / prev.past;
                thru = std::min(thru, r.past);   // float-sum guard; the invariant holds exactly in real arithmetic
                prev = r;
            };
            for (const auto& h : history) step(read(h.xy, h.cov));
            step(now);
        }

        c.s       = static_cast<float>(now.s);
        c.u       = static_cast<float>(now.u);
        c.sig_s   = static_cast<float>(now.sig_s);
        c.sig_t   = static_cast<float>(now.sig_t);
        c.p_past  = static_cast<float>(now.past);
        c.p_span  = static_cast<float>(now.span);
        c.p_through = static_cast<float>(thru);
        c.p_geom  = c.p_through;

        // ★ THE PASSAGE IS EVIDENCE ABOUT THE LEAF — three classes, not two (Task C 2026-10-23; Task 16 2026-10-29).
        //   The path posterior splits the robot's whereabouts into
        //     A = past, entered between the jambs (q_A = thru),
        //     W = past, entered OUTSIDE the jambs — through a wall (q_W = P(past) − thru),
        //     N = not past (q_N = Φ(−s/σ_s), computed directly).
        //   The door supplies the physical prior over them, with ONE contact constant ε (a body does not pass a solid
        //   barrier, leaf or wall, without contact):
        //     open:  A (1−ε)/2   N (1−ε)/2   W ε          (A vs N: the same flat prior the pose reading assumes)
        //     shut:  A ε         N 1−2ε      W ε
        //   p_cross = P(A | path) ∝ P(A)·q_A over the three. Task C's two-class form lumped W into "not crossed" with
        //   the flat ½ — it read a wall-passage as ordinary evidence AGAINST the door, so any sideways uncertainty at the
        //   mouth was charged at full weight. Now the passage overrides a wrong P(open) prior exactly as the contact
        //   model says it should: p_open 1e-4 ⇒ prior odds of A ≈ 1e-3, and a path through the jambs with q_N ≈ 0
        //   (the body is metres past) gives p_cross → 1; a pose jitter of a few σ past a SHUT door (q_N ~ 1e-3) cannot.
        //   P(open | path) is reported too (p_open_post), the posterior the passage itself implies about the leaf.
        //   ε = kShutPassageEps (flagged there); the 0.95 decision boundary at the call site acts on this posterior.
        //   Unmeasured door (p_open < 0): ½.
        {
            const double e  = kShutPassageEps;
            const double po = ap.p_open < 0.f ? 0.5 : std::clamp(static_cast<double>(ap.p_open), 0.0, 1.0);
            const double qA = thru;
            const double qW = std::max(0.0, now.past - thru);
            const double qN = phi_d(-now.s / now.sig_s);
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

    inline void log_row(const std::string& name, float p_open, const Crossing& c,
                        float r_body, const Eigen::Vector2f& xy)
    {
        static std::ofstream f = []
        {
            std::ofstream o("tmp/crossing.csv", std::ios::out | std::ios::trunc);
            o.imbue(std::locale::classic());
            o << "ts_ms,door,p_open,p_past,p_span,p_geom,p_cross,s,u,r_body,"
                 "sig_s,sig_t,span_w,x,y,p_through,p_open_post\n";
            return o;
        }();
        if (not f)
            return;
        const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
        f << ts << ',' << (name.empty() ? std::string{"?"} : name) << ',' << p_open << ','
          << c.p_past << ',' << c.p_span << ',' << c.p_geom << ',' << c.p_cross << ','
          << c.s << ',' << c.u << ',' << r_body << ',' << c.sig_s << ',' << c.sig_t << ','
          << c.span_w << ',' << xy.x() << ',' << xy.y() << ',' << c.p_through << ',' << c.p_open_post << '\n';
        f.flush();
    }
}   // namespace rc::crossing
