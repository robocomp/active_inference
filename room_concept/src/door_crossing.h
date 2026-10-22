#pragma once

// ── HAS THE ROBOT GONE THROUGH A DOOR? ──────────────────────────────────────────────────────────
// The first half of the proto-room: evidence that the robot is in space the current room does not
// explain, BECAUSE it went through one of that room's open doorways. Losing the pose and crossing a
// door look identical from the SDF misfit alone (both read "the map no longer fits"), so the misfit is
// NOT used here at all. This asks a purely geometric question about the belief we already hold:
//
//   p_geom  = P(body centre past the aperture line) · P(centre within the aperture span)
//   p_cross = p_open · p_geom   (a CONJUNCTION since 2026-10-13 — see evaluate(); the logit sum below is history)
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
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <locale>
#include <vector>

#include "door_apertures.h"   // rc::DoorAperture

namespace rc::crossing
{
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
        float p_cross  = 0.f;   ///< p_open · p_geom, p_geom = P(centre past)·P(within span)
        float p_geom   = 0.f;   ///< P(past) · P(within span), the pose's own answer
        float p_past   = 0.f;   ///< P(body centre past the aperture line), no p_open, no span
        float p_span   = 0.f;   ///< P(centre within the aperture span), the other half of p_geom
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

    // ── DIAGNOSTIC CSV: tmp/crossing.csv ────────────────────────────────────────────────────────
    // EVERY aperture evaluated, EVERY cycle — not just the winner. A proto-room that is not born is
    // diagnosed by which FACTOR fell short (p_open vs p_past vs p_span), and the call site sees only
    // the argmax, which cannot show that. The terms are fused in log-odds, so the row carries each one
    // separately: p_cross = p_open · p_geom, p_geom = p_past · p_span.
    // Truncated once per process run, like tmp/door_filter.csv.
    // ★WRITTEN THROUGH THE CLASSIC LOCALE. These machines run LANG=es_ES.UTF-8 and Qt calls
    // setlocale(LC_ALL,"") at startup; an un-imbued stream can emit a COMMA decimal separator into a
    // comma-separated file, which corrupts the file silently. Read it back with std::from_chars.
    inline void log_row(const std::string& name, float p_open, const struct Crossing& c,
                        float r_body, const Eigen::Vector2f& xy);

    /// Evaluate one aperture. `xy`/`cov_xy` are the robot position and its 2×2 covariance, ROOM frame.
    inline Crossing evaluate(const DoorAperture& ap, const std::vector<Eigen::Vector2f>& polygon,
                             const Eigen::Vector2f& xy, const Eigen::Matrix2f& cov_xy, float r_body)
    {
        Crossing c;
        const Eigen::Vector2f ab = ap.b - ap.a;
        const float w = ab.norm();
        if (w <= 0.f or not outward_normal(ap, polygon, c.n_out))
            return c;
        c.tangent = ab / w;
        c.centre  = 0.5f * (ap.a + ap.b);
        const Eigen::Vector2f d = xy - c.centre;
        c.s = d.dot(c.n_out);
        c.u = d.dot(c.tangent);
        // Projected pose variances. The additive 1e-8 is numerical (a zero covariance is a legal input
        // from a frozen or synthetic pose), not a model floor: at 0.1 mm it cannot change any answer.
        const float sig_s = std::sqrt(c.n_out.dot(cov_xy * c.n_out) + 1e-8f);
        const float sig_t = std::sqrt(c.tangent.dot(cov_xy * c.tangent) + 1e-8f);
        const float half = 0.5f * w;
        // ★ WHICH SIDE OF THE DOOR LINE IS THE BODY CENTRE ON (2026-10-21, Task R1b). Rooms partition the floor at
        //   the door line, so "which room is the robot in" is the side its CENTRE is on, under the pose's own
        //   covariance — not "is the whole footprint past". The footprint rule (Φ((s − r_body)/σ)) left a robot that
        //   PAUSED straddling the doorway in the room behind it: live rtD1g2, centre 0.175 m past for 1.5 s (needs
        //   > r_body 0.247), the old room then had to explain a view that was mostly the next room's (median |SDF|
        //   ~1 m), the tracker was dragged back 2–4 m into it, and the hand-over never came. The centre test hands
        //   over as soon as the sensor itself has passed the mouth — exactly when the room left behind stops owing
        //   the view (DoorApertures::weight's doorway branch switches there too). r_body stays the doorway term's.
        c.p_past = phi(c.s / sig_s);
        const float p_span = phi((half - c.u) / sig_t) * phi((half + c.u) / sig_t);
        c.p_geom  = c.p_past * p_span;
        // ★ CONJUNCTION, NOT A LOGIT SUM (2026-10-13). "Crossed through this door" = door passable AND footprint
        //   past it. The logit sum fuse() treated two readings of DIFFERENT propositions as evidence about one, so a
        //   confident open door (+13.8 at the clamp) outvoted a confident "not past" — live try35: p_open 1,
        //   p_past 0.006, p_cross 0.9998 ⇒ a room BORN through a door the robot was 3.5 m from. The sum existed
        //   because door_open_prob never rose above 0.14 (flat leaf posterior); it now reaches ≈1 for an open door.
        //   Unmeasured door (p_open < 0): ½ — no evidence either way.
        c.p_cross = (ap.p_open < 0.f ? 0.5f : std::clamp(ap.p_open, 0.f, 1.f)) * c.p_geom;
        c.p_span  = p_span;
        c.sig_s   = sig_s;
        c.sig_t   = sig_t;
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
                 "sig_s,sig_t,span_w,x,y\n";
            return o;
        }();
        if (not f)
            return;
        const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
        f << ts << ',' << (name.empty() ? std::string{"?"} : name) << ',' << p_open << ','
          << c.p_past << ',' << c.p_span << ',' << c.p_geom << ',' << c.p_cross << ','
          << c.s << ',' << c.u << ',' << r_body << ',' << c.sig_s << ',' << c.sig_t << ','
          << c.span_w << ',' << xy.x() << ',' << xy.y() << '\n';
        f.flush();
    }
}   // namespace rc::crossing
