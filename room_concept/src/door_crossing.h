#pragma once

// ── HAS THE ROBOT GONE THROUGH A DOOR? ──────────────────────────────────────────────────────────
// The first half of the proto-room: evidence that the robot is in space the current room does not
// explain, BECAUSE it went through one of that room's open doorways. Losing the pose and crossing a
// door look identical from the SDF misfit alone (both read "the map no longer fits"), so the misfit is
// NOT used here at all. This asks a purely geometric question about the belief we already hold:
//
//   p_geom  = P(footprint past the aperture line) · P(centre within the aperture span)
//   p_cross : logit(p_cross) = logit(p_open) + logit(p_geom)
//
//   · p_open — door_concept's own marginalised belief that the leaf leaves a passable gap. A robot the
//     pose says is "past" a SHUT door is a robot whose pose is wrong, and p_open is what says so.
//   ★FUSED AS EVIDENCE, NOT MULTIPLIED IN AS A CAP. A product p_open·p_geom can never exceed p_open, and
//     door_open_prob is a weak belief: measured 2026-09-14 on the live apartamento it never rose above 0.14
//     (the leaf-angle posterior is nearly flat — see door_concept). A cap would make the proto-room
//     unreachable. Two independent pieces of evidence about the same binary question add in log-odds: a
//     confident "the footprint is past the line" outweighs a lukewarm door belief, while a door believed
//     firmly SHUT (logit ≪ 0) still vetoes a pose that claims to be through it.
//   · past the line — s is the signed distance of the robot centre beyond the aperture, measured AWAY
//     from the room interior; the whole footprint is past once s > r_body. σ_s is the pose covariance
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

    /// Outward unit normal of the aperture: perpendicular to a→b, pointing AWAY from the room interior.
    /// The interior side comes from the polygon's own orientation (signed area) and the polygon edge
    /// nearest the aperture centre — the aperture lies ON the boundary, so the edge it lies on says which
    /// side is inside. No probe distance, nothing to tune. Returns false when the polygon is degenerate.
    inline bool outward_normal(const DoorAperture& ap, const std::vector<Eigen::Vector2f>& polygon,
                               Eigen::Vector2f& n_out)
    {
        const std::size_t n = polygon.size();
        if (n < 3) return false;
        float area2 = 0.f;
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto& p = polygon[i];
            const auto& q = polygon[(i + 1) % n];
            area2 += p.x() * q.y() - q.x() * p.y();
        }
        if (area2 == 0.f) return false;

        const Eigen::Vector2f c = 0.5f * (ap.a + ap.b);
        std::size_t best = 0;
        float best_d2 = std::numeric_limits<float>::max();
        for (std::size_t i = 0; i < n; ++i)
        {
            const Eigen::Vector2f p = polygon[i];
            const Eigen::Vector2f e = polygon[(i + 1) % n] - p;
            const float L2 = e.squaredNorm();
            if (L2 <= 0.f) continue;
            const float u = std::clamp((c - p).dot(e) / L2, 0.f, 1.f);
            if (const float d2 = (c - (p + u * e)).squaredNorm(); d2 < best_d2)
            {
                best_d2 = d2;
                best = i;
            }
        }
        Eigen::Vector2f e = polygon[(best + 1) % n] - polygon[best];
        if (e.squaredNorm() <= 0.f) return false;
        e.normalize();
        // Counter-clockwise polygon (area2 > 0): interior on the LEFT of each edge, so outward = right.
        const Eigen::Vector2f right(e.y(), -e.x());
        n_out = area2 > 0.f ? right : Eigen::Vector2f(-right);
        return true;
    }

    struct Crossing
    {
        float p_cross  = 0.f;   ///< logit-sum of p_open and p_geom = P(past)·P(within span)
        float p_geom   = 0.f;   ///< P(past) · P(within span), the pose's own answer
        float p_past   = 0.f;   ///< P(footprint past the aperture line), no p_open, no span
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

    // ── DIAGNOSTIC CSV: tmp/crossing.csv ────────────────────────────────────────────────────────
    // EVERY aperture evaluated, EVERY cycle — not just the winner. A proto-room that is not born is
    // diagnosed by which FACTOR fell short (p_open vs p_past vs p_span), and the call site sees only
    // the argmax, which cannot show that. The terms are fused in log-odds, so the row carries each one
    // separately: logit(p_cross) = logit(p_open) + logit(p_geom), p_geom = p_past · p_span.
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
        c.p_past = phi((c.s - r_body) / sig_s);
        const float p_span = phi((half - c.u) / sig_t) * phi((half + c.u) / sig_t);
        c.p_geom  = c.p_past * p_span;
        c.p_cross = fuse(ap.p_open, c.p_geom);
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
