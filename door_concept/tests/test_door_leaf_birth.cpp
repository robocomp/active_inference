// test_door_leaf_birth.cpp — Task 9: a leaf seen side-on points to an opening in its frame wall (door_leaf_birth.h) and the LiDAR opening
// likelihood that scores it (door_opening_edges.h opening_llr / opening_view_llr). Pure functions, no DoorCore.
//   g++ -std=c++23 -O1 -w -I/usr/include/eigen3 -I../common tests/test_door_leaf_birth.cpp src/door_belief.cpp -o /tmp/test_door_leaf_birth && /tmp/test_door_leaf_birth
#include <cmath>
#include <numbers>
#include <print>
#include <vector>

#include "../src/core/bernoulli_filter.h"
#include "../src/door_leaf_birth.h"
#include "../src/door_opening_edges.h"

using namespace rc::door;
using Vec2f = Eigen::Vector2f;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}: {}\n", __FILE__, __LINE__, #c); } } while (0)

static const std::vector<Vec2f> kRoom5{{0.f, 0.f}, {5.f, 0.f}, {5.f, 5.f}, {0.f, 5.f}};
static core::SizePrior typical() { return {0.85f, 0.12f, 2.05f, 0.06f}; }
static RoomBand band() { return RoomBand{0.20f, 0.10f}; }

// A sweep standing at (x0, 2) that reads the wall line y = 5 along [a0, a1] as `through` (return 1 m beyond) or blocked (return on it),
// every 1 cm along the window, at 1 m height.
static RegisteredSweep sweep(const Aperture& ap, bool through_in_span, bool through_everywhere = false, bool empty = false)
{
    RegisteredSweep sw;
    sw.origin = Eigen::Vector3f(1.4f, 2.0f, 0.3f);
    if (empty) return sw;
    for (float a = ap.s - 0.9f; a <= ap.s + ap.w + 0.9f; a += 0.01f)
    {
        const bool in_span = a >= ap.s and a <= ap.s + ap.w;
        const bool thr = through_everywhere or (through_in_span and in_span);
        const float y = thr ? 6.0f : 5.0f;
        // the ray from the origin through the wall line at x = a: the return at depth y along the same ray
        const float t = (y - sw.origin.y()) / (5.0f - sw.origin.y());
        sw.pts.emplace_back(sw.origin.x() + (a - sw.origin.x()) * t, y, 1.0f);
    }
    return sw;
}
static Aperture span_ap()
{
    Aperture ap; ap.wall_O = {0.f, 5.f}; ap.wall_u = {1.f, 0.f}; ap.s = 1.0f; ap.w = 0.85f; ap.h = 2.05f; ap.thickness = 0.1f;
    return ap;
}

int main()
{
    std::vector<int> groups(4);
    for (int i = 0; i < 4; ++i) groups[static_cast<std::size_t>(i)] = i;

    // ── a leaf perpendicular to the y = 5 wall, hinge on it: the aperture is in THAT wall, either side of the hinge ──
    {
        const auto hyps = apertures_from_leaf({2.0f, 5.0f}, {2.0f, 4.15f}, 2.05f, kRoom5, groups, typical(), band());
        CHECK(hyps.size() == 4);
        const auto best = std::ranges::max_element(hyps, {}, &LeafApertureHyp::log_w);
        CHECK(best != hyps.end());
        std::print("leaf_perpendicular: best centre ({:.3f},{:.3f}) w {:.3f} phi {:.1f} deg wall {} log_w {:.3f}\n", best->aperture.centre_xy().x(),
                   best->aperture.centre_xy().y(), best->aperture.w, best->leaf.phi * 180.f / std::numbers::pi_v<float>, best->wall, best->log_w);
        CHECK(std::abs(best->aperture.centre_xy().y() - 5.0f) < 0.05f);   // on the y = 5 wall, not in the leaf's plane (x = 2)
        CHECK(best->wall == groups[2] or best->wall == groups[3]);          // never an x wall
        CHECK(std::abs(best->leaf.phi - std::numbers::pi_v<float> / 2) < 0.05f);
        int near1575 = 0, near2425 = 0;
        for (const auto& h : hyps)
            if (h.log_w > best->log_w - 1e-3)   // the two hypotheses of the hinge END (the far end is 0.85 m off the wall: far below)
            {
                near1575 += std::abs(h.aperture.centre_xy().x() - 1.575f) < 0.05f;
                near2425 += std::abs(h.aperture.centre_xy().x() - 2.425f) < 0.05f;
                CHECK(std::abs(h.aperture.centre_xy().y() - 5.0f) < 0.05f);
            }
        CHECK(near1575 == 1 and near2425 == 1);
        // the far end as the hinge (0.85 m off the wall) is far below the hinge end: the band prior
        double worst = 1e9;
        for (const auto& h : hyps) worst = std::min(worst, h.log_w);
        CHECK(worst < best->log_w - 20.0);
        // swing: the leaf stands on the room side (y < 5): across_u of the wall (nearest edge (5,5)->(0,5), u = -x) is (0,-1)... either way the side is a sign
        CHECK(best->leaf.swing == 1.0f or best->leaf.swing == -1.0f);
    }

    // ── opening_llr sign: through along the span > 0, all blocked < 0, no returns = 0 ──
    {
        const Aperture ap = span_ap();
        const std::pair<float, float> open_curve{std::numbers::pi_v<float> / 2, 1.0f};   // the leaf is wide open: P(open) = 1 - cos(90 deg) = 1
        const std::span<const std::pair<float, float>> oc(&open_curve, 1);
        const float through = opening_llr(ap, sweep(ap, true), 0.802f, 0.0011f, oc);
        const float blocked = opening_llr(ap, sweep(ap, false), 0.802f, 0.0011f, oc);
        const float none = opening_llr(ap, sweep(ap, false, false, true), 0.802f, 0.0011f, oc);
        std::print("opening_llr: through {:.2f} blocked {:.2f} none {:.2f}; per-view {:.3f} / {:.3f} / {:.3f}\n", through, blocked, none,
                   opening_view_llr(ap, sweep(ap, true), 0.802f, 0.0011f, oc), opening_view_llr(ap, sweep(ap, false), 0.802f, 0.0011f, oc),
                   opening_view_llr(ap, sweep(ap, false, false, true), 0.802f, 0.0011f, oc));
        CHECK(through > 0.f);
        CHECK(blocked < 0.f);
        CHECK(none == 0.f);
        // a shut-leaf belief (P(open) = 0): the span is wall to the LiDAR whatever it reads, so BOTH readings are no evidence (p_open = p_wall)
        const std::pair<float, float> shut_curve{0.0f, 1.0f};
        const std::span<const std::pair<float, float>> sc(&shut_curve, 1);
        CHECK(std::abs(opening_llr(ap, sweep(ap, false), 0.802f, 0.0011f, sc)) < 1e-3f);
        CHECK(std::abs(opening_llr(ap, sweep(ap, true), 0.802f, 0.0011f, sc)) < 1e-3f);
        // one sweep is ONE trial: the per-view term is the MEAN per-bin ratio over the span, bounded by the per-bin ratio
        const float v = opening_view_llr(ap, sweep(ap, true), 0.802f, 0.0011f, oc);
        CHECK(v > 0.f and v <= std::log(0.802f / 0.0011f) + 1e-3f);
        // line of sight: a sweep that saw the half of the span only is worth about half of one that saw all of it
        Aperture half = ap;
        RegisteredSweep sw = sweep(ap, true);
        std::erase_if(sw.pts, [&](const Eigen::Vector3f& p) { return p.x() > ap.s + 0.5f * ap.w; });
        const float vh = opening_view_llr(half, sw, 0.802f, 0.0011f, oc);
        std::print("opening_view_llr: full {:.3f} half the span in view {:.3f}\n", v, vh);
        CHECK(vh > 0.35f * v and vh < 0.65f * v);
    }

    // ── a proto room: no wall within the band's reach of either end: no hypothesis can beat the clutter one; with no outline at all, none ──
    {
        const auto none = apertures_from_leaf({2.5f, 2.5f}, {2.5f, 1.65f}, 2.05f, {}, {}, typical(), band());
        CHECK(none.empty());
        const auto hyps = apertures_from_leaf({2.5f, 2.5f}, {2.5f, 1.65f}, 2.05f, kRoom5, groups, typical(), band());
        const double log_clutter = std::log(0.194) + std::log1p(-0.05) - std::log(5.0);   // the fitted ZED false alarm, over a 5 m wall
        double best = -1e9;
        for (const auto& h : hyps) best = std::max(best, -std::log1p(std::exp(-h.log_w)) - std::log(5.0));   // log b / Lambda, before any evidence
        std::print("proto: best leaf hypothesis log b/Lambda {:.1f} vs clutter {:.2f}\n", best, log_clutter);
        CHECK(best < log_clutter - 20.0);
    }

    // ── the plan segment of a mask and the angular likelihoods: flush = aperture, perpendicular = leaf, and they sum to one ──
    {
        std::vector<Eigen::Vector3f> pts;
        for (int i = 0; i <= 20; ++i) for (int j = 0; j <= 10; ++j) pts.emplace_back(2.0f, 5.0f - 0.85f * float(i) / 20.f, 0.2f * float(j));
        const MaskSegment s = plan_segment(pts);
        CHECK(s.ok and std::abs(s.length - 0.85f) < 0.01f and std::abs(s.height - 2.0f) < 0.01f);
        const float sd = segment_angle_sd(s.length, 0.10f);
        CHECK(aperture_angle_likelihood(0.0f, sd) == 1.0);
        CHECK(aperture_angle_likelihood(std::numbers::pi_v<float> / 2, sd) < 1e-6);
        std::print("segment: length {:.3f} sigma_theta {:.1f} deg\n", s.length, sd * 180.f / std::numbers::pi_v<float>);
    }

    if (fails == 0) std::print("ALL PASS\n"); else std::print("FAILED {}\n", fails);
    return fails == 0 ? 0 : 1;
}
