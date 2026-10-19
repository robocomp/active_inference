// try102 (2026-10-19): the leaf channel (src/door_leaf_channel.h) — the code DoorFitter::estimate_phi runs — against the
// over-confidence and the forgetting that cost leg 5 two 40-s waits.
//
//   (a) units: one frame's contaminated evidence is bounded by log(1/ε); one viewpoint's by log((1−ε_v)/ε_v); g is odd.
//   (b) LONG RUN, static binary: 6000 frames, each +15 nats for the wrong hinge (try102 leg 1 had single frames of 13-17),
//       from 3 viewpoints → the hinge log-odds stays ≤ 3·log((1−ε_v)/ε_v) (the old sum: 90 000); a parked robot (one
//       viewpoint, 3000 frames) stays ≤ one viewpoint's bound; the robot then moves through NEW viewpoints seeing the TRUE
//       hinge and the belief flips within a bounded number of informative frames. RED reference: the old frame sum.
//   (c) REPLAY of try102 leg 5 from the logged frames (tests/data/try102_phi_frames.csv: every frame's (hinge × swing)
//       marginal φ likelihood, the sensor origin, the room): the OLD predict (2 rad/s diffusion, phi_prior_sigma +
//       phi_posterior_step) reproduces the live failure — parked 5.5 m out, φ decays to the uniform's 60° and P(passable)
//       to its 0.35; the channel holds the open door: P(passable) ≥ the offer level 0.6 on every frame of the window.
//   (d) φ is HELD, not frozen: a door that really shuts in front of a parked robot is followed within a bounded number of
//       frames; one seen shut from a NEW viewpoint after an absence likewise.
//   (e) cross_offer_prob: from leg 5's 5.5 m `cross` is not offered (P(at the door) ≈ 0); from the approach standpoint it is.
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tests/test_door_leaf_evidence.cpp src/door_belief.cpp -o /tmp/test_door_leaf_evidence && /tmp/test_door_leaf_evidence
// The replay reads tests/data/try102_phi_frames.csv relative to the working directory (door_concept/); LEAF_TRACE=1 prints it.
#include <algorithm>
#include <charconv>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <format>
#include <fstream>
#include <limits>
#include <numbers>
#include <print>
#include <string>
#include <string_view>
#include <vector>

#include "../src/door_belief.h"
#include "../src/door_leaf_channel.h"
#include "../src/door_passability.h"
#include "../src/door_phi_posterior.h"

using V2 = Eigen::Vector2f;
static int g_fail = 0;
static void check(bool ok, const std::string& what)
{
    std::print("  [{}] {}\n", ok ? "PASS" : "FAIL", what);
    if (not ok) ++g_fail;
}

constexpr float PHI_MIN = 0.0f, PHI_MAX = 2.0944f, PHI_STEP = 0.0873f, NEEDED = 0.70f, OFFER = 0.60f, WITHDRAW = 0.35f;
constexpr int   NSTEP = 25;
static float grid(int k) { return PHI_MIN + (PHI_MAX - PHI_MIN) * float(k) / float(NSTEP - 1); }

static rc::DoorBelief belief_at(V2 centre, float yaw, float w)
{
    rc::DoorBeliefParams p;
    p.thickness = 0.05f; p.floor_z = 0.0f;
    p.wall_u = V2(std::cos(yaw), std::sin(yaw));
    p.wall_O = centre - 0.5f * w * p.wall_u;   // s = 0 ⇒ the centre is wall_O + w/2·u
    p.leaf = {1.5708f, rc::door::HingeSide::Near, +1.0f};
    return rc::DoorBelief({0.0f, w, 2.1f}, p);
}

// A frame whose four (hinge × swing) curves favour hypothesis (h, s) by `nats` (flat in φ — this is about the binaries).
static rc::door::LeafChannelInput binary_frame(int h_true, int s_true, float nats, V2 sensor, float dt)
{
    rc::door::LeafChannelInput in;
    auto curve = [&](int h, int s)
    {
        std::vector<std::pair<float, float>> c;
        for (int k = 0; k < NSTEP; ++k) c.emplace_back(grid(k), (h == h_true and s == s_true) ? 0.0f : -nats);
        return c;
    };
    in.bb = curve(0, 0); in.ob = curve(1, 0); in.bs = curve(0, 1); in.os = curve(1, 1);
    in.have_sensor = true; in.sensor = sensor; in.dt = dt;
    in.phi_min = PHI_MIN; in.phi_max = PHI_MAX; in.nstep = NSTEP; in.view_rad = PHI_STEP;
    return in;
}

int main()
{
    std::setlocale(LC_ALL, "");   // CLAUDE.md: the agent's locale — the replay parses with std::from_chars
    using namespace rc::door;
    const double eps = kLeafFrameOutlierRate, eps_v = kLeafViewOutlierRate;
    const double frame_bound = std::log(1.0 / eps), view_bound = std::log((1.0 - eps_v) / eps_v);

    // ── (a) units ───────────────────────────────────────────────────────────────────────────────────────────────────
    std::print("(a) units: ε {:.2f} ⇒ one frame ≤ {:.2f} nats, one viewpoint ≤ {:.2f} nats\n", eps, frame_bound, view_bound);
    check(std::abs(contaminated_llr(1e6, eps) - frame_bound) < 1e-6 and std::abs(contaminated_llr(-1e6, eps) + frame_bound) < 1e-6,
          "a frame's contaminated evidence saturates at ±log(1/ε)");
    check(std::abs(contaminated_llr(0.01, eps) - 0.01 * (1.0 - eps)) < 1e-3, "…and is (1−ε)·δ for a weak frame");
    check(std::abs(viewpoint_llr(1e4, eps_v) - view_bound) < 1e-6 and std::abs(viewpoint_llr(-3.7, eps_v) + viewpoint_llr(3.7, eps_v)) < 1e-12,
          "a viewpoint saturates at ±log((1−ε_v)/ε_v) and g is odd");

    // ── (b) long run, static binary ─────────────────────────────────────────────────────────────────────────────────
    std::print("\n(b) long run: 6000 frames of +15 nats for the WRONG hinge from 3 viewpoints, then the truth from new ones\n");
    {
        auto b = belief_at(V2(0, 0), 0.0f, 1.0f);   // room frame: the wall along x, the room y > 0
        LeafChannel ch;
        float hl = 0.f, sl = 0.f, pe = 1.5708f, ps = 0.05f;
        const V2 parked[3] = {V2(0.0f, 2.5f), V2(0.9f, 2.0f), V2(-0.8f, 1.8f)};
        double old_sum = 0.0;
        float worst = 0.f;
        // The WRONG hinge is Far (believed Near, curves for "other hinge" favoured), the swing believed is right.
        for (int f = 0; f < 6000; ++f)
        {
            const V2 at = parked[f / 2000] + V2(0.002f * std::sin(0.7f * f), 0.002f * std::cos(1.3f * f));   // localiser jitter
            const bool believed_far = b.params().leaf.hinge == HingeSide::Far;
            auto in = binary_frame(believed_far ? 0 : 1, 0, 15.f, at, 0.1f);
            old_sum += 15.0;
            leaf_channel_step(ch, b, hl, sl, pe, ps, in);
            worst = std::max(worst, std::abs(hl));
        }
        std::print("  after 6000 frames: hinge {} at log-odds {:+.2f} ({} viewpoints); the old frame sum would read {:+.0f}\n",
                   b.params().leaf.hinge == HingeSide::Far ? "Far (wrong)" : "Near", hl, ch.views.views.size(), old_sum);
        check(worst <= 3.0f * float(view_bound) + 1e-3f,
              std::format("bounded by 3 viewpoints: max |log-odds| {:.2f} ≤ {:.2f}", worst, 3.0 * view_bound));
        check(old_sum > 1000.0, "RED reference: the old per-frame sum is in the thousands (try102: +7049 / +10639)");
        // Parked: one viewpoint, 3000 more frames.
        LeafChannel ch1;
        auto b1 = belief_at(V2(0, 0), 0.0f, 1.0f);
        float hl1 = 0.f, sl1 = 0.f, pe1 = 1.5708f, ps1 = 0.05f;
        for (int f = 0; f < 3000; ++f)
        {
            const bool believed_far = b1.params().leaf.hinge == HingeSide::Far;
            leaf_channel_step(ch1, b1, hl1, sl1, pe1, ps1, binary_frame(believed_far ? 0 : 1, 0, 15.f, V2(0.3f, 2.2f), 0.1f));
        }
        check(std::abs(hl1) <= float(view_bound) + 1e-3f,
              std::format("a PARKED robot (one viewpoint, 3000 frames) adds one viewpoint at most: |{:.2f}| ≤ {:.2f}", hl1, view_bound));
        // Now the truth (Near) from new viewpoints: the robot drives an arc in front of the door, 10 frames per viewpoint.
        int flip_frame = -1;
        for (int f = 0; f < 400 and flip_frame < 0; ++f)
        {
            const float a = 0.35f + 0.012f * float(f);   // the bearing walks 0.7° per frame ⇒ a new viewpoint every ~7 frames
            const V2 at(2.2f * std::cos(a), 2.2f * std::sin(a));
            const bool believed_far = b.params().leaf.hinge == HingeSide::Far;
            leaf_channel_step(ch, b, hl, sl, pe, ps, binary_frame(believed_far ? 1 : 0, 0, 3.f, at, 0.1f));   // a modest 3 nats/frame
            if (b.params().leaf.hinge == HingeSide::Near and hl > 0.f) flip_frame = f;
        }
        std::print("  the true hinge from new viewpoints (3 nats/frame): flipped back after {} frames, {} viewpoints in all\n",
                   flip_frame, ch.views.views.size());
        check(flip_frame >= 0 and flip_frame < 60, std::format("a wrong hinge is overturned within 60 informative frames ({})", flip_frame));
    }

    // ── (c) replay try102 leg 5 ────────────────────────────────────────────────────────────────────────────────────
    std::print("\n(c) replay try102 leg 5 (tests/data/try102_phi_frames.csv)\n");
    {
        struct Row { std::int64_t t; int room; float ox, oy; int cmd; float cmd_mean, cmd_sigma, post_mean; std::vector<float> ll; };
        std::vector<Row> rows;
        std::ifstream f("tests/data/try102_phi_frames.csv");
        std::string line;
        auto num = [](std::string_view s, auto& v) { return std::from_chars(s.data(), s.data() + s.size(), v).ec == std::errc{}; };
        while (std::getline(f, line))
        {
            if (line.empty() or line[0] == '#' or line.starts_with("t_ms")) continue;
            std::vector<std::string_view> c;
            std::string_view sv(line);
            for (std::size_t p = 0;;) { const auto q = sv.find(',', p); c.push_back(sv.substr(p, q - p)); if (q == std::string_view::npos) break; p = q + 1; }
            if (c.size() < 9) continue;
            Row r;
            bool ok = num(c[0], r.t) and num(c[1], r.room) and num(c[2], r.ox) and num(c[3], r.oy) and num(c[4], r.cmd)
                  and num(c[5], r.cmd_mean) and num(c[6], r.cmd_sigma) and num(c[7], r.post_mean);
            for (std::size_t p = 0; ok and not c[8].empty();)
            {
                const auto q = c[8].find(';', p);
                const auto tok = c[8].substr(p, q - p);
                float v;
                if (tok == "-inf") v = -std::numeric_limits<float>::infinity();
                else if (not num(tok, v)) { ok = false; break; }
                r.ll.push_back(v);
                if (q == std::string_view::npos) break;
                p = q + 1;
            }
            if (ok) rows.push_back(std::move(r));
        }
        check(rows.size() > 1500, std::format("{} frames read", rows.size()));
        constexpr std::int64_t T0 = 1792421552068;   // the run's first phi frame (ai2_log_phi.csv row 1)
        // The two rooms' door_1 apertures as the agent held them (etc/ai2_log.csv: cx, cy, yaw, w).
        auto room_belief = [](int room) { return room == 1 ? belief_at(V2(-0.47f, 4.61f), 3.128f, 1.0f) : belief_at(V2(0.02f, 0.18f), 0.007f, 1.01f); };
        // OLD: the 2 rad/s diffusion + phi_posterior_step. NEW: the channel.
        float o_mean = rows.front().post_mean, o_sigma = 0.05f;
        // The live hinge / swing were settled (and right) by leg 5; under the channel a settled binary reads a few viewpoints'
        // worth (+20 here, not the live +7049 / +10639). The logged curve is already the leaf mixture: it goes in as the
        // believed leaf's.
        float n_mean = o_mean, n_sigma = 0.05f, hl = 20.f, sl = 20.f;
        LeafChannel ch;
        auto nb = room_belief(rows.front().room);
        int room = rows.front().room;
        float o_pmin = 2.f, n_pmin = 2.f, o_sum_low = 0.f, n_sum_low = 0.f, o_phi_end = 0.f, n_phi_end = 0.f, worst_dev = 0.f;
        float o_sum_wd = 0.f, n_sum_wd = 0.f, win_s = 0.f, log_end = 0.f;
        std::vector<float> devs;
        std::int64_t prev_t = rows.front().t;
        for (const auto& r : rows)
        {
            const float t = 1e-3f * float(r.t - T0), dt = r.t == prev_t ? 0.1f : 1e-3f * float(r.t - prev_t);
            prev_t = r.t;
            if (r.room != room) { room = r.room; nb = room_belief(room); ch = LeafChannel{}; }   // the room switch: a new instance, φ carried
            std::vector<std::pair<float, float>> ll;
            if (r.ll.size() == std::size_t(NSTEP)) for (int k = 0; k < NSTEP; ++k) ll.emplace_back(grid(k), r.ll[std::size_t(k)]);
            // OLD
            const float sp = phi_prior_sigma(o_sigma, 2.0f, dt, PHI_STEP, PHI_MAX);
            const auto op = phi_posterior_step(ll, PHI_MIN, PHI_MAX, NSTEP, o_mean, sp);
            o_mean = op.mean; o_sigma = op.sigma;
            if (std::abs(o_mean - r.post_mean) > worst_dev and std::getenv("LEAF_TRACE"))
                std::print("    old-replay deviation {:.3f} rad at t {:.1f} (logged {:.3f}, replay {:.3f})\n", std::abs(o_mean - r.post_mean), t, r.post_mean, o_mean);
            worst_dev = std::max(worst_dev, std::abs(o_mean - r.post_mean));
            devs.push_back(std::abs(o_mean - r.post_mean));
            // NEW
            LeafChannelInput in;
            in.bb = ll; in.have_sensor = true; in.sensor = V2(r.ox, r.oy); in.dt = dt;
            in.phi_min = PHI_MIN; in.phi_max = PHI_MAX; in.nstep = NSTEP; in.view_rad = PHI_STEP;
            const auto np = leaf_channel_step(ch, nb, hl, sl, n_mean, n_sigma, in).post;
            const float po = passable_prob(op.curve, 1.0f, NEEDED).value_or(0.f), pn = passable_prob(np.curve, 1.0f, NEEDED).value_or(0.f);
            if (t >= 407.f and t <= 494.f)   // leg 5, attempts 1-2: parked 5.5 m out beside chair_2, the door open at 90°
            {
                o_pmin = std::min(o_pmin, po); n_pmin = std::min(n_pmin, pn);
                o_sum_low += po < OFFER ? dt : 0.f; n_sum_low += pn < OFFER ? dt : 0.f;
                o_sum_wd += po < WITHDRAW ? dt : 0.f; n_sum_wd += pn < WITHDRAW ? dt : 0.f;
                win_s += dt;
                o_phi_end = o_mean; n_phi_end = n_mean; log_end = r.post_mean;
            }
            if (std::getenv("LEAF_TRACE") and (int(t * 10) % 10 == 0))
                std::print("    t {:6.1f} room {} o ({:+.2f},{:+.2f}) | old {:5.1f}° P {:.2f} | new {:5.1f}° ± {:4.1f} P {:.2f} views {} hazard 1/{:.0f} s\n",
                           t, room, r.ox, r.oy, o_mean * 57.2958f, po, n_mean * 57.2958f, n_sigma * 57.2958f, pn,
                           ch.views.views.size(), 1.0 / ch.hazard());
        }
        std::ranges::nth_element(devs, devs.begin() + long(devs.size() / 2));
        const float med_dev = devs[devs.size() / 2];
        std::print("  the parked window (t 407-494 s, {:.0f} s): the OLD predict (as live) / the channel\n", win_s);
        std::print("    OLD: min P(passable) {:.2f}; {:.1f} s below the offer level {:.2f}, {:.1f} s below the withdrawal level {:.2f}; φ at the end {:.1f}° (logged {:.1f}°)\n",
                   o_pmin, o_sum_low, OFFER, o_sum_wd, WITHDRAW, o_phi_end * 57.2958f, log_end * 57.2958f);
        std::print("    NEW: min P(passable) {:.2f}; {:.1f} s below the offer level, {:.1f} s below the withdrawal level; φ at the end {:.1f}°\n",
                   n_pmin, n_sum_low, n_sum_wd, n_phi_end * 57.2958f);
        std::print("    (the old replay vs the logged posterior: median |Δφ| {:.4f} rad, worst {:.3f} rad at a transient)\n", med_dev, worst_dev);
        check(med_dev < 0.005f and std::abs(o_phi_end - log_end) < 0.01f, "the OLD replay reproduces the logged posterior (median and the window's end)");
        check(o_sum_wd > 40.f and std::abs(o_phi_end * 57.2958f - 60.f) < 6.f,
              "RED reference: the old predict decays to the uniform — φ 60°, P(passable) ≈ 0.35, below the withdrawal level for > 40 s");
        check(n_sum_wd < 1.0f, std::format("the channel is below the withdrawal level for < 1 s ({:.1f} s; old: > 40 s)", n_sum_wd));
        check(n_sum_low < 0.1f * win_s, std::format("…and is below the offer level for < 10 % of the window ({:.1f} of {:.0f} s; the 4-frame "
                                                   "burst of 20-ray frames at t 455 s that placed a leaf at 20-30°)", n_sum_low, win_s));
        check(std::abs(n_phi_end * 57.2958f - 90.f) < 15.f, std::format("…at φ {:.1f}° (90 ± 15)", n_phi_end * 57.2958f));
    }

    // ── (d) held, not frozen ───────────────────────────────────────────────────────────────────────────────────────
    std::print("\n(d) a door that really shuts\n");
    {
        auto peaked = [](float at, float nats)   // a frame whose likelihood peaks at `at`, `nats` below elsewhere (σ 5°)
        {
            std::vector<std::pair<float, float>> c;
            for (int k = 0; k < NSTEP; ++k) { const float d = (grid(k) - at) / 0.0873f; c.emplace_back(grid(k), std::max(-nats, -0.5f * d * d)); }
            return c;
        };
        auto run = [&](const V2& at_open, const V2& at_shut, float absent_s, const char* what)
        {
            auto b = belief_at(V2(0, 0), 0.0f, 1.0f);
            LeafChannel ch;
            float hl = 20.f, sl = 20.f, pe = 0.f, ps = 2.0944f;   // a settled leaf; the frames score the believed one
            LeafChannelInput in; in.have_sensor = true; in.phi_min = PHI_MIN; in.phi_max = PHI_MAX; in.nstep = NSTEP; in.view_rad = PHI_STEP;
            for (int f = 0; f < 100; ++f)   // 10 s seeing it open
            {
                in.bb = peaked(1.5708f, 20.f); in.sensor = at_open; in.dt = f == 0 ? std::numeric_limits<float>::infinity() : 0.1f;
                leaf_channel_step(ch, b, hl, sl, pe, ps, in);
            }
            in.dt = absent_s > 0.f ? absent_s : 0.1f;
            int k_follow = -1;
            for (int f = 0; f < 100 and k_follow < 0; ++f)   // then it is shut
            {
                in.bb = peaked(0.0f, 20.f); in.sensor = at_shut;
                const auto r = leaf_channel_step(ch, b, hl, sl, pe, ps, in);
                in.dt = 0.1f;
                if (passable_prob(r.post.curve, 1.0f, NEEDED).value_or(1.f) < 0.35f) k_follow = f + 1;
            }
            std::print("  {}: P(passable) below the withdrawal level after {} frames (φ {:.1f}°)\n", what, k_follow, pe * 57.2958f);
            return k_follow;
        };
        const int k_parked = run(V2(0.2f, 2.0f), V2(0.2f, 2.0f), 0.f, "shut in front of a parked robot");
        check(k_parked > 0 and k_parked <= 10, std::format("followed within 10 frames ({})", k_parked));
        const int k_new = run(V2(0.2f, 2.0f), V2(-0.9f, 1.6f), 60.f, "seen shut from a new viewpoint after 60 s away");
        check(k_new > 0 and k_new <= 10, std::format("followed within 10 frames ({})", k_new));
    }

    // ── (e) cross is offered at the door, not from across the room ─────────────────────────────────────────────────
    std::print("\n(e) cross_offer_prob (P(passable) 1.00; reach 2.00 m, σ 0.10 m)\n");
    {
        const float far = cross_offer_prob(1.0f, prob_within(5.50f, 2.0f, 0.10f));
        const float band = cross_offer_prob(1.0f, prob_within(1.75f, 2.0f, 0.10f));
        const float shut = cross_offer_prob(0.0f, prob_within(1.75f, 2.0f, 0.10f));
        std::print("  5.50 m (leg 5): {:.3f}; 1.75 m (the approach standpoint): {:.3f}; shut at 1.75 m: {:.3f}\n", far, band, shut);
        check(far < 0.35f and band > OFFER and shut < 0.35f, "offered at the door only, and never for a shut door");
    }

    std::print("\n{}\n", g_fail == 0 ? "ALL PASS" : std::format("{} FAILURE(S)", g_fail));
    return g_fail == 0 ? 0 : 1;
}
