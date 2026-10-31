// Sampler of the generative block (imagined observations, Task 3).   LANG=es_ES.UTF-8 ./generative_test
// A chair-equivalent Generative is built by hand (chair_card.h make_generative, same numbers as chair.concept.toml).
#include "perception_card/generative.h"
#include <chrono>
#include <clocale>
#include <cstdio>
#include <numbers>
#include <random>

using namespace rc::card;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++bad; } } while (0)

static Generative chair_gen()
{
    const auto K = [](float b) { return Bind{"", 0.0f, b}; };
    const auto V = [](const char* v, float a = 1.0f, float b = 0.0f) { return Bind{v, a, b}; };
    const float T = 0.075f;
    Generative g;
    g.shape.push_back({Prim::Slab, {{"w", V("seat_w")}, {"d", V("seat_d")}, {"h", K(T)}, {"x", K(0)}, {"y", K(0)}, {"z", V("seat_h", 1, -T)}}});
    g.shape.push_back({Prim::Panel, {{"w", V("seat_w")}, {"t", K(T)}, {"h", V("back_h")}, {"x", K(0)}, {"y", V("seat_d", -0.5f, 0.5f * T)}, {"z", V("seat_h")}}});
    g.shape.push_back({Prim::LegSet, {{"n", K(4)}, {"half", K(0.02f)}, {"fw", V("seat_w")}, {"fd", V("seat_d")}, {"z0", K(0)}, {"z1", V("seat_h", 1, -T)}}});
    g.vars = {{"seat_w", 0.60f, 0.15f, 0.30f, 0.65f, false}, {"seat_d", 0.52f, 0.15f, 0.30f, 0.65f, false},
              {"seat_h", 0.595f, 0.15f, 0.38f, 0.70f, false}, {"back_h", 0.655f, 0.15f, 0.30f, 0.80f, false}};
    g.placement.support = "floor";
    g.placement.contexts = {{"free", 0.5f}, {"against_wall", 0.5f}};
    g.placement.wall_gap_mean = 0.05f; g.placement.wall_gap_sigma = 0.05f;
    return g;
}
// Mean of N(mean, sigma) truncated to [lo, hi], by midpoint integration (the reference the sampler is checked against).
static std::pair<double, double> truncated_normal_moments(double mean, double sigma, double lo, double hi)
{
    double num = 0, den = 0, sq = 0;
    const int N = 200000;
    for (int i = 0; i < N; ++i)
    {
        const double x = lo + (hi - lo) * (i + 0.5) / N, p = std::exp(-0.5 * (x - mean) * (x - mean) / (sigma * sigma));
        num += x * p; sq += x * x * p; den += p;
    }
    const double m = num / den;
    return {m, std::sqrt(sq / den - m * m)};
}
static Card chair_card() { Card c; c.concept_ = "chair"; c.walls = WallOcclusion{0.15f, -1.0f, 0.0f}; c.generative = chair_gen(); return c; }

static std::vector<Eigen::Vector2f> room(float w, float h)
{
    return {{0, 0}, {w, 0}, {w, h}, {0, h}};
}
static bool same(const SampleSet& a, const SampleSet& b)
{
    if (a.samples.size() != b.samples.size() or a.drawn != b.drawn or a.rejected != b.rejected) return false;
    for (std::size_t i = 0; i < a.samples.size(); ++i)
        if (a.samples[i].x != b.samples[i].x or a.samples[i].y != b.samples[i].y or a.samples[i].yaw != b.samples[i].yaw
            or a.samples[i].state != b.samples[i].state) return false;
    return true;
}

int main()
{
    std::setlocale(LC_ALL, "");
    const Card card = chair_card();
    World w; w.polygon = room(6, 4);
    w.obstacles.push_back({"box", rc::nbv::Obstacle{3.0f, 2.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f}});

    // 1. determinism
    { const auto a = sample_concept(card, w, {}, 50, 7), b = sample_concept(card, w, {}, 50, 7), c = sample_concept(card, w, {}, 50, 8);
      CHECK(same(a, b)); CHECK(not same(a, c)); CHECK(a.seed == 7); }

    // 2. no part overlaps a wall or the obstacle; acceptance numbers
    {
        const auto s = sample_concept(card, w, {}, 200, 1);
        CHECK(s.samples.size() == 200);
        const auto walls = rc::nbv::wall_obstacles(w.polygon, 1e-3f);
        bool clean = true;
        for (const auto& sm : s.samples)
        {
            CHECK(not sm.parts.boxes.empty());
            for (const auto& b : sm.parts.boxes)
            {
                auto bb = b;
                for (const auto& o : walls) { auto ow = o; ow.z0 = 0; ow.z1 = 1e9f; if (boxes_overlap(bb, ow)) clean = false; }
                auto o = w.obstacles[0].second; o.z1 = 1e9f;
                if (boxes_overlap(bb, o)) clean = false;
            }
        }
        CHECK(clean);
        std::printf("acceptance 6x4 + 1x1 box: drawn %d rejected %d accepted %zu (%.1f%%)\n", s.drawn, s.rejected,
                    s.samples.size(), 100.0 * s.samples.size() / s.drawn);
    }

    // 3. size moments and bounds
    {
        const auto s = sample_concept(card, w, {}, 2000, 3, 100000);
        CHECK(s.samples.size() == 2000);
        std::map<std::string, double> sum, sumsq;
        bool inb = true;
        for (const auto& sm : s.samples)
            for (const auto& v : card.generative->vars)
            {
                const float x = sm.state.at(v.var);
                sum[v.var] += x; sumsq[v.var] += double(x) * x;
                if (x < v.lo or x > v.hi) inb = false;
            }
        CHECK(inb);
        // Truncation to [lo,hi] moves the mean (seat_w 0.60 -> ~0.52) and shrinks the SD of a wide prior (0.15 against
        // bounds ~1-2 sigma away), so both are checked against the analytic TRUNCATED-normal moments, never the prior's.
        for (const auto& v : card.generative->vars)
        {
            const double m = sum[v.var] / 2000.0, sd = std::sqrt(sumsq[v.var] / 2000.0 - m * m);
            const auto [rm, rsd] = truncated_normal_moments(v.mean, v.sigma, v.lo, v.hi);
            std::printf("  %s mean %.4f (ref %.4f, prior %.4f)  sd %.4f (ref %.4f)\n", v.var.c_str(), m, rm, v.mean, sd, rsd);
            CHECK(std::abs(m - rm) < 0.01);
            CHECK(std::abs(sd - rsd) < 0.01);
        }
    }

    // 4. against_wall: backrest within gap+3 sigma of a wall and facing it
    {
        Context ctx; ctx.force_context = "against_wall";
        const auto s = sample_concept(card, w, ctx, 200, 5);
        CHECK(s.samples.size() == 200);
        const auto& P = w.polygon;
        for (const auto& sm : s.samples)
        {
            const Eigen::Vector2f back_dir(std::sin(sm.yaw), -std::cos(sm.yaw));            // local -y
            const Eigen::Vector2f face = Eigen::Vector2f(sm.x, sm.y) + back_dir * (0.5f * sm.state.at("seat_d"));
            float best = 1e9f; float dot = -1;
            for (std::size_t i = 0, j = P.size() - 1; i < P.size(); j = i++)
            {
                const Eigen::Vector2f d = (P[i] - P[j]).normalized(), nrm_out(d.y(), -d.x());   // CCW polygon: right normal = outward
                const float dist = std::abs((face - P[j]).dot(nrm_out));
                if (dist < best) { best = dist; dot = back_dir.dot(nrm_out); }
            }
            CHECK(best <= 0.05f + 3 * 0.05f);
            CHECK(dot > 0.9f);
        }
    }

    // 5. Review focus 1: a 2x2 room with 4 obstacles terminates short, rejected > 0, with a note
    {
        World t; t.polygon = room(2, 2);
        for (const auto& [x, y] : std::vector<std::pair<float, float>>{{0.5f, 0.5f}, {1.5f, 0.5f}, {0.5f, 1.5f}, {1.5f, 1.5f}})
            t.obstacles.push_back({"o", rc::nbv::Obstacle{x, y, 0.9f, 0.9f, 0.0f, 0, 0}});
        const auto s = sample_concept(card, t, {}, 100, 2);
        CHECK(s.samples.size() < 100); CHECK(s.rejected > 0); CHECK(not s.notes.empty()); CHECK(s.drawn <= 50 * 100);
        std::printf("acceptance 2x2 + 4 obstacles: drawn %d rejected %d accepted %zu notes %zu\n", s.drawn, s.rejected, s.samples.size(), s.notes.size());
    }

    // 6. Review focus 2: no polygon -> free samples + note; against_wall skipped + note
    {
        World e;
        const auto s = sample_concept(card, e, {}, 20, 4);
        CHECK(s.samples.size() == 20); CHECK(not s.notes.empty());
        bool in6 = true;
        for (const auto& sm : s.samples) in6 = in6 and std::abs(sm.x) <= 3.0f and std::abs(sm.y) <= 3.0f;
        CHECK(in6);
        Context ctx; ctx.force_context = "against_wall";
        const auto a = sample_concept(card, e, ctx, 20, 4);
        CHECK(a.samples.empty()); CHECK(not a.notes.empty());
    }

    // 7. at: position fixed
    {
        Context ctx; ctx.at = Eigen::Vector2f(1.5f, 1.0f);
        const auto s = sample_concept(card, w, ctx, 10, 9);
        CHECK(not s.samples.empty());
        for (const auto& sm : s.samples) CHECK(sm.x == 1.5f and sm.y == 1.0f);
    }

    // 8. instantiate: chair at identity, yaw 0: the backrest is on the -y edge, top at seat_h + back_h
    {
        const std::map<std::string, float> st{{"seat_w", 0.6f}, {"seat_d", 0.52f}, {"seat_h", 0.595f}, {"back_h", 0.655f}};
        const auto p = instantiate(*card.generative, st, 1.0f, 2.0f, 0.0f);
        CHECK(p.boxes.size() == 1 + 1 + 4);
        float top = 0, miny = 1e9f;
        for (const auto& b : p.boxes) { top = std::max(top, b.z1); miny = std::min(miny, b.cy - 0.5f * b.h); }
        CHECK(std::abs(top - 1.25f) < 1e-4f);
        CHECK(std::abs(miny - (2.0f - 0.26f)) < 1e-4f);
    }
    // 9. truncated-normal tails, degenerate and inverted intervals
    {
        std::mt19937_64 rng(11);
        const auto stats = [&](double lo, double hi, int N = 5000)
        {
            double m = 0; bool in = true;
            for (int i = 0; i < N; ++i) { const double x = detail::trunc_normal(0, 1, lo, hi, rng); m += x; in = in and x >= lo and x <= hi and std::isfinite(x); }
            return std::pair{m / N, in};
        };
        const auto [m1, in1] = stats(-30, -20);   CHECK(in1); CHECK(std::abs(m1 - (-20.0488)) < 0.01);
        const auto [m2, in2] = stats(8, 12);      CHECK(in2); CHECK(std::abs(m2 - 8.1175) < 0.01);
        const auto [m3, in3] = stats(1.5, 1.5);   CHECK(in3); CHECK(m3 == 1.5);
        CHECK(detail::trunc_normal(0, 1, 2.0, 1.0, rng) == 2.0);   // lo > hi: returns lo, no clamp UB
        Card bc = chair_card(); bc.generative->vars[0].lo = 0.7f; bc.generative->vars[0].hi = 0.3f;
        const auto bs = sample_concept(bc, w, {}, 5, 1);
        CHECK(not bs.notes.empty()); CHECK(bs.samples.size() == 5 and bs.samples[0].state.at("seat_w") == 0.7f);
        std::printf("tails: [-30,-20] mean %.4f  [8,12] mean %.4f\n", m1, m2);
    }

    // 10. boxes_overlap directly
    {
        using O = rc::nbv::Obstacle;
        const float q = std::numbers::pi_v<float> / 4;
        const O a{0, 0, 2.0f, 0.2f, q, 0, 0}, b{0.42f, -0.42f, 2.0f, 0.2f, q, 0, 0};     // parallel, 0.6 m apart across their width
        CHECK(not boxes_overlap(a, b));                                                   // AABBs (half 0.78) overlap, boxes do not
        const O c{0.14f, -0.14f, 2.0f, 0.2f, q, 0, 0};                                    // 0.2 apart: they interpenetrate
        CHECK(boxes_overlap(a, c));
        const O u1{0, 0, 1, 1, 0, 0, 0}, u2{1.0f, 0, 1, 1, 0, 0, 0}, u3{0.99f, 0, 1, 1, 0, 0, 0};
        CHECK(not boxes_overlap(u1, u2)); CHECK(boxes_overlap(u1, u3));                   // touching vs 1 cm overlap
        const O top{0, 0, 1, 1, 0, 1.0f, 2.0f}, low{0, 0, 1, 1, 0, 0.0f, 1.0f};
        CHECK(not boxes_overlap(top, low));                                               // z spans only touch
    }

    // 11. ruling P4: against_wall with a 12 cm wall; the gap is to the wall's INNER face; doors are skipped
    {
        Card pc = chair_card(); pc.walls = WallOcclusion{0.15f, -1.0f, 0.12f};
        Context ctx; ctx.force_context = "against_wall";
        World e; e.polygon = room(6, 4);
        const auto s = sample_concept(pc, e, ctx, 1000, 21);
        CHECK(s.samples.size() == 1000);
        const double frac = double(s.rejected + s.outside_polygon) / s.drawn;
        double mg = 0, mn = 1e9;
        for (const auto& sm : s.samples)
        {
            const Eigen::Vector2f bd(std::sin(sm.yaw), -std::cos(sm.yaw));
            const Eigen::Vector2f face = Eigen::Vector2f(sm.x, sm.y) + bd * (0.5f * sm.state.at("seat_d"));
            const float dl = std::min({face.x(), 6 - face.x(), face.y(), 4 - face.y()});   // to the nearest edge line
            const float gap = dl - 0.06f; mg += gap; mn = std::min<double>(mn, gap);
        }
        mg /= s.samples.size();
        const auto [gm, gsd] = truncated_normal_moments(0.05, 0.05, 0.0, 1.0);   // upper bound 20 sigma: effectively [0, inf)
        std::printf("P4 against_wall (12 cm walls): rejection %.2f%% (drawn %d)  gap mean %.4f ref %.4f  min gap %.4f\n",
                    100 * frac, s.drawn, mg, gm, mn);
        CHECK(frac < 0.05); CHECK(std::abs(mg - gm) < 0.01); CHECK(mn > -1e-4);

        World d = e; d.gaps.push_back({Eigen::Vector2f(3.0f, 0.0f), 0.5f});              // a doorway in the y = 0 wall
        const auto sd = sample_concept(pc, d, ctx, 1000, 22);
        CHECK(sd.samples.size() == 1000);
        int near_door_wall = 0; bool clear = true;
        for (const auto& sm : sd.samples)
            if (sm.y < 0.8f and sm.x > 1.0f and sm.x < 5.0f)   // against the doorway's wall
            {
                ++near_door_wall;
                for (const auto& b : sm.parts.boxes) if (b.cx > 2.5f and b.cx < 3.5f) clear = false;
            }
        CHECK(near_door_wall > 50); CHECK(clear);
        std::printf("P4 doorway: %d of 1000 chairs on the doorway's wall, none over the gap: %s\n", near_door_wall, clear ? "yes" : "NO");
    }
    // 11. evaluate_mc: the evaluator averaged over the size prior
    {
        Card mc = chair_card(); mc.graph_type = "object"; mc.level = Level::Countable;
        mc.channels = {{"zed", Role::Identity, true}};
        mc.geometry = {0.6f, 0.6f, 1.2f, 0.0f, 1.2f};
        SubtypeParams sp; sp.env = {0.0f, 1.32f, 0.26f}; mc.subtypes[""] = sp;
        World w; w.polygon = room(6, 4);
        rc::nbv::Sensor s; constexpr float f = 448.0f;
        s.hfov_rad = 2.0f * std::atan(640.0f / f); s.vfov_rad = 2.0f * std::atan(360.0f / f); s.height_m = 0.945f;
        w.zed = s; w.zed_mount_z = 0.945f;
        Hypothesis base; base.concept_ = "chair"; base.x = 3.0f; base.y = 2.0f; base.yaw = 0.0f;
        const Viewpoint vp{{3.0f - 1.3f, 2.0f}, std::nullopt};

        { // n = 1, all sigmas 0: equals evaluate() at the bbox-CENTRED hypothesis. The reference is built by hand from the
          // chair's primitives (seat 0.60 x 0.52 at the origin, backrest inside it, legs at the corners): the bbox is
          // symmetric, so the centre is the origin and the size is the seat plus the tallest part (seat_h + back_h).
            Card z = mc; for (auto& v : z.generative->vars) v.sigma = 0.0f;
            Hypothesis h = base; h.size = GeometryPrior{0.60f, 0.52f, 0.595f + 0.655f, 0.0f, 0.595f + 0.655f};
            const auto r = evaluate(z, w, h, vp);
            const auto m = evaluate_mc(z, w, base, vp, 1, 5);
            CHECK(m.n == 1 and std::abs(m.p_notice_mean - r.p_notice) < 1e-6f and std::abs(m.p_recognise_mean - r.p_recognise) < 1e-6f);
            CHECK(m.p_notice_sd == 0.0f and m.p_recognise_sd == 0.0f and m.modal_limit == r.limit);
            const auto m2 = evaluate_mc(z, w, base, vp, 32, 5);
            CHECK(m2.p_recognise_sd == 0.0f and m2.p_recognise_mean == r.p_recognise);
            Card ng = mc; ng.generative.reset();
            const auto m3 = evaluate_mc(ng, w, base, vp, 32, 5); const auto r3 = evaluate(ng, w, base, vp);
            CHECK(m3.n == 1 and m3.p_recognise_mean == r3.p_recognise and m3.p_recognise_sd == 0.0f);
        }
        { // asymmetric template: one box 0.4 x 0.2 x 1.0 whose centre is 0.5 m along local +y of the origin; yaw 0.5
            Card az = mc; az.generative = Generative{};
            const auto K = [](float v) { return Bind{"", 0.0f, v}; };
            az.generative->shape.push_back({Prim::Box, {{"w", K(0.4f)}, {"d", K(0.2f)}, {"h", K(1.0f)}, {"x", K(0.0f)}, {"y", K(0.5f)}, {"z", K(0.0f)}}});
            az.generative->vars = {{"dummy", 1.0f, 0.0f, 0.0f, 2.0f, false}};
            az.generative->placement = chair_gen().placement;
            Hypothesis b2 = base; b2.x = 3.0f; b2.y = 3.2f; b2.yaw = 0.5f;               // near the y = 4 wall
            const float cx = b2.x - 0.5f * std::sin(0.5f), cy = b2.y + 0.5f * std::cos(0.5f);   // centre by hand: R(yaw) * (0, 0.5)
            const Viewpoint v2{{1.7f, 3.7f}, std::nullopt};
            Hypothesis hc = b2; hc.x = cx; hc.y = cy; hc.size = GeometryPrior{0.4f, 0.2f, 1.0f, 0.0f, 1.0f};
            Hypothesis hu = hc; hu.x = b2.x; hu.y = b2.y;                                   // the uncorrected (origin) centre
            const auto rc_ = evaluate(az, w, hc, v2), ru = evaluate(az, w, hu, v2);
            const auto m = evaluate_mc(az, w, b2, v2, 1, 5);
            const auto t = detail::target_of(instantiate(*az.generative, {{"dummy", 1.0f}}, b2.x, b2.y, b2.yaw));
            CHECK(std::abs(t.cy - b2.y) > 0.3f and std::abs(t.cx - cx) < 1e-5f and std::abs(t.cy - cy) < 1e-5f);
            CHECK(std::abs(m.p_notice_mean - rc_.p_notice) < 1e-6f and std::abs(m.p_recognise_mean - rc_.p_recognise) < 1e-6f);
            CHECK(std::abs(rc_.p_recognise - ru.p_recognise) > 1e-3f or std::abs(rc_.p_notice - ru.p_notice) > 1e-3f);   // the test can see the offset
            CHECK(std::abs(m.p_recognise_mean - ru.p_recognise) > 1e-3f or std::abs(m.p_notice_mean - ru.p_notice) > 1e-3f);
            std::printf("offset box: corrected p=%.4f uncorrected p=%.4f mc p=%.4f\n", rc_.p_recognise, ru.p_recognise, m.p_recognise_mean);
        }
        { // reproducible; a different seed differs
            const auto a = evaluate_mc(mc, w, base, vp, 32, 9), b = evaluate_mc(mc, w, base, vp, 32, 9), c = evaluate_mc(mc, w, base, vp, 32, 10);
            CHECK(a.n == 32 and a.p_recognise_mean == b.p_recognise_mean and a.p_notice_sd == b.p_notice_sd and a.modal_limit == b.modal_limit);
            CHECK(a.p_recognise_mean != c.p_recognise_mean);
        }
        { // near the envelope's upper shoulder the size prior spreads P; and 32 samples are cheap
            const auto t0 = std::chrono::steady_clock::now();
            const auto m = evaluate_mc(mc, w, base, vp, 32, 3);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("evaluate_mc n=32: notice %.3f +- %.3f  recognise %.3f +- %.3f  limit %d  %.2f ms\n",
                        m.p_notice_mean, m.p_notice_sd, m.p_recognise_mean, m.p_recognise_sd, int(m.modal_limit), ms);
            CHECK(m.p_recognise_sd > 0.0f or m.p_notice_sd > 0.0f);
            CHECK(ms < 200.0);
        }
    }
    std::printf(bad ? "generative_test: %d FAILED\n" : "generative_test: OK\n", bad);
    return bad ? 1 : 0;
}
