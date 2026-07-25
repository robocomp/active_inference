#ifndef REFRIGERATOR_BIRTH_SURPRISE_PROBE_H
#define REFRIGERATOR_BIRTH_SURPRISE_PROBE_H

/*
 * birth_surprise_probe.h — read-only, EXPERIMENTAL (off by default).
 *
 * Reinterpret residual_concept's `grid` node (under room) as a BIRTH-SURPRISE field. Its grid_occupancy_prob is
 * P(occupied ∧ ¬explained-by-any-concept) — the accumulated, concept-subtracted, multi-sensor evidence the
 * current generative model does NOT account for. A coherent, confidently-unexplained region NOT already covered
 * by a believed refrigerator is a candidate for a NEW refrigerator: a concept condensing out of the residual field (birth =
 * model expansion when instantiating a component lowers total free energy).
 *
 * This module ONLY reads + clusters + scores. It never births and never writes the graph. Purpose: watch the
 * surprise ramp on a real refrigerator's region and confirm it flags births cleanly (and rejects phantoms) BEFORE
 * letting it gate/replace the tracker's frames-counter. Pure math (no DSR) → unit-testable. The removal half is
 * NOT here: the grid subtracts explained cells, so it cannot see a modelled refrigerator's absence — removal stays the
 * per-instance existence-surprise job. See [[refrigerator-birth-surprise-probe]].
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace rc {

// The residual field as read off the `grid` node: dense row-major P (+ optional Var) and its room-frame geometry.
struct GridField
{
    std::vector<float> prob;   // P(occ ∧ ¬explained), row-major i = y*width + x
    std::vector<float> var;    // Var[P] (epistemic); may be empty
    float xmin = 0.f, ymin = 0.f, cell = 0.f;   // room-frame origin of cell (0,0) + cell size (m)
    int   width = 0, height = 0;
    bool  valid() const
    { return width > 0 and height > 0 and cell > 0.f and static_cast<int>(prob.size()) >= width * height; }
    float cell_cx(int x) const { return xmin + (static_cast<float>(x) + 0.5f) * cell; }
    float cell_cy(int y) const { return ymin + (static_cast<float>(y) + 0.5f) * cell; }
};

// A believed refrigerator's footprint (room frame) — used to mark residual regions an existing concept already explains.
struct FootprintBox { float cx, cy, w, h, yaw; };

// One clustered region of confidently-unexplained occupancy = one birth candidate.
struct BirthCandidate
{
    float cx = 0.f, cy = 0.f;          // surprise-mass centroid (room frame, m)
    float ext_x = 0.f, ext_y = 0.f;    // bounding extent (m)
    int   cells = 0;                   // # cells over the P threshold
    float mass = 0.f;                  // Σ P over the region — the accumulated SURPRISE MASS (the score)
    float mean_p = 0.f, mean_var = 0.f;
    bool  covered_by_refrigerator = false;    // centroid inside a believed refrigerator footprint (+margin) ⇒ NOT a birth
};

struct BirthProbeParams
{
    float p_threshold    = 0.5f;   // a cell counts as confidently-unexplained-occupied above this P
    float cover_margin_m = 0.30f;  // grow each refrigerator footprint by this before the "already explained" test
    int   min_cells      = 4;      // ignore specks (sensor noise); a real refrigerator footprint spans many cells
};

class BirthSurpriseProbe
{
public:
    // Cluster the field's confidently-unexplained cells (4-connectivity flood fill), score each region by surprise
    // MASS, flag whether a believed refrigerator already covers it, and return them sorted by mass (strongest first).
    // Pure function of its inputs — no DSR, no state.
    static std::vector<BirthCandidate> scan(const GridField& g,
                                            const std::vector<FootprintBox>& refrigerators,
                                            const BirthProbeParams& p = {})
    {
        std::vector<BirthCandidate> out;
        if (not g.valid()) return out;
        const int W = g.width, H = g.height, N = W * H;
        std::vector<char> seen(static_cast<std::size_t>(N), 0);
        std::vector<int>  stack;
        for (int s = 0; s < N; ++s)
        {
            if (seen[s] or g.prob[s] <= p.p_threshold) continue;
            stack.clear(); stack.push_back(s); seen[s] = 1;
            double sx = 0, sy = 0, mass = 0, sp = 0, sv = 0;
            int cells = 0;
            float minx = 1e9f, maxx = -1e9f, miny = 1e9f, maxy = -1e9f;
            while (not stack.empty())
            {
                const int i = stack.back(); stack.pop_back();
                const int x = i % W, y = i / W;
                const float px = g.cell_cx(x), py = g.cell_cy(y), pr = g.prob[i];
                sx += px * pr; sy += py * pr; mass += pr; sp += pr;
                sv += (i < static_cast<int>(g.var.size())) ? g.var[i] : 0.0f;
                ++cells;
                minx = std::min(minx, px); maxx = std::max(maxx, px);
                miny = std::min(miny, py); maxy = std::max(maxy, py);
                const int nb[4] = { (x > 0 ? i - 1 : -1), (x < W - 1 ? i + 1 : -1),
                                    (y > 0 ? i - W : -1), (y < H - 1 ? i + W : -1) };
                for (const int n : nb)
                    if (n >= 0 and not seen[n] and g.prob[n] > p.p_threshold) { seen[n] = 1; stack.push_back(n); }
            }
            if (cells < p.min_cells) continue;
            BirthCandidate c;
            c.cells    = cells;
            c.mass     = static_cast<float>(mass);
            c.cx       = static_cast<float>(sx / std::max(1e-6, mass));
            c.cy       = static_cast<float>(sy / std::max(1e-6, mass));
            c.ext_x    = maxx - minx + g.cell;
            c.ext_y    = maxy - miny + g.cell;
            c.mean_p   = static_cast<float>(sp / cells);
            c.mean_var = static_cast<float>(sv / cells);
            c.covered_by_refrigerator = covered(c.cx, c.cy, refrigerators, p.cover_margin_m);
            out.push_back(c);
        }
        std::sort(out.begin(), out.end(), [](const BirthCandidate& a, const BirthCandidate& b)
                  { return a.mass > b.mass; });
        return out;
    }

    // Detection-conditioned readout: the accumulated surprise MASS (Σ P over confidently-unexplained cells) within
    // radius_m of a room-frame point — i.e. how much unexplained occupancy sits under a YOLO detection. This is the
    // FUSION signal: a real refrigerator detection should land on high residual mass; a flicker/phantom detection on ~0.
    static float residual_mass_near(const GridField& g, float x, float y, float radius_m, float p_threshold = 0.5f)
    {
        if (not g.valid() or radius_m <= 0.f) return 0.f;
        const int cx = static_cast<int>((x - g.xmin) / g.cell);
        const int cy = static_cast<int>((y - g.ymin) / g.cell);
        const int r  = static_cast<int>(std::ceil(radius_m / g.cell));
        const float r2 = radius_m * radius_m;
        double m = 0.0;
        for (int yy = std::max(0, cy - r); yy <= std::min(g.height - 1, cy + r); ++yy)
            for (int xx = std::max(0, cx - r); xx <= std::min(g.width - 1, cx + r); ++xx)
            {
                const float pr = g.prob[yy * g.width + xx];
                if (pr <= p_threshold) continue;
                const float dx = g.cell_cx(xx) - x, dy = g.cell_cy(yy) - y;
                if (dx * dx + dy * dy <= r2) m += pr;
            }
        return static_cast<float>(m);
    }

    // Self-test: two blobs (one already under a refrigerator, one free); scan must find both, mark exactly one covered,
    // reject a sub-min_cells speck, and rank the free blob's mass correctly.
    static bool self_test()
    {
        GridField g;
        g.xmin = 0.f; g.ymin = 0.f; g.cell = 0.1f; g.width = 40; g.height = 40;
        g.prob.assign(static_cast<std::size_t>(g.width * g.height), 0.0f);
        const auto set = [&](int x, int y, float v) { g.prob[y * g.width + x] = v; };
        for (int y = 5; y < 12; ++y) for (int x = 5; x < 12; ++x) set(x, y, 0.9f);   // blob A (room ~0.5..1.2)
        for (int y = 25; y < 35; ++y) for (int x = 25; x < 35; ++x) set(x, y, 0.8f); // blob B (bigger, ~2.5..3.5)
        set(20, 20, 0.95f);                                                          // speck (1 cell → dropped)
        std::vector<FootprintBox> refrigerators{ {0.85f, 0.85f, 0.9f, 0.9f, 0.0f} };        // covers blob A
        const auto c = scan(g, refrigerators, {});
        bool ok = c.size() == 2;                          // two real regions, speck dropped
        ok = ok and c[0].mass > c[1].mass;                // sorted by mass; B (bigger) first
        ok = ok and not c[0].covered_by_refrigerator;            // blob B is free
        ok = ok and c[1].covered_by_refrigerator;                // blob A sits under the refrigerator
        // residual_mass_near: high under blob B's centre (~room 3.0,3.0), ~0 in an empty corner (~room 0.2,3.5)
        const float m_on  = residual_mass_near(g, 3.0f, 3.0f, 0.5f);
        const float m_off = residual_mass_near(g, 0.2f, 3.5f, 0.5f);
        ok = ok and m_on > 5.0f and m_off < 1e-3f;
        std::printf("BirthSurpriseProbe::self_test %s (regions=%zu, mass_on=%.1f mass_off=%.1f)\n",
                    ok ? "PASS" : "FAIL", c.size(), m_on, m_off);
        return ok;
    }

private:
    static bool covered(float x, float y, const std::vector<FootprintBox>& refrigerators, float margin)
    {
        for (const auto& t : refrigerators)
        {
            const float c = std::cos(t.yaw), s = std::sin(t.yaw);
            const float dx = x - t.cx, dy = y - t.cy;
            const float lx =  c * dx + s * dy;   // world → refrigerator-local (rotate by −yaw)
            const float ly = -s * dx + c * dy;
            if (std::abs(lx) <= 0.5f * t.w + margin and std::abs(ly) <= 0.5f * t.h + margin) return true;
        }
        return false;
    }
};

}  // namespace rc
#endif  // REFRIGERATOR_BIRTH_SURPRISE_PROBE_H
