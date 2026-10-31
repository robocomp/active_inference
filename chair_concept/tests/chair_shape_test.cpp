// chair_shape_test: the SHAPE IDENTITY. The card's generative block (closed primitives bound to the four size
// variables) must describe the same body as ChairModel's compound SDF. If they disagree, the primitives are wrong,
// never the SDF (spec 11b.4).
//   points: 2000 uniform in the chair's bounding box + 0.2 m, at the prior mean and at each of the 16 +-2 sigma
//           corners of (seat_w, seat_d, seat_h, back_h), clamped to the variables' [lo, hi].
//   (1) |sdf| > 0.01 and |prim| > 0.01  =>  the inside/outside signs agree.
//   (2) |sdf| < 0.03                    =>  |prim - sdf| < 0.01.
// "prim" is the EXACT signed distance to the union of the primitives (Euclidean outside, -min depth inside). The SDF
// is exact outside but NOT inside: box_sdf's inside term is -0.3 * (softmax of the axis depths), so just inside a
// thin body it is far from the metric distance. Check (2) is therefore applied in full OUTSIDE (sdf >= 0), and the
// inside is checked against a documented model of that quirk (inside_sdf_quirk), with both max deviations printed.
// Run under the agent locale:  LANG=es_ES.UTF-8 ./chair_shape_test
#include "chair_card.h"
#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
using namespace rc;

using Vars = std::map<std::string, float>;
static float val(const card::Primitive& p, const char* k, const Vars& v)
{
    const auto& b = p.dims.at(k);
    return b.var.empty() ? b.b : b.a * v.at(b.var) + b.b;
}
// exact signed distance to an axis-aligned box given centre and half extents (negative inside = -min depth)
static float box_dist(const Eigen::Vector3f& q, const Eigen::Vector3f& c, const Eigen::Vector3f& h)
{
    const Eigen::Vector3f d = (q - c).cwiseAbs() - h;
    const float out = d.cwiseMax(0.0f).norm();
    return out + std::min(std::max({d.x(), d.y(), d.z()}), 0.0f);
}
// the SDF's own inside value for a box: -0.3 * k * log(sum exp(depth_i / k)), k = 0.02 (only when strictly inside)
static float inside_sdf_quirk(const Eigen::Vector3f& q, const Eigen::Vector3f& c, const Eigen::Vector3f& h)
{
    const Eigen::Vector3f d = (q - c).cwiseAbs() - h;
    if (not (d.x() < 0 and d.y() < 0 and d.z() < 0)) return std::max(d.maxCoeff(), 0.0f) > 0 ? d.cwiseMax(0.0f).norm() : 0.0f;
    constexpr float k = ChairModel::SDF_SMOOTH_K;
    float mx = std::max({-d.x(), -d.y(), -d.z()}) / k, sum = 0;
    for (int i = 0; i < 3; ++i) sum += std::exp(-d[i] / k - mx);
    return ChairModel::SDF_INSIDE_SCALE * (-k * (mx + std::log(sum)));
}
// the boxes of a primitive list: (centre, half extents)
static std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> boxes(const card::Generative& g, const Vars& v)
{
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> out;
    for (const auto& p : g.shape)
    {
        using card::Prim;
        if (p.type == Prim::Slab or p.type == Prim::Box or p.type == Prim::Panel)
        {
            const float w = val(p, "w", v), d = val(p, p.type == Prim::Panel ? "t" : "d", v), h = val(p, "h", v);
            out.push_back({{val(p, "x", v), val(p, "y", v), val(p, "z", v) + 0.5f * h}, {0.5f * w, 0.5f * d, 0.5f * h}});
        }
        else if (p.type == Prim::LegSet)
        {
            CHECK(static_cast<int>(val(p, "n", v)) == 4);
            const float hf = val(p, "half", v), ox = 0.5f * val(p, "fw", v) - hf, oy = 0.5f * val(p, "fd", v) - hf;
            const float z0 = val(p, "z0", v), z1 = val(p, "z1", v);
            for (const auto& s : {std::pair{1, 1}, std::pair{-1, 1}, std::pair{-1, -1}, std::pair{1, -1}})
                out.push_back({{s.first * ox, s.second * oy, 0.5f * (z0 + z1)}, {hf, hf, 0.5f * (z1 - z0)}});
        }
        else CHECK(false);   // no cylinders in a chair
    }
    return out;
}

int main()
{
    CHECK(std::setlocale(LC_ALL, "") != nullptr);
    const auto prior = load_chair_prior(std::string(AGENT_DIR) + "/../common/concept_manifest/chair.concept.toml");
    CHECK(prior.has_value());
    if (not prior) return 1;
    ChairConfig cfg;
    cfg.prior = *prior;
    const card::Card c = make_card(cfg);
    CHECK(c.generative.has_value());
    if (not c.generative) return 1;
    const card::Generative& g = *c.generative;
    CHECK(g.vars.size() == 4 and g.shape.size() == 3);

    // the exported bounds ARE apply_constraints' (one source): a model clamped by apply_constraints stays inside [lo, hi]
    {
        ChairModel m;
        m.set_state(ChairState::from_array({0, 0, 0, 0, 9.f, 9.f, 9.f, 9.f}));
        const auto hi = m.state();
        m.set_state(ChairState::from_array({0, 0, 0, 0, -9.f, -9.f, -9.f, -9.f}));
        const auto lo = m.state();
        for (const auto& v : g.vars)
        {
            const float h = v.var == "seat_w" ? hi.seat_w : v.var == "seat_d" ? hi.seat_d : v.var == "seat_h" ? hi.seat_h : hi.back_h;
            const float l = v.var == "seat_w" ? lo.seat_w : v.var == "seat_d" ? lo.seat_d : v.var == "seat_h" ? lo.seat_h : lo.back_h;
            CHECK(v.hi == h and v.lo == l);
            CHECK(v.sigma == prior->size_std);
        }
    }

    // configurations: the prior mean, then the 16 +-2 sigma corners clamped to [lo, hi]
    std::vector<Vars> configs;
    {
        Vars m;
        for (const auto& v : g.vars) m[v.var] = v.mean;
        configs.push_back(m);
        for (int mask = 0; mask < 16; ++mask)
        {
            Vars k;
            for (std::size_t i = 0; i < g.vars.size(); ++i)
            {
                const auto& v = g.vars[i];
                k[v.var] = std::clamp(v.mean + ((mask >> i & 1) ? 2.0f : -2.0f) * v.sigma, v.lo, v.hi);
            }
            configs.push_back(k);
        }
    }

    ChairModel model;
    std::mt19937 rng(12345);
    std::size_t n_pts = 0, sign_checked = 0, sign_bad = 0, near_out = 0, near_in = 0;
    float max_out = 0, max_in = 0, max_in_quirk = 0;
    for (const auto& v : configs)
    {
        ChairState s{};
        s.seat_w = v.at("seat_w"); s.seat_d = v.at("seat_d"); s.seat_h = v.at("seat_h"); s.back_h = v.at("back_h");
        const auto bx = boxes(g, v);
        std::uniform_real_distribution<float> ux(-0.5f * s.seat_w - 0.2f, 0.5f * s.seat_w + 0.2f),
            uy(-0.5f * s.seat_d - 0.2f, 0.5f * s.seat_d + 0.2f), uz(-0.2f, s.seat_h + s.back_h + 0.2f);
        for (int i = 0; i < 2000; ++i, ++n_pts)
        {
            const Eigen::Vector3f q(ux(rng), uy(rng), uz(rng));
            const float sdf = model.sdf_point_at(q, s);
            float pd = 1e9f, pq = 1e9f;
            for (const auto& [ctr, h] : bx) { pd = std::min(pd, box_dist(q, ctr, h)); pq = std::min(pq, inside_sdf_quirk(q, ctr, h)); }
            if (std::fabs(sdf) > 0.01f and std::fabs(pd) > 0.01f)
            {
                ++sign_checked;
                if ((sdf < 0) != (pd < 0)) ++sign_bad;
            }
            if (std::fabs(sdf) < 0.03f)
            {
                if (sdf >= 0) { ++near_out; max_out = std::max(max_out, std::fabs(pd - sdf)); }
                else { ++near_in; max_in = std::max(max_in, std::fabs(pd - sdf)); max_in_quirk = std::max(max_in_quirk, std::fabs(pq - sdf)); }
            }
        }
    }
    std::printf("shape identity: %zu points (%zu configs x 2000), sign checked %zu, sign disagreements %zu\n",
                n_pts, configs.size(), sign_checked, sign_bad);
    std::printf("near surface (|sdf|<0.03): outside n=%zu max|dd|=%.6f m;  inside n=%zu max|dd| vs exact distance=%.6f m, vs the SDF's inside convention=%.6f m\n",
                near_out, max_out, near_in, max_in, max_in_quirk);
    CHECK(n_pts == 17u * 2000u);
    CHECK(sign_checked > 10000);
    CHECK(sign_bad == 0);
    CHECK(near_out > 100 and max_out < 0.01f);
    CHECK(near_in > 0 and max_in_quirk < 0.01f);
    std::printf(bad ? "chair_shape_test: %d FAILED\n" : "chair_shape_test: OK\n", bad);
    return bad != 0;
}
