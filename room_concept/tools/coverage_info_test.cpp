// AuRo journal item A (2026-11-23): the room-building offer in nats (coverage_info.h).
//   g++ -std=c++23 -I../src coverage_info_test.cpp -o coverage_info_test && ./coverage_info_test
#include "coverage_info.h"
#include "room_explored.h"
#include "free_space_polygon.h"
#include <optional>
#include <vector>
#include <cmath>
#include <cstdio>

static int failures = 0;
static void check(bool ok, const char *what)
{ if (!ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what); }
static bool near(double a, double b, double tol = 1e-6) { return std::abs(a - b) <= tol; }

int main()
{
    using namespace rc::coverage;
    std::printf("binary entropy\n");
    check(near(binary_entropy(0.0), 0.0) and near(binary_entropy(1.0), 0.0), "h(0) = h(1) = 0");
    check(near(binary_entropy(0.5), std::log(2.0)), "h(1/2) = ln 2");
    check(near(binary_entropy(0.2), binary_entropy(0.8)), "symmetric");
    check(binary_entropy(0.1) < binary_entropy(0.3) and binary_entropy(0.3) < binary_entropy(0.5), "rises towards 1/2");

    std::printf("Beta prior\n");
    BetaPrior flat;
    check(near(flat.nats_per_unit(), std::log(2.0)), "Beta(1,1): ln 2 per unobserved unit");
    BetaPrior learnt; learnt.add(100.0, 900.0);   // a finished room: 10 % of its cells occupied
    check(near(learnt.mean(), 101.0 / 1002.0), "counts update the mean");
    check(learnt.nats_per_unit() < flat.nats_per_unit(), "a mostly-free room makes an unknown cell worth less");
    BetaPrior same = flat; same.add(-5.0, -5.0);
    check(near(same.mean(), 0.5) and near(same.a, 1.0), "negative counts are ignored");

    std::printf("unit conversion\n");
    const float cell = 0.05f;
    LengthPrior ell0;   // the stated prior: 0.3 m
    check(near(area_to_nats(1.0f, cell, flat, ell0), std::log(2.0) / 0.09, 1e-3), "1 m^2 at l = 0.3 m = 11.1 regions x ln 2 nats");
    LengthPrior tiny; tiny.add(0.001, 1e6);
    check(near(area_to_nats(1.0f, cell, flat, tiny), 400.0 * std::log(2.0), 1e-2), "l below the raster cell counts cells, never more");
    check(near(area_to_nats(0.f, cell, flat), 0.0), "nothing unknown, nothing offered");
    check(near(area_to_nats(1.f, 0.f, flat), 0.0), "degenerate raster offers nothing");
    check(near(wall_to_nats(2.0f, 1.0f, flat), 2.0 * std::log(2.0), 1e-6), "2 m of wall at 1 m doors = 2 segments x ln 2");

    std::printf("correlation length prior\n");
    LengthPrior lp;
    check(near(lp.value(), 0.3, 1e-9), "prior l0 = 0.3 m");
    lp.add(0.6);
    check(near(lp.value(), std::sqrt(0.3 * 0.6), 1e-9), "one room at 0.6 m: geometric mean with the prior");
    LengthPrior bad = lp; bad.add(-1.0); bad.add(0.5, 0.0);
    check(near(bad.value(), lp.value(), 1e-12), "non-positive lengths or weights are ignored");

    std::printf("correlation length from a scanned raster\n");
    {
        using Poly = std::vector<Eigen::Vector2f>;
        const auto ray_seg = [](const Eigen::Vector2f& o, const Eigen::Vector2f& d, const Eigen::Vector2f& a,
                                const Eigen::Vector2f& b) -> std::optional<float>
        {
            const Eigen::Vector2f e = b - a; const float den = d.x() * e.y() - d.y() * e.x();
            if (std::abs(den) < 1e-12f) return std::nullopt;
            const Eigen::Vector2f ao = a - o;
            const float t = (ao.x() * e.y() - ao.y() * e.x()) / den, u = (ao.x() * d.y() - ao.y() * d.x()) / den;
            if (t < 0.f or u < 0.f or u > 1.f) return std::nullopt;
            return t;
        };
        const auto ell_of = [&](bool boxes)
        {
            std::vector<Poly> polys = {{{0, 0}, {4, 0}, {4, 4}, {0, 4}}};
            if (boxes) { polys.push_back({{1, 2.5f}, {2, 2.5f}, {2, 3}, {1, 3}}); polys.push_back({{2.8f, 0.6f}, {3.3f, 0.6f}, {3.3f, 1.1f}, {2.8f, 1.1f}}); }
            rc::freespace::Params prm; rc::freespace::FreeSpacePolygon fs(prm);
            for (const Eigen::Vector2f o : Poly{{0.6f, 0.6f}, {2.f, 1.5f}, {3.4f, 3.4f}, {0.6f, 3.5f}})
            {
                std::vector<Eigen::Vector2f> ends;
                for (int b = 0; b < 720; ++b)
                {
                    const float a = 2.f * static_cast<float>(M_PI) * b / 720.f; const Eigen::Vector2f u(std::cos(a), std::sin(a));
                    float t = 8.f; bool hit = false;
                    for (const auto& P : polys) for (std::size_t e = 0; e < P.size(); ++e)
                        if (const auto tt = ray_seg(o, u, P[e], P[(e + 1) % P.size()]); tt and *tt < t) { t = *tt; hit = true; }
                    if (hit) ends.push_back(o + u * t);
                }
                fs.add_scan(o, ends);
            }
            return fs.occupancy_correlation_length(fs.countable_mask(polys[0], 0.40f));
        };
        const auto empty = ell_of(false), furnished = ell_of(true);
        check(empty.has_value() and furnished.has_value(), "estimated on both rooms");
        if (empty and furnished)
        {
            std::printf("     empty room l = %.3f m, furnished l = %.3f m\n", *empty, *furnished);
            check(*furnished < *empty, "furniture shortens the correlation length (more independent patches)");
            check(*furnished > 0.05f and *empty < 3.f, "within (one cell, the largest lag tried)");
        }
        rc::freespace::Params prm; rc::freespace::FreeSpacePolygon blank(prm);
        check(not blank.occupancy_correlation_length({}).has_value(), "no raster, no estimate");
    }

    std::printf("explored belief is unit-free\n");
    // p_stop = price / (price + gain): converting gain AND price by the same factor must not move it.
    const float price_m2 = 0.4f, gain_m2 = 1.3f;
    const auto p_m2 = rc::explored::p_stop(price_m2, gain_m2);
    const auto p_nats = rc::explored::p_stop(area_to_nats(price_m2, cell, learnt), area_to_nats(gain_m2, cell, learnt));
    check(p_m2.has_value() and p_nats.has_value() and near(*p_m2, *p_nats, 1e-5), "p_stop identical in m^2 and in nats");

    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
