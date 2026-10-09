// g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/free_space_polygon_selftest.cpp -o /tmp/fsp && /tmp/fsp
//
// The start-up proto-room's extent (src/free_space_polygon.h) on a synthetic L-shaped room, scanned by a
// ray-cast LiDAR (2 cm noise) from a robot driving from one arm of the L into the other. After EVERY scan:
//   1. NEVER CROSSES A WALL — every sample of the ring and of its interior is inside the true room and at
//      least the erosion radius from every wall (one cell of slack for the raster);
//   2. NEVER CONTAINS UNSEEN SPACE — every interior sample is in line of sight of some scan origin so far
//      (the truth's visibility, not the accumulator's own raster: a check computed the code's own way
//      cannot refute the code);
//   3. GROWS MONOTONICALLY — the confirmed-free raster never loses a cell (no matter appears in an empty
//      room), and the ring's area never falls by more than the simplification can move it;
//   4. is a simple CCW ring of <= 64 vertices around the robot.
// And at the end it must have SEEN most of the room: area >= 70% of the room eroded by the radius.
#include "free_space_polygon.h"

#include <cmath>
#include <cstdio>
#include <optional>
#include <random>
#include <vector>

namespace
{
    using Poly = std::vector<Eigen::Vector2f>;
    using rc::freespace::point_in_polygon;

    std::optional<float> ray_segment_t(const Eigen::Vector2f& o, const Eigen::Vector2f& d,
                                       const Eigen::Vector2f& a, const Eigen::Vector2f& b)
    {
        const Eigen::Vector2f e = b - a;
        const float den = d.x() * e.y() - d.y() * e.x();
        if (std::abs(den) < 1e-12f) return std::nullopt;
        const Eigen::Vector2f ao = a - o;
        const float t = (ao.x() * e.y() - ao.y() * e.x()) / den;
        const float u = (ao.x() * d.y() - ao.y() * d.x()) / den;
        if (t < 0.f or u < 0.f or u > 1.f) return std::nullopt;
        return t;
    }
    float dist_to_boundary(const Eigen::Vector2f& q, const Poly& room)
    {
        float best = 1e9f;
        for (std::size_t e = 0; e < room.size(); ++e)
        {
            const Eigen::Vector2f a = room[e], ab = room[(e + 1) % room.size()] - a;
            const float t = std::clamp((q - a).dot(ab) / ab.squaredNorm(), 0.f, 1.f);
            best = std::min(best, (q - (a + t * ab)).norm());
        }
        return best;
    }
    bool visible(const Eigen::Vector2f& o, const Eigen::Vector2f& q, const Poly& room, float range)
    {
        const Eigen::Vector2f d = q - o;
        const float L = d.norm();
        if (L > range) return false;
        if (L < 1e-6f) return true;
        const Eigen::Vector2f u = d / L;
        for (std::size_t e = 0; e < room.size(); ++e)
            if (const auto t = ray_segment_t(o, u, room[e], room[(e + 1) % room.size()]); t and *t < L)
                return false;
        return true;
    }
}   // namespace

int main()
{
    // L room: a 6 x 3 bottom arm and a 3 x 3 upper arm on the left.
    const Poly room = {{0.f, 0.f}, {6.f, 0.f}, {6.f, 3.f}, {3.f, 3.f}, {3.f, 6.f}, {0.f, 6.f}};
    const float range = 8.f;
    rc::freespace::Params prm;
    prm.erode_m = 0.24f;
    rc::freespace::FreeSpacePolygon fs(prm);
    std::mt19937 rng(11);
    std::normal_distribution<float> noise(0.f, 0.02f);

    // Drive: start deep in the bottom-right arm (the upper arm is out of sight), go left, then up.
    Poly path;
    for (float x = 5.0f; x >= 1.5f; x -= 0.25f) path.emplace_back(x, 1.0f);
    for (float y = 1.25f; y <= 5.0f; y += 0.25f) path.emplace_back(1.5f, y);

    int fails = 0;
    long prev_free = 0;
    float prev_area = 0.f;
    Poly origins;
    int k = 0;
    for (const auto& o : path)
    {
        origins.push_back(o);
        std::vector<Eigen::Vector2f> ends;
        for (int b = 0; b < 720; ++b)
        {
            const float a = 2.f * static_cast<float>(M_PI) * b / 720.f;
            const Eigen::Vector2f u(std::cos(a), std::sin(a));
            float t = range;
            bool hit = false;
            for (std::size_t e = 0; e < room.size(); ++e)
                if (const auto tt = ray_segment_t(o, u, room[e], room[(e + 1) % room.size()]); tt and *tt < t)
                { t = *tt; hit = true; }
            if (hit) ends.push_back(o + u * (t + noise(rng)));
        }
        fs.add_scan(o, ends);
        const auto ring = fs.polygon(o);
        ++k;
        // 3a. the raster is monotone in an empty room — away from the wall surface. A cell ON the surface
        //     is legitimately both crossed by one noisy beam and hit by another, and MATTER wins there by
        //     design (sticky); more than 3 sigma + one cell inside the room nothing may ever be lost.
        long interior_free = 0;
        for (float x = 0.025f; x < 6.f; x += prm.cell)
            for (float y = 0.025f; y < 6.f; y += prm.cell)
                if (point_in_polygon({x, y}, room) and dist_to_boundary({x, y}, room) > 0.06f + prm.cell
                    and fs.state_at({x, y}) == 1)
                    ++interior_free;
        if (interior_free < prev_free) { ++fails; std::printf("FAIL scan %d: interior free cells fell %ld -> %ld\n", k, prev_free, interior_free); }
        prev_free = interior_free;
        if (ring.empty()) { ++fails; std::printf("FAIL scan %d: no ring\n", k); continue; }
        // 4. shape
        const float area = rc::freespace::signed_area(ring);
        if (ring.size() > 64 or area <= 0.f or not rc::freespace::is_simple(ring) or not point_in_polygon(o, ring))
        { ++fails; std::printf("FAIL scan %d: verts=%zu area=%.3f simple=%d contains_robot=%d\n", k, ring.size(), area,
                               rc::freespace::is_simple(ring), point_in_polygon(o, ring)); }
        // 3b. area: never falls by more than the eps-band of one doubling (generous: 2 cells of perimeter)
        float perim = 0.f;
        for (std::size_t i = 0; i < ring.size(); ++i) perim += (ring[(i + 1) % ring.size()] - ring[i]).norm();
        if (area < prev_area - 2.f * prm.cell * perim)
        { ++fails; std::printf("FAIL scan %d: area fell %.3f -> %.3f\n", k, prev_area, area); }
        prev_area = std::max(prev_area, area);
        // 1 + 2. dense samples of the ring and its interior
        int bad_wall = 0, bad_unseen = 0, n = 0;
        float worst_clear = 1e9f;
        Poly samples;
        for (std::size_t i = 0; i < ring.size(); ++i)
            for (int s = 0; s < 20; ++s)
                samples.push_back(ring[i] + (ring[(i + 1) % ring.size()] - ring[i]) * (s / 20.f));
        for (float x = -0.5f; x <= 6.5f; x += 0.05f)
            for (float y = -0.5f; y <= 6.5f; y += 0.05f)
                if (point_in_polygon({x, y}, ring)) samples.emplace_back(x, y);
        for (const auto& q : samples)
        {
            ++n;
            const bool inside = point_in_polygon(q, room);
            const float clear = dist_to_boundary(q, room);
            worst_clear = std::min(worst_clear, inside ? clear : -clear);
            if (not inside or clear < prm.erode_m - prm.cell) ++bad_wall;
            bool seen = false;
            for (const auto& oo : origins) if (visible(oo, q, room, range)) { seen = true; break; }
            if (not seen) ++bad_unseen;
        }
        if (bad_wall > 0 or bad_unseen > 0)
        { ++fails; std::printf("FAIL scan %d: %d/%d samples too close to / past a wall (worst clearance %.3f m), "
                               "%d unseen\n", k, bad_wall, n, worst_clear, bad_unseen); }
        if (k % 6 == 0 or k == static_cast<int>(path.size()))
            std::printf("scan %2d at (%.2f,%.2f): verts=%2zu area=%6.3f m2 worst clearance=%.3f m free_cells=%ld\n",
                        k, o.x(), o.y(), ring.size(), area, worst_clear, fs.free_cells());
    }
    // The room eroded by the radius: 6x3 + 3x3 = 27 m2, minus a 0.24 m band around 24 m of perimeter.
    const float target = 27.f - 0.24f * 24.f;
    const auto final_ring = fs.polygon(path.back());
    const float final_area = final_ring.empty() ? 0.f : rc::freespace::signed_area(final_ring);
    for (const auto& v : final_ring) std::printf("  (%.2f,%.2f)", v.x(), v.y());
    std::printf("\n");
    std::printf("final area %.2f m2 vs %.2f m2 eroded room (%.0f%%)\n", final_area, target, 100.f * final_area / target);
    if (final_area < 0.7f * target) { ++fails; std::printf("FAIL: the ring saw too little of the room\n"); }
    // ── A PILLAR: the documented limit. An island inside the free region is FILLED (one ring cannot carry
    //    a hole); the ring must still be simple, contain the robot, stay inside the room and off its outer
    //    walls. Samples inside the pillar or its shadow are counted and reported, not failed.
    {
        const Poly box = {{0.f, 0.f}, {6.f, 0.f}, {6.f, 4.f}, {0.f, 4.f}};
        const Poly pillar = {{2.8f, 1.8f}, {3.2f, 1.8f}, {3.2f, 2.2f}, {2.8f, 2.2f}};
        rc::freespace::FreeSpacePolygon fp(prm);
        Poly origins2;
        for (float x = 1.0f; x <= 5.0f; x += 0.5f)
        {
            const Eigen::Vector2f o(x, 1.0f);
            origins2.push_back(o);
            std::vector<Eigen::Vector2f> ends;
            for (int b = 0; b < 720; ++b)
            {
                const float a = 2.f * static_cast<float>(M_PI) * b / 720.f;
                const Eigen::Vector2f u(std::cos(a), std::sin(a));
                float t = range; bool hit = false;
                for (const Poly* P : {&box, &pillar})
                    for (std::size_t e = 0; e < P->size(); ++e)
                        if (const auto tt = ray_segment_t(o, u, (*P)[e], (*P)[(e + 1) % P->size()]); tt and *tt < t)
                        { t = *tt; hit = true; }
                if (hit) ends.push_back(o + u * (t + noise(rng)));
            }
            fp.add_scan(o, ends);
        }
        const auto ring = fp.polygon(origins2.back());
        int outside = 0, in_pillar = 0, n = 0;
        for (float x = 0.f; x <= 6.f; x += 0.05f)
            for (float y = 0.f; y <= 4.f; y += 0.05f)
                if (point_in_polygon({x, y}, ring))
                {
                    ++n;
                    if (not point_in_polygon({x, y}, box) or dist_to_boundary({x, y}, box) < prm.erode_m - prm.cell) ++outside;
                    if (point_in_polygon({x, y}, pillar)) ++in_pillar;
                }
        const bool ok = not ring.empty() and rc::freespace::is_simple(ring) and ring.size() <= 64
                        and point_in_polygon(origins2.back(), ring) and outside == 0;
        std::printf("pillar room: verts=%zu samples=%d too-close-to-outer-wall=%d inside-pillar=%d (filled hole, "
                    "documented) %s\n", ring.size(), n, outside, in_pillar, ok ? "PASS" : "FAIL");
        if (not ok) ++fails;
    }
    std::printf("%s: %d failure(s)\n", fails == 0 ? "ALL PASS" : "FAILED", fails);
    return fails == 0 ? 0 : 1;
}
