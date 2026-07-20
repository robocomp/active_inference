/*
 * residual_floor_plane.cpp  —  see residual_floor_plane.h
 */

#include "residual_floor_plane.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace rc
{

namespace
{
// Weighted least-squares fit of z = a·x + b·y + c over the indexed points (weight 1). Returns false if the normal
// matrix is singular (degenerate geometry — e.g. all points collinear).
bool fit_plane(const std::vector<Eigen::Vector3f>& pts, const std::vector<int>& idx,
               float& a, float& b, float& c)
{
    if (idx.size() < 3) return false;
    Eigen::Matrix3d AtA = Eigen::Matrix3d::Zero();
    Eigen::Vector3d Atz = Eigen::Vector3d::Zero();
    for (const int i : idx)
    {
        const Eigen::Vector3d r(pts[i].x(), pts[i].y(), 1.0);
        AtA += r * r.transpose();
        Atz += r * static_cast<double>(pts[i].z());
    }
    // LDLT is stable for this symmetric PSD 3×3; bail if it is not positive-definite enough (singular geometry).
    const Eigen::LDLT<Eigen::Matrix3d> ldlt(AtA);
    if (ldlt.info() != Eigen::Success) return false;
    const Eigen::Vector3d s = ldlt.solve(Atz);
    if (not s.allFinite()) return false;
    a = static_cast<float>(s.x()); b = static_cast<float>(s.y()); c = static_cast<float>(s.z());
    return true;
}
}  // namespace

FloorPlane estimate_floor_plane(const std::vector<Eigen::Vector3f>& points_room, const Eigen::Vector3f& origin,
                                float floor_z0, float floor_slope, const FloorPlaneParams& p, const FloorPlane& prev)
{
    // 1) Candidate floor points: within candidate_band_m ABOVE the current best floor estimate (prev, or z=0 on
    //    the first cycle). This admits an offset/tilted floor while excluding tall obstacles up front.
    const float ref_a = prev.valid ? prev.a : 0.0f;
    const float ref_b = prev.valid ? prev.b : 0.0f;
    const float ref_c = prev.valid ? prev.c : 0.0f;
    std::vector<int> idx;
    idx.reserve(points_room.size());
    for (int i = 0; i < static_cast<int>(points_room.size()); ++i)
    {
        const auto& q = points_room[i];
        const float range   = std::hypot(q.x() - origin.x(), q.y() - origin.y());
        const float floor_z = ref_a * q.x() + ref_b * q.y() + ref_c;
        const float top     = floor_z + floor_z0 + floor_slope * range + p.candidate_band_m;
        const float bot     = floor_z - p.candidate_band_m;                 // symmetric guard below (dips/noise)
        if (q.z() >= bot and q.z() <= top) idx.push_back(i);
    }
    if (static_cast<int>(idx.size()) < p.min_candidates) return prev;       // too little floor seen → hold

    // 2) Robust fit: least-squares plane, then trim by residual MAD and refit (rejects legs / low clutter).
    float a = ref_a, b = ref_b, c = ref_c;
    if (not fit_plane(points_room, idx, a, b, c)) return prev;
    for (int it = 0; it < std::max(0, p.iters); ++it)
    {
        std::vector<float> res; res.reserve(idx.size());
        for (const int i : idx) res.push_back(points_room[i].z() - (a * points_room[i].x() + b * points_room[i].y() + c));
        std::vector<float> absres(res.size());
        for (std::size_t k = 0; k < res.size(); ++k) absres[k] = std::abs(res[k]);
        std::vector<float> tmp = absres;
        std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
        const float mad = std::max(1e-4f, tmp[tmp.size() / 2]);             // median abs residual (robust scale)
        const float thr = p.trim_k * 1.4826f * mad;                        // MAD→σ, then k·σ
        std::vector<int> keep; keep.reserve(idx.size());
        for (std::size_t k = 0; k < idx.size(); ++k) if (absres[k] <= thr) keep.push_back(idx[k]);
        if (static_cast<int>(keep.size()) < p.min_candidates) break;       // trimmed too hard → stop with current fit
        idx.swap(keep);
        if (not fit_plane(points_room, idx, a, b, c)) return prev;
    }

    // 3) Sanity: reject a wild fit (keep the previous plane so a bad cycle can't teleport the floor).
    if (not std::isfinite(a) or not std::isfinite(b) or not std::isfinite(c)
        or std::abs(a) > p.max_tilt or std::abs(b) > p.max_tilt or std::abs(c) > p.max_offset_m)
        return prev;

    // 4) Temporal EMA toward the new fit (slow → stable). First valid fit initialises directly.
    FloorPlane out;
    out.valid = true;
    if (prev.valid)
    {
        const float e = std::clamp(p.ema, 0.0f, 1.0f);
        out.a = prev.a + e * (a - prev.a);
        out.b = prev.b + e * (b - prev.b);
        out.c = prev.c + e * (c - prev.c);
    }
    else { out.a = a; out.b = b; out.c = c; }
    return out;
}

// ─── Self-test: recover an offset+tilted floor from a cloud with obstacles/legs mixed in ────────────────────
bool floor_plane_self_test()
{
    bool ok = true;
    auto check = [&](bool cnd, const char* m) { if (not cnd) { ok = false; std::printf("  FAIL: %s\n", m); } };

    FloorPlaneParams p; p.enabled = true; p.ema = 1.0f;   // ema=1 → converge in one call for the test
    const Eigen::Vector3f origin(0, 0, 0.5f);

    // Ground truth floor: offset c=0.12 m, tilted a=0.03 (3 cm/m in x). Sample a dense floor grid …
    const float ga = 0.03f, gb = -0.01f, gc = 0.12f;
    std::vector<Eigen::Vector3f> pts;
    for (int ix = -20; ix <= 20; ++ix)
        for (int iy = -20; iy <= 20; ++iy)
        {
            const float x = ix * 0.15f, y = iy * 0.15f;
            pts.emplace_back(x, y, ga * x + gb * y + gc);
        }
    // … plus a table + some legs standing WELL above the floor (obstacles the trim must reject).
    for (int k = 0; k < 300; ++k)
    { const float x = -1.0f + 0.004f * k, y = 0.5f; pts.emplace_back(x, y, ga * x + gb * y + gc + 0.75f); }
    for (int k = 0; k < 60; ++k)
    { const float x = 0.8f, y = -0.6f + 0.01f * k; pts.emplace_back(x, y, ga * x + gb * y + gc + 0.30f); }

    // (1) The band+trim recover the offset+tilt despite the obstacles.
    FloorPlane fp = estimate_floor_plane(pts, origin, 0.06f, 0.04f, p, FloorPlane{});
    check(fp.valid, "must produce a valid plane");
    check(std::abs(fp.c - gc) < 0.02f, "recover the floor OFFSET c");
    check(std::abs(fp.a - ga) < 0.01f and std::abs(fp.b - gb) < 0.01f, "recover the floor TILT a,b");

    // (2) A FLAT floor at z=0 returns ≈(0,0,0) → zero regression vs the fixed band.
    std::vector<Eigen::Vector3f> flat;
    for (int ix = -20; ix <= 20; ++ix) for (int iy = -20; iy <= 20; ++iy) flat.emplace_back(ix * 0.15f, iy * 0.15f, 0.0f);
    FloorPlane fz = estimate_floor_plane(flat, origin, 0.06f, 0.04f, p, FloorPlane{});
    check(fz.valid and std::abs(fz.a) < 1e-3f and std::abs(fz.b) < 1e-3f and std::abs(fz.c) < 1e-3f,
          "flat z=0 floor ⇒ plane ≈ (0,0,0)");

    // (3) Too few candidates ⇒ keep the previous plane (never fit on noise).
    FloorPlane seed; seed.valid = true; seed.c = 0.09f;
    FloorPlane held = estimate_floor_plane({{0, 0, 0.0f}}, origin, 0.06f, 0.04f, p, seed);
    check(held.valid and std::abs(held.c - 0.09f) < 1e-6f, "too few candidates ⇒ hold previous plane");

    std::printf("[residual_floor_plane] self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
