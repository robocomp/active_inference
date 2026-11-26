#pragma once
// Birth on the furniture priors, for a wall-standing class (spec 2026-11-15 lidar-measures §7c.8 rev 3.3.14).
//
// The size is the class mean; the data sizes the box afterwards. The cloud's spread does not set the size; it sets
// how uncertain the POSITION is.
//  · ALONG the wall the centre is the cloud's middle. An object of length L and a seen extent E place that middle anywhere
//    over |L − E|: E > L, the object lies somewhere inside a cloud that took in more (a neighbour: live14fixA, 2.46 m for a
//    0.51 m nightstand); E < L, the cloud is the part of the object in view (a mask clipped at the image border, a partial
//    view) and the object extends past it on either side. Uniform either way, variance (L − E)²/12 (plan rev 2 F1.3: this
//    replaces the fitter's "skip the whole birth on a truncated mask" switch, which started GN 0.8 m away on engine1's bed).
//  · ACROSS the wall the centre is NOT the cloud's: it is h̄/2 in from the wall (the back on the wall). A shallower cloud (one
//    face) says nothing about it; only a cloud reaching further from the wall than the object is deep says the object may
//    stand off it, over E − h̄. So that axis keeps max(0, E − h̄): the cloud-placed centre's spread, not a cutoff.
#include <algorithm>
#include <cmath>
#include <span>
#include <vector>
#include <Eigen/Dense>

namespace rc
{
struct BirthFootprint
{
    bool ok = false;
    float w = 0.0f, h = 0.0f, yaw = 0.0f;              // seed size (class means) and yaw (back on the wall)
    Eigen::Vector2f centre = Eigen::Vector2f::Zero();  // room frame: the cloud's middle along the wall, h/2 in from it
    float extent_t = 0.0f, extent_n = 0.0f;            // the cloud's full extent along / across the wall, (p95 − p5)/0.9
    float mid_n = 0.0f;                                // the cloud's middle ACROSS the wall, (p5 + p95)/2, from the wall (F2)
    float var_t = 0.0f, var_n = 0.0f;                  // position variance along / across the wall
    Eigen::Matrix2f pos_cov = Eigen::Matrix2f::Zero(); // the same, rotated into the room frame
};

// xy: the cloud's points (room frame, already z-banded). wall_p: a point on the nearest wall; wall_n: its unit normal
// into the room. class_w / class_d: the class's mean width / depth; pos_std: the class position std (σ_pos).
inline BirthFootprint birth_footprint(std::span<const Eigen::Vector2f> xy, const Eigen::Vector2f& wall_p,
                                      const Eigen::Vector2f& wall_n, float class_w, float class_d, float pos_std)
{
    BirthFootprint out;
    if (xy.empty()) return out;
    const Eigen::Vector2f n = wall_n, t(-n.y(), n.x());   // into the room, and along the wall
    std::vector<float> a, b;
    a.reserve(xy.size()); b.reserve(xy.size());
    for (const auto& q : xy) { a.push_back((q - wall_p).dot(t)); b.push_back((q - wall_p).dot(n)); }
    const auto pct = [](std::vector<float> v, float f)
    {
        const std::size_t k = static_cast<std::size_t>(f * static_cast<float>(v.size() - 1));
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
        return v[k];
    };
    const float a0 = pct(a, 0.05f), a1 = pct(a, 0.95f), b0 = pct(b, 0.05f), b1 = pct(b, 0.95f);
    out.w = class_w;
    out.h = class_d;
    out.yaw = std::atan2(n.x(), -n.y());                  // R(yaw)·ŷ = −n: the back faces the wall
    out.centre = wall_p + t * (0.5f * (a0 + a1)) + n * (0.5f * class_d);
    out.extent_t = (a1 - a0) / 0.9f;                      // p5–p95 holds 90 % of a uniform spread
    out.extent_n = (b1 - b0) / 0.9f;
    out.mid_n = 0.5f * (b0 + b1);
    const float ex_t = out.extent_t - class_w, ex_n = std::max(0.0f, out.extent_n - class_d);
    const float pp = pos_std * pos_std;
    out.var_t = pp + ex_t * ex_t / 12.0f;
    out.var_n = pp + ex_n * ex_n / 12.0f;
    Eigen::Matrix2f R;
    R.col(0) = t; R.col(1) = n;
    out.pos_cov = R * Eigen::Vector2f(out.var_t, out.var_n).asDiagonal() * R.transpose();
    out.ok = true;
    return out;
}
}  // namespace rc
