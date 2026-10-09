#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
// PLAIN 2-D SLAM FOR ESTIMATE MODE (2026-10-09) — the pose while the room is being learnt.
//
// While the layout is being estimated the pose used to be solved against the PROVISIONAL LAYOUT itself (the wall
// landmarks, room_concept.cpp run_gn_loop: in.walls = &wall_map_). Pose and layout were estimated against each other:
// a half-room seed, a weak "twin" wall replacing a strong one, or a missing wall dragged the pose, and the dragged
// pose corrupted the layout. Measured on three live recordings against Webots ground truth, that coupling made the
// pose 4x WORSE than odometry alone (median 0.22 / 1.55 / 3.13 m vs odometry 0.08 / 1.77 / 0.20 m).
//
// This is the standard alternative (user's proposal): register each scan against an OCCUPANCY MAP of the scans
// already registered — no layout anywhere in the loop. The layout estimator consumes the aligned scans; it never
// feeds back into the pose until the room is promoted and given-mode localisation takes over.
//
// MODEL (validated offline, scratchpad/slam/sdfslam.py: median 0.010 / 0.031 / 0.013 m on the same three runs):
//   map         5 cm hit counts of REGISTERED wall-band returns; a cell is SURFACE once seen twice (one return can
//               be noise). Distance field D = exact Euclidean distance to the nearest surface cell.
//   scan term   each return's distance to the mapped surface ~ N(0, sigma_lidar), with a CAUCHY tail of scale
//               `cauchy_m`: a return on structure not yet mapped fades out smoothly — no association gate.
//   odometry    a STUDENT-t prior (nu dof) on the deviation from the odometry prediction, sigma proportional to the
//               motion. Wheel slip (wheels turning, body blocked — measured: 1.5 m reported, 0.001 m moved) is an
//               outlier process; the t lets 400 metric returns overrule it in proportion to the disagreement.
//   solve       Gauss-Newton (IRLS) from the odometry prediction; posterior covariance = H^-1 at the solution.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace rc::slam
{
struct Params
{
    float cell        = 0.05f;   ///< m, map resolution
    float half_span   = 12.f;    ///< m, the map covers [-half_span, half_span]^2 round the start
    int   surface_hits = 2;      ///< hits for a cell to count as surface (one return can be noise)
    float sigma_lidar = 0.03f;   ///< m, range/registration noise of one return against the map
    float cauchy_m    = 0.10f;   ///< m, scale of the heavy tail (structure not yet mapped)
    float nu          = 3.f;     ///< Student-t dof of the odometry factor (wheel slip = outliers)
    float odo_t_floor = 0.005f;  ///< m, per-frame translation sigma floor
    float odo_t_frac  = 0.10f;   ///< sigma_t += frac * |translation|
    float odo_r_floor = 0.0035f; ///< rad (0.2°), per-frame rotation sigma floor
    float odo_r_frac  = 0.05f;   ///< sigma_r += frac * |rotation|
    float odo_r_per_m = 0.01f;   ///< rad per metre travelled
    int   warm_frames = 20;      ///< odometry only while the map is still a single view
    int   iters       = 15;
};

class ScanSlam
{
public:
    explicit ScanSlam(Params p = {}) : p_(p) { reset(); }

    void reset()
    {
        n_ = static_cast<int>(std::ceil(2.f * p_.half_span / p_.cell));
        o_ = -p_.half_span;
        hits_.assign(static_cast<size_t>(n_) * n_, 0);
        D_.assign(hits_.size(), 0.f);
        dirty_ = true; any_surface_ = false;
        pose_ = Eigen::Vector3f::Zero();
        cov_  = Eigen::Matrix3f::Identity() * 1e-4f;
        frames_ = 0;
    }
    void set_pose(const Eigen::Vector3f& pose) { pose_ = pose; }

    /// One frame: map-frame odometry delta (dx, dy, dtheta) since the previous call, and the scan's returns in the
    /// ROBOT frame (xy). Returns the registered pose; posterior covariance via covariance().
    Eigen::Vector3f step(const Eigen::Vector3f& odom_delta, const std::vector<Eigen::Vector2f>& pts)
    {
        const Eigen::Vector3f pred = pose_ + odom_delta;
        const float dt = odom_delta.head<2>().norm(), dr = std::abs(odom_delta.z());
        const float st = p_.odo_t_floor + p_.odo_t_frac * dt;
        const float sr = p_.odo_r_floor + p_.odo_r_frac * dr + p_.odo_r_per_m * dt;
        const Eigen::Vector3f sig(st, st, sr);
        if (frames_ < p_.warm_frames or not any_surface_ or pts.size() < 10)
        {
            pose_ = pred;
            cov_ = sig.cwiseAbs2().asDiagonal();
        }
        else
            match(pred, sig, pts);
        insert(pts);
        ++frames_;
        return pose_;
    }

    [[nodiscard]] const Eigen::Vector3f& pose() const { return pose_; }
    [[nodiscard]] const Eigen::Matrix3f& covariance() const { return cov_; }
    [[nodiscard]] const Params& params() const { return p_; }

private:
    Params p_;
    int n_ = 0;
    float o_ = 0.f;
    std::vector<std::uint16_t> hits_;
    std::vector<float> D_;
    bool dirty_ = true, any_surface_ = false;
    Eigen::Vector3f pose_ = Eigen::Vector3f::Zero();
    Eigen::Matrix3f cov_ = Eigen::Matrix3f::Identity();
    long frames_ = 0;

    [[nodiscard]] size_t idx(int i, int j) const { return static_cast<size_t>(j) * n_ + i; }

    void insert(const std::vector<Eigen::Vector2f>& pts)
    {
        const float c = std::cos(pose_.z()), s = std::sin(pose_.z());
        for (const auto& q : pts)
        {
            const float x = c * q.x() - s * q.y() + pose_.x(), y = s * q.x() + c * q.y() + pose_.y();
            const int i = static_cast<int>(std::floor((x - o_) / p_.cell)), j = static_cast<int>(std::floor((y - o_) / p_.cell));
            if (i < 0 or j < 0 or i >= n_ or j >= n_) continue;
            auto& h = hits_[idx(i, j)];
            if (h < std::numeric_limits<std::uint16_t>::max()) ++h;
            if (h == p_.surface_hits) { dirty_ = true; any_surface_ = true; }   // a NEW surface cell: field changes
        }
    }

    // Exact squared Euclidean distance transform (Felzenszwalb & Huttenlocher), 1-D pass.
    static void edt1d(const float* f, float* d, int n, std::vector<int>& v, std::vector<float>& z)
    {
        constexpr float INF = 1e20f;
        const auto sect = [&](int q, int r)
        { return ((f[q] + float(q) * q) - (f[r] + float(r) * r)) / (2.f * float(q) - 2.f * float(r)); };
        int k = 0; v[0] = 0; z[0] = -INF; z[1] = INF;
        for (int q = 1; q < n; ++q)
        {
            float s = sect(q, v[k]);
            while (s <= z[k]) { --k; s = sect(q, v[k]); }   // z[0] = -INF stops it at k = 0
            ++k; v[k] = q; z[k] = s; z[k + 1] = INF;
        }
        k = 0;
        for (int q = 0; q < n; ++q)
        {
            while (z[k + 1] < float(q)) ++k;
            d[q] = (float(q) - v[k]) * (float(q) - v[k]) + f[v[k]];
        }
    }
    void rebuild()
    {
        constexpr float INF = 1e20f;
        std::vector<float> f(static_cast<size_t>(n_)), d(static_cast<size_t>(n_)), z(static_cast<size_t>(n_) + 1);
        std::vector<int> v(static_cast<size_t>(n_));
        for (size_t k = 0; k < hits_.size(); ++k) D_[k] = hits_[k] >= p_.surface_hits ? 0.f : INF;
        for (int j = 0; j < n_; ++j)            // rows
        {
            for (int i = 0; i < n_; ++i) f[i] = D_[idx(i, j)];
            edt1d(f.data(), d.data(), n_, v, z);
            for (int i = 0; i < n_; ++i) D_[idx(i, j)] = d[i];
        }
        for (int i = 0; i < n_; ++i)            // columns
        {
            for (int j = 0; j < n_; ++j) f[j] = D_[idx(i, j)];
            edt1d(f.data(), d.data(), n_, v, z);
            for (int j = 0; j < n_; ++j) D_[idx(i, j)] = std::sqrt(d[j]) * p_.cell;
        }
        dirty_ = false;
    }
    // Bilinear sample of D at cell-centre coordinates, with its gradient (m/m). Clamped at the border.
    void sample(float x, float y, float& d, float& gx, float& gy) const
    {
        const float u = std::clamp((x - o_) / p_.cell - 0.5f, 0.f, float(n_ - 1) - 1e-3f);
        const float w = std::clamp((y - o_) / p_.cell - 0.5f, 0.f, float(n_ - 1) - 1e-3f);
        const int i = static_cast<int>(u), j = static_cast<int>(w);
        const float a = u - i, b = w - j;
        const float d00 = D_[idx(i, j)], d10 = D_[idx(i + 1, j)], d01 = D_[idx(i, j + 1)], d11 = D_[idx(i + 1, j + 1)];
        d  = (1 - a) * (1 - b) * d00 + a * (1 - b) * d10 + (1 - a) * b * d01 + a * b * d11;
        gx = ((1 - b) * (d10 - d00) + b * (d11 - d01)) / p_.cell;
        gy = ((1 - a) * (d01 - d00) + a * (d11 - d10)) / p_.cell;
    }
    void match(const Eigen::Vector3f& pred, const Eigen::Vector3f& sig, const std::vector<Eigen::Vector2f>& pts)
    {
        if (dirty_) rebuild();
        const Eigen::Matrix3f W0 = sig.cwiseAbs2().cwiseInverse().asDiagonal();
        const float wl0 = 1.f / (p_.sigma_lidar * p_.sigma_lidar);
        Eigen::Vector3f p = pred;
        Eigen::Matrix3f H = W0;
        for (int it = 0; it < p_.iters; ++it)
        {
            const float c = std::cos(p.z()), s = std::sin(p.z());
            Eigen::Matrix3f Hs = Eigen::Matrix3f::Zero();
            Eigen::Vector3f bs = Eigen::Vector3f::Zero();
            for (const auto& q : pts)
            {
                const float x = c * q.x() - s * q.y() + p.x(), y = s * q.x() + c * q.y() + p.y();
                float d, gx, gy;
                sample(x, y, d, gx, gy);
                const float r = d / p_.cauchy_m;
                const float w = wl0 / (1.f + r * r);                       // Cauchy IRLS weight
                const Eigen::Vector3f J(gx, gy, gx * (-s * q.x() - c * q.y()) + gy * (c * q.x() - s * q.y()));
                Hs.noalias() += w * J * J.transpose();
                bs.noalias() += w * J * d;
            }
            const Eigen::Vector3f e = p - pred;
            const float m2 = e.dot(W0 * e);
            const Eigen::Matrix3f W = ((p_.nu + 3.f) / (p_.nu + m2)) * W0;  // Student-t IRLS weight
            H = Hs + W;
            const Eigen::Vector3f dp = -H.ldlt().solve(bs + W * e);
            if (not dp.allFinite()) break;
            p += dp;
            if (dp.head<2>().cwiseAbs().maxCoeff() < 1e-4f and std::abs(dp.z()) < 1e-5f) break;
        }
        pose_ = p;
        const Eigen::Matrix3f C = H.inverse();
        if (C.allFinite()) cov_ = C;
    }
};
} // namespace rc::slam
