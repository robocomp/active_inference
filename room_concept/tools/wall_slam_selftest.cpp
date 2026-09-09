/*
 *  wall_slam_selftest.cpp — offline validation of the wall-SLAM layer.
 *
 *  A known L-shaped Manhattan room, a robot driving a loop inside it, ray-cast LiDAR with noise and
 *  noisy odometry. Everything the agent would do with that data — segment, associate, solve the pose
 *  window jointly with the wall landmarks, absorb dropped slots, derive the polygon — runs here
 *  against an exact truth. Build:
 *      make -C build wall_slam_selftest && ./bin/wall_slam_selftest
 *
 *  Sections:
 *    1. Segmenter: recall, sign convention, observed corners on one scan.
 *    2. Jacobians of the new factor families (wall points, room↔wall, priors, gauge) vs central
 *       differences, in isolation and together, including the saturated Huber branch.
 *    3. The full loop: does the polygon close, how far is it from the truth (Hausdorff, per-wall
 *       offset), and how far the poses drifted.
 *    4. A chamfer: a 45° wall must be born as "no class" (k = −1) and still close the polygon.
 *    5. Structure change after closure: two walls appear (a notch); the map must bear them.
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <queue>
#include <random>
#include <charconv>
#include <cstring>
#include <unordered_map>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <fstream>
#include <locale>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "corner_visibility.h"
#include "room_concept.h"
#include "room_gn_solver.h"
#include "room_model.h"
#include "wall_map.h"
#include "wall_segmenter.h"

using rc::RoomConcept;
using Poly = std::vector<Eigen::Vector2f>;

namespace
{
    constexpr float kPi = static_cast<float>(M_PI);

    int failures = 0;
    void check(const char* what, bool ok, const std::string& detail)
    {
        std::printf("  %-52s %s   %s\n", what, ok ? "PASS" : "FAIL", detail.c_str());
        if (not ok) ++failures;
    }
    std::string fmt(const char* f, auto... a)
    {
        char buf[512];
        std::snprintf(buf, sizeof buf, f, a...);
        return buf;
    }

    // ── Rooms (CCW) ──────────────────────────────────────────────────────────────────────────────

    // Bench-only overrides of the structural counts the code length cannot derive, for sweeps.
    static void apply_env_overrides(rc::wallmap::Params& p)
    {
        const auto envf = [](const char* name, auto& dst)
        {
            if (const char* e = std::getenv(name))
            {
                float v = 0.f;
                const auto r = std::from_chars(e, e + std::strlen(e), v);
                if (r.ec == std::errc{}) dst = static_cast<std::remove_reference_t<decltype(dst)>>(v);
            }
        };
        envf("WS_REPLACE_EDGES", p.replace_code_edges);   // lines a replacement names (default 1)
        envf("WS_WRAP_EDGES",    p.wrap_code_edges);      // lines a spur wrap names (default 2)
        envf("WS_KEEP",          p.order_keep_fraction);  // down-jump refund fraction (default 0.7)
        envf("WS_ADOPT_JUDGE",   p.adopt_judge);          // 0 incumbent (IoU margin + veto), 1 one energy
        envf("WS_ADOPT_REPAIR",  p.adopt_repair);         // repair self-crossing cycles before judging
        envf("WS_MANHATTAN_GAIN", p.manhattan_gain);      // scale on the in-loop Manhattan factor (#4 test)
        envf("WS_LEVEL2",        p.enable_level2);        // level-2 residual pass on the published copy
        envf("WS_LEVEL2_FIT",    p.level2_fit);            // fit the step's 3 DoF to the returns
        envf("WS_L2_MIN",        p.level2_min_m);         // smallest feature level 2 keeps (m)
        envf("WS_L2_CLEAR",      p.level2_clear_cells);    // residual-cell clearance from every edge (cells)
        envf("WS_THETA0_POST",   p.theta0_posterior);     // 1 posterior over all directions, 0 the old OBB+mean
        envf("WS_WAIVER_PRICED", p.adopt_waiver_priced);  // 1 the health waiver pays its code length, 0 free
        envf("WS_SPLICE_SURR",   p.splice_surrender);     // 1 a splice pays the existence support it erases
        envf("WS_SIGMA_K",       p.adopt_sigma_k);        // judge 3: margin in standard deviations of dE
        envf("WS_TRIAL",         p.trial_adoption);       // adopt a refused-but-promising cycle ON TRIAL
        envf("WS_TRIAL_FRAMES",  p.trial_frames);         // how long the challenger has to prove itself
        envf("WS_SURR_EXTENT",   p.surrender_by_extent);  // price support on the extent dropped, not just vanished lines
    }

    Poly l_room()      { return {{-4.f, -3.f}, {4.f, -3.f}, {4.f, 1.f}, {1.f, 1.f}, {1.f, 3.f}, {-4.f, 3.f}}; }
    Poly l_room_notch(){ return {{-4.f, -3.f}, {4.f, -3.f}, {4.f, 1.f}, {1.f, 1.f}, {1.f, 3.f}, {-2.f, 3.f}, {-2.f, 2.f}, {-4.f, 2.f}}; }
    Poly chamfer_room(){ return {{-4.f, -3.f}, {4.f, -3.f}, {4.f, 2.f}, {3.f, 3.f}, {-4.f, 3.f}}; }

    /// Ray-cast the polygon from `pose`; robot-frame 2-D points with perpendicular-ish range noise.
    std::vector<Eigen::Vector2f> scan(const Poly& room, const Eigen::Vector3f& pose, int n, float sigma,
                                      std::mt19937& rng)
    {
        std::normal_distribution<float> noise(0.f, sigma);
        std::vector<Eigen::Vector2f> out;
        const int N = static_cast<int>(room.size());
        for (int i = 0; i < n; ++i)
        {
            const float bearing = -kPi + 2.f * kPi * static_cast<float>(i) / static_cast<float>(n);
            const float wd = pose.z() + bearing;
            const Eigen::Vector2f d(std::cos(wd), std::sin(wd));
            float best = 1e9f;
            for (int e = 0; e < N; ++e)
                if (auto t = rc::corner_visibility::ray_segment_t(pose.head<2>(), d, room[e], room[(e + 1) % N]); t and *t < best)
                    best = *t;
            if (best > 1e8f) continue;
            const float r = best + noise(rng);
            out.emplace_back(r * std::cos(bearing), r * std::sin(bearing));
        }
        return out;
    }

    /// A loop inside the L: waypoints joined by straight legs, heading along the leg.
    std::vector<Eigen::Vector3f> trajectory(int n_per_leg)
    {
        const std::vector<Eigen::Vector2f> wp = {{-2.5f, -1.5f}, {2.5f, -1.5f}, {2.5f, -0.5f}, {-0.5f, -0.5f},
                                                 {-0.5f, 1.5f}, {-2.5f, 1.5f}, {-2.5f, -1.5f}};
        std::vector<Eigen::Vector3f> tr;
        for (size_t l = 0; l + 1 < wp.size(); ++l)
        {
            const Eigen::Vector2f e = wp[l + 1] - wp[l];
            const float th = std::atan2(e.y(), e.x());
            for (int i = 0; i < n_per_leg; ++i)
            {
                const float a = static_cast<float>(i) / static_cast<float>(n_per_leg);
                tr.emplace_back(wp[l].x() + a * e.x(), wp[l].y() + a * e.y(), th);
            }
        }
        return tr;
    }

    torch::Tensor pose_tensor(const Eigen::Vector3f& p)
    {
        return torch::tensor({p.x(), p.y(), p.z()}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
    }
    torch::Tensor points_tensor(const std::vector<Eigen::Vector2f>& pts)
    {
        auto t = torch::zeros({static_cast<long>(pts.size()), 3}, torch::kFloat32);
        auto a = t.accessor<float, 2>();
        for (size_t i = 0; i < pts.size(); ++i) { a[i][0] = pts[i].x(); a[i][1] = pts[i].y(); a[i][2] = 0.f; }
        return t;
    }
    torch::Tensor mat3(const Eigen::Matrix3f& m)
    {
        auto t = torch::zeros({3, 3}, torch::kFloat32);
        auto a = t.accessor<float, 2>();
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) a[r][c] = m(r, c);
        return t;
    }

    /// True walls of a CCW polygon in Hesse form with the normal INTO the room.
    struct TrueWall { float phi, d; Eigen::Vector2f a, b; };
    std::vector<TrueWall> true_walls(const Poly& room)
    {
        std::vector<TrueWall> w;
        const int N = static_cast<int>(room.size());
        for (int i = 0; i < N; ++i)
        {
            const Eigen::Vector2f a = room[i], b = room[(i + 1) % N];
            const Eigen::Vector2f e = (b - a).normalized();
            const Eigen::Vector2f n(-e.y(), e.x());     // left of the CCW edge = inside
            w.push_back({std::atan2(n.y(), n.x()), n.dot(a), a, b});
        }
        return w;
    }

    float wrap(float a) { return std::atan2(std::sin(a), std::cos(a)); }

    /// Hausdorff distance between two closed polygons (vertex-to-boundary, both ways).
    float point_to_segment(const Eigen::Vector2f& p, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
    {
        const Eigen::Vector2f ab = b - a;
        const float t = std::clamp((p - a).dot(ab) / std::max(1e-9f, ab.squaredNorm()), 0.f, 1.f);
        return (p - (a + t * ab)).norm();
    }
    float point_to_poly(const Eigen::Vector2f& p, const Poly& poly)
    {
        float best = 1e9f;
        for (size_t i = 0; i < poly.size(); ++i)
            best = std::min(best, point_to_segment(p, poly[i], poly[(i + 1) % poly.size()]));
        return best;
    }
    /// Rasterised intersection-over-union: the metric that PUNISHES a missing concavity (Hausdorff
    /// under-reports a deep interior wall the estimate paves over). 4 cm cells.
    float polygon_iou(const Poly& a, const Poly& b)
    {
        if (a.size() < 3 or b.size() < 3) return 0.f;
        float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
        for (const auto& P : {a, b})
            for (const auto& v : P)
            { x0 = std::min(x0, v.x()); y0 = std::min(y0, v.y()); x1 = std::max(x1, v.x()); y1 = std::max(y1, v.y()); }
        const float cell = 0.04f;
        long inter = 0, uni = 0;
        for (float x = x0 + cell * 0.5f; x < x1; x += cell)
            for (float y = y0 + cell * 0.5f; y < y1; y += cell)
            {
                const Eigen::Vector2f p(x, y);
                const bool ia = rc::corner_visibility::point_in_polygon(p, a);
                const bool ib = rc::corner_visibility::point_in_polygon(p, b);
                if (ia and ib) ++inter;
                if (ia or ib) ++uni;
            }
        return (uni > 0) ? static_cast<float>(inter) / static_cast<float>(uni) : 0.f;
    }

    float hausdorff(const Poly& a, const Poly& b)
    {
        float h = 0.f;
        for (const auto& p : a) h = std::max(h, point_to_poly(p, b));
        for (const auto& p : b) h = std::max(h, point_to_poly(p, a));
        return h;
    }

    // ── The loop the agent runs, in miniature ──────────────────────────────────────────────────
    struct RunResult
    {
        rc::wallmap::WallMap map;
        rc::wallmap::Polygon poly;
        float pose_rmse_xy = 0.f, pose_max_xy = 0.f, pose_max_th = 0.f;
        int frames = 0, closed_at = -1, births = 0, deaths = 0, rejected = 0;
        Eigen::Vector2f last_xy = Eigen::Vector2f::Zero();   // final robot position (map frame)
        long occluded_pts = 0;   // LiDAR returns the camera showed to be on furniture, not on a wall
    };

    using Boxes = std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>>;   // (lo, hi)

    struct RunConfig
    {
        int window = 5;
        int n_rays = 360;
        float scan_sigma = 0.02f;
        float odom_sigma_xy = 0.005f, odom_sigma_th = 0.3f * kPi / 180.f;
        bool verbose = false;
        Boxes occluders;                 // furniture the LiDAR cannot see through
        bool  info_gain = true;          // drive by expected information gain per metre (rank 6)
        bool  ceiling_line = false;      // add the camera's wall-to-ceiling ranges
        float ceiling_dh = 1.70f;        // ceiling height above the camera (3.0 m room, 1.30 m mount)
        float ceiling_sigma_rad = 0.0011f;   // 0.065 deg — the measured fitted-contour scatter
        int   ceiling_rays = 180;
        const char* trace_csv = nullptr;   // per-frame precision trace (WallMap::precisions)
        const char* poly_csv  = nullptr;   // per-frame PUBLISHED layout + its per-corner/per-edge sigma
    };

    /// Append one frame's precision snapshot. The file is opened once per run and imbued with the
    /// classic locale, so a comma decimal separator can never reach it (see CLAUDE.md).
    void trace_precisions(std::ofstream& f, const rc::wallmap::WallMap& map, float iou, float pose_err)
    {
        if (not f.is_open()) return;
        const auto p = map.precisions();
        f << p.frame << ',' << p.theta0_deg << ',' << p.theta0_disp_deg << ',' << p.theta0_gate_deg
          << ',' << p.n_walls << ',' << p.n_cand << ',' << p.n_order
          << ',' << p.sigma_phi_deg << ',' << p.sigma_d_m << ',' << p.corner_sigma_m
          << ',' << p.corner_sigma_med_m << ',' << p.corners_over_bar << ',' << p.n_corners
          << ',' << p.exist_nats << ',' << p.class_err_deg << ',' << iou << ',' << pose_err << '\n';
    }
    void trace_open(std::ofstream& f, const char* path)
    {
        if (path == nullptr) return;
        const bool fresh = not std::filesystem::exists(path);
        f.open(path, std::ios::app);
        if (not f.is_open()) return;
        f.imbue(std::locale::classic());
        if (not fresh) return;
        f << "frame,theta0_deg,theta0_disp_deg,theta0_gate_deg,walls,cand,order,"
             "sigma_phi_deg,sigma_d_m,corner_sigma_m,corner_sigma_med_m,corners_over_bar,n_corners,"
             "exist_nats,class_err_deg,iou,pose_err\n";
    }

    /// FURNITURE. Axis-aligned boxes standing inside the room. The LiDAR band cuts through them at
    /// 1-2 m, so a wall behind one is never seen; the wall-to-ceiling junction, three metres up, is
    /// never blocked by them. That difference is the whole reason to want a ceiling line, and a
    /// bench without occluders cannot show it — the two sensors would see the same walls and the
    /// camera would only add noise.

    /// Range to the first box hit along a ray, or infinity.
    float box_range(const Eigen::Vector2f& o, const Eigen::Vector2f& d, const Boxes& boxes)
    {
        float best = std::numeric_limits<float>::infinity();
        for (const auto& [lo, hi] : boxes)
        {
            float t0 = 0.f, t1 = best;
            bool ok = true;
            for (int k = 0; k < 2 and ok; ++k)
            {
                if (std::abs(d[k]) < 1e-9f) { ok = (o[k] >= lo[k] and o[k] <= hi[k]); continue; }
                float ta = (lo[k] - o[k]) / d[k], tb = (hi[k] - o[k]) / d[k];
                if (ta > tb) std::swap(ta, tb);
                t0 = std::max(t0, ta); t1 = std::min(t1, tb);
                ok = t0 <= t1;
            }
            if (ok and t0 > 1e-3f) best = std::min(best, t0);
        }
        return best;
    }

    /// The LiDAR band, with furniture in the way: the nearer of the wall and the box.
    std::vector<Eigen::Vector2f> scan_occluded(const Poly& room, const Boxes& boxes,
                                               const Eigen::Vector3f& pose, int n, float sigma,
                                               std::mt19937& rng)
    {
        std::normal_distribution<float> noise(0.f, sigma);
        std::vector<Eigen::Vector2f> out;
        const int N = static_cast<int>(room.size());
        for (int i = 0; i < n; ++i)
        {
            const float bearing = -kPi + 2.f * kPi * static_cast<float>(i) / static_cast<float>(n);
            const float wd = pose.z() + bearing;
            const Eigen::Vector2f d(std::cos(wd), std::sin(wd));
            float best = 1e9f;
            for (int e = 0; e < N; ++e)
                if (auto t = rc::corner_visibility::ray_segment_t(pose.head<2>(), d, room[e], room[(e + 1) % N]); t and *t < best)
                    best = *t;
            best = std::min(best, box_range(pose.head<2>(), d, boxes));
            if (best > 1e8f) continue;
            const float r = best + noise(rng);
            out.emplace_back(r * std::cos(bearing), r * std::sin(bearing));
        }
        return out;
    }

    /// THE WALL-TO-CEILING LINE, as the 360 camera would deliver it. For each azimuth the junction is
    /// seen at elevation atan((h_ceil - h_cam) / r); the camera measures that ANGLE, with a scatter
    /// set by how well the fitted contour is localised in rows, and the range follows from the
    /// ceiling height. So the error grows with range as (dh^2 + r^2)/dh per radian — the reason this
    /// is a poor per-column range and a good long-baseline one. Furniture cannot block it.
    std::vector<Eigen::Vector2f> scan_ceiling(const Poly& room, const Eigen::Vector3f& pose, int n,
                                              float dh, float sigma_rad, std::mt19937& rng)
    {
        std::normal_distribution<float> noise(0.f, sigma_rad);
        std::vector<Eigen::Vector2f> out;
        const int N = static_cast<int>(room.size());
        for (int i = 0; i < n; ++i)
        {
            const float bearing = -kPi + 2.f * kPi * static_cast<float>(i) / static_cast<float>(n);
            const float wd = pose.z() + bearing;
            const Eigen::Vector2f d(std::cos(wd), std::sin(wd));
            float best = 1e9f;
            for (int e = 0; e < N; ++e)
                if (auto t = rc::corner_visibility::ray_segment_t(pose.head<2>(), d, room[e], room[(e + 1) % N]); t and *t < best)
                    best = *t;
            if (best > 1e8f) continue;
            const float alpha = std::atan2(dh, best) + noise(rng);
            if (alpha < 0.02f) continue;                    // grazing: the junction is not resolvable
            const float r = dh / std::tan(alpha);
            if (not std::isfinite(r) or r < 0.2f or r > 20.f) continue;
            out.emplace_back(r * std::cos(bearing), r * std::sin(bearing));
        }
        return out;
    }

    /// ONE SWEEP OF BOTH SENSORS, on a shared bearing grid, with the disagreement between them used
    /// as evidence rather than discarded. The LiDAR stops at the nearest thing on the ray, furniture
    /// included; the ceiling junction is above the furniture and stops at the wall. So when the two
    /// disagree by more than their combined noise on the SAME bearing, that is not conflict to be
    /// averaged away — it is the signature of an occluder, and it says the near return is not a
    /// wall. Those LiDAR points are dropped from the wall cloud. Without this the two sensors hand
    /// the segmenter a furniture face and the true wall behind it competing for one stretch of
    /// boundary, with nothing in the model to explain the near one away.
    std::vector<Eigen::Vector2f> scan_fused(const Poly& room, const RunConfig& cfg,
                                            const Eigen::Vector3f& pose, std::mt19937& rng,
                                            Eigen::VectorXf& weights, int& n_occluded)
    {
        std::normal_distribution<float> rn(0.f, cfg.scan_sigma), an(0.f, cfg.ceiling_sigma_rad);
        const int N = static_cast<int>(room.size());
        const int n = cfg.n_rays;
        std::vector<Eigen::Vector2f> out;
        std::vector<float> w;
        n_occluded = 0;
        const int ceil_every = cfg.ceiling_line ? std::max(1, n / std::max(1, cfg.ceiling_rays)) : 0;
        for (int i = 0; i < n; ++i)
        {
            const float bearing = -kPi + 2.f * kPi * static_cast<float>(i) / static_cast<float>(n);
            const float wd = pose.z() + bearing;
            const Eigen::Vector2f d(std::cos(wd), std::sin(wd));
            float wall = 1e9f;
            for (int e = 0; e < N; ++e)
                if (auto t = rc::corner_visibility::ray_segment_t(pose.head<2>(), d, room[e], room[(e + 1) % N]); t and *t < wall)
                    wall = *t;
            if (wall > 1e8f) continue;
            const float occ = std::min(wall, box_range(pose.head<2>(), d, cfg.occluders));
            const float r_lidar = occ + rn(rng);
            // The camera, on the bearings it samples.
            bool have_ceiling = false; float r_ceiling = 0.f, s_ceiling = 0.f;
            if (cfg.ceiling_line and (i % ceil_every) == 0)
            {
                const float alpha = std::atan2(cfg.ceiling_dh, wall) + an(rng);
                if (alpha > 0.02f)
                {
                    const float rr = cfg.ceiling_dh / std::tan(alpha);
                    if (std::isfinite(rr) and rr > 0.2f and rr < 20.f)
                    {
                        have_ceiling = true; r_ceiling = rr;
                        s_ceiling = (cfg.ceiling_dh * cfg.ceiling_dh + rr * rr) / cfg.ceiling_dh * cfg.ceiling_sigma_rad;
                    }
                }
            }
            // The occluder test: the camera sees FURTHER than the LiDAR by more than the two of them
            // can disagree by chance.
            const bool occluded = have_ceiling
                and (r_ceiling - r_lidar) > 3.f * std::sqrt(cfg.scan_sigma * cfg.scan_sigma + s_ceiling * s_ceiling);
            if (occluded) ++n_occluded;
            else { out.emplace_back(r_lidar * std::cos(bearing), r_lidar * std::sin(bearing)); w.push_back(1.f); }
            if (have_ceiling)
            {
                out.emplace_back(r_ceiling * std::cos(bearing), r_ceiling * std::sin(bearing));
                w.push_back(std::min(1.f, (cfg.scan_sigma * cfg.scan_sigma) / std::max(s_ceiling * s_ceiling, 1e-9f)));
            }
        }
        weights = Eigen::VectorXf::Zero(static_cast<long>(w.size()));
        for (size_t k = 0; k < w.size(); ++k) weights[static_cast<long>(k)] = w[k];
        return out;
    }

    RunResult run_loop(const std::vector<Poly>& rooms_by_frame, const std::vector<Eigen::Vector3f>& truth,
                       const RunConfig& cfg, std::mt19937& rng, RunResult* resume = nullptr)
    {
        RunResult R;
        if (resume) R = *resume;
        rc::wallseg::Params sp;
        const auto obb_rect = [](const std::vector<Eigen::Vector2f>& pts) -> Poly
        {
            Eigen::Vector2f mu = Eigen::Vector2f::Zero();
            for (const auto& p : pts) mu += p;
            mu /= static_cast<float>(pts.size());
            Eigen::Matrix2f C = Eigen::Matrix2f::Zero();
            for (const auto& p : pts) { const Eigen::Vector2f d = p - mu; C += d * d.transpose(); }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eig(C);
            const Eigen::Vector2f ax = eig.eigenvectors().col(1), ay = eig.eigenvectors().col(0);
            float lo0 = 1e9f, hi0 = -1e9f, lo1 = 1e9f, hi1 = -1e9f;
            for (const auto& p : pts)
            {
                const float u = ax.dot(p - mu), v = ay.dot(p - mu);
                lo0 = std::min(lo0, u); hi0 = std::max(hi0, u);
                lo1 = std::min(lo1, v); hi1 = std::max(hi1, v);
            }
            // CCW for right-handed (ax, ay)
            return {mu + ax * lo0 + ay * lo1, mu + ax * hi0 + ay * lo1,
                    mu + ax * hi0 + ay * hi1, mu + ax * lo0 + ay * hi1};
        };
        sp.sensor_sigma = cfg.scan_sigma;
        R.map.params.obs_sigma = 0.05f;
        R.map.params.huber_delta = 0.15f;
        R.map.params.debug_splice = std::getenv("WS_DEBUG_SPLICE") != nullptr;
        apply_env_overrides(R.map.params);
        // The forward-model referee scans every stored beam per judged decision: minutes per seed
        // under churn against 13 s for the whole bench without it. Opt in with WS_REFEREE=1.
        R.map.params.forward_referee = std::getenv("WS_NO_BEAMS") == nullptr;
        R.map.params.referee_log     = std::getenv("WS_REFEREE") != nullptr;
        std::ofstream trace; trace_open(trace, cfg.trace_csv);

        rc::Model model;
        model.init_from_polygon({{-20.f, -20.f}, {20.f, -20.f}, {20.f, 20.f}, {-20.f, 20.f}}, 0.f, 0.f, 0.f, 2.4f);
        RoomConcept::Params params;
        params.rfe_obs_sigma = 0.05f;
        params.rfe_huber_delta = 0.15f;
        params.enable_corner_tracking = false;
        params.object_anchor.enable = false;
        params.image_edge.enable = false;

        std::deque<RoomConcept::WindowSlot> window;
        RoomConcept::BoundaryPrior bp;   // invalid until the first drop ⇒ gauge factor pins slot 0
        std::normal_distribution<float> nxy(0.f, cfg.odom_sigma_xy), nth(0.f, cfg.odom_sigma_th);

        // The estimate lives in the MAP frame, which is the first pose's frame: truth is expressed
        // relative to truth[0] so the two are directly comparable.
        const Eigen::Vector3f origin = truth[0];
        const auto to_map = [&](const Eigen::Vector3f& p)
        {
            const float c = std::cos(-origin.z()), s = std::sin(-origin.z());
            const Eigen::Vector2f dxy = p.head<2>() - origin.head<2>();
            return Eigen::Vector3f(c * dxy.x() - s * dxy.y(), s * dxy.x() + c * dxy.y(), wrap(p.z() - origin.z()));
        };

        Eigen::Vector3f est = Eigen::Vector3f::Zero();
        Eigen::Vector3f prev_truth_map = to_map(truth[0]);
        if (resume) est = resume->map.walls.empty() ? est : est;   // (resume keeps the map; poses restart at the origin of the new leg)
        double se = 0.0; int n_err = 0;

        for (size_t f = 0; f < truth.size(); ++f)
        {
            const Poly& room = rooms_by_frame[std::min(f, rooms_by_frame.size() - 1)];
            const Eigen::Vector3f tm = to_map(truth[f]);
            // Odometry in the map frame with noise; prediction = previous estimate + odom.
            Eigen::Vector3f odom = tm - prev_truth_map; odom.z() = wrap(odom.z());
            if (f > 0) odom += Eigen::Vector3f(nxy(rng), nxy(rng), nth(rng));
            prev_truth_map = tm;
            const Eigen::Vector3f pred = (f == 0) ? Eigen::Vector3f::Zero()
                                                  : Eigen::Vector3f(est.x() + odom.x(), est.y() + odom.y(), wrap(est.z() + odom.z()));

            const auto pts = scan(room, truth[f], cfg.n_rays, cfg.scan_sigma, rng);
            // MODEL-FIRST: the very first scan's OBB seeds the rectangle (map frame = first pose).
            if (R.map.walls.empty())
            {
                Poly rect = obb_rect(pts);
                // Ensure CCW (positive area)
                float a2 = 0.f;
                for (size_t i = 0; i < rect.size(); ++i)
                { const auto& p = rect[i]; const auto& q = rect[(i + 1) % rect.size()]; a2 += p.x() * q.y() - q.x() * p.y(); }
                if (a2 < 0.f) std::reverse(rect.begin(), rect.end());
                R.map.initialize_rect(rect);
            }
            const auto seg = rc::wallseg::segment(pts, sp, rng);
            const Eigen::Matrix3f pcov = Eigen::Vector3f(0.05f * 0.05f, 0.05f * 0.05f, 0.03f * 0.03f).asDiagonal();
            const auto fr = R.map.observe(seg, pts, Eigen::VectorXf{}, pred, pcov, static_cast<std::int64_t>(f) * 50);
            R.births += fr.births;
            R.deaths += fr.deaths;
            R.rejected += fr.splice_rejected;

            RoomConcept::WindowSlot slot;
            slot.pose = pose_tensor(pred);
            slot.lidar_points = points_tensor(pts);
            slot.odometry_delta = (f == 0) ? Eigen::Vector3f::Zero() : odom;
            slot.motion_cov = Eigen::Vector3f(cfg.odom_sigma_xy * cfg.odom_sigma_xy, cfg.odom_sigma_xy * cfg.odom_sigma_xy,
                                              cfg.odom_sigma_th * cfg.odom_sigma_th).asDiagonal();
            slot.odom_delta_tensor = torch::tensor({odom.x(), odom.y(), odom.z()}, torch::kFloat32);
            slot.motion_prec_tensor = mat3(slot.motion_cov.inverse());
            slot.wall_assoc = fr.assoc;

            rc::gn::Input in;
            in.model = &model; in.params = &params; in.window = &window; in.boundary_prior = &bp;
            in.device = torch::kCPU;
            in.walls = &R.map; in.no_sdf = true; in.gauge_fix = true;

            if (static_cast<int>(window.size()) >= cfg.window)
            {
                // Drop the oldest: fold its wall observations into the map first (at its converged pose),
                // then anchor the new front with a plain prior at its converged pose (the legacy
                // boundary anchor; the agent uses FEJ+Schur).
                auto front_pose = window.front().pose.detach();
                const Eigen::Vector3f fp(front_pose[0].item<float>(), front_pose[1].item<float>(), front_pose[2].item<float>());
                rc::gn::absorb_wall_observations(in, window.front(), fp);
                window.pop_front();
                auto nf = window.front().pose.detach();
                bp.valid = true;
                bp.mu = Eigen::Vector3f(nf[0].item<float>(), nf[1].item<float>(), nf[2].item<float>());
                bp.precision = Eigen::Vector3f(400.f, 400.f, 1600.f).asDiagonal();
            }
            window.push_back(std::move(slot));

            std::vector<Eigen::Vector3f> poses;
            for (const auto& s : window)
            {
                auto p = s.pose.detach();
                poses.emplace_back(p[0].item<float>(), p[1].item<float>(), p[2].item<float>());
            }
            rc::gn::Options opts;
            const auto r = rc::gn::solve(in, poses, opts);
            if (r.ok)
                for (size_t i = 0; i < window.size(); ++i) window[i].pose = pose_tensor(poses[i]);
            est = poses.back();
            R.map.merge_indistinguishable();

            const Eigen::Vector3f err = est - tm;
            const float exy = err.head<2>().norm();
            se += exy * exy; ++n_err;
            R.pose_max_xy = std::max(R.pose_max_xy, exy);
            R.pose_max_th = std::max(R.pose_max_th, std::abs(wrap(err.z())));

            const auto poly = R.map.build_polygon();
            if (poly.closed and R.closed_at < 0) R.closed_at = static_cast<int>(f);
            if (cfg.verbose and (f % 10 == 0 or fr.births > 0))
                std::printf("    f=%3zu segs=%2zu walls=%2zu cand=%2d births=%d solve=%s loss=%.3f it=%d err=%.3fm/%.2fdeg poly=%s%s\n",
                            f, seg.segments.size(), R.map.walls.size(), fr.candidates, fr.births,
                            r.ok ? "ok" : "FAIL", r.loss, r.iterations, exy, std::abs(wrap(err.z())) * 180.f / kPi,
                            poly.closed ? "closed" : "open", poly.status.empty() ? "" : (" [" + poly.status + "]").c_str());
            trace_precisions(trace, R.map, 0.f, exy);
            R.frames = static_cast<int>(f) + 1;
        }
        R.pose_rmse_xy = (n_err > 0) ? static_cast<float>(std::sqrt(se / n_err)) : 0.f;
        R.poly = R.map.manhattan_polygon();   // published layout: exactly Manhattan
        return R;
    }


    // ── Epistemic exploration: the robot is DRIVEN BY WHAT THE MODEL DOES NOT KNOW ─────────────
    // No scripted tour: after each solve the explorer collects the map's unknowns — existence bins
    // not yet solid, homeless candidates, corners above the publish bar — scores reachable
    // viewpoints by how much unknown they see, and A*-walks to the best. It stops when the model
    // has nothing left to ask (publishable and no unknowns) or at the frame cap. Physical
    // feasibility (collision, visibility) uses the TRUTH polygon — the simulator's job; the
    // TARGETS come only from the estimator's own uncertainty, as they must.
    struct Explorer
    {
        const Poly& room;
        float cell = 0.30f, clearance = 0.35f;
        float x0, y0; int nx, ny;
        std::vector<char> free_;
        explicit Explorer(const Poly& r) : room(r)
        {
            float xa = 1e9f, ya = 1e9f, xb = -1e9f, yb = -1e9f;
            for (const auto& v : room) { xa = std::min(xa, v.x()); ya = std::min(ya, v.y()); xb = std::max(xb, v.x()); yb = std::max(yb, v.y()); }
            x0 = xa; y0 = ya;
            nx = static_cast<int>((xb - xa) / cell) + 1;
            ny = static_cast<int>((yb - ya) / cell) + 1;
            free_.assign(static_cast<size_t>(nx * ny), 0);
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                {
                    const Eigen::Vector2f p = at(i, j);
                    if (rc::corner_visibility::point_in_polygon(p, room) and point_to_poly(p, room) > clearance)
                        free_[static_cast<size_t>(j * nx + i)] = 1;
                }
        }
        Eigen::Vector2f at(int i, int j) const { return {x0 + (i + 0.5f) * cell, y0 + (j + 0.5f) * cell}; }
        bool is_free(int i, int j) const
        { return i >= 0 and i < nx and j >= 0 and j < ny and free_[static_cast<size_t>(j * nx + i)] != 0; }
        std::pair<int,int> cell_of(const Eigen::Vector2f& p) const
        { return {static_cast<int>((p.x() - x0) / cell), static_cast<int>((p.y() - y0) / cell)}; }
        bool sees(const Eigen::Vector2f& from, const Eigen::Vector2f& to) const
        {
            const int N = static_cast<int>(room.size());
            const Eigen::Vector2f d = to - from;
            const float L = d.norm();
            if (L < 1e-3f or L > 8.f) return L <= 8.f;
            const Eigen::Vector2f dir = d / L;
            for (int e = 0; e < N; ++e)
                if (auto t = rc::corner_visibility::ray_segment_t(from, dir, room[static_cast<size_t>(e)], room[static_cast<size_t>((e + 1) % N)]);
                    t and *t < L - 0.15f)
                    return false;
            return true;
        }
        std::vector<Eigen::Vector2f> astar(const Eigen::Vector2f& from, const Eigen::Vector2f& to) const
        {
            auto [si, sj] = cell_of(from);
            auto [ti, tj] = cell_of(to);
            if (not is_free(ti, tj) or not is_free(si, sj)) return {};
            const int n = nx * ny;
            std::vector<float> g(static_cast<size_t>(n), 1e18f);
            std::vector<int> par(static_cast<size_t>(n), -1);
            auto idx = [&](int i, int j) { return j * nx + i; };
            auto h = [&](int i, int j) { return std::hypot(static_cast<float>(i - ti), static_cast<float>(j - tj)); };
            using QN = std::pair<float, int>;
            std::priority_queue<QN, std::vector<QN>, std::greater<>> q;
            g[static_cast<size_t>(idx(si, sj))] = 0.f;
            q.push({h(si, sj), idx(si, sj)});
            const int di[8] = {1,-1,0,0,1,1,-1,-1}, dj[8] = {0,0,1,-1,1,-1,1,-1};
            while (not q.empty())
            {
                const auto [f, u] = q.top(); q.pop();
                const int ui = u % nx, uj = u / nx;
                if (ui == ti and uj == tj) break;
                if (f > g[static_cast<size_t>(u)] + h(ui, uj) + 1e-4f) continue;
                for (int k = 0; k < 8; ++k)
                {
                    const int vi = ui + di[k], vj = uj + dj[k];
                    if (not is_free(vi, vj)) continue;
                    if (k >= 4 and (not is_free(ui, vj) or not is_free(vi, uj))) continue;  // no corner cutting
                    const float w = (k < 4) ? 1.f : 1.41421f;
                    if (g[static_cast<size_t>(u)] + w < g[static_cast<size_t>(idx(vi, vj))])
                    {
                        g[static_cast<size_t>(idx(vi, vj))] = g[static_cast<size_t>(u)] + w;
                        par[static_cast<size_t>(idx(vi, vj))] = u;
                        q.push({g[static_cast<size_t>(idx(vi, vj))] + h(vi, vj), idx(vi, vj)});
                    }
                }
            }
            if (par[static_cast<size_t>(idx(ti, tj))] < 0 and not (si == ti and sj == tj)) return {};
            std::vector<Eigen::Vector2f> path;
            for (int u = idx(ti, tj); u >= 0; u = par[static_cast<size_t>(u)])
            { path.push_back(at(u % nx, u / nx)); if (u == idx(si, sj)) break; }
            std::reverse(path.begin(), path.end());
            return path;
        }
    };

    struct Unknown { Eigen::Vector2f p; float w; };
    /// Binary entropy of a variable held at these log-odds, in nats. Zero when the map is certain
    /// either way, ln 2 at total ignorance. This is the quantity an information-gain explorer
    /// maximises, and it is what the hand-set target weights below were standing in for.
    float entropy_nats(float lodds)
    {
        const float pp = std::clamp(1.f / (1.f + std::exp(-lodds)), 1e-6f, 1.f - 1e-6f);
        return -pp * std::log(pp) - (1.f - pp) * std::log(1.f - pp);
    }

    /// EXPECTED INFORMATION GAIN (Stachniss 2005; Julian & Karaman 2014), in place of a list of
    /// epistemic targets carrying five hand-set weights. Every uncertain thing in the map — a grid
    /// cell, an existence bin, a candidate line, a corner — contributes its OWN entropy, in the same
    /// nats, and the explorer drives to where a scan would remove the most of it per metre of
    /// travel. A cell the map is already sure about contributes nothing, so the objective terminates
    /// on its own instead of needing a separate "no frontier left" rule, and a far region full of
    /// unknown cells outweighs a near sliver without needing a coverage turn to force the issue.
    std::vector<Unknown> collect_entropy_targets(const rc::wallmap::WallMap& map)
    {
        std::vector<Unknown> out;
        // (a) THE GRID. Subsampled so the score loop stays the size it was.
        const auto poly_now = map.build_polygon();
        const auto interior_far_from_boundary = [&](const Eigen::Vector2f& q)
        {
            if (not poly_now.closed or poly_now.verts.size() < 3) return false;
            if (not rc::corner_visibility::point_in_polygon(q, poly_now.verts)) return false;
            const float clear = 2.f * map.fgrid.cell;
            for (size_t e = 0; e < poly_now.verts.size(); ++e)
            {
                const Eigen::Vector2f a = poly_now.verts[e], ab = poly_now.verts[(e + 1) % poly_now.verts.size()] - a;
                const float l2 = ab.squaredNorm();
                const float tt = l2 > 1e-9f ? std::clamp((q - a).dot(ab) / l2, 0.f, 1.f) : 0.f;
                if ((q - (a + tt * ab)).norm() < clear) return false;
            }
            return true;
        };
        if (map.fgrid.ready())
        {
            const int stride = 3;
            for (int i = 0; i < map.fgrid.nx; i += stride)
                for (int j = 0; j < map.fgrid.ny; j += stride)
                {
                    const size_t id = static_cast<size_t>(map.fgrid.idx(i, j));
                    float h = entropy_nats(map.fgrid.lodds[id]);
                    // ★ CONFLICT IS UNCERTAINTY THE LOG-ODDS CANNOT SHOW (the evidential-grid idea,
                    // Moras & Cherfaoui 2011, in miniature). The grid keeps two channels on purpose:
                    // endpoint RETURNS, which localise matter, and traversals, which are weak and
                    // explicable — a thin wall shares its cell with air. When they DISAGREE, a return
                    // says matter and the beams say free, the entropy of the log-odds alone reads
                    // that cell as settled when the map plainly does not know. Such a cell is a
                    // suspected thin wall, and it is the one thing a visit can settle outright, so it
                    // is scored at full ignorance. This is what gives a spur its interest: not a
                    // hand-set weight for "weak matter", but the admission that two channels in
                    // conflict carry no information until someone goes and looks.
                    // ...but ONLY AWAY FROM THE BOUNDARY. Contested cells are not rare: every wall
                    // surface has them, because a grazing beam disagrees with the return beside it.
                    // Promoting all of them measured as a broad reweighting rather than an interest
                    // in spurs — one more spur and two more corner columns bought with six wall
                    // columns and 0.18 m of Hausdorff. A conflict ON the boundary is the wall's own
                    // surface; a conflict INSIDE the room is a suspected free-standing or protruding
                    // structure, which is what a spur is.
                    const unsigned short hits = map.fgrid.hits[id];
                    if (hits >= 1 and hits < 3 and map.fgrid.lodds[id] < 0.f
                        and interior_far_from_boundary(map.fgrid.at(i, j)))
                        h = std::max(h, std::log(2.f));
                    if (h > 0.15f) out.push_back({map.fgrid.at(i, j), h * static_cast<float>(stride * stride)});
                }
        }
        // (b) THE EXISTENCE BINS. A bin IS a log-odds, so its entropy needs no conversion.
        for (const auto& w : map.walls)
        {
            const Eigen::Vector2f n = w.normal(), t = w.tangent();
            for (size_t b = 0; b < w.exist_bins.size(); ++b)
            {
                const float h = entropy_nats(w.exist_bins[b] - map.params.birth_nats);
                if (h > 0.15f)
                    out.push_back({n * w.d + t * (w.bins_s0 + (static_cast<float>(b) + 0.5f) * map.params.exist_bin_m), h});
            }
            if (w.exist_bins.empty() and w.has_extent)
                out.push_back({n * w.d + t * (0.5f * (w.s_min + w.s_max)), std::log(2.f)});
        }
        // (c) CANDIDATE LINES: their existence is exactly what a visit would settle.
        for (const auto& c : map.candidates)
            if (c.npts >= 3)
                out.push_back({rc::linefit::normal_of(c.phi) * c.d
                               + rc::linefit::tangent_of(c.phi) * (0.5f * (c.s_min + c.s_max)),
                               entropy_nats(c.evidence() - map.params.birth_nats)});
        // (d) CORNERS: a position, so its uncertainty is differential entropy — the excess nats of a
        // corner wider than the publish bar, ln(sigma / bar), and nothing once it is inside it.
        for (const auto& c : poly_now.corners)
        {
            const float sig = std::isfinite(c.sigma) ? c.sigma : 1e3f;
            if (sig > map.params.publish_corner_sigma)
                out.push_back({c.p, std::log(sig / map.params.publish_corner_sigma)});
        }
        return out;
    }

    std::vector<Unknown> collect_unknowns(const rc::wallmap::WallMap& map)
    {
        std::vector<Unknown> out;
        const float bar = map.params.birth_nats;
        for (const auto& w : map.walls)
        {
            const Eigen::Vector2f n = w.normal(), t = w.tangent();
            for (size_t b = 0; b < w.exist_bins.size(); ++b)
                if (w.exist_bins[b] < bar)   // not yet solid — the map still has a question here
                    out.push_back({n * w.d + t * (w.bins_s0 + (static_cast<float>(b) + 0.5f) * map.params.exist_bin_m), 1.f});
            if (w.exist_bins.empty() and w.has_extent)
                out.push_back({n * w.d + t * (0.5f * (w.s_min + w.s_max)), 1.f});
        }
        for (const auto& c : map.candidates)
            if (c.npts >= 3)
                out.push_back({rc::linefit::normal_of(c.phi) * c.d
                               + rc::linefit::tangent_of(c.phi) * (0.5f * (c.s_min + c.s_max)), 2.f});
        const auto poly = map.build_polygon();
        for (const auto& c : poly.corners)
            if (not std::isfinite(c.sigma) or c.sigma > map.params.publish_corner_sigma)
                out.push_back({c.p, 3.f});
        for (const auto& fpt : map.frontiers())
            out.push_back({fpt, 1.5f});    // free space touching the unknown: go and look
        {
            // Weakly-held matter: a thin wall exists only if someone goes and CONFIRMS it.
            auto wm = map.weak_matter();
            const size_t stride = 1 + wm.size() / 250;     // cap the target flood, keep the spread
            for (size_t k = 0; k < wm.size(); k += stride)
                out.push_back({wm[k], 2.5f});
        }
        return out;
    }

    /// The estimation loop with the robot DRIVEN by the model's uncertainty instead of a script.
    RunResult run_explore(const Poly& room, const RunConfig& cfg, std::mt19937& rng, int max_frames,
                          const Eigen::Vector2f& start)
    {
        long occluded_total = 0;
        RunResult R;
        Explorer ex(room);
        rc::wallseg::Params sp;
        sp.sensor_sigma = cfg.scan_sigma;
        R.map.params.obs_sigma = 0.05f;
        R.map.params.huber_delta = 0.15f;
        R.map.params.debug_splice = std::getenv("WS_DEBUG_SPLICE") != nullptr;
        apply_env_overrides(R.map.params);
        // The forward-model referee scans every stored beam per judged decision: minutes per seed
        // under churn against 13 s for the whole bench without it. Opt in with WS_REFEREE=1.
        R.map.params.forward_referee = std::getenv("WS_NO_BEAMS") == nullptr;
        R.map.params.referee_log     = std::getenv("WS_REFEREE") != nullptr;
        std::ofstream trace; trace_open(trace, cfg.trace_csv);
        // PER-FRAME LAYOUT TRACE: the published polygon exactly as the world would receive it that
        // frame, with the uncertainty the model attaches to it — per corner the σ of its position
        // (Corner::sigma, from the two edges' information), per edge the σ of its own offset
        // (1/√Λ_dd). One line per frame; the origin is on the header line because the map frame is
        // start-relative. Semicolon-separated so the vertex list can keep its commas.
        std::ofstream ptrace;
        if (cfg.poly_csv != nullptr)
        {
            ptrace.open(cfg.poly_csv);
            if (ptrace.is_open())
            {
                ptrace.imbue(std::locale::classic());
                ptrace << "# origin " << start.x() << ' ' << start.y() << '\n'
                       << "# frame;est_x,est_y,est_th;tru_x,tru_y,tru_th;verts;corner_sigma;edge_sigma_d;closed,publishable\n";
            }
        }

        rc::Model model;
        model.init_from_polygon({{-20.f, -20.f}, {20.f, -20.f}, {20.f, 20.f}, {-20.f, 20.f}}, 0.f, 0.f, 0.f, 2.4f);
        RoomConcept::Params params;
        params.rfe_obs_sigma = 0.05f;
        params.rfe_huber_delta = 0.15f;
        params.enable_corner_tracking = false;
        params.object_anchor.enable = false;
        params.image_edge.enable = false;

        std::deque<RoomConcept::WindowSlot> window;
        RoomConcept::BoundaryPrior bp;
        std::normal_distribution<float> nxy(0.f, cfg.odom_sigma_xy), nth(0.f, cfg.odom_sigma_th);

        Eigen::Vector3f tru(start.x(), start.y(), 0.f);   // simulated TRUE pose; first pose = map origin
        const Eigen::Vector3f origin = tru;
        const auto to_map = [&](const Eigen::Vector3f& p)
        {
            const float c = std::cos(-origin.z()), s = std::sin(-origin.z());
            const Eigen::Vector2f dxy = p.head<2>() - origin.head<2>();
            return Eigen::Vector3f(c * dxy.x() - s * dxy.y(), s * dxy.x() + c * dxy.y(), wrap(p.z() - origin.z()));
        };
        const auto from_map = [&](const Eigen::Vector2f& p)   // estimator frame → truth frame
        {
            const float c = std::cos(origin.z()), s = std::sin(origin.z());
            return Eigen::Vector2f(c * p.x() - s * p.y() + origin.x(), s * p.x() + c * p.y() + origin.y());
        };

        Eigen::Vector3f est = Eigen::Vector3f::Zero();
        Eigen::Vector3f prev_tru_map = to_map(tru);
        double se = 0.0; int n_err = 0;
        std::vector<Eigen::Vector2f> path;   // truth-frame waypoints ahead
        int replan_in = 0;
        int quiet_frames = 0;
        int last_rederive = 0, rejected_since_rederive = 0, rederives = 0, replan_count = 0;
        // ── ATTENTION MEMORY: which BEARINGS each place has already been looked at from ──────────
        // Subtracting only "what is visible from where I stand" gave the objective no memory, and a
        // robot with no memory shuttles: standing at A, everything visible only from B is new;
        // arriving at B, everything visible only from A is new again. Measured on this bench:
        // 154.9 m walked inside a 0.65 m² corridor — 1.1% of a 60.5 m² flat — 102 cells visited a
        // mean of 10.8 times each, 29 crossings of the corridor's own midpoint.
        // What makes a look NEW is not a new position but a new ANGLE: a cell already seen from
        // this bearing has nothing more to say to a second look from the same side, while the same
        // cell seen from a different side is exactly what an unresolved corner needs (its two walls
        // must be seen well TOGETHER). So each 8 cm place keeps 16 bits, one per 22.5° bearing
        // sector it has been looked at from, and a candidate viewpoint earns a target only through
        // a sector still unmarked.
        // ⚠ THE ONE CONSTANT: 16 sectors. It is a discretisation of "a different look", of the same
        // kind as the 8 cm cell and the 0.25 m existence bin, not a tuning knob — but it is a
        // choice, and WS_SECTORS exists so it can be moved and the result measured.
        int n_sectors = 16;
        if (const char* e = std::getenv("WS_SECTORS"))
        { int v = 0; if (std::from_chars(e, e + std::strlen(e), v).ec == std::errc{} and v > 0 and v <= 16) n_sectors = v; }
        std::unordered_map<long long, unsigned short> looked;
        const auto akey = [](const Eigen::Vector2f& p)
        { return (static_cast<long long>(std::lround(p.x() / 0.08f)) << 22)
               ^  static_cast<long long>(std::lround(p.y() / 0.08f)); };
        const auto sector_of = [&](const Eigen::Vector2f& d)
        {
            float a = std::atan2(d.y(), d.x());
            if (a < 0.f) a += 2.f * kPi;
            const int k = static_cast<int>(a / (2.f * kPi) * static_cast<float>(n_sectors));
            return std::min(k, n_sectors - 1);
        };

        for (int f = 0; f < max_frames; ++f)
        {
            // ── act: follow the current plan one step ─────────────────────────────────────────
            if (not path.empty())
            {
                const Eigen::Vector2f tgt = path.front();
                const Eigen::Vector2f d = tgt - tru.head<2>();
                const float L = d.norm();
                if (L < 0.15f) path.erase(path.begin());
                else
                {
                    const float step = std::min(0.28f, L);
                    tru.head<2>() += d / L * step;
                    tru.z() = std::atan2(d.y(), d.x());
                }
            }

            const Eigen::Vector3f tm = to_map(tru);
            Eigen::Vector3f odom = tm - prev_tru_map; odom.z() = wrap(odom.z());
            if (f > 0) odom += Eigen::Vector3f(nxy(rng), nxy(rng), nth(rng));
            prev_tru_map = tm;
            const Eigen::Vector3f pred = (f == 0) ? Eigen::Vector3f::Zero()
                : Eigen::Vector3f(est.x() + odom.x(), est.y() + odom.y(), wrap(est.z() + odom.z()));

            // Both sensors in one sweep, with the disagreement between them read as an occluder
            // rather than averaged away (see scan_fused). With the camera off this is the plain
            // LiDAR sweep and the weights are all one.
            Eigen::VectorXf pw;
            int n_occluded = 0;
            auto pts = scan_fused(room, cfg, tru, rng, pw, n_occluded);
            occluded_total += n_occluded;
            if (R.map.walls.empty())
            {
                // model-first init from the first scan's OBB (same as run_loop)
                Eigen::Vector2f mu = Eigen::Vector2f::Zero();
                for (const auto& p : pts) mu += p;
                mu /= static_cast<float>(pts.size());
                Eigen::Matrix2f C = Eigen::Matrix2f::Zero();
                for (const auto& p : pts) { const Eigen::Vector2f dd = p - mu; C += dd * dd.transpose(); }
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eig(C);
                const Eigen::Vector2f ax = eig.eigenvectors().col(1), ay = eig.eigenvectors().col(0);
                float lo0 = 1e9f, hi0 = -1e9f, lo1 = 1e9f, hi1 = -1e9f;
                for (const auto& p : pts)
                { const float u = ax.dot(p - mu), v2 = ay.dot(p - mu);
                  lo0 = std::min(lo0, u); hi0 = std::max(hi0, u); lo1 = std::min(lo1, v2); hi1 = std::max(hi1, v2); }
                Poly rect = {mu + ax * lo0 + ay * lo1, mu + ax * hi0 + ay * lo1, mu + ax * hi0 + ay * hi1, mu + ax * lo0 + ay * hi1};
                float a2 = 0.f;
                for (size_t i = 0; i < rect.size(); ++i)
                { const auto& p = rect[i]; const auto& q = rect[(i + 1) % rect.size()]; a2 += p.x() * q.y() - q.x() * p.y(); }
                if (a2 < 0.f) std::reverse(rect.begin(), rect.end());
                R.map.initialize_rect(rect);
            }
            const auto seg = rc::wallseg::segment(pts, sp, rng);
            const Eigen::Matrix3f pcov = Eigen::Vector3f(0.05f * 0.05f, 0.05f * 0.05f, 0.03f * 0.03f).asDiagonal();
            const auto fr = R.map.observe(seg, pts, pw, pred, pcov, static_cast<std::int64_t>(f) * 50);
            R.births += fr.births; R.deaths += fr.deaths; R.rejected += fr.splice_rejected;

            RoomConcept::WindowSlot slot;
            slot.pose = pose_tensor(pred);
            slot.lidar_points = points_tensor(pts);
            slot.odometry_delta = (f == 0) ? Eigen::Vector3f::Zero() : odom;
            slot.motion_cov = Eigen::Vector3f(cfg.odom_sigma_xy * cfg.odom_sigma_xy, cfg.odom_sigma_xy * cfg.odom_sigma_xy,
                                              cfg.odom_sigma_th * cfg.odom_sigma_th).asDiagonal();
            slot.odom_delta_tensor = torch::tensor({odom.x(), odom.y(), odom.z()}, torch::kFloat32);
            slot.motion_prec_tensor = mat3(slot.motion_cov.inverse());
            slot.wall_assoc = fr.assoc;

            rc::gn::Input in;
            in.model = &model; in.params = &params; in.window = &window; in.boundary_prior = &bp;
            in.device = torch::kCPU;
            in.walls = &R.map; in.no_sdf = true; in.gauge_fix = true;
            if (static_cast<int>(window.size()) >= cfg.window)
            {
                auto fp = window.front().pose.detach();
                rc::gn::absorb_wall_observations(in, window.front(),
                    Eigen::Vector3f(fp[0].item<float>(), fp[1].item<float>(), fp[2].item<float>()));
                window.pop_front();
                auto nf = window.front().pose.detach();
                bp.valid = true;
                bp.mu = Eigen::Vector3f(nf[0].item<float>(), nf[1].item<float>(), nf[2].item<float>());
                bp.precision = Eigen::Vector3f(400.f, 400.f, 1600.f).asDiagonal();
            }
            window.push_back(std::move(slot));
            std::vector<Eigen::Vector3f> poses;
            for (const auto& sl : window)
            { auto pp = sl.pose.detach(); poses.emplace_back(pp[0].item<float>(), pp[1].item<float>(), pp[2].item<float>()); }
            rc::gn::Options opts;
            const auto r = rc::gn::solve(in, poses, opts);
            if (r.ok) for (size_t i = 0; i < window.size(); ++i) window[i].pose = pose_tensor(poses[i]);
            est = poses.back();
            R.map.merge_indistinguishable();

            // ── GLOBAL re-derivation: when local jumps are stuck (rejections pile up) or on a slow
            // cadence, trace the observed free space and adopt its cycle iff it explains more.
            rejected_since_rederive += fr.splice_rejected;
            if (R.map.params.enable_rederive
                and (f - last_rederive >= R.map.params.rederive_every_frames
                     or rejected_since_rederive >= R.map.params.rederive_after_rejections))
            {
                last_rederive = f;
                rejected_since_rederive = 0;
                if (R.map.re_derive(est.head<2>()))
                {
                    ++rederives;
                    if (cfg.verbose)
                        std::printf("    f=%3d GLOBAL re-derivation adopted: %zu walls in cycle\n",
                                    f, R.map.order.size());
                }
            }

            const Eigen::Vector3f err = est - tm;
            const float exy = err.head<2>().norm();
            se += exy * exy; ++n_err;
            R.pose_max_xy = std::max(R.pose_max_xy, exy);
            R.pose_max_th = std::max(R.pose_max_th, std::abs(wrap(err.z())));

            // ── perceive → decide: replan toward the largest EXPECTED INFORMATION GAIN ───────
            if (--replan_in <= 0 or path.empty())
            {
                // ═══ WHAT A LOOK FROM v WOULD ACTUALLY TEACH, predicted through the model ═══════
                // The previous objective summed the entropy VISIBLE from a viewpoint, with
                // visibility ray-cast against the TRUE room. Two things were wrong with that and
                // both were visible in the robot's behaviour. It used an oracle: it knew what it
                // would see from a place it had never been. And it scored H(x), not the expected
                // reduction H(x) − E_o[H(x|o)], so a cell it could see but could not RESOLVE — one
                // behind a wall the estimate already knows about, or past the beam's reach — paid
                // its full ignorance for ever and never paid it back. That is what parked the robot
                // (3.4 m, then 1077 stationary frames) and, once patched by hand, what made it
                // shuttle (154.9 m inside 1.1% of the floor).
                //
                // This is the expectation itself, taken with the same beam model the estimator
                // uses, through the ESTIMATED polygon and the occupancy grid — never the true room:
                //   · cast the scan the robot would take from v, stopping at the model's own walls,
                //     so everything behind them contributes nothing because it would not be seen;
                //   · per traversed grid cell, the expected entropy drop under the grid's OWN
                //     update rule: H(p) − [p·H(σ(l+1.0)) + (1−p)·H(σ(l−0.4))], which is ~0 for a
                //     cell already certain and largest for one that is genuinely unknown;
                //   · per beam that lands on a wall, the Fisher information that point adds to that
                //     wall's (φ, d) — h = [s, −1], J = h·hᵀ/σ² — and then the CORNERS: a corner's
                //     σ is recomputed from its two walls' augmented information with the model's own
                //     intersect_walls, and the look earns ln(σ_before/σ_after) nats for it.
                // The last term is what a corner needs and no visibility count can express: the
                // gain is large only where a viewpoint sees a long stretch of BOTH walls that meet.
                // Nothing here is a weight; the units are nats throughout, and the ratio to
                // distance stays what it was — nats per metre.
                const auto eig_of = [&](const Eigen::Vector2f& v_world) -> float
                {
                    const Eigen::Vector2f v = to_map(Eigen::Vector3f(v_world.x(), v_world.y(), 0.f)).head<2>();
                    const auto poly = R.map.build_polygon();
                    if (not poly.closed or poly.verts.size() < 3) return 0.f;
                    const auto& fg = R.map.fgrid;
                    if (not fg.ready()) return 0.f;
                    const int NB = 36;
                    const bool occl_pred = std::getenv("WS_NO_OCCL_PRED") == nullptr;
                    const float sig2 = R.map.params.obs_sigma * R.map.params.obs_sigma;
                    std::unordered_map<std::uint64_t, Eigen::Matrix2f> J;   // wall id → added information
                    float gain = 0.f;
                    for (int b = 0; b < NB; ++b)
                    {
                        const float a = 2.f * kPi * static_cast<float>(b) / static_cast<float>(NB);
                        const Eigen::Vector2f dir(std::cos(a), std::sin(a));
                        // range predicted by the MODEL's own boundary
                        float r_pred = R.map.params.sensor_range; int hit = -1;
                        for (size_t e = 0; e < poly.verts.size(); ++e)
                            if (const auto t = rc::corner_visibility::ray_segment_t(
                                    v, dir, poly.verts[e], poly.verts[(e + 1) % poly.verts.size()]);
                                t and *t > 1e-3f and *t < r_pred) { r_pred = *t; hit = static_cast<int>(e); }
                        // Every cell the beam would cross — and it STOPS at the first matter the
                        // grid already holds, not only at the polygon. Predicting through the
                        // boundary alone made furniture invisible to the planner while it stays
                        // decisive for the sensor: the prediction sent beams sailing past a cupboard
                        // to the wall behind it and credited every cell on the way, the real beam
                        // stopped at the cupboard, and the uncollected promise was re-offered from
                        // the other side next replan. Measured: two cells 0.6 m apart, ~125 nats
                        // each, alternating for hundreds of frames. A prediction must be made
                        // through everything the model believes, or it is not an expectation.
                        bool blocked = false;
                        for (float rr = fg.cell; rr < r_pred; rr += fg.cell)
                        {
                            const Eigen::Vector2f q = v + dir * rr;
                            const int i = static_cast<int>((q.x() - fg.x0) / fg.cell);
                            const int j = static_cast<int>((q.y() - fg.y0) / fg.cell);
                            if (not fg.in(i, j)) continue;
                            const float l = fg.lodds[static_cast<size_t>(fg.idx(i, j))];
                            const float p = 1.f / (1.f + std::exp(-l));
                            const float h_now = entropy_nats(l);
                            const float h_after = p * entropy_nats(std::min(4.f, l + 1.0f))
                                                + (1.f - p) * entropy_nats(std::max(-4.f, l - 0.4f));
                            gain += std::max(0.f, h_now - h_after);
                            // WS_NO_OCCL_PRED=1 restores the polygon-only prediction, so the
                            // occlusion term can be measured on its own rather than against a run
                            // taken under a different address layout.
                            if (occl_pred and fg.is_occupied(i, j)) { blocked = true; break; }
                        }
                        // what the returning point would tell the wall it lands on — only if it gets there
                        if (not blocked and hit >= 0 and static_cast<size_t>(hit) < poly.wall_of_edge.size())
                        {
                            const auto* w = R.map.find(poly.wall_of_edge[static_cast<size_t>(hit)]);
                            if (w != nullptr)
                            {
                                const Eigen::Vector2f q = v + dir * r_pred;
                                const float sc_ = w->tangent().dot(q);
                                Eigen::Vector2f h(sc_, -1.f);
                                J[w->id] += (h * h.transpose()) / sig2;
                            }
                        }
                    }
                    // corners: the model's own σ, recomputed with the information the look would add
                    for (const auto& c : poly.corners)
                    {
                        if (not std::isfinite(c.sigma) or c.sigma <= R.map.params.publish_corner_sigma) continue;
                        const auto* wa = R.map.find(c.wall_a); const auto* wb = R.map.find(c.wall_b);
                        if (wa == nullptr or wb == nullptr) continue;
                        const auto ja = J.find(wa->id), jb = J.find(wb->id);
                        if (ja == J.end() and jb == J.end()) continue;     // this look says nothing about it
                        rc::wallmap::WallLandmark a2 = *wa, b2 = *wb;
                        if (ja != J.end()) a2.information += ja->second;
                        if (jb != J.end()) b2.information += jb->second;
                        const auto c2 = rc::wallmap::WallMap::intersect_walls(a2, b2, c.inferred);
                        if (std::isfinite(c2.sigma) and c2.sigma > 0.f and c2.sigma < c.sigma)
                            gain += std::log(c.sigma / c2.sigma);          // differential entropy, nats
                    }
                    return gain;
                };
                replan_in = 15;
                const auto unknowns = cfg.info_gain ? collect_entropy_targets(R.map) : collect_unknowns(R.map);
                // Exploration is COMPLETE when free space has no true frontier left; the map may
                // keep refining, but there is nowhere informative left to drive to.
                // ── THE GUARD: a run may only declare itself finished with a CLOSED polygon ──
                // "Nothing left worth looking at" and "the map is done" are different statements,
                // and conflating them cost a whole run: with an honest predictor the offers fall to
                // fractions of a nat, the no-path condition starts firing, and the run stopped at
                // frame 803 holding an OPEN cycle — which publishes nothing and scores 0.000. An
                // open polygon is itself the loudest thing the map can say about its own state, so
                // exploration is not over while it stands, whatever the information ledger says.
                // The frame budget remains the outer bound; nothing here can run for ever.
                const bool closed_now = R.map.build_polygon().closed;
                if (unknowns.empty() and closed_now) { R.frames = f + 1; break; }
                float best_sc = -1.f; Eigen::Vector2f best_v = tru.head<2>();
                const bool dbg_plan = std::getenv("WS_DEBUG_PLAN") != nullptr;
                // COVERAGE GUARANTEE: greedy argmax-by-visible-mass starves sparse far regions — a
                // wrong early wall then amputates a whole space for ever, because nothing ever goes
                // where it would be contradicted (seed-7: one diagonal cut off the SE space for 1100
                // frames). Every 4th replan goes to the NEAREST frontier, mass be damned.
                // Coverage has diminishing value as frontiers vanish; dwell does not. Once the
                // space is essentially explored, every replan goes back to refinement — that is
                // what the good seeds paid for coverage turns before this condition existed.
                const auto fronts = R.map.frontiers();
                // The coverage turn is a patch for an objective that undervalues far regions. An
                // entropy objective does not need it: a far leg full of unknown cells carries more
                // nats than a near sliver, and the ratio decides honestly.
                const bool coverage_turn = not cfg.info_gain and (++replan_count % 4 == 0) and fronts.size() > 3;
                // ⚠ TRIED AND REVERTED 2026-09-03: every second coverage turn to the FARTHEST
                // frontier instead. It is the obvious cure for what the coverage diagnostic shows —
                // an L room leaves 14% of its own interior unseen while only 5% is seen and left
                // outside the polygon — but it does not pay: the real flat's worst seed rose 0.902
                // to 0.966 while its best fell 0.980 to 0.966 and a second check began to fail, and
                // over 50 rooms the median fell 0.947 to 0.941, the Hausdorff median rose 0.89 to
                // 1.14 m and spurs dropped 13% to 10%. A far frontier costs a long drive whose
                // frames come out of refinement. The coverage problem is real; a nearest/farthest
                // heuristic is not its answer — an information-gain objective that prices the drive
                // against what it would reveal is (literature review, rank 6).
                if (coverage_turn)
                {
                    float dbest = 1e9f;
                    for (const auto& fp : fronts)
                    {
                        const Eigen::Vector2f ft = from_map(fp);
                        const float dd = (ft - tru.head<2>()).norm();
                        if (dd < dbest) { dbest = dd; best_v = ft; best_sc = 1.f; }
                    }
                }
                // ── WHAT A LOOK WOULD ADD, not what happens to be in view ────────────────────
                // The old score summed every target VISIBLE from a candidate cell, and the robot's
                // own cell is a candidate: it sees whatever it already sees, divides by 1 + 0 m,
                // and wins. Worse, a target it cannot resolve from here — occluded, or past the
                // beam's reach — keeps its entropy for ever, so the score never decayed and the
                // drive never resumed. Measured before this rewrite: 3.40 m travelled on every one
                // of twelve apartamento seeds, motion ending at frame 23, 97.9% of all frames
                // stationary, 13-63 frontiers still open, and the population's worst room (22)
                // replanning 898 times having moved once.
                //
                // A look is worth the entropy it would NEWLY expose, so everything already in view
                // from where the robot stands is subtracted first. This carries no constant and no
                // radius: the robot's own cell scores exactly zero by construction, and so does any
                // cell that merely re-sees this standpoint's view. When every cell scores zero
                // there is nothing left to learn by moving — which is the honest definition of a
                // finished exploration, and what "stabilised" should mean.
                // Everything the robot can see FROM HERE is now recorded against the bearing it is
                // seeing it from, so this standpoint — and every standpoint it has already used —
                // stops paying. The record is what the old version lacked.
                const bool use_eig = std::getenv("WS_NO_EIG") == nullptr;
                std::vector<Eigen::Vector2f> tp(unknowns.size());
                std::vector<unsigned short>  tmask(unknowns.size(), 0);
                const unsigned short full = static_cast<unsigned short>((1u << n_sectors) - 1u);
                for (size_t k = 0; k < unknowns.size() and not use_eig; ++k)
                {
                    tp[k] = from_map(unknowns[k].p);
                    if (ex.sees(tru.head<2>(), tp[k]))
                        looked[akey(tp[k])] |= static_cast<unsigned short>(1u << sector_of(tru.head<2>() - tp[k]));
                }
                for (size_t k = 0; k < unknowns.size() and not use_eig; ++k)
                    if (const auto it = looked.find(akey(tp[k])); it != looked.end()) tmask[k] = it->second;
                // Keep the best few, not just the argmax: a viewpoint A* cannot reach used to end
                // the plan silently and park the robot for the rest of the run (room 22: 898
                // replans, `path=0` every time). An unreachable winner now yields to the next.
                std::vector<std::pair<float, Eigen::Vector2f>> cands;
                if (best_sc < 0.f)
                for (int i = 0; i < ex.nx; i += 2)
                    for (int j = 0; j < ex.ny; j += 2)
                    {
                        if (not ex.is_free(i, j)) continue;
                        const Eigen::Vector2f v = ex.at(i, j);
                        float sc = 0.f;
                        if (use_eig) sc = eig_of(v);
                        else
                        for (size_t k = 0; k < unknowns.size(); ++k)
                        {
                            // ── the hand-built stand-in this replaces (WS_NO_EIG=1 to compare) ──
                            if (tmask[k] == full) continue;
                            if (tmask[k] != 0 and (tmask[k] & static_cast<unsigned short>(1u << sector_of(v - tp[k])))) continue;
                            if (ex.sees(v, tp[k])) sc += unknowns[k].w;
                        }
                        if (sc <= 0.f) continue;
                        // GAIN PER METRE. The constant is not a tuning knob: it is the distance the
                        // robot covers between replans, so the ratio is nats per look. Without it
                        // the explorer would teleport-shop; with a hand-set 0.10 discount it valued
                        // distance by taste.
                        sc /= (cfg.info_gain ? (1.0f + (v - tru.head<2>()).norm())
                                             : (1.f + 0.10f * (v - tru.head<2>()).norm()));
                        cands.emplace_back(sc, v);
                    }
                std::sort(cands.begin(), cands.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
                if (cands.size() > 16) cands.resize(16);
                if (best_sc > 0.f)   // a coverage turn already chose one
                {
                    path = ex.astar(tru.head<2>(), best_v);
                    if (path.size() > 1) path.erase(path.begin());
                    else path.clear();
                }
                else
                    for (const auto& [sc, v] : cands)
                    {
                        path = ex.astar(tru.head<2>(), v);
                        if (path.size() > 1) { path.erase(path.begin()); best_sc = sc; best_v = v; break; }
                        path.clear();
                    }
                // STABILISED: no reachable viewpoint anywhere would show the map anything it has
                // not already seen from here. Counted over consecutive replans for the same reason
                // the old test was — one replan can fall in a moment when the targets are briefly
                // all in view — and only after the map has had a chance to exist.
                quiet_frames = (f > 60 and path.empty()) ? quiet_frames + 1 : 0;
                if (quiet_frames >= 3 and closed_now) { R.frames = f + 1; break; }
                if (dbg_plan)
                    std::printf("[plan] f=%4d targets=%zu frontiers=%zu best_sc=%.3f best_v=(%.2f,%.2f) "
                                "here=(%.2f,%.2f) path=%zu quiet=%d\n",
                                f, unknowns.size(), R.map.frontiers().size(), best_sc, best_v.x(), best_v.y(),
                                tru.x(), tru.y(), path.size(), quiet_frames);
            }
            if (cfg.verbose and (f % 25 == 0 or fr.births > 0))
                std::printf("    f=%3d walls=%zu cand=%d births=%d deaths=%d err=%.3fm poly=%s\n",
                            f, R.map.walls.size(), fr.candidates, fr.births, fr.deaths, exy,
                            R.map.build_polygon().closed ? "closed" : "open");
            trace_precisions(trace, R.map, 0.f, exy);
            if (ptrace.is_open())
            {
                const auto pub = R.map.manhattan_polygon();
                ptrace << f << ';' << est.x() << ',' << est.y() << ',' << est.z()
                       << ';' << tru.x() << ',' << tru.y() << ',' << tru.z() << ';';
                for (size_t v = 0; v < pub.verts.size(); ++v)
                    ptrace << (v ? " " : "") << pub.verts[v].x() << ',' << pub.verts[v].y();
                ptrace << ';';
                for (size_t v = 0; v < pub.corners.size(); ++v)
                    ptrace << (v ? " " : "") << pub.corners[v].sigma;
                ptrace << ';';
                for (size_t e = 0; e < pub.wall_of_edge.size(); ++e)
                {
                    const auto* w = R.map.find(pub.wall_of_edge[e]);
                    const float lam = (w != nullptr) ? w->information(1, 1) : 0.f;
                    ptrace << (e ? " " : "") << ((lam > 1e-9f) ? 1.f / std::sqrt(lam) : -1.f);
                }
                std::string st = pub.status;
                for (char& ch : st) if (ch == ';' or ch == '\n') ch = ' ';
                const auto raw = R.map.build_polygon();
                std::string st_raw = raw.status;
                for (char& ch : st_raw) if (ch == ';' or ch == '\n') ch = ' ';
                ptrace << ';' << (pub.closed ? 1 : 0) << ',' << (pub.publishable ? 1 : 0)
                       << ';' << R.map.walls.size() << ',' << R.map.order.size() << ',' << (raw.closed ? 1 : 0)
                       << ';' << st << ';' << st_raw << '\n';
            }
            R.frames = f + 1;
        }
        R.pose_rmse_xy = (n_err > 0) ? static_cast<float>(std::sqrt(se / n_err)) : 0.f;
        R.last_xy = est.head<2>();
        R.occluded_pts = occluded_total;
        if (cfg.verbose) std::printf("    global re-derivations adopted: %d\n", rederives);
        R.poly = R.map.manhattan_polygon();   // published layout: exactly Manhattan
        return R;
    }

    /// Map-frame polygon → the truth's frame (the run's origin was truth[0]).
    Poly to_world(const Poly& p, const Eigen::Vector3f& origin)
    {
        Poly out;
        const float c = std::cos(origin.z()), s = std::sin(origin.z());
        for (const auto& v : p) out.emplace_back(c * v.x() - s * v.y() + origin.x(), s * v.x() + c * v.y() + origin.y());
        return out;
    }
} // namespace

int main()
{
    // ── DETERMINISM. torch::set_num_threads(1) alone was NOT enough: the same command returned
    // 0.974 / 0.906 / 0.906 / 0.923 / 0.932 on one room, and 0.000 inside a loaded sweep, because
    // OTHER OpenMP-parallel code in the process (Eigen's GEMM among it) still took its team size
    // from the machine — so the reduction ORDER, and therefore the answer, followed the load.
    // Setting OMP_NUM_THREADS in the environment fixed it 3/3, which is the same statement made
    // from outside; these two calls make it from inside, so the guarantee travels with the binary.
    // A bench whose result depends on what else the machine is doing cannot referee anything.
    // The in-process calls are NOT enough on their own: the OpenMP runtime and the BLAS pools size
    // themselves before main(), so shrinking them here leaves the reduction order already chosen.
    // Only the ENVIRONMENT is read early enough — so if it is not set, set it and re-exec ourselves
    // once. Verified: with it, four identical invocations of the room that exposed this return the
    // same IoU; without it they returned 0.974 / 0.906 / 0.906 / 0.923 / 0.932 and 0.000 in a loaded
    // sweep. WS_THREADS=n opts out deliberately, trading the guarantee for speed.
    if (std::getenv("OMP_NUM_THREADS") == nullptr and std::getenv("WS_THREADS") == nullptr)
    {
        ::setenv("OMP_NUM_THREADS", "1", 1);
        ::setenv("MKL_NUM_THREADS", "1", 1);
        ::setenv("OPENBLAS_NUM_THREADS", "1", 1);
        char self[4096];
        const ssize_t n = ::readlink("/proc/self/exe", self, sizeof self - 1);
        if (n > 0) { self[n] = '\0'; char* av[] = {self, nullptr}; ::execv(self, av); }
        // exec failed: carry on unpinned rather than not run at all, and say so.
        std::printf("[warn] could not re-exec to pin the thread pools; results may not be reproducible\n");
    }
    Eigen::setNbThreads(1);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    torch::set_num_threads(1);
    std::mt19937 rng(7);

    // WS_ONLY7=1 skips the synthetic sections — iteration aid for the (slow) real-layout bench.
    if (std::getenv("WS_ONLY7") == nullptr)
    {
    // ═══ 1. Segmenter ═════════════════════════════════════════════════════════════════════════
    std::printf("\n1. Segmenter on one scan of the L room\n");
    {
        const Poly room = l_room();
        const Eigen::Vector3f pose(-1.f, 0.f, 0.4f);
        const auto pts = scan(room, pose, 720, 0.02f, rng);
        rc::wallseg::Params sp; sp.sensor_sigma = 0.02f;
        const auto res = rc::wallseg::segment(pts, sp, rng);
        const auto tw = true_walls(room);

        int visible = 0;
        for (const auto& w : tw)
        {
            // A wall is visible if any scan point lies on it.
            const float c = std::cos(pose.z()), s = std::sin(pose.z());
            for (const auto& p : pts)
            {
                const Eigen::Vector2f q(c * p.x() - s * p.y() + pose.x(), s * p.x() + c * p.y() + pose.y());
                if (std::abs(rc::linefit::normal_of(w.phi).dot(q) - w.d) < 0.06f) { ++visible; break; }
            }
        }
        int matched = 0, sign_ok = 0;
        float worst = 0.f;
        for (const auto& sg : res.segments)
        {
            if (sg.d < 0.f) ++sign_ok;
            // Segment in the world frame vs the nearest true wall.
            const float phi_w = wrap(sg.phi + pose.z());
            const float d_w = sg.d + rc::linefit::normal_of(phi_w).dot(pose.head<2>());
            float best = 1e9f;
            for (const auto& w : tw)
                best = std::min(best, std::abs(d_w - w.d) + 2.f * std::abs(wrap(phi_w - w.phi)));
            worst = std::max(worst, best);
            if (best < 0.06f) ++matched;
        }
        check("every visible wall yields a segment", static_cast<int>(res.segments.size()) >= visible and matched >= visible,
              fmt("visible %d, segments %zu, matched %d, worst misfit %.3f", visible, res.segments.size(), matched, worst));
        check("sign convention: d < 0 for every segment", sign_ok == static_cast<int>(res.segments.size()),
              fmt("%d / %zu", sign_ok, res.segments.size()));
        check("observed corners on a scan that sees corners", res.corners.size() >= 2,
              fmt("%zu corners, %zu unexplained points, %d ransac iters", res.corners.size(), res.unexplained.size(), res.ransac_iters));
        int info_ok = 0;
        for (const auto& sg : res.segments) if (sg.info_phi_d.determinant() > 0.f) ++info_ok;
        check("every segment carries a positive-definite Λ(φ,d)", info_ok == static_cast<int>(res.segments.size()),
              fmt("%d / %zu", info_ok, res.segments.size()));
    }

    // ═══ 2. Jacobians ═════════════════════════════════════════════════════════════════════════
    std::printf("\n2. Jacobians of the wall factor family vs central differences\n");
    {
        const Poly room = l_room();
        const auto tw = true_walls(room);
        const std::vector<Eigen::Vector3f> truth = {{-1.f, 0.f, 0.3f}, {-0.8f, 0.1f, 0.35f}, {-0.6f, 0.2f, 0.4f}};

        rc::wallmap::WallMap map;
        map.theta0_born = true; map.theta0 = tw[0].phi; map.theta0_information = 1e3f;
        for (size_t i = 0; i < tw.size(); ++i)
        {
            rc::wallmap::WallLandmark w;
            w.id = i + 1; w.phi = tw[i].phi; w.d = tw[i].d;
            w.k = static_cast<int>(std::lround(wrap(w.phi - map.theta0) / (kPi / 2.f))) & 3;
            w.manhattan_var = std::pow(2.f * kPi / 180.f, 2.f);
            w.information = Eigen::Vector2f(500.f, 2000.f).asDiagonal();
            map.walls.push_back(w);
        }

        rc::Model model;
        model.init_from_polygon(room, 0.f, 0.f, 0.f, 2.4f);
        RoomConcept::Params params;
        params.rfe_obs_sigma = 0.05f; params.rfe_huber_delta = 0.15f;
        params.enable_corner_tracking = false; params.object_anchor.enable = false; params.image_edge.enable = false;

        std::deque<RoomConcept::WindowSlot> window;
        for (size_t i = 0; i < truth.size(); ++i)
        {
            RoomConcept::WindowSlot slot;
            slot.pose = pose_tensor(truth[i]);
            const auto pts = scan(room, truth[i], 360, 0.01f, rng);
            slot.lidar_points = points_tensor(pts);
            slot.odometry_delta = (i == 0) ? Eigen::Vector3f::Zero() : Eigen::Vector3f(truth[i] - truth[i - 1]);
            slot.motion_cov = Eigen::Vector3f(4e-4f, 4e-4f, 1e-4f).asDiagonal();
            slot.odom_delta_tensor = torch::tensor({slot.odometry_delta.x(), slot.odometry_delta.y(), slot.odometry_delta.z()}, torch::kFloat32);
            slot.motion_prec_tensor = mat3(slot.motion_cov.inverse());
            // Associate every point to its true wall (the harness knows), one WallAssoc per wall.
            const float c = std::cos(truth[i].z()), s = std::sin(truth[i].z());
            std::vector<std::vector<Eigen::Vector2f>> per_wall(tw.size());
            for (const auto& p : pts)
            {
                const Eigen::Vector2f q(c * p.x() - s * p.y() + truth[i].x(), s * p.x() + c * p.y() + truth[i].y());
                int best = -1; float bd = 1e9f;
                for (size_t w = 0; w < tw.size(); ++w)
                {
                    const float dd = std::abs(rc::linefit::normal_of(tw[w].phi).dot(q) - tw[w].d);
                    if (dd < bd) { bd = dd; best = static_cast<int>(w); }
                }
                if (best >= 0 and bd < 0.1f) per_wall[static_cast<size_t>(best)].push_back(p);
            }
            for (size_t w = 0; w < tw.size(); ++w)
            {
                if (per_wall[w].size() < 3) continue;
                rc::wallmap::WallAssoc a;
                a.wall_id = w + 1; a.pda = 1.f;
                a.pts.resize(static_cast<long>(per_wall[w].size()), 2);
                for (size_t k = 0; k < per_wall[w].size(); ++k) { a.pts(static_cast<long>(k), 0) = per_wall[w][k].x(); a.pts(static_cast<long>(k), 1) = per_wall[w][k].y(); }
                slot.wall_assoc.push_back(std::move(a));
            }
            window.push_back(std::move(slot));
        }
        RoomConcept::BoundaryPrior bp; bp.valid = false;

        rc::gn::Input in;
        in.model = &model; in.params = &params; in.window = &window; in.boundary_prior = &bp; in.device = torch::kCPU;
        in.walls = &map; in.no_sdf = true; in.gauge_fix = true;

        std::vector<Eigen::Vector3f> at = truth;
        at[2] += Eigen::Vector3f(0.03f, -0.02f, 0.015f);

        struct Case { const char* name; bool points, room, priors, gauge, motion; float corrupt; };
        for (const Case& cs : std::vector<Case>{
                {"wall points only",              true,  false, false, false, false, 0.f},
                {"wall points only, SATURATED",   true,  false, false, false, false, 0.4f},
                {"room<->wall (Manhattan) only",  false, true,  false, false, false, 0.f},
                {"carried priors only",           false, false, true,  false, false, 0.f},
                {"gauge only",                    false, false, false, true,  false, 0.f},
                {"all wall factors + motion",     true,  true,  true,  true,  true,  0.f},
                {"all wall factors + motion, SATURATED", true, true, true, true, true, 0.4f}})
        {
            rc::wallmap::WallMap m = map;
            std::deque<RoomConcept::WindowSlot> w = window;
            if (not cs.points) for (auto& s : w) s.wall_assoc.clear();
            if (not cs.motion) for (auto& s : w) s.motion_prec_tensor = mat3(Eigen::Matrix3f::Zero());
            if (not cs.room) for (auto& wl : m.walls) wl.manhattan_var = 0.f;
            if (not cs.priors) { for (auto& wl : m.walls) wl.information.setZero(); m.theta0_information = 0.f; }
            if (cs.corrupt > 0.f) for (auto& wl : m.walls) wl.d += cs.corrupt;   // every point past the Huber knee
            // Off the optimum in the MAP variables too, so their columns of the Jacobian are exercised.
            for (auto& wl : m.walls) { wl.phi = wrap(wl.phi + 0.01f); wl.d += 0.02f; }
            m.theta0 = wrap(m.theta0 - 0.008f);
            rc::gn::Input i2 = in; i2.window = &w; i2.walls = &m; i2.gauge_fix = cs.gauge;
            const float err = rc::gn::gradient_check(i2, at);
            check(cs.name, std::isfinite(err) and err < 1e-2f, fmt("rel err = %.3e", err));
        }

        // Convergence: perturb poses AND walls, solve, compare with truth.
        {
            rc::wallmap::WallMap m = map;
            for (auto& wl : m.walls) { wl.phi = wrap(wl.phi + 0.02f); wl.d += 0.05f; wl.information = Eigen::Vector2f(1.f, 1.f).asDiagonal(); }
            m.theta0 = wrap(m.theta0 + 0.02f); m.theta0_information = 1.f;
            std::vector<Eigen::Vector3f> poses = truth;
            for (auto& p : poses) p += Eigen::Vector3f(0.05f, -0.04f, 0.03f);
            // The truth here is NOT expressed in a first-pose frame, so pin slot 0 at truth[0] with a
            // boundary prior instead of the origin gauge; the walls can then be compared directly.
            RoomConcept::BoundaryPrior bp2; bp2.valid = true; bp2.mu = truth[0];
            bp2.precision = Eigen::Vector3f(1e6f, 1e6f, 1e6f).asDiagonal();
            rc::gn::Input i2 = in; i2.walls = &m; i2.boundary_prior = &bp2; i2.gauge_fix = false;
            rc::gn::Options opts;
            const float loss0 = rc::gn::evaluate(i2, poses);
            const auto r = rc::gn::solve(i2, poses, opts);
            float worst_d = 0.f, worst_phi = 0.f;
            for (size_t k = 0; k < tw.size(); ++k)
            {
                worst_d = std::max(worst_d, std::abs(m.walls[k].d - tw[k].d));
                worst_phi = std::max(worst_phi, std::abs(wrap(m.walls[k].phi - tw[k].phi)));
            }
            const Eigen::Vector3f rel = poses[2] - poses[0], rel_t = truth[2] - truth[0];
            check("joint solve recovers walls and relative poses",
                  r.ok and r.loss < loss0 and worst_d < 0.03f and worst_phi < 0.01f and (rel - rel_t).head<2>().norm() < 0.03f,
                  fmt("loss %.3f->%.3f, %d it, worst |dd| %.3f m, |dphi| %.4f rad, rel pose err %.3f m",
                      loss0, r.loss, r.iterations, worst_d, worst_phi, (rel - rel_t).head<2>().norm()));
        }
    }

    // ═══ 3. Full loop in the L room ════════════════════════════════════════════════════════════
    std::printf("\n3. Estimate the L room while driving a loop (window 5, 360 rays, sigma 2 cm)\n");
    {
        const Poly room = l_room();
        const auto truth = trajectory(12);   // 6 legs x 12 = 72 frames
        RunConfig cfg; cfg.verbose = true;
        const auto R = run_loop({room}, truth, cfg, rng);
        std::printf("    walls=%zu candidates=%zu births=%d closed_at=%d status='%s' worst corner sigma=%.3f\n",
                    R.map.walls.size(), R.map.candidates.size(), R.births, R.closed_at, R.poly.status.c_str(), R.poly.worst_corner_sigma);
        for (const auto& w : R.map.walls)
            std::printf("    wall %llu k=%d phi=%.3f d=%.3f extent[%.2f,%.2f] frames=%d dF=%.2f info=(%.0f,%.0f)\n",
                        static_cast<unsigned long long>(w.id), w.k, w.phi, w.d, w.s_min, w.s_max, w.frames_seen, w.room_factor_dF,
                        w.information(0, 0), w.information(1, 1));
        check("exactly the 6 walls of the L", R.map.walls.size() == 6, fmt("%zu walls", R.map.walls.size()));
        check("polygon closed", R.poly.closed, R.poly.status);
        const Poly est_world = to_world(R.poly.verts, truth[0]);
        const float h = R.poly.closed ? hausdorff(est_world, room) : 1e9f;
        check("polygon within 5 cm of the truth (Hausdorff)", h < 0.05f, fmt("%.3f m, %zu vertices", h, est_world.size()));
        check("pose error stays small over the loop", R.pose_rmse_xy < 0.05f and R.pose_max_xy < 0.15f,
              fmt("rmse %.3f m, max %.3f m / %.2f deg", R.pose_rmse_xy, R.pose_max_xy, R.pose_max_th * 180.f / kPi));
        check("polygon publishable (every corner sharp enough)", R.poly.publishable,
              fmt("worst corner sigma %.3f m (bar %.3f)", R.poly.worst_corner_sigma, R.map.params.publish_corner_sigma));
        int manhattan = 0;
        for (const auto& w : R.map.walls) if (w.k >= 0) ++manhattan;
        check("every wall classified (k >= 0)", manhattan == static_cast<int>(R.map.walls.size()), fmt("%d / %zu", manhattan, R.map.walls.size()));
    }

    // ═══ 4. Chamfer ═══════════════════════════════════════════════════════════════════════════
    // STRICT MANHATTAN (2026-09-01): this stage estimates the room's MAIN LINES — the polygon is
    // the Manhattan outline (a square corner where the chamfer is), and the chamfer itself is
    // preserved as a strong OBLIQUE CANDIDATE for the later refinement stage.
    std::printf("\n4. A 45-degree chamfer: Manhattan outline now, the chamfer kept for postprocessing\n");
    {
        const Poly room = chamfer_room();
        std::vector<Eigen::Vector3f> truth;
        const std::vector<Eigen::Vector2f> wp = {{-2.5f, -1.5f}, {2.5f, -1.5f}, {2.5f, 1.f}, {-2.5f, 1.5f}, {-2.5f, -1.5f}};
        for (size_t l = 0; l + 1 < wp.size(); ++l)
        {
            const Eigen::Vector2f e = wp[l + 1] - wp[l];
            const float th = std::atan2(e.y(), e.x());
            for (int i = 0; i < 14; ++i)
            {
                const float a = static_cast<float>(i) / 14.f;
                truth.emplace_back(wp[l].x() + a * e.x(), wp[l].y() + a * e.y(), th);
            }
        }
        RunConfig cfg;
        const auto R = run_loop({room}, truth, cfg, rng);
        int off = 0;
        for (const auto& w : R.map.walls) if (w.k < 0) ++off;
        std::printf("    walls=%zu closed=%d status='%s'\n", R.map.walls.size(), R.poly.closed, R.poly.status.c_str());
        for (const auto& w : R.map.walls)
            std::printf("    wall %llu k=%d phi=%.3f d=%.3f frames=%d dF=%.2f\n",
                        static_cast<unsigned long long>(w.id), w.k, w.phi, w.d, w.frames_seen, w.room_factor_dF);
        {
            const Poly ew = to_world(R.poly.verts, truth[0]);
            std::printf("    order:"); for (auto id : R.map.order) std::printf(" %llu", (unsigned long long)id); std::printf("\n    verts:");
            for (const auto& v : ew) std::printf(" (%.2f,%.2f)", v.x(), v.y());
            std::printf("\n");
        }
        check("Manhattan outline: 4 walls, every one classified", R.map.walls.size() == 4 and off == 0,
              fmt("%zu walls, %d without class", R.map.walls.size(), off));
        const Poly est_world = to_world(R.poly.verts, truth[0]);
        const float h = R.poly.closed ? hausdorff(est_world, room) : 1e9f;
        // The square corner lies 0.71 m from the 45-degree chamfer line — that is the lawful cost
        // of the main-lines stage, not an error.
        check("closed within the square-corner bound (0.75 m)", R.poly.closed and h < 0.75f,
              fmt("closed=%d hausdorff=%.3f", R.poly.closed, h));
        {
            int strong_oblique = 0;
            const auto scan_oblique = [&](const auto& pool)
            {
                for (const auto& cnd : pool)
                {
                    if (cnd.npts < 50) continue;
                    float eps = std::numeric_limits<float>::infinity();
                    for (int k = 0; k < 4; ++k)
                        eps = std::min(eps, std::abs(rc::linefit::wrap_pi(
                            cnd.phi - R.map.theta0 - static_cast<float>(k) * kPi * 0.5f)));
                    if (eps > R.map.params.manhattan_gate_rad) ++strong_oblique;
                }
            };
            scan_oblique(R.map.candidates);
            scan_oblique(R.map.corner_residue);   // silenced corner chords land here for postprocessing
            check("the chamfer survives as an oblique candidate for postprocessing", strong_oblique >= 1,
                  fmt("%d strong oblique candidates", strong_oblique));
        }
    }

    // ═══ 5. Structure change after closure ═════════════════════════════════════════════════════
    std::printf("\n5. After closure the room gains a notch (two new walls)\n");
    {
        const auto truth = trajectory(12);
        std::vector<Eigen::Vector3f> twice = truth;
        twice.insert(twice.end(), truth.begin(), truth.end());
        std::vector<Poly> rooms(twice.size(), l_room());
        for (size_t f = truth.size(); f < twice.size(); ++f) rooms[f] = l_room_notch();
        RunConfig cfg;
        const auto R = run_loop(rooms, twice, cfg, rng);
        std::printf("    walls=%zu births=%d closed_at=%d status='%s'\n", R.map.walls.size(), R.births, R.closed_at, R.poly.status.c_str());
        for (const auto& w : R.map.walls)
            std::printf("    wall %llu k=%d phi=%.3f d=%.3f extent[%.2f,%.2f] frames=%d\n",
                        (unsigned long long)w.id, w.k, w.phi, w.d, w.s_min, w.s_max, w.frames_seen);
        {
            const Poly ew = to_world(R.poly.verts, twice[0]);
            std::printf("    order:"); for (auto id : R.map.order) std::printf(" %llu", (unsigned long long)id); std::printf("\n    verts:");
            for (const auto& v : ew) std::printf(" (%.2f,%.2f)", v.x(), v.y());
            std::printf("\n");
        }
        const Poly est_world = to_world(R.poly.verts, twice[0]);
        const float h = R.poly.closed ? hausdorff(est_world, l_room_notch()) : 1e9f;
        check("the two notch walls were born after closure", R.map.walls.size() == 8, fmt("%zu walls (expected 8)", R.map.walls.size()));
        check("polygon re-closed on the notched room within 8 cm", R.poly.closed and h < 0.08f,
              fmt("closed=%d hausdorff=%.3f status='%s'", R.poly.closed, h, R.poly.status.c_str()));
    }

    // ═══ 6. Step-back on the MODEL: a wrong splice (fake notch) heals itself ════════════════════
    std::printf("\n6. Step-back operator: a fake notch spliced into the south wall heals\n");
    {
        const Poly room = l_room();
        const auto truth = trajectory(12);
        RunConfig cfg;
        auto Rr = run_loop({room}, truth, cfg, rng);
        const size_t walls_before = Rr.map.walls.size();
        // Wound: an inward fake notch on the longest wall — [E, jogB, C, jogA, E].
        std::uint64_t host_id = 0; float host_len = 0.f;
        for (const auto& w : Rr.map.walls)
            if (w.has_extent and w.s_max - w.s_min > host_len) { host_len = w.s_max - w.s_min; host_id = w.id; }
        auto* E = Rr.map.find(host_id);
        const float mid = 0.5f * (E->s_min + E->s_max);
        const Eigen::Vector2f tE = E->tangent();
        auto mk = [&](float phi, float d, std::uint64_t id)
        {
            rc::wallmap::WallLandmark w;
            w.id = id; w.phi = phi; w.d = d;
            w.information = Eigen::Vector2f(300.f, 100.f).asDiagonal();
            w.exist_lodds = Rr.map.params.birth_nats;
            return w;
        };
        // C parallel to E, 0.8 m INTO the room; jogs at mid±0.6 along E.
        auto C  = mk(E->phi, E->d + 0.8f, 999);
        C.has_extent = true; C.s_min = mid - 0.6f; C.s_max = mid + 0.6f;
        auto J1 = mk(std::atan2(tE.y(), tE.x()), tE.dot(tE * (mid + 0.6f)), 998);
        auto J0 = mk(std::atan2(-tE.y(), -tE.x()), -(mid - 0.6f), 997);
        // The jogs' extents: between the host line and C, in each jog's own tangent coordinates.
        for (auto* J : {&J1, &J0})
        {
            const Eigen::Vector2f tJ = rc::linefit::tangent_of(J->phi);
            const Eigen::Vector2f nE = E->normal();
            const float sa = tJ.dot(nE * E->d), sb = tJ.dot(nE * C.d);
            J->s_min = std::min(sa, sb); J->s_max = std::max(sa, sb); J->has_extent = true;
        }
        Rr.map.walls.push_back(J1); Rr.map.walls.push_back(C); Rr.map.walls.push_back(J0);
        std::vector<std::uint64_t> no;
        for (auto id : Rr.map.order)
        {
            no.push_back(id);
            if (id == host_id and no.size() and std::find(no.begin(), no.end(), 999ULL) == no.end())
            { no.push_back(998); no.push_back(999); no.push_back(997); no.push_back(host_id); }
        }
        Rr.map.order = no;
        const bool wounded = Rr.map.build_polygon().closed;
        auto R2 = run_loop({room}, truth, cfg, rng, &Rr);
        {
            const Poly ew6 = to_world(R2.poly.verts, truth[0]);
            std::printf("    healed verts:");
            for (const auto& v : ew6) std::printf(" (%.2f,%.2f)", v.x(), v.y());
            std::printf("\n");
            for (const auto& w : R2.map.walls)
                std::printf("    wall %llu k=%d phi=%.3f d=%.3f extent[%.2f,%.2f] frames=%d pts=%d\n",
                            (unsigned long long)w.id, w.k, w.phi, w.d, w.s_min, w.s_max, w.frames_seen, w.points_seen);
            std::printf("    order:");
            for (auto id : R2.map.order) std::printf(" %llu", (unsigned long long)id);
            std::printf("\n");
        }
        const bool fake_gone = R2.map.find(999) == nullptr and R2.map.find(998) == nullptr and R2.map.find(997) == nullptr;
        check("the wounded polygon was valid to start", wounded, "");
        check("the fake notch died and was spliced out", fake_gone,
              fmt("walls %zu -> %zu", walls_before + 3, R2.map.walls.size()));
        const Poly est_world = to_world(R2.poly.verts, truth[0]);
        const float h = R2.poly.closed ? hausdorff(est_world, room) : 1e9f;
        check("the healed polygon matches the room", R2.poly.closed and h < 0.06f,
              fmt("closed=%d hausdorff=%.3f", R2.poly.closed, h));
    }

    }   // end WS_ONLY7 skip

    // ═══ 6b. SPUR: a thin interior wall the boundary must wrap as THREE walls ═══════════════════
    // The user's grammar: one wall toward the room centre, a small perpendicular cap, one going
    // back. Deterministic and fast — the development harness for the wrap operators; runs under
    // WS_ONLY7 too.
    std::printf("\n6b. Spur room: thin interior wall wrapped as [in, cap, back]\n");
    {
        const Poly room = {{-4.f, -3.f}, {4.f, -3.f}, {4.f, 3.f}, {0.08f, 3.f}, {0.08f, 0.5f},
                           {-0.04f, 0.5f}, {-0.04f, 3.f}, {-4.f, 3.f}};
        const std::vector<Eigen::Vector2f> wp = {{-2.5f, 1.5f}, {-2.5f, -1.5f}, {0.f, -2.f},
                                                 {2.5f, -1.5f}, {2.5f, 1.5f}, {1.2f, 2.3f},
                                                 {0.6f, 1.2f}, {1.2f, 2.3f}, {2.5f, 1.5f},
                                                 {0.f, -2.f}, {-2.5f, -1.5f}, {-1.2f, 2.3f},
                                                 {-0.6f, 1.2f}, {-1.2f, 2.3f}, {-2.5f, 1.5f}};
        std::vector<Eigen::Vector3f> truth;
        for (size_t l = 0; l + 1 < wp.size(); ++l)
        {
            const Eigen::Vector2f e = wp[l + 1] - wp[l];
            const float th = std::atan2(e.y(), e.x());
            const int n = std::max(2, static_cast<int>(e.norm() / 0.25f));
            for (int i = 0; i < n; ++i)
                truth.emplace_back(wp[l].x() + e.x() * static_cast<float>(i) / static_cast<float>(n),
                                   wp[l].y() + e.y() * static_cast<float>(i) / static_cast<float>(n), th);
        }
        std::mt19937 rng6(99);
        RunConfig cfg;
        cfg.verbose = true;
        cfg.trace_csv = std::getenv("WS_TRACE_SPUR");
        auto R1 = run_loop({room}, truth, cfg, rng6);
        auto R2 = run_loop({room}, truth, cfg, rng6, &R1);   // second lap: steady state
        const Poly ew = to_world(R2.poly.verts, truth[0]);
        const float h = R2.poly.closed ? hausdorff(ew, room) : 1e9f;
        const float tip_a = R2.poly.closed ? point_to_poly(room[4], ew) : 1e9f;
        const float tip_b = R2.poly.closed ? point_to_poly(room[5], ew) : 1e9f;
        std::printf("    walls=%zu births=%d deaths=%d verts:", R2.map.walls.size(),
                    R1.births + R2.births, R1.deaths + R2.deaths);
        for (const auto& v : ew) std::printf(" (%.2f,%.2f)", v.x(), v.y());
        std::printf("\n");
        for (const auto& w : R2.map.walls)
            std::printf("    wall %llu k=%d phi=%.3f d=%.3f extent[%.2f,%.2f] frames=%d pts=%d\n",
                        (unsigned long long)w.id, w.k, w.phi, w.d, w.s_min, w.s_max, w.frames_seen, w.points_seen);
        check("spur room closed", R2.poly.closed, R2.poly.status);
        check("spur wrapped: both truth tips on the estimate (<12 cm)", tip_a < 0.12f and tip_b < 0.12f,
              fmt("tips %.3f / %.3f m", tip_a, tip_b));
        check("spur room shape within 12 cm (Hausdorff)", h < 0.12f, fmt("%.3f m", h));
    }

    // ═══ 7. THE REAL LAYOUT: apartamento_layout.svg, toured and estimated to convergence ════════
    if (std::getenv("WS_NO7") == nullptr)
    {
    std::printf("\n7. The real apartamento layout (32 vertices incl. trace artefacts)\n");
    {
        // Local SVG polygon read (std::from_chars — the agents' locale rule; no Qt in the harness).
        Poly room;
        Eigen::Vector2f room_centre = Eigen::Vector2f::Zero();   // the SVG→truth-frame shift
        {
            std::ifstream in("/home/pbustos/robocomp/components/active_inference/layouts/apartamento_layout.svg");
            std::stringstream ss; ss << in.rdbuf();
            const std::string svg = ss.str();
            const auto idpos = svg.find("id=\"room_contour\"");
            check("layout file readable", idpos != std::string::npos, "");
            if (idpos != std::string::npos)
            {
                // points="..." nearest to the id, searching the enclosing tag both ways.
                const auto tag0 = svg.rfind('<', idpos);
                const auto tag1 = svg.find('>', idpos);
                const std::string tag = svg.substr(tag0, tag1 - tag0);
                const auto pp = tag.find("points=\"");
                if (pp != std::string::npos)
                {
                    const auto pend = tag.find('"', pp + 8);
                    const std::string pts_str = tag.substr(pp + 8, pend - pp - 8);
                    const char* c = pts_str.data();
                    const char* end = c + pts_str.size();
                    while (c < end)
                    {
                        while (c < end and (*c == ' ' or *c == ',' or *c == '\n' or *c == '\t')) ++c;
                        float x = 0.f, y = 0.f;
                        auto r1 = std::from_chars(c, end, x);
                        if (r1.ec != std::errc{}) break;
                        c = r1.ptr;
                        while (c < end and (*c == ' ' or *c == ',')) ++c;
                        auto r2 = std::from_chars(c, end, y);
                        if (r2.ec != std::errc{}) break;
                        c = r2.ptr;
                        room.emplace_back(x, y);
                    }
                }
            }
            // Recentre on the bbox centre, as the agent does.
            if (not room.empty())
            {
                Eigen::Vector2f lo = room.front(), hi = room.front();
                for (const auto& v : room) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
                const Eigen::Vector2f c0 = 0.5f * (lo + hi);
                for (auto& v : room) v -= c0;
                room_centre = c0;
            }
        }
        float a2 = 0.f;
        for (size_t i = 0; i < room.size(); ++i)
        { const auto& p = room[i]; const auto& q = room[(i + 1) % room.size()]; a2 += p.x() * q.y() - q.x() * p.y(); }
        if (a2 < 0.f) std::reverse(room.begin(), room.end());
        check("layout loaded", room.size() >= 20, fmt("%zu vertices, area %.1f m2", room.size(), std::abs(a2) * 0.5f));

        // An interior tour generated from the polygon itself: inward-offset waypoints on the angle
        // bisectors, greedily connected by legs that stay inside with clearance.
        const auto inside = [&](const Eigen::Vector2f& p, float clear)
        { return rc::corner_visibility::point_in_polygon(p, room) and point_to_poly(p, room) > clear; };
        std::vector<Eigen::Vector2f> wps;
        const int NV = static_cast<int>(room.size());
        for (int i = 0; i < NV; ++i)
        {
            const Eigen::Vector2f prev = room[static_cast<size_t>((i + NV - 1) % NV)];
            const Eigen::Vector2f cur = room[static_cast<size_t>(i)];
            const Eigen::Vector2f next = room[static_cast<size_t>((i + 1) % NV)];
            const Eigen::Vector2f din = (cur - prev).normalized(), dout = (next - cur).normalized();
            Eigen::Vector2f bis = Eigen::Vector2f(-din.y(), din.x()) + Eigen::Vector2f(-dout.y(), dout.x());
            if (bis.norm() < 1e-3f) bis = Eigen::Vector2f(-din.y(), din.x());
            bis.normalize();
            const Eigen::Vector2f w = cur + bis * 0.8f;
            if (inside(w, 0.4f)) wps.push_back(w);
        }
        // Densify: midpoints of consecutive reachable waypoints, so long legs get intermediate views.
        {
            std::vector<Eigen::Vector2f> dense;
            for (size_t j = 0; j < wps.size(); ++j)
            {
                dense.push_back(wps[j]);
                const Eigen::Vector2f mid = 0.5f * (wps[j] + wps[(j + 1) % wps.size()]);
                if (inside(mid, 0.4f)) dense.push_back(mid);
            }
            wps = std::move(dense);
        }
        const auto leg_ok = [&](const Eigen::Vector2f& a, const Eigen::Vector2f& b)
        {
            const float L = (b - a).norm();
            const int n = std::max(2, static_cast<int>(L / 0.1f));
            for (int k = 0; k <= n; ++k)
                if (not inside(a + (b - a) * (static_cast<float>(k) / static_cast<float>(n)), 0.3f)) return false;
            return true;
        };
        std::vector<Eigen::Vector2f> path;
        size_t at = 0;
        path.push_back(wps[0]);
        for (size_t j = 1; j < wps.size(); ++j)
            if (leg_ok(wps[at], wps[j])) { path.push_back(wps[j]); at = j; }
        if (leg_ok(wps[at], wps[0])) path.push_back(wps[0]);
        std::vector<Eigen::Vector3f> truth;
        for (int lap = 0; lap < 2; ++lap)   // two laps: the second closes what the first only glimpsed
            for (size_t l = 0; l + 1 < path.size(); ++l)
            {
                const Eigen::Vector2f e = path[l + 1] - path[l];
                const float th = std::atan2(e.y(), e.x());
                const int n = std::max(2, static_cast<int>(e.norm() / 0.25f));
                for (int i = 0; i < n; ++i)
                {
                    const float a = static_cast<float>(i) / static_cast<float>(n);
                    truth.emplace_back(path[l].x() + a * e.x(), path[l].y() + a * e.y(), th);
                }
            }
        check("tour generated", truth.size() > 60, fmt("%zu waypoints, %zu poses", path.size(), truth.size()));

        RunConfig cfg;
        cfg.n_rays = 720;
        cfg.verbose = false;
        cfg.info_gain = std::getenv("WS_NO_INFOGAIN") == nullptr;
        // EPISTEMIC DRIVE: no scripted tour — the robot goes where the model is uncertain.
        // THREE SEEDS: single runs swing 0.89–0.97 IoU on identical configs; a mechanism is judged
        // on the distribution, never on one draw (the unaligned-measurements lesson).
        (void)truth;
        // ── LEVEL-2 FEATURES of the real layout (SVG coordinates, shifted into the truth frame):
        // what the coarse Manhattan cycle lawfully leaves for the residual pass. Two pillars on a
        // wall (matter the polygon must step around) and one alcove (free space it must step
        // into). Chamfers are not graded yet. The metric is the mis-explained fraction of the
        // feature's own area: |truth Δ estimate| inside the feature box (5 cm pad) over the box.
        struct Feature { const char* name; Eigen::Vector2f lo, hi; };
        const std::vector<Feature> features = {
            {"left pillar (0.32 x 0.60 m)",  Eigen::Vector2f(0.178f, 0.594f) - room_centre, Eigen::Vector2f(0.511f, 1.194f) - room_centre},
            {"right pillar (0.38 x 0.60 m)", Eigen::Vector2f(8.110f, 0.610f) - room_centre, Eigen::Vector2f(8.505f, 1.210f) - room_centre},
            {"alcove (0.45 x 0.85 m)",       Eigen::Vector2f(5.884f, 7.163f) - room_centre, Eigen::Vector2f(6.336f, 8.026f) - room_centre},
        };
        const auto feature_miss = [&](const Poly& est, const Feature& f)
        {
            if (est.size() < 3) return 1.f;
            const float pad = 0.05f, step = 0.02f;
            int miss = 0, n = 0;
            for (float x = f.lo.x() - pad; x <= f.hi.x() + pad; x += step)
                for (float y = f.lo.y() - pad; y <= f.hi.y() + pad; y += step)
                {
                    const Eigen::Vector2f q(x, y);
                    const bool it = rc::corner_visibility::point_in_polygon(q, room);
                    const bool ie = rc::corner_visibility::point_in_polygon(q, est);
                    if (it != ie) ++miss;
                    ++n;
                }
            const float box = (f.hi.x() - f.lo.x()) * (f.hi.y() - f.lo.y());
            return static_cast<float>(miss) * step * step / std::max(box, 1e-6f);
        };
        RunResult R7;
        float best_iou = -1.f;
        std::vector<std::pair<float,float>> per_seed;   // (iou, hausdorff)
        struct RefereeRow { std::string site; bool oracle, inc, fwd; float diou, evidence, cost, fwd_dll, fwd_cost; };
        std::vector<RefereeRow> referee_rows;
        // Three seeds by default (the campaign's distribution); WS_SEEDS=n extends the list, for
        // experiments that need more than an anecdote — the batch-at-saturation measurement below.
        std::vector<unsigned> seed_list{7u, 1001u, 424242u, 5u, 13u, 99u, 777u, 2024u, 31337u, 8675309u, 42u, 6u};
        {
            size_t ns = 3;
            if (const char* e = std::getenv("WS_SEEDS"))
            {
                int v = 0;
                if (std::from_chars(e, e + std::strlen(e), v).ec == std::errc{} and v > 0)
                    ns = std::min<size_t>(static_cast<size_t>(v), seed_list.size());
            }
            seed_list.resize(ns);
        }
        for (unsigned seed : seed_list)
        {
            std::mt19937 rng7(seed);
            // WS_TRACE7 names a per-frame precision trace; one file per seed keeps them apart.
            std::string trace7;
            if (const char* tp = std::getenv("WS_TRACE7"))
            { trace7 = std::string(tp) + "_" + std::to_string(seed) + ".csv"; cfg.trace_csv = trace7.c_str(); }
            std::string poly7;
            if (const char* pp = std::getenv("WS_POLY7"))
            { poly7 = std::string(pp) + "_" + std::to_string(seed) + ".csv"; cfg.poly_csv = poly7.c_str(); }
            int nframes7 = 1100;
            if (const char* e = std::getenv("WS_FRAMES7"))
            { int v = 0; if (std::from_chars(e, e + std::strlen(e), v).ec == std::errc{} and v > 0) nframes7 = v; }
            // ── OBSTACLES (WS_OBSTACLES=1): furniture the LiDAR cannot see through ────────────
            // Half of them stand in the open floor, half sit FLUSH AGAINST a wall — the second kind
            // is what actually tests the estimator, because a wall behind a cabinet is never seen at
            // all in the band, while the room's truth polygon is unchanged. So the score still asks
            // exactly one question: did the walls come back? Placement is deterministic (its own
            // stream, seeded from the run) and rejected unless the whole box lies inside the room.
            cfg.occluders.clear();
            if (std::getenv("WS_OBSTACLES") != nullptr)
            {
                std::mt19937 rgo(31415u);
                const auto U = [&](float a, float b) { return std::uniform_real_distribution<float>(a, b)(rgo); };
                const auto inside_all = [&](const Eigen::Vector2f& lo, const Eigen::Vector2f& hi)
                {
                    for (int c = 0; c < 4; ++c)
                    {
                        const Eigen::Vector2f q((c & 1) ? hi.x() : lo.x(), (c & 2) ? hi.y() : lo.y());
                        if (not rc::corner_visibility::point_in_polygon(q, room)) return false;
                    }
                    return true;
                };
                const auto clear_of_others = [&](const Eigen::Vector2f& lo, const Eigen::Vector2f& hi)
                {
                    for (const auto& [l2, h2] : cfg.occluders)
                        if (lo.x() < h2.x() + 0.3f and l2.x() < hi.x() + 0.3f and
                            lo.y() < h2.y() + 0.3f and l2.y() < hi.y() + 0.3f) return false;
                    return true;
                };
                // free-standing: tables, sofas, a bed
                for (int k = 0, tries = 0; k < 5 and tries < 4000; ++tries)
                {
                    const Eigen::Vector2f lo(U(-4.0f, 3.4f), U(-4.4f, 4.2f));
                    const Eigen::Vector2f hi = lo + Eigen::Vector2f(U(0.5f, 1.2f), U(0.5f, 1.2f));
                    if (not inside_all(lo, hi) or not clear_of_others(lo, hi)) continue;
                    cfg.occluders.push_back({lo, hi}); ++k;
                }
                // against a wall: cupboards and shelves, which HIDE the wall behind them
                for (int k = 0, tries = 0; k < 5 and tries < 6000; ++tries)
                {
                    const size_t e = static_cast<size_t>(U(0.f, static_cast<float>(room.size()) - 0.01f));
                    const Eigen::Vector2f a = room[e], b = room[(e + 1) % room.size()];
                    const float L = (b - a).norm();
                    if (L < 1.4f) continue;
                    const Eigen::Vector2f t = (b - a) / L;
                    const Eigen::Vector2f nn(-t.y(), t.x());               // one side or the other
                    const float w = U(0.6f, 1.3f), dpt = U(0.30f, 0.55f), sc_ = U(0.3f, L - w - 0.3f);
                    const Eigen::Vector2f p0 = a + t * sc_, p1 = p0 + t * w;
                    for (float sgn : {1.f, -1.f})
                    {
                        const Eigen::Vector2f q0 = p0 + nn * (sgn * 0.02f), q1 = p1 + nn * (sgn * dpt);
                        const Eigen::Vector2f lo = q0.cwiseMin(q1), hi = q0.cwiseMax(q1);
                        if (not inside_all(lo, hi) or not clear_of_others(lo, hi)) continue;
                        cfg.occluders.push_back({lo, hi}); ++k; break;
                    }
                }
                std::printf("      obstacles[%u]: %zu boxes (5 free-standing + those that fit against a wall)\n",
                            seed, cfg.occluders.size());
                for (const auto& [lo, hi] : cfg.occluders)
                    std::printf("        box %.2f,%.2f %.2f,%.2f\n", lo.x(), lo.y(), hi.x(), hi.y());
            }
            auto Rx = run_explore(room, cfg, rng7, nframes7, path[0]);
            const Poly ew = to_world(Rx.poly.verts, Eigen::Vector3f(path[0].x(), path[0].y(), 0.f));
            const float iou_x = Rx.poly.closed ? polygon_iou(ew, room) : 0.f;
            const float h_x = Rx.poly.closed ? hausdorff(ew, room) : 1e9f;
            per_seed.push_back({iou_x, h_x});
            std::printf("    seed %-7u frames=%d IoU=%.3f hausdorff=%.3f pose_rmse=%.3f walls=%zu births=%d deaths=%d rejected=%d\n",
                        seed, Rx.frames, iou_x, h_x, Rx.pose_rmse_xy, Rx.map.walls.size(), Rx.births, Rx.deaths, Rx.rejected);
            std::printf("      verts[%u]:", seed);
            for (const auto& v : ew) std::printf(" (%.2f,%.2f)", v.x(), v.y());
            std::printf("\n");
            std::printf("      level-2 features[%u]: mis-explained fraction of each feature's area:", seed);
            for (const auto& f : features) std::printf("  %s %.2f", f.name, Rx.poly.closed ? feature_miss(ew, f) : 1.f);
            std::printf("\n");
            std::printf("      order[%u]:\n", seed);
            for (const auto oid : Rx.map.order)
                for (const auto& w : Rx.map.walls)
                    if (w.id == oid)
                        std::printf("        id=%llu phi=%.3f d=%.3f pts=%d frames=%d lodds=%.1f ext=[%.2f,%.2f]%d\n",
                                    static_cast<unsigned long long>(w.id), w.phi, w.d, w.points_seen,
                                    w.frames_seen, w.exist_lodds, w.s_min, w.s_max, static_cast<int>(w.has_extent));
            float tilt_max = 0.f;
            for (const auto oid : Rx.map.order)
                for (const auto& w : Rx.map.walls)
                    if (w.id == oid and w.k >= 0)
                        tilt_max = std::max(tilt_max, std::abs(rc::linefit::wrap_pi(
                            w.phi - Rx.map.theta0 - static_cast<float>(w.k) * kPi * 0.5f)));
            std::printf("      tilt[%u]: max |phi - theta0 - k*pi/2| = %.4f rad (%.2f deg)\n",
                        seed, tilt_max, tilt_max * 180.f / kPi);
            float pub_tilt = 0.f;
            for (size_t vi = 0; ew.size() >= 2 and vi < ew.size(); ++vi)
            {
                const Eigen::Vector2f e2 = ew[(vi + 1) % ew.size()] - ew[vi];
                if (e2.norm() < 1e-6f) continue;
                const float ang = std::atan2(e2.y(), e2.x());
                float beste = std::numeric_limits<float>::infinity();
                for (int k2 = 0; k2 < 4; ++k2)
                    beste = std::min(beste, std::abs(rc::linefit::wrap_pi(
                        ang - Rx.map.theta0 - static_cast<float>(k2) * kPi * 0.5f)));
                pub_tilt = std::max(pub_tilt, beste);
            }
            std::printf("      published-tilt[%u]: max edge off-axis = %.4f rad (%.2f deg)\n",
                        seed, pub_tilt, pub_tilt * 180.f / kPi);
            // INTERNAL Manhattan-ness: the same maximum, but against the polygon's OWN axis frame
            // (the length-weighted circular mean of its edge directions mod 90 deg) instead of the
            // live theta0. The projection makes the published polygon internally rectilinear; any
            // gap between these two numbers is the copy's theta0' minus the live theta0 — review #6.
            {
                float sx = 0.f, sy = 0.f;
                for (size_t vi = 0; ew.size() >= 2 and vi < ew.size(); ++vi)
                {
                    const Eigen::Vector2f e2 = ew[(vi + 1) % ew.size()] - ew[vi];
                    const float L2 = e2.norm();
                    if (L2 < 1e-6f) continue;
                    const float a4 = 4.f * std::atan2(e2.y(), e2.x());   // mod 90 deg -> full turn
                    sx += L2 * std::cos(a4); sy += L2 * std::sin(a4);
                }
                const float own = std::atan2(sy, sx) / 4.f;
                float own_tilt = 0.f;
                for (size_t vi = 0; ew.size() >= 2 and vi < ew.size(); ++vi)
                {
                    const Eigen::Vector2f e2 = ew[(vi + 1) % ew.size()] - ew[vi];
                    if (e2.norm() < 1e-6f) continue;
                    const float ang = std::atan2(e2.y(), e2.x());
                    float b2 = std::numeric_limits<float>::infinity();
                    for (int k2 = -2; k2 <= 2; ++k2)
                        b2 = std::min(b2, std::abs(rc::linefit::wrap_pi(ang - own - static_cast<float>(k2) * kPi * 0.5f)));
                    own_tilt = std::max(own_tilt, b2);
                }
                std::printf("      internal-tilt[%u]: max edge off its OWN axes = %.4f rad (%.2f deg); frame gap theta0'-theta0 = %.2f deg\n",
                            seed, own_tilt, own_tilt * 180.f / kPi,
                            std::abs(rc::linefit::wrap_pi(own - Rx.map.theta0)) * 180.f / kPi);
            }
            {
                // Per-seed diagnostics: is the deep spur resolved (grid cells on its two truth
                // faces; distance of the truth tip vertices to the estimate), and which frontiers
                // refuse to close (they keep the coverage gate armed for ever).
                const Eigen::Vector2f o = path[0];
                int occ = 0, fre = 0, unk = 0, n = 0;
                for (const auto& pr : {std::make_pair(room[21], room[22]), std::make_pair(room[23], room[24])})
                    for (int k2 = 0; k2 <= 20; ++k2)
                    {
                        const Eigen::Vector2f pt = pr.first + (pr.second - pr.first) * (static_cast<float>(k2) / 20.f) - o;
                        const int i2 = static_cast<int>((pt.x() - Rx.map.fgrid.x0) / Rx.map.fgrid.cell);
                        const int j2 = static_cast<int>((pt.y() - Rx.map.fgrid.y0) / Rx.map.fgrid.cell);
                        if (not Rx.map.fgrid.in(i2, j2)) continue;
                        ++n;
                        const float l = Rx.map.fgrid.lodds[static_cast<size_t>(Rx.map.fgrid.idx(i2, j2))];
                        if (l > 1.f) ++occ; else if (l < -1.f) ++fre; else ++unk;
                    }
                const float tip_a = Rx.poly.closed ? point_to_poly(room[22], ew) : 1e9f;
                const float tip_b = Rx.poly.closed ? point_to_poly(room[23], ew) : 1e9f;
                const auto fl = Rx.map.frontiers();
                std::printf("      diag[%u]: spur cells occ/free/unk=%d/%d/%d of %d; tip->est %.2f / %.2f m; frontiers=%zu weak=%zu\n",
                            seed, occ, fre, unk, n, tip_a, tip_b, fl.size(), Rx.map.weak_matter().size());
                {
                    // The SHORT spur (truth verts 2..5): same instrumentation as the deep one.
                    int occ2 = 0, fre2 = 0, unk2 = 0, n2 = 0;
                    for (const auto& pr : {std::make_pair(room[2], room[3]), std::make_pair(room[4], room[5])})
                        for (int k2 = 0; k2 <= 20; ++k2)
                        {
                            const Eigen::Vector2f pt = pr.first + (pr.second - pr.first) * (static_cast<float>(k2) / 20.f) - o;
                            const int i2 = static_cast<int>((pt.x() - Rx.map.fgrid.x0) / Rx.map.fgrid.cell);
                            const int j2 = static_cast<int>((pt.y() - Rx.map.fgrid.y0) / Rx.map.fgrid.cell);
                            if (not Rx.map.fgrid.in(i2, j2)) continue;
                            ++n2;
                            const float l = Rx.map.fgrid.lodds[static_cast<size_t>(Rx.map.fgrid.idx(i2, j2))];
                            if (l > 1.f) ++occ2; else if (l < -1.f) ++fre2; else ++unk2;
                        }
                    const float stip_a = Rx.poly.closed ? point_to_poly(room[3], ew) : 1e9f;
                    const float stip_b = Rx.poly.closed ? point_to_poly(room[4], ew) : 1e9f;
                    std::printf("      diag[%u] SHORT spur: cells occ/free/unk=%d/%d/%d of %d; tip->est %.2f / %.2f m\n",
                                seed, occ2, fre2, unk2, n2, stip_a, stip_b);
                }
                for (const auto& fp : fl)
                    std::printf("        frontier world(%.2f,%.2f)\n", fp.x() + o.x(), fp.y() + o.y());
                for (const auto& w : Rx.map.walls)
                    std::printf("        wall %llu k=%d phi=%.3f d=%.3f extent[%.2f,%.2f] frames=%d pts=%d\n",
                                (unsigned long long)w.id, w.k, w.phi, w.d, w.s_min, w.s_max, w.frames_seen, w.points_seen);
                std::printf("        order[%u]:", seed);
                for (auto id : Rx.map.order) std::printf(" %llu", (unsigned long long)id);
                std::printf("\n");
            }
            // ── BATCH PASS AT SATURATION (experiment 2026-09-03): the run is over, the grid holds
            // everything the robot ever saw. Re-derive the cycle from that final grid, repeatedly,
            // until nothing more is adopted, and grade the result. This asks whether the spread
            // between seeds is a DATA difference or a PATH difference: if a seed that finished at
            // 0.889 re-derives to ~0.96 from its own final grid, the online commitments were the
            // cost, not the evidence. Judges: the incumbent (grid-IoU margin + surrender veto),
            // the same with self-crossing repair, and "adopt any closed cycle" — which shows what
            // the contour itself contains, with no judge in the way.
            {
                const Eigen::Vector3f org(path[0].x(), path[0].y(), 0.f);
                std::printf("      batch pass on the FINAL grid[%u]  (online result: IoU %.3f, Hausdorff %.3f m)\n",
                            seed, iou_x, h_x);
                for (const auto& [name, judge, repair] : std::vector<std::tuple<const char*, int, bool>>{
                        {"incumbent judge          ", 0, false},
                        {"incumbent + repair       ", 0, true},
                        {"adopt any closed cycle   ", 2, false},
                        {"adopt any closed + repair", 2, true}})
                {
                    rc::wallmap::WallMap m = Rx.map;
                    m.params.adopt_judge = judge;
                    m.params.adopt_repair = repair;
                    m.params.forward_referee = false;
                    m.decisions.clear();
                    int adopted = 0;
                    for (int k = 0; k < 12; ++k) { if (not m.re_derive(Rx.last_xy)) break; ++adopted; }
                    const auto pb = m.manhattan_polygon();
                    const Poly bw = to_world(pb.verts, org);
                    const float iou_b = pb.closed ? polygon_iou(bw, room) : 0.f;
                    const float h_b = pb.closed ? hausdorff(bw, room) : 1e9f;
                    std::printf("        %s adopted %2d -> IoU %.3f (%+.3f)  Hausdorff %.3f m  walls %2zu\n",
                                name, adopted, iou_b, iou_b - iou_x, h_b, m.walls.size());
                }
            }
            // ── CAN A JUDGE PICK THE BETTER OF THE TWO AT SATURATION? The run is over, so there
            // is no churn to fear: this is one static choice between two complete polygons, the
            // online cycle and the batch cycle re-derived from the final grid. Score both under
            // the forward beam model against the code length of the extra edges, and compare the
            // verdict with the truth. This is the question the per-decision referee could not
            // answer, because there every verdict fed back into the map's dynamics.
            if (not Rx.map.beams.empty())
            {
                float prof_moved = 0.f, prof_signed = 0.f;
                rc::wallmap::WallMap mb = Rx.map;
                mb.params.adopt_judge = 2; mb.params.adopt_repair = true;
                mb.params.forward_referee = false; mb.beams.clear(); mb.decisions.clear();
                for (int k = 0; k < 12; ++k) if (not mb.re_derive(Rx.last_xy)) break;
                const auto pb = mb.manhattan_polygon();
                if (pb.closed and Rx.poly.closed)
                {
                    const Eigen::Vector3f org3(path[0].x(), path[0].y(), 0.f);
                    const Poly bw = to_world(pb.verts, org3);
                    const float iou_b = polygon_iou(bw, room);
                    const float dll = Rx.map.forward_delta(Rx.poly.verts, pb.verts);
                    const float code = (static_cast<float>(pb.verts.size()) - static_cast<float>(Rx.poly.verts.size()))
                                     * Rx.map.edge_code_nats(true);
                    const bool judge_takes_batch = dll > code;
                    const bool truth_prefers_batch = iou_b > iou_x;
                    std::printf("        saturation choice: online %.3f vs batch %.3f | forward dll %+.1f vs code %+.1f -> take %s | truth prefers %s | %s\n",
                                iou_x, iou_b, dll, code, judge_takes_batch ? "BATCH " : "online",
                                truth_prefers_batch ? "BATCH " : "online",
                                judge_takes_batch == truth_prefers_batch ? "AGREE" : "WRONG");

                    // ── PROFILE OUT THE OFFSETS, then compare again. A polygon's beam likelihood is
                    // dominated by where its edges SIT (the grid contour's walls are a cell off the
                    // returns, the online walls are line-fitted), which buries the topology by four
                    // orders of magnitude. So slide every edge along its own normal onto the returns
                    // assigned to it — a nuisance parameter per edge, profiled out in the likelihood
                    // sense — and only then score. Rectilinearity is preserved: the lines keep their
                    // directions and the vertices are re-intersected.
                    const auto profile = [&](const Poly& poly)
                    {
                        const size_t N = poly.size();
                        if (N < 3) return poly;
                        std::vector<Eigen::Vector2f> nrm(N);
                        std::vector<float> cst(N);
                        for (size_t e = 0; e < N; ++e)
                        {
                            const Eigen::Vector2f dvec = poly[(e + 1) % N] - poly[e];
                            const float L = dvec.norm();
                            nrm[e] = L > 1e-9f ? Eigen::Vector2f(-dvec.y() / L, dvec.x() / L) : Eigen::Vector2f(1.f, 0.f);
                            cst[e] = nrm[e].dot(poly[e]);
                        }
                        std::vector<std::vector<float>> res(N);
                        for (const auto& b : Rx.map.beams)
                        {
                            const Eigen::Vector2f q = b.o + b.d * b.r;
                            int best = -1; float bd = 0.25f;
                            for (size_t e = 0; e < N; ++e)
                            {
                                const Eigen::Vector2f a = poly[e], ab = poly[(e + 1) % N] - a;
                                const float l2 = ab.squaredNorm();
                                const float tt = l2 > 1e-9f ? std::clamp((q - a).dot(ab) / l2, 0.f, 1.f) : 0.f;
                                const float d2 = (q - (a + tt * ab)).norm();
                                if (d2 < bd) { bd = d2; best = static_cast<int>(e); }
                            }
                            if (best >= 0) res[static_cast<size_t>(best)].push_back(nrm[static_cast<size_t>(best)].dot(q) - cst[static_cast<size_t>(best)]);
                        }
                        float moved = 0.f, signed_sum = 0.f; int nmoved = 0;
                        for (size_t e = 0; e < N; ++e)
                            if (res[e].size() >= 20)
                            {
                                std::nth_element(res[e].begin(), res[e].begin() + static_cast<long>(res[e].size() / 2), res[e].end());
                                const float off = res[e][res[e].size() / 2];
                                cst[e] += off; moved += std::abs(off); signed_sum += off; ++nmoved;
                            }
                        Poly out(N);
                        for (size_t e = 0; e < N; ++e)
                        {
                            const size_t pv = (e + N - 1) % N;
                            const float cr = nrm[pv].x() * nrm[e].y() - nrm[pv].y() * nrm[e].x();
                            if (std::abs(cr) < 1e-6f) { out[e] = poly[e]; continue; }
                            out[e] = Eigen::Vector2f((cst[pv] * nrm[e].y() - cst[e] * nrm[pv].y()) / cr,
                                                     (cst[e] * nrm[pv].x() - cst[pv] * nrm[e].x()) / cr);
                        }
                        prof_moved = nmoved > 0 ? moved / static_cast<float>(nmoved) : 0.f;
                        // Sign convention: nrm is left-of-travel on a CCW cycle, i.e. the INTERIOR
                        // side. A positive median residual means the returns lie inside the edge —
                        // the polygon is too big there; negative means it is too small.
                        prof_signed = nmoved > 0 ? signed_sum / static_cast<float>(nmoved) : 0.f;
                        return out;
                    };
                    const auto score = [&](const Poly& poly)
                    {
                        double sum = 0.0;
                        for (const auto& b : Rx.map.beams) sum += static_cast<double>(Rx.map.beam_loglik(b, poly));
                        return sum;
                    };
                    const Poly on_p = profile(Rx.poly.verts);  const float mv_on = prof_moved, sg_on = prof_signed;
                    const Poly ba_p = profile(pb.verts);       const float mv_ba = prof_moved, sg_ba = prof_signed;
                    const float iou_on_p = polygon_iou(to_world(on_p, org3), room);
                    const float iou_ba_p = polygon_iou(to_world(ba_p, org3), room);
                    const double dll_p = score(ba_p) - score(on_p);
                    const bool takes_batch_p = dll_p > static_cast<double>(code);
                    const bool truth_p = iou_ba_p > iou_on_p;
                    {
                        const Poly ow = to_world(on_p, org3);
                        std::printf("        profiled-verts[%u]:", seed);
                        for (const auto& v : ow) std::printf(" (%.3f,%.3f)", v.x(), v.y());
                        std::printf("\n");
                    }
                    std::printf("        profiled:          online %.3f vs batch %.3f | edges moved %.3f (signed %+.3f) / %.3f (signed %+.3f) m | dll %+.1f vs code %+.1f -> take %s | truth prefers %s | %s\n",
                                iou_on_p, iou_ba_p, mv_on, sg_on, mv_ba, sg_ba, dll_p, code,
                                takes_batch_p ? "BATCH " : "online", truth_p ? "BATCH " : "online",
                                takes_batch_p == truth_p ? "AGREE" : "WRONG");
                }
            }
            // ── THE REFEREE'S REPORT CARD: every judged structure change, both judges against the
            // truth. The oracle is the IoU with the real layout: a trial was RIGHT iff it raised it.
            {
                struct Tally { int n = 0, inc_ok = 0, fwd_ok = 0, disagree = 0, fwd_right_inc_wrong = 0, inc_right_fwd_wrong = 0; };
                std::map<std::string, Tally> tally;
                const Eigen::Vector3f org2(path[0].x(), path[0].y(), 0.f);
                for (const auto& d : Rx.map.decisions)
                {
                    const float iou_c = polygon_iou(to_world(d.cur, org2), room);
                    const float iou_t = polygon_iou(to_world(d.trial, org2), room);
                    const bool oracle = iou_t > iou_c + 1e-4f;
                    const bool inc = d.accepted, fwd = d.fwd_dll > d.fwd_cost;
                    auto& t = tally[d.site];
                    ++t.n; t.inc_ok += (inc == oracle); t.fwd_ok += (fwd == oracle); t.disagree += (inc != fwd);
                    t.fwd_right_inc_wrong += (fwd == oracle and inc != oracle);
                    t.inc_right_fwd_wrong += (inc == oracle and fwd != oracle);
                    referee_rows.push_back({std::string(d.site), oracle, inc, fwd, iou_t - iou_c, d.evidence, d.cost, d.fwd_dll, d.fwd_cost});
                }
                std::printf("      referee[%u]: %zu beams stored, %zu decisions\n", seed, Rx.map.beams.size(), Rx.map.decisions.size());
                for (const auto& [site, t] : tally)
                    std::printf("        %-7s n=%3d  incumbent right %3d  forward right %3d  disagree %3d  (forward right & incumbent wrong %d, the reverse %d)\n",
                                site.c_str(), t.n, t.inc_ok, t.fwd_ok, t.disagree, t.fwd_right_inc_wrong, t.inc_right_fwd_wrong);
            }
            if (iou_x > best_iou) { best_iou = iou_x; R7 = std::move(Rx); }
        }
        {
            // Across seeds: where the judges disagree, who was right, and by how much IoU.
            int n = 0, inc_ok = 0, fwd_ok = 0, dis = 0, dis_fwd = 0, dis_inc = 0;
            float iou_when_fwd_right = 0.f, iou_when_inc_right = 0.f;
            for (const auto& r : referee_rows)
            {
                ++n; inc_ok += (r.inc == r.oracle); fwd_ok += (r.fwd == r.oracle);
                if (r.inc != r.fwd)
                {
                    ++dis;
                    if (r.fwd == r.oracle) { ++dis_fwd; iou_when_fwd_right += std::abs(r.diou); }
                    else if (r.inc == r.oracle) { ++dis_inc; iou_when_inc_right += std::abs(r.diou); }
                }
            }
            std::printf("    referee across seeds: %d decisions, incumbent right %d (%.0f%%), forward right %d (%.0f%%); disagreements %d — forward right in %d (Σ|ΔIoU| %.3f), incumbent right in %d (Σ|ΔIoU| %.3f)\n",
                        n, inc_ok, 100.f * inc_ok / std::max(n, 1), fwd_ok, 100.f * fwd_ok / std::max(n, 1),
                        dis, dis_fwd, iou_when_fwd_right, dis_inc, iou_when_inc_right);
            std::ofstream csv("/tmp/wall_slam_referee.csv");
            csv.imbue(std::locale::classic());
            csv << "site,oracle,incumbent,forward,diou,evidence,cost,fwd_dll,fwd_cost\n";
            for (const auto& r : referee_rows)
                csv << r.site << ',' << r.oracle << ',' << r.inc << ',' << r.fwd << ',' << r.diou << ',' << r.evidence << ',' << r.cost << ',' << r.fwd_dll << ',' << r.fwd_cost << '\n';
        }
        float iou_min = 2.f, iou_med = 0.f;
        { std::vector<float> v; for (auto& q : per_seed) v.push_back(q.first);
          std::sort(v.begin(), v.end()); iou_min = v.front(); iou_med = v[v.size() / 2]; }
        std::printf("    across seeds: IoU min=%.3f median=%.3f best=%.3f\n", iou_min, iou_med, best_iou);
        std::printf("    explorer finished after %d frames; frontiers left=%zu\n", R7.frames, R7.map.frontiers().size());
        {
            // Does the grid RESOLVE the thin interior wall? Sample along the truth spur's two faces.
            const Eigen::Vector2f o = path[0];
            int occ = 0, fre = 0, unk = 0, n = 0;
            for (const auto& pr : {std::make_pair(room[21], room[22]), std::make_pair(room[23], room[24])})
                for (int k = 0; k <= 20; ++k)
                {
                    const Eigen::Vector2f pt = pr.first + (pr.second - pr.first) * (static_cast<float>(k) / 20.f) - o;
                    const int i = static_cast<int>((pt.x() - R7.map.fgrid.x0) / R7.map.fgrid.cell);
                    const int j = static_cast<int>((pt.y() - R7.map.fgrid.y0) / R7.map.fgrid.cell);
                    if (not R7.map.fgrid.in(i, j)) continue;
                    ++n;
                    const float l = R7.map.fgrid.lodds[static_cast<size_t>(R7.map.fgrid.idx(i, j))];
                    if (l > 1.f) ++occ; else if (l < -1.f) ++fre; else ++unk;
                }
            std::printf("    spur-face grid cells: occupied %d, FREE %d, unknown %d of %d — free>0 means beams carved through the thin wall\n",
                        occ, fre, unk, n);
        }
        std::printf("    births=%d deaths=%d splice_rejected=%d\n", R7.births, R7.deaths, R7.rejected);
        {
            auto cs = R7.map.candidates;
            std::sort(cs.begin(), cs.end(), [](const auto& a, const auto& b) { return a.npts > b.npts; });
            for (size_t i = 0; i < std::min<size_t>(10, cs.size()); ++i)
                std::printf("    cand phi=%.3f d=%.3f extent[%.2f,%.2f] npts=%d frames=%d gain=%.0f\n",
                            cs[i].phi, cs[i].d, cs[i].s_min, cs[i].s_max, cs[i].npts, cs[i].frames, cs[i].gain);
        }
        for (const auto& w : R7.map.walls)
            std::printf("    wall %llu k=%d phi=%.3f d=%.3f extent[%.2f,%.2f] frames=%d pts=%d lodds=%.1f\n",
                        (unsigned long long)w.id, w.k, w.phi, w.d, w.s_min, w.s_max, w.frames_seen, w.points_seen, w.exist_lodds);
        std::printf("    order:");
        for (auto id : R7.map.order) std::printf(" %llu", (unsigned long long)id);
        std::printf("\n    verts:");
        for (const auto& v : R7.poly.verts) std::printf(" (%.2f,%.2f)", v.x(), v.y());
        std::printf("\n");
        std::printf("    walls=%zu candidates=%zu births=%d closed_at=%d status='%s' worst corner sigma=%.3f\n",
                    R7.map.walls.size(), R7.map.candidates.size(), R7.births, R7.closed_at,
                    R7.poly.status.c_str(), R7.poly.worst_corner_sigma);
        const Poly est_world = to_world(R7.poly.verts, Eigen::Vector3f(path[0].x(), path[0].y(), 0.f));
        const float h = R7.poly.closed ? hausdorff(est_world, room) : 1e9f;
        // Per-vertex nearest-boundary error of the TRUTH against the estimate: where is it worst?
        float worst_v = 0.f; int worst_i = -1;
        for (size_t i = 0; i < room.size(); ++i)
        {
            const float d = R7.poly.closed ? point_to_poly(room[i], est_world) : 1e9f;
            if (d > worst_v) { worst_v = d; worst_i = static_cast<int>(i); }
        }
        const float iou = R7.poly.closed ? polygon_iou(est_world, room) : 0.f;
        const float sym_diff = (1.f - iou) * 60.5f / std::max(iou, 1e-3f) * iou;   // ≈ union·(1−IoU) m²
        std::printf("    hausdorff=%.3f m; IoU=%.3f (sym diff ~%.1f m2); worst truth vertex #%d off by %.3f m; pose rmse %.3f max %.3f m\n",
                    h, iou, sym_diff, worst_i, worst_v, R7.pose_rmse_xy, R7.pose_max_xy);
        // ── WHERE DOES THE 5 cm INWARD BIAS LIVE? The published polygon's edges sit ~5 cm inside
        // the returns. Test the WALL LINES themselves, before projection and decoration: for each
        // wall in the cycle, the median signed residual of the beam endpoints near its own line and
        // inside its own extent (the normal points INTO the room, so negative = the returns are
        // outside the line = the wall is too far in). If the walls are unbiased the bias is made by
        // the projection or by level 2; if they are biased it is upstream, in association or in what
        // the contour adoption creates.
        if (not R7.map.beams.empty())
        {
            std::printf("    wall-line bias (best seed): median signed residual of the returns each wall owns\n");
            std::vector<float> per_wall; std::vector<int> pts_of;
            for (const auto id : R7.map.order)
            {
                const auto* w = R7.map.find(id);
                if (w == nullptr) continue;
                if (std::find(R7.map.order.begin(), R7.map.order.end(), id) != std::find(R7.map.order.begin(), R7.map.order.end(), id)) {}
                const Eigen::Vector2f nn = w->normal(), tv = w->tangent();
                std::vector<float> res;
                for (const auto& b : R7.map.beams)
                {
                    const Eigen::Vector2f q = b.o + b.d * b.r;
                    const float r = nn.dot(q) - w->d;
                    if (std::abs(r) > 0.25f) continue;
                    const float sc = tv.dot(q);
                    if (sc < w->s_min or sc > w->s_max) continue;
                    res.push_back(r);
                }
                if (res.size() < 50) continue;
                std::nth_element(res.begin(), res.begin() + static_cast<long>(res.size() / 2), res.end());
                const float med = res[res.size() / 2];
                per_wall.push_back(med); pts_of.push_back(w->points_seen);
                std::printf("      wall %-6llu k=%d pts=%-7d frames=%-5d beams=%-6zu median residual %+.3f m\n",
                            static_cast<unsigned long long>(w->id), w->k, w->points_seen, w->frames_seen,
                            res.size(), med);
            }
            if (not per_wall.empty())
            {
                std::vector<float> sorted = per_wall;
                std::sort(sorted.begin(), sorted.end());
                float mean = 0.f; for (float v : per_wall) mean += v; mean /= static_cast<float>(per_wall.size());
                int well = 0; float mean_well = 0.f;
                for (size_t k = 0; k < per_wall.size(); ++k)
                    if (pts_of[k] > 5000) { ++well; mean_well += per_wall[k]; }
                std::printf("      %zu walls: mean %+.3f m, median %+.3f m | of these %d have >5000 points, mean %+.3f m\n",
                            per_wall.size(), mean, sorted[sorted.size() / 2], well,
                            well > 0 ? mean_well / static_cast<float>(well) : 0.f);
            }
        }
        check("polygon closed on the real layout", R7.poly.closed, R7.poly.status);
        // 0.20 m bar: the SVG itself carries 6-15 cm trace artefacts the estimator may lawfully
        // smooth over; a real miss (a whole alcove) is metres.
        check("estimate within 20 cm of the real layout (Hausdorff)", h < 0.20f, fmt("%.3f m", h));
        check("estimate overlaps the real layout (IoU >= 0.95)", iou >= 0.95f, fmt("IoU %.3f", iou));
        // LEVEL-2 pre-registration (2026-09-02): the residual pass is graded on the real layout's own
        // small features. A feature counts as explained when less than a third of its area is
        // mis-explained; the coarse cycle alone leaves each of them essentially whole (~1.0).
        for (const auto& f : features)
        {
            const float m = feature_miss(est_world, f);
            check(fmt("level-2: %s explained", f.name).c_str(), m < 0.33f, fmt("mis-explained %.2f of its area", m));
        }
        check("pose stayed on track through the tour", R7.pose_rmse_xy < 0.08f,
              fmt("rmse %.3f m, max %.3f m", R7.pose_rmse_xy, R7.pose_max_xy));
        // RE-ANCHOR (the agent does this once, the bench never did): the grid must move with the
        // walls. Measured by the fraction of grid cells inside the published polygon that are free,
        // before and after a re-anchor by the polygon's centre and 0.3 rad — the agreement of the
        // grid with the polygon is frame-invariant iff the grid was transformed.
        {
            const auto free_inside = [](const rc::wallmap::WallMap& m)
            {
                const auto poly = m.manhattan_polygon();
                long in = 0, fr = 0;
                for (int i = 0; i < m.fgrid.nx; ++i)
                    for (int j = 0; j < m.fgrid.ny; ++j)
                        if (rc::corner_visibility::point_in_polygon(m.fgrid.at(i, j), poly.verts))
                        { ++in; if (m.fgrid.is_free(i, j)) ++fr; }
                return in > 0 ? static_cast<float>(fr) / static_cast<float>(in) : 0.f;
            };
            rc::wallmap::WallMap m2 = R7.map;
            const float before = free_inside(m2);
            Eigen::Vector2f cc = Eigen::Vector2f::Zero();
            for (const auto& v : R7.poly.verts) cc += v;
            cc /= static_cast<float>(std::max<size_t>(1, R7.poly.verts.size()));
            m2.reanchor(cc, 0.3f);
            const float after = free_inside(m2);
            check("re-anchor keeps the grid aligned with the walls", std::abs(after - before) < 0.03f,
                  fmt("free fraction inside the polygon %.3f -> %.3f", before, after));
        }
    }
    }   // end WS_NO7 skip

    // ═══ 8. RANDOM ROOMS: wall columns, alcoves, corner columns, spurs ══════════════════════════
    // A separate population test (WS_ROOMS=n, default off so the standard bench stays at ~12 s).
    // Each room is a rectangle carrying a random set of the four feature kinds the model claims to
    // handle, all Manhattan and non-overlapping by construction. Every room is graded twice: the
    // whole layout (IoU, Hausdorff) and each feature on its own (the mis-explained fraction of its
    // area, the same measure and the same 0.33 bar as the real apartamento's features).
    if (const char* rooms_env = std::getenv("WS_ROOMS"))
    {
        int n_rooms = 50;
        { int v = 0; if (std::from_chars(rooms_env, rooms_env + std::strlen(rooms_env), v).ec == std::errc{} and v > 0) n_rooms = v; }
        std::printf("\n8. %d random rooms (wall column, alcove, corner column, spur)\n", n_rooms);
        struct Feat { int kind; Eigen::Vector2f lo, hi; };   // 0 column, 1 alcove, 2 corner, 3 spur
        static const char* kind_name[4] = {"wall column", "alcove", "corner column", "spur"};
        int found[4] = {0, 0, 0, 0}, total[4] = {0, 0, 0, 0};
        std::vector<float> ious, hauss;
        int only_room = -1;
        if (const char* e = std::getenv("WS_ROOM_ONLY"))
        { int v = 0; if (std::from_chars(e, e + std::strlen(e), v).ec == std::errc{}) only_room = v; }
        for (int r = 0; r < n_rooms; ++r)
        {
            if (only_room >= 0 and r != only_room) continue;
            std::mt19937 rg(9000u + static_cast<unsigned>(r));
            const auto U = [&](float a, float b) { return std::uniform_real_distribution<float>(a, b)(rg); };
            const float W = U(6.f, 11.f), H = U(5.f, 9.f);
            // Base outline: a rectangle, or (40%) an L — a rectangle with one corner quadrant cut
            // away, both legs at least a third of the room wide, rotated onto a random corner.
            const bool ell = U(0.f, 1.f) < 0.4f;
            std::vector<Eigen::Vector2f> base;
            if (not ell) base = {{0.f, 0.f}, {W, 0.f}, {W, H}, {0.f, H}};
            else
            {
                const float cx = U(0.40f, 0.65f) * W, cy = U(0.40f, 0.65f) * H;
                base = {{0.f, 0.f}, {W, 0.f}, {W, cy}, {cx, cy}, {cx, H}, {0.f, H}};
                const int turns = std::uniform_int_distribution<int>(0, 3)(rg);
                for (int q = 0; q < turns; ++q)
                {
                    for (auto& v : base) v = Eigen::Vector2f(-v.y(), v.x());   // rotate 90 deg CCW
                    Eigen::Vector2f mn = base.front();
                    for (const auto& v : base) mn = mn.cwiseMin(v);
                    for (auto& v : base) v -= mn;
                }
            }
            const int NW = static_cast<int>(base.size());
            std::vector<Eigen::Vector2f> V(base), T(NW), N(NW);
            std::vector<float> LEN(NW);
            for (int k = 0; k < NW; ++k)
            {
                const Eigen::Vector2f d = base[(k + 1) % NW] - base[k];
                LEN[k] = d.norm(); T[k] = d / LEN[k]; N[k] = Eigen::Vector2f(-T[k].y(), T[k].x());
            }
            std::vector<float> corner(NW, 0.f);
            std::vector<Feat> feats;
            for (int k = 0; k < NW; ++k)
            {
                const Eigen::Vector2f& tp = T[(k + NW - 1) % NW];
                const bool convex = tp.x() * T[k].y() - tp.y() * T[k].x() > 0.f;   // CCW left turn
                if (convex and U(0.f, 1.f) < 0.35f) corner[k] = U(0.3f, 0.8f);
            }
            Poly room;
            for (int w = 0; w < NW; ++w)
            {
                const int wp = (w + NW - 1) % NW;
                if (corner[w] > 0.f)
                {
                    const float c = corner[w];
                    const Eigen::Vector2f p0 = V[w] - T[wp] * c, p1 = p0 + T[w] * c, p2 = V[w] + T[w] * c;
                    room.push_back(p0); room.push_back(p1); room.push_back(p2);
                    feats.push_back({2, p0.cwiseMin(p2), p0.cwiseMax(p2)});
                }
                else room.push_back(V[w]);
                // Features along wall w, left to right, never overlapping and clear of both corners.
                float s = std::max(corner[w], 0.f) + 0.6f;
                const float s_end = LEN[w] - std::max(corner[(w + 1) % NW], 0.f) - 0.6f;
                while (s < s_end - 0.5f)
                {
                    const float roll = U(0.f, 1.f);
                    if (roll > 0.55f) { s += U(0.8f, 2.5f); continue; }   // a plain stretch of wall
                    int kind; float wid, dep;
                    if (roll < 0.20f)      { kind = 0; wid = U(0.30f, 0.80f); dep = U(0.20f, 0.50f); }
                    else if (roll < 0.42f) { kind = 1; wid = U(0.50f, 1.50f); dep = U(0.30f, 0.80f); }
                    else                   { kind = 3; wid = U(0.10f, 0.16f); dep = U(1.00f, 2.60f); }
                    if (s + wid > s_end) break;
                    // An inward feature may not reach across the room (an L's leg can be narrow):
                    // cast from the middle of its base into the room and keep well short of what it
                    // hits. Outward features only have to clear the corners, which they already do.
                    if (kind != 1)
                    {
                        const Eigen::Vector2f mid = V[w] + T[w] * (s + 0.5f * wid) + N[w] * 0.01f;
                        float reach = 1e9f;
                        for (int q = 0; q < NW; ++q)
                            if (const auto tt = rc::corner_visibility::ray_segment_t(mid, N[w], base[q], base[(q + 1) % NW]);
                                tt and *tt > 1e-3f) reach = std::min(reach, *tt);
                        dep = std::min(dep, 0.6f * reach);
                        if (dep < 0.18f) { s += wid + 0.4f; continue; }
                    }
                    const float sg = (kind == 1) ? -1.f : 1.f;   // an alcove steps out, the rest step in
                    const Eigen::Vector2f a0 = V[w] + T[w] * s, a1 = V[w] + T[w] * (s + wid);
                    const Eigen::Vector2f b0 = a0 + N[w] * (sg * dep), b1 = a1 + N[w] * (sg * dep);
                    room.push_back(a0); room.push_back(b0); room.push_back(b1); room.push_back(a1);
                    feats.push_back({kind, a0.cwiseMin(b1), a0.cwiseMax(b1)});
                    s += wid + U(0.5f, 1.5f);
                }
            }
            // A start pose well inside: the deepest interior point of a coarse scan of the room.
            Eigen::Vector2f start(W * 0.5f, H * 0.5f); float best_clear = -1.f;
            for (float x = 0.5f; x < W; x += 0.25f)
                for (float y = 0.5f; y < H; y += 0.25f)
                {
                    const Eigen::Vector2f q(x, y);
                    if (not rc::corner_visibility::point_in_polygon(q, room)) continue;
                    const float cl = point_to_poly(q, room);
                    if (cl > best_clear) { best_clear = cl; start = q; }
                }
            // FURNITURE, optional (WS_FURNITURE=1): boxes standing against the walls, which the
            // LiDAR band cannot see through and the ceiling junction is far above. Structure is
            // unchanged — the truth polygon is the same — so the score measures exactly whether the
            // walls behind them were recovered.
            Boxes furniture;
            if (std::getenv("WS_FURNITURE") != nullptr)
            {
                const int nf = 2 + static_cast<int>(U(0.f, 3.99f));
                for (int k = 0; k < nf; ++k)
                {
                    const int w = static_cast<int>(U(0.f, static_cast<float>(NW) - 0.01f));
                    const float dep = U(0.35f, 0.75f), wid = U(0.7f, 2.0f);
                    if (LEN[w] < wid + 1.2f) continue;
                    const float sc = U(0.6f, LEN[w] - wid - 0.6f);
                    const Eigen::Vector2f a0 = V[w] + T[w] * sc, a1 = V[w] + T[w] * (sc + wid) + N[w] * dep;
                    furniture.push_back({a0.cwiseMin(a1), a0.cwiseMax(a1)});
                }
            }
            RunConfig cfg8;
            cfg8.n_rays = 480;
            cfg8.verbose = std::getenv("WS_ROOM_VERBOSE") != nullptr;
            std::string poly8;
            if (const char* pp = std::getenv("WS_POLY_ROOM"))
            { poly8 = std::string(pp) + "_" + std::to_string(r) + ".csv"; cfg8.poly_csv = poly8.c_str(); }
            cfg8.occluders = furniture;
            cfg8.ceiling_line = std::getenv("WS_CEILING") != nullptr;
            cfg8.info_gain = std::getenv("WS_NO_INFOGAIN") == nullptr;
            std::mt19937 rrun(4242u + static_cast<unsigned>(r));
            int room_frames = 900;
            if (const char* e = std::getenv("WS_ROOM_FRAMES"))
            { int v = 0; if (std::from_chars(e, e + std::strlen(e), v).ec == std::errc{} and v > 0) room_frames = v; }
            auto Rr = run_explore(room, cfg8, rrun, room_frames, start);
            const Poly ew = to_world(Rr.poly.verts, Eigen::Vector3f(start.x(), start.y(), 0.f));
            // COVERAGE: of the truth's own interior, how much did the robot's grid ever learn about?
            // It separates the two ways a room can be lost — never seen, or seen and left outside the
            // polygon — which no IoU can tell apart.
            float coverage = 0.f, seen_but_excluded = 0.f;
            {
                long known = 0, total = 0, excl = 0;
                for (float x = -20.f; x < 20.f; x += 0.15f)
                    for (float y = -20.f; y < 20.f; y += 0.15f)
                    {
                        const Eigen::Vector2f w(x, y);
                        if (not rc::corner_visibility::point_in_polygon(w, room)) continue;
                        ++total;
                        const Eigen::Vector2f m = w - start;   // map frame = start-relative
                        const int gi = static_cast<int>((m.x() - Rr.map.fgrid.x0) / Rr.map.fgrid.cell);
                        const int gj = static_cast<int>((m.y() - Rr.map.fgrid.y0) / Rr.map.fgrid.cell);
                        const bool kn = Rr.map.fgrid.in(gi, gj) and not Rr.map.fgrid.is_unknown(gi, gj);
                        if (kn) ++known;
                        if (kn and not rc::corner_visibility::point_in_polygon(w, ew)) ++excl;
                    }
                if (total > 0) { coverage = static_cast<float>(known) / static_cast<float>(total);
                                 seen_but_excluded = static_cast<float>(excl) / static_cast<float>(total); }
            }
            const float iou_r = Rr.poly.closed ? polygon_iou(ew, room) : 0.f;
            const float h_r = Rr.poly.closed ? hausdorff(ew, room) : 1e9f;
            ious.push_back(iou_r); hauss.push_back(h_r);
            std::string fs;
            for (const auto& f : feats)
            {
                ++total[f.kind];
                float miss = 1.f;
                if (Rr.poly.closed)
                {
                    const float pad = 0.05f, step = 0.02f;
                    int bad = 0;
                    for (float x = f.lo.x() - pad; x <= f.hi.x() + pad; x += step)
                        for (float y = f.lo.y() - pad; y <= f.hi.y() + pad; y += step)
                        {
                            const Eigen::Vector2f q(x, y);
                            if (rc::corner_visibility::point_in_polygon(q, room)
                                != rc::corner_visibility::point_in_polygon(q, ew)) ++bad;
                        }
                    const float box = std::max((f.hi.x() - f.lo.x()) * (f.hi.y() - f.lo.y()), 1e-6f);
                    miss = static_cast<float>(bad) * step * step / box;
                }
                if (miss < 0.33f) ++found[f.kind];
                fs += fmt(" %s:%.2f", kind_name[f.kind], miss);
            }
            std::printf("    room %-3d %5.1f x %4.1f m %s feats %2zu furn %zu  IoU %.3f  cover %.2f excl %.2f  Hausdorff %.3f m  walls %2zu |%s\n",
                        r, W, H, ell ? "L  " : "rect", feats.size(), furniture.size(), iou_r, coverage,
                        seen_but_excluded, h_r, Rr.map.walls.size(), fs.c_str());
            std::printf("      truth[%d]:", r);
            for (const auto& v : room) std::printf(" (%.2f,%.2f)", v.x(), v.y());
            std::printf("\n      est[%d]:", r);
            for (const auto& v : ew) std::printf(" (%.2f,%.2f)", v.x(), v.y());
            std::printf("\n");
        }
        std::sort(ious.begin(), ious.end());
        std::sort(hauss.begin(), hauss.end());
        float mean = 0.f; for (float v : ious) mean += v; mean /= static_cast<float>(std::max<size_t>(1, ious.size()));
        std::printf("    %zu rooms: IoU mean %.3f median %.3f min %.3f max %.3f | Hausdorff median %.3f m\n",
                    ious.size(), mean, ious[ious.size() / 2], ious.front(), ious.back(), hauss[hauss.size() / 2]);
        for (int k = 0; k < 4; ++k)
            if (total[k] > 0)
                std::printf("    %-14s found %3d of %3d (%.0f%%)\n", kind_name[k], found[k], total[k],
                            100.f * static_cast<float>(found[k]) / static_cast<float>(total[k]));
        check("random rooms: median IoU >= 0.90", ious[ious.size() / 2] >= 0.90f, fmt("median %.3f", ious[ious.size() / 2]));
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
