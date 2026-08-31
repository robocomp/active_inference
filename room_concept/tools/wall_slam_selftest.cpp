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
#include <deque>
#include <queue>
#include <random>
#include <charconv>
#include <fstream>
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
    };

    struct RunConfig
    {
        int window = 5;
        int n_rays = 360;
        float scan_sigma = 0.02f;
        float odom_sigma_xy = 0.005f, odom_sigma_th = 0.3f * kPi / 180.f;
        bool verbose = false;
    };

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
            R.frames = static_cast<int>(f) + 1;
        }
        R.pose_rmse_xy = (n_err > 0) ? static_cast<float>(std::sqrt(se / n_err)) : 0.f;
        R.poly = R.map.build_polygon();
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
        return out;
    }

    /// The estimation loop with the robot DRIVEN by the model's uncertainty instead of a script.
    RunResult run_explore(const Poly& room, const RunConfig& cfg, std::mt19937& rng, int max_frames,
                          const Eigen::Vector2f& start)
    {
        RunResult R;
        Explorer ex(room);
        rc::wallseg::Params sp;
        sp.sensor_sigma = cfg.scan_sigma;
        R.map.params.obs_sigma = 0.05f;
        R.map.params.huber_delta = 0.15f;

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
        int last_rederive = 0, rejected_since_rederive = 0, rederives = 0;

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

            const auto pts = scan(room, tru, cfg.n_rays, cfg.scan_sigma, rng);
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
            const auto fr = R.map.observe(seg, pts, Eigen::VectorXf{}, pred, pcov, static_cast<std::int64_t>(f) * 50);
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
            if (f - last_rederive >= 40 or rejected_since_rederive >= 30)
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

            // ── perceive → decide: replan toward the largest visible unknown ──────────────────
            if (--replan_in <= 0 or path.empty())
            {
                replan_in = 15;
                const auto unknowns = collect_unknowns(R.map);
                // Exploration is COMPLETE when free space has no true frontier left; the map may
                // keep refining, but there is nowhere informative left to drive to.
                quiet_frames = (f > 60 and R.map.frontiers().empty()) ? quiet_frames + 1 : 0;
                if (quiet_frames >= 3 or unknowns.empty()) { R.frames = f + 1; break; }
                // score viewpoints on the free grid (subsampled) by visible unknown mass
                float best_sc = -1.f; Eigen::Vector2f best_v = tru.head<2>();
                for (int i = 0; i < ex.nx; i += 2)
                    for (int j = 0; j < ex.ny; j += 2)
                    {
                        if (not ex.is_free(i, j)) continue;
                        const Eigen::Vector2f v = ex.at(i, j);
                        float sc = 0.f;
                        for (const auto& u : unknowns)
                            if (ex.sees(v, from_map(u.p))) sc += u.w;
                        if (sc <= 0.f) continue;
                        sc /= (1.f + 0.10f * (v - tru.head<2>()).norm());
                        if (sc > best_sc) { best_sc = sc; best_v = v; }
                    }
                if (best_sc > 0.f)
                {
                    path = ex.astar(tru.head<2>(), best_v);
                    if (path.size() > 1) path.erase(path.begin());   // skip the cell we stand in
                }
            }
            if (cfg.verbose and (f % 25 == 0 or fr.births > 0))
                std::printf("    f=%3d walls=%zu cand=%d births=%d deaths=%d err=%.3fm poly=%s\n",
                            f, R.map.walls.size(), fr.candidates, fr.births, fr.deaths, exy,
                            R.map.build_polygon().closed ? "closed" : "open");
            R.frames = f + 1;
        }
        R.pose_rmse_xy = (n_err > 0) ? static_cast<float>(std::sqrt(se / n_err)) : 0.f;
        if (cfg.verbose) std::printf("    global re-derivations adopted: %d\n", rederives);
        R.poly = R.map.build_polygon();
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
    torch::set_num_threads(1);
    std::mt19937 rng(7);

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
    std::printf("\n4. A 45-degree chamfer: a wall with no Manhattan class\n");
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
        check("5 walls, one of them class-less (the chamfer)", R.map.walls.size() == 5 and off == 1,
              fmt("%zu walls, %d without class", R.map.walls.size(), off));
        const Poly est_world = to_world(R.poly.verts, truth[0]);
        const float h = R.poly.closed ? hausdorff(est_world, room) : 1e9f;
        check("chamfered polygon closed and within 5 cm", R.poly.closed and h < 0.05f, fmt("closed=%d hausdorff=%.3f", R.poly.closed, h));
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
        const bool fake_gone = R2.map.find(999) == nullptr and R2.map.find(998) == nullptr and R2.map.find(997) == nullptr;
        check("the wounded polygon was valid to start", wounded, "");
        check("the fake notch died and was spliced out", fake_gone,
              fmt("walls %zu -> %zu", walls_before + 3, R2.map.walls.size()));
        const Poly est_world = to_world(R2.poly.verts, truth[0]);
        const float h = R2.poly.closed ? hausdorff(est_world, room) : 1e9f;
        check("the healed polygon matches the room", R2.poly.closed and h < 0.06f,
              fmt("closed=%d hausdorff=%.3f", R2.poly.closed, h));
    }

    // ═══ 7. THE REAL LAYOUT: apartamento_layout.svg, toured and estimated to convergence ════════
    std::printf("\n7. The real apartamento layout (32 vertices incl. trace artefacts)\n");
    {
        // Local SVG polygon read (std::from_chars — the agents' locale rule; no Qt in the harness).
        Poly room;
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
        cfg.verbose = true;
        // EPISTEMIC DRIVE: no scripted tour — the robot goes where the model is uncertain.
        (void)truth;
        const auto R7 = run_explore(room, cfg, rng, 900, path[0]);
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
        check("polygon closed on the real layout", R7.poly.closed, R7.poly.status);
        // 0.20 m bar: the SVG itself carries 6-15 cm trace artefacts the estimator may lawfully
        // smooth over; a real miss (a whole alcove) is metres.
        check("estimate within 20 cm of the real layout (Hausdorff)", h < 0.20f, fmt("%.3f m", h));
        check("estimate overlaps the real layout (IoU >= 0.95)", iou >= 0.95f, fmt("IoU %.3f", iou));
        check("pose stayed on track through the tour", R7.pose_rmse_xy < 0.08f,
              fmt("rmse %.3f m, max %.3f m", R7.pose_rmse_xy, R7.pose_max_xy));
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
