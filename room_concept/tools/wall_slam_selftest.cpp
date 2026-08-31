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
#include <random>
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
        int frames = 0, closed_at = -1, births = 0;
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
            const auto seg = rc::wallseg::segment(pts, sp, rng);
            const Eigen::Matrix3f pcov = Eigen::Vector3f(0.05f * 0.05f, 0.05f * 0.05f, 0.03f * 0.03f).asDiagonal();
            const auto fr = R.map.observe(seg, pts, Eigen::VectorXf{}, pred, pcov, static_cast<std::int64_t>(f) * 50);
            R.births += fr.births;

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
        const Poly est_world = to_world(R.poly.verts, twice[0]);
        const float h = R.poly.closed ? hausdorff(est_world, l_room_notch()) : 1e9f;
        check("the two notch walls were born after closure", R.map.walls.size() == 8, fmt("%zu walls (expected 8)", R.map.walls.size()));
        check("polygon re-closed on the notched room within 8 cm", R.poly.closed and h < 0.08f,
              fmt("closed=%d hausdorff=%.3f status='%s'", R.poly.closed, h, R.poly.status.c_str()));
    }

    // ═══ 6. Existence: a phantom wall dies by pass-through; the real ones survive ═══════════════
    std::printf("\n6. Step-back operator: beams walk through a phantom wall\n");
    {
        const Poly room = l_room();
        const auto truth = trajectory(12);
        RunConfig cfg;
        auto Rr = run_loop({room}, truth, cfg, rng);
        const size_t walls_before = Rr.map.walls.size();
        // Inject a phantom "furniture wall" across the interior — the LiDAR sees straight through
        // where it claims to stand. phi/d chosen with d < 0 (the sign convention) and the extent well
        // inside the room, crossing the driven loop's lines of sight.
        rc::wallmap::WallLandmark ph;
        ph.id = 999;
        ph.phi = 0.3f;
        ph.d = -0.4f;
        ph.information = Eigen::Vector2f(500.f, 2000.f).asDiagonal();
        ph.has_extent = true; ph.s_min = -1.2f; ph.s_max = 1.2f;
        ph.exist_lodds = Rr.map.params.birth_nats;
        Rr.map.walls.push_back(ph);
        auto R2 = run_loop({room}, truth, cfg, rng, &Rr);
        const bool phantom_gone = R2.map.find(999) == nullptr;
        int real_alive = 0;
        for (const auto& w : R2.map.walls) if (w.id <= 6) ++real_alive;
        for (const auto& w : R2.map.walls)
            std::printf("    wall %llu lodds=%.1f extent[%.2f,%.2f] bins=%zu\n",
                        static_cast<unsigned long long>(w.id), w.exist_lodds, w.s_min, w.s_max, w.exist_bins.size());
        check("the phantom wall died", phantom_gone, fmt("walls %zu -> %zu", walls_before + 1, R2.map.walls.size()));
        check("the six real walls survived", real_alive == 6, fmt("%d / 6 alive", real_alive));
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
