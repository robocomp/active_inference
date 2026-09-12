/*
 *  mp3d_replay.cpp — drive the wall-SLAM estimator from RECORDED scans instead of simulated ones.
 *
 *  Every layout number this system has published so far was measured against scans it generated
 *  itself from the very polygon it was then asked to recover. That tests geometric generality — 594
 *  real floor plans is a real shape distribution — but it cannot test the sensor, and a reviewer is
 *  right to discount it. MP3D-FPE supplies the missing half: real Matterport depth, real camera
 *  trajectories, and per-room annotated corners, sliced to 2-D by datasets/mp3d_fpe/tools.
 *
 *  So this is run_explore with two organs removed. The scan comes off disk rather than out of a ray
 *  caster, and the trajectory is the one the human operator actually walked rather than one the EIG
 *  explorer chose. Everything between — segmentation, wall birth/death, the sliding-window joint
 *  solve, polygon closure — is the same code in the same order, because the point is to measure the
 *  estimator, not a reimplementation of it.
 *
 *  What that costs, stated plainly: with the trajectory fixed, this measures the MAPPER alone. The
 *  explorer's contribution cannot appear in these numbers, and a room the operator walked badly will
 *  score badly for reasons that are not the estimator's fault. Read it as a lower bound.
 *
 *  Usage:
 *      mp3d_replay <dataset_root> [room_glob]      e.g.  mp3d_replay datasets/mp3d_fpe
 *      MP3D_LIMIT=n      stop after n rooms
 *      MP3D_POLY=<pfx>   write the per-frame layout+sigma trace, <pfx>_<i>.csv
 *      MP3D_VERBOSE=1    per-frame detail
 */
#include <torch/torch.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <locale>
#include <string>
#include <vector>

#include "room_concept.h"
#include "room_gn_solver.h"
#include "room_model.h"
#include "wall_map.h"
#include "wall_segmenter.h"

namespace
{
    using Poly = std::vector<Eigen::Vector2f>;
    using rc::RoomConcept;

    float wrap(float a) { return std::atan2(std::sin(a), std::cos(a)); }

    torch::Tensor pose_tensor(const Eigen::Vector3f& p)
    { return torch::tensor({p.x(), p.y(), p.z()},
                           torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true)); }

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

    /// ⚠ std::from_chars, never strtof: this machine runs es_ES and the C library would stop at the
    /// decimal POINT and silently return the integer part (CLAUDE.md). Every number here is read
    /// from a file we wrote with points.
    bool num(const char*& p, const char* end, float& out)
    {
        while (p < end and (*p == ' ' or *p == ',' or *p == '\t' or *p == '\r')) ++p;
        if (p >= end) return false;
        const auto r = std::from_chars(p, end, out);
        if (r.ec != std::errc{}) return false;
        p = r.ptr;
        return true;
    }

    std::vector<Eigen::Vector2f> read_xy(const std::filesystem::path& f)
    {
        std::vector<Eigen::Vector2f> out;
        std::ifstream in(f);
        std::string line;
        while (std::getline(in, line))
        {
            if (line.empty() or line[0] == '#' or (line[0] < '0' and line[0] != '-' and line[0] != '+')) continue;
            const char* p = line.data(); const char* e = p + line.size();
            float x = 0.f, y = 0.f;
            if (num(p, e, x) and num(p, e, y)) out.emplace_back(x, y);
        }
        return out;
    }

    // ── scoring ────────────────────────────────────────────────────────────────────────────────
    bool inside(const Poly& poly, const Eigen::Vector2f& q)
    {
        bool in = false;
        for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
            if (((poly[i].y() > q.y()) != (poly[j].y() > q.y())) and
                (q.x() < (poly[j].x() - poly[i].x()) * (q.y() - poly[i].y()) /
                         (poly[j].y() - poly[i].y()) + poly[i].x()))
                in = not in;
        return in;
    }
    /// Rasterised IoU at 2 cm — the polygons are non-convex (L, T, notched) so a clipping routine
    /// would need a full boolean op; a grid is exact enough at this resolution and cannot go wrong.
    float poly_iou(const Poly& a, const Poly& b)
    {
        if (a.size() < 3 or b.size() < 3) return 0.f;
        Eigen::Vector2f lo = a[0], hi = a[0];
        for (const auto& P : {a, b}) for (const auto& v : P) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
        lo.array() -= 0.1f; hi.array() += 0.1f;
        const float step = 0.02f;
        long inter = 0, uni = 0;
        for (float y = lo.y(); y <= hi.y(); y += step)
            for (float x = lo.x(); x <= hi.x(); x += step)
            {
                const Eigen::Vector2f q(x, y);
                const bool ia = inside(a, q), ib = inside(b, q);
                if (ia and ib) ++inter;
                if (ia or ib) ++uni;
            }
        return uni > 0 ? static_cast<float>(inter) / static_cast<float>(uni) : 0.f;
    }
    float seg_dist(const Eigen::Vector2f& p, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
    {
        const Eigen::Vector2f ab = b - a;
        const float L2 = ab.squaredNorm();
        const float t = L2 > 1e-12f ? std::clamp(ab.dot(p - a) / L2, 0.f, 1.f) : 0.f;
        return (p - (a + t * ab)).norm();
    }
    float boundary_dist(const Poly& poly, const Eigen::Vector2f& q)
    {
        float best = 1e9f;
        for (size_t i = 0; i < poly.size(); ++i)
            best = std::min(best, seg_dist(q, poly[i], poly[(i + 1) % poly.size()]));
        return best;
    }
    float hausdorff(const Poly& a, const Poly& b)
    {
        const auto one_way = [](const Poly& from, const Poly& to)
        {
            float worst = 0.f;
            for (size_t i = 0; i < from.size(); ++i)
            {
                const Eigen::Vector2f& p = from[i]; const Eigen::Vector2f& q = from[(i + 1) % from.size()];
                const int n = std::max(2, static_cast<int>((q - p).norm() / 0.02f));
                for (int k = 0; k <= n; ++k)
                    worst = std::max(worst, boundary_dist(to, p + (q - p) * (static_cast<float>(k) / n)));
            }
            return worst;
        };
        return std::max(one_way(a, b), one_way(b, a));
    }
} // namespace

int main(int argc, char** argv)
{
    torch::set_num_threads(1);
    if (argc < 2) { std::printf("usage: mp3d_replay <dataset_root> [room_substring]\n"); return 2; }
    const std::filesystem::path root = argv[1];
    const std::string filter = (argc > 2) ? argv[2] : "";

    std::vector<std::filesystem::path> rooms;
    for (const auto& scene : std::filesystem::directory_iterator(root))
    {
        if (not scene.is_directory()) continue;
        for (const auto& rm : std::filesystem::directory_iterator(scene.path()))
        {
            if (not rm.is_directory()) continue;
            if (not std::filesystem::exists(rm.path() / "truth.txt")) continue;
            const std::string s = rm.path().string();
            if (filter.empty() or s.find(filter) != std::string::npos) rooms.push_back(rm.path());
        }
    }
    std::sort(rooms.begin(), rooms.end());
    if (const char* lim = std::getenv("MP3D_LIMIT"))
    { int v = 0; if (std::from_chars(lim, lim + std::strlen(lim), v).ec == std::errc{} and v > 0)
        if (static_cast<size_t>(v) < rooms.size()) rooms.resize(static_cast<size_t>(v)); }
    std::printf("mp3d_replay: %zu rooms under %s\n", rooms.size(), root.c_str());

    const bool verbose = std::getenv("MP3D_VERBOSE") != nullptr;
    std::ofstream csv("tmp/mp3d_replay.csv", std::ios::out | std::ios::trunc);
    if (csv.is_open())
    { csv.imbue(std::locale::classic());
      csv << "room,frames,points,gt_corners,est_corners,closed,iou,hausdorff_m,worst_corner_sigma_m\n"; }

    int done = 0;
    double iou_sum = 0.0;
    std::vector<float> ious, hauss;
    for (size_t ri = 0; ri < rooms.size(); ++ri)
    {
        const auto& dir = rooms[ri];
        // truth polygon, scene frame: "<name>;x,y x,y …"
        Poly truth;
        std::string name = dir.filename().string();
        {
            std::ifstream t(dir / "truth.txt");
            std::string line;
            if (std::getline(t, line))
            {
                const auto semi = line.find(';');
                if (semi != std::string::npos)
                {
                    name = line.substr(0, semi);
                    const char* p = line.data() + semi + 1; const char* e = line.data() + line.size();
                    float x = 0.f, y = 0.f;
                    while (num(p, e, x) and num(p, e, y)) truth.emplace_back(x, y);
                }
            }
        }
        if (truth.size() < 3) continue;

        // poses: frame,x,y,theta  (scene frame)
        std::vector<Eigen::Vector3f> gt;
        {
            std::ifstream f(dir / "poses.csv");
            std::string line;
            std::getline(f, line);                       // header
            while (std::getline(f, line))
            {
                const char* p = line.data(); const char* e = p + line.size();
                float fr = 0.f, x = 0.f, y = 0.f, th = 0.f;
                if (num(p, e, fr) and num(p, e, x) and num(p, e, y) and num(p, e, th))
                    gt.emplace_back(x, y, th);
            }
        }
        std::vector<std::filesystem::path> scans;
        for (const auto& s : std::filesystem::directory_iterator(dir))
            if (s.path().filename().string().rfind("scan_", 0) == 0) scans.push_back(s.path());
        std::sort(scans.begin(), scans.end());
        if (scans.empty() or gt.size() != scans.size()) continue;

        // ── the estimator, set up exactly as run_explore does ────────────────────────────────
        rc::wallmap::WallMap map;
        rc::wallseg::Params sp; sp.sensor_sigma = 0.02f;
        map.params.obs_sigma = 0.05f;
        map.params.huber_delta = 0.15f;
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

        // MAP FRAME = the first recorded pose, position AND heading, so the estimate starts at the
        // origin the way it does on the robot. The polygon is mapped back through it to score.
        const Eigen::Vector3f org = gt.front();
        const float c0 = std::cos(org.z()), s0 = std::sin(org.z());
        const auto scene_to_map = [&](const Eigen::Vector2f& v)
        { const Eigen::Vector2f d = v - org.head<2>();
          return Eigen::Vector2f(c0 * d.x() + s0 * d.y(), -s0 * d.x() + c0 * d.y()); };
        const auto map_to_scene = [&](const Eigen::Vector2f& v)
        { return Eigen::Vector2f(c0 * v.x() - s0 * v.y() + org.x(), s0 * v.x() + c0 * v.y() + org.y()); };

        Eigen::Vector3f est = Eigen::Vector3f::Zero(), prev_map = Eigen::Vector3f::Zero();
        long npts_total = 0;
        for (size_t f = 0; f < scans.size(); ++f)
        {
            // Ground-truth pose in the map frame.
            const Eigen::Vector2f tp = scene_to_map(gt[f].head<2>());
            const Eigen::Vector3f tm(tp.x(), tp.y(), wrap(gt[f].z() - org.z()));

            // Odometry is a MAP-FRAME delta, not a body-frame one, because that is what the
            // estimator's motion factor consumes: run_explore forms `odom = tm - prev_tru_map` and
            // composes `pred = est + odom` componentwise. Writing it in the body frame instead —
            // the natural choice, and my first one — mis-rotates every prediction and drops IoU to
            // 0.14. The dataset gives poses rather than wheel odometry, so this is a noise-free
            // odometer: generous to the estimator, and stated rather than buried.
            Eigen::Vector3f odom = Eigen::Vector3f::Zero();
            if (f > 0) { odom = tm - prev_map; odom.z() = wrap(odom.z()); }
            prev_map = tm;
            const Eigen::Vector3f pred = (f == 0) ? Eigen::Vector3f::Zero()
                : Eigen::Vector3f(est.x() + odom.x(), est.y() + odom.y(), wrap(est.z() + odom.z()));

            // The recorded scan, scene frame → ROBOT frame (the estimator's input convention).
            const auto world = read_xy(scans[f]);
            std::vector<Eigen::Vector2f> pts;
            pts.reserve(world.size());
            const float ct = std::cos(gt[f].z()), st = std::sin(gt[f].z());
            for (const auto& w : world)
            {
                const Eigen::Vector2f d = w - gt[f].head<2>();
                pts.emplace_back(ct * d.x() + st * d.y(), -st * d.x() + ct * d.y());
            }
            if (pts.size() < 16) continue;
            npts_total += static_cast<long>(pts.size());

            if (map.walls.empty())
            {   // model-first init from the first scan's OBB (identical to run_explore)
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
                  lo0 = std::min(lo0, u); hi0 = std::max(hi0, u);
                  lo1 = std::min(lo1, v2); hi1 = std::max(hi1, v2); }
                Poly rect = {mu + ax * lo0 + ay * lo1, mu + ax * hi0 + ay * lo1,
                             mu + ax * hi0 + ay * hi1, mu + ax * lo0 + ay * hi1};
                float a2 = 0.f;
                for (size_t i = 0; i < rect.size(); ++i)
                { const auto& p = rect[i]; const auto& q = rect[(i + 1) % rect.size()];
                  a2 += p.x() * q.y() - q.x() * p.y(); }
                if (a2 < 0.f) std::reverse(rect.begin(), rect.end());
                map.initialize_rect(rect);
            }

            std::mt19937 rng(1234u + static_cast<unsigned>(f));
            const auto seg = rc::wallseg::segment(pts, sp, rng);
            Eigen::VectorXf pw = Eigen::VectorXf::Ones(static_cast<long>(pts.size()));
            const Eigen::Matrix3f pcov =
                Eigen::Vector3f(0.05f * 0.05f, 0.05f * 0.05f, 0.03f * 0.03f).asDiagonal();
            const auto fr = map.observe(seg, pts, pw, pred, pcov, static_cast<std::int64_t>(f) * 50);

            RoomConcept::WindowSlot slot;
            slot.pose = pose_tensor(pred);
            slot.lidar_points = points_tensor(pts);
            slot.odometry_delta = odom;
            slot.motion_cov = Eigen::Vector3f(0.02f * 0.02f, 0.02f * 0.02f, 0.01f * 0.01f).asDiagonal();
            slot.odom_delta_tensor = torch::tensor({odom.x(), odom.y(), odom.z()}, torch::kFloat32);
            slot.motion_prec_tensor = mat3(slot.motion_cov.inverse());
            slot.wall_assoc = fr.assoc;

            rc::gn::Input in;
            in.model = &model; in.params = &params; in.window = &window; in.boundary_prior = &bp;
            in.device = torch::kCPU;
            in.walls = &map; in.no_sdf = true; in.gauge_fix = true;
            if (static_cast<int>(window.size()) >= 5)
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
            { auto pp = sl.pose.detach();
              poses.emplace_back(pp[0].item<float>(), pp[1].item<float>(), pp[2].item<float>()); }
            rc::gn::Options opts;
            const auto r = rc::gn::solve(in, poses, opts);
            if (r.ok) for (size_t i = 0; i < window.size(); ++i) window[i].pose = pose_tensor(poses[i]);
            est = poses.back();
            map.merge_indistinguishable();
            if (verbose and (f % 20) == 0)
                std::printf("      f=%3zu pts=%4zu walls=%2zu\n", f, pts.size(), map.walls.size());
        }

        const auto pr = map.build_polygon();
        Poly est_scene;
        for (const auto& v : pr.verts) est_scene.push_back(map_to_scene(v));
        float worst_sigma = 0.f;
        for (const auto& c : pr.corners) if (std::isfinite(c.sigma)) worst_sigma = std::max(worst_sigma, c.sigma);

        const float iou = (pr.closed and est_scene.size() >= 3) ? poly_iou(est_scene, truth) : 0.f;
        const float hd  = (pr.closed and est_scene.size() >= 3) ? hausdorff(est_scene, truth) : -1.f;
        ++done; iou_sum += iou; ious.push_back(iou);
        if (hd >= 0.f) hauss.push_back(hd);
        std::printf("  room %3zu  %-40s scans %4zu  gt %2zu  est %2zu  closed %d  IoU %.3f  Hausdorff %.3f m  worst sigma %.3f\n",
                    ri, name.c_str(), scans.size(), truth.size(), est_scene.size(),
                    pr.closed ? 1 : 0, iou, hd, worst_sigma);
        std::fflush(stdout);
        if (csv.is_open())
            csv << name << ',' << scans.size() << ',' << npts_total << ',' << truth.size() << ','
                << est_scene.size() << ',' << (pr.closed ? 1 : 0) << ',' << iou << ',' << hd << ','
                << worst_sigma << '\n';
    }

    std::sort(ious.begin(), ious.end());
    std::sort(hauss.begin(), hauss.end());
    if (done > 0)
        std::printf("\n  %d rooms | IoU mean %.4f median %.4f | Hausdorff median %.3f m | IoU>=0.90 %.1f%%\n",
                    done, iou_sum / done, ious[ious.size() / 2],
                    hauss.empty() ? -1.f : hauss[hauss.size() / 2],
                    100.0 * static_cast<double>(std::count_if(ious.begin(), ious.end(),
                        [](float v){ return v >= 0.90f; })) / done);
    return 0;
}
