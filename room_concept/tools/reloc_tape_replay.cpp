/*
 *  reloc_tape_replay.cpp -- offline A/B of the LEGACY and the RIVAL relocaliser on recorded scan tapes (spec 2026-11-01-relocaliser-
 *  redesign.md sec 9; SDD Task 7, steps 1-2).
 *
 *  WHAT IS REPLAYED. The tape (scan_tape.h) holds, per relocalisation_step call, what the live tracker committed (pose, covariance),
 *  the odometry increment its predict step used and the scan. Each frame is handed to the REAL RoomConcept::relocalisation_step of a
 *  fresh RoomConcept in either RelocMode (the harness is reloc_selftest's: RelocSelftestAccess + set_robot_pose + step8), exactly the
 *  entry the agent calls. update() is NOT run: the tracker's pose per frame is the tape's, carried by a rigid correction C
 *  ("published = C o tape"). C starts at identity; when a relocaliser moves the pose (reloc_epoch_ ticks), C is re-seated on the new
 *  pose, so everything the tape's tracker does afterwards (its motion AND its own corrections) is applied on top of the relocaliser's
 *  move, the way a real tracker continues from a reset pose. An injected kidnap is a change of C.
 *
 *  GROUND TRUTH. gt_error.csv has the world-frame ground truth per tape frame (gt_x_world, gt_y_world, gt_theta). The tape's poses
 *  are in each ROOM's frame, one frame per visit (the layout is re-estimated between visits). A visit = a run of frames with the same
 *  room polygon; per visit a rigid map room->world is fitted (heading offset = circular mean of gt_theta - theta_room - 90 deg, the
 *  controller's theta being forward-at-+90; then translation = mean residual; one trimmed refit at 15 cm) and the truth in the room
 *  frame is its inverse. The fit uses the tracker's own poses, so "truth" cannot disagree with the tracker by more than the
 *  tracker's own wobble inside a visit: see the report for what that does to the metrics.
 *
 *  Build (EXCLUDE_FROM_ALL):  make -C build reloc_tape_replay
 *  Run:  bin/reloc_tape_replay [--kidnaps N] [--seed S] [--mirror M] [--skip-healthy] TAPE GT [TAPE GT ...]
 */
#include <QtGlobal>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <numbers>
#include <print>
#include <random>
#include <ranges>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "door_apertures.h"
#include "reloc_search.h"
#include "room_concept.h"
#include "scan_tape.h"

namespace rc
{
    /// Test access to the relocalisation runtime (declared friend in RoomConcept) -- the subset of reloc_selftest's.
    struct RelocSelftestAccess
    {
        static void  step8(RoomConcept& r, const RoomConcept::UpdateResult& res, const std::vector<Eigen::Vector3f>& pts) { r.relocalisation_step(res, pts); }
        static std::uint64_t epoch(RoomConcept& r) { return r.reloc_epoch_.load(); }
        static void  set_mode(RoomConcept& r, const std::string& m) { r.params.reloc_mode = m; }
        static void  timing_on(RoomConcept& r) { r.rv_timing_on_ = true; }
        static auto& timing(RoomConcept& r) { return r.rv_timing_us_; }
        static Eigen::Matrix3f spread(RoomConcept& r) { return r.rt_.held_spread; }
        static rc::reloc::Params rparams(RoomConcept& r) { return r.reloc_params(); }
        static void set_crossing(RoomConcept& r, bool on) { r.set_crossing(on); r.set_transition_open(on); }   // the scene graph sets both while a cross executes
        static float last_search_ms(RoomConcept& r) { return r.rv_last_search_ms_; }
        static std::uint64_t swaps(RoomConcept& r) { return r.rv_swaps_; }
        static std::uint64_t searches(RoomConcept& r) { return r.rt_.searches; }
        static void  set_odo(RoomConcept& r, const Eigen::Vector3f& global_delta, const Eigen::Matrix3f& cov)
        { r.last_selected_prior_.valid = true; r.last_selected_prior_.delta_pose = global_delta; r.last_selected_prior_.covariance_eigen = cov; }
        static bool  proposer_busy(RoomConcept& r) { return r.proposer_.busy(); }
        static void  wait_proposer(RoomConcept& r) { while (r.proposer_.busy()) std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    };
}

namespace
{
    using V3 = Eigen::Vector3f;
    using V2 = Eigen::Vector2f;
    using M3 = Eigen::Matrix3f;
    using Poly = std::vector<V2>;
    constexpr float kPi = std::numbers::pi_v<float>;
    float wrap(float a) { return std::remainder(a, 2.f * kPi); }
    float deg(float r) { return r * 180.f / kPi; }

    // ---- SE(2) -------------------------------------------------------------------------------------------------------------
    V3 comp(const V3& a, const V3& d)   // a o d, d in a's frame
    {
        const float c = std::cos(a.z()), s = std::sin(a.z());
        return {a.x() + c * d.x() - s * d.y(), a.y() + s * d.x() + c * d.y(), wrap(a.z() + d.z())};
    }
    V3 inv(const V3& a)
    {
        const float c = std::cos(a.z()), s = std::sin(a.z());
        return {-(c * a.x() + s * a.y()), -(-s * a.x() + c * a.y()), wrap(-a.z())};
    }
    M3 rot_cov(const M3& c, float th)   // covariance of a (x, y, theta) quantity rotated by a rigid rotation th
    {
        M3 J = M3::Identity();
        J(0, 0) = std::cos(th); J(0, 1) = -std::sin(th); J(1, 0) = std::sin(th); J(1, 1) = std::cos(th);
        return J * c * J.transpose();
    }

    // ---- numbers (locale-independent) ----------------------------------------------------------------------------------------
    bool num(std::string_view s, double& v)
    {
        const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
        return r.ec == std::errc();
    }
    std::vector<std::string_view> split(std::string_view l)
    {
        std::vector<std::string_view> o;
        std::size_t p = 0;
        for (;;)
        {
            const auto q = l.find(',', p);
            o.push_back(l.substr(p, q == std::string_view::npos ? q : q - p));
            if (q == std::string_view::npos) break;
            p = q + 1;
        }
        return o;
    }

    struct GtRow { double gx = 0, gy = 0, gth = 0, sdf = 0, px = 0, py = 0, pth = 0; };
    std::unordered_map<std::int64_t, GtRow> load_gt(const std::string& path)
    {
        std::unordered_map<std::int64_t, GtRow> out;
        std::ifstream f(path);
        std::string l;
        if (not std::getline(f, l)) return out;
        const auto h = split(l);
        auto col = [&](std::string_view n) { for (std::size_t i = 0; i < h.size(); ++i) if (h[i] == n) return static_cast<int>(i); return -1; };
        const int c_ts = col("ts_ms"), c_gx = col("gt_x_world"), c_gy = col("gt_y_world"), c_gt = col("gt_theta"), c_s = col("sdf_mse"),
                  c_px = col("pred_x"), c_py = col("pred_y"), c_pt = col("pred_theta");
        if (std::min({c_ts, c_gx, c_gy, c_gt, c_s, c_px, c_py, c_pt}) < 0) { std::println(stderr, "gt_error.csv: missing columns"); return out; }
        while (std::getline(f, l))
        {
            const auto c = split(l);
            if (c.size() <= static_cast<std::size_t>(std::max({c_ts, c_gx, c_gy, c_gt, c_s, c_px, c_py, c_pt}))) continue;
            GtRow r; double ts = 0;
            if (not (num(c[c_ts], ts) and num(c[c_gx], r.gx) and num(c[c_gy], r.gy) and num(c[c_gt], r.gth) and num(c[c_s], r.sdf)
                     and num(c[c_px], r.px) and num(c[c_py], r.py) and num(c[c_pt], r.pth))) continue;
            out[static_cast<std::int64_t>(ts)] = r;
        }
        return out;
    }

    // ---- the tape with its truth ---------------------------------------------------------------------------------------------
    struct FInfo { bool has = false; V3 truth = V3::Zero(); V3 pred = V3::Zero(); V3 live = V3::Zero(); float s_tape = 0.034f; };
    struct Visit { int a = 0, b = 0; int rec = 0; double rho = 0; V2 t = V2::Zero(); double rms = 0; };
    struct Data
    {
        std::string name, label;
        rc::tape::Tape tape;
        std::vector<FInfo> fi;
        std::vector<Visit> visits;
        std::vector<int> visit_of;
        int push_f = -1;               ///< first frame of a sustained tracker-vs-gt departure (a push / drag), -1 = none
        int n_live_moves = 0;          ///< live relocaliser moves found in events.jsonl and removed from the tape's tracker poses
        std::vector<char> crossing;   ///< per frame: the controller was executing a door cross (graph_trace.csv) -> the live relocaliser was disarmed
    };

    Data load(const std::string& tape_path, const std::string& gt_path)
    {
        Data d;
        d.name = tape_path;
        d.label = std::filesystem::path(tape_path).parent_path().filename().string();
        d.tape = rc::tape::read_tape(tape_path);
        const auto gt = load_gt(gt_path);
        const auto& T = d.tape;
        const int n = static_cast<int>(T.frames.size());
        d.fi.resize(n);
        d.visit_of.assign(n, -1);
        // The tape's poses are the LIVE tracker's, so they carry whatever the live (legacy) relocaliser did: its moves enter the
        // tape as pose jumps and the replay would hand them to BOTH relocalisers as tracker motion. Find the live moves (the
        // `[reloc] ... moved` / `swap` lines of events.jsonl, matched to the frame within 300 ms whose pose step departs most from the
        // odometry's) and replace that one step by the odometry increment, so the replayed tracker is the tracker WITHOUT the relocaliser.
        std::vector<V3> live(n);
        for (int f = 0; f < n; ++f) live[f] = T.frames[f].pose;
        std::vector<char> jump(n, 0);
        {
            std::ifstream ef(std::filesystem::path(gt_path).parent_path() / "events.jsonl");
            std::string l;
            while (std::getline(ef, l))
            {
                if (l.find("\"kind\":\"log\"") == std::string::npos) continue;
                if (l.find("[reloc") == std::string::npos or (l.find("moved") == std::string::npos and l.find("swap") == std::string::npos)) continue;
                const auto k = l.find("\"t\":");
                double te = 0; if (k == std::string::npos) continue;
                const auto kend = l.find_first_of(",}", k + 4);
                if (not num(std::string_view(l).substr(k + 4, kend - k - 4), te)) continue;
                int best = -1; double bv = 0;
                for (int f = 1; f < n; ++f)
                {
                    if (std::abs(static_cast<double>(T.frames[f].t_ms) - te) > 300) continue;
                    const V3 dv = live[f] - live[f - 1] - T.frames[f].odo_delta;
                    const double v = V2(dv.x(), dv.y()).norm() + std::abs(wrap(dv.z()));
                    if (v > bv) { bv = v; best = f; }
                }
                if (best >= 0 and bv > 0.05) { jump[best] = 1; ++d.n_live_moves; }
            }
        }
        std::vector<V3> Cf(n, V3::Zero());   // rebased = Cf o live
        {
            V3 cur = live[0];
            for (int f = 1; f < n; ++f)
            {
                V3 r = comp(inv(live[f - 1]), live[f]);
                if (jump[f])
                {
                    const float c = std::cos(live[f - 1].z()), s2 = std::sin(live[f - 1].z());
                    const V3 g = T.frames[f].odo_delta;
                    r = V3(c * g.x() + s2 * g.y(), -s2 * g.x() + c * g.y(), g.z());
                }
                cur = comp(cur, r);
                Cf[f] = comp(cur, inv(live[f]));
                d.tape.frames[f].pose = cur;
            }
            Cf[0] = V3::Zero();
        }
        for (int f = 0; f < n; ++f) d.fi[f].live = live[f];
        d.visit_of.assign(n, -1);
        auto poly_of = [&](int f) -> const Poly& { return T.rooms[std::max(0, T.frame_room[f])].polygon; };
        // a visit = a run of frames over one room polygon
        for (int f = 0, s = 0; f <= n; ++f)
            if (f == n or poly_of(f) != poly_of(s))
            {
                if (f > s and poly_of(s).size() >= 3) { Visit v; v.a = s; v.b = f; v.rec = T.frame_room[s]; d.visits.push_back(v); }
                s = f;
            }
        for (std::size_t vi = 0; vi < d.visits.size(); ++vi)
        {
            Visit& v = d.visits[vi];
            std::vector<int> rows;
            for (int f = v.a; f < v.b; ++f) { d.visit_of[f] = static_cast<int>(vi); if (gt.contains(T.frames[f].t_ms)) rows.push_back(f); }
            if (rows.size() < 10) continue;
            auto fit = [&](const std::vector<int>& use)
            {
                double sx = 0, sy = 0;
                for (int f : use)
                {
                    const auto& g = gt.at(T.frames[f].t_ms);
                    const double dth = std::remainder(g.gth - T.frames[f].pose.z() - std::numbers::pi / 2, 2 * std::numbers::pi);
                    sx += std::cos(dth); sy += std::sin(dth);
                }
                v.rho = std::atan2(sy, sx);
                const double c = std::cos(v.rho), s = std::sin(v.rho);
                double tx = 0, ty = 0;
                for (int f : use)
                {
                    const auto& g = gt.at(T.frames[f].t_ms); const auto& e = T.frames[f].pose;
                    tx += g.gx - (c * e.x() - s * e.y()); ty += g.gy - (s * e.x() + c * e.y());
                }
                v.t = V2(static_cast<float>(tx / use.size()), static_cast<float>(ty / use.size()));
            };
            auto resid = [&](int f)
            {
                const auto& g = gt.at(T.frames[f].t_ms); const auto& e = T.frames[f].pose;
                const double c = std::cos(v.rho), s = std::sin(v.rho);
                return std::hypot(c * e.x() - s * e.y() + v.t.x() - g.gx, s * e.x() + c * e.y() + v.t.y() - g.gy);
            };
            fit(rows);
            std::vector<int> keep;
            auto hres = [&](int f)
            {
                const auto& g = gt.at(T.frames[f].t_ms);
                return std::abs(std::remainder(g.gth - T.frames[f].pose.z() - std::numbers::pi / 2 - v.rho, 2 * std::numbers::pi));
            };
            for (int f : rows) if (resid(f) < 0.15 and hres(f) < 10 * std::numbers::pi / 180) keep.push_back(f);
            if (keep.size() >= 10) fit(keep);
            double ss = 0; for (int f : rows) ss += resid(f) * resid(f);
            v.rms = std::sqrt(ss / rows.size());
            // truth in the room frame = inverse of (room -> world)
            for (int f : rows)
            {
                const auto& g = gt.at(T.frames[f].t_ms);
                const double c = std::cos(-v.rho), s = std::sin(-v.rho);
                const double wx = g.gx - v.t.x(), wy = g.gy - v.t.y();
                auto& fi = d.fi[f];
                fi.has = true;
                fi.truth = V3(static_cast<float>(c * wx - s * wy), static_cast<float>(s * wx + c * wy),
                              wrap(static_cast<float>(g.gth - v.rho - std::numbers::pi / 2)));
                fi.pred = comp(Cf[f], V3(static_cast<float>(g.px), static_cast<float>(g.py), static_cast<float>(g.pth)));
                fi.s_tape = static_cast<float>(g.sdf);
            }
        }
        // the live relocaliser is disarmed while a cross EXECUTES (controller `executing` edge on aff_*_cross) until the room switch
        d.crossing.assign(n, 0);
        {
            std::ifstream gf(std::filesystem::path(gt_path).parent_path() / "graph_trace.csv");
            std::string l;
            std::vector<std::pair<std::int64_t, std::int64_t>> win;   // [executing inserted, deleted)
            std::map<std::string, std::int64_t> open;   // b_id -> inserted (an `executing` edge is deleted with its names blank, so key by node id)
            std::map<std::string, bool> is_cross;
            while (std::getline(gf, l))
            {
                const auto c = split(l);
                if (c.size() < 9 or c[8] != "executing") continue;
                double ts = 0; if (not num(c[0], ts)) continue;
                const std::string key(c[6]);
                if (c[1] == "edge_insert") { open[key] = static_cast<std::int64_t>(ts); is_cross[key] = c[7].starts_with("aff_") and c[7].ends_with("_cross"); }
                else if (c[1] == "edge_delete" and open.contains(key))
                {
                    if (is_cross[key]) win.emplace_back(open[key], static_cast<std::int64_t>(ts));
                    open.erase(key);
                }
            }
            for (const auto& [k, t0] : open) if (is_cross[k]) win.emplace_back(t0, std::numeric_limits<std::int64_t>::max());
            for (const auto& [t0, t1] : win)
            {
                // from the cross starting to the room switch that ends the visit it started in (or the edge's own end, if earlier)
                int f0 = 0; while (f0 < n and T.frames[f0].t_ms < t0) ++f0;
                if (f0 >= n) continue;
                const int v0 = d.visit_of[f0];
                const std::int64_t end = std::min(t1, v0 >= 0 ? T.frames[d.visits[v0].b - 1].t_ms : T.frames[n - 1].t_ms);
                for (int f = f0; f < n and T.frames[f].t_ms <= end; ++f) d.crossing[f] = 1;
            }
        }
        // frames without a gt row keep the tracker's own pose as prediction
        for (int f = 0; f < n; ++f) if (not d.fi[f].has) d.fi[f].pred = T.frames[f].pose;
        // a push / drag: the tracker (odometry only, parked) stays while the ground truth leaves it for good
        for (int f = 0, run_len = 0; f < n; ++f)
        {
            const auto& fi = d.fi[f];
            const bool off = fi.has and ((fi.truth.head<2>() - T.frames[f].pose.head<2>()).norm() > 0.10f or std::abs(wrap(fi.truth.z() - T.frames[f].pose.z())) > 10.f * kPi / 180.f);
            run_len = off ? run_len + 1 : 0;
            if (run_len >= 40) { d.push_f = f - 39; break; }
        }
        return d;
    }

    // ---- stats ---------------------------------------------------------------------------------------------------------------
    double pct(std::vector<double> v, double p)   // nearest-rank percentile (infinities allowed: an unrecovered event is infinitely late)
    {
        if (v.empty()) return std::nan("");
        std::ranges::sort(v);
        const auto k = static_cast<std::size_t>(std::max(0.0, std::ceil(p * static_cast<double>(v.size())) - 1.0));
        return v[std::min(k, v.size() - 1)];
    }

    // ---- scan perturbations ("people") -----------------------------------------------------------------------------------------
    enum class Pert { None, Clutter, Person };
    const char* pname(Pert p) { return p == Pert::None ? "none" : p == Pert::Clutter ? "clutter30" : "person"; }
    std::vector<V3> perturb(const std::vector<V2>& pts, Pert p, std::uint64_t seed, int f, double t_s)
    {
        std::vector<V3> out; out.reserve(pts.size());
        std::mt19937_64 rng(seed * 1000003ULL + static_cast<std::uint64_t>(f));
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        if (p == Pert::Clutter)
        {
            for (const auto& q : pts)
            {
                if (u01(rng) < 0.30f)
                {
                    const float b = -kPi + 2.f * kPi * u01(rng), r = 0.3f + 2.2f * u01(rng);
                    out.emplace_back(r * std::cos(b), r * std::sin(b), 0.f);
                }
                else out.emplace_back(q.x(), q.y(), 0.f);
            }
        }
        else if (p == Pert::Person)
        {
            // a 0.5 m cylinder walking round the robot at 1.5 m (one lap per 20 s): the beams it blocks return its near surface
            const float phi = wrap(static_cast<float>(2 * std::numbers::pi * t_s / 20.0));
            const float D = 1.5f, R = 0.25f;
            for (const auto& q : pts)
            {
                const float b = std::atan2(q.y(), q.x()), r = q.norm();
                const float off = wrap(b - phi);
                const float lat = D * std::sin(off);   // lateral distance of the beam from the cylinder centre line
                if (std::abs(off) < kPi / 2 and std::abs(lat) < R and r > D - R)
                {
                    const float rr = D * std::cos(off) - std::sqrt(std::max(0.f, R * R - lat * lat));
                    out.emplace_back(rr * std::cos(b), rr * std::sin(b), 0.f);
                }
                else out.emplace_back(q.x(), q.y(), 0.f);
            }
        }
        else for (const auto& q : pts) out.emplace_back(q.x(), q.y(), 0.f);
        return out;
    }

    float median_d(const Poly& poly, const std::vector<V2>& scan, const V3& pose)
    {
        float med = 0.f;
        rc::reloc::Params rp;
        rc::reloc::log_likelihood(poly, scan, pose, rp, 10.f, &med);
        return med;
    }

    std::vector<rc::DoorAperture> apertures_of(const rc::tape::RoomRecord& r)
    {
        std::vector<rc::DoorAperture> o;
        for (const auto& a : r.apertures)
        {
            rc::DoorAperture d;
            d.a = a.a; d.b = a.b; d.n_out = a.n_out; d.p_open = a.p_open; d.prior_closed = a.prior_closed;
            d.half_depth = a.half_depth; d.recess = a.recess; d.sigma_along = a.sigma_along; d.name = a.name;
            o.push_back(std::move(d));
        }
        return o;
    }

    // ---- the replay ------------------------------------------------------------------------------------------------------------
    struct FOut
    {
        int f = 0;
        V3 pub = V3::Zero(), pre = V3::Zero();
        bool moved = false;
        float move_m = 0.f, move_rad = 0.f, sigma = 0.f, cov_maxdiag = 0.f, latency_ms = 0.f;
        double step_ms = 0.0;
    };
    struct RunCfg
    {
        std::string mode;
        const Data* d = nullptr;
        int a = 0, b = 0;              // frames [a,b) of one visit
        int kidnap_f = -1;             // frame at which C is changed so that the tracker's pose = `wrong`
        V3 wrong = V3::Zero();
        Pert pert = Pert::None;
        std::uint64_t seed = 1;
    };
    struct RunOut { std::vector<FOut> fr; std::vector<double> hot_us; };

    RunOut run(const RunCfg& cfg)
    {
        RunOut out;
        const auto& T = cfg.d->tape;
        const Poly& poly = T.rooms[T.frame_room[cfg.a]].polygon;
        rc::RoomConcept rcn;
        rc::RelocSelftestAccess::set_mode(rcn, cfg.mode);
        rcn.set_polygon_room(poly);
        int rec = -1;
        V3 C = V3::Zero();   // published = C o tape
        const bool rivals = cfg.mode == "rivals";
        if (rivals) rc::RelocSelftestAccess::timing_on(rcn);
        rcn.set_robot_pose(T.frames[cfg.a].pose.x(), T.frames[cfg.a].pose.y(), T.frames[cfg.a].pose.z(), false);
        for (int f = cfg.a; f < cfg.b; ++f)
        {
            const auto& fr = T.frames[f];
            if (T.frame_room[f] != rec) { rec = T.frame_room[f]; rcn.set_door_apertures(apertures_of(T.rooms[rec])); }
            rc::RelocSelftestAccess::set_crossing(rcn, cfg.d->crossing[f] != 0);
            const V3 tp = fr.pose;
            if (f == cfg.kidnap_f) C = comp(cfg.wrong, inv(tp));
            const V3 pt = comp(C, tp);
            rcn.set_robot_pose(pt.x(), pt.y(), pt.z(), false);
            // the scan the relocaliser gets (perturbed for the "people" runs)
            const double t_s = (fr.t_ms - T.frames[cfg.a].t_ms) * 1e-3;
            const auto pts3 = perturb(fr.pts_robot, cfg.pert, cfg.seed, f, t_s);
            std::vector<V2> pts2; pts2.reserve(pts3.size());
            const int stride = std::max<int>(1, static_cast<int>(pts3.size()) / 150);
            for (std::size_t i = 0; i < pts3.size(); i += stride) pts2.emplace_back(pts3[i].x(), pts3[i].y());
            std::vector<V2> pts2c;   // the same subsample of the CLEAN scan: s_tape was measured on it, at the live pose
            for (std::size_t i = 0; i < fr.pts_robot.size(); i += std::max<std::size_t>(1, fr.pts_robot.size() / 150)) pts2c.push_back(fr.pts_robot[i]);
            const auto& fi = cfg.d->fi[f];
            rc::RoomConcept::UpdateResult r;
            r.ok = true;
            r.robot_pose = Eigen::Affine2f::Identity();
            r.robot_pose.translation() = pt.head<2>();
            r.robot_pose.linear() = Eigen::Rotation2Df(pt.z()).toRotationMatrix();
            r.covariance = rot_cov(fr.cov, C.z());
            const V3 pr = comp(C, fi.pred);
            r.pred_x = pr.x(); r.pred_y = pr.y(); r.pred_theta = pr.z();
            // s: what the tracker's SDF residual was (the tape side's value, exact when the pose is the tape's), plus the extra
            // misfit the displacement of the published pose adds on this scan -- the same median-|SDF| statistic the legacy trigger
            // reads, evaluated on the replay's own points (reloc_selftest's median_d).
            const float d_pub = median_d(poly, pts2, pt), d_tape = median_d(poly, pts2c, fi.live);   // s_tape belongs to the LIVE pose
            r.sdf_mse = std::max(0.f, fi.s_tape + (d_pub - d_tape));
            r.pred_sdf_median = r.sdf_mse;
            r.timestamp_ms = fr.t_ms;
            // the odometry increment the predict step used, rotated into the corrected frame
            {
                const float c = std::cos(C.z()), s = std::sin(C.z());
                const V3 od = fr.odo_delta;
                rc::RelocSelftestAccess::set_odo(rcn, V3(c * od.x() - s * od.y(), s * od.x() + c * od.y(), od.z()), rot_cov(fr.odo_cov, C.z()));
            }
            const auto e0 = rc::RelocSelftestAccess::epoch(rcn);
            const auto t0 = std::chrono::steady_clock::now();
            rc::RelocSelftestAccess::step8(rcn, r, pts3);
            const auto t1 = std::chrono::steady_clock::now();
            if (rivals) rc::RelocSelftestAccess::wait_proposer(rcn);   // the worker finishes before the next 50 ms frame, as live
            FOut o;
            o.f = f; o.pre = pt; o.pub = pt;
            o.step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (rc::RelocSelftestAccess::epoch(rcn) != e0)
            {
                const auto st = rcn.get_current_state();
                const V3 P(st[2], st[3], st[4]);
                o.moved = true;
                o.move_m = (P.head<2>() - pt.head<2>()).norm();
                o.move_rad = std::abs(wrap(P.z() - pt.z()));
                o.pub = P;
                if (std::getenv("RELOC_TRACE") != nullptr)
                {
                    const auto rp = rc::RelocSelftestAccess::rparams(rcn);
                    const auto bl = rc::reloc::beyond_lines(pts2, rp);
                    auto ell = [&](const V3& x) { return rc::reloc::log_likelihood(poly, pts2, x, rp, 20.f, nullptr, &bl); };
                    std::println("      {} move f{} t+{:.1f}s {:.2f} m {:.0f} deg  pre ({:+.2f},{:+.2f},{:+.0f}) pub ({:+.2f},{:+.2f},{:+.0f}) truth ({:+.2f},{:+.2f},{:+.0f}) cross={} ell pre {:.0f} pub {:.0f} truth {:.0f}  d(pre) {:.3f} s_tape {:.3f}  doors {}  | no-door ell pre {:.0f} pub {:.0f}; d(pub) {:.3f}",
                                 cfg.mode, f, (fr.t_ms - T.frames[cfg.a].t_ms) * 1e-3, o.move_m, deg(o.move_rad), pt.x(), pt.y(), deg(pt.z()), P.x(), P.y(), deg(P.z()),
                                 fi.truth.x(), fi.truth.y(), deg(fi.truth.z()), static_cast<int>(cfg.d->crossing[f]), ell(pt), ell(P), fi.has ? ell(fi.truth) : 0.f, median_d(poly, pts2, pt), fi.s_tape, rp.apertures.size(),
                                 [&] { auto q = rp; q.apertures.clear(); return rc::reloc::log_likelihood(poly, pts2, pt, q, 20.f); }(),
                                 [&] { auto q = rp; q.apertures.clear(); return rc::reloc::log_likelihood(poly, pts2, P, q, 20.f); }(), median_d(poly, pts2, P));
                    std::fflush(stdout);
                }
                C = comp(P, inv(tp));
                o.latency_ms = rivals ? std::max(0.f, rc::RelocSelftestAccess::last_search_ms(rcn)) : 0.f;
            }
            const M3 cov = r.covariance + rc::RelocSelftestAccess::spread(rcn);
            o.cov_maxdiag = cov.diagonal().maxCoeff();
            o.sigma = std::sqrt(std::max(0.f, Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f>(r.covariance.topLeftCorner<2, 2>()).eigenvalues().maxCoeff()));
            out.fr.push_back(o);
        }
        if (rivals) for (float u : rc::RelocSelftestAccess::timing(rcn)) out.hot_us.push_back(u);
        return out;
    }

    // ---- E1: healthy replays -------------------------------------------------------------------------------------------------------
    struct Healthy
    {
        double healthy_s = 0, total_s = 0;
        int moves = 0, false_moves = 0, harmful = 0, false_after_5s = 0;
        double max_move = 0;
        std::vector<double> err_pub, err_trk, err_ang_pub;
        double max_cov = 0;
        std::vector<double> hot_us, step_ms;
        void add(const Data& d, const RunCfg& cfg, const RunOut& ro)
        {
            const auto& T = d.tape;
            for (std::size_t i = 0; i < ro.fr.size(); ++i)
            {
                const auto& o = ro.fr[i]; const auto& fi = d.fi[o.f];
                const double dt = i + 1 < ro.fr.size() ? (T.frames[ro.fr[i + 1].f].t_ms - T.frames[o.f].t_ms) * 1e-3 : 0.05;
                total_s += dt;
                const bool healthy_before = fi.has and (o.pre.head<2>() - fi.truth.head<2>()).norm() < 0.05f;
                if (healthy_before) healthy_s += dt;
                if (fi.has and (d.push_f < 0 or o.f < d.push_f))
                {
                    err_pub.push_back((o.pub.head<2>() - fi.truth.head<2>()).norm());
                    err_trk.push_back((T.frames[o.f].pose.head<2>() - fi.truth.head<2>()).norm());
                    err_ang_pub.push_back(std::abs(wrap(o.pub.z() - fi.truth.z())));
                }
                if (d.push_f < 0 or o.f < d.push_f) max_cov = std::max(max_cov, static_cast<double>(o.cov_maxdiag));
                if (o.moved)
                {
                    ++moves; max_move = std::max<double>(max_move, o.move_m);
                    if (healthy_before and o.move_m > std::max(3.f * o.sigma, 0.05f))
                    {
                        ++false_moves;
                        if (T.frames[o.f].t_ms - T.frames[cfg.a].t_ms > 5000) ++false_after_5s;
                        if (fi.has and (o.pub.head<2>() - fi.truth.head<2>()).norm() >= 0.05f) ++harmful;
                    }
                }
                step_ms.push_back(o.step_ms);
            }
            hot_us.insert(hot_us.end(), ro.hot_us.begin(), ro.hot_us.end());
        }
    };

    // ---- E2: injected kidnaps ----------------------------------------------------------------------------------------------------
    struct Kid { int vi = 0, k = 0; int kind = 0; V3 wrong = V3::Zero(); bool parked = false; };
    const char* kname(int k) { return k == 0 ? "trans 1-3.5 m" : k == 1 ? "rot +-90" : k == 2 ? "flip 180" : "trans+rot"; }
    struct KRes { bool ok = false; double t_rec = std::nan(""); bool ambiguous = false; };

    constexpr int kWarm = 100, kHorizon = 170;

    std::vector<Kid> make_kidnaps(const Data& d, int n, std::uint64_t seed)
    {
        std::mt19937_64 rng(seed);
        std::vector<Kid> out;
        std::vector<int> cand;   // frames with enough warm-up and horizon inside a visit
        for (std::size_t vi = 0; vi < d.visits.size(); ++vi)
            for (int f = d.visits[vi].a + kWarm; f < d.visits[vi].b - kHorizon; ++f)
                if (d.fi[f].has and (d.push_f < 0 or f + kHorizon < d.push_f) and std::none_of(d.crossing.begin() + f - kWarm, d.crossing.begin() + f + kHorizon, [](char c) { return c != 0; })) cand.push_back(f);
        if (cand.empty()) return out;
        auto parked_at = [&](int f)
        {
            const auto& T = d.tape.frames;
            int g = f; while (g > 0 and T[f].t_ms - T[g].t_ms < 500) --g;
            int h = f; while (h + 1 < static_cast<int>(T.size()) and T[h].t_ms - T[f].t_ms < 500) ++h;
            const double dt = (T[h].t_ms - T[g].t_ms) * 1e-3;
            if (dt <= 0) return false;
            return (T[h].pose.head<2>() - T[g].pose.head<2>()).norm() / dt < 0.03 and std::abs(wrap(T[h].pose.z() - T[g].pose.z())) / dt < 0.05;
        };
        std::vector<int> park, move;
        for (int f : cand) (parked_at(f) ? park : move).push_back(f);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        for (int i = 0; i < n; ++i)
        {
            Kid k;
            k.kind = i % 4;
            const bool want_park = (i / 4) % 2 == 0 and not park.empty();
            const auto& pool = want_park ? park : (move.empty() ? park : move);
            k.k = pool[rng() % pool.size()];
            k.parked = want_park;
            k.vi = d.visit_of[k.k];
            const V3 tp = d.tape.frames[k.k].pose;
            const Poly& poly = d.tape.rooms[d.tape.frame_room[k.k]].polygon;
            for (int tries = 0; tries < 1000; ++tries)
            {
                V3 w = tp;
                if (k.kind == 0 or k.kind == 3)
                {
                    const float r = k.kind == 0 ? 1.f + 2.5f * u01(rng) : 1.f + 3.f * u01(rng), a = -kPi + 2 * kPi * u01(rng);
                    w.x() += r * std::cos(a); w.y() += r * std::sin(a);
                }
                if (k.kind == 1) w.z() = wrap(tp.z() + (u01(rng) < 0.5f ? kPi / 2 : -kPi / 2));
                if (k.kind == 2) w.z() = wrap(tp.z() + kPi);
                if (k.kind == 3) w.z() = wrap(tp.z() + (u01(rng) < 0.5f ? 1.f : -1.f) * (kPi / 2 + (kPi / 2) * u01(rng)));
                if (rc::reloc::inside_polygon(poly, w.head<2>()) and rc::reloc::polygon_distance(poly, w.head<2>()) > 0.3f) { k.wrong = w; break; }
            }
            out.push_back(k);
        }
        return out;
    }

    KRes run_kidnap(const Data& d, const Kid& k, const std::string& mode, Pert pert, std::uint64_t seed)
    {
        RunCfg c;
        c.mode = mode; c.d = &d; c.a = k.k - kWarm; c.b = k.k + kHorizon; c.kidnap_f = k.k; c.wrong = k.wrong; c.pert = pert; c.seed = seed;
        const auto ro = run(c);
        KRes r;
        const auto& T = d.tape.frames;
        auto good = [&](const FOut& o)
        {
            const auto& fi = d.fi[o.f];
            return fi.has and (o.pub.head<2>() - fi.truth.head<2>()).norm() < 0.10f and std::abs(wrap(o.pub.z() - fi.truth.z())) < 5.f * kPi / 180.f;
        };
        int first = -1;
        for (std::size_t i = 0; i < ro.fr.size(); ++i)
        {
            if (ro.fr[i].f < k.k) continue;
            // a recovery = within 10 cm / 5 deg for 3 frames running
            if (i + 2 < ro.fr.size() and good(ro.fr[i]) and good(ro.fr[i + 1]) and good(ro.fr[i + 2])) { first = static_cast<int>(i); break; }
        }
        if (std::getenv("RELOC_KTRACE") != nullptr)
        {
            const auto& e = ro.fr.back(); const auto& fi = d.fi[e.f];
            std::println("      K {} kind {} {} v{} k{} wrong ({:+.2f},{:+.2f},{:+.0f}) tp ({:+.2f},{:+.2f},{:+.0f}) first {} end err {:.2f} m {:.0f} deg",
                         mode, k.kind, k.parked ? "parked" : "moving", k.vi, k.k, k.wrong.x(), k.wrong.y(), deg(k.wrong.z()), T[k.k].pose.x(), T[k.k].pose.y(), deg(T[k.k].pose.z()),
                         first, (e.pub.head<2>() - fi.truth.head<2>()).norm(), deg(std::abs(wrap(e.pub.z() - fi.truth.z()))));
        }
        {   // not recovered: is the pose it ended on a pose the SCAN cannot tell from the truth (room-only ell within 3 nats)? a symmetric twin
            const auto& e = ro.fr.back(); const auto& fi = d.fi[e.f];
            std::vector<V2> p2; const auto& pr = T[e.f].pts_robot;
            for (std::size_t i = 0; i < pr.size(); i += std::max<std::size_t>(1, pr.size() / 150)) p2.push_back(pr[i]);
            const Poly& poly = d.tape.rooms[d.tape.frame_room[e.f]].polygon;
            rc::reloc::Params rp;
            r.ambiguous = fi.has and rc::reloc::log_likelihood(poly, p2, e.pub, rp, 20.f) >= rc::reloc::log_likelihood(poly, p2, fi.truth, rp, 20.f) - 3.f;
        }
        if (first < 0) return r;
        int ok = 0, n = 0;
        for (std::size_t i = first; i < ro.fr.size(); ++i) { ++n; ok += good(ro.fr[i]) ? 1 : 0; }
        r.ok = ok >= 0.8 * n;   // and it STAYS there (a measurement-noise frame or two aside)
        r.t_rec = (T[ro.fr[first].f].t_ms - T[k.k].t_ms) * 1e-3 + ro.fr[first].latency_ms * 1e-3;
        // latency: the rival path's worker search runs async live; the replay waits for it between frames
        for (int i = 0; i <= first; ++i) if (ro.fr[i].moved and ro.fr[i].f >= k.k) { r.t_rec = (T[ro.fr[i].f].t_ms - T[k.k].t_ms) * 1e-3 + ro.fr[i].latency_ms * 1e-3; break; }
        return r;
    }

    // ---- E3: mirrored (180-degree point symmetric) room, healthy segments --------------------------------------------------------
    std::vector<V2> synth_scan(const Poly& poly, const V3& pose, int n, float noise, float clutter, std::mt19937& rng)
    {
        std::normal_distribution<float> nz(0.f, noise);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        std::vector<V2> out;
        for (int i = 0; i < n; ++i)
        {
            const float bearing = -kPi + 2.f * kPi * static_cast<float>(i) / static_cast<float>(n);
            const float wd = pose.z() + bearing;
            const V2 dir(std::cos(wd), std::sin(wd));
            float best = std::numeric_limits<float>::max();
            for (std::size_t k = 0; k < poly.size(); ++k)
            {
                const V2 a = poly[k], b = poly[(k + 1) % poly.size()], e = b - a;
                const float den = dir.x() * e.y() - dir.y() * e.x();
                if (std::abs(den) < 1e-9f) continue;
                const V2 w = a - pose.head<2>();
                const float s = (w.x() * e.y() - w.y() * e.x()) / den, t = (w.x() * dir.y() - w.y() * dir.x()) / den;
                if (s > 1e-4f and t >= 0.f and t <= 1.f) best = std::min(best, s);
            }
            if (best == std::numeric_limits<float>::max()) continue;
            float r = best + nz(rng);
            if (u01(rng) < clutter) r = std::max(0.3f, best * (0.2f + 0.7f * u01(rng)));
            out.emplace_back(r * std::cos(bearing), r * std::sin(bearing));
        }
        return out;
    }

    struct Mirror { int segments = 0; double seconds = 0; int moves = 0, flips = 0; double max_move = 0; std::vector<double> err; };
    Mirror run_mirror(const std::string& mode, int segments, std::uint64_t seed)
    {
        Mirror m;
        const Poly poly = {{-3.f, -2.f}, {3.f, -2.f}, {3.f, 2.f}, {-3.f, 2.f}};   // 6 x 4 m: indistinguishable under the half-turn
        for (int sgi = 0; sgi < segments; ++sgi)
        {
            std::mt19937 rng(static_cast<unsigned>(seed + 7919 * sgi));
            std::normal_distribution<float> n01(0.f, 1.f);
            std::uniform_real_distribution<float> u01(0.f, 1.f);
            rc::RoomConcept rcn;
            rc::RelocSelftestAccess::set_mode(rcn, mode);
            rcn.set_polygon_room(poly);
            const float ph0 = 2 * kPi * u01(rng), wx = 0.9f + 0.3f * u01(rng), wy = 0.7f + 0.3f * u01(rng), dir = u01(rng) < 0.5f ? 1.f : -1.f;
            auto truth_at = [&](double t)   // a slow Lissajous drive, heading along the path
            {
                const float ph = ph0 + 0.07f * static_cast<float>(t);
                const V2 p(2.0f * std::sin(wx * ph), 1.1f * std::sin(wy * ph + 0.5f));
                const V2 v(2.0f * wx * std::cos(wx * ph), 1.1f * wy * std::cos(wy * ph + 0.5f));
                return V3(p.x(), p.y(), dir > 0 ? std::atan2(v.y(), v.x()) : wrap(std::atan2(v.y(), v.x()) + kPi));
            };
            V3 prev = truth_at(0);
            rcn.set_robot_pose(prev.x(), prev.y(), prev.z(), false);
            const int nf = 17 * 60;   // 60 s at the live rate
            for (int f = 0; f < nf; ++f)
            {
                const double t = f / 17.0;
                const V3 tr = truth_at(t);
                const V3 pose(tr.x() + 0.005f * n01(rng), tr.y() + 0.005f * n01(rng), wrap(tr.z() + 0.002f * n01(rng)));
                rcn.set_robot_pose(pose.x(), pose.y(), pose.z(), false);
                rc::RoomConcept::UpdateResult r;
                r.ok = true;
                r.robot_pose = Eigen::Affine2f::Identity();
                r.robot_pose.translation() = pose.head<2>();
                r.robot_pose.linear() = Eigen::Rotation2Df(pose.z()).toRotationMatrix();
                r.covariance = Eigen::Vector3f(9e-4f, 9e-4f, 1e-4f).asDiagonal();
                r.pred_x = pose.x(); r.pred_y = pose.y(); r.pred_theta = pose.z();
                const auto sc = synth_scan(poly, tr, 360, 0.02f, 0.30f, rng);
                std::vector<V3> p3; for (const auto& q : sc) p3.emplace_back(q.x(), q.y(), 0.f);
                const float s = median_d(poly, sc, pose);
                r.sdf_mse = s; r.pred_sdf_median = s;
                r.timestamp_ms = 1000 + static_cast<std::int64_t>(f * 1000 / 17);
                const V3 dl = comp(inv(prev), tr);
                rc::RelocSelftestAccess::set_odo(rcn, V3(tr.x() - prev.x(), tr.y() - prev.y(), wrap(tr.z() - prev.z())), M3(Eigen::Vector3f(1e-5f, 1e-5f, 1e-6f).asDiagonal()));
                (void)dl;
                prev = tr;
                const auto e0 = rc::RelocSelftestAccess::epoch(rcn);
                rc::RelocSelftestAccess::step8(rcn, r, p3);
                if (mode == "rivals") rc::RelocSelftestAccess::wait_proposer(rcn);
                if (rc::RelocSelftestAccess::epoch(rcn) != e0)
                {
                    const auto st = rcn.get_current_state();
                    const V3 P(st[2], st[3], st[4]);
                    const double mv = (P.head<2>() - pose.head<2>()).norm();
                    ++m.moves; m.max_move = std::max(m.max_move, mv);
                    if (mv > 1.0 or std::abs(wrap(P.z() - pose.z())) > kPi / 2) ++m.flips;
                }
                const auto st = rcn.get_current_state();
                m.err.push_back((V2(st[2], st[3]) - tr.head<2>()).norm());
            }
            m.seconds += nf / 17.0;
            ++m.segments;
        }
        return m;
    }
}

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");   // the agents are Qt programs: make the C locale es_ES here too, or a parsing bug would not show
    qInstallMessageHandler([](QtMsgType, const QMessageLogContext&, const QString&) {});
    int n_kid = 300, n_mirror = 12, n_kid_people = 100;
    std::uint64_t seed = 20261104;
    bool skip_healthy = false;
    std::vector<std::pair<std::string, std::string>> inputs;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--kidnaps" and i + 1 < argc) n_kid = std::atoi(argv[++i]);
        else if (a == "--seed" and i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--mirror" and i + 1 < argc) n_mirror = std::atoi(argv[++i]);
        else if (a == "--people-kidnaps" and i + 1 < argc) n_kid_people = std::atoi(argv[++i]);
        else if (a == "--skip-healthy") skip_healthy = true;
        else if (i + 1 < argc) { inputs.emplace_back(a, argv[i + 1]); ++i; }
    }
    if (inputs.empty()) { std::println(stderr, "usage: reloc_tape_replay [--kidnaps N] [--seed S] [--mirror M] TAPE GT [TAPE GT ...]"); return 2; }

    std::vector<Data> data;
    for (const auto& [t, g] : inputs) data.push_back(load(t, g));
    for (const auto& d : data)
    {
        std::println("== {}: {} frames, {} visits, {} live relocaliser move(s) removed from the tracker poses, push detected at frame {}", d.name, d.tape.frames.size(),
                     d.visits.size(), d.n_live_moves, d.push_f);
        for (const auto& v : d.visits)
            std::println("   visit frames [{},{}) room '{}' polygon {} v: gt alignment rho {:+.2f} deg, residual rms {:.1f} cm", v.a, v.b,
                         d.tape.rooms[v.rec].room, d.tape.rooms[v.rec].polygon.size(), v.rho * 180 / std::numbers::pi, v.rms * 100);
    }

    // sanity: the statistic the replay feeds the legacy trigger, at the tape's own pose, against the tracker's recorded one
    for (const auto& d : data)
        for (const auto& v : d.visits)
        {
            std::vector<double> dt, st, ell;
            const Poly& poly = d.tape.rooms[v.rec].polygon;
            rc::reloc::Params rp;
            for (int f = v.a; f < v.b; f += 5)
            {
                std::vector<V2> p2; const auto& pr = d.tape.frames[f].pts_robot;
                const int stride = std::max<int>(1, static_cast<int>(pr.size()) / 150);
                for (std::size_t i = 0; i < pr.size(); i += stride) p2.push_back(pr[i]);
                dt.push_back(median_d(poly, p2, d.tape.frames[f].pose)); st.push_back(d.fi[f].s_tape);
                ell.push_back(rc::reloc::log_likelihood(poly, p2, d.tape.frames[f].pose, rp, 20.f));
            }
            std::println("   [{}] visit [{},{}): median|SDF| at the tape pose on the replay's points: p50 {:.3f} p90 {:.3f} m; tracker's recorded sdf_mse p50 {:.3f} p90 {:.3f}; ell p50 {:.0f}",
                         d.label, v.a, v.b, pct(dt, 0.5), pct(dt, 0.9), pct(st, 0.5), pct(st, 0.9), pct(ell, 0.5));
        }
    const std::vector<std::string> modes = {"legacy", "rivals"};
    // ---- E1 -------------------------------------------------------------------------------------------------------------------
    // ---- E1b: a push / drag recorded on a tape ---------------------------------------------------------------------------------
    auto push_report = [&](const Data& d, const std::string& mode, const RunOut& ro, int a)
    {
        const auto& T = d.tape.frames;
        const std::int64_t t_push = T[d.push_f].t_ms;
        int first = -1, moves_after = 0, moves_before = 0;
        auto good = [&](const FOut& o)
        {
            const auto& fi = d.fi[o.f];
            return fi.has and (o.pub.head<2>() - fi.truth.head<2>()).norm() < 0.10f and std::abs(wrap(o.pub.z() - fi.truth.z())) < 5.f * kPi / 180.f;
        };
        for (std::size_t i = 0; i < ro.fr.size(); ++i)
        {
            if (ro.fr[i].moved) (ro.fr[i].f >= d.push_f ? moves_after : moves_before)++;
            if (first < 0 and ro.fr[i].f >= d.push_f and i + 2 < ro.fr.size() and good(ro.fr[i]) and good(ro.fr[i + 1]) and good(ro.fr[i + 2])) first = static_cast<int>(i);
        }
        std::vector<double> ep, ea;
        for (std::size_t i = ro.fr.size() > 40 ? ro.fr.size() - 40 : 0; i < ro.fr.size(); ++i)
        {
            const auto& fi = d.fi[ro.fr[i].f];
            if (fi.has) { ep.push_back((ro.fr[i].pub.head<2>() - fi.truth.head<2>()).norm()); ea.push_back(std::abs(wrap(ro.fr[i].pub.z() - fi.truth.z()))); }
        }
        std::println("   [{}] {:<7} push at tape+{:.1f} s (frame {}; tracker-vs-gt off by >10 cm or >10 deg for 40 frames); moves before {} / after {}; "
                     "{}; final error (last 40 frames median) {:.1f} cm, {:.1f} deg", d.label, mode, (t_push - T[a].t_ms) * 1e-3, d.push_f, moves_before, moves_after,
                     first < 0 ? std::string("NEVER within 10 cm / 5 deg of the truth")
                               : std::format("within 10 cm / 5 deg {:.2f} s after the push", (T[ro.fr[first].f].t_ms - t_push) * 1e-3),
                     pct(ep, 0.5) * 100, pct(ea, 0.5) * 180 / std::numbers::pi);
    };

    if (not skip_healthy)
    {
        std::println("\n== E1 healthy replays (the tape's tracker, scans as recorded or perturbed)");
        std::println("{:<10} {:<10} {:<9} {:>8} {:>8} {:>6} {:>6} {:>7} {:>7} {:>9} {:>9} {:>9} {:>9} {:>8}", "tape", "mode", "pert", "total_s", "healthy_s",
                     "moves", "false", "harmful", "f>5s", "worst_cm", "p99_cm", "pub_worst", "trk_worst", "maxcov");
        for (const auto& d : data)
            for (const Pert pert : {Pert::None, Pert::Clutter, Pert::Person})
                for (const auto& mode : modes)
                {
                    Healthy h;
                    for (const auto& v : d.visits)
                    {
                        RunCfg c; c.mode = mode; c.d = &d; c.a = v.a; c.b = v.b; c.pert = pert; c.seed = seed;
                        const auto ro = run(c);
                        h.add(d, c, ro);
                        if (pert == Pert::None and d.push_f >= v.a and d.push_f < v.b) push_report(d, mode, ro, v.a);
                    }
                    const double w = h.err_pub.empty() ? 0 : *std::ranges::max_element(h.err_pub), tw = h.err_trk.empty() ? 0 : *std::ranges::max_element(h.err_trk);
                    std::println("{:<10} {:<10} {:<9} {:>8.1f} {:>8.1f} {:>6} {:>6} {:>7} {:>7} {:>9.1f} {:>9.1f} {:>9.1f} {:>9.1f} {:>8.4f}",
                                 d.label, mode, pname(pert), h.total_s, h.healthy_s, h.moves,
                                 h.false_moves, h.harmful, h.false_after_5s, w * 100, pct(h.err_pub, 0.99) * 100, w * 100, tw * 100, h.max_cov);
                    std::println("    angle p99 {:.2f} deg; step ms p50 {:.2f} p99 {:.2f} max {:.2f}; hot-path us p50 {:.0f} p99 {:.0f} max {:.0f} (n={}); false/h {:.0f}",
                                 pct(h.err_ang_pub, 0.99) * 180 / std::numbers::pi, pct(h.step_ms, 0.5), pct(h.step_ms, 0.99), h.step_ms.empty() ? 0.0 : *std::ranges::max_element(h.step_ms),
                                 pct(h.hot_us, 0.5), pct(h.hot_us, 0.99), h.hot_us.empty() ? 0.0 : *std::ranges::max_element(h.hot_us), h.hot_us.size(),
                                 h.healthy_s > 0 ? h.false_moves * 3600.0 / h.healthy_s : 0.0);
                    std::fflush(stdout);
                }
    }

    // ---- E2 -------------------------------------------------------------------------------------------------------------------
    auto kidnap_table = [&](const char* title, int n, Pert pert)
    {
        std::println("\n== E2 injected kidnaps: {} ({} events per mode, {})", title, n, pname(pert));
        // events spread over the tapes in proportion to their frames
        std::vector<std::pair<const Data*, Kid>> ev;
        std::size_t total = 0; for (const auto& d : data) total += d.tape.frames.size();
        for (std::size_t di = 0; di < data.size(); ++di)
        {
            const int nd = static_cast<int>(std::lround(static_cast<double>(n) * data[di].tape.frames.size() / total));
            for (auto& k : make_kidnaps(data[di], nd, seed + di)) ev.emplace_back(&data[di], k);
        }
        for (const auto& mode : modes)
        {
            std::vector<KRes> res(ev.size());
            std::atomic<std::size_t> next{0};
            auto worker = [&]() { for (std::size_t i; (i = next.fetch_add(1)) < ev.size();) res[i] = run_kidnap(*ev[i].first, ev[i].second, mode, pert, seed); };
            std::vector<std::thread> th;
            const unsigned nthreads = std::max(1u, std::min(6u, std::thread::hardware_concurrency() / 2));
            for (unsigned t = 0; t < nthreads; ++t) th.emplace_back(worker);
            for (auto& t : th) t.join();
            auto summarize = [&](const char* label, auto pred)
            {
                std::vector<double> tr; int n_ev = 0, n_ok = 0, n_15 = 0, n_amb = 0;
                for (std::size_t i = 0; i < ev.size(); ++i)
                {
                    if (not pred(ev[i].second)) continue;
                    ++n_ev;
                    if (res[i].ok) { ++n_ok; tr.push_back(res[i].t_rec); if (res[i].t_rec <= 1.5) ++n_15; }
                    else if (res[i].ambiguous) ++n_amb;
                }
                // unrecovered events count as infinity in the percentiles
                std::vector<double> all = tr; all.resize(static_cast<std::size_t>(n_ev), std::numeric_limits<double>::infinity());
                std::println("   {:<8} {:<16} n={:<4} recovered {:>5.1f}% (<=1.5 s {:>5.1f}%) + ended on a scan-indistinguishable twin {:>4.1f}% = {:>5.1f}%  t_rec median {:.2f} s  p95 {:.2f} s  max(rec) {:.2f} s", mode, label, n_ev,
                             n_ev ? 100.0 * n_ok / n_ev : 0.0, n_ev ? 100.0 * n_15 / n_ev : 0.0, n_ev ? 100.0 * n_amb / n_ev : 0.0, n_ev ? 100.0 * (n_ok + n_amb) / n_ev : 0.0, pct(all, 0.5), pct(all, 0.95), tr.empty() ? 0.0 : *std::ranges::max_element(tr));
            };
            summarize("ALL", [](const Kid&) { return true; });
            summarize("moving", [](const Kid& k) { return not k.parked; });
            summarize("parked", [](const Kid& k) { return k.parked; });
            for (int kd = 0; kd < 4; ++kd) summarize(kname(kd), [kd](const Kid& k) { return k.kind == kd; });
            std::fflush(stdout);
        }
    };
    kidnap_table("clean scans", n_kid, Pert::None);
    if (n_kid_people > 0) kidnap_table("people", n_kid_people, Pert::Person);

    // ---- E3 -------------------------------------------------------------------------------------------------------------------
    if (n_mirror > 0)
    {
        std::println("\n== E3 mirrored (180-degree point-symmetric 6x4 m) room, healthy segments, {} x 60 s per mode", n_mirror);
        for (const auto& mode : modes)
        {
            const auto m = run_mirror(mode, n_mirror, seed);
            std::println("   {:<8} {:>5.0f} s healthy: moves {}, flips (>1 m or >90 deg) {}, largest move {:.2f} m, pose error p99 {:.1f} cm max {:.1f} cm", mode, m.seconds, m.moves, m.flips,
                         m.max_move, pct(m.err, 0.99) * 100, m.err.empty() ? 0.0 : *std::ranges::max_element(m.err) * 100);
            std::fflush(stdout);
        }
    }
    return 0;
}
