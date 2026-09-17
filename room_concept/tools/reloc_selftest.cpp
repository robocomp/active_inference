/*
 *  reloc_selftest.cpp — offline A/B of relocalisation on INJECTED lost-robot episodes.
 *
 *  A harness without the defect cannot validate its cure, so every episode here IS the defect: the
 *  localiser is handed a WRONG pose (a 180° flip, a mirror image, a small drift, or a uniform kidnap)
 *  together with a scan taken at the TRUE pose, and each method has to recover.
 *
 *  CONTROL   RoomConcept::legacy_grid_search_pose() — the pre-09-16 4-stage lattice, frozen verbatim.
 *  TREATMENT rc::reloc::search() — batched robust likelihood + structural yaws + mixture posterior.
 *
 *  ENDPOINTS (per room × kidnap kind)
 *    ok      the committed / top-weight pose is within 0.10 m and 5° of the truth OR of a pose that is
 *            genuinely indistinguishable from it (the room's symmetry images). Committing a symmetric
 *            twin is not an error; committing anything else is.
 *    truth   (treatment) some returned mode is within tolerance of the truth itself — the belief still
 *            CONTAINS the right answer even when a symmetric twin carries the top weight.
 *    honest  (treatment, symmetric room only) the truth's mode and its twin each carry weight in
 *            [0.25, 0.75] — i.e. the posterior reports the ambiguity instead of arbitrarily resolving it.
 *    nees    mean e'Σ⁻¹e / 3 of the top mode on correct recoveries (≈1 = calibrated covariance).
 *    ms      wall time per search.
 *  REFUTED if the treatment does not beat the control's ok-rate at lower ms, or its NEES is far from 1,
 *  or on the symmetric room it is NOT honest.
 *
 *  Build:  make -C build reloc_selftest && ./bin/reloc_selftest
 */
#include <charconv>
#include <chrono>
#include <string_view>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "reloc_search.h"
#include "room_concept.h"
#include "epistemic_planner.h"

namespace rc
{
    /// Test access to the relocalisation runtime (declared friend in RoomConcept).
    struct RelocSelftestAccess
    {
        static float belief(RoomConcept& r, float s, float outside) { return r.update_lost_belief(s, outside); }
        static void  reset(RoomConcept& r) { r.reset_lost_belief(); }
        static void  step(RoomConcept& r, const std::vector<Eigen::Vector3f>& pts) { r.step_rival_modes(pts); }
        static auto& rivals(RoomConcept& r) { return r.rivals_; }
        static void  set_ref(RoomConcept& r, const Eigen::Vector3f& p) { r.rival_ref_pose_ = p; r.rival_ref_valid_ = true; }
        static bool  confirmed(RoomConcept& r) { return r.last_search_confirmed_incumbent_; }
        static bool  moved(RoomConcept& r) { return r.last_search_moved_; }
        static std::uint64_t epoch(RoomConcept& r) { return r.reloc_epoch_.load(); }
        static void  step8(RoomConcept& r, const RoomConcept::UpdateResult& res, const std::vector<Eigen::Vector3f>& pts)
        { r.relocalisation_step(res, pts); }
        static void  force_searching_with_map_ready(RoomConcept& r)
        { r.runtime_estimate_.store(true); r.map_ready_.store(true); r.wall_frozen_ = false; }
        static bool  searching(RoomConcept& r) { return r.searching(); }
        static float track_var(RoomConcept& r) { return r.track_var_ln_s_; }
        static std::vector<Eigen::Vector2f> poly(RoomConcept& r) { return r.current_room_polygon(); }
        static bool  explains(RoomConcept& r, float s) { return r.explains_like_tracking(s); }
        static rc::reloc::Params rparams(RoomConcept& r) { return r.reloc_params(); }
        static float& track_mu(RoomConcept& r) { return r.track_mu_ln_s_; }
        static void  relocalise(RoomConcept& r, const std::vector<Eigen::Vector3f>& pts, float s, float p)
        { r.relocalise_from_belief(pts, s, p); }
    };
}

namespace
{
    constexpr float kPi = std::numbers::pi_v<float>;
    float wrap(float a) { return std::remainder(a, 2.f * kPi); }

    using Poly = std::vector<Eigen::Vector2f>;
    using SymFn = std::function<Eigen::Vector3f(const Eigen::Vector3f&)>;

    struct Room
    {
        std::string name;
        Poly poly;
        std::vector<SymFn> symmetries;   // pose maps under which the room (and so every scan) is identical
    };

    std::vector<Room> rooms()
    {
        std::vector<Room> r;
        // 6 × 4 m rectangle centred at the origin: indistinguishable under the half-turn.
        r.push_back({"rect_6x4",
                     {{-3.f, -2.f}, {3.f, -2.f}, {3.f, 2.f}, {-3.f, 2.f}},
                     {[](const Eigen::Vector3f& p) { return Eigen::Vector3f(-p.x(), -p.y(), wrap(p.z() + kPi)); }}});
        // L-shape: no symmetry at all.
        r.push_back({"L_7x5",
                     {{0.f, 0.f}, {7.f, 0.f}, {7.f, 2.5f}, {3.f, 2.5f}, {3.f, 5.f}, {0.f, 5.f}},
                     {}});
        // 8 × 5 rectangle with a 1 × 1 notch in one corner: NEAR-symmetric — the lattice's hard case.
        r.push_back({"notched_8x5",
                     {{0.f, 0.f}, {8.f, 0.f}, {8.f, 4.f}, {7.f, 4.f}, {7.f, 5.f}, {0.f, 5.f}},
                     {}});
        return r;
    }

    /// Ray-cast the polygon from `pose`; robot-frame hits. `clutter` of the beams stop short (furniture);
    /// a 12° sector looks through an "open door" and returns a point 0.5–3 m beyond the wall.
    std::vector<Eigen::Vector2f> synth_scan(const Poly& poly, const Eigen::Vector3f& pose, int n,
                                            float noise, float clutter, std::mt19937& rng,
                                            float door_half_width_deg = 6.f)
    {
        std::normal_distribution<float> nz(0.f, noise);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        const float door_centre = -kPi + 2.f * kPi * u01(rng);
        std::vector<Eigen::Vector2f> out;
        out.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            const float bearing = -kPi + 2.f * kPi * static_cast<float>(i) / static_cast<float>(n);
            const float wd = pose.z() + bearing;
            const Eigen::Vector2f dir(std::cos(wd), std::sin(wd));
            float best = std::numeric_limits<float>::max();
            for (std::size_t k = 0; k < poly.size(); ++k)
            {
                const Eigen::Vector2f a = poly[k], b = poly[(k + 1) % poly.size()];
                const Eigen::Vector2f e = b - a;
                const float den = dir.x() * e.y() - dir.y() * e.x();
                if (std::abs(den) < 1e-9f) continue;
                const Eigen::Vector2f w = a - pose.head<2>();
                const float s = (w.x() * e.y() - w.y() * e.x()) / den;
                const float t = (w.x() * dir.y() - w.y() * dir.x()) / den;
                if (s > 1e-4f and t >= 0.f and t <= 1.f) best = std::min(best, s);
            }
            if (best == std::numeric_limits<float>::max()) continue;
            float r = best + nz(rng);
            if (std::abs(wrap(bearing - door_centre)) < door_half_width_deg * kPi / 180.f)
                r = best + 0.5f + 2.5f * u01(rng);
            else if (u01(rng) < clutter)
                r = std::max(0.3f, best * (0.2f + 0.7f * u01(rng)));
            out.emplace_back(r * std::cos(bearing), r * std::sin(bearing));
        }
        return out;
    }

    bool near(const Eigen::Vector3f& a, const Eigen::Vector3f& b, float tol_m = 0.10f, float tol_deg = 5.f)
    {
        return (a.head<2>() - b.head<2>()).norm() < tol_m and std::abs(wrap(a.z() - b.z())) < tol_deg * kPi / 180.f;
    }
    bool near_or_twin(const Room& room, const Eigen::Vector3f& est, const Eigen::Vector3f& truth,
                      float tol_m = 0.10f, float tol_deg = 5.f)
    {
        if (near(est, truth, tol_m, tol_deg)) return true;
        return std::ranges::any_of(room.symmetries, [&](const SymFn& f) { return near(est, f(truth), tol_m, tol_deg); });
    }

    Eigen::Vector3f random_pose(const Poly& poly, std::mt19937& rng, float clearance)
    {
        Eigen::Vector2f lo = poly.front(), hi = poly.front();
        for (const auto& v : poly) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
        std::uniform_real_distribution<float> ux(lo.x(), hi.x()), uy(lo.y(), hi.y()), ut(-kPi, kPi);
        for (;;)
        {
            const Eigen::Vector2f q(ux(rng), uy(rng));
            if (rc::reloc::inside_polygon(poly, q) and rc::reloc::polygon_distance(poly, q) > clearance)
                return {q.x(), q.y(), ut(rng)};
        }
    }

    enum class Kind { Flip, Mirror, Drift, Uniform };
    const char* name(Kind k)
    {
        switch (k) { case Kind::Flip: return "flip180"; case Kind::Mirror: return "mirror";
                     case Kind::Drift: return "drift"; case Kind::Uniform: return "uniform"; }
        return "?";
    }

    Eigen::Vector3f kidnap(Kind k, const Poly& poly, const Eigen::Vector3f& truth, std::mt19937& rng)
    {
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        Eigen::Vector2f lo = poly.front(), hi = poly.front();
        for (const auto& v : poly) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
        const Eigen::Vector2f c = 0.5f * (lo + hi);
        switch (k)
        {
            case Kind::Flip:   return {truth.x(), truth.y(), wrap(truth.z() + kPi)};
            case Kind::Mirror: return {2.f * c.x() - truth.x(), truth.y(), wrap(kPi - truth.z())};
            case Kind::Drift:
            {
                const float r = 0.5f + 0.5f * u01(rng), a = 2.f * kPi * u01(rng);
                const float dth = (20.f + 20.f * u01(rng)) * kPi / 180.f * (u01(rng) < 0.5f ? -1.f : 1.f);
                return {truth.x() + r * std::cos(a), truth.y() + r * std::sin(a), wrap(truth.z() + dth)};
            }
            case Kind::Uniform: return random_pose(poly, rng, 0.3f);
        }
        return truth;
    }

    struct Tally
    {
        int n = 0, ok = 0, basin = 0, truth = 0, honest = 0, honest_n = 0, nees_n = 0;
        double ms = 0.0, nees = 0.0;
    };
} // namespace

int main()
{
    std::setlocale(LC_ALL, "");   // behave like the Qt agent (es_ES decimal comma) — see CLAUDE.md
    torch::set_num_threads(1);
    constexpr int kEpisodes = 25;
    std::printf("reloc_selftest: %d episodes per room x kind, 360-beam scans, 2 cm noise, 30%% clutter, "
                "one open-door sector\n\n", kEpisodes);
    std::printf("%-12s %-8s | %-36s | %-60s\n", "room", "kidnap", "CONTROL ok   basin  +polish     ms", "TREATMENT ok   basin  truth  honest   nees     ms  modes");

    for (const auto& room : rooms())
    {
        rc::reloc::Params rp;
        if (const char* sc = std::getenv("RELOC_SIGMA_C"); sc != nullptr)
        {
            std::string_view sv(sc);
            float v = rp.wall_offset_sigma;
            if (std::from_chars(sv.data(), sv.data() + sv.size(), v).ec == std::errc{}) rp.wall_offset_sigma = v;
        }
        rc::reloc::LikelihoodRaster raster;
        raster.build(room.poly, rp);

        for (const Kind kind : {Kind::Flip, Kind::Mirror, Kind::Drift, Kind::Uniform})
        {
            std::mt19937 rng(1000u + static_cast<unsigned>(kind) * 17u + static_cast<unsigned>(room.name.size()));
            Tally ctl, trt;
            double modes_sum = 0.0;
            for (int e = 0; e < kEpisodes; ++e)
            {
                const Eigen::Vector3f truth = random_pose(room.poly, rng, 0.6f);
                const auto scan = synth_scan(room.poly, truth, 360, 0.02f, 0.30f, rng);
                Eigen::Vector3f handed = kidnap(kind, room.poly, truth, rng);
                if (not rc::reloc::inside_polygon(room.poly, handed.head<2>()))
                    handed.head<2>() = truth.head<2>();   // a mirror of an L can leave the room; keep yaw error

                // ── CONTROL ──
                {
                    rc::RoomConcept rc_ctl;
                    rc_ctl.set_polygon_room(room.poly);
                    rc_ctl.set_robot_pose(handed.x(), handed.y(), handed.z(), false);
                    std::vector<Eigen::Vector3f> pts3;
                    pts3.reserve(scan.size());
                    for (const auto& p : scan) pts3.emplace_back(p.x(), p.y(), 0.f);
                    const auto t0 = std::chrono::steady_clock::now();
                    rc_ctl.legacy_grid_search_pose(pts3, 0.5f);   // frozen live constants inside
                    ctl.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    const auto st = rc_ctl.get_current_state();
                    const Eigen::Vector3f est(st[2], st[3], st[4]);
                    ++ctl.n;
                    if (near_or_twin(room, est, truth)) ++ctl.ok;
                    if (near_or_twin(room, est, truth, 0.30f, 10.f)) ++ctl.basin;
                    // Same continuous polish the treatment gets: the live window optimizer would refine the
                    // committed lattice point, so what the control is really judged on is the BASIN it chose.
                    const auto pol = rc::reloc::polish(room.poly, scan, est, rp, raster.extent());
                    if (near_or_twin(room, pol.pose, truth)) ++ctl.truth;
                }

                // ── TREATMENT ──
                {
                    std::vector<Eigen::Vector3f> seeds{handed};
                    const auto res = rc::reloc::search(raster, room.poly, scan, seeds, rp,
                                                       static_cast<std::uint32_t>(e + 1));
                    ++trt.n;
                    trt.ms += res.duration_ms;
                    modes_sum += static_cast<double>(res.modes.size());
                    if (res.modes.empty()) continue;
                    const auto& top = res.modes.front();
                    if (near_or_twin(room, top.pose, truth, 0.30f, 10.f)) ++trt.basin;
                    if (not near_or_twin(room, top.pose, truth) and std::getenv("RELOC_DUMP") != nullptr)
                    {
                        const auto yaws = rc::reloc::structural_yaws(room.poly, scan, rp, static_cast<std::uint32_t>(e + 1));
                        float best_dyaw = kPi;
                        for (const float y : yaws) best_dyaw = std::min(best_dyaw, std::abs(wrap(y - truth.z())));
                        // ★ BOTH on the same point set. The first version scored the top mode on the search's
                        //   150-point subsample and the truth on all 360 — a point-count gap that read as
                        //   "the polish stops 14–50 nats short" and was nothing of the kind.
                        float ll_truth_med = 0.f;
                        const float ll_truth = rc::reloc::log_likelihood(room.poly, scan, truth, rp, raster.extent(), &ll_truth_med);
                        const float ll_top_same = rc::reloc::log_likelihood(room.poly, scan, top.pose, rp, raster.extent());
                        std::printf("   FAIL %s/%s e%d truth(%.2f,%.2f,%.0f°) top(%.2f,%.2f,%.0f°) w=%.2f ll_top=%.1f ll_truth=%.1f "
                                    "segs=%d yaws=%d nearest_yaw=%.1f°\n",
                                    room.name.c_str(), name(kind), e, truth.x(), truth.y(), truth.z() * 180. / kPi,
                                    top.pose.x(), top.pose.y(), top.pose.z() * 180. / kPi, top.weight,
                                    ll_top_same, ll_truth, res.n_segments, res.n_yaws, best_dyaw * 180. / kPi);
                    }
                    if (near_or_twin(room, top.pose, truth)) ++trt.ok;
                    // NEES over every BASIN-correct episode, not only the ones inside the 0.10 m tolerance:
                    // conditioning on a small error would bias the NEES low and hide overconfidence.
                    if (near_or_twin(room, top.pose, truth, 0.30f, 10.f))
                    {
                        // NEES against whichever of truth / twin it matched.
                        Eigen::Vector3f ref = truth;
                        for (const auto& f : room.symmetries)
                            if (near(top.pose, f(truth), 0.30f, 10.f)) ref = f(truth);
                        Eigen::Vector3f err = top.pose - ref;
                        err.z() = wrap(err.z());
                        const double q = err.dot(top.cov.ldlt().solve(err)) / 3.0;
                        if (std::isfinite(q)) { trt.nees += q; ++trt.nees_n; }
                        if (std::getenv("RELOC_DOF") != nullptr)   // per-DOF: which direction is overconfident
                            std::printf("   DOF %s/%s ex=%+.3f sx=%.3f  ey=%+.3f sy=%.3f  eth=%+.2f° sth=%.2f°\n",
                                        room.name.c_str(), name(kind), err.x(), std::sqrt(top.cov(0, 0)),
                                        err.y(), std::sqrt(top.cov(1, 1)), err.z() * 180. / kPi,
                                        std::sqrt(top.cov(2, 2)) * 180. / kPi);
                    }
                    float w_truth = 0.f, w_twin = 0.f;
                    for (const auto& m : res.modes)
                    {
                        if (near(m.pose, truth)) { w_truth += m.weight; }
                        for (const auto& f : room.symmetries)
                            if (near(m.pose, f(truth))) w_twin += m.weight;
                    }
                    if (w_truth > 0.f) ++trt.truth;
                    if (not room.symmetries.empty())
                    {
                        ++trt.honest_n;
                        if (w_truth >= 0.25f and w_truth <= 0.75f and w_twin >= 0.25f and w_twin <= 0.75f) ++trt.honest;
                    }
                }
            }
            char honest[16] = "   n/a";
            if (trt.honest_n > 0) std::snprintf(honest, sizeof honest, "%3d/%-3d", trt.honest, trt.honest_n);
            std::printf("%-12s %-8s | %3d/%-3d %3d/%-3d %3d/%-3d %8.1f | %3d/%-3d %3d/%-3d %3d/%-3d %s %6.2f %6.1f  %4.2f\n",
                        room.name.c_str(), name(kind),
                        ctl.ok, ctl.n, ctl.basin, ctl.n, ctl.truth, ctl.n, ctl.ms / std::max(1, ctl.n),
                        trt.ok, trt.n, trt.basin, trt.n, trt.truth, trt.n, honest,
                        trt.nees_n > 0 ? trt.nees / trt.nees_n : std::nan(""),
                        trt.ms / std::max(1, trt.n), modes_sum / std::max(1, trt.n));
        }
    }
    // ══ CLOSED LOOP: the runtime that decides whether the published pose flips ══════════════════════
    std::printf("\nclosed loop (RoomConcept lost belief + rival modes, live defaults)\n");
    using Acc = rc::RelocSelftestAccess;
    const auto all_rooms = rooms();
    auto to3 = [](const std::vector<Eigen::Vector2f>& v)
    {
        std::vector<Eigen::Vector3f> o; o.reserve(v.size());
        for (const auto& p : v) o.emplace_back(p.x(), p.y(), 0.f);
        return o;
    };
    auto median_d = [](const Poly& poly, const std::vector<Eigen::Vector2f>& scan, const Eigen::Vector3f& pose)
    {
        float med = 0.f;
        rc::reloc::Params rp;
        rc::reloc::log_likelihood(poly, scan, pose, rp, 10.f, &med);
        return med;
    };
    auto compose = [](const Eigen::Vector3f& a, const Eigen::Vector3f& d)   // a ∘ d, d in a's frame
    {
        const float c = std::cos(a.z()), s = std::sin(a.z());
        return Eigen::Vector3f(a.x() + c * d.x() - s * d.y(), a.y() + s * d.x() + c * d.y(), wrap(a.z() + d.z()));
    };

    if (std::getenv("RELOC_POLY") != nullptr)
        for (const auto& room : all_rooms)
        {
            rc::RoomConcept rcn;
            rcn.set_polygon_room(room.poly);
            const auto pg = Acc::poly(rcn);
            std::printf("  poly %s: set %zu verts, model returns %zu:", room.name.c_str(), room.poly.size(), pg.size());
            for (const auto& v : pg) std::printf(" (%.2f,%.2f)", v.x(), v.y());
            std::printf("\n");
        }

    // A / B: a wrong mode committed, the truth carried as a rival. Robot drives a slow arc.
    for (const auto& [room_idx, label, llr0, expect_switch] :
         std::vector<std::tuple<int, const char*, float, bool>>{{0, "A symmetric: twin committed, truth rival (llr 0)", 0.f, false},
                                                                {2, "B notched: mirror committed, truth rival (llr -5)", -5.f, true}})
    {
        const auto& room = all_rooms[static_cast<std::size_t>(room_idx)];
        std::mt19937 rng(777);
        int switches = 0, first_switch = -1, final_ok = 0, runs = 20, epoch_mismatch = 0;
        for (int run = 0; run < runs; ++run)
        {
            rc::RoomConcept rcn;
            rcn.set_polygon_room(room.poly);
            Eigen::Vector3f truth = random_pose(room.poly, rng, 1.2f);
            Eigen::Vector3f wrong;
            if (room_idx == 0) wrong = room.symmetries.front()(truth);
            else
            {
                Eigen::Vector2f lo = room.poly.front(), hi = room.poly.front();
                for (const auto& v : room.poly) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
                wrong = {lo.x() + hi.x() - truth.x(), truth.y(), wrap(kPi - truth.z())};
                if (not rc::reloc::inside_polygon(room.poly, wrong.head<2>())) { --run; continue; }
            }
            rcn.set_robot_pose(wrong.x(), wrong.y(), wrong.z(), false);
            Acc::rivals(rcn).clear();
            Acc::rivals(rcn).push_back({truth, llr0});
            Acc::set_ref(rcn, wrong);
            Eigen::Vector3f committed = wrong;
            const Eigen::Vector3f step(0.03f, 0.f, 0.02f);   // per frame, robot frame
            int local_switches = 0;
            const auto epoch0 = Acc::epoch(rcn);
            for (int f = 0; f < 150; ++f)
            {
                truth = compose(truth, step);
                if (not rc::reloc::inside_polygon(room.poly, truth.head<2>())
                    or rc::reloc::polygon_distance(room.poly, truth.head<2>()) < 0.4f)
                    break;
                committed = compose(committed, step);
                rcn.set_robot_pose(committed.x(), committed.y(), committed.z(), false);
                const auto before = rcn.get_current_state();
                Acc::step(rcn, to3(synth_scan(room.poly, truth, 360, 0.02f, 0.30f, rng)));
                const auto after = rcn.get_current_state();
                if (std::abs(after[2] - before[2]) + std::abs(after[3] - before[3]) > 1e-4f)
                {
                    ++local_switches;
                    if (first_switch < 0) first_switch = f;
                    committed = Eigen::Vector3f(after[2], after[3], after[4]);
                }
            }
            switches += local_switches;
            if (Acc::epoch(rcn) - epoch0 != static_cast<std::uint64_t>(local_switches)) ++epoch_mismatch;
            if (near_or_twin(room, committed, truth, 0.30f, 10.f)
                and (room_idx == 0 or near(committed, truth, 0.30f, 10.f))) ++final_ok;
        }
        std::printf("  %-52s switches=%d (%s), first at frame %d, final pose ok %d/%d, clamp releases != switches in %d runs (expect 0)\n", label, switches,
                    expect_switch ? "expect >=runs, no flip-back" : "expect 0", first_switch, final_ok, runs, epoch_mismatch);
    }

    // C: door OPENING with the robot inside — a large through-door sector, pose CORRECT.
    {
        const auto& room = all_rooms[1];
        std::mt19937 rng(4242);
        int searches = 0, moved = 0, frames = 0, released = 0;
        for (int run = 0; run < 10; ++run)
        {
            rc::RoomConcept rcn;
            rcn.set_polygon_room(room.poly);
            const Eigen::Vector3f truth = random_pose(room.poly, rng, 1.0f);
            rcn.set_robot_pose(truth.x(), truth.y(), truth.z(), false);
            Acc::reset(rcn);
            const auto epoch0 = Acc::epoch(rcn);
            for (int f = 0; f < 60; ++f, ++frames)
            {
                const auto scan = synth_scan(room.poly, truth, 360, 0.02f, 0.30f, rng, f < 10 ? 6.f : 100.f);
                const float s = median_d(room.poly, scan, truth);
                const float pl = Acc::belief(rcn, s, 0.f);
                if (pl >= 0.5f)
                {
                    ++searches;
                    const auto before = rcn.get_current_state();
                    Acc::relocalise(rcn, to3(scan), s, pl);
                    const auto after = rcn.get_current_state();
                    if (std::hypot(after[2] - before[2], after[3] - before[3]) > 0.3f
                        or std::abs(wrap(after[4] - before[4])) > 10.f * kPi / 180.f)
                        ++moved;
                }
            }
            released += static_cast<int>(Acc::epoch(rcn) - epoch0);
        }
        std::printf("  %-52s searches=%d over %d frames, pose MOVED %d times (expect 0), clamp released %d times (expect 0)\n",
                    "C door opens (200° through-door), robot inside", searches, frames, moved, released);
    }

    // F: end-to-end kidnap. Robot tracked correctly, then carried elsewhere; the committed pose is stale.
    //    The belief must fire, the search must MOVE the pose to the truth (or its true twin), and stay.
    for (int room_idx = 0; room_idx < 3; ++room_idx)
    {
        const auto& room = all_rooms[static_cast<std::size_t>(room_idx)];
        std::mt19937 rng(31337u + static_cast<unsigned>(room_idx));
        int ok = 0, fired = 0, frames_sum = 0, runs = 15, right_without_release = 0;
        for (int run = 0; run < runs; ++run)
        {
            rc::RoomConcept rcn;
            rcn.set_polygon_room(room.poly);
            const Eigen::Vector3f stale = random_pose(room.poly, rng, 0.8f);
            Eigen::Vector3f truth = random_pose(room.poly, rng, 0.8f);
            while ((truth.head<2>() - stale.head<2>()).norm() < 1.5f) truth = random_pose(room.poly, rng, 0.8f);
            rcn.set_robot_pose(stale.x(), stale.y(), stale.z(), false);
            Acc::reset(rcn);
            // WARM-UP: tracking correctly at `stale` first, so the tracking emission is the one THIS scan
            // model produces (a live robot learns it before any kidnap). Toggle off with RELOC_F_COLD.
            if (std::getenv("RELOC_F_COLD") == nullptr)
                for (int f = 0; f < 200; ++f)
                {
                    const auto scan = synth_scan(room.poly, stale, 360, 0.02f, 0.30f, rng);
                    Acc::belief(rcn, median_d(room.poly, scan, stale), 0.f);
                }
            int f_fire = -1;
            for (int f = 0; f < 30; ++f)
            {
                const auto scan = synth_scan(room.poly, truth, 360, 0.02f, 0.30f, rng);
                const auto st = rcn.get_current_state();
                const float s = median_d(room.poly, scan, Eigen::Vector3f(st[2], st[3], st[4]));
                const float pl = Acc::belief(rcn, s, 0.f);
                if (pl >= 0.5f)
                {
                    if (f_fire < 0) f_fire = f;
                    if (std::getenv("RELOC_F_TRACE") != nullptr)
                    {
                        const auto st0 = rcn.get_current_state();
                        rc::reloc::LikelihoodRaster ras;
                        const auto rp2 = Acc::rparams(rcn);
                        ras.build(room.poly, rp2);
                        const auto res = rc::reloc::search(ras, room.poly, scan, {Eigen::Vector3f(st0[2], st0[3], st0[4])}, rp2);
                        for (const auto& m : res.modes)
                            std::printf("        mode (%.2f,%.2f,%.0f°) w=%.2f med=%.3f explains=%d truth_near=%d\n",
                                        m.pose.x(), m.pose.y(), m.pose.z() * 180. / kPi, m.weight, m.median_abs,
                                        Acc::explains(rcn, m.median_abs) ? 1 : 0, near_or_twin(room, m.pose, truth, 0.3f, 10.f) ? 1 : 0);
                    }
                    Acc::relocalise(rcn, to3(scan), s, pl);
                    if (std::getenv("RELOC_F_TRACE") != nullptr)
                    {
                        const auto a = rcn.get_current_state();
                        std::printf("      F %s run%d f%d s=%.3f -> pose (%.2f,%.2f,%.0f°) confirmed=%d moved=%d rivals=%zu | stale (%.2f,%.2f,%.0f°) truth (%.2f,%.2f,%.0f°)\n",
                                    room.name.c_str(), run, f, s, a[2], a[3], a[4] * 180. / kPi,
                                    Acc::confirmed(rcn) ? 1 : 0, Acc::moved(rcn) ? 1 : 0, Acc::rivals(rcn).size(),
                                    stale.x(), stale.y(), stale.z() * 180. / kPi, truth.x(), truth.y(), truth.z() * 180. / kPi);
                    }
                }
            }
            const auto st = rcn.get_current_state();
            if (f_fire >= 0) { ++fired; frames_sum += f_fire + 1; }
            const bool right = near_or_twin(room, Eigen::Vector3f(st[2], st[3], st[4]), truth, 0.30f, 10.f);
            if (right) ++ok;
            if (right and Acc::epoch(rcn) == 0) ++right_without_release;
            else if (std::getenv("RELOC_DUMP") != nullptr)
            {
                const auto scan = synth_scan(room.poly, truth, 360, 0.02f, 0.30f, rng);
                std::printf("    F MISS %s run%d: final (%.2f,%.2f,%.0f°) truth (%.2f,%.2f,%.0f°) s@truth=%.3f "
                            "sd_ln_s=%.3f mu_s=%.3f\n", room.name.c_str(), run, st[2], st[3], st[4] * 180. / kPi,
                            truth.x(), truth.y(), truth.z() * 180. / kPi, median_d(room.poly, scan, truth),
                            std::sqrt(Acc::track_var(rcn)), std::exp(Acc::track_mu(rcn)));
            }
        }
        std::printf("  F kidnap end-to-end %-32s fired %d/%d (after %.1f frames), final pose right %d/%d, right but clamp never released %d (expect 0)\n",
                    room.name.c_str(), fired, runs, fired > 0 ? static_cast<double>(frames_sum) / fired : std::nan(""), ok, runs, right_without_release);
    }

    // G: planner mode disambiguation. Committed pose = the mirror of the truth in the notched room, truth carried
    //    as a rival with llr 0. The notch is the only asymmetry, so J must be largest where the notch is seen
    //    DIFFERENTLY by the two hypotheses; in the symmetric rectangle J must be ~0 everywhere.
    {
        auto run_g = [&](const Room& room, const Eigen::Vector3f& committed, const Eigen::Vector3f& rival, const char* label)
        {
            rc::EpistemicPlanner pl;
            Eigen::Vector2f lo = room.poly.front(), hi = room.poly.front();
            for (const auto& v : room.poly) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
            pl.set_room_bounds(lo, hi);
            pl.set_room_polygon(room.poly);
            Eigen::Affine2f T = Eigen::Affine2f::Identity();
            T.translation() = committed.head<2>();
            T.linear() = Eigen::Rotation2Df(committed.z()).toRotationMatrix();
            pl.set_robot_state(T, Eigen::Matrix3f::Identity() * 0.01f);
            pl.set_pose_hypotheses({{rival, 0.f}});
            float jmax = 0.f, jmin = 1e9f; Eigen::Vector2f argmax = committed.head<2>();
            for (float x = lo.x() + 0.5f; x < hi.x(); x += 0.5f)
                for (float y = lo.y() + 0.5f; y < hi.y(); y += 0.5f)
                {
                    if (not rc::reloc::inside_polygon(room.poly, {x, y})) continue;
                    const float j = pl.mode_disambiguation_nats({x, y});
                    if (j > jmax) { jmax = j; argmax = {x, y}; }
                    jmin = std::min(jmin, j);
                }
            std::printf("  G %-50s J here=%.2f  min=%.2f  max=%.2f nats at (%.1f,%.1f)\n", label,
                        pl.mode_disambiguation_nats(committed.head<2>()), jmin, jmax, argmax.x(), argmax.y());
        };
        const auto& rect = all_rooms[0];
        const Eigen::Vector3f t_rect(-1.5f, 0.5f, 0.3f);
        run_g(rect, rect.symmetries.front()(t_rect), t_rect, "rect: twin vs truth (expect ~0 everywhere)");
        const auto& notch = all_rooms[2];
        const Eigen::Vector3f t_n(2.0f, 2.0f, 0.2f);
        run_g(notch, Eigen::Vector3f(8.f - t_n.x(), t_n.y(), wrap(kPi - t_n.z())), t_n,
              "notched: mirror vs truth (expect max near a notch view)");
    }

    // H: END TO END THROUGH update(). The earlier cases drive the relocaliser's pieces directly and never run
    //    update(), so they could not see that a commit written only to the model tensors is re-predicted from
    //    last_update_result on the next frame and silently undone — which is what happened live (09-17).
    //    Given layout: warm up tracking, kidnap, keep feeding scans; the pose must reach the truth AND stay.
    //    Then the live failure's configuration: SEARCHING with the map ready must never relocalise.
    {
        const auto& room = all_rooms[2];
        std::mt19937 rng(2024);
        int right = 0, stayed = 0, runs = 8, gate_violations = 0;
        std::uint64_t epochs_max = 0;
        std::int64_t ts = 1000;
        for (int run = 0; run < runs; ++run)
        {
            rc::RoomConcept rcn;
            rcn.set_polygon_room(room.poly);
            const Eigen::Vector3f start = random_pose(room.poly, rng, 1.0f);
            Eigen::Vector3f truth = random_pose(room.poly, rng, 1.0f);
            while ((truth.head<2>() - start.head<2>()).norm() < 2.0f) truth = random_pose(room.poly, rng, 1.0f);
            rcn.set_robot_pose(start.x(), start.y(), start.z(), false);
            auto frame = [&](const Eigen::Vector3f& at)
            {
                const auto pts = to3(synth_scan(room.poly, at, 360, 0.02f, 0.30f, rng));
                ts += 50;
                // A zero-velocity odometry stream covering the frame, like the parked robot's. WITHOUT odometry the
                // measured prior is invalid, the prediction falls back to the model tensors, and a commit that
                // forgets last_update_result looks correct — this harness first passed with that bug in it.
                std::vector<rc::OdometryReading> odom;
                for (std::int64_t t = ts - 100; t <= ts; t += 10)
                {
                    rc::OdometryReading o;
                    o.source_ts_ms = t; o.recv_ts_ms = t;
                    o.var_adv = 1e-4f; o.var_side = 1e-4f;
                    odom.push_back(o);
                }
                const auto res = rcn.update(rc::LidarData{pts, ts}, {}, odom);
                Acc::step8(rcn, res, pts);
            };
            for (int f = 0; f < 60; ++f) frame(start);           // tracking at `start`
            const auto e_warm = Acc::epoch(rcn);
            for (int f = 0; f < 40; ++f)                         // kidnapped: the scans now come from `truth`
            {
                const auto e_before = Acc::epoch(rcn);
                frame(truth);
                if (std::getenv("RELOC_H_TRACE") != nullptr and Acc::epoch(rcn) != e_before)
                {
                    const auto s2 = rcn.get_current_state();
                    std::printf("      H run%d f%d epoch %llu: now (%.2f,%.2f,%.0f°) truth (%.2f,%.2f,%.0f°) moved=%d rivals=%zu\n",
                                run, f, static_cast<unsigned long long>(Acc::epoch(rcn)), s2[2], s2[3], s2[4] * 180. / kPi,
                                truth.x(), truth.y(), truth.z() * 180. / kPi, Acc::moved(rcn) ? 1 : 0, Acc::rivals(rcn).size());
                }
            }
            if (std::getenv("RELOC_H_TRACE") != nullptr and e_warm != 0)
                std::printf("      H run%d: %llu relocalisations during WARM-UP\n", run, static_cast<unsigned long long>(e_warm));
            auto st = rcn.get_current_state();
            const bool ok_now = near_or_twin(room, Eigen::Vector3f(st[2], st[3], st[4]), truth, 0.30f, 10.f);
            if (ok_now) ++right;
            for (int f = 0; f < 40; ++f) frame(truth);           // and it must STAY there
            st = rcn.get_current_state();
            if (ok_now and near_or_twin(room, Eigen::Vector3f(st[2], st[3], st[4]), truth, 0.30f, 10.f)) ++stayed;
            epochs_max = std::max(epochs_max, Acc::epoch(rcn));
            if (std::getenv("RELOC_DUMP") != nullptr and not ok_now)
                std::printf("    H MISS run%d: est (%.2f,%.2f,%.0f°) truth (%.2f,%.2f,%.0f°) start (%.2f,%.2f,%.0f°) epochs=%llu\n",
                            run, st[2], st[3], st[4] * 180. / kPi, truth.x(), truth.y(), truth.z() * 180. / kPi,
                            start.x(), start.y(), start.z() * 180. / kPi, static_cast<unsigned long long>(Acc::epoch(rcn)));

            // SEARCHING with the map ready (estimate mode, freeze off): feed the kidnapped scans; nothing may fire.
            Acc::force_searching_with_map_ready(rcn);
            const auto e0 = Acc::epoch(rcn);
            const auto before = rcn.get_current_state();
            const Eigen::Vector3f elsewhere = random_pose(room.poly, rng, 1.0f);
            for (int f = 0; f < 10; ++f)
            {
                const auto pts = to3(synth_scan(room.poly, elsewhere, 360, 0.02f, 0.30f, rng));
                rc::RoomConcept::UpdateResult fake;
                fake.ok = true; fake.sdf_mse = 1.0f; fake.pred_sdf_median = 1.0f;   // a terrible fit
                Acc::step8(rcn, fake, pts);
            }
            const auto after = rcn.get_current_state();
            if (not Acc::searching(rcn) or Acc::epoch(rcn) != e0 or (after - before).norm() > 1e-6f) ++gate_violations;
        }
        std::printf("  H kidnap THROUGH update() (notched)                 pose right %d/%d, stayed right %d/%d; "
                    "most relocalisations in one run %llu (expect 1); searching+map-ready relocalised in %d runs (expect 0)\n",
                    right, runs, stayed, runs, static_cast<unsigned long long>(epochs_max), gate_violations);
    }

    // D / E: healthy residual stream with turn spikes, then a kidnap.
    {
        std::mt19937 rng(99);
        std::normal_distribution<float> nz(0.f, 0.30f);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        int false_fires = 0, kidnap_delay_sum = 0, kidnaps = 0, missed = 0;
        for (int run = 0; run < 20; ++run)
        {
            rc::RoomConcept rcn;
            Acc::reset(rcn);
            for (int f = 0; f < 2000; ++f)
            {
                float s = 0.034f * std::exp(nz(rng));
                if (u01(rng) < 0.05f) s *= 3.f;          // a hard turn spikes the prediction error
                if (Acc::belief(rcn, s, 0.f) >= 0.5f) { ++false_fires; Acc::reset(rcn); }   // a search that confirms
            }
            int delay = -1;
            for (int f = 0; f < 40; ++f)
            {
                const float s = 0.40f * std::exp(nz(rng));   // lost: wrong pose
                if (Acc::belief(rcn, s, 0.f) >= 0.5f) { delay = f + 1; break; }
            }
            if (delay < 0) ++missed; else { kidnap_delay_sum += delay; ++kidnaps; }
        }
        std::printf("  %-52s false fires=%d in 40000 healthy frames (5%% turn spikes x3)\n", "D healthy stream", false_fires);
        std::printf("  %-52s fired in %.1f frames on average, missed %d/20\n", "E kidnap (s -> 0.40 m)",
                    kidnaps > 0 ? static_cast<double>(kidnap_delay_sum) / kidnaps : std::nan(""), missed);
    }
    return 0;
}
