// replay_test — the replay tap (shared-engine plan revision 2, Task R1; src/countable_replay.h).
//
// 1. Bit-identity: a short persistent sequence on the golden nightstand (engine_golden_test's fixture: birth with the
//    birth position covariance, the room interior, parked camera + LiDAR, an aged gap, camera-only, LiDAR-only (gated), a
//    predict, moving updates with a chain, a front cue, a step-bound reject restored and widened by a predict, and two
//    more updates) is driven with the tap on, exactly as the fitter calls it. The dump is then replayed by rc::replay::play
//    (the replay tool's own loop) and every update's θ and the whole joint P must equal the live run's BIT FOR BIT, as must
//    the live results the dump recorded.
// 2. Completeness: every member of each recorded struct has a fields() line (Boost.PFR's member count).
// 3. By name: a record with a field the reader lacks and without fields it has decodes, keeping the defaults and
//    reporting both.
// 4. Size: the bytes of one update event for a bed-sized frame (35k camera points + a staged Helios sweep).
// 5. A failing target (/dev/full): the writer goes inert, logs the failure once, and the belief is bit-identical to one
//    updated with no writer at all.
// 6. The reader: a truncated last event and a bad version are reported, not read.
// Run: ./replay_test [DUMP.bin]   (with a path, the sequence's dump is written there and kept, for ./replay DUMP.bin)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <boost/pfr.hpp>
#include <Eigen/Dense>
#include "countable_belief.h"
#include "countable_replay.h"
#include "test_scene.h"
#include "beam_fixture.h"

using namespace rc;
namespace ts = rc::test;

static int g_fail = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static constexpr float ty = -0.255f, tH = 0.62f, tw = 0.51f, tx = 0.30f;

static CountableBeliefParams params()
{
    CountableBeliefParams p;   // engine_golden_test's nightstand
    p.prior_footprint_m = 0.51f; p.prior_footprint_std = 0.06f; p.prior_height_m = 0.62f; p.prior_height_std = 0.10f;
    p.plaus_height_min = 0.40f; p.plaus_height_soft = 0.08f;
    p.wall_flush_prior = 0.90f;
    p.room_axis_std    = 3.0f * 0.017453293f;
    p.volatility_infer = true;
    p.persistent       = true;
    p.beam             = ts::test_beam_config();
    p.layer_spread_m   = {0.025f, 0.025f, 0.0f, 0.04f, 0.05f, 0.0f};
    return p;
}

struct Lcg
{
    std::uint64_t s = 0x9E3779B97F4A7C15ull;
    float uniform(float a)
    {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return float((2.0 * (double(s >> 11) * (1.0 / 9007199254740992.0)) - 1.0) * a);
    }
};

static ts::SceneBox truth_box() { return {Eigen::Vector2f(tx, ty), 0.0f, tw, tw, tH}; }
static const ts::SceneWall kWall{Eigen::Vector2f(-4.0f, 0.0f), Eigen::Vector2f(4.0f, 0.0f), 2.5f};

struct Step
{
    Eigen::Vector2f robot;
    bool camera = true, lidar = true;
    float rho = 1.0f, fresh = 0.0f;
    Eigen::Vector2f chain_dr = Eigen::Vector2f::Zero();
    int grid = 12;   // camera points per face edge (the size check uses a bed-sized cloud)
};

static CountableFrame make_frame(const Step& st, Lcg& rng)
{
    const CountableBeliefParams p = params();
    CountableFrame f;
    if (st.camera)
    {
        const float x0 = tx - 0.5f * tw, y0 = ty - 0.5f * tw;
        for (int i = 0; i < st.grid; ++i) for (int j = 0; j < st.grid; ++j)
            f.points.emplace_back(x0 + tw * (i + 0.5f) / float(st.grid) + rng.uniform(0.004f), y0 + rng.uniform(0.004f),
                                  tH * (j + 0.5f) / float(st.grid) + rng.uniform(0.004f));
        f.R.assign(f.points.size(), p.sigma_base_m * p.sigma_base_m);
        f.z_top_obs = tH;
    }
    if (st.lidar)
    {
        const std::vector<ts::SceneBox> obj{truth_box()};
        const std::vector<ts::SceneWall> walls{kWall};
        f.beam = ts::helios_beam(Eigen::Vector3f(st.robot.x(), st.robot.y(), 1.1075f), obj, walls, truth_box(), 0.10f,
                                 ts::hand_labels(obj), 1.0, p.model_sigma_m);
        f.lidar_only = not st.camera;
    }
    f.wall.ok = true; f.wall.p = Eigen::Vector2f(0, 0); f.wall.n = Eigen::Vector2f(0, -1);
    f.wall.sigma_m = 0.02f; f.wall.length_m = 4.0f;
    f.room_axis_ok = true; f.room_axis = 0.0f;
    const float pp = p.common_mode_pos_std * p.common_mode_pos_std, ss = p.common_mode_size_std * p.common_mode_size_std,
                yy = p.common_mode_yaw_std * p.common_mode_yaw_std;
    f.sig2_cam = {pp, pp, ss, ss, ss, yy};
    f.sig2_L   = {0.5f * pp, 0.5f * pp, ss, ss, ss, 0.5f * yy};
    f.rho_cam.fill(st.rho); f.rho_L.fill(st.rho); f.rho_chain = st.rho;
    f.fresh_cam.fill(st.fresh); f.fresh_L.fill(0.5f * st.fresh);
    f.sigma_pose = Eigen::Matrix3d(Eigen::Vector3d(1e-4, 1e-4, 4e-4).asDiagonal());
    f.chain_dr = st.chain_dr;
    f.robot_xy = st.robot;
    f.camera_xy = st.robot;
    f.foreign_boxes.push_back({Eigen::Vector2f(1.4f, -0.3f), 0.7f, 0.5f, 0.4f, 0.1f,
                               (Eigen::Matrix<float, 6, 1>() << 0.02f, 0.02f, 0.03f, 0.03f, 0.03f, 0.05f).finished(), 0.0f});
    f.mask_error_rate = 0.085;
    return f;
}

// What one update left in the belief, for the bitwise comparison.
struct Snap
{
    CountableBeliefState theta;
    CountableBelief::MatJd P;
    Eigen::Matrix<float, 6, 1> omega;
    int steps = 0;
    float fe = 0.0f;
};
static Snap snap(const CountableBelief& b, float fe)
{ return {b.state(), b.joint_covariance(), b.log_volatility(), b.last_gn_iters(), fe}; }
static bool same_bits(const Snap& a, const Snap& b)
{
    return std::memcmp(&a.theta, &b.theta, sizeof a.theta) == 0 and a.P.size() == b.P.size()
           and std::memcmp(a.P.data(), b.P.data(), sizeof(double) * std::size_t(a.P.size())) == 0
           and std::memcmp(a.omega.data(), b.omega.data(), sizeof(float) * 6) == 0 and a.steps == b.steps
           and std::memcmp(&a.fe, &b.fe, sizeof a.fe) == 0;
}

// The live run with the tap on, calling the writer exactly where the fitter does (countable_fitter.cpp run_inference).
static std::vector<Snap> live_run(const std::string& path)
{
    std::vector<Snap> out;
    Lcg rng;
    auto w = replay::Writer::open(path);
    CHECK(w != nullptr);
    const CountableBeliefState s0{tx + 0.05f, ty - 0.04f, tH + 0.03f, tw + 0.02f, tw - 0.01f, 0.03f};
    CountableBelief b(s0, params());
    w->seed("nightstand_test", s0, params());
    Eigen::Matrix2f excess;
    excess << 4.0e-3f, 1.5e-3f, 1.5e-3f, 2.5e-3f;
    b.set_birth_position_cov(excess);
    w->birth_pos_cov(excess);
    std::int64_t cycle = 0;
    const auto update = [&](const Step& st)
    {
        const CountableFrame f = make_frame(st, rng);
        const Eigen::Vector2f interior(0.0f, -3.0f);
        w->room_interior(interior);                       // stage_room, every cycle (written once: it never changes)
        b.set_room_interior(interior);
        w->update(++cycle, f.lidar_only, f);
        const float fe = b.update(f);
        w->result(b, fe);
        out.push_back(snap(b, fe));
    };
    const Eigen::Vector2f park(tx, ty - 0.5f * tw - 1.5f);
    for (int k = 0; k < 4; ++k) update(Step{park});
    w->inflate(2.0f, 0.5f);
    b.inflate_for_age(2.0f, 0.5f);
    for (int k = 0; k < 2; ++k) { Step s{park}; s.lidar = false; update(s); }
    for (int k = 0; k < 2; ++k) { Step s{park}; s.camera = false; update(s); }
    w->predict();
    b.predict();
    Eigen::Vector2f robot = park;
    for (int k = 0; k < 4; ++k)
    {
        const Eigen::Vector2f next(tx - 0.6f + 0.3f * float(k), park.y() + 0.05f * float(k % 2));
        Step s{next};
        s.rho = 0.7f; s.fresh = 4.0e-5f; s.chain_dr = next - robot;
        update(s);
        robot = next;
    }
    FrontCue cue;
    cue.bearing_rad = -1.5f; cue.confidence = 0.8f; cue.view_bearing_rad = -1.4f;
    w->resolve_front(cue, 0.9f);
    b.resolve_front(cue, 0.9f);
    {   // a step-bound reject: update, restore the copy, widen by a predict (update_guarded)
        const CountableBelief pre = b;
        Step s{robot};
        update(s);
        w->restore();
        b = pre;
        w->predict();
        b.predict();
    }
    for (int k = 0; k < 2; ++k) update(Step{robot});
    return out;
}

// ── by-name decoding: an older WallRef with a field this code lacks, and without three it has ───────────────────────────
namespace rc::replay::testing
{
struct WallRefV0 { bool ok = true; Eigen::Vector2f p = Eigen::Vector2f(1.5f, -2.0f); float extra = 7.0f; };
template <class V> void fields(V& v, WallRefV0& x) { v("ok", x.ok); v("p", x.p); v("extra", x.extra); }
}

int main(int argc, char** argv)
{
    // 2. completeness
    CHECK(replay::field_count<CountableBeliefState>()   == int(boost::pfr::tuple_size_v<CountableBeliefState>));
    CHECK(replay::field_count<CountableBeliefParams>()  == int(boost::pfr::tuple_size_v<CountableBeliefParams>));
    CHECK(replay::field_count<CountableFrame>()         == int(boost::pfr::tuple_size_v<CountableFrame>));
    CHECK(replay::field_count<WallRef>()                == int(boost::pfr::tuple_size_v<WallRef>));
    CHECK(replay::field_count<FrontCue>()               == int(boost::pfr::tuple_size_v<FrontCue>));
    CHECK(replay::field_count<rc::ai::LidarRays>()      == int(boost::pfr::tuple_size_v<rc::ai::LidarRays>));
    CHECK(replay::field_count<rc::ai::beam::BeamRays>() == int(boost::pfr::tuple_size_v<rc::ai::beam::BeamRays>));
    CHECK(replay::field_count<rc::ai::beam::ForeignBox>() == int(boost::pfr::tuple_size_v<rc::ai::beam::ForeignBox>));
    CHECK(replay::field_count<rc::ai::beam::BeamConfig>() == int(boost::pfr::tuple_size_v<rc::ai::beam::BeamConfig>));
    CHECK(replay::field_count<rc::ai::beam::StudentT>() == int(boost::pfr::tuple_size_v<rc::ai::beam::StudentT>));
    CHECK(replay::field_count<rc::ai::beam::GaussLegendre01>() == int(boost::pfr::tuple_size_v<rc::ai::beam::GaussLegendre01>));
    CHECK(replay::field_count<rc::manifest::ShapePart>() == int(boost::pfr::tuple_size_v<rc::manifest::ShapePart>));
    CHECK(replay::field_count<rc::manifest::ShapeProportion>() == int(boost::pfr::tuple_size_v<rc::manifest::ShapeProportion>));
    CHECK(replay::field_count<rc::ai::view::Viewpoint>()  == int(boost::pfr::tuple_size_v<rc::ai::view::Viewpoint>));
    CHECK(replay::field_count<rc::ai::view::ViewAnchor>() == int(boost::pfr::tuple_size_v<rc::ai::view::ViewAnchor>));
    CHECK(replay::field_count<rc::ai::view::ViewState>()  == int(boost::pfr::tuple_size_v<rc::ai::view::ViewState>));

    // 1. bit-identity
    const bool keep = argc > 1;
    const std::string path = keep ? std::string(argv[1])
                                  : (std::filesystem::temp_directory_path() / "countable_replay_test.bin").string();
    const std::vector<Snap> live = live_run(path);
    std::vector<Snap> rep;
    int results = 0, results_same = 0;
    replay::Reader r;
    CHECK(r.open(path));
    CountableBelief b;
    const bool ok = replay::play(r, b,
        [&](const replay::UpdateEv&, const CountableBelief& after, float fe) { rep.push_back(snap(after, fe)); },
        [&](const replay::ResultEv& lv, const CountableBelief& after)
        {
            ++results;
            const auto& P = after.joint_covariance();
            bool same = std::memcmp(&lv.theta, &after.state(), sizeof lv.theta) == 0 and lv.p_diag.size() == std::size_t(P.rows())
                        and lv.gn_it == after.last_gn_iters();
            for (int i = 0; same and i < P.rows(); ++i)
                same = std::memcmp(&lv.p_diag[std::size_t(i)], &P(i, i), sizeof(double)) == 0;
            results_same += same;
        });
    CHECK(ok);
    CHECK(r.error().empty());
    CHECK(r.diag().missing.empty() and r.diag().unknown.empty() and r.diag().bad.empty());
    CHECK(rep.size() == live.size());
    int same = 0;
    for (std::size_t i = 0; i < std::min(rep.size(), live.size()); ++i) same += same_bits(rep[i], live[i]);
    CHECK(same == int(live.size()));
    CHECK(results == int(live.size()) and results_same == results);
    int measured = 0;
    for (const auto& s : live) measured += s.steps > 0;
    CHECK(measured >= int(live.size()) - 1);   // the sequence really runs the GN (every update measures)
    std::printf("replay_test: %zu updates (%d with GN steps), %d/%zu bit-identical (theta, joint P %ldx%ld, omega, steps, fe), "
                "%d/%d live results matched\n", live.size(), measured, same, live.size(), long(live.front().P.rows()),
                long(live.front().P.cols()), results_same, results);
    if (not keep) std::filesystem::remove(path);

    // 3. by name
    {
        const std::string bytes = replay::encode(replay::testing::WallRefV0{});
        WallRef wr;
        wr.sigma_m = 0.33f;
        replay::Diag d;
        CHECK(replay::decode(bytes, wr, d));
        CHECK(wr.ok and wr.p == Eigen::Vector2f(1.5f, -2.0f) and wr.sigma_m == 0.33f);
        CHECK(d.missing == (std::set<std::string>{"n", "sigma_m", "length_m", "room", "edge"}));   // room, edge: F1.4
        CHECK(d.unknown == (std::set<std::string>{"extra"}));
        CHECK(d.bad.empty());
    }

    // 4. size of one bed update (35k camera points, a staged Helios sweep)
    {
        const std::string spath = (std::filesystem::temp_directory_path() / "countable_replay_size.bin").string();
        auto w = replay::Writer::open(spath);
        Lcg rng;
        Step s{Eigen::Vector2f(tx, ty - 0.5f * tw - 1.5f)};
        s.grid = 187;   // 187² = 34969 points
        const CountableFrame f = make_frame(s, rng);
        w->update(1, false, f);
        const double per_pt = 3 * 4 + 4;                 // a point (3 float32) + its R (float32)
        const double per_ray = 3 * 8 + 8 + 8;            // a beam endpoint (3 float64) + m + p_f
        std::printf("replay_test: one bed update = %llu bytes (%zu points x %.0f B + %zu rays x %.0f B + %llu B other)\n",
                    static_cast<unsigned long long>(w->last_event_bytes()), f.points.size(), per_pt, f.beam.endpoints.size(),
                    per_ray, static_cast<unsigned long long>(w->last_event_bytes() - std::uint64_t(per_pt * double(f.points.size()))
                                                             - std::uint64_t(per_ray * double(f.beam.endpoints.size()))));
        CHECK(w->last_event_bytes() > std::uint64_t(per_pt * double(f.points.size())));
        std::filesystem::remove(spath);
    }

    // 5. a full disk
    if (std::filesystem::exists("/dev/full"))
    {
        auto w = replay::Writer::open("/dev/full");
        CHECK(w != nullptr and w->dead());                   // the header's flush already fails
        CHECK(w and w->failure_logs() == 1 and w->failure().find("/dev/full") != std::string::npos);
        Lcg r1, r2;
        const CountableBeliefState s0{tx + 0.05f, ty - 0.04f, tH + 0.03f, tw + 0.02f, tw - 0.01f, 0.03f};
        CountableBelief with(s0, params()), without(s0, params());
        w->seed("full", s0, params());
        const Eigen::Vector2f park(tx, ty - 0.5f * tw - 1.5f);
        for (int k = 0; k < 3; ++k)
        {
            const CountableFrame f1 = make_frame(Step{park}, r1), f2 = make_frame(Step{park}, r2);
            w->update(k, false, f1);
            const float fe1 = with.update(f1);
            w->result(with, fe1);
            w->predict();
            const float fe2 = without.update(f2);
            CHECK(same_bits(snap(with, fe1), snap(without, fe2)));
        }
        CHECK(w->failure_logs() == 1 and w->bytes_written() == 0);   // logged once; nothing counted as written
    }
    else std::printf("replay_test: /dev/full absent, write-failure case skipped\n");

    // 6. reader: truncation and version
    {
        const std::string src = (std::filesystem::temp_directory_path() / "countable_replay_trunc.bin").string();
        live_run(src);
        const auto full = std::filesystem::file_size(src);
        std::filesystem::resize_file(src, full - 100);       // cut inside the last event
        replay::Reader rt;
        CHECK(rt.open(src));
        replay::Event e;
        int n = 0;
        while (rt.next(e)) ++n;
        CHECK(n > 0 and rt.error().find("truncated") != std::string::npos);
        {   // version 0 is refused
            std::fstream f(src, std::ios::binary | std::ios::in | std::ios::out);
            f.seekp(8);
            const std::uint32_t zero = 0;
            f.write(reinterpret_cast<const char*>(&zero), sizeof zero);
        }
        replay::Reader rv;
        CHECK(not rv.open(src) and rv.error().find("version") != std::string::npos);
        std::filesystem::remove(src);
    }

    if (g_fail == 0) std::printf("replay_test: PASS\n");
    else std::printf("replay_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
