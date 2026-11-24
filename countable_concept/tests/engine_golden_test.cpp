// engine_golden_test — the persistent update's trajectory before and after the shared-engine extraction (Task 3 of the
// shared-engine plan, spec docs/superpowers/specs/2026-11-23-shared-engine-design.md). Recorded with the code as it stood
// BEFORE the extraction (HEAD b36db5d9); the default mode re-runs the same sequence and compares.
//
// The sequence drives one persistent CountableBelief (the golden nightstand, persistent_belief_test's fixture) through:
//   - a birth with set_birth_position_cov;
//   - 8 parked updates with camera + LiDAR, 5 camera-only (after an inflate_for_age), 5 LiDAR-only, one predict-only frame;
//   - 11 updates while the robot moves round the box (ρ < 1, fresh ego-motion variances, a chain with a ψ part), mixing
//     both channels, camera-only and LiDAR-only;
//   - one mode rotation (w ↔ h);
//   - 10 parked updates at the last standpoint on the box moved 6 cm along x (the GN has somewhere to go), GnTrace = 2 on
//     the first of them and GnTrace = 1 on the rest.
// After each step it records θ, the diagonal of P (the 33 entries the pre-extraction layout had), the GN step count, the
// returned energy and the GnTrace rows. Compare: every double within 1e-9 relative, every count equal; any P diagonal entry
// beyond the recorded 33 (the shape latents, zero variance until Task 5) must be exactly 0.
//
// Run: ./engine_golden_test            (compare against golden/engine_golden.bin)
//      ./engine_golden_test --record   (rewrite the golden file; only with the pre-extraction code)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "countable_belief.h"
#include "test_scene.h"
#include "beam_fixture.h"

using namespace rc;
namespace ts = rc::test;

static constexpr float ty = -0.255f, tH = 0.62f, tw = 0.51f;
static float tx = 0.30f;                   // the box centre x; the last stretch moves the box
static constexpr int kDiagRecorded = 33;   // N + 27: θ, e, b_cam, b_L, u_cam, u_L (the pre-extraction joint)

static CountableBeliefParams params()
{
    CountableBeliefParams p;
    p.prior_footprint_m = 0.51f; p.prior_footprint_std = 0.06f; p.prior_height_m = 0.62f; p.prior_height_std = 0.10f;
    p.plaus_height_min = 0.40f; p.plaus_height_soft = 0.08f;
    p.wall_flush_prior = 0.90f;
    p.room_axis_std    = 3.0f * 0.017453293f;
    p.volatility_infer = true;
    p.persistent       = true;
    p.beam             = ts::test_beam_config();
    return p;
}

// A deterministic noise source (no library distribution: the sequence must not depend on the standard library).
struct Lcg
{
    std::uint64_t s = 0x9E3779B97F4A7C15ull;
    float uniform(float a)   // in [−a, a]
    {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const double u = double(s >> 11) * (1.0 / 9007199254740992.0);
        return float((2.0 * u - 1.0) * a);
    }
};

static ts::SceneBox truth_box() { return {Eigen::Vector2f(tx, ty), 0.0f, tw, tw, tH}; }
static const ts::SceneWall kWall{Eigen::Vector2f(-4.0f, 0.0f), Eigen::Vector2f(4.0f, 0.0f), 2.5f};

static Eigen::Matrix3d full_sigma_pose()
{
    Eigen::Matrix3d S;
    S << 9.0e-4, 2.0e-4, 1.0e-4,
         2.0e-4, 1.2e-3, -5.0e-5,
         1.0e-4, -5.0e-5, 4.0e-4;
    return S;
}

// What one step of the sequence stages: where the robot stands, which channels measure, the latents' ρ and fresh shares.
struct Step
{
    Eigen::Vector2f robot;
    bool camera = true, lidar = true;
    float rho_cam = 1.0f, rho_L = 1.0f, rho_chain = 1.0f;
    float fresh = 0.0f;                       // fresh (ego-motion) variance on every DOF of both channels
    Eigen::Vector2f chain_dr = Eigen::Vector2f::Zero();
    bool moving_pose = false;                 // the full Σ_pose (ψ part, cross terms) instead of 1 cm on (x, y)
};

// Camera points on the faces visible from the robot (front always; the left or right side when the robot stands beyond
// it), with ±4 mm deterministic noise, R = σ_base²; the room wall staged (flush anchor) and the room axis.
static void add_camera(CountableFrame& f, const Eigen::Vector2f& robot, Lcg& rng)
{
    const float x0 = tx - 0.5f * tw, x1 = tx + 0.5f * tw, y0 = ty - 0.5f * tw;
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 10; ++j)
        f.points.emplace_back(x0 + tw * (i + 0.5f) / 12.0f + rng.uniform(0.004f), y0 + rng.uniform(0.004f),
                              tH * (j + 0.5f) / 10.0f + rng.uniform(0.004f));
    if (robot.x() < x0 or robot.x() > x1)
    {
        const float xs = robot.x() < x0 ? x0 : x1;
        for (int i = 0; i < 8; ++i) for (int j = 0; j < 10; ++j)
            f.points.emplace_back(xs + rng.uniform(0.004f), y0 + tw * (i + 0.5f) / 8.0f + rng.uniform(0.004f),
                                  tH * (j + 0.5f) / 10.0f + rng.uniform(0.004f));
    }
    f.R.assign(f.points.size(), params().sigma_base_m * params().sigma_base_m);
    f.z_top_obs = tH;
}

static CountableFrame make_frame(const Step& st, Lcg& rng)
{
    const CountableBeliefParams p = params();
    CountableFrame f;
    if (st.camera) add_camera(f, st.robot, rng);
    else f.z_top_obs = -1.0f;
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
    f.rho_cam.fill(st.rho_cam); f.rho_L.fill(st.rho_L); f.rho_chain = st.rho_chain;
    f.fresh_cam.fill(st.fresh); f.fresh_L.fill(0.5f * st.fresh);
    f.sigma_pose = st.moving_pose ? full_sigma_pose() : Eigen::Matrix3d(Eigen::Vector3d(1e-4, 1e-4, 0.0).asDiagonal());
    f.chain_dr = st.chain_dr;
    f.robot_xy = st.robot;
    return f;
}

// One record: what the step left in the belief.
struct Record
{
    std::int32_t tag = 0;                     // 0 update, 1 mode rotation, 2 predict / inflate
    std::int32_t steps = 0, n_trace = 0, n_fd = 0;
    double fe = 0.0;
    std::vector<double> theta, pdiag, trace, fd;
};

static Record snapshot(const CountableBelief& b, int tag, double fe)
{
    Record r;
    r.tag = tag;
    r.steps = b.last_gn_iters();
    r.fe = fe;
    for (int i = 0; i < 6; ++i) r.theta.push_back(double(b.state().vec()(i)));
    const auto& P = b.joint_covariance();
    for (int i = 0; i < P.rows(); ++i) r.pdiag.push_back(P(i, i));
    r.n_trace = std::int32_t(b.dbg_gn_steps().size());
    for (const auto& s : b.dbg_gn_steps())
        for (const double v : {double(s.step), s.lambda, s.rho, s.pred, s.J_before, s.J_after, double(s.accepted)})
            r.trace.push_back(v);
    r.n_fd = std::int32_t(b.dbg_gn_fd().size());
    for (const auto& d : b.dbg_gn_fd())
        for (const double v : {double(d.at), double(d.channel), double(d.k), d.grad_an, d.grad_marg, d.grad_fd3, d.grad_fd4,
                               d.curv_an, d.curv_fd})
            r.fd.push_back(v);
    return r;
}

static std::vector<Record> run_sequence()
{
    std::vector<Record> out;
    Lcg rng;
    CountableBelief b(CountableBeliefState{tx + 0.05f, ty - 0.04f, tH + 0.03f, tw + 0.02f, tw - 0.01f, 0.03f}, params());
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
    Eigen::Matrix2f excess;
    excess << 4.0e-3f, 1.5e-3f,
              1.5e-3f, 2.5e-3f;
    b.set_birth_position_cov(excess);                                   // fix B (rev 3.3.14)

    const Eigen::Vector2f park(tx, ty - 0.5f * tw - 1.5f);
    const auto update = [&](const Step& st)
    {
        const CountableFrame f = make_frame(st, rng);
        const float fe = b.update(f);
        out.push_back(snapshot(b, 0, double(fe)));
    };
    // Parked, both channels.
    for (int k = 0; k < 8; ++k) update(Step{park});
    // Parked, camera only, after an aged gap.
    b.inflate_for_age(2.0f, 0.5f);
    out.push_back(snapshot(b, 2, 0.0));
    for (int k = 0; k < 5; ++k) { Step s{park}; s.lidar = false; update(s); }
    // Parked, LiDAR only.
    for (int k = 0; k < 5; ++k) { Step s{park}; s.camera = false; update(s); }
    // A predict-only frame (no points, not LiDAR-only) and a bare predict.
    { Step s{park}; s.camera = false; s.lidar = false; update(s); }
    b.predict();
    out.push_back(snapshot(b, 2, 0.0));
    // Moving round the box: new views (ρ < 1), ego-motion fresh shares, a chain with a ψ part.
    Eigen::Vector2f robot = park;
    for (int k = 0; k < 11; ++k)
    {
        const Eigen::Vector2f next(tx - 0.9f + 0.15f * float(k), ty - 0.5f * tw - 1.5f + 0.03f * float(k % 4));
        Step s{next};
        s.camera = (k % 3) != 2;
        s.lidar  = (k % 3) != 1;
        s.rho_cam = 0.7f; s.rho_L = 0.8f; s.rho_chain = 0.95f;
        s.fresh = 4.0e-5f;
        s.chain_dr = next - robot;
        s.moving_pose = true;
        update(s);
        robot = next;
    }
    // The mode rotation (w ↔ h), the countable box's symmetry.
    b.apply_mode_rotation_for_test(1);
    out.push_back(snapshot(b, 1, 0.0));
    tx += 0.06f;
    // Parked at the last standpoint, the box moved 6 cm along x; GnTrace = 2 on the first update, 1 (the FD check on unconverged updates) on the rest.
    for (int k = 0; k < 10; ++k)
    {
        CountableBeliefParams p = b.params();
        p.gn_trace = (k == 0) ? 2 : 1;
        b.set_params(p);
        Step s{robot};
        s.moving_pose = true;
        update(s);
    }
    return out;
}

static void write_records(const std::string& path, const std::vector<Record>& rs)
{
    std::ofstream f(path, std::ios::binary);
    const std::int32_t n = std::int32_t(rs.size());
    f.write(reinterpret_cast<const char*>(&n), sizeof n);
    for (const auto& r : rs)
    {
        const std::int32_t hdr[6] = {r.tag, r.steps, r.n_trace, r.n_fd, std::int32_t(r.theta.size()), std::int32_t(r.pdiag.size())};
        f.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
        f.write(reinterpret_cast<const char*>(&r.fe), sizeof r.fe);
        for (const auto* v : {&r.theta, &r.pdiag, &r.trace, &r.fd})
            f.write(reinterpret_cast<const char*>(v->data()), std::streamsize(v->size() * sizeof(double)));
    }
}

static bool read_records(const std::string& path, std::vector<Record>& rs)
{
    std::ifstream f(path, std::ios::binary);
    if (not f) return false;
    std::int32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof n);
    for (int i = 0; i < n and f; ++i)
    {
        Record r;
        std::int32_t hdr[6];
        f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
        r.tag = hdr[0]; r.steps = hdr[1]; r.n_trace = hdr[2]; r.n_fd = hdr[3];
        f.read(reinterpret_cast<char*>(&r.fe), sizeof r.fe);
        r.theta.resize(hdr[4]); r.pdiag.resize(hdr[5]); r.trace.resize(std::size_t(r.n_trace) * 7); r.fd.resize(std::size_t(r.n_fd) * 9);
        for (auto* v : {&r.theta, &r.pdiag, &r.trace, &r.fd})
            f.read(reinterpret_cast<char*>(v->data()), std::streamsize(v->size() * sizeof(double)));
        rs.push_back(std::move(r));
    }
    return bool(f) and int(rs.size()) == n;
}

// Relative equality at 1e-9 (both NaN counts as equal: GnTrace's ρ is NaN on the undamped proposal).
static bool close(double a, double b, double& worst)
{
    if (std::isnan(a) or std::isnan(b)) return std::isnan(a) and std::isnan(b);
    if (a == b) return true;
    const double rel = std::abs(a - b) / std::max(std::abs(a), std::abs(b));
    worst = std::max(worst, rel);
    return rel <= 1e-9;
}

int main(int argc, char** argv)
{
    const bool record = argc > 1 and std::strcmp(argv[1], "--record") == 0;
    const std::vector<Record> got = run_sequence();
    int updates = 0;
    for (const auto& r : got) updates += r.tag == 0;
    if (record)
    {
        write_records(GOLDEN_FILE, got);
        std::printf("engine_golden_test: recorded %zu records (%d updates) to %s\n", got.size(), updates, GOLDEN_FILE);
        return 0;
    }
    std::vector<Record> want;
    if (not read_records(GOLDEN_FILE, want))
    {
        std::printf("FAIL: cannot read %s\n", GOLDEN_FILE);
        return 1;
    }
    int bad = 0;
    double worst = 0.0;
    if (want.size() != got.size()) { std::printf("FAIL: %zu records, golden %zu\n", got.size(), want.size()); return 1; }
    for (std::size_t i = 0; i < got.size(); ++i)
    {
        const Record& g = got[i];
        const Record& w = want[i];
        bool ok = g.tag == w.tag and g.steps == w.steps and g.n_trace == w.n_trace and g.n_fd == w.n_fd
                  and g.theta.size() == w.theta.size() and int(g.pdiag.size()) >= kDiagRecorded
                  and int(w.pdiag.size()) == kDiagRecorded;
        if (not ok)
        {
            std::printf("FAIL record %zu: tag %d/%d steps %d/%d trace %d/%d fd %d/%d diag %zu/%zu\n", i, g.tag, w.tag, g.steps,
                        w.steps, g.n_trace, w.n_trace, g.n_fd, w.n_fd, g.pdiag.size(), w.pdiag.size());
            ++bad;
            continue;
        }
        ok = close(g.fe, w.fe, worst);
        for (std::size_t k = 0; k < g.theta.size(); ++k) ok = close(g.theta[k], w.theta[k], worst) and ok;
        for (int k = 0; k < kDiagRecorded; ++k) ok = close(g.pdiag[k], w.pdiag[k], worst) and ok;
        for (std::size_t k = kDiagRecorded; k < g.pdiag.size(); ++k) ok = g.pdiag[k] == 0.0 and ok;   // the shape latents
        for (std::size_t k = 0; k < g.trace.size(); ++k) ok = close(g.trace[k], w.trace[k], worst) and ok;
        for (std::size_t k = 0; k < g.fd.size(); ++k) ok = close(g.fd[k], w.fd[k], worst) and ok;
        if (not ok)
        {
            std::printf("FAIL record %zu (tag %d): values differ beyond 1e-9 relative\n", i, g.tag);
            ++bad;
        }
    }
    int traced = 0;
    for (const auto& r : got) traced += r.n_trace > 0;
    std::printf("engine_golden_test: %zu records (%d updates, %d with GnTrace rows), worst relative deviation %.3e, %d failing\n",
                got.size(), updates, traced, worst, bad);
    return bad == 0 ? 0 : 1;
}
