// C1 bench (door-crossing contract, 2026-11-02): POLICY-ONLY (the look-ahead policy drives the whole passage to the through
// goal) versus TODAY'S CHAIN (door_approach_law to the vantage, then the AIF cross servo door_cross_aif on the anchor, the
// drive-in being the same servo), both in the policy harness's plant (sim2d: 2 LiDARs at Shadow's mounts, latency 60 ms +-
// 20 ms, deadband, 5 % slip, the TRUE bare hull as the contact oracle). Same worlds, same start states, same anchor error.
//
//   ANCHOR ERROR  one per-run realisation: the door's centre (x, y) ~ N(0, noise) and the normal's heading ~ N(0, noise_th),
//                 used for BOTH arms (the policy's Doorway goal is built from it; the servo's RelPose is built from it).
//   SUCCESS       the TRUE centre is >= 1.2 m past the door plane (the cross's drive-in target is 1.3 m), no contact.
//   (d) SHUT      success = no contact and the robot parked in the last 5 s.
//   CHAIN CAVEATS the chain arm has NO live-return guards / lidar caps / escape / MAKING ROOM (they are LiDAR-side pieces of the
//                 old chain this harness does not rebuild): it is the chain's LAWS on the same plant, not the whole chain.
//
// Build+run (from controller/):
//   g++ -std=c++23 -O2 -Wall -Wextra -pthread -I/usr/include/eigen3 -I../common -Isrc $(pkg-config --cflags Qt6Core) -fPIC
//     tests/bench_c1_door.cpp src/local_policy/local_policy.cpp src/local_policy/contact.cpp src/local_policy/cost_to_go.cpp
//     src/local_policy/navigator.cpp ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp
//     -o /tmp/bench_c1 -lfastdds -lfastcdr $(pkg-config --libs Qt6Core) && /tmp/bench_c1
//   Env: SEEDS=<n> (default 20), ONLY=<scenario>, TRACE=<scenario>:<arm>:<seed>
#include "sim2d/sim2d.h"
#include "../src/local_policy/local_policy.h"
#include "../src/local_policy/navigator.h"
#include "../src/local_policy/unseen_contact.h"
#include "../src/local_policy/doorway_corridor.h"
#include "../src/door_approach_law.h"
#include "../src/door_cross_aif.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <functional>
#include <string>
#include <vector>

using namespace rc::policy;
using sim2d::V2;
using sim2d::World;
constexpr float kPi = sim2d::kPi, kDeg = kPi / 180.f;
constexpr float kYc = -0.055f;          // the door plane (the wall's mid-thickness); room_1 at y < kYc, the far room at y > 0
constexpr float kThroughS = 1.3f;       // the cross target (standoff)
constexpr float kSuccessS = 1.2f;       // success: the TRUE centre this far past the plane
constexpr float kVantage = 1.72f;       // door_1's vantage distance (warm33)
enum class Arm { Policy, Chain, ChainUngated };
static const char* arm_name(Arm a) { return a == Arm::Policy ? "POLICY" : a == Arm::Chain ? "CHAIN" : "CHAIN-UNGATED"; }

static BaseLimits limits()
{
    BaseLimits b; b.v_max = 0.5f; b.w_max = 1.5f; b.a_lin = 1.f; b.d_lin = 1.f; b.a_rot = 2.f; b.d_rot = 2.f;
    b.v_min_eff = 0.03f; b.w_min_eff = 0.08f; return b;
}
static const rc::RobotFootprint& body() { static const rc::RobotFootprint f = rc::RobotFootprint::shadow(); return f; }
static const sim2d::Mount kHelios{V2(0.f, -0.155f)}, kBpearl{V2(0.f, 0.140f)};
constexpr int kBufferCycles = 2;
constexpr float kSlipSigma = 0.05f, kLatencyJitterS = 0.02f, kDeadbandSpread = 0.2f;

struct LeafSpec
{
    bool on = false;
    float hinge_x = 0.f;      // anchor frame x of the hinge (a jamb)
    int side = -1;            // -1: swings into the robot's room (y < plane), +1: into the far room
    float phi0 = 90.f * kDeg, omega = 0.f, phi_lo = 0.f, phi_hi = 3.14f, len = 0.9f;
    float phi_at(float t) const { return std::clamp(phi0 + omega * t, phi_lo, phi_hi); }
    // direction from the hinge (world), the leaf closed lies along the wall toward the other jamb
    V2 dir(float phi) const { return V2((hinge_x > 0.f ? -1.f : 1.f) * std::cos(phi), float(side) * std::sin(phi)); }
    V2 hinge() const { return V2(hinge_x, kYc); }
    V2 tip(float phi) const { return hinge() + len * dir(phi); }
};

struct Scn
{
    std::string name;
    bool room_b = false, shut = false;
    Pose2 start;
    std::vector<V2> route;            // chain: the route from the start to the lead point (the vantage is appended)
    float push_t = -1.f; V2 push = V2::Zero();
    LeafSpec leaf;
    float tmax = 60.f;
    float gate_t = 0.f;               // the protocol's cross gate: no through goal / servo before this time (leaf opening done)
};

struct Cfg { float gap, noise_xy, noise_th; float hz = 20.f; };

static World make_world(const Scn& s, float gap)
{
    World w;
    const float h = 0.5f * gap;
    w.boxes = {{-8.f, -0.11f, -h, 0.f}, {h, -0.11f, 8.f, 0.f},                       // the wall with the opening
               {-3.1f, 0.f, -3.f, 3.5f}, {3.f, 0.f, 3.1f, 3.5f}, {-3.1f, 3.5f, 3.1f, 3.6f}};   // the far room
    if (s.room_b) w.boxes.insert(w.boxes.end(), {{-4.7f, -5.9f, -4.6f, -0.11f}, {0.95f, -5.9f, 1.05f, -0.11f},     // small room, corner 0.95 m east of the door
                                                  {-4.7f, -5.9f, 1.05f, -5.8f},
                                                  {-2.4f, -4.2f, -1.2f, -2.4f}});                                     // a table across the straight line
    else w.boxes.insert(w.boxes.end(), {{-3.1f, -4.6f, -3.f, -0.11f}, {3.f, -4.6f, 3.1f, -0.11f}, {-3.1f, -4.7f, 3.1f, -4.6f}});
    if (s.shut) w.boxes.push_back({-h, -0.11f, h, 0.f, false});                      // the shut leaf: not in the map
    return w;
}

struct Result
{
    bool success = false, contact = false, unreachable = false, stuck = false;
    int contacts = 0, held = 0;
    float path = 0.f;
    float t = 0.f, min_clear = 1e9f, max_yawrate_near = 0.f;
    float parked_move = 1e9f;
    std::vector<float> step_ms;
    std::string why;
};

static bool trace_on(const std::string& n, Arm a, unsigned seed)
{
    const char* t = std::getenv("TRACE");
    return t != nullptr and std::string(t) == n + ":" + arm_name(a) + ":" + std::to_string(seed);
}

// ---- the chain's route helper (polyline, arc-length)
struct Poly
{
    std::vector<V2> p; std::vector<float> s;
    explicit Poly(std::vector<V2> pts) : p(std::move(pts)) { s.assign(p.size(), 0.f); for (size_t i = 1; i < p.size(); ++i) s[i] = s[i - 1] + (p[i] - p[i - 1]).norm(); }
    float length() const { return s.back(); }
    V2 at(float a) const
    {
        a = std::clamp(a, 0.f, length());
        for (size_t i = 1; i < p.size(); ++i)
            if (a <= s[i]) { const float u = (a - s[i - 1]) / std::max(1e-6f, s[i] - s[i - 1]); return p[i - 1] + u * (p[i] - p[i - 1]); }
        return p.back();
    }
    float progress(const V2& q) const   // arc of the nearest point
    {
        float best = 1e9f, arc = 0.f;
        for (size_t i = 1; i < p.size(); ++i)
        {
            const V2 ab = p[i] - p[i - 1];
            const float u = std::clamp((q - p[i - 1]).dot(ab) / std::max(1e-9f, ab.squaredNorm()), 0.f, 1.f);
            const float d = (p[i - 1] + u * ab - q).norm();
            if (d < best) { best = d; arc = s[i - 1] + u * std::sqrt(ab.squaredNorm()); }
        }
        return arc;
    }
};

class MapFree
{
public:
    CostToGo::Grid grid;
    std::vector<char> free_;
    MapFree(const World& w, const std::function<bool(const V2&)>& extra_block, float x0, float y0, float x1, float y1)
    {
        grid = {V2(x0, y0), 0.05f, int(std::lround((x1 - x0) / 0.05f)) + 1, int(std::lround((y1 - y0) / 0.05f)) + 1};
        free_.assign(size_t(grid.w * grid.h), 0);
        const float r_ins = body().inscribed_radius();
        for (int j = 0; j < grid.h; ++j)
            for (int i = 0; i < grid.w; ++i)
            {
                const V2 q = grid.origin + V2(i * grid.cell, j * grid.cell);
                free_[size_t(j * grid.w + i)] = w.clearance(q, true) >= r_ins and not (extra_block and extra_block(q));
            }
    }
    bool operator()(const V2& c) const
    {
        const int i = int(std::lround((c.x() - grid.origin.x()) / grid.cell)), j = int(std::lround((c.y() - grid.origin.y()) / grid.cell));
        return i >= 0 and j >= 0 and i < grid.w and j < grid.h and free_[size_t(j * grid.w + i)];
    }
};

static float seg_dist(const V2& p, const V2& a, const V2& b)
{
    const V2 ab = b - a;
    const float t = std::clamp((p - a).dot(ab) / std::max(ab.squaredNorm(), 1e-12f), 0.f, 1.f);
    return (a + t * ab - p).norm();
}

static Result run(const Scn& S, const Cfg& cfg, Arm arm, unsigned seed)
{
    Result res;
    std::mt19937 rng(seed * 7919u + 13u + (unsigned)(cfg.gap * 1000));
    const bool tr = trace_on(S.name, arm, seed);
    const auto& B = body();
    rc::RobotFootprint bare = B; bare.set_safety_margin(0.f);
    const World wstat = make_world(S, cfg.gap);
    sim2d::ContactTester contact(wstat, B);
    sim2d::Base base; base.lim = limits(); base.latency_s = 0.06f;
    base.rng = &rng; base.jitter_s = kLatencyJitterS; base.slip_sigma = kSlipSigma;
    base.deadband_scale = std::uniform_real_distribution<float>(1.f - kDeadbandSpread, 1.f + kDeadbandSpread)(rng);
    std::normal_distribution<float> slip(0.f, kSlipSigma), nd(0.f, 1.f);

    // ---- the anchor error of this run (shared by both arms)
    const float bx = cfg.noise_xy * nd(rng), by = cfg.noise_xy * nd(rng), bth = cfg.noise_th * nd(rng);
    const V2 c_meas(bx, kYc + by), n_meas(-std::sin(bth), std::cos(bth));
    const float gap = cfg.gap;

    // ---- the leaf at time t: segment in the world (scan) + its hull-contact oracle + door_concept's hypothesis
    const auto leaf_pts = [&](float t)
    {
        std::vector<V2> pts;
        if (not S.leaf.on) return pts;
        const float phi = S.leaf.phi_at(t);
        const V2 a = S.leaf.hinge(), b = S.leaf.tip(phi);
        const int n = int((b - a).norm() / 0.02f);
        for (int i = 0; i <= n; ++i) pts.push_back(a + (b - a) * (float(i) / float(std::max(1, n))));
        return pts;
    };
    const auto wscan_at = [&](float t)
    {
        World w = wstat;
        if (S.leaf.on) { const float phi = S.leaf.phi_at(t); w.segments.push_back({S.leaf.hinge(), S.leaf.tip(phi), false}); }
        return w;
    };
    const auto leaf_touches = [&](const Pose2& pose, float t)
    {
        for (const auto& q : leaf_pts(t)) if ((q - pose.p).squaredNorm() <= 0.6f * 0.6f and bare.contains_yaw(q, pose.p, pose.yaw)) return true;
        return false;
    };
    const auto leaf_clear = [&](const Pose2& pose, float t)
    {
        float m = 1e9f;
        for (const auto& q : leaf_pts(t)) if ((q - pose.p).squaredNorm() <= 1.0f) m = std::min(m, hull_distance(B, pose, q));
        return m;
    };
    const auto world_touches = [&](const Pose2& pose, float t) { return contact.touches(pose) or leaf_touches(pose, t); };

    // ---- the policy arm's navigator (Doorway goal on the measured opening, jambs, corridor, trap veto: as the controller / harness)
    Navigator nav;
    const float tol = Goal{}.tol_xy;
    const float ahead = kThroughS + 2.f * tol + 0.05f;
    std::optional<DoorwayRisk> jambs;
    Goal goal{.kind = Goal::Kind::Doorway, .p = c_meas, .yaw = {}};
    goal.normal = n_meas; goal.width = gap;
    const float x0 = -5.f, x1 = 3.5f, y0 = -6.f, y1 = 3.6f;
    // the leaf as door_concept would publish it: hard in the field over its swept sector (phi +- 2 sigma)
    const auto leaf_block = [&](float t_now)
    {
        return std::function<bool(const V2&)>([&S, t_now, r = B.inscribed_radius()](const V2& q)
        {
            if (not S.leaf.on) return false;
            const float phi = S.leaf.phi_at(t_now);
            // door_concept's sigma_phi: wide while the leaf moves (+-0.5 rad), the settled value (0.17) once it has stopped
            const float hw = S.leaf.phi_at(t_now + 0.5f) != phi ? 0.5f : 0.17f;
            for (float d = -hw; d <= hw + 1e-4f; d += hw / 5.f)
                if (seg_dist(q, S.leaf.hinge(), S.leaf.tip(std::clamp(phi + d, S.leaf.phi_lo, S.leaf.phi_hi))) < r) return true;
            return false;
        });
    };
    std::unique_ptr<MapFree> mf;
    const auto free_fn = [&]() -> std::function<bool(const V2&)>
    {
        return [&mf, &jambs](const V2& c)
        {
            if (c.y() >= kYc) return jambs.has_value() and in_doorway_risk_corridor(*jambs, c) and (*mf)(c);   // the far side is not in the map
            return (*mf)(c);
        };
    };
    bool policy_started = false;
    const auto start_policy = [&](float t_now)
    {
        policy_started = true;
        const float sg = 0.02f * 0.70710678f;
        jambs = DoorwayRisk{};
        jambs->anchor = doorway_jambs(RoomGap{goal.p, goal.normal.normalized(), goal.width}, sg, sg,
                                      std::hypot(0.01f, PolicyInput{}.body_geometry_sigma_m), 0.02f, bare, 1.0f, ahead);
        mf = std::make_unique<MapFree>(wstat, S.leaf.on ? leaf_block(t_now) : nullptr, x0, y0, x1, y1);
        std::vector<V2> pts;
        {
            const V2 n = goal.normal.normalized(), lat(-n.y(), n.x());
            const float half = std::max(0.f, 0.5f * goal.width - B.inscribed_radius()), past = kThroughS + 2.f * goal.tol_xy;
            for (float t = -half; t <= half + 1e-4f; t += 0.05f) pts.push_back(goal.p + past * n + t * lat);
            if (pts.empty()) pts.push_back(goal.p + past * n);
        }
        nav.set_goal(goal, mf->grid, free_fn(), [jb = *jambs](const V2& c) { return doorway_risk_density(jb, c); }, pts, B.circumscribed_radius());
        const RoomGap g{goal.p, goal.normal.normalized(), goal.width};
        const float r_ins = bare.inscribed_radius(), body_w = bare.support_radius(0.f, V2(1.f, 0.f)) + bare.support_radius(0.f, V2(-1.f, 0.f));
        nav.set_trap_veto([g, r_ins, body_w, ahead](const V2& c, float r) { return disc_meets_doorway_corridor(g, c, r, 1.0f, ahead, r_ins, body_w); });
    };

    // ---- the chain's pieces
    const V2 vantage = c_meas - kVantage * n_meas;
    std::vector<V2> rpts = S.route;
    if (rpts.empty()) rpts.push_back(S.start.p);
    rpts.push_back(vantage);
    const Poly route(rpts);
    const float sV = route.length();
    const float half_w = 0.5f * (B.support_radius(0.f, V2(1.f, 0.f)) + B.support_radius(0.f, V2(-1.f, 0.f)));
    const float front = B.support_radius(0.f, V2(0.f, 1.f));
    int phase = 0;   // 0 approach law, 1 servo

    Pose2 truth = S.start, odom = S.start;
    sim2d::PoseError err; err.on = false; err.pivot = truth.p;
    UnseenContact unseen;
    std::deque<std::vector<Return>> buffer;
    const float kTick = 1.f / cfg.hz;
    constexpr float kSub = 0.005f;
    const bool period_in_latency = std::getenv("PERIOD_IN") == nullptr or std::string(std::getenv("PERIOD_IN")) != "0";   // C9 (a)
    const float min_hz = std::getenv("MINHZ") ? float(std::atof(std::getenv("MINHZ"))) : 10.f;                              // C9 (b)
    const bool rate_hold = cfg.hz < min_hz;
    bool pushed = false;
    std::vector<V2> hist;
    float t = 0.f;
    if (world_touches(truth, 0.f)) { res.why = "start overlaps"; res.contact = true; res.contacts = 1; return res; }
    V2 park_ref = truth.p; bool park_set = false;

    for (; t < S.tmax; t += kTick)
    {
        if (not park_set and t >= S.tmax - 5.f) { park_ref = truth.p; park_set = true; }
        if (S.push_t >= 0.f and not pushed and t >= S.push_t) { truth.p += S.push; odom.p += S.push; pushed = true; err.pivot = truth.p; }
        const float s_true = truth.p.y() - kYc;
        if (not S.shut and s_true >= kSuccessS) { res.success = true; res.why = "through"; break; }
        const Pose2 bel = err.believed(truth);
        float v_cmd = 0.f, w_cmd = 0.f;
        const auto t_step = std::chrono::steady_clock::now();
        if (arm == Arm::Policy and (t < S.gate_t or rate_hold)) { v_cmd = w_cmd = 0.f; if (rate_hold) ++res.held; }
        else if (arm == Arm::Policy)
        {
            if (not policy_started) start_policy(t);
            // the field re-priced with the leaf's present sector, as the controller does through update_world (twice a second)
            if (S.leaf.on and S.leaf.omega != 0.f and std::fmod(t, 0.5f) < kTick)
            {
                mf = std::make_unique<MapFree>(wstat, leaf_block(t), x0, y0, x1, y1);
                nav.update_world(free_fn(), [jb = *jambs](const V2& c) { return doorway_risk_density(jb, c); });
            }
            const World wsc = wscan_at(t);
            std::vector<Return> sweep;
            const V2 bf(std::cos(bel.yaw), std::sin(bel.yaw)), br(std::sin(bel.yaw), -std::cos(bel.yaw));
            for (const auto& m : {kHelios, kBpearl})
                for (const V2& q : sim2d::scan_mount(wsc, truth, m, rng, 0.02f)) sweep.push_back({bel.p + q.x() * br + q.y() * bf, 0.02f, 1.f});
            buffer.push_back(std::move(sweep));
            while (int(buffer.size()) > kBufferCycles) buffer.pop_front();
            std::vector<Return> ret;
            for (const auto& sw : buffer) ret.insert(ret.end(), sw.begin(), sw.end());
            for (const auto& h : unseen.hypotheses(bel, odom)) ret.push_back(h);
            PolicyInput in;
            in.pose = bel; in.v_now = base.v; in.w_now = base.w; in.sigma_rel_xy = 0.01f; in.sigma_rel_yaw = 0.01f;
            in.camera_half_fov_rad = 0.96f; in.affordance_facing = true;
            in.returns_age_s = float(kBufferCycles - 1) * kTick;
            in.returns = ret; in.body = &B; in.base = base.lim; in.latency_s = base.latency_s + (period_in_latency ? kTick : 0.f); in.now_s = t;
            const auto t0 = std::chrono::steady_clock::now();
            PolicyParams pp;
            const Navigator::Step st = nav.step(in, pp);
            res.step_ms.push_back(float(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()));
            const Decision& d = st.d;
            v_cmd = d.v; w_cmd = d.w;
            if (tr) std::printf("   t=%6.2f true(%+.3f,%+.3f,%+6.1f) %-7s v=%+.2f w=%+.2f | %s\n", t, truth.p.x(), truth.p.y(), truth.yaw / kDeg, mode_name(d.mode), d.v, d.w, d.why.c_str());
            if (d.mode == Decision::Mode::Blind) { res.stuck = true; res.why = d.why; break; }
            if (d.mode == Decision::Mode::Stuck) { res.stuck = true; v_cmd = w_cmd = 0.f; }
            if (st.trap) { /* the field was rebuilt inside the navigator */ }
            if (st.unreachable or (st.hand_back)) { res.unreachable = true; res.why = "handed back (unreachable)"; break; }
        }
        else
        {
            const auto t0 = std::chrono::steady_clock::now();
            const float phi = truth.yaw - kPi / 2.f;
            const float cph = std::cos(phi), sph = std::sin(phi);
            if (phase == 0)
            {
                namespace da = rc::door_approach;
                da::State st;
                st.p = truth.p; st.theta = phi; st.vantage = vantage; st.door = c_meas;
                const float prog = route.progress(truth.p);
                st.aim = prog <= sV ? route.at(std::min(prog + 0.4f, sV)) : vantage;
                st.d_route = prog <= sV ? std::max(0.f, sV - prog) + (truth.p - route.at(prog)).norm() : (truth.p - vantage).norm();
                st.rho = 0.25f; st.yaw_tol = 0.12f; st.v_max = 0.5f; st.w_max = 1.0f; st.tau = 1.f;
                const auto c = da::law(st);
                v_cmd = c.adv; w_cmd = c.rot;
                if (c.done) phase = 1;
            }
            if (phase == 1 and t < S.gate_t) { v_cmd = 0.f; w_cmd = 0.f; }
            else if (phase == 1 and (arm == Arm::ChainUngated or not S.shut))
            {
                const V2 dd = c_meas - truth.p;
                rc::interaction::RelPose r;
                r.x = Eigen::Vector3f(cph * dd.x() + sph * dd.y(), -sph * dd.x() + cph * dd.y(), std::atan2(-n_meas.x(), n_meas.y()) - phi);
                r.P = Eigen::Vector3f(cfg.noise_xy * cfg.noise_xy, cfg.noise_xy * cfg.noise_xy, cfg.noise_th * cfg.noise_th).asDiagonal();
                std::vector<rc::interaction::AnchorLeaf> leaves;
                if (S.leaf.on)
                {
                    rc::interaction::AnchorLeaf l;
                    const float ph = S.leaf.phi_at(t);
                    l.hinge = Eigen::Vector2f(S.leaf.hinge_x, 0.f);
                    const V2 tp = S.leaf.tip(ph);
                    l.tip = Eigen::Vector2f(tp.x(), tp.y() - kYc);
                    l.phi = ph; l.sigma_phi = 0.17f; l.p = 1.f;
                    leaves.push_back(l);
                }
                const auto cmd = rc::door_cross_aif(r, gap, leaves, rc::DoorCrossAifBody{half_w, front}, kThroughS, 0.5f, 1.0f, kTick).cmd;
                v_cmd = cmd.adv; w_cmd = cmd.rot;
            }
            else if (phase == 1) { v_cmd = 0.f; w_cmd = 0.f; }
            if (rate_hold) { v_cmd = w_cmd = 0.f; ++res.held; }   // the protocol's NO MOUTH hold (door shut: P(open) = 0, cross never offered)
            res.step_ms.push_back(float(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()));
            if (tr) std::printf("   t=%6.2f true(%+.3f,%+.3f,%+6.1f) ph=%d v=%+.2f w=%+.2f\n", t, truth.p.x(), truth.p.y(), truth.yaw / kDeg, phase, v_cmd, w_cmd);
        }
        (void)t_step;

        // ---- the plant
        base.send(t, v_cmd, w_cmd);
        const Pose2 before = truth;
        const float v_meas = base.v, w_meas = base.w;
        const float sv = 1.f + slip(rng), sw = 1.f + slip(rng);
        bool touched = false;
        std::vector<UnseenContact::CmdSeg> in_force;
        const Pose2 odom_before = odom;
        for (float s = 0.f; s < kTick - 1e-6f; s += kSub)
        {
            base.step_velocity(t + s, kSub);
            if (not in_force.empty() and in_force.back().v == base.cmd_v and in_force.back().w == base.cmd_w) in_force.back().dt += kSub;
            else in_force.push_back({kSub, base.cmd_v, base.cmd_w});
            const Pose2 nxt = sim2d::integrate(truth, base.v * sv, base.w * sw, kSub);
            if (world_touches(nxt, t + s)) { touched = true; base.v = base.w = 0.f; continue; }
            odom = sim2d::integrate(odom, base.v, base.w, kSub);
            truth = nxt;
        }
        if (touched) { ++res.contacts; res.contact = true; }
        const V2 fwd(std::cos(odom_before.yaw), std::sin(odom_before.yaw));
        unseen.update(std::span<const UnseenContact::CmdSeg>(in_force), v_meas, w_meas, (odom.p - odom_before.p).dot(fwd),
                      sim2d::wrap(odom.yaw - odom_before.yaw), odom, base.lim, B, 0.005f);
        res.path += (truth.p - before.p).norm();
        const float clear = std::min(contact.clearance(truth), leaf_clear(truth, t));
        res.min_clear = std::min(res.min_clear, clear);
        if (clear < 0.3f) res.max_yawrate_near = std::max(res.max_yawrate_near, std::abs(sim2d::wrap(truth.yaw - before.yaw)) / kTick);
        if (res.contact and S.shut) break;   // a contact with the shut door is the failure; no need to keep pushing
    }
    res.t = t;
    res.parked_move = park_set ? (truth.p - park_ref).norm() : 0.f;
    if (S.shut)
    {
        const bool parked = res.unreachable or res.stuck or res.parked_move < 0.05f;
        res.success = not res.contact and parked;
        if (res.why.empty()) res.why = parked ? "parked" : "still moving at the end";
    }
    else if (res.contact) res.success = false;   // success requires a clean passage
    if (not res.success and res.why.empty()) res.why = "time limit";
    return res;
}

static void report(const Scn& S, const Cfg& cfg, Arm arm, int nseeds)
{
    std::vector<std::future<Result>> fut;
    for (int s = 0; s < nseeds; ++s) fut.push_back(std::async(std::launch::async, [&, s] { return run(S, cfg, arm, unsigned(s)); }));
    int ok = 0, cont = 0, unr = 0, stk = 0, ncont = 0, held = 0; float sp = 0.f; int nsp = 0;
    std::vector<float> ts, ms; float minc = 1e9f, yaw = 0.f;
    std::string why;
    for (auto& f : fut)
    {
        const auto r = f.get();
        held += r.held > 0; if (r.t > 0.f) { sp += r.path / r.t; ++nsp; }
        ok += r.success; cont += r.contact; ncont += r.contacts; unr += r.unreachable; stk += r.stuck;
        if (r.success) ts.push_back(r.t);
        else if (why.empty()) why = r.why;
        minc = std::min(minc, r.min_clear); yaw = std::max(yaw, r.max_yawrate_near);
        ms.insert(ms.end(), r.step_ms.begin(), r.step_ms.end());
    }
    std::ranges::sort(ts); std::ranges::sort(ms);
    std::printf("%-24s %2.0fHz gap %.2f noise %.0fcm %-13s ok %2d/%d | v_mean %.2f held %d | contact-runs %2d (%d ticks) | handed-back %2d | t_med %5.1f s | min clr %+6.3f | max|dyaw| near walls %4.2f rad/s | step med %.3f p99 %.3f max %.2f ms%s%s\n",
                S.name.c_str(), cfg.hz, cfg.gap, cfg.noise_xy * 100, arm_name(arm), ok, nseeds, nsp ? sp / nsp : 0.f, held, cont, ncont, unr, ts.empty() ? NAN : ts[ts.size() / 2], minc, yaw,
                ms.empty() ? 0.f : ms[ms.size() / 2], ms.empty() ? 0.f : ms[ms.size() * 99 / 100], ms.empty() ? 0.f : ms.back(),
                ok < nseeds ? "  [fail: " : "", ok < nseeds ? (why + "]").c_str() : "");
    std::fflush(stdout);
}

int main()
{
    int nseeds = 20;
    if (const char* s = std::getenv("SEEDS")) nseeds = std::max(1, std::atoi(s));
    const char* only = std::getenv("ONLY");
    const float yawN = kPi / 2.f;   // facing the door normal (+y)
    std::vector<Scn> sc;
    {   // (a) 2 m out, 80 deg off the door normal (warm33)
        Scn s; s.name = "a_80deg_2m"; s.start = {V2(0.3f, kYc - 2.0f), yawN - 80.f * kDeg}; s.route = {s.start.p}; sc.push_back(s);
    }
    {   // (b) 7 m curved approach, small room, corner 0.95 m east of the door, table across the straight line
        Scn s; s.name = "b_curved_7m"; s.room_b = true; s.start = {V2(-4.0f, -5.2f), 0.f};
        s.route = {V2(-4.0f, -5.2f), V2(-0.5f, -4.5f), V2(0.f, -3.0f)}; s.tmax = 90.f; sc.push_back(s);
    }
    {   // (c) leaf beside after the plane (warm24 leg 2): centre 0.9 m past the plane, -59 deg, leaf swung into the far room
        Scn s; s.name = "c_leaf_beside"; s.start = {V2(0.f, kYc + 0.9f), (-59.f * kDeg) + yawN};
        s.leaf = {.on = true, .hinge_x = 0.45f, .side = +1, .phi0 = 110.f * kDeg, .omega = 0.f, .phi_lo = 0.f, .phi_hi = 3.14f, .len = 0.9f};
        s.route = {s.start.p}; sc.push_back(s);
    }
    {   // (d) shut door at the vantage: must not enter, must stop
        Scn s; s.name = "d_shut"; s.shut = true; s.start = {V2(0.f, kYc - kVantage), yawN}; s.route = {s.start.p}; s.tmax = 240.f; sc.push_back(s);
    }
    {   // (e) pushed 0.5 m sideways at the vantage (t = 1 s)
        Scn s; s.name = "e_push_0.5"; s.start = {V2(0.f, kYc - kVantage), yawN}; s.route = {s.start.p}; s.push_t = 1.0f; s.push = V2(0.5f, 0.f); sc.push_back(s);
    }
    {   // (f) the leaf (hung from the +x jamb, swinging into the robot's room) opening away 70 -> 120 deg while the robot approaches
        Scn s; s.name = "f_leaf_away"; s.start = {V2(0.1f, kYc - 2.0f), yawN};
        s.leaf = {.on = true, .hinge_x = 0.45f, .side = -1, .phi0 = 70.f * kDeg, .omega = 0.3f, .phi_lo = 70.f * kDeg, .phi_hi = 120.f * kDeg, .len = 0.9f};
        s.route = {s.start.p}; sc.push_back(s);
    }
    {   // (g) THE CONTRACT (invariant 3): the door is shut, its leaf swings TOWARD the robot, and is only opened once the robot stands
        //     clear of its arc (the vantage, 1.72 m, beyond leaf 0.9 + sigma); the cross goal is given when the opening is done (gate)
        Scn s; s.name = "g_contract_open"; s.start = {V2(0.f, kYc - kVantage), yawN};
        s.leaf = {.on = true, .hinge_x = 0.45f, .side = -1, .phi0 = 0.f, .omega = 0.3f, .phi_lo = 0.f, .phi_hi = 100.f * kDeg, .len = 0.9f};
        s.gate_t = 100.f * kDeg / 0.3f + 0.5f; s.route = {s.start.p}; s.tmax = 60.f; sc.push_back(s);
    }
    if (std::getenv("RATES"))   // C9: scenarios (a) (b) (e) at 20 16 12 10 8 6 Hz, 0.88 m gap, 2 cm noise, policy arm (and chain for reference)
    {
        for (const auto& s : sc)
            if (s.name[0] == 'a' or s.name[0] == 'b' or s.name[0] == 'e')
                for (const float hz : {20.f, 16.f, 12.f, 10.f, 8.f, 6.f})
                    for (const Arm a : {Arm::Policy, Arm::Chain}) report(s, Cfg{0.88f, 0.02f, 0.0175f, hz}, a, nseeds);
        return 0;
    }
    for (const auto& s : sc)
    {
        if (only and std::strcmp(only, s.name.c_str()) != 0) continue;
        for (const float g : {0.88f, 1.0f})
            for (const auto [nx, nt] : {std::pair{0.02f, 0.0175f}, std::pair{0.05f, 0.052f}})
            {
                Scn sg = s;
                if (sg.leaf.on) sg.leaf.hinge_x = 0.5f * g;
                for (const Arm a : {Arm::Policy, Arm::Chain}) report(sg, Cfg{g, nx, nt}, a, nseeds);
                if (s.shut) report(sg, Cfg{g, nx, nt}, Arm::ChainUngated, nseeds);
            }
    }
    return 0;
}
