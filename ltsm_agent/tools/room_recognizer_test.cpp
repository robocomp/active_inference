// room_recognizer: evidence in, posterior out, staged commit requests (ltsm).   run from the component root:
//   ./bin/room_recognizer_test
// Scene = the three-room loop of room_chain_test, in a world frame W (each room has its OWN arbitrarily posed frame):
//   r1_room_1 [-2,2]x[-2,2]   r2_room_2 [2,6]x[-2,2]   r3_room_3 [-2,6]x[2,6]
//   passage_0 r1<->r2 at (2,0)   passage_1 r1<->r3 at (0,2)    DOOR_2 at (4,2): memory holds r2's face ONLY (no passage).
// LIVE graph: robot "Shadow" with rooms hung from it by RT (room_1 first, then room_2 current, then a PROTO provisional
// room whose frame is an ARBITRARY SE(2) pose in W) and, under it, the entry mirror of DOOR_2 (the same hole, seen from the
// new side, in either orientation). The fresh seam T(r2 <- provisional) is built by RoomRecognizer FROM THE TWO FACES --
// nothing here hands it the seam, so the convention (L <- prov, the turn by pi) is exercised.
#include "../src/room_recognizer.h"
#include "../src/recognition_cycle.h"
#include "../src/seam_fusion.h"
#include <QCoreApplication>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using namespace ltsm::seam;
static int failures = 0;
static void check(bool ok, const std::string &what) { if (!ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str()); }

static Vec3 inv(const Vec3 &p) { return inverse(p); }
static Vec3 mul(const Vec3 &a, const Vec3 &b) { return compose_cov({a, Mat3::Zero()}, {b, Mat3::Zero()}).pose; }

static void put_edge(DSR::DSRGraph &G, std::uint64_t from, std::uint64_t to, const Vec3 &p, std::optional<Mat3> cov = std::nullopt)
{
    auto n = G.get_node(from).value();
    G.get_rt_api()->insert_or_assign_edge_RT(n, to, std::vector<float>{(float)p.x(), (float)p.y(), 0.f},
                                             std::vector<float>{0.f, 0.f, (float)p.z()});
    if (cov)
    {
        Eigen::Matrix<double, 6, 6> C = Eigen::Matrix<double, 6, 6>::Zero();
        C(0, 0) = (*cov)(0, 0); C(1, 1) = (*cov)(1, 1); C(5, 5) = (*cov)(2, 2);
        G.get_rt_api()->insert_or_assign_edge_RT_covariance(from, to, DSR::RT_API::CovarianceKind::Pose, C);
    }
}

struct Room { std::string name; std::uint64_t rid; std::uint64_t id = 0; Vec3 in_world; std::vector<Eigen::Vector2d> world_poly; };

static std::shared_ptr<DSR::DSRGraph> live, mem;
static std::vector<Room> rooms;
static std::uint64_t robot_id = 0, live_r1 = 0, live_r2 = 0, live_D = 0;
static std::uint64_t face_r2_door = 0;
static const Vec3 hole_w(4.0, 2.0, 0.0);                 // DOOR_2 in W (r2's remembered face, y out of r2)
static const Vec3 prov_w(7.0, -3.0, 1.0);                // the provisional room's frame in W (arbitrary)
static Mat3 Codo;

static void build_memory()
{
    const auto root = mem->get_node("root").value().id();
    auto rect = [](double x0, double y0, double x1, double y1) {
        return std::vector<Eigen::Vector2d>{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}; };
    rooms = {{"r1_room_1", 1, 0, Vec3(10.0, 5.0, 0.3), rect(-2, -2, 2, 2)},
             {"r2_room_2", 2, 0, Vec3(-4.0, 7.0, -1.1), rect(2, -2, 6, 2)},
             {"r3_room_3", 3, 0, Vec3(3.0, -8.0, 2.0), rect(-2, 2, 6, 6)}};
    for (auto &r : rooms)
    {
        DSR::Node n; n.type("room"); n.name(r.name);
        mem->add_or_modify_attrib_local<room_id_att>(n, r.rid);
        std::vector<float> px, py;
        for (const auto &w : r.world_poly)
        {
            const Vec3 l = mul(inv(r.in_world), Vec3(w.x(), w.y(), 0.0));
            px.push_back((float)l.x()); py.push_back((float)l.y());
        }
        mem->add_or_modify_attrib_local<delimiting_polygon_x_att>(n, px);
        mem->add_or_modify_attrib_local<delimiting_polygon_y_att>(n, py);
        r.id = mem->insert_node(n).value();
        put_edge(*mem, root, r.id, Vec3::Zero());
    }
    auto add_face = [&](const Room &room, const std::string &name, const Vec3 &face_in_world) {
        DSR::Node f; f.type("object"); f.name(name);
        mem->add_or_modify_attrib_local<object_subtype_att>(f, std::string("door"));
        mem->add_or_modify_attrib_local<parent_att>(f, room.id);
        const auto id = mem->insert_node(f).value();
        put_edge(*mem, room.id, id, mul(inv(room.in_world), face_in_world));
        return id; };
    auto add_passage = [&](int i, const Room &a, const Room &b, const Vec3 &hole) {
        const auto fa = add_face(a, "f" + std::to_string(i) + "_" + a.name, hole);
        const auto fb = add_face(b, "f" + std::to_string(i) + "_" + b.name, hole);
        DSR::Node pw; pw.type("metaconcept"); pw.name("passage_" + std::to_string(i));
        mem->add_or_modify_attrib_local<object_subtype_att>(pw, std::string("passage"));
        const auto pid = mem->insert_node(pw).value();
        mem->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, fa));
        mem->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, fb));
    };
    add_passage(0, rooms[0], rooms[1], Vec3(2.0, 0.0, M_PI / 2));
    add_passage(1, rooms[0], rooms[2], Vec3(0.0, 2.0, 0.0));
    face_r2_door = add_face(rooms[1], "r2_door_3", hole_w);       // DOOR_2's remembered face: in NO passage
}

static std::uint64_t live_room(const std::string &name, std::uint64_t rid, std::optional<std::uint64_t> room_id)
{
    DSR::Node n; n.type("room"); n.name(name);
    if (room_id) live->add_or_modify_attrib_local<room_id_att>(n, *room_id);
    live->add_or_modify_attrib_local<parent_att>(n, robot_id);
    (void)rid;
    return live->insert_node(n).value();
}

static std::uint64_t live_door(const std::string &name, std::uint64_t parent)
{
    DSR::Node d; d.type("object"); d.name(name);
    live->add_or_modify_attrib_local<object_subtype_att>(d, std::string("door"));
    live->add_or_modify_attrib_local<parent_att>(d, parent);
    return live->insert_node(d).value();
}

static std::uint64_t live_node_for_test(const std::string &name, std::uint64_t rid) { return live_room(name, rid, rid); }

static void set_current(std::uint64_t room)
{
    for (const auto &e : live->get_edges_by_type("current")) live->delete_edge(e.from(), e.to(), "current");
    live->insert_or_assign_edge(DSR::Edge::create<current_edge_type>(robot_id, room));
}

static void build_live()
{
    DSR::Node r; r.type("robot"); r.name("Shadow");
    robot_id = live->insert_node(r).value();
    put_edge(*live, live->get_node("root").value().id(), robot_id, Vec3::Zero());
    live_r1 = live_room("room_1", 1, 1);
    live_r2 = live_room("room_2", 2, 2);
    put_edge(*live, robot_id, live_r1, Vec3::Zero(), Codo);      // the RT edge gives a node its level (children need it)
    put_edge(*live, robot_id, live_r2, Vec3::Zero(), Codo);
    // room_2's frame IS memory's r2 frame (seed copies it under the same RT chain)
    live_D = live_door("door_3", live_r2);
    put_edge(*live, live_r2, live_D, mul(inv(rooms[1].in_world), hole_w));
}

struct Truth { Vec3 robot_w; };

static void place_robot(const Vec3 &robot_w, std::uint64_t prov)
{
    put_edge(*live, robot_id, live_r1, inv(mul(inv(rooms[0].in_world), robot_w)), Codo);
    put_edge(*live, robot_id, live_r2, inv(mul(inv(rooms[1].in_world), robot_w)), Codo);
    if (prov) put_edge(*live, robot_id, prov, inv(mul(inv(prov_w), robot_w)), Codo);
}

// The robot's pose in P stated WITHOUT a covariance (promotion_evidence's reading: no covariance, no pose): the robot chain
// is unmeasurable, so the metric expert is unavailable. (Deleting DOOR_2's memory face no longer does that: the fresh seam
// is built from the two LIVE faces, so a first-ever door with no memory face still gives the metric.)
static void unmeasured_robot(std::uint64_t prov, const Vec3 &robot_w)
{
    live->delete_edge(robot_id, prov, "RT");
    put_edge(*live, robot_id, prov, inv(mul(inv(prov_w), robot_w)));
}

static void write_evidence(const std::vector<std::uint64_t> &ids, const std::vector<float> &llr, const std::vector<std::optional<Vec3>> &mu_in_k)
{
    std::optional<DSR::Node> n = live->get_node("place_evidence");
    const bool fresh = not n.has_value();
    if (fresh) { DSR::Node x; x.type("metaconcept"); x.name("place_evidence"); n = x; }
    Mat3 S = Mat3::Zero(); S.diagonal() << 0.2 * 0.2, 0.2 * 0.2, 0.1 * 0.1;
    std::vector<float> mix;
    std::vector<std::uint64_t> cnt;
    for (const auto &m : mu_in_k)
    {
        if (not m) { cnt.push_back(0); continue; }
        cnt.push_back(1);
        for (float f : {1.f, (float)m->x(), (float)m->y(), (float)m->z(), (float)S(0, 0), (float)S(0, 1), (float)S(0, 2), (float)S(1, 1), (float)S(1, 2), (float)S(2, 2)}) mix.push_back(f);
    }
    live->add_or_modify_attrib_local<object_subtype_att>(n.value(), std::string("place_evidence"));
    live->add_or_modify_attrib_local<recog_room_ids_att>(n.value(), std::vector<std::uint64_t>(ids));
    live->add_or_modify_attrib_local<recog_loglr_att>(n.value(), std::vector<float>(llr));
    live->add_or_modify_attrib_local<recog_mix_att>(n.value(), std::move(mix));
    live->add_or_modify_attrib_local<recog_mix_n_att>(n.value(), std::move(cnt));
    if (fresh) live->insert_node(n.value()); else live->update_node(n.value());
}

struct Posterior { std::vector<std::uint64_t> ids; std::vector<float> post; bool present = false; };
static Posterior read_recognition()
{
    Posterior p;
    const auto n = live->get_node("recognition");
    if (not n) return p;
    const auto ids = live->get_attrib_by_name<recog_room_ids_att>(n.value());
    const auto po = live->get_attrib_by_name<recog_posterior_att>(n.value());
    if (ids and po) { p.present = true; p.ids = ids->get(); p.post = po->get(); }
    return p;
}

// One scenario: the recogniser sees room_1 current (history), the robot enters room_2, then crosses DOOR_2 into a PROVISIONAL
// room (frame prov_w) whose entry mirror is the hole in the given orientation.
struct Scenario
{
    ltsm::RoomEviction ev;
    ltsm::RoomRecognizer rec;
    std::uint64_t prov = 0, mirror = 0;
    Scenario(bool turned_mirror, const Vec3 &robot_w)
        : ev(live, mem), rec(live, mem, ev)
    {
        set_current(live_r1);
        rec.step();                                    // the crossing history: room_1 was current
        set_current(live_r2);
        prov = live_room("room_4", 4, std::nullopt);
        put_edge(*live, robot_id, prov, Vec3::Zero(), Codo);
        live->insert_or_assign_edge(DSR::Edge::create<proto_edge_type>(prov, prov));
        mirror = live_door("door_9", prov);
        const Vec3 hole = turned_mirror ? Vec3(hole_w.x(), hole_w.y(), hole_w.z() + M_PI) : hole_w;
        put_edge(*live, prov, mirror, mul(inv(prov_w), hole));
        place_robot(robot_w, prov);
    }
    ~Scenario()
    {
        rec.sweep_stale();
        live->delete_node(mirror);
        live->delete_node(prov);
        set_current(live_r1);
    }
};


// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
// THE LIVE SEQUENCE OF recog_try2 (2026-10-24), fresh -> fresh. World W, every room in its OWN frame:
//   r1 [-2,2]x[-2,2] (A)   r2 [2,6]x[-2,2] (B)   room_3 [-2,6]x[2,6] (C, born a proto, promoted, live)
//   passage_0 r1<->r2 at (2,0) in memory.  DOOR_A room_1<->room_3 at (0,2): first-ever, no memory face.
//   DOOR_2 room_3<->r2 at (4,2): first-ever from room_3's side, the live door hangs from a WALL (non-identity frame) with
//   door_concept-like RT covariance.  DOOR_E on room_3's east wall at (6,4): behind it nothing memory knows.
// 1. room_1 current, proto room_3 behind DOOR_A: room_3's episode begins (candidate r2 only), the metric says NEW.
// 2. room_3 is promoted: room_1 leaves the live graph; memory gains r1's face of DOOR_A, room_3's promotion STUB twin
//    (room node + its entry face, no face of DOOR_2, a stale outline) and passage_1. room_3's episode keeps running.
// 3. a proto room_4 appears behind DOOR_2 (or room_5 behind DOOR_E): room_3's episode ends, room_4's begins in the same cycle.
static bool this_swap_state(const ltsm::RoomSwap &sw) { return sw.state() == ltsm::RoomSwap::State::Staged; }
enum class Seq { Known, KnownSameOrientationMirror, New, Wiring };
static int live_sequence(Seq mode, int8_t dom_live, int8_t dom_mem)
{
    const bool to_known_room = mode != Seq::New;
    live = std::make_shared<DSR::DSRGraph>("seq_live", 196 + dom_live, "ltsm_root.json", true, dom_live);
    mem  = std::make_shared<DSR::DSRGraph>("seq_mem", 196 + dom_mem, "ltsm_root.json", true, dom_mem);
    const int before = failures;
    auto rect = [](double x0, double y0, double x1, double y1) {
        return std::vector<Eigen::Vector2d>{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}; };
    const Vec3 A(10.0, 5.0, 0.3), B(-4.0, 7.0, -1.1), C(3.0, -8.0, 2.0);
    auto local = [](const Vec3 &frame, const Vec3 &w) { return mul(inv(frame), w); };
    auto set_poly = [](DSR::DSRGraph &G, DSR::Node &n, const Vec3 &frame, const std::vector<Eigen::Vector2d> &wp) {
        std::vector<float> px, py;
        for (const auto &w : wp) { const Vec3 l = mul(inv(frame), Vec3(w.x(), w.y(), 0.0)); px.push_back((float)l.x()); py.push_back((float)l.y()); }
        G.add_or_modify_attrib_local<delimiting_polygon_x_att>(n, px);
        G.add_or_modify_attrib_local<delimiting_polygon_y_att>(n, py); };

    // ── memory: r1, r2, passage_0 ──
    const auto mroot = mem->get_node("root").value().id();
    auto mem_room = [&](const std::string &name, std::uint64_t rid, const Vec3 &frame, const std::vector<Eigen::Vector2d> &wp) {
        DSR::Node n; n.type("room"); n.name(name);
        mem->add_or_modify_attrib_local<room_id_att>(n, rid);
        if (not wp.empty()) set_poly(*mem, n, frame, wp);
        const auto id = mem->insert_node(n).value();
        put_edge(*mem, mroot, id, Vec3::Zero());
        return id; };
    auto mem_face = [&](std::uint64_t room, const std::string &name, const Vec3 &frame, const Vec3 &face_w) {
        DSR::Node f; f.type("object"); f.name(name);
        mem->add_or_modify_attrib_local<object_subtype_att>(f, std::string("door"));
        mem->add_or_modify_attrib_local<parent_att>(f, room);
        const auto id = mem->insert_node(f).value();
        put_edge(*mem, room, id, local(frame, face_w));
        return id; };
    auto mem_passage = [&](const std::string &name, std::uint64_t fa, std::uint64_t fb) {
        DSR::Node pw; pw.type("metaconcept"); pw.name(name);
        mem->add_or_modify_attrib_local<object_subtype_att>(pw, std::string("passage"));
        const auto pid = mem->insert_node(pw).value();
        mem->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, fa));
        mem->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, fb)); };
    const auto m1 = mem_room("r1_room_1", 1, A, rect(-2, -2, 2, 2));
    const auto m2 = mem_room("r2_room_2", 2, B, rect(2, -2, 6, 2));
    mem_passage("passage_0", mem_face(m1, "r1_door_1", A, Vec3(2.0, 0.0, M_PI / 2)), mem_face(m2, "r2_door_1", B, Vec3(2.0, 0.0, M_PI / 2)));

    // ── live: robot, room_1 current with DOOR_A, proto room_3 behind it ──
    DSR::Node rn; rn.type("robot"); rn.name("Shadow");
    robot_id = live->insert_node(rn).value();
    put_edge(*live, live->get_node("root").value().id(), robot_id, Vec3::Zero());
    auto robot_at = [&](const Vec3 &w, std::initializer_list<std::pair<std::uint64_t, Vec3>> rooms_frames) {
        for (const auto &[room, frame] : rooms_frames) put_edge(*live, robot_id, room, inv(local(frame, w)), Codo); };
    const Vec3 robot_1(0.0, 2.6, M_PI / 2);                                  // 0.6 m into room_3
    const auto l1 = live_room("room_1", 1, 1);
    const auto l3 = live_room("room_3", 3, 3);
    robot_at(robot_1, {{l1, A}, {l3, C}});                                   // (the RT edge gives a room its level)
    { auto n = live->get_node(l1).value(); set_poly(*live, n, A, rect(-2, -2, 2, 2)); live->update_node(n); }
    const auto doorA = live_door("door_2", l1);
    put_edge(*live, l1, doorA, local(A, Vec3(0.0, 2.0, 0.0)));
    live->insert_or_assign_edge(DSR::Edge::create<proto_edge_type>(l3, l3));
    const auto mirA = live_door("door_9", l3);
    put_edge(*live, l3, mirA, local(C, Vec3(0.0, 2.0, M_PI)));             // the same hole seen from room_3, turned by pi
    set_current(l1);

    ltsm::RoomEviction ev(live, mem);
    ltsm::RoomRecognizer rec(live, mem, ev);
    ltsm::RoomRecognizer::Step st;
    bool any = false;
    for (int i = 0; i < 5; ++i) any = any or (st = rec.step()).act != ltsm::RoomRecognizer::Step::Act::None;
    auto p = read_recognition();
    std::printf("  [1] room_3 behind DOOR_A: ids"); for (auto id : p.ids) std::printf(" %llu", (unsigned long long)id);
    std::printf("  post"); for (float x : p.post) std::printf(" %.4f", x); std::printf("\n");
    check(rec.provisional_id() == l3, "[1] room_3's episode runs (L = room_1, first-ever door: no memory face)");
    check(p.present and p.ids == std::vector<std::uint64_t>{2} and p.post.size() == 2 and p.post[1] > 0.9f and not any,
          "[1] metric available from the live faces: room_3 is NEW (> 0.9), no request");

    // ── 2. PROMOTION of room_3 (what eviction #3 wrote) ──
    live->delete_edge(l3, l3, "proto");
    set_current(l3);
    live->delete_node(doorA);
    live->delete_edge(robot_id, l1, "RT");
    live->delete_node(l1);
    { auto n = live->get_node(l3).value(); set_poly(*live, n, C, rect(-2, 2, 6, 6)); live->update_node(n); }
    const auto m3 = mem_room("r3_room_3", 3, C, rect(-1, 2, 1, 3));          // the promotion STUB: a stale, partial outline
    mem_passage("passage_1", mem_face(m1, "r1_door_2", A, Vec3(0.0, 2.0, 0.0)), mem_face(m3, "r3_door_9", C, Vec3(0.0, 2.0, M_PI)));
    for (int i = 0; i < 3; ++i) any = any or rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
    check(rec.provisional_id() == l3 and not any, "[2] promoted: room_3's episode spans the promotion, no request");

    // ── 3. the robot crosses a door of room_3 nobody has crossed: a NEWER provisional room ──
    // the door hangs from a wall of room_3 (wall frame on the south wall, x along it), with an RT covariance in the WALL frame
    DSR::Node wn; wn.type("wall"); wn.name("room_3_wall_0");
    live->add_or_modify_attrib_local<parent_att>(wn, l3);
    const auto wall = live->insert_node(wn).value();
    const Vec3 wall_w = to_known_room ? Vec3(1.0, 2.0, 0.0) : Vec3(6.0, 2.0, M_PI / 2);
    put_edge(*live, l3, wall, local(C, wall_w));
    const Vec3 hole_w = to_known_room ? Vec3(4.0, 2.0, 0.0) : Vec3(6.0, 4.0, M_PI / 2);
    const auto D = live_door(to_known_room ? "door_4" : "door_5", wall);
    Mat3 Cd = Mat3::Zero(); Cd.diagonal() << 0.054 * 0.054, 0.01 * 0.01, 0.0173 * 0.0173;
    put_edge(*live, wall, D, local(wall_w, hole_w), Cd);
    const auto l4 = live_room(to_known_room ? "room_4" : "room_5", 4, 4);
    const Vec3 P_w(7.0, -3.0, 1.0);
    // 0.39 m past the aperture: where room_4's first promotion reading put it (and it was promoted that cycle)
    const Vec3 robot_2 = to_known_room ? Vec3(4.0, 2.0 - 0.39, -M_PI / 2) : Vec3(6.39, 4.0, 0.0);
    robot_at(robot_2, {{l3, C}, {l4, P_w}});
    live->insert_or_assign_edge(DSR::Edge::create<proto_edge_type>(l4, l4));
    const auto mir = live_door("door_7", l4);
    const bool turned = mode != Seq::KnownSameOrientationMirror;
    put_edge(*live, l4, mir, local(P_w, Vec3(hole_w.x(), hole_w.y(), hole_w.z() + (turned ? M_PI : 0.0))));
    std::optional<ltsm::RoomRecall> recall;
    std::optional<ltsm::RoomSwap> sw;
    if (mode == Seq::Wiring) { recall.emplace(live, mem, ev); sw.emplace(live, mem, ev, *recall); }
    auto cycle = [&] { return mode == Seq::Wiring ? ltsm::recognition_cycle(rec, &*sw) : rec.step(); };

    st = cycle();
    p = read_recognition();
    std::printf("  [3] %s: first step act %d ids", to_known_room ? "room_4 behind DOOR_2" : "room_5 behind DOOR_E", int(st.act));
    for (auto id : p.ids) std::printf(" %llu", (unsigned long long)id);
    std::printf("  post"); for (float x : p.post) std::printf(" %.4f", x); std::printf("\n");
    check(rec.provisional_id() == l4, "[3] in ONE step: room_3's episode ended and the episode for the newer provisional room began");
    check(p.present and p.ids == std::vector<std::uint64_t>({1, 2}) and p.post.size() == 3, "[3] candidates {r1, r2} (L = room_3 excluded), posterior + new");
    bool uniform = p.post.size() == 3;
    for (float x : p.post) uniform = uniform and std::abs(x - 1.f / 3.f) < 1e-3f;
    check(not uniform, "[3] metric available (the posterior is not the prior): no memory face of the crossed door, no full archive of L");
    if (to_known_room)
    {
        check(p.post.size() == 3 and p.post[1] > p.post[0] and p.post[1] > p.post[2], "[3] the TRUE room (r2) is the argmax");
        ltsm::RoomRecognizer::Step sw = st;
        int n = 1;
        for (; n < 5 and sw.act == ltsm::RoomRecognizer::Step::Act::None; ++n) sw = cycle();
        check(sw.act == ltsm::RoomRecognizer::Step::Act::Swap and sw.room_id == 2, "[3] Swap requested for r2 (step " + std::to_string(n) + ")");
        const Vec3 truth = local(B, robot_2);
        check(not sw.pose_in_k.empty() and (sw.pose_in_k[0].mu.head<2>().cast<double>() - truth.head<2>()).norm() < 0.05
                  and std::abs(wrap(double(sw.pose_in_k[0].mu.z()) - truth.z())) < 0.03,
              "[3] the Swap carries the robot's pose in r2 (< 5 cm, < 0.03 rad)");
    }
    if (mode == Seq::Wiring)
    {
        auto late_recalls = [&] {
            int n = 0;
            for (const auto &m : live->get_nodes_by_type("metaconcept"))
                if (live->get_attrib_by_name<recall_late_att>(m).value_or(false)) ++n;
            return n; };
        check(this_swap_state(*sw) and rec.swapped() and late_recalls() == 1 and not live->get_node(l4).has_value(),
              "[W] RoomSwap executed it (case A): Staged, recogniser held, K's late recall posted, P discarded");
        // room_concept re-births K from the late recall (a proto carrying K's room id), it is promoted, room_3 leaves
        const auto K = live_room("room_2", 2, 2);
        robot_at(robot_2, {{K, B}});
        live->insert_or_assign_edge(DSR::Edge::create<proto_edge_type>(K, K));
        bool any = false;
        for (int i = 0; i < 2; ++i) any = any or cycle().act != ltsm::RoomRecognizer::Step::Act::None;
        live->delete_edge(K, K, "proto");
        set_current(K);
        for (const auto id : {D, wall, mirA}) live->delete_node(id);
        live->delete_edge(robot_id, l3, "RT");
        live->delete_node(l3);
        { auto n = live->get_node(K).value(); set_poly(*live, n, B, rect(2, -2, 6, 2)); live->update_node(n); }
        ltsm::RoomRecognizer::Step h;
        for (int i = 0; i < 3; ++i) { h = cycle(); any = any or h.act != ltsm::RoomRecognizer::Step::Act::None or h.released; }
        check(not any and this_swap_state(*sw) and rec.swapped(), "[W] K re-born, promoted, current: the swap is held, the window open");
        // the robot crosses a door of K nobody has crossed: a newer provisional room with door + mirror
        DSR::Node kw; kw.type("wall"); kw.name("room_2_wall_1");
        live->add_or_modify_attrib_local<parent_att>(kw, K);
        const auto kwall = live->insert_node(kw).value();
        put_edge(*live, K, kwall, local(B, Vec3(6.0, -2.0, M_PI / 2)));
        const Vec3 holeE(6.0, 0.0, M_PI / 2);
        const auto DK = live_door("door_8", kwall);
        put_edge(*live, kwall, DK, local(Vec3(6.0, -2.0, M_PI / 2), holeE), Cd);
        const auto P6 = live_room("room_6", 6, 6);
        const Vec3 P6_w(-5.0, 4.0, -2.2), robot_3(6.4, 0.0, 0.0);
        robot_at(robot_3, {{K, B}, {P6, P6_w}});
        live->insert_or_assign_edge(DSR::Edge::create<proto_edge_type>(P6, P6));
        const auto M6 = live_door("door_10", P6);
        put_edge(*live, P6, M6, local(P6_w, Vec3(holeE.x(), holeE.y(), holeE.z() + M_PI)));
        const auto r = cycle();
        p = read_recognition();
        std::printf("  [W] newer room_6 behind door_8: released %d act %d prov %llu ids", int(r.released), int(r.act), (unsigned long long)rec.provisional_id());
        for (auto id : p.ids) std::printf(" %llu", (unsigned long long)id);
        std::printf("  post"); for (float x : p.post) std::printf(" %.4f", x); std::printf("\n");
        check(r.released and r.act == ltsm::RoomRecognizer::Step::Act::None, "[W] the step reports the held episode RELEASED, and no Undo");
        check(not this_swap_state(*sw) and late_recalls() == 0, "[W] the wiring closed RoomSwap's window (on_room_left): Entering, K's late recall retired");
        {   // the swap anchored on the CROSSED door (door_4, wall-hung), not room_3's entry mirror: passage_1 (r1<->r3) untouched
            const auto p1 = mem->get_node("passage_1");
            check(p1.has_value() and DSR::DSRGraph::get_node_edges_by_type(p1.value(), "match").size() == 2,
                  "[W] passage_1 (r1<->r3) still has exactly its two faces: K's face was not carried into the door the robot came IN by");
        }
        check(rec.provisional_id() == P6 and not rec.swapped() and p.present, "[W] and the new episode for room_6 runs, in the same cycle");
        const auto r2 = cycle();
        check(not r2.released, "[W] released is one-shot");
    }
    if (mode == Seq::New)
    {
        bool req = st.act != ltsm::RoomRecognizer::Step::Act::None;
        for (int i = 0; i < 50; ++i) req = req or rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        p = read_recognition();
        check(not req and p.post.size() == 3 and p.post[2] > 0.9f, "[3] a genuinely new room behind DOOR_E stays NEW (> 0.9), no request in 50 steps");
    }
    rec.sweep_stale();
    return failures - before;
}

int main(int argc, char **argv)
{
    std::setlocale(LC_ALL, "");
    setvbuf(stdout, nullptr, _IONBF, 0);
    QCoreApplication app(argc, argv);
    live = std::make_shared<DSR::DSRGraph>("recog_live", 194, "ltsm_root.json", true, 44);
    mem  = std::make_shared<DSR::DSRGraph>("recog_mem", 195, "ltsm_root.json", true, 45);
    Codo = Mat3::Zero();  Codo.diagonal() << 0.05 * 0.05, 0.05 * 0.05, 0.02 * 0.02;
    build_memory();
    build_live();

    const Vec3 robot_in_r3_w(3.0, 4.0, 0.5);              // truly inside r3
    const Vec3 far_w(9.0, 9.0, 0.0);                      // outside every outline (>= 3 m from r3's edge)
    auto in_room = [&](int idx, const Vec3 &w) { return mul(inv(rooms[idx].in_world), w); };

    std::printf("fresh_seam_convention_metric_alone_commits_known_room\n");
    for (const bool turned : {false, true})
    {
        std::printf(" mirror %s\n", turned ? "turned by pi" : "same orientation");
        Scenario s(turned, robot_in_r3_w);
        ltsm::RoomRecognizer::Step st;
        int steps = 0;
        for (; steps < 5 and st.act == ltsm::RoomRecognizer::Step::Act::None; ++steps) st = s.rec.step();
        check(st.act == ltsm::RoomRecognizer::Step::Act::Swap and st.room_id == 3, "Swap with R3's room_id within a few steps");
        const Vec3 truth = in_room(2, robot_in_r3_w);
        check(not st.pose_in_k.empty() and (st.pose_in_k[0].mu.head<2>().cast<double>() - truth.head<2>()).norm() < 0.02,
              "the Swap carries the robot's pose in R3 (< 2 cm): the fresh seam lands the robot in the TRUE room");
        const auto p = read_recognition();
        check(p.present and p.ids == std::vector<std::uint64_t>{1, 3} and p.post.size() == 3, "recognition node: ids {1,3}, 3 posterior entries (known..., new)");
        check(p.present and p.post.size() == 3 and p.post[1] > 0.99f, "recog_posterior(R3) > 0.99");
        check(s.rec.step().act == ltsm::RoomRecognizer::Step::Act::None, "committed: no second Swap request");
    }

    std::printf("fresh_seam_arbitrary_provisional_frame_is_irrelevant\n");
    {   // the same robot, another provisional frame: the posterior must not move
        Scenario s(false, robot_in_r3_w);
        for (int i = 0; i < 3; ++i) s.rec.step();
        const auto a = read_recognition();
        check(a.present and a.post.size() == 3 and a.post[1] > 0.99f, "R3 recognised under prov_w");
    }

    std::printf("entry_without_evidence_does_nothing\n");
    for (const double y : {2.0, 2.05, 2.4})      // in the aperture, just past it (either side of the margin: new .58 | R3 .64), well past (R3 starts at y = 2)
    {
        Scenario s(false, Vec3(4.0, y, 1.5707963));
        ltsm::RoomRecognizer::Step st;
        for (int i = 0; i < 5 and st.act == ltsm::RoomRecognizer::Step::Act::None; ++i) st = s.rec.step();
        const auto p = read_recognition();
        const bool ok = p.present and p.post.size() == 3;
        std::printf("  y=%.2f posterior R1 %.4f R3 %.4f new %.4f   act %d\n", y, ok ? p.post[0] : -1.f, ok ? p.post[1] : -1.f, ok ? p.post[2] : -1.f, int(st.act));
        check(ok and p.post[0] < 1e-6f, "R1 is excluded by the metric");
        if (y < 2.01)
            check(ok and p.post[2] > p.post[1] and st.act == ltsm::RoomRecognizer::Step::Act::None, "in the aperture: `new` leads (0.58 vs 0.42 asserted as an ordering), no request");
        else
            check(ok and p.post[1] > (y < 2.2 ? 0.5f : 0.99f) and st.act == ltsm::RoomRecognizer::Step::Act::Swap and st.room_id == 3, "past the aperture: R3 leads and the Swap is requested; well past: R3 > 0.99");
    }

    std::printf("unresolvable_face_means_no_metric\n");
    for (const bool lose_D : {true, false})
    {   // the live passage pairs D with the mirror (so both are FOUND without any transform); then one face loses its RT edge
        Scenario s(false, robot_in_r3_w);
        DSR::Node pw; pw.type("metaconcept"); pw.name("passage_live");
        live->add_or_modify_attrib_local<object_subtype_att>(pw, std::string("passage"));
        const auto pid = live->insert_node(pw).value();
        live->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, s.mirror));
        live->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, live_D));
        if (lose_D) live->delete_edge(live_r2, live_D, "RT"); else live->delete_edge(s.prov, s.mirror, "RT");
        bool any = false;
        for (int i = 0; i < 5; ++i) any = any or s.rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        const auto p = read_recognition();
        bool uniform = p.present and p.post.size() == 3;
        for (float x : p.post) uniform = uniform and std::abs(x - 1.f / 3.f) < 1e-5f;
        check(s.rec.provisional_id() == s.prov and uniform and not any,
              std::string(lose_D ? "D" : "the mirror") + " unresolvable: the episode runs, the metric is unavailable (posterior = prior), no request");
        live->delete_node(pid);
        if (lose_D) put_edge(*live, live_r2, live_D, mul(inv(rooms[1].in_world), hole_w));
    }

    std::printf("first_ever_door_needs_no_memory_face\n");
    {   // the crossed door's face is missing in memory (a first-ever door): the seam comes from the LIVE faces, the metric stands
        mem->delete_node(face_r2_door);
        Scenario s(false, robot_in_r3_w);
        ltsm::RoomRecognizer::Step st;
        for (int i = 0; i < 5 and st.act == ltsm::RoomRecognizer::Step::Act::None; ++i) st = s.rec.step();
        const auto p = read_recognition();
        check(st.act == ltsm::RoomRecognizer::Step::Act::Swap and st.room_id == 3 and p.present and p.post.size() == 3 and p.post[1] > 0.99f,
              "no memory face of DOOR_2: R3 > 0.99 and the Swap is requested (fresh seam from live D + live mirror)");
    }

    std::printf("no_information_is_the_prior\n");
    {   // (face still missing) the robot pose in P states no covariance: the robot chain is unmeasurable => metric unavailable
        Scenario s(false, robot_in_r3_w);
        unmeasured_robot(s.prov, robot_in_r3_w);
        bool any = false;
        for (int i = 0; i < 20; ++i) any = any or s.rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        const auto p = read_recognition();
        bool uniform = p.present and p.post.size() == 3;
        for (float x : p.post) uniform = uniform and std::abs(x - 1.f / 3.f) < 1e-5f;
        check(not any and uniform, "posterior = the uniform prior over {R1, R3, new}, never NaN, never a request");
    }

    std::printf("appearance_alone_commits_when_the_metric_is_unavailable\n");
    {   // (robot chain unmeasurable) an expert that is absent is omitted, not zeroed: the other decides
        Scenario s(false, robot_in_r3_w);
        unmeasured_robot(s.prov, robot_in_r3_w);
        write_evidence({1, 3}, {-2.f, 12.f}, {std::nullopt, in_room(2, robot_in_r3_w)});
        ltsm::RoomRecognizer::Step st;
        for (int i = 0; i < 5 and st.act == ltsm::RoomRecognizer::Step::Act::None; ++i) st = s.rec.step();
        check(st.act == ltsm::RoomRecognizer::Step::Act::Swap and st.room_id == 3, "Swap R3 on appearance alone when there is no metric");
        live->delete_node(live->get_node("place_evidence").value().id());
    }

    std::printf("symmetric_candidates_no_commit\n");
    {   // no metric; two candidates with IDENTICAL evidence: the posterior splits, nothing is committed
        Scenario s(false, robot_in_r3_w);
        unmeasured_robot(s.prov, robot_in_r3_w);
        write_evidence({1, 3}, {3.f, 3.f}, {in_room(2, robot_in_r3_w), in_room(2, robot_in_r3_w)});
        bool any = false;
        for (int i = 0; i < 200; ++i) any = any or s.rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        const auto p = read_recognition();
        check(not any, "no request over 200 steps");
        check(p.present and p.post.size() == 3 and p.post[0] == p.post[1] and p.post[0] > 0.4f and p.post[2] < 0.1f, "posterior splits evenly between the two");
        live->delete_node(live->get_node("place_evidence").value().id());
    }

    // restore DOOR_2's face (the metric is back from here on)
    {
        DSR::Node f; f.type("object"); f.name("r2_door_3");
        mem->add_or_modify_attrib_local<object_subtype_att>(f, std::string("door"));
        mem->add_or_modify_attrib_local<parent_att>(f, rooms[1].id);
        face_r2_door = mem->insert_node(f).value();
        put_edge(*mem, rooms[1].id, face_r2_door, mul(inv(rooms[1].in_world), hole_w));
    }

    std::printf("appearance_alone_cannot_commit_against_metric\n");
    {
        Scenario s(false, far_w);                                        // the chain puts the robot outside every polygon
        write_evidence({1, 3}, {0.f, 25.f}, {std::nullopt, in_room(2, far_w)});
        bool any = false;
        for (int i = 0; i < 20; ++i) any = any or s.rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        const auto p = read_recognition();
        std::printf("  posterior:"); for (float x : p.post) std::printf(" %.4g", x); std::printf("\n");
        check(not any, "appearance +25 nats for R3 does not commit against a metric that excludes it");
        check(p.present and p.post.size() == 3 and p.post[2] > 0.9f, "the metric keeps `new` on top");
        live->delete_node(live->get_node("place_evidence").value().id());
    }

    std::printf("new_room_stays_new\n");
    {
        Scenario s(false, far_w);
        write_evidence({1, 3}, {-5.f, -5.f}, {std::nullopt, std::nullopt});
        bool any = false;
        for (int i = 0; i < 200; ++i) any = any or s.rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        const auto p = read_recognition();
        check(not any, "never a request over 200 steps");
        check(p.present and p.post.size() == 3 and p.post[2] > 0.9f, "recog_posterior(new) > 0.9");
        live->delete_node(live->get_node("place_evidence").value().id());
    }

    std::printf("commit_then_contradiction_is_undone_and_not_recommitted\n");
    {
        Scenario s(false, robot_in_r3_w);
        ltsm::RoomRecognizer::Step st;
        for (int i = 0; i < 5 and st.act == ltsm::RoomRecognizer::Step::Act::None; ++i) st = s.rec.step();
        check(st.act == ltsm::RoomRecognizer::Step::Act::Swap and st.room_id == 3, "Swap R3 on the metric");
        write_evidence({1, 3}, {-1000.f, -1000.f}, {std::nullopt, std::nullopt});     // after the commit: nothing here looks like R1 or R3
        ltsm::RoomRecognizer::Step u;
        for (int i = 0; i < 5 and u.act == ltsm::RoomRecognizer::Step::Act::None; ++i) u = s.rec.step();
        check(u.act == ltsm::RoomRecognizer::Step::Act::Undo and u.room_id == 3, "Undo R3: evidence after the commit says it is not R3");
        bool again = false;
        for (int i = 0; i < 20; ++i) again = again or s.rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        check(not again, "unchanged evidence after the Undo: no re-commit (posterior since the change = the prior)");
        write_evidence({1, 3}, {-1000.f, 1500.f}, {std::nullopt, in_room(2, robot_in_r3_w)});   // fresh evidence for R3
        ltsm::RoomRecognizer::Step r;
        for (int i = 0; i < 5 and r.act == ltsm::RoomRecognizer::Step::Act::None; ++i) r = s.rec.step();
        check(r.act == ltsm::RoomRecognizer::Step::Act::Swap and r.room_id == 3, "fresh evidence for R3 after the Undo: re-commit");
        live->delete_node(live->get_node("place_evidence").value().id());
    }

    std::printf("appearance_carried_commit_is_not_undone_by_unchanged_evidence\n");
    {   // the metric mildly prefers `new` (the robot is just short of R3), appearance strongly says R3
        const Vec3 near_w(4.0, 2.0, 1.5707963);
        Scenario s(false, near_w);
        s.rec.step();
        const auto p0 = read_recognition();
        std::printf("  metric alone: R3 %.3f new %.3f\n", p0.post.size() == 3 ? p0.post[1] : -1.f, p0.post.size() == 3 ? p0.post[2] : -1.f);
        check(p0.post.size() == 3 and p0.post[2] > p0.post[1], "the metric alone prefers `new`");
        write_evidence({1, 3}, {0.f, 12.f}, {std::nullopt, in_room(2, near_w)});
        ltsm::RoomRecognizer::Step st;
        for (int i = 0; i < 5 and st.act == ltsm::RoomRecognizer::Step::Act::None; ++i) st = s.rec.step();
        check(st.act == ltsm::RoomRecognizer::Step::Act::Swap and st.room_id == 3, "appearance +12 carries the commit against the mild metric");
        bool any = false;
        for (int i = 0; i < 50; ++i) any = any or s.rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        check(not any, "50 cycles of unchanged evidence: no Undo");
        live->delete_node(live->get_node("place_evidence").value().id());
    }

    std::printf("episode_spans_promotion\n");
    {
        Scenario s(false, robot_in_r3_w);
        ltsm::RoomRecognizer::Step st;
        for (int i = 0; i < 5 and st.act == ltsm::RoomRecognizer::Step::Act::None; ++i) st = s.rec.step();
        check(st.act == ltsm::RoomRecognizer::Step::Act::Swap, "Swap requested before promotion");
        // PROMOTION: the proto edge goes, `current` moves to the provisional room, the old room leaves the live graph
        live->delete_edge(s.prov, s.prov, "proto");
        set_current(s.prov);
        live->delete_node(s.mirror);                        // door_concept retires the mirror with its source
        bool any = false;
        for (int i = 0; i < 10; ++i) any = any or s.rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        const auto p = read_recognition();
        check(not any and p.present and p.post.size() == 3 and p.post[1] > 0.99f, "after promotion: episode and commit kept, R3 still recognised (cached seam)");
        set_current(live_r1);                               // the eviction fence: the robot leaves the room
        const auto f = s.rec.step();
        check(f.act == ltsm::RoomRecognizer::Step::Act::None and not read_recognition().present, "at the fence the episode ends with no Undo");
        s.mirror = live_door("door_9", s.prov);             // the destructor deletes it
        live->insert_or_assign_edge(DSR::Edge::create<proto_edge_type>(s.prov, s.prov));
    }

    std::printf("episode_lost_under_a_commit_requests_undo\n");
    {
        Scenario s(false, robot_in_r3_w);
        ltsm::RoomRecognizer::Step st;
        for (int i = 0; i < 5 and st.act == ltsm::RoomRecognizer::Step::Act::None; ++i) st = s.rec.step();
        live->delete_node(s.mirror);                        // pre-promotion, the crossing evidence vanishes
        const auto u = s.rec.step();
        check(u.act == ltsm::RoomRecognizer::Step::Act::Undo and u.room_id == 3 and not read_recognition().present, "Undo requested when the episode is lost under a commit");
        s.mirror = live_door("door_9", s.prov);
    }

    std::printf("executed_swap_changes_are_not_episode_loss\n");
    {
        Scenario s(false, robot_in_r3_w);
        ltsm::RoomRecognizer::Step st;
        for (int i = 0; i < 5 and st.act == ltsm::RoomRecognizer::Step::Act::None; ++i) st = s.rec.step();
        check(st.act == ltsm::RoomRecognizer::Step::Act::Swap and st.room_id == 3, "Swap requested");
        s.rec.on_swap_executed(3);                          // RoomSwap executed it: P is discarded with its mirror
        live->delete_node(s.mirror);
        live->delete_node(s.prov);
        bool any = false;
        for (int i = 0; i < 3; ++i) any = any or s.rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        check(not any and s.rec.swapped(), "P deleted by the swap itself, before promotion: no Undo, the swap is held");
        // the crossed door is now in a passage (the swap's carry-over): known_door must not read as loss either
        DSR::Node pw; pw.type("metaconcept"); pw.name("passage_9");
        mem->add_or_modify_attrib_local<object_subtype_att>(pw, std::string("passage"));
        const auto pid = mem->insert_node(pw).value();
        mem->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, face_r2_door));
        for (int i = 0; i < 3; ++i) any = any or s.rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        check(not any and s.rec.swapped(), "the crossed door gaining a passage: still no Undo");
        mem->delete_node(pid);
        set_current(live_r1);                               // the robot leaves: the eviction fence
        const auto f = s.rec.step();
        check(f.act == ltsm::RoomRecognizer::Step::Act::None and not s.rec.swapped() and s.rec.provisional_id() == 0, "the fence ends the held episode with no Undo");
        s.mirror = live_door("door_9", live_r2);            // so the destructor has something to delete
        s.prov = s.mirror;
    }

    std::printf("late_adoption_failure_requests_undo\n");
    {
        Scenario s(false, robot_in_r3_w);
        ltsm::RoomRecognizer::Step st;
        for (int i = 0; i < 5 and st.act == ltsm::RoomRecognizer::Step::Act::None; ++i) st = s.rec.step();
        s.rec.on_swap_executed(3);
        live->delete_node(s.mirror);
        live->delete_node(s.prov);
        DSR::Node rn; rn.type("metaconcept"); rn.name("recall_room_3");
        live->add_or_modify_attrib_local<object_subtype_att>(rn, std::string("room_recall"));
        live->add_or_modify_attrib_local<room_id_att>(rn, std::uint64_t(3));
        live->add_or_modify_attrib_local<recall_late_att>(rn, true);
        const auto rid = live->insert_node(rn).value();
        bool any = false;
        for (int i = 0; i < 3; ++i) any = any or s.rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        check(not any and s.rec.swapped(), "recall present, no objection: the swap is held");
        const auto Kr = live_node_for_test("room_3", 3);       // room_concept re-births K; the recall is still there
        auto n2 = live->get_node(rid).value();
        live->add_or_modify_attrib_local<valid_att>(n2, true);
        live->update_node(n2);
        check(s.rec.step().act == ltsm::RoomRecognizer::Step::Act::None, "valid = true: still no objection");
        live->add_or_modify_attrib_local<valid_att>(n2, false);          // room_concept: the late-adoption fit rejected K
        live->update_node(n2);
        const auto u = s.rec.step();
        check(u.act == ltsm::RoomRecognizer::Step::Act::Undo and u.room_id == 3 and not s.rec.swapped(), "valid = false on the late recall: Undo requested");
        s.rec.on_undo_executed();
        live->delete_node(Kr);
        live->delete_node(rid);
        s.mirror = live_door("door_9", live_r2);
        s.prov = s.mirror;
    }

    std::printf("held_swap_ends_without_undo_when_a_newer_provisional_room_appears\n");
    {
        Scenario s(false, robot_in_r3_w);
        ltsm::RoomRecognizer::Step st;
        for (int i = 0; i < 5 and st.act == ltsm::RoomRecognizer::Step::Act::None; ++i) st = s.rec.step();
        s.rec.on_swap_executed(3);
        live->delete_node(s.mirror);
        live->delete_node(s.prov);
        // room_concept re-births K as a PROTO carrying K's room id (adopt_late_recall), then it is promoted and current
        const auto Kr = live_node_for_test("room_3", 3);
        put_edge(*live, robot_id, Kr, Vec3::Zero(), Codo);
        live->insert_or_assign_edge(DSR::Edge::create<proto_edge_type>(Kr, Kr));
        bool any = false;
        for (int i = 0; i < 3; ++i) any = any or s.rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        live->delete_edge(Kr, Kr, "proto");
        set_current(Kr);
        for (int i = 0; i < 3; ++i) any = any or s.rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        check(not any and s.rec.swapped(), "K's own proto re-birth and promotion are not a newer provisional room: the swap is held");
        const auto Pn = live_room("room_7", 7, 7);                          // the robot crosses another door out of K
        put_edge(*live, robot_id, Pn, Vec3::Zero(), Codo);
        live->insert_or_assign_edge(DSR::Edge::create<proto_edge_type>(Pn, Pn));
        const auto a = s.rec.step();
        check(a.act == ltsm::RoomRecognizer::Step::Act::None and not s.rec.swapped(), "a newer provisional room ends the held episode with no Undo (the fence)");
        live->delete_node(Pn);
        live->delete_node(Kr);
        set_current(live_r1);
        s.mirror = live_door("door_9", live_r2);
        s.prov = s.mirror;
    }

    std::printf("failed_swap_reverts_to_entering\n");
    {
        Scenario s(false, robot_in_r3_w);
        ltsm::RoomRecognizer::Step st;
        for (int i = 0; i < 5 and st.act == ltsm::RoomRecognizer::Step::Act::None; ++i) st = s.rec.step();
        check(st.act == ltsm::RoomRecognizer::Step::Act::Swap, "Swap requested");
        s.rec.on_swap_failed();
        bool any = false;
        for (int i = 0; i < 10; ++i) any = any or s.rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        check(not any and not s.rec.swapped(), "failed: back to Entering, no Undo, no re-commit on unchanged evidence");
        live->delete_node(s.mirror);                        // the episode is then lost: nothing stands, so no Undo
        check(s.rec.step().act == ltsm::RoomRecognizer::Step::Act::None, "losing the episode after a failed swap requests no Undo");
        s.mirror = live_door("door_9", s.prov);
    }

    std::printf("episode_ends_with_the_provisional_room\n");
    {
        {
            Scenario s(false, robot_in_r3_w);
            s.rec.step();
            check(read_recognition().present, "recognition node present while the provisional room exists");
            live->delete_node(s.mirror);
            s.rec.step();
            check(not read_recognition().present, "recognition node removed when the entry mirror is gone");
            s.mirror = live_door("door_9", s.prov);                      // so the destructor has something to delete
        }
        check(not read_recognition().present, "and nothing is left behind after the episode");
    }

    std::printf("known_door_is_not_audited\n");
    {   // DOOR_2 already in a passage: recall is trusted, recognition does not run
        DSR::Node pw; pw.type("metaconcept"); pw.name("passage_9");
        mem->add_or_modify_attrib_local<object_subtype_att>(pw, std::string("passage"));
        const auto pid = mem->insert_node(pw).value();
        mem->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, face_r2_door));
        Scenario s(false, robot_in_r3_w);
        bool any = false;
        for (int i = 0; i < 5; ++i) any = any or s.rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        check(not any and not read_recognition().present, "no episode for a door that already has a passage");
        mem->delete_node(pid);
    }

    std::printf("same_name_different_hole_is_not_a_known_door\n");
    {   // recog_try12: door_concept re-used a door NAME (an entry door renamed, its old name free for a NEW doorway). Memory holds a
        // passage face carrying D's name, but at ANOTHER place: the hole is unknown, the episode must run.
        const Vec3 elsewhere_w(1.0, -2.0, 0.0);      // r2's remembered face, moved 3 m along r2's south wall
        put_edge(*mem, rooms[1].id, face_r2_door, mul(inv(rooms[1].in_world), elsewhere_w));
        DSR::Node pw; pw.type("metaconcept"); pw.name("passage_9");
        mem->add_or_modify_attrib_local<object_subtype_att>(pw, std::string("passage"));
        const auto pid = mem->insert_node(pw).value();
        mem->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, face_r2_door));
        Scenario s(false, robot_in_r3_w);
        ltsm::RoomRecognizer::Step st;
        for (int i = 0; i < 5 and st.act == ltsm::RoomRecognizer::Step::Act::None; ++i) st = s.rec.step();
        check(read_recognition().present and s.rec.provisional_id() == s.prov, "same name, other place: the episode runs for the provisional room");
        mem->delete_node(pid);
        put_edge(*mem, rooms[1].id, face_r2_door, mul(inv(rooms[1].in_world), hole_w));
    }

    std::printf("stale_recognition_swept_at_start\n");
    {
        for (const char *sub : {"recognition", "place_evidence"})
        {
            DSR::Node n; n.type("metaconcept"); n.name(sub);
            live->add_or_modify_attrib_local<object_subtype_att>(n, std::string(sub));
            live->insert_node(n);
        }
        ltsm::RoomEviction ev(live, mem);
        ltsm::RoomRecognizer rec(live, mem, ev);
        const int n = rec.sweep_stale();
        check(n == 2 and not live->get_node("recognition").has_value() and not live->get_node("place_evidence").has_value(),
              "both leftovers removed (" + std::to_string(n) + ")");
    }

    std::printf("no_dsr_signal_connections\n");
    {
        bool found = false, readable = false;
        for (const char *f : {"src/room_recognizer.h", "src/room_recognizer.cpp"})
        {
            std::ifstream in(f);
            readable = readable or in.good();
            std::stringstream ss; ss << in.rdbuf();
            const std::string t = ss.str();
            found = found or t.find("Direct" "Connection") != std::string::npos or t.find("connect(") != std::string::npos
                    or t.find("QObject::connect") != std::string::npos;
        }
        check(readable and not found, "no DirectConnection / connect( in room_recognizer.*");
    }

    std::printf("live_sequence_fresh_to_fresh_known_room_behind_first_ever_door\n");
    live_sequence(Seq::Known, 46, 47);
    std::printf("live_sequence_fresh_to_fresh_same_orientation_mirror\n");
    live_sequence(Seq::KnownSameOrientationMirror, 50, 51);
    std::printf("live_sequence_fresh_to_fresh_new_room_stays_new\n");
    live_sequence(Seq::New, 48, 49);
    std::printf("wiring_held_swap_released_by_a_newer_provisional_room\n");
    live_sequence(Seq::Wiring, 52, 53);

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "all ok", failures);
    std::fflush(stdout);
    std::_Exit(failures ? 1 : 0);
}
