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

    std::printf("no_information_is_the_prior\n");
    {   // the crossed door's face is missing in memory: the fresh seam is unmeasurable => metric unavailable, no evidence at all
        mem->delete_node(face_r2_door);
        Scenario s(false, robot_in_r3_w);
        bool any = false;
        for (int i = 0; i < 20; ++i) any = any or s.rec.step().act != ltsm::RoomRecognizer::Step::Act::None;
        const auto p = read_recognition();
        bool uniform = p.present and p.post.size() == 3;
        for (float x : p.post) uniform = uniform and std::abs(x - 1.f / 3.f) < 1e-5f;
        check(not any and uniform, "posterior = the uniform prior over {R1, R3, new}, never NaN, never a request");
    }

    std::printf("appearance_alone_commits_when_the_metric_is_unavailable\n");
    {   // (face still missing) an expert that is absent is omitted, not zeroed: the other decides
        Scenario s(false, robot_in_r3_w);
        write_evidence({1, 3}, {-2.f, 12.f}, {std::nullopt, in_room(2, robot_in_r3_w)});
        ltsm::RoomRecognizer::Step st;
        for (int i = 0; i < 5 and st.act == ltsm::RoomRecognizer::Step::Act::None; ++i) st = s.rec.step();
        check(st.act == ltsm::RoomRecognizer::Step::Act::Swap and st.room_id == 3, "Swap R3 on appearance alone when there is no metric");
        live->delete_node(live->get_node("place_evidence").value().id());
    }

    std::printf("symmetric_candidates_no_commit\n");
    {   // no metric; two candidates with IDENTICAL evidence: the posterior splits, nothing is committed
        Scenario s(false, robot_in_r3_w);
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

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "all ok", failures);
    std::fflush(stdout);
    std::_Exit(failures ? 1 : 0);
}
