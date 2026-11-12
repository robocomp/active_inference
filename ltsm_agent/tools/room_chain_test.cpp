// room_chain: the path-taken seam chain against each known room's OWN chain (metric room-recognition evidence).
//   run from the component root:  ./bin/room_chain_test
// Scene = the live loop, in a common world frame W (each room has its OWN, arbitrarily posed frame, as in memory):
//   r1_room_1 [-2,2]x[-2,2]   r2_room_2 [2,6]x[-2,2]   r3_room_3 [-2,6]x[2,6]
//   passage_0 r1<->r2 at (2,0)   passage_1 r1<->r3 at (0,2)
//   DOOR_2 at (4,2): memory holds r2's face ONLY (no passage); the robot crossed it into a PROVISIONAL room whose frame is
//   `prov_w`; the fresh seam T(r2 <- prov) comes from the two faces and is passed in.
// The robot stands at (3,4) in W, i.e. inside r3, having come r1 -> r2 -> DOOR_2.
#include "../src/room_chain.h"
#include "../src/seam_fusion.h"
#include <QCoreApplication>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
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

struct Room { std::string name; std::uint64_t id; Vec3 in_world; std::vector<Eigen::Vector2d> world_poly; };

int main(int argc, char **argv)
{
    std::setlocale(LC_ALL, "");
    setvbuf(stdout, nullptr, _IONBF, 0);
    QCoreApplication app(argc, argv);
    auto mem = std::make_shared<DSR::DSRGraph>("chain_mem", 193, "ltsm_root.json", true, 43);
    const auto root = mem->get_node("root").value().id();

    auto rect = [](double x0, double y0, double x1, double y1) {
        return std::vector<Eigen::Vector2d>{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}; };
    std::vector<Room> rooms{{"r1_room_1", 0, Vec3(10.0, 5.0, 0.3), rect(-2, -2, 2, 2)},
                            {"r2_room_2", 0, Vec3(-4.0, 7.0, -1.1), rect(2, -2, 6, 2)},
                            {"r3_room_3", 0, Vec3(3.0, -8.0, 2.0), rect(-2, 2, 6, 6)}};
    for (auto &r : rooms)
    {
        DSR::Node n; n.type("room"); n.name(r.name);
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
    std::uint64_t face_ids[3][2];   // passage -> {face in first room, face in second room}
    auto add_passage = [&](int i, const Room &a, const Room &b, const Vec3 &hole) {
        face_ids[i][0] = add_face(a, "f" + std::to_string(i) + "_" + a.name, hole);
        face_ids[i][1] = add_face(b, "f" + std::to_string(i) + "_" + b.name, hole);
        DSR::Node pw; pw.type("metaconcept"); pw.name("passage_" + std::to_string(i));
        mem->add_or_modify_attrib_local<object_subtype_att>(pw, std::string("passage"));
        const auto pid = mem->insert_node(pw).value();
        mem->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, face_ids[i][0]));
        mem->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, face_ids[i][1]));
    };
    add_passage(0, rooms[0], rooms[1], Vec3(2.0, 0.0, M_PI / 2));
    add_passage(1, rooms[0], rooms[2], Vec3(0.0, 2.0, 0.0));
    add_face(rooms[1], "r2_door_3", Vec3(4.0, 2.0, 0.0));     // DOOR_2's remembered face: unmatched, no passage

    const Vec3 robot_w(3.0, 4.0, 0.5);
    const Vec3 prov_w(7.0, -3.0, 1.0);      // the provisional room's frame (arbitrary)
    const Vec3 robot_in_r3 = mul(inv(rooms[2].in_world), robot_w);
    const Vec3 robot_in_prov = mul(inv(prov_w), robot_w);
    Mat3 Cface = Mat3::Zero();  Cface.diagonal() << 0.03 * 0.03, 0.03 * 0.03, 0.01 * 0.01;
    // fresh seam T(r2 <- prov) from the crossed door's two faces (each 3 cm / 0.01 rad)
    const ltsm::ChainPose fresh{mul(inv(rooms[1].in_world), prov_w), 2.0 * Cface};
    const Vec3 robot_in_ref_truth = mul(inv(rooms[0].in_world), robot_w);
    Mat3 Codo = Mat3::Zero();  Codo.diagonal() << 0.05 * 0.05, 0.05 * 0.05, 0.02 * 0.02;
    const std::vector<std::string> path{"passage_0"};

    auto argmax = [](const std::vector<rc::recog::MetricEvidence> &ev) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < ev.size(); ++i)
            if (ev[i].available and (not ev[best].available or ev[i].log_inside > ev[best].log_inside)) best = i;
        return best; };
    auto index_of = [](const ltsm::MetricQuery &q, const std::string &name) {
        for (std::size_t i = 0; i < q.rooms.size(); ++i) if (q.rooms[i].mem_room == name) return int(i);
        return -1; };

    std::printf("loop_consistency\n");
    {
        const auto q = ltsm::build_metric_query(*mem, path, "r1_room_1", fresh, {robot_in_prov, Codo});
        const auto q_direct = ltsm::build_metric_query(*mem, {"passage_1"}, "r1_room_1", ltsm::ChainPose{mul(inv(rooms[2].in_world), prov_w), 2.0 * Cface}, {robot_in_prov, Codo});
        check(q.robot_ok and q_direct.robot_ok, "both chains resolve");
        const Vec3 a = q.robot_in_ref.pose, b = q_direct.robot_in_ref.pose;
        std::printf("  path chain (%+.4f,%+.4f,%+.4f)  direct chain (%+.4f,%+.4f,%+.4f)  truth (%+.4f,%+.4f,%+.4f)\n",
                    a.x(), a.y(), a.z(), b.x(), b.y(), b.z(), robot_in_ref_truth.x(), robot_in_ref_truth.y(), robot_in_ref_truth.z());
        check((a - b).head<2>().norm() < 0.01 and std::abs(wrap(a.z() - b.z())) < 0.01, "path chain and R1->R3 chain agree < 1 cm");
        check((a - robot_in_ref_truth).head<2>().norm() < 0.01, "path chain = the truth < 1 cm");
        const int i3 = index_of(q, "r3_room_3");
        check(i3 >= 0 and q.rooms[i3].chain_ok, "R3 has its own chain");
        // R3's own chain must not be the path's last passage: it is passage_1 (one seam), cov_T = one seam
        const Vec3 t3 = mul(inv(rooms[0].in_world), rooms[2].in_world);
        check(i3 >= 0 and (Vec3(q.rooms[i3].T_ref_k.translation().x(), q.rooms[i3].T_ref_k.translation().y(), 0) - Vec3(t3.x(), t3.y(), 0)).norm() < 0.01,
              "T_ref_k(R3) = the exact seam R1<-R3");
        float log_new = 0.f;
        const auto ev = ltsm::metric_evidence(q, log_new);
        std::printf("  log_inside:"); for (std::size_t i = 0; i < ev.size(); ++i) std::printf(" %s %.4g", q.rooms[i].mem_room.c_str(), ev[i].log_inside);
        std::printf("   log_new %.4g\n", log_new);
        check(q.rooms.size() == 3 and q.robot_ok, "DOOR_2 has no passage in memory, yet the robot chain is available");
        check(i3 >= 0 and ev[i3].available and ev[i3].log_inside > std::log(0.99f), "log_inside(R3) > log 0.99");
        check(i3 >= 0 and argmax(ev) == std::size_t(i3), "R3 is the argmax");
        check(ev[i3].pose_in_k.size() == 1 and (ev[i3].pose_in_k[0].mu - robot_in_r3.cast<float>()).head<2>().norm() < 0.02f,
              "pose_in_k(R3) is the robot's pose in R3");
    }

    std::printf("missing_fresh_seam_is_unavailable\n");
    {
        bool threw = false;
        ltsm::MetricQuery q;
        std::vector<rc::recog::MetricEvidence> ev;
        float log_new = 1.f;
        try { q = ltsm::build_metric_query(*mem, path, "r1_room_1", std::nullopt, {robot_in_prov, Codo}); ev = ltsm::metric_evidence(q, log_new); }
        catch (...) { threw = true; }
        bool none = true;
        for (const auto &e : ev) none = none and not e.available;
        check(not threw and not q.robot_ok and none and log_new == 0.f, "no fresh seam: robot_ok=false, nothing available, log_new 0, no throw");
    }

    std::printf("seam_error_widens_not_flips\n");
    {
        float log_new0 = 0.f;
        const auto q0 = ltsm::build_metric_query(*mem, path, "r1_room_1", fresh, {robot_in_prov, Codo});
        (void)ltsm::metric_evidence(q0, log_new0);
        // a seam that is 0.1 rad off AND says so: the R2 face of passage_0 turned by 0.1 rad about its own origin, with a covariance to match
        const Vec3 good = mul(inv(rooms[1].in_world), Vec3(2.0, 0.0, M_PI / 2));
        const Vec3 bad(good.x(), good.y(), good.z() + 0.1);
        Mat3 Cbad = Mat3::Zero();  Cbad.diagonal() << 0.1 * 0.1, 0.1 * 0.1, 0.1 * 0.1;
        put_edge(*mem, rooms[1].id, face_ids[0][1], bad, Cbad);
        const auto q = ltsm::build_metric_query(*mem, path, "r1_room_1", fresh, {robot_in_prov, Codo});
        float log_new = 0.f;
        const auto ev = ltsm::metric_evidence(q, log_new);
        const int i3 = index_of(q, "r3_room_3");
        std::printf("  trace(cov) %.5f -> %.5f   log_new %.4g\n", q0.robot_in_ref.cov.trace(), q.robot_in_ref.cov.trace(), log_new);
        check(q.robot_ok and q.robot_in_ref.cov.trace() > 5.0 * q0.robot_in_ref.cov.trace(), "robot-chain cov rises");
        check((q.robot_in_ref.pose - q0.robot_in_ref.pose).head<2>().norm() > 0.1, "the pose is displaced by the seam error");
        check(i3 >= 0 and argmax(ev) == std::size_t(i3), "R3 stays the argmax");
        check(log_new < std::log(0.05f), "log_new stays below log 0.05");
        put_edge(*mem, rooms[1].id, face_ids[0][1], good, Codo * 0.0);   // restore (zero cov = unstated -> prior)
    }

    std::printf("outside_everything_is_new\n");
    {
        const Vec3 far(-9.0, -9.0, 0.0);    // in the reference frame, >5 m from every outline
        const auto q = ltsm::build_metric_query(*mem, {}, "r1_room_1", ltsm::ChainPose{}, {far, Codo});
        float log_new = 0.f;
        const auto ev = ltsm::metric_evidence(q, log_new);
        std::printf("  log_new %.4g  log_inside:", log_new); for (const auto &e : ev) std::printf(" %.4g", e.log_inside); std::printf("\n");
        check(std::abs(log_new) < 1e-6, "log_new ~ 0");
        bool all = true;
        for (const auto &e : ev) all = all and e.available and e.log_inside < std::log(1e-6f);
        check(all, "every log_inside < log 1e-6");
    }

    // ── THE OUTLINE'S SOFT EXTENT (2026-11-12): edge support read from the remembered walls' obj_exist_prob ──
    std::printf("edge_support_from_wall_existence\n");
    {
        // r3's floor and its four walls, built from its outline as room_concept does (one wall per edge, mid + yaw + width_m)
        const Room &r3 = rooms[2];
        DSR::Node fl; fl.type("floor"); fl.name("r3_room_3_floor");
        mem->add_or_modify_attrib_local<parent_att>(fl, r3.id);
        const auto fid = mem->insert_node(fl).value();
        put_edge(*mem, r3.id, fid, Vec3::Zero());
        std::vector<Eigen::Vector2d> local;
        for (const auto &w : r3.world_poly) { const Vec3 l = mul(inv(r3.in_world), Vec3(w.x(), w.y(), 0.0)); local.emplace_back(l.x(), l.y()); }
        std::vector<std::uint64_t> wall_ids;
        for (std::size_t i = 0; i < local.size(); ++i)
        {
            // wall_i on edge (i+2) % 4: the wall order is NOT the edge order, the match is by geometry
            const std::size_t e = (i + 2) % local.size();
            const Eigen::Vector2d a = local[e], b = local[(e + 1) % local.size()], m = 0.5 * (a + b);
            DSR::Node w; w.type("wall"); w.name("r3_room_3_wall_" + std::to_string(i));
            mem->add_or_modify_attrib_local<width_m_att>(w, float((b - a).norm()));
            mem->add_or_modify_attrib_local<parent_att>(w, fid);
            if (i == 0) mem->add_or_modify_attrib_local<obj_exist_prob_att>(w, 0.f);    // edge 2: no wall was seen there
            if (i == 1) mem->add_or_modify_attrib_local<obj_exist_prob_att>(w, 0.7f);   // edge 3: partly seen
            const auto wid = mem->insert_node(w).value();
            wall_ids.push_back(wid);
            put_edge(*mem, fid, wid, Vec3(m.x(), m.y(), std::atan2(b.y() - a.y(), b.x() - a.x())));
        }
        const auto q = ltsm::build_metric_query(*mem, path, "r1_room_1", fresh, {robot_in_prov, Codo});
        const int i3 = index_of(q, "r3_room_3");
        const auto &s = q.rooms[i3].edge_support;
        std::printf("  r3 edge_support:"); for (const double v : s) std::printf(" %.2f", v); std::printf("\n");
        check(s.size() == 4 and s[0] == 1.0 and s[1] == 1.0 and std::abs(s[2]) < 1e-9 and std::abs(s[3] - 0.7) < 1e-6,
              "support = the matched wall's obj_exist_prob; unstated = 1");
        // r3 holds one door face, f1_r3_room_3 at W (0,2): on its outline's edge 0, (-2,2) -> (6,2)
        const auto &dr = q.rooms[i3].edge_door;
        check(dr.size() == 4 and dr[0] and not dr[1] and not dr[2] and not dr[3], "edge_door: the face of passage_1 marks r3's edge 0 only");
        const int i1 = index_of(q, "r1_room_1");
        check(i1 >= 0 and q.rooms[i1].edge_support.size() == 4
              and std::ranges::all_of(q.rooms[i1].edge_support, [](double v) { return v == 1.0; }), "a room with no walls: every edge 1");
        // restore: the later scenes read r3's outline as fully seen
        for (const auto wid : wall_ids)
            if (auto w = mem->get_node(wid); w.has_value()) { mem->add_or_modify_attrib_local<obj_exist_prob_att>(w.value(), 1.f); mem->update_node(w.value()); }
    }

    std::printf("staged_twin_is_not_a_candidate\n");
    {
        DSR::Node n; n.type("room"); n.name("r9_room_9");
        mem->add_or_modify_attrib_local<object_subtype_att>(n, std::string("staged"));
        const auto id = mem->insert_node(n).value();
        put_edge(*mem, root, id, Vec3::Zero());
        const auto q = ltsm::build_metric_query(*mem, path, "r1_room_1", fresh, {robot_in_prov, Codo});
        check(index_of(q, "r9_room_9") < 0 and q.rooms.size() == 3, "the staged twin is absent from rooms");
    }

    std::printf("missing_face_marks_unavailable\n");
    {
        mem->delete_node(face_ids[1][1]);    // R3's face of passage_1 (R1-R3)
        bool threw = false;
        ltsm::MetricQuery q;
        std::vector<rc::recog::MetricEvidence> ev;
        float log_new = 0.f;
        try { q = ltsm::build_metric_query(*mem, path, "r1_room_1", fresh, {robot_in_prov, Codo}); ev = ltsm::metric_evidence(q, log_new); }
        catch (...) { threw = true; }
        check(not threw, "no throw");
        const int i3 = index_of(q, "r3_room_3");
        check(i3 >= 0 and not q.rooms[i3].chain_ok and not ev[i3].available, "R3: chain_ok=false, available=false");
        const int i1 = index_of(q, "r1_room_1"), i2 = index_of(q, "r2_room_2");
        check(i1 >= 0 and i2 >= 0 and ev[i1].available and ev[i2].available, "R1 and R2 stay available");
    }

    // ── clean53 (2026-11-11), the final recognition episode, replayed offline from its log + memory snapshot ──────────
    // ref = r2_room_2 (the bath). Logged: robot_in_ref (+2.898,-2.816,-1.589) with its covariance; T_ref_k(r3) (+3.308,
    // -1.698,-1.568), T_ref_k(r1) (-0.676,-4.522,-1.205). Outlines = ltsm_20261111_231727_274.json, in each room's frame.
    // Truth (diag_recognition.md): the robot is in r3 (the big room), 0.76 m past r3's remembered outline, in a corridor r3
    // closed with walls nobody saw: r3_room_3_wall_0 / wall_20 / wall_2 = outline edges 0, 1, 2 (matched by length:
    // 0.88, 1.90, 3.35 m). The bath is just across a seen partition. Logged with the HARD inside test: r1 -16.57,
    // r2 -4.04, r3 -6.22: the bath wins.
    {
        auto place = [](const Vec3 &T, const std::vector<Eigen::Vector2d> &poly) {
            std::vector<Eigen::Vector2d> out;
            for (const auto &v : poly) { const Vec3 p = mul(T, Vec3(v.x(), v.y(), 0.0)); out.emplace_back(p.x(), p.y()); }
            return out; };
        const std::vector<Eigen::Vector2d> r1_local{{1.0480, -2.5354}, {3.2244, 3.3261}, {1.2382, 4.0636}, {0.9189, 3.2037}, {0.6703, 3.2960}, {0.2591, 2.1885}, {-3.6771, 3.6500}, {-5.1230, -0.2440}};
        const std::vector<Eigen::Vector2d> r2_local{{3.4392, 3.5476}, {-2.7812, 3.4862}, {-2.7485, 0.1750}, {1.5861, 0.2178}, {1.6107, -2.2719}, {3.4964, -2.2533}};
        const std::vector<Eigen::Vector2d> r3_local{{0.6051, 2.2650}, {1.4882, 2.2650}, {1.4882, 0.3649}, {4.8352, 0.3650}, {4.8351, 4.4150}, {4.8379, 4.4150}, {4.8379, 5.6650}, {4.8124, 5.6650}, {4.8123, 7.3440}, {1.4551, 7.3439}, {1.4551, 6.5650}, {0.8051, 6.5649}, {0.8051, 4.9150}, {0.5846, 4.9150}, {0.5845, 6.6008}, {-0.3943, 6.6007}, {-0.3943, 7.3439}, {-3.4858, 7.3438}, {-3.4858, 5.8650}, {-3.5449, 5.8650}, {-3.5448, 0.1861}, {0.6052, 0.1862}};
        const Vec3 T1(-0.676, -4.522, -1.205), T2 = Vec3::Zero(), T3(+3.308, -1.698, -1.568);
        // cov_T as build_metric_query composes it: the two faces of the room's own passage, each with its RT edge's stated
        // (x, y, yaw) covariance, from the same snapshot (passage_1 = r2_door_3 | r3_door_4; passage_0 = r2_door_2 | r1_door_1)
        auto face = [](double x, double y, double th, double cxx) { Mat3 C = Mat3::Zero(); C.diagonal() << cxx, 0.0001, 0.0003; return Measured{Vec3(x, y, th), C}; };
        const Mat3 Cseam3 = compose_cov(face(3.4905, -1.6513, 1.5807, 0.059891), inverse_cov(face(-0.046, 0.1822, 0.0066, 0.037205))).cov;
        const Mat3 Cseam1 = compose_cov(face(0.0312, 0.2025, 0.0099, 0.008615), inverse_cov(face(-4.1602, 2.3491, -1.9263, 0.004636))).cov;
        auto make_room = [&](const char *name, std::uint64_t id, const Vec3 &T, const std::vector<Eigen::Vector2d> &local, const Mat3 &C) {
            ltsm::KnownRoomMetric k;
            k.room_id = id; k.mem_room = name; k.chain_ok = true;
            k.T_ref_k = Eigen::Affine2d::Identity(); k.T_ref_k.rotate(T.z()); k.T_ref_k.pretranslate(Eigen::Vector2d(T.x(), T.y()));
            k.cov_T = C; k.polygon_in_ref = place(T, local);
            double a2 = 0.0;
            for (std::size_t i = 0; i < local.size(); ++i) a2 += local[i].x() * local[(i + 1) % local.size()].y() - local[(i + 1) % local.size()].x() * local[i].y();
            k.area = 0.5 * std::abs(a2);
            return k; };
        auto scene = [&](const Vec3 &robot, const Mat3 &Crob, bool r3_corridor_unseen) {
            ltsm::MetricQuery q;
            q.reference_room = "r2_room_2"; q.robot_ok = true; q.robot_in_ref = {robot, Crob};
            q.rooms.push_back(make_room("r1_room_1", 1, T1, r1_local, Cseam1));
            q.rooms.push_back(make_room("r2_room_2", 2, T2, r2_local, Mat3::Zero()));
            q.rooms.push_back(make_room("r3_room_3", 3, T3, r3_local, Cseam3));
            if (r3_corridor_unseen)
            {
                // P(real wall) per edge, computed by room_concept's model (wall_support.h) from clean53's own raster
                // (refine_input.bin frames 12 / 26 / 54, room_concept/tools/wall_support_test). The corridor's two phantoms
                // read 0.061 / 0.001; edge 2 is MIXED (2.2 m of the big room's real south wall + the 0.95 m corridor mouth)
                // and reads 1.0 as a whole; r1's edge 2, seen only edge-on, reads ~ its prior.
                q.rooms[2].edge_support = {0.061, 0.001, 1.000, 1.000, 0.561, 0.998, 0.456, 0.973, 1.000, 0.706, 0.924,
                                           0.004, 0.491, 1.000, 0.993, 0.904, 1.000, 0.996, 0.621, 1.000, 1.000, 1.000};
                q.rooms[0].edge_support = {1.0, 1.0, 0.603, 0.778, 0.971, 1.0, 1.0, 1.0};
                // the snapshot's doors, each on the edge nearest it: r1_door_9 (DOOR_1, seen not crossed) e5, r1_door_1 e6;
                // r2_door_2 e2, r2_door_3 e5; r3_door_4 e20
                q.rooms[0].edge_door.assign(r1_local.size(), false); q.rooms[0].edge_door[5] = q.rooms[0].edge_door[6] = true;
                q.rooms[1].edge_door.assign(r2_local.size(), false); q.rooms[1].edge_door[2] = q.rooms[1].edge_door[5] = true;
                q.rooms[2].edge_door.assign(r3_local.size(), false); q.rooms[2].edge_door[20] = true;
            }
            return q; };
        Mat3 Crob;
        Crob << 0.03512, -0.01847, -0.003853,  -0.01847, 0.06232, 0.003148,  -0.003853, 0.003148, 0.001003;
        const Vec3 robot(2.898, -2.816, -1.589);
        auto run = [&](const ltsm::MetricQuery &q, float &log_new) {
            const auto ev = ltsm::metric_evidence(q, log_new);
            std::printf("   "); for (std::size_t i = 0; i < ev.size(); ++i) std::printf(" %s %+.2f", q.rooms[i].mem_room.c_str(), ev[i].log_inside);
            std::printf("  new %+.2f\n", log_new);
            return ev; };

        std::printf("clean53_replay_fully_seen_outline_reproduces_the_log\n");
        {
            float log_new = 0.f;
            const auto ev = run(scene(robot, Crob, false), log_new);
            // the soft extent at a seen wall adds a few centimetres of tail: the logged hard values to within a fraction of a nat
            check(std::abs(ev[1].log_inside - (-4.04f)) < 0.5f and std::abs(ev[2].log_inside - (-6.22f)) < 0.5f,
                  "every edge seen: r2 ~ -4.04, r3 ~ -6.22 as logged (the replay is the episode)");
            check(ev[1].log_inside > ev[2].log_inside, "...and, as live, the bath outscores r3");
        }
        std::printf("clean53_computed_supports_recognise_r3\n");
        {
            float log_new = 0.f;
            const auto ev = run(scene(robot, Crob, true), log_new);
            check(ev[2].log_inside > ev[1].log_inside and ev[2].log_inside > ev[0].log_inside, "r3 outscores the bath and r1");
            check(ev[2].log_inside > log_new, "r3 outscores NEW");
            check(std::abs(ev[1].log_inside - (-4.04f)) < 0.5f, "the bath is untouched: its partition is a seen wall");
        }
        std::printf("clean53_robot_in_the_bath_stays_in_the_bath\n");
        {
            float log_new = 0.f;
            const auto ev = run(scene(Vec3(0.3, 2.0, 0.0), Crob, true), log_new);   // inside r2, 2-3 m from r3's open edges
            check(ev[1].log_inside > std::log(0.99f), "inside r2: log_inside(r2) ~ 0");
            check(ev[1].log_inside > ev[2].log_inside + 1.0f, "r2 beats r3's unseen extension by > 1 nat");
        }
        std::printf("seen_wall_keeps_the_robot_out\n");
        {
            ltsm::MetricQuery qk;
            qk.rooms.resize(1);
            auto &k = qk.rooms[0];
            k.chain_ok = true; k.polygon_in_ref = {{0, 0}, {4, 0}, {4, 4}, {0, 4}}; k.area = 16.0;
            k.edge_support = {1.0, 1.0, 1.0, 0.0};            // the edge x=0 (3 -> 0) was never seen as a wall
            const double behind_seen = ltsm::soft_inside(qk, 0, {4.3, 2.0}), behind_unseen = ltsm::soft_inside(qk, 0, {-0.3, 2.0});
            const double behind_seen_far = ltsm::soft_inside(qk, 0, {4.76, 2.0});
            std::printf("  P_in 0.30 m past a seen wall %.3g, 0.76 m %.3g; 0.30 m past an unseen edge %.3f\n", behind_seen, behind_seen_far, behind_unseen);
            check(behind_seen < 1e-4 and behind_seen_far < 1e-9, "past a SEEN wall the robot is outside (no false recognition)");
            check(behind_unseen > 0.9, "past an UNSEEN edge, 0.3 m in a 4 m room, it is still plausibly inside");
            check(ltsm::soft_inside(qk, 0, {2.0, 2.0}) == 1.0, "inside the outline: 1");
            // a robot in the bath-like pocket behind a seen wall, on the outward side of the unseen edge's LINE but reached
            // from it only through the room itself: U-shaped room, open edge at the bottom of the U's notch
            ltsm::MetricQuery qu;
            qu.rooms.resize(1);
            auto &u = qu.rooms[0];
            u.chain_ok = true; u.area = 12.0;
            u.polygon_in_ref = {{0, 0}, {4, 0}, {4, 4}, {3, 4}, {3, 1}, {1, 1}, {1, 4}, {0, 4}};
            u.edge_support.assign(8, 1.0);
            u.edge_support[4] = 0.0;                          // (3,1) -> (1,1): the notch's floor, unseen
            // (3.5, 4.5) lies beyond the unseen edge's line, but the straight way from that edge to it crosses the U's
            // right arm (its top wall (4,4)-(3,4)): the extension cannot pass through the room's own wall
            const double in_notch = ltsm::soft_inside(qu, 0, {2.0, 1.5}), behind_arm = ltsm::soft_inside(qu, 0, {3.5, 4.5});
            std::printf("  U room: 0.5 m into the notch past its unseen floor %.3f; behind the right arm's seen top wall %.3g\n", in_notch, behind_arm);
            check(in_notch > 0.5, "the notch beyond an unseen edge is plausibly the room");
            check(behind_arm < 1e-6, "behind a seen wall: the unseen edge cannot reach through the room's own wall");
        }
    }

    // ── A DOOR EDGE IS A BOUNDARY (2026-11-12): beyond a remembered door lies the passage's other room ──────────────────
    std::printf("door_edge_is_a_boundary\n");
    {
        ltsm::MetricQuery qd;
        qd.rooms.resize(2);
        auto &k = qd.rooms[0];
        k.chain_ok = true; k.polygon_in_ref = {{0, 0}, {4, 0}, {4, 4}, {0, 4}}; k.area = 16.0;
        k.edge_support = {1.0, 1.0, 1.0, 0.0};          // edge x=0 never seen as a wall...
        const double open_before = ltsm::soft_inside(qd, 0, {-0.5, 2.0});
        k.edge_door = {false, false, false, true};      // ...but it holds one of k's doors
        const double open_after = ltsm::soft_inside(qd, 0, {-0.5, 2.0});
        // another room whose unseen edge faces k's door edge from the far side: its extension must stop there too
        auto &o = qd.rooms[1];
        o.chain_ok = true; o.polygon_in_ref = {{-4, 0}, {-2, 0}, {-2, 4}, {-4, 4}}; o.area = 8.0;
        o.edge_support = {1.0, 0.0, 1.0, 1.0};          // its edge x=-2 unseen
        const double through_door = ltsm::soft_inside(qd, 1, {1.0, 2.0});   // inside k, reached from o through k's door edge
        std::printf("  0.5 m past an unseen edge %.3f; past the same edge holding a door %.3g; o reaching into k through k's door edge %.3g\n",
                    open_before, open_after, through_door);
        check(open_before > 0.8, "unseen edge: the room plausibly continues");
        check(open_after < 1e-6, "door edge: it does not");
        check(through_door < 1e-6, "a door edge blocks another room's extension like a seen wall");
    }

    // ── clean56 (2026-11-12): the robot started in the big room (r1_room_1, 59.7 m2) and crossed DOOR_0 into the bath; the
    //    metric expert read the bath as INSIDE the big room (P_in 0.94, log_inside -0.06; posterior 0.565 -> proposed and
    //    ACCEPTED). Outline + walls + doors = ltsm_20261112_093110_768.json (the memory after the eviction, which the
    //    proposal at l.267 was computed on: replayed posterior 0.566 vs 0.565 logged); robot from ltsm_agent.log l.252.
    {
        const std::vector<Eigen::Vector2d> big{{4.0123, 4.0025}, {2.5479, 5.0023}, {1.8446, 3.9749}, {1.3904, 3.1748}, {-0.6447, 4.5642}, {-2.9410, 1.2007}, {-2.9604, 1.2139}, {-3.9459, -0.2295}, {-3.7241, -0.3810}, {-4.3647, -1.3194}, {-1.8185, -3.0577}, {-1.4081, -2.4566}, {-0.8813, -2.8163}, {0.0218, -1.4936}, {0.1593, -1.5874}, {-0.7495, -2.9185}, {0.0177, -3.4423}, {-0.4051, -4.0616}, {2.2443, -5.8704}, {2.8033, -5.0515}, {2.6030, -4.9147}, {2.7447, -4.7071}, {3.0100, -4.8882}, {6.3269, -0.0298}, {2.8704, 2.3300}};
        auto scene56 = [&](bool new_model) {
            ltsm::MetricQuery q;
            q.reference_room = "r1_room_1"; q.robot_ok = true;
            Mat3 C;
            C << 0.004379, -0.002253, -0.0002195,  -0.002253, 0.00256, 9.508e-05,  -0.0002195, 9.508e-05, 0.0004129;
            q.robot_in_ref = {Vec3(3.652, 2.514, -0.487), C};
            ltsm::KnownRoomMetric k;
            k.room_id = 1; k.mem_room = "r1_room_1"; k.chain_ok = true; k.polygon_in_ref = big; k.area = 59.7;
            if (new_model)
            {   // room_concept's P(real wall) from clean56's own raster (refine_input.bin frame 3) + the doors on edges 0, 23
                k.edge_support = {0.9999, 0.9687, 0.0360, 1.0000, 1.0000, 1.0000, 0.9999, 0.8399, 0.4276, 1.0000, 0.9630, 0.9746,
                                  0.0009, 0.3591, 0.9695, 0.9961, 0.5586, 1.0000, 0.9904, 0.6135, 0.8473, 0.8599, 1.0000, 1.0000, 0.9713};
                k.edge_door.assign(big.size(), false); k.edge_door[0] = k.edge_door[23] = true;
            }
            else   // the remembered walls' "fraction seen" as live (matched to edges by edge_support_of)
                k.edge_support = {0.7867, 0.0400, 0.0223, 0.8769, 0.8997, 1.0000, 0.9031, 0.6575, 0.2278, 0.7618, 0.1182, 0.8584,
                                  0.0000, 0.2037, 0.7399, 0.9037, 0.0335, 0.8533, 0.7069, 0.2000, 0.9213, 0.6010, 0.6861, 0.8425, 0.2562};
            q.rooms.push_back(k);
            return q; };
        std::printf("clean56_old_supports_reproduce_the_false_recognition\n");
        {
            float log_new = 0.f;
            const auto ev = ltsm::metric_evidence(scene56(false), log_new);
            std::printf("    r1 %+.2f new %+.2f\n", ev[0].log_inside, log_new);
            check(std::abs(ev[0].log_inside - (-0.21f)) < 0.05f and ev[0].log_inside > log_new, "as live: the bath reads inside the big room (-0.21, as the recog_extent_replay of the live memory)");
        }
        std::printf("clean56_bath_is_not_the_big_room\n");
        {
            float log_new = 0.f;
            const auto ev = ltsm::metric_evidence(scene56(true), log_new);
            std::printf("    r1 %+.2f new %+.2f\n", ev[0].log_inside, log_new);
            check(ev[0].log_inside < log_new - 2.0f, "the bath is NEW by > 2 nats");
        }
    }

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "all ok", failures);
    std::fflush(stdout);
    std::_Exit(failures ? 1 : 0);
}
