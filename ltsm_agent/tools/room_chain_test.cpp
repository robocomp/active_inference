// room_chain: the path-taken seam chain against each known room's OWN chain (metric room-recognition evidence).
//   run from the component root:  ./bin/room_chain_test
// Scene = the live loop, in a common world frame W (each room has its OWN, arbitrarily posed frame, as in memory):
//   r1_room_1 [-2,2]x[-2,2]   r2_room_2 [2,6]x[-2,2]   r3_room_3 [-2,6]x[2,6]
//   passage_0 r1<->r2 at (2,0)   passage_1 r1<->r3 at (0,2)   passage_2 r2<->r3 at (4,2)  (= DOOR_2, the never-crossed door)
// The robot stands at (3,4) in W, i.e. inside r3, having come r1 -> r2 -> DOOR_2.
#include "../src/room_chain.h"
#include "../src/seam_fusion.h"
#include <QCoreApplication>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    add_passage(2, rooms[1], rooms[2], Vec3(4.0, 2.0, 0.0));

    const Vec3 robot_w(3.0, 4.0, 0.5);
    const Vec3 robot_in_r3 = mul(inv(rooms[2].in_world), robot_w);
    const Vec3 robot_in_ref_truth = mul(inv(rooms[0].in_world), robot_w);
    Mat3 Codo = Mat3::Zero();  Codo.diagonal() << 0.05 * 0.05, 0.05 * 0.05, 0.02 * 0.02;
    const std::vector<std::string> path{"passage_0", "passage_2"};

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
        const auto q = ltsm::build_metric_query(*mem, path, "r1_room_1", {robot_in_r3, Codo});
        const auto q_direct = ltsm::build_metric_query(*mem, {"passage_1"}, "r1_room_1", {robot_in_r3, Codo});
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
        check(i3 >= 0 and ev[i3].available and ev[i3].log_inside > std::log(0.99f), "log_inside(R3) > log 0.99");
        check(i3 >= 0 and argmax(ev) == std::size_t(i3), "R3 is the argmax");
        check(ev[i3].pose_in_k.size() == 1 and (ev[i3].pose_in_k[0].mu - robot_in_r3.cast<float>()).head<2>().norm() < 0.02f,
              "pose_in_k(R3) is the robot's pose in R3");
    }

    std::printf("seam_error_widens_not_flips\n");
    {
        float log_new0 = 0.f;
        const auto q0 = ltsm::build_metric_query(*mem, path, "r1_room_1", {robot_in_r3, Codo});
        (void)ltsm::metric_evidence(q0, log_new0);
        // a seam that is 0.1 rad off AND says so: the R2 face of passage_0 turned by 0.1 rad about its own origin, with a covariance to match
        const Vec3 good = mul(inv(rooms[1].in_world), Vec3(2.0, 0.0, M_PI / 2));
        const Vec3 bad(good.x(), good.y(), good.z() + 0.1);
        Mat3 Cbad = Mat3::Zero();  Cbad.diagonal() << 0.1 * 0.1, 0.1 * 0.1, 0.1 * 0.1;
        put_edge(*mem, rooms[1].id, face_ids[0][1], bad, Cbad);
        const auto q = ltsm::build_metric_query(*mem, path, "r1_room_1", {robot_in_r3, Codo});
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
        const auto q = ltsm::build_metric_query(*mem, {}, "r1_room_1", {far, Codo});
        float log_new = 0.f;
        const auto ev = ltsm::metric_evidence(q, log_new);
        std::printf("  log_new %.4g  log_inside:", log_new); for (const auto &e : ev) std::printf(" %.4g", e.log_inside); std::printf("\n");
        check(std::abs(log_new) < 1e-6, "log_new ~ 0");
        bool all = true;
        for (const auto &e : ev) all = all and e.available and e.log_inside < std::log(1e-6f);
        check(all, "every log_inside < log 1e-6");
    }

    std::printf("staged_twin_is_not_a_candidate\n");
    {
        DSR::Node n; n.type("room"); n.name("r9_room_9");
        mem->add_or_modify_attrib_local<object_subtype_att>(n, std::string("staged"));
        const auto id = mem->insert_node(n).value();
        put_edge(*mem, root, id, Vec3::Zero());
        const auto q = ltsm::build_metric_query(*mem, path, "r1_room_1", {robot_in_r3, Codo});
        check(index_of(q, "r9_room_9") < 0 and q.rooms.size() == 3, "the staged twin is absent from rooms");
    }

    std::printf("missing_face_marks_unavailable\n");
    {
        mem->delete_node(face_ids[1][1]);    // R3's face of passage_1 (R1-R3)
        bool threw = false;
        ltsm::MetricQuery q;
        std::vector<rc::recog::MetricEvidence> ev;
        float log_new = 0.f;
        try { q = ltsm::build_metric_query(*mem, path, "r1_room_1", {robot_in_r3, Codo}); ev = ltsm::metric_evidence(q, log_new); }
        catch (...) { threw = true; }
        check(not threw, "no throw");
        const int i3 = index_of(q, "r3_room_3");
        check(i3 >= 0 and not q.rooms[i3].chain_ok and not ev[i3].available, "R3: chain_ok=false, available=false");
        const int i1 = index_of(q, "r1_room_1"), i2 = index_of(q, "r2_room_2");
        check(i1 >= 0 and i2 >= 0 and ev[i1].available and ev[i2].available, "R1 and R2 stay available");
    }

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "all ok", failures);
    std::fflush(stdout);
    std::_Exit(failures ? 1 : 0);
}
