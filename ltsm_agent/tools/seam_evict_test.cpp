// Task S INTEGRATION: drives RoomEviction::evict (detect -> evict) on a real in-process DSR graph with the gold1
// garbage robot-chain seam and asserts the STORED seam is the aperture seam. The pure-math test cannot see whether the
// fusion is wired in (review C1: it was gated on a flag nothing set).
//   run from the component root:  ./bin/seam_evict_test sane | garbage | reentry | reentry-garbage | established | flagged | nocov
// Scene = etc/stage_two_rooms.json, re-posed so room_1's frame IS the crossed aperture (a proto's frame): door_1 sits on
// wall_1 (the +x wall) at (3, 0.5), its x axis = the wall tangent (+y); the robot stands 0.6 m past it.
#include "../src/room_eviction.h"
#include "../src/seam_fusion.h"
#include <QCoreApplication>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace ltsm::seam;
static int failures = 0;
static void check(bool ok, const std::string &what) { if (!ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str()); }

static void put_edge(DSR::DSRGraph &G, std::uint64_t from, std::uint64_t to, const Vec3 &p, std::optional<Mat3> cov)
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

int main(int argc, char **argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    QCoreApplication app(argc, argv);
    const std::string mode = argc > 1 ? argv[1] : "garbage";
    auto live = std::make_shared<DSR::DSRGraph>("seam_live", 191, "etc/stage_two_rooms.json", true, 41);
    auto mem  = std::make_shared<DSR::DSRGraph>("seam_mem", 192, "ltsm_root.json", true, 42);

    const Vec3 door_in_old(3.0, 0.5, M_PI / 2);
    const Vec3 robot_in_old(3.6, 0.5, 0.0);
    const Vec3 proto = proto_in_old(door_in_old, robot_in_old.head<2>());
    const Vec3 truth = inverse(proto);                        // T(room_1 <- room_0)
    Mat3 Cr = Mat3::Zero();  Cr.diagonal() << 0.05 * 0.05, 0.05 * 0.05, 0.02 * 0.02;
    Mat3 Cd = Mat3::Zero();  Cd.diagonal() << 0.03 * 0.03, 0.03 * 0.03, 0.01 * 0.01;

    // robot -> room edges hold T(robot <- room) = inverse(robot pose in room)
    const Vec3 robot_in_new = [&] { Vec3 q; q << (rot(truth.z()) * robot_in_old.head<2>() + truth.head<2>()), wrap(robot_in_old.z() + truth.z()); return q; }();
    const bool cov = mode != "nocov";
    put_edge(*live, 200, 300, inverse(robot_in_old), cov ? std::optional{Cr} : std::nullopt);
    // established: a sane chain, but room_1's frame is ARBITRARY (not the aperture): shift it by (1.3, -0.7, 0.4)
    const Vec3 arb(1.3, -0.7, 0.4);
    const Vec3 robot_in_arb = [&] { Vec3 q; q << (rot(arb.z()) * robot_in_new.head<2>() + arb.head<2>()), wrap(robot_in_new.z() + arb.z()); return q; }();
    put_edge(*live, 200, 400, mode == "sane" ? inverse(robot_in_new) : (mode == "established" or mode == "reentry") ? inverse(robot_in_arb) : Vec3(-14262.68, 1667.96, -1.1285), cov ? std::optional{Cr} : std::nullopt);
    // door_1 hangs from wall_1: pose in the wall frame as the fixture authored it; give that edge the door's covariance
    const auto d1 = live->get_node("door_1").value();
    if (cov) put_edge(*live, 311, d1.id(), [&] { const auto e = live->get_edge(311, d1.id(), "RT").value();
                                                  const auto T = live->get_rt_api()->get_edge_RT_as_rtmat(e).value();
                                                  return Vec3(T.translation().x(), T.translation().y(), 0.0); }(), Cd);

    // gold1: room_1 is still a proto (self-edge). "garbage_flag"/"flagged": no proto edge, crossing_born set on the candidate.
    if (mode == "garbage" or mode == "sane" or mode == "reentry" or mode == "reentry-garbage")
        std::printf("  proto self-edge inserted: %d\n", int(live->insert_or_assign_edge(DSR::Edge::create<proto_edge_type>(400, 400))));
    ltsm::RoomEviction ev(live, mem);
    const bool reentry = mode == "reentry" or mode == "reentry-garbage";
    // the seam memory already holds for this passage: an OLDER estimate of the sane chain, off by (0.03, -0.02, 0.01)
    const Vec3 chain_arb = chain_seam({inverse(robot_in_arb), Mat3::Zero()}, {inverse(robot_in_old), Mat3::Zero()}).pose;
    const Vec3 remembered_seam = chain_arb + Vec3(0.03, -0.02, 0.01);
    if (reentry)
    {   // memory ALREADY holds both rooms and the passage between them (an earlier visit): twins r7_room_1 / r6_room_0,
        // one face of the hole in each, a passage matching both. T(new<-old) read through the faces = remembered_seam.
        for (auto [id, rid] : {std::pair{400, 7}, std::pair{300, 6}})
        {
            auto rn = live->get_node(id).value();
            live->add_or_modify_attrib_local<room_id_att>(rn, std::uint64_t(rid));
            live->update_node(rn);
        }
        const auto root = mem->get_node("root").value().id();
        auto add_room = [&](const std::string &name, std::uint64_t rid) {
            DSR::Node t; t.type("room"); t.name(name);
            mem->add_or_modify_attrib_local<room_id_att>(t, rid);
            const auto id = mem->insert_node(t).value();
            put_edge(*mem, root, id, Vec3::Zero(), std::nullopt);
            return id; };
        auto add_face = [&](const std::string &name, std::uint64_t room, const Vec3 &pose) {
            DSR::Node f; f.type("object"); f.name(name);
            mem->add_or_modify_attrib_local<object_subtype_att>(f, std::string("door"));
            const auto id = mem->insert_node(f).value();
            put_edge(*mem, room, id, pose, std::nullopt);
            return id; };
        const auto r_new = add_room("r7_room_1", 7), r_old = add_room("r6_room_0", 6);
        const Vec3 f_old = door_in_old;
        const Vec3 f_new = [&] { Vec3 q; q << (rot(remembered_seam.z()) * f_old.head<2>() + remembered_seam.head<2>()), wrap(f_old.z() + remembered_seam.z()); return q; }();
        const auto face_old = add_face("r6_door_1", r_old, f_old), face_new = add_face("r7_door_1", r_new, f_new);
        DSR::Node pw; pw.type("metaconcept"); pw.name("passage_0");
        mem->add_or_modify_attrib_local<object_subtype_att>(pw, std::string("passage"));
        const auto pid = mem->insert_node(pw).value();
        mem->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, face_old));
        mem->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, face_new));
    }
    // a proto-room is not offered by detect(); the live path is promotion_candidate(reading)
    std::optional<ltsm::Candidate> cand;
    if (mode == "garbage" or mode == "sane" or mode == "reentry" or mode == "reentry-garbage")
    {
        ev.ensure_current_edge();
        ltsm::PromotionReading r; r.proto_id = 400; r.proto_name = "room_1"; r.valid = true; r.p_inside = 1.0;
        cand = ev.promotion_candidate(r);
    }
    else
        cand = ev.detect();
    check(cand.has_value(), "a room_0 -> room_1 candidate");
    if (not cand) return 1;
    auto c = cand.value();
    if (mode == "flagged") c.crossing_born = true;
    const auto out = ev.evict(c);
    check(out.ok, "evict ok (" + out.reason + ")");
    const Vec3 stored(out.seam.translation().x(), out.seam.translation().y(),
                      std::atan2(out.seam.rotation()(1, 0), out.seam.rotation()(0, 0)));
    std::printf("  stored seam (%+.3f,%+.3f,%+.4f)  aperture truth (%+.3f,%+.3f,%+.4f)\n", stored.x(), stored.y(), stored.z(),
                truth.x(), truth.y(), truth.z());
    const auto close = [&](const Vec3 &a, const Vec3 &b, double txy, double tyaw) {
        return (a - b).head<2>().norm() < txy and std::abs(wrap(a.z() - b.z())) < tyaw; };
    // What the aperture alone gives, with the recess nuisance's MEAN shift along the normal (model constant 0.25 m):
    // a chain with weight ~0 must leave EXACTLY this (tolerance 1 cm: numerical + float edge storage), not a loose bound.
    const Vec3 ap_expected = aperture_seam(door_in_old, Cd, robot_in_old.head<2>(), 0.25, 0.25).pose;
    if (mode == "established")
        check(close(stored, chain_arb, 1e-3, 1e-3) and (stored - truth).head<2>().norm() > 0.5,
              "established, arbitrary frame, sane chain: stored seam = the chain, NOT the aperture seam");
    else if (mode == "reentry")
    {   // sane chain vs the remembered seam (3.6 cm apart, inside sigma 0.1): fused, i.e. between the two, refining it
        const double dc = (stored - chain_arb).head<2>().norm(), dm = (stored - remembered_seam).head<2>().norm();
        const double dcm = (remembered_seam - chain_arb).head<2>().norm();
        check(dc > 1e-3 and dm > 1e-3 and dc < dcm and dm < dcm,
              "re-entry, sane chain: stored seam is closer to each of the chain and the remembered seam than they are to each other (fused, neither replaced)");
    }
    else if (mode == "reentry-garbage")
        check(close(stored, remembered_seam, 0.01, 0.005),
              "re-entry, gold1 garbage chain: the REMEMBERED seam is kept (chain weight ~0)");
    else if (mode == "nocov")
        check(std::abs(stored.x()) > 1000, "no covariance: the robot chain is kept (pre-existing behaviour, warned)");
    else if (mode == "sane")
    {   // chain = mouth at the door node (recess 0), aperture shifted 0.25 m: sigma_chain 0.05 vs 0.25 along the normal =>
        // the fused seam moves ~0.01 m off the chain; the bound is the chain's own sigma (0.05 m), state it.
        check(close(stored, truth, 0.05, 0.02), "sane chain: stored seam within the chain sigma (0.05 m) of the chain");
    }
    else
        check(close(stored, ap_expected, 0.01, 0.005),
              "garbage chain: stored seam = the aperture seam incl. the recess mean shift (1 cm)");
    std::printf("%s [%s] (%d failures)\n", failures ? "FAILED" : "PASSED", mode.c_str(), failures);
    std::fflush(stdout);
    std::_Exit(failures ? 1 : 0);
}
