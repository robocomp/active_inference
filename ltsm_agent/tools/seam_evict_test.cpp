// Task S INTEGRATION: drives RoomEviction::evict (detect -> evict) on a real in-process DSR graph with the gold1
// garbage robot-chain seam and asserts the STORED seam is the aperture seam. The pure-math test cannot see whether the
// fusion is wired in (review C1: it was gated on a flag nothing set).
//   run from the component root:  ./bin/seam_evict_test sane | garbage | nocov
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
    put_edge(*live, 200, 400, mode == "sane" ? inverse(robot_in_new) : Vec3(-14262.68, 1667.96, -1.1285), cov ? std::optional{Cr} : std::nullopt);
    // door_1 hangs from wall_1: pose in the wall frame as the fixture authored it; give that edge the door's covariance
    const auto d1 = live->get_node("door_1").value();
    if (cov) put_edge(*live, 311, d1.id(), [&] { const auto e = live->get_edge(311, d1.id(), "RT").value();
                                                  const auto T = live->get_rt_api()->get_edge_RT_as_rtmat(e).value();
                                                  return Vec3(T.translation().x(), T.translation().y(), 0.0); }(), Cd);

    ltsm::RoomEviction ev(live, mem);
    ev.seed_from_working_memory();
    const auto cand = ev.detect();
    check(cand.has_value(), "detect() yields the room_0 -> room_1 candidate");
    if (not cand) return 1;
    const auto out = ev.evict(cand.value());
    check(out.ok, "evict ok (" + out.reason + ")");
    const Vec3 stored(out.seam.translation().x(), out.seam.translation().y(),
                      std::atan2(out.seam.rotation()(1, 0), out.seam.rotation()(0, 0)));
    std::printf("  stored seam (%+.3f,%+.3f,%+.4f)  aperture truth (%+.3f,%+.3f,%+.4f)\n", stored.x(), stored.y(), stored.z(),
                truth.x(), truth.y(), truth.z());
    if (mode == "nocov")
        check(std::abs(stored.x()) > 1000, "no covariance: the robot chain is kept (pre-existing behaviour, warned)");
    else
        check((stored - truth).head<2>().norm() < 0.05 and std::abs(wrap(stored.z() - truth.z())) < 0.02,
              "stored seam = the aperture seam within sigma");
    std::printf("%s [%s] (%d failures)\n", failures ? "FAILED" : "PASSED", mode.c_str(), failures);
    std::fflush(stdout);
    std::_Exit(failures ? 1 : 0);
}
