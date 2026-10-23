// Plan Task 1.3: a body->sensor RT edge that robot_concept COMMITS at runtime must reach room_concept's
// consumers without a restart.
//
// What this proves, against a REAL DSR graph (private DDS domain, shadow.json) and the SAME call the
// LiDAR reader makes (common/media_transport/lidar_plane_reader.cpp append_transformed):
//   1. helios path: get_transformation_matrix(body, helios, stamp_ms != 0, "RT", Nearest) -- a point's body-frame
//      bearing moves by 1 deg +- 0.05 on the very next query after body->helios yaw is raised by 1 deg.
//      A real stamp bypasses the InnerEigenAPI ts==0 cache, and a static edge (no rt_timestamps) answers
//      every stamp with its single block, so this is instantaneous (no Qt event loop is needed).
//   2. camera path: get_transformation_matrix(camera, body, ts == 0) on an instance owned by the MAIN thread
//      (what CameraIngestor::inner_ is) sees the change once the queued invalidation has run -- i.e. after
//      one pass of the event loop. This is the contract refresh_graph_extrinsic() relies on.
// Build+run (EXCLUDE_FROM_ALL target, or by hand):
//   make -C build lidar_mount_edge_live_test && ../bin/lidar_mount_edge_live_test
#include "../src/camera_ingestor.h"
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_rt_api.h>
#include <QCoreApplication>
#include <QThread>
#include <cmath>
#include <print>

namespace
{
int failures = 0;
void check(bool ok, int line, std::string_view what)
{
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)

double bearing_deg(const Eigen::Matrix4d& M, const Eigen::Vector3d& p_sensor)
{
    const Eigen::Vector4d q = M * Eigen::Vector4d(p_sensor.x(), p_sensor.y(), p_sensor.z(), 1.0);
    return std::atan2(q.y(), q.x()) * 180.0 / M_PI;
}
double wrap(double d) { while (d > 180) d -= 360; while (d < -180) d += 360; return d; }

// Commit +dyaw to the static body->`sensor` edge exactly as robot_concept does (insert_or_assign_edge_RT_static).
bool commit_yaw(DSR::DSRGraph& G, const std::string& sensor, float dyaw_rad)
{
    auto bn = G.get_node("body"); auto sn = G.get_node(sensor);
    if (not bn or not sn) return false;
    const auto e = G.get_edge(bn->id(), sn->id(), "RT");
    if (not e) return false;
    auto rot = G.get_attrib_by_name<rt_rotation_euler_xyz_att>(*e);
    auto tr  = G.get_attrib_by_name<rt_translation_att>(*e);
    if (not rot or not tr) return false;
    std::vector<float> r = rot->get(); r[2] += dyaw_rad;
    const std::vector<float> t = tr->get();
    G.get_rt_api()->insert_or_assign_edge_RT_static(*bn, sn->id(), t, r);
    return true;
}
}  // namespace

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);   // the MAIN thread: owner of the instance below, runs the queued slots
    auto G = std::make_shared<DSR::DSRGraph>("lidar_mount_edge_live_test", 990, std::string(SHADOW_JSON), true,
                                             static_cast<int8_t>(90), DSR::SignalMode::QT);   // QT: what the agents use
    const double dyaw_deg = 1.0;
    const Eigen::Vector3d wall(2.0, 0.5, 0.0);          // a known point in each sensor's own frame
    constexpr std::uint64_t stamp_ms = 1'700'000'000'000ull;   // a real sweep stamp: != 0 => no cache

    auto reader_inner = G->get_inner_eigen_api();        // plays LidarIngestor::inner_eigen_
    auto camera_inner = G->get_inner_eigen_api();        // plays CameraIngestor::inner_
    auto lidar_T = [&](const std::string& s)
    { return reader_inner->get_transformation_matrix("body", s, stamp_ms, "RT", DSR::RT_API::TimeQuery::Nearest); };
    auto cam_T = [&](const std::string& s)
    { return camera_inner->get_transformation_matrix(s, "body", 0, "RT", DSR::RT_API::TimeQuery::Nearest); };

    // Prime both (the camera instance caches its ts==0 answer; a stale cache is the failure being hunted).
    const auto h0 = lidar_T("helios"); const auto c0 = cam_T("ricoh");
    CHECK(h0.has_value()); CHECK(c0.has_value());
    if (not h0 or not c0) { std::println("graph incomplete"); return 2; }
    const double b0 = bearing_deg(h0->matrix(), wall);

    CHECK(commit_yaw(*G, "helios", static_cast<float>(dyaw_deg * M_PI / 180.0)));
    CHECK(commit_yaw(*G, "ricoh",  static_cast<float>(dyaw_deg * M_PI / 180.0)));

    // 1. helios: the first query after the commit (the brief allows 2 polls)
    double db = 0; int polls = 0;
    for (; polls < 2; ++polls)
    {
        const auto h1 = lidar_T("helios");
        if (not h1) break;
        db = wrap(bearing_deg(h1->matrix(), wall) - b0);
        if (std::abs(std::abs(db) - dyaw_deg) <= 0.05) break;
    }
    std::println("helios: bearing moved {:+.4f} deg after {} poll(s)", db, polls + 1);
    CHECK(std::abs(std::abs(db) - dyaw_deg) <= 0.05);

    // 2. camera: ts==0 on the main-thread instance, after the event loop has run the queued invalidations
    // Pump the loop for up to 2 s: the invalidation is queued, and the edge signal is itself emitted off-thread.
    for (int i = 0; i < 200; ++i)
    {
        QCoreApplication::processEvents();
        const auto t = cam_T("ricoh");
        if (t and (t->matrix() - c0->matrix()).cwiseAbs().maxCoeff() > 0) { std::println("ricoh: cache invalidated after {} pumps", i + 1); break; }
        QThread::msleep(10);
    }
    const auto c1 = cam_T("ricoh");
    CHECK(c1.has_value());
    if (c1)
    {
        const Eigen::Matrix3d Rrel = c1->matrix().block<3, 3>(0, 0) * c0->matrix().block<3, 3>(0, 0).transpose();
        const double ang = Eigen::AngleAxisd(Rrel).angle() * 180.0 / M_PI;   // rotation angle of the change
        std::println("ricoh: cam<-body rotation changed by {:.4f} deg", ang);
        CHECK(std::abs(ang - dyaw_deg) <= 0.05);
    }
    // 3. THE FIX ITSELF: CameraIngestor::refresh_graph_extrinsic() against this graph.
    {
        rc::CameraIngestor ing(G, "zed");
        CHECK(ing.bind_extrinsic("body"));
        const Eigen::Matrix3f R0 = ing.base_R();
        CHECK(not ing.refresh_graph_extrinsic());                       // edge untouched => unchanged
        CHECK((ing.base_R() - R0).cwiseAbs().maxCoeff() == 0.f);
        CHECK(commit_yaw(*G, "zed", static_cast<float>(dyaw_deg * M_PI / 180.0)));
        bool changed = false;
        for (int i = 0; i < 200 and not changed; ++i)
        {
            QCoreApplication::processEvents();
            changed = ing.refresh_graph_extrinsic();
            if (not changed) QThread::msleep(10);
        }
        CHECK(changed);
        const Eigen::Matrix3d Rrel = (ing.base_R() * R0.transpose()).cast<double>();
        const double ang = Eigen::AngleAxisd(Rrel).angle() * 180.0 / M_PI;
        std::println("CameraIngestor(zed): base extrinsic rotated {:.4f} deg by refresh_graph_extrinsic()", ang);
        CHECK(std::abs(ang - dyaw_deg) <= 0.05);
        CHECK(not ing.refresh_graph_extrinsic());                       // and settles
        // mountPublish on => refresh disabled: a further edge change must NOT move the base (no double application)
        ing.set_live_graph_refresh(false);
        const Eigen::Matrix3f R1 = ing.base_R();
        CHECK(commit_yaw(*G, "zed", static_cast<float>(dyaw_deg * M_PI / 180.0)));
        for (int i = 0; i < 20; ++i) { QCoreApplication::processEvents(); QThread::msleep(10); }
        CHECK(not ing.refresh_graph_extrinsic());
        CHECK((ing.base_R() - R1).cwiseAbs().maxCoeff() == 0.f);
    }
    std::println("{}", failures == 0 ? "PASS" : "FAILED");
    G.reset();
    return failures == 0 ? 0 : 1;
}
