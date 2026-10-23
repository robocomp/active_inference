// Plan Task 2.2: sensor_calibration's input layer, verified WITHOUT Webots and WITHOUT the live DSR domain.
//
//  A. pure parts: odometry attribute parsing, the camera decimator (30 Hz in, 1.5 Hz config => 1.5 +- 0.1 Hz out),
//     stream monitor (rate, stall, recovery);
//  B. in-process DSR graph (shadow.json, PRIVATE domain 91): odometry / pose + 6x6 covariance / room polygon read
//     back through SensorInputs::poll();
//  C. in-process MEDIA plane on a PRIVATE domain (44): synthetic helios / zed rgb+depth / ricoh / imu publishers,
//     descriptors written onto the graph nodes, SensorInputs creates its subscribers through the shared factories:
//     counts, decimation, deep copy, sensor->body transform, stall + recovery.
// Build+run:  make -C build sensor_inputs_test && ../bin/sensor_inputs_test
#include "../src/sensor_inputs.h"
#include "../../common/media_transport/media_transport.h"

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_rt_api.h>
#include <QCoreApplication>
#include <QThread>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <print>

namespace
{
int failures = 0;
void check(bool ok, int line, std::string_view what)
{
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)

constexpr std::uint32_t MEDIA_DOMAIN = 44;   // PRIVATE: the live media plane is 7

void part_a()
{
    using sc::make_odometry;
    // A1 odometry parsing
    const auto o = make_odometry(0.5f, -0.1f, 0.3f, 1234ull, 99ull, true, std::vector<float>{0.01f, -1.f, 0.02f});
    CHECK(o.has_value());
    if (o) { CHECK(o->adv == 0.5f and o->side == -0.1f and o->rot == 0.3f and o->source_ts_ms == 1234 and o->sim_ts_ms == 99 and o->simulated);
             CHECK(o->var_adv == 0.01f and o->var_side == -1.f and o->var_rot == 0.02f); }
    CHECK(not make_odometry(0.5f, 0.f, std::nullopt, 10ull, std::nullopt, std::nullopt, std::nullopt));   // rot absent
    CHECK(not make_odometry(0.5f, 0.f, 0.f, 0ull, std::nullopt, std::nullopt, std::nullopt));            // ts == 0
    const auto o2 = make_odometry(1.f, 0.f, 0.f, 5ull, std::nullopt, std::nullopt, std::nullopt);
    CHECK(o2 and o2->var_adv == -1.f and not o2->simulated and o2->sim_ts_ms == 0);
    std::println("A1 odometry parsing ok");

    // A2 decimator: 30 Hz input for 10 s, 1.5 Hz config
    sc::RateDecimator dec(1.5);
    int accepted = 0; const int n = 300;
    for (int i = 0; i < n; ++i) accepted += dec.accept(static_cast<std::int64_t>(1'700'000'000'000ll + std::llround(i * 1000.0 / 30.0)));
    const double hz = accepted / (n / 30.0);
    std::println("A2 decimator: {} of {} frames handed over = {:.2f} Hz", accepted, n, hz);
    CHECK(std::abs(hz - 1.5) <= 0.1 and accepted < n / 10);
    // a paused stream resyncs: after a 5 s gap the next frame is accepted and the schedule restarts from it
    CHECK(dec.accept(1'700'000'000'000ll + 20'000)); CHECK(not dec.accept(1'700'000'000'000ll + 20'033));

    // A3 monitor
    using Clk = sc::StreamMonitor::Clock;
    sc::StreamMonitor mon("imu", std::chrono::milliseconds(1000));
    const auto t0 = Clk::now();
    CHECK(mon.update(t0).empty() and not mon.stalled());                    // never seen => not "stalled"
    for (int i = 0; i < 100; ++i) mon.note_arrival(t0 + std::chrono::milliseconds(10 * i));
    const double r = mon.take_rate_hz(t0 + std::chrono::seconds(1));   // 100 arrivals in the 1 s window
    CHECK(std::abs(r - 100.0) < 1.0);
    CHECK(mon.update(t0 + std::chrono::milliseconds(1500)).empty());        // 0.5 s of silence: fine
    const auto m1 = mon.update(t0 + std::chrono::milliseconds(2500));
    CHECK(not m1.empty() and mon.stalled());                                // 1.5 s of silence: stalled, logged once
    CHECK(mon.update(t0 + std::chrono::milliseconds(3000)).empty());        // ...and only once
    mon.note_arrival(t0 + std::chrono::milliseconds(3100));
    CHECK(not mon.update(t0 + std::chrono::milliseconds(3100)).empty() and not mon.stalled());
    std::println("A3 monitor: rate {:.1f} Hz, stall/recover messages: '{}'", r, m1);
}

template <class Pub, class F>
bool publish_with(Pub& p, F fill)
{
    auto* s = p.loan();
    if (s == nullptr) return false;
    fill(*s);
    return p.publish(s);
}

}  // namespace

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    part_a();

    auto G = std::make_shared<DSR::DSRGraph>("sensor_inputs_test", 991, std::string(SHADOW_JSON), true,
                                             static_cast<int8_t>(91), DSR::SignalMode::QT);
    constexpr std::int64_t T0 = 1'700'000'000'000ll;
    const float kYaw = static_cast<float>(M_PI / 6.0);   // 30 deg: the stored edge (room in robot frame)

    // ── B. graph reads ──────────────────────────────────────────────────────────────────────────────
    auto robot = G->get_nodes_by_type("robot").front();
    {
        DSR::Node room = DSR::Node::create<room_node_type>("room_1");
        G->add_or_modify_attrib_local<delimiting_polygon_x_att>(room, std::vector<float>{0.f, 4.f, 4.f, 0.f});
        G->add_or_modify_attrib_local<delimiting_polygon_y_att>(room, std::vector<float>{0.f, 0.f, 6.f, 6.f});
        const auto rid = G->insert_node(room);
        CHECK(rid.has_value());
        auto rt = G->get_rt_api();
        auto rn = G->get_node(robot.id()).value();
        rt->insert_or_assign_edge_RT(rn, *rid, std::vector<float>{1.f, 2.f, 0.f}, std::vector<float>{0.f, 0.f, kYaw});
        std::vector<float> cov(36, 0.f);
        cov[0] = 0.01f; cov[7] = 0.02f; cov[35] = 0.003f;
        CHECK(rt->insert_or_assign_edge_RT_covariance(robot.id(), *rid, DSR::RT_API::CovarianceKind::Pose, cov));
        auto rn2 = G->get_node(robot.id()).value();
        G->add_or_modify_attrib_local<robot_current_advance_speed_att>(rn2, 0.25f);
        G->add_or_modify_attrib_local<robot_current_side_speed_att>(rn2, -0.05f);
        G->add_or_modify_attrib_local<robot_current_angular_speed_att>(rn2, 0.4f);
        G->add_or_modify_attrib_local<robot_current_speed_timestamp_att>(rn2, static_cast<unsigned long>(T0));
        G->update_node(rn2);
    }

    // Media descriptors on the sensor nodes, all on the PRIVATE media domain.
    auto write_desc = [&](const std::string& node, const std::string& type, const char* tag,
                          std::map<std::string, std::string> streams)
    {
        rc::media::MediaDescriptor d;
        d.domain_id = MEDIA_DOMAIN; d.type_name = type; d.type_tag = tag; d.ready = true;
        for (auto& [k, v] : streams) d.stream_types[k] = type;
        d.streams = std::move(streams);
        auto n = G->get_node(node).value();
        G->add_or_modify_attrib_local<media_descriptor_att>(n, d.to_json());
        G->update_node(n);
    };
    write_desc("helios", "LidarFrame", rc::media::LIDAR_FRAME_TYPE_TAG, {{"lidar", "rc/test/helios"}});
    write_desc("zed", "ImageFrame", rc::media::IMAGE_FRAME_TYPE_TAG, {{"rgb", "rc/test/zed_rgb"}, {"depth", "rc/test/zed_depth"}});
    write_desc("ricoh", "Image360Frame", rc::media::IMAGE360_FRAME_TYPE_TAG, {{"rgb360", "rc/test/ricoh"}});
    write_desc("imu", "ImuFrame", rc::media::IMU_FRAME_TYPE_TAG, {{"imu", "rc/test/imu"}});

    // Publishers (what robot_concept does in production), same private domain.
    auto pcfg = [](const char* topic) { rc::media::PublisherConfig c; c.domain_id = MEDIA_DOMAIN; c.topic_name = topic; return c; };
    rc::media::LidarPublisher pub_helios; rc::media::MediaPublisher pub_rgb, pub_depth; rc::media::Image360Publisher pub_ricoh;
    rc::media::ImuPublisher pub_imu;
    CHECK(pub_helios.init(pcfg("rc/test/helios"))); CHECK(pub_rgb.init(pcfg("rc/test/zed_rgb")));
    CHECK(pub_depth.init(pcfg("rc/test/zed_depth"))); CHECK(pub_ricoh.init(pcfg("rc/test/ricoh"))); CHECK(pub_imu.init(pcfg("rc/test/imu")));

    sc::InputsConfig cfg; cfg.camera_rate_hz = 1.5; cfg.stall_imu_ms = 300; cfg.log_period_ms = 1'000'000;
    auto inputs = std::make_unique<sc::SensorInputs>(G, cfg);   // main thread, like the agent

    std::vector<sc::LidarIn> lid; std::vector<sc::ImageIn> img; std::vector<sc::ImuIn> imu; std::vector<sc::OdometryIn> odo;
    auto pump = [&]()
    {
        QCoreApplication::processEvents();
        inputs->poll();
        for (auto& v : inputs->take_lidar()) lid.push_back(std::move(v));
        for (auto& v : inputs->take_images()) img.push_back(std::move(v));
        for (auto& v : inputs->take_imu()) imu.push_back(std::move(v));
        for (auto& v : inputs->take_odometry()) odo.push_back(std::move(v));
    };

    // Let DDS discovery match readers to writers (subscribers are created by the first poll()).
    for (int i = 0; i < 400; ++i) { pump(); QThread::msleep(5); }

    // B. graph reads, through poll()
    CHECK(odo.size() == 1);
    if (!odo.empty()) CHECK(odo[0].adv == 0.25f and odo[0].side == -0.05f and odo[0].rot == 0.4f and odo[0].source_ts_ms == T0);
    const auto& pose = inputs->pose();
    CHECK(pose.has_value());
    if (pose)
    {
        // room_T_robot is the INVERSE of the stored edge: R(-30deg), t = -R^T t' = -(R(-30)^T (1,2)).
        const double th = M_PI / 6.0, c = std::cos(th), s = std::sin(th);
        const Eigen::Vector2d t_expect(-(c * 1.0 + s * 2.0), -(-s * 1.0 + c * 2.0));
        const double yaw = std::atan2(pose->room_T_robot.linear()(1, 0), pose->room_T_robot.linear()(0, 0));
        CHECK((pose->room_T_robot.translation().head<2>() - t_expect).norm() < 1e-5 and std::abs(yaw + th) < 1e-5);
        // Covariance: stored diag(.01,.02,.003) is of the EDGE pose. Independent expectation by a NUMERIC Jacobian
        // of g(t, th) = (-R(th)^T t, -th) at the stored pose (central differences; no use of the library's formula).
        auto g = [](Eigen::Vector3d v)
        { const double a = std::cos(v[2]), b = std::sin(v[2]);
          return Eigen::Vector3d(-(a * v[0] + b * v[1]), -(-b * v[0] + a * v[1]), -v[2]); };
        const Eigen::Vector3d p0(1.0, 2.0, th);
        Eigen::Matrix3d J;
        for (int j = 0; j < 3; ++j) { Eigen::Vector3d d = Eigen::Vector3d::Zero(); d[j] = 1e-6; J.col(j) = (g(p0 + d) - g(p0 - d)) / 2e-6; }
        const Eigen::Matrix3d Sexp = J * Eigen::Vector3d(0.01, 0.02, 0.003).asDiagonal() * J.transpose();
        const int k[3] = {0, 1, 5};
        bool ok = pose->has_cov;
        for (int r = 0; r < 3; ++r) for (int q = 0; q < 3; ++q)
            ok = ok and std::abs(pose->cov_room_robot[k[r] * 6 + k[q]] - Sexp(r, q)) < 1e-5;
        CHECK(ok);
        CHECK(std::abs(pose->cov_room_robot[0] - 0.01f) > 3e-3f);     // would equal .01 if the frame were left unconverted
        CHECK(std::abs(pose->cov_room_robot[35] - 0.003f) < 1e-7f);   // yaw variance is invariant
        std::println("B pose room_T_robot t=({:.3f},{:.3f}) yaw={:.3f}; cov_room_robot xx={:.4f} yy={:.4f} xy={:.4f} yawyaw={:.4f} (stored xx=.01 yy=.02)",
                     pose->room_T_robot.translation().x(), pose->room_T_robot.translation().y(), yaw,
                     pose->cov_room_robot[0], pose->cov_room_robot[7], pose->cov_room_robot[1], pose->cov_room_robot[35]);
    }
    CHECK(inputs->polygon().vertices.size() == 4);
    if (inputs->polygon().vertices.size() == 4) CHECK(inputs->polygon().vertices[2].isApprox(Eigen::Vector2f(4.f, 6.f)));
    std::println("B odometry/pose/polygon read through poll(): ok");

    // C. media plane. 10 s of STAMPED time at 30 Hz (published faster than real time, paced 5 ms).
    const int N = 300;
    int sent_imu = 0, sent_cam = 0, sent_lidar = 0;
    for (int i = 0; i < N; ++i)
    {
        const std::uint64_t stamp = static_cast<std::uint64_t>(T0 + 1000 + std::llround(i * 1000.0 / 30.0));
        sent_imu += publish_with(pub_imu, [&](rc::media::ImuFrame& f) { f.frame_id(i); f.stamp_ms(stamp); f.acc_z(9.81f); f.gyro_z(0.01f * i); });
        const bool a = publish_with(pub_rgb, [&](rc::media::ImageFrame& f)
        {
            f.frame_id(i); f.stamp_ms(stamp); f.width(8); f.height(4); f.step(24); f.format(rc::media::FORMAT_BGR8); f.size(96);
            std::memset(f.data().data(), i & 0xFF, 96);
        });
        publish_with(pub_depth, [&](rc::media::ImageFrame& f)
        {
            f.frame_id(i); f.stamp_ms(stamp); f.width(8); f.height(4); f.step(16); f.format(rc::media::FORMAT_Z16); f.size(64);
            std::memset(f.data().data(), 7, 64);
        });
        publish_with(pub_ricoh, [&](rc::media::Image360Frame& f)
        {
            f.frame_id(i); f.stamp_ms(stamp); f.width(8); f.height(4); f.step(24); f.format(rc::media::IMG360_FORMAT_BGR8); f.size(96);
            std::memset(f.data().data(), 3, 96);
        });
        sent_cam += a;
        if (i % 3 == 0)   // 10 Hz LiDAR
            sent_lidar += publish_with(pub_helios, [&](rc::media::LidarFrame& f)
            {
                f.frame_id(i); f.stamp_ms(stamp); f.count(2); f.format(rc::media::LIDAR_FORMAT_XYZ_F32); f.stride(3);
                f.points()[0] = 2.f; f.points()[1] = 0.5f; f.points()[2] = 0.f;
                f.points()[3] = 1.f; f.points()[4] = 1.f;  f.points()[5] = 0.3f;
            });
        QThread::msleep(5);
        pump();
    }
    for (int i = 0; i < 50; ++i) { pump(); QThread::msleep(5); }

    auto count = [&](const char* s) { return std::ranges::count_if(img, [&](const sc::ImageIn& m) { return m.sensor + "/" + m.stream == s; }); };
    const auto nrgb = count("zed/rgb"), ndep = count("zed/depth"), nric = count("ricoh/rgb360");
    std::println("C published: imu {} cam {} lidar {} | received: imu {} lidar {} | handed over: rgb {} depth {} ricoh {}",
                 sent_imu, sent_cam, sent_lidar, imu.size(), lid.size(), nrgb, ndep, nric);
    CHECK(sent_imu == N and static_cast<int>(imu.size()) >= N - 3);                     // IMU at full rate
    CHECK(sent_lidar > 0 and static_cast<int>(lid.size()) >= sent_lidar - 1);           // LiDAR: every sweep
    for (auto n : {nrgb, ndep, nric}) CHECK(n >= 14 and n <= 16);                       // 1.5 Hz x 10 s = 15 +- 1 (= +-0.1 Hz)
    CHECK(sent_cam > 10 * nrgb);                                                         // the rest was drained unconverted
    // payload integrity + deep copy: each rgb frame is filled with its own frame_id
    bool payload_ok = true;
    for (auto& m : img)
        if (m.stream == "rgb") payload_ok = payload_ok and m.data.size() == 96 and m.data[0] == (m.frame_id & 0xFF) and m.data[95] == (m.frame_id & 0xFF);
    CHECK(payload_ok);
    // LiDAR: sensor-frame points + the body<-sensor mount, equal to an independent graph query
    CHECK(not lid.empty() and lid[0].points_sensor.size() == 2 and lid[0].points_sensor[0].isApprox(Eigen::Vector3f(2.f, 0.5f, 0.f)));
    if (!lid.empty())
    {
        auto inner = G->get_inner_eigen_api();
        const auto T = inner->get_transformation_matrix("body", "helios", static_cast<std::uint64_t>(lid[0].stamp_ms), "RT", DSR::RT_API::TimeQuery::Nearest);
        CHECK(T.has_value() and lid[0].body_from_sensor.has_value());
        if (T and lid[0].body_from_sensor)
        {
            const Eigen::Vector4d q = T->matrix() * Eigen::Vector4d(2.0, 0.5, 0.0, 1.0);
            const auto pb = lid[0].points_body();
            CHECK(pb.size() == 2 and (pb[0].cast<double>() - q.head<3>()).norm() < 1e-4);
            std::println("C helios sweep: sensor ({:.2f},{:.2f},{:.2f}) -> body ({:.3f},{:.3f},{:.3f}), mount agrees with the graph query",
                         2.0, 0.5, 0.0, pb[0].x(), pb[0].y(), pb[0].z());
        }
    }
    CHECK(not inputs->any_stalled());
    std::println("C {}", inputs->status_line());

    // Stall: stop publishing; the IMU (stall_imu_ms = 300) is declared stalled, then recovers on a new frame.
    for (int i = 0; i < 100; ++i) { pump(); QThread::msleep(5); }
    CHECK(inputs->stalled("imu") and inputs->any_stalled());
    publish_with(pub_imu, [&](rc::media::ImuFrame& f) { f.frame_id(1000); f.stamp_ms(static_cast<std::uint64_t>(T0 + 20000)); });
    for (int i = 0; i < 100 and inputs->stalled("imu"); ++i) { pump(); QThread::msleep(5); }
    CHECK(not inputs->stalled("imu"));

    inputs.reset();   // subscribers down before the graph, as the agent does
    pub_helios.close(); pub_rgb.close(); pub_depth.close(); pub_ricoh.close(); pub_imu.close();
    std::println("{}", failures == 0 ? "PASS" : "FAILED");
    G.reset();
    std::fflush(stdout);
    // _Exit: the Ice static teardown in this link throws at normal exit (the agent uses rc::agent::terminal_exit for the same reason).
    std::_Exit(failures == 0 ? 0 : 1);
}
