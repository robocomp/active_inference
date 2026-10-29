#include "sensor_inputs.h"

#include "../../common/media_transport/media_transport.h"
#include "../../common/room_resolve/room_resolve.h"

#include <algorithm>
#include <cmath>
#include <print>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_rt_api.h>

namespace sc
{
using Clock = std::chrono::steady_clock;

namespace
{
std::int64_t wall_ms()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
}  // namespace

// ───────────────────────────── pure parts ─────────────────────────────

std::vector<Eigen::Vector3f> LidarIn::points_body() const
{
    std::vector<Eigen::Vector3f> out;
    if (not body_from_sensor.has_value())
        return out;
    out.reserve(points_sensor.size());
    for (const auto& p : points_sensor)
        out.emplace_back(*body_from_sensor * p);
    return out;
}

std::vector<Eigen::Vector3f> LidarIn::points_robot() const
{
    std::vector<Eigen::Vector3f> out;
    if (not robot_from_sensor.has_value())
        return out;
    out.reserve(points_sensor.size());
    for (const auto& p : points_sensor)
        out.emplace_back(*robot_from_sensor * p);
    return out;
}

Eigen::Matrix3d invert_pose_covariance_se2(const Eigen::Matrix3d& cov, double x, double y, double th)
{
    // Jacobian of g(t, th) = (-R(th)^T t, -th) at (x, y, th). g is an involution, so the same formula at the stored
    // pose maps the stored covariance to that of the inverse pose (room_scene_graph.cpp builds its J the same way).
    const double c = std::cos(th), s = std::sin(th);
    Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
    J(0, 0) = -c; J(0, 1) = -s; J(0, 2) =  s * x - c * y;
    J(1, 0) =  s; J(1, 1) = -c; J(1, 2) =  c * x + s * y;
    J(2, 2) = -1.0;
    return J * cov * J.transpose();
}

std::optional<OdometryIn> make_odometry(std::optional<float> adv, std::optional<float> side, std::optional<float> rot,
                                        std::optional<std::uint64_t> ts, std::optional<std::uint64_t> sim_ts,
                                        std::optional<bool> simulated, const std::optional<std::vector<float>>& variance)
{
    if (not adv.has_value() or not side.has_value() or not rot.has_value() or not ts.has_value() or *ts == 0)
        return std::nullopt;
    OdometryIn o;
    o.adv = *adv; o.side = *side; o.rot = *rot;
    o.source_ts_ms = static_cast<std::int64_t>(*ts);
    o.sim_ts_ms    = sim_ts.has_value() ? static_cast<std::int64_t>(*sim_ts) : 0;
    o.simulated    = simulated.value_or(false);
    if (variance.has_value() and variance->size() >= 3)
    {
        // Absent = -1; a negative entry means the producer said "unknown" for that channel.
        const auto stated = [](float x) { return x >= 0.f ? x : -1.f; };
        o.var_adv = stated((*variance)[0]); o.var_side = stated((*variance)[1]); o.var_rot = stated((*variance)[2]);
    }
    return o;
}

bool RateDecimator::accept(std::int64_t stamp_ms)
{
    if (period_ms_ <= 0.0)
        return true;
    const double t = static_cast<double>(stamp_ms);
    if (not started_)
    {
        started_ = true;
        next_due_ms_ = t + period_ms_;
        return true;
    }
    if (t < next_due_ms_)
        return false;
    // On schedule: advance by exactly one period, so the long-run rate is rate_hz whatever the input rate.
    // A gap of a whole slot or more (stream paused) restarts the schedule from this frame.
    next_due_ms_ = (t >= next_due_ms_ + period_ms_) ? t + period_ms_ : next_due_ms_ + period_ms_;
    return true;
}

std::string StreamMonitor::update(Clock::time_point now)
{
    if (not ever_)
        return {};
    const bool now_stalled = (now - last_arrival_) > stall_after_;
    if (now_stalled == stalled_)
        return {};
    stalled_ = now_stalled;
    const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_arrival_).count();
    return stalled_ ? std::format("[inputs] stream '{}' STALLED (no frame for {} ms); consumers must not use its stale data", name_, age)
                    : std::format("[inputs] stream '{}' recovered", name_);
}

double StreamMonitor::take_rate_hz(Clock::time_point now)
{
    if (not window_started_)
    {
        window_started_ = true; window_start_ = now; window_count_ = 0;
        return 0.0;
    }
    const double s = std::chrono::duration<double>(now - window_start_).count();
    const double r = s > 0.0 ? static_cast<double>(window_count_) / s : 0.0;
    window_start_ = now; window_count_ = 0;
    return r;
}

// ───────────────────────────── the input layer ─────────────────────────────

SensorInputs::SensorInputs(std::shared_ptr<DSR::DSRGraph> graph, InputsConfig cfg)
    : G_(std::move(graph)), cfg_(cfg)
{
    inner_ = G_->get_inner_eigen_api();   // created on (and so owned by) the main thread
    rt_    = G_->get_rt_api();
}

SensorInputs::~SensorInputs() = default;

template <class T>
void SensorInputs::push(std::deque<T>& q, T&& v, std::size_t cap)
{
    q.push_back(std::move(v));
    while (q.size() > cap)   // memory bound, drop-oldest
        q.pop_front();
}

std::optional<Eigen::Affine3f> SensorInputs::body_from_sensor(const std::string& sensor, std::int64_t stamp_ms) const
{
    if (stamp_ms <= 0)
        stamp_ms = wall_ms();   // never ts==0: that is the cached path
    const auto T = inner_->get_transformation_matrix("body", sensor, static_cast<std::uint64_t>(stamp_ms), "RT",
                                                     DSR::RT_API::TimeQuery::Nearest);
    if (not T.has_value())
        return std::nullopt;
    return Eigen::Affine3f(Eigen::Matrix4f(T->matrix().cast<float>()));
}

std::optional<Eigen::Affine3f> SensorInputs::robot_from_sensor(const std::string& sensor, std::int64_t stamp_ms) const
{
    if (robot_name_.empty())
        return std::nullopt;
    if (stamp_ms <= 0)
        stamp_ms = wall_ms();   // never ts==0: that is the cached path
    const auto T = inner_->get_transformation_matrix(robot_name_, sensor, static_cast<std::uint64_t>(stamp_ms), "RT",
                                                     DSR::RT_API::TimeQuery::Nearest);
    if (not T.has_value())
        return std::nullopt;
    return Eigen::Affine3f(Eigen::Matrix4f(T->matrix().cast<float>()));
}

std::optional<Eigen::Affine3d> SensorInputs::room_T_robot_at(std::int64_t stamp_ms, double* walk_ms) const
{
    if (walk_ms) *walk_ms = 0.0;
    if (not room_id_.has_value() or robot_id_ == 0 or stamp_ms <= 0)
        return std::nullopt;
    const auto robot = G_->get_node(robot_id_);
    if (not robot.has_value())
        return std::nullopt;
    const auto e = DSR::RT_API::get_edge_RT(robot.value(), *room_id_);
    if (not e.has_value())
        return std::nullopt;
    DSR::RT_API::TimeQueryInfo info;
    const auto T = rt_->get_edge_RT_as_rtmat(e.value(), static_cast<std::uint64_t>(stamp_ms),
                                             DSR::RT_API::TimeQuery::Extrapolated, &info);
    // Stale/Clamped return an end block AS-IS: a pose from another instant is not evidence for this one.
    if (not T.has_value() or info.clamped())
        return std::nullopt;
    if (walk_ms) *walk_ms = static_cast<double>(info.applied_dt_ms);
    const Eigen::Affine3d& robot_T_room = *T;   // the edge as stored: the room in the robot frame
    return robot_T_room.inverse();
}

void SensorInputs::ensure_subscribers()
{
    const auto now = Clock::now();
    if (last_discovery_ != Clock::time_point{} and now - last_discovery_ < std::chrono::milliseconds(cfg_.discovery_period_ms))
        return;
    last_discovery_ = now;

    const auto present = [&](const std::string& node) -> bool
    {
        if (G_->get_node(node).has_value())
            return true;
        if (not announced_absent_[node])
        {
            announced_absent_[node] = true;
            std::print("[inputs] DSR node '{}' absent: stream not consumed (retrying quietly)\n", node);
        }
        return false;
    };
    const auto add = [&](const std::string& key, int stall_ms, double rate)
    {
        streams_.try_emplace(key, Stream{StreamMonitor(key, std::chrono::milliseconds(stall_ms)), RateDecimator(rate)});
    };

    for (const auto* name : {"helios", "bpearl"})
    {
        auto& sub = std::string(name) == "helios" ? helios_ : bpearl_;
        if (sub == nullptr and present(name))
            if ((sub = rc::media::make_lidar_subscriber_from_graph(*G_, name, "lidar")))
                add(name, cfg_.stall_lidar_ms, 0.0);
    }
    if (cfg_.camera_rate_hz > 0.0)
    {
        if ((zed_rgb_ == nullptr or zed_depth_ == nullptr) and present("zed"))
        {
            if (zed_rgb_ == nullptr and (zed_rgb_ = rc::media::make_image_subscriber_from_graph(*G_, "zed", "rgb")))
                add("zed/rgb", cfg_.stall_camera_ms, cfg_.camera_rate_hz);
            if (zed_depth_ == nullptr and (zed_depth_ = rc::media::make_image_subscriber_from_graph(*G_, "zed", "depth")))
                add("zed/depth", cfg_.stall_camera_ms, cfg_.camera_rate_hz);
        }
        // The ricoh panorama is an Image360 stream: the shared Image360 factory (same file, same contract).
        if (ricoh_ == nullptr and present("ricoh"))
            if ((ricoh_ = rc::media::make_image360_subscriber_from_graph(*G_, "ricoh", "rgb360")))
                add("ricoh", cfg_.stall_camera_ms, cfg_.camera_rate_hz);
    }
    if (imu_sub_ == nullptr and present("imu"))
        if ((imu_sub_ = rc::media::make_imu_subscriber_from_graph(*G_, "imu", "imu")))
            add("imu", cfg_.stall_imu_ms, 0.0);
}

void SensorInputs::drain_lidar(const std::string& sensor, rc::media::LidarSubscriber& sub, StreamMonitor& mon)
{
    sub.poll([&](const rc::media::LidarFrame& f, std::int64_t)
    {
        mon.note_arrival(Clock::now());
        LidarIn s;
        s.sensor = sensor;
        s.frame_id = f.frame_id();
        s.stamp_ms = static_cast<std::int64_t>(f.stamp_ms());
        const std::uint32_t stride = f.stride() ? f.stride() : 3u;
        const auto& pts = f.points();
        s.points_sensor.reserve(f.count());
        for (std::uint32_t i = 0; i < f.count(); ++i)
        {
            const std::size_t b = static_cast<std::size_t>(i) * stride;
            if (b + 2 >= pts.size())
                break;
            const float x = pts[b], y = pts[b + 1], z = pts[b + 2];
            if (std::isfinite(x) and std::isfinite(y) and std::isfinite(z))   // NaN is "no return", not a point
                s.points_sensor.emplace_back(x, y, z);
        }
        s.body_from_sensor = body_from_sensor(sensor, s.stamp_ms);
        s.robot_from_sensor = robot_from_sensor(sensor, s.stamp_ms);
        s.room_T_robot = room_T_robot_at(s.stamp_ms, &s.pose_walk_ms);
        push(lidar_q_, std::move(s), cfg_.max_lidar_queue);
    });
}

void SensorInputs::drain_image(const std::string& sensor, const std::string& stream, rc::media::MediaSubscriber& sub,
                               StreamMonitor& mon, RateDecimator& dec)
{
    sub.poll([&](const rc::media::ImageFrame& f, std::int64_t)
    {
        mon.note_arrival(Clock::now());
        if (not dec.accept(static_cast<std::int64_t>(f.stamp_ms())))
            return;   // drained, NOT converted: the camera cost is the copy
        ImageIn im;
        im.sensor = sensor; im.stream = stream;
        im.frame_id = f.frame_id(); im.stamp_ms = static_cast<std::int64_t>(f.stamp_ms());
        im.width = f.width(); im.height = f.height(); im.step = f.step(); im.format = f.format();
        const std::size_t n = std::min<std::size_t>(f.size(), f.data().size());
        im.data.assign(f.data().begin(), f.data().begin() + n);   // deep copy of the valid bytes
        im.body_from_sensor = body_from_sensor(sensor, im.stamp_ms);
        push(image_q_, std::move(im), cfg_.max_image_queue);
    });
}

void SensorInputs::drain_image360(rc::media::Image360Subscriber& sub, StreamMonitor& mon, RateDecimator& dec)
{
    sub.poll([&](const rc::media::Image360Frame& f, std::int64_t)
    {
        mon.note_arrival(Clock::now());
        if (not dec.accept(static_cast<std::int64_t>(f.stamp_ms())))
            return;
        ImageIn im;
        im.sensor = "ricoh"; im.stream = "rgb360";
        im.frame_id = f.frame_id(); im.stamp_ms = static_cast<std::int64_t>(f.stamp_ms());
        im.width = f.width(); im.height = f.height(); im.step = f.step(); im.format = f.format();
        const std::size_t n = std::min<std::size_t>(f.size(), f.data().size());
        im.data.assign(f.data().begin(), f.data().begin() + n);
        im.body_from_sensor = body_from_sensor("ricoh", im.stamp_ms);
        push(image_q_, std::move(im), cfg_.max_image_queue);
    });
}

void SensorInputs::drain_imu()
{
    auto& mon = streams_.at("imu").mon;
    imu_sub_->poll([&](const rc::media::ImuFrame& f, std::int64_t)
    {
        mon.note_arrival(Clock::now());
        ImuIn m;
        m.frame_id = f.frame_id(); m.stamp_ms = static_cast<std::int64_t>(f.stamp_ms());
        m.sim_stamp_ms = static_cast<std::int64_t>(f.sim_stamp_ms());
        m.acc  = {f.acc_x(), f.acc_y(), f.acc_z()};
        m.gyro = {f.gyro_x(), f.gyro_y(), f.gyro_z()};
        m.roll = f.roll(); m.pitch = f.pitch(); m.yaw = f.yaw();
        m.gyro_var = f.gyro_var(); m.acc_var = f.acc_var();
        push(imu_q_, std::move(m), cfg_.max_imu_queue);
    });
}

void SensorInputs::poll_graph()
{
    const auto robots = G_->get_nodes_by_type("robot");
    if (robots.empty())
        return;
    const auto& robot = robots.front();
    robot_name_ = robot.name();
    robot_id_ = robot.id();

    // ── odometry: the same attributes room_concept's slot reads, polled here (no DSR signal) ──
    {
        const auto adv = G_->get_attrib_by_name<robot_current_advance_speed_att>(robot);
        const auto sid = G_->get_attrib_by_name<robot_current_side_speed_att>(robot);
        const auto rot = G_->get_attrib_by_name<robot_current_angular_speed_att>(robot);
        const auto ts  = G_->get_attrib_by_name<robot_current_speed_timestamp_att>(robot);
        const auto sts = G_->get_attrib_by_name<robot_current_speed_sim_timestamp_att>(robot);
        const auto sim = G_->get_attrib_by_name<robot_current_speed_simulated_att>(robot);
        const auto var = G_->get_attrib_by_name<robot_current_speed_variance_att>(robot);
        std::optional<std::vector<float>> varv;
        if (var.has_value())
            varv = var.value().get();
        const auto toU = [](const auto& o) { return o.has_value() ? std::optional<std::uint64_t>(static_cast<std::uint64_t>(o.value())) : std::nullopt; };
        const auto toF = [](const auto& o) { return o.has_value() ? std::optional<float>(static_cast<float>(o.value())) : std::nullopt; };
        auto o = make_odometry(toF(adv), toF(sid), toF(rot), toU(ts), toU(sts),
                               sim.has_value() ? std::optional<bool>(sim.value()) : std::nullopt, varv);
        if (o.has_value() and o->source_ts_ms > last_odom_ts_)   // one entry per producer sample
        {
            last_odom_ts_ = o->source_ts_ms;
            ++odom_total_; odom_last_ = Clock::now();
            push(odom_q_, std::move(*o), cfg_.max_odom_queue);
        }
    }

    // ── pose + covariance on the robot -> room RT edge, as residual_concept reads it ──
    const auto room = rc::room::current_room(*G_);
    room_id_ = room;
    if (not room.has_value())
        return;
    // The edge carries a HISTORY RING (rt_translation holds several blocks), so the raw attributes are not "the
    // pose": ask the RT API for the newest block, and the covariance of the SAME block (same stamp + query).
    if (const auto e = DSR::RT_API::get_edge_RT(robot, *room); e.has_value())
    {
        if (const auto T = rt_->get_edge_RT_as_rtmat(e.value()); T.has_value())
        {
            PoseIn p;
            p.robot_id = robot.id(); p.room_id = *room; p.read_stamp_ms = wall_ms();
            const Eigen::Affine3d& robot_T_room = *T;    // the edge as stored: the room in the robot frame
            p.room_T_robot = robot_T_room.inverse();
            if (const auto C = rt_->get_edge_RT_covariance(e.value()); C.has_value())
            {
                // Stored layout is protocol: 6x6 ROW-MAJOR over [x y z rx ry rz]; SE(2) lives in slots 0, 1, 5.
                static constexpr int k[3] = {0, 1, 5};
                Eigen::Matrix3d S;
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        S(r, c) = (*C)(k[r], k[c]);
                const auto& R = robot_T_room.linear();
                const Eigen::Matrix3d Sc = invert_pose_covariance_se2(S, robot_T_room.translation().x(), robot_T_room.translation().y(),
                                                                      std::atan2(R(1, 0), R(0, 0)));
                Eigen::Matrix<double, 6, 6> M = *C;
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        M(k[r], k[c]) = Sc(r, c);
                for (int r = 0; r < 6; ++r)
                    for (int c = 0; c < 6; ++c)
                        p.cov_room_robot[r * 6 + c] = static_cast<float>(M(r, c));
                p.has_cov = true;
            }
            if (not pose_.has_value() or not pose_->room_T_robot.matrix().isApprox(p.room_T_robot.matrix(), 0.0))
                ++pose_total_;   // distinct pose (display counter only)
            pose_ = p;
        }
    }
    // ── room polygon: a 1 Hz read (it changes on layout publishes, not per tick) ──
    if (const auto now = Clock::now(); last_poly_ == Clock::time_point{} or now - last_poly_ >= std::chrono::seconds(1) or polygon_.room_id != *room)
    {
        last_poly_ = now;
        if (const auto rn = G_->get_node(*room); rn.has_value())
        {
            const auto px = G_->get_attrib_by_name<delimiting_polygon_x_att>(rn.value());
            const auto py = G_->get_attrib_by_name<delimiting_polygon_y_att>(rn.value());
            PolygonIn poly; poly.room_id = *room;
            poly.ceiling_m = G_->get_attrib_by_name<room_height_att>(rn.value()).value_or(0.f);
            poly.ceiling_sigma_m = G_->get_attrib_by_name<room_height_sigma_att>(rn.value()).value_or(0.f);
            if (px.has_value() and py.has_value())
            {
                const auto& xs = px->get(); const auto& ys = py->get();
                const std::size_t n = std::min(xs.size(), ys.size());
                for (std::size_t i = 0; i < n; ++i)
                    poly.vertices.emplace_back(xs[i], ys[i]);
            }
            polygon_ = std::move(poly);
        }
    }
}

void SensorInputs::monitor_and_log()
{
    const auto now = Clock::now();
    for (auto& [key, s] : streams_)
        if (const auto msg = s.mon.update(now); not msg.empty())
            std::print("{}\n", msg);
    if (last_log_ == Clock::time_point{})
        last_log_ = now;
    if (now - last_log_ < std::chrono::milliseconds(cfg_.log_period_ms))
        return;
    last_log_ = now;
    std::print("[inputs] rates: {}\n", status_line());
}

std::string SensorInputs::status_line()
{
    std::string out;
    const auto now = Clock::now();
    for (auto& [key, s] : streams_)
    {
        auto& mon = s.mon;
        out += std::format("{}={:.1f}Hz{} ", key, mon.take_rate_hz(now), mon.stalled() ? "(STALLED)" : "");
    }
    out += std::format("| odom={} pose={} poly={}v", last_odom_ts_ > 0 ? "yes" : "no",
                       pose_.has_value() ? (pose_->has_cov ? "yes+cov" : "yes") : "no", polygon_.vertices.size());
    if (streams_.empty())
        out = "no stream up yet; " + out;
    return out;
}

bool SensorInputs::any_stalled() const
{
    return std::ranges::any_of(streams_, [](const auto& kv) { return kv.second.mon.stalled(); });
}

void SensorInputs::poll()
{
    ensure_subscribers();
    if (helios_) drain_lidar("helios", *helios_, streams_.at("helios").mon);
    if (bpearl_) drain_lidar("bpearl", *bpearl_, streams_.at("bpearl").mon);
    if (zed_rgb_)   { auto& s = streams_.at("zed/rgb");   drain_image("zed", "rgb",   *zed_rgb_,   s.mon, s.dec); }
    if (zed_depth_) { auto& s = streams_.at("zed/depth"); drain_image("zed", "depth", *zed_depth_, s.mon, s.dec); }
    if (ricoh_)     { auto& s = streams_.at("ricoh");     drain_image360(*ricoh_, s.mon, s.dec); }
    if (imu_sub_)   drain_imu();
    poll_graph();
    monitor_and_log();
}

std::vector<LidarIn>    SensorInputs::take_lidar()    { std::vector<LidarIn> v(std::make_move_iterator(lidar_q_.begin()), std::make_move_iterator(lidar_q_.end())); lidar_q_.clear(); return v; }
std::vector<ImageIn>    SensorInputs::take_images()   { std::vector<ImageIn> v(std::make_move_iterator(image_q_.begin()), std::make_move_iterator(image_q_.end())); image_q_.clear(); return v; }
std::vector<ImuIn>      SensorInputs::take_imu()      { std::vector<ImuIn> v(imu_q_.begin(), imu_q_.end()); imu_q_.clear(); return v; }
std::vector<OdometryIn> SensorInputs::take_odometry() { std::vector<OdometryIn> v(odom_q_.begin(), odom_q_.end()); odom_q_.clear(); return v; }

}  // namespace sc
