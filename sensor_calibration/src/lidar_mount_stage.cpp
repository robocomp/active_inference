#include "lidar_mount_stage.h"

#include <cmath>
#include <ctime>
#include <filesystem>
#include <format>
#include <locale>

namespace sc
{
namespace
{
    constexpr double kRad2Deg = 180.0 / M_PI;
}

std::vector<rc::mountf::VerticalMount::Wall> walls_in_robot(const std::vector<Eigen::Vector2f>& polygon_room,
                                                            const Eigen::Affine3d& room_T_robot)
{
    std::vector<rc::mountf::VerticalMount::Wall> walls;
    if (polygon_room.size() < 3)
        return walls;
    const Eigen::Affine3d robot_T_room = room_T_robot.inverse();
    const auto to_robot = [&](const Eigen::Vector2f& v)
    { return Eigen::Vector2f((robot_T_room * Eigen::Vector3d(v.x(), v.y(), 0.0)).head<2>().cast<float>()); };
    walls.reserve(polygon_room.size());
    for (std::size_t k = 0; k < polygon_room.size(); ++k)
        walls.push_back({to_robot(polygon_room[k]), to_robot(polygon_room[(k + 1) % polygon_room.size()])});
    return walls;
}

std::int64_t place_of(const Eigen::Affine3d& room_T_robot)
{
    const auto& R = room_T_robot.linear();
    return rc::mountf::place_key(static_cast<float>(room_T_robot.translation().x()),
                                 static_cast<float>(room_T_robot.translation().y()),
                                 static_cast<float>(std::atan2(R(1, 0), R(0, 0))));
}

LidarMountStage::LidarMountStage(LidarMountStageConfig cfg) : cfg_(std::move(cfg))
{
    rc::mountf::VerticalMount::Params ph; ph.role = rc::mountf::VerticalMount::Role::HeliosTilt;
    rc::mountf::VerticalMount::Params pb; pb.role = rc::mountf::VerticalMount::Role::BpearlPlanar;
    vert_h_.set_params(ph);
    vert_bp_.set_params(pb);
}

std::vector<Eigen::Vector3f> LidarMountStage::strided(const std::vector<Eigen::Vector3f>& in) const
{
    // A COMPUTATIONAL stride, not a selection: every k-th return so one step costs ~ms; the per-sweep nuisance
    // (not the point count) already bounds what one sweep can claim.
    if (cfg_.max_points == 0 or in.size() <= cfg_.max_points)
        return in;
    const std::size_t k = (in.size() + cfg_.max_points - 1) / cfg_.max_points;
    std::vector<Eigen::Vector3f> out;
    out.reserve(in.size() / k + 1);
    for (std::size_t i = 0; i < in.size(); i += k)
        out.push_back(in[i]);
    return out;
}

void LidarMountStage::open_csv(std::ofstream& f, const std::string& stem, const char* header)
{
    if (f.is_open() or cfg_.log_dir.empty())
        return;
    std::error_code ec;
    std::filesystem::create_directories(cfg_.log_dir, ec);
    char stamp[32];
    const std::time_t now_t = std::time(nullptr);
    std::strftime(stamp, sizeof stamp, "%Y-%m-%d_%H-%M-%S", std::localtime(&now_t));
    f.open(cfg_.log_dir + "/" + stem + "_" + stamp + ".csv");
    f.imbue(std::locale::classic());   // CLAUDE.md: data files never acquire a comma separator
    f << header << '\n';
}

bool LidarMountStage::observe(const MountSweep& sweep, const std::vector<Eigen::Vector2f>& polygon_room, float ceiling_hint_m)
{
    const bool helios = sweep.sensor == "helios";
    if (not helios and sweep.sensor != "bpearl")
        return false;
    std::int64_t& last = helios ? last_helios_ms_ : last_bpearl_ms_;
    const double period_ms = cfg_.rate_hz > 0.0 ? 1000.0 / cfg_.rate_hz : 0.0;
    if (last != 0 and static_cast<double>(sweep.stamp_ms - last) < period_ms)
        return false;
    if (sweep.points_robot.empty())
        return false;
    last = sweep.stamp_ms;

    const float sxy = static_cast<float>(cfg_.pose_sigma_xy);
    const float syaw = static_cast<float>(cfg_.pose_sigma_yaw_deg / kRad2Deg);
    const Eigen::Matrix3f pose_cov = Eigen::Vector3f(sxy, sxy, syaw).cwiseAbs2().asDiagonal();

    std::vector<rc::mountf::VerticalMount::Wall> walls;
    std::int64_t place = -1;
    if (sweep.room_T_robot.has_value())
    {
        walls = walls_in_robot(polygon_room, *sweep.room_T_robot);
        place = place_of(*sweep.room_T_robot);
    }
    const auto pts = strided(sweep.points_robot);

    if (helios)
    {
        s_helios_ = sweep.origin_robot;
        if (not walls.empty())   // tilt from walls at every height + the ceiling (h_c a nuisance)
            vert_h_.observe_sweep(pts, s_helios_, walls, pose_cov, ceiling_hint_m, place);
        return true;
    }

    // bpearl: floor (absolute, needs no pose) + verticals (planar relative to the wall map)
    s_bpearl_ = sweep.origin_robot;
    floor_bp_.observe_sweep(pts, s_bpearl_, place);
    ++n_floor_;
    if (walls.empty())
        return true;
    vert_bp_.observe_sweep(pts, s_bpearl_, walls, pose_cov, 0.f, place);
    ++n_walls_b_;
    // Per-sweep planar reading (the 2026-10-08 diagnosis: a timing cause scales with speed x stamp gap and points
    // along the motion; a map/polygon cause depends on WHERE).
    Eigen::Vector3d m, sg;
    if (vert_bp_.last_sweep_planar(m, sg))
    {
        open_csv(sweep_csv_, "bpearl_sweeps", "stamp_ms,x,y,heading,dx,dy,dyaw,sd_dx,sd_dy,sd_dyaw,n_walls");
        if (sweep_csv_.is_open())
        {
            const auto& T = *sweep.room_T_robot;
            sweep_csv_ << sweep.stamp_ms << ',' << T.translation().x() << ',' << T.translation().y() << ','
                       << std::atan2(T.linear()(1, 0), T.linear()(0, 0)) << ',' << m(0) << ',' << m(1) << ',' << m(2)
                       << ',' << sg(0) << ',' << sg(1) << ',' << sg(2) << ',' << walls.size() << '\n';
        }
    }
    return true;
}

MountFactorsState LidarMountStage::state() const
{
    MountFactorsState s;
    s.floor = floor_bp_.info();
    s.vert_helios = vert_h_.info();
    s.vert_bpearl = vert_bp_.info();
    s.s_helios = s_helios_;
    s.s_bpearl = s_bpearl_;
    s.floor_share = floor_bp_.floor_share();
    s.floor_sigma = floor_bp_.sigma_floor();
    s.wall_sigma_h = vert_h_.sigma_wall();
    s.wall_sigma_b = vert_bp_.sigma_wall();
    s.sweeps_helios = vert_h_.sweeps();
    s.sweeps_bpearl_floor = n_floor_;
    s.sweeps_bpearl_walls = n_walls_b_;
    return s;
}

std::string LidarMountStage::summary() const
{
    const auto s = state();
    const Eigen::Vector3d fm = s.floor.mean(), fs = s.floor.sigma();
    const auto hm = s.vert_helios.mean(), hs = s.vert_helios.sigma();
    const auto bm = s.vert_bpearl.mean(), bs = s.vert_bpearl.sigma();
    return std::format("[mounts] bpearl floor roll {:+.3f}±{:.3f} pitch {:+.3f}±{:.3f} deg z {:+.1f}±{:.1f} mm ({} sw) | "
                       "helios tilt roll {:+.3f}±{:.3f} pitch {:+.3f}±{:.3f} deg ({} sw) | "
                       "bpearl planar x {:+.1f}±{:.1f} y {:+.1f}±{:.1f} mm yaw {:+.3f}±{:.3f} deg ({} sw)",
                       fm(0) * kRad2Deg, fs(0) * kRad2Deg, fm(1) * kRad2Deg, fs(1) * kRad2Deg, fm(2) * 1e3, fs(2) * 1e3,
                       s.sweeps_bpearl_floor,
                       hm(3) * kRad2Deg, hs(3) * kRad2Deg, hm(4) * kRad2Deg, hs(4) * kRad2Deg, s.sweeps_helios,
                       bm(0) * 1e3, bs(0) * 1e3, bm(1) * 1e3, bs(1) * 1e3, bm(5) * kRad2Deg, bs(5) * kRad2Deg,
                       s.sweeps_bpearl_walls);
}

void LidarMountStage::log_state(std::int64_t stamp_ms)
{
    open_csv(state_csv_, "mounts",
             "stamp_ms,floor_droll,floor_dpitch,floor_dz,sd_floor_droll,sd_floor_dpitch,sd_floor_dz,n_floor,"
             "helios_droll,helios_dpitch,sd_helios_droll,sd_helios_dpitch,n_helios,"
             "bpearl_dx,bpearl_dy,bpearl_dyaw,sd_bpearl_dx,sd_bpearl_dy,sd_bpearl_dyaw,n_bpearl_walls,"
             "floor_share,floor_sigma,wall_sigma_h,wall_sigma_b");
    if (not state_csv_.is_open())
        return;
    const auto s = state();
    const Eigen::Vector3d fm = s.floor.mean(), fs = s.floor.sigma();
    const auto hm = s.vert_helios.mean(), hs = s.vert_helios.sigma();
    const auto bm = s.vert_bpearl.mean(), bs = s.vert_bpearl.sigma();
    state_csv_ << stamp_ms << ',' << fm(0) << ',' << fm(1) << ',' << fm(2) << ',' << fs(0) << ',' << fs(1) << ',' << fs(2)
               << ',' << s.sweeps_bpearl_floor << ',' << hm(3) << ',' << hm(4) << ',' << hs(3) << ',' << hs(4) << ','
               << s.sweeps_helios << ',' << bm(0) << ',' << bm(1) << ',' << bm(5) << ',' << bs(0) << ',' << bs(1) << ','
               << bs(5) << ',' << s.sweeps_bpearl_walls << ',' << s.floor_share << ',' << s.floor_sigma << ','
               << s.wall_sigma_h << ',' << s.wall_sigma_b << '\n';
    state_csv_.flush();
}
}  // namespace sc
