#include "place_stage.h"

#include <genericworker.h>                       // DSR graph API
#include <dsr/api/dsr_api.h>

#include "../../common/diag_log/rotating_csv.h"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <iomanip>
#include <limits>
#include <print>

namespace rc
{
namespace
{
constexpr const char* kRed   = "\033[1;31m";
constexpr const char* kReset = "\033[0m";

/// SE(2) triple from a 3-D affine transform: planar translation + yaw about +Z.
inline Eigen::Vector3f se2_of(const Eigen::Transform<double, 3, Eigen::Affine>& T)
{
    const Eigen::Matrix3d R = T.linear();
    return { float(T.translation().x()), float(T.translation().y()),
             float(std::atan2(R(1, 0), R(0, 0))) };
}

/// fp32 -> fp16 bit pattern. The query grid is 16x32x384 per frame; fp16 halves 786 KB to 393 KB and
/// costs ~3 decimal digits, which is far below the noise on a ViT activation.
inline std::uint16_t to_half(float f)
{
    std::uint32_t x; std::memcpy(&x, &f, 4);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t exp = std::int32_t((x >> 23) & 0xFF) - 127 + 15;
    std::uint32_t man = x & 0x7FFFFFu;
    if (exp <= 0)  return std::uint16_t(sign);
    if (exp >= 31) return std::uint16_t(sign | 0x7C00u);
    return std::uint16_t(sign | (std::uint32_t(exp) << 10) | (man >> 13));
}
}   // namespace

PlaceStage::PlaceStage(const PlaceStageConfig& cfg, std::shared_ptr<DSR::DSRGraph> graph)
    : cfg_(cfg), graph_(std::move(graph))
{
    encoder_ = std::make_unique<rc::place::PlaceEncoder>(cfg_.encoder);
    if (graph_)
    {
        rt_api_ = graph_->get_rt_api();
        // ★ Its OWN InnerEigenAPI instance. get_inner_eigen_api() hands out a fresh unique_ptr per
        // call, and the ts==0 path reads an UNLOCKED cache — sharing one across threads is the one
        // real cliff in the DSR API. Every lookup here pins a real stamp anyway (ts != 0 bypasses the
        // cache entirely), so this instance never touches it.
        inner_ = graph_->get_inner_eigen_api();
    }

    rc::place::Header h;
    h.dim       = std::uint32_t(encoder_->dim());
    h.n_sectors = std::uint32_t(cfg_.encoder.n_sectors);
    h.pool_p    = cfg_.encoder.pool_p;
    h.band_lo   = cfg_.encoder.band_lo;
    h.band_hi   = cfg_.encoder.band_hi;
    h.center    = cfg_.encoder.center ? 1 : 0;
    h.input_w   = cfg_.encoder.input_w;
    h.input_h   = cfg_.encoder.input_h;
    h.model_name = std::filesystem::path(cfg_.encoder.model_path).filename().string();
    h.azimuth_tune_deg = cfg_.azimuth_tune_deg;
    if (graph_)
        if (const auto rooms = graph_->get_nodes_by_type("room"); not rooms.empty())
            h.room_name = rooms.front().name();
    map_.set_header(h);

    if (cfg_.build_map)
    {
        std::string why;
        if (map_.load(cfg_.map_path, cfg_.map_blob_path, &h, &why))
            std::println("[PlaceStage] loaded {} keyframes from {}", map_.size(), cfg_.map_path);
        else if (std::filesystem::exists(cfg_.map_path))
            // ★ Loud and specific. A map that silently fails to load looks like a map that is simply
            // empty, and the run then quietly rebuilds it in the wrong frame.
            std::println("{}[PlaceStage] REFUSED to load {}: {}{}", kRed, cfg_.map_path, why, kReset);
        next_id_ = map_.empty() ? 0 : map_.keyframes().back().id + 1;
        map_size_ = map_.size();
    }
    if (cfg_.log_queries) open_logs();
}

PlaceStage::~PlaceStage()
{
    if (cfg_.build_map and map_.size() > 0) save_map();
}

// ── pose + covariance ───────────────────────────────────────────────────────────────────────────
bool PlaceStage::robot_pose_in_room(std::uint64_t stamp,
                                    Eigen::Vector3f& pose, Eigen::Matrix3f& cov) const
{
    if (not graph_ or not inner_) return false;
    const auto rooms = graph_->get_nodes_by_type("room");
    if (rooms.empty()) return false;
    const auto& room = rooms.front();
    const auto robots = graph_->get_nodes_by_type("robot");
    if (robots.empty()) return false;
    const auto& robot = robots.front();

    // room<-robot, pinned to the panorama stamp (ts != 0 ⇒ no InnerEigenAPI cache).
    const auto T = inner_->get_transformation_matrix(room.name(), robot.name(), stamp);
    if (not T.has_value()) return false;                  // ALWAYS check: the documented crash mode
    pose = se2_of(T.value());

    cov = Eigen::Matrix3f::Identity() * 1e-2f;            // conservative default if none published
    if (not rt_api_) return true;

    // Which way does the edge run? Once the room node exists room_concept re-parents it UNDER the
    // robot, and from then on the published covariance is cov(robot->room) -- already pushed through
    // the inversion Jacobian (room_scene_graph.cpp:343-352).
    bool inverted = true;
    auto edge = graph_->get_edge(robot.id(), room.id(), "RT");
    if (not edge.has_value())
    {
        inverted = false;
        if (const auto roots = graph_->get_nodes_by_type("root"); not roots.empty())
            edge = graph_->get_edge(roots.front().id(), robot.id(), "RT");
    }
    if (not edge.has_value()) return true;

    // ★ Through cortex's own accessor, never by hand-indexing the flat attribute: it is a 6x6
    // ROW-MAJOR SE3 block, often a RING of them, and block 0 is not necessarily the newest.
    const auto c6 = rt_api_->get_edge_RT_covariance(edge.value());
    if (not c6.has_value()) return true;

    // SE(2) lives at SE3 slots {0, 1, 5} -- (2,2) is var_Z, and reading yaw from there is a defect
    // this fleet has already paid for once.
    static constexpr int se3_of_se2[3] = { 0, 1, 5 };
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            cov(r, c) = float((*c6)(se3_of_se2[r], se3_of_se2[c]));

    if (inverted)
        // Undo the transport: J(p)^-1 == J(inv(p)), so evaluate the SAME Jacobian at the PUBLISHED
        // (robot->room) pose to get back a room-frame covariance for the robot.
        cov = rc::place::invert_se2_cov(cov, rc::place::inv_se2(pose));

    // Keep it a usable covariance even if the producer published something degenerate.
    for (int i = 0; i < 3; ++i)
        if (not std::isfinite(cov(i, i)) or cov(i, i) <= 0.f) cov(i, i) = 1e-2f;
    return true;
}

// ── logging ─────────────────────────────────────────────────────────────────────────────────────
void PlaceStage::open_logs()
{
    std::error_code ec;
    std::filesystem::create_directories(cfg_.log_dir, ec);
    // The query log IS a diagnostic dataset, so it rotates (the map, being persisted state, does not).
    rc::diag::open_rotating(qlog_, cfg_.log_dir + "/place_query_log.csv",
        "stamp_ms,est_x,est_y,est_theta,est_rx,est_ry,est_rtheta,"
        "cxx,cxy,cxt,cyy,cyt,ctt,gt_x,gt_y,gt_angle_raw,gt_valid,"
        "blob_index,grid_rows,grid_cols,dim,n_sectors");
    qlog_ << std::setprecision(std::numeric_limits<float>::max_digits10);
    if (cfg_.log_grid)
    {
        qblob_.open(cfg_.log_dir + "/place_query_grid.f16", std::ios::binary | std::ios::trunc);
        const char magic[4] = { 'R', 'C', 'P', 'G' };
        qblob_.write(magic, 4);
        const std::uint32_t f[5] = { rc::place::kFormatVersion, std::uint32_t(encoder_->grid_rows()),
                                     std::uint32_t(encoder_->grid_cols()),
                                     std::uint32_t(encoder_->dim()), 1u /* fp16 */ };
        for (std::uint32_t v : f) qblob_.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }
}

void PlaceStage::log_query(const PerceptionFrame& in, const std::vector<float>& desc,
                           const std::vector<float>& grid,
                           const Eigen::Vector3f& ricoh_pose, const Eigen::Vector3f& robot_pose,
                           const Eigen::Matrix3f& cov)
{
    if (not qlog_.is_open()) return;

    // ★ GROUND TRUTH IS LOGGED EXACTLY AS PUBLISHED -- un-negated, un-offset. robot_gt_angle arrives
    // with an inverted sign and robot_gt_* is in the WEBOTS WORLD frame, not the room frame. Applying
    // either correction here would make this log LIE the moment robot_concept is fixed. place_eval
    // grades on RELATIVE poses (so the constant frame offset cancels exactly) and scores both sign
    // conventions by constancy, naming the winner -- the gt_convention_report() pattern.
    float gx = 0.f, gy = 0.f, ga = 0.f; int gv = 0;
    if (graph_)
        if (const auto robots = graph_->get_nodes_by_type("robot"); not robots.empty())
        {
            const auto& r = robots.front();
            // Type-attributed accessors, never runtime_checked_* (CLAUDE.md): a typo is a compile
            // error here rather than a runtime throw.
            const auto x = graph_->get_attrib_by_name<robot_gt_x_att>(r);
            const auto y = graph_->get_attrib_by_name<robot_gt_y_att>(r);
            const auto a = graph_->get_attrib_by_name<robot_gt_angle_att>(r);
            if (x and y and a) { gx = *x; gy = *y; ga = *a; gv = 1; }
        }

    std::uint64_t idx = 0;
    if (cfg_.log_grid and qblob_.is_open() and not grid.empty())
    {
        idx = blob_index_++;
        std::vector<std::uint16_t> h(grid.size());
        std::transform(grid.begin(), grid.end(), h.begin(), to_half);
        qblob_.write(reinterpret_cast<const char*>(h.data()),
                     std::streamsize(h.size() * sizeof(std::uint16_t)));
    }

    qlog_ << in.stamp << ','
          << ricoh_pose.x() << ',' << ricoh_pose.y() << ',' << ricoh_pose.z() << ','
          << robot_pose.x() << ',' << robot_pose.y() << ',' << robot_pose.z() << ','
          << cov(0,0) << ',' << cov(0,1) << ',' << cov(0,2) << ','
          << cov(1,1) << ',' << cov(1,2) << ',' << cov(2,2) << ','
          << gx << ',' << gy << ',' << ga << ',' << gv << ','
          << idx << ',' << encoder_->grid_rows() << ',' << encoder_->grid_cols() << ','
          << encoder_->dim() << ',' << cfg_.encoder.n_sectors << '\n';

    // Descriptors ride in the grid blob when logging grids; when not, the pooled vector is what the
    // eval needs, so append it after the row-oriented columns in a sidecar.
    if (not cfg_.log_grid and not desc.empty())
    {
        if (not qblob_.is_open())
        {
            qblob_.open(cfg_.log_dir + "/place_query_desc.f32", std::ios::binary | std::ios::trunc);
            const char magic[4] = { 'R', 'C', 'P', 'D' };
            qblob_.write(magic, 4);
            const std::uint32_t f[3] = { rc::place::kFormatVersion,
                                         std::uint32_t(cfg_.encoder.n_sectors),
                                         std::uint32_t(encoder_->dim()) };
            for (std::uint32_t v : f) qblob_.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }
        qblob_.write(reinterpret_cast<const char*>(desc.data()),
                     std::streamsize(desc.size() * sizeof(float)));
    }
}

// ── the stage ───────────────────────────────────────────────────────────────────────────────────
void PlaceStage::run(const PerceptionFrame& in, PerceptionResult& /*out*/)
{
    // 360 only: the sector representation is meaningless on a perspective frame, and the ZED path
    // would re-introduce yaw as a nuisance -- which is the thing the panorama removes.
    if (not ready() or not in.is_360 or in.rgbd.bgr.empty()) return;

    // ★ STARTUP SELF-TEST: one extra forward pass, once. Catches a transposed patch grid, a wrong
    // register-token offset and a collapsed pooling band -- three failures that do not crash and that
    // still produce entirely plausible numbers.
    if (not self_tested_)
    {
        self_tested_ = true;
        std::string detail;
        const bool ok = encoder_->self_test(in.rgbd.bgr, &detail);
        if (ok) std::println("[PlaceStage] self-test PASS — {}", detail);
        else    std::println("{}[PlaceStage] self-test FAIL — {}. A yaw change is NOT a clean cyclic "
                             "shift of the sector array; do not trust this map.{}", kRed, detail, kReset);
    }

    if (cfg_.decimation > 1 and (counter_++ % std::uint64_t(cfg_.decimation)) != 0) return;

    std::vector<float> grid;
    const auto desc = encoder_->encode(in.rgbd.bgr, cfg_.log_grid ? &grid : nullptr);
    if (desc.empty()) return;

    const Eigen::Vector3f ricoh_pose = se2_of(in.room_T_sensor);
    Eigen::Vector3f robot_pose;
    Eigen::Matrix3f cov;
    if (not robot_pose_in_room(in.stamp, robot_pose, cov)) return;   // no pose ⇒ no keyframe

    if (cfg_.log_queries and (cfg_.log_stride <= 1 or (counter_ / std::uint64_t(std::max(cfg_.decimation, 1)))
                                                      % std::uint64_t(cfg_.log_stride) == 0))
        log_query(in, desc, grid, ricoh_pose, robot_pose, cov);

    if (cfg_.build_map and map_.should_insert(ricoh_pose, cfg_.insert_min_dist_m))
    {
        rc::place::Keyframe k;
        k.id = next_id_++;
        k.stamp_ms   = in.stamp;
        k.ricoh_pose = ricoh_pose;
        k.robot_pose = robot_pose;
        k.cov        = cov;
        map_.add(k, desc);
        if (++since_save_ >= std::max(cfg_.save_every_n, 1)) save_map();   // never per frame
        std::lock_guard lk(status_mx_);
        map_size_ = map_.size();
        summary_ = std::format("kf {} @ ({:.2f}, {:.2f}, {:.0f} deg)  sigma_xy={:.3f} m",
                               k.id, robot_pose.x(), robot_pose.y(),
                               robot_pose.z() * 180.0f / float(M_PI),
                               std::sqrt(std::max(cov(0,0), cov(1,1))));
    }
}

void PlaceStage::save_map()
{
    if (not map_.save(cfg_.map_path, cfg_.map_blob_path))
        std::println("{}[PlaceStage] FAILED to save the map to {}{}", kRed, cfg_.map_path, kReset);
    else
        std::println("[PlaceStage] saved {} keyframes -> {} + {}",
                     map_.size(), cfg_.map_path, cfg_.map_blob_path);
    since_save_ = 0;
}

std::size_t PlaceStage::map_size() const { std::lock_guard lk(status_mx_); return map_size_; }
std::string PlaceStage::last_summary() const { std::lock_guard lk(status_mx_); return summary_; }

}   // namespace rc
