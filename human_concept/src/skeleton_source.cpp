/*
 * skeleton_source.cpp — CSV replay backend + factory.
 */

#include "skeleton_source.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <print>
#include <sstream>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_eigen_defs.h>

#include "csv_parse.h"   // ★locale-independent parsing — NEVER strtof here (see the header)

namespace rc {

namespace
{
std::vector<std::string> split(const std::string& line)
{
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) cols.push_back(cell);
    return cols;
}
}  // namespace

ReplaySkeletonSource::ReplaySkeletonSource(std::string path, bool loop)
    : loop_(loop)
{
    if (path.empty())
    {
        std::print("human_concept: [replay] no replay path configured\n");
        return;
    }
    std::ifstream f(path);
    if (not f.is_open())
    {
        std::print("human_concept: [replay] cannot open '{}'\n", path);
        return;
    }
    std::string line;
    int line_no = 0;
    while (std::getline(f, line))
    {
        ++line_no;
        if (line.empty() or line[0] == '#') continue;
        const auto c = split(line);
        // Column-count auto-detect: with/without an id column, with/without confidence.
        int base = -1;       // index of kp0_x
        bool has_id = false;
        if      (c.size() == 1 + 54 or c.size() == 1 + 54 + human::NUM_KP) { base = 1; has_id = false; }
        else if (c.size() == 2 + 54 or c.size() == 2 + 54 + human::NUM_KP) { base = 2; has_id = true; }
        else
        {
            std::print("human_concept: [replay] line {} has {} cols (skipped)\n", line_no, c.size());
            continue;
        }

        SkeletonBody b;
        b.id = has_id ? csv::parse_int(c[1]) : 0;
        b.kp.resize(human::NUM_KP, 3);
        for (int i = 0; i < human::NUM_KP; ++i)
            for (int k = 0; k < 3; ++k)
                b.kp(i, k) = csv::parse_float(c[base + i * 3 + k]);

        if (static_cast<int>(c.size()) >= base + 54 + human::NUM_KP)
        {
            std::array<float, human::NUM_KP> cf{};
            for (int i = 0; i < human::NUM_KP; ++i)
                cf[i] = csv::parse_float(c[base + 54 + i]);
            b.conf = cf;
        }
        rows_.push_back(std::move(b));
    }
    std::print("human_concept: [replay] loaded {} frames from '{}'\n", rows_.size(), path);
}

std::vector<SkeletonBody> ReplaySkeletonSource::poll()
{
    if (rows_.empty())
        return {};
    if (cursor_ >= rows_.size())
    {
        if (not loop_)
            return {};
        cursor_ = 0;
    }
    return { rows_[cursor_++] };
}

// ── DSR backend: voxelizer 'skeleton' node → BODY_18 bodies (CAMERA→world transformed) ──────────

DsrSkeletonSource::DsrSkeletonSource(std::shared_ptr<DSR::DSRGraph> graph,
                                     DSR::InnerEigenAPI* inner_eigen,
                                     std::string world_frame,
                                     std::string camera_frame)
    : G_(std::move(graph)), inner_eigen_(inner_eigen),
      world_frame_(std::move(world_frame)), camera_frame_(std::move(camera_frame))
{}

std::vector<SkeletonBody> DsrSkeletonSource::poll()
{
    if (not G_)
        return {};

    const auto node_opt = G_->get_node("skeleton");
    if (not node_opt.has_value())
        return {};

    // TYPE-ATTRIBUTED reads (CLAUDE.md), compile-checked against dsr_attr_name.h. Optionals held so the
    // reference_wrapper payloads stay alive; skeleton_timestamp_ms is uint64 (read via .value()).
    const auto& node = node_opt.value();
    using VecOpt = std::optional<std::reference_wrapper<const std::vector<float>>>;
    const auto frame_opt = G_->get_attrib_by_name<skeleton_frame_id_att>(node);
    const auto count_opt = G_->get_attrib_by_name<skeleton_count_att>(node);
    const VecOpt ids_opt  = G_->get_attrib_by_name<skeleton_ids_att>(node);
    const VecOpt xyz_opt  = G_->get_attrib_by_name<skeleton_kp_xyz_att>(node);
    const VecOpt conf_opt = G_->get_attrib_by_name<skeleton_kp_conf_att>(node);
    const auto ts_opt     = G_->get_attrib_by_name<skeleton_timestamp_ms_att>(node);   // capture stamp (optional)
    if (not frame_opt or not count_opt or not xyz_opt)
        return {};

    const int frame_id = frame_opt.value();
    if (frame_id == last_frame_id_)   // same frame already consumed
        return {};
    last_frame_id_ = frame_id;

    const int count = std::max(0, count_opt.value());
    const auto& xyz  = xyz_opt.value().get();
    static const std::vector<float> empty_flat;
    const auto& ids  = ids_opt  ? ids_opt->get()  : empty_flat;
    const auto& conf = conf_opt ? conf_opt->get() : empty_flat;

    constexpr int K = human::NUM_KP;   // 18
    if (static_cast<int>(xyz.size()) < count * K * 3)
        return {};

    // Camera→world transform, PINNED to the frame's capture stamp (like the masks consumer) so the
    // keypoints are placed against the camera pose at acquisition time, not the latest pose — matters
    // while the robot is moving. ts==0 ⇒ the API falls back to the nearest/latest pose. Identity
    // fallback keeps camera-frame data flowing if the RT tree isn't ready yet (better than dropping
    // the frame); placement just won't be world-correct.
    const std::uint64_t capture_ts = ts_opt ? ts_opt.value() : 0;
    Mat::RTMat world_T_cam = Mat::RTMat::Identity();
    bool have_tf = false;
    if (inner_eigen_)
        if (const auto T = inner_eigen_->get_transformation_matrix(world_frame_, camera_frame_, capture_ts); T.has_value())
        {
            world_T_cam = T.value();
            have_tf = true;
        }

    std::vector<SkeletonBody> bodies;
    bodies.reserve(static_cast<std::size_t>(count));
    for (int b = 0; b < count; ++b)
    {
        SkeletonBody body;
        body.id = (b < static_cast<int>(ids.size())) ? static_cast<int>(std::lround(ids[b])) : b;
        body.kp.resize(K, 3);

        std::array<float, K> cf{};
        for (int j = 0; j < K; ++j)
        {
            const std::size_t base = (static_cast<std::size_t>(b) * K + j) * 3;
            const float x = xyz[base + 0], y = xyz[base + 1], z = xyz[base + 2];
            if (std::isfinite(x) and std::isfinite(y) and std::isfinite(z))
            {
                const Mat::Vector3d p = world_T_cam * Mat::Vector3d(x, y, z);
                body.kp(j, 0) = static_cast<float>(p.x());
                body.kp(j, 1) = static_cast<float>(p.y());
                body.kp(j, 2) = static_cast<float>(p.z());
            }
            else
            {
                body.kp(j, 0) = body.kp(j, 1) = body.kp(j, 2) = std::numeric_limits<float>::quiet_NaN();
            }
            // Producer conf is [0,1]; the fitter convention (and replay) is [0,100].
            const std::size_t cidx = static_cast<std::size_t>(b) * K + j;
            cf[j] = (cidx < conf.size()) ? conf[cidx] * 100.0f : 0.0f;
        }
        body.conf = cf;
        bodies.push_back(std::move(body));
    }

    if (not have_tf and not bodies.empty())
        std::print("human_concept: [dsr] no {}<-{} transform yet — skeletons left in camera frame\n",
                   world_frame_, camera_frame_);
    return bodies;
}

std::unique_ptr<SkeletonSource> make_skeleton_source(const std::string& kind,
                                                     const std::string& replay_path,
                                                     bool replay_loop,
                                                     std::shared_ptr<DSR::DSRGraph> graph,
                                                     DSR::InnerEigenAPI* inner_eigen)
{
    if (kind == "dsr" or kind == "live")
    {
        if (graph)
            return std::make_unique<DsrSkeletonSource>(std::move(graph), inner_eigen);
        std::print("human_concept: [source] kind '{}' needs the DSR graph — falling back to replay\n", kind);
        return std::make_unique<ReplaySkeletonSource>(replay_path, replay_loop);
    }
    if (kind != "replay")
        std::print("human_concept: [source] unknown kind '{}' — defaulting to replay\n", kind);
    return std::make_unique<ReplaySkeletonSource>(replay_path, replay_loop);
}

}  // namespace rc
