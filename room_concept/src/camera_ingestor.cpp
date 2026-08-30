/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify it under
 *    the terms of the GNU General Public License as published by the Free
 *    Software Foundation, either version 3 of the License, or (at your option)
 *    any later version. See <http://www.gnu.org/licenses/>.
 */

#include "camera_ingestor.h"
#include <pthread.h>   // pthread_setname_np: name the worker so a per-thread CPU sample attributes itself

#include <cstring>
#include <print>

#include <QDebug>

#include <dsr/api/dsr_camera_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include <media_transport/media_transport.h>

#include "camera_jacobian.h"
#include "image_edge_ops.h"

namespace rc
{
namespace
{
    std::int64_t now_ms()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch()).count();
    }
}   // namespace

CameraIngestor::CameraIngestor(std::shared_ptr<DSR::DSRGraph> graph, std::string camera_node)
    : G_(std::move(graph)), camera_node_(std::move(camera_node))
{}

CameraIngestor::~CameraIngestor() { stop(); }

bool CameraIngestor::bind_camera(const std::string& robot_frame)
{
    if (not G_) return false;
    if (robot_frame.empty())
        return false;      // the type-"robot" node has not been resolved yet; caller retries
    robot_frame_ = robot_frame;

    // ── The camera node + its intrinsics ─────────────────────────────────────────────────────────
    const auto node = G_->get_node(camera_node_);
    if (not node.has_value())
    {
        qWarning() << "[imgedge] camera node" << QString::fromStdString(camera_node_) << "not in the graph yet";
        return false;
    }
    if (not camera_api_)
        camera_api_ = G_->get_camera_api(node.value());
    if (not camera_api_)
    {
        qWarning() << "[imgedge] could not bind CameraAPI to" << QString::fromStdString(camera_node_);
        return false;
    }

    // Reduce to plain numbers, by ASKING project(), not by assuming the panorama convention.
    float model_err = 0.f;
    model_ = rc::img::calibrate_camera_model(*camera_api_, &model_err);
    if (not model_.valid)
    {
        qWarning() << "[imgedge] CameraModel failed to reproduce CameraAPI::project() (max err"
                   << model_err << "px) — RGB edge term stays OFF. The panorama convention may have "
                      "changed; fix calibrate_camera_model() rather than guessing a sign.";
        return false;
    }

    // ── The static camera <- robot extrinsic ─────────────────────────────────────────────────────
    // ★ MAIN THREAD ONLY: ts == 0 sets use_cache=true inside InnerEigenAPI and touches an UNLOCKED
    //   map (CLAUDE.md). Read once here, cache, and never query it from the compute/ingest thread.
    // ★ This is the ONLY transform the measurement path takes from the graph. room <- robot is the
    //   STATE VARIABLE; reading it here would make the residual a function of the answer.
    auto inner = G_->get_inner_eigen_api();
    if (not inner)
    {
        qWarning() << "[imgedge] no InnerEigenAPI";
        return false;
    }
    const auto cam_T_robot = inner->get_transformation_matrix(
        camera_node_, robot_frame_, 0, "RT", DSR::RT_API::TimeQuery::Nearest);
    if (not cam_T_robot.has_value())          // ALWAYS check the optional (CLAUDE.md)
    {
        qWarning() << "[imgedge] RT chain" << QString::fromStdString(camera_node_) << "<-"
                   << QString::fromStdString(robot_frame_) << "not resolvable yet";
        return false;
    }
    const Eigen::Matrix4d M = cam_T_robot.value().matrix();
    cam_R_robot_ = M.block<3, 3>(0, 0).cast<float>();
    cam_t_robot_ = M.block<3, 1>(0, 3).cast<float>();

    // ── Measured boresight-yaw correction, applied LOCALLY ───────────────────────────────────────
    // ★ THE AXIS IS DERIVED, NOT ASSUMED. A yaw is a rotation about the VERTICAL, and which camera
    //   axis that is depends on the camera's internal convention (the ZED is x-right, y-DEPTH,
    //   z-up — but ricoh is not, and the file header above already warns "fix the model rather
    //   than guessing a sign"). So take the robot's own up-axis and express it in camera
    //   coordinates: cam_R_robot * e_z IS the vertical, whatever the camera calls it. Rotating
    //   about that is unambiguously a pan, for any camera on any mount.
    // ★ LEFT-multiplied: R' = Rot(axis, d) * R turns the CAMERA after the robot->camera mapping,
    //   which is a boresight. Right-multiplying would rotate the ROBOT frame instead — that is a
    //   pose correction wearing a mount's clothes, and it would fight the localiser rather than
    //   inform it.
    if (std::abs(mount_yaw_correction_) > 0.f)
    {
        const Eigen::Vector3f up_cam = (cam_R_robot_ * Eigen::Vector3f::UnitZ()).normalized();
        cam_R_robot_ = Eigen::AngleAxisf(mount_yaw_correction_, up_cam).toRotationMatrix()
                     * cam_R_robot_;
        std::print("[imgedge] boresight yaw correction {:+.5f} rad ({:+.3f} deg) applied about the "
                   "robot vertical, in camera coords [{:+.3f} {:+.3f} {:+.3f}]\n",
                   mount_yaw_correction_, mount_yaw_correction_ * 180.0 / M_PI,
                   up_cam.x(), up_cam.y(), up_cam.z());
    }
    extrinsic_ok_ = cam_R_robot_.allFinite() and cam_t_robot_.allFinite();

    if (extrinsic_ok_)
        std::print("[imgedge] bound '{}': {}x{} model={} (reproduced to {:.2e} px), cam<-{} t=[{:.3f} {:.3f} {:.3f}]\n",
                   camera_node_, static_cast<int>(model_.width), static_cast<int>(model_.height),
                   model_.kind == CameraModel::Kind::Pinhole ? "pinhole"
                     : (model_.kind == CameraModel::Kind::Equirect ? "equirect" : "cylindrical"),
                   model_err, robot_frame_, cam_t_robot_.x(), cam_t_robot_.y(), cam_t_robot_.z());
    return extrinsic_ok_;
}

Eigen::Vector3f CameraIngestor::ray_from_pixel(double u, double v) const
{
    if (not camera_api_) return Eigen::Vector3f::Zero();
    return camera_api_->ray_from_pixel(u, v).cast<float>();
}

void CameraIngestor::start()
{
    if (running_.exchange(true)) return;      // idempotent
    thread_ = std::thread([this]
    {
        // Name the thread so a per-thread CPU sample (/proc/<pid>/task/*/stat) attributes itself.
        // There is one ingestor per camera and they cost different amounts, so carry the node name.
        const auto tname = ("cam:" + camera_node_).substr(0, 15);
        pthread_setname_np(pthread_self(), tname.c_str());
        ingest_loop();
    });
}

void CameraIngestor::stop()
{
    if (not running_.exchange(false)) return;
    wake_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    sub_.reset();
    sub360_.reset();
}

void CameraIngestor::ingest_loop()
{
    // Tight drain (mirrors CameraVisualizer::ingest_loop / LidarIngestor::ingest_loop): keep the
    // RELIABLE reader's SHM pool empty, or the producer's loan_sample() eventually fails and it
    // silently stops publishing, freezing every other consumer on the plane.
    while (running_.load(std::memory_order_acquire))
    {
        if (ingest_pump()) continue;
        std::unique_lock<std::mutex> lk(wake_mtx_);
        wake_cv_.wait_for(lk, std::chrono::milliseconds(5),
                          [this] { return not running_.load(std::memory_order_acquire); });
    }
}

bool CameraIngestor::try_discover()
{
    if (sub_ or sub360_ or not G_) return false;
    const auto now = std::chrono::steady_clock::now();
    if (now - last_discovery_ < std::chrono::seconds(1)) return false;   // self-throttle
    last_discovery_ = now;

    // ── Which stream, and therefore which READER TYPE, is the producer's statement ────────────────
    // Descriptor-driven: domain + topic come from the media_descriptor JSON the PRODUCER authored on
    // the node, never from a config entry (CLAUDE.md). The stream KEY is read from the same place,
    // for the same reason — the ZED advertises {"rgb","depth"} and the Ricoh advertises {"rgb360"},
    // and those two keys carry DIFFERENT DDS TYPES (ImageFrame vs the ~5.5 MB Image360Frame).
    //
    // Asking the descriptor first also keeps the log clean: the make_*_from_graph factories print to
    // stderr on a miss, so calling one speculatively once a second is a 1 Hz error stream for the
    // life of the run. Here a camera whose descriptor has not been published yet is simply not ready.
    const auto desc = rc::media::descriptor_from_graph(*G_, camera_node_);
    if (not desc.has_value()) return false;         // node/descriptor not up yet — retry next second

    if (desc->streams.contains("rgb360"))
        sub360_ = rc::media::make_image360_subscriber_from_graph(*G_, camera_node_, "rgb360");
    else if (desc->streams.contains("rgb"))
        sub_ = rc::media::make_image_subscriber_from_graph(*G_, camera_node_, "rgb");
    else if (not no_stream_warned_)
    {
        // ONCE. A camera node that publishes only depth (or only a stream key nobody here knows) is
        // a real, permanent condition and it must say so — but exactly one line, not one per second.
        no_stream_warned_ = true;
        std::string keys;
        for (const auto& [k, v] : desc->streams) { if (not keys.empty()) keys += ", "; keys += k; }
        qWarning() << "[imgedge] node" << QString::fromStdString(camera_node_)
                   << "advertises no image stream this subsystem can read (has:"
                   << QString::fromStdString(keys) << ") — the RGB edge term will stay silent";
    }
    return sub_ != nullptr or sub360_ != nullptr;
}

int CameraIngestor::probe_depth(const std::vector<Eigen::Vector2f>& uv, int patch_radius,
                                std::vector<float>& depth_m, std::int64_t& stamp_ms)
{
    depth_m.assign(uv.size(), -1.f);       // -1 = NOT AVAILABLE. Never 0: a 0 here would read as a
    stamp_ms = 0;                          // point at the camera centre and be believed.
    if (not G_ or uv.empty()) return 0;

    if (not sub_depth_)
    {
        const auto desc = rc::media::descriptor_from_graph(*G_, camera_node_);
        if (not desc.has_value()) return 0;                 // descriptor not up yet, retry next tick
        if (not desc->streams.contains("depth"))
        {
            if (not depth_absent_warned_)                   // ONCE: a permanent condition, not a tick
            {
                depth_absent_warned_ = true;
                qWarning() << "[imgedge] node" << QString::fromStdString(camera_node_)
                           << "advertises no 'depth' stream — triple points will carry no range";
            }
            return 0;
        }
        sub_depth_ = rc::media::make_image_subscriber_from_graph(*G_, camera_node_, "depth");
        if (not sub_depth_) return 0;
        // ★ MAX_IMAGE_BYTES is 3686400 = exactly 1280x720x4, so FORMAT_DEPTH_F32 at that resolution
        //   fits with ZERO margin and anything larger is dropped SILENTLY by the plane. Say the size
        //   out loud once, so "producer healthy, plane reads 0 Hz" is diagnosable from the log.
        std::print("[imgedge] depth subscriber up on '{}' (Z16 -> 1.84 MB, F32 -> 3.69 MB at "
                   "1280x720; the plane's ceiling is 3.69 MB)\n", camera_node_);
    }

    ++depth_polls_;
    int filled = 0;
    // poll() drains every pending sample and calls back per frame; the LAST wins, which is what we
    // want (the newest depth). Everything below runs inside the loaned view — no frame copy.
    sub_depth_->poll([&](const rc::media::ImageFrame& f, std::int64_t)
    {
        const int dw = static_cast<int>(f.width()), dh = static_cast<int>(f.height());
        if (dw <= 0 or dh <= 0) return;
        const std::size_t npix = static_cast<std::size_t>(dw) * static_cast<std::size_t>(dh);
        const std::uint32_t fmt = f.format();
        const bool f32 = (fmt == rc::media::FORMAT_DEPTH_F32);
        const bool z16 = (fmt == rc::media::FORMAT_Z16);
        if (not f32 and not z16) return;
        if (f.size() < npix * (f32 ? 4u : 2u)) return;

        // The depth image need not share the RGB resolution. Scale rather than assume — and only
        // uniformly, because a non-uniform difference would mean a different FoV, not a resize.
        const double sx = (model_.width  > 0.f) ? dw / static_cast<double>(model_.width)  : 1.0;
        const double sy = (model_.height > 0.f) ? dh / static_cast<double>(model_.height) : 1.0;

        const auto* p32 = reinterpret_cast<const float*>(f.data().data());
        const auto* p16 = reinterpret_cast<const std::uint16_t*>(f.data().data());
        int local = 0;
        std::vector<float> patch;
        for (std::size_t k = 0; k < uv.size(); ++k)
        {
            const int cu = static_cast<int>(std::lround(uv[k].x() * sx));
            const int cv = static_cast<int>(std::lround(uv[k].y() * sy));
            patch.clear();
            for (int dv = -patch_radius; dv <= patch_radius; ++dv)
                for (int du = -patch_radius; du <= patch_radius; ++du)
                {
                    const int x = cu + du, y = cv + dv;
                    if (x < 0 or y < 0 or x >= dw or y >= dh) continue;
                    const std::size_t idx = static_cast<std::size_t>(y) * dw + x;
                    // Z16 is millimetres with 0 meaning NO RETURN; F32 is metres and may be inf/nan
                    // on a miss. Both are invalid, and neither may be averaged in as a number.
                    const float d = f32 ? p32[idx] : (p16[idx] == 0 ? 0.f : p16[idx] * 1e-3f);
                    if (std::isfinite(d) and d > 0.05f) patch.push_back(d);
                }
            if (patch.empty()) { depth_m[k] = -1.f; continue; }
            // MEDIAN, not mean: a triple point sits where three surfaces meet, so a patch straddling
            // an edge mixes two populations and a mean lands between them, on nothing.
            std::nth_element(patch.begin(), patch.begin() + patch.size() / 2, patch.end());
            depth_m[k] = patch[patch.size() / 2];
            ++local;
        }
        stamp_ms = f.stamp_ms();
        filled = local;
    });
    if (filled > 0) ++depth_hits_;
    return filled;
}

void CameraIngestor::absorb_gray(std::vector<std::uint8_t> gray, int w, int h, std::int64_t stamp_ms)
{
    // Per-frame sensor noise, MEASURED. It is the denominator of every precision this subsystem
    // reports, so it must track auto-exposure rather than be pinned in a config file.
    const float sigma_i = rc::img::estimate_noise_sigma_immerkaer(gray.data(), w, h);

    {
        std::lock_guard<std::mutex> lk(frame_mtx_);
        frame_.gray    = std::move(gray);
        frame_.width   = w;
        frame_.height  = h;
        frame_.stamp   = static_cast<std::uint64_t>(stamp_ms);
        frame_.sigma_i = sigma_i;
        frame_.valid   = true;
        frame_fresh_   = true;
    }
    last_frame_wall_ms_.store(now_ms(), std::memory_order_relaxed);
    frames_.fetch_add(1, std::memory_order_relaxed);
}

// Is this frame due for conversion? Counts every frame it is asked about, so the ratio of converted
// to delivered is observable. The clock is the FRAME's own stamp, not wall time: a stalled producer
// must not accumulate "credit" and then convert a burst it has no use for.
bool CameraIngestor::convert_due(std::int64_t stamp_ms) noexcept
{
    ++conv_seen_;
    if (min_convert_ms_ <= 0) { ++conv_done_; return true; }
    if (last_convert_ms_ != 0 and stamp_ms - last_convert_ms_ < min_convert_ms_
        and stamp_ms >= last_convert_ms_)                       // a stamp going backwards = new run
        return false;
    last_convert_ms_ = stamp_ms;
    ++conv_done_;
    return true;
}

bool CameraIngestor::ingest_pump()
{
    if (not sub_ and not sub360_) { try_discover(); return false; }

    // Convert to grey HERE, on the ingest thread: the boundary payload becomes a third the size and
    // the localizer never has to know about channel order. Both readers land in absorb_gray(), so
    // the two camera kinds cannot drift apart in how their frames are measured.
    if (sub360_)
    {
        const int delivered = sub360_->poll([this](const rc::media::Image360Frame& f, std::int64_t)
        {
            const int w = static_cast<int>(f.width());
            const int h = static_cast<int>(f.height());
            if (w <= 0 or h <= 0) return;
            const std::size_t npix = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
            if (f.size() < npix * 3) return;
            // Returning here still ENDS THE LOAN, so the pool drains exactly as before — only the
            // ~5.5 MB grey conversion is skipped.
            if (not convert_due(f.stamp_ms())) return;
            // ⚠ The 360 format enum is its OWN numbering — IMG360_FORMAT_BGR8 == 0 is the producer
            //   default, whereas the plain-image enum numbers them differently. Comparing an
            //   Image360Frame's format against rc::media::FORMAT_BGR8 would compile and be wrong.
            std::vector<std::uint8_t> gray;
            rc::img::gray_from_rgb8(f.data().data(), w, h,
                                    f.format() == rc::media::IMG360_FORMAT_BGR8, gray);
            absorb_gray(std::move(gray), w, h, f.stamp_ms());
        });
        return delivered > 0;
    }

    const int delivered = sub_->poll([this](const rc::media::ImageFrame& f, std::int64_t)
    {
        const int w = static_cast<int>(f.width());
        const int h = static_cast<int>(f.height());
        if (w <= 0 or h <= 0) return;
        const std::size_t npix = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
        if (not convert_due(f.stamp_ms())) return;   // loan still released; conversion skipped

        std::vector<std::uint8_t> gray;
        switch (f.format())
        {
            case rc::media::FORMAT_RGB8:
            case rc::media::FORMAT_BGR8:
            {
                if (f.size() < npix * 3) return;
                rc::img::gray_from_rgb8(f.data().data(), w, h,
                                        f.format() == rc::media::FORMAT_BGR8, gray);
                break;
            }
            case rc::media::FORMAT_GRAY8:
            {
                if (f.size() < npix) return;
                gray.resize(npix);
                std::memcpy(gray.data(), f.data().data(), npix);
                break;
            }
            default: return;      // depth / unknown are not consumed here
        }
        absorb_gray(std::move(gray), w, h, f.stamp_ms());
    });
    return delivered > 0;
}

bool CameraIngestor::take_latest(GrayFrame& out)
{
    std::lock_guard<std::mutex> lk(frame_mtx_);
    if (not frame_fresh_ or not frame_.ok()) return false;
    out = std::move(frame_);          // vector MOVE: no copy, and value semantics across the boundary
    frame_.valid = false;
    frame_.gray.clear();
    frame_fresh_ = false;
    return true;
}

std::int64_t CameraIngestor::ms_since_last_frame() const noexcept
{
    const auto t = last_frame_wall_ms_.load(std::memory_order_relaxed);
    return t == 0 ? -1 : (now_ms() - t);
}

}  // namespace rc
