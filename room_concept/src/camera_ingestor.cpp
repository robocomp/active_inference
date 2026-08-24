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

CameraIngestor::CameraIngestor(std::shared_ptr<DSR::DSRGraph> graph,
                               std::string camera_node, std::string robot_frame)
    : G_(std::move(graph)), camera_node_(std::move(camera_node)), robot_frame_(std::move(robot_frame))
{}

CameraIngestor::~CameraIngestor() { stop(); }

bool CameraIngestor::bind_camera()
{
    if (not G_) return false;

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
    extrinsic_ok_ = cam_R_robot_.allFinite() and cam_t_robot_.allFinite();

    if (extrinsic_ok_)
        std::print("[imgedge] bound '{}': {}x{} model={} (reproduced to {:.2e} px), cam<-{} t=[{:.3f} {:.3f} {:.3f}]\n",
                   camera_node_, static_cast<int>(model_.width), static_cast<int>(model_.height),
                   model_.kind == CameraModel::Kind::Pinhole ? "pinhole"
                     : (model_.kind == CameraModel::Kind::Equirect ? "equirect" : "cylindrical"),
                   model_err, robot_frame_, cam_t_robot_.x(), cam_t_robot_.y(), cam_t_robot_.z());
    return extrinsic_ok_;
}

void CameraIngestor::start()
{
    if (running_.exchange(true)) return;      // idempotent
    thread_ = std::thread(&CameraIngestor::ingest_loop, this);
}

void CameraIngestor::stop()
{
    if (not running_.exchange(false)) return;
    wake_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    sub_.reset();
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
    if (sub_ or not G_) return false;
    const auto now = std::chrono::steady_clock::now();
    if (now - last_discovery_ < std::chrono::seconds(1)) return false;   // self-throttle
    last_discovery_ = now;
    // Descriptor-driven factory: domain + topic come from the media_descriptor JSON the PRODUCER
    // authored on the node. Never a config entry (CLAUDE.md).
    sub_ = rc::media::make_image_subscriber_from_graph(*G_, camera_node_, "rgb");
    return sub_ != nullptr;
}

bool CameraIngestor::ingest_pump()
{
    if (not sub_) { try_discover(); return false; }

    const int delivered = sub_->poll([this](const rc::media::ImageFrame& f, std::int64_t)
    {
        const int w = static_cast<int>(f.width());
        const int h = static_cast<int>(f.height());
        if (w <= 0 or h <= 0) return;
        const std::size_t npix = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);

        // Convert to grey HERE, on the ingest thread: the boundary payload becomes a third the size
        // and the localizer never has to know about channel order.
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

        // Per-frame sensor noise, MEASURED. It is the denominator of every precision this subsystem
        // reports, so it must track auto-exposure rather than be pinned in a config file.
        const float sigma_i = rc::img::estimate_noise_sigma_immerkaer(gray.data(), w, h);

        {
            std::lock_guard<std::mutex> lk(frame_mtx_);
            frame_.gray    = std::move(gray);
            frame_.width   = w;
            frame_.height  = h;
            frame_.stamp   = static_cast<std::uint64_t>(f.stamp_ms());
            frame_.sigma_i = sigma_i;
            frame_.valid   = true;
            frame_fresh_   = true;
        }
        last_frame_wall_ms_.store(now_ms(), std::memory_order_relaxed);
        frames_.fetch_add(1, std::memory_order_relaxed);
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
