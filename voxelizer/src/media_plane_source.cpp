#include "media_plane_source.h"

#include "../../common/media_transport/media_transport.h"
#include "../../common/media_transport/lidar_plane_reader.h"

#include <genericworker.h>          // DSR graph API + generated cam_* attribute types

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <print>
#include <string>
#include <utility>

MediaPlaneSource::MediaPlaneSource(std::shared_ptr<DSR::DSRGraph> graph)
    : graph_(std::move(graph))
{
}

MediaPlaneSource::~MediaPlaneSource() = default;   // here: rc::media::* subscriber types are complete

bool MediaPlaneSource::init_media_plane(std::uint32_t domain_id,
                                      const std::string& rgb_topic,
                                      const std::string& depth_topic)
{
    media_rgb_sub_   = std::make_unique<rc::media::MediaSubscriber>();
    media_depth_sub_ = std::make_unique<rc::media::MediaSubscriber>();

    rc::media::SubscriberConfig rgb_cfg;
    rgb_cfg.domain_id     = domain_id;
    rgb_cfg.topic_name    = rgb_topic;
    rgb_cfg.history_depth = 8;
    rc::media::SubscriberConfig depth_cfg = rgb_cfg;
    depth_cfg.topic_name = depth_topic;

    // Prefer the producer's media descriptor on the "zed" node — take the FULL config
    // (domain, topic, history_depth, shared_memory_only, data_sharing) so this consumer
    // tracks the producer's QoS automatically; fall back to the configured domain/topic
    // (with defaults) if the descriptor isn't advertised yet.
    if (graph_)
        if (auto desc = rc::media::descriptor_from_graph(*graph_, "zed"); desc.has_value())
        {
            if (auto c = desc->subscriber_config("rgb");   c.has_value()) rgb_cfg   = *c;
            if (auto c = desc->subscriber_config("depth"); c.has_value()) depth_cfg = *c;
        }

    const bool rgb_ok   = media_rgb_sub_->init(rgb_cfg);
    const bool depth_ok = media_depth_sub_->init(depth_cfg);
    if (!rgb_ok || !depth_ok)
    {
        std::print(stderr, "[voxelizer] media plane subscriber init FAILED (rgb={}, depth={})\n", rgb_ok, depth_ok);
        return false;
    }
    std::print("[voxelizer] media plane ready rgb domain={} '{}' | depth domain={} '{}' data_sharing={}\n",
               rgb_cfg.domain_id, rgb_cfg.topic_name, depth_cfg.domain_id, depth_cfg.topic_name,
               media_rgb_sub_->data_sharing_active() && media_depth_sub_->data_sharing_active());
    return true;
}

void MediaPlaneSource::drain_media_plane() const
{
    // Diagnostic: positively confirm media-plane reception (mirrors robot_concept's
    // producer-side "[Media] 5s stats"). Accumulate poll() delivery counts and print
    // every 5 s on stdout, alongside [Tracks].
    static std::uint64_t rx_rgb = 0, rx_depth = 0;
    static auto last_rx_report = std::chrono::steady_clock::now();

    if (media_rgb_sub_)
    {
        rx_rgb += media_rgb_sub_->poll([this](const rc::media::ImageFrame& f, std::int64_t)
        {
            rx_rgb_total_.fetch_add(1, std::memory_order_relaxed);   // EXACT source rate (see header)
            const int w = static_cast<int>(f.width());
            const int h = static_cast<int>(f.height());
            if (w <= 0 || h <= 0)
                return;
            const std::size_t npix = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
            if (f.format() == rc::media::FORMAT_BGR8 || f.format() == rc::media::FORMAT_RGB8)
            {
                if (f.size() < npix * 3)
                    return;
                cv::Mat view(h, w, CV_8UC3, const_cast<std::uint8_t*>(f.data().data()));
                ZedRgbFrame rgb{.bgr = {}, .width = w, .height = h};
                if (f.format() == rc::media::FORMAT_RGB8)
                    cv::cvtColor(view, rgb.bgr, cv::COLOR_RGB2BGR);   // owned buffer (not the transient DDS view)
                else
                    rgb.bgr = view.clone();
                const std::uint64_t st = f.stamp_ms();
                zed_buf_.put_rgb(std::move(rgb), st);   // keyed by the grab stamp
                latest_rgb_stamp_.store(st, std::memory_order_relaxed);
            }
        });
    }

    if (media_depth_sub_)
    {
        rx_depth += media_depth_sub_->poll([this](const rc::media::ImageFrame& f, std::int64_t)
        {
            const int w = static_cast<int>(f.width());
            const int h = static_cast<int>(f.height());
            if (w <= 0 || h <= 0)
                return;
            const std::size_t npix = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
            ZedDepthFrame depth{.depth = {}, .width = w, .height = h};
            if (f.format() == rc::media::FORMAT_DEPTH_F32)
            {
                if (f.size() < npix * sizeof(float))
                    return;
                const float* p = reinterpret_cast<const float*>(f.data().data());
                depth.depth.assign(p, p + npix);
            }
            else if (f.format() == rc::media::FORMAT_Z16)
            {
                if (f.size() < npix * sizeof(std::uint16_t))
                    return;
                const std::uint16_t* p = reinterpret_cast<const std::uint16_t*>(f.data().data());
                depth.depth.resize(npix);
                for (std::size_t i = 0; i < npix; ++i)
                    depth.depth[i] = static_cast<float>(p[i]) * 0.001f;  // mm -> m fallback
            }
            else
                return;
            const std::uint64_t st = f.stamp_ms();
            zed_buf_.put_depth(std::move(depth), st);   // keyed by the same grab stamp
            latest_depth_stamp_.store(st, std::memory_order_relaxed);
        });
    }

    const auto now_rx = std::chrono::steady_clock::now();
    if (now_rx - last_rx_report >= std::chrono::seconds(5))
    {
        // data_sharing=1 ⇒ true zero-copy SHM loans on BOTH streams; 0 ⇒ SHM-transport memcpy.
        const bool ds = media_rgb_sub_ and media_depth_sub_
                        and media_rgb_sub_->data_sharing_active()
                        and media_depth_sub_->data_sharing_active();
        std::println("[MediaRx] 5s stats rgb={} depth={} rgb_stamp={} depth_stamp={} data_sharing={}",
                     rx_rgb, rx_depth, latest_rgb_stamp_.load(), latest_depth_stamp_.load(), ds ? 1 : 0);
        rx_rgb = rx_depth = 0;
        last_rx_report = now_rx;
    }
}

bool MediaPlaneSource::init_ricoh_media_plane(std::uint32_t domain_id, const std::string& topic)
{
    // Prefer the producer's descriptor on the "ricoh" node (its authored media
    // domain/topic + Image360Frame type); fall back to the configured domain/topic.
    if (graph_)
        media_ricoh_sub_ = rc::media::make_image360_subscriber_from_graph(*graph_, "ricoh", "rgb360");

    if (!media_ricoh_sub_)
    {
        rc::media::SubscriberConfig cfg;
        cfg.domain_id     = domain_id;
        cfg.topic_name    = topic;
        cfg.history_depth = 8;
        media_ricoh_sub_ = std::make_unique<rc::media::Image360Subscriber>();
        if (!media_ricoh_sub_->init(cfg))
        {
            std::print(stderr, "[voxelizer] ricoh media subscriber init FAILED (domain={}, '{}')\n",
                       domain_id, topic);
            media_ricoh_sub_.reset();
            return false;
        }
        std::print("[voxelizer] ricoh media plane ready (fallback) domain={} '{}'\n", domain_id, topic);
    }
    return true;
}

void MediaPlaneSource::poll_ricoh(bool force)
{
    if (!media_ricoh_sub_)
        return;
    const bool wanted = force || ricoh_wanted_.load(std::memory_order_relaxed);
    media_ricoh_sub_->poll([this, wanted](const rc::media::Image360Frame& f, std::int64_t)
    {
        ricoh_last_stamp_ms_.store(f.stamp_ms(), std::memory_order_relaxed);   // cheap — feeds the RGB360 rate HUD
        rx_ricoh_total_.fetch_add(1, std::memory_order_relaxed);                // EXACT source rate (see header)
        if (!wanted)                       // window hidden: drain + discard, no decode/clone cost
            return;
        const int w = static_cast<int>(f.width());
        const int h = static_cast<int>(f.height());
        if (w <= 0 || h <= 0)
            return;
        const std::size_t npix = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
        if (f.size() < npix * 3)
            return;
        cv::Mat view(h, w, CV_8UC3, const_cast<std::uint8_t*>(f.data().data()));
        std::scoped_lock lk(media_ricoh_mutex_);   // may race a concurrent ricoh_bgr_copy() from another thread
        if (f.format() == rc::media::IMG360_FORMAT_RGB8)
            cv::cvtColor(view, media_ricoh_.bgr, cv::COLOR_RGB2BGR);
        else                               // IMG360_FORMAT_BGR8 (producer default) or unspecified
            media_ricoh_.bgr = view.clone();
        media_ricoh_.width    = w;
        media_ricoh_.height   = h;
        media_ricoh_.stamp    = f.stamp_ms();
        media_ricoh_.frame_id = f.frame_id();
        media_ricoh_.valid    = true;
    });
}

cv::Mat MediaPlaneSource::ricoh_bgr_copy() const
{
    std::scoped_lock lk(media_ricoh_mutex_);
    return media_ricoh_.bgr;   // shallow (refcounted) copy — cheap, safe to use after the lock is released
}

bool MediaPlaneSource::init_lidar_media_plane(DSR::InnerEigenAPI* inner_eigen, std::uint32_t /*domain_id*/, const std::string& /*topic*/, bool use_media)
{
    lidar_use_media_ = use_media;
    if (!use_media)
    {
        std::print("[voxelizer] lidar media plane DISABLED (Voxel.lidar_use_media=false) — no LiDAR source\n");
        return false;
    }

    // Shared multi-plane reader (the same one every agent uses): prefers the two per-device planes
    // helios+bpearl (DEVICE frame), transformed to the robot frame + merged; falls back to the fused
    // "lidar3D" plane. Domain/topic come from each plane's media descriptor (no config). Subscribers
    // come up lazily inside get_lidar3D()->poll() — nothing touches DDS here. inner_eigen (passed by
    // SceneProcessor, whose configure() ran first) backs the device->robot RT transform.
    lidar_reader_ = std::make_unique<rc::media::LidarPlaneReader>(
        graph_, inner_eigen, std::vector<std::string>{"helios", "bpearl"}, "lidar3D", "lidar");
    std::print("[voxelizer] lidar media plane ready (shared reader: helios+bpearl → robot, lidar3D fallback)\n");
    return true;
}

std::optional<LidarData> MediaPlaneSource::get_lidar3D(const std::string& robot_name)
{
    if (!lidar_use_media_ || !lidar_reader_)
        return std::nullopt;

    // Diagnostic (every 5 s): fresh = merged sweeps this window; served = cycles that returned a scan.
    // fresh==0 while served>0 ⇒ serving a STALE scan (producer stopped publishing).
    static std::uint64_t fresh = 0, served = 0;
    static auto last_report = std::chrono::steady_clock::now();

    // Merge helios+bpearl (or fused lidar3D) into the ROBOT frame; callers apply the dynamic
    // room<-robot pose at the scan stamp (interpolate=false here — only static mount edges crossed).
    if (!robot_name.empty())
        if (auto sweep = lidar_reader_->poll(robot_name, /*interpolate=*/false);
            sweep.has_value() && !sweep->points.empty())
        {
            // Count a PRIMARY-PLANE arrival, not a merged sweep — see rx_lidar_plane0_total_ in the header.
            if (not sweep->plane_stamp_ms.empty())
            {
                const auto p0 = static_cast<std::uint64_t>(std::max<std::int64_t>(0, sweep->plane_stamp_ms[0]));
                if (p0 != 0 and p0 != lidar_plane0_last_stamp_.load(std::memory_order_relaxed))
                {
                    lidar_plane0_last_stamp_.store(p0, std::memory_order_relaxed);
                    rx_lidar_plane0_total_.fetch_add(1, std::memory_order_relaxed);
                }
            }
            LidarData ld;
            ld.xs.reserve(sweep->points.size());
            ld.ys.reserve(sweep->points.size());
            ld.zs.reserve(sweep->points.size());
            for (const auto& p : sweep->points)
            {
                ld.xs.push_back(p.x());
                ld.ys.push_back(p.y());
                ld.zs.push_back(p.z());
            }
            ld.plane_id = sweep->plane_id;   // per-point source plane (helios=0, bpearl=1) for viewer colouring
            ld.timestamp_ms = static_cast<std::uint64_t>(sweep->stamp_ms);
            media_lidar_ = std::move(ld);
            media_lidar_valid_ = true;
            ++fresh;
        }

    std::optional<LidarData> out;
    if (media_lidar_valid_ && !media_lidar_.xs.empty())
    {
        out = media_lidar_;
        ++served;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - last_report >= std::chrono::seconds(5))
    {
        std::println("[LidarSrc] 5s media fresh={} served={} ({} pts)",
                     fresh, served, out ? out->xs.size() : 0u);
        fresh = served = 0;
        last_report = now;
    }
    return out;
}

std::uint64_t MediaPlaneSource::get_frame_timestamp_ms() const
{
    if (const auto t = last_frame_ts_.load(std::memory_order_relaxed); t != 0)
        return t;   // stamp of the last aligned RGBD we assembled
    if (const auto r = latest_rgb_stamp_.load(std::memory_order_relaxed); r != 0)
        return r;
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

// Max stamp gap (ms) for an rgb/depth pair to be considered the SAME grab. They share the grab stamp,
// so this only tolerates ms rounding jitter — a full-frame gap (a dropped depth) fails the match and
// the cycle is skipped rather than pairing rgb-N with depth-(N-1).
static constexpr std::uint64_t kZedPairMaxDiffMs = 20;

std::optional<RGBDData> MediaPlaneSource::get_rgbd_frame_from_dsr() const
{
    // Pixels arrive over the zero-copy media plane. RGB and depth are two streams sharing the per-grab
    // acquisition stamp; read the time-ALIGNED pair so the mask always meets its own frame's depth.
    drain_media_plane();

    // Reference stamp = the newest rgb ACTUALLY committed to the buffer (put is async, so the just-put
    // frame may not be visible yet — take what's landed), then read the depth aligned to it.
    const std::uint64_t ref = zed_buf_.newest_rgb_stamp();
    if (ref == 0)
        return std::nullopt;

    auto pair = zed_buf_.read_aligned(ref, kZedPairMaxDiffMs);
    if (!pair.has_value())
        return std::nullopt;   // no depth within one grab of this rgb → skip (no skewed pairing)
    auto& [rgb, depth] = *pair;
    if (rgb.bgr.empty() || depth.depth.empty())
        return std::nullopt;

    const int rgb_w = rgb.width;
    const int rgb_h = rgb.height;
    const int depth_w = depth.width;
    const int depth_h = depth.height;
    if (rgb_w <= 0 || rgb_h <= 0 || depth_w <= 0 || depth_h <= 0)
        return std::nullopt;

    // Voxel pipeline assumes one xyz point per RGB pixel.
    if (depth_w != rgb_w || depth_h != rgb_h)
    {
        qWarning() << "RGB/depth resolution mismatch. RGB=" << rgb_w << "x" << rgb_h
                   << " depth=" << depth_w << "x" << depth_h;
        return std::nullopt;
    }

    const std::size_t depth_size = static_cast<std::size_t>(depth_w) * static_cast<std::size_t>(depth_h);
    if (depth.depth.size() < depth_size)
        return std::nullopt;

    // Camera intrinsics from the static 'zed' node (not per-frame).
    float focal_x = 0.f;
    float focal_y = 0.f;
    if (graph_)
    {
        if (auto zed_node = graph_->get_node("zed"); zed_node.has_value())
        {
            if (auto fx = graph_->get_attrib_by_name<cam_depth_focalx_att>(zed_node.value()); fx.has_value())
                focal_x = static_cast<float>(fx.value());
            if (auto fy = graph_->get_attrib_by_name<cam_depth_focaly_att>(zed_node.value()); fy.has_value())
                focal_y = static_cast<float>(fy.value());
        }
    }
    if (focal_x <= 0.f || focal_y <= 0.f)
        return std::nullopt;

    RGBDData data;
    data.bgr     = rgb.bgr.clone();   // owned copy for downstream
    data.width   = rgb_w;
    data.height  = rgb_h;
    data.focal_x = focal_x;
    data.focal_y = focal_y;
    data.depth   = std::move(depth.depth);   // `depth` is a throwaway read-copy — move it
    data.depth.resize(depth_size);
    last_frame_ts_.store(ref, std::memory_order_relaxed);
    return data;
}
