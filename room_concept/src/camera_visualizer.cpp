#include "camera_visualizer.h"
#include <pthread.h>   // pthread_setname_np: name the worker so a per-thread CPU sample attributes itself

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_camera_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_rt_api.h>
#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include "../../common/media_transport/media_transport.h"

#include <QBrush>
#include <QColor>
#include <QFontMetrics>
#include <QPainter>
#include <QPolygonF>
#include <QHBoxLayout>
#include <QVBoxLayout>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <print>
#include <string>
#include <string_view>

namespace rc {

namespace {

struct RoomToCameraBasis
{
    Mat::Vector3d origin{0.0, 0.0, 0.0};
    Mat::Vector3d axis_x{0.0, 0.0, 0.0};
    Mat::Vector3d axis_y{0.0, 0.0, 0.0};
    Mat::Vector3d axis_z{0.0, 0.0, 0.0};
};

bool compute_room_to_camera_basis(DSR::InnerEigenAPI* inner_eigen_api,
                                  const std::string& camera_node_name,
                                  const std::string& room_frame_name,
                                  std::uint64_t rt_timestamp,
                                  RoomToCameraBasis& basis)
{
    if (inner_eigen_api == nullptr)
        return false;

    const auto origin_opt = inner_eigen_api->transform(camera_node_name, Mat::Vector3d(0.0, 0.0, 0.0), room_frame_name, rt_timestamp);
    const auto x_opt = inner_eigen_api->transform(camera_node_name, Mat::Vector3d(1.0, 0.0, 0.0), room_frame_name, rt_timestamp);
    const auto y_opt = inner_eigen_api->transform(camera_node_name, Mat::Vector3d(0.0, 1.0, 0.0), room_frame_name, rt_timestamp);
    const auto z_opt = inner_eigen_api->transform(camera_node_name, Mat::Vector3d(0.0, 0.0, 1.0), room_frame_name, rt_timestamp);
    if (!origin_opt.has_value() || !x_opt.has_value() || !y_opt.has_value() || !z_opt.has_value())
        return false;

    basis.origin = origin_opt.value();
    basis.axis_x = x_opt.value() - basis.origin;
    basis.axis_y = y_opt.value() - basis.origin;
    basis.axis_z = z_opt.value() - basis.origin;
    return true;
}

Mat::Vector3d transform_room_point(const RoomToCameraBasis& basis, const Mat::Vector3d& point_room)
{
    return basis.origin
         + point_room.x() * basis.axis_x
         + point_room.y() * basis.axis_y
         + point_room.z() * basis.axis_z;
}

// Max forward-prediction horizon for the pose dead-reckoning (s). Caps extrapolation if the
// RT feed stalls or a stamp is bogus, so the overlay can never run away from the image.
constexpr double kMaxPredictHorizonS = 0.25;

// Overlay colour per object category (mirrors the retina's 3D-viewer palette intent:
// tables warm/orange, bottles cyan, generic objects green).
QColor color_for_category(const std::string& category)
{
    if (category == "model_table" || category == "table")        return QColor(255, 165, 0);    // orange
    if (category == "bottle" || category == "cylinder")          return QColor(0, 220, 220);    // cyan
    if (category == "chair")                                     return QColor(200, 120, 255);  // violet
    if (category == "object")                                    return QColor(60, 220, 90);    // green
    return QColor(255, 230, 0);                                                                 // yellow fallback
}

}  // namespace

static std::vector<Eigen::Vector2f> read_room_polygon_from_dsr(const std::shared_ptr<DSRGraph>& graph, const std::string& room_frame_name)
{
    if (!graph)
        return {};

    const auto room_node_opt = graph->get_node(room_frame_name);
    if (!room_node_opt.has_value())
        return {};

    const auto polygon_x_opt = graph->get_attrib_by_name<delimiting_polygon_x_att>(room_node_opt.value());
    const auto polygon_y_opt = graph->get_attrib_by_name<delimiting_polygon_y_att>(room_node_opt.value());
    if (!polygon_x_opt.has_value() || !polygon_y_opt.has_value())
        return {};

    const auto& polygon_x = polygon_x_opt.value();
    const auto& polygon_y = polygon_y_opt.value();
    if (polygon_x.get().size() < 3 || polygon_x.get().size() != polygon_y.get().size())
        return {};

    std::vector<Eigen::Vector2f> polygon;
    polygon.reserve(polygon_x.get().size());
    for (std::size_t i = 0; i < polygon_x.get().size(); ++i)
        polygon.emplace_back(polygon_x.get()[i], polygon_y.get()[i]);

    return polygon;
}

CameraVisualizer::CameraVisualizer(std::shared_ptr<DSRGraph> graph, const std::vector<Eigen::Vector2f>& room_polygon,
                                   std::vector<std::string> overlay_object_types,
                                   std::string camera_node, QWidget* parent)
    : QDialog(parent), graph_(graph), room_polygon_(room_polygon),
      overlay_object_types_(std::move(overlay_object_types))
{
    if (not camera_node.empty()) camera_node_name_ = std::move(camera_node);
    setWindowTitle(QString("%1 — room layout projection")
                       .arg(QString::fromStdString(camera_node_name_).toUpper()));
    setGeometry(100, 100, 800, 600);

    auto* layout = new QVBoxLayout(this);
    image_label_ = new QLabel();
    image_label_->setMinimumSize(640, 480);
    image_label_->setScaledContents(false);
    image_label_->setAlignment(Qt::AlignCenter);
    // ── All / Matched corner filter ──────────────────────────────────────────────────────────────
    // Checkable, and the LABEL is the state rather than a caption beside a tick: this window has no
    // other controls, so there is no settings idiom for a reader to lean on.
    corner_filter_btn_ = new QPushButton(this);
    corner_filter_btn_->setCheckable(true);
    corner_filter_btn_->setChecked(corners_matched_only_);
    corner_filter_btn_->setToolTip(
        "Which corners the overlays draw — BOTH channels, LiDAR and RGB.\n\n"
        "MATCHED (default): only corners matched to a model vertex and believed — the same rules the\n"
        "2-D canvas uses (assoc_prob for LiDAR, occlusion + residual chi2 for RGB), so the two views\n"
        "agree.\n"
        "ALL: every corner the detector produced, including ambiguous ones it is not using and RGB\n"
        "crossings whose vertex stands behind a wall. Switch to this when asking why a corner was\n"
        "dropped.\n\n"
        "Display only: the estimator receives every match either way.");
    const auto sync_btn = [this]
    { corner_filter_btn_->setText(corners_matched_only_ ? "Corners: Matched" : "Corners: All"); };
    sync_btn();
    connect(corner_filter_btn_, &QPushButton::toggled, this, [this, sync_btn](bool on)
    {
        corners_matched_only_ = on;
        sync_btn();
    });
    auto* controls = new QHBoxLayout;
    controls->addWidget(corner_filter_btn_);
    controls->addStretch(1);
    layout->addLayout(controls);
    layout->addWidget(image_label_);
    setLayout(layout);

    if (graph_)
    {
        if (auto zed_node = graph_->get_node(camera_node_name_); zed_node.has_value())
        {
            camera_api_ = graph_->get_camera_api(zed_node.value());
            inner_eigen_api_ = graph_->get_inner_eigen_api();
        }
        else
        {
            image_label_->setText("No 'zed' node found in DSR");
        }
    }

    fetch_camera_intrinsics();

    // Refresh periodically to show the live camera stream and overlay.
    refresh_timer_ = new QTimer(this);
    refresh_timer_->setTimerType(Qt::PreciseTimer);
    connect(refresh_timer_, &QTimer::timeout, this, &CameraVisualizer::update_frame);
    fps_timer_.start();
    timing_window_timer_.start();
}

CameraVisualizer::~CameraVisualizer()
{
    stop_ingest();          // join the ingest thread BEFORE the subscriber it owns is destroyed
    media_rgb_sub_.reset();
}

void CameraVisualizer::set_corner_matches(std::vector<rc::CornerDetector::CornerMatch> matches,
                                          const Eigen::Affine2f& robot_pose)
{
    std::lock_guard<std::mutex> lk(corner_matches_mtx_);
    corner_matches_ = std::move(matches);
    corner_pose_ = robot_pose;
}

void CameraVisualizer::set_triple_points(std::vector<rc::TriplePoint> pts, const std::string& from_camera)
{
    std::lock_guard<std::mutex> lk(triple_mtx_);
    triple_points_ = std::move(pts);
    triple_from_   = from_camera;
}

void CameraVisualizer::start_media_plane()
{
    // Spin up the always-on ingest thread. With a RELIABLE reader, samples that are never
    // take()'d stay pinned in the producer's preallocated SHM pool; once it fills, the
    // producer's loan_sample() fails and it silently stops publishing, freezing every other
    // consumer (e.g. the retina). The thread drains continuously regardless of dialog
    // visibility (rendering is still gated by isVisible() in update_frame()). It also owns
    // the lazy subscriber discovery, so DDS entity bring-up happens off the GUI thread —
    // serialized against the LiDAR ingest thread by media_transport's entity mutex.
    if (ingest_running_.exchange(true))
        return;   // idempotent: already started
    ingest_thread_ = std::thread([this]
    {
        pthread_setname_np(pthread_self(), "camview-ingest");
        ingest_loop();
    });
}

void CameraVisualizer::stop_ingest()
{
    if (!ingest_running_.exchange(false))
        return;   // not running
    ingest_wake_cv_.notify_all();
    if (ingest_thread_.joinable())
        ingest_thread_.join();
}

void CameraVisualizer::ingest_loop()
{
    // Tight drain loop on a dedicated thread (mirrors rc::LidarIngestor::ingest_loop): react
    // to a fresh RGB frame with ~0-2 ms latency and keep the RELIABLE reader's SHM pool empty.
    while (ingest_running_.load(std::memory_order_acquire))
    {
        if (ingest_pump())
            continue;   // drained a frame — try again immediately in case more are queued
        // Idle (no subscriber yet, or no fresh frame): brief wait, woken instantly by stop_ingest().
        std::unique_lock<std::mutex> lk(ingest_wake_mtx_);
        ingest_wake_cv_.wait_for(lk, std::chrono::milliseconds(5),
                                 [this] { return !ingest_running_.load(std::memory_order_acquire); });
    }
}

bool CameraVisualizer::try_discover_media_plane()
{
    // ★ BOTH subscribers. Checking only media_rgb_sub_ meant the ricoh path never saw itself as
    //   discovered and rebuilt its 360 reader once a second for ever, which is not a slow start —
    //   a reader recreated before it delivers never delivers.
    if (media_rgb_sub_ || media_rgb360_sub_ || !graph_)
        return false;

    // Self-throttle discovery attempts (ingest thread, ~200 Hz idle poll).
    const auto now = std::chrono::steady_clock::now();
    if (now - last_media_discovery_attempt_ < std::chrono::seconds(1))
        return false;
    last_media_discovery_attempt_ = now;

    // Shared descriptor-driven factory (same init code as every other agent): verifies
    // the "zed" node + descriptor exist and reads the DDS domain/topic from the JSON.
    // ★ THE STREAM KEY COMES FROM THE NODE'S OWN DESCRIPTOR, not from an assumption. The zed
    //   advertises "rgb" and the ricoh advertises "rgb360", and those two keys carry DIFFERENT DDS
    //   types — ImageFrame against the ~5.5 MB Image360Frame — so asking for the wrong one does not
    //   merely return nothing, it asks the wrong reader for the wrong thing.
    const auto desc = rc::media::descriptor_from_graph(*graph_, camera_node_name_);
    if (not desc.has_value())
    {
        // Not an error: the producer may simply not have advertised yet. Said ONCE, because a
        // window that reports "not initialized" with no reason leaves the reader guessing between
        // a missing node, a missing stream and a failed reader — three different problems.
        if (not discovery_reason_logged_)
        {
            discovery_reason_logged_ = true;
            qWarning() << "[camviz]" << QString::fromStdString(camera_node_name_)
                       << "has no media descriptor yet — retrying every second";
        }
        return false;
    }
    if (desc->streams.contains("rgb360"))
        media_rgb360_sub_ = rc::media::make_image360_subscriber_from_graph(*graph_, camera_node_name_,
                                                                          "rgb360");
    else if (desc->streams.contains("rgb"))
        media_rgb_sub_ = rc::media::make_image_subscriber_from_graph(*graph_, camera_node_name_, "rgb");
    else if (not discovery_reason_logged_)
    {
        discovery_reason_logged_ = true;
        std::string keys;
        for (const auto& [k, v] : desc->streams) { if (not keys.empty()) keys += ", "; keys += k; }
        qWarning() << "[camviz]" << QString::fromStdString(camera_node_name_)
                   << "advertises no stream this window can read (has:"
                   << QString::fromStdString(keys) << ")";
    }
    const bool up = (media_rgb_sub_ != nullptr or media_rgb360_sub_ != nullptr);
    if (up)
    {
        subscriber_ready_.store(true, std::memory_order_release);
        qInfo() << "[camviz]" << QString::fromStdString(camera_node_name_) << "subscriber up on"
                << (media_rgb360_sub_ ? "rgb360" : "rgb");
    }
    else if (not discovery_reason_logged_)
    {
        discovery_reason_logged_ = true;
        qWarning() << "[camviz]" << QString::fromStdString(camera_node_name_)
                   << "descriptor found and the stream is advertised, but the reader could not be"
                      " created — check the media plane domain and MAX_IMAGE_BYTES";
    }
    return up;
}

bool CameraVisualizer::ingest_pump()
{
    if (!media_rgb_sub_ && !media_rgb360_sub_)
    {
        try_discover_media_plane();   // lazy: up once the node's descriptor exists
        return false;
    }

    if (media_rgb360_sub_)
    {
        // The 360 frame is RGB8 by construction (Image360Frame carries no format field), so it is
        // NOT checked against rc::media::FORMAT_* — doing so would compile and be wrong.
        const int got = media_rgb360_sub_->poll([this](const rc::media::Image360Frame& f, std::int64_t)
        {
            const int w = static_cast<int>(f.width()), h = static_cast<int>(f.height());
            if (w <= 0 || h <= 0) return;
            const std::size_t need = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3;
            if (f.data().size() < need) return;
            std::lock_guard<std::mutex> lk(media_rgb_mtx_);
            media_rgb_.bytes.resize(need);
            std::memcpy(media_rgb_.bytes.data(), f.data().data(), need);
            media_rgb_.width  = w;
            media_rgb_.height = h;
            media_rgb_.format = rc::media::FORMAT_RGB8;
            media_rgb_.stamp  = f.stamp_ms();
            media_rgb_.valid  = true;
        });
        return got > 0;
    }

    const int delivered = media_rgb_sub_->poll([this](const rc::media::ImageFrame& f, std::int64_t)
    {
        const int w = static_cast<int>(f.width());
        const int h = static_cast<int>(f.height());
        if (w <= 0 || h <= 0)
            return;

        const std::size_t npix = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
        std::size_t expected = 0;
        switch (f.format())
        {
            case rc::media::FORMAT_BGR8:
            case rc::media::FORMAT_RGB8:  expected = npix * 3; break;
            case rc::media::FORMAT_GRAY8: expected = npix;     break;
            default: return;  // depth / unknown formats are not consumed here
        }
        if (f.size() < expected)
            return;

        // Publish the newest frame to the GUI thread under the cache lock.
        std::lock_guard<std::mutex> lk(media_rgb_mtx_);
        media_rgb_.bytes.resize(expected);
        std::memcpy(media_rgb_.bytes.data(), f.data().data(), expected);
        media_rgb_.width  = w;
        media_rgb_.height = h;
        media_rgb_.format = f.format();
        media_rgb_.stamp  = f.stamp_ms();
        media_rgb_.valid  = true;
    });
    return delivered > 0;
}

void CameraVisualizer::showEvent(QShowEvent* event)
{
    QDialog::showEvent(event);
    callback_gap_timer_.invalidate();
    have_last_source_timestamp_ = false;
    last_source_timestamp_ = 0;
    estimated_source_period_ms_ = 50.0;
    have_estimated_source_period_ = false;
    reset_timing_window();
    if (refresh_timer_ != nullptr && !refresh_timer_->isActive())
        refresh_timer_->start(50);  // target 20 Hz to guarantee >=10 Hz in practice
    update_frame();
}

void CameraVisualizer::hideEvent(QHideEvent* event)
{
    log_timing_summary("hide");
    if (refresh_timer_ != nullptr)
        refresh_timer_->stop();
    QDialog::hideEvent(event);
}

void CameraVisualizer::reset_timing_window()
{
    timing_stats_ = TimingStats{};
    timing_window_timer_.restart();
}

double CameraVisualizer::raw_timestamp_delta_to_ms(std::uint64_t raw_delta)
{
    if (raw_delta == 0)
        return 0.0;

    // Heuristic unit detection: DSR camera timestamps are typically ns, but keep
    // compatibility with ms/us producers.
    if (raw_delta >= 1000000ULL)
        return static_cast<double>(raw_delta) / 1000000.0;
    if (raw_delta >= 1000ULL)
        return static_cast<double>(raw_delta) / 1000.0;
    return static_cast<double>(raw_delta);
}

void CameraVisualizer::adapt_refresh_interval(std::uint64_t raw_delta)
{
    if (refresh_timer_ == nullptr || raw_delta == 0)
        return;

    const double observed_period_ms = raw_timestamp_delta_to_ms(raw_delta);
    if (!std::isfinite(observed_period_ms) || observed_period_ms < 5.0)
        return;

    if (!have_estimated_source_period_)
    {
        estimated_source_period_ms_ = observed_period_ms;
        have_estimated_source_period_ = true;
    }
    else
    {
        constexpr double alpha = 0.25;
        estimated_source_period_ms_ = (1.0 - alpha) * estimated_source_period_ms_ + alpha * observed_period_ms;
    }

    // Poll at roughly 2x the estimated source rate. Matching the source period
    // makes the viewer sensitive to phase drift and can cause a feedback loop
    // where skipped frames inflate the estimated source period and the poll rate
    // collapses toward 4 Hz.
    const int target_interval_ms = std::clamp(static_cast<int>(std::lround(0.5 * estimated_source_period_ms_)), 20, 250);
    if (std::abs(refresh_timer_->interval() - target_interval_ms) >= 5)
        refresh_timer_->setInterval(target_interval_ms);
}

void CameraVisualizer::log_timing_summary(const char* reason)
{
    (void)reason;
}

bool CameraVisualizer::fetch_rgb_from_dsr(QImage& rgb_image, std::uint64_t& frame_timestamp)
{
    frame_timestamp = 0;

    // Camera intrinsics still come from the static 'zed' DSR node (cam_rgb_width/focalx
    // descriptor attributes survive even after the cam_rgb blob is off the graph).
    if (graph_ && !camera_api_)
    {
        if (auto zed_node = graph_->get_node(camera_node_name_); zed_node.has_value())
        {
            camera_api_ = graph_->get_camera_api(zed_node.value());
            inner_eigen_api_ = graph_->get_inner_eigen_api();
            fetch_camera_intrinsics();
        }
    }
    if (!camera_data_.valid)
        fetch_camera_intrinsics();

    // Pixels come from the zero-copy media plane (replaces camera_api_->get_rgb_image()). The
    // ingest thread owns the subscriber and drains DDS; here we only snapshot the newest decoded
    // frame it published. subscriber_ready_ becomes true once that subscriber is live.
    if (!subscriber_ready_.load(std::memory_order_acquire))
    {
        image_label_->setText("Media plane RGB subscriber not initialized");
        return false;
    }

    // Copy the latest frame out under the cache lock, then release it before the (heavier)
    // QImage deep-copy + overlay drawing so the ingest thread is never blocked on rendering.
    int width = 0, height = 0;
    std::uint32_t format = 0;
    {
        std::lock_guard<std::mutex> lk(media_rgb_mtx_);
        if (!media_rgb_.valid)
        {
            image_label_->setText("Waiting for RGB frames on media plane...");
            return false;
        }
        width          = media_rgb_.width;
        height         = media_rgb_.height;
        format         = media_rgb_.format;
        frame_timestamp = media_rgb_.stamp;
        render_bytes_  = media_rgb_.bytes;   // GUI-thread-private copy for the rest of this call
    }

    if (width <= 0 || height <= 0)
    {
        image_label_->setText("Invalid media-plane frame dimensions");
        return false;
    }

    const uchar* data = reinterpret_cast<const uchar*>(render_bytes_.data());
    switch (format)
    {
        case rc::media::FORMAT_BGR8:
        {
            // All Webots-derived RGB producers now tag their true order FORMAT_RGB8 (the branch below),
            // so FORMAT_BGR8 now means GENUINE BGR (e.g. real Theta/ZED hardware) and must swap R/B.
            QImage img(data, width, height, width * 3, QImage::Format_RGB888);
            rgb_image = img.rgbSwapped();
            return true;
        }
        case rc::media::FORMAT_RGB8:
        {
            QImage img(data, width, height, width * 3, QImage::Format_RGB888);
            rgb_image = img.copy();
            return true;
        }
        case rc::media::FORMAT_GRAY8:
        {
            QImage img(data, width, height, width, QImage::Format_Grayscale8);
            rgb_image = img.copy();
            return true;
        }
        default:
            image_label_->setText(QString("Unsupported media RGB format: %1").arg(format));
            return false;
    }
}

bool CameraVisualizer::fetch_camera_intrinsics()
{
    if (!camera_api_)
        return false;

    float fx = camera_api_->get_focal_x();
    float fy = camera_api_->get_focal_y();
    const int width = static_cast<int>(camera_api_->get_width());
    const int height = static_cast<int>(camera_api_->get_height());
    const float cx = static_cast<float>(width) * 0.5f;
    const float cy = static_cast<float>(height) * 0.5f;

    // ★ A PANORAMA HAS NO FOCAL LENGTH, AND ITS ABSENCE IS NOT AN ERROR. cortex sets
    //   focal_x = focal_y = 0 for Equirectangular and Cylindrical deliberately
    //   (dsr_camera_api.cpp:51). Testing fx > 0 as a validity condition is therefore a PINHOLE test
    //   wearing the clothes of a sanity check: on the ricoh it set camera_data_.valid = false and
    //   silently disabled the entire overlay, while the image itself displayed perfectly — contours
    //   and corners simply never drawn, with nothing saying why.
    //   Nothing in the projection needs a focal length: CameraAPI::project() dispatches on the model.
    const bool panoramic =
        camera_api_->get_projection_model() != DSR::CameraAPI::ProjectionModel::Pinhole;
    if (panoramic and (fx <= 0.f or fy <= 0.f) and width > 0)
    {
        // Pixels per RADIAN, which is the panoramic analogue and the only thing K is used for here:
        // turning a metric sigma at range d into an overlay radius (px = (W/2pi) * sigma / d). It is
        // NOT a focal length and must not be read as one — hence the name of the field it feeds is
        // the only place this value is allowed to matter.
        fx = fy = static_cast<float>(width) / (2.f * static_cast<float>(M_PI));
    }

    if (fx <= 0.f || fy <= 0.f || width <= 0 || height <= 0)
    {
        camera_data_.valid = false;
        return false;
    }

    camera_data_.K << fx, 0, cx,
                      0, fy, cy,
                      0, 0, 1;
    camera_data_.width = width;
    camera_data_.height = height;
    camera_data_.valid = true;

    return true;
}

std::vector<Eigen::Vector3f> CameraVisualizer::get_room_corners_3d() const
{
    std::vector<Eigen::Vector3f> corners_3d;
    float room_height = 2.4f;
    auto room_polygon = room_polygon_;

    if (graph_)
    {
        if (auto room_node = graph_->get_node(room_frame_name_); room_node.has_value())
        {
            if (const auto h = graph_->get_attrib_by_name<room_height_att>(room_node.value()); h.has_value())
                room_height = h.value();
        }

        if (auto polygon_from_dsr = read_room_polygon_from_dsr(graph_, room_frame_name_);
            polygon_from_dsr.size() >= 3)
        {
            room_polygon = std::move(polygon_from_dsr);
        }
    }

    // All coords in metres (polygon X/Y from DSR are in metres, room_height in metres).
    for (const auto& corner_2d : room_polygon)
    {
        corners_3d.emplace_back(corner_2d.x(), corner_2d.y(), 0.0f);
        corners_3d.emplace_back(corner_2d.x(), corner_2d.y(), room_height);
    }

    return corners_3d;
}

std::vector<CameraVisualizer::ObjectBox> CameraVisualizer::get_dsr_object_boxes(std::uint64_t rt_timestamp) const
{
    std::vector<ObjectBox> boxes;
    if (!graph_ || !inner_eigen_api_)
        return boxes;

    // Same node family the retina draws in its 3D viewer: generic type()=="object" nodes (concept
    // agents publish tables/chairs/bottles/… as objects, class in object_subtype). Each carries
    // width/depth/height + a room←node RT edge; colour/label are derived per-node from object_subtype.
    const auto build_for = [&](const std::string& node_type)
    {
        for (const auto& node : graph_->get_nodes_by_type(node_type))
        {
            const auto width_opt  = graph_->get_attrib_by_name<width_m_att>(node);
            const auto depth_opt  = graph_->get_attrib_by_name<depth_m_att>(node);
            const auto height_opt = graph_->get_attrib_by_name<height_m_att>(node);
            if (!width_opt.has_value() || !depth_opt.has_value() || !height_opt.has_value())
                continue;

            const float width = width_opt.value(), depth = depth_opt.value(), height = height_opt.value();
            if (width <= 0.f || depth <= 0.f || height <= 0.f)
                continue;

            const auto room_T_object = inner_eigen_api_->get_transformation_matrix(
                room_frame_name_, node.name(), rt_timestamp, "RT", DSR::RT_API::TimeQuery::Nearest);
            if (!room_T_object.has_value())
                continue;

            const float hw = width * 0.5f, hd = depth * 0.5f, hh = height * 0.5f;

            // Concept nodes are now generic type()=="object"; the class is in object_subtype (name prefix
            // unchanged). Read it once and label/anchor by class instead of the old per-class node types.
            std::string subtype;
            if (const auto s = graph_->get_attrib_by_name<object_subtype_att>(node); s.has_value())
                subtype = s.value();
            const auto name_is = [&](std::string_view p) { return std::string_view(node.name()).starts_with(p); };

            // Floor-standing furniture (tables, chairs, fridges) anchors its node origin at the base → box
            // extends upward [0, h]; free objects are center-anchored → [-h/2, h/2]. Matches the retina.
            const bool stands_on_floor = subtype == "table" || subtype == "chair" || subtype == "refrigerator"
                                         || name_is("table") || name_is("chair") || name_is("refrigerator");
            const float z_lo = stands_on_floor ? 0.f     : -hh;
            const float z_hi = stands_on_floor ? height  :  hh;

            const std::array<Eigen::Vector3d, 8> local = {
                Eigen::Vector3d{-hw, -hd, z_lo}, Eigen::Vector3d{ hw, -hd, z_lo},
                Eigen::Vector3d{ hw,  hd, z_lo}, Eigen::Vector3d{-hw,  hd, z_lo},
                Eigen::Vector3d{-hw, -hd, z_hi}, Eigen::Vector3d{ hw, -hd, z_hi},
                Eigen::Vector3d{ hw,  hd, z_hi}, Eigen::Vector3d{-hw,  hd, z_hi}
            };

            ObjectBox box;
            box.node_name = node.name();
            box.category  = node.name();
            if (const auto it = node.attrs().find("semantic_class");
                it != node.attrs().end() && it->second.selected() == 0)
                box.category = it->second.str();
            // Colour/label by class (object_subtype / name), mapping onto the palette's historic keys.
            if (subtype == "table" || name_is("table"))          box.category = "model_table";
            else if (subtype == "bottle" || subtype == "cylinder") box.category = "bottle";
            else if (subtype == "chair" || name_is("chair"))      box.category = "chair";
            else if (!subtype.empty())                            box.category = subtype;

            for (std::size_t i = 0; i < local.size(); ++i)
            {
                const Eigen::Vector3d c = room_T_object->linear() * local[i] + room_T_object->translation();
                box.corners[i] = Eigen::Vector3f(static_cast<float>(c.x()),
                                                 static_cast<float>(c.y()),
                                                 static_cast<float>(c.z()));
            }
            boxes.push_back(std::move(box));
        }
    };

    // Config-driven: project every DSR node type in Overlay.ObjectTypes (default just "object" now that
    // furniture is generic — legacy per-class types resolve to empty). Walls: get_dsr_wall_quads.
    for (const auto& node_type : overlay_object_types_)
        build_for(node_type);
    return boxes;
}

std::vector<CameraVisualizer::WallQuad> CameraVisualizer::get_dsr_wall_quads(std::uint64_t rt_timestamp) const
{
    std::vector<WallQuad> quads;
    if (!graph_ || !inner_eigen_api_)
        return quads;

    for (const auto& node : graph_->get_nodes_by_type("wall"))
    {
        const auto width_opt  = graph_->get_attrib_by_name<width_m_att>(node);   // length along local X
        const auto height_opt = graph_->get_attrib_by_name<height_m_att>(node);  // room height along local Z
        if (!width_opt.has_value() || !height_opt.has_value())
            continue;
        const float L = width_opt.value(), H = height_opt.value();
        if (L <= 0.f || H <= 0.f)
            continue;

        const auto room_T_wall = inner_eigen_api_->get_transformation_matrix(
            room_frame_name_, node.name(), rt_timestamp, "RT", DSR::RT_API::TimeQuery::Nearest);
        if (!room_T_wall.has_value())
            continue;

        const float hw = L * 0.5f, hh = H * 0.5f;
        // Wall plane is local Y=0; X spans the length, Z the height, centred on the RT origin.
        const std::array<Eigen::Vector3d, 4> local = {
            Eigen::Vector3d{-hw, 0.0, -hh},   // bottom-left
            Eigen::Vector3d{ hw, 0.0, -hh},   // bottom-right
            Eigen::Vector3d{ hw, 0.0,  hh},   // top-right
            Eigen::Vector3d{-hw, 0.0,  hh}    // top-left
        };

        WallQuad q;
        q.node_name = node.name();
        for (std::size_t i = 0; i < local.size(); ++i)
        {
            const Eigen::Vector3d c = room_T_wall->linear() * local[i] + room_T_wall->translation();
            q.corners[i] = Eigen::Vector3f(static_cast<float>(c.x()),
                                           static_cast<float>(c.y()),
                                           static_cast<float>(c.z()));
        }
        quads.push_back(std::move(q));
    }
    return quads;
}

std::optional<Eigen::Affine3d> CameraVisualizer::predicted_camera_from_room(std::uint64_t frame_ts) const
{
    if (!graph_ || !inner_eigen_api_)
        return std::nullopt;

    const auto room_node = graph_->get_node(room_frame_name_);
    if (!room_node.has_value())
        return std::nullopt;

    const auto robot_nodes = graph_->get_nodes_by_type("robot");
    if (robot_nodes.empty())
        return std::nullopt;
    const auto& robot_node = robot_nodes.front();
    const std::string robot_name = robot_node.name();

    // room←robot at the frame time (DSR clamps/interpolates), and the static robot←zed mount.
    const auto room_T_robot = inner_eigen_api_->get_transformation_matrix(
        room_frame_name_, robot_name, frame_ts, "RT", DSR::RT_API::TimeQuery::Interpolated);
    const auto robot_T_zed = inner_eigen_api_->get_transformation_matrix(
        robot_name, camera_node_name_, 0, "RT", DSR::RT_API::TimeQuery::Nearest);
    if (!room_T_robot.has_value() || !robot_T_zed.has_value())
        return std::nullopt;

    Eigen::Affine3d room_T_robot_pred = room_T_robot.value();

    // Dead-reckon room←robot forward by (frame_ts − leading_edge_stamp) using the body twist
    // written on the room→robot RT edge (rt_translation_velocity=[adv,side,0] m/s, body frame;
    // rt_rotation_euler_xyz_velocity=[0,0,rot] rad/s). SE2 increment: t += R·v·dt, θ += rot·dt.
    if (const auto rt_edge = graph_->get_edge(room_node->id(), robot_node.id(), "RT"); rt_edge.has_value())
    {
        const auto vel_t  = graph_->get_attrib_by_name<rt_translation_velocity_att>(rt_edge.value());
        const auto vel_r  = graph_->get_attrib_by_name<rt_rotation_euler_xyz_velocity_att>(rt_edge.value());
        const auto stamps = graph_->get_attrib_by_name<rt_timestamps_att>(rt_edge.value());

        if (vel_t.has_value() && vel_r.has_value() && stamps.has_value()
            && vel_t->get().size() >= 2 && vel_r->get().size() >= 3 && !stamps->get().empty())
        {
            std::uint64_t t_leading = 0;
            for (const auto s : stamps->get())
                t_leading = std::max(t_leading, s);

            // Only forward-predict: if the frame predates the leading edge the pose was already
            // bracketed/interpolated exactly, so dt=0. Clamp to the safety horizon.
            double dt = (frame_ts > t_leading) ? (static_cast<double>(frame_ts - t_leading) * 1e-3) : 0.0;
            dt = std::clamp(dt, 0.0, kMaxPredictHorizonS);

            if (dt > 1e-4)
            {
                const float adv  = vel_t->get()[0];
                const float side = vel_t->get()[1];
                const float rot  = vel_r->get()[2];

                const Eigen::Matrix3d R_old = room_T_robot->linear();
                const Eigen::Vector3d t_old = room_T_robot->translation();
                room_T_robot_pred.linear() =
                    Eigen::AngleAxisd(rot * dt, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R_old;
                room_T_robot_pred.translation() =
                    t_old + R_old * Eigen::Vector3d(adv, side, 0.0) * dt;
            }
        }
    }

    const Eigen::Affine3d room_T_zed_pred = room_T_robot_pred * robot_T_zed.value();
    return room_T_zed_pred.inverse();   // camera_T_room: maps room points → camera frame
}

std::vector<Eigen::Vector2f> CameraVisualizer::project_points_to_image(
    const std::vector<Eigen::Vector3f>& world_points, std::uint64_t rt_timestamp) const
{
    std::vector<Eigen::Vector2f> image_points;
    image_points.reserve(world_points.size());

    if (!inner_eigen_api_ || !camera_api_)
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (std::size_t i = 0; i < world_points.size(); ++i)
            image_points.emplace_back(nan, nan);
        return image_points;
    }

    if (!graph_ || !graph_->get_node(room_frame_name_).has_value() || !graph_->get_node(camera_node_name_).has_value())
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (std::size_t i = 0; i < world_points.size(); ++i)
            image_points.emplace_back(nan, nan);
        return image_points;
    }

    RoomToCameraBasis basis;
    if (!compute_room_to_camera_basis(inner_eigen_api_.get(), camera_node_name_, room_frame_name_, rt_timestamp, basis))
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (std::size_t i = 0; i < world_points.size(); ++i)
            image_points.emplace_back(nan, nan);
        return image_points;
    }

    for (std::size_t i = 0; i < world_points.size(); ++i)
    {
        const auto& p = world_points[i];
        const auto p_cam = transform_room_point(basis, Mat::Vector3d(p.x(), p.y(), p.z()));

        // CameraAPI::project uses Y as depth axis in camera frame.
        if (p_cam.y() <= 1e-6)
        {
            const float nan = std::numeric_limits<float>::quiet_NaN();
            image_points.emplace_back(nan, nan);
            continue;
        }

        const Eigen::Vector2d uv = camera_api_->project(p_cam);
        image_points.emplace_back(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
    }

    return image_points;
}

void CameraVisualizer::draw_projections(QImage& image, std::uint64_t rt_timestamp)
{
    if (!camera_data_.valid)
        return;

    if (!graph_ || !graph_->get_node(room_frame_name_).has_value() || !graph_->get_node(camera_node_name_).has_value())
        return;

    auto corners_3d = get_room_corners_3d();
    if (corners_3d.size() < 6)
        return;

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing, true);

    const int num_corners = static_cast<int>(corners_3d.size() / 2);
    if (!inner_eigen_api_ || !camera_api_ || num_corners < 2)
        return;

    // Prefer the latency-compensated camera pose (dead-reckoned to the frame's capture time) so the
    // overlay does not trail the image during motion. Fall back to the plain time-indexed lookup.
    RoomToCameraBasis basis;
    bool basis_ok = false;
    if (overlay_predict_pose_)
    {
        if (const auto cam_T_room = predicted_camera_from_room(rt_timestamp); cam_T_room.has_value())
        {
            const Eigen::Affine3d& M = cam_T_room.value();
            basis.origin = M.translation();
            basis.axis_x = M.linear().col(0);
            basis.axis_y = M.linear().col(1);
            basis.axis_z = M.linear().col(2);
            basis_ok = true;
        }
    }
    if (!basis_ok)
        basis_ok = compute_room_to_camera_basis(inner_eigen_api_.get(), camera_node_name_, room_frame_name_, rt_timestamp, basis);
    if (!basis_ok)
        return;

    // 1) Transform all room floor/top corners from room frame -> camera frame.
    std::vector<Mat::Vector3d> floor_in_cam;
    std::vector<Mat::Vector3d> top_in_cam;
    floor_in_cam.reserve(num_corners);
    top_in_cam.reserve(num_corners);
    for (int i = 0; i < num_corners; ++i)
    {
        const auto& p_floor_room = corners_3d[i * 2];      // floor corner z=0
        const auto& p_top_room = corners_3d[i * 2 + 1];    // top corner z=room_height

        floor_in_cam.push_back(transform_room_point(basis, Mat::Vector3d(p_floor_room.x(), p_floor_room.y(), p_floor_room.z())));
        top_in_cam.push_back(transform_room_point(basis, Mat::Vector3d(p_top_room.x(), p_top_room.y(), p_top_room.z())));
    }

    auto finite = [](const Eigen::Vector2f& p)
    {
        return std::isfinite(p.x()) && std::isfinite(p.y());
    };

    auto draw_segment = [&](const Eigen::Vector2f& a, const Eigen::Vector2f& b, const QPen& pen)
    {
        if (!finite(a) || !finite(b))
            return;

        QPointF p0(a.x(), a.y());
        QPointF p1(b.x(), b.y());

        // Draw the projected edge segment itself. Extending the line to the full
        // image can create visual crossings near corner views.
        painter.setPen(pen);
        painter.drawLine(p0, p1);
    };

    // ★ A 360 CAMERA SEES BEHIND ITSELF, so the near-plane clip below is a PINHOLE operation and
    //   applying it to a panorama discards the half of the room that is behind the robot — which is
    //   most of what a panorama is for. Detected from the model, not configured.
    const bool panoramic =
        camera_api_ and camera_api_->get_projection_model() != DSR::CameraAPI::ProjectionModel::Pinhole;
    const double img_w = static_cast<double>(camera_data_.width);

    auto project_clipped_segment = [&](Mat::Vector3d a, Mat::Vector3d b, Eigen::Vector2f& out_a, Eigen::Vector2f& out_b)
    {
        constexpr double near_y = 1e-4;

        if (not panoramic)
        {
            if (a.y() <= near_y && b.y() <= near_y)
                return false;

            if (a.y() <= near_y)
            {
                const double t = (near_y - a.y()) / (b.y() - a.y());
                a = a + t * (b - a);
            }
            else if (b.y() <= near_y)
            {
                const double t = (near_y - b.y()) / (a.y() - b.y());
                b = b + t * (a - b);
            }
        }

        const Eigen::Vector2d uv0 = camera_api_->project(a);
        Eigen::Vector2d uv1 = camera_api_->project(b);
        // ★ THE SEAM. On a cyclic column axis two ends of one short wall can land at u=1918 and u=2,
        //   and a straight line between them is drawn right across the image — a wall that is not
        //   there, which is worse than a wall that is missing. Unwrap the second endpoint so the
        //   segment stays continuous; Qt clips the part that leaves the widget. The piece that
        //   re-enters on the far side is not drawn, which is a small gap at the seam rather than a
        //   line across the middle.
        if (panoramic and img_w > 0.0)
            uv1.x() -= std::round((uv1.x() - uv0.x()) / img_w) * img_w;
        out_a = Eigen::Vector2f(static_cast<float>(uv0.x()), static_cast<float>(uv0.y()));
        out_b = Eigen::Vector2f(static_cast<float>(uv1.x()), static_cast<float>(uv1.y()));
        return true;
    };

    // ★ A STRAIGHT 3-D LINE IS A CURVE IN EQUIRECTANGULAR. Joining two projected endpoints with a
    //   straight image-space line is right for a pinhole and wrong for a panorama: the true image of
    //   the segment is a great-circle arc, and the error is largest exactly where the wall passes
    //   above or below the camera — which for a room contour is most of the frame. The fix is the
    //   one retina already uses (ricoh_projection_overlay.cpp:91): subdivide the 3-D segment,
    //   project every sample, and join consecutive samples ONLY when they lie on the same side of
    //   the wrap seam. Same construction, same reason, so the two cannot drift apart.
    //   Pinhole keeps the single clipped line — subdividing there would be pure cost for no change.
    auto draw_edge_cam = [&](const Mat::Vector3d& a, const Mat::Vector3d& b, const QPen& pen,
                             int subdiv)
    {
        if (not panoramic)
        {
            Eigen::Vector2f pa, pb;
            if (project_clipped_segment(a, b, pa, pb)) draw_segment(pa, pb, pen);
            return;
        }
        painter.setPen(pen);
        std::optional<QPointF> prev;
        for (int k = 0; k <= subdiv; ++k)
        {
            const double t = static_cast<double>(k) / static_cast<double>(subdiv);
            const Eigen::Vector2d uv = camera_api_->project(a + t * (b - a));
            if (!std::isfinite(uv.x()) || !std::isfinite(uv.y())) { prev.reset(); continue; }
            const QPointF cur(uv.x(), uv.y());
            if (prev.has_value() and std::abs(cur.x() - prev->x()) < 0.5 * img_w)
                painter.drawLine(*prev, cur);
            prev = cur;
        }
    };

    // 2) Draw a very light grid on floor and ceiling (different colors).
    float min_x = std::numeric_limits<float>::infinity();
    float max_x = -std::numeric_limits<float>::infinity();
    float min_y = std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < num_corners; ++i)
    {
        const auto& p = corners_3d[i * 2];
        min_x = std::min(min_x, p.x());
        max_x = std::max(max_x, p.x());
        min_y = std::min(min_y, p.y());
        max_y = std::max(max_y, p.y());
    }

    const float floor_z = 0.f;
    const float ceil_z = corners_3d[1].z();
    const float grid_step = 1.0f;  // metres

    auto draw_grid_plane = [&](float z, const QPen& pen)
    {
        auto project_room_segment = [&](const Mat::Vector3d& p0_room, const Mat::Vector3d& p1_room,
                                        Eigen::Vector2f& out0, Eigen::Vector2f& out1)
        {
            return project_clipped_segment(transform_room_point(basis, p0_room),
                                           transform_room_point(basis, p1_room),
                                           out0,
                                           out1);
        };

        const float x0 = std::floor(min_x / grid_step) * grid_step;
        const float x1 = std::ceil(max_x / grid_step) * grid_step;
        const float y0 = std::floor(min_y / grid_step) * grid_step;
        const float y1 = std::ceil(max_y / grid_step) * grid_step;

        // Lines parallel to room Y axis.
        for (float x = x0; x <= x1 + 1e-4f; x += grid_step)
        {
            Eigen::Vector2f a;
            Eigen::Vector2f b;
            if (project_room_segment(Mat::Vector3d(x, y0, z), Mat::Vector3d(x, y1, z), a, b))
                draw_edge_cam(transform_room_point(basis, Mat::Vector3d(x, y0, z)),
                              transform_room_point(basis, Mat::Vector3d(x, y1, z)), pen, 24);
        }

        // Lines parallel to room X axis.
        for (float y = y0; y <= y1 + 1e-4f; y += grid_step)
        {
            Eigen::Vector2f a;
            Eigen::Vector2f b;
            if (project_room_segment(Mat::Vector3d(x0, y, z), Mat::Vector3d(x1, y, z), a, b))
                draw_edge_cam(transform_room_point(basis, Mat::Vector3d(x0, y, z)),
                              transform_room_point(basis, Mat::Vector3d(x1, y, z)), pen, 24);
        }
    };

    draw_grid_plane(floor_z, QPen(QColor(80, 140, 255, 65), 1.4));   // slightly stronger blue floor grid
    draw_grid_plane(ceil_z, QPen(QColor(255, 110, 110, 65), 1.4));   // slightly stronger red ceiling grid

    // 3) Draw all visible floor/ceiling edge segments.
    for (int i = 0; i < num_corners; ++i)
    {
        const int j = (i + 1) % num_corners;
        Eigen::Vector2f floor_a;
        Eigen::Vector2f floor_b;
        Eigen::Vector2f top_a;
        Eigen::Vector2f top_b;

        const bool floor_ok = project_clipped_segment(floor_in_cam[i], floor_in_cam[j], floor_a, floor_b);
        const bool top_ok = project_clipped_segment(top_in_cam[i], top_in_cam[j], top_a, top_b);

        // Subdivision counts follow retina's: 40 along a wall run, which is where the curvature is.
        if (floor_ok)
            draw_edge_cam(floor_in_cam[i], floor_in_cam[j], QPen(Qt::blue, 3), 40);
        if (top_ok)
            draw_edge_cam(top_in_cam[i], top_in_cam[j], QPen(Qt::red, 3), 40);

        // Optional endpoints for visual debugging.
        painter.setPen(QPen(Qt::blue, 2));
        painter.setBrush(QColor(0, 100, 255, 100));
        if (floor_ok && finite(floor_a))
            painter.drawEllipse(QPointF(floor_a.x(), floor_a.y()), 4, 4);
        if (floor_ok && finite(floor_b))
            painter.drawEllipse(QPointF(floor_b.x(), floor_b.y()), 4, 4);

        painter.setPen(QPen(Qt::red, 2));
        painter.setBrush(QColor(255, 100, 0, 100));
        if (top_ok && finite(top_a))
            painter.drawEllipse(QPointF(top_a.x(), top_a.y()), 4, 4);
        if (top_ok && finite(top_b))
            painter.drawEllipse(QPointF(top_b.x(), top_b.y()), 4, 4);
    }

    // 3.4) Matched-corner uncertainty overlay. Markers are anchored at the CEILING point of each corner.
    // Radius = the projected 1σ positional uncertainty (from Σ_corner = Λ_det⁻¹), so shallow/aperture-
    // ambiguous corners read as big fuzzy blobs and clean corners as tight dots.
    {
        std::vector<rc::CornerDetector::CornerMatch> matches;
        {
            std::lock_guard<std::mutex> lk(corner_matches_mtx_);
            matches = corner_matches_;
        }
        // One rule, shared with the 2-D canvas (corner_detector.h), so the two views cannot drift.
        if (corners_matched_only_)
            std::erase_if(matches, [](const auto& m)
                          { return not rc::CornerDetector::matched_for_display(m); });
        constexpr double near_y = 1e-4;
        // px per metre at unit range: a focal length on a pinhole, pixels-per-radian on a panorama
        // (fetch_camera_intrinsics fills the latter, since a panorama has no focal length).
        const float fx = camera_data_.K(0, 0);
        const bool pano =
            camera_api_ and camera_api_->get_projection_model() != DSR::CameraAPI::ProjectionModel::Pinhole;
        painter.setPen(Qt::NoPen);
        for (const auto& m : matches)
        {
            const auto cam = transform_room_point(basis, Mat::Vector3d(m.model_world.x(), m.model_world.y(), ceil_z));
            // Behind the camera is INVISIBLE on a pinhole and perfectly visible on a panorama; the
            // radius below uses |cam| rather than cam.y() there, since y is not the range off-axis.
            if (not pano and cam.y() <= near_y)
                continue;
            if (pano and cam.norm() <= near_y)
                continue;
            const Eigen::Vector2d uv = camera_api_->project(cam);
            if (!std::isfinite(uv.x()) || !std::isfinite(uv.y()))
                continue;

            // Σ_corner = Λ_det⁻¹ via the eigenvalues with a precision floor (rank-1 shallow corner → σ capped),
            // mirroring the 2D viewer. Characteristic 1σ length = (det Σ)^{1/4} = geometric mean of the σ axes.
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(m.information);
            const Eigen::Vector2f lam = es.eigenvalues();
            constexpr float kPrecFloor = 1e-3f;
            const float cov0 = 1.f / std::max(lam(0), kPrecFloor);
            const float cov1 = 1.f / std::max(lam(1), kPrecFloor);
            const float det_cov = cov0 * cov1;             // m⁴ (= 1/det Λ_det)
            const float sigma_m = std::pow(std::max(det_cov, 1e-12f), 0.25f);   // metres

            // Project the metric σ to pixels at the corner's depth; clamp so it stays visible but bounded.
            // The radius carries the uncertainty (bigger blob = shallower corner), so the fill is a uniform
            // orange tone — brightening slightly with uncertainty — rather than a hue ramp.
            const double range = pano ? cam.norm() : cam.y();
            const float radius_px = std::clamp(fx * sigma_m / static_cast<float>(range), 5.f, 140.f);
            const float u = std::clamp((sigma_m - 0.02f) / (0.60f - 0.02f), 0.f, 1.f);
            // A RETIRED corner (information yield never materialised) is still projected — it is still
            // being detected and can still recover — but as a faint grey blob, so it never reads as a
            // live landmark. Matches the 2D viewer's dashed-grey treatment.
            const QColor fill = m.suppressed ? QColor(150, 150, 150, 55)
                                             : QColor(255, static_cast<int>(120 + 40 * u), 0, 130);
            painter.setBrush(fill);
            painter.drawEllipse(QPointF(uv.x(), uv.y()), radius_px, radius_px);
        }
        painter.setBrush(Qt::NoBrush);
    }

    // 3.4b) RGB TRIPLE POINTS — where the image says each wall-wall corner is, against where the
    // model puts it. Two markers and the line between them IS the residual, at true scale in the
    // image, which is the quantity the whole camera calibration is about.
    //
    // ★ DRAWN ONLY IN THE WINDOW WHOSE CAMERA PRODUCED THEM. uv_meas is a pixel coordinate in ONE
    //   camera's image; painting the ricoh's corners on the zed's frame would place them at
    //   plausible-looking positions that mean nothing. The producing camera is carried alongside the
    //   points and compared, rather than assumed to be this one.
    {
        std::vector<rc::TriplePoint> tps;
        {
            std::lock_guard<std::mutex> lk(triple_mtx_);
            if (triple_from_ == camera_node_name_) tps = triple_points_;
        }
        // One rule, shared with the 2-D canvas (rc::visible_and_matched, image_edge_types.h), and
        // governed by the SAME All/Matched button as the LiDAR corners above — one control for both
        // channels, because a reader comparing the two markers must know they were filtered alike.
        // Applied to this window's own copy, taken under triple_mtx_ above: the shared list is never
        // mutated to make a picture.
        // ★ It drops the WHOLE corner — measured square, predicted ring AND the residual line
        //   between them — not just the marker. A residual line hanging off a corner that is not
        //   drawn is worse than the corner: it points at nothing and still reads as evidence.
        if (corners_matched_only_)
            std::erase_if(tps, [](const auto& t) { return not rc::visible_and_matched(t); });
        for (const auto& t : tps)
        {
            if (!std::isfinite(t.uv_meas.x()) || !std::isfinite(t.uv_meas.y())) continue;
            const bool ceiling = (t.from == ContourClass::WallCeiling);
            // Ceiling corners in cyan, floor in magenta: the two populations answer differently
            // (the ceiling one is far less occluded) and a single colour would hide which is which.
            // Orange floor / cyan ceiling, matching the 2-D canvas. The two displays are read side
            // by side and the same feature must not change colour between them.
            const QColor col = ceiling ? QColor(0, 220, 255) : QColor(255, 150, 0);
            // MEASURED: a filled square, the same shape used for these in the 2-D canvas.
            painter.setPen(QPen(col, 2.0));
            painter.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 150)));
            painter.drawRect(QRectF(t.uv_meas.x() - 7.0, t.uv_meas.y() - 7.0, 14.0, 14.0));
            // PREDICTED: a hollow circle, plus the residual as a line. Skipped when the two are far
            // apart in u on a cyclic axis — that is the seam, not a 1900 px error.
            if (std::isfinite(t.uv_pred.x()) && std::isfinite(t.uv_pred.y()))
            {
                painter.setBrush(Qt::NoBrush);
                painter.drawEllipse(QPointF(t.uv_pred.x(), t.uv_pred.y()), 8.0, 8.0);
                if (std::abs(t.uv_meas.x() - t.uv_pred.x()) < 0.5 * camera_data_.width)
                {
                    painter.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 120), 1.0));
                    painter.drawLine(QPointF(t.uv_pred.x(), t.uv_pred.y()),
                                     QPointF(t.uv_meas.x(), t.uv_meas.y()));
                }
            }
        }
        painter.setBrush(Qt::NoBrush);
    }

    // 3.4c) The LiDAR's DETECTED corners, projected into this camera's image. Drawn at BOTH heights
    // because a LiDAR corner is a 2-D (x, y) on a vertical edge, and the camera sees that edge end
    // at the floor and again at the ceiling — the two points a triple point can be. So each green
    // marker has a same-height camera marker to be compared against, and the gap between them IS
    // the camera-vs-LiDAR disagreement for that corner, in this camera's own pixels.
    //
    // ★ DETECTED, not model_world. The struct carries the model position, which is where the corner
    //   is BELIEVED to be; drawing that would show the map agreeing with itself and say nothing
    //   about either sensor.
    // ★ GREEN, chosen against the palette already in use here: magenta and cyan are the camera's own
    //   floor and ceiling corners, and orange is the uncertainty blob above.
    {
        std::vector<rc::CornerDetector::CornerMatch> ms;
        Eigen::Affine2f rp;
        {
            std::lock_guard<std::mutex> lk(corner_matches_mtx_);
            ms = corner_matches_;
            rp = corner_pose_;
        }
        if (corners_matched_only_)
            std::erase_if(ms, [](const auto& m)
                          { return not rc::CornerDetector::matched_for_display(m); });
        const QColor lid(0, 230, 120);
        for (const auto& m : ms)
        {
            const Eigen::Vector2f w = rp * m.detected;      // robot frame -> room
            for (const double h : {0.0, static_cast<double>(ceil_z)})
            {
                const auto cam = transform_room_point(basis, Mat::Vector3d(w.x(), w.y(), h));
                // ★ BEHIND THE CAMERA. On a pinhole, project() of a point with y < 0 returns a
                //   FINITE but mirrored pixel — it does not fail, it lies — so a corner behind the
                //   ZED was being drawn at a plausible position on the wrong side of the image. A
                //   panorama genuinely sees behind itself, so the test applies only to the pinhole.
                if (not panoramic and cam.y() <= 1e-4) continue;
                if (panoramic and cam.norm() <= 1e-4) continue;
                const Eigen::Vector2d uv = camera_api_->project(cam);
                if (!std::isfinite(uv.x()) || !std::isfinite(uv.y())) continue;
                // A retired corner is still detected and can still recover, so it is drawn faint
                // rather than dropped — the same treatment it gets in the 2-D canvas.
                const int a = m.suppressed ? 70 : 210;
                painter.setPen(QPen(QColor(lid.red(), lid.green(), lid.blue(), a), 2.0));
                painter.setBrush(Qt::NoBrush);
                painter.drawEllipse(QPointF(uv.x(), uv.y()), 9.0, 9.0);
                painter.drawLine(QPointF(uv.x() - 13.0, uv.y()), QPointF(uv.x() + 13.0, uv.y()));
                painter.drawLine(QPointF(uv.x(), uv.y() - 13.0), QPointF(uv.x(), uv.y() + 13.0));
            }
        }
    }

    // 3.4d) Legend. Four marker types on one image, two of them differing only by hue, is more than
    // a reader should have to reconstruct from memory — and the whole point of these overlays is
    // comparing markers against each other, which needs knowing which is which.
    {
        struct Item { QColor col; const char* text; int shape; };   // 0 filled square, 1 ring, 2 cross
        const Item items[] = {
            { QColor(255, 150, 0), "camera corner — floor",    0 },
            { QColor(0, 220, 255), "camera corner — ceiling",  0 },
            { QColor(200, 200, 200), "same corner, model says", 1 },
            { QColor(0, 230, 120), "LiDAR corner (both ends)",  2 },
        };
        const int pad = 8, row = 18, box = 12;
        int wmax = 0;
        const QFontMetrics fm(painter.font());
        for (const auto& it : items) wmax = std::max(wmax, fm.horizontalAdvance(it.text));
        // Right-hand side. Anchored to the IMAGE width rather than a constant, so it stays put on
        // the 1280 px ZED and the 1920 px panorama alike; clamped to 0 so a window narrower than the
        // panel still shows it rather than pushing it off the left edge.
        const double panel_w = static_cast<double>(pad * 3 + box + wmax);
        const double panel_h = static_cast<double>(pad * 2 + row * std::size(items));
        const QRectF panel(std::max(0.0, image.width() - panel_w - 8.0), 8.0, panel_w, panel_h);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(0, 0, 0, 140));
        painter.drawRoundedRect(panel, 4, 4);
        int y = static_cast<int>(panel.y()) + pad + row / 2;
        for (const auto& it : items)
        {
            const int x = static_cast<int>(panel.x()) + pad;
            painter.setPen(QPen(it.col, 2.0));
            switch (it.shape)
            {
                case 0:
                    painter.setBrush(QColor(it.col.red(), it.col.green(), it.col.blue(), 150));
                    painter.drawRect(QRectF(x, y - box / 2.0, box, box));
                    break;
                case 1:
                    painter.setBrush(Qt::NoBrush);
                    painter.drawEllipse(QPointF(x + box / 2.0, y), box / 2.0, box / 2.0);
                    break;
                default:
                    painter.setBrush(Qt::NoBrush);
                    painter.drawEllipse(QPointF(x + box / 2.0, y), box / 2.0, box / 2.0);
                    painter.drawLine(QPointF(x - 2, y), QPointF(x + box + 2, y));
                    painter.drawLine(QPointF(x + box / 2.0, y - box / 2.0 - 2),
                                     QPointF(x + box / 2.0, y + box / 2.0 + 2));
                    break;
            }
            painter.setPen(QPen(QColor(235, 235, 235), 1.0));
            painter.setBrush(Qt::NoBrush);
            painter.drawText(QPointF(x + box + pad, y + 4), it.text);
            y += row;
        }
    }

    // 3.5) Translucent mesh + name label on each DSR wall. Each wall is a vertical quad; clip it
    // against the camera near plane (Sutherland-Hodgman, keep camera-y >= near) so walls partly
    // behind the camera still fill correctly, then project + fill translucent and label by name.
    {
        constexpr double near_y = 1e-4;
        const QColor wall_fill(120, 90, 200, 60);    // translucent purple mesh
        const QColor wall_edge(150, 120, 230, 170);
        for (const auto& wall : get_dsr_wall_quads(rt_timestamp))
        {
            std::vector<Mat::Vector3d> cam;
            cam.reserve(wall.corners.size());
            for (const auto& c : wall.corners)
                cam.push_back(transform_room_point(basis, Mat::Vector3d(c.x(), c.y(), c.z())));

            // ★ The near-plane clip is a PINHOLE operation: it keeps the part of the quad in front
            //   of the camera. A panorama has no front, so clipping there would delete every wall
            //   behind the robot. Panoramic keeps all four corners.
            std::vector<Mat::Vector3d> clipped;
            if (panoramic)
                clipped = cam;
            else
                for (std::size_t i = 0; i < cam.size(); ++i)
                {
                    const Mat::Vector3d& curr = cam[i];
                    const Mat::Vector3d& next = cam[(i + 1) % cam.size()];
                    const bool in_curr = curr.y() >= near_y;
                    const bool in_next = next.y() >= near_y;
                    if (in_curr)
                        clipped.push_back(curr);
                    if (in_curr != in_next)
                    {
                        const double t = (near_y - curr.y()) / (next.y() - curr.y());
                        clipped.push_back(curr + t * (next - curr));
                    }
                }
            if (clipped.size() < 3)
                continue;

            // ★ THE FILL MUST FOLLOW THE SAME CURVE AS THE OUTLINE. Projecting only the four
            //   corners and filling between them draws a straight-edged trapezoid underneath the
            //   correctly-curved contour lines — visibly wrong, and wrong in the direction that
            //   makes a wall look like it covers floor it does not. Walk each EDGE of the quad with
            //   the same subdivision the outlines use, so the polygon boundary is the projected arc
            //   rather than its chord.
            QPolygonF poly;
            bool poly_ok = true;
            const int quad_subdiv = panoramic ? 24 : 1;
            for (std::size_t i = 0; i < clipped.size() && poly_ok; ++i)
            {
                const Mat::Vector3d& a = clipped[i];
                const Mat::Vector3d& b = clipped[(i + 1) % clipped.size()];
                // k < subdiv, not <=: the next edge contributes its own start point, so the shared
                // vertex is added once rather than duplicated at every corner.
                for (int k = 0; k < quad_subdiv; ++k)
                {
                    const double t = static_cast<double>(k) / static_cast<double>(quad_subdiv);
                    const Eigen::Vector2d uv = camera_api_->project(a + t * (b - a));
                    if (!std::isfinite(uv.x()) || !std::isfinite(uv.y())) { poly_ok = false; break; }
                    poly << QPointF(uv.x(), uv.y());
                }
            }
            if (!poly_ok || poly.size() < 3)
                continue;
            // ★ A quad spanning the seam projects to columns at both edges, and FILLING that
            //   polygon paints a purple band straight across the image — a wall where there is
            //   none. Unlike a line segment a wrapping polygon cannot simply be unwrapped, so it is
            //   SKIPPED. A missing wall is honest; an invented one is not.
            if (panoramic and img_w > 0.0)
            {
                // ★ A JUMP between CONSECUTIVE boundary points, not the overall span. The span test
                //   cannot tell a quad that wraps the seam from a genuinely wide wall seen close up
                //   — both span more than half the image — and would silently delete the second.
                //   Now that the boundary is finely sampled, a wrap shows as one adjacent pair
                //   leaping most of the width, which nothing else produces.
                bool wraps = false;
                for (int i = 0; i < poly.size() and not wraps; ++i)
                {
                    const QPointF& q0 = poly[i];
                    const QPointF& q1 = poly[(i + 1) % poly.size()];
                    if (std::abs(q1.x() - q0.x()) > 0.5 * img_w) wraps = true;
                }
                if (wraps)
                    continue;   // a wall that is missing is honest; one painted across the image is not
            }

            painter.setPen(QPen(wall_edge, 1.5));
            painter.setBrush(QBrush(wall_fill));
            painter.drawPolygon(poly);

            // Name label at the projected polygon centroid (same style intent as object labels).
            QPointF centroid(0.0, 0.0);
            for (const auto& pt : poly)
                centroid += pt;
            centroid /= static_cast<double>(poly.size());
            painter.setBrush(Qt::NoBrush);
            painter.setPen(QPen(QColor(235, 220, 255)));
            painter.drawText(centroid, QString::fromStdString(wall.node_name));
        }
        painter.setBrush(Qt::NoBrush);
    }

    // 4) Project every modelled DSR object (object/table/cylinder) as an oriented 3D box.
    // 12 edges per box: bottom face (0-1-2-3), top face (4-5-6-7), 4 verticals (i ↔ i+4).
    // Each edge is room→camera transformed, near-plane clipped and projected like the room edges.
    painter.setBrush(Qt::NoBrush);
    static constexpr int box_edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},   // bottom
        {4, 5}, {5, 6}, {6, 7}, {7, 4},   // top
        {0, 4}, {1, 5}, {2, 6}, {3, 7}    // verticals
    };
    for (const auto& box : get_dsr_object_boxes(rt_timestamp))
    {
        const QPen box_pen(color_for_category(box.category), 2.5);
        for (const auto& e : box_edges)
        {
            const auto& ca = box.corners[e[0]];
            const auto& cb = box.corners[e[1]];
            const auto a_cam = transform_room_point(basis, Mat::Vector3d(ca.x(), ca.y(), ca.z()));
            const auto b_cam = transform_room_point(basis, Mat::Vector3d(cb.x(), cb.y(), cb.z()));
            Eigen::Vector2f a;
            Eigen::Vector2f b;
            // 12 for a box edge: shorter than a wall run, so the arc is gentler and fewer samples
            // carry it. Still curved — an object edge above or below the camera bends visibly.
            if (project_clipped_segment(a_cam, b_cam, a, b))
                draw_edge_cam(a_cam, b_cam, box_pen, 12);
        }

        // Label the box at its projected top-front corner (corner 4), if visible.
        const auto& c0 = box.corners[4];
        const auto c0_cam = transform_room_point(basis, Mat::Vector3d(c0.x(), c0.y(), c0.z()));
        if (c0_cam.y() > 1e-4)
        {
            const Eigen::Vector2d uv = camera_api_->project(c0_cam);
            if (std::isfinite(uv.x()) && std::isfinite(uv.y()))
            {
                painter.setPen(QPen(color_for_category(box.category)));
                painter.drawText(QPointF(uv.x() + 3, uv.y() - 3),
                                 QString::fromStdString(box.node_name));
            }
        }
    }

    painter.end();
}

void CameraVisualizer::draw_status_overlay(QImage& image) const
{
    QPainter painter(&image);
    painter.setRenderHint(QPainter::TextAntialiasing, true);

    const double source_fps = (have_estimated_source_period_ && estimated_source_period_ms_ > 1e-3)
        ? 1000.0 / estimated_source_period_ms_
        : 0.0;
    const double poll_fps = (refresh_timer_ != nullptr && refresh_timer_->interval() > 0)
        ? 1000.0 / static_cast<double>(refresh_timer_->interval())
        : 0.0;

    const QString overlay_text = QString("RGB %1 fps | Poll %2 fps")
        .arg(source_fps, 0, 'f', 1)
        .arg(poll_fps, 0, 'f', 1);

    QFont font = painter.font();
    const int target_point_size = std::clamp(image.height() / 22, 11, 26);
    font.setPointSize(target_point_size);
    font.setBold(true);
    painter.setFont(font);

    const QFontMetrics metrics(font);
    const QRect text_rect = metrics.boundingRect(overlay_text).adjusted(-10, -6, 10, 6);
    const QRect panel_rect(12, 12, text_rect.width(), text_rect.height());

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(0, 0, 0, 170));
    painter.drawRoundedRect(panel_rect, 8, 8);

    painter.setPen(Qt::white);
    painter.drawText(panel_rect, Qt::AlignCenter, overlay_text);
    painter.end();
}

void CameraVisualizer::update_frame()
{
    if (!isVisible())
        return;

    ++timing_stats_.timer_callbacks;

    if (callback_gap_timer_.isValid())
    {
        const float callback_gap_ms = static_cast<float>(callback_gap_timer_.restart());
        timing_stats_.total_callback_gap_ms += callback_gap_ms;
        timing_stats_.max_callback_gap_ms = std::max(timing_stats_.max_callback_gap_ms, callback_gap_ms);
        ++timing_stats_.callback_gap_samples;
    }
    else
    {
        callback_gap_timer_.start();
    }

    QElapsedTimer frame_timer;
    frame_timer.start();

    QImage rgb_image;
    std::uint64_t frame_timestamp = 0;
    QElapsedTimer stage_timer;
    stage_timer.start();
    const bool fetched = fetch_rgb_from_dsr(rgb_image, frame_timestamp);
    const float fetch_ms = static_cast<float>(stage_timer.elapsed());
    timing_stats_.total_fetch_ms += fetch_ms;

    if (!fetched)
    {
        ++timing_stats_.fetch_failures;
    }
    else
    {
        bool should_render = true;
        if (frame_timestamp == 0)
        {
            ++timing_stats_.zero_timestamps;
        }
        else if (have_last_source_timestamp_)
        {
            if (frame_timestamp == last_source_timestamp_)
            {
                ++timing_stats_.repeated_source_frames;
                should_render = false;
            }
            else if (frame_timestamp > last_source_timestamp_)
            {
                ++timing_stats_.unique_source_frames;
                const auto source_delta = frame_timestamp - last_source_timestamp_;
                timing_stats_.total_source_delta += static_cast<double>(source_delta);
                ++timing_stats_.source_delta_samples;
                adapt_refresh_interval(source_delta);
            }
            else
            {
                ++timing_stats_.source_regressions;
            }
        }
        else
        {
            ++timing_stats_.unique_source_frames;
        }

        if (frame_timestamp > 0 && (!have_last_source_timestamp_ || frame_timestamp >= last_source_timestamp_))
        {
            last_source_timestamp_ = frame_timestamp;
            have_last_source_timestamp_ = true;
        }

        if (should_render)
        {
            stage_timer.restart();
            draw_projections(rgb_image, frame_timestamp);
            const float draw_ms = static_cast<float>(stage_timer.elapsed());
            timing_stats_.total_draw_ms += draw_ms;

            stage_timer.restart();
            QImage display_image = rgb_image;
            if (image_label_ != nullptr)
            {
                const QSize target_size = image_label_->size();
                if (target_size.isValid() && !target_size.isEmpty())
                {
                    display_image = rgb_image.scaled(target_size,
                                                     Qt::KeepAspectRatio,
                                                     Qt::SmoothTransformation);
                }
            }
            draw_status_overlay(display_image);
            image_label_->setPixmap(QPixmap::fromImage(display_image));
            const float present_ms = static_cast<float>(stage_timer.elapsed());
            timing_stats_.total_present_ms += present_ms;
            ++timing_stats_.rendered_frames;
        }

        ++frames_since_fps_log_;
        const qint64 elapsed_ms = fps_timer_.elapsed();
        if (elapsed_ms >= 2000)
        {
            fps_timer_.restart();
            frames_since_fps_log_ = 0;
        }
    }

    const float total_ms = static_cast<float>(frame_timer.elapsed());
    timing_stats_.total_callback_ms += total_ms;
    timing_stats_.max_callback_ms = std::max(timing_stats_.max_callback_ms, total_ms);

    if (timing_window_timer_.elapsed() >= 2000)
    {
        log_timing_summary("periodic");
        reset_timing_window();
    }
}

}  // namespace rc
