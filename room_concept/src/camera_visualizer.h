#ifndef CAMERA_VISUALIZER_H
#define CAMERA_VISUALIZER_H

#include <QDialog>
#include <QHideEvent>
#include <QLabel>
#include <QImage>
#include <QShowEvent>
#include <QTimer>
#include <QElapsedTimer>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <Eigen/Dense>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
#include <memory>
#include <string>

#include "corner_detector.h"   // rc::CornerDetector::CornerMatch (matched-corner overlay)

// Forward declarations
namespace DSR {
    class DSRGraph;
    class CameraAPI;
    class InnerEigenAPI;
}
using DSRGraph = DSR::DSRGraph;

namespace rc::media { class MediaSubscriber; }  // media-plane RGB consumer (keeps fastdds out of MOC header)

namespace rc {

/**
 * @brief Camera visualization dialog that displays RGB frames from the DSR zed node
 *        and overlays room layout projections at two z-heights (0 and 2.5m)
 */
class CameraVisualizer : public QDialog
{
    Q_OBJECT

    public:
        explicit CameraVisualizer(std::shared_ptr<DSRGraph> graph, const std::vector<Eigen::Vector2f>& room_polygon,
                                  std::vector<std::string> overlay_object_types = {"object"},
                                  QWidget* parent = nullptr);
        ~CameraVisualizer();  // defined in .cpp for unique_ptr<MediaSubscriber> of incomplete type

        // Start the dedicated media-plane ingest thread. The RGB subscriber itself is
        // created lazily on that thread (shared descriptor-driven factory) once the "zed"
        // node + media descriptor exist (domain/topic from that JSON — no config), and the
        // thread then continuously drains DDS into a mutex-protected frame cache. Keeping the
        // subscriber lifecycle + take() OFF the GUI thread both decouples camera latency from
        // repaint and (with the media_transport entity mutex) makes concurrent bring-up
        // alongside the LiDAR ingest thread safe. Call once after construction.
        void start_media_plane();

        // Hand the latest matched corners (localizer thread → GUI) for the RGB uncertainty overlay.
        // Each is projected as a translucent circle whose radius grows with det(Σ_corner). Thread-safe.
        void set_corner_matches(std::vector<rc::CornerDetector::CornerMatch> matches);

        void update_frame();  // Call this periodically to refresh the visualization

    protected:
        void showEvent(QShowEvent* event) override;
        void hideEvent(QHideEvent* event) override;

    private:
        std::shared_ptr<DSRGraph> graph_;
        std::vector<Eigen::Vector2f> room_polygon_;
        std::vector<std::string> overlay_object_types_;  // DSR node types projected as boxes (config Overlay.ObjectTypes)

        // Latest matched corners for the uncertainty overlay (set from the GUI thread each frame).
        std::mutex corner_matches_mtx_;
        std::vector<rc::CornerDetector::CornerMatch> corner_matches_;
        QLabel* image_label_;

        // Cached camera data
        struct CameraData
        {
            int width = 0;
            int height = 0;
            Eigen::Matrix3f K = Eigen::Matrix3f::Zero();  // Intrinsic matrix
            bool valid = false;
        };
        CameraData camera_data_;
        QTimer* refresh_timer_ = nullptr;
        QElapsedTimer fps_timer_;
        int frames_since_fps_log_ = 0;
        std::unique_ptr<DSR::CameraAPI> camera_api_;
        std::unique_ptr<DSR::InnerEigenAPI> inner_eigen_api_;
        std::string camera_node_name_ = "zed";
        std::string room_frame_name_ = "room";

        // ── Media-plane RGB consumer (dedicated ingest thread) ────────────────
        // media_rgb_sub_ is owned EXCLUSIVELY by the ingest thread (created in
        // try_discover_media_plane, polled in ingest_pump, destroyed after the thread joins).
        // The GUI thread never touches the subscriber — it only reads the latest decoded frame
        // from media_rgb_ under media_rgb_mtx_, gated by the subscriber_ready_ flag.
        std::unique_ptr<rc::media::MediaSubscriber> media_rgb_sub_;
        struct MediaRgbCache
        {
            std::vector<std::uint8_t> bytes;
            int width = 0;
            int height = 0;
            std::uint32_t format = 0;
            std::uint64_t stamp = 0;
            bool valid = false;
        };
        MediaRgbCache media_rgb_;                 // written by ingest thread, read by GUI — under media_rgb_mtx_
        std::mutex media_rgb_mtx_;
        std::vector<std::uint8_t> render_bytes_;  // GUI-thread scratch: a copy of the latest frame's bytes

        // Ingest thread machinery (mirrors rc::LidarIngestor).
        std::thread ingest_thread_;
        std::atomic<bool> ingest_running_{false};
        std::atomic<bool> subscriber_ready_{false};   // set once the RGB subscriber is live
        std::mutex ingest_wake_mtx_;
        std::condition_variable ingest_wake_cv_;
        void ingest_loop();   // thread body: discover → drain → brief idle wait
        bool ingest_pump();   // one discovery/drain step; true if a frame was taken
        void stop_ingest();   // signal + join the ingest thread (idempotent)

        // Lazily create the RGB subscriber from the "zed" media descriptor (self-throttled).
        // Runs on the ingest thread only.
        bool try_discover_media_plane();
        std::chrono::steady_clock::time_point last_media_discovery_attempt_{};

        struct TimingStats
        {
            std::uint64_t timer_callbacks = 0;
            std::uint64_t rendered_frames = 0;
            std::uint64_t fetch_failures = 0;
            std::uint64_t repeated_source_frames = 0;
            std::uint64_t unique_source_frames = 0;
            std::uint64_t zero_timestamps = 0;
            std::uint64_t source_regressions = 0;
            std::uint64_t callback_gap_samples = 0;
            std::uint64_t source_delta_samples = 0;
            float total_fetch_ms = 0.f;
            float total_draw_ms = 0.f;
            float total_present_ms = 0.f;
            float total_callback_ms = 0.f;
            float total_callback_gap_ms = 0.f;
            float max_callback_ms = 0.f;
            float max_callback_gap_ms = 0.f;
            double total_source_delta = 0.0;
        };
        TimingStats timing_stats_;
        QElapsedTimer timing_window_timer_;
        QElapsedTimer callback_gap_timer_;
        std::uint64_t last_source_timestamp_ = 0;
        bool have_last_source_timestamp_ = false;
        double estimated_source_period_ms_ = 50.0;
        bool have_estimated_source_period_ = false;

        // Methods
        bool fetch_rgb_from_dsr(QImage& rgb_image, std::uint64_t& frame_timestamp);
        bool fetch_camera_intrinsics();
        std::vector<Eigen::Vector3f> get_room_corners_3d() const;
        std::vector<Eigen::Vector2f> project_points_to_image(const std::vector<Eigen::Vector3f>& world_points,
                                                            std::uint64_t rt_timestamp) const;

        // Oriented 3D box of a DSR object/table/cylinder node, with its 8 corners already
        // transformed to the ROOM frame (same 8-corner convention as the retina's
        // GraphObjectBox: 0-3 = bottom face CCW, 4-7 = top face CCW). Drawn on top of the
        // room-layout overlay so every modelled object is projected onto the live image.
        struct ObjectBox
        {
            std::array<Eigen::Vector3f, 8> corners;  // room frame, metres
            std::string category;                    // drives the overlay colour
            std::string node_name;
        };
        // Read every DSR node whose type is in overlay_object_types_ (config Overlay.ObjectTypes:
        // object/table/cylinder/chair/…) and build its oriented room-frame box from
        // width/depth/height + the room←node RT transform at rt_timestamp.
        std::vector<ObjectBox> get_dsr_object_boxes(std::uint64_t rt_timestamp) const;

        // Planar quad of a DSR wall node, 4 corners in the ROOM frame (CCW). A wall is a vertical
        // rectangle: width_m along its local X, height_m along local Z, centred on its RT origin.
        struct WallQuad
        {
            std::array<Eigen::Vector3f, 4> corners;  // room frame, metres (bottom-left, bottom-right, top-right, top-left)
            std::string node_name;
        };
        std::vector<WallQuad> get_dsr_wall_quads(std::uint64_t rt_timestamp) const;

        // Forward-predict the camera←room transform (maps room points → camera frame) to the RGB
        // frame's capture time using the robot body twist on the room→robot RT edge. The localizer
        // pose trails the camera stream and DSR clamps at the leading edge (never extrapolates), so
        // a plain RT lookup returns a stale pose and the overlay lags the image during motion. This
        // dead-reckons the latest pose forward by (frame_ts − leading_edge_stamp). Returns nullopt
        // when inputs are missing → caller falls back to the plain RT lookup.
        std::optional<Eigen::Affine3d> predicted_camera_from_room(std::uint64_t frame_ts) const;
        bool overlay_predict_pose_ = true;   // toggle the dead-reckoning latency compensation

        void draw_projections(QImage& image, std::uint64_t rt_timestamp);
        void draw_status_overlay(QImage& image) const;
        void reset_timing_window();
        void log_timing_summary(const char* reason);
        static double raw_timestamp_delta_to_ms(std::uint64_t raw_delta);
        void adapt_refresh_interval(std::uint64_t raw_delta);
    };

}  // namespace rc

#endif // CAMERA_VISUALIZER_H
