#ifndef CAMERA_VISUALIZER_H
#define CAMERA_VISUALIZER_H

#include <QDialog>
#include <QHideEvent>
#include <QLabel>
#include <QImage>
#include <QShowEvent>
#include <QTimer>
#include <QElapsedTimer>
#include <cstdint>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <string>

// Forward declarations
namespace DSR {
    class DSRGraph;
    class CameraAPI;
    class InnerEigenAPI;
}
using DSRGraph = DSR::DSRGraph;

namespace rc {

/**
 * @brief Camera visualization dialog that displays RGB frames from the DSR zed node
 *        and overlays room layout projections at two z-heights (0 and 2.5m)
 */
class CameraVisualizer : public QDialog
{
    Q_OBJECT

    public:
        explicit CameraVisualizer(std::shared_ptr<DSRGraph> graph, const std::vector<Eigen::Vector2f>& room_polygon, QWidget* parent = nullptr);
        ~CameraVisualizer() = default;

        void update_frame();  // Call this periodically to refresh the visualization

    protected:
        void showEvent(QShowEvent* event) override;
        void hideEvent(QHideEvent* event) override;

    private:
        std::shared_ptr<DSRGraph> graph_;
        std::vector<Eigen::Vector2f> room_polygon_;
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
        void draw_projections(QImage& image, std::uint64_t rt_timestamp);
        void draw_status_overlay(QImage& image) const;
        void reset_timing_window();
        void log_timing_summary(const char* reason);
        static double raw_timestamp_delta_to_ms(std::uint64_t raw_delta);
        void adapt_refresh_interval(std::uint64_t raw_delta);
    };

}  // namespace rc

#endif // CAMERA_VISUALIZER_H
