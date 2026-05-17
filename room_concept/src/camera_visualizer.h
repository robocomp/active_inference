#ifndef CAMERA_VISUALIZER_H
#define CAMERA_VISUALIZER_H

#include <QDialog>
#include <QLabel>
#include <QImage>
#include <QTimer>
#include <QElapsedTimer>
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

        // Methods
        bool fetch_rgb_from_dsr(QImage& rgb_image, std::uint64_t& frame_timestamp);
        bool fetch_camera_intrinsics();
        std::vector<Eigen::Vector3f> get_room_corners_3d() const;
        std::vector<Eigen::Vector2f> project_points_to_image(const std::vector<Eigen::Vector3f>& world_points,
                                                            std::uint64_t rt_timestamp) const;
        void draw_projections(QImage& image, std::uint64_t rt_timestamp);
    };

}  // namespace rc

#endif // CAMERA_VISUALIZER_H
