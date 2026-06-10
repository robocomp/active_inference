#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <QWidget>

#include <array>
#include <string>
#include <vector>

class QLabel;
class QPushButton;

// A 3D viewer widget for the arm belief state, with an embedded time-series plot for the gripper sensor readings.  The 3D view shows the arm pose and the scene geometry (table, bottle) as the controller sees it, and the plot tracks the gripper forces and contacts over time.  The viewer also has a big "Start" button that arms/disarms the controller, emitting a signal when toggled.
class ArmBeliefViewer3D : public QWidget
{
    Q_OBJECT
public:
    struct LinkPose
    {
        std::string mesh_filename;
        Eigen::Isometry3d pose;
    };

    explicit ArmBeliefViewer3D(QWidget* parent = nullptr);
    ~ArmBeliefViewer3D() override;

    void update_beliefs(const std::array<double, 7>& joint_angles,
                        const std::vector<LinkPose>& link_poses,
                        const Eigen::Vector3d& target,
                        const Eigen::Vector3d& ee_position);

    // Push one sample of the gripper sensors into the rolling time-series plot under the 3D
    // view: 4 force channels (motor grip L/R + fingertip tactile L/R) + the 2 binary tip
    // bumper contacts (drawn as steps).
    void update_forces(float lforce, float rforce, float ltipforce, float rtipforce,
                       float wrist_force, bool ltipcontact, bool rtipcontact);

    // table_corners must be 8 points in robot frame (m), ordered as
    // [bottom 4 CCW] then [top 4 CCW] — the panel connects them as a wireframe.
    // bottle_origin / bottle_axis are in the robot frame, m / unit-vector.
    // bottle_radius and bottle_height come from the DSR bottle node.
    void update_scene_objects(const std::vector<Eigen::Vector3d>& table_corners,
                              const Eigen::Vector3d& bottle_origin,
                              const Eigen::Vector3d& bottle_axis,
                              double bottle_radius,
                              double bottle_height);

    // Vertical support column (the arm's mast in Webots), drawn as a cylinder
    // from `base` to `top` (world frame, m). radius<=0 hides it.
    void set_column(const Eigen::Vector3d& base,
                    const Eigen::Vector3d& top,
                    double radius);

    void set_mesh_root(std::string mesh_root);

signals:
    // True when the user has armed the controller (button checked),
    // false when they've toggled it back off (button unchecked).
    void run_state_changed(bool running);

private:
    class GLPanel;
    class ForcePlot;

    GLPanel*     gl_panel_     = nullptr;
    QLabel*      status_label_ = nullptr;
    QPushButton* start_button_ = nullptr;
    ForcePlot*   force_plot_   = nullptr;
    std::string  mesh_root_;
};
