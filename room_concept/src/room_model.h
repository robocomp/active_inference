#pragma once

#include <optional>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace rc
{

/// Robot-frame scan points [x, y, z]. The estimator's one point-set type.
using ScanPoints = std::vector<Eigen::Vector3f>;

/**
 * Model - Room model for Active Inference SLAM
 * Supports both axis-aligned box and polygon rooms
 *
 * State (5): [width, length, x, y, phi]
 *  - width/length: room dimensions (m)
 *  - x,y,phi: robot pose wrt room center
 *
 * Plain Eigen since 2026-10-17. It used to be a torch::nn::Module whose pose was an autograd
 * parameter; Gauss-Newton (room_gn_solver) became the only backend and takes analytic Jacobians from
 * SdfQueryResult::grad, so nothing differentiates through this class any more. The SDF formulas are
 * the torch ones term for term (same epsilons, same clamps, same first-minimum tie rule).
 */
class Model
{
public:
  struct SdfQueryResult
  {
    Eigen::VectorXf               sdf;              // [N]
    Eigen::Matrix<float, -1, 2>   closest_normals;  // [N, 2] wall normals in room frame
    // [N, 2] ∇_p SDF in the room frame. This is NOT closest_normals: the polygon branch returns an
    // UNSIGNED distance, whose gradient is the unit vector from the closest point TOWARDS p — the segment
    // normal has a winding-dependent sign and is wrong for points on the other side, and points whose
    // closest feature is a vertex have no segment normal at all. Analytic-Jacobian consumers
    // (room_gn_solver) must use this; the loss uses closest_normals for incidence weighting, where only
    // |cos| matters. Filled only when asked for (want_grad).
    Eigen::Matrix<float, -1, 2>   grad;
    [[nodiscard]] Eigen::Index size() const noexcept { return sdf.size(); }
  };

    // Room parameters (fixed)
    Eigen::Vector2f half_extents = Eigen::Vector2f::Zero();  // [half_width, half_length] - box mode / bbox
    std::vector<Eigen::Vector2f> polygon_vertices;           // polygon vertices in room frame
    bool use_polygon = false;

    // Robot pose wrt room center
    Eigen::Vector2f robot_pos = Eigen::Vector2f::Zero();     // [x, y]
    float           robot_theta = 0.f;                       // phi

    float half_height = 1.2f;    // half of wall height (fixed)

    // Prediction fields (Active Inference prior)
    Eigen::Vector2f predicted_pos = Eigen::Vector2f::Zero();
    float           predicted_theta = 0.f;
    Eigen::Matrix3f prediction_precision_matrix = Eigen::Matrix3f::Identity();
    bool has_prediction = false;

    // State for EKF-style prediction
    Eigen::Matrix3f prev_cov = 0.1f * Eigen::Matrix3f::Identity();
    std::optional<Eigen::Affine2f> robot_prev_pose;

    // Initialize from rectangular room
    void init_from_state(float width, float length, float x, float y, float phi, float wall_height);

    // Initialize from polygon room
    void init_from_polygon(const std::vector<Eigen::Vector2f>& vertices,
                           float x, float y, float phi, float wall_height);

    // Replace the polygon WITHOUT touching the pose, the covariance or anything else the estimator
    // carries between frames. Wall-SLAM re-derives the polygon from the wall landmarks every cycle;
    // rebuilding the whole Model for that (set_polygon_room) would reset the window and the covariance
    // on every update. Only the geometry and the bbox half_extents change here.
    void update_polygon_vertices(const std::vector<Eigen::Vector2f>& vertices);

    // True once init_from_state / init_from_polygon has run. Existence of a Model object is NOT the
    // same as the model being usable, and every caller that reads the state must ask this, not merely
    // whether the shared_ptr is non-null.
    [[nodiscard]] bool has_state() const noexcept { return initialised_; }

    // Set prediction for Active Inference prior
    void set_prediction(const Eigen::Vector2f& pred_pos, float pred_theta,
                        const Eigen::Matrix3f& precision);

    // [x, y, theta] of the model's own pose
    [[nodiscard]] Eigen::Vector3f pose() const { return {robot_pos.x(), robot_pos.y(), robot_theta}; }
    void set_pose(const Eigen::Vector3f& p) { robot_pos = p.head<2>(); robot_theta = p.z(); }

    // SDF of robot-frame points at the model's own pose
    [[nodiscard]] Eigen::VectorXf sdf(const ScanPoints& points_robot) const;

    // SDF of robot-frame points at an external pose [x, y, theta] (sliding window)
    [[nodiscard]] Eigen::VectorXf sdf_at_pose(const ScanPoints& points_robot, const Eigen::Vector3f& pose) const;

    // SDF and closest-wall normals at an external pose. want_grad additionally fills
    // SdfQueryResult::grad (the analytic-Jacobian consumer needs it; the loss path does not).
    [[nodiscard]] SdfQueryResult sdf_query_at_pose(const ScanPoints& points_robot,
                                                   const Eigen::Vector3f& pose,
                                                   bool want_grad = false) const;

    // Get current state as Eigen vector [width, length, x, y, phi]
    [[nodiscard]] Eigen::Matrix<float, 5, 1> get_state() const;

private:
    void init_common();
    void set_polygon_geometry(const std::vector<Eigen::Vector2f>& vertices);

    void sdf_box_query(const ScanPoints& points_robot, const Eigen::Matrix<float, -1, 2>& room_xy,
                       bool want_grad, SdfQueryResult& out) const;
    void sdf_polygon_query(const Eigen::Matrix<float, -1, 2>& room_xy, bool want_grad,
                           SdfQueryResult& out) const;

    bool initialised_ = false;

    // Pre-computed segment data for polygon SDF
    std::vector<Eigen::Vector2f> seg_a_;    // segment start points
    std::vector<Eigen::Vector2f> seg_ab_;   // segment vectors (end - start)
    std::vector<float>           seg_ab_sq_;// squared length of each segment
};

} // namespace rc
