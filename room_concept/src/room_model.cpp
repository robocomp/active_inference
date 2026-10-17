#include "room_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rc
{

void Model::init_from_state(float width, float length, float x, float y, float phi, float wall_height)
{
    use_polygon = false;
    half_height = wall_height * 0.5f;
    half_extents = {width * 0.5f, length * 0.5f};   // fixed room size
    robot_pos = {x, y};
    robot_theta = phi;
    init_common();
}

void Model::set_polygon_geometry(const std::vector<Eigen::Vector2f>& vertices)
{
    polygon_vertices = vertices;
    // Pre-compute segment data for the SDF: segment k runs from vertex k to vertex k+1 (closed).
    const auto n = vertices.size();
    seg_a_.resize(n);
    seg_ab_.resize(n);
    seg_ab_sq_.resize(n);
    for (std::size_t k = 0; k < n; ++k)
    {
        seg_a_[k] = vertices[k];
        seg_ab_[k] = vertices[(k + 1) % n] - vertices[k];
        seg_ab_sq_[k] = seg_ab_[k].x() * seg_ab_[k].x() + seg_ab_[k].y() * seg_ab_[k].y();
    }
    // Bounding box for half_extents (for compatibility)
    float min_x = vertices[0].x(), max_x = vertices[0].x();
    float min_y = vertices[0].y(), max_y = vertices[0].y();
    for (const auto& v : vertices)
    {
        min_x = std::min(min_x, v.x()); max_x = std::max(max_x, v.x());
        min_y = std::min(min_y, v.y()); max_y = std::max(max_y, v.y());
    }
    half_extents = {(max_x - min_x) / 2.f, (max_y - min_y) / 2.f};
}

void Model::init_from_polygon(const std::vector<Eigen::Vector2f>& vertices,
                              float x, float y, float phi, float wall_height)
{
    use_polygon = true;
    half_height = wall_height * 0.5f;
    set_polygon_geometry(vertices);
    robot_pos = {x, y};
    robot_theta = phi;
    init_common();
}

void Model::update_polygon_vertices(const std::vector<Eigen::Vector2f>& vertices)
{
    if (vertices.size() < 3) return;
    use_polygon = true;
    set_polygon_geometry(vertices);
}

void Model::init_common()
{
    predicted_pos.setZero();
    predicted_theta = 0.f;
    prediction_precision_matrix.setIdentity();
    has_prediction = false;
    prev_cov = 0.1f * Eigen::Matrix3f::Identity();   // moderate initial uncertainty for EKF prediction
    robot_prev_pose = std::nullopt;
    initialised_ = true;
}

void Model::set_prediction(const Eigen::Vector2f& pred_pos, float pred_theta,
                           const Eigen::Matrix3f& precision)
{
    predicted_pos = pred_pos;
    predicted_theta = pred_theta;
    prediction_precision_matrix = precision;
    has_prediction = true;
}

Eigen::VectorXf Model::sdf(const ScanPoints& points_robot) const
{
    return sdf_at_pose(points_robot, pose());
}

Eigen::VectorXf Model::sdf_at_pose(const ScanPoints& points_robot, const Eigen::Vector3f& pose) const
{
    return sdf_query_at_pose(points_robot, pose).sdf;
}

Model::SdfQueryResult Model::sdf_query_at_pose(const ScanPoints& points_robot,
                                               const Eigen::Vector3f& pose,
                                               bool want_grad) const
{
    SdfQueryResult out;
    const auto n = static_cast<Eigen::Index>(points_robot.size());
    out.sdf.resize(n);
    out.closest_normals.resize(n, 2);
    if (want_grad) out.grad.resize(n, 2);
    if (n == 0) return out;

    // Map points to room frame: p_room = R(phi) * p_robot + t
    const float c = std::cos(pose.z()), s = std::sin(pose.z());
    Eigen::Matrix<float, -1, 2> room_xy(n, 2);
    for (Eigen::Index i = 0; i < n; ++i)
    {
        const auto& p = points_robot[static_cast<std::size_t>(i)];
        room_xy(i, 0) = c * p.x() - s * p.y() + pose.x();
        room_xy(i, 1) = s * p.x() + c * p.y() + pose.y();
    }

    if (use_polygon)
        sdf_polygon_query(room_xy, want_grad, out);
    else
        sdf_box_query(points_robot, room_xy, want_grad, out);
    return out;
}

void Model::sdf_box_query(const ScanPoints& points_robot, const Eigen::Matrix<float, -1, 2>& room_xy,
                          bool want_grad, SdfQueryResult& out) const
{
    const float hx = half_extents.x(), hy = half_extents.y(), hz = half_height;
    for (Eigen::Index i = 0; i < room_xy.rows(); ++i)
    {
        const float px = room_xy(i, 0), py = room_xy(i, 1);
        const float pz = points_robot[static_cast<std::size_t>(i)].z();
        const float dx = std::abs(px) - hx, dy = std::abs(py) - hy, dz = std::abs(pz) - hz;
        const float ox = std::max(dx, 0.f), oy = std::max(dy, 0.f), oz = std::max(dz, 0.f);
        const float outside = std::sqrt(ox * ox + oy * oy + oz * oz);
        const float inside = std::min(std::max(std::max(dx, dy), dz), 0.f);
        out.sdf(i) = outside + inside;

        // Closest face (first minimum, as torch::min): left, right, bottom, top.
        const float faces[4] = {std::abs(px + hx), std::abs(hx - px), std::abs(py + hy), std::abs(hy - py)};
        int k = 0;
        for (int f = 1; f < 4; ++f) if (faces[f] < faces[k]) k = f;
        static constexpr float nx[4] = {-1.f, 1.f, 0.f, 0.f};
        static constexpr float ny[4] = {0.f, 0.f, -1.f, 1.f};
        out.closest_normals(i, 0) = nx[k];
        out.closest_normals(i, 1) = ny[k];
    }
    // For the box the signed distance's gradient IS the closest face's outward unit normal: exact for
    // interior points (the branch lidar hits fall in) and for exterior points outside a single face.
    // It degrades only in the corner wedge outside two faces at once, where the true gradient is the
    // normalized max(d,0) vector — an approximation the Gauss-Newton step tolerates (LM damping
    // covers it), and one the polygon branch below does not need.
    if (want_grad) out.grad = out.closest_normals;
}

void Model::sdf_polygon_query(const Eigen::Matrix<float, -1, 2>& room_xy, bool want_grad,
                              SdfQueryResult& out) const
{
    const auto S = seg_a_.size();
    if (S == 0)   // no geometry yet: report zero distance rather than index an empty segment list
    {
        out.sdf.setZero(); out.closest_normals.setZero();
        if (want_grad) out.grad.setZero();
        return;
    }
    for (Eigen::Index i = 0; i < room_xy.rows(); ++i)
    {
        const float px = room_xy(i, 0), py = room_xy(i, 1);
        float best_d2 = std::numeric_limits<float>::infinity();
        std::size_t best = 0;
        float best_cx = 0.f, best_cy = 0.f;
        for (std::size_t k = 0; k < S; ++k)
        {
            const float apx = px - seg_a_[k].x(), apy = py - seg_a_[k].y();
            const float abx = seg_ab_[k].x(), aby = seg_ab_[k].y();
            // t = clamp(dot(ap, ab) / |ab|², 0, 1)
            float t = (apx * abx + apy * aby) / (seg_ab_sq_[k] + 1e-8f);
            t = std::clamp(t, 0.0f, 1.0f);
            const float cx = seg_a_[k].x() + t * abx, cy = seg_a_[k].y() + t * aby;
            const float ddx = px - cx, ddy = py - cy;
            const float d2 = ddx * ddx + ddy * ddy;
            if (d2 < best_d2) { best_d2 = d2; best = k; best_cx = cx; best_cy = cy; }   // first minimum
        }
        const float len = std::sqrt(seg_ab_sq_[best] + 1e-8f);
        out.closest_normals(i, 0) = -seg_ab_[best].y() / len;
        out.closest_normals(i, 1) =  seg_ab_[best].x() / len;
        const float dist = std::sqrt(best_d2 + 1e-8f);
        out.sdf(i) = dist;
        // ∇SDF for the UNSIGNED polygon distance = (p − closest)/|p − closest|: the unit vector pointing
        // from the closest boundary point towards p.
        if (want_grad)
        {
            const float inv = 1.f / std::max(dist, 1e-6f);
            out.grad(i, 0) = (px - best_cx) * inv;
            out.grad(i, 1) = (py - best_cy) * inv;
        }
    }
}

Eigen::Matrix<float, 5, 1> Model::get_state() const
{
    if (not has_state())
        throw std::runtime_error("Model::get_state() on an uninitialised model "
                                 "(init_from_state/init_from_polygon has not run)");
    Eigen::Matrix<float, 5, 1> s;
    s << 2.f * half_extents.x(), 2.f * half_extents.y(), robot_pos.x(), robot_pos.y(), robot_theta;
    return s;
}

} // namespace rc
