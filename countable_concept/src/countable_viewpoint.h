/*
 * countable_viewpoint.h — how much a sensor's SHARED error has decorrelated since the channel last updated the
 * instance (spec 2026-11-15 lidar-measures §7c.2). Pure Eigen, no OpenCV/DSR/Qt.
 *
 * A sensor's systematic error (a bias in where the camera sees the edge, a ring-gap aliasing on the LiDAR) is
 * the same on consecutive frames and only decorrelates when the VIEWPOINT changes. rho in [0,1] is the
 * remaining correlation: 1 = parked (the shared error is unchanged, so re-measuring adds nothing), 0 = fully
 * decorrelated. It is a continuous function of the odometry path relative to the object — no gate.
 * Rotation in place moves neither the sensor's bearing to the object nor its range, so it leaves rho = 1.
 */
#pragma once

#include <Eigen/Dense>

namespace rc {

// Accumulated since the channel's last update of the instance; the caller resets it after an update.
struct ViewpointPath
{
    float dphi_rad = 0.f;   // |bearing change| as seen from the object, summed (tangential path / range)
    float dz_m     = 0.f;   // |range change| to the object, summed (radial path)
};

// Adds |step·t̂|/z to dphi and |step·r̂| to dz, r̂ = unit(sensor − object), t̂ ⊥ r̂, z = |sensor − object|.
void accumulate_path(ViewpointPath& p, const Eigen::Vector2f& sensor_xy_room,
                     const Eigen::Vector2f& step_xy_room, const Eigen::Vector2f& object_xy);

// Room-frame displacement of the robot over dt. Robot frame: +x right, +y FORWARD (controller/FRAMES.md,
// ROBOT_GEOMETRY.md), CCW-positive yaw; v_adv is along +y, v_side along +x; rotated into the room by yaw_room.
Eigen::Vector2f odometry_step_room(float v_adv, float v_side, float yaw_room, float dt_s);

// exp(−|Δφ|/atan(w⊥/2z) − |Δz|/z): e-fold when the camera has moved by the object's angular half-width.
float rho_camera(const ViewpointPath& p, float z_m, float w_perp_m);
// exp(−|Δφ|/g), g = ring gap (rad). g ≤ 0 (SensorModel absent) ⇒ 1.
float rho_lidar_lateral(const ViewpointPath& p, float ring_gap_rad);
// exp(−|Δz||tan ε|/(g z)), ε = elevation of the beam. g ≤ 0 ⇒ 1.
float rho_lidar_height(const ViewpointPath& p, float z_m, float eps_rad, float ring_gap_rad);
// exp(−½(cx²/var_xx + cy²/var_yy)). A zero variance contributes 0 when its correction is 0, else rho = 0.
float rho_chain(const Eigen::Vector2f& correction_xy, float var_xx, float var_yy);

}  // namespace rc
