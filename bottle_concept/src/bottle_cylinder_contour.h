/*
 * bottle_cylinder_contour.h — pure geometry (Eigen + OpenCV, no DSR) of the bottle's image outline.
 *
 * ★THE OUTLINE OF AN UPRIGHT CYLINDER IS THE HULL OF ITS TWO RIMS. Seen from any camera outside it, the
 * silhouette is bounded by two straight generators (the limb lines) and, top and bottom, by the arcs of the
 * two circular rims — exactly the convex hull of the projected rims. Every edge of that hull is an OCCLUDING
 * boundary of the solid (the depth scorer tests recession on all of them) and none is a joint inside the
 * object. The hull is chosen IN THE IMAGE, because which rim arcs bound the solid depends on the camera; the
 * hull's 3-D vertices are then handed, in hull order, to the shared loop entry (project_loop_slid).
 *
 * ★THE NULL is the shared one: the same shape slid horizontally on the support plane, perpendicular to the
 * viewing ray (a cylinder has no yaw, so no preferred in-plane axis; perpendicular to the ray keeps the
 * displaced copy at the same range and foreshortening).
 *
 * ★THE PROJECTION STAYS A CALLBACK (PointProjector), as the shared unit requires: near-clip, the stamp the pose
 * is pinned to and the intrinsic-vs-delivered-frame scale belong to the caller. That also makes this testable
 * with a plain pinhole and no DSR.
 */

#pragma once

#include <Eigen/Dense>

#include "../../common/contour_edge/contour_edge_project.h"

namespace rc
{

// An upright cylinder (centre cx,cy; z0..z1; radius r) → contour + slid null. Empty `face` = not projectable
// this frame (a rim sample behind the camera): NOT MEASURED, never a refutation.
[[nodiscard]] rc::edges::ContourSet cylinder_contour_set(float cx, float cy, float z0, float z1, float radius,
                                                         const Eigen::Vector3d& camera_pos_world,
                                                         const rc::edges::PointProjector& project);

}   // namespace rc
