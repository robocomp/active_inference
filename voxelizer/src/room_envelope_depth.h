#pragma once

/*
 * room_envelope_depth.{h,cpp} — what the ROOM BELIEF predicts the camera should see.
 *
 * For every pixel, cast a ray from the ZED through it and intersect the room envelope that
 * room_concept believes in: the floor (z = 0), the ceiling (z = room_height), and one vertical wall
 * per edge of the room polygon. The nearest positive hit is the depth the belief PREDICTS at that
 * pixel.
 *
 * Differenced against the ZED's MEASURED depth, this is the room belief's per-pixel residual made
 * visible — the same quantity the free energy is built from, rendered as an image. Everything the
 * room model does not explain stands out: furniture, people, clutter (bright), and genuine model
 * error such as a mis-fitted wall or a wrong ceiling height (also bright, but structured — a whole
 * surface tilting or offsetting rather than an object-shaped blob).
 *
 * ★This is the room ENVELOPE only. Furniture is deliberately NOT included, so a table reads as a
 * large residual. That is the intended reading: the difference image answers "what is in this room
 * beyond its shell", which is exactly the question residual_concept exists to answer, shown per pixel
 * in the camera's own view instead of as grid cells.
 *
 * Camera convention (must match the deprojection in graph_publisher.cpp): the DSR `zed` frame is
 * x-right, **y-DEPTH**, z-up. A pixel's ray is d_cam = ((col−cx)/fx, 1, (cy−row)/fy), so a point at
 * ZED-depth t is exactly t·d_cam — meaning the ray parameter t IS the depth the ZED reports, and no
 * conversion is needed anywhere below.
 */

#include <opencv2/core.hpp>

#include <span>

#include <Eigen/Geometry>
#include <dsr/api/dsr_eigen_defs.h>   // Mat::RTMat

namespace DSR { class CameraAPI; }

namespace rc::depth
{

// Predicted depth (metres, ZED convention) for every pixel; NaN where the ray escapes the envelope.
// `decimate` computes on a 1/decimate grid and upsamples — the envelope is piecewise planar, so the
// interpolation error is negligible away from edges and it keeps a 640x480 pass to a few ms.
// Returns an empty Mat if the polygon or the intrinsics are unusable.
[[nodiscard]] cv::Mat room_envelope_depth(int width, int height, float fx, float fy,
                                          const Mat::RTMat& room_T_zed,
                                          std::span<const float> poly_x,
                                          std::span<const float> poly_y,
                                          float room_height, int decimate = 2);

// Equirectangular counterpart, for the ricoh panorama. Returns RANGE along the ray (not a z-depth),
// which is what the monocular map produces once zdepth_to_range is on — so the two are directly
// comparable with no conversion.
//
// ★Rays come from DSR::CameraAPI::ray_from_pixel(), never from a formula written here. It is the
// exact inverse of project() and — critically — it DISPATCHES ON THE NODE'S PROJECTION MODEL:
// Equirectangular uses asin for elevation, Cylindrical (the Webots 360 camera, which is what the
// ricoh is) uses tan, and Pinhole is a third case. A hand-rolled "equirect" inverse is silently wrong
// in the vertical axis on a cylindrical camera — the room looks plausible and is bent. It also reads
// the azimuth sign/offset from the graph, so the panorama's mirror and seam-zero convention come from
// the same place project() gets them.
[[nodiscard]] cv::Mat room_envelope_range_equirect(DSR::CameraAPI& cam, int width, int height,
                                                   const Mat::RTMat& room_T_cam,
                                                   std::span<const float> poly_x,
                                                   std::span<const float> poly_y,
                                                   float room_height, int decimate = 2);

}   // namespace rc::depth
