#pragma once

#include <Eigen/Dense>
#include <QString>

#include <string>
#include <vector>

namespace rc
{
class SvgRoomLoader
{
public:
    // Reads a polygon from an SVG file ("polygon" element with the given id) and returns 2D points.
    static std::vector<Eigen::Vector2f> load_polygon_points(const std::string& svg_file,
                                                             const std::string& polygon_id = "room_contour",
                                                             bool flip_y = false,
                                                             bool mirror_x = false);

    // Shifts `points` in place so the origin lands on their bounding-box centre, and returns the
    // offset that was subtracted (i.e. the OLD-frame coordinates of the NEW origin).
    //
    // The room frame is defined by nothing but these coordinates, so wherever the SVG author put
    // (0,0) becomes the room origin — a corner for a plan traced from (0,0) (apartamento_layout),
    // the centre for a plan drawn about its middle (beta_layout, room_4x4). Recentring makes the
    // origin the room's geometric centre for EVERY layout, which is what the cold-start bootstrap
    // already assumes: it seeds the pose by placing the LiDAR OBB centre at the room-frame origin.
    //
    // Bounding-box centre, not area centroid, on purpose: the centroid of a non-convex plan (the
    // L-shaped apartment) moves whenever a wall is re-traced, so the frame would silently drift
    // between edits of the SVG. The bbox centre only moves if the room's extent changes.
    static Eigen::Vector2f recenter_to_bbox_center(std::vector<Eigen::Vector2f>& points);

private:
    static std::vector<Eigen::Vector2f> parse_points_attribute(const QString& points_attr,
                                                               bool flip_y,
                                                               bool mirror_x);
};

} // namespace rc
