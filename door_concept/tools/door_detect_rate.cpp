// door_detect_rate — Task D-C defect 4: how often the YOLO "door" masks detect each GROUND-TRUTH Webots door, by leaf state
// (shut until an actuator OPEN for it appears on the tape, open after), wall line of sight and range, from a DCIN tape. The
// geometry is the simulator's, not the agent's beliefs: the GT robot pose (robot_gt) gives room_1's offset to the world
// (median of the first 50 room_1 cycles; cycles whose pose disagrees with it by > 0.25 m are skipped — e.g. a relocalisation
// gone wrong), the GT door centre at 1.05 m is projected with the cycle's room←zed, and a cycle counts as IN VIEW when that
// centre falls in the central 80 % of the image width, in front of the camera. "los"/"occl": whether a room_1 polygon edge
// (the door's own wall excluded) crosses the camera→door sightline. DETECTED: a zed "door" slice whose pixel columns
// (±40 px) contain the projected centre. Only room_1 cycles are used (its frame is the world's, translated).
// Usage: door_detect_rate <door_inputs.bin> piso|three      (piso.wbt = rtD1i, tourR1/R2; threerooms-piso.wbt = tour3A)
// Build (from door_concept/): g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/door_detect_rate.cpp src/core/door_cycle_io.cpp -lzstd -o /tmp/door_detect_rate
#include <fstream>
#include <map>
#include <print>
#include <cmath>
#include "../src/core/door_cycle_io.h"
using namespace rc::door::core;
struct GtDoor { const char* name; double x, y; };
static bool seg_cross(Eigen::Vector2d a, Eigen::Vector2d b, Eigen::Vector2d c, Eigen::Vector2d d)
{
    auto cr = [](Eigen::Vector2d p, Eigen::Vector2d q) { return p.x() * q.y() - p.y() * q.x(); };
    const Eigen::Vector2d r = b - a, s = d - c;
    const double den = cr(r, s);
    if (std::abs(den) < 1e-12) return false;
    const double t = cr(c - a, s) / den, u = cr(c - a, r) / den;
    return t > 1e-3 and t < 1 - 1e-3 and u >= 0 and u <= 1;
}
int main(int argc, char** argv)
{
    if (argc < 3 or (std::string(argv[2]) != "piso" and std::string(argv[2]) != "three"))
    { std::println("usage: door_detect_rate <door_inputs.bin> piso|three"); return 2; }
    std::ifstream f(argv[1], std::ios::binary);
    std::uint32_t v = 0; auto h = read_header(f, RecordingKind::Input, &v);
    const bool three = std::string(argv[2]) == "three";
    std::vector<GtDoor> doors = three ? std::vector<GtDoor>{{"DOOR_0", -3.59333, 7.22555}, {"DOOR_1", -4.76901, 9.3796}}
                                      : std::vector<GtDoor>{{"DOOR_0", -3.58277, 7.166}, {"DOOR_1", -4.77112, 9.23958}};
    std::map<std::string, bool> open;   // after an actuator OPEN for it
    // room_1 offset: gt - room, median over the first 200 room_1 cycles with a body pose
    std::vector<double> ox, oy; std::uint64_t room1 = 0;
    std::map<std::string, std::array<long, 2>> stat;   // key → (in view, detected)
    long n = 0;
    double offx = 0, offy = 0; bool have_off = false;
    while (auto rec = read_input(f, v))
    {
        const auto& in = *rec;
        for (const auto& a : in.actuator_calls)
            if (a.kind == ActuatorCallKind::Request and a.open)
            {
                // map to the GT door nearest the request place (world_T_room ∘ xy)
                const Eigen::Vector4d p = a.world_T_room * Eigen::Vector4d(a.xy_m.x(), a.xy_m.y(), 0, 1);
                double best = 1e9; std::string nm;
                for (const auto& d : doors) { const double e = std::hypot(p.x() - d.x, p.y() - d.y); if (e < best) { best = e; nm = d.name; } }
                if (not open[nm]) std::println("cycle {} OPEN {} ({:.2f} m)", in.cycle, nm, best);
                open[nm] = true;
            }
        if (not in.robot.room_T_body or not in.robot.room_T_zed or not in.robot.gt or not in.masks or not in.masks->valid) continue;
        std::string rname;
        const RoomInfo* cur = nullptr;
        if (in.rooms.graph_current_room)
            for (const auto& r : in.rooms.rooms) if (r.id == *in.rooms.graph_current_room) { rname = r.name; cur = &r; }
        if (rname != "room_1" or not cur) continue;
        const auto& B = *in.robot.room_T_body;
        const double gx = in.robot.gt->x() - B(0, 3), gy = in.robot.gt->y() - B(1, 3);
        if (not have_off) { ox.push_back(gx); oy.push_back(gy); if (ox.size() == 50) { std::ranges::sort(ox); std::ranges::sort(oy); offx = ox[25]; offy = oy[25]; have_off = true; std::println("room_1 offset {:.3f} {:.3f}", offx, offy); } continue; }
        if (std::hypot(gx - offx, gy - offy) > 0.25) continue;     // pose not tracking: skip
        if (not in.camera.present or in.camera.width == 0) continue;
        const Mat4d Zi = in.robot.room_T_zed->inverse();
        ++n;
        for (const auto& d : doors)
        {
            const Eigen::Vector2d c(d.x - offx, d.y - offy);
            // occlusion by the room polygon's walls, the door's own wall excluded (edges within 0.3 m of the centre's line y)
            const Eigen::Vector2d cam((*in.robot.room_T_zed)(0, 3), (*in.robot.room_T_zed)(1, 3));
            bool occl = false;
            const auto& P = cur->polygon;
            for (std::size_t i = 0; i < P.size() and not occl; ++i)
            {
                const Eigen::Vector2d a = P[i].cast<double>(), b = P[(i + 1) % P.size()].cast<double>();
                // the door's OWN wall: both ends inside the doorway plane's depth band (±0.20 m, the leaf field's
                // FieldNuisance depth_lo/hi — rc::door::door_wall_edges' rule; both GT doors lie in walls along x)
                if (std::abs(a.y() - c.y()) <= 0.20 and std::abs(b.y() - c.y()) <= 0.20) continue;
                occl = seg_cross(cam, c - 0.15 * (c - cam).normalized(), a, b);
            }
            const Eigen::Vector4d pc = Zi * Eigen::Vector4d(c.x(), c.y(), 1.05, 1.0);
            if (pc.y() < 0.3) continue;   // behind (or at) the camera
            const double u = in.camera.focal_x * pc.x() / pc.y() + in.camera.width / 2, vv = -in.camera.focal_y * pc.z() / pc.y() + in.camera.height / 2;
            if (u < 0.1 * in.camera.width or u > 0.9 * in.camera.width or vv < 0 or vv > in.camera.height) continue;
            const double range = (c - cam).norm();
            bool det = false;
            for (const auto& sl : in.masks->slices)
            {
                if (sl.label != "door" or sl.source != 0) continue;
                double c0 = 1e9, c1 = -1e9;
                for (auto i = sl.pixel_begin; i < sl.pixel_end and i < in.masks->mask_pixels.size(); ++i)
                { c0 = std::min<double>(c0, in.masks->mask_pixels[i].x()); c1 = std::max<double>(c1, in.masks->mask_pixels[i].x()); }
                if (u >= c0 - 40 and u <= c1 + 40) det = true;
            }
            const int rb = std::min(4, static_cast<int>(range));
            const std::string key = std::format("{} {} {} r{}", d.name, open[d.name] ? "open" : "shut", occl ? "occl" : "los ", rb);
            stat[key][0]++; stat[key][1] += det;
        }
    }
    std::println("room_1 fresh-mask cycles used: {}", n);
    for (const auto& [k, s] : stat) std::println("{:28} in_view {:6} detected {:6} rate {:.3f}", k, s[0], s[1], s[0] ? double(s[1]) / s[0] : 0.0);
}
