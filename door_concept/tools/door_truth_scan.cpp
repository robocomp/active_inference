// door_truth_scan — Task 5: ground-truth labels from a DCIN tape of the 3-room apartment world (threerooms-piso.wbt).
// Writes two CSVs into OUT_DIR:
//   room_pose.csv   cycle,t_ms,room,tx,ty,yaw     world<-current-room SE(2), from the sim's robot_gt (x, y, angle) and the cycle's room<-body
//                   (lets a script put a room-frame door centre — door_existence_log cx,cy — into the Webots world and match it
//                   to a true door).
//   lidar_bins.csv  cycle,t_ms,door,state,region,obs,thr   per low-LiDAR sweep and true door: how many wall-line bins of the sweep
//                   were OBSERVED (a through or a blocked return) and how many read THROUGH, split into the true aperture
//                   (region ap, bins whose centre is 5 cm inside the jambs) and the solid wall beside it (region wall: the stretches listed in
//                   gt_doors(), 0.15 m clear of the jambs, within the registration window). state = open once an actuator OPEN request for that door is >= 10 s
//                   old, shut before it; the 10 s after a request (the leaf swinging) are skipped. The bins are door_opening_edges.h's own (through_bin_counts), on the registered sweep, in the
//                   WORLD frame, so the room's mapping error does not enter.
// robot_gt_angle is NOT the heading in the graph's axes (see the conversion below); the labelling is validated on tape_e2e18 (room_1 yaw sd
// 0.03 rad, 2-3 cm in translation over 777 cycles).
// Usage: door_truth_scan <door_inputs.bin> <out_dir>   (env TRUTH_PROFILE=1 also prints the through fraction along each door's wall)
// Build (from door_concept/): g++ -std=c++23 -O2 -w -I/usr/include/eigen3 tools/door_truth_scan.cpp src/core/door_cycle_io.cpp -lzstd -o door_truth_scan
#include <clocale>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <array>
#include <cstdlib>
#include <map>
#include <print>
#include "../src/core/door_cycle_io.h"
#include "../src/door_opening_edges.h"
using namespace rc::door::core;

// wall_lo/hi: the stretches (along-wall coordinate a from the aperture's near jamb, m) that are SOLID WALL within the +-1 m window,
// 0.15 m clear of the jambs, from door_apertures.json's wall extents: DOOR_0's wall ends at x -4.172 (a -0.089: beyond it is the next
// room's open floor), DOOR_1's wall runs a -0.6 .. 1.075 on the far side only as far as the json segment, DOOR_2's extent is unknown.
struct GtDoor { const char* name; double cx, cy, ux, uy; double w0lo, w0hi, w1lo, w1hi; };   // centre (world), the wall's unit direction, two solid-wall stretches
// threerooms-piso.wbt: DOOR_0 y = 7.166, opening x -4.083..-3.083; DOOR_1 wall a(-5.871,9.256) b(-4.196,9.231) (door_apertures.json),
// opening centred on x = -4.771; DOOR_2 node (-1.705, 10.77), wall along y (rotation pi, size 0.1 1 2.1).
static std::vector<GtDoor> gt_doors()
{
    const double bx = -4.196 + 5.871, by = 9.231 - 9.256, bl = std::hypot(bx, by);
    const double t = (-4.771 + 5.871) / bx;
    const double nan = std::nan("");
    return {{"DOOR_0", -3.583, 7.166, 1.0, 0.0, 1.15, 2.0, nan, nan}, {"DOOR_1", -4.771, 9.256 + t * by, bx / bl, by / bl, -0.5, -0.15, nan, nan},
            {"DOOR_2", -1.705, 10.77, 0.0, 1.0, nan, nan, nan, nan}};
}

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");
    if (argc < 3) { std::println("usage: door_truth_scan <door_inputs.bin> <out_dir>"); return 2; }
    std::ifstream f(argv[1], std::ios::binary);
    std::uint32_t v = 0;
    auto h = read_header(f, RecordingKind::Input, &v);
    (void)h;
    std::filesystem::create_directories(argv[2]);
    std::ofstream rp(std::filesystem::path(argv[2]) / "room_pose.csv"), lb(std::filesystem::path(argv[2]) / "lidar_bins.csv");
    rp.imbue(std::locale::classic()); lb.imbue(std::locale::classic());
    rp << "cycle,t_ms,room,tx,ty,yaw,gx,gy,gyaw,bx,by,byaw\n";
    lb << "cycle,t_ms,door,state,region,obs,thr,dist,side\n";
    const auto doors = gt_doors();
    std::map<std::string, std::int64_t> opened_ms, closed_ms;   // door -> time of its last OPEN request / of the CLOSE after it (none: still open)
    std::map<std::string, std::array<std::array<int, 2>, 60>> prof;   // door/state/side -> per 5 cm along-wall bin: (observed, through)
    long n_sweeps = 0, n_unreg = 0, n_rows = 0;
    while (auto rec = read_input(f, v))
    {
        const auto& in = *rec;
        if (not in.robot.room_T_body or not in.robot.gt or not in.rooms.graph_current_room) continue;
        const auto& B = *in.robot.room_T_body;
        const double yb = std::atan2(B(1, 0), B(0, 0));
        // robot_gt_angle is the sim's Webots heading seen from the graph's axes: gt = -(heading) - pi/2 (measured on tape_e2e18: room_1 yaw
        // sd 0.03 rad, translation sd 2-3 cm over 777 cycles; a Kabsch fit of the positions agrees), so heading = -gt - pi/2.
        const double yw = -static_cast<double>(in.robot.gt->z()) - M_PI / 2 - yb;
        const double c = std::cos(yw), s = std::sin(yw);
        const double tx = in.robot.gt->x() - (c * B(0, 3) - s * B(1, 3));
        const double ty = in.robot.gt->y() - (s * B(0, 3) + c * B(1, 3));
        rp << in.cycle << ',' << in.t_ms() << ',' << *in.rooms.graph_current_room << ',' << tx << ',' << ty << ',' << yw << ',' << in.robot.gt->x() << ',' << in.robot.gt->y() << ',' << in.robot.gt->z() << ',' << B(0, 3) << ',' << B(1, 3) << ',' << yb << '\n';
        for (const auto& a : in.actuator_calls)
            if (a.kind == ActuatorCallKind::Request)
            {
                const Eigen::Vector4d p = a.world_T_room * Eigen::Vector4d(a.xy_m.x(), a.xy_m.y(), 0, 1);
                double best = 1e9; std::string nm;
                for (const auto& d : doors) { const double e = std::hypot(p.x() - d.cx, p.y() - d.cy); if (e < best) { best = e; nm = d.name; } }
                if (best >= 1.5) continue;
                std::println("cycle {} {} {} ({:.2f} m)", in.cycle, a.open ? "OPEN" : "CLOSE", nm, best);
                // the state is the LAST request's: a close ends the open state (the next open starts a new swing)
                if (a.open and not (opened_ms.contains(nm) and not closed_ms.contains(nm))) { opened_ms[nm] = in.t_ms(); closed_ms.erase(nm); }
                if (not a.open and opened_ms.contains(nm) and not closed_ms.contains(nm)) closed_ms[nm] = in.t_ms();
            }
        if (not in.bpearl or not in.bpearl->room_fresh or in.bpearl->pts_room.empty()) continue;
        ++n_sweeps;
        const auto W = [&](const Eigen::Vector3f& p) { return Eigen::Vector3f(static_cast<float>(c * p.x() - s * p.y() + tx), static_cast<float>(s * p.x() + c * p.y() + ty), p.z()); };
        std::vector<Eigen::Vector3f> pts; pts.reserve(in.bpearl->pts_room.size());
        for (const auto& p : in.bpearl->pts_room) pts.push_back(W(p));
        const Eigen::Vector3f org = W(in.bpearl->origin_room);
        for (const auto& d : doors)
        {
            if (std::hypot(org.x() - d.cx, org.y() - d.cy) > 8.0) continue;
            rc::door::Aperture ap;
            ap.w = 1.0f; ap.h = 2.1f; ap.s = 0.0f;
            ap.wall_u = Eigen::Vector2f(static_cast<float>(d.ux), static_cast<float>(d.uy));
            ap.wall_O = Eigen::Vector2f(static_cast<float>(d.cx), static_cast<float>(d.cy)) - 0.5f * ap.wall_u;
            const auto rs = rc::door::register_sweep(ap, org, pts);
            if (not rs.reg.ok) { ++n_unreg; continue; }
            const auto tb = rc::door::through_bin_counts(ap, rs.origin, rs.pts);
            if (tb.nb == 0) continue;
            int obs_ap = 0, thr_ap = 0, obs_w = 0, thr_w = 0;
            for (int b = 0; b < tb.nb; ++b)
            {
                const int T = tb.nT[static_cast<std::size_t>(b)], Bk = tb.nB[static_cast<std::size_t>(b)];
                if (T + Bk == 0) continue;
                const float a = tb.a_lo + (static_cast<float>(b) + 0.5f) * rc::door::kEdgeBinM;
                if (std::getenv("TRUTH_PROFILE")) { const int pb = static_cast<int>(std::floor((a + 0.5f) / 0.05f)); if (pb >= 0 and pb < 60) { auto& pr = prof[std::string(d.name) + (opened_ms.contains(d.name) and in.t_ms() >= opened_ms[d.name] + 10000 and not closed_ms.contains(d.name) ? " open " : " shut ") + ((-d.uy * (org.x() - d.cx) + d.ux * (org.y() - d.cy)) > 0 ? "+" : "-")][static_cast<std::size_t>(pb)]; ++pr[0]; pr[1] += (T > 0 and Bk == 0); } } 
                const bool thr = T > 0 and Bk == 0;
                if (a >= ap.s + 0.05f and a <= ap.s + ap.w - 0.05f) { ++obs_ap; thr_ap += thr; }
                else if ((a >= d.w0lo and a <= d.w0hi) or (a >= d.w1lo and a <= d.w1hi)) { ++obs_w; thr_w += thr; }
            }
            const auto it = opened_ms.find(d.name);
            // the leaf takes ~7 s to swing (tape_e2e18 DOOR_0: request at cycle 702, through bins from 716, full by 732): the 10 s after
            // a request are a transition, not a state
            const auto ct = closed_ms.find(d.name);
            if (it != opened_ms.end() and in.t_ms() >= it->second and in.t_ms() < it->second + 10000) continue;
            if (ct != closed_ms.end() and in.t_ms() >= ct->second and in.t_ms() < ct->second + 10000) continue;   // a CLOSE swings too
            const bool is_open = it != opened_ms.end() and in.t_ms() >= it->second + 10000 and (ct == closed_ms.end() or in.t_ms() < ct->second);
            const char* state = is_open ? "open" : "shut";
            const double dist = std::hypot(org.x() - d.cx, org.y() - d.cy);
            const int side = (-d.uy * (org.x() - d.cx) + d.ux * (org.y() - d.cy)) > 0 ? 1 : -1;   // which side of the wall the sensor is on
            lb << in.cycle << ',' << in.t_ms() << ',' << d.name << ',' << state << ",ap," << obs_ap << ',' << thr_ap << ',' << dist << ',' << side << '\n'
               << in.cycle << ',' << in.t_ms() << ',' << d.name << ',' << state << ",wall," << obs_w << ',' << thr_w << ',' << dist << ',' << side << '\n';
            n_rows += 2;
        }
    }
    for (const auto& [k, a] : prof) { std::print("{}:", k); for (const auto& x : a) std::print(" {}", x[0] ? static_cast<int>(100.0 * x[1] / x[0]) : -1); std::println(""); }
    std::println("sweeps {} unregistered {} lidar rows {}", n_sweeps, n_unreg, n_rows);
}
