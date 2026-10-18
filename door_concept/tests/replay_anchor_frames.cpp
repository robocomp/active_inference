// Offline replay of door_concept's interaction-anchor frame record (Task 8, fix 3): reads etc/anchor_frames.bin (written
// live by step_passage_anchor, format in src/anchor_frame_record.h), re-runs update_door_anchor on every record with the
// recorded inputs, and prints per frame the recorded vs replayed P(door in view) — p_lidar with its three factors
// (opening coverage · P(door) · walls), p_cam, p_view — and the replayed posterior's distance to the recorded one.
// A replay that disagrees with its own record (pose Δ ≫ 0, or p_lidar off) means the update is no longer the one that
// ran live: rebuild, or check the format version.
//
//   Build + run (from door_concept/):
//     Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O2 -fPIC -I/usr/include/eigen3 -Isrc -I../common -I$Q -I$Q/QtCore
//       -I$Q/QtGui -I$Q/QtWidgets tests/replay_anchor_frames.cpp -o build/replay_anchor_frames -lfastdds -lfastcdr -lQt5Core
//     ./build/replay_anchor_frames [etc/anchor_frames.bin] [--every N]
//
// The recorded p_cam / p_view are AS PUBLISHED (fix 1: the last mask frame's camera term held for kAnchorStaleMs); the
// replayed ones are this frame's own (p_cam > 0 only on a frame that carried a mask), so they differ on mask-less frames
// by design — compare p_lidar for the update itself.
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <print>
#include <string>
#include <vector>

#include "anchor_frame_record.h"

int main(int argc, char** argv)
{
    std::string path = "etc/anchor_frames.bin";
    int every = 1;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--every") == 0 and i + 1 < argc)
        {
            const char* a = argv[++i];
            if (const auto [p, ec] = std::from_chars(a, a + std::strlen(a), every); ec != std::errc{} or every < 1) every = 1;
        }
        else path = argv[i];
    }
    std::ifstream in(path, std::ios::binary);
    if (not in.is_open()) { std::print(stderr, "cannot open {}\n", path); return 2; }
    std::vector<rc::door::rec::Frame> frames;
    std::uint32_t version = 0;
    if (not rc::door::rec::read_all(in, frames, &version)) { std::print(stderr, "{}: not an anchor frame record (or another version)\n", path); return 2; }
    std::print("{}: version {}, {} frames\n", path, version, frames.size());
    std::print("{:>8} {:>5} {:>4} {:>4} | {:>6} {:>6} {:>6} | {:>6} = {:>5} x {:>5} x {:>5} | {:>5} {:>5} | {:>6} {:>6}\n", "t_ms", "rays", "F", "mask",
               "pv_rec", "pl_rec", "pc_rec", "pl_rep", "open", "door", "walls", "pc", "pv", "dxy_m", "dyaw");
    const std::int64_t t0 = frames.empty() ? 0 : frames.front().t_ms;
    double worst_xy = 0, worst_yaw = 0, worst_pl = 0;
    for (std::size_t k = 0; k < frames.size(); ++k)
    {
        const auto& f = frames[k];
        const auto [post, ev] = rc::door::update_door_anchor(f.prior, f.width, f.thickness, f.wall_u_min, f.wall_u_max, f.lidar_origin,
                                                             f.rays, f.mask, f.cam, f.pi_open, f.leaf, f.floor_below);
        const double dxy = (post.x.head<2>() - f.post.x.head<2>()).norm();
        const double dyaw = std::abs(rc::interaction::wrap_pi(post.x.z() - f.post.x.z()));
        worst_xy = std::max(worst_xy, dxy); worst_yaw = std::max(worst_yaw, dyaw);
        worst_pl = std::max(worst_pl, double(std::abs(ev.p_lidar - f.p_lidar)));
        if (k % std::size_t(every) != 0) continue;
        std::print("{:8d} {:5d} {:4d} {:4d} | {:6.3f} {:6.3f} {:6.3f} | {:6.3f} = {:5.3f} x {:5.3f} x {:5.3f} | {:5.3f} {:5.3f} | {:6.4f} {:6.4f}{}\n",
                   f.t_ms - t0, f.rays.size(), ev.n_budget, f.mask.size(), f.p_view, f.p_lidar, f.p_cam, ev.p_lidar, ev.lidar_open,
                   ev.lidar_door, ev.lidar_wall, ev.p_cam, ev.p_view, dxy, dyaw, f.p_view < 0.5f ? "  (not published)" : "");
    }
    std::print("replay vs record: worst pose Δ {:.4f} m / {:.4f} rad, worst |Δ p_lidar| {:.4f}\n", worst_xy, worst_yaw, worst_pl);
    return 0;
}
