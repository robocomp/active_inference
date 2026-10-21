// tape_stats — read-only size breakdown of a door_concept DCIN tape (etc/door_inputs.bin, core/door_cycle_io.h).
// For every record it measures each field family as the serialised size of the record MINUS the size of the same record
// with that family cleared, plus how often each media stream actually CHANGED between consecutive records (equal-to-
// previous counts). ★Sizes are always those of the CURRENT encoding (v2): on a v1 tape the table therefore says what that
// same content costs in v2, next to the file's own v1 rate. Reads v1 and v2 tapes; streams the file.
// --reencode OUT writes the tape again in the current version and checks every re-read record equals the original.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/tape_stats.cpp src/core/door_cycle_io.cpp -lzstd -o /tmp/tape_stats
//   /tmp/tape_stats <door_inputs.bin> [--reencode OUT]
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <print>
#include <sstream>
#include <string>
#include <vector>
#include "../src/core/door_cycle_io.h"

using namespace rc::door::core;

namespace
{
std::size_t size_of(const DoorCycleInput& in)
{
    std::ostringstream o(std::ios::binary);
    (void)write(o, in);
    return o.str().size();
}
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::println("usage: tape_stats <door_inputs.bin> [--reencode OUT]"); return 2; }
    const std::string reencode = (argc >= 4 and std::string(argv[2]) == "--reencode") ? argv[3] : std::string{};
    std::ifstream f(argv[1], std::ios::binary);
    if (not f) { std::println("cannot open {}", argv[1]); return 1; }
    std::uint32_t version = 0;
    const auto hdr = read_header(f, RecordingKind::Input, &version);
    if (not hdr) { std::println("header refused (wrong magic/version?)"); return 1; }
    std::ofstream out;
    if (not reencode.empty())
    {
        out.open(reencode, std::ios::binary);
        if (not out or not write_header(out, *hdr, RecordingKind::Input)) { std::println("cannot write {}", reencode); return 1; }
    }
    std::ifstream verify;   // reads OUT back as it is written
    const auto header_bytes = static_cast<std::uint64_t>(f.tellg());

    using Clear = std::function<void(DoorCycleInput&)>;
    const std::vector<std::pair<std::string, Clear>> fam{
        {"masks (all but pixels/points)", [](DoorCycleInput& x) { if (x.masks) { x.masks->slices.clear(); x.masks->room_zed_chain.clear(); } }},
        {"masks.support_points (room+cam)", [](DoorCycleInput& x) { if (x.masks) { x.masks->support_points.clear(); x.masks->support_points_cam.clear(); } }},
        {"masks.mask_pixels", [](DoorCycleInput& x) { if (x.masks) x.masks->mask_pixels.clear(); }},
        // ★ONE family: in v2 the room plane is coded against the robot plane, so clearing one alone moves cost to the other
        {"bpearl room+robot planes", [](DoorCycleInput& x) { if (x.bpearl) { x.bpearl->pts_room.clear(); x.bpearl->pts_robot.clear(); } }},
        {"semantic plane", [](DoorCycleInput& x) { x.semantic.reset(); }},
        {"rgb_gray patches", [](DoorCycleInput& x) { x.rgb_gray.reset(); }},
        {"depth patches", [](DoorCycleInput& x) { x.depth.reset(); }},
        {"rooms", [](DoorCycleInput& x) { x.rooms = {}; }},
        {"walls", [](DoorCycleInput& x) { x.walls.clear(); }},
        {"objects", [](DoorCycleInput& x) { x.objects.clear(); }},
        {"robot+camera", [](DoorCycleInput& x) { x.robot = {}; x.camera = {}; }},
        {"affordances+executing", [](DoorCycleInput& x) { x.affordances.clear(); x.executing.clear(); }},
        {"events+identity_load", [](DoorCycleInput& x) { x.events.clear(); x.identity_load.reset(); }},
        {"actuator_calls", [](DoorCycleInput& x) { x.actuator_calls.clear(); }},
    };
    std::vector<std::uint64_t> bytes(fam.size(), 0);
    std::uint64_t total = 0, n = 0;
    std::int64_t t0 = 0, t1 = 0;
    // present / changed-vs-previous counters per stream
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> pres;   // name → (present, same-as-previous)
    std::optional<MaskPacket> pm; std::optional<BpearlSweep> pb; std::optional<SemanticPlane> ps;
    std::optional<ImageFrame> pg, pd;
    RoomResolution pr; std::vector<WallInfo> pw; std::vector<ObjectInfo> po;
    auto note = [&](const std::string& k, bool present, bool same) { auto& c = pres[k]; c.first += present; c.second += same; };

    while (true)
    {
        auto rec = read_input(f, version);
        if (not rec) break;
        const auto& in = rec.value();
        if (out.is_open())
        {
            if (not write(out, in)) { std::println("write failed at record {}", n); return 1; }
            out.flush();
            if (not verify.is_open())
            {
                verify.open(reencode, std::ios::binary);
                std::uint32_t v2 = 0;
                if (not read_header(verify, RecordingKind::Input, &v2)) { std::println("re-read header failed"); return 1; }
            }
            const auto back = read_input(verify);
            if (not back or not (*back == in)) { std::println("RE-ENCODE MISMATCH at record {} (cycle {})", n, in.cycle); return 1; }
        }
        if (n == 0) t0 = in.t_steady_ns;
        t1 = in.t_steady_ns;
        const std::size_t whole = size_of(in);
        total += whole;
        for (std::size_t i = 0; i < fam.size(); ++i)
        {
            DoorCycleInput c = in;
            fam[i].second(c);
            bytes[i] += whole - size_of(c);
        }
        note("masks", in.masks.has_value(), in.masks.has_value() and pm.has_value() and *pm == *in.masks);
        if (in.masks) pm = in.masks;
        note("bpearl", in.bpearl.has_value(), in.bpearl.has_value() and pb.has_value() and *pb == *in.bpearl);
        if (in.bpearl) pb = in.bpearl;
        note("semantic", in.semantic.has_value(), in.semantic.has_value() and ps.has_value() and ps->plane == in.semantic->plane);
        if (in.semantic) ps = in.semantic;
        note("rgb_gray", in.rgb_gray.has_value(), in.rgb_gray.has_value() and pg.has_value() and *pg == *in.rgb_gray);
        if (in.rgb_gray) pg = in.rgb_gray;
        note("depth", in.depth.has_value(), in.depth.has_value() and pd.has_value() and *pd == *in.depth);
        if (in.depth) pd = in.depth;
        note("rooms", true, n > 0 and pr == in.rooms); pr = in.rooms;
        note("walls", true, n > 0 and pw == in.walls); pw = in.walls;
        note("objects", true, n > 0 and po == in.objects); po = in.objects;
        ++n;
    }
    const double secs = (t1 - t0) * 1e-9;
    const auto file_size = static_cast<double>(std::ifstream(argv[1], std::ios::binary | std::ios::ate).tellg());
    std::println("tape {} (v{})  file {:.1f} MB = {:.2f} MB/s  header {} B  records {}  span {:.1f} s  {:.2f} cycles/s",
                 argv[1], version, file_size / 1e6, secs > 0 ? file_size / 1e6 / secs : 0.0, header_bytes, n, secs,
                 secs > 0 ? n / secs : 0.0);
    std::println("in the CURRENT encoding (v{}): {:.1f} MB = {:.2f} MB/s ({:.2f}x smaller than the file)", kCycleIoVersion,
                 total / 1e6, secs > 0 ? total / 1e6 / secs : 0.0, total > 0 ? file_size / total : 0.0);
    if (out.is_open())
        std::println("re-encoded to {}: {:.1f} MB, every record re-read EQUAL to the original",
                     reencode, static_cast<double>(out.tellp()) / 1e6);
    std::println("{:<34} {:>12} {:>8} {:>10} {:>10}", "family", "MB", "%", "KB/rec", "MB/s");
    std::uint64_t acc = 0;
    for (std::size_t i = 0; i < fam.size(); ++i)
    {
        acc += bytes[i];
        std::println("{:<34} {:>12.1f} {:>7.1f}% {:>10.1f} {:>10.3f}", fam[i].first, bytes[i] / 1e6, 100.0 * bytes[i] / total,
                     n ? bytes[i] / 1e3 / n : 0.0, secs > 0 ? bytes[i] / 1e6 / secs : 0.0);
    }
    std::println("{:<34} {:>12.1f} {:>7.1f}%", "(rest: clocks, framing, small fields)", (total - acc) / 1e6, 100.0 * (total - acc) / total);
    std::println("\nstream presence (records carrying it / of those, identical to the previous one carried):");
    for (const auto& [k, c] : pres)
        std::println("  {:<32} present {:>6}  same-as-previous {:>6}", k, c.first, c.second);
    return 0;
}
