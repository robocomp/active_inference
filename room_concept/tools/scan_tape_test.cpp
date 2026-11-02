// Build (from room_concept/):
//   g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/scan_tape_test.cpp -o /tmp/scan_tape_test && /tmp/scan_tape_test
//
// The scan tape (scan_tape.h): round trip is bit-identical, a truncated tail is dropped (not an error), and the
// per-frame write cost is measured (the relocaliser's compute thread pays it at 20 Hz).
#include "scan_tape.h"

#include <chrono>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, const char* what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(c) check((c), __LINE__, #c)

bool same(const rc::tape::Frame& a, const rc::tape::Frame& b)
{
    return a.t_ms == b.t_ms and a.room == b.room
       and std::memcmp(a.pose.data(), b.pose.data(), sizeof(float) * 3) == 0
       and std::memcmp(a.cov.data(), b.cov.data(), sizeof(float) * 9) == 0
       and std::memcmp(a.odo_delta.data(), b.odo_delta.data(), sizeof(float) * 3) == 0
       and std::memcmp(a.odo_cov.data(), b.odo_cov.data(), sizeof(float) * 9) == 0
       and a.pts_robot.size() == b.pts_robot.size()
       and (a.pts_robot.empty() or std::memcmp(a.pts_robot.data(), b.pts_robot.data(), sizeof(float) * 2 * a.pts_robot.size()) == 0);
}

rc::tape::Frame make_frame(std::int64_t t, const std::string& room, int n, float seed)
{
    rc::tape::Frame f;
    f.t_ms = t; f.room = room;
    f.pose = {seed, seed * 2.f, 0.1f * seed};
    f.cov = Eigen::Matrix3f::Identity() * (0.01f + seed);
    f.odo_delta = {0.01f * seed, -0.02f, 0.003f};
    f.odo_cov = Eigen::Matrix3f::Identity() * 1e-4f;
    for (int i = 0; i < n; ++i) f.pts_robot.emplace_back(seed + 0.001f * i, -seed + 0.002f * i);
    return f;
}

rc::tape::RoomRecord make_room(const std::string& name, float s)
{
    rc::tape::RoomRecord r;
    r.room = name;
    r.polygon = {{-s, -s}, {s, -s}, {s, s}, {-s, s}};
    rc::tape::Aperture ap;
    ap.a = {s, -0.5f}; ap.b = {s, 0.5f}; ap.p_open = 0.7f; ap.prior_closed = 0.3f; ap.half_depth = 0.03f;
    ap.recess = 0.1f; ap.n_out = {1.f, 0.f}; ap.sigma_along = 0.05f; ap.name = "door_" + name;
    r.apertures.push_back(ap);
    return r;
}
}

int main()
{
    std::setlocale(LC_ALL, "");
    const auto path = (std::filesystem::temp_directory_path() / "scan_tape_test.bin").string();
    std::filesystem::remove(path);
    const auto r1 = make_room("room_1", 2.f), r2 = make_room("room_2", 3.f);
    const std::vector<rc::tape::Frame> fr = {make_frame(1000, "room_1", 2000, 0.5f), make_frame(1050, "room_1", 1500, 0.6f),
                                             make_frame(1100, "room_2", 0, 0.7f)};
    {
        rc::tape::Writer w(path);
        w.write_room(r1); w.write(fr[0]); w.write(fr[1]);
        w.write_room(r2); w.write(fr[2]);
    }
    const auto size_full = std::filesystem::file_size(path);
    {
        const auto got = rc::tape::read_all(path);
        CHECK(got.size() == 3);
        for (std::size_t i = 0; i < got.size() and i < 3; ++i) CHECK(same(got[i], fr[i]));
        const auto tape = rc::tape::read_tape(path);
        CHECK(tape.rooms.size() == 2 and tape.frame_room.size() == 3);
        if (tape.rooms.size() == 2 and tape.frame_room.size() == 3)
        {
            CHECK(tape.frame_room[0] == 0 and tape.frame_room[1] == 0 and tape.frame_room[2] == 1);
            CHECK(tape.rooms[1].room == "room_2" and tape.rooms[1].polygon == r2.polygon);
            CHECK(tape.rooms[0].apertures.size() == 1 and tape.rooms[0].apertures[0].name == "door_room_1"
                  and tape.rooms[0].apertures[0].a == r1.apertures[0].a and tape.rooms[0].apertures[0].recess == 0.1f);
        }
    }
    // Truncated last frame (a crash / SIGKILL mid-write): dropped, earlier frames kept, no throw.
    std::filesystem::resize_file(path, size_full - 3);
    {
        const auto got = rc::tape::read_all(path);
        CHECK(got.size() == 2);
        for (std::size_t i = 0; i < got.size() and i < 2; ++i) CHECK(same(got[i], fr[i]));
    }
    std::filesystem::resize_file(path, 5);   // inside the header's version field: empty, not an error
    CHECK(rc::tape::read_all(path).empty());
    CHECK(rc::tape::read_all(path + ".missing").empty());

    // Cost of one frame (~2 k points), what the compute thread pays per call.
    {
        std::filesystem::remove(path);
        rc::tape::Writer w(path);
        w.write_room(r1);
        auto f = make_frame(0, "room_1", 2000, 0.5f);
        std::vector<double> us;
        for (int i = 0; i < 2000; ++i)
        {
            f.t_ms = i * 50;
            const auto t0 = std::chrono::steady_clock::now();
            w.write(f);
            us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        std::ranges::sort(us);
        std::println("write cost over {} frames: p50 {:.1f} us, p99 {:.1f} us, max {:.1f} us", us.size(), us[us.size() / 2],
                     us[us.size() * 99 / 100], us.back());
        CHECK(us[us.size() * 99 / 100] < 200.0);
    }
    std::filesystem::remove(path);
    std::println("{}: {} checks, {} failures", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
