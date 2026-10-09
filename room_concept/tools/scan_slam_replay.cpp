// Offline check of src/scan_slam.h on a WIN1 recording (room_concept RecordWallInput): replays odometry delta + the
// wall-band returns and writes ts,x,y,theta per frame to stdout-named CSV, for grading against Webots ground truth.
//   g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/scan_slam_replay.cpp -o /tmp/scan_slam_replay
//   scan_slam_replay tmp/wall_input_<ts>.bin out.csv
#include "scan_slam.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <locale>
#include <vector>
int main(int argc, char** argv)
{
    if (argc < 3) { std::fprintf(stderr, "usage: %s wall_input.bin out.csv\n", argv[0]); return 2; }
    std::ifstream in(argv[1], std::ios::binary);
    char magic[4]; std::uint32_t ver = 0;
    in.read(magic, 4); in.read(reinterpret_cast<char*>(&ver), 4);
    if (std::memcmp(magic, "WIN1", 4) != 0) { std::fprintf(stderr, "not WIN1\n"); return 2; }
    std::ofstream out(argv[2]); out.imbue(std::locale::classic());
    out << "ts_ms,x,y,theta,sxx,syy,stt\n";
    rc::slam::ScanSlam slam;
    long frames = 0; double ms = 0.0;
    while (true)
    {
        std::int64_t ts; std::uint8_t st; float od[3], po[3], cv[9]; std::uint32_t n;
        if (not in.read(reinterpret_cast<char*>(&ts), 8)) break;
        in.read(reinterpret_cast<char*>(&st), 1); in.read(reinterpret_cast<char*>(od), 12);
        in.read(reinterpret_cast<char*>(po), 12); in.read(reinterpret_cast<char*>(cv), 36);
        if (not in.read(reinterpret_cast<char*>(&n), 4)) break;
        std::vector<float> buf(static_cast<size_t>(n) * 4);
        if (not in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size() * 4))) break;
        std::vector<Eigen::Vector2f> pts; pts.reserve(n);
        for (std::uint32_t i = 0; i < n; ++i) pts.emplace_back(buf[4 * i], buf[4 * i + 1]);
        const auto t0 = std::chrono::steady_clock::now();
        const Eigen::Vector3f p = slam.step(Eigen::Vector3f(od[0], od[1], od[2]), pts);
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const auto& C = slam.covariance();
        out << ts << ',' << p.x() << ',' << p.y() << ',' << p.z() << ',' << C(0, 0) << ',' << C(1, 1) << ',' << C(2, 2) << '\n';
        ++frames;
    }
    std::fprintf(stderr, "%ld frames, %.2f ms/frame\n", frames, frames ? ms / frames : 0.0);
    return 0;
}
