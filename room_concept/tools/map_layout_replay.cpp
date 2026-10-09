// Offline check of FreeSpacePolygon::manhattan_layout on a WIN1 recording: scans inserted at the recorded pose (the
// agent's own pose — plain SLAM when PlainSlam was on) or at an external pose CSV (ts,x,y,theta), then the layout.
//   g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/map_layout_replay.cpp -o /tmp/map_layout_replay
//   map_layout_replay wall_input.bin out_poly.csv [poses.csv]
#include "free_space_polygon.h"
#include <charconv>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <locale>
#include <map>
#include <string>
#include <vector>
int main(int argc, char** argv)
{
    if (argc < 3) { std::fprintf(stderr, "usage: %s wall_input.bin out_poly.csv [poses.csv]\n", argv[0]); return 2; }
    std::map<std::int64_t, Eigen::Vector3f> ext;
    if (argc > 3)
    {
        std::ifstream pf(argv[3]); std::string line; std::getline(pf, line);
        while (std::getline(pf, line))
        {
            std::int64_t ts; float v[3]; const char* p = line.data(); const char* e = p + line.size();
            auto r = std::from_chars(p, e, ts); p = r.ptr + 1;
            for (float& x : v) { auto q = std::from_chars(p, e, x); p = q.ptr + 1; }
            ext[ts] = Eigen::Vector3f(v[0], v[1], v[2]);
        }
    }
    std::ifstream in(argv[1], std::ios::binary);
    char magic[4]; std::uint32_t ver = 0; in.read(magic, 4); in.read(reinterpret_cast<char*>(&ver), 4);
    if (std::memcmp(magic, "WIN1", 4) != 0) return 2;
    rc::freespace::Params fp; fp.erode_m = 0.f;
    rc::freespace::FreeSpacePolygon fs(fp);
    const Eigen::Vector2f lidar_off(0.f, -0.155f);
    Eigen::Vector2f last_pos(0, 0); long frames = 0;
    while (true)
    {
        std::int64_t ts; std::uint8_t st; float od[3], po[3], cv[9]; std::uint32_t n;
        if (not in.read(reinterpret_cast<char*>(&ts), 8)) break;
        in.read(reinterpret_cast<char*>(&st), 1); in.read(reinterpret_cast<char*>(od), 12);
        in.read(reinterpret_cast<char*>(po), 12); in.read(reinterpret_cast<char*>(cv), 36);
        if (not in.read(reinterpret_cast<char*>(&n), 4)) break;
        std::vector<float> buf(static_cast<size_t>(n) * 4);
        if (not in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size() * 4))) break;
        Eigen::Vector3f pose(po[0], po[1], po[2]);
        if (not ext.empty()) { auto it = ext.find(ts); if (it == ext.end()) continue; pose = it->second; }
        const float c = std::cos(pose.z()), s = std::sin(pose.z());
        std::vector<Eigen::Vector2f> ends; ends.reserve(n);
        for (std::uint32_t i = 0; i < n; ++i)
        { const float x = buf[4 * i], y = buf[4 * i + 1]; ends.emplace_back(c * x - s * y + pose.x(), s * x + c * y + pose.y()); }
        const Eigen::Vector2f org(c * lidar_off.x() - s * lidar_off.y() + pose.x(), s * lidar_off.x() + c * lidar_off.y() + pose.y());
        fs.add_scan(org, ends);
        last_pos = pose.head<2>(); ++frames;
    }
    float th = 0.f;
    const auto poly = fs.manhattan_layout(Eigen::Vector2f(0, 0), 0.12f, 0.40f, &th);
    std::ofstream out(argv[2]); out.imbue(std::locale::classic());
    out << "x,y\n"; for (const auto& v : poly) out << v.x() << ',' << v.y() << '\n';
    std::fprintf(stderr, "%ld frames, layout %zu verts, theta %.2f deg, area %.2f m2\n", frames, poly.size(),
                 th * 57.2958f, poly.size() >= 3 ? rc::freespace::signed_area(poly) : 0.f);
    return 0;
}
