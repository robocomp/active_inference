#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
// LEARNT LAYOUTS (2026-10-10) — the room an Estimate run promoted (and refined) is kept as a dated SVG, so a later
// run can start from it: RoomShape.MapMode = "last" loads the newest one as the GIVEN room.
//
//   <LayoutDir>/learnt/<scenario stem>_<YYYY-MM-DD_HH-MM-SS>.svg      (one file per run; refinements overwrite it)
//
// The file is in the format SvgRoomLoader reads for a surveyed plan: a <polygon id="room_contour"> loaded with
// mirror_x = true, which negates x and reverses the order — so it is WRITTEN with x negated and the order reversed,
// and loads back as exactly the vertices that were published. Coordinates are the learnt room's own frame (the robot's
// start pose of the run that learnt it); a run that loads it localises in it like in any given layout.
// Written to a temporary file and renamed, so a reader never sees half a file. Numbers through the classic locale.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#include <Eigen/Dense>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <locale>
#include <optional>
#include <string>
#include <vector>

namespace rc::learnt
{
/// "apartamento_layout.svg" or "some/dir/apartamento_layout.svg" -> "apartamento_layout"
inline std::string scenario_stem(const std::string& svg_name)
{
    return std::filesystem::path(svg_name).stem().string();
}

/// A new file name for this run: <dir>/learnt/<stem>_<local date-time>.svg
inline std::string new_path(const std::string& layout_dir, const std::string& stem)
{
    const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", &tm);
    return (std::filesystem::path(layout_dir) / "learnt" / (stem + "_" + buf + ".svg")).string();
}

/// Write `poly` (room frame, m) as a loadable layout. Returns false (and leaves any previous file) on failure.
inline bool save(const std::string& path, const std::vector<Eigen::Vector2f>& poly, const std::string& note)
{
    if (poly.size() < 3) return false;
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    Eigen::Vector2f lo = poly.front(), hi = lo;
    for (const auto& v : poly) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::out | std::ios::trunc);
        if (not f.is_open()) return false;
        f.imbue(std::locale::classic());
        f.setf(std::ios::fixed); f.precision(4);
        const float m = 0.5f;   // margin of the view box, m (display only)
        f << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
          << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"" << (-hi.x() - m) << ' ' << (lo.y() - m) << ' '
          << (hi.x() - lo.x() + 2 * m) << ' ' << (hi.y() - lo.y() + 2 * m) << "\">\n"
          << "  <!-- LEARNT layout (room_concept, Estimate mode). " << note << " -->\n"
          << "  <!-- room_concept loads polygon id=room_contour with mirror_x=true: x is stored negated, order reversed. -->\n"
          << "  <polygon id=\"room_contour\" fill=\"none\" stroke=\"black\" stroke-width=\"0.02\" points=\"";
        for (auto it = poly.rbegin(); it != poly.rend(); ++it)
            f << (it == poly.rbegin() ? "" : " ") << -it->x() << ',' << it->y();
        f << "\"/>\n</svg>\n";
        if (not f.good()) return false;
    }
    std::filesystem::rename(tmp, path, ec);
    return not ec;
}

/// The newest learnt layout for this scenario, or nullopt if there is none.
inline std::optional<std::string> newest(const std::string& layout_dir, const std::string& stem)
{
    const auto dir = std::filesystem::path(layout_dir) / "learnt";
    std::error_code ec;
    if (not std::filesystem::is_directory(dir, ec)) return std::nullopt;
    std::optional<std::string> best;   // names carry a sortable date-time, so the largest name is the newest
    for (const auto& e : std::filesystem::directory_iterator(dir, ec))
    {
        const auto name = e.path().filename().string();
        if (e.path().extension() != ".svg" or not name.starts_with(stem + "_")) continue;
        if (not best or name > std::filesystem::path(*best).filename().string()) best = e.path().string();
    }
    return best;
}
}   // namespace rc::learnt
