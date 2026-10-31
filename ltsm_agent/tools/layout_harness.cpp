// The MAP memory layout on REAL memory files, against the spanning tree it replaces. Loads each file into a private DSR
// graph (domain 9: nobody's), builds the map both ways and prints what moved and why.
//   make -C build -j8 layout_harness && bin/layout_harness <ltsm_memory.json>...
#include "topology_layout.h"

#include <QCoreApplication>

#include <clocale>
#include <cstdlib>
#include <cstdio>
#include <numbers>

namespace
{
double yaw_of(const Mat::RTMat &T) { return std::atan2(T.rotation()(1, 0), T.rotation()(0, 0)); }
double deg(double r) { return r * 180.0 / std::numbers::pi; }
}   // namespace

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    std::setlocale(LC_ALL, "");   // the agent runs under Qt's locale (es_ES here): parse under the same one
    if (argc < 2) { std::fprintf(stderr, "usage: %s <ltsm_memory.json>...\n", argv[0]); return 2; }
    int bad = 0;
    for (int f = 1; f < argc; ++f)
    {
        std::printf("\n══ %s\n", argv[f]);
        auto G = std::make_shared<DSR::DSRGraph>("layout_harness", 4000 + f, argv[f], true, 9);
        const auto tree = ltsm::topo::build(*G, "room_1");
        ltsm::topo::LayoutCache cache;
        ltsm::layout::Model model;
        model.verbose = std::getenv("LAYOUT_VERBOSE") != nullptr;
        const auto map = ltsm::topo::build(*G, "room_1", {}, &cache, model);
        const auto &c = cache;
        if (not c.converged) ++bad;
        // A metre-scale shift is NOT by itself a fault: on gold2 the tree put room_2 inside room_1 and the MAP moved it
        // 8 m to where a clean run (gold6) has it. Compare the world boxes printed below against a clean run instead.
        std::printf("  solve #%d: %d rooms, %d passages, %d it%s, %.1f ms, %d starts%s\n", c.solves, c.rooms, c.passages,
                    c.iterations, c.converged ? "" : " NOT CONVERGED", c.ms, c.starts,
                    c.restarted ? " (won by a re-start)" : "");
        std::printf("  F %.3f -> %.3f   seam %.3f->%.3f  square %.3f->%.3f  overlap %.3f->%.3f  wall %.3f->%.3f\n",
                    c.before.total(), c.after.total(), c.before.seam, c.after.seam, c.before.manhattan, c.after.manhattan,
                    c.before.overlap, c.after.overlap, c.before.wall, c.after.wall);
        std::printf("  adjacency %.3f->%.3f\n", c.before.adjacency, c.after.adjacency);
        if (c.building_axis.has_value()) std::printf("  building axis %.2f deg\n", deg(c.building_axis.value()));
        std::printf("  %-14s %28s %28s %16s %20s %8s\n", "room", "tree (x, y, yaw)", "MAP (x, y, yaw)", "shift m / deg",
                    "sigma x, y, yaw", "P(sq)");
        for (const auto &[i, r] : std::views::enumerate(map.rooms))
        {
            const auto &t = tree.rooms[i].world;
            const Eigen::Vector2d dt = r.world.translation().head<2>() - t.translation().head<2>();
            const double dy = std::remainder(yaw_of(r.world) - yaw_of(t), 2 * std::numbers::pi);
            std::printf("  %-14s (%7.3f, %7.3f, %7.2f) (%7.3f, %7.3f, %7.2f) %7.3f / %6.2f   (%.3f, %.3f, %.2f) %8.2f\n",
                        r.name.c_str(), t.translation().x(), t.translation().y(), deg(yaw_of(t)), r.world.translation().x(),
                        r.world.translation().y(), deg(yaw_of(r.world)), dt.norm(), deg(dy), std::sqrt(r.cov(0, 0)),
                        std::sqrt(r.cov(1, 1)), deg(std::sqrt(r.cov(2, 2))), r.p_manhattan);
            const auto bbox = [](const std::vector<Eigen::Vector2d> &o)
            {
                if (o.empty()) return std::string("-");
                Eigen::Vector2d lo = o[0], hi = o[0];
                for (const auto &p : o) { lo = lo.cwiseMin(p); hi = hi.cwiseMax(p); }
                char b[96];
                std::snprintf(b, sizeof b, "x[%.2f, %.2f] y[%.2f, %.2f]", lo.x(), hi.x(), lo.y(), hi.y());
                return std::string(b);
            };
            std::printf("  %-14s   outline tree %s   MAP %s\n", "", bbox(r.outline_raw).c_str(), bbox(r.outline).c_str());
        }
        for (const auto &[k, p] : std::views::enumerate(map.passages))
            std::printf("  %-12s seam_ok %d  loop %d  faces with cov %d  seam d2 %8.2f  P(right) %.4f  closure tree %.3f m -> MAP %.3f m\n",
                        p.name.c_str(), p.seam_ok, p.loop, p.stated_faces, p.seam_d2, p.p_inlier, tree.passages[k].closure_m,
                        p.closure_m);
        // Cached: the same structure must NOT re-solve.
        cache.fresh = false;
        const int solves = cache.solves;
        (void)ltsm::topo::build(*G, "room_1", {}, &cache);
        std::printf("  rebuild with an unchanged structure re-solved: %s\n", cache.solves != solves ? "YES (BUG)" : "no");
        if (cache.solves != solves) ++bad;
    }
    std::printf("\n%s\n", bad ? "SOMETHING TO CHASE (a solve did not converge, or an unchanged structure re-solved)" : "ok");
    std::fflush(stdout);
    std::_Exit(bad ? 1 : 0);   // skip static destruction (libIce/DDS finalizer order; see README)
}
