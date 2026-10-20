// cost_to_go.h — geodesic distance-to-goal over free space, with trap memory.
#pragma once
#include <Eigen/Dense>
#include <functional>
#include <span>
#include <vector>

namespace rc::policy
{
// Progress measure for the local policy. Cell (i,j) has its CENTRE at origin + (i,j)*cell. 8-connected Dijkstra:
// entering a cell costs step length (cell or cell*sqrt2) + extra_cost_m(cell centre) + trap cost of that cell.
// ★ extra/trap costs are charged per ENTERED cell (metres-equivalent), so a trap of 2 m spanning a few cells makes the
//   region much dearer than a detour of a metre or two; that is what lets a rebuilt field route elsewhere.
class CostToGo
{
public:
    struct Grid { Eigen::Vector2f origin; float cell; int w, h; };

    // free(p): body-agnostic free test. Goal points off-grid / in non-free cells snap to the nearest free cell within
    // one cell; if none exists the field is all +inf (never throws).
    void build(const Grid& g, const std::function<bool(const Eigen::Vector2f&)>& free,
               const std::function<float(const Eigen::Vector2f&)>& extra_cost_m,
               std::span<const Eigen::Vector2f> goal_points);
    // ★ Bilinear over the 4 surrounding cell values; if ANY of the four is +inf the blend would be inf/NaN, so fall
    //   back to the nearest finite neighbour among them (+inf if none finite). Off-grid -> +inf.
    float at(const Eigen::Vector2f& p) const;
    bool reachable(const Eigen::Vector2f& p) const;
    void add_trap(const Eigen::Vector2f& p, float radius_m, float cost_m);   // applied at the next build()
    void decay_traps(float factor);                                           // multiplies costs; drops < 1e-3 m

private:
    struct Trap { Eigen::Vector2f p; float radius, cost; };
    Grid g_{{0.f, 0.f}, 1.f, 0, 0};
    std::vector<float> field_;
    std::vector<Trap> traps_;
};
}  // namespace rc::policy
