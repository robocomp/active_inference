// cost_to_go.h — geodesic distance-to-goal over free space, with trap memory.
#pragma once
#include <Eigen/Dense>
#include <functional>
#include <span>
#include <vector>

namespace rc::policy
{
// Progress measure for the local policy. Cell (i,j) has its CENTRE at origin + (i,j)*cell. 8-connected Dijkstra:
// a step of length L (cell or cell*sqrt2) costs L * (1 + density), density = mean of extra_cost_m and trap density
// at the two cell centres.
// ★ UNITS: extra_cost_m returns a DENSITY, metres-equivalent per metre travelled. add_trap's cost_m is the TOTAL cost
//   of crossing the trap's diameter, i.e. density = cost_m / (2*radius). Charging per metre (not per entered cell)
//   keeps diagonals and axis steps consistent and makes a trap independent of grid resolution.
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
    // A pose whose 4 surrounding cells are all non-free (normal in a doorway, closer to the wall than the inscribed
    // radius) reads the nearest finite cell's value PLUS the Euclidean distance to its centre: continuous, and progress
    // stays meaningful. Poses in the last row/column are clamped into the grid. +inf only if no finite cell exists.
    float at(const Eigen::Vector2f& p) const;
    int seeded_goal_cells() const { return seeded_; }
    // ★ RESOLUTION of a field DIFFERENCE f(a) - f(b) between two points `separation_m` apart (round-5 ruling: a decrease
    //   below it is not a measurement of progress). Derivation: the 8-connected Dijkstra measures the octile metric,
    //   whose length gain over the Euclidean one varies with direction between 1 and sqrt(4 - 2 sqrt2) = 1.0824
    //   (worst at 22.5 deg); so the error of a node value differs between two nodes by at most 0.0824 x their
    //   separation. at() is a convex combination of the 4 nodes around a point (or the nearest node), whose separation
    //   from the point is at most one cell diagonal sqrt2 * cell. Hence |error of f(a) - f(b)| <= 0.0824 x (separation +
    //   2 sqrt2 cell): 1.2 cm at zero separation for 0.05 m cells (one cell x 0.233), +0.82 cm per 0.1 m apart.
    float resolution_m(float separation_m) const { return kOctileGainExcess * (separation_m + 2.f * 1.41421356f * g_.cell); }
    float cell() const { return g_.cell; }
    static constexpr float kOctileGainExcess = 0.0823922f;   // sqrt(4 - 2 sqrt2) - 1   // goal cells seeded by the last build (0 = goal fully blocked)
    bool reachable(const Eigen::Vector2f& p) const;
    void add_trap(const Eigen::Vector2f& p, float radius_m, float cost_m);   // applied at the next build()
    void decay_traps(float factor);                                           // multiplies costs; drops < 1e-3 m

private:
    struct Trap { Eigen::Vector2f p; float radius, cost; };
    Grid g_{{0.f, 0.f}, 1.f, 0, 0};
    std::vector<float> field_;
    std::vector<Trap> traps_;
    int seeded_ = 0;
    std::vector<char> free_;
};
}  // namespace rc::policy
