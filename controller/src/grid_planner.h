/*
 * grid_planner.h — A* over an occupancy grid with EXACT robot-footprint collision.
 *
 * Replaces the visibility-graph planner (room_path_planner). The visibility graph is the right tool for a
 * world that is a handful of polygons, which is what a fully-modelled world looks like — but the residual is a
 * FIELD, and every attempt to squeeze it through a polygon interface has been lossy in a different way:
 *   - traced outlines dropped interior holes → free space enclosed by residual was published SOLID, which put
 *     obstacle polygons into rooms the robot had never observed
 *   - a diagonal pinch fell back to the component's CONVEX HULL → half a room went solid (15× its true area)
 *   - an exact rectangular cover fixed both and produced 154–461 polygons, and the visibility graph is
 *     O(V²·E) in obstacle vertices: ~1.2e8 segment tests at 154, ~3.1e9 at 461. It stopped returning, so the
 *     robot reported itself stuck standing in open floor.
 * The grid does not have a lossy conversion because there is no conversion.
 *
 * COLLISION IS THE ROBOT'S ACTUAL SHAPE, NOT AN INFLATED OBSTACLE.
 * C-space inflation buys a cheap point test by pretending the robot is a disc, and the price is that every
 * stage has to guess a radius. Six such guesses had accumulated across three agents (see robot_footprint.h),
 * summing to ~0.95 m of demanded gap for a robot that physically passes 0.461 m — and cancelling exactly often
 * enough to report the robot as being inside its own exclusion disc. Here there is ONE predicate,
 * `footprint(pose) ∩ occupied == ∅`, evaluated against the authored polygon. It cannot stack, because there is
 * only one of it, and any standoff beyond it is a single explicit safety_margin_m rather than six hidden ones.
 *
 * HEADING IS PART OF THE STATE. A rigid footprint is not rotation-invariant — a rectangle fits through a gap
 * sideways that it cannot enter head-on — and collapsing that to a disc is precisely what was denying the robot
 * passages it fits. So A* searches (x, y, heading) over 8 headings, with the heading of a move set by its
 * direction of travel, which is also how the controller actually drives: turn toward the next waypoint, then
 * translate. Footprint cell offsets are precomputed once per heading, so a collision test is a few dozen array
 * lookups.
 *
 * DSR-free, pure Eigen/STL → unit-testable in isolation (self_test()).
 */

#pragma once

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>
#include <Eigen/Dense>

#include "../../common/robot_footprint/robot_footprint.h"

namespace rc
{

class GridPlanner
{
public:
    struct Params
    {
        // Planning resolution. Independent of residual's 0.05 m evidence grid: the planner does not need
        // centimetre fidelity, and cell count drives both memory and search time quadratically.
        float cell_size_m = 0.10f;
        // The ONE standoff. Everything the old pipeline expressed as six separate inflations is this number,
        // and it is a stated preference rather than a geometric fact — so it can be tuned for comfort without
        // ever being able to make a reachable goal unreachable, which is what the old stack could do.
        float safety_margin_m = 0.05f;
        // A start already in collision must not brick the planner. Perception transients happen (an obstacle
        // grows over the robot, localisation jumps), and refusing to plan leaves the robot permanently stuck
        // with no way out. Collision is ignored for the start cell ONLY, so the search can walk out of it.
        bool allow_start_in_collision = true;
        int  max_expansions = 400000;   // runaway guard; the apartment needs a small fraction of this
    };

    Params params;
    RobotFootprint footprint = RobotFootprint::shadow();

    // Rasterise the world the controller already holds. `room_polygon` bounds free space — anything OUTSIDE it
    // is occupied, which is how walls enter without needing a separate representation. `obstacles` are filled
    // polygons in room coordinates (residual hulls, concept-object boxes: whatever the controller is avoiding).
    // Obstacles are rasterised at their TRUE extent — no inflation. The robot's shape is applied at query time.
    void set_world(const std::vector<Eigen::Vector2f>& room_polygon,
                   const std::vector<std::vector<Eigen::Vector2f>>& obstacles);

    // Shortest footprint-feasible path from start to goal, in room coordinates. Returns nullopt with
    // last_failure() set. The returned path is simplified to its turning points (a dense cell chain is useless
    // to a trajectory controller and makes the MPPI's carrot jitter).
    std::optional<std::vector<Eigen::Vector2f>> plan(const Eigen::Vector2f& start_room,
                                                     const Eigen::Vector2f& goal_room);

    // Why the last plan() failed. Distinguishing the cases matters: a goal in collision, a start in collision
    // and a genuinely disconnected free space are three different faults with three different fixes, and
    // reporting them all as "no path" is what sent one debugging round chasing a performance red herring.
    const std::string& last_failure() const { return last_failure_; }

    // Is this pose footprint-feasible? Exposed so the caller can test a target BEFORE committing to it —
    // one predicate shared by planning and target repair, so the two can no longer disagree.
    bool pose_free(const Eigen::Vector2f& pos_room, float theta) const;

    // Nearest footprint-feasible pose to `pos_room`, searched outward. Replaces repair_target's ring search,
    // and because it uses the SAME predicate the planner does, a repaired target is feasible by construction.
    std::optional<Eigen::Vector2f> nearest_free(const Eigen::Vector2f& pos_room, float theta,
                                                float max_radius_m = 3.0f) const;

    // ── DISTANCE FIELD ────────────────────────────────────────────────────────────────────────────
    // Metres from `p` to the nearest occupied cell (outside-the-room counts as occupied, so walls are
    // included). Zero inside an obstacle. Returns a large positive value if there is no world yet, so a
    // caller that forgot to set_world gets "wide open" rather than a spurious obstacle.
    //
    // EXACT, not chamfer. The planner's own collision test needs no distance at all, so this exists for
    // callers that OPTIMISE against clearance — and an optimiser follows the gradient of whatever field
    // it is handed, so a chamfer's ~8% direction-dependent error would be baked into the shape of the
    // result, not merely mis-score it. Felzenszwalb's two-pass squared-distance transform is O(cells)
    // and exact, so there is no reason to approximate.
    //
    // Built LAZILY and cached: set_world runs every control cycle, while the field is wanted only when a
    // route is built or repaired. Nothing pays for it until something asks.
    float distance_at(const Eigen::Vector2f& pos_room) const;
    // ∇distance (metres per metre), by central differences over `fd_cells` cells. The field is C0 across
    // cell boundaries, so a step of about one cell is deliberate: it is what keeps a gradient-based
    // caller from chattering on the facets of the interpolant.
    Eigen::Vector2f distance_gradient_at(const Eigen::Vector2f& pos_room, float fd_cells = 1.0f) const;
    bool  has_distance_field() const { return w_ > 0 and h_ > 0; }

    int   width()  const { return w_; }
    int   height() const { return h_; }
    long  occupied_cells() const;

    // ── SNAPSHOT ──────────────────────────────────────────────────────────────────────────────────
    // Serialise/restore the rasterised world, so a route can be rebuilt OFFLINE against exactly the
    // world the robot had. The RASTER is written rather than the source polygons on purpose: the
    // polygons would have to be re-rasterised offline, and any difference between the two rasters —
    // a padding rule, a cell-centre test, the room-mask fallback — would show up as a difference in
    // the ROUTE and be read as an effect of whatever was being studied. There is nothing to re-derive
    // here, so there is nothing to disagree about.
    void write_grid(std::ostream& os) const;
    bool read_grid(std::istream& is);   // false on a malformed stream; the planner is left empty

    static bool self_test();

private:
    static constexpr int kHeadings = 8;

    bool  in_bounds(int ix, int iy) const { return ix >= 0 and ix < w_ and iy >= 0 and iy < h_; }
    int   idx(int ix, int iy) const { return iy * w_ + ix; }
    bool  world_to_cell(const Eigen::Vector2f& p, int& ix, int& iy) const;
    Eigen::Vector2f cell_to_world(int ix, int iy) const;
    // Footprint at heading bucket `h` centred on cell (ix,iy) overlaps no occupied cell and stays in bounds.
    bool  cell_free(int ix, int iy, int h) const;
    void  rebuild_offsets();
    void  build_distance_field() const;   // lazy; fills dist_ with metres, sets dist_valid_

    std::vector<std::uint8_t> occ_;
    // Cached distance field, in metres, one entry per cell. Mutable because it is a memoised view of occ_:
    // asking for a distance does not change the planner's world, it only realises part of it.
    mutable std::vector<float> dist_;
    mutable bool dist_valid_ = false;
    float xmin_ = 0, ymin_ = 0, cell_ = 0.1f;
    int   w_ = 0, h_ = 0;
    // Precomputed footprint coverage per heading bucket — the reason exact collision is affordable here.
    std::vector<std::vector<Eigen::Vector2i>> offsets_;
    float offsets_cell_ = 0.f, offsets_margin_ = -1.f;
    bool room_mask_usable_ = true;  // false ⇒ the room polygon was discarded (see set_world)
    int last_obstacle_count_ = 0;   // polygons in the last set_world — for the failure report
    std::string last_failure_;
};

}  // namespace rc
