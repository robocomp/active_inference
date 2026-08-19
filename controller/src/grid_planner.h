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
#include <functional>
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

        // ── CLEARANCE PREFERENCE IN THE SEARCH ITSELF ────────────────────────────────────────────
        // The step cost used to be distance + a small turn penalty, i.e. pure shortest path, with
        // `cell_free` (footprint + safety_margin_m) as the ONLY thing keeping it off the furniture. So
        // A* returned the shortest admissible path, which hugs every corner and wall it legally can,
        // and the route optimiser downstream then spent its clearance term trying to undo that — while
        // RouteSpline's feasibility pass pulled any sample it could not fix back TOWARD the same
        // hugging polyline. Reported live as "it moves too close to furniture and thin walls" with the
        // safety slider already at maximum: the slider reweights the optimiser, which is the wrong end.
        //
        // A step through a tight cell now costs more:  step * (1 + w * max(0, d_pref - d)/d_pref)
        // ★A PREFERENCE, NOT A CONSTRAINT — and deliberately so. It never makes a passable gap
        // unplannable (that is `safety_margin_m`'s job, and inflating it is how a stack ends up
        // demanding 0.95 m of gap for a robot that fits through 0.461 m); it only makes the roomy route
        // cheaper when one exists. A corridor with no alternative still gets planned, at a higher cost
        // that nothing competes with.
        // ★The field is the planner's own exact EDT, already built for `distance_at`, so this costs one
        // array lookup per expansion and no new state.
        // 0 restores pure shortest path.
        float clearance_weight = 1.5f;
        // Distance (m, from the robot's CENTRE) beyond which a cell is "roomy" and costs nothing extra.
        // Below it the penalty ramps linearly to `clearance_weight` at d = 0.
        float clearance_pref_m = 0.9f;
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

    // ── WHAT `theta` MEANS EVERYWHERE IN THIS CLASS ───────────────────────────────────────────────
    // A room YAW: the direction the robot FACES, forward = +x at 0, counter-clockwise. That is what
    // every caller already passes (an affordance's target.yaw_rad, a robot pose's theta + pi/2) and what
    // plan()'s move table means. It is NOT RobotFootprint's own theta — that class's forward axis is +y,
    // so the two differ by a quarter turn, and rebuild_offsets() is the single place the conversion is
    // applied. Getting it wrong is silent (the hull is nearly symmetric), so it is pinned by self_test.
    // Is this pose footprint-feasible? Exposed so the caller can test a target BEFORE committing to it —
    // one predicate shared by planning and target repair, so the two can no longer disagree.
    bool pose_free(const Eigen::Vector2f& pos_room, float theta) const;

    // Is there a rasterised world at all yet? ★pose_free() answers FALSE for every pose before
    // set_world() has run — world_to_cell fails on a zero-sized grid — so a caller that turns "not
    // feasible" into a report to another agent would report an infeasible standpoint for the first
    // cycles of every run, when the only true statement is "I do not know yet". Ask this first.
    bool has_world() const { return w_ > 0 and h_ > 0; }

    // Identity of the rasterised world (see set_world). Two calls returning the same value describe
    // the same occupancy, so an answer derived from the grid can be cached against it instead of
    // against a clock — a timer would either re-ask a question the world has not changed the answer
    // to, or trust an answer after the world moved. This can do neither.
    std::size_t world_hash() const { return world_hash_; }

    // ── MIGRATION MONITOR (temporary; delete once the yaw correction is trusted) ───────────────────
    // The same question answered with the body oriented as it WAS before the correction — turned 90 deg
    // from its direction of travel. A change of this kind is easy to argue about and hard to see, so it
    // is instrumented instead: these two say what it actually did to the world the robot is in.
    bool pose_free_legacy(const Eigen::Vector2f& pos_room, float theta) const;
    // Free (cell, heading) states under each rasterisation.
    // ★THE TOTALS ARE EQUAL BY CONSTRUCTION, AND THAT IS THE POINT. With 8 headings a quarter turn is
    // exactly two buckets, so the corrected offsets at heading h ARE the legacy offsets at heading h-2:
    // summed over all headings the two rasterisations cover the identical set of body placements. The
    // correction therefore cannot change HOW MUCH space is free — only WHICH HEADING is free WHERE. A
    // gap running east-west stops admitting east-west travel and starts admitting north-south.
    // So read `lost` (== `gained`), not the totals: it is how much of the free C-space changed hands,
    // and it is the only number here that can be zero if the correction failed to land.
    struct OrientationCensus { long states = 0, free_now = 0, free_legacy = 0, lost = 0, gained = 0; };
    OrientationCensus orientation_census() const;

    // Nearest footprint-feasible pose to `pos_room`, searched outward. Replaces repair_target's ring search,
    // and because it uses the SAME predicate the planner does, a repaired target is feasible by construction.
    std::optional<Eigen::Vector2f> nearest_free(const Eigen::Vector2f& pos_room, float theta,
                                                float max_radius_m = 3.0f) const;

    // Same expanding-ring search, ANDed with an extra predicate the caller supplies. It exists because the
    // grid is not the only evidence there is: the occupancy it rasterises comes from beliefs that DECAY, so a
    // cell an object still occupies reads free once the belief that put it there has faded — and re-asking the
    // grid, however often, keeps returning the same wrong answer. The caller passes the evidence the grid does
    // not carry (live LiDAR at the final approach), and the search remains ONE search, so a pose it returns is
    // admissible under both questions by construction rather than by two stages agreeing.
    // `admissible` is called on candidate CENTRES only; it is never asked about the footprint's cells, so it
    // must itself account for the body's extent.
    std::optional<Eigen::Vector2f> nearest_free_where(const Eigen::Vector2f& pos_room, float theta,
                                                      const std::function<bool(const Eigen::Vector2f&)>& admissible,
                                                      float max_radius_m = 3.0f) const;

    // Nearest pose to `goal_room` that is ACTUALLY REACHABLE from `start_room`, under exactly the move
    // model plan() uses. This is the missing half of nearest_free: that one tests whether the footprint
    // FITS, which is a purely LOCAL question, so it happily returns a pose sealed inside a pocket the
    // robot can never enter. Repair then produced the same infeasible-to-route target every cycle, the
    // planner correctly reported "no route", and the robot held forever — with the repair line and the
    // hold line both repeating, which is exactly what a disconnected free space looks like from outside.
    // One flood fill over the (cell, heading) graph; every state it reaches is routable by construction.
    // Returns nullopt only if the robot cannot leave its own cell.
    std::optional<Eigen::Vector2f> nearest_reachable(const Eigen::Vector2f& start_room,
                                                     const Eigen::Vector2f& goal_room);

    // ── FINAL-APPROACH DIAGNOSTICS ────────────────────────────────────────────────────────────────
    // How much slack the FOOTPRINT has at this pose: the smallest distance-to-obstacle over the cells
    // the body actually covers, in metres. 0 means a body cell is on an obstacle. This is the continuous
    // form of pose_free, which only ever answers yes/no and so cannot say "feasible, but barely".
    float pose_clearance(const Eigen::Vector2f& pos_room, float theta) const;

    // Can the robot TURN IN PLACE here, from `theta_from` to `theta_to` the short way round? Returns the
    // tightest clearance over the swept headings, and whether every one of them is footprint-feasible.
    // ★This is the question nothing in the arrival path has ever asked. A standpoint is verified feasible
    // at ONE heading — the facing yaw for the repair, the travel heading for the plan — but the terminal
    // rotation sweeps everything in between, and the body is 0.65 m wide against a 0.46 m inscribed
    // width. Feasible at both ends does not imply feasible in the middle.
    struct RotationSweep { float min_clearance_m = 0.f; bool feasible = false; float worst_heading_rad = 0.f; };
    RotationSweep rotation_sweep(const Eigen::Vector2f& pos_room, float theta_from, float theta_to) const;

    // Nearest pose to `pos_room` the robot can TURN AROUND IN — footprint-feasible at EVERY heading —
    // and, among equally distant candidates, the one with the most room to spare.
    // WHY THIS AND NOT nearest_free: a standpoint is only ever verified at ONE heading, but with
    // GoalFacingYawEnabled the robot performs a terminal rotation in place there, sweeping every
    // heading between its arrival and the facing yaw — with NO obstacle check while it does. The
    // arrival heading is not known when the target is repaired, so the guarantee has to be
    // heading-independent: "you can turn to anything from here".
    // Clearance is a PREFERENCE, not a cutoff — the ring search already bounds the distance, so among
    // the candidates it admits we take the roomiest rather than the first. No threshold to pick.
    std::optional<Eigen::Vector2f> nearest_rotatable(const Eigen::Vector2f& pos_room,
                                                     float max_radius_m = 3.0f) const;

    // Can the robot turn all the way round HERE? The predicate nearest_rotatable searches on, exposed
    // so a caller re-checking a standpoint later asks the identical question — the heading count is not
    // something two files should each know.
    bool can_turn_here(const Eigen::Vector2f& pos_room) const;

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
    bool  cell_free_legacy(int ix, int iy, int h) const;   // migration monitor — see rebuild_offsets()
    // The footprint at `offsets[h]` centred on (ix,iy) overlaps nothing occupied and stays in bounds.
    // Shared by cell_free and its legacy twin so the body exists once — the twin is scheduled for
    // deletion, and one substituted array is not a reason to copy a loop.
    bool  cell_free_in(const std::vector<std::vector<Eigen::Vector2i>>& offsets, int ix, int iy, int h) const;
    // Room YAW -> heading bucket. ONE place: the fmod/lround quantisation was written out three times,
    // and the whole class exists on the premise that a convention lives in a single spot.
    int   heading_bucket(float yaw) const;
    bool  cell_free_at(const Eigen::Vector2f& pos_room, int heading_index) const;
    void  rebuild_offsets();
    void  build_distance_field() const;   // lazy; fills dist_ with metres, sets dist_valid_

    std::vector<std::uint8_t> occ_;
    // Cached distance field, in metres, one entry per cell. Mutable because it is a memoised view of occ_:
    // asking for a distance does not change the planner's world, it only realises part of it.
    mutable std::vector<float> dist_;
    mutable bool dist_valid_ = false;
    float xmin_ = 0, ymin_ = 0, cell_ = 0.1f;
    int   w_ = 0, h_ = 0;
    std::size_t world_hash_ = 0;
    // Precomputed footprint coverage per heading bucket — the reason exact collision is affordable here.
    std::vector<std::vector<Eigen::Vector2i>> offsets_;
    // The SAME coverage as it was rasterised before the yaw correction, i.e. with the body turned 90 deg
    // from its direction of travel. Exists only so the correction can be MEASURED against the live world
    // (orientation_census / pose_free_legacy) instead of asserted. Delete with them.
    std::vector<std::vector<Eigen::Vector2i>> offsets_legacy_;
    float offsets_cell_ = 0.f, offsets_margin_ = -1.f;
    bool room_mask_usable_ = true;  // false ⇒ the room polygon was discarded (see set_world)
    int last_obstacle_count_ = 0;   // polygons in the last set_world — for the failure report
    std::string last_failure_;
};

}  // namespace rc
