/*
 * residual_occupancy_grid.h  —  the residual_concept SAFETY layer, rebuilt (Phase 0).
 *
 * A fixed room-frame 2-D log-odds OCCUPANCY GRID. It replaces the cluster→track→box-belief core, which was a
 * MISSPECIFIED model: it fit persistent parametric objects to identity-less residual points, so its posterior
 * was overconfident (σ≈0.03 m while boxes wandered 1.7 m) and instances teleported (association failure). The
 * residual is the COMPLEMENT of all modelled objects — a FIELD, not an object list — so a grid is the right
 * primitive: cells are fixed (no jitter, no association), evidence accumulates (log-odds), and it is provably
 * COMPLETE (every in-band return marks a cell → nothing a clustering gate could hide).
 *
 * Sensor model (inverse-sensor-model, per beam origin→return, room frame):
 *   - HIT: a return whose z is in the NAV BAND (floor_band(range) < z < ceil_z) adds l_hit to its cell and
 *     records the hit's z-band. Floor/ceiling returns are NOT hits (that is the floor/ceiling "explainer",
 *     applied at the sensor model). One hit crosses the occupied threshold in a single frame (completeness).
 *   - MISS (free): cells the beam TRAVERSES get l_miss — but z-AWARE: a cell that already holds a hit is only
 *     cleared by a beam passing through its hit z-band (± margin); a beam going OVER a low obstacle does NOT
 *     erase it. This is the ray-carve principle, per cell — so a high beam can't miss a low obstacle.
 *   - Cells never traversed (behind an occluder / out of range) stay UNKNOWN. Occluded ≠ free.
 * Log-odds are clamped (bounded memory → dynamic obstacles clear in seconds). The probabilistic content is
 * the same DetectionProb/ClutterProb sensor model proven in the old carve; it is now aimed at cells, not boxes.
 *
 * Pure Eigen/STL, DSR-free → unit-testable in isolation (self_test): completeness, carve-clears, occluded-unknown.
 *
 * BELIEF FIELD (for planning over belief, not geometry): alongside the log-odds latch, each cell carries a
 * Beta(α,β) posterior over its occupancy probability, driven by the SAME per-cycle hit/miss flags. It exposes
 * the two terms a belief-space planner needs: mean P=α/(α+β) (collision RISK) and Var[P] (EPISTEMIC — where to
 * look). Var separates UNKNOWN (few counts → high var) from CONFLICTED (many mixed counts → low var, same P≈0.5)
 * — which the plain grid's Bernoulli variance P(1−P) cannot. See OccGridParams beta_* and self_test property (5).
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>
#include <Eigen/Dense>

namespace rc
{

struct OccGridParams
{
    float cell_size_m = 0.05f;      // grid resolution
    // Inverse-sensor-model log-odds (from the DetectionProb/ClutterProb sensor model). l_hit chosen so ONE hit
    // crosses occ_threshold (completeness: a single confident return → occupied in one frame).
    float l_hit       = 0.85f;      // += on a hit  (≈ log(pd/pc))
    float l_miss      = 0.40f;      // −= on a traversed (see-through) cell
    float l_clamp     = 5.0f;       // |log-odds| clamp — bounded memory (dynamic obstacles clear in seconds)
    // HYSTERESIS (two thresholds) → stable blobs. A cell LATCHES occupied when lo rises above occ_set and only
    // clears when lo falls below occ_clear (occ_clear ≪ occ_set). So a marginal cell oscillating hit/miss near
    // the boundary can't flicker; only sustained see-through (lo → occ_clear) clears it. occ_set is low enough
    // that ONE hit sets it (completeness); occ_clear is well below 0 so several see-through frames are needed.
    float occ_set     = 0.40f;      // latch OCCUPIED when lo > this  (one NEAR hit at l_hit crosses it)
    float occ_clear   = -1.20f;     // release to FREE  when lo < this (sustained see-through). Far-persistence now
                                    // comes from range-weighted misses (below), not a wider gap — so this stays proven.
    // MOTION STABILITY — evidence weighted by measurement PRECISION. A LiDAR hit's room-frame position error grows
    // with range (angular footprint r·Δθ + the localisation lever arm r·σ_ψ), so a FAR return is unreliable and a
    // moving robot's pose jitter makes far cells flicker. So a hit's evidence is scaled by w(r)=r0²/(r0²+r²): a
    // NEAR hit (collision-relevant) keeps full weight and latches in one frame (completeness, safety); a FAR hit
    // adds little, so it needs SEVERAL consistent frames to latch — transient far jitter can't create a phantom.
    // MISSES are weighted the SAME way (by the cleared cell's range): a FAR see-through barely clears — at range
    // a beam's position is uncertain and may pass BESIDE the obstacle, not through it — so an obstacle PERSISTS
    // until the robot is close enough to confirm free space. Occluded cells are never traversed → held. 0 ⇒ off.
    float hit_reliable_range_m = 2.5f;   // range where an observation's evidence halves (precision ∝ 1/(1+(r/r0)²))
    // Nav band (the floor/ceiling "explainer" at the sensor model). A return is an obstacle only in this band.
    float floor_z0    = 0.06f;      // floor height at the sensor
    float floor_slope = 0.04f;      // + per metre of horizontal range (grazing)
    float ceil_z      = 1.80f;
    float z_band_margin_m = 0.10f;  // z-aware clearing tolerance
    // ── FLOOR EXPLAIN-AWAY AS A MIXTURE RESPONSIBILITY, not a hard band ──
    // The nav band above is a STEP: a return one millimetre over floor_z0+floor_slope·r is a FULL-weight obstacle,
    // and since l_hit·w(r) > occ_set for every range under ~2.7 m, it LATCHES its cell in a single frame. The floor
    // is not a step, though — it is a MEASURED surface with MEASURED scatter: the plane fit reports its own
    // residual RMS (≈7 cm in the apartment — larger than floor_z0 itself). A return just above the band is
    // therefore far more likely a floor return in that scatter's tail than an obstacle, and the band cannot know
    // that because it discards the scatter it was fitted with.
    // So a return's HIT weight is multiplied by the RESPONSIBILITY of the OBSTACLE component in the two-component
    // per-return mixture {floor, obstacle}: floor ~ N(z_floor(x,y), σ_f(r)²), obstacle ~ Uniform over the nav band,
    //     r_obst(z) = u / (u + N(z; z_floor, σ_f)),        u = 1/(ceil_z − z_floor)
    //     σ_f(r)²   = rms² + (floor_slope·r)² + floor_sigma_min²
    // — the floor's own measured roughness, the grazing term, and the sensor's irreducible range noise. This is the
    // construction that fixed the refrigerator's wall explain-away: a competing explanation belongs INSIDE the
    // per-point mixture, never in a gate. It self-calibrates with no threshold to tune — over a crisp floor
    // (small rms) a 6 cm obstacle already earns nearly full weight; over a rough or badly-registered floor
    // (large rms) an obstacle must stand ~3σ clear before the model credits it. And it can only ever REDUCE a
    // weight: the hard band above is KEPT as the term's support, so nothing that was not already a hit can become
    // one, and the change is a strict tightening of the occupied condition.
    bool  floor_responsibility = true;   // false ⇒ the old hard step (full weight everywhere above the band)
    float floor_sigma_min_m    = 0.03f;  // irreducible sensor range noise (m) — the σ floor of the floor model
    // ── A FLOOR RETURN IS FREE EVIDENCE FOR ITS OWN CELL (the other half of the inverse sensor model) ──
    // A beam that terminates ON the floor at (x,y) says the column above the floor in that cell is EMPTY — that is
    // the whole content of the floor explainer. The original model dropped it: a below-band return produced no hit
    // (correct) and no miss either (the DDA marks the cells it TRAVERSES and breaks at the endpoint cell), so the
    // endpoint cell received NO evidence at all. That is a RATCHET. Once a cell has latched from a single noisy
    // near-floor return, the thousands of clean floor returns that land in it every second — each one direct proof
    // that the floor, not an obstacle, is there — are recorded as nothing. Its only clearing route is a beam that
    // both traverses it toward something farther away AND passes inside its remembered z-band, which is exactly
    // what miss_blocked_zaware measures being discarded (95%+ of all clearing evidence). Hence floor phantoms that
    // freeze forever: `occupied` grows monotonically and then never changes again.
    // With this on, a below-band return delivers free evidence to its own cell, gated on SUPPORT: the return
    // reached the floor, so anything RESTING on the floor there would have blocked it first (refuted, clear it),
    // whereas a FLOATING surface — a tabletop, a shelf — was merely passed underneath (held). The cell already
    // records which of the two its evidence is: zmn_, the lowest return ever seen there. See
    // mark_floor_endpoint_flag; the ordinary z-overlap gate cannot serve here, because a beam that terminates on
    // the floor arrives at floor height by definition and would be discarded every single time.
    bool  floor_return_clears  = true;   // false ⇒ the old behaviour (a floor return carries no evidence at all)
    // C-space INFLATION: regrow the occupied set outward by this radius (≈ half the robot width) so narrow gaps
    // the robot can't pass close, and a table's sparse LiDAR ring-cells bridge into a solid footprint. The
    // inflated "border" is displayed in a second colour and the published obstacle components use it. 0 = off.
    float inflate_radius_m = 0.25f;
    // Resolution at which the PUBLISHED obstacle polygons are built. The grid reasons at cell_size_m (0.05 m)
    // because occupancy evidence wants to be fine; the planner does not — its own grid_resolution_m is 0.35 m,
    // so 5 cm polygon fidelity buys nothing and costs enormously. Decomposing the occupied set at 5 cm produced
    // 154 polygons median / 461 peak, and the controller's visibility graph is O(V²·E) in obstacle vertices:
    // ~1.2e8 segment tests at 154 polygons, ~3.1e9 at 461. plan_path stops returning, so there is no route, and
    // the robot reports itself stuck standing in open floor. Coarsening to 0.20 m cuts the count ~16×.
    // CONSERVATIVE by construction: a coarse cell is occupied if ANY fine cell inside it is, so the published
    // set never under-covers the real obstacle — it only rounds outward. 0 ⇒ publish at cell_size_m.
    // 0 = publish at the grid's own resolution. It was 0.20 to cut polygon COUNT for the controller's
    // visibility-graph planner, which was O(V²·E) in obstacle vertices. That planner is gone: the controller
    // now rasterises the polygons into its own grid, where cost is independent of polygon count (measured:
    // 24 vs 960 polygons, identical build and plan time). So the coarsening buys nothing and costs a great
    // deal — roundingevery isolated 5 cm cell up to a 20 cm block is a 16× area inflation, which turned ~4.75 m²
    // of scattered residual into ~60 m² of merged rectangles covering the whole room and made every target
    // infeasible. Publish the true extent; the consumer that knows the robot's shape decides what fits.
    float publish_cell_size_m = 0.0f;
    // ── BETA–BERNOULLI belief (the field the PLANNER consumes) ──
    // Alongside the log-odds latch, each cell carries a Beta(α,β) posterior over its occupancy PROBABILITY. It
    // gives the planner BOTH terms it needs: the mean P=α/(α+β) (collision RISK) and the variance Var[P] (the
    // EPISTEMIC term — where to look). Crucially Var separates UNKNOWN (few counts → wide Beta → high var) from
    // CONFLICTED (many mixed counts → low var) even though both have mean≈0.5 — the plain grid's Bernoulli
    // variance P(1−P) cannot. Evidence weights mirror the log-odds asymmetry (a hit is stronger than a miss →
    // the mean is biased occupied under equal evidence, the SAFE direction). The prior is Jeffreys (0.5,0.5):
    // an unobserved cell reads P=0.5 at MAX variance = "unknown". κ_max caps the concentration α+β → a variance
    // FLOOR (the field is never infinitely confident) AND bounded memory (old evidence is discounted as new
    // arrives → dynamic obstacles clear, same role the log-odds clamp plays for the latch).
    float beta_prior_a = 0.5f;      // Jeffreys prior α  (unobserved ⇒ P=0.5, max variance)
    float beta_prior_b = 0.5f;      // Jeffreys prior β
    float beta_hit_w   = 0.85f;     // α += this on a hit-cycle  (mirrors l_hit → safe occupied bias)
    float beta_miss_w  = 0.40f;     // β += this on a miss-cycle (mirrors l_miss)
    float beta_kappa_max = 40.0f;   // cap on α+β → variance floor (σ_P≈0.08) + bounded memory (dynamics clear)
    // ── FORGETTING: evidence loses PRECISION with time since last observation ──
    // κ_max above bounds how confident a cell can get, but it scales α and β TOGETHER, so it preserves the mean:
    // it caps confidence and never relaxes the belief. A cell that stops being observed therefore keeps its
    // posterior EXACTLY, forever — and an OCCLUDED cell can never be observed again by construction, so anything
    // ever latched behind an occluder (a door that swung shut, someone who walked past, a mis-registered scan) is
    // immortal. That is not a Bayesian subtlety, it is a missing term: with no new data the right posterior for a
    // region you cannot see is not "still occupied", it is "I no longer know".
    // So relax the posterior toward the Jeffreys prior at a fixed half-life: (α−α₀, β−β₀) ×= ½^(Δt/T). P drifts to
    // 0.5 and Var rises — which is ALSO what makes Var[P] mean what this header claims it means (the EPISTEMIC
    // term, "where to look"): stale space becomes uncertain and therefore attractive to re-observe, instead of
    // reading as confidently occupied. Continuously-observed cells are unaffected (their evidence is renewed every
    // cycle). 0 ⇒ off (original never-forget behaviour).
    float forget_half_life_s = 10.0f;
    // (There was a z_band_relax knob here that made the remembered z-band CONTRACT. It was removed: the theory
    // behind it had the sign backwards and it caused a measured 3.0× clearing regression. See the long note at
    // the zmn_/zmx_ update in commit_cycle before considering anything like it again.)
    // ── SELF-BODY term in the SENSOR model ──
    // A beam that terminates on the robot's own surface says nothing about the world, so it must not contribute
    // occupancy evidence. Modelling the robot only as a read-out MASK around its CURRENT pose (which is what
    // residual_concept did) cannot work against a LATCHED map: self-returns are written into the map permanently,
    // stay hidden while the robot is standing over them, and re-emerge as residual the moment it drives away —
    // the robot lays a trail of phantoms behind itself. The fix belongs in the sensor model, at integration.
    // Not a self-filter radius: the HIT weight is scaled by P(this return came from the world, not from us) =
    // Φ(s/σ) where s is the signed distance from the return to the body envelope. σ is the body surface's own
    // positional uncertainty (mount + pose + beam footprint), so the term is a precision, not a cutoff — it fades
    // to nothing within ~2σ of the surface and never suppresses a clearly-external return. Clearing is left alone
    // (freeing space is always safe). radius 0 ⇒ off.
    float self_body_radius_m = 0.0f;   // set per cycle via set_self_body(); 0 disables the term
    float self_body_sigma_m  = 0.08f;  // positional uncertainty of the body surface
};

// Per-sweep integration counters — a DIAGNOSTIC of the sensor-model dynamics (esp. the tabletop-clearing bug:
// grazing beams that skim a horizontal surface ray-trace THROUGH its cells and clear them). Reset each sweep.
struct SweepDiag
{
    long hits = 0;               // CELLS given a +l_hit this cycle (one per cell, hit precedence — not per ray)
    long misses = 0;             // CELLS given a −l_miss this cycle (one per cell — not per ray)
    long miss_blocked_zaware = 0;// beam→cell see-throughs suppressed because the beam missed the cell's z-band
    long cells_latched = 0;      // cells that crossed occ_set → OCCUPIED this cycle
    long cells_released = 0;      // cells that fell below occ_clear → FREE this cycle
    long hit_then_cleared = 0;    // structurally 0 now (hit precedence): kept as a regression sentinel — must stay 0
    long cells_forgotten = 0;     // cells RELEASED by the time-decay (unobserved too long) rather than by a
                                  // see-through. Non-zero ⇒ forgetting is reaching occluded/stale space, which
                                  // no amount of clearing evidence could ever have reached. 0 for a whole run
                                  // with forget_half_life_s>0 ⇒ nothing is going stale (or the decay is too slow).
    long self_hits_damped = 0;    // returns whose HIT weight was attenuated by the self-body term (<0.99×)
    long floor_damped_hits = 0;   // in-band returns whose HIT weight the FLOOR RESPONSIBILITY cut (<0.9×) — these
                                  // are the near-floor returns that used to latch a cell outright. 0 for a whole
                                  // run ⇒ the term is not engaging (check floor_responsibility / the fit's rms).
    long floor_endpoint_clears = 0;// below-band (floor) returns that cleared THEIR OWN cell — the evidence the old
                                  // model discarded. Compare against miss_blocked_zaware: this is clearing that
                                  // no traversing beam could ever have delivered.
};

// One connected occupied region → footprint + z-band, ready for the scene-graph publish (box + hull).
struct OccComponent
{
    std::vector<Eigen::Vector2f> hull;                 // convex-hull footprint (room xy) — Layer B
    // CONCAVE outline (room xy) traced from the component's cell boundary: unlike `hull`, it does NOT
    // fill C/U/L concavities, so a planner consuming it leaves the real free space inside a concavity
    // navigable. One or more CCW loops (outer boundaries only; holes dropped = conservative). Falls back
    // to `hull` if the trace is degenerate. Consumed by grid_obstacle_hulls (the controller's planner).
    std::vector<std::vector<Eigen::Vector2f>> outline;
    float cx = 0, cy = 0, w = 0.05f, d = 0.05f, yaw = 0;  // AABB footprint (Phase 0: axis-aligned)
    float z_min = 0, z_max = 0;                        // vertical band (from the cells' hit z-bands)
    int   n_cells = 0;
};

class OccupancyGrid
{
public:
    OccupancyGrid() = default;

    // (Re)allocate to cover [xmin,xmax]×[ymin,ymax] (room frame, m). Clears all evidence.
    void reset(float xmin, float ymin, float xmax, float ymax, const OccGridParams& p);
    bool valid() const { return w_ > 0 and h_ > 0; }
    const OccGridParams& params() const { return p_; }

    // Reference the floor/obstacle band to a DATA-DRIVEN floor plane z=a·x+b·y+c instead of z=0. The obstacle
    // test becomes: z > floor_z(x,y) + floor_z0 + floor_slope·range. This makes the floor explainer follow an
    // offset/tilted floor (a new scenario), so floor returns never latch. (0,0,0) ⇒ the original fixed band
    // (zero regression). Set once per cycle before integrate_sweep; persists until changed.
    // `rms` is the fit's OWN residual scatter (metres) — how planar the surface it just fitted actually is. It is
    // the σ of the floor component in the mixture responsibility (see OccGridParams::floor_responsibility), so the
    // model's tolerance for a near-floor return is set by MEASURED floor quality rather than by a constant. 0 ⇒
    // unknown ⇒ σ falls back to floor_sigma_min_m ⊕ the grazing term.
    void set_floor_plane(float a, float b, float c, float rms = 0.0f)
    { fp_a_ = a; fp_b_ = b; fp_c_ = c; fp_rms_ = rms; }

    // P(this return came from an OBSTACLE, not from the floor) for a return at height z over (x,y) seen at
    // horizontal range `range` — the obstacle component's responsibility in the {floor, obstacle} mixture. Public
    // because the READ-OUT floor explainer must score a cell with the SAME model that scored the returns that
    // built it; two datums/tolerances for one surface is how a floor cell ends up unexplainable. 1 ⇒ certainly an
    // obstacle, 0 ⇒ certainly the floor. Returns 1 when floor_responsibility is off (the old hard-step behaviour).
    float floor_obstacle_responsibility(float x, float y, float z, float range) const;

    // Place the robot's body envelope for this cycle (room frame) so integrate_sweep can discount returns that
    // came off the robot itself. radius<=0 ⇒ the term is off. Set once per cycle before integrate_sweep, like
    // set_floor_plane; the value persists until changed. See OccGridParams self_body_* for why this belongs in
    // the sensor model rather than in a read-out mask.
    void set_self_body(float x, float y, float radius_m)
    { self_x_ = x; self_y_ = y; self_r_ = radius_m; }

    // Integrate one sensor sweep (room frame). origin = sensor position (room). See the header comment.
    //
    // CRITICAL (occupancy-grid correctness): the inverse sensor model must be applied ONCE PER CELL PER SCAN,
    // not once per ray. A scan has thousands of beams and a single cell is crossed by dozens of them; applying
    // −l_miss for every crossing over-counts free evidence ~100× and drives any not-continuously-re-hit cell
    // (e.g. a grazed tabletop) straight to the clamp. So integrate_sweep only ACCUMULATES per-cell hit/miss
    // FLAGS; the single log-odds update per cell happens in commit_cycle(). Hits take precedence over misses
    // (a cell hit by any beam this scan is a HIT, never cleared this scan) — this is the costmap_2d/OctoMap rule.
    //
    // Call once per sensor per cycle: begin_cycle=true (first sensor) clears the flags + diagnostics; pass
    // begin_cycle=false for later sensors (e.g. ZED after LiDAR) so their evidence ACCUMULATES into the same
    // cycle. Then call commit_cycle() exactly once to fold the accumulated flags into the log-odds field.
    // `reliability` (0..1) is a GLOBAL evidence scale for the whole sweep — the EGO-MOTION precision: 1 when the
    // robot is still, <1 when it moves fast (pose jitter + motion blur make the whole scan less trustworthy), so
    // the stable accumulated field dominates during motion and sharpens when stopped. Combined with the per-hit
    // range weight, this is the motion-stability control. 1 = full trust.
    //
    // `hit_weight_scale` (optional, parallel to points_room) multiplies into each return's HIT weight ONLY — a
    // per-point precision cue (e.g. RGB-semantic floor down-weighting: a floor-labelled near-floor ZED return
    // contributes a weaker hit, so a phantom floor obstacle needs more evidence to latch). It does NOT touch the
    // MISS/clearing weight — clearing free space is always safe, independent of a point's semantics. nullptr /
    // wrong-size ⇒ all hits keep their full range×ego weight.
    void integrate_sweep(const Eigen::Vector3f& origin, const std::vector<Eigen::Vector3f>& points_room,
                         bool begin_cycle = true, float reliability = 1.0f,
                         const std::vector<float>* hit_weight_scale = nullptr);
    // Fold this cycle's accumulated per-cell hit/miss flags into the log-odds field: exactly one +l_hit or
    // −l_miss per cell (hit precedence), then update the hysteresis latch. Call once after all sensors' sweeps.
    // `dt_s` is the elapsed time since the previous commit and drives the FORGETTING term (see
    // OccGridParams::forget_half_life_s): cells observed by NEITHER a hit nor a miss this cycle relax toward the
    // prior. Pass 0 (the default) to disable forgetting entirely — the original never-forget behaviour, which is
    // what the self_test's older properties assume.
    void commit_cycle(float dt_s = 0.0f);

    // A read-out predicate: the PROBABILITY (0..1) that this cell (world xy, hit z-band) is EXPLAINED by a known
    // model — hard for walls/floor/robot (0 or 1), SOFT for objects (Φ(−sdf/σ) marginalised over the object's
    // published position covariance, so a distant/uncertain object collapses over a σ-wide fuzzy boundary rather
    // than a hard kσ box). The field is attenuated by (1−p_explained); boolean read-outs (cells/hulls) decide at
    // the MAP point p>0.5. Evidence is masked at read-out, NEVER deleted, so a mis-fit can't erase a real
    // obstacle. Empty ⇒ nothing explained (full occupied set).
    using CellExplained = std::function<float(float x, float y, float z_lo, float z_hi)>;

    // Connected components (8-neighbour) of RESIDUAL cells (occupied ∧ ¬explained), optionally after C-space
    // INFLATION by inflate_radius_m (bridges gaps + adds clearance) → footprint polygon + z-band.
    std::vector<OccComponent> occupied_components(int min_cells = 1, const CellExplained& explained = {},
                                                  float inflate_radius_m = 0.0f) const;

    // Flat [x0,y0,x1,y1,...] centres of every residual (occupied ∧ ¬explained) cell — the raw obstacle (colour A).
    std::vector<float> residual_cell_centres(const CellExplained& explained = {}) const;
    // Same residual cells but flat [x0,y0,z0,...] with z = the cell's REAL top height (hit z-band max),
    // so a consumer can raise a 3-D surface to the actual obstacle height instead of a flat display z.
    std::vector<float> residual_cell_centres_xyz(const CellExplained& explained = {}) const;
    // Flat [x0,y0,…] centres of the INFLATED BORDER — cells added by dilating the residual set by
    // inflate_radius_m that were not themselves occupied (the clearance ring, drawn in colour B).
    std::vector<float> inflated_border_centres(const CellExplained& explained, float inflate_radius_m) const;

    // ── read-out ──
    bool  occupied(int ix, int iy) const;
    float logodds (int ix, int iy) const { return in_bounds(ix, iy) ? lo_[idx(ix, iy)] : 0.0f; }
    // Remembered hit z-band of a cell (0 if never hit). Exposed for the self_test's bounded-memory property —
    // this band gates z-aware clearing, so its width IS the cell's clearability.
    float zband_lo(int ix, int iy) const { return in_bounds(ix, iy) and hit_[idx(ix, iy)] ? zmn_[idx(ix, iy)] : 0.0f; }
    float zband_hi(int ix, int iy) const { return in_bounds(ix, iy) and hit_[idx(ix, iy)] ? zmx_[idx(ix, iy)] : 0.0f; }
    // Beta–Bernoulli belief the planner consumes. prob = mean occupancy P (RISK). prob_variance = Var[P] (the
    // EPISTEMIC term). prob_std = √Var. Out-of-bounds ⇒ the unobserved prior (P=0.5, max variance) = "unknown".
    float prob         (int ix, int iy) const;
    float prob_variance(int ix, int iy) const;
    float prob_std     (int ix, int iy) const;
    // Dense fields (row-major, size w·h) for the planner / publish: P and Var over the whole grid extent. Cells
    // EXPLAINED by a modelled object (the predicate) are COLLAPSED to (P=0, Var=0) — the object agent owns that
    // region, so the residual/null field defers to it. Evidence is masked at read-out, never deleted internally
    // (a specialist mis-fit that vanishes lets the residual reappear). Empty predicate ⇒ raw field.
    void  occupancy_fields(std::vector<float>& prob_out, std::vector<float>& var_out,
                           const CellExplained& explained = {}) const;
    int   width()  const { return w_; }
    int   height() const { return h_; }
    float xmin()   const { return xmin_; }
    float ymin()   const { return ymin_; }
    float cell_size() const { return inv_cell_ > 0 ? 1.0f / inv_cell_ : 0.0f; }
    bool  world_to_cell(float x, float y, int& ix, int& iy) const;
    void  cell_to_world_pub(int ix, int iy, float& x, float& y) const { cell_to_world(ix, iy, x, y); }
    long  occupied_count() const { long n = 0; for (auto v : occ_) n += v; return n; }
    // Height histogram of the LATCHED cells, binned by the cell's running-max hit height zmx_ (the tallest thing
    // ever seen there). Diagnostic for "are these phantoms?": a cell whose tallest return is only a few cm above
    // the floor is a GRAZING/floor return, not an obstacle — a mass of such cells is the floor-band signature.
    // `edges` are the upper bin edges in metres; the returned vector has edges.size()+1 entries (last = above all).
    std::vector<long> occupied_height_hist(const std::vector<float>& edges) const
    {
        std::vector<long> bins(edges.size() + 1, 0);
        for (std::size_t i = 0; i < occ_.size(); ++i)
        {
            if (not occ_[i] or not hit_[i]) continue;
            std::size_t k = 0;
            while (k < edges.size() and zmx_[i] > edges[k]) ++k;
            ++bins[k];
        }
        return bins;
    }
    // Same histogram over the RESIDUAL set (occupied ∧ ¬explained) — the cells that actually leave this agent as
    // obstacles. occupied_count()/occupied_height_hist() above count EVERY latched cell, walls included, and in an
    // apartment the walls are the large majority of them: a diagnostic built on those numbers cannot see the
    // phantoms at all (it reports ~1850 tall wall cells and calls the run healthy). These two are the ones to read.
    long residual_count(const CellExplained& explained = {}) const;
    std::vector<long> residual_height_hist(const std::vector<float>& edges,
                                           const CellExplained& explained = {}) const;

    const SweepDiag& last_sweep_diag() const { return sd_; }

    static bool self_test();

private:
    int  idx(int ix, int iy) const { return iy * w_ + ix; }
    bool in_bounds(int ix, int iy) const { return ix >= 0 and ix < w_ and iy >= 0 and iy < h_; }
    void cell_to_world(int ix, int iy, float& x, float& y) const;
    // The z-band a cell presents to the READ-OUT explainer. Deliberately NOT (zmn_, zmx_): zmx_ is a running MAX
    // that never contracts (correct for gating clearing beams — see the note in commit_cycle) so a single transient
    // tall return permanently relabels a floor-height cell as a tall one. The read-out asks a different question —
    // "what is there NOW?" — whose answer is dispz_, the EMA of the cell's current top. Scoring the floor explainer
    // against a ratcheted maximum is how a cell built entirely from floor-height returns becomes an obstacle that
    // no explainer can ever account for again.
    void readout_zband(int i, float& zlo, float& zhi) const
    {
        if (not hit_[i]) { zlo = 0.0f; zhi = 0.05f; return; }
        zhi = dispz_[i];
        zlo = std::min(zmn_[i], zhi);
    }
    void mark_hit_flag (int ix, int iy, float zlo, float zhi, float w);   // accumulate a hit + its precision weight
    void mark_miss_flag(int ix, int iy, float beam_z, float w);           // accumulate a see-through + its weight
    // A beam that TERMINATED on the floor in this cell: free evidence gated on SUPPORT (is the cell's own lowest
    // evidence floor-standing, hence refuted, or floating, hence merely passed under?) rather than on z-overlap,
    // which a floor return can never satisfy. See the long note at the definition.
    void mark_floor_endpoint_flag(int ix, int iy, float band_top, float w);
    std::vector<std::uint8_t> residual_mask(const CellExplained& explained) const;   // occupied ∧ ¬explained
    std::vector<std::uint8_t> dilate_mask(const std::vector<std::uint8_t>& m, int radius_cells) const;

    OccGridParams p_;
    float xmin_ = 0, ymin_ = 0, inv_cell_ = 0;
    float fp_a_ = 0, fp_b_ = 0, fp_c_ = 0;        // data-driven floor plane z=a·x+b·y+c (0 ⇒ fixed z=0 band)
    float fp_rms_ = 0;                            // that fit's own residual scatter (m) = σ of the floor component
    float self_x_ = 0, self_y_ = 0, self_r_ = 0;  // robot body envelope this cycle (room frame); r<=0 ⇒ term off
    int   w_ = 0, h_ = 0;
    std::vector<float>        lo_;                 // log-odds (drives the hard occupied() latch — unchanged)
    std::vector<float>        a_, b_;              // Beta(α,β) posterior over occupancy probability (the belief field)
    std::vector<float>        zmn_, zmx_;          // per-cell hit z-band (zmx_ is a running MAX — for miss gating)
    std::vector<float>        dispz_;              // per-cell CURRENT top height (EMA of this cycle's hits) for display
    std::vector<std::uint8_t> hit_;                // 1 once a cell has received any hit (z-band valid)
    std::vector<std::uint8_t> occ_;                // LATCHED occupied state (hysteresis: set at occ_set, clear at occ_clear)
    // Per-CYCLE scratch (reset on begin_cycle, folded once in commit_cycle) — the fix for miss over-counting.
    std::vector<std::uint8_t> shit_;               // this cycle: cell received an endpoint hit (any beam)
    std::vector<std::uint8_t> smiss_;              // this cycle: cell was seen-through (z-aware) by any beam
    std::vector<float>        shz_lo_, shz_hi_;    // this cycle: accumulated hit z-band for shit_ cells
    std::vector<float>        shit_w_;             // this cycle: MAX precision weight of the hits on this cell (range×motion)
    std::vector<float>        smiss_w_;            // this cycle: MAX precision weight of the see-throughs on this cell
    SweepDiag sd_;                                 // counters for the cycle just committed
};

}  // namespace rc
