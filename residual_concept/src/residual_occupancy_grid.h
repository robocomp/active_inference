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
    // ── ...but a STATIC structure must get HARDER to remove the longer it has stood ──
    // With one fixed clamp, a voxel confirmed for 3000 cycles banks exactly as much evidence as one confirmed
    // for 12, so the outcome is decided by the instantaneous hit:miss ratio and not by history. Measured live
    // 2026-08-23 with two static tables in the room: their cells hovered at the release threshold (mean
    // lo_before = -1.11) and churned for thousands of cycles — 322 kills on one table, 270 on the other. A map
    // that cannot become more certain about a thing that never moves is the wrong map.
    // So the CAPACITY is per voxel and grows with CONSISTENCY: every observation that agrees with the voxel's
    // current belief raises its stability, every one that contradicts it drops it sharply (asymmetric — trust is
    // slow to earn and quick to lose). The clamp becomes l_clamp·(1 + stable_gain·s), s in [0,1].
    // This is the essential part of the per-cell learned dynamics in the literature (Saarinen et al., IROS 2012,
    // "Independent Markov Chain Occupancy Grid Maps"; Meyer-Delius et al., AAAI 2012): a cell that never changes
    // learns it is static and becomes stiff; a cell that flickers stays responsive. It REPLACES an arbitrary
    // constant with a measured property of the cell, rather than adding another rule on top of it.
    // The gain is bounded on purpose: removal must stay possible in tens of seconds, not minutes. A table
    // carried away still clears — the probe measures it — because clearing scales with the same capacity.
    float stable_gain = 3.0f;       // a fully consistent voxel may bank (1+gain)× the base evidence. 0 ⇒ off
    float stable_rise = 0.02f;      // per agreeing observation (≈50 to earn full stiffness)
    float stable_fall = 0.30f;      // ...per contradicting one: earned slowly, lost fast
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
    float z_band_margin_m = 0.10f;  // z-aware clearing tolerance (the SUPPORT of the soft term below)
    static constexpr int Z_BINS = 64;   // height-support resolution (see bin_span_m)
    // ── DO NOT CLEAR CLOSE TO THE HIT (costmap_2d's rule, made adaptive) ──
    // A voxel is 6.25 cm tall and a tabletop is 2 cm thick, so a beam grazing just over a plate is INSIDE the
    // plate's own voxel and raytraces straight through it. This is the documented weakness of a voxel costmap,
    // and it is why the pure 3-D model measured 2530 releases on a standing table (vs 25) even though its
    // removal behaviour was excellent. costmap_2d guards it by not clearing the endpoint cell; the same idea
    // has to reach further when the beam is shallow, because then it stays inside one voxel height for a long
    // way. So skip the clearing run over which the beam has not yet changed voxel height:
    //     d_stop = voxel_height / |uz|,  capped
    // Steep beam ⇒ ~one voxel, exactly costmap_2d's rule. Shallow beam onto a flat surface ⇒ the whole grazing
    // run, which is precisely the stretch where the sensor cannot say which cell it first touched. The cap is
    // what keeps a near-horizontal beam from refusing to clear along its whole length; it is the one number
    // here that is a judgement rather than a measurement, and it is bounded by the fact that clearing resumes
    // beyond it. A surface that has genuinely gone is still cleared by the beams that now pass THROUGH its
    // voxels — measured: a table carried away is gone within ~100 cycles.
    // ── A BEAM CANNOT REFUTE WHAT THE SENSOR COULD NOT HAVE SEEN ──
    // Every ray starts AT the sensor, so the DDA marks the voxels around the robot free — and it weights them by
    // w(r), which is largest at r = 0. The near field, where the lidar is blind, is therefore cleared HARDEST of
    // all. Below its minimum range a lidar returns nothing whatever is there, so a ray "passing through" that
    // shell is not an observation at all and must carry no free evidence.
    // This is what removes a table the moment the robot closes on it: the cells fall inside the dead shell, get
    // cleared at full weight, and the map deletes a table the robot is about to touch. Going to 3-D did NOT fix
    // it — I deleted the envelope term in the rewrite on the argument that the geometry made it unnecessary, and
    // that was wrong: the rays still pass through those voxels.
    // Set per sweep, like set_device_floor_z0, from the device's own datasheet minimum. 0 ⇒ no dead shell.
    // ── CLEARING AUTHORITY IS PROPORTIONAL TO MEASUREMENT PRECISION ──
    // Measured live 2026-08-23 with the new `sensor` column on the release trace: of 3028 removals, **2780 were
    // the ZED and 248 helios**, and the ZED's kill hotspots land exactly on the two tables (402 at (0,0), 277 at
    // (2,0), 151 at (2.5,0)) at a mean range of 2.1 m. The ZED raytraces ~8000 rays a cycle and is allowed to
    // MARK about 250 of them, so it is almost purely a clearing sensor — and its depth noise grows as range²
    // (sigma = 0.03 + 0.006·r²): at 4 m its endpoint is uncertain by 13 cm. A ray that cannot say where it
    // stopped to better than 13 cm cannot certify the space in front of it, and an OVERESTIMATED depth sends it
    // straight through the tabletop it actually hit.
    // So a sweep declares its own noise model (set_sensor_noise) and gets clearing authority in proportion:
    //     scale = sigma_ref² / max(sigma_ref², sigma(r)²)
    // A sensor as precise as the reference clears at full weight (helios: unchanged, scale 1.0); the ZED at 4 m
    // clears at 0.06 of that. No threshold, no on/off switch — the same precision-weighting the range term
    // already applies, extended across sensors instead of only across range within one.
    // The endpoint margin follows the same quantity: do not clear within k·sigma(r) of where the beam stopped.
    float reference_sigma_m   = 0.03f;  // the LiDAR's range noise — the yardstick for "full authority"
    float clear_stop_sigma_k  = 2.0f;   // endpoint margin, in sigmas of the sweep's own noise
    float clear_stop_max_m    = 0.10f;
    // ── ...and the guard may only fire when the beam stopped on a THIN HORIZONTAL surface ──
    // The height test alone has a systematic failure with a ring lidar. helios stands at 1.075 m, so a beam at
    // ring height that crosses a phantom in mid-room terminates on a WALL at that same height: |cz-ez| <= 1, the
    // guard fires, and the phantom is protected FOR EVER. Measured live 2026-08-23: 1402 residual cells above
    // 0.9 m, 1052 of them more than 1.5 m from any wall, accumulated over two minutes and then frozen at
    // latched = released = 0. A map the planner cannot cross.
    // The premise of the guard is "the surface that stopped the beam extends back to here", and that is a claim
    // about a HORIZONTAL surface — a plate the beam skimmed and then struck. A beam cannot skim a wall: a wall is
    // perpendicular to it, and its endpoint column is material from floor to ceiling. So look at what the beam
    // actually stopped on. A tabletop's endpoint column holds material in one or two bins; a wall's holds it in
    // dozens. Fire the guard only for the thin one.
    int   plate_bins_max      = 4;      // an endpoint column this thin (≈12 cm) is a horizontal surface, not a wall
    // ── THE LIDAR CLEARANCE RADIUS: nothing inside it may be cleared, by any sensor ──
    // The plain statement of the rule, and the right one. A lidar sees nothing within the disc the driver
    // self-filters, and every ray in this grid STARTS inside that disc and marks its way out — weighted by w(r),
    // which is largest at r = 0. So the ring of cells the robot is standing in was being cleared at the highest
    // weight the grid awards, in the one place no sensor can see. That is what deletes a table the moment the
    // robot closes on it, and it is a statement about the ROBOT, not about one sensor's minimum range: it holds
    // for the forward-mounted ZED exactly as it does for the two lidars, and it is measured from the robot to
    // the cell rather than along the ray.
    // Match it to the lidar3d driver's own [Footprint] radius, the same number self_body_radius_m uses, so both
    // agree on one body model. 0 ⇒ off.
    float lidar_clearance_m   = 0.55f;  // 0 ⇒ textbook VoxelLayer (endpoint voxel only, no grazing guard)
    // A cell may only be called FREE on the strength of the column the sensors actually resolved — but only the
    // part where an unseen obstacle would actually be HIT matters. Mixing over the whole 1.8 m column made every
    // cell read ~0.5 (no fan from a point sensor ever resolves a whole column), which would hand the planner a
    // map of pure risk. Restricting it to the collision band is what makes the term about the hazard: helios at
    // 1.075 m cannot see below 0.65 m at 0.3 m range, and that is precisely where a chair leg would be.
    float collision_band_top_m = 0.50f;  // 0 ⇒ no observability discount on the free side
    // ...and the band is scored in a few SUB-BANDS, not voxel by voxel. A 32-layer lidar spans 70 deg, so at
    // 2.5 m its beams are ~10 cm apart and can never touch every 3 cm voxel — demanding that would report most
    // of the map as unknown and hand the planner a wall of risk. The question the discount asks is "did we look
    // at this height REGION", which a sub-band answers and a voxel does not.
    static constexpr int COLLISION_GROUPS = 4;  // cap on the no-clear run before the endpoint (0 ⇒ endpoint voxel only)
    float bin_span_m          = 2.0f;   // height covered by the 32 support bins (0 ⇒ fall back to the zmn/zmx hull)
    // ── ...and ONE CONTINUOUS SURFACE may explain both the skim and the return ──
    // p_block alone did not save the table: it cut the killing weight from 0.87 to 0.48 for beams 5 cm over the
    // plate, but the beams that actually finished the cells were at 0.75-0.77 against a band of [0.75, 0.75] —
    // AT the surface, where p_block is ~0.95 and clearing is at full strength. The geometry is the reason. Helios
    // stands at 1.075 m and the plate is at 0.75, so at 1 m range the beam descends at only 18 deg: it crosses
    // four or five tabletop cells within 2 cm of the plate before terminating ON that same plate. Every one of
    // those crossings is recorded as free space, and the cell that stopped the beam is the only one that is not.
    // A flat surface at grazing incidence therefore erases itself, and the sensor cannot say which cell it first
    // touched: that is a real ambiguity, not a bug to gate away.
    // The evidence is what resolves it. The beam terminated on material at height end_z. If the material this
    // cell remembers (its top, zmx) lies BETWEEN the beam's height here and where the beam ended, then a single
    // continuous surface explains both facts at once — the beam skimmed this cell's surface and struck it a
    // little further along — and the crossing refutes nothing:
    //     p_surface = Phi((zmx - min(bz,end_z))/sigma) * Phi((max(bz,end_z) - zmx)/sigma),   w *= (1 - p_surface)
    // A horizontal beam ending on a far WALL is untouched: the traversed cell's material top is nowhere near the
    // narrow interval between two equal heights, so p_surface collapses to 0 and free space clears exactly as
    // before. And a table that has genuinely GONE still clears: the beams that used to stop on it now run down
    // to the floor, crossing its cells at heights well below the remembered plate, where no continuous surface
    // reaches the return and p_surface is 0 again.
    // ★ I ALSO TRIED AND REVERTED requiring the surface to be CONTIGUOUS along the ray — walking back from the
    // endpoint and firing the term only on the unbroken run of cells whose remembered top still matched the
    // return height. It reads better than the plain height test and it measured WORSE, twice, on the same probe:
    // releases inside the table footprint 36 -> 1084 breaking the run at un-hit cells, and 36 -> 1485 breaking it
    // instead at cells positively known free. The second is the instructive one. A tabletop's interior is mostly
    // cells a horizontal LiDAR never strikes, so any chain rule has to pass through unknown cells; and once ONE
    // plate cell is cleared, every later ray's run breaks there and exposes all the cells nearer than it. That is
    // positive feedback — the rule erases the surface faster the more of it has already been erased.
    // What survives from that attempt is the honest statement of the limitation: the plain height test also
    // protects a cell whose remembered top happens to match a return arriving from much further away at the same
    // height (a horizontal ring beam grazing a low box, returning from a wall beyond). That case is genuinely
    // ambiguous to a 2-D grid, it errs toward HOLDING an obstacle, and the removal control below shows it does
    // not become a leak: a table carried away still clears completely within ~10 s.
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
    // ── ...but forget OCCUPANCY only. ──
    // The paragraph above is right about an unobserved OCCUPIED cell and wrong about an unobserved FREE one.
    // Decaying (α,β) toward Jeffreys is symmetric, so a well-cleared cell (α≈0.5, β≈39.5, P≈0.012) also drifts
    // back toward P=0.5 — the map quietly un-learns free space it paid beams to establish. That costs nothing in
    // the POLYGON channel (lo_ decays toward 0 from below and can never cross the positive occ_set, so a free
    // cell cannot re-latch) but it is published in the FIELD channel, which five concept agents consume for
    // birth-surprise gating via common/birth_surprise/residual_field_reader.h.
    // octomap::OcTreeStamped::degradeOutdatedNodes makes exactly this distinction — it degrades nodes for which
    // isNodeOccupied() holds and leaves everything else alone. Free space that quietly reverts to unknown is what
    // makes a planner refuse a corridor it cleared twenty seconds ago.
    // So gate the decay on the cell holding NET OCCUPANCY evidence (lo_ > 0). Cells at or below zero — free, and
    // never-observed — are left completely untouched. (Since 2026-08-22 the decay no longer touches lo_ at all
    // by default, so this gate now selects which cells' FIELD ages; it still selects the lo_ decay under
    // forget_can_unlatch.)
    // Measured 2026-08-19: the decay was doing 42% of ALL un-latching, so this term is load-bearing and the fix
    // is to narrow WHAT it touches, not to lengthen the half-life (ForgetHalfLifeS=0 ratcheted occupied +56%).
    bool forget_occupied_only = true;   // false ⇒ the old symmetric relax-everything behaviour
    // ── ...and forget only what we could actually SEE. ──
    // Measured 2026-08-20 on the live run: 750 of 2548 occupied cells (29%) were decaying every cycle purely
    // because nothing was looking at them, and at a 10 s half-life a latched cell reaches occ_set after ~36 s
    // unobserved. So an obstacle the robot drives away from vanishes from the published map in half a minute,
    // in a static apartment where it certainly has not moved. The decay cannot distinguish "I stopped looking"
    // from "it left", and treating the two the same is inventing evidence.
    // The ray-DDA already knows the difference. slook_ marks every cell a beam passed through this cycle, set
    // BEFORE the z-aware gate can reject that beam as free evidence — so it is a statement about VISIBILITY,
    // not occupancy. Gate the decay on it: looked-at-and-not-confirmed decays, never-reached is held.
    // This is what STVL achieves with a frustum test, but derived from the real rays, which also repairs STVL's
    // documented weakness (it tests a point against an FOV cone with no occlusion model, so a voxel behind a
    // wall is punished as if it had been observed).
    // COST, stated plainly: a cell in a region the robot never revisits is now held indefinitely. That is the
    // correct posterior for "no information", and one hit or one see-through re-decides it the moment the robot
    // returns — but it does mean the time decay is no longer a blanket garbage collector.
    bool forget_visible_only = true;    // false ⇒ decay every unobserved cell, wherever it is
    // ── ...and at the precision the failed observation actually had. ──
    // Every piece of EVIDENCE in this grid is scaled by w(r)=r0²/(r0²+r²) — hits AND misses, deliberately, so a
    // far see-through barely clears and a distant obstacle PERSISTS until the robot closes on it. The decay was
    // the one mechanism that ignored range entirely. So at 8 m every term that would defend a cell runs at 0.089
    // of full strength while the term that erodes it runs at 1.0: an 11× mismatch, pointing at erasure.
    // That is the same defect as a constant process noise (CONCEPT_AGENT_LIFECYCLE): the rate of forgetting must
    // carry the same covariate as the rate of learning. "I looked from 8 m and did not confirm" is weak evidence
    // of absence — the beam may simply have passed BESIDE the obstacle — and must erode the posterior weakly.
    // So scale the per-cycle decay by the SAME w(r), r measured from the observer to the cell:
    //     γ_cell = 1 − (1 − γ) · w(r)
    // Near cells keep the configured half-life exactly; far cells forget proportionally slower, with no new
    // parameter (r0 is hit_reliable_range_m, already the precision scale for every other term).
    bool forget_range_weighted = true;  // false ⇒ one half-life everywhere, regardless of how well we saw it
    // ── ...and it may touch the BELIEF FIELD only, never the evidence ledger and never the latch. ──
    // The three gates above narrow WHICH cells the decay touches. This one settles what it is allowed to DECIDE.
    // Until 2026-08-22 the decay released a latched cell as soon as lo fell back below occ_set (+0.40) — a bar
    // three times easier than the occ_clear (−1.20) that free-space EVIDENCE has to clear, and reached without a
    // single beam ever contradicting the obstacle. Worse, that release also wiped the cell's z-band (hit_=0), and
    // the z-band is the only thing stopping every beam that passes over or under the obstacle from counting as a
    // see-through. So one forgetting release flipped ~94% of this grid's clearing traffic from "carries no
    // information about this column" to "clears it at full weight", drove lo to the −5 clamp, and left a real
    // obstacle unable to re-latch until it was directly struck several more times. A ratchet toward erasure, the
    // exact mirror of the marking ratchet that mark_floor_endpoint_flag exists to break.
    // The rule now: an occupied cell is released ONLY by accumulated evidence that it is FREE — see-through beams
    // that actually passed through its own z-band, each weighted by the precision w(r) it was collected at, summed
    // until lo < occ_clear against everything the cell has accumulated (up to l_clamp). Absence of observation is
    // not evidence of absence: the decay still relaxes lo toward 0 and the Beta toward Jeffreys, so a stale cell
    // reads as UNKNOWN with rising Var[P] in the field channel (the epistemic "go and look" signal is preserved,
    // and it is what should retire the cell — by sending the robot to observe it, not by assuming).
    // COST, stated plainly: a cell nothing can ever observe again — mass latched behind a door that then closed —
    // is now immortal in the polygon channel. That is the deliberate trade: a safety layer under maximum
    // uncertainty must call a cell occupied. Set true to restore the old behaviour for an A/B.
    // What the term still does, and why it is not simply deleted: (α,β) relax to Jeffreys, so P → 0.5 and
    // Var[P] → max. A stale cell reads as UNKNOWN in the field five concept agents consume, and — since Var[P]
    // IS the epistemic term — it becomes attractive to go and re-observe. That is the Active-Inference route to
    // retiring it: send the robot to LOOK, and let the beams decide. The log-odds is left untouched, because it
    // is the ledger of what was actually measured and no measurement arrived; eroding it spends the cell's
    // accumulated defence on the passage of time, so a single later see-through clears a cell that should have
    // needed the whole l_clamp → occ_clear span. "How sure am I" ages; "what have I measured" does not.
    bool forget_can_unlatch = false;    // true ⇒ the decay also erodes lo_ and may release a latched cell
                                        //        (the coupled pre-2026-08-22 behaviour, kept for A/B)
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
// ── WHY WAS THIS CELL REMOVED? ────────────────────────────────────────────────────────────────────────────────
// The counters below say HOW MANY cells were released; they cannot say why any particular one was, and "the
// residual under the table disappeared" is a question about one particular obstacle. Every release is therefore
// traced: the cell, what it had banked, what the beam that finished it looked like, and how long it had stood.
// Cheap by construction — only RELEASED cells are recorded, and a healthy run releases a handful per cycle.
struct ReleaseEvent
{
    float x = 0.0f, y = 0.0f;      // cell centre, room frame
    float lo_before = 0.0f;        // the evidence ledger before this cycle's update...
    float lo_after  = 0.0f;        // ...and after it crossed occ_clear
    float zmn = 0.0f, zmx = 0.0f;  // the cell's remembered occupied z-band (what we believed was there)
    float clear_z = 0.0f;          // height at which the see-through beam crossed this column
    float clear_w = 0.0f;          // precision weight that beam carried (range x ego-motion)
    float range_m = 0.0f;          // horizontal range from the observer to this cell
    long  age_cycles = 0;          // how many cycles it had been continuously latched. A LARGE age is the alarm:
                                   // a structure that stood for minutes was deleted by seconds of see-through.
    std::uint8_t cause = 0;        // 0 = see-through evidence (miss), 1 = time decay (forget_can_unlatch)
    std::uint8_t src = 0;          // which SWEEP delivered the see-through that finished it (set_sensor_id).
                                   // Without this "the table is being eroded" cannot be attributed to a sensor,
                                   // and this grid is fed by helios, bpearl and a ZED with very different
                                   // failure modes — the ZED raytraces ~8000 rays a cycle while only ~250 of
                                   // them are allowed to mark, so it is overwhelmingly a CLEARING sensor.
    float last_z = 0.0f;           // the cell's TOP before this cycle wiped it (see trace_release)
};

// ── WHAT CREATED THIS CELL? ───────────────────────────────────────────────────────────────────────────────────
// The release trace answers "what removed it". With an empty room every latched cell is by definition a phantom,
// so the question that matters is the opposite one, and nothing recorded it. One row per cell the moment it
// crosses occ_set: where, at what height, from how far, and WHICH SENSOR delivered the confirming return.
struct LatchEvent
{
    float x = 0.0f, y = 0.0f;      // cell centre, room frame
    float z = 0.0f;                // height of the voxel that crossed the threshold — WHAT was "seen"
    float lo = 0.0f;               // the column's log-odds at the moment it latched
    float range_m = 0.0f;          // how far the robot was from it
    float w = 0.0f;                // precision weight of the confirming return
    int   bins = 0;                // how many height bins of the column hold material (1-2 = a thin sliver)
    std::uint8_t src = 0;          // 1 bpearl, 2 helios, 3 zed
    float robot_x = 0.0f, robot_y = 0.0f;   // WHERE THE ROBOT STOOD when it created this cell. A phantom field
                                   // that is really displaced wall returns correlates with the pose, not with
                                   // the sensor — and that cannot be seen without logging the pose beside it.
};

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
    // ── Stage-1 instrumentation (2026-08-19). These exist because `floor_clears` sat at 0 for 9381 straight
    //    cycles and nobody could tell whether the term was rejecting its input or never being offered any. A
    //    counter that conflates "refused" with "never asked" cannot answer that, so the two are now separate. ──
    long floor_endpoint_returns = 0;// below-band returns the grid was OFFERED. floor_endpoint_returns == 0 means
                                  // the points are being filtered out UPSTREAM (the device_sweep defect); it is a
                                  // completely different fault from floor_endpoint_returns > 0 with
                                  // floor_endpoint_clears == 0, which means the support gate is refusing them.
    long floor_endpoint_blocked = 0;// below-band returns REFUSED by the support gate (cell's lowest evidence sits
                                  // above the floor ⇒ the thing is FLOATING and the beam merely passed under it).
                                  // Split out of miss_blocked_zaware, which now counts only the traverse gate —
                                  // the two were one counter and mean opposite things about the same cell.
    long bad_points = 0;          // returns dropped as non-finite before the sensor model saw them. NONZERO is
                                  // normal for the ZED (invalid depth); a nonzero count on a LiDAR-only cycle
                                  // means the sweep itself is corrupt.
    long cells_repaired = 0;      // cells whose log-odds had gone non-finite and were reset to unknown. Must be
                                  // 0 on a healthy run: anything else means a NaN is still reaching the model.
    long cells_unsupported = 0;   // cells released because EVERY height bin they claimed has been refuted —
                                  // "no lidar hits there". The removal channel once the support model is on.
    long clear_imprecise = 0;     // voxel clearings whose weight was cut because the sweep's own depth noise at
                                  // that range is worse than the reference sensor's. Large for a stereo camera
                                  // at range, ~0 for a lidar. Zero with the ZED feeding ⇒ the term is not wired.
    long clear_blind_shell = 0;   // voxel clearings REFUSED because they lay inside the sensor's dead shell —
                                  // the near zone it cannot return from. With the robot working close to
                                  // furniture this must be NONZERO; zero means the dead shell is not configured
                                  // and the map is deleting obstacles it is about to drive into.
    long clear_stopped = 0;       // voxel clearings skipped because they fell inside the no-clear run before the
                                  // hit. With a flat surface in view this must be NONZERO; zero means grazing
                                  // beams are raytracing straight through the surfaces they are about to strike.
    long bins_confirmed = 0;      // height bins a return confirmed this cycle
    long bins_refuted = 0;        // height bins a beam crossed without returning from. Its ratio to
                                  // bins_confirmed is how fast pollution is being cleaned out of the columns.
    long clear_blind = 0;         // see-throughs on never-hit cells whose weight the sensor envelope cut (the
                                  // near-field cone the device is physically unable to look into). ZERO while a
                                  // sensor envelope is configured means the term is not reaching the blind zone.
    long clear_surface_damped = 0;// see-throughs refused because one continuous surface explained both the skim
                                  // and the return (p_surface > 0.5). With a table in the room this must be
                                  // NONZERO; a zero here means grazing beams are still erasing flat surfaces.
    long clear_damped = 0;        // see-throughs whose weight the soft blocking probability actually reduced
                                  // (p_block < 0.9). ZERO here with a live table in the room means the term is
                                  // not engaging and grazing beams are still deleting surfaces at full weight.
    double clear_p_sum = 0.0;     // Sum of p_block over every admitted see-through on a cell that believes it
                                  // holds material. clear_p_sum / (misses on such cells) is the mean fraction of
                                  // a refutation those beams were actually worth.
    long clear_on_material = 0;   // ...the denominator: admitted see-throughs on cells that have a z-band
    long marks_suppressed = 0;    // returns whose ray was traced but whose endpoint was not allowed to mark
                                  // (mark_mask==0). 0 while the ZED feed is on ⇒ the mask is not plumbed through.
    long cells_decayed = 0;       // unobserved cells the forgetting term actually touched this cycle
    double decay_weight_sum = 0.0;// Σ w(r) over the cells decayed this cycle. decay_weight_sum/cells_decayed is
                                  // the MEAN precision of the observations that failed to confirm — well below 1
                                  // means the decaying population is mostly far away and is now eroding slowly,
                                  // which is the whole point of forget_range_weighted.
    long cells_unseen  = 0;       // unobserved cells SPARED because no beam reached them at all this cycle
                                  // (occluded / out of range / behind the robot) — forget_visible_only
    long cells_zheld   = 0;       // occupied cells SPARED because every beam that reached their column this
                                  // cycle passed OUTSIDE their z-band (over or under the obstacle). Those beams
                                  // are already counted in miss_blocked_zaware as carrying no free evidence; a
                                  // beam cannot be uninformative for the latch and informative for the decay at
                                  // the same time. Split out of cells_decayed 2026-08-22 — while they shared one
                                  // counter, "we looked and it was not there" and "we looked straight past it"
                                  // were indistinguishable, and this grid is 94% the second one.
    long cells_held    = 0;       // unobserved cells it SPARED because they carry free/no evidence
                                  // (forget_occupied_only). decayed+held = the whole unobserved population, so
                                  // the ratio says how much of the map the old symmetric rule was un-learning.
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

    // PER-DEVICE nav band. Two lidars do not see the floor the same way: bpearl is a downward dome measuring it
    // head-on (reads it within ~12 cm), helios is upright at ~1.1 m and only ever GRAZES it, landing 13-17 cm high.
    // The worker used to handle that by DELETING each device's near-floor returns before integration
    // (SpecificWorker::device_sweep). That deletion also threw away the return's CLEARING meaning: a beam that
    // reached the floor inside a cell proves nothing was STANDING there, and mark_floor_endpoint_flag exists
    // precisely to bank that — but it can only fire on a return the grid actually receives. Measured 2026-08-19:
    // `floor_clears` was 0 on 9381 of 9381 cycles, i.e. the mechanism had never run once in the live pipeline,
    // while the self-test (which feeds raw floor returns straight in) passed the whole time.
    // This is the same separation costmap_2d draws with min/max_obstacle_height: those gate MARKING, and the beam
    // is still raytraced for CLEARING. So hand the device's band in here instead of pre-filtering its cloud.
    // z0 < 0 ⇒ unset ⇒ fall back to OccGridParams::floor_z0. Set per device before that device's integrate_sweep,
    // exactly like set_floor_plane; persists until changed.
    void set_device_floor_z0(float z0) { dev_floor_z0_ = z0; }
    // The DEAD SHELL of the device about to be integrated: inside this range it returns nothing, so its rays
    // carry no free evidence there. See OccGridParams::clear_stop_max_m's neighbour note above.
    void set_sensor_min_range(float r) { sensor_min_r_ = std::max(0.0f, r); }
    // The sweep's own range-noise model, sigma(r) = s0 + quad·r². Governs how much clearing authority it gets
    // and how far short of its endpoint it must stop. See OccGridParams::reference_sigma_m.
    void set_sensor_noise(float s0, float quad) { sens_s0_ = s0; sens_quad_ = quad; }
    // Tag the sweep about to be integrated, so a release can name the sensor that finished the cell.
    void set_sensor_id(std::uint8_t id) { sensor_id_ = id; }
    float device_floor_z0() const { return dev_floor_z0_ >= 0.0f ? dev_floor_z0_ : p_.floor_z0; }

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
    { self_x_ = x; self_y_ = y; self_r_ = radius_m; observer_valid_ = true; }

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
    //
    // `mark_mask` (optional, parallel to points_room) is the MARKING gate, and it is deliberately NOT the same
    // thing as a zero `hit_weight_scale`. Zero weight still calls mark_hit_flag, which sets shit_[cell] — that
    // takes hit PRECEDENCE (suppressing this cycle's clearing for the cell) and installs hit_/zmn_/zmx_, i.e. it
    // would create a brand-new z-band that blocks future clearing beams, on the strength of no evidence at all.
    // mark_mask[i]==0 instead means "trace this beam, but do not let its endpoint mark": exactly costmap_2d's
    // clearing-only observation. Use it for returns a model already owns (ZED floor/ceiling/wall points), which
    // were previously DELETED from the cloud — and deleting a return does not just lose a mark, it loses the whole
    // ray and every free cell along it. The dropped ones are the worst to lose: a grazing floor return and a wall
    // return are the LONGEST rays in the sweep, so they carried the most clearing. nullptr / wrong-size ⇒ every
    // return may mark (the previous behaviour). A masked-off return still clears its own cell when it is a floor
    // return, since that is clearing evidence, not marking.
    void integrate_sweep(const Eigen::Vector3f& origin, const std::vector<Eigen::Vector3f>& points_room,
                         bool begin_cycle = true, float reliability = 1.0f,
                         const std::vector<float>* hit_weight_scale = nullptr,
                         const std::vector<std::uint8_t>* mark_mask = nullptr);
    // Fold this cycle's accumulated per-cell hit/miss flags into the log-odds field: exactly one +l_hit or
    // −l_miss per cell (hit precedence), then update the hysteresis latch. Call once after all sensors' sweeps.
    // `dt_s` is the elapsed time since the previous commit and drives the FORGETTING term (see
    // OccGridParams::forget_half_life_s): cells observed by NEITHER a hit nor a miss this cycle relax toward the
    // prior. Pass 0 (the default) to disable forgetting entirely — the original never-forget behaviour, which is
    // what the self_test's older properties assume.
    void commit_cycle(float dt_s = 0.0f);
    void trace_release(std::size_t i, float lo_before, float clear_z, float w, std::uint8_t cause,
                       float prev_lo, float prev_hi, float prev_top);
    void update_bins(std::size_t i, float w_hit, float w_miss);   // per-height confirm / refute
    void clear_bins(std::size_t i);
    bool has_support(std::size_t i) const;
    bool voxel_has_material(int ix, int iy, int iz) const;   // does this voxel hold material?
    int  column_thickness_bins(int ix, int iy) const;        // plate (1-2) or wall (dozens)?                        // any height bin still holding material?

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
    void  cell_belief  (int ix, int iy, float& alpha, float& beta) const;  // the exported Beta, reconstructed
    float column_prob  (int i) const;                                      // P(occupied) from the voxel column
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
    // Every cell released by the cycle just committed, with the evidence that finished it. The instrument for
    // "the table's residual vanished": a release with a large age_cycles and a clear_z outside [zmn, zmx] — or a
    // range small enough that the sensor could not have seen that height at all — is a wrongful removal.
    const std::vector<ReleaseEvent>& last_releases() const { return releases_; }
    // Every cell the cycle just committed LATCHED, with what created it. See LatchEvent.
    const std::vector<LatchEvent>& last_latches() const { return latches_; }

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
    void mark_hit_voxel (int ix, int iy, int iz, float z, float w);   // a return landed IN this voxel
    void mark_free_voxel(int ix, int iy, int iz, float z, float w);   // a beam passed THROUGH this voxel
    // Height-support bin of z, clamped into [0, OccGridParams::Z_BINS). See OccGridParams::bin_span_m.
    int z_bin(float z) const
    { const float bw = p_.bin_span_m / OccGridParams::Z_BINS;
      return std::clamp(static_cast<int>(z / std::max(1e-6f, bw)), 0, OccGridParams::Z_BINS - 1); }           // accumulate a see-through + its weight
    // A beam that TERMINATED on the floor in this cell: free evidence gated on SUPPORT (is the cell's own lowest
    // evidence floor-standing, hence refuted, or floating, hence merely passed under?) rather than on z-overlap,
    // which a floor return can never satisfy. See the long note at the definition.
    std::vector<std::uint8_t> residual_mask(const CellExplained& explained) const;   // occupied ∧ ¬explained
    std::vector<std::uint8_t> dilate_mask(const std::vector<std::uint8_t>& m, int radius_cells) const;

    OccGridParams p_;
    float xmin_ = 0, ymin_ = 0, inv_cell_ = 0;
    float fp_a_ = 0, fp_b_ = 0, fp_c_ = 0;        // data-driven floor plane z=a·x+b·y+c (0 ⇒ fixed z=0 band)
    float fp_rms_ = 0;                            // that fit's own residual scatter (m) = σ of the floor component
    float dev_floor_z0_ = -1.0f;                  // per-device nav band (m); <0 ⇒ unset ⇒ use p_.floor_z0
    float self_x_ = 0, self_y_ = 0, self_r_ = 0;  // robot body envelope this cycle (room frame); r<=0 ⇒ term off
    bool  observer_valid_ = false;                // set_self_body has been called ⇒ self_x_/self_y_ are a real
                                                  // observer position (range-weighted decay needs it; without it
                                                  // the origin would masquerade as the robot and skew the range)
    int   w_ = 0, h_ = 0;
    std::vector<float>        lo_;                 // log-odds (drives the hard occupied() latch — unchanged)
    std::vector<float>        kobs_;              // accumulated OBSERVATION weight per cell (the confidence behind
                                                   // the projection's answer). Risk comes from the column; this
                                                   // carries only how much evidence stands behind it.              // Beta(α,β) posterior over occupancy probability (the belief field)
    std::vector<float>        zmn_, zmx_;          // per-cell hit z-band (zmx_ is a running MAX — trace + display)
    std::vector<float>        zstab_;             // per-voxel CONSISTENCY in [0,1] — how reliably this voxel's
                                                   // observations have agreed with its own belief. Scales the
                                                   // evidence capacity: see OccGridParams::stable_gain.
    std::vector<float>        zsup_;              // per-cell HEIGHT SUPPORT, Z_BINS per cell: the log-odds that
                                                   // material sits in that height bin. > 0 ⇒ material. This is the
                                                   // test the clearing weight is measured against; see bin_span_m.
    std::vector<std::uint64_t> shbits_;            // this cycle: bins a return landed in (per cell)
    std::vector<std::uint64_t> smbits_;            // this cycle: bins a beam crossed without returning from
    std::vector<float>        dispz_;              // per-cell CURRENT top height (EMA of this cycle's hits) for display
    std::vector<std::uint8_t> hit_;                // 1 once a cell has received any hit (z-band valid)
    std::vector<std::uint8_t> occ_;                // LATCHED occupied state (hysteresis: set at occ_set, clear at occ_clear)
    // Per-CYCLE scratch (reset on begin_cycle, folded once in commit_cycle) — the fix for miss over-counting.
    std::vector<std::uint8_t> shit_;               // this cycle: cell received an endpoint hit (any beam)
    std::vector<std::uint8_t> smiss_;              // this cycle: cell was seen-through (z-aware) by any beam
    std::vector<std::uint8_t> slook_;              // this cycle: a beam reached this cell AND could have observed
                                                   // its obstacle (it passed through the z-band, or the cell holds
                                                   // no band yet) — a real observation OPPORTUNITY. Gates the
                                                   // forgetting term. NOT set by a beam the z-gate rejected: see
                                                   // szblock_.
    std::vector<std::uint8_t> szblock_;            // this cycle: a beam reached this column but passed OUTSIDE the
                                                   // cell's z-band, so it observed the space over/under the
                                                   // obstacle and nothing about the obstacle itself
    std::vector<float>        shz_lo_, shz_hi_;    // this cycle: accumulated hit z-band for shit_ cells
    std::vector<float>        shit_w_;             // this cycle: MAX precision weight of the hits on this cell (range×motion)
    std::vector<float>        smiss_w_;            // this cycle: MAX precision weight of the see-throughs on this cell
    std::vector<std::uint8_t> shit_src_;            // which sweep delivered this cycle's hit (birth trace)
    std::vector<std::uint8_t> smiss_src_;           // which sweep set smiss_w_ (for the release trace)
    std::uint8_t              sensor_id_ = 0;      // the sweep currently being integrated
    float                     sensor_min_r_ = 0.0f; // ...and its dead shell (no returns closer than this)
    float                     sens_s0_ = 0.0f, sens_quad_ = 0.0f;   // its range-noise model (0 ⇒ the reference)
    float                     clear_r2_ = 0.0f;      // lidar clearance radius, squared (see lidar_clearance_m)
    std::vector<float>        smiss_z_;            // this cycle: height of the see-through beam that set smiss_w_
    std::vector<std::uint32_t> occ_since_;         // cycle index at which this cell latched (release age, for the trace)
    std::uint32_t             cycle_ = 0;          // committed-cycle counter (ages the release trace)
    std::vector<LatchEvent>   latches_;            // this cycle's births, with the reason for each
    std::vector<ReleaseEvent> releases_;           // this cycle's releases, with the reason for each
    // The BEST observable fraction with which this cell has ever been cleared — the union over sensors and over
    // time, since driving past from a new range genuinely does buy more. It bounds what clearing can establish
    // here: see OccGridParams::sensor_min_range_m. 0 until the first see-through; 1 when no envelope is declared.
    std::vector<float>        seenf_;
    SweepDiag sd_;                                 // counters for the cycle just committed
};

}  // namespace rc
