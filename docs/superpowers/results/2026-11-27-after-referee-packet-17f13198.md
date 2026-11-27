# Blind referee packet 2: rows of the after-count at the graded-binary tree 17f13198

Source: active_inference commit 17f13198 (tree hash `5572d3e8d46cebde6a5d52804e30a9e067a3ea7b`), the code the graded batch was built from; the batch commit 91598c88 is docs-only on top of it. The earlier after-count (cf65a471, tree `5c0029ab522691fb4747d2e7215493162f13f0b0`) stays recorded as an intermediate count; its pass-2 labels (A0001-A2588) stand for every row that is unchanged between the two trees. This packet holds only the rows that need a label: new rows, rows whose reading statement, consumers or enclosing function changed since cf65a471, and baseline-matched rows that stopped matching. Ids continue from A2589.

No label, no proposed label. The role column quotes the code's own comment or the loader text, not a reading. "Why in this packet" says which test sent the row here.

## Definition (confirmed by the owner, applied literally)

A PATCH constant is any constant that is neither class knowledge nor a property of a sensor. That means any precision, gate, anchor, threshold, gain, margin, floor, clamp or reference/scale whose role is to correct the estimator's behaviour.

- CLASS knowledge: a prior on the class's size, shape, placement (e.g. stands against a wall), appearance, or persistence/permanence.
- SENSOR: a property of a sensor or its mount: noise σ, range, field of view, depth-noise model, detection probability of the detector, intrinsics.
- Owner's examples of PATCH (nightstand config): AI2DepthUnobsPrecision, AI2TopNoFloatPrecision / AI2TopNoFloatMargin, AI2TopOversegSigmaPerM, FeRef / FeScale, PlausHeightPriorGain.

## Exclusion rule

NOT counted at all: plumbing and engineering parameters with no estimation role. Examples: agent id, DDS domain, node names, file paths, log/dump/diagnostic switches (GnTrace, *Log*, *Dump*), period/Hz of the compute loop, thread counts, display/visualisation-only settings, GN iteration caps and numerical-solver tolerances. They are listed in each section with their names (label `excluded`) so a referee can audit them.


## Labels

`sensor` `class` `patch` `excluded`; `switch (patch)` = boolean that enables a patch mechanism; `switch (other)` = boolean with no patch mechanism behind it (publication, channel choice, history stage). A trailing `?` = ambiguous (counted in the "ambiguous" column only, never twice).

## Referee's rules

Rules 0-8 (header of referee_labels_2026-11-27.tsv, verbatim), then rules 9-13 (header of after_referee_labels_2026-11-27.tsv, verbatim).

```
# Rules applied to the hard cases
# 0. A row is labelled by the meaning of its key when the mechanism is on, not by its present value. A value of 0 that
#    only disables the mechanism does not change the label. A constant that no code reads at all is 'excluded' (dead).
# 1. Process noise. Class only where the code states an object fact with a value that follows from it (extent Q = 0:
#    'dimensions are constants of the object'). The per-frame pose Q (0.005 m, 0.01 rad, identical in every class) is
#    patch: the code gives it a filter role (tracks localisation jitter; rate at which evidence is discarded).
# 2. Sensor noise. Sensor when the code declares it as the measurement noise of a channel and gives no other purpose.
#    Patch when the code says it is enlarged on purpose, covers a model or pose error, or is a floor.
# 3. Sensor precision used as a weight. Patch when the device sigma exists as a separate constant and the precision is
#    combined with it or matched to another term (LidarPrecision). Sensor only when it is the device's 1/sigma^2 itself.
# 4. Prior stds. Class when they parameterise a prior over size, shape, placement, appearance or persistence in
#    physical units. Patch when the code says the prior only breaks a degeneracy or acts as a covariance ceiling,
#    or when a class fact is multiplied by a free gain (owner's example PlausHeightPriorGain).
# 5. Arbitration, planner, calibration. Gains, thresholds, margins, hysteresis and targets that shape decisions are
#    patch. Excluded: log-only scales, sample counts, numeric guards, write dedup, solver schedules, robot-body sizes.
# 6. Floors and clamps on an estimated state are patch even where the code calls them physical floors. Guards on pure
#    numerics (projection singularity, exact-zero cut-offs, float-precision culls) are excluded.
# 7. A constant in a class-size or class-shape slot is class even when its value was copied from another class
#    (low confidence).
# 8. Room-model and lexical knowledge (wall sigma, floor sigma, colour vocabulary) is neither class nor sensor; the
#    first sentence of the definition makes it patch (low confidence).
# Blind referee labels, second pass: 286 rows of the after-count. Code read at commit cf65a471 (tree 5c0029ab), source only.
# Rules 0-8 are those of the first pass, unchanged (see referee_labels.tsv). New rules, this pass only:
# 9.  A literal that is an array index, dimension, loop bound, latent-layout offset, container size, hash constant,
#     format version, unit conversion or a mathematical identity (1/sqrt(12) of a uniform, 1/sqrt(2) of the normal CDF,
#     1/4 from z = H/2) is excluded: it has no free value.
# 10. Booleans. A dump or diagnostic switch is excluded. A boolean that turns on a publication, a channel or a feature
#     is 'switch (other)'. 'switch (patch)' only when what it turns on is itself a corrective mechanism.
# 11. One constant listed at several sites (toml key, loader default literal, struct default) gets one label at all sites.
# 12. 'moved or removed?' rows: 'removed' only when no code in the after tree carries the constant. If it is still
#     there, the reason gives the after location and the row is labelled by rules 0-11.
# 13. A solver tolerance is excluded for its stop-rule role; where the code itself flags that the same number decides
#     whether information enters the estimate, that role is labelled patch on the row that names it.
```

## Rows to label

Columns: id | key or constant | value | location (file:line at 17f13198) | why in this packet | role (quoted from the code)


### common/ai_belief/persistent_engine.h

| A2590 | literal | 9.0, 3.0 | common/ai_belief/persistent_engine.h:279 | new at 17f13198 | (no comment in the code) |
| A2599 | Σ capped at the class prior on every predict | Σ_ii ≤ Σ₀,ii | common/ai_belief/persistent_engine.h:1514 | changed since cf65a471: enclosing function differs (was A2580) | **changed since cf65a471** (the enclosing function differs between cf65a471 and 17f13198; before/after text below). Code comment: (none) |

<details><summary>A2599: context changed, function `void learn_volatility(const M& m, const VecNf& q_used, const VecNd& theta_pred, const MatJd& P_bar, const VecNd& post_var, const MatJd& Lambda_data, const VecXd` (11 lines before, 49 after)</summary>

Evidence line (before, common/ai_belief/persistent_engine.h:1514 (cf65a471)): `for (int i = 0; i < N; ++i)`; found unchanged at common/ai_belief/persistent_engine.h:1514 (17f13198).

```
BEFORE (from line 1511)
    static void predict_theta_on(MatJd& P_, const VecNf& q, const VecNf& p0, float q_scale)
AFTER (from line 1482)
    void learn_volatility(const M& m, const VecNf& q_used, const VecNd& theta_pred, const MatJd& P_bar, const VecNd& post_var,
                          const MatJd& Lambda_data, const VecXd& eta_data, const VecNd& theta_post)
```
```
BEFORE (from line 1513)
        for (int i = 0; i < N; ++i) P_(i, i) += double(std::max(0.0f, q_scale)) * double(q(i));
AFTER (from line 1485)
        const VolatilityConfig vc = volatility_of(m);
        if (not vc.infer) return;
        const VecNd q0 = m.process_noise_diag().template cast<double>();
        if (not vol_init_)
        {
            vol_omega_ = q0.array().log().matrix();
            vol_s_.setZero();
            vol_n_ = 0.0;
            vol_init_ = true;
        }
        const double n0 = 1.0 / (vc.sigma_omega * vc.sigma_omega);
        const bool jumps = static_cast<int>(vc.jump_var.size()) == N;
        const double k_steps = std::max(1.0, steps_since_update_);      // the update's own predict is always one
        // P(at least one move in the interval) = 1 − (1 − π)^k
        const double pi_int = jumps ? -std::expm1(k_steps * std::log1p(-std::min(vc.jump_prob, 1.0))) : 0.0;
        // Spike-and-slab, JOINT over θ (one move moves the object as a whole): the data of this update as a factor (Λ, η) on the
        // joint x at x̄; "stayed" x ~ N(x̄, P̄), "moved" adds v ~ N(0, M) on θ (M = diag jump_var). p = P(moved | data) by Bayes with
        // prior π_int (detail::jump_evidence, information form); per component E[w_i²] = p·E[v_i² | moved, data], ÷ k per step.
        VecNd ew2_jump = VecNd::Zero();
        if (jumps)
        {
            Eigen::Matrix<double, N, N> Mj = Eigen::Matrix<double, N, N>::Zero();
            for (int i = 0; i < N; ++i) Mj(i, i) = vc.jump_var[std::size_t(i)];
            // the Bayes factor in information form on the joint x (detail::jump_evidence, round 7 A2)
            const auto je = detail::jump_evidence<N, N + K>(Lambda_data, eta_data, P_bar, VecNd(Mj.diagonal()), pi_int);
            for (int i = 0; i < N; ++i) ew2_jump(i) = je.p * (je.v_mean(i) * je.v_mean(i) + je.v_cov(i, i)) / k_steps;
            last_jump_log_bf_ = je.log_bf; last_jump_p_ = je.p;
        }
        vol_n_ = vc.memory * vol_n_ + 1.0;
```
```
BEFORE (from line 1515)
            if (p0(i) > 0.0f and P_(i, i) > double(p0(i)))
AFTER (from line 1515)

```
```
BEFORE (from line 1517)
                const double sc = std::sqrt(double(p0(i)) / P_(i, i));   // symmetric rescale keeps P PSD
                P_.row(i) *= sc;
                P_.col(i) *= sc;
AFTER (from line 1516)
            double dth = theta_post(i) - theta_pred(i);
            if (i == kYaw) dth = std::remainder(dth, 2.0 * M_PI);
            const double q = double(q_used(i)), sbar = P_bar(i, i), spost = post_var(i);
            double ew2;
            if (not jumps)
            {   // (unreachable for the countable classes: CountableEngineModel::volatility() always sets jump_var, so `jumps` holds)
                // Gaussian random walk (item 2): E[w²] = q + (q/Σ̄)²(δθ² − (Σ̄ − Σ⁺))
                const double k = q / sbar;
                ew2 = q + k * k * (dth * dth - (sbar - spost));
            }
            else ew2 = ew2_jump(i);
            vol_s_(i) = vc.memory * vol_s_(i) + ew2;
            vol_omega_(i) = std::log((n0 * q0(i) + vol_s_(i)) / (n0 + vol_n_));
```
</details>


### countable_concept/src/countable_birth_footprint.h

| A2589 | birth footprint percentiles | 0.05 / 0.95 / 0.9 | countable_concept/src/countable_birth_footprint.h:50 | changed since cf65a471: enclosing function differs (was A2279) | **changed since cf65a471** (the enclosing function differs between cf65a471 and 17f13198; before/after text below). Code comment: (none) |

<details><summary>A2589: context changed, function `inline BirthFootprint birth_footprint(std::span<const Eigen::Vector2f> xy, const Eigen::Vector2f& wall_p, const Eigen::Vector2f& wall_n, float class_w, float cl` (32 lines before, 33 after)</summary>

Evidence line (before, countable_concept/src/countable_birth_footprint.h:49 (cf65a471)): `const float a0 = pct(a, 0.05f), a1 = pct(a, 0.95f), b0 = pct(b, 0.05f), b1 = pct(b, 0.95f);`; found unchanged at countable_concept/src/countable_birth_footprint.h:50 (17f13198).

```
BEFORE (from line 56)

AFTER (from line 57)
    out.mid_n = 0.5f * (b0 + b1);
```
</details>


### countable_concept/src/countable_birth_seed.h

| A2591 | literal | 10 | countable_concept/src/countable_birth_seed.h:47 | changed since cf65a471: enclosing function differs (was A2530) | **changed since cf65a471** (the enclosing function differs between cf65a471 and 17f13198; before/after text below). Code comment: (none) |
| A2592 | literal | 12.0 | countable_concept/src/countable_birth_seed.h:105 | new at 17f13198 | (no comment in the code) |

<details><summary>A2591: context changed, function `inline std::optional<WallSeed> wall_seed(const std::vector<Eigen::Vector3f>& pts, float z_lo, float z_hi, const CountableConfig& cfg, const NearestWall& nearest` (23 lines before, 16 after)</summary>

Evidence line (before, countable_concept/src/countable_birth_seed.h:44 (cf65a471)): `if (xy.size() < 10) return std::nullopt;`; found unchanged at countable_concept/src/countable_birth_seed.h:47 (17f13198).

```
BEFORE (from line 38)
inline std::optional<Eigen::Matrix2f> birth_on_priors(CountableBeliefState& s0, const std::vector<Eigen::Vector3f>& pts, float z_lo,
                                                      float z_hi, const CountableConfig& cfg, const NearestWall& nearest_wall,
                                                      const std::string& name)
AFTER (from line 42)
inline std::optional<WallSeed> wall_seed(const std::vector<Eigen::Vector3f>& pts, float z_lo, float z_hi, const CountableConfig& cfg,
                                         const NearestWall& nearest_wall)
```
```
BEFORE (from line 53)
    s0.w = bf.w; s0.h = bf.h; s0.yaw = bf.yaw;
    s0.cx = bf.centre.x(); s0.cy = bf.centre.y();
    std::print("[{}] birth on priors: wall n=({:.2f},{:.2f}) → yaw={:.1f}° w={:.2f} h={:.2f} (class means) centre ({:.2f},{:.2f}) "
               "[cloud centre ({:.2f},{:.2f}), extent along {:.2f} across {:.2f} → extra pos std along {:.3f} across {:.3f}]\n",
               name, w.n.x(), w.n.y(), s0.yaw * 57.29578f, s0.w, s0.h, s0.cx, s0.cy, c.x(), c.y(), bf.extent_t, bf.extent_n,
               std::sqrt(bf.var_t), std::sqrt(bf.var_n));
    return bf.pos_cov;
AFTER (from line 56)
    return WallSeed{bf, w, c};
```
</details>


### countable_concept/src/countable_config.cpp

| A2593 | literal | 6 | countable_concept/src/countable_config.cpp:211 | changed since cf65a471: enclosing function differs (was A2532) | **changed since cf65a471** (the enclosing function differs between cf65a471 and 17f13198; before/after text below). Code comment: (none) |
| A2594 | literal | 0.10f | countable_concept/src/countable_config.cpp:431 | changed since cf65a471: enclosing function differs (was A2533) | **changed since cf65a471** (the enclosing function differs between cf65a471 and 17f13198; before/after text below). Code comment: (none) |
| A2595 | literal | 0.10f | countable_concept/src/countable_config.cpp:432 | changed since cf65a471: enclosing function differs (was A2534) | **changed since cf65a471** (the enclosing function differs between cf65a471 and 17f13198; before/after text below). Code comment: (none) |
| A2596 | literal | 0.10f | countable_concept/src/countable_config.cpp:433 | changed since cf65a471: enclosing function differs (was A2535) | **changed since cf65a471** (the enclosing function differs between cf65a471 and 17f13198; before/after text below). Code comment: (none) |
| A2597 | literal | 0.15f | countable_concept/src/countable_config.cpp:434 | changed since cf65a471: enclosing function differs (was A2536) | **changed since cf65a471** (the enclosing function differs between cf65a471 and 17f13198; before/after text below). Code comment: (none) |

<details><summary>A2593: context changed, function `CountableConfig load_countable_config(const ConfigLoader& cfg)` (602 lines before, 606 after)</summary>

Evidence line (before, countable_concept/src/countable_config.cpp:207 (cf65a471)): `const auto ls = rc::manifest::layer_spread(manifest_path, 6);`; found unchanged at countable_concept/src/countable_config.cpp:211 (17f13198).

```
BEFORE (from line 200)

AFTER (from line 200)
        out.placement_p_against_wall = rc::manifest::placement_p_against_wall(manifest_path);
        if (out.placement_p_against_wall)
            std::print("[manifest] {} placement mixture: P(against a wall) = {:.3f} — the birth seed is the mixture's MAP\n",
                       out.concept_class, *out.placement_p_against_wall);
```
</details>


<details><summary>A2594: context changed, function `CountableConfig load_countable_config(const ConfigLoader& cfg)` (602 lines before, 606 after)</summary>

Evidence line (before, countable_concept/src/countable_config.cpp:427 (cf65a471)): `out.link_pi_move = getf("", 0.10f, "");`; found unchanged at countable_concept/src/countable_config.cpp:431 (17f13198).

```
BEFORE (from line 200)

AFTER (from line 200)
        out.placement_p_against_wall = rc::manifest::placement_p_against_wall(manifest_path);
        if (out.placement_p_against_wall)
            std::print("[manifest] {} placement mixture: P(against a wall) = {:.3f} — the birth seed is the mixture's MAP\n",
                       out.concept_class, *out.placement_p_against_wall);
```
</details>


<details><summary>A2595: context changed, function `CountableConfig load_countable_config(const ConfigLoader& cfg)` (602 lines before, 606 after)</summary>

Evidence line (before, countable_concept/src/countable_config.cpp:428 (cf65a471)): `out.link_pi_gone = getf("", 0.10f, "");`; found unchanged at countable_concept/src/countable_config.cpp:432 (17f13198).

```
BEFORE (from line 200)

AFTER (from line 200)
        out.placement_p_against_wall = rc::manifest::placement_p_against_wall(manifest_path);
        if (out.placement_p_against_wall)
            std::print("[manifest] {} placement mixture: P(against a wall) = {:.3f} — the birth seed is the mixture's MAP\n",
                       out.concept_class, *out.placement_p_against_wall);
```
</details>


<details><summary>A2596: context changed, function `CountableConfig load_countable_config(const ConfigLoader& cfg)` (602 lines before, 606 after)</summary>

Evidence line (before, countable_concept/src/countable_config.cpp:429 (cf65a471)): `out.link_lambda_new = getf("", 0.10f, "");`; found unchanged at countable_concept/src/countable_config.cpp:433 (17f13198).

```
BEFORE (from line 200)

AFTER (from line 200)
        out.placement_p_against_wall = rc::manifest::placement_p_against_wall(manifest_path);
        if (out.placement_p_against_wall)
            std::print("[manifest] {} placement mixture: P(against a wall) = {:.3f} — the birth seed is the mixture's MAP\n",
                       out.concept_class, *out.placement_p_against_wall);
```
</details>


<details><summary>A2597: context changed, function `CountableConfig load_countable_config(const ConfigLoader& cfg)` (602 lines before, 606 after)</summary>

Evidence line (before, countable_concept/src/countable_config.cpp:430 (cf65a471)): `out.ghost_evidence_gain = getf("", 0.15f,`; found unchanged at countable_concept/src/countable_config.cpp:434 (17f13198).

```
BEFORE (from line 200)

AFTER (from line 200)
        out.placement_p_against_wall = rc::manifest::placement_p_against_wall(manifest_path);
        if (out.placement_p_against_wall)
            std::print("[manifest] {} placement mixture: P(against a wall) = {:.3f} — the birth seed is the mixture's MAP\n",
                       out.concept_class, *out.placement_p_against_wall);
```
</details>


### countable_concept/src/specificworker.cpp

| A2598 | literal | 1000003ull | countable_concept/src/specificworker.cpp:1034 | changed since cf65a471: enclosing function differs (was A2564) | **changed since cf65a471** (the enclosing function differs between cf65a471 and 17f13198; before/after text below). Code comment: (none) |

<details><summary>A2598: context changed, function `void SpecificWorker::step_epistemic(rc::CountableInstance& inst, DSR::Node& node)` (55 lines before, 55 after)</summary>

Evidence line (before, countable_concept/src/specificworker.cpp:1034 (cf65a471)): `occupied_stamp_cells_ * 1000003ull ^ occupied_stamp_prob_,`; found unchanged at countable_concept/src/specificworker.cpp:1034 (17f13198).

```
BEFORE (from line 1054)
        // G5: the promise, beside the delivery the fitter logs per update (persistent path; offers.csv)
AFTER (from line 1054)
        // G5: the promise, beside the delivery the fitter logs per update (persistent path; offers.csv), and its p_stand (F7)
```
```
BEFORE (from line 1056)
                           published.epistemic_target_yaw_rad, published.promised_nats);
AFTER (from line 1056)
                           published.epistemic_target_yaw_rad, published.promised_nats, published.p_stand);
```
</details>

