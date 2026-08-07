#pragma once

/*
 * controller_lockon.h
 *
 * Generic "servo" control primitive for the affordance executor (controller). After the robot
 * reaches an affordance pose, this runs a quasi-static MOVE-STOP-MEASURE micro-search to satisfy the
 * affordance's completion predicate — e.g. nudging the base so an intermittent YOLO detection fires.
 *
 * It is OBJECT-AGNOSTIC: it consumes a normalised Reading {valid, err_x, err_y, scalar} that the
 * session fills from the contract-named feedback attributes, and a `goal_met` boolean the session
 * computes from the contract's completion predicate. The primitive owns only the rhythm and the
 * corrective law:
 *   SETTLE (hold still so the image de-blurs / detection settles) → at each measurement, if the goal
 *   has held for `stable_needed` measurements → LOCKED; else choose a nudge — centre via yaw
 *   (err_x), reach the scalar target via advance, else a small yaw dither to escape a dead-spot —
 *   → STEP (apply briefly) → SETTLE. Times out to GIVE_UP so the robot never orbits forever.
 *
 * "HOW" (gains/caps/timing) comes from ControllerParams via LockOnConfig; "WHAT/WHEN"
 * (scalar_target, stable_needed, timeout) is per-affordance and passed to begin().
 */

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace rc
{

struct LockOnConfig
{
    // Distance sweep is the PRIMARY search: detectability depends mostly on stand-off distance, not
    // heading, so we oscillate the base fwd/back over a range and measure, with yaw only as a light
    // continuous centring correction.
    float sweep_speed_mps = 0.12f;   // advance speed while sweeping distance (primary axis)
    float sweep_range_m   = 0.45f;   // oscillate ± this far from the arrival pose
    float k_yaw           = 0.8f;    // rps per unit horizontal error (secondary centring)
    float max_yaw_rps     = 0.12f;
    float dither_yaw_rps  = 0.10f;   // reacquire (yaw) when feedback is invalid
    float settle_ms       = 400.0f;  // MINIMUM dwell before each measurement (see settle_max_ms)
    float step_ms         = 400.0f;  // duration of one sweep step
    float offset_tol      = 0.15f;   // |err_x| deadband for centring (loose; distance dominates)
    int   max_attempts    = 30;
    // ── HOW LONG A "MEASUREMENT" REALLY TAKES ─────────────────────────────────────────────────────
    // settle_ms alone was a GUESS at the perception pipeline's latency, and the guess decided whether
    // the loop could see anything at all: camera → YOLO → mask node → concept agent → feedback attribute
    // is easily longer than one settle, so the predicate sampled at the end of the dwell could still be
    // describing a frame captured during the PRECEDING nudge — while the base was moving and the mask
    // was smeared. The loop then moved again, and a good detection arriving 200 ms later was never
    // sampled at all. That is the "the mask is visible but the loop cannot capture it" failure.
    // The cure is to stop guessing the latency and WAIT FOR THE EVIDENCE: hold still until the mask
    // producer has published `settle_new_frames` frames since the base stopped, so the measurement is
    // taken on looks the robot actually held still for. Counted in FRAMES, not milliseconds, because a
    // frame counter needs no clock agreement between two agents — see ControllerSession::masks_frame_seq.
    float settle_max_ms      = 2500.0f;  // bound: never wait forever on a dead producer
    int   settle_new_frames  = 2;        // post-stop frames to wait for (0 = old fixed-dwell behaviour)
};

class LockOn
{
public:
    enum class Phase { Idle, Settle, Step, Locked, GiveUp };

    // Normalised servo inputs (filled from the contract-named feedback attributes).
    struct Reading
    {
        bool  valid  = false;   // feedback currently usable (e.g. model projects in front of camera)
        float err_x  = 0.0f;    // horizontal error to null ([-1,1], 0 = centred) → yaw
        float err_y  = 0.0f;    // secondary error (reserved; body-cam pitch is not base-controllable)
        float scalar = 0.0f;    // scalar to drive toward scalar_target → advance
    };

    struct Command { float adv = 0.0f, side = 0.0f, rot = 0.0f; };

    void configure(const LockOnConfig& c) { cfg_ = c; }

    // WHAT/WHEN from the affordance contract. (scalar_target is unused by the distance sweep.)
    void begin(std::uint64_t t_ms, float /*scalar_target*/, int stable_needed, float timeout_ms)
    {
        phase_ = Phase::Settle;
        phase_start_ms_ = t_ms; begin_ms_ = t_ms;
        stable_needed_ = std::max(1, stable_needed);
        timeout_ms_ = timeout_ms;
        stable_ = 0; attempts_ = 0; dither_sign_ = 1; cur_ = {};
        sweep_offset_ = 0.0f; sweep_dir_ = 1;
        settle_seq_at_stop_ = -1; settle_hit_ = false;
    }
    void reset() { phase_ = Phase::Idle; cur_ = {}; }

    Phase phase()  const { return phase_; }
    bool  active() const { return phase_ == Phase::Settle || phase_ == Phase::Step; }
    bool  done()   const { return phase_ == Phase::Locked || phase_ == Phase::GiveUp; }
    bool  locked() const { return phase_ == Phase::Locked; }
    // How many CONSECUTIVE cycles the contract predicate has held. Read-only, for the affordance panel:
    // a predicate that flickers true is not one that HOLDS, and without this the difference is invisible.
    int   stable() const { return stable_; }
    // What it is counting TOWARD. Exposed because the panel used to read the requirement from a
    // SECOND resolution of the contract (target_contract_, resolved at selection) while the loop
    // counted against a different one (active_contract_, resolved at arrival). Two readings of one
    // contract are two chances to disagree, and when they did the row read "1 of 1" while the loop
    // was waiting for 3 — a row that says the goal is met beside a loop that says it is not.
    int   stable_needed() const { return stable_needed_; }

    // Advance one control cycle. `goal_met` = the contract predicate holds THIS cycle (session-evaluated).
    // `evidence_seq` = the perception producer's monotonic frame counter (mask_frame_id), or <0 when it
    // is unavailable — in which case the settle falls back to the plain settle_ms dwell.
    Command update(const Reading& r, bool goal_met, std::uint64_t t_ms, int evidence_seq = -1)
    {
        const float since = static_cast<float>(t_ms - phase_start_ms_);
        if (static_cast<float>(t_ms - begin_ms_) > timeout_ms_ || attempts_ > cfg_.max_attempts)
        {
            phase_ = Phase::GiveUp;
            return {};
        }

        if (phase_ == Phase::Settle)
        {
            // ★A MEASUREMENT IS THE WHOLE STILL WINDOW, NOT THE INSTANT IT ENDS. The predicate is
            // sampled EVERY cycle the robot is holding still and latched, so a detection that fires at
            // any point during the dwell counts. Sampling only at the end asked the perception stack to
            // be true at one nominated millisecond; an intermittent YOLO mask is exactly the thing that
            // will not oblige, and the loop was throwing away sightings it had already been given.
            // Nothing false can latch early: the session's predicate already ANDs in robot_still(), and
            // the base-speed EMA is still decaying just after the stop.
            if (goal_met)
                settle_hit_ = true;
            if (not settle_ready(since, evidence_seq))
                return {};                       // hold still: the look is not finished yet
            if (settle_hit_)
            {
                if (++stable_ >= stable_needed_) { phase_ = Phase::Locked; return {}; }
            }
            else
                stable_ = 0;
            cur_ = decide_step(r);
            ++attempts_;
            phase_ = Phase::Step; phase_start_ms_ = t_ms;
            return cur_;
        }

        if (phase_ == Phase::Step)
        {
            if (since < cfg_.step_ms)
                return cur_;                     // keep nudging
            // Stop and settle. The evidence counter is stamped HERE, at the moment the nudge ends, so
            // "frames since the base stopped" means what it says.
            phase_ = Phase::Settle; phase_start_ms_ = t_ms;
            settle_seq_at_stop_ = evidence_seq;
            settle_hit_ = false;
            return {};
        }
        return {};
    }

private:
    // Is this still-window's look finished? The minimum dwell must have passed AND the producer must
    // have delivered `settle_new_frames` frames since the base stopped — bounded by settle_max_ms so a
    // dead or stalled producer degrades to the old fixed dwell instead of freezing the search.
    bool settle_ready(float since, int evidence_seq) const
    {
        if (since >= cfg_.settle_max_ms) return true;      // bound first: it outranks every other wait
        if (since < cfg_.settle_ms)      return false;
        if (cfg_.settle_new_frames <= 0) return true;      // feature off
        if (evidence_seq < 0 or settle_seq_at_stop_ < 0) return true;   // no counter ⇒ cannot wait on it
        return evidence_seq - settle_seq_at_stop_ >= cfg_.settle_new_frames;
    }

    Command decide_step(const Reading& r)
    {
        Command c;
        if (!r.valid)   // feedback unusable → small alternating yaw sweep to reacquire
        {
            c.rot = cfg_.dither_yaw_rps * static_cast<float>(dither_sign_);
            dither_sign_ = -dither_sign_;
            return c;
        }
        // PRIMARY: distance sweep — detectability depends mostly on stand-off distance, so oscillate
        // the base forward/back over ±sweep_range and measure at each step.
        c.adv = cfg_.sweep_speed_mps * static_cast<float>(sweep_dir_);
        sweep_offset_ += cfg_.sweep_speed_mps * (cfg_.step_ms / 1000.0f) * static_cast<float>(sweep_dir_);
        if (sweep_dir_ > 0 && sweep_offset_ >=  cfg_.sweep_range_m) sweep_dir_ = -1;
        if (sweep_dir_ < 0 && sweep_offset_ <= -cfg_.sweep_range_m) sweep_dir_ = +1;
        // SECONDARY: keep the table roughly centred while sweeping (omni base: yaw + advance together).
        if (std::abs(r.err_x) > cfg_.offset_tol)
            c.rot = std::clamp(-cfg_.k_yaw * r.err_x, -cfg_.max_yaw_rps, cfg_.max_yaw_rps);
        return c;
    }

    LockOnConfig  cfg_;
    Phase         phase_ = Phase::Idle;
    std::uint64_t phase_start_ms_ = 0, begin_ms_ = 0;
    float         timeout_ms_ = 10000.0f;
    float         sweep_offset_ = 0.0f;   // accumulated (open-loop) advance from the start pose
    int           stable_ = 0, stable_needed_ = 3, attempts_ = 0, dither_sign_ = 1, sweep_dir_ = 1;
    int           settle_seq_at_stop_ = -1;   // producer frame counter when this still-window began
    bool          settle_hit_ = false;        // the predicate held at some point during this window
    Command       cur_;
};

}  // namespace rc
