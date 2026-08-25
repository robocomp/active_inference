#pragma once
// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE CALIBRATION CHANNEL — room's second affordance, and the arithmetic that decides whether to
// offer it at all.
//
// Two things live here, and they are the two halves of one idea:
//
//   1. THE PASSIVE OBSERVER. The robot drives all day serving standpoints, and every one of those
//      tours turns. The estimator does not care WHY the robot moved, so ordinary exploration is free
//      calibration data. This feeds it, every cycle, whatever the robot happens to be doing.
//
//   2. THE DELIBERATE MANOEUVRE, offered only for what it adds BEYOND that free data. This is the
//      whole reason the gain is computed marginally rather than absolutely: a robot whose day is full
//      of turning already knows its rotation scale, and asking it to stop and spin buys nothing. The
//      offer therefore extinguishes itself — not on a timer and not on a counter, but because the
//      quantity it advertises really does fall to zero as the posterior sharpens.
//
// ★THE TRIGGER IS NOT A THRESHOLD ON THE POSTERIOR WIDTH. There is no `if (s_std > k)` here. The
// offer carries its expected information gain in nats and competes in the same EFE arbitration as
// every exploration standpoint; if a cell is worth more, the cell wins and the pivot never runs. That
// is the only "trigger" — the robot feels miscalibrated exactly when knowing its own motion model
// better is worth more than the next thing it could look at.
//
// ★WHY THE PASSIVE RATE IS MEASURED AND NOT ASSUMED. The marginal gain needs to know how much turning
// the ordinary tours deliver per second, and that is a property of THIS robot's day — a lap of a
// corridor and a shuffle between two nearby standpoints are not the same diet. Measuring it costs one
// EMA over a quantity already being accumulated, and assuming it would put a guessed number in the
// one place the whole design claims not to have one.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#include "calib_pivot.h"
#include "../../common/affordance_protocol/affordance_goal_parse.h"   // rc::affordance::Outcome — the PURE half; the full protocol header pulls in DSR/Qt
#include "../../common/motion_calib/scale_estimator.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace rc::calib
{

struct CalibChannelParams
{
    /// OFF until a watched run says otherwise. Not a tuning knob — a feature that has never driven a
    /// robot, kept behind a switch the user owns.
    bool   enabled = false;
    /// One passive window per this many seconds. Same length the offline replay uses, so the online
    /// and offline answers stay comparable on the same log.
    double window_s = 1.0;
    /// The base's own envelope, for the kinematic admissibility test. A relocalisation is not travel:
    /// 1.77% of localiser cycles on this robot imply a speed above 1 m/s on a 0.6 m/s base, and the
    /// worst moved 11.63 m in 50 ms. Charging that to the odometry would put a metre-scale outlier
    /// into a fit whose typical increment is a metre.
    double max_speed_mps = 0.6;
    /// How fast the base turns under the Orient policy — used ONLY to predict how long the manoeuvre
    /// would take, which is what converts its excitation into a rate the marginal gain can compare
    /// against the tours. It prices the offer; it does not command anything.
    /// ★MIRRORS THE CONSUMER'S `LockOnMaxYawRps`, WHICH IS 0.06 AND NOT 0.12. The first version of
    /// this file assumed 0.12 from the code default and the live config says half that — so a 120
    /// degree step takes 35 s, not 17. The producer cannot read the consumer's config, so this is an
    /// assumption and is named as one; when it is wrong the price is wrong, nothing else.
    /// The rate the pivot is BOTH priced at and executed at -- one number, declared to the consumer
    /// through Contract::yaw_rate rather than assumed about it.
    ///
    /// ★IT WAS TWO NUMBERS AND THEY DISAGREED. This constant only ever priced the offer; execution
    /// was capped by the consumer's Controller.LockOnMaxYawRps = 0.06, tuned for the LockOn
    /// micro-search where creeping protects the masks being collected. A calibration pivot observes
    /// nothing while it turns, so slow is pure cost there -- two different manoeuvres sharing one
    /// cap. Raising the price alone made the offer advertise a 50 s detour that still took 7 minutes,
    /// which lets it win contests it does not deserve; reverting the price alone made it honestly
    /// expensive and permanently uncompetitive. Neither is the fix. The producer knows what the
    /// manoeuvre is FOR, so the producer states the rate and the consumer clamps it to the BASE's
    /// limit (MaxRotSpeed), not to its own servo tuning.
    /// ★Measured live at the old cap: 0.056 rad/s, 16 deg per 5 s, 2 mm of translation -- correct
    /// behaviour, and slow enough that it looked stopped to someone watching.
    double pivot_rot_rate = 0.5;
    /// ★THE RATES, ONE PER BLOCK. A closure at rate w measures k + b/w, so a single rate reports the
    /// scale and the bias fused into one number and cannot say which is which — see PivotParams.
    /// Two blocks at two rates separate them exactly, on headings alone.
    /// ★THEY MUST DIFFER ENOUGH TO MATTER. The bias falls out of a difference divided by
    /// (1/w1 - 1/w2), so rates that are close divide a small difference by a small number and
    /// amplify the noise. 0.5 and 0.25 give 1/w spread of 2 s/rad, which is the widest this base
    /// usefully offers: slower than 0.25 makes a four-turn block take over three minutes.
    /// A single-element list restores the old single-rate manoeuvre exactly.
    std::vector<double> pivot_rates = {0.5, 0.25};
    /// Time constant of the passive-excitation EMA. Long, because the question it answers is "what is
    /// this robot's diet", not "what is it doing right now".
    double passive_tau_s = 120.0;
    /// ★TESTING ONLY, AND IT LIES ON PURPOSE. >0 replaces the advertised gain with this constant so
    /// the pivot wins the EFE contest on demand while the manoeuvre itself is still being debugged.
    /// A well-calibrated robot correctly prices this offer near zero (measured +0.00007 +/- 0.00039
    /// on P3Bot => fractions of a nat), so it loses every contest to an exploration cell and the
    /// four-turn closure has never once been exercised end to end. This forces the contest, not the
    /// manoeuvre: everything downstream of selection runs exactly as it would in earnest.
    /// ★NOTHING MEASURED WHILE THIS IS SET MAY BE QUOTED AS A VALUATION. The true figure is still
    /// computed and still logged beside the forced one, precisely so a run cannot be graded on the
    /// number we invented. Leave at 0 outside a debugging session.
    double forced_gain_nats = 0.0;
    PivotParams          pivot;
    ScaleEstimatorParams estimator;
};

class CalibChannel
{
public:
    explicit CalibChannel(CalibChannelParams p = CalibChannelParams{})
        : p_(p), rot_(p.estimator), pivot_(p.pivot) {}

    [[nodiscard]] const PivotAffordance &pivot() const { return pivot_; }
    [[nodiscard]] ChannelPosterior       posterior() const { return rot_.posterior(); }
    [[nodiscard]] double passive_rate_rad_s() const { return passive_rate_; }
    /// The rate this channel prices AND asks the consumer to execute at — one number, so the offer
    /// cannot advertise a cost the manoeuvre will not have. Per BLOCK: the manoeuvre deliberately
    /// runs its blocks at different rates so the closure can separate scale from bias.
    [[nodiscard]] double pivot_rate_rps() const { return rate_for_block(pivot_.block()); }

    /// The rate block `b` runs at. Falls back to the single configured rate when no list is given.
    [[nodiscard]] double rate_for_block(int b) const
    {
        if (p_.pivot_rates.empty()) return p_.pivot_rot_rate;
        return p_.pivot_rates[static_cast<std::size_t>(
            std::clamp(b, 0, static_cast<int>(p_.pivot_rates.size()) - 1))];
    }

    /// ── SCALE AND BIAS, SEPARATED BY THE RATES ALONE ────────────────────────────────────────────
    /// Each closed block gives s_i = k + b/w_i. With two or more distinct rates that is a straight
    /// line in u = 1/w, and k is its intercept while b is its slope. Solved in closed form for the
    /// two-block case, which is the one the manoeuvre runs.
    /// ★THE UNCERTAINTIES ARE THE CLOSURES' OWN RESOLUTIONS, PROPAGATED — not a confidence invented
    /// here. A block that resolves 0.2% cannot contribute better than 0.2% to either term, and the
    /// bias divides a DIFFERENCE of two such numbers by (u1-u2), so it is always the coarser of the
    /// two. Saying so is the difference between an instrument and a number.
    struct RateSeparation
    {
        bool   solved = false;      ///< two or more blocks closed at DISTINCT rates
        double k_omega = 0.0;       ///< rotation scale, fractional — free of the bias at last
        double sigma_k = 0.0;
        double b_omega = 0.0;       ///< gyro bias, rad/s
        double sigma_b = 0.0;
        bool   usable_k = false;    ///< the term exceeds what the closures can resolve
        bool   usable_b = false;
    };

    [[nodiscard]] RateSeparation separate_scale_and_bias() const
    {
        RateSeparation r;
        const auto &cs = pivot_.closures();
        if (cs.size() < 2) return r;
        // Two blocks is what the manoeuvre runs; with more, the first and last give the widest
        // lever in u and therefore the best-conditioned solve.
        const double u1 = 1.0 / std::max(rate_for_block(0), 1e-6);
        const double u2 = 1.0 / std::max(rate_for_block(static_cast<int>(cs.size()) - 1), 1e-6);
        const double du = u1 - u2;
        if (std::abs(du) < 1e-6) return r;      // same rate twice separates nothing
        const double s1 = cs.front().s_omega,  s2 = cs.back().s_omega;
        const double r1 = cs.front().resolution, r2 = cs.back().resolution;
        r.b_omega = (s1 - s2) / du;
        r.sigma_b = std::hypot(r1, r2) / std::abs(du);
        r.k_omega = (u1 * s2 - u2 * s1) / du;
        r.sigma_k = std::hypot(u1 * r2, u2 * r1) / std::abs(du);
        r.usable_k = std::abs(r.k_omega) > r.sigma_k;
        r.usable_b = std::abs(r.b_omega) > r.sigma_b;
        r.solved = true;
        return r;
    }
    [[nodiscard]] double pivot_step_rad() const { return p_.pivot.step_rad; }
    [[nodiscard]] bool   offering() const { return offer_open_; }
    [[nodiscard]] bool   enabled_public() const { return p_.enabled; }
    /// True while this side is holding a refusal, i.e. waiting to be carried somewhere with room.
    [[nodiscard]] bool   refused_here() const { return refused_at_.has_value(); }
    [[nodiscard]] double authoritative_information() const noexcept { return authoritative_info_; }

    /// One cycle of ordinary life. `posterior_theta` is the localiser's fused heading, `odom_dtheta`
    /// the measured odometry increment for THIS frame, `ref_travel_m` the reference displacement over
    /// the same frame (used only to reject a pose discontinuity).
    ///
    /// ★CALLED WHATEVER THE ROBOT IS DOING, including during the pivot. The estimator is watching
    /// motion, not serving an affordance, and excluding the pivot's own turning would throw away the
    /// most informative windows of the day.
    void note_motion(double t_s, double x, double y, double posterior_theta, double odom_dtheta,
                     bool odom_valid)
    {
        if (not std::isfinite(posterior_theta) or not std::isfinite(odom_dtheta)) return;
        if (not have_prev_)
        { have_prev_ = true; prev_theta_ = posterior_theta; prev_x_ = x; prev_y_ = y;
          prev_t_ = win_t0_ = t_s; return; }

        const double dt    = t_s - prev_t_;
        const double d_ref = wrap(posterior_theta - prev_theta_);
        const double step  = std::hypot(x - prev_x_, y - prev_y_);
        prev_theta_ = posterior_theta; prev_x_ = x; prev_y_ = y; prev_t_ = t_s;

        // ★A RELOCALISATION IS NOT ODOMETRY ERROR, AND IT IS JUDGED PER FRAME. The reference is the
        // localiser's posterior and it jumps: measured 2026-08-20, 2172 of 122841 cycles imply a speed
        // above 1 m/s on a 0.6 m/s base, the worst 11.63 m in 50 ms. Testing the jump against the
        // WINDOW's length instead of the frame's would let a 3 m teleport through unnoticed, because
        // 3 m in a second is not impossible — it is only impossible in 50 ms. One poisoned frame
        // discards the whole window: a jump anywhere in it corrupts both ends of the difference.
        if (not ScaleEstimator::window_is_physical(step, dt, p_.max_speed_mps)) win_poisoned_ = true;

        if (odom_valid)
        {
            win_ref_ += d_ref;
            win_odo_ += odom_dtheta;
        }
        else win_poisoned_ = true;   // a window missing part of its odometry is not a window
        // ★THE PIVOT'S OWN ACCUMULATOR RUNS ON THE SAME INCREMENT, not on a second one derived later.
        // Taking the difference again from state that has already advanced returns zero — which is
        // what a first version of the offline loop accumulated, silently, for a whole log.
        // ★★★A CLOSURE IS A TOTAL, SO IT COUNTS EVERYTHING BETWEEN ITS TWO ENDS.
        // These were gated on `offer_open_ and claim_held_` -- only motion made while the consumer
        // demonstrably held our claim. That was written to stop a competing traversal being credited
        // to a pivot step, and it broke the measurement it was protecting. The closure's argument is
        // "the heading left `start` and came back, therefore the robot turned a whole number of
        // turns", and that is a statement about EVERY radian in between: the claim latency, the
        // deceleration tail after the consumer clears its flag, the seconds between one step
        // completing and the next being claimed. Measured live 2026-08-25: fifteen anchored steps,
        // five complete turns (1800 deg), and ref_turn_ credited only 1440 -- a whole turn lost to
        // poll and DDS latency on the two claim edges. whole_turns() then read 4, the odometry's
        // 1502 deg was divided by four turns instead of five, and the pivot reported +4.34% against
        // an online estimator reading +0.34% on the same robot from the same odometry.
        // ★AND THE GATE IS NOW UNNECESSARY, which is the part worth keeping. It existed because
        // closure() ASSERTED four turns as the truth, so stray rotation could fake the total. The
        // truth is COUNTED now: a traversal in the middle raises ref_turn_, raises whole_turns(),
        // raises truth_rad, and is counted in odom_turn_ as well -- the ratio is unharmed and the
        // closure stays exact. Counting instead of asserting is what removed the need to gate.
        if (pivot_.state() == PivotAffordance::State::Offering and odom_valid)
            step_odom_ += odom_dtheta;
        // ★AND THE REFERENCE SIDE THE SAME WAY. The pivot used to take the reference turn as
        // wrap(heading_now - heading_at_the_last_step), which spans the GAP between steps -- so any
        // driving the robot did in between was counted as pivot rotation, and wrap() capped a 200 deg
        // excursion at -160. Measured live: 932 deg of in-place turning summed to only -662 signed,
        // with 269 deg of rotation-while-driving in between. Both sides of the comparison are now
        // accumulated over the SAME interval, which is the only way the ratio means anything.
        if (pivot_.state() == PivotAffordance::State::Offering) step_ref_ += d_ref;

        const double T = t_s - win_t0_;
        if (T < p_.window_s) return;

        if (not win_poisoned_)
        {
            rot_.predict(T);
            rot_.add(win_ref_, win_odo_ - win_ref_, T);
            // The diet: how much turning per second the robot's ordinary work delivers. An EMA over
            // the SAME windows the estimator is fed, so the two can never disagree about what happened.
            const double a = 1.0 - std::exp(-T / std::max(p_.passive_tau_s, 1e-3));
            passive_rate_ += a * (std::abs(win_ref_) / T - passive_rate_);
        }
        else ++poisoned_windows_;
        win_ref_ = win_odo_ = 0.0; win_poisoned_ = false; win_t0_ = t_s;
    }

    [[nodiscard]] long poisoned_windows() const { return poisoned_windows_; }

    /// How long a whole pivot would take, at the rate the consumer actually turns.
    /// ★THE WHOLE MANOEUVRE, ALL BLOCKS. Pricing only the current block would understate the detour
    /// by however many blocks are still to come, and the offer must cost what it will actually cost.
    [[nodiscard]] double pivot_duration_s() const
    {
        const int n = std::max(1, p_.pivot.blocks);
        double t = 0.0;
        for (int b = 0; b < n; ++b)
            t += std::abs(2.0 * M_PI * p_.pivot.turns) / std::max(rate_for_block(b), 1e-3);
        return t;
    }

    /// What the manoeuvre is worth BEYOND the free data, in nats — the number that goes on the wire
    /// and competes with every exploration cell.
    /// Supply the information (1/sigma^2) that the AUTHORITATIVE estimator currently holds on the
    /// rotation scale — room_concept's joint BatchEstimator, whose parameters actually correct the
    /// prediction. <=0 or non-finite means "not available", and the channel falls back to its own
    /// passive posterior.
    ///
    /// ★Without this the offer prices a posterior that nothing consumes. This channel observes and
    /// feeds nothing back by design, so a robot could correctly decide a pivot was worthless while
    /// the estimator steering it still had an uninformed parameter.
    void set_authoritative_information(double info) noexcept { authoritative_info_ = info; }

    /// What the offer ADVERTISES — the true valuation, unless a debugging session has forced it.
    [[nodiscard]] double marginal_gain_nats() const
    {
        if (p_.forced_gain_nats > 0.0) return p_.forced_gain_nats;
        return true_marginal_gain_nats();
    }

    /// True while the advertised gain is a fabrication. Anything reporting a valuation must say so.
    [[nodiscard]] bool gain_is_forced() const noexcept { return p_.forced_gain_nats > 0.0; }

    /// What the manoeuvre is ACTUALLY worth, always computed, never overridden — the number a run is
    /// graded on even when the wire carries the forced one.
    [[nodiscard]] double true_marginal_gain_nats() const
    {
        const double T = pivot_duration_s();
        const double excite = 2.0 * M_PI * p_.pivot.turns * std::max(1, p_.pivot.blocks);
        if (authoritative_info_ > 0.0 and std::isfinite(authoritative_info_))
            return rot_.expected_information_gain_from(authoritative_info_, excite, T,
                                                       passive_rate_ * T);
        return rot_.expected_information_gain(excite, T, passive_rate_ * T);
    }

    /// Should an offer go out, and at what bearing? nullopt is the normal state of a calibrated robot.
    std::optional<double> offer(double robot_heading_rad)
    {
        if (not p_.enabled) return std::nullopt;
        if (offer_open_) return std::nullopt;          // one live offer at a time
        // ★DO NOT LATCH HERE. offer_open_ means "an offer is LIVE ON THE WIRE", and that is only
        // true once the producer has actually published it. Latching on the mere intention deadlocked
        // the channel permanently: ensure_calib_node() deliberately returns false on the cycle it
        // CREATES the node ("arm on the next cycle"), so publish_target was skipped, no has_intention
        // edge was written, and the next cycle's offer() refused because offer_open_ was already set.
        // Nothing could clear it, because only an outcome does and no consumer could ever see an
        // affordance that was never published. Observed live 2026-08-24: afford_calib present in the
        // graph with no edge to room, for ever. The caller calls mark_offered() once the publish has
        // succeeded.
        return pivot_.next_bearing(robot_heading_rad, marginal_gain_nats());
    }

    /// The offer reached the graph. Only now is one live.
    /// ★DOES NOT RESET THE ACCUMULATORS. They run from the pivot's first offer to its closure and are
    /// banked at each outcome; zeroing them here would drop whatever the robot did between one step
    /// finishing and the next going out, which the closure's total must include.
    void mark_offered() noexcept { offer_open_ = true; }

    /// The consumer answered. The heading is the robot's MEASURED one now — the sequence advances on
    /// that and never on the count of steps issued.
    ///
    /// ★INFEASIBLE IS THE ONE THAT MEANS SOMETHING SPECIFIC HERE: the consumer tested whether the body
    /// can sweep its diagonal where it stands and said no. Believe it, stop asking, and wait to be
    /// carried somewhere with room by the ordinary work — which is the whole opportunism. Every other
    /// non-satisfied outcome leaves the sequence exactly where it was, short of closure.
    void on_outcome(rc::affordance::Outcome o, double heading_rad)
    {
        if (not offer_open_) return;
        offer_open_ = false;
        using O = rc::affordance::Outcome;
        pivot_.on_outcome(o == O::Satisfied, o == O::Infeasible, heading_rad, step_odom_, step_ref_);
        step_odom_ = step_ref_ = 0.0;
    }

    /// The robot has been carried elsewhere by its ordinary work; a spot that would not do no longer
    /// applies. Distance measured against where the refusal happened, in the caller's own frame.
    void note_robot_pos(double x, double y)
    {
        if (pivot_.state() == PivotAffordance::State::SpotRefused)
        {
            if (not refused_at_.has_value()) { refused_at_ = std::pair{x, y}; return; }
            // ★THE BODY'S OWN WIDTH IS THE ONLY DISTANCE THAT MEANS ANYTHING HERE. "Somewhere else"
            // for a question about whether the body fits is one body away; anything shorter is the
            // same spot with noise on it, and anything longer is a number nobody can justify.
            const double dx = x - refused_at_->first, dy = y - refused_at_->second;
            if (std::hypot(dx, dy) > kBodyWidthM) { refused_at_.reset(); pivot_.robot_moved(); }
        }
        else refused_at_.reset();
    }

    [[nodiscard]] PivotClosure closure() const { return pivot_.closure(); }

    /// The closure has been read and recorded; make the channel available again.
    ///
    /// ★★★`Closed` MEANT "NEVER AGAIN", AND IT SHOULD MEAN "THAT ONE IS FINISHED". A closed pivot
    /// returned nullopt from every future offer, so the manoeuvre was once per process: afford_calib
    /// simply vanished from the graph and nothing said why. That contradicts the premise the whole
    /// channel is built on -- the scale is a slowly varying quantity (tyres wear, payloads shift,
    /// which is why ScaleEstimator carries a random-walk density at all), so a measurement made once
    /// at boot cannot stay true, and a robot that improves by itself in time must be able to ask
    /// again.
    /// ★AND NOTHING NEEDS A SCHEDULE OR A THRESHOLD TO DECIDE WHEN. The marginal gain already governs
    /// it: next_bearing refuses to offer unless there is gain to advertise, and the gain falls as the
    /// posterior sharpens (measured across this pivot, 5.204 -> 1.092 nats) and widens again on its
    /// own as the random walk ages the evidence. Re-arming just returns that decision to the rule
    /// that was always supposed to make it.
    void restart_after_closure()
    {
        if (pivot_.state() == PivotAffordance::State::Closed) pivot_.reset();
    }

private:
    static constexpr double kBodyWidthM = 0.543;   // Shadow's measured across-body extent
    static double wrap(double a) { while (a > M_PI) a -= 2*M_PI; while (a < -M_PI) a += 2*M_PI; return a; }

    CalibChannelParams p_;
    ScaleEstimator     rot_;
    PivotAffordance    pivot_;

    bool   have_prev_ = false;
    double prev_theta_ = 0.0, prev_x_ = 0.0, prev_y_ = 0.0, prev_t_ = 0.0, win_t0_ = 0.0;
    double win_ref_ = 0.0, win_odo_ = 0.0;
    bool   win_poisoned_ = false;
    long   poisoned_windows_ = 0;   ///< windows discarded for a pose jump or missing odometry
    double authoritative_info_ = 0.0;   // 1/sigma^2 from the estimator that steers; 0 = unset
    double passive_rate_ = 0.0;      ///< rad/s of turning the ordinary tours deliver — MEASURED
    bool   offer_open_ = false;
    double step_odom_  = 0.0;        ///< odometry turn accumulated since this offer went out
    double step_ref_   = 0.0;        ///< reference (localiser) turn over the SAME interval
    std::optional<std::pair<double,double>> refused_at_;
};

}   // namespace rc::calib
