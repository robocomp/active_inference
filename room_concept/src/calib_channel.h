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

#include <cmath>
#include <optional>

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
    /// Time constant of the passive-excitation EMA. Long, because the question it answers is "what is
    /// this robot's diet", not "what is it doing right now".
    double passive_tau_s = 120.0;
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
    /// cannot advertise a cost the manoeuvre will not have.
    [[nodiscard]] double pivot_rate_rps() const { return p_.pivot_rot_rate; }
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
        if (offer_open_ and odom_valid) step_odom_ += odom_dtheta;
        // ★AND THE REFERENCE SIDE THE SAME WAY. The pivot used to take the reference turn as
        // wrap(heading_now - heading_at_the_last_step), which spans the GAP between steps -- so any
        // driving the robot did in between was counted as pivot rotation, and wrap() capped a 200 deg
        // excursion at -160. Measured live: 932 deg of in-place turning summed to only -662 signed,
        // with 269 deg of rotation-while-driving in between. Both sides of the comparison are now
        // accumulated over the SAME interval, which is the only way the ratio means anything.
        if (offer_open_) step_ref_ += d_ref;

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
    [[nodiscard]] double pivot_duration_s() const
    {
        return std::abs(2.0 * M_PI * p_.pivot.turns) / std::max(p_.pivot_rot_rate, 1e-3);
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

    [[nodiscard]] double marginal_gain_nats() const
    {
        const double T = pivot_duration_s();
        const double excite = 2.0 * M_PI * p_.pivot.turns;
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
    void mark_offered() noexcept { offer_open_ = true; step_odom_ = step_ref_ = 0.0; }

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
