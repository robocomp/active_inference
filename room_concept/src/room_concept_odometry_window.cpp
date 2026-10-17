// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// Odometry preintegration over the window.

#include "room_concept.h"
#include "../../common/status_stream/status_sink.h"
#include <pthread.h>   // pthread_setname_np: name the worker so a per-thread CPU sample attributes itself
#include "pointcloud_center_estimator.h"
#include <numeric>
#include <set>
#include "room_gn_solver.h"
#include "room_obs_weights.h"
#include "room_gn_solver.h"
#include "image_edge_accumulate.h"
#include "image_edge_ops.h"
#include <algorithm>
#include <ranges>
#include <fstream>
#include <filesystem>
#include <limits>
#include <locale>
#include <ctime>
#include <print>
#include <sstream>
#include <sys/stat.h>
#include <QDebug>

namespace rc
{

    // The interval's clock and sensor streams, and the per-segment integrals over them. Until refactor step 2c
    // (2026-10-17) these were lambdas inside integrate_odometry_over_window; the bodies are unchanged.
    struct RoomConcept::OdomSources
    {
        bool use_sim_clock = false;
        const std::vector<ImuReading> *imu_history = nullptr;
        double odom_dt_sample = -1.0;           // measured wheel sample period (s); <0 = unmeasurable
        bool odom_variance_injection = false;   // Params::odom_variance_injection

        std::int64_t stamp_of(const OdometryReading &o) const
        { return use_sim_clock ? o.integration_ts_ms() : o.effective_ts_ms(); }
        std::int64_t imu_stamp(const ImuReading &s) const
        { return use_sim_clock ? s.integration_ts_ms() : s.effective_ts_ms(); }
        // ── Linear channel from the accelerometer ──────────────────────────────────────────────────
        // Returns the velocity CHANGE over [seg_start, seg_end] in the body frame, and the
        // displacement that change contributes within the same interval (the double integration).
        //
        // Both are INCREMENTS confined to one interval and are never chained. That distinction is the
        // whole design: over ~50 ms the accelerometer estimates dv ~15x more precisely than the
        // wheels (0.1-0.4 mm/s of noise against their 67 mm/s per sample), but an accelerometer
        // cannot observe velocity itself, only its change, and chaining the same 0.5 deg tilt error
        // reaches 0.086 m/s after 1 s and 5.1 m/s after 60 s. So this supplies dv; the wheels supply
        // the absolute v that dv is a change TO.
        //
        // dp here is the correction to the constant-velocity assumption inside the interval
        // (0.5*a*T^2), not a position estimate. At 50 ms it is sub-millimetre and its only job is to
        // stop a hard acceleration being modelled as if the velocity had been constant throughout.
        //
        // GRAVITY IS NOT REMOVED. The samples carry it, and the horizontal components are honest
        // horizontal acceleration only insofar as the mount is level; the residual is a slowly
        // varying bias, which is why this is logged and cross-checked before it is ever fused.
        bool imu_dvel(std::int64_t seg_start, std::int64_t seg_end,
                                  float &dvx_out, float &dvy_out,
                                  float &dpx_out, float &dpy_out) const
        {
            if (imu_history == nullptr or imu_history->size() < 2) return false;
            if (imu_stamp(imu_history->front()) > seg_start or imu_stamp(imu_history->back()) < seg_end)
                return false;                       // partial coverage refused, as for imu_dtheta
            double vx = 0, vy = 0, px = 0, py = 0;
            bool any = false;
            for (size_t k = 0; k + 1 < imu_history->size(); ++k)
            {
                const auto &a0 = (*imu_history)[k];
                const auto &a1 = (*imu_history)[k + 1];
                const std::int64_t t0 = std::max(imu_stamp(a0), seg_start);
                const std::int64_t t1 = std::min(imu_stamp(a1), seg_end);
                if (t1 <= t0) continue;
                const double h = static_cast<double>(t1 - t0) * 1e-3;
                // Trapezoid on acceleration: the sample rate is ~5x the interval, so the ramp between
                // samples is real information and a zero-order hold would systematically lag it.
                const double ax = 0.5 * (a0.acc_x + a1.acc_x);
                const double ay = 0.5 * (a0.acc_y + a1.acc_y);
                px += vx * h + 0.5 * ax * h * h;    // v already accumulated within THIS interval only
                py += vy * h + 0.5 * ay * h * h;
                vx += ax * h;
                vy += ay * h;
                any = true;
            }
            if (not any) return false;
            dvx_out = static_cast<float>(vx); dvy_out = static_cast<float>(vy);
            dpx_out = static_cast<float>(px); dpy_out = static_cast<float>(py);
            return true;
        }

        // Mean per-sample variance a channel reported over [seg_start, seg_end], converted to the
        // NOISE DENSITY the preintegrator wants: sigma = sqrt(var_sample * dt_sample). Returns <0
        // when the producer said "unknown" (negative variance) or the segment is not bracketed, and
        // the caller then falls back to the asserted constant. See the conversion note in
        // se2_preintegration.h: handing it the raw per-sample variance over-states the noise by
        // dt_sample/dt, a factor of five at 100 Hz across a 50 ms segment.
        float imu_sigma(std::int64_t seg_start, std::int64_t seg_end,
                                   bool want_gyro) const
        {
            if (imu_history == nullptr or imu_history->size() < 2) return -1.f;
            double var_sum = 0.0; int n = 0;
            std::int64_t first = 0, last = 0;
            for (const auto &sm : *imu_history)
            {
                const std::int64_t st = imu_stamp(sm);
                if (st < seg_start or st > seg_end) continue;
                const float v = want_gyro ? sm.gyro_var : sm.acc_var;
                if (not (v >= 0.f)) return -1.f;      // producer does not know -> use the model
                var_sum += v; ++n;
                if (first == 0) first = st;
                last = st;
            }
            if (n < 2 or last <= first) return -1.f;
            const double dt_sample = static_cast<double>(last - first) * 1e-3 / (n - 1);
            return static_cast<float>(std::sqrt(var_sum / n * dt_sample));
        }
        // Returns <0 for "use the model": the channel is off, the producer said "unknown" (negative
        // variance), or the period could not be measured. A zero variance is REFUSED for the same
        // reason: it would claim infinite confidence in a channel, and no wheel is that good.
        float wheel_sigma(float var_sample) const
        {
            if (not odom_variance_injection) return -1.f;
            if (not (var_sample > 0.f) or odom_dt_sample <= 0.0) return -1.f;
            return static_cast<float>(std::sqrt(static_cast<double>(var_sample) * odom_dt_sample));
        }
        // Two independent noise contributions to the same channel add in quadrature. Either being
        // "unstated" (<0) leaves the other alone, and both unstated falls back to the model.
        static float quad(float a, float b)
        {
            if (a < 0.f) return b;
            if (b < 0.f) return a;
            return std::sqrt(a * a + b * b);
        }

        bool imu_dtheta(std::int64_t seg_start, std::int64_t seg_end, float &dtheta_out) const
        {
            if (imu_history == nullptr or imu_history->size() < 2) return false;
            if (imu_stamp(imu_history->front()) > seg_start or imu_stamp(imu_history->back()) < seg_end)
                return false;
            double acc = 0.0;
            bool any = false;
            for (size_t k = 0; k + 1 < imu_history->size(); ++k)
            {
                const std::int64_t a = imu_stamp((*imu_history)[k]);
                const std::int64_t b = imu_stamp((*imu_history)[k + 1]);
                if (b <= a) continue;
                const std::int64_t s = std::max(a, seg_start);
                const std::int64_t e = std::min(b, seg_end);
                if (e <= s) continue;
                // Trapezoid across the sample pair, so a rate ramping through the segment is not
                // biased by holding the leading sample flat.
                const double w0 = (*imu_history)[k].gyro_z, w1 = (*imu_history)[k + 1].gyro_z;
                const double span = static_cast<double>(b - a);
                const double f0 = static_cast<double>(s - a) / span;
                const double f1 = static_cast<double>(e - a) / span;
                const double wm = 0.5 * ((w0 + (w1 - w0) * f0) + (w0 + (w1 - w0) * f1));
                acc += wm * static_cast<double>(e - s) * 0.001;
                any = true;
            }
            if (not any) return false;
            dtheta_out = static_cast<float>(acc);
            return true;
        }
    };

    Eigen::Vector3f RoomConcept::integrate_odometry_over_window(
                const Eigen::Affine2f& robot_pose,
                const std::vector<OdometryReading> &odometry_history,
                const int64_t &t_start_ms,
                const int64_t &t_end_ms,
                rc::preint::Interval *preint_out,
                const std::vector<ImuReading> *imu_history,
                const SimClockMap *sim_clock)
    {
        Eigen::Vector3f total_delta = Eigen::Vector3f::Zero();
        float running_theta = std::atan2(robot_pose.linear()(1,0), robot_pose.linear()(0,0));

        // Preintegration runs on the SAME segments, seeded at the SAME heading. It is a pure observer:
        // it never writes total_delta, so the mean cannot change depending on whether it is asked for.
        rc::preint::Integrator preint(running_theta);
        preint.set_noise(params.odom_preint_noise);

        // ── ONE CLOCK, and it is the producer's ────────────────────────────────────────────────────
        // The two ends of this interval come from different places. The BOUNDS are lidar sweep stamps,
        // which are local wall-clock. The RATES are in whatever clock the producer measured them in --
        // and a simulator measures per SIMULATION second while running behind real time, so
        // integrating its rates over wall intervals under-counts by exactly the sim/wall ratio (~6%
        // here; the mirror image of the bug webots-bridge commit 5968a65 fixed from the other side).
        //
        // Converting the two BOUNDS is the cheap direction: everything below then works in one clock,
        // and because this function returns a DISPLACEMENT rather than a rate, no caller has to know.
        // On real hardware the map is identity and none of this costs anything.
        const bool use_sim_clock = sim_clock != nullptr and sim_clock->valid()
                                   and not odometry_history.empty()
                                   and odometry_history.front().simulated;
        OdomSources src;
        src.use_sim_clock = use_sim_clock;
        src.imu_history = imu_history;
        src.odom_variance_injection = params.odom_variance_injection;
        const auto to_clock = [&](std::int64_t wall_ms) -> std::int64_t
        { return use_sim_clock ? sim_clock->to_sim(wall_ms) : wall_ms; };
        const auto stamp_of = [&](const OdometryReading &o) -> std::int64_t { return src.stamp_of(o); };


        const std::int64_t win_start_ms = to_clock(t_start_ms);
        const std::int64_t win_end_ms   = to_clock(t_end_ms);
        // The window's own duration, kept for the calibrator: elapsed time is the only covariate that
        // separates a gyro bias from a gyro scale.
        last_cycle_dt_s_ = static_cast<float>(win_end_ms - win_start_ms) * 1e-3f;

        // ── Heading change from the gyro, over an arbitrary sub-interval ───────────────────────────
        // Yaw is the channel wheel odometry gets worst: a differential base turns by scrubbing its
        // wheels sideways, so it over-reports rotation -- measured 8.2% on this robot, against
        // translation exact to 0.1%. The gyro measures the body rate directly and carries no such
        // error. It also runs ~10x faster, so this integrates the rate as it actually varied instead
        // of holding one 10 Hz `rot` sample flat across a whole sweep interval.
        //
        // Returns false unless the IMU brackets the WHOLE segment; partial coverage is deliberately
        // refused, because a Δθ covering part of a segment silently under-rotates, which is worse
        // than a consistent wheel estimate.
        // (imu_stamp, imu_dvel, imu_sigma, wheel_sigma, quad and imu_dtheta are OdomSources members now.)

        // ── The WHEELS' own stated noise, as a density ─────────────────────────────────────────────
        // Same conversion as imu_sigma, and it exists for the same reason: the producer states a
        // PER-SAMPLE variance at ITS OWN rate (FullPoseEuler::velCov, forwarded onto the robot node by
        // robot_concept), while the preintegrator integrates densities. sigma = sqrt(var * dt_sample).
        //
        // dt_sample is MEASURED from the stamps, never configured. The bridge's [FullPose]
        // PublishPeriod went 100 ms -> 20 ms on 2026-08-25, and a hard-coded period would have made
        // the density 5x too large the moment it changed -- silently, since a variance that is merely
        // wrong still looks like a number. Measuring it means the rate can move again and this line
        // stays correct; it is also the only way a FASTER stream can tighten anything, because a
        // density is invariant to sample rate and only a smaller per-sample variance or a shorter
        // dt_sample moves it.
        //
        // Estimated across the whole history rather than per pair, so one late sample cannot inflate
        // the period for the segment it lands in.
        double odom_dt_sample = -1.0;
        if (odometry_history.size() >= 2)
        {
            const std::int64_t first = stamp_of(odometry_history.front());
            const std::int64_t last  = stamp_of(odometry_history.back());
            if (last > first)
                odom_dt_sample = static_cast<double>(last - first) * 1e-3
                               / static_cast<double>(odometry_history.size() - 1);
        }
        src.odom_dt_sample = odom_dt_sample;
        int imu_segments = 0, wheel_segments = 0;
        int odom_var_segments = 0;   // segments whose noise the WHEELS themselves stated
        cyc_imu_dtheta_ = cyc_wheel_dtheta_ = cyc_wheel_shadow_dtheta_ = 0.f;
        cyc_dx_local_ = cyc_dy_local_ = 0.f;
        cyc_imu_dvx_ = cyc_imu_dvy_ = cyc_imu_dpx_ = cyc_imu_dpy_ = 0.f;
        cyc_wheel_dvx_ = cyc_wheel_dvy_ = 0.f; cyc_imu_lin_segs_ = 0;
        cyc_imu_segs_ = cyc_wheel_segs_ = 0;
        cyc_heading_cov_ = {};
        cyc_gyro_w_dt_ = cyc_heading_dt_ = 0.0;
        cyc_hdiag_ = {};
        cyc_dens_w_dt_ = cyc_dens_g_dt_ = 0.0;

        // Integrate over all odometry readings in [win_start_ms, win_end_ms], on the clock chosen above.
        for (size_t i = 0; i < odometry_history.size(); ++i)
        {
            const auto& odom = odometry_history[i];

            // Get time window for this reading
            const std::int64_t cmd_start_ms = stamp_of(odom);
            const std::int64_t cmd_end_ms = (i + 1 < odometry_history.size())
                           ? stamp_of(odometry_history[i + 1])
                           : win_end_ms;

            if (cmd_start_ms <= 0 || cmd_end_ms <= 0 || cmd_end_ms <= cmd_start_ms)
                continue;

            // Clip to [win_start_ms, win_end_ms]
            if (cmd_end_ms < win_start_ms) continue;
            if (cmd_start_ms > win_end_ms) break;

            const std::int64_t effective_start_ms = std::max(cmd_start_ms, win_start_ms);
            const std::int64_t effective_end_ms = std::min(cmd_end_ms, win_end_ms);

            const float dt = static_cast<float>(effective_end_ms - effective_start_ms) * 0.001f;
            if (dt <= 0) continue;

            // Odometry velocities are in robot frame: adv=forward(Y), side=lateral(X), rot=angular
            // Learned scale from motion_calib_ (1.0 until it has seen anything, and exactly 1.0
            // while the feature is off, so this line is a no-op in the default build).
            // Forward and lateral carry SEPARATE scales. They used to share one, which was harmless
            // only because lateral travel is usually ~0; on a mecanum the lateral channel is the one
            // roller slip corrupts and it is physically a different error.
            const float k_v   = motion_calib_.forward_scale();
            const float k_lat = motion_calib_.lateral_scale();
            const float dx_local = odom.side * dt * k_lat;  // lateral (X in robot frame)
            const float dy_local = odom.adv  * dt * k_v;    // forward (Y in robot frame)
            cyc_dx_local_ += dx_local;
            cyc_dy_local_ += dy_local;
            // The wheels' own velocity CHANGE across this segment, so the two channels are compared
            // as like for like: an accelerometer gives dv, never v. A disagreement here is the first
            // thing translation has ever had that can see wheel slip -- during slip the wheels report
            // a dv that did not happen while the accelerometer sees the truth. Heading has had this
            // cross-check since the gyro went in; translation has had NO second opinion at all.
            if (i + 1 < odometry_history.size())
            {
                const auto &nx = odometry_history[i + 1];
                cyc_wheel_dvx_ += nx.side - odom.side;
                cyc_wheel_dvy_ += nx.adv  - odom.adv;
            }
            float imu_dpx_seg = 0.f, imu_dpy_seg = 0.f;
            if (float dvx = 0.f, dvy = 0.f, dpx = 0.f, dpy = 0.f;
                src.imu_dvel(effective_start_ms, effective_end_ms, dvx, dvy, dpx, dpy))
            {
                cyc_imu_dvx_ += dvx; cyc_imu_dvy_ += dvy;
                cyc_imu_dpx_ += dpx; cyc_imu_dpy_ += dpy;
                ++cyc_imu_lin_segs_;
                imu_dpx_seg = dpx; imu_dpy_seg = dpy;
            }
            // THE INJECTION, linear half. Body-frame axes line up one-for-one now that the producer
            // rotates the device frame: dx_local is lateral (+X), dy_local forward (+Y), and so are
            // dpx/dpy. Refused unless the IMU brackets the WHOLE segment, exactly as for dtheta -- a
            // partial integral silently under-reports rather than degrading gracefully.
            const float dx_use = dx_local + (params.imu_linear_injection ? imu_dpx_seg : 0.f);
            const float dy_use = dy_local + (params.imu_linear_injection ? imu_dpy_seg : 0.f);

            // ── HEADING: two factors on one rotation ──────────────────────────────────────────────
            // The wheels and the gyro each measure this segment's rotation, wrong in different ways,
            // so each is corrected by ITS OWN calibration and the two are combined by precision.
            //   wheels: dpsi_w = k_omega_w * rot*dt + dk_wheel * forward     (scrubbing, asymmetry)
            //   gyro:   dpsi_g = k_omega * (integrated gyro) - b_omega * dt   (scale, bias)
            // Their product is a Gaussian on the heading whose mean is the precision-weighted mean --
            // which is what "a weighted combination of wheels and gyro" means as a generative model.
            // The weights come from each channel's noise density PLUS its scale uncertainty, so the
            // gyro takes a fast turn (scrubbing uncertainty grows with |rotation|), and the wheels
            // take a stop (they read exactly zero there; ZUPT density). No gate decides it.
            //
            // Translation stays on the wheels either way -- an accelerometer cannot supply it without
            // a drifting double integration, and the wheels are already exact there.
        const auto [dtheta, w_g, sig_om_fused, rot_eff] =
            fuse_segment_heading(odom, dt, dy_local, effective_start_ms, effective_end_ms, src,
                                 imu_segments, wheel_segments);


            // Transform to global frame using MIDPOINT theta (reduces integration bias)
            // The learned yaw offset rotates the body->world mapping: it absorbs a mount or
            // body-axis misalignment, which shows up as travel drifting off the fitted heading.
            const float theta_mid = running_theta + 0.5f * dtheta + motion_calib_.yaw_offset();
            total_delta[0] += dx_use * std::cos(theta_mid) - dy_use * std::sin(theta_mid);
            total_delta[1] += dx_use * std::sin(theta_mid) + dy_use * std::cos(theta_mid);
            total_delta[2] += dtheta;

            // Preintegration sees the SAME rate the mean used, or its covariance would describe a
            // trajectory that was not integrated.
            if (preint_out != nullptr)
            {
                // Densities from what the producers actually stated this segment, falling back to
                // the asserted constants per channel. The gyro's covers the heading it supplied; the
                // accelerometer's covers the translation only when its correction is actually being
                // used, because otherwise the displacement came from the wheels and it is the wheels'
                // noise that describes it.
                // Yaw: the density of the heading that was actually integrated -- the fused one
                // (product of the two channels' variances) under fusion, the supplying channel's
                // under the legacy switch. <0 leaves the model constant in charge.
                const float sig_om = sig_om_fused;
                // Translation always comes from the wheels. When the accelerometer's within-segment
                // correction is enabled it is ADDED to that displacement, so its noise is an extra
                // independent contribution to the same channel, not a replacement for the wheels'.
                const float sig_acc = params.imu_linear_injection
                                    ? src.imu_sigma(effective_start_ms, effective_end_ms, false) : -1.f;
                const float sig_lat  = src.quad(src.wheel_sigma(odom.var_side), sig_acc);
                const float sig_long = src.quad(src.wheel_sigma(odom.var_adv),  sig_acc);
                // The accelerometer's within-segment correction enters as an EFFECTIVE mean velocity,
                // so the preintegrator's constant-velocity step reproduces the corrected displacement
                // without changing its structure -- and the correction is then covered by Q and by
                // the transport term A, which it would not be if it were added to the mean afterwards.
                const float v_lat_eff  = odom.side + (params.imu_linear_injection ? imu_dpx_seg / dt : 0.f);
                const float v_long_eff = odom.adv  + (params.imu_linear_injection ? imu_dpy_seg / dt : 0.f);
                preint.add(v_lat_eff, v_long_eff, rot_eff, dt, sig_lat, sig_long, sig_om);
                if (sig_lat >= 0.f or sig_long >= 0.f or (w_g < 1.f and src.wheel_sigma(odom.var_rot) >= 0.f))
                    ++odom_var_segments;
            }

            running_theta += dtheta;
        }
        report_odometry_channels(src, odometry_history, t_end_ms, imu_segments, wheel_segments,
                                 odom_var_segments);

        if (preint_out != nullptr)
            *preint_out = preint.result();

        return total_delta;
    }

    RoomConcept::SegmentHeading RoomConcept::fuse_segment_heading(
                const OdometryReading &odom, float dt, float dy_local,
                std::int64_t effective_start_ms, std::int64_t effective_end_ms,
                const OdomSources &src, int &imu_segments, int &wheel_segments)
    {
            const float k_g   = motion_calib_.omega_scale();
            const float b_g   = motion_calib_.omega_bias();          // physical sign: gyro reads w + b
            const float k_ww  = motion_calib_.wheel_omega_scale();
            const float dk_wheel = motion_calib_.wheel_mismatch();
            const float curve = dk_wheel * dy_local;
            const float wheel_raw = odom.rot * dt;
            // The ZUPT condition no longer GATES anything under fusion: it only says which noise
            // describes a wheel reading -- a stopped encoder's residual (zupt_density_omega) rather
            // than a rolling one's. Under the legacy switch it still decides the channel.
            const bool wheel_stationary = params.zupt_enabled
                                        and std::abs(odom.rot)  < params.zupt_wheel_rot_eps
                                        and std::abs(odom.adv)  < params.zupt_wheel_lin_eps
                                        and std::abs(odom.side) < params.zupt_wheel_lin_eps;
            if (wheel_stationary)
                ++zupt_segs_;
            float dth_imu = 0.f;
            const bool have_gyro = src.imu_dtheta(effective_start_ms, effective_end_ms, dth_imu);
            cyc_hdiag_.raw_wheel += wheel_raw;
            cyc_hdiag_.total_dt  += dt;
            if (have_gyro) { cyc_hdiag_.raw_gyro += dth_imu; cyc_hdiag_.gyro_dt += dt; }
            if (wheel_stationary) cyc_hdiag_.zupt_dt += dt;

            float dtheta = 0.f;
            float w_g = 0.f;            // the gyro's share of this segment's heading
            float sig_om_fused = -1.f;  // density handed to the preintegrator; <0 = use the model
            rc::calib::HeadingCovariates hc{};
            if (params.heading_fusion)
            {
                const float dpsi_w = wheel_raw * k_ww + curve;
                const float ws = src.wheel_sigma(odom.var_rot);
                const float dens_w_stated = wheel_stationary ? params.odom_preint_noise.zupt_density_omega
                                          : (ws >= 0.f ? ws : params.odom_preint_noise.sigma_omega);
                // Learned: the wheels' density at THIS motion, from their disagreement with the gyro.
                // Its prior is the stated/model rest density, so on a cold start it begins there.
                if (params.heading_noise_learning)
                {
                    wheel_noise_.set_prior_density(ws >= 0.f ? ws : params.odom_preint_noise.sigma_omega);
                    // Until moving rows exist, a moving wheel is as noisy as the motion model's constant.
                    wheel_noise_.set_motion_prior_density(params.odom_preint_noise.sigma_omega);
                }
                const float dens_w = params.heading_noise_learning
                                   ? wheel_noise_.density(odom.adv, odom.rot) : dens_w_stated;
                const float s_kw = motion_calib_.param_sigma(rc::calib::P_K_OMEGA_W) * wheel_raw;
                const float var_w = dens_w * dens_w * dt + s_kw * s_kw;
                cyc_dens_w_dt_ += static_cast<double>(dens_w) * dt;
                if (have_gyro)
                {
                    const float dpsi_g = dth_imu * k_g - b_g * dt;
                    const float gs = src.imu_sigma(effective_start_ms, effective_end_ms, true);
                    const float dens_g = gs >= 0.f ? gs : params.odom_preint_noise.sigma_omega;
                    const float s_kg = motion_calib_.param_sigma(rc::calib::P_K_OMEGA) * dth_imu;
                    const float s_bg = motion_calib_.param_sigma(rc::calib::P_B_OMEGA) * dt;
                    const float var_g = dens_g * dens_g * dt + s_kg * s_kg + s_bg * s_bg;
                    // Teach the learner: wheel minus gyro cancels the true rotation. Only with a
                    // STATED gyro density (gs) -- the model constant is not an anchor.
                    if (params.heading_noise_learning and gs > 0.f)
                        wheel_noise_.add(dpsi_w - (dth_imu * k_g - b_g * dt), dt, odom.adv, odom.rot, gs);
                    cyc_dens_g_dt_ += static_cast<double>(dens_g) * dt;
                    w_g = var_w / std::max(var_w + var_g, 1e-30f);
                    dtheta = w_g * dpsi_g + (1.f - w_g) * dpsi_w;
                    // A stated density reaches the preintegrator only if some producer stated one;
                    // otherwise the model's own constant stays in charge, as before.
                    if (gs >= 0.f or (ws >= 0.f and not wheel_stationary))
                        sig_om_fused = std::sqrt(var_w * var_g / std::max(var_w + var_g, 1e-30f) / dt);
                    wheel_dtheta_sum_ += dpsi_w;            // diagnostics: both channels, same segment
                    imu_dtheta_sum_   += dpsi_g;
                    cyc_wheel_shadow_dtheta_ += dpsi_w;
                    ++cyc_imu_segs_;
                    ++imu_segments;
                }
                else
                {
                    dtheta = dpsi_w;
                    sig_om_fused = (wheel_stationary or ws < 0.f) ? -1.f : ws;
                    ++cyc_wheel_segs_;
                    ++wheel_segments;
                }
                hc.th_gyro   = w_g * dth_imu;
                hc.t_gyro    = w_g * dt;
                hc.th_wheel  = (1.f - w_g) * wheel_raw;
                hc.fwd_wheel = (1.f - w_g) * dy_local;
            }
            else
            {
                // LEGACY: one channel per segment, one shared scale. The bias now carries its
                // physical sign, so it is subtracted -- on wheel segments too, which is one of the
                // defects the fusion path removes (a parked robot's wheels have no gyro bias).
                if (not wheel_stationary and have_gyro)
                {
                    wheel_dtheta_sum_ += wheel_raw * k_g - b_g * dt + curve;
                    imu_dtheta_sum_   += dth_imu;
                    cyc_wheel_shadow_dtheta_ += wheel_raw * k_g - b_g * dt + curve;
                    dtheta = dth_imu * k_g - b_g * dt + curve;
                    w_g = 1.f;
                    sig_om_fused = src.imu_sigma(effective_start_ms, effective_end_ms, true);
                    hc.th_gyro = dth_imu;
                    ++cyc_imu_segs_;
                    ++imu_segments;
                }
                else
                {
                    dtheta = wheel_raw * k_g - b_g * dt + curve;
                    sig_om_fused = src.wheel_sigma(odom.var_rot);
                    hc.th_gyro = wheel_raw;           // the shared scale acted on the wheels here
                    ++cyc_wheel_segs_;
                    ++wheel_segments;
                }
                hc.t_gyro    = dt;
                hc.fwd_wheel = dy_local;
            }
            const float rot_eff = dt > 0.f ? dtheta / dt : 0.f;
            cyc_imu_dtheta_   += w_g * dtheta;            // the heading's gyro-attributed share
            cyc_wheel_dtheta_ += (1.f - w_g) * dtheta;    // and the wheels'; they sum to dtheta
            cyc_heading_cov_.th_gyro  += hc.th_gyro;  cyc_heading_cov_.t_gyro    += hc.t_gyro;
            cyc_heading_cov_.th_wheel += hc.th_wheel; cyc_heading_cov_.fwd_wheel += hc.fwd_wheel;
            cyc_gyro_w_dt_ += static_cast<double>(w_g) * dt; cyc_heading_dt_ += dt;
            gyro_w_dt_sum_ += static_cast<double>(w_g) * dt; heading_dt_sum_ += dt;
        return {dtheta, w_g, sig_om_fused, rot_eff};
    }

    void RoomConcept::report_odometry_channels(const OdomSources &src,
                const std::vector<OdometryReading> &odometry_history, std::int64_t t_end_ms,
                int imu_segments, int wheel_segments, int odom_var_segments)
    {
        // ── Proof of life, then periodic health ────────────────────────────────────────────────────
        // A channel that silently fails to bind is a recurring failure mode here, and an unused or
        // degrading IMU is invisible in every outcome metric until it has already cost accuracy. The
        // one-shot says it bound; the periodic line says it is STILL bound, how much of each interval
        // the IMU actually covers, and how much heading it is removing from the wheel estimate.
        imu_seg_used_  += imu_segments;
        imu_seg_total_ += imu_segments + wheel_segments;
        imu_stats_sim_clock_ = src.use_sim_clock;

        // Proof of life for the wheel-variance channel. It must be printed from the segment counter
        // and not from "the flag is on", because every way this can fail -- producer silent, producer
        // says "unknown", period unmeasurable, attribute never registered -- ends in the same silent
        // fallback to the constants. A flag that is on and a channel that is working are different
        // claims, and only the second one is worth logging.
        if (not odom_var_announced_ and odom_var_segments > 0)
        {
            odom_var_announced_ = true;
            const auto& o = odometry_history.back();
            const float s_lat  = src.wheel_sigma(o.var_side);
            const float s_long = src.wheel_sigma(o.var_adv);
            const float s_om   = src.wheel_sigma(o.var_rot);
            qInfo().nospace() << "[OdomVar] wheels' stated variance ACTIVE | dt_sample="
                              << QString::number(src.odom_dt_sample * 1e3, 'f', 1) << " ms"
                              << " | var(adv,side,rot)=" << o.var_adv << "," << o.var_side << "," << o.var_rot
                              << " -> sigma(lat,long,om)="
                              << QString::number(s_lat,  'g', 3) << ","
                              << QString::number(s_long, 'g', 3) << ","
                              << QString::number(s_om,   'g', 3)
                              << " vs model " << params.odom_preint_noise.sigma_v_lat << ","
                              << params.odom_preint_noise.sigma_v_long << ","
                              << params.odom_preint_noise.sigma_omega
                              << " | " << odom_var_segments << "/" << (imu_segments + wheel_segments)
                              << " segments";
        }

        if (not imu_injection_announced_ and imu_segments > 0)
        {
            imu_injection_announced_ = true;
            qInfo().nospace() << "[ImuInject] gyro heading ACTIVE | clock="
                              << (src.use_sim_clock ? "SIM (producer)" : "WALL — sim map NOT bound")
                              << " | " << imu_segments << "/" << (imu_segments + wheel_segments)
                              << " segments from IMU";
        }

        // Per-WINDOW coverage, mirrored for the debug log. The 5 s stats above reset themselves, so
        // they cannot be sampled per row — and without a coverage column the rotation channel is
        // uninterpretable: a scale that fails to recover looks identical whether the estimator is
        // wrong or the injected channel simply was not the one in use. That ambiguity already cost
        // one full recovery run.
        last_imu_cover_ = (imu_segments + wheel_segments) > 0
                        ? static_cast<float>(imu_segments) / static_cast<float>(imu_segments + wheel_segments)
                        : -1.f;

        if (imu_stats_last_log_ms_ == 0)
            imu_stats_last_log_ms_ = t_end_ms;
        else if (t_end_ms - imu_stats_last_log_ms_ >= 5000 and imu_seg_total_ > 0)
        {
            const double cover = 100.0 * imu_seg_used_ / imu_seg_total_;
            // Ratio of what the wheels would have said to what the gyro said, over the same segments.
            // Expect ~1.05-1.08 on this robot: a differential base over-reports rotation because it
            // turns by scrubbing. A ratio pinned at 1.000 means the gyro is agreeing suspiciously
            // exactly -- more likely the same source twice than two sensors agreeing.
            // ★ THE GUARD MUST BE A MEANINGFUL ROTATION, NOT A NON-ZERO ONE. At 1e-4 rad this printed
            // a ratio for a robot standing still: observed live, "wheel/gyro=-9.6824 over -0.001 rad"
            // — one milliradian of true rotation, so the quotient was 0/0 and the number was noise
            // wearing four decimal places. It is worse now that the wheel channel carries injected
            // noise: over a 5 s window a parked robot accumulates sigma_w*sqrt(T) = 0.010*sqrt(5) =
            // 0.022 rad of random walk against ~0 of signal, so a parked window CANNOT produce a
            // meaningful ratio however long it runs. 0.5 rad is roughly 30 degrees — enough that the
            // scrubbing error being measured is well clear of the noise floor.
            constexpr double kMinRotForRatio = 0.5;   // rad
            const bool ratio_valid = std::abs(imu_dtheta_sum_) > kMinRotForRatio;
            const double ratio = ratio_valid ? wheel_dtheta_sum_ / imu_dtheta_sum_
                                             : std::numeric_limits<double>::quiet_NaN();
            last_wheel_gyro_ratio_ = static_cast<float>(ratio);   // NaN while the guard is unmet
            qInfo().nospace() << "[ImuInject] clock="
                              << (imu_stats_sim_clock_ ? "SIM" : "WALL(unbound)")
                              << " coverage=" << QString::number(cover, 'f', 1) << "%"
                              << " (" << imu_seg_used_ << "/" << imu_seg_total_ << " seg)"
                              << " dtheta wheel/gyro="
                              << (ratio_valid ? QString::number(ratio, 'f', 4)
                                              : QString("n/a (needs >%1 rad of turning)")
                                                    .arg(kMinRotForRatio, 0, 'f', 1))
                              << " over " << QString::number(imu_dtheta_sum_, 'f', 3) << " rad"
                              << " zupt=" << zupt_segs_ << "/" << imu_seg_total_ << " seg"
                              << " gyro_weight="
                              << (heading_dt_sum_ > 0.0
                                      ? QString::number(100.0 * gyro_w_dt_sum_ / heading_dt_sum_, 'f', 1) + "%"
                                      : QString("n/a"))
                              << (params.heading_fusion ? " (fused)" : " (switch)");
            imu_stats_last_log_ms_ = t_end_ms;
            imu_seg_used_ = imu_seg_total_ = 0;
            imu_dtheta_sum_ = wheel_dtheta_sum_ = 0.0;
            zupt_segs_ = 0;
            gyro_w_dt_sum_ = heading_dt_sum_ = 0.0;
        }
    }

} // namespace rc
