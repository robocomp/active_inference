// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// Camera mount: kinematic mount observation, pooled mount solve, calibration service.

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

    // ── r2 KINEMATIC MOUNT FACTOR (mount_factors.h; plan docs/superpowers/plans/2026-10-08-lidar-mounts-kinematic-floor.md
    //    Tasks 1, 1b, 5; r2.2 items 1, 4). Called from observe_innovation on `fed` cycles only. ──────────────
    // T's columns are (FORWARD, LATERAL, heading) in world coordinates, so T.col(0).dot(delta) is the forward
    // innovation and odom_body = (forward, lateral, dtheta) -- named fields in Cycle, never a positional pair.
    // Row variance = the noise learner's model variance for the axis this cycle (last_m), weight 1 - its outlier
    // responsibility (last_r); the odometry calibration's posterior / applied values / re-solve counter feed the
    // common-mode Schur over (k_v, k_lat).
    void RoomConcept::observe_kinematic_mount(const Eigen::Vector3f &delta, const Eigen::Matrix3f &T,
                                              const Eigen::Vector3f &odom_body, UpdateResult &res)
    {
        using rc::calib::P_K_V;
        using rc::calib::P_K_LAT;
        // ── the odometry calibration as it stands: applied (acting) and estimated (taught) values, posterior ──
        const auto &sol = motion_calib_.last_solve();
        const Eigen::Vector2d applied(motion_calib_.acting(P_K_V) ? double(sol.value[P_K_V]) : 0.0,
                                      motion_calib_.acting(P_K_LAT) ? double(sol.value[P_K_LAT]) : 0.0);
        const Eigen::Vector2d estimate(motion_calib_.taught(P_K_V) ? double(sol.value[P_K_V]) : 0.0,
                                       motion_calib_.taught(P_K_LAT) ? double(sol.value[P_K_LAT]) : 0.0);
        Eigen::Matrix2d Sig = Eigen::Matrix2d::Constant(std::numeric_limits<double>::quiet_NaN());   // NaN = keep
        if (const auto &inf = motion_calib_.last_information(); inf.episodes > 0)
        {
            const Eigen::Matrix<double, rc::calib::P_COUNT, rc::calib::P_COUNT> Hd = inf.H.cast<double>();
            const Eigen::LDLT<Eigen::Matrix<double, rc::calib::P_COUNT, rc::calib::P_COUNT>> ldlt(Hd);
            if (ldlt.info() == Eigen::Success)
            {
                const auto C = ldlt.solve(Eigen::Matrix<double, rc::calib::P_COUNT, rc::calib::P_COUNT>::Identity()).eval();
                Sig << C(P_K_V, P_K_V), C(P_K_V, P_K_LAT), C(P_K_LAT, P_K_V), C(P_K_LAT, P_K_LAT);
            }
        }
        kin_mount_.set_odometry(Sig, applied, estimate - applied, motion_calib_.episodes());
        rc::mountf::KinematicMount::Cycle c;
        c.innov_fwd = double(T.col(0).dot(delta));
        c.innov_lat = double(T.col(1).dot(delta));
        c.d_fwd = double(odom_body[0]); c.d_lat = double(odom_body[1]); c.dth = double(odom_body[2]);
        c.var_fwd = noise_innov_.last_m(0);  c.var_lat = noise_innov_.last_m(1);
        c.w_fwd = 1.0 - noise_innov_.last_r(0);  c.w_lat = 1.0 - noise_innov_.last_r(1);
        kin_mount_.observe(c);
        // ── per-cycle replay log: what the factor saw, plus the ACTING eps (Task 6 must show it is 0) ──
        if (not kin_csv_.is_open())
        {
            std::filesystem::create_directories("tmp/mount_factors");
            char stamp[32]; const std::time_t now = std::time(nullptr);
            std::strftime(stamp, sizeof stamp, "%Y-%m-%d_%H-%M-%S", std::localtime(&now));
            kin_csv_.open(std::string("tmp/mount_factors/kin_") + stamp + ".csv");
            kin_csv_.imbue(std::locale::classic());
            kin_csv_ << "ts_ms,innov_fwd,innov_lat,odom_fwd,odom_lat,odom_th,var_fwd,var_lat,w_fwd,w_lat,"
                        "yaw_offset,kv_applied,klat_applied,mu_kv,mu_klat,sig_kv,sig_klat,episodes,"
                        "dx,dy,psi,dx_sd,dy_sd,psi_sd\n";
        }
        const Eigen::Vector3d m = kin_mount_.mean(), sd = kin_mount_.sigma();
        const Eigen::Matrix2d So = kin_mount_.sigma_odometry();
        kin_csv_ << res.timestamp_ms << ',' << c.innov_fwd << ',' << c.innov_lat << ',' << c.d_fwd << ',' << c.d_lat << ','
                 << c.dth << ',' << c.var_fwd << ',' << c.var_lat << ',' << c.w_fwd << ',' << c.w_lat << ','
                 << motion_calib_.yaw_offset() << ',' << applied[0] << ',' << applied[1] << ','
                 << estimate[0] - applied[0] << ',' << estimate[1] - applied[1] << ','
                 << std::sqrt(std::max(0.0, So(0, 0))) << ',' << std::sqrt(std::max(0.0, So(1, 1))) << ','
                 << motion_calib_.episodes() << ',' << m[0] << ',' << m[1] << ',' << m[2] << ','
                 << sd[0] << ',' << sd[1] << ',' << sd[2] << '\n';
    }

    /// One weighted least squares per WINDOW — the accumulators reset after every solve — with the
    /// robot's pose logged beside it.
    ///
    /// ★ IT WAS CUMULATIVE, AND THAT HID THE ANSWER. Pooling over the whole run produced an estimate
    ///   that wandered from −0.087° to +0.565° while its own standard error shrank to ±0.0044°, and
    ///   crossed zero on the way — 149x its claimed uncertainty, for a quantity that is bolted to the
    ///   robot and cannot change at all. The formal error assumes each sample is an independent draw,
    ///   and they are not: every sample taken against one wall shares whatever is wrong with THAT
    ///   wall — its position in the map, the floor height beneath it, the residual pose error there.
    ///   Two hundred samples along one wall are close to one measurement repeated, which is exactly
    ///   the "N samples are not N measurements" correction image_edge_accumulate.h applies WITHIN a
    ///   frame; pooling reintroduced it ACROSS frames where nothing was correcting it.
    ///
    /// ★ A per-window row can be pooled back into the cumulative answer offline; the cumulative row
    ///   can never be taken apart. So this form strictly dominates, and it makes the between-window
    ///   scatter — the honest uncertainty — visible instead of hidden.
    ///
    /// What to do with the output: plot dpitch_deg against pose_x/pose_y. If it tracks which wall is
    /// in view, the wandering is MAP error and the mount cannot be measured this way until the map is
    /// fixed. If the windows agree once pose is accounted for, the mount number survives.
    ///
    /// ★ THE STANDARD ERRORS ARE INFLATED BY THE OBSERVED SCATTER, and that is not a fudge. The
    ///   per-frame monitor reported se_const_px = 0.013 px while consecutive frames disagreed by
    ///   tens of pixels: it was describing the spread WITHIN one frame's four samples and knew
    ///   nothing about the frames disagreeing. chi2/dof measures exactly that discrepancy — how much
    ///   larger the residuals are than the sigmas claim — so scaling the parameter sigma by
    ///   sqrt(chi2/dof) reports the precision the data actually supports. When the residual model is
    ///   right chi2/dof -> 1 and the inflation vanishes on its own.
    void RoomConcept::mount_pooled_solve(std::int64_t timestamp_ms)
    {
        if (mnt_win_start_ms_ == 0) { mnt_win_start_ms_ = timestamp_ms; return; }
        const std::int64_t win_ms = timestamp_ms - mnt_win_start_ms_;
        if (win_ms < 5000) return;
        // Both conditions, and NEITHER resets on its own: a window short of samples keeps
        // accumulating rather than being solved thin and reported as if it were a measurement. The
        // window length is therefore variable and is logged, so a row can never be misread as
        // covering 5 s when it covered thirty.
        // ★ EITHER monitor having enough samples opens the window. The two draw on DIFFERENT
        // populations — the fy/d fit on floor junctions only, the translation fit on every class —
        // so gating both on the first would silence the translation fit in exactly the
        // corners-only configuration (useFloorJunction = false) where it is the only one of the two
        // still able to answer.
        if (mnt_n_ < 200 and mnt_tn_ < 200) return;

        // ── Monitor 1: b_const + b_invd * (fy/d), floor junctions ────────────────────────────────
        // ★ NaN, not 0, when the window could not be solved. A 0 here would read as "measured no
        // pitch error" when it means "never asked", and that conflation has cost this codebase real
        // time before. NaN survives the CSV round trip and cannot be averaged by accident.
        const double NA = std::numeric_limits<double>::quiet_NaN();
        double rho = NA, cond = NA, b_const = NA, b_invd = NA, c2d = NA, se_const = NA, se_invd = NA;
        double dpitch_deg = NA, se_pitch_deg = NA;
        const double det  = mnt_S11_ * mnt_S22_ - mnt_S12_ * mnt_S12_;
        const bool   m_ok = mnt_n_ >= 200 and det > 0.0 and std::isfinite(det);
        if (m_ok)
        {

        // ── COULD THIS WINDOW SEPARATE PITCH FROM HEIGHT AT ALL? ─────────────────────────────────
        // The two parameters are told apart ONLY by how their covariates scale with range: pitch is
        // constant in pixels, height goes as fy/d. If every sample in the window sat at the same
        // distance, x2 = fy/d would be constant, S12 would equal S11*x2 exactly, and the normal
        // matrix would be singular — the fit would still return two numbers, and they would be
        // meaningless, trading off along the degenerate direction. That is what was happening: the
        // per-window pitch estimate wandered -0.087..+0.565 deg while its own standard error shrank
        // to +/-0.0044, and the worst rows were exactly the sparse, narrow-range ones.
        //
        // ★ CORRELATION-NORMALISED, following calibration_estimator.h, and for the same reason it
        // records: on the RAW matrix the units span orders of magnitude regardless of geometry, and
        // the raw number once ranked a separable window as WORSE than a deliberately collinear one.
        // Normalising by the diagonal leaves a 2x2 with 1s on it and the correlation off it, whose
        // eigenvalues are 1 +/- |rho|. So for this fit the whole diagnostic is one number:
        //     rho  = S12 / sqrt(S11*S22)        how collinear the two covariates were
        //     cond = (1+|rho|) / (1-|rho|)
        // rho -> 1 means the window saw one range and cannot answer; it is a direct readout of the
        // RANGE DIVERSITY the trajectory happened to supply.
        // ★ It is REPORTED, not gated. "The estimate wanders" and "this window was never able to
        // answer" are different facts, and a filter that silently dropped the second would leave the
        // first looking like noise.
            rho  = mnt_S12_ / std::sqrt(std::max(1e-300, mnt_S11_ * mnt_S22_));
            const double arho = std::min(std::abs(rho), 1.0 - 1e-12);
            cond = (1.0 + arho) / (1.0 - arho);
            b_const = ( mnt_S22_ * mnt_Sy1_ - mnt_S12_ * mnt_Sy2_) / det;
            b_invd  = (-mnt_S12_ * mnt_Sy1_ + mnt_S11_ * mnt_Sy2_) / det;
            // Weighted residual sum of squares of the FITTED model, so chi2/dof is a statement about
            // the part the two mount parameters could not explain.
            const double chi2 = std::max(0.0, mnt_Syy_ - (b_const * mnt_Sy1_ + b_invd * mnt_Sy2_));
            const double dof  = std::max(1.0, static_cast<double>(mnt_n_) - 2.0);
            c2d  = chi2 / dof;
            const double infl = std::sqrt(std::max(1.0, c2d));
            se_const = std::sqrt(mnt_S22_ / det) * infl;
            se_invd  = std::sqrt(mnt_S11_ / det) * infl;
            const double fy = last_mount_fy_ > 0.f ? last_mount_fy_ : 1.f;
            dpitch_deg   = (b_const / fy) * 180.0 / M_PI;
            se_pitch_deg = (se_const / fy) * 180.0 / M_PI;
        }

        // ── Monitor 2: tx*nx + ty*ny — the rigid image displacement of the PREDICTION ────────────
        // Sign: r = n_hat . (uv_pred - uv_meas), so a positive tx means the projected room sits to
        // the RIGHT of the edge actually found, and positive ty means it sits BELOW.
        // Interpretation is deliberately NOT unique and must not be reported as if it were: an
        // image translation is produced identically by a principal-point offset, a boresight yaw
        // (tx ~ fx*dyaw) and a mount pitch (ty ~ fy*dpitch). What it settles is the question the
        // fy/d fit cannot even pose — whether a rigid pointing error is present AT ALL.
        double tx = NA, ty = NA, se_tx = NA, se_ty = NA, c2d_t = NA, rho_t = NA, cond_t = NA;
        const double det_t = mnt_T11_ * mnt_T22_ - mnt_T12_ * mnt_T12_;
        const bool   t_ok  = mnt_tn_ >= 200 and det_t > 0.0 and std::isfinite(det_t);
        if (t_ok)
        {
            // Same correlation-normalised conditioning as monitor 1, and here it reads the SPREAD
            // OF CONTOUR ORIENTATIONS rather than of ranges: rho_t -> +/-1 means every normal in
            // the window pointed the same way (all floor junctions, or all corners) and the two
            // components only trade off. Reported, never gated.
            rho_t = mnt_T12_ / std::sqrt(std::max(1e-300, mnt_T11_ * mnt_T22_));
            const double ar = std::min(std::abs(rho_t), 1.0 - 1e-12);
            cond_t = (1.0 + ar) / (1.0 - ar);
            tx = ( mnt_T22_ * mnt_Tx_ - mnt_T12_ * mnt_Ty_) / det_t;
            ty = (-mnt_T12_ * mnt_Tx_ + mnt_T11_ * mnt_Ty_) / det_t;
            const double chi2_t = std::max(0.0, mnt_Tyy_ - (tx * mnt_Tx_ + ty * mnt_Ty_));
            c2d_t = chi2_t / std::max(1.0, static_cast<double>(mnt_tn_) - 2.0);
            const double infl_t = std::sqrt(std::max(1.0, c2d_t));
            se_tx = std::sqrt(mnt_T22_ / det_t) * infl_t;
            se_ty = std::sqrt(mnt_T11_ / det_t) * infl_t;
        }

        // ── Monitor 3 / STAGE 1: the four mount nuisances solved as parameters ───────────────────
        // ★ SIGN. The residual is r = n_hat.(uv_pred(0) - uv_meas) and h = d(n_hat.uv_pred)/d(nuisance).
        //   The measurement is the projection under the TRUE mount, so n_hat.uv_meas =
        //   n_hat.uv_pred(0) + h*x_true, giving r = -h*x_true. A least-squares fit of r on h therefore
        //   returns MINUS the parameter: the physical value, and the correction to apply, is -x.
        //   Self-checking: mountYawCorrection is already applied, so p_yaw must now read ~0. If it
        //   reads about twice the applied correction instead, this sign is inverted.
        // ★ The prior is the IDENTITY because h carries sigma_i (see the header note).
        Eigen::Vector4d mp = Eigen::Vector4d::Constant(NA), msig = Eigen::Vector4d::Constant(NA);
        double c2d_h = NA, cond_h = NA, rho_h = 0.0;
        int    informed_mask = 0, rho_i = 0, rho_j = 1;
        const bool h_ok = mnt_hn_ >= 200;
        if (h_ok)
        {
            const Eigen::Matrix4d A = mnt_H_ + Eigen::Matrix4d::Identity();
            const Eigen::Matrix4d C = A.inverse();
            if (C.allFinite())
            {
                const Eigen::Vector4d x = C * mnt_b_;
                // mnt_Tyy_ is the same weighted sum of squares over the same samples.
                const double chi2_h = std::max(0.0, mnt_Tyy_ - x.dot(mnt_b_));
                c2d_h = chi2_h / std::max(1.0, static_cast<double>(mnt_hn_) - 4.0);
                const double infl_h = std::sqrt(std::max(1.0, c2d_h));
                mp = -x;
                for (int i = 0; i < 4; ++i)
                {
                    msig(i) = std::sqrt(std::max(0.0, C(i, i))) * infl_h;
                    // `informed` = the data shrank this parameter's uncertainty below 0.9x its prior.
                    // Per-parameter, never global: three of these can be answered while the fourth is
                    // pure prior, and a global flag would licence acting on the one that is not.
                    if (msig(i) < 0.9) informed_mask |= (1 << i);
                }
                // ── COULD THESE FOUR BE TOLD APART AT ALL? ───────────────────────────────────
                // ★ `informed` PER PARAMETER CANNOT SEE A DEGENERATE PAIR. A strong prior shrinks
                //   the posterior sigma even along a direction the data never constrained, so a
                //   parameter can read INFORMED while only its COMBINATION with another is
                //   determined. Pitch and height are separated solely by the fy/d covariate — the
                //   same two columns whose 2-parameter fit sits at rho 0.9864 / cond 146 standing
                //   still. Reporting them individually without this would repeat, in a 4x4, exactly
                //   the error the fy/d monitor already made.
                // Correlation-normalised, per calibration_estimator.h: on the raw matrix the units
                // span orders of magnitude regardless of geometry, and the raw number once ranked a
                // separable window as WORSE than a deliberately collinear one.
                Eigen::Matrix4d R = Eigen::Matrix4d::Identity();
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        R(i, j) = C(i, j) / std::sqrt(std::max(1e-300, C(i, i) * C(j, j)));
                const Eigen::Vector4d ev = Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>(R)
                                               .eigenvalues();
                cond_h = ev(3) / std::max(1e-12, ev(0));
                for (int i = 0; i < 4; ++i)              // worst pair, named rather than implied
                    for (int j = i + 1; j < 4; ++j)
                        if (std::abs(R(i, j)) > std::abs(rho_h))
                        { rho_h = R(i, j); rho_i = i; rho_j = j; }
                mnt_p_sum_  += mp;
                mnt_p_sum2_ += mp.cwiseProduct(mp);
                ++mnt_p_n_;
            }
        }

        if (not mount_csv_.is_open())
        {
            mount_csv_.open("etc/image_edge_mount.csv", std::ios::out | std::ios::trunc);
            if (mount_csv_.is_open())
            {
                mount_csv_.imbue(std::locale::classic());   // CLAUDE.md: never a comma decimal
                mount_csv_ << "ts_ms,window_ms,n_samples,chi2_per_dof,b_const_px,se_const_px,"
                              "b_invd_m,se_invd_m,dpitch_deg,se_pitch_deg,"
                              "pose_x,pose_y,pose_theta,rho,cond,"
                              "t_n,tx_px,se_tx_px,ty_px,se_ty_px,chi2_t,rho_t,cond_t,"
                              "h_n,p_pitch_rad,p_height_m,p_yaw_rad,p_dt,"
                              "s_pitch,s_height,s_yaw,s_dt,chi2_h,informed,cond_h,rho_h\n";
            }
        }
        if (mount_csv_.is_open())
        {
            mount_csv_ << timestamp_ms << ',' << win_ms << ',' << mnt_n_ << ',' << c2d << ','
                       << b_const << ',' << se_const << ',' << b_invd << ',' << se_invd << ','
                       << dpitch_deg << ',' << se_pitch_deg << ','
                       << mnt_pose_x_ << ',' << mnt_pose_y_ << ',' << mnt_pose_th_ << ','
                       << rho << ',' << cond << ','
                       << mnt_tn_ << ',' << tx << ',' << se_tx << ',' << ty << ',' << se_ty << ','
                       << c2d_t << ',' << rho_t << ',' << cond_t << ','
                       // Parameters written in PHYSICAL units; sigmas stay in units of the prior,
                       // where 1.0 means "the data said nothing" and <0.9 is `informed`.
                       << mnt_hn_ << ','
                       << mp(0) * params.image_edge.mount_pitch_sigma  << ','
                       << mp(1) * params.image_edge.mount_height_sigma << ','
                       << mp(2) * params.image_edge.mount_yaw_sigma    << ',' << mp(3) << ','
                       << msig(0) << ',' << msig(1) << ',' << msig(2) << ',' << msig(3) << ','
                       << c2d_h << ',' << informed_mask << ',' << cond_h << ',' << rho_h << '\n';
            mount_csv_.flush();
        }
        // Between-window scatter, which is the uncertainty that turned out to matter. Kept as a
        // running mean and sum of squares so the log line can say how much the WINDOWS disagree
        // beside how much each window claims to know — the two differed by 149x in the pooled form.
        ++mnt_wins_;
        if (m_ok) { mnt_pitch_sum_ += dpitch_deg; mnt_pitch_sum2_ += dpitch_deg * dpitch_deg; ++mnt_pitch_n_; }
        // Report the WITHIN-window error beside the BETWEEN-window scatter. If the second is much
        // larger than the first, the samples are not independent and the first is meaningless — say
        // so on the line rather than leaving a reader to discover it from a table of rows.
        double spread = 0.0;
        if (mnt_pitch_n_ >= 2)
        {
            const double m = mnt_pitch_sum_ / mnt_pitch_n_;
            spread = std::sqrt(std::max(0.0, mnt_pitch_sum2_ / mnt_pitch_n_ - m * m));
        }
        qInfo().nospace().noquote() << "[mount] window " << mnt_wins_ << " (" << win_ms << " ms, "
                          << mnt_n_ << " samples) | pitch "
                          << QString::number(dpitch_deg, 'f', 4) << " +/- "
                          << QString::number(se_pitch_deg, 'f', 4) << " deg within"
                          << (mnt_pitch_n_ >= 2
                              ? QString(", %1 deg BETWEEN windows (%2x)")
                                    .arg(spread, 0, 'f', 4)
                                    .arg(spread / std::max(1e-9, se_pitch_deg), 0, 'f', 0)
                              : QString())
                          << " | height " << QString::number(b_invd, 'f', 4)
                          << " | chi2/dof " << QString::number(c2d, 'f', 2)
                          << " | rho " << QString::number(rho, 'f', 4)
                          << " cond " << QString::number(cond, 'f', 0)
                          << (m_ok and cond > 50.0
                                  ? "  <- DEGENERATE: one range in view, pitch and height are not"
                                    " separable here and these two numbers only trade off"
                                  : m_ok ? "" : "  <- NOT SOLVED (too few floor-junction samples)")
                          << " | pose " << QString::number(mnt_pose_x_, 'f', 2) << ","
                          << QString::number(mnt_pose_y_, 'f', 2)
                          << (m_ok and c2d > 4.0
                                  ? "  <- residual model is WRONG; the estimate inherits it" : "");
        // Monitor 2 on its OWN line. It answers a different question from a different population,
        // and folding it into the line above would invite reading one window's two fits as one
        // estimate of one thing.
        if (t_ok)
            qInfo().nospace().noquote() << "[mount/shift] window " << mnt_wins_ << " (" << mnt_tn_
                              << " samples, all classes) | tx "
                              << QString::number(tx, 'f', 3) << " +/- "
                              << QString::number(se_tx, 'f', 3) << " px | ty "
                              << QString::number(ty, 'f', 3) << " +/- "
                              << QString::number(se_ty, 'f', 3) << " px | |t| "
                              << QString::number(std::hypot(tx, ty), 'f', 3) << " px"
                              << " | chi2/dof " << QString::number(c2d_t, 'f', 2)
                              << " | rho " << QString::number(rho_t, 'f', 4)
                              << " cond " << QString::number(cond_t, 'f', 1)
                              << (cond_t > 50.0
                                      ? "  <- one contour ORIENTATION in view; tx and ty are not"
                                        " separable here and only trade off"
                                      : "")
                              << (std::hypot(tx, ty) > 3.0 * std::max(se_tx, se_ty)
                                      ? "  <- SIGNIFICANT rigid displacement: pointing or principal"
                                        " point, NOT distinguishable from each other by this fit"
                                      : "  <- no rigid displacement resolved");
        else
            qInfo().nospace().noquote() << "[mount/shift] window " << mnt_wins_ << ": only " << mnt_tn_
                              << " weighted samples (need 200) — NOT solved, which is not the same"
                                 " as 'no displacement found'";

        // ── Monitor 3 line: the calibration itself ───────────────────────────────────────────────
        if (h_ok and std::isfinite(mp(0)))
        {
            const double sig[4] = {params.image_edge.mount_pitch_sigma,
                                   params.image_edge.mount_height_sigma,
                                   params.image_edge.mount_yaw_sigma, 1.0};
            const char*  nm[4]  = {"pitch", "height", "yaw", "dt"};
            const char*  un[4]  = {"deg", "m", "deg", "x"};
            QString body;
            for (int i = 0; i < 4; ++i)
            {
                double v = mp(i) * sig[i];
                if (i == 0 or i == 2) v *= 180.0 / M_PI;          // report angles in degrees
                // Between-window scatter is the honest uncertainty on any parameter that cannot be
                // separated within a window (yaw vs heading); the within-window sigma is not.
                double spread = 0.0;
                if (mnt_p_n_ >= 2)
                {
                    const double m = mnt_p_sum_(i) / mnt_p_n_;
                    spread = std::sqrt(std::max(0.0, mnt_p_sum2_(i) / mnt_p_n_ - m * m)) * sig[i];
                    if (i == 0 or i == 2) spread *= 180.0 / M_PI;
                }
                body += QString(" | %1 %2 %3 (%4 prior sig%5)")
                            .arg(nm[i]).arg(v, 0, 'f', i == 1 ? 4 : 4).arg(un[i])
                            .arg(msig(i), 0, 'f', 3)
                            .arg((informed_mask >> i) & 1 ? ", INFORMED" : "");
                if (mnt_p_n_ >= 2)
                    body += QString(" [%1 between windows]").arg(spread, 0, 'f', 4);
            }
            qInfo().nospace().noquote()
                << "[mount/calib] window " << mnt_wins_ << " (" << mnt_hn_ << " samples)"
                << body << " | chi2/dof " << QString::number(c2d_h, 'f', 2)
                << " | cond " << QString::number(cond_h, 'f', 1)
                << " (worst pair " << nm[rho_i] << "/" << nm[rho_j] << " rho "
                << QString::number(rho_h, 'f', 4) << ")"
                << (cond_h > 50.0
                        ? "   <- DEGENERATE: that pair trades off; only their COMBINATION is"
                          " determined, and a per-parameter INFORMED flag cannot see this"
                        : "")
                << (informed_mask == 0
                        ? "   <- nothing INFORMED yet: every posterior is still its prior"
                        : "");
        }

        // ── RESET: every row is an independent window ────────────────────────────────────────────
        mnt_S11_ = mnt_S12_ = mnt_S22_ = mnt_Sy1_ = mnt_Sy2_ = mnt_Syy_ = 0.0;
        mnt_T11_ = mnt_T12_ = mnt_T22_ = mnt_Tx_  = mnt_Ty_  = mnt_Tyy_ = 0.0;
        mnt_H_.setZero(); mnt_b_.setZero(); mnt_hn_ = 0;
        mnt_n_ = 0;
        mnt_tn_ = 0;
        mnt_win_start_ms_ = timestamp_ms;
    }

    /// Calibration bookkeeping, run every cycle: apply a pending reset, restore the window once,
    /// drain any closed pivots into the estimator, and keep the window persisted.
    ///
    /// ★ UNGATED, and that matters. All of this used to live inside adapt_motion_model(), which was
    /// gated by LearnMotionModel — a flag switched off in 2026-08 to disable a DIFFERENT learner (the
    /// EMA slip-k path, A/B'd and rejected, now deleted). One switch was therefore turning off two
    /// unrelated things, and the batch estimator's whole bookkeeping went with it: closures reached
    /// nothing, the state file was never written, and nothing said so.
    void RoomConcept::service_calibration()
    {
        // ── Reset, if the calibration window asked for one ────────────────────────────────────────
        if (calib_reset_pending_.exchange(false))
        {
            motion_calib_.reset_state(params.calib_state_file);
            qInfo() << "[calib] RESET: window emptied and" 
                    << QString::fromStdString(params.calib_state_file)
                    << "deleted. Every parameter is back at its prior and reports NOT informed, which"
                       " is the honest state of a robot that has just been told to un-learn.";
        }

        // ── Restore the window once, then keep it saved ───────────────────────────────────────────
        // Evidence, not parameters: the solve after a restart is the solve it would have been had the
        // run never stopped, with the prior still at zero and nothing ratcheting.
        if (not calib_state_loaded_)
        {
            calib_state_loaded_ = true;
            const std::size_t n = motion_calib_.load_state(params.calib_state_file);
            if (n > 0)
                qInfo().nospace() << "[calib] restored " << n << " measurements from "
                                  << QString::fromStdString(params.calib_state_file)
                                  << " (" << motion_calib_.closures() << " closed pivot(s)) — the "
                                     "window resumes, the priors do not move";
            // Localiser thread: rc::status::event only pushes into the stream's queue.
            rc::status::event("loaded", rc::status::Obj{}
                                            .s("what", "motion_calib")
                                            .s("path", params.calib_state_file)
                                            .s("detail", std::format("{} measurements, {} closed pivot(s){}", n,
                                                                     motion_calib_.closures(),
                                                                     n > 0 ? "" : " (nothing on disk: priors)")));
            if (const std::size_t d = motion_calib_.legacy_dropped(); d > 0)
                qWarning().nospace() << "[calib] DROPPED " << d << " episode rows from "
                                     << QString::fromStdString(params.calib_state_file)
                                     << " — written before the heading became two factors (one shared "
                                        "heading covariate, bias of opposite sign); no exact split exists, "
                                        "so the window re-earns them";
        }

        // Drain any closed pivots first, on THIS thread, before the episode path runs. A closure is
        // the strongest measurement of the rotation model this robot can make — no map, no survey,
        // no localiser in the number — and until 2026-08-26 it reached nothing: the pivot measured
        // the scale to +/-0.18%, logged it, and the estimator that PRICES the manoeuvre never heard.
        // So the offer's value never fell and the robot pivoted for ever, unable to extinguish its
        // own worth. Feeding it back is what closes that loop.
        {
            std::vector<ClosureObs> pending;
            {
                std::scoped_lock lk(pending_closures_mutex_);
                pending.swap(pending_closures_);
            }
            for (const auto& c : pending)
            {
                motion_calib_.observe_closure(c.truth_rad, c.turned_rad, c.rate_rad_s, c.sigma_s);
                // Say whether the closure actually TAUGHT k_omega, not just that it arrived. A
                // parameter's value cannot distinguish "measured" from "left where the prior put
                // it", and the accessors now apply it only once it is taught — so this is the line
                // that says whether the pivot changed the robot's behaviour or only its logs.
                qInfo().nospace() << "[calib] CLOSURE -> batch estimator | truth "
                                  << QString::number(c.truth_rad * 180.0 / M_PI, 'f', 1) << " deg, odometry "
                                  << QString::number(c.turned_rad * 180.0 / M_PI, 'f', 1) << " deg, rate "
                                  << QString::number(c.rate_rad_s, 'f', 3) << " rad/s, sigma "
                                  << QString::number(c.sigma_s * 100.0, 'f', 3) << "% | k_omega now "
                                  << QString::number((motion_calib_.omega_scale() - 1.f) * 100.f, 'f', 4)
                                  << "% +/- " << QString::number(motion_calib_.k_w_sigma() * 100.f, 'f', 4)
                                  << "% over " << motion_calib_.closures() << " closure(s), "
                                  << (motion_calib_.taught(rc::calib::P_K_OMEGA)
                                          ? "TAUGHT — the scale is now applied"
                                          : "still UNTAUGHT — nominal is applied, the value is only reported");
            }
            // A closure is minutes of robot time; persist immediately rather than waiting for the
            // periodic save, so a crash between here and then cannot throw it away.
            if (not pending.empty())
                motion_calib_.save_state(params.calib_state_file);
        }

        // Periodic save of the window. Every 30 s: often enough that a crash costs little, rare
        // enough to be invisible next to a solve.
        {
            const std::int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            if (calib_state_last_save_ms_ == 0) calib_state_last_save_ms_ = now_ms;
            else if (now_ms - calib_state_last_save_ms_ >= 30000)
            {
                motion_calib_.save_state(params.calib_state_file);
                calib_state_last_save_ms_ = now_ms;
            }
        }

    }

} // namespace rc
