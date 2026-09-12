/*
 *  Copyright (C) 2026 by Pablo Bustos
 *
 *  This program is free software: you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software Foundation, either version 3 or
 *  any later version. See <http://www.gnu.org/licenses/>.
 */
#pragma once

/**
 *  The CAMERA half of the self-calibration: the mount, estimated from RGB corners against LiDAR
 *  corners with the pose FROZEN — in fact with the pose absent from the residual altogether.
 *
 *  ── Why this is a separate block from the motion parameters, not more entries in their enum ──────
 *  The two are fed by different streams (driving episodes vs image frames, so one `episodes` count
 *  cannot mean both) and share no covariate, so a merged information matrix would be block-diagonal
 *  with zero cross-terms — coupling that buys nothing and makes a camera recalibration able to
 *  invalidate the motion parameters through a shared file.
 *
 *  ★ There is ONE apparent cross-term, and it argues FOR the split rather than against it.
 *    `P_EPS_YAW` in the motion block is the body/mount yaw offset, with which a naive camera
 *    boresight would be confounded. This block is immune: it compares the camera against the LiDAR,
 *    both mounted on `body`, so any body-vs-localisation-frame yaw cancels in the residual and never
 *    reaches these parameters. That independence is real, and it is what makes two blocks correct.
 *
 *  ── What is deliberately NOT here ───────────────────────────────────────────────────────────────
 *  The per-corner detection offsets. Those are per-corner, not global, and they are the reason the
 *  pose factor was reverted on 2026-08-28 (a ~1.7 px per-corner bias converting into heading error
 *  through the corr(x,theta)=0.98 ridge). Estimating them means corners as landmarks with their own
 *  state — a different and larger design, and putting them in a global vector would be wrong.
 *
 *  ── Persistence saves EVIDENCE, never parameters ────────────────────────────────────────────────
 *  Following calibration_estimator.h: "evidence, not parameters. Delete to return to the priors."
 *  H and b ARE the sufficient statistics, so restoring them and re-adding the prior at solve time is
 *  exactly resuming the measurement. Saving the fitted values and restoring them as prior means
 *  would be a ratchet — each session would inherit the last one's answer as though it were data.
 */

#include <QDebug>
#include <charconv>
#include <iomanip>
#include <limits>
#include <cstdio>
#include <fstream>
#include <locale>
#include <string>
#include <string_view>

#include "mount_lidar_pair.h"

namespace rc::camcal
{
    enum Param : int
    {
        P_PITCH = 0,   ///< boresight pitch (rad): rotation about the camera's own x axis
        P_HEIGHT,      ///< mount height (m)
        P_YAW,         ///< boresight yaw (rad), RELATIVE TO THE LIDAR — see the note below
        P_DT,          ///< image/LiDAR offset x velocity scale (dimensionless multiplier)
        P_COUNT
    };

    [[nodiscard]] constexpr std::string_view param_name(int p)
    {
        switch (p)
        {
            case P_PITCH:  return "cam_pitch";
            case P_HEIGHT: return "cam_height";
            case P_YAW:    return "cam_yaw";
            case P_DT:     return "cam_dt";
            default:       return "?";
        }
    }
    /// Display unit and the scale that takes the internal value to it.
    [[nodiscard]] constexpr std::string_view param_unit(int p)
    {
        switch (p) { case P_PITCH: case P_YAW: return "deg"; case P_HEIGHT: return "mm";
                     case P_DT: return "x"; default: return ""; }
    }

    /// What motion or view WOULD identify this parameter. The motion block's tooltips are its most
    /// useful feature — "not asked" is a question and this answers it where it is asked — so these
    /// say what is actually true here, including where the honest answer is "this is hard".
    [[nodiscard]] constexpr std::string_view param_why(int p)
    {
        switch (p)
        {
            case P_PITCH: return
                "Rotation of the camera about its own x axis. Identified by corners at DIFFERENT "
                "RANGES: pitch shifts the image by a constant number of pixels while a height error "
                "shifts it by fy/d, so only a spread of distances tells the two apart. "
                "<b>This is the hard one.</b> Measured 2026-08-28: the pair holds a correlation of "
                "-0.95 to -0.98 with height in every window, and driving deliberately close to walls "
                "did NOT fix it — a LiDAR corner's positional uncertainty projects as fy/d, so the "
                "near-range corners that would separate the two carry ~0% of the fit weight "
                "(sigma_v 18-161 px under 2 m against 1.1 px at 4-6 m). The combination is "
                "determined; the split may not be.";
            case P_HEIGHT: return
                "Height of the camera above the robot frame. Shares its covariate with pitch (see "
                "the pitch note) and is determined mainly in combination with it. NOT confounded "
                "with the floor datum here, unlike the older image-vs-model monitor: this block "
                "compares the camera against the LIDAR, and both see the same physical corner "
                "without reference to the map.";
            case P_YAW: return
                "Boresight yaw — and note it is measured RELATIVE TO THE LIDAR, not absolutely. A "
                "yaw common to both mounts is invisible here and stays inside the localiser's own "
                "gauge. That is the right target (the goal is the two terms agreeing about one "
                "room) but the number must not be read as a physical bolt angle. "
                "Identified by corners at a spread of BEARINGS; it is the best-determined of the "
                "four because a corner measures bearing far better than range.";
            case P_DT: return
                "The image/LiDAR time offset multiplied by the odometry velocity scale — the two are "
                "NOT separable here, since the residual only ever sees their product. Identified by "
                "EGO-MOTION: parked it is unobservable and correctly stays at its prior, and it "
                "became informed within one window of driving starting. Compare against the motion "
                "block's k_v before correcting either.";
            default: return "";
        }
    }

    /// Thin naming + persistence layer over the pose-free accumulator. The estimation lives in
    /// rc::mount::Accum; nothing here re-implements it.
    class Estimator
    {
    public:
        /// Which camera AND which robot this evidence belongs to. Set BEFORE load().
        ///
        /// ★ THE ROBOT IS PART OF THE KEY TOO. Keying by camera alone still lets two ROBOTS share a
        ///   file: Shadow's ricoh sits at 1.275 m and P3Bot's at 1.42 m, so "the ricoh's mount" is
        ///   not one quantity across platforms. This is the same defect as the camera key, one level
        ///   up, and it would mix silently in exactly the same way.
        void set_camera(std::string name, std::string robot = "")
        { camera_ = std::move(name); robot_ = std::move(robot); }
        [[nodiscard]] const std::string& camera() const noexcept { return camera_; }
        /// Path for this (robot, camera) pair. One place, so save and load cannot disagree.
        [[nodiscard]] std::string path() const
        { return "etc/camera_calib_" + (robot_.empty() ? std::string("unknown") : robot_)
                 + "_" + camera_ + ".txt"; }

        // ── THE MOUNT THIS EVIDENCE'S `applied` IS MEASURED AGAINST ──────────────────────────────
        // ★★★ WITHOUT THIS, PUBLISHING AND RESUMING CANNOT BOTH BE CORRECT. `applied` is a total
        //     relative to a nominal mount. Read the nominal from the GRAPH at bind and it is right
        //     only until someone writes the graph — and ImageEdge.mountPublish writes exactly this
        //     edge. The next restart then binds to `nominal ⊕ applied`, pushes `applied` on top of
        //     it, and applies the correction TWICE, with the prior anchored at the doubled total: the
        //     ratchet, one correction per restart. So the nominal travels WITH the evidence.
        // ★ Two frames, because two different consumers need it. `base_*` is cam ← robot_frame, which
        //   is what the ingestor measures against. `edge_*` is the parent → camera RT edge, which is
        //   what the publish composes. Deriving one from the other needs the rest of the RT chain, so
        //   both are stored rather than reconstructed.
        void set_base(const Eigen::Matrix3f& R, const Eigen::Vector3f& t) noexcept
        { base_R_ = R; base_t_ = t; have_base_ = true; }
        [[nodiscard]] bool have_base() const noexcept { return have_base_; }
        [[nodiscard]] const Eigen::Matrix3f& base_R() const noexcept { return base_R_; }
        [[nodiscard]] const Eigen::Vector3f& base_t() const noexcept { return base_t_; }
        void set_edge_nominal(const Eigen::Vector3f& t, const Eigen::Vector3f& r) noexcept
        { edge_t_ = t; edge_r_ = r; have_edge_ = true; }
        [[nodiscard]] bool have_edge_nominal() const noexcept { return have_edge_; }
        [[nodiscard]] const Eigen::Vector3f& edge_nominal_t() const noexcept { return edge_t_; }
        [[nodiscard]] const Eigen::Vector3f& edge_nominal_r() const noexcept { return edge_r_; }
        /// Adopt a nominal that came from OUTSIDE (the robot's JSON reseeding the graph with a mount
        /// that now carries a measurement) and drop the correction the new nominal absorbs.
        /// ⚠ NEVER call this for our OWN published value: re-centring the prior on the estimator's
        ///   last answer removes the prior's pull, and publishes happen every window.
        ///
        /// ★★★ THE EVIDENCE MUST MOVE WITH THE MOUNT OR IT DESCRIBES ONE THAT NO LONGER EXISTS.
        ///     H and b were accumulated against the old base. Changing the base and zeroing `applied`
        ///     without re-referencing them leaves a pool that answers about a mount nobody has — and
        ///     with millions of pairs behind it, it answers with a certainty no window can outvote:
        ///     measured 2026-09-12, a pool at 0.00 +/- 0.03 mm of height against a fresh window
        ///     saying 52.8 +/- 7.3 mm. Accum::rebase exists for exactly this and its own note says the
        ///     pair must happen together; this used to do one alone.
        /// ★ REBASE IF THE CHANGE IS IN THE MODEL'S FAMILY, RESET IF IT IS NOT. The family is
        ///   `Rx(pitch)*Rz(yaw)` plus `height*z_cam` — the transformation the J columns are the
        ///   derivative of. A change expressible in it is re-referenced exactly and no information is
        ///   lost. A change that is NOT (a real re-mount, an x/y shift) cannot be expressed as a
        ///   correction at all, so the evidence is dropped instead of re-referenced approximately:
        ///   information measured against a mount that no longer exists is not information.
        /// ⚠ The composition of two corrections is first-order in their product (rotations do not
        ///   commute), which at these magnitudes — under a degree — is ~1e-5 rad of slop. The residual
        ///   of the decomposition is CHECKED against a tolerance rather than assumed small.
        /// @return true if the evidence was rebased, false if it was reset.
        bool adopt_external_nominal(const Eigen::Matrix3f& R1, const Eigen::Vector3f& t1,
                                    float sigma_pitch, float sigma_height, float sigma_yaw,
                                    Eigen::Vector3f* delta_out = nullptr, float* residual_out = nullptr)
        {
            if (not have_base_ or acc_.n <= 0)
            {
                // No old base to move FROM. The caller's branch states the assumption that the new
                // base is the mount the evidence already describes, so the evidence stays where it is
                // and only the anchor moves.
                // ⚠ `applied` is zeroed rather than reduced by a measured delta, because without an
                //   old base there is no delta to measure. That is the caller's stated assumption —
                //   the new base IS the mount in force — and it is exactly the assumption that was
                //   applied unconditionally before this function knew the difference.
                set_base(R1, t1); acc_.applied.setZero(); have_edge_ = false;
                if (delta_out) delta_out->setZero();
                if (residual_out) *residual_out = 0.f;
                return true;
            }
            const Eigen::Matrix3f Rc = R1 * base_R_.transpose();
            // Rx(p)*Rz(y) has a zero in (0,2); recover p and y from the same entries that build it.
            const float p = std::atan2(-Rc(1, 2), Rc(2, 2));
            const float y = std::atan2(-Rc(0, 1), Rc(0, 0));
            const Eigen::Matrix3f chk =
                (Eigen::AngleAxisf(p, Eigen::Vector3f::UnitX())
                 * Eigen::AngleAxisf(y, Eigen::Vector3f::UnitZ())).toRotationMatrix();
            const Eigen::Vector3f dt = t1 - Rc * base_t_;
            const float rot_res = (chk - Rc).cwiseAbs().maxCoeff();
            const float xy_res  = std::max(std::abs(dt.x()), std::abs(dt.y()));
            const float res = std::max(rot_res, xy_res);
            if (delta_out) *delta_out = Eigen::Vector3f(p, dt.z(), y);
            if (residual_out) *residual_out = res;
            set_base(R1, t1);
            have_edge_ = false;
            if (res < 1e-5f and sigma_pitch > 0.f and sigma_height > 0.f and sigma_yaw > 0.f)
            {
                const Eigen::Vector4d d(static_cast<double>(p) / sigma_pitch,
                                        static_cast<double>(dt.z()) / sigma_height,
                                        static_cast<double>(y) / sigma_yaw, 0.0);
                // ★★★ NO rebase HERE, AND THAT DISTINCTION IS THE WHOLE OPERATION. Accum::rebase is
                //     for when the MOUNT moves: the residuals then refer to a mount that no longer
                //     exists and b must move with it. Here the mount is untouched — only the point it
                //     is measured FROM has moved — so b is still correct and re-referencing it would
                //     subtract the change a second time. Measured by the selftest before this comment
                //     existed: rebasing drifted the estimated mount by -0.355 deg on a +0.400 deg
                //     re-reference, i.e. almost exactly twice the change.
                //     Only the accumulated total is expressed against the new base.
                acc_.applied -= d;
                return true;
            }
            acc_.reset();                // includes applied; offset_sigma_px is configuration and survives
            return false;
        }

        void add(const rc::mount::PairObs& o) { acc_.add(o); }
        /// Prior sigma on a corner's own image offset, in pixels. 0 = the nuisance is off and the
        /// solve is the old one exactly. Set it on the pool AND on every calib channel, or two
        /// cameras would be reported under two different models on the same screen.
        void set_vertex_offset_sigma_px(double px) noexcept { acc_.offset_sigma_px = px; }
        /// Apply a correction to the mount this evidence describes: re-references the evidence and
        /// moves the prior's anchor together, so the accumulated total remains the posterior mean
        /// of the error relative to the ORIGINAL graph extrinsic. See Accum::applied.
        void apply_correction(const Eigen::Vector4d& dp) { acc_.apply_correction(dp); }
        [[nodiscard]] const Eigen::Vector4d& applied() const noexcept { return acc_.applied; }
        [[nodiscard]] double vertex_offset_sigma_px() const noexcept { return acc_.offset_sigma_px; }
        void reset() { acc_.reset(); }
        [[nodiscard]] long pairs() const noexcept { return acc_.n; }
        [[nodiscard]] rc::mount::Accum::Solution solve() const { return acc_.solve(); }

        /// Sufficient statistics only. See the header note on why this is not the fitted values.
        /// ★ ATOMIC: the evidence is written to a sibling temp file and RENAMED into place. A rename
        ///   within one directory is atomic on POSIX, so a reader sees either the whole previous file
        ///   or the whole new one, and a process killed mid-save leaves the previous evidence intact
        ///   instead of a truncated file.
        ///   MEASURED 2026-09-03: reading the file during an in-place rewrite returned a header with
        ///   NO per-vertex blocks — which is exactly the shape the loader now (correctly) refuses, so
        ///   the old behaviour could turn a mistimed `kill` into a pool that refuses to resume.
        ///   ⚠ This protects against a kill and against a concurrent reader. It does NOT protect
        ///     against a power cut: there is no fsync, so the rename may reach the disk first.
        bool save(const std::string& path) const
        {
            const std::string tmp = path + ".tmp";
            {
            std::ofstream f(tmp, std::ios::out | std::ios::trunc);
            if (not f.is_open()) return false;
            f.imbue(std::locale::classic());   // CLAUDE.md: never a comma decimal separator
            // ★ FULL precision. The default 6 significant figures silently truncates H, whose
            //   entries run to 1e3 and above, and a reload then answers a slightly different
            //   question than the one that was saved — round-trip error 3.6e-8 measured before this
            //   line existed. Evidence must survive a save exactly, or resuming is not resuming.
            f << std::setprecision(std::numeric_limits<double>::max_digits10);
            f << "# camera mount calibration — EVIDENCE (H, b), not parameters.\n"
                 "# Delete to return to the priors. Restoring H and b and re-adding the prior at\n"
                 "# solve time resumes the measurement; restoring fitted VALUES would be a ratchet.\n";
            // ★ THE CAMERA IS PART OF THE EVIDENCE. These parameters describe ONE mount; body->zed
            //   is at 1.08 m and body->ricoh at 1.42 m, so summing their information matrices
            //   estimates a single mount from two different ones and the result means nothing.
            //   Recorded here as well as in the filename, so a copied or renamed file is still
            //   caught rather than silently accepted.
            f << "robot," << robot_ << '\n';
            f << "camera," << camera_ << '\n';
            f << "n," << acc_.n << '\n' << "rTr," << acc_.rTr << '\n';
            for (int i = 0; i < 4; ++i)
                for (int j = i; j < 4; ++j) f << "H," << i << ',' << j << ',' << acc_.H(i, j) << '\n';
            for (int i = 0; i < 4; ++i) f << "b," << i << ',' << acc_.b(i) << '\n';
            // ── PER-VERTEX PARTIALS (format 2) ───────────────────────────────────────────────────
            // ★ The aggregate above is kept and still written, so an older reader loads this file and
            //   gets exactly what it got before. What it CANNOT do is marginalise, and `format` is
            //   how a newer reader knows the difference — an aggregate carries no vertex, and a
            //   cluster structure cannot be recovered from a sum over clusters.
            f << "format,3\n";
            // ★ THE APPLIED CORRECTION IS PART OF THE EVIDENCE. Restoring H and b without it would
            //   resume a measurement referenced to a mount the agent no longer has, and the loop
            //   would re-apply the same correction on every restart — a ratchet across sessions
            //   rather than within one.
            f << "applied," << acc_.applied(0) << ',' << acc_.applied(1) << ','
              << acc_.applied(2) << ',' << acc_.applied(3) << '\n';
            // ── THE NOMINAL (format 3) ───────────────────────────────────────────────────────────
            // Written whenever known. Its ABSENCE is meaningful and the loader says so rather than
            // assuming the graph still holds the nominal — see load().
            if (have_base_)
            {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) f << "base_R," << i << ',' << j << ',' << base_R_(i, j) << '\n';
                f << "base_t," << base_t_.x() << ',' << base_t_.y() << ',' << base_t_.z() << '\n';
            }
            if (have_edge_)
            {
                f << "edge_t," << edge_t_.x() << ',' << edge_t_.y() << ',' << edge_t_.z() << '\n';
                f << "edge_r," << edge_r_.x() << ',' << edge_r_.y() << ',' << edge_r_.z() << '\n';
            }
            for (const auto& [vtx, v] : acc_.per_vertex)
            {
                if (v.n <= 0 or not v.finite()) continue;
                f << "V," << vtx << ",n," << v.n << '\n';
                f << "V," << vtx << ",rTr," << v.rTr << '\n';
                for (int i = 0; i < 4; ++i)
                    for (int j = i; j < 4; ++j) f << "V," << vtx << ",A," << i << ',' << j << ',' << v.A(i, j) << '\n';
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 2; ++j) f << "V," << vtx << ",c," << i << ',' << j << ',' << v.c(i, j) << '\n';
                for (int i = 0; i < 2; ++i)
                    for (int j = i; j < 2; ++j) f << "V," << vtx << ",D," << i << ',' << j << ',' << v.D(i, j) << '\n';
                for (int i = 0; i < 4; ++i) f << "V," << vtx << ",b," << i << ',' << v.b(i) << '\n';
                for (int i = 0; i < 2; ++i) f << "V," << vtx << ",e," << i << ',' << v.e(i) << '\n';
            }
            // The write is only now known to have succeeded. The previous version returned true
            // unconditionally, so a full disk reported a saved pool it had not saved.
            f.flush();
            if (not f.good()) { f.close(); std::remove(tmp.c_str()); return false; }
            }   // f closed here — the rename must not race the stream's own flush
            if (std::rename(tmp.c_str(), path.c_str()) != 0) { std::remove(tmp.c_str()); return false; }
            return true;
        }

        /// Returns the pair count restored; 0 means "no file", the ordinary first-run state.
        std::size_t load(const std::string& path)
        {
            // format 3: the nominal. All nine cells and both vectors, or it is not a nominal — a
            // partially written one would be a mount nobody measured.
            int  base_R_cells = 0;
            bool got_base_t = false, got_edge_t = false, got_edge_r = false;
            std::ifstream f(path);
            if (not f.is_open()) return 0;
            rc::mount::Accum in;
            std::string line;
            const auto num = [](std::string_view s, double& out)
            {   // from_chars, never strtod: these machines run es_ES (CLAUDE.md)
                const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
                return ec == std::errc{};
            };
            while (std::getline(f, line))
            {
                if (line.empty() or line[0] == '#') continue;
                std::vector<std::string> tok;
                for (std::size_t a = 0, b; a <= line.size(); a = b + 1)
                { b = line.find(',', a); if (b == std::string::npos) b = line.size();
                  tok.emplace_back(line.substr(a, b - a)); }
                double v = 0.0;
                if (tok[0] == "robot" and tok.size() == 2)
                {
                    if (not robot_.empty() and tok[1] != robot_)
                    {
                        qWarning() << "[camcal] refusing" << QString::fromStdString(path)
                                   << "— evidence for robot" << QString::fromStdString(tok[1])
                                   << "but this is" << QString::fromStdString(robot_);
                        return 0;
                    }
                    continue;
                }
                if (tok[0] == "camera" and tok.size() == 2)
                {
                    if (not camera_.empty() and tok[1] != camera_)
                    {
                        // REFUSED, not merged. A wrong-camera file is not partial evidence, it is
                        // evidence about a different object.
                        qWarning() << "[camcal] refusing" << QString::fromStdString(path)
                                   << "— it holds evidence for camera"
                                   << QString::fromStdString(tok[1]) << "but this is"
                                   << QString::fromStdString(camera_);
                        return 0;
                    }
                    continue;
                }
                if (tok[0] == "n" and tok.size() == 2 and num(tok[1], v))   in.n = static_cast<long>(v);
                else if (tok[0] == "rTr" and tok.size() == 2 and num(tok[1], v)) in.rTr = v;
                else if (tok[0] == "applied" and tok.size() == 5)
                {
                    double a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                    if (num(tok[1], a0) and num(tok[2], a1) and num(tok[3], a2) and num(tok[4], a3))
                        in.applied = Eigen::Vector4d(a0, a1, a2, a3);
                }
                else if (tok[0] == "base_R" and tok.size() == 4)
                {
                    double di = 0, dj = 0;
                    if (num(tok[1], di) and num(tok[2], dj) and num(tok[3], v))
                    {
                        const int i = static_cast<int>(di), j = static_cast<int>(dj);
                        if (i >= 0 and i < 3 and j >= 0 and j < 3)
                        { base_R_(i, j) = static_cast<float>(v); ++base_R_cells; }
                    }
                }
                else if (tok[0] == "base_t" and tok.size() == 4)
                {
                    double a = 0, b = 0, c = 0;
                    if (num(tok[1], a) and num(tok[2], b) and num(tok[3], c))
                    { base_t_ = Eigen::Vector3f(a, b, c); got_base_t = true; }
                }
                else if (tok[0] == "edge_t" and tok.size() == 4)
                {
                    double a = 0, b = 0, c = 0;
                    if (num(tok[1], a) and num(tok[2], b) and num(tok[3], c))
                    { edge_t_ = Eigen::Vector3f(a, b, c); got_edge_t = true; }
                }
                else if (tok[0] == "edge_r" and tok.size() == 4)
                {
                    double a = 0, b = 0, c = 0;
                    if (num(tok[1], a) and num(tok[2], b) and num(tok[3], c))
                    { edge_r_ = Eigen::Vector3f(a, b, c); got_edge_r = true; }
                }
                else if (tok[0] == "H" and tok.size() == 4)
                {
                    double di = 0, dj = 0;
                    if (num(tok[1], di) and num(tok[2], dj) and num(tok[3], v))
                    {
                        const int i = static_cast<int>(di), j = static_cast<int>(dj);
                        if (i >= 0 and i < 4 and j >= 0 and j < 4) { in.H(i, j) = v; in.H(j, i) = v; }
                    }
                }
                else if (tok[0] == "b" and tok.size() == 3)
                {
                    double di = 0;
                    if (num(tok[1], di) and num(tok[2], v))
                    {
                        const int i = static_cast<int>(di);
                        if (i >= 0 and i < 4) in.b(i) = v;
                    }
                }
                // V,<vertex>,<what>,<i>[,<j>],<value> — the per-vertex partials (format 2).
                // ⚠ >= 4, NOT >= 5. `V,<vertex>,n,<count>` and `V,<vertex>,rTr,<value>` are FOUR
                //   tokens; the A/c/D rows are six and b/e are five. With the bound at 5 the two
                //   4-token lines never reached their branches, so every RESTORED block came back
                //   with n = 0 — and solve() skips a block with n <= 0, which meant resumed evidence
                //   was silently dropped from the marginalisation while its rows stayed in the
                //   aggregate H. A fresh start always worked, so nothing ever pointed at it; the
                //   coverage check above is what made it visible (2026-09-03).
                else if (tok[0] == "V" and tok.size() >= 4)
                {
                    double dv = 0;
                    if (not num(tok[1], dv)) continue;
                    rc::mount::VertexBlock& vb = in.per_vertex[static_cast<int>(dv)];
                    const std::string& what = tok[2];
                    double di = 0, dj = 0;
                    if (what == "n"   and tok.size() == 4 and num(tok[3], v)) { vb.n = static_cast<long>(v); continue; }
                    if (what == "rTr" and tok.size() == 4 and num(tok[3], v)) { vb.rTr = v; continue; }
                    if (what == "b" and tok.size() == 5 and num(tok[3], di) and num(tok[4], v))
                    { const int i = static_cast<int>(di); if (i >= 0 and i < 4) vb.b(i) = v; continue; }
                    if (what == "e" and tok.size() == 5 and num(tok[3], di) and num(tok[4], v))
                    { const int i = static_cast<int>(di); if (i >= 0 and i < 2) vb.e(i) = v; continue; }
                    if (tok.size() != 6 or not num(tok[3], di) or not num(tok[4], dj) or not num(tok[5], v))
                        continue;
                    const int i = static_cast<int>(di), j = static_cast<int>(dj);
                    if (what == "A" and i >= 0 and i < 4 and j >= 0 and j < 4) { vb.A(i, j) = v; vb.A(j, i) = v; }
                    else if (what == "c" and i >= 0 and i < 4 and j >= 0 and j < 2) vb.c(i, j) = v;
                    else if (what == "D" and i >= 0 and i < 2 and j >= 0 and j < 2) { vb.D(i, j) = v; vb.D(j, i) = v; }
                }
            }
            if (in.n <= 0 or not in.H.allFinite() or not in.b.allFinite()) return 0;
            // ⚠ EVIDENCE WRITTEN BEFORE THE PER-VERTEX PARTITION CANNOT BE MARGINALISED. Its rows
            //   are already summed across corners, and no post-hoc step recovers which corner each
            //   came from. Flagging it makes the solve REFUSE the nuisance rather than quietly
            //   returning the old, 127x-overconfident answer under the new model's name.
            // ★ THE TEST IS COVERAGE, NOT PRESENCE. "No vertex blocks at all" only catches a pure
            //   format-1 file. A MIXED one — a format-1 pool resumed by the new binary and then saved
            //   with the partials of the pairs seen since — has blocks, passes a presence check, and
            //   is the dangerous case: the Schur complement would subtract per-vertex terms from an
            //   aggregate containing rows those terms do not describe, giving a number that belongs
            //   to neither model and looks entirely plausible. Every attributed pair increments BOTH
            //   counters, so sum(V.n) < n is exact evidence of unattributed mass.
            //   MEASURED 2026-09-03 on a live start: ricoh n=833400 with 2030 attributed, zed
            //   n=150009 with 90006 — both would have marginalised against 831370 and 60003
            //   unaccounted rows.
            long attributed = 0;
            for (const auto& [vid, vb] : in.per_vertex) attributed += vb.n;
            if (attributed < in.n)
            {
                in.legacy_unattributed = true;
                qWarning().nospace()
                    << "[camcal] " << QString::fromStdString(path) << ": " << (in.n - attributed)
                    << " of " << in.n << " pairs carry no per-vertex partials"
                    << (in.per_vertex.empty() ? " (format 1)" : " (MIXED format-1 and format-2 evidence)")
                    << ". They are restored for the aggregate solve, but the per-vertex offset nuisance "
                    << "CANNOT run on them and will REFUSE. Delete the file to re-accumulate under the "
                    << "new model.";
            }
            // ★ `in` is a fresh Accum, so assigning it would reset the nuisance's prior sigma to its
            //   default and the solve would silently revert to the old model on any run that resumed
            //   from disk. The knob is CONFIGURATION, not evidence; it must survive a load.
            const double keep_sigma = acc_.offset_sigma_px;
            acc_ = in;
            acc_.offset_sigma_px = keep_sigma;
            // ★ ALL NINE CELLS AND BOTH VECTORS, or there is no nominal. A half-written one is a
            //   mount nobody measured, and silently completing it from the graph is the very
            //   substitution this field exists to prevent.
            have_base_ = (base_R_cells == 9 and got_base_t);
            have_edge_ = (got_edge_t and got_edge_r);
            return static_cast<std::size_t>(acc_.n);
        }

    private:
        rc::mount::Accum acc_;
        std::string      camera_, robot_;
        /// The nominal `applied` is measured against — see set_base(). cam ← robot_frame for the
        /// ingestor, and the parent → camera RT edge for the publish.
        Eigen::Matrix3f  base_R_ = Eigen::Matrix3f::Identity();
        Eigen::Vector3f  base_t_ = Eigen::Vector3f::Zero();
        Eigen::Vector3f  edge_t_ = Eigen::Vector3f::Zero(), edge_r_ = Eigen::Vector3f::Zero();
        bool             have_base_ = false, have_edge_ = false;
    };
}   // namespace rc::camcal
