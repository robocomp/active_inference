/*  joint_calib_monitor.h — the LOG-ONLY joint calibration monitor (plan 2026-10-05 Task 4).
 *
 *  Once per new motion solve: fuse the motion block's normal equations with every camera block through
 *  the shared helios yaw (joint_calibration.h) and write one CSV row. Changes NOTHING the robot does.
 *  Grade with tools/joint_calib_report.py.
 *
 *  Pure (no Qt, no DSR) so the row format is tested offline against the report (joint_calib_selftest).
 *  ⚠ THREADS. The motion block is fed on the LOCALIZER thread; this runs on the MAIN thread (where the
 *  camera pools are fed). It therefore takes the motion block as a COPY of its Information carried in
 *  the localizer's result (UpdateResult::calib_information), never by reaching into the calibrator.
 *
 *  Row: ts_ms,episodes, <name>,<name>_sd for every Solution::names entry, corr_eps_<cam>yaw per camera,
 *  chain_<cam>.yaw per camera (the camera's OWN, chained answer). Written through the classic locale.
 */
#pragma once
#include "joint_calibration.h"

#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <string>
#include <vector>

namespace rc::joint
{
    class Monitor
    {
    public:
        /// Directory the CSV goes to (relative to the agent's working directory). Settable for tests.
        explicit Monitor(std::string dir = "tmp/joint_calib") : dir_(std::move(dir)) {}

        /// Solve and log. Returns the solution (ok=false when nothing was solved).
        Solution observe(std::int64_t ts_ms, int episodes,
                         const rc::calib::BatchEstimator::Information &motion,
                         const std::vector<CameraBlock> &cams)
        {
            const Solution s = solve(motion, cams);
            if (s.ok) write_row(ts_ms, episodes, s, cams);
            return s;
        }
        [[nodiscard]] const std::string& path() const noexcept { return path_; }

    private:
        static std::string stamp()
        {
            const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::tm tm{};
            localtime_r(&t, &tm);
            char buf[32];
            std::strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%S", &tm);
            return buf;
        }
        /// (Re)open when the column set changes (a camera block appearing) so a header never lies.
        void ensure_open(const Solution &s, const std::vector<CameraBlock> &cams)
        {
            std::vector<std::string> cols = s.names;
            for (const auto &c : cams) cols.push_back(c.name);
            if (csv_.is_open() and cols == cols_) return;
            if (csv_.is_open()) csv_.close();
            cols_ = std::move(cols);
            std::error_code ec;
            std::filesystem::create_directories(dir_, ec);
            path_ = dir_ + "/joint_" + stamp() + (reopened_++ ? "_" + std::to_string(reopened_) : "") + ".csv";
            csv_.open(path_, std::ios::out | std::ios::trunc);
            if (not csv_.is_open()) return;
            csv_.imbue(std::locale::classic());   // CLAUDE.md: never a comma decimal separator
            csv_ << "ts_ms,episodes";
            for (const auto &n : s.names) csv_ << ',' << n << ',' << n << "_sd";
            for (const auto &c : cams) csv_ << ",corr_eps_" << c.name << "yaw";
            for (const auto &c : cams) csv_ << ",chain_" << c.name << ".yaw";
            csv_ << '\n';
        }
        void write_row(std::int64_t ts_ms, int episodes, const Solution &s, const std::vector<CameraBlock> &cams)
        {
            ensure_open(s, cams);
            if (not csv_.is_open()) return;
            constexpr double nan = std::numeric_limits<double>::quiet_NaN();
            constexpr int ie = rc::calib::P_EPS_YAW;
            csv_ << ts_ms << ',' << episodes;
            for (std::size_t i = 0; i < s.names.size(); ++i)
                csv_ << ',' << s.value[static_cast<Eigen::Index>(i)] << ',' << s.sigma[static_cast<Eigen::Index>(i)];
            for (std::size_t c = 0; c < cams.size(); ++c)
            {
                const Eigen::Index iy = rc::calib::P_COUNT + 4 * static_cast<Eigen::Index>(c) + kCamYawRow;
                const double den = s.sigma[ie] * s.sigma[iy];
                csv_ << ',' << (den > 0.0 ? s.cov(ie, iy) / den : nan);
            }
            for (const auto &c : cams)
            {
                double chain = nan;   // the camera's own answer: (H_data + H_prior)^-1 (b_data + b_prior), yaw row
                if (c.info.ok)
                    chain = (c.info.H_data + c.info.H_prior).ldlt().solve(c.info.b_data + c.info.b_prior)[kCamYawRow];
                csv_ << ',' << chain;
            }
            csv_ << '\n';
            csv_.flush();
        }

        std::string dir_, path_;
        std::ofstream csv_;
        std::vector<std::string> cols_;
        int reopened_ = 0;
    };

    /* MountMonitor — the LOG-ONLY r2 mount monitor (plan 2026-10-08 Task 5). One row per call: solve_mounts over
     * the mount-factor blocks + the cameras -> tmp/joint_calib/mounts_<ts>.csv. Changes NOTHING the robot does.
     * Row: ts_ms, yaw_offset (the motion calibrator's ACTING eps on the odometry -- must be 0 while the
     * kinematic factor owns the helios yaw, r2.2 items 4/9), n_kin, n_floor, n_vert_h, n_vert_b, then per
     * parameter: <name>, <name>_sd, <name>_sh_<factor> for each of kShareNames (its information share).
     * ⚠ THREADS: the blocks are COPIES (kinematic from UpdateResult, floor/vertical from the ingestor's
     * snapshot); this runs on the main thread. Grade with tools/joint_calib_report.py --mounts. */
    class MountMonitor
    {
    public:
        explicit MountMonitor(std::string dir = "tmp/joint_calib") : dir_(std::move(dir)) {}
        struct Extra { double yaw_offset = 0.0; };
        MountSolution observe(std::int64_t ts_ms, const MountBlocks &B, const std::vector<CameraBlock> &cams, const Extra &ex)
        {
            const MountSolution s = solve_mounts(B, cams);
            if (s.ok) write_row(ts_ms, B, s, ex);
            return s;
        }
        [[nodiscard]] const std::string& path() const noexcept { return path_; }
    private:
        void ensure_open(const MountSolution &s)
        {
            if (csv_.is_open() and s.names == cols_) return;
            if (csv_.is_open()) csv_.close();
            cols_ = s.names;
            std::error_code ec;
            std::filesystem::create_directories(dir_, ec);
            const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::tm tm{}; localtime_r(&t, &tm);
            char buf[32]; std::strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%S", &tm);
            path_ = dir_ + "/mounts_" + buf + (reopened_++ ? "_" + std::to_string(reopened_) : "") + ".csv";
            csv_.open(path_, std::ios::out | std::ios::trunc);
            if (not csv_.is_open()) return;
            csv_.imbue(std::locale::classic());   // CLAUDE.md: never a comma decimal separator
            csv_ << "ts_ms,yaw_offset,n_kin,n_floor,n_vert_h,n_vert_b";
            for (const auto &n : s.names)
            {
                csv_ << ',' << n << ',' << n << "_sd";
                for (const char *f : kShareNames) csv_ << ',' << n << "_sh_" << f;
            }
            csv_ << '\n';
        }
        void write_row(std::int64_t ts_ms, const MountBlocks &B, const MountSolution &s, const Extra &ex)
        {
            ensure_open(s);
            if (not csv_.is_open()) return;
            csv_ << ts_ms << ',' << ex.yaw_offset << ',' << B.kinematic.n << ',' << B.floor.n << ','
                 << B.vert_helios.n << ',' << B.vert_bpearl.n;
            for (Eigen::Index i = 0; i < s.value.size(); ++i)
            {
                csv_ << ',' << s.value[i] << ',' << s.sigma[i];
                for (int f = 0; f < kShareCols; ++f) csv_ << ',' << s.share(i, f);
            }
            csv_ << '\n';
            csv_.flush();
        }
        std::string dir_, path_;
        std::ofstream csv_;
        std::vector<std::string> cols_;
        int reopened_ = 0;
    };
}
