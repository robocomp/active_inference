/*
 * bottle_evaluator.cpp — Webots-GT / sweep / eval-CSV validation harness.
 */

#include "bottle_evaluator.h"

#include <algorithm>
#include <fstream>
#include <print>

#include <Eigen/Dense>

namespace rc {

BottleEvaluator::BottleEvaluator(BottleConfig& cfg,
                                 RoboCompWebots2Robocomp::Webots2RobocompPrxPtr proxy,
                                 DSR::InnerEigenAPI* inner_eigen,
                                 TableTopFn table_top)
    : cfg_(cfg), proxy_(std::move(proxy)), inner_eigen_(inner_eigen), table_top_(std::move(table_top))
{}

void BottleEvaluator::place_bottle_on_start()
{
    place_bottle_done_ = true;   // one-shot regardless of outcome — never fight perception with retries
    if (not proxy_)
        return;
    try
    {
        RoboCompWebots2Robocomp::ObjectPose p = proxy_->getObjectPose(cfg_.eval_bottle_def);
        p.position.x = cfg_.place_world_x * 1000.0f;   // config metres → bridge mm
        p.position.y = cfg_.place_world_y * 1000.0f;   // z + orientation kept from current pose
        proxy_->setObjectPose(cfg_.eval_bottle_def, p);
        std::print("bottle_concept: placed '{}' on start at Webots world ({:.3f}, {:.3f}) m (arm-side approach)\n",
                   cfg_.eval_bottle_def, cfg_.place_world_x, cfg_.place_world_y);
    }
    catch (const std::exception& e)
    { std::print("bottle_concept: place_bottle_on_start setObjectPose failed ({})\n", e.what()); }
}

bool BottleEvaluator::acquire_webots_gt()
{
    if (not inner_eigen_ or not proxy_)
        return false;

    RoboCompWebots2Robocomp::ObjectPose bottle_w, robot_w;
    try
    {
        bottle_w = proxy_->getObjectPose(cfg_.eval_bottle_def);
        robot_w  = proxy_->getObjectPose(cfg_.eval_robot_def);
    }
    catch (const std::exception& e)
    {
        std::print("bottle_concept: [eval] getObjectPose failed ({}); will retry\n", e.what());
        return false;
    }

    auto to_iso = [](const RoboCompWebots2Robocomp::ObjectPose& p)
    {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() = Eigen::Quaterniond(p.orientation.w, p.orientation.x,
                                        p.orientation.y, p.orientation.z).normalized().toRotationMatrix();
        T.translation() = Eigen::Vector3d(p.position.x / 1000.0,   // Webots reports mm
                                          p.position.y / 1000.0,
                                          p.position.z / 1000.0);
        return T;
    };

    // The WaterBottle proto origin is the bottle BASE (geometry shifted +height/2), so the
    // queried pose is the base. Express the base in the DSR `body` frame (== Webots Shadow
    // frame), then walk the graph room←body to land in the room frame the tracker fits in.
    const Eigen::Isometry3d T_world_bottle = to_iso(bottle_w);
    const Eigen::Isometry3d T_world_body   = to_iso(robot_w);
    const Eigen::Vector3d base_in_body = (T_world_body.inverse() * T_world_bottle).translation();

    const auto room_T_body = inner_eigen_->get_transformation_matrix("room", "body", 0);
    if (not room_T_body.has_value())
        return false;

    const Eigen::Vector3d base_in_room =
        room_T_body.value().linear() * base_in_body + room_T_body.value().translation();

    cfg_.gt_cx = static_cast<float>(base_in_room.x());
    cfg_.gt_cy = static_cast<float>(base_in_room.y());
    cfg_.gt_cz = static_cast<float>(base_in_room.z()) + 0.5f * cfg_.gt_height;   // base → centre
    // The room←body localization carries a ~4 cm z offset (the GT base lands BELOW the table top —
    // physically impossible), so the z above is unreliable. The bottle stands ON the table, so its
    // true base-z is the table top in the room frame: derive gt_cz from there when a table is found.
    if (const auto tt = table_top_ ? table_top_(cfg_.gt_cx, cfg_.gt_cy) : std::nullopt; tt.has_value())
        cfg_.gt_cz = *tt + 0.5f * cfg_.gt_height;
    return true;
}

void BottleEvaluator::step_move_experiment(std::unordered_map<std::uint64_t, BottleInstance>& instances)
{
    if (not cfg_.move_experiment or not proxy_)
        return;

    // Place the bottle in the WORLD frame (Webots), keeping the home z/orientation. In absolute mode the
    // pair is the world (x,y) target directly; otherwise it is a (dx,dy) offset from the home pose.
    const auto place = [this](float ax_mm, float ay_mm)
    {
        RoboCompWebots2Robocomp::ObjectPose p = move_home_;
        p.position.x = cfg_.move_absolute ? ax_mm : move_home_.position.x + ax_mm;
        p.position.y = cfg_.move_absolute ? ay_mm : move_home_.position.y + ay_mm;
        try { proxy_->setObjectPose(cfg_.eval_bottle_def, p); }
        catch (const std::exception& e) { std::print("bottle_concept: [move-exp] setObjectPose failed ({})\n", e.what()); }
    };

    // Re-seed countdown: a few cycles after each teleport (so the mask + voxels have refreshed at the new
    // pose), force a fresh cold-start so the fit re-snaps to the new position instead of being stranded
    // near the previous pose by the model-centred ownership gate. Makes each grid pose per-pose-honest.
    constexpr int MOVE_RESEED_DELAY = 8;
    if (move_reseed_in_ > 0 and --move_reseed_in_ == 0)
        for (auto& [id, inst] : instances)
        {
            inst.matched_frames  = 0;      // re-arm cold-start; the bank is purged inside it post-snap
            inst.reseed_requested = true;
        }

    // Lazy start: capture the home pose and build the grid over the table.
    if (not move_started_)
    {
        try { move_home_ = proxy_->getObjectPose(cfg_.eval_bottle_def); }
        catch (const std::exception& e)
        {
            std::print("bottle_concept: [move-exp] getObjectPose(home) failed ({}); retrying\n", e.what());
            return;
        }
        const float step_mm = cfg_.move_step_m * 1000.0f;
        move_offsets_.clear();
        if (cfg_.move_absolute)
        {
            // Absolute world rectangle [xmin,xmax]×[ymin,ymax] at step spacing (mm), row-major in y.
            for (float y = cfg_.move_ymin; y <= cfg_.move_ymax + 1e-4f; y += cfg_.move_step_m)
                for (float x = cfg_.move_xmin; x <= cfg_.move_xmax + 1e-4f; x += cfg_.move_step_m)
                    move_offsets_.emplace_back(x * 1000.0f, y * 1000.0f);
        }
        else
        {
            const int n = std::max(1, cfg_.move_grid_n);
            const float c = (n - 1) / 2.0f;     // centre the grid on home
            for (int iy = 0; iy < n; ++iy)
                for (int ix = 0; ix < n; ++ix)
                    move_offsets_.emplace_back((ix - c) * step_mm, (iy - c) * step_mm);
        }
        move_idx_ = 0;
        move_settle_ = 0;
        move_started_ = true;
        place(move_offsets_[0].first, move_offsets_[0].second);
        move_reseed_in_ = MOVE_RESEED_DELAY;
        std::print("bottle_concept: [move-exp] started — {} poses, step {:.3f} m, settle {} cycles, log -> {}\n",
                   move_offsets_.size(), cfg_.move_step_m, cfg_.move_settle_cycles, cfg_.eval_log_path);
        return;
    }

    if (move_idx_ >= move_offsets_.size())
        return;   // finished

    // Hold each pose for move_settle_cycles so the fit converges, then step to the next grid pose.
    if (++move_settle_ < cfg_.move_settle_cycles)
        return;

    ++move_idx_;
    move_settle_ = 0;
    if (move_idx_ >= move_offsets_.size())
    {
        std::print("bottle_concept: [move-exp] DONE ({} poses) — returning bottle home\n", move_offsets_.size());
        try { proxy_->setObjectPose(cfg_.eval_bottle_def, move_home_); } catch (...) {}
        cfg_.move_experiment = false;   // stop stepping; static eval (if enabled) continues
        return;
    }
    place(move_offsets_[move_idx_].first, move_offsets_[move_idx_].second);
    move_reseed_in_ = MOVE_RESEED_DELAY;
}

bool BottleEvaluator::place_static_test_pose()
{
    if (not cfg_.static_pose_test or not proxy_)
        return false;
    if (move_started_)
        return true;   // already placed for this run

    // Home = persisted grid origin. Captured the first time the home file is absent, then reused
    // across restarts so the grid targets stay consistent even though prior runs left the bottle
    // elsewhere. (Before starting the sweep, delete the file with the bottle at its true home.)
    const std::string home_path = "etc/bottle_home_pose.txt";
    bool have_home = false;
    {
        std::ifstream hf(home_path);
        if (hf and (hf >> move_home_.position.x >> move_home_.position.y >> move_home_.position.z
                       >> move_home_.orientation.x >> move_home_.orientation.y
                       >> move_home_.orientation.z >> move_home_.orientation.w))
            have_home = true;
    }
    if (not have_home)
    {
        try { move_home_ = proxy_->getObjectPose(cfg_.eval_bottle_def); }
        catch (const std::exception& e)
        {
            std::print("bottle_concept: [static-test] getObjectPose(home) failed ({}); retrying\n", e.what());
            return false;
        }
        std::ofstream hf(home_path);
        hf << move_home_.position.x << ' ' << move_home_.position.y << ' ' << move_home_.position.z << ' '
           << move_home_.orientation.x << ' ' << move_home_.orientation.y << ' '
           << move_home_.orientation.z << ' ' << move_home_.orientation.w << '\n';
        std::print("bottle_concept: [static-test] captured home ({:.0f},{:.0f},{:.0f}) mm -> {}\n",
                   move_home_.position.x, move_home_.position.y, move_home_.position.z, home_path);
    }

    // This run's grid pose (same centred N×N grid as the moving experiment).
    const int n = std::max(1, cfg_.move_grid_n);
    const int idx = std::clamp(cfg_.static_pose_index, 0, n * n - 1);
    const float step_mm = cfg_.move_step_m * 1000.0f;
    const float c = (n - 1) / 2.0f;
    const int ix = idx % n, iy = idx / n;
    RoboCompWebots2Robocomp::ObjectPose target = move_home_;
    target.position.x = move_home_.position.x + (ix - c) * step_mm;
    target.position.y = move_home_.position.y + (iy - c) * step_mm;
    try { proxy_->setObjectPose(cfg_.eval_bottle_def, target); }
    catch (const std::exception& e)
    {
        std::print("bottle_concept: [static-test] setObjectPose failed ({})\n", e.what());
        return false;
    }
    move_idx_ = idx;
    move_started_ = true;
    std::print("bottle_concept: [static-test] pose {}/{} -> bottle at ({:.0f},{:.0f},{:.0f}) mm\n",
               idx, n * n, target.position.x, target.position.y, target.position.z);
    return true;
}

void BottleEvaluator::log_eval(const BottleInstance& inst, float free_energy)
{
    if (not cfg_.eval_enabled)
        return;

    // Live GT from Webots. In the moving-bottle experiment the bottle changes pose, so refresh GT
    // EVERY cycle; otherwise acquire the static pose once (lazy — the bridge may not be ready at start).
    if (cfg_.eval_gt_source == "webots")
    {
        if (cfg_.move_experiment or cfg_.static_pose_test)
        {
            if (not acquire_webots_gt())
                return;
        }
        else if (not gt_acquired_)
        {
            if (acquire_webots_gt())
                gt_acquired_ = true;
            else
                return;
        }
    }

    if (not eval_log_.is_open())
    {
        // Static-restart test APPENDS (each restart adds rows to the same CSV); otherwise truncate.
        const bool append = cfg_.static_pose_test;
        eval_log_.open(cfg_.eval_log_path, std::ios::out | (append ? std::ios::app : std::ios::trunc));
        if (not eval_log_.is_open())
        {
            std::print("bottle_concept: [eval] cannot open log '{}'\n", cfg_.eval_log_path);
            cfg_.eval_enabled = false;   // don't retry every cycle
            return;
        }
        if (not append or eval_log_.tellp() == std::streampos(0))
            eval_log_ << "frame,node,move_idx,settled,fe,cx,cy,cz,radius,height,"
                         "gt_cx,gt_cy,gt_cz,gt_radius,gt_height,"
                         "ex,ey,ez,pos_err,radius_err,height_err,nees,trace_P\n";
    }

    const auto& s = inst.model.state();

    // 3×3 position covariance (same P_bottle written on the RT edge) — the AI2 belief's Σ[cx,cy,cz].
    if (not inst.ai2_initialized)
        return;
    const Eigen::Matrix3f cov = inst.ai2_belief.covariance().block<3, 3>(0, 0);

    const Eigen::Vector3f e(s.cx - cfg_.gt_cx, s.cy - cfg_.gt_cy, s.cz - cfg_.gt_cz);

    // NEES = eᵀ P⁻¹ e (3-DOF position; χ² expectation ≈ 3 when P is calibrated).
    const Eigen::Matrix3f cov_reg = cov + Eigen::Matrix3f::Identity() * 1e-9f;
    const float nees = e.dot(cov_reg.ldlt().solve(e));

    const int settled = (cfg_.move_experiment and move_settle_ >= cfg_.move_settle_cycles) ? 1 : 0;
    eval_log_ << inst.processed_cycles << ',' << inst.node_name << ','
              << move_idx_ << ',' << settled << ',' << free_energy << ','
              << s.cx << ',' << s.cy << ',' << s.cz << ',' << s.radius << ',' << s.height << ','
              << cfg_.gt_cx << ',' << cfg_.gt_cy << ',' << cfg_.gt_cz << ','
              << cfg_.gt_radius << ',' << cfg_.gt_height << ','
              << e.x() << ',' << e.y() << ',' << e.z() << ',' << e.norm() << ','
              << (s.radius - cfg_.gt_radius) << ',' << (s.height - cfg_.gt_height) << ','
              << nees << ',' << cov.trace() << '\n';
    eval_log_.flush();
}

}  // namespace rc
