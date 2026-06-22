/*
 *    Copyright (C) 2026 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
	\brief
	@author authorname
*/

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

// If you want to reduce the period automatically due to lack of use, you must uncomment the following line
//#define HIBERNATION_ENABLED

#include <genericworker.h>

#include "media_transport.h"

#include <Eigen/Dense>

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

class Kinematics;                         // shared FK model (common/kinematics)
class SelfProjectionViewer;               // RGB + FK-aura display dock (src/self_projection_viewer)
namespace DSR { class InnerEigenAPI; }    // arm-base→camera extrinsic from the graph

/**
 * \brief Class SpecificWorker implements the core functionality of the component.
 */
class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    /**
     * \brief Constructor for SpecificWorker.
     * \param configLoader Configuration loader for the component.
     * \param tprx Tuple of proxies required for the component.
     * \param startup_check Indicates whether to perform startup checks.
     */
	SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);

	/**
     * \brief Destructor for SpecificWorker.
     */
	~SpecificWorker();


public slots:

	/**
	 * \brief Initializes the worker one time.
	 */
	void initialize();

	/**
	 * \brief Main compute loop of the worker.
	 */
	void compute();

	/**
	 * \brief Handles the emergency state loop.
	 */
	void emergency();

	/**
	 * \brief Restores the component from an emergency state.
	 */
	void restore();

    /**
     * \brief Performs startup checks for the component.
     * \return An integer representing the result of the checks.
     */
	int startup_check();

	void modify_node_slot(std::uint64_t, const std::string &type){};
	void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
	void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type){};
	void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names){};
	void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
	void del_node_slot(std::uint64_t from){};     
private:

	/**
     * \brief Flag indicating whether startup checks are enabled.
     */
	bool startup_check_flag;

	// ── Media plane (RGB + depth) ─────────────────────────────────────────────
	// Discovered from the MediaDescriptor on the camera node (no hardcoded topic/
	// domain) and polled in compute(). This agent has no 100 Hz loop to protect, so
	// it polls the subscriber directly (unlike the controller's threaded source).
	void init_media_plane();
	std::unique_ptr<rc::media::MediaSubscriber> media_rgb_sub_;
	std::unique_ptr<rc::media::MediaSubscriber> media_depth_sub_;
	std::uint64_t rx_rgb_count_   = 0, rx_depth_count_   = 0;
	std::uint32_t last_rgb_w_     = 0, last_rgb_h_       = 0;
	std::uint32_t last_depth_w_   = 0, last_depth_h_     = 0;
	std::uint64_t last_rgb_stamp_ms_   = 0;   // epoch ms (the t for the t↔q match)
	std::uint64_t last_depth_stamp_ms_ = 0;
	// Latest frames (copied out of the loaned view so we can use them after poll).
	std::vector<std::uint8_t> depth_data_;
	std::uint32_t             depth_format_ = 0;
	std::vector<std::uint8_t> rgb_data_;
	std::uint32_t             rgb_format_ = 0;

	// ── Step 3: depth-channel reprojection residual ───────────────────────────
	// q matched to a frame stamp, with a finite-difference joint-speed for the
	// residual-vs-q̇ split (constant bias = spatial calibration, q̇-proportional = τ).
	struct JointSample
	{
		std::array<double, 7> q{};
		std::array<double, 7> q_prev{};   // neighbouring ring sample (for the depth-velocity)
		std::uint64_t stamp_ms     = 0;
		std::uint64_t match_err_ms = 0;   // |nearest ring stamp − frame stamp|
		double        qdot_norm    = 0.0; // rad/s, ‖Δq/Δt‖ across the matched ring step
		double        dt_s         = 0.0; // signed time of the matched ring step (t_best − t_prev)
		bool          has_prev     = false;
		bool          valid        = false;
	};
	// Read the (stamp_ms,q) ring off kinova_arm_r and nearest-match to t_img.
	[[nodiscard]] JointSample match_joint(std::uint64_t t_img) const;
	// Compose arm-base→camera: body_T_zed⁻¹ · body_T_arm (both hang off "body").
	[[nodiscard]] bool build_extrinsic(Eigen::Matrix4d& zed_T_arm) const;
	// Read depth intrinsics once (cam_depth_focalx/y; cx,cy = w/2,h/2).
	bool ensure_depth_intrinsics();
	// FK the matched q, project the link capsules, score the depth residual; print it.
	void compute_depth_residual();

	std::unique_ptr<Kinematics>         kin_;
	std::unique_ptr<DSR::InnerEigenAPI> inner_eigen_;
	std::uint64_t last_processed_depth_stamp_ = 0;
	double dfx_ = 0.0, dfy_ = 0.0, dcx_ = 0.0, dcy_ = 0.0;
	bool   depth_intr_ok_ = false;

	// ── Visualization: RGB + projected FK-capsule aura ────────────────────────
	// The agent has the frame + FK + extrinsic, so it builds the overlay here and
	// pushes (image + capsules) to a display-only dock each new RGB frame.
	void build_self_projection();
	bool ensure_rgb_intrinsics();
	std::unique_ptr<SelfProjectionViewer> viewer_;
	std::uint64_t last_processed_rgb_stamp_ = 0;
	double rfx_ = 0.0, rfy_ = 0.0, rcx_ = 0.0, rcy_ = 0.0;
	bool   rgb_intr_ok_ = false;

	// ── τ / spatial-bias regression ───────────────────────────────────────────
	// Per inlier sample, the residual r ≈ spatial_bias − τ·vdepth, where vdepth =
	// ∂(predicted depth)/∂t measured from the ring step. Keep a sliding window of
	// (vdepth, r) pairs and fit r = a − τ·vdepth ROBUSTLY (IRLS Huber) every
	// REG_REPORT_EVERY frames, so the fast-motion model-breakdown frames can't lever
	// the slope. τ = −slope, spatial_bias = intercept.
	std::deque<std::pair<float, float>> reg_samples_;     // (vdepth, residual)
	int    reg_frames_      = 0;
	double residual_gate_m_ = 0.06;                       // inlier band (SelfCalib.residual_gate_m)
	double tau_vspread_min_ = 0.1;                        // min depth-velocity std (m/s) to identify τ (SelfCalib.tau_vspread_min)
	static constexpr std::size_t REG_MAX          = 3000; // ~2–3 s of inlier samples
	static constexpr int         REG_REPORT_EVERY = 20;

signals:
	//void customSignal();
};

#endif
