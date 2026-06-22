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
#include "specificworker.h"

#include "kinematics.h"
#include "self_projection_capsules.h"
#include "self_projection_viewer.h"

#include <dsr/api/dsr_inner_eigen_api.h>

#include <QImage>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <print>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
    // Capsule fitted to a given link mesh (small linear scan over the 8 links).
    const LinkCapsule* capsule_for(std::string_view mesh)
    {
        for (const auto& c : LINK_CAPSULES) if (c.mesh == mesh) return &c;
        return nullptr;
    }

    // Near-surface depth (camera-frame Y) where the camera ray through point P first
    // meets the capsule (segment A..B, radius R) — the proper ray–capsule intersection
    // that replaces the crude Y−R: cylinder body, with sphere caps past the ends.
    // Returns `fallback` when the ray grazes/misses (numerically degenerate).
    double capsule_near_depth(const Eigen::Vector3d& P, const Eigen::Vector3d& A,
                              const Eigen::Vector3d& B, double R, double fallback)
    {
        const Eigen::Vector3d d  = P.normalized();   // ray dir from the camera origin (O=0)
        const Eigen::Vector3d ab = B - A;
        const double L = ab.norm();
        double tau = -1.0;
        if (L > 1e-6)                                 // infinite-cylinder body, clamped to [A,B]
        {
            const Eigen::Vector3d u = ab / L;
            const double du = d.dot(u);
            const double alpha = 1.0 - du * du;
            if (alpha > 1e-6)
            {
                const double Au = A.dot(u), dA = d.dot(A);
                const double beta  = 2.0 * (du * Au - dA);
                const double gamma = A.squaredNorm() - Au * Au - R * R;
                const double disc  = beta * beta - 4.0 * alpha * gamma;
                if (disc >= 0.0)
                {
                    const double t = (-beta - std::sqrt(disc)) / (2.0 * alpha);
                    const double sproj = (t * d - A).dot(u);   // axial coord of the hit
                    if (t > 1e-6 and sproj >= 0.0 and sproj <= L) tau = t;
                }
            }
        }
        if (tau < 0.0)                                // off the body → sphere cap at the nearer end
        {
            const Eigen::Vector3d C = (L > 1e-6 and (P - A).dot(ab) > 0.5 * L * L) ? B : A;
            const double dC = d.dot(C);
            const double disc = dC * dC - (C.squaredNorm() - R * R);
            if (disc >= 0.0)
            {
                const double t = dC - std::sqrt(disc);
                if (t > 1e-6) tau = t;
            }
        }
        return tau < 0.0 ? fallback : (tau * d).y();
    }

    struct LineFit { double slope = 0, intercept = 0, scale = 0, vspread = 0, inlier_frac = 0; bool ok = false; };

    // Robust line fit r = intercept + slope·v via IRLS with Huber weights, so the
    // fast-motion model-breakdown samples can't lever the slope. ok=false if too few
    // points or no v-spread (τ is then unseparable from the bias).
    LineFit huber_fit(const std::deque<std::pair<float, float>>& pts, double vspread_min)
    {
        LineFit f;
        const std::size_t n = pts.size();
        if (n < 30) return f;

        double mv = 0.0; for (const auto& p : pts) mv += p.first; mv /= static_cast<double>(n);
        double vv = 0.0; for (const auto& p : pts) vv += (p.first - mv) * (p.first - mv);
        f.vspread = std::sqrt(vv / static_cast<double>(n));
        // Below this depth-velocity leverage τ is unseparable from the bias: the slope
        // just fits noise and emits a confident-looking but meaningless value (e.g. +220 ms
        // at vspread≈0.04). Gate it so τ only reports when the arm moves enough.
        if (f.vspread < vspread_min) return f;

        double sv=0,sr=0,svv=0,svr=0;     // OLS seed
        for (const auto& p : pts){ sv+=p.first; sr+=p.second; svv+=p.first*p.first; svr+=p.first*p.second; }
        const double det = static_cast<double>(n)*svv - sv*sv;
        if (std::abs(det) < 1e-12) return f;
        double b = (static_cast<double>(n)*svr - sv*sr)/det;
        double a = (sr - b*sv)/static_cast<double>(n);

        std::vector<double> ae(n);
        double scale = 0.0;
        for (int iter = 0; iter < 5; ++iter)   // IRLS Huber
        {
            std::size_t i = 0;
            for (const auto& p : pts) ae[i++] = std::abs(p.second - (a + b*p.first));
            std::nth_element(ae.begin(), ae.begin() + n/2, ae.end());
            scale = 1.4826 * std::max(ae[n/2], 1e-4);
            const double k = 1.345 * scale;
            double Wsw=0,Wsv=0,Wsr=0,Wsvv=0,Wsvr=0;
            for (const auto& p : pts)
            {
                const double e = p.second - (a + b*p.first);
                const double w = std::abs(e) <= k ? 1.0 : k/std::abs(e);
                Wsw+=w; Wsv+=w*p.first; Wsr+=w*p.second; Wsvv+=w*p.first*p.first; Wsvr+=w*p.first*p.second;
            }
            const double d2 = Wsw*Wsvv - Wsv*Wsv;
            if (std::abs(d2) < 1e-12) break;
            b = (Wsw*Wsvr - Wsv*Wsr)/d2;
            a = (Wsr - b*Wsv)/Wsw;
        }
        std::size_t in = 0;
        for (const auto& p : pts) if (std::abs(p.second - (a + b*p.first)) <= 1.345*scale) ++in;
        f.slope=b; f.intercept=a; f.scale=scale; f.inlier_frac=static_cast<double>(in)/static_cast<double>(n); f.ok=true;
        return f;
    }
}

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check) : GenericWorker(configLoader, tprx)
{
	this->startup_check_flag = startup_check;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
		#ifdef HIBERNATION_ENABLED
			hibernationChecker.start(500);
		#endif
		
		// Example statemachine:
		/***
		//Your definition for the statesmachine (if you dont want use a execute function, use nullptr)
		states["CustomState"] = std::make_unique<GRAFCETStep>("CustomState", period, 
															std::bind(&SpecificWorker::customLoop, this),  // Cyclic function
															std::bind(&SpecificWorker::customEnter, this), // On-enter function
															std::bind(&SpecificWorker::customExit, this)); // On-exit function

		//Add your definition of transitions (addTransition(originOfSignal, signal, dstState))
		states["CustomState"]->addTransition(states["CustomState"].get(), SIGNAL(entered()), states["OtherState"].get());
		states["Compute"]->addTransition(this, SIGNAL(customSignal()), states["CustomState"].get()); //Define your signal in the .h file under the "Signals" section.

		//Add your custom state
		statemachine.addState(states["CustomState"].get());
		***/

		statemachine.setChildMode(QState::ExclusiveStates);
		statemachine.start();

		auto error = statemachine.errorString();
		if (error.length() > 0){
			qWarning() << error;
			throw error;
		}
	}
}

SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;
	/*
	for (auto const& [name, g] : Graphs) {
	    g->write_to_json_file("./"+agent_name+"_"+name+".json");
	}
	*/
}


void SpecificWorker::initialize()
{
    std::cout << "initialize worker" << std::endl;
	GenericWorker::initialize();

	//Subscription to DSR graph update signals. 
	// If multiple graphs exist, it is necessary to specify the graph name 
	// using 'Graphs.at("name")' to connect its signals to the Worker's slots.
	//connect(Graphs.at("").get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
	//connect(Graphs.at("").get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
	//connect(Graphs.at("").get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
	//connect(Graphs.at("").get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
	//connect(Graphs.at("").get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
	//connect(Graphs.at("").get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

	/***
	Custom Widget
	In addition to the predefined viewers, Graph Viewer allows you to add various widgets designed by the developer.
	The add_custom_widget_to_dock method is used. This widget can be defined like any other Qt widget,
	either with a QtDesigner or directly from scratch in a class of its own.
	The add_custom_widget_to_dock method receives a name for the widget and a reference to the class instance.
	***/
	//If you have more than one graph, you need to connect to the specific graph with the name
	//graph_viewers.at("")->add_custom_widget_to_dock("CustomWidget", &custom_widget);

    //initializeCODE
    init_media_plane();
    try { residual_gate_m_ = configLoader.get<double>("SelfCalib.residual_gate_m"); } catch (...) {}
    try { tau_vspread_min_ = configLoader.get<double>("SelfCalib.tau_vspread_min"); } catch (...) {}

    // FK model (the SAME Kinematics the controller uses) + the graph extrinsic API.
    if (G) inner_eigen_ = G->get_inner_eigen_api();
    std::string urdf = "/home/pbustos/robocomp/components/active_inference/common/kinematics/gen3_robotiq_2f_85-mod.urdf";
    try { urdf = configLoader.get<std::string>("SelfCalib.urdf_path"); } catch (...) {}
    try
    {
        kin_ = std::make_unique<Kinematics>(urdf);   // base_tf_ stays identity ⇒ base_link frame
        std::print("[self_calib] Kinematics loaded: {}\n", urdf);
    }
    catch (const std::exception& e)
    {
        std::print(stderr, "[self_calib] Kinematics load FAILED ({}) — residual disabled.\n", e.what());
    }

    // Visualization dock: RGB + the projected FK-capsule aura. Own top-level window
    // (isolated from the graph view's repaint storm); falls back to a standalone
    // window if no graph viewer is present.
    viewer_ = std::make_unique<SelfProjectionViewer>();
    if (graph_viewers.contains(""))
        graph_viewers.at("")->add_custom_widget_in_own_window("self_projection", viewer_.get());
    else
        viewer_->show();
}

// Discover the zero-copy media plane from the DSR graph (the camera node carries a
// MediaDescriptor JSON written by robot_concept) and open RGB + depth subscribers.
// All failures are soft: the agent stays up and retries are a future refinement.
void SpecificWorker::init_media_plane()
{
    if (not G)
    {
        std::print(stderr, "[self_calib] DSR graph G is null — cannot discover media plane.\n");
        return;
    }

    std::string camera_node = "zed";
    try { camera_node = configLoader.get<std::string>("SelfCalib.camera_node"); } catch (...) {}

    const auto desc = rc::media::descriptor_from_graph(*G, camera_node);
    if (not desc.has_value())
    {
        std::print(stderr, "[self_calib] no media_descriptor on node '{}' — is robot_concept up?\n", camera_node);
        return;
    }
    if (desc->type_tag != rc::media::IMAGE_FRAME_TYPE_TAG)
    {
        std::print(stderr, "[self_calib] type_tag mismatch ('{}' != '{}') — refusing stream.\n",
                   desc->type_tag, rc::media::IMAGE_FRAME_TYPE_TAG);
        return;
    }

    if (const auto cfg = desc->subscriber_config("rgb"))
    {
        media_rgb_sub_ = std::make_unique<rc::media::MediaSubscriber>();
        if (not media_rgb_sub_->init(*cfg)) media_rgb_sub_.reset();
    }
    if (const auto cfg = desc->subscriber_config("depth"))
    {
        media_depth_sub_ = std::make_unique<rc::media::MediaSubscriber>();
        if (not media_depth_sub_->init(*cfg)) media_depth_sub_.reset();
    }
    std::print("[self_calib] media plane: domain={} rgb={} depth={}\n",
               desc->domain_id, static_cast<bool>(media_rgb_sub_), static_cast<bool>(media_depth_sub_));
}


void SpecificWorker::compute()
{
    // Drain the media plane (non-blocking). The frame is a loaned view valid only
    // for the callback; record metadata now, do the FK/projection/residual later.
    if (media_rgb_sub_)
        rx_rgb_count_ += media_rgb_sub_->poll([this](const rc::media::ImageFrame& f, std::int64_t)
        {
            last_rgb_w_ = f.width(); last_rgb_h_ = f.height();
            last_rgb_stamp_ms_ = f.stamp_ms();
            rgb_format_ = f.format();
            const std::uint32_t sz = f.size();           // copy out of the loaned view for the overlay
            if (sz > 0 and sz <= f.data().size())
                rgb_data_.assign(f.data().data(), f.data().data() + sz);
        });
    if (media_depth_sub_)
        rx_depth_count_ += media_depth_sub_->poll([this](const rc::media::ImageFrame& f, std::int64_t)
        {
            last_depth_w_ = f.width(); last_depth_h_ = f.height();
            last_depth_stamp_ms_ = f.stamp_ms();
            depth_format_ = f.format();
            const std::uint32_t sz = f.size();           // copy out of the loaned view to sample later
            if (sz > 0 and sz <= f.data().size())
                depth_data_.assign(f.data().data(), f.data().data() + sz);
        });

    // New depth frame ⇒ score the reprojection residual against the matched q.
    if (last_depth_stamp_ms_ != last_processed_depth_stamp_ and not depth_data_.empty())
    {
        last_processed_depth_stamp_ = last_depth_stamp_ms_;
        compute_depth_residual();
    }

    // New RGB frame ⇒ rebuild the aura overlay (FK + project capsules) and show it.
    if (last_rgb_stamp_ms_ != last_processed_rgb_stamp_ and not rgb_data_.empty())
    {
        last_processed_rgb_stamp_ = last_rgb_stamp_ms_;
        build_self_projection();
    }

    // Reception heartbeat (foundation milestone): confirm the plane is flowing
    // before we build FK projection + the reprojection residual on top.
    static auto last_report = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - last_report >= std::chrono::seconds(5))
    {
        std::print("[calib-rx] 5s rgb={} ({}x{} t={}ms) depth={} ({}x{} t={}ms)\n",
                   rx_rgb_count_, last_rgb_w_, last_rgb_h_, last_rgb_stamp_ms_,
                   rx_depth_count_, last_depth_w_, last_depth_h_, last_depth_stamp_ms_);
        last_report = now;
    }
}


// Read the (stamp_ms,q) ring off kinova_arm_r (written by the controller) and
// nearest-match q to the frame stamp t_img. Reconstructs stamps from base + offsets.
SpecificWorker::JointSample SpecificWorker::match_joint(std::uint64_t t_img) const
{
    JointSample out;
    if (not G) return out;
    auto node = G->get_node("kinova_arm_r");
    if (not node.has_value()) return out;
    const auto find_attr = [&](const char* n) -> const DSR::Attribute*
    {
        auto it = node->attrs().find(n);
        return it == node->attrs().end() ? nullptr : &it->second;
    };
    const auto* base_a = find_attr("joint_buffer_base_ms");
    const auto* off_a  = find_attr("joint_buffer_stamp_off_ms");
    const auto* q_a    = find_attr("joint_buffer_q");
    const auto* dof_a  = find_attr("joint_buffer_dof");
    if (not (base_a and off_a and q_a and dof_a))
        return out;

    const std::uint64_t base = base_a->uint64();
    const auto& off = off_a->float_vec();
    const auto& qv  = q_a->float_vec();
    const int   dof = dof_a->dec();
    const std::size_t N = off.size();
    if (dof <= 0 or N == 0 or qv.size() < N * static_cast<std::size_t>(dof))
        return out;

    const auto stamp_at = [&](std::size_t i) { return base + static_cast<std::uint64_t>(std::llround(off[i])); };
    std::size_t best = 0; std::uint64_t best_d = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t i = 0; i < N; ++i)
    {
        const std::uint64_t st = stamp_at(i);
        const std::uint64_t d  = st > t_img ? st - t_img : t_img - st;
        if (d < best_d) { best_d = d; best = i; }
    }
    for (int j = 0; j < 7 and j < dof; ++j)
        out.q[j] = qv[best * dof + j];
    out.stamp_ms     = stamp_at(best);
    out.match_err_ms = best_d;

    // Neighbouring ring sample → the matched step's q_prev + dt (for the depth-velocity
    // ∂depth/∂t) and a scalar ‖q̇‖. Prefer the earlier neighbour so dt > 0.
    if (N >= 2)
    {
        const std::size_t k = best > 0 ? best - 1 : best + 1;
        out.dt_s = (static_cast<double>(off[best]) - static_cast<double>(off[k])) / 1000.0;
        double s2 = 0.0;
        for (int j = 0; j < 7 and j < dof; ++j)
        {
            out.q_prev[j] = qv[k * dof + j];
            const double dq = static_cast<double>(qv[best * dof + j]) - static_cast<double>(qv[k * dof + j]);
            s2 += dq * dq;
        }
        out.qdot_norm = std::abs(out.dt_s) > 1e-4 ? std::sqrt(s2) / std::abs(out.dt_s) : 0.0;
        out.has_prev  = std::abs(out.dt_s) > 1e-4;
    }
    out.valid = true;
    return out;
}

// arm-base→camera extrinsic: body_T_zed⁻¹ · body_T_arm (both hang off "body", so
// room/world cancels). Element-wise copy = alignment-safe (bottle_concept pattern).
bool SpecificWorker::build_extrinsic(Eigen::Matrix4d& zed_T_arm) const
{
    if (not inner_eigen_) return false;
    const auto btz = inner_eigen_->get_transformation_matrix("body", "zed", 0);
    const auto bta = inner_eigen_->get_transformation_matrix("body", "kinova_arm_r", 0);
    if (not (btz.has_value() and bta.has_value())) return false;
    const auto to_m4 = [](const Mat::RTMat& T)
    {
        Eigen::Matrix4d m; const auto& s = T.matrix();
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) m(i, j) = s(i, j);
        return m;
    };
    zed_T_arm = to_m4(btz.value()).inverse() * to_m4(bta.value());
    return true;
}

// Depth intrinsics from the zed node (once): focal from cam_depth_focalx/y, the
// principal point at the depth image centre (cx,cy = w/2,h/2 — CameraAPI convention).
bool SpecificWorker::ensure_depth_intrinsics()
{
    if (depth_intr_ok_) return true;
    if (not G or last_depth_w_ == 0 or last_depth_h_ == 0) return false;
    auto zed = G->get_node("zed");
    if (not zed.has_value()) return false;
    const auto fx = G->get_attrib_by_name<cam_depth_focalx_att>(zed.value());
    const auto fy = G->get_attrib_by_name<cam_depth_focaly_att>(zed.value());
    if (not (fx.has_value() and fy.has_value())) return false;
    dfx_ = fx.value(); dfy_ = fy.value();
    dcx_ = last_depth_w_ / 2.0; dcy_ = last_depth_h_ / 2.0;
    depth_intr_ok_ = true;
    std::print("[self_calib] depth intrinsics: fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f} ({}x{})\n",
               dfx_, dfy_, dcx_, dcy_, last_depth_w_, last_depth_h_);
    return true;
}

// First reprojection residual (depth channel, FIXED params). FK the matched q,
// sample each link capsule's axis, project into the depth image, and compare the
// predicted near-surface depth (axis depth − radius) to the observed depth. An
// inlier gate rejects background/occlusion. Prints rms + signed bias (the spatial
// calibration error) alongside q̇ (the residual's q̇-slope is τ).
void SpecificWorker::compute_depth_residual()
{
    if (not kin_ or not ensure_depth_intrinsics())
        return;
    const auto js = match_joint(last_depth_stamp_ms_);
    if (not js.valid)
        return;
    Eigen::Matrix4d zed_T_arm;
    if (not build_extrinsic(zed_T_arm))
        return;

    const int w = static_cast<int>(last_depth_w_), h = static_cast<int>(last_depth_h_);
    const auto depth_at = [&](int u, int v) -> double          // observed forward distance, m (<0 = invalid)
    {
        if (u < 0 or v < 0 or u >= w or v >= h) return -1.0;
        const std::size_t idx = static_cast<std::size_t>(v) * w + u;
        if (depth_format_ == rc::media::FORMAT_DEPTH_F32)
        {
            if ((idx + 1) * sizeof(float) > depth_data_.size()) return -1.0;
            const float d = reinterpret_cast<const float*>(depth_data_.data())[idx];
            return (std::isfinite(d) and d > 0.05f) ? static_cast<double>(d) : -1.0;
        }
        if (depth_format_ == rc::media::FORMAT_Z16)
        {
            if ((idx + 1) * sizeof(std::uint16_t) > depth_data_.size()) return -1.0;
            const std::uint16_t mm = reinterpret_cast<const std::uint16_t*>(depth_data_.data())[idx];
            return mm > 0 ? mm * 0.001 : -1.0;
        }
        return -1.0;
    };

    const auto cap_for = [](std::string_view m) -> const LinkCapsule*
    {
        for (const auto& c : LINK_CAPSULES) if (c.mesh == m) return &c;
        return nullptr;
    };

    // FK both the matched q and its ring neighbour, so every sample carries its own
    // predicted depth-velocity vdepth = (Y(q) − Y(q_prev)) / dt → the τ regressor.
    const auto links      = kin_->arm_mesh_link_poses(js.q);
    const bool have_prev  = js.has_prev and std::abs(js.dt_s) > 1e-4;
    const auto links_prev = have_prev ? kin_->arm_mesh_link_poses(js.q_prev)
                                      : std::vector<Kinematics::MeshLinkPose>{};

    constexpr int    S    = 12;                 // samples along each capsule axis
    const double     gate = residual_gate_m_;   // m inlier band (tightened; rejects model breakdown)
    int n_in = 0; double sum2 = 0.0, sum_signed = 0.0;

    for (std::size_t li = 0; li < links.size(); ++li)
    {
        const LinkCapsule* cap = cap_for(links[li].mesh_filename);
        if (not cap) continue;
        const Eigen::Vector3d axis(cap->ax, cap->ay, cap->az), ctr(cap->cx, cap->cy, cap->cz);
        const Eigen::Vector3d a0 = ctr - axis * cap->half_length, a1 = ctr + axis * cap->half_length;
        // Capsule axis endpoints in the camera frame, under q (and q_prev for the velocity).
        const Eigen::Vector3d e0c = (zed_T_arm * (links[li].pose * a0).homogeneous()).head<3>();
        const Eigen::Vector3d e1c = (zed_T_arm * (links[li].pose * a1).homogeneous()).head<3>();
        const bool lp = have_prev and li < links_prev.size();
        const Eigen::Vector3d e0p = lp ? Eigen::Vector3d((zed_T_arm * (links_prev[li].pose * a0).homogeneous()).head<3>()) : Eigen::Vector3d::Zero();
        const Eigen::Vector3d e1p = lp ? Eigen::Vector3d((zed_T_arm * (links_prev[li].pose * a1).homogeneous()).head<3>()) : Eigen::Vector3d::Zero();
        for (int s = 0; s < S; ++s)
        {
            const double t = -cap->half_length + 2.0 * cap->half_length * s / (S - 1);
            const Eigen::Vector3d P = (zed_T_arm * (links[li].pose * (ctr + axis * t)).homogeneous()).head<3>();
            const double X = P.x(), Y = P.y(), Z = P.z();
            if (Y <= 1e-3) continue;                            // behind the camera
            const int u = static_cast<int>(std::lround(dcx_ + dfx_ * X / Y));
            const int v = static_cast<int>(std::lround(dcy_ - dfy_ * Z / Y));
            const double d_obs = depth_at(u, v);
            if (d_obs < 0.0) continue;
            const double d_pred = capsule_near_depth(P, e0c, e1c, cap->radius, Y - cap->radius);
            const double r = d_pred - d_obs;                    // ray–capsule near surface − observed
            if (std::abs(r) > gate) continue;
            sum2 += r * r; sum_signed += r; ++n_in;

            // τ regressor = ∂d_pred/∂t along the SAME pixel ray: the q_prev capsule's
            // near-surface depth at this ray, differenced. Consistent with d_pred (the
            // earlier axis-Y velocity was not, and injected a spurious slope).
            if (lp)
            {
                const double d_pred_prev = capsule_near_depth(P, e0p, e1p, cap->radius, d_pred);
                const double vdepth = (d_pred - d_pred_prev) / js.dt_s;
                reg_samples_.emplace_back(static_cast<float>(vdepth), static_cast<float>(r));
                if (reg_samples_.size() > REG_MAX) reg_samples_.pop_front();
            }
        }
    }

    if (n_in < 5)
    {
        std::print("[calib] depth residual: too few inliers ({}) — arm out of view / extrinsic off?\n", n_in);
        return;
    }
    std::print("[calib] depth rms={:.4f}m bias={:+.4f}m n={} | qdot={:.3f}rad/s match_dt={}ms\n",
               std::sqrt(sum2 / n_in), sum_signed / n_in, n_in, js.qdot_norm, js.match_err_ms);

    if (++reg_frames_ % REG_REPORT_EVERY == 0)
    {
        const auto f = huber_fit(reg_samples_, tau_vspread_min_);
        if (f.ok)
            std::print("[calib-tau] τ={:+.1f}ms  spatial_bias={:+.4f}m  (n={}, inliers={:.0f}%, scale={:.4f}m, vspread={:.3f})\n",
                       -f.slope * 1000.0, f.intercept, reg_samples_.size(),
                       100.0 * f.inlier_frac, f.scale, f.vspread);
        else
            std::print("[calib-tau] τ unidentifiable (n={}, vspread={:.3f} < {:.3f} m/s — move the arm more)\n",
                       reg_samples_.size(), f.vspread, tau_vspread_min_);
    }
}

// RGB intrinsics from the zed node (once): focal from cam_rgb_focalx/y, principal
// point at the RGB image centre.
bool SpecificWorker::ensure_rgb_intrinsics()
{
    if (rgb_intr_ok_) return true;
    if (not G or last_rgb_w_ == 0 or last_rgb_h_ == 0) return false;
    auto zed = G->get_node("zed");
    if (not zed.has_value()) return false;
    const auto fx = G->get_attrib_by_name<cam_rgb_focalx_att>(zed.value());
    const auto fy = G->get_attrib_by_name<cam_rgb_focaly_att>(zed.value());
    if (not (fx.has_value() and fy.has_value())) return false;
    rfx_ = fx.value(); rfy_ = fy.value();
    rcx_ = last_rgb_w_ / 2.0; rcy_ = last_rgb_h_ / 2.0;
    rgb_intr_ok_ = true;
    return true;
}

// Build the aura overlay for the latest RGB frame and push it to the viewer: FK the
// q matched to the RGB stamp, project each link capsule's two axis endpoints (+ its
// perspective radius) into the RGB image, wrap the frame in a QImage. Same FK +
// capsule model as the depth residual, projected with the RGB intrinsics.
void SpecificWorker::build_self_projection()
{
    if (not viewer_ or not kin_ or rgb_data_.empty())
        return;
    if (not ensure_rgb_intrinsics())
        return;
    const auto js = match_joint(last_rgb_stamp_ms_);
    Eigen::Matrix4d zed_T_arm;
    if (not js.valid or not build_extrinsic(zed_T_arm))
        return;

    const int w = static_cast<int>(last_rgb_w_), h = static_cast<int>(last_rgb_h_);
    QImage::Format qfmt = QImage::Format_Invalid;
    if      (rgb_format_ == rc::media::FORMAT_RGB8) qfmt = QImage::Format_RGB888;
    else if (rgb_format_ == rc::media::FORMAT_BGR8) qfmt = QImage::Format_BGR888;
    else return;
    const int step = w * 3;
    if (static_cast<std::size_t>(step) * h > rgb_data_.size())
        return;
    QImage img(rgb_data_.data(), w, h, step, qfmt);   // the viewer deep-copies

    const auto project = [&](const Eigen::Vector3d& p_arm, double r_m, QPointF& uv, double& rpx) -> bool
    {
        const Eigen::Vector4d pc = zed_T_arm * p_arm.homogeneous();
        const double X = pc.x(), Y = pc.y(), Z = pc.z();
        if (Y <= 1e-3) return false;
        uv  = QPointF(rcx_ + rfx_ * X / Y, rcy_ - rfy_ * Z / Y);
        rpx = std::clamp(rfx_ * r_m / Y, 1.0, 80.0);   // ceiling: near links (small Y) would otherwise balloon
        return true;
    };

    std::vector<SelfProjectionViewer::ProjCapsule> caps;
    for (const auto& link : kin_->arm_mesh_link_poses(js.q))
    {
        const LinkCapsule* cap = capsule_for(link.mesh_filename);
        if (not cap) continue;
        const Eigen::Vector3d axis(cap->ax, cap->ay, cap->az), ctr(cap->cx, cap->cy, cap->cz);
        const Eigen::Vector3d e0 = link.pose * (ctr - axis * cap->half_length);
        const Eigen::Vector3d e1 = link.pose * (ctr + axis * cap->half_length);
        SelfProjectionViewer::ProjCapsule pc;
        if (project(e0, cap->radius, pc.a, pc.ra) and project(e1, cap->radius, pc.b, pc.rb))
            caps.push_back(pc);
    }

    // Crisp FK skeleton (base→joint_1..7→tool) for the static projection check.
    std::vector<QPointF> skel;
    for (const auto& p_arm : kin_->arm_skeleton_points(js.q))
    {
        const Eigen::Vector4d pc = zed_T_arm * p_arm.homogeneous();
        const double X = pc.x(), Y = pc.y(), Z = pc.z();
        if (Y <= 1e-3) { skel.clear(); break; }   // a joint behind the camera ⇒ skip the skeleton this frame
        skel.emplace_back(rcx_ + rfx_ * X / Y, rcy_ - rfy_ * Z / Y);
    }

    viewer_->set_overlay(img, std::move(caps), std::move(skel));
}

void SpecificWorker::emergency()
{
    fps.print("Emergency worker", 3000);
    //emergencyCODE
    //
    //if (SUCCESSFUL) //The componet is safe for continue
    //  emmit goToRestore()
}


//Execute one when exiting to emergencyState
void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
    //restoreCODE
    //Restore emergency component

}


int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
	return 0;
}



