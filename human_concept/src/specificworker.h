/*
 *    Copyright (C) 2026 by RoboComp CORTEX Team
 *
 *    This file is part of RoboComp — GNU GPL v3 (see headers of sibling agents).
 */

/**
 * human_concept — Active Inference agent for human body-pose estimation.
 *
 * Consumes BODY_18 3D skeletons through a decoupled SkeletonSource (replay CSV first; a live ZED /
 * media-plane backend later) and, per tracked person, fits an 11-DOF kinematic generative model by
 * the Laplace free-energy update in cpp/core (HumanKinematicModel + AInfLaplacePoseEstimator). The
 * fitted pelvis pose + an uncertainty proxy are written on the room→person RT edge; a reduce-occlusion
 * affordance advertises the next-best-view so the controller can clear occluded joints.
 *
 * Structurally a focused clone of bottle_concept: presence protocol, RT machinery, dashboard and the
 * canonical ensure_instance → observe → run_inference → persist loop are kept; the SDF/mask/support-bank
 * pipeline, the evaluator and the Webots/Ice proxies are dropped (this agent links no Ice interfaces).
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <atomic>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <genericworker.h>
#include <fps/fps.h>
#include <Eigen/Dense>

#include <dsr/api/dsr_rt_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "../../common/dashboard/custom_widget.h"
#include "../../common/dashboard/timeseries_plot.h"
#include "human_config.h"
#include "human_instance.h"
#include "epistemic_planner.h"
#include "skeleton_source.h"
#include "human_scene_graph.h"
#include "human_fitter.h"
#include "prior_store.h"

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
    ~SpecificWorker();

public slots:
    void initialize();
    void compute();
    void emergency();
    void restore();
    int  startup_check();

    void modify_node_slot(std::uint64_t, const std::string&){};
    void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names);
    void modify_edge_slot(std::uint64_t, std::uint64_t, const std::string&){};
    void modify_edge_attrs_slot(std::uint64_t, std::uint64_t, const std::string&, const std::vector<std::string>&){};
    void del_edge_slot(std::uint64_t, std::uint64_t, const std::string&){};
    void del_node_slot(std::uint64_t from);

private:
    // ── Presence protocol (copied from bottle_concept) ──────────────────────────
    void waiting_enter();
    void waiting_loop();
    void operating_enter();
    void operating_loop();
    void degraded_enter();
    void degraded_loop();
    void cleanup_owned_nodes();
    void remove_stale_affordance_nodes();
    void remove_owned_person_nodes();
    void request_shutdown();
    void terminal_shutdown();
    static constexpr int REQUIRED_LOSS_GRACE_MS = 3000;
    void on_optional_peer_lost(const std::string& name, std::uint32_t id);
    void on_optional_peer_ready(const std::string& name, std::uint32_t id);

    // Per-node orchestration (canonical concept-agent loop).
    void process_person_node(const DSR::Node& node);
    void prune_absent_persons();   // delete persons unseen for > DeathFrames cycles

    void step_epistemic(rc::HumanInstance& inst);
    void publish_human_diagnostics(rc::HumanInstance& inst, float free_energy);
    void log_epistemic_csv(const rc::HumanInstance& inst, const rc::EpistemicProposal& prop,
                           const Eigen::Vector2f& camera_xy);
    // Per-cycle fit diagnostics CSV (gated by HumanModel.FitCsvPath): dt, FE, θ, per-joint σ, limits.
    void log_fit_csv(const rc::HumanInstance& inst, float free_energy);

    // ── Members ──────────────────────────────────────────────────────────────
    bool startup_check_flag = false;
    bool owned_nodes_cleaned_ = false;
    std::atomic<bool> shutting_down_{false};
    AgentPresenceCoordinator presence_coordinator_;

    rc::HumanConfig                cfg_;
    rc::EpistemicPlanner           epistemic_planner_;
    std::unique_ptr<rc::PriorStore> prior_store_;
    std::vector<rc::HumanPrior>    priors_cache_;

    Custom_widget*      custom_widget_  = nullptr;
    rc::TimeSeriesPlot* ts_fe_plot_     = nullptr;   // free energy (data misfit)
    rc::TimeSeriesPlot* ts_unc_plot_    = nullptr;   // tr(cov) + valid joints
    rc::TimeSeriesPlot* ts_sigma_plot_  = nullptr;   // posterior σ (elbow L/R)

    std::ofstream epistemic_csv_;
    std::ofstream fit_csv_;

    std::unique_ptr<DSR::RT_API> rt_api_;
    std::unique_ptr<DSR::InnerEigenAPI> inner_eigen_;
    std::uint64_t room_node_id_ = 0;

    std::unique_ptr<rc::SkeletonSource>  skeleton_source_;
    std::unique_ptr<rc::HumanSceneGraph> scene_graph_;
    std::unique_ptr<rc::HumanFitter>     fitter_;

    FPSCounter fps_counter_;   // [Compute] period/fps/cpu/mem heartbeat (printed every 3 s)

signals:
    void presenceReady();
    void presenceLost();
};

#endif // SPECIFICWORKER_H
