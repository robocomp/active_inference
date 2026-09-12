#pragma once

// ── CAMERA↔LiDAR MOUNT CALIBRATION, LIFTED OUT OF SpecificWorker ─────────────────────────────────
// 691 lines of specificworker.cpp were this: the pose-free camera-vs-LiDAR mount solve, its two
// accumulators, the per-camera publish bookkeeping, the replayable pair logs and the sensor-triangle
// loop closure. None of it is coordination and none of it is estimation or localisation — it is a
// self-contained measurement channel that happens to run on the worker's thread, which is exactly
// what belongs in its own class.
//
// It owns every piece of state it needs. The only things it borrows are the graph (it mirrors a
// measured mount into the body→camera RT edge), the config (the prior sigmas that give its
// corrections physical units), and the viewer (it displays what it has measured, and nothing reads
// back from there). The ingestors are passed per call, because calibration and driving are different
// jobs and need not use the same camera.

#include <Eigen/Dense>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "camera_calibration.h"
#include "camera_ingestor.h"
#include "corner_detector.h"
#include "image_edge_types.h"
#include "mount_lidar_pair.h"
#include "room_config.h"

namespace DSR { class DSRGraph; }
namespace rc { class RoomViewer; }

namespace rc
{
    class MountCalibrator
    {
    public:
        /// `viewer` may be null (headless); every use of it is guarded.
        MountCalibrator(std::shared_ptr<DSR::DSRGraph> graph, RoomConfig& params, RoomViewer** viewer)
            : G(std::move(graph)), params(params), viewer_slot_(viewer) {}

        /// The persistent evidence pool of the DRIVING camera. Public because the camera-reset
        /// handler clears it and because the pumps feed and read it directly.
        rc::camcal::Estimator& pool()       { return mp_pool_; }
        rc::mount::Accum&      window()     { return mp_win_; }
        bool  evidence_loaded() const       { return mp_loaded_; }
        void  set_evidence_loaded(bool v)   { mp_loaded_ = v; }

        /// Clear both accumulators and delete the evidence file — the "Reset" button's action.
        void reset_evidence()
        {
            mp_pool_.reset();
            mp_win_.reset();
            std::remove(mp_pool_.path().c_str());
        }

        /// The DRIVING camera's ingestor. mount_pair_update() works on that one specifically — the
        /// auxiliary channels pass their own ingestor per call — and it is set once it exists, which
        /// is after this object is constructed. Null until then, and every use is guarded.
        void set_driving_ingestor(rc::CameraIngestor* ing) { driving_ = ing; }

        void mount_pair_update(const rc::ImageEdgeObs& obs,
                               const std::vector<rc::CornerDetector::CornerMatch>& matches,
                               std::int64_t timestamp_ms);
        void open_pair_log(std::ofstream& csv, const std::string& cam, const rc::CameraIngestor& ing);
        void push_mount_correction(rc::CameraIngestor& ing, const Eigen::Vector4d& applied,
                                   const std::string& cam, const char* why);
        void apply_mount_solve(rc::camcal::Estimator& pool, rc::CameraIngestor& ing,
                               const rc::mount::Accum::Solution& sol, const std::string& cam);
        void publish_mount_to_graph(rc::camcal::Estimator& pool, const rc::CameraIngestor& ing,
                                    const std::string& cam);
        void reconcile_mount_nominal(rc::camcal::Estimator& pool, rc::CameraIngestor& ing,
                                     const std::string& cam);
        void loop_closure_observe(const std::string& cam, int vertex, bool ceiling,
                                  double du_rad, double dv_rad, std::int64_t ts);
        static void write_pair_row(std::ofstream& csv, const std::string& cam, std::int64_t ts,
                                   const rc::mount::PairObs& pr, bool ceiling, float angle_deg,
                                   float assoc_chi2, int n_rivals, float runnerup_chi2,
                                   const Eigen::Vector3f& corr);

    private:
        RoomViewer* viewer() const { return viewer_slot_ != nullptr ? *viewer_slot_ : nullptr; }

        std::shared_ptr<DSR::DSRGraph> G;
        RoomConfig&  params;
        RoomViewer** viewer_slot_ = nullptr;   ///< the worker's unique_ptr slot; may hold null
        rc::CameraIngestor* driving_ = nullptr;

        // ── THE SENSOR TRIANGLE ──────────────────────────────────────────────────────────────────
        // Each camera's residual against the LiDAR is (camera error) + (LiDAR corner error).
        // Differencing two of them CANCELS the LiDAR term and leaves camera-vs-camera — the only
        // statement here that needs no ground truth. Held in RADIANS: a pixel is a different angle on
        // each camera and the two are otherwise not comparable at all.
        struct CornerAngle { double du_rad = 0, dv_rad = 0; std::int64_t ts = 0; };
        std::map<std::pair<std::string, int>, CornerAngle> loop_last_;   // (camera, vertex*2+ceiling)
        double loop_du_sum_ = 0, loop_dv_sum_ = 0, loop_du_sq_ = 0, loop_dv_sq_ = 0;
        long   loop_n_ = 0;
        std::ofstream loop_csv_;

        // TWO accumulators on purpose. mp_win_ resets every window and is directly comparable with
        // stage 1's per-window solve; mp_pool_ never resets. Pooling is only legitimate if the pose
        // was the dominant between-window nuisance, which is a CLAIM — running both is what tests it.
        rc::mount::Accum      mp_win_;
        rc::camcal::Estimator mp_pool_;
        bool             mp_loaded_ = false;
        std::int64_t     mp_win_start_ms_ = 0;
        long             mp_wins_ = 0, mp_seen_ = 0, mp_paired_ = 0;
        std::ofstream    mp_csv_;
        Eigen::Vector4d  mp_sum_ = Eigen::Vector4d::Zero(), mp_sum2_ = Eigen::Vector4d::Zero();
        long             mp_sum_n_ = 0;

        /// Per camera: the mount this SESSION started from, plus the last correction written. The
        /// nominal is captured once so a later window composes `nominal x correction` and never
        /// `edge x correction`, which would compound this class's own previous write.
        struct MountPublishState
        {
            Eigen::Vector3f nominal_t = Eigen::Vector3f::Zero();
            Eigen::Vector3f nominal_r = Eigen::Vector3f::Zero();
            Eigen::Vector3f last      = Eigen::Vector3f::Zero();
            bool have_nominal = false, have_last = false;
            bool checked = false;
        };
        std::map<std::string, MountPublishState> mount_publish_;
        bool mount_apply_refused_logged_ = false;
        bool mount_publish_refused_logged_ = false;
    };
}   // namespace rc
