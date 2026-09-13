#pragma once

// ── THE RGB CHANNELS, LIFTED OUT OF SpecificWorker ───────────────────────────────────────────────
// One driving camera and any number of calibration-only cameras, each with its own ingestor, its own
// contour extraction, its own evidence and its own replayable pair log. Calibration and driving are
// different jobs and need not use the same sensor, which is the whole reason this is a LIST and not
// a pair of members.
//
// It owns the plumbing (ingestors, extractors, per-channel state, the room polygon they project) and
// borrows what it reports to: the graph, the config, the estimator it asks for poses and corner
// matches, the mount calibrator it feeds pairs to, and the viewer it displays on.
//
// ⚠ The room polygon arrives LATE in Estimate mode. It is loaded from SVG in Given mode and is empty
// until the layout freezes otherwise — see take_room_polygon_from(); three consumers in a row were
// found silently idle for exactly this reason on 2026-09-12.

#include <Eigen/Dense>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "camera_calibration.h"
#include "camera_ingestor.h"
#include "image_edge_source.h"
#include "room_config.h"

namespace DSR { class DSRGraph; }

namespace rc
{
    class RoomConcept;
    class RoomViewer;
    class MountCalibrator;

    class CalibChannels
    {
    public:
        CalibChannels(std::shared_ptr<DSR::DSRGraph> graph, RoomConfig& params, RoomConcept& room,
                      MountCalibrator& mount, RoomViewer** viewer)
            : G(std::move(graph)), params(params), room_concept_(room), mount_(mount),
              viewer_slot_(viewer) {}

        /// Build the driving camera and every entry of ImageEdge.calibCameras. Does nothing when
        /// ImageEdge.enable is false: no subscriber, no thread, no extraction — the feature is exactly
        /// free when off, which is why the construction is here and not unconditional.
        void configure();
        void start();                 ///< begin the ingest threads (Operating-enter)
        void stop();                  ///< drop the readers before the graph goes (shutdown)

        void pump_image_edges();      ///< the driving camera: extraction, once per compute() tick
        void pump_calib_channels();   ///< every other camera: calibration only, never the pose

        /// The room polygon the extraction projects. Set from the SVG in Given mode; in Estimate mode
        /// it must be taken from the frozen layout once LOCALIZING begins.
        void set_room_polygon(std::vector<Eigen::Vector2f> poly, const Eigen::Vector2f& offset);
        const std::vector<Eigen::Vector2f>& room_polygon() const { return room_polygon_; }
        const Eigen::Vector2f& room_polygon_offset() const { return room_polygon_offset_; }
        bool have_room_polygon() const { return room_polygon_.size() >= 3; }

        rc::CameraIngestor* driving_ingestor() const { return camera_ingestor_.get(); }
        rc::ImageEdgeSource* driving_source() const { return image_edge_source_.get(); }

        /// converted/delivered per camera, for the pump report in compute().
        std::string convert_stats_line() const;

        /// Carry a triple point's image corner into the room plane at the LiDAR corner's range, so the
        /// canvas can draw a camera corner beside its LiDAR one. Static: it needs no channel state.
        static void place_triple_points_in_room(rc::ImageEdgeObs& obs, const rc::CameraIngestor& ing,
                                                const Eigen::Vector3f& pose);

    private:
        RoomViewer* viewer() const { return viewer_slot_ != nullptr ? *viewer_slot_ : nullptr; }

        std::shared_ptr<DSR::DSRGraph> G;
        RoomConfig&      params;
        RoomConcept&     room_concept_;
        MountCalibrator& mount_;
        RoomViewer**     viewer_slot_ = nullptr;

        std::unique_ptr<rc::CameraIngestor>  camera_ingestor_;
        std::unique_ptr<rc::ImageEdgeSource> image_edge_source_;
        bool            image_edge_bound_ = false;   ///< bind_camera() succeeded (retried until it does)
        std::int64_t    last_image_edge_log_ms_ = 0;
        Eigen::Vector3f image_edge_prev_pose_ = Eigen::Vector3f::Zero();
        std::int64_t    image_edge_prev_ts_   = 0;

        std::vector<Eigen::Vector2f> room_polygon_;
        Eigen::Vector2f              room_polygon_offset_ = Eigen::Vector2f::Zero();

        /// The driving camera keeps the members above; every OTHER camera in ImageEdge.calibCameras
        /// gets one of these. Auxiliary channels wrote NO pair rows before 2026-09-02 — they
        /// accumulated evidence but left no replayable record, so arm 7's attribution table could not
        /// be replayed from a single drive. One file per camera, like the evidence file beside it.
        struct CalibChannel
        {
            std::string                          name;
            std::unique_ptr<rc::CameraIngestor>  ingestor;
            std::unique_ptr<rc::ImageEdgeSource> source;
            rc::camcal::Estimator                calib;
            std::ofstream                        csv;
            bool                                 bound = false, loaded = false;
            long                                 pairs = 0;
            /// Last push to the Calib window (ms, WALL clock — so a channel resuming evidence from
            /// disk still shows it on a tick where no frame arrived).
            std::int64_t                         viz_ms = 0;
        };
        std::vector<std::unique_ptr<CalibChannel>> calib_channels_;
    };
}   // namespace rc
