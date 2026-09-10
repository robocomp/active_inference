#pragma once

/*
 * place_stage.h — the retina end of the panoramic place memory.
 *
 * Encodes the ricoh panorama into per-sector appearance descriptors (PlaceEncoder), tags them with
 * the robot's room pose AND that pose's covariance, and either logs them for offline evaluation or
 * accumulates them into a persistent map (rc::place::PlaceMap).
 *
 * ★ NO PerceptionResult SLOT, DELIBERATELY. perception_stage.h states the recipe as "a Stage + a
 * PerceptionResult slot + a Publisher route". This stage has no slot and no publish, because in this
 * pass NOTHING READS IT: the outputs go to files this stage owns. 161 of 292 published graph
 * attributes in this fleet have no consumer, and the fix for that is to not add the 162nd before the
 * reader exists. The slot arrives when room_concept's grid_search Stage 1 is wired to consume the
 * mixture. This is a reasoned deviation, not an omission.
 *
 * ★ THE ORDERING OF THE WHOLE FEATURE IS "LOGGER BEFORE MAP". The map's constants -- the softmax
 * temperature, the similarity-decay radius, the insertion density -- are OUTPUTS of place_eval, not
 * inputs to it. So `build_map` defaults off and `log_queries` is what you turn on first.
 *
 * THREADING: run() is on the ricoh PerceptionWorker thread. It touches the DSR graph, which is safe
 * (DSRGraph serialises with a shared_mutex) provided transforms are pinned to a real timestamp --
 * ts != 0 bypasses InnerEigenAPI's unlocked cache, which is the one genuine cliff. See CLAUDE.md.
 */

#include "perception_stage.h"
#include "place_encoder.h"

#include "../../common/place_memory/place_map.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

namespace DSR { class DSRGraph; class RT_API; class InnerEigenAPI; }

namespace rc
{

struct PlaceStageConfig
{
    rc::place::EncoderConfig encoder;
    int         decimation        = 10;      // ~2 Hz on the 50 ms ricoh worker; the map needs no more
    bool        build_map         = false;
    float       insert_min_dist_m = 0.5f;
    int         save_every_n      = 20;      // never write the map from inside a frame
    std::string map_path          = "etc/place_map.csv";
    std::string map_blob_path     = "etc/place_map.bin";
    bool        log_queries       = false;
    bool        log_grid          = false;   // raw patch grid, fp16 — 393 KB/frame. Capture runs only.
    int         log_stride        = 1;
    std::string log_dir           = "etc/place_log";
    float       azimuth_tune_deg  = 0.f;     // recorded into the map header; see place_map.h
};

class PlaceStage : public Stage
{
public:
    PlaceStage(const PlaceStageConfig& cfg, std::shared_ptr<DSR::DSRGraph> graph);
    ~PlaceStage() override;

    const char* name() const override { return "place"; }
    bool ready() const override { return encoder_ and encoder_->ready(); }
    void run(const PerceptionFrame& in, PerceptionResult& out) override;

    /// Persist the map. Called on shutdown; also every save_every_n insertions.
    void save_map();

    // Status for a UI line. const + copy-out: the MAIN thread calls these while this worker thread
    // may be inside run() (the compose() convention every other stage here follows).
    [[nodiscard]] std::size_t map_size() const;
    [[nodiscard]] std::string last_summary() const;

private:
    /// room<-robot pose and its ROOM-FRAME SE(2) covariance, at `stamp`.
    /// ★ The RT edge stores robot->room once the room node exists, with a covariance already pushed
    /// through the inversion Jacobian; this undoes both. Returns false if the chain is incomplete --
    /// a keyframe with no pose is worse than no keyframe.
    bool robot_pose_in_room(std::uint64_t stamp, Eigen::Vector3f& pose, Eigen::Matrix3f& cov) const;

    void open_logs();
    void log_query(const PerceptionFrame& in, const std::vector<float>& desc,
                   const std::vector<float>& grid,
                   const Eigen::Vector3f& ricoh_pose, const Eigen::Vector3f& robot_pose,
                   const Eigen::Matrix3f& cov);

    PlaceStageConfig                       cfg_;
    std::shared_ptr<DSR::DSRGraph>         graph_;
    std::unique_ptr<DSR::RT_API>           rt_api_;
    std::unique_ptr<DSR::InnerEigenAPI>    inner_;
    std::unique_ptr<rc::place::PlaceEncoder> encoder_;
    rc::place::PlaceMap                    map_;

    std::uint64_t counter_       = 0;
    std::uint32_t next_id_       = 0;
    int           since_save_    = 0;
    bool          self_tested_   = false;

    std::ofstream qlog_;
    std::ofstream qblob_;
    std::uint64_t blob_index_    = 0;

    mutable std::mutex status_mx_;
    std::string   summary_;
    std::size_t   map_size_      = 0;
};

}   // namespace rc
