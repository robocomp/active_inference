// input_snapshot.h — the ADAPTER half of door_concept's input recording (Task 3 of the DoorCore record/replay plan,
// docs/superpowers/specs/2026-10-21-door-core-replay-design.md §3.1/§3.3/§6.4): it fills one core::DoorCycleInput per
// compute() cycle from the graph and the media plane, latches what arrives between cycles, and hands the finished
// record to DoorInputRecorder AFTER the cycle (so the actuator transcript is complete).
//
// ★RECORDED ALONGSIDE, NOTHING CHANGES. Today's decision logic keeps reading the graph exactly as before; this class
//   only READS (the same reads the enumeration in task-2-report.md §1a lists) and never writes the graph. With
//   DoorConcept.RecordInputs = false (the default) every method returns at its first line.
// ★WHERE EACH PART IS TAKEN. The graph part (rooms, walls, objects, robot, camera, affordance flags, executing edges)
//   at the TOP of compute(), right after the decision-log clock; the media parts at the point the old code consumes
//   them — the mask packet after MaskIngestor::refresh() (and at the top, if a refresh between cycles changed it), the
//   bpearl sweep after each ConceptLidarIngestor::pump() that staged one, the semantic plane after its refresh, and
//   RGB/depth only on cycles the existence update actually scores them, CROPPED to the door ROIs it sampled (ruling:
//   "images recorded only on cycles that use them, cropped to the door ROI patches").
// ★FOREIGN ONLY (ruling, Task 2 review I-3): object/wall rows and node events of nodes created by THIS agent
//   (core::created_by) are dropped. The one place own nodes appear is `affordances[]`: our affordance nodes and our
//   door nodes carry flags the CONTROLLER writes (active / epistemic_pending), which are foreign facts.
//   And one deletion kind: one of OUR nodes deleted by somebody else (NodeDeleted{by_other_agent}, ruling Task 3).
//   door_concept's own deletions are told apart by self_scope() (input_latch.h). Strip-button presses are latched as
//   UiActuationRequest events; the target resolution they do between cycles (nearest_door_for_actuation) is NOT
//   snapshotted — the core redoes it on the next cycle's snapshot.
// Main thread only (ts==0 transforms through the main thread's InnerEigenAPI; CLAUDE.md "DSR graph thread-safety").
#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_rt_api.h>
#include "core/door_cycle_io.h"
#include "input_latch.h"
#include "input_recorder.h"

namespace rc { class MaskIngestor; class ConceptLidarIngestor; class SemanticProbField; }

namespace rc::door
{

class InputSnapshot
{
public:
    /// Start latching (called from initialize() when DoorConcept.RecordInputs is on, before the identity load, so
    /// the startup events are caught even though the file is opened at the end of initialize()).
    void arm(std::uint32_t agent_id) { armed_ = true; latch_.set_agent_id(agent_id); agent_id_ = agent_id; }
    [[nodiscard]] bool armed() const { return armed_; }
    /// Open the tape. A failure DISARMS (nothing would ever drain the latch).
    [[nodiscard]] bool open(const std::string& path, const core::DoorRecordingHeader& h, std::uint64_t max_bytes)
    {
        const bool ok = armed_ and recorder_.open(path, h, max_bytes);
        if (not ok) armed_ = false;
        return ok;
    }
    /// Recording right now (armed, file open, cap not reached).
    [[nodiscard]] bool recording() const { return armed_ and recorder_.enabled(); }
    void close() { recorder_.close(); }
    [[nodiscard]] const DoorInputRecorder& recorder() const { return recorder_; }

    // ── between-cycle events (latched while armed) ────────────────────────────────────────────────────────────────
    void on_node_deleted(std::uint64_t id);
    /// modify_node_slot's door branch: a door node that ensure_instance is about to adopt.
    void on_door_upserted(DSR::DSRGraph& G, DSR::InnerEigenAPI* inner, const DSR::Node& node);
    void on_ui_request(bool open, const std::string& provider_id);
    void on_owned_swept(const std::string& why, std::vector<std::uint64_t> removed);
    /// The startup load: raw text of both files as initialize() reads them, and the room it then enters.
    void on_identity_load(const std::string& identity_path, const std::string& registration_path,
                          std::optional<std::uint64_t> initial_room, const std::string& initial_room_name);
    /// door_concept's own code is running: deletions it causes are self-requested (input_latch.h). RAII, nests.
    [[nodiscard]] InputLatch::SelfScope self_scope() { return InputLatch::SelfScope{latch_}; }
    /// One call through the actuator port (in a cycle → that cycle; from a Qt callback → the next one).
    void on_actuator_call(core::ActuatorCall c) { if (armed_ and not recorder_.stopped()) latch_.actuator_call(std::move(c)); }

    // ── the cycle ────────────────────────────────────────────────────────────────────────────────────────────────
    void begin_cycle(DSR::DSRGraph& G, DSR::InnerEigenAPI* inner, DSR::RT_API* rt, std::uint64_t cycle,
                     std::int64_t t_steady_ns, std::int64_t t_wall_ms, const rc::MaskIngestor* masks);
    /// After MaskIngestor::refresh() (or at any point the packet may have changed): records it if it differs.
    void capture_masks(DSR::DSRGraph& G, DSR::InnerEigenAPI* inner, DSR::RT_API* rt, const rc::MaskIngestor& m);
    /// After ConceptLidarIngestor::pump(): records the bpearl planes it staged (nothing if none).
    void capture_bpearl(DSR::DSRGraph& G, DSR::InnerEigenAPI* inner, const rc::ConceptLidarIngestor& l);
    /// After SemanticProbField::refresh().
    void capture_semantic(const rc::SemanticProbField& f, bool refreshed_ok);
    /// The existence update is about to score the images: which frames (nullptr = not used this cycle).
    void note_images(const cv::Mat* bgr, std::uint64_t rgb_stamp_ms, float frame_ref,
                     const cv::Mat* depth, std::uint64_t depth_stamp_ms);
    /// A closed polygon (full RGB frame pixels) whose BOUNDARY a check samples (the leaf quad / its controls), with the
    /// extra reach of that check beyond the boundary, and which image(s) the check reads there. Recorded as short
    /// chunks along each edge, not the polygon's box: the checks never read the interior.
    void note_roi(const std::vector<cv::Point>& poly, int extra_px, bool rgb, bool depth);
    /// One sampled pixel (the aperture frame's samples / controls).
    void note_roi(const cv::Point2f& p, int extra_px, bool rgb, bool depth);
    /// Crop the noted images, write the record, close the cycle. Safe to call when not recording.
    void end_cycle();
    [[nodiscard]] bool in_cycle() const { return in_cycle_; }

    /// The margin around every ROI: 5x5 blur + 3x3 Sobel + the ±2 px sample search + the 3..6 px depth reads, with
    /// slack. Interior gradients/depths of a patch are then bit-identical to the full frame's (task-2-report.md §2.5).
    static constexpr int kRoiMarginPx = 12;

private:
    struct Rect { int x0, y0, x1, y1; };   // inclusive-exclusive, RGB frame pixels
    static void add_rect(std::vector<Rect>& rois, Rect r);
    void add(Rect r, bool rgb, bool depth) { if (rgb) add_rect(rois_rgb_, r); if (depth) add_rect(rois_depth_, r); }
    static constexpr int kEdgeChunkPx = 32;   // an edge is recorded as boxes this long (+ margin) — see note_roi

    bool armed_ = false;
    std::uint32_t agent_id_ = 0;
    InputLatch latch_;
    DoorInputRecorder recorder_;
    core::DoorCycleInput cur_;
    bool in_cycle_ = false;

    // "changed since the last record" keys
    struct MaskKey { bool valid; int frame_id; std::uint64_t ts; std::size_t n_slices, n_support, n_pixels;
                     bool operator==(const MaskKey&) const = default; };
    std::optional<MaskKey> last_mask_;
    bool semantic_touched_ = false;
    bool last_semantic_valid_ = false;
    std::optional<std::uint64_t> last_semantic_stamp_;

    // images noted this cycle
    const cv::Mat* rgb_ = nullptr;
    const cv::Mat* depth_ = nullptr;
    std::uint64_t rgb_stamp_ = 0, depth_stamp_ = 0;
    float frame_ref_ = 0.f;
    std::vector<Rect> rois_rgb_, rois_depth_;
};

}  // namespace rc::door
