// door_cycle_io.h — the DoorCore BOUNDARY: what one compute() cycle of door_concept reads (DoorCycleInput) and what it
// decides (DoorCycleOutput), plus their versioned binary recordings ("DCIN" / "DCOU", v2; v1 still readable).
//
// ★WHY IT EXISTS. door_concept's per-cycle logic is being moved into a DSR-free DoorCore driven by ONE plain struct per
// cycle (docs/superpowers/specs/2026-10-21-door-core-replay-design.md §3.1). This file IS that boundary: every field
// below was derived from an enumeration of the graph/media reads made inside SpecificWorker::compute() and its callees,
// and of every slot that changes state between cycles (.superpowers/sdd/2026-10-21-door-core-replay/task-2-report.md
// carries the file:line table). Nothing in src/core/ may include DSR / Qt / Ice / the media plane; the test
// tests/test_door_cycle_io.cpp enforces it by scanning the includes.
//
// ★WHAT IS RECORDED AND WHAT IS NOT (spec §6.1). FOREIGN inputs — written by other agents or sensors (room geometry,
// ltsm's `current`/recall, the robot pose and twist, masks, sweeps, the semantic posterior, the controller's claims and
// `executing` edges, actuator replies) — are recorded. door_concept's OWN outputs read back from the graph (its door
// nodes' w/h/φ/open_prob, its RT edges, its affordance proposals) are NOT: the core mirrors what it wrote. Two
// deliberate exceptions, both cheap and both because a foreign agent can touch them: the `epistemic_pending` flag the
// controller clears on a door node, and the object-node list (names + poses) the name allocator and the obstacle
// collectors read.
// ★OWN READ-BACKS IN objects[] AND AffordanceFlags ARE FOR FOREIGN MIRROR SOURCES ONLY. objects[] lists our own doors
// and AffordanceFlags our own target x/y/yaw/epoch because one walk records them with the foreign rows; the core must
// take ITS doors' geometry and ITS proposals from its own mirror, and may read those rows only where the source is
// foreign (another room's door seen as a mirror source, a node left by an earlier run, the controller's flags).
//
// ★ASYNCHRONOUS INPUTS BECOME EVENTS (spec §6.4). A node deleted, a door node appearing, a button press — anything
// that today reaches SpecificWorker between two compute() calls — is latched by the adapter and delivered in the NEXT
// cycle's `events`, in arrival order, each stamped with its own steady time. The core API stays a single step(input).
// ★Node events are CLOSED-LOOP on our own state (spec §6.1): the adapter records NodeDeleted / DoorNodeUpserted only for
// nodes NOT created by this agent (created_by() below: the creator is the low agent-id bits of a DSR id, the rule
// common/affordance_manager/affordance_manager.cpp:620 uses), AND the core ignores upserts/deletes of ids it owns —
// both sides, so a recording made by a buggy adapter still replays the core's own lifecycle from its own decisions.
// ★ONE EXCEPTION (ruling, Task 3): a deletion of one of OUR nodes that we did NOT request (ltsm evicting the room we
// leave and our doors with it — the tourP4 defect) is a foreign act. It is recorded as NodeDeleted{by_other_agent =
// true}, and the core MUST apply it. Only the deletions door_concept itself requested are left out.
//
// ★THE ACTUATOR IS A SYNCHRONOUS PORT, NOT AN EVENT (ruling, Task 2 review). DoorActuator::request/poll/refresh_caps
// are blocking Ice calls whose replies today's code reads in the same cycle; delaying them a cycle would change
// behaviour. So the core calls an injected ActuatorPort inside step(), and the live port appends each call — kind,
// arguments, reply — to `actuator_calls`. ★RECORDING ORDER: the adapter builds the input, runs step() with the live
// port, and only THEN writes the input record (transcript filled in). The replay port consumes the transcript in order
// inside step(), checking each call's kind and arguments against the recorded ones (a mismatch = the core diverged).
//
// ★TIME IS AN INPUT (spec §3.3). The core reads no clock: `t_steady_ns` is the steady clock at the top of the cycle
// (nanoseconds, because update_ego_motion divides by sub-millisecond dt), `t_wall_ms` the system clock the anchor
// prediction uses (rc::provenance::now_ms). t_ms() reproduces the decision log's stamp exactly.
//
// ★"PRESENT" vs "UNCHANGED". Big inputs (mask packet, LiDAR sweep, images, semantic plane) are optional: nullopt means
// "nothing new this cycle — keep the last one you were given", which is exactly how today's ingestors behave
// (packet()/frame() return the last staged value). A value means it REPLACES the core's copy.
//
// FILE LAYOUT (little-endian native; static_assert below):
//   "DCIN" | u32 version (2) | u64 len | DoorRecordingHeader payload           ("DCOU" for an output file)
// ★v2 = v1 + LOSSLESS PACKING OF THE BULK ARRAYS (Task 3 live size: tourR1 wrote 25.4 MB/s, 50 % of it mask support
//   points, 17 % mask pixels, 17 % the two bpearl planes, 15 % image patches — and none of them ever repeats between
//   records, so a "same as previous" marker buys nothing). Each bulk array is zstd-compressed after a reversible
//   transform: point clouds / float planes per component, delta-coded on their 32-bit patterns, byte-shuffled; mask
//   pixels as int16 deltas when every value is an exact small integer; gray patches as bytes. And two arrays are
//   stored as the XOR of their bits against a PREDICTION from data already in the record — mask support_points from
//   support_points_cam through room_T_zed_capture (the ingestor's own arithmetic: residual 0 on 100 % of tourR1's
//   points), the bpearl room plane from the robot plane through room_T_robot_capture — with an FNV-64 of the true
//   bits, so a decoder whose prediction differs by one bit reports Corrupt instead of returning wrong data. zstd frames
//   carry their own checksum. Field ORDER inside MaskPacket/BpearlSweep/ImageFrame differs between v1 and v2 (the
//   prediction inputs come first); the reader follows the header's version.
//   then records:  u32 tag 'CYC1' | u64 len | payload
// A record is read whole or not at all, so a file cut mid-write (SIGKILL, disk full, the size cap) yields every complete
// record and nothing else. Doubles/floats are stored as raw bits (NaN payloads and -0.0 survive). No locale anywhere.
#pragma once
#include <array>
#include <bit>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <Eigen/Dense>
#include "../decision_log.h"   // rc::door::DoorDecision — the DDEC stream, so a replay compares like with like

static_assert(std::endian::native == std::endian::little, "DCIN/DCOU v1 store little-endian native; add byte swaps first");

namespace rc::door::core
{

using Vec2f = Eigen::Vector2f;
using Vec3f = Eigen::Vector3f;
using Mat3f = Eigen::Matrix3f;
using Mat4d = Eigen::Matrix4d;
using Mat6d = Eigen::Matrix<double, 6, 6>;

inline constexpr std::uint32_t kCycleIoVersion = 3;     // written
inline constexpr std::uint32_t kCycleIoOldestReadable = 1;   // read (v1: unpacked arrays)

/// DSR ids carry the generating agent's id in their low bits (cortex core/id_generator.h, agent_id_size = 12).
inline constexpr int kDsrAgentIdBits = 12;
[[nodiscard]] constexpr bool created_by(std::uint64_t node_id, std::uint32_t agent_id)
{ return static_cast<std::uint32_t>(node_id & ((std::uint64_t{1} << kDsrAgentIdBits) - 1)) == agent_id; }

// ═══ GRAPH SNAPSHOT (foreign) ═════════════════════════════════════════════════════════════════════════════════════

/// An RT edge as stored: parent id, translation, rotation_euler_xyz (what the wall/door code reads raw).
struct RtEdge
{
    std::uint64_t parent = 0;
    Vec3f translation = Vec3f::Zero();
    Vec3f rotation_euler_xyz = Vec3f::Zero();
    bool operator==(const RtEdge&) const = default;
};

/// Every node of type "room". ★ALL rooms, not just the core's: which room the core is in is core STATE, so recording
/// only that room's data would make the recording depend on the code being replayed.
struct RoomInfo
{
    std::uint64_t id = 0;
    std::string   name;
    bool          is_proto = false;              // rc::room::is_proto
    std::optional<Mat4d> root_T_room;            // root ← room (ts=0); composes room↔room and room↔world transforms
    std::vector<Vec2f>   polygon;                // delimiting_polygon_x/_y (empty = attribute absent)
    std::optional<int>   level;                  // level_att (the entry-mirror layout)
    bool operator==(const RoomInfo&) const = default;
};

struct RoomResolution
{
    std::optional<std::uint64_t> graph_current_room;   // rc::room::current_room(G) — the frame every "room" query resolves
    int n_current_edges = 0;                           // `current` edges pointing at a ROOM node (ltsm_agent's output)
    std::optional<std::uint64_t> current_edge_room;    // its target when exactly one (SpecificWorker::resolve_current_room)
    std::vector<std::string> recall_names;             // ltsm metaconcept `recall_<room>` (subtype room_recall), the <room> part
    std::vector<RoomInfo> rooms;
    bool operator==(const RoomResolution&) const = default;
};

/// Every node of type "wall" (DoorSceneGraph::resolve_wall reads them for the door's parent).
struct WallInfo
{
    std::uint64_t id = 0;
    std::optional<std::uint64_t> parent;           // parent_att (a room, or a floor whose parent is the room)
    std::optional<std::uint64_t> parent_parent;    // that parent's parent_att
    std::optional<RtEdge> rt;                      // RT edge parent→wall
    std::optional<RtEdge> rt_from_room;            // RT edge room→wall (pre-floor graphs)
    std::optional<float>  width_m;
    bool operator==(const WallInfo&) const = default;
};

/// Every node of type "object" or "box" — furniture claims (rc::exclusion::foreign_claims), NBV obstacles
/// (rc::nbv::collect_graph_obstacles), the name allocator (door_* names), restore's live doors. Includes our own.
struct ObjectInfo
{
    std::uint64_t id = 0;
    std::string   name;
    std::string   type;                            // "object" | "box"
    std::optional<std::uint64_t> parent;           // parent_att (is_proto_mirror)
    std::optional<float> width_m, depth_m, height_m;
    std::optional<Mat4d> room_T_node;              // graph-current room ← node (ts=0)
    bool is_anchor = false;                        // rc::interaction::is_anchor
    std::optional<bool>  epistemic_pending;        // the controller CLEARS this on our door nodes (rc::poll_protocol)
    std::optional<float> door_open_prob, door_phi_rad;   // read by the entry-mirror copy (mirrors of foreign rooms)
    bool operator==(const ObjectInfo&) const = default;
};

struct Twist
{
    Vec3f linear = Vec3f::Zero();     // CHILD axes, m/s  (rc::rt::robot_body_twist: {side, advance, 0})
    Vec3f angular = Vec3f::Zero();    // rad/s
    std::uint64_t stamp_ms = 0;
    bool operator==(const Twist&) const = default;
};

struct SimStamps
{
    std::int64_t wall_ms = 0;         // robot_current_speed_timestamp
    std::int64_t sim_ms = 0;          // robot_current_speed_sim_timestamp (present only when simulated)
    bool operator==(const SimStamps&) const = default;
};

/// The robot node (type "robot") and every transform the cycle resolves at ts=0 in the GRAPH-CURRENT room frame.
/// ★ONE snapshot per cycle. Today's code asks cortex for room←zed up to six times a cycle; between those calls
/// room_concept may rewrite the edge. The core sees one value — a (tiny) behaviour change, the price of determinism.
struct RobotState
{
    bool present = false;
    std::string   name;
    std::uint64_t id = 0;
    std::optional<Mat4d> room_T_robot, room_T_body, room_T_zed;   // ts = 0
    std::optional<Mat4d> robot_T_zed, body_T_zed;                 // static mounts
    std::optional<Mat4d> root_T_body;                             // recreate_lost_door
    std::optional<Mat6d> rt_cov;                                  // robot↔room RT edge covariance (rt_api)
    bool rt_cov_edge_robot_to_room = true;                        // which direction the edge carrying it runs
    std::optional<Twist>     twist;                               // robot_current_{advance,side,angular}_speed
    std::optional<SimStamps> sim_stamps;                          // rc::rt::robot_twist_sim_stamps
    std::vector<float>       speed_variance;                      // robot_current_speed_variance
    std::optional<Vec3f>     gt;                                  // robot_gt_x/_y/_angle (sim witness, actuator world frame)
    bool operator==(const RobotState&) const = default;
};

/// The "zed" node's camera model (DSR CameraAPI + the raw attributes the anchor reads).
struct CameraIntrinsics
{
    bool present = false;
    std::uint32_t width = 0, height = 0;
    float focal_x = 0.f, focal_y = 0.f;
    std::uint8_t projection_model = 0;            // DSR::CameraAPI::ProjectionModel
    std::optional<int>   cam_rgb_width, cam_rgb_focalx;
    std::optional<float> cam_fov;
    bool operator==(const CameraIntrinsics&) const = default;
};

// ═══ MEDIA (foreign) ══════════════════════════════════════════════════════════════════════════════════════════════

/// rc::MaskIngestor::MaskSlice, field for field (the core must not include mask_ingestor.h: it pulls DSR).
struct MaskSlice
{
    std::string label;
    float class_id = -1.f, confidence = 0.f;
    Vec3f centroid = Vec3f::Zero(), bbox_min = Vec3f::Zero(), bbox_max = Vec3f::Zero();
    std::uint64_t support_begin = 0, support_end = 0, pixel_begin = 0, pixel_end = 0;
    float motion_var = 0.f, motion_bias = 0.f, motion_dotd = 0.f, trunc_frac = 0.f, centroid_radius = 0.f, range = 0.f;
    bool  has_depth = true;
    float azimuth_room_rad = 0.f;
    std::uint8_t source = 0;                       // rc::MaskSource (0 = zed)
    float depth_var = 0.f;
    Vec3f color_chroma = Vec3f::Zero(), color_var = Vec3f::Zero();
    float color_neff = 0.f;
    bool operator==(const MaskSlice&) const = default;
};

/// One step of an RT chain with its covariance — what InnerGaussianAPI composes (it perturbs each step numerically).
struct RtChainStep
{
    std::string parent, child;
    Mat4d T = Mat4d::Identity();
    std::optional<Mat6d> cov;
    bool inverted = false;                         // traversed child→parent
    bool operator==(const RtChainStep&) const = default;
};

/// rc::MaskIngestor::MasksPacket as delivered (room-frame points already transformed by the ingestor), plus the
/// transforms resolved AT THE CAPTURE STAMP that the core needs and cannot ask for.
struct MaskPacket
{
    bool valid = false;
    int  frame_id = -1;
    std::uint64_t timestamp_ms = 0;
    std::vector<MaskSlice> slices;
    std::vector<Vec3f> support_points;             // room frame
    std::vector<Vec3f> support_points_cam;         // camera frame
    std::vector<Vec2f> mask_pixels;                // raw (col,row)
    std::array<float, 6> cam_twist{};
    float frame_dt_s = 0.f;
    std::optional<Mat4d> room_T_zed_capture;       // the zed→room the ingestor applied (resolve_transform)
    std::optional<Mat4d> room_T_robot_capture;     // room←robot at timestamp_ms, Interpolated (anchor mask frames)
    std::vector<RtChainStep> room_zed_chain;       // room←zed at timestamp_ms with per-edge covariance (chain cov)
    std::int64_t ms_since_last_frame = -1;         // the ingestor's liveness probe at refresh
    /// The RICOH's observation footprint of THIS frame: the ROOM-frame azimuth intervals (x = from, y = to, rad, counter-clockwise)
    /// of the panorama strips the detector actually ran on. The 360 detector segments ONE of three vertical strips per frame
    /// (Ricoh.yolo_strips_per_frame = 1), so a door outside the strip was NOT LOOKED AT - which is not the same as not seen.
    /// EMPTY = unknown (a producer that does not publish it): every bearing counts as looked at, as before. v3 of the tape format.
    std::vector<Vec2f> ricoh_windows;
    bool operator==(const MaskPacket&) const = default;
};

/// rc::ConceptLidarIngestor's low (bpearl) plane after a pump() that staged a new capture.
struct BpearlSweep
{
    bool room_fresh = false;
    std::vector<Vec3f> pts_room;
    Vec3f origin_room = Vec3f::Zero();
    bool robot_fresh = false;
    std::vector<Vec3f> pts_robot;
    Vec3f origin_robot = Vec3f::Zero();
    std::int64_t robot_stamp_ms = 0;
    std::optional<Mat4d> room_T_robot_capture;     // room←robot at robot_stamp_ms, Interpolated (anchor ray frames)
    bool operator==(const BpearlSweep&) const = default;
};

/// The retina's graded semantic posterior, ONLY the door class's plane (door_semantic_field.cpp extracts it).
enum class SemanticState : std::uint8_t { Invalid = 0, Unchanged = 1, Fresh = 2 };
struct SemanticPlane
{
    int pw = 0, ph = 0, frame_w = 0, frame_h = 0;
    std::optional<std::uint64_t> stamp_ms;
    std::vector<float> plane;                      // pw*ph, the door channel
    bool operator==(const SemanticPlane&) const = default;
};

/// RGB/depth for the contour/frame/depth existence checks. ★They score polygons the CORE projects from its own beliefs,
/// so the evidence cannot be derived outside the core: the image itself must cross. The minimal exact form:
///  · RGB as GRAY8 (prepare_frame's first step is BGR→gray; nothing else reads colour) + `frame_ref`, the whole-frame
///    mean |∇| that prepare_frame computes — a frame property, so a patch set does not have to cover the frame;
///  · depth as the WIRE format (Z16 mm or F32 m), which the core decodes exactly as DepthIngestor does.
/// `patches` lets the recorder store crops (with a ≥ 6 px margin: 5×5 blur + 3×3 Sobel + the ±2 px sample search)
/// instead of the full frame; one patch at (0,0,width,height) is the full frame.
enum class PixelFormat : std::uint8_t { Gray8 = 1, DepthF32 = 2, DepthZ16 = 3 };
struct ImagePatch
{
    std::int32_t x0 = 0, y0 = 0, w = 0, h = 0;
    std::vector<std::uint8_t> bytes;               // row-major, w*h*bytes_per_pixel
    bool operator==(const ImagePatch&) const = default;
};
struct ImageFrame
{
    PixelFormat   format = PixelFormat::Gray8;
    std::int32_t  width = 0, height = 0;           // the FULL frame's size (depth scale factors use it)
    std::uint64_t stamp_ms = 0;
    float         frame_ref = 0.f;                 // Gray8 only
    std::vector<ImagePatch> patches;
    bool operator==(const ImageFrame&) const = default;
};

// ═══ AFFORDANCE PROTOCOL (foreign writes on our nodes, and the consumer's edges) ═════════════════════════════════

/// One affordance node door_concept created (epistemic aff_door_N, approach/open/cross/close, door_look_*).
/// The flags are the CONTROLLER's; the target is ours but read back by PragmaticAffordance::write_proposal for its
/// epoch bump, so it is kept (it is 20 bytes).
struct AffordanceFlags
{
    std::uint64_t node_id = 0;
    bool exists = false;
    std::string name;
    std::optional<bool>  active, epistemic_pending;
    std::optional<int>   target_epoch;
    std::optional<float> target_x, target_y, target_yaw;
    bool operator==(const AffordanceFlags&) const = default;
};

/// An `executing` edge (consumer agent node → affordance node), AffordanceManager::read_executing's attributes.
struct ExecutingEdge
{
    std::uint64_t from = 0, to = 0;
    std::optional<int>   epoch;
    std::optional<float> x, y, yaw;
    bool operator==(const ExecutingEdge&) const = default;
};

// ═══ EVENTS (asynchronous, latched by the adapter) ════════════════════════════════════════════════════════════════
// ★Only for nodes NOT created by this agent (see the header note: created_by), and the core ignores ids it owns —
// EXCEPT NodeDeleted{by_other_agent}: one of OUR nodes deleted by somebody else, which the core must apply.

struct NodeDeleted
{
    std::uint64_t id = 0;
    bool by_other_agent = false;                   // true ⇔ the id is OURS and we did not request its deletion
    bool operator==(const NodeDeleted&) const = default;
};

/// update_node_signal on an "object" node named door* (modify_node_slot → ensure_instance). The adapter keeps only the
/// LAST upsert per id per cycle. ⚠`rt_parent` is the RT edge from the node's parent as read; ensure_instance reads
/// room→node, so the core may use it ONLY when rt_parent->parent is the core's current room (a door hung from a wall,
/// or from another room, is placed from objects[].room_T_node or not at all).
struct DoorNodeUpserted
{
    ObjectInfo node;
    std::optional<RtEdge> rt_parent;
    bool operator==(const DoorNodeUpserted&) const = default;
};

/// The strip window's open/close buttons (a Qt callback between cycles).
struct UiActuationRequest
{
    bool open = true;
    std::string provider_id;                       // the combo pick; empty = "(by place)"
    bool operator==(const UiActuationRequest&) const = default;
};

/// The DSR id a node creation the core asked for got (creations happen in the adapter, after step()).
struct NodeIdAssigned
{
    std::uint64_t create_seq = 0;                  // NodeCreateOut::seq / AffordanceOut::create_seq
    std::uint64_t node_id = 0;                     // 0 = the insert failed
    std::string   name;                            // the name it got (DSR names are unique)
    bool operator==(const NodeIdAssigned&) const = default;
};

/// A sweep of our own leftover nodes (startup stale-sweep, first-Operating affordance sweep).
struct OwnedNodesSwept
{
    std::string why;
    std::vector<std::uint64_t> removed;
    bool operator==(const OwnedNodesSwept&) const = default;
};

using DoorEventPayload = std::variant<NodeDeleted, DoorNodeUpserted, UiActuationRequest, NodeIdAssigned, OwnedNodesSwept>;
struct DoorEvent
{
    std::int64_t t_steady_ns = 0;                  // when the adapter latched it
    DoorEventPayload payload;
    bool operator==(const DoorEvent&) const = default;
};

// ═══ ACTUATOR PORT TRANSCRIPT (synchronous calls made INSIDE step()) ═══════════════════════════════════════════════

struct ProviderDoor
{
    std::string id;
    Vec2f xy_mm = Vec2f::Zero();
    float angle = 0.f;
    bool operator==(const ProviderDoor&) const = default;
};
/// DoorActuator::Caps.
struct ActuatorCaps
{
    bool available = false, continuous_angle = false, requires_human = false;
    float typical_latency_s = 0.f;
    int n_doors = 0;
    std::string note;
    std::vector<ProviderDoor> doors;
    bool operator==(const ActuatorCaps&) const = default;
};
/// DoorActuator::Pending.
struct ActuatorPending
{
    int  id = -1;
    std::string door_name;
    bool opening = true;
    std::string state = "—";
    bool settled = false;
    bool operator==(const ActuatorPending&) const = default;
};

/// One call through the port, in call order. Arguments are stored so the replay port can CHECK them (a different
/// call = the core diverged from the recorded run); the reply is what the replay port returns.
enum class ActuatorCallKind : std::uint8_t { RefreshCaps = 0, Request = 1, Poll = 2 };
struct ActuatorCall
{
    ActuatorCallKind kind = ActuatorCallKind::RefreshCaps;
    // Request arguments (DoorActuator::request)
    std::string door_name;
    Vec2f xy_m = Vec2f::Zero();
    float yaw = 0.f, width_m = 0.f;
    bool  open = true;
    Mat4d world_T_room = Mat4d::Identity();
    std::string purpose, provider_id;
    // Poll argument: the standing request as passed in
    std::optional<ActuatorPending> poll_in;
    // Replies: RefreshCaps → caps; Request → pending (nullopt = "no provider"); Poll → pending (the updated request)
    std::optional<ActuatorCaps>    caps;
    std::optional<ActuatorPending> pending;
    bool operator==(const ActuatorCall&) const = default;
};

/// Startup loads (spec §6.6 + review I-4): the identity table and the learned world←room registration as LOADED
/// (raw text — the core parses them with the same parsers), and the room resolved at startup that initialize()
/// enters fresh right after the load.
struct IdentityLoad
{
    bool file_present = false;
    std::string csv_text;                          // etc/door_identities.csv
    std::optional<std::uint64_t> initial_room;
    std::string initial_room_name;
    bool registration_present = false;
    std::string registration_csv_text;             // etc/door_world_registration.csv (DoorWorldRegistration)
    bool operator==(const IdentityLoad&) const = default;
};

// ═══ THE INPUT ═══════════════════════════════════════════════════════════════════════════════════════════════════

struct DoorCycleInput
{
    std::uint64_t cycle = 0;                       // decision-log cycle index (one per compute() past the graph guard)
    std::int64_t  t_steady_ns = 0;
    std::int64_t  t_wall_ms = 0;
    [[nodiscard]] std::int64_t t_ms() const { return t_steady_ns / 1'000'000; }   // == the DDEC stamp

    RoomResolution            rooms;
    std::vector<WallInfo>     walls;
    std::vector<ObjectInfo>   objects;
    RobotState                robot;
    CameraIntrinsics          camera;

    std::optional<MaskPacket>  masks;              // nullopt = no new frame this cycle
    std::int64_t               mask_stream_ms_since_last_frame = -1;
    std::optional<BpearlSweep> bpearl;             // nullopt = pump() staged nothing new
    SemanticState              semantic_state = SemanticState::Invalid;
    std::optional<SemanticPlane> semantic;         // present iff semantic_state == Fresh
    std::optional<ImageFrame>  rgb_gray;           // nullopt = unchanged since the last one given
    std::optional<ImageFrame>  depth;

    std::vector<AffordanceFlags> affordances;
    std::vector<ExecutingEdge>   executing;

    std::optional<IdentityLoad>  identity_load;    // once, on the first recorded cycle
    std::vector<DoorEvent>       events;           // since the previous cycle, in arrival order
    std::vector<ActuatorCall>    actuator_calls;   // filled DURING step() by the live port; record written after it

    bool operator==(const DoorCycleInput&) const = default;
};

// ═══ THE OUTPUT ══════════════════════════════════════════════════════════════════════════════════════════════════
// Every decision the adapter turns into a graph write, plus the diagnostic rows. Actuator calls are NOT here: they are
// made synchronously through the ActuatorPort inside step() and recorded in DoorCycleInput::actuator_calls (and the
// DDEC ActuatorRequest decision). v1 carries the
// shapes the slices are known to need; a slice that needs more bumps the version (the reader refuses other versions).

/// A door's published state after the cycle (what persist_door_belief / the RT-cov publish / attributes write).
struct DoorStateOut
{
    std::uint64_t node_id = 0;
    std::string   name;
    std::uint64_t parent = 0;                      // room or wall
    Mat4d pose = Mat4d::Identity();                // parent ← door
    float width_m = 0.f, depth_m = 0.f, height_m = 0.f;
    float open_prob = 0.f, phi = 0.f, phi_sigma = 0.f;
    float exist_logodds = 0.f, p_exists = 0.f;
    Mat6d cov = Mat6d::Zero();                     // the RT covariance published with it
    std::uint32_t flags = 0;                       // bit 0 ai2_initialized, 1 bearing hypothesis, 2 phi measured, ...
    bool operator==(const DoorStateOut&) const = default;
};

enum class CreateKind : std::uint8_t { DoorFromDetection = 0, DoorCopy = 1, EntryMirror = 2, LookAffordance = 3,
                                       Anchor = 4 };   // Anchor: the passage door's robot-relative copy
struct NodeCreateOut
{
    std::uint64_t seq = 0;                         // answered by NodeIdAssigned{seq} next cycle
    CreateKind kind = CreateKind::DoorFromDetection;
    std::uint64_t parent = 0;
    std::string name, preferred_name;
    Mat4d pose = Mat4d::Identity();
    float width_m = 0.f, depth_m = 0.f, height_m = 0.f;
    std::optional<float> open_prob, phi;
    std::uint64_t source_id = 0;                   // the mirrored / copied door
    std::string note;
    bool operator==(const NodeCreateOut&) const = default;
};

struct NodeDeleteOut
{
    std::uint64_t id = 0;
    std::string why;
    bool operator==(const NodeDeleteOut&) const = default;
};

enum class AffordanceAction : std::uint8_t { Offer = 0, Refresh = 1, Withdraw = 2, Remove = 3 };
struct AffordanceOut
{
    std::uint64_t create_seq = 0;                  // ≠0 on the Offer that CREATES the node: answered by NodeIdAssigned
    std::uint64_t node_id = 0;                     // the affordance node once known (0 until its id is assigned)
    std::uint64_t owner_id = 0;                    // the door (or room, for a look)
    std::string   kind;                            // epistemic | approach | open | cross | close | look
    AffordanceAction action = AffordanceAction::Offer;
    int   epoch = 0;
    float p = 0.f, value = 0.f, x = 0.f, y = 0.f, yaw = 0.f;
    std::string why;
    bool operator==(const AffordanceOut&) const = default;
};

struct AnchorOut
{
    std::uint64_t door_id = 0;
    Vec3f x = Vec3f::Zero();                       // robot frame x, y, yaw
    Mat3f P = Mat3f::Zero();
    bool  drivable = false;
    int   unseen_ms = 0;
    int   epoch = 0;
    float clear_width = 0.f;
    std::string note;
    bool operator==(const AnchorOut&) const = default;
};

struct DiagRow
{
    std::string stream;                            // which CSV/log today's code appends it to
    std::string line;
    bool operator==(const DiagRow&) const = default;
};

struct DoorCycleOutput
{
    std::uint64_t cycle = 0;
    std::int64_t  t_ms = 0;
    std::uint64_t room_id = 0;                     // the room the core follows after this cycle (0 = none)
    std::vector<rc::door::DoorDecision> decisions; // the DDEC stream (decision_log.h kinds)
    std::vector<DoorStateOut>       doors;
    std::vector<NodeCreateOut>      creates;
    std::vector<NodeDeleteOut>      deletes;
    std::vector<AffordanceOut>      affordances;
    std::optional<std::string>      identity_table_csv;   // save etc/door_identities.csv with this content
    std::optional<AnchorOut>        anchor;               // nullopt = no passage anchor
    std::vector<DiagRow>            diag;
    bool operator==(const DoorCycleOutput&) const = default;
};

// ═══ RECORDINGS ══════════════════════════════════════════════════════════════════════════════════════════════════

/// Once per file (spec §3.1 "configuration snapshot (once per file)").
struct DoorRecordingHeader
{
    std::uint64_t agent_id = 0;
    std::string   agent_name;
    std::string   config_text;                     // the config file as loaded, verbatim
    std::int64_t  start_steady_ns = 0;
    std::int64_t  start_wall_ms = 0;
    std::string   note;                            // git rev / run tag / "replay of …"
    bool operator==(const DoorRecordingHeader&) const = default;
};

enum class RecordingKind : std::uint8_t { Input, Output };   // "DCIN" | "DCOU"

/// Header: magic + version + the header record. false on a stream error.
[[nodiscard]] bool write_header(std::ostream& os, const DoorRecordingHeader& h, RecordingKind kind);
/// nullopt on a wrong magic, a wrong version or a short header.
[[nodiscard]] std::optional<DoorRecordingHeader> read_header(std::istream& is, RecordingKind kind,
                                                             std::uint32_t* version = nullptr);

/// One record. The writer serialises to a buffer first and issues ONE write, so a record is on disk whole or cut.
[[nodiscard]] bool write(std::ostream& os, const DoorCycleInput& x);
[[nodiscard]] bool write(std::ostream& os, const DoorCycleOutput& x);
/// One record; nullopt at a clean end, on a truncated record, or on a corrupt one (never a half-filled struct).
/// ⚠READS A RECORD OF THE CURRENT VERSION (v2) ONLY. Handed a v1 file's records it misreads them as Corrupt (the
/// field order of the packed structs differs). For a file of unknown version use read_input_file(), or read_header()'s
/// `version` out-parameter + read_input(is, version).
[[nodiscard]] std::optional<DoorCycleInput>  read_input(std::istream& is);
/// A record of a file whose header said `version` (read_header's out-parameter).
[[nodiscard]] std::optional<DoorCycleInput>  read_input(std::istream& is, std::uint32_t version);

/// The prediction DCIN v2 codes a derived array against (mask support_points from support_points_cam; the bpearl room
/// plane from the robot plane): T · (p, 1) in IEEE double with a FIXED summation order and no fused multiply-add,
/// rounded to float. BUILD-INVARIANT by construction (see door_cycle_io.cpp); exposed for the golden-bits test.
[[nodiscard]] Vec3f derived_prediction(const Mat4d& T, const Vec3f& p);
[[nodiscard]] std::optional<DoorCycleOutput> read_output(std::istream& is);
/// The names the brief asked for: read(std::istream&) for an input record.
[[nodiscard]] inline std::optional<DoorCycleInput> read(std::istream& is) { return read_input(is); }

/// Why a whole-file read stopped: the end of the file on a record boundary, a record cut short (the writer was
/// killed / hit the size cap — expected), or a record that is present but does not decode (a real defect).
enum class StopReason : std::uint8_t { CleanEof = 0, Truncated = 1, Corrupt = 2 };
struct InputRecording
{
    DoorRecordingHeader header;
    std::vector<DoorCycleInput> cycles;
    StopReason    stop = StopReason::CleanEof;
    std::uint64_t stop_offset = 0;                 // byte offset of the record that stopped the read (file size on CleanEof)
};
struct OutputRecording
{
    DoorRecordingHeader header;
    std::vector<DoorCycleOutput> cycles;
    StopReason    stop = StopReason::CleanEof;
    std::uint64_t stop_offset = 0;
};
/// Whole file: nullopt if it cannot be opened or its header is refused; otherwise every COMPLETE record + why it stopped.
[[nodiscard]] std::optional<InputRecording>  read_input_file(const std::string& path);
[[nodiscard]] std::optional<OutputRecording> read_output_file(const std::string& path);

}  // namespace rc::door::core
