// door_cycle_io.h — the DoorCore BOUNDARY: what one compute() cycle of door_concept reads (DoorCycleInput) and what it
// decides (DoorCycleOutput), plus their versioned binary recordings ("DCIN" v1 / "DCOU" v1).
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
// collectors read — it includes our own doors, which the core may ignore in favour of its mirror.
//
// ★ASYNCHRONOUS INPUTS BECOME EVENTS (spec §6.4). A node deleted, a door node appearing, an actuator reply, a button
// press — anything that today reaches SpecificWorker between two compute() calls — is latched by the adapter and
// delivered in the NEXT cycle's `events`, in arrival order, each stamped with its own steady time. The core API stays a
// single step(input).
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
//   "DCIN" | u32 version (1) | u64 len | DoorRecordingHeader payload           ("DCOU" for an output file)
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

inline constexpr std::uint32_t kCycleIoVersion = 1;

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

struct NodeDeleted { std::uint64_t id = 0; bool operator==(const NodeDeleted&) const = default; };

/// update_node_signal on an "object" node named door* (modify_node_slot → ensure_instance). The adapter keeps only the
/// LAST upsert per id per cycle — the signal fires on every attribute write, including our own.
struct DoorNodeUpserted
{
    ObjectInfo node;
    std::optional<RtEdge> rt_parent;               // the RT edge from its parent (ensure_instance reads room→node)
    bool operator==(const DoorNodeUpserted&) const = default;
};

struct ProviderDoor
{
    std::string id;
    Vec2f xy_mm = Vec2f::Zero();
    float angle = 0.f;
    bool operator==(const ProviderDoor&) const = default;
};
struct ActuatorCaps
{
    bool available = false, continuous_angle = false, requires_human = false;
    float typical_latency_s = 0.f;
    int n_doors = 0;
    std::string note;
    std::vector<ProviderDoor> doors;
    bool operator==(const ActuatorCaps&) const = default;
};
struct ActuatorCapsRefreshed { ActuatorCaps caps; bool operator==(const ActuatorCapsRefreshed&) const = default; };

/// DoorActuator::request()/poll() result. ★request() is a SYNCHRONOUS Ice call today; through the core boundary its
/// reply arrives one cycle later as this event (request_seq ties it to the ActuatorRequestOut that asked).
struct ActuatorReply
{
    std::uint64_t request_seq = 0;                 // 0 = a poll() update of the standing request
    bool has_pending = false;                      // false = request() returned nullopt ("no provider")
    int  pending_id = -1;
    std::string door_name;
    bool opening = true;
    std::string state;
    bool settled = false;
    bool operator==(const ActuatorReply&) const = default;
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
    std::uint64_t create_seq = 0;                  // NodeCreateOut::seq
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

using DoorEventPayload = std::variant<NodeDeleted, DoorNodeUpserted, ActuatorCapsRefreshed, ActuatorReply,
                                      UiActuationRequest, NodeIdAssigned, OwnedNodesSwept>;
struct DoorEvent
{
    std::int64_t t_steady_ns = 0;                  // when the adapter latched it
    DoorEventPayload payload;
    bool operator==(const DoorEvent&) const = default;
};

/// spec §6.6: the identity table as LOADED (raw text — the core parses it with the same parse_identity_table), and
/// the room resolved at startup that initialize() enters fresh right after the load.
struct IdentityLoad
{
    bool file_present = false;
    std::string csv_text;
    std::optional<std::uint64_t> initial_room;
    std::string initial_room_name;
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

    bool operator==(const DoorCycleInput&) const = default;
};

// ═══ THE OUTPUT ══════════════════════════════════════════════════════════════════════════════════════════════════
// Every decision the adapter turns into a graph write or an actuator call, plus the diagnostic rows. v1 carries the
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

enum class CreateKind : std::uint8_t { DoorFromDetection = 0, DoorCopy = 1, EntryMirror = 2, LookAffordance = 3 };
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
    std::uint64_t owner_id = 0;                    // the door (or room, for a look)
    std::string   kind;                            // epistemic | approach | open | cross | close | look
    AffordanceAction action = AffordanceAction::Offer;
    int   epoch = 0;
    float p = 0.f, value = 0.f, x = 0.f, y = 0.f, yaw = 0.f;
    std::string why;
    bool operator==(const AffordanceOut&) const = default;
};

struct ActuatorRequestOut
{
    std::uint64_t seq = 0;
    std::string door_name;
    Vec2f xy = Vec2f::Zero();
    float yaw = 0.f, width_m = 0.f;
    bool open = true;
    Mat4d world_T_room = Mat4d::Identity();
    std::string purpose, provider_id;
    bool operator==(const ActuatorRequestOut&) const = default;
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
    std::vector<ActuatorRequestOut> actuator;
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
[[nodiscard]] std::optional<DoorRecordingHeader> read_header(std::istream& is, RecordingKind kind);

/// One record. The writer serialises to a buffer first and issues ONE write, so a record is on disk whole or cut.
[[nodiscard]] bool write(std::ostream& os, const DoorCycleInput& x);
[[nodiscard]] bool write(std::ostream& os, const DoorCycleOutput& x);
/// One record; nullopt at a clean end, on a truncated record, or on a corrupt one (never a half-filled struct).
[[nodiscard]] std::optional<DoorCycleInput>  read_input(std::istream& is);
[[nodiscard]] std::optional<DoorCycleOutput> read_output(std::istream& is);
/// The names the brief asked for: read(std::istream&) for an input record.
[[nodiscard]] inline std::optional<DoorCycleInput> read(std::istream& is) { return read_input(is); }

struct InputRecording  { DoorRecordingHeader header; std::vector<DoorCycleInput>  cycles; };
struct OutputRecording { DoorRecordingHeader header; std::vector<DoorCycleOutput> cycles; };
/// Whole file: nullopt if it cannot be opened or its header is refused; otherwise every COMPLETE record.
[[nodiscard]] std::optional<InputRecording>  read_input_file(const std::string& path);
[[nodiscard]] std::optional<OutputRecording> read_output_file(const std::string& path);

}  // namespace rc::door::core
