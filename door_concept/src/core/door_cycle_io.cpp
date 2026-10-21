// door_cycle_io.cpp — binary (de)serialisation of the DoorCore boundary types. See door_cycle_io.h for the layout.
//
// ★ONE FIELD LIST PER STRUCT, USED BY BOTH DIRECTIONS. Each struct has a single `fields(ar, x)` that names its members
// once; the Writer and the Reader are both "archives" applied to it. A writer and a reader maintained separately drift
// — a field added to one and not the other is a silent misread of everything after it — and this file exists so that
// recorded data can be trusted bit for bit.
#include "door_cycle_io.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <istream>
#include <ostream>
#include <type_traits>

namespace rc::door::core
{
namespace
{

constexpr std::uint32_t kRecordTag = 0x31435943u;   // "CYC1" little-endian
constexpr std::uint64_t kMaxRecord = 1ull << 32;    // 4 GB: a length beyond it is corruption, not data

template <class T> struct is_vector : std::false_type {};
template <class T, class A> struct is_vector<std::vector<T, A>> : std::true_type {};
template <class T> struct is_optional : std::false_type {};
template <class T> struct is_optional<std::optional<T>> : std::true_type {};
template <class T> struct is_std_array : std::false_type {};
template <class T, std::size_t N> struct is_std_array<std::array<T, N>> : std::true_type {};
template <class T> struct is_variant : std::false_type {};
template <class... Ts> struct is_variant<std::variant<Ts...>> : std::true_type {};
template <class T> concept EigenFixed = requires { typename T::Scalar; T::RowsAtCompileTime; T::ColsAtCompileTime; }
                                        and (T::RowsAtCompileTime > 0) and (T::ColsAtCompileTime > 0);

// ─── the archives ────────────────────────────────────────────────────────────────────────────────────────────────
struct Writer
{
    std::string b;
    template <class T> void raw(const T& x)
    {
        const auto* p = reinterpret_cast<const char*>(&x);
        b.append(p, sizeof(T));
    }
    template <class... Ts> void operator()(const Ts&... xs) { (one(xs), ...); }
    template <class T> void one(const T& x);
};

struct Reader
{
    const char* p;
    const char* e;
    bool ok = true;
    template <class T> void raw(T& x)
    {
        if (not ok or static_cast<std::size_t>(e - p) < sizeof(T)) { ok = false; return; }
        std::memcpy(&x, p, sizeof(T));
        p += sizeof(T);
    }
    [[nodiscard]] std::size_t left() const { return static_cast<std::size_t>(e - p); }
    template <class... Ts> void operator()(Ts&... xs) { (one(xs), ...); }
    template <class T> void one(T& x);
};

// ─── the field lists (one per struct; read AND write) ───────────────────────────────────────────────────────────
// A struct's members are listed by a lambda-free template so `Ar` may be a Writer (const view) or a Reader.
#define DCIO_FIELDS(Type, ...)                                                                 \
    template <class Ar> void fields(Ar& ar, Type& x) { ar(__VA_ARGS__); }                      \
    template <class Ar> void fields(Ar& ar, const Type& cx) { auto& x = const_cast<Type&>(cx); ar(__VA_ARGS__); }

DCIO_FIELDS(RtEdge, x.parent, x.translation, x.rotation_euler_xyz)
DCIO_FIELDS(RoomInfo, x.id, x.name, x.is_proto, x.root_T_room, x.polygon, x.level)
DCIO_FIELDS(RoomResolution, x.graph_current_room, x.n_current_edges, x.current_edge_room, x.recall_names, x.rooms)
DCIO_FIELDS(WallInfo, x.id, x.parent, x.parent_parent, x.rt, x.rt_from_room, x.width_m)
DCIO_FIELDS(ObjectInfo, x.id, x.name, x.type, x.parent, x.width_m, x.depth_m, x.height_m, x.room_T_node, x.is_anchor,
            x.epistemic_pending, x.door_open_prob, x.door_phi_rad)
DCIO_FIELDS(Twist, x.linear, x.angular, x.stamp_ms)
DCIO_FIELDS(SimStamps, x.wall_ms, x.sim_ms)
DCIO_FIELDS(RobotState, x.present, x.name, x.id, x.room_T_robot, x.room_T_body, x.room_T_zed, x.robot_T_zed, x.body_T_zed,
            x.root_T_body, x.rt_cov, x.rt_cov_edge_robot_to_room, x.twist, x.sim_stamps, x.speed_variance, x.gt)
DCIO_FIELDS(CameraIntrinsics, x.present, x.width, x.height, x.focal_x, x.focal_y, x.projection_model, x.cam_rgb_width,
            x.cam_rgb_focalx, x.cam_fov)
DCIO_FIELDS(MaskSlice, x.label, x.class_id, x.confidence, x.centroid, x.bbox_min, x.bbox_max, x.support_begin,
            x.support_end, x.pixel_begin, x.pixel_end, x.motion_var, x.motion_bias, x.motion_dotd, x.trunc_frac,
            x.centroid_radius, x.range, x.has_depth, x.azimuth_room_rad, x.source, x.depth_var, x.color_chroma,
            x.color_var, x.color_neff)
DCIO_FIELDS(RtChainStep, x.parent, x.child, x.T, x.cov, x.inverted)
DCIO_FIELDS(MaskPacket, x.valid, x.frame_id, x.timestamp_ms, x.slices, x.support_points, x.support_points_cam,
            x.mask_pixels, x.cam_twist, x.frame_dt_s, x.room_T_zed_capture, x.room_T_robot_capture, x.room_zed_chain,
            x.ms_since_last_frame)
DCIO_FIELDS(BpearlSweep, x.room_fresh, x.pts_room, x.origin_room, x.robot_fresh, x.pts_robot, x.origin_robot,
            x.robot_stamp_ms, x.room_T_robot_capture)
DCIO_FIELDS(SemanticPlane, x.pw, x.ph, x.frame_w, x.frame_h, x.stamp_ms, x.plane)
DCIO_FIELDS(ImagePatch, x.x0, x.y0, x.w, x.h, x.bytes)
DCIO_FIELDS(ImageFrame, x.format, x.width, x.height, x.stamp_ms, x.frame_ref, x.patches)
DCIO_FIELDS(AffordanceFlags, x.node_id, x.exists, x.name, x.active, x.epistemic_pending, x.target_epoch, x.target_x,
            x.target_y, x.target_yaw)
DCIO_FIELDS(ExecutingEdge, x.from, x.to, x.epoch, x.x, x.y, x.yaw)
DCIO_FIELDS(NodeDeleted, x.id, x.by_other_agent)
DCIO_FIELDS(DoorNodeUpserted, x.node, x.rt_parent)
DCIO_FIELDS(ProviderDoor, x.id, x.xy_mm, x.angle)
DCIO_FIELDS(ActuatorCaps, x.available, x.continuous_angle, x.requires_human, x.typical_latency_s, x.n_doors, x.note,
            x.doors)
DCIO_FIELDS(ActuatorPending, x.id, x.door_name, x.opening, x.state, x.settled)
DCIO_FIELDS(ActuatorCall, x.kind, x.door_name, x.xy_m, x.yaw, x.width_m, x.open, x.world_T_room, x.purpose, x.provider_id,
            x.poll_in, x.caps, x.pending)
DCIO_FIELDS(UiActuationRequest, x.open, x.provider_id)
DCIO_FIELDS(NodeIdAssigned, x.create_seq, x.node_id, x.name)
DCIO_FIELDS(OwnedNodesSwept, x.why, x.removed)
DCIO_FIELDS(DoorEvent, x.t_steady_ns, x.payload)
DCIO_FIELDS(IdentityLoad, x.file_present, x.csv_text, x.initial_room, x.initial_room_name, x.registration_present,
            x.registration_csv_text)
DCIO_FIELDS(DoorCycleInput, x.cycle, x.t_steady_ns, x.t_wall_ms, x.rooms, x.walls, x.objects, x.robot, x.camera, x.masks,
            x.mask_stream_ms_since_last_frame, x.bpearl, x.semantic_state, x.semantic, x.rgb_gray, x.depth,
            x.affordances, x.executing, x.identity_load, x.events, x.actuator_calls)
DCIO_FIELDS(DoorStateOut, x.node_id, x.name, x.parent, x.pose, x.width_m, x.depth_m, x.height_m, x.open_prob, x.phi,
            x.phi_sigma, x.exist_logodds, x.p_exists, x.cov, x.flags)
DCIO_FIELDS(NodeCreateOut, x.seq, x.kind, x.parent, x.name, x.preferred_name, x.pose, x.width_m, x.depth_m, x.height_m,
            x.open_prob, x.phi, x.source_id, x.note)
DCIO_FIELDS(NodeDeleteOut, x.id, x.why)
DCIO_FIELDS(AffordanceOut, x.create_seq, x.node_id, x.owner_id, x.kind, x.action, x.epoch, x.p, x.value, x.x, x.y, x.yaw, x.why)
DCIO_FIELDS(AnchorOut, x.door_id, x.x, x.P, x.drivable, x.unseen_ms, x.epoch, x.clear_width, x.note)
DCIO_FIELDS(DiagRow, x.stream, x.line)
DCIO_FIELDS(DoorCycleOutput, x.cycle, x.t_ms, x.room_id, x.decisions, x.doors, x.creates, x.deletes, x.affordances,
            x.identity_table_csv, x.anchor, x.diag)
DCIO_FIELDS(DoorRecordingHeader, x.agent_id, x.agent_name, x.config_text, x.start_steady_ns, x.start_wall_ms, x.note)
#undef DCIO_FIELDS
}  // namespace
}  // namespace rc::door::core

// The DDEC record lives in rc::door; its field list is declared there so argument-dependent lookup finds it.
namespace rc::door
{
template <class Ar> void fields(Ar& ar, const DoorDecision& cx)
{
    auto& x = const_cast<DoorDecision&>(cx);
    ar(x.cycle, x.t_ms, x.kind, x.door_id, x.name, x.a, x.b, x.c, x.d, x.note);
}
}  // namespace rc::door

namespace rc::door::core
{
namespace
{

// ─── enum ranges: a byte outside them is corruption, not a new kind ─────────────────────────────────────────────
// ★One line per serialised enum. A new enum without a line here fails to COMPILE (enum_ok has no overload), so the
//   check cannot be forgotten.
constexpr bool enum_ok(SemanticState v)    { return static_cast<unsigned>(v) <= 2u; }
constexpr bool enum_ok(PixelFormat v)      { return static_cast<unsigned>(v) >= 1u and static_cast<unsigned>(v) <= 3u; }
constexpr bool enum_ok(CreateKind v)       { return static_cast<unsigned>(v) <= 4u; }
constexpr bool enum_ok(AffordanceAction v) { return static_cast<unsigned>(v) <= 3u; }
constexpr bool enum_ok(ActuatorCallKind v) { return static_cast<unsigned>(v) <= 2u; }

// ─── the generic encoders ───────────────────────────────────────────────────────────────────────────────────────
// bool is stored as one byte 0/1 and REFUSED otherwise on read (a corrupt byte must not become `true`).
template <class T> void Writer::one(const T& x)
{
    if constexpr (std::is_same_v<T, bool>)                     { const std::uint8_t v = x ? 1 : 0; raw(v); }
    else if constexpr (std::is_enum_v<T>)                      { raw(static_cast<std::underlying_type_t<T>>(x)); }
    else if constexpr (std::is_arithmetic_v<T>)                { raw(x); }   // raw bits: NaN payloads, -0.0 survive
    else if constexpr (std::is_same_v<T, std::string>)
    { raw(static_cast<std::uint64_t>(x.size())); b.append(x.data(), x.size()); }
    else if constexpr (EigenFixed<T>)
    { for (Eigen::Index i = 0; i < x.size(); ++i) raw(x.data()[i]); }   // column-major, as stored
    else if constexpr (is_std_array<T>::value)                 { for (const auto& v : x) one(v); }
    else if constexpr (is_vector<T>::value)
    {
        raw(static_cast<std::uint64_t>(x.size()));
        using E = typename T::value_type;
        if constexpr (std::is_arithmetic_v<E> and not std::is_same_v<E, bool>)
            b.append(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(E));   // bulk: point clouds, images
        else
            for (const auto& v : x) one(v);
    }
    else if constexpr (is_optional<T>::value)                  { one(x.has_value()); if (x) one(*x); }
    else if constexpr (is_variant<T>::value)
    {
        const auto idx = static_cast<std::uint8_t>(x.index());
        raw(idx);
        std::visit([this](const auto& v) { one(v); }, x);
    }
    else                                                       { fields(*this, x); }
}

template <class V, std::size_t I = 0> bool emplace_variant(Reader& r, V& v, std::size_t idx)
{
    if constexpr (I < std::variant_size_v<V>)
    {
        if (idx == I) { std::variant_alternative_t<I, V> a{}; r.one(a); v = std::move(a); return r.ok; }
        return emplace_variant<V, I + 1>(r, v, idx);
    }
    else return false;
}

template <class T> void Reader::one(T& x)
{
    if (not ok) return;
    if constexpr (std::is_same_v<T, bool>)
    {
        std::uint8_t v = 0; raw(v);
        if (v > 1) ok = false;
        x = (v == 1);
    }
    else if constexpr (std::is_enum_v<T>)
    {
        std::underlying_type_t<T> v{}; raw(v);
        x = static_cast<T>(v);
        if (not enum_ok(x)) ok = false;
    }
    else if constexpr (std::is_arithmetic_v<T>)                { raw(x); }
    else if constexpr (std::is_same_v<T, std::string>)
    {
        std::uint64_t n = 0; raw(n);
        if (not ok or n > left()) { ok = false; return; }
        x.assign(p, static_cast<std::size_t>(n));
        p += n;
    }
    else if constexpr (EigenFixed<T>)
    { for (Eigen::Index i = 0; i < x.size(); ++i) raw(x.data()[i]); }
    else if constexpr (is_std_array<T>::value)                 { for (auto& v : x) one(v); }
    else if constexpr (is_vector<T>::value)
    {
        std::uint64_t n = 0; raw(n);
        // ★Every element costs ≥ 1 byte on disk, so a count beyond the bytes left is corruption: refuse it BEFORE
        //   allocating (a flipped bit in a length must not become a 2^60-element resize).
        if (not ok or n > left()) { ok = false; return; }
        using E = typename T::value_type;
        if constexpr (std::is_arithmetic_v<E> and not std::is_same_v<E, bool>)
        {
            if (n * sizeof(E) > left()) { ok = false; return; }
            x.resize(static_cast<std::size_t>(n));
            std::memcpy(x.data(), p, static_cast<std::size_t>(n) * sizeof(E));
            p += n * sizeof(E);
        }
        else
        {
            // ★M-5: grow as elements actually DECODE, never resize to the claimed count up front — n ≤ bytes left only
            //   bounds the count, and n × sizeof(E) of a large struct could still be gigabytes from one bad length.
            x.clear();
            x.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(n, left() / sizeof(E) + 1)));
            for (std::uint64_t i = 0; i < n; ++i)
            {
                E v{};
                one(v);
                if (not ok) return;
                x.push_back(std::move(v));
            }
        }
    }
    else if constexpr (is_optional<T>::value)
    {
        bool has = false; one(has);
        if (not ok) return;
        if (has) { typename T::value_type v{}; one(v); x = std::move(v); }
        else     x.reset();
    }
    else if constexpr (is_variant<T>::value)
    {
        std::uint8_t idx = 0; raw(idx);
        if (not ok or idx >= std::variant_size_v<T> or not emplace_variant(*this, x, idx)) ok = false;
    }
    else                                                       { fields(*this, x); }
}

// ─── framing ────────────────────────────────────────────────────────────────────────────────────────────────────
template <class T> bool write_record(std::ostream& os, const T& x)
{
    Writer w;
    w.raw(kRecordTag);
    w.raw(std::uint64_t{0});              // length, patched below
    const std::size_t body = w.b.size();
    w.one(x);
    const std::uint64_t len = w.b.size() - body;
    std::memcpy(w.b.data() + sizeof(kRecordTag), &len, sizeof len);
    os.write(w.b.data(), static_cast<std::streamsize>(w.b.size()));   // ONE write: whole or cut, never interleaved
    os.flush();
    return static_cast<bool>(os);
}

enum class Got { Ok, Eof, Truncated, Corrupt };

// Read exactly n bytes, growing the buffer in 1 MB steps as data ARRIVES (M-5: a corrupt length near kMaxRecord
// must not allocate gigabytes before the stream says it has nothing like that many).
bool read_exact(std::istream& is, std::string& buf, std::uint64_t n)
{
    constexpr std::uint64_t kChunk = 1ull << 20;
    buf.clear();
    while (buf.size() < n)
    {
        const std::size_t want = static_cast<std::size_t>(std::min<std::uint64_t>(kChunk, n - buf.size()));
        const std::size_t at = buf.size();
        buf.resize(at + want);
        if (not is.read(buf.data() + at, static_cast<std::streamsize>(want))) return false;
    }
    return true;
}

template <class T> Got read_record_ex(std::istream& is, T& out)
{
    std::uint32_t tag = 0;
    std::uint64_t len = 0;
    if (not is.read(reinterpret_cast<char*>(&tag), sizeof tag))
        return is.gcount() == 0 ? Got::Eof : Got::Truncated;
    if (not is.read(reinterpret_cast<char*>(&len), sizeof len)) return Got::Truncated;
    if (tag != kRecordTag or len > kMaxRecord) return Got::Corrupt;
    std::string buf;
    if (not read_exact(is, buf, len)) return Got::Truncated;
    Reader r{buf.data(), buf.data() + buf.size()};
    T x{};
    r.one(x);
    if (not r.ok or r.p != r.e) return Got::Corrupt;   // a record must be consumed EXACTLY: anything else is a mismatch
    out = std::move(x);
    return Got::Ok;
}

template <class T> std::optional<T> read_record(std::istream& is)
{
    T x{};
    if (read_record_ex(is, x) != Got::Ok) return std::nullopt;
    return x;
}

const char* magic_of(RecordingKind k) { return k == RecordingKind::Input ? "DCIN" : "DCOU"; }

template <class Rec, class T> std::optional<Rec> read_file(const std::string& path, RecordingKind kind)
{
    std::ifstream f(path, std::ios::binary);
    if (not f.is_open()) return std::nullopt;
    auto h = read_header(f, kind);
    if (not h) return std::nullopt;
    Rec rec;
    rec.header = std::move(*h);
    for (;;)
    {
        const auto at = static_cast<std::uint64_t>(f.tellg());
        T x{};
        const Got g = read_record_ex(f, x);
        if (g == Got::Ok) { rec.cycles.push_back(std::move(x)); continue; }
        // clean EOF, a truncated tail, or a corrupt record: stop, keep what is whole, and SAY which
        rec.stop = g == Got::Eof ? StopReason::CleanEof : g == Got::Truncated ? StopReason::Truncated : StopReason::Corrupt;
        rec.stop_offset = at;
        return rec;
    }
}

}  // namespace

bool write_header(std::ostream& os, const DoorRecordingHeader& h, RecordingKind kind)
{
    os.write(magic_of(kind), 4);
    const std::uint32_t v = kCycleIoVersion;
    os.write(reinterpret_cast<const char*>(&v), sizeof v);
    return write_record(os, h);
}

std::optional<DoorRecordingHeader> read_header(std::istream& is, RecordingKind kind)
{
    char m[4]{};
    std::uint32_t v = 0;
    if (not is.read(m, 4) or not is.read(reinterpret_cast<char*>(&v), sizeof v)) return std::nullopt;
    if (std::memcmp(m, magic_of(kind), 4) != 0 or v != kCycleIoVersion) return std::nullopt;
    return read_record<DoorRecordingHeader>(is);
}

bool write(std::ostream& os, const DoorCycleInput& x)  { return write_record(os, x); }
bool write(std::ostream& os, const DoorCycleOutput& x) { return write_record(os, x); }
std::optional<DoorCycleInput>  read_input(std::istream& is)  { return read_record<DoorCycleInput>(is); }
std::optional<DoorCycleOutput> read_output(std::istream& is) { return read_record<DoorCycleOutput>(is); }

std::optional<InputRecording> read_input_file(const std::string& path)
{ return read_file<InputRecording, DoorCycleInput>(path, RecordingKind::Input); }
std::optional<OutputRecording> read_output_file(const std::string& path)
{ return read_file<OutputRecording, DoorCycleOutput>(path, RecordingKind::Output); }

}  // namespace rc::door::core
