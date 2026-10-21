// door_cycle_io.cpp — binary (de)serialisation of the DoorCore boundary types. See door_cycle_io.h for the layout.
//
// ★ONE FIELD LIST PER STRUCT, USED BY BOTH DIRECTIONS. Each struct has a single `fields(ar, x)` that names its members
// once; the Writer and the Reader are both "archives" applied to it. A writer and a reader maintained separately drift
// — a field added to one and not the other is a silent misread of everything after it — and this file exists so that
// recorded data can be trusted bit for bit.
#include "door_cycle_io.h"

#include <bit>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <istream>
#include <ostream>
#include <type_traits>
#include <zstd.h>

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
    std::uint32_t version = kCycleIoVersion;   // a writer always writes the current version
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
    std::uint32_t version = kCycleIoVersion;   // the FILE's version (read_header)
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
// MaskPacket, BpearlSweep, SemanticPlane, ImageFrame: hand-written below (v2 packs their bulk arrays).
DCIO_FIELDS(ImagePatch, x.x0, x.y0, x.w, x.h, x.bytes)
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

// ═══ v2 PACKING (see the header: lossless transforms + zstd; predictions verified by FNV-64) ═══════════════════════
constexpr int kZstdLevel = 1;                         // tourR1: levels 1..12 within 2 % of each other; 1 is the fastest
constexpr std::uint64_t kMaxUnpacked = 1ull << 31;    // one array; a larger claimed size is corruption

std::uint64_t fnv64(const void* data, std::size_t n)
{
    std::uint64_t h = 1469598103934665603ull;
    const auto* b = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

void put_blob(Writer& w, const std::string& raw)
{
    ZSTD_CCtx* c = ZSTD_createCCtx();
    ZSTD_CCtx_setParameter(c, ZSTD_c_compressionLevel, kZstdLevel);
    ZSTD_CCtx_setParameter(c, ZSTD_c_checksumFlag, 1);            // the frame checks its own content
    std::string out(ZSTD_compressBound(raw.size()), '\0');
    const std::size_t n = ZSTD_compress2(c, out.data(), out.size(), raw.data(), raw.size());
    ZSTD_freeCCtx(c);
    if (ZSTD_isError(n)) { out.clear(); }                         // cannot happen with a bound-sized buffer
    else out.resize(n);
    w.raw(static_cast<std::uint64_t>(out.size()));
    w.b.append(out);
}

/// Exactly `expect` bytes or ok = false (wrong content size, bad frame, checksum mismatch).
bool get_blob(Reader& r, std::string& out, std::uint64_t expect)
{
    std::uint64_t clen = 0;
    r.raw(clen);
    if (not r.ok or clen > r.left() or expect > kMaxUnpacked) return r.ok = false;
    const auto fcs = ZSTD_getFrameContentSize(r.p, static_cast<std::size_t>(clen));
    if (fcs != expect) return r.ok = false;
    out.assign(static_cast<std::size_t>(expect), '\0');
    const std::size_t n = ZSTD_decompress(out.data(), out.size(), r.p, static_cast<std::size_t>(clen));
    if (ZSTD_isError(n) or n != expect) return r.ok = false;
    r.p += clen;
    return true;
}

/// n elements of k 32-bit words: planar per component, delta on the bit pattern (mod 2^32), byte-shuffled.
std::string pack_words(const std::uint32_t* w, std::size_t n, int k)
{
    const std::size_t m = n * static_cast<std::size_t>(k);
    std::vector<std::uint32_t> d(m);
    for (int c = 0; c < k; ++c)
    {
        std::uint32_t prev = 0;
        for (std::size_t i = 0; i < n; ++i) { const std::uint32_t v = w[i * k + c]; d[c * n + i] = v - prev; prev = v; }
    }
    std::string out(m * 4, '\0');
    const auto* b = reinterpret_cast<const unsigned char*>(d.data());
    for (std::size_t j = 0; j < 4; ++j)
        for (std::size_t i = 0; i < m; ++i) out[j * m + i] = static_cast<char>(b[i * 4 + j]);
    return out;
}
void unpack_words(const std::string& in, std::uint32_t* w, std::size_t n, int k)
{
    const std::size_t m = n * static_cast<std::size_t>(k);
    std::vector<std::uint32_t> d(m);
    auto* b = reinterpret_cast<unsigned char*>(d.data());
    for (std::size_t j = 0; j < 4; ++j)
        for (std::size_t i = 0; i < m; ++i) b[i * 4 + j] = static_cast<unsigned char>(in[j * m + i]);
    for (int c = 0; c < k; ++c)
    {
        std::uint32_t prev = 0;
        for (std::size_t i = 0; i < n; ++i) { prev += d[c * n + i]; w[i * k + c] = prev; }
    }
}

template <class V> constexpr int words_of() { return static_cast<int>(sizeof(V) / 4); }

/// A vector of float-only elements (Vec3f, Vec2f, float): u64 n | blob.
template <class V> void put_words(Writer& w, const std::vector<V>& v)
{
    static_assert(sizeof(V) % 4 == 0);
    w.raw(static_cast<std::uint64_t>(v.size()));
    put_blob(w, pack_words(reinterpret_cast<const std::uint32_t*>(v.data()), v.size(), words_of<V>()));
}
template <class V> void get_words(Reader& r, std::vector<V>& v)
{
    std::uint64_t n = 0;
    r.raw(n);
    if (not r.ok or n > kMaxUnpacked / sizeof(V)) { r.ok = false; return; }   // no n·size overflow
    std::string raw;
    if (not get_blob(r, raw, n * sizeof(V))) return;
    v.resize(static_cast<std::size_t>(n));
    unpack_words(raw, reinterpret_cast<std::uint32_t*>(v.data()), v.size(), words_of<V>());
}

Vec3f predict(const Mat4d& T, const Vec3f& c) { return derived_prediction(T, c); }

/// `target` coded against T·src: u8 mode | mode 0: words(target) | mode 1: u64 fnv64(target) + words(target ⊕ pred).
void put_derived(Writer& w, const std::vector<Vec3f>& target, const std::vector<Vec3f>& src, const std::optional<Mat4d>& T)
{
    const bool predictable = T.has_value() and not target.empty() and target.size() == src.size();
    w.raw(static_cast<std::uint8_t>(predictable ? 1 : 0));
    if (not predictable) { put_words(w, target); return; }
    w.raw(fnv64(target.data(), target.size() * sizeof(Vec3f)));
    std::vector<Vec3f> res(target.size());
    for (std::size_t i = 0; i < target.size(); ++i)
    {
        const Vec3f p = predict(*T, src[i]);
        for (int c = 0; c < 3; ++c)
            res[i][c] = std::bit_cast<float>(std::bit_cast<std::uint32_t>(target[i][c]) ^ std::bit_cast<std::uint32_t>(p[c]));
    }
    put_words(w, res);
}
void get_derived(Reader& r, std::vector<Vec3f>& target, const std::vector<Vec3f>& src, const std::optional<Mat4d>& T)
{
    std::uint8_t mode = 0;
    r.raw(mode);
    if (not r.ok) return;
    if (mode == 0) { get_words(r, target); return; }
    if (mode != 1 or not T.has_value()) { r.ok = false; return; }
    std::uint64_t h = 0;
    r.raw(h);
    get_words(r, target);
    if (not r.ok or target.size() != src.size()) { r.ok = false; return; }
    for (std::size_t i = 0; i < target.size(); ++i)
    {
        const Vec3f p = predict(*T, src[i]);
        for (int c = 0; c < 3; ++c)
            target[i][c] = std::bit_cast<float>(std::bit_cast<std::uint32_t>(target[i][c]) ^ std::bit_cast<std::uint32_t>(p[c]));
    }
    if (fnv64(target.data(), target.size() * sizeof(Vec3f)) != h) r.ok = false;   // this build predicts differently
}

/// Mask pixels: u8 mode | mode 1 (every value an exact int16, no -0): u64 n + blob of int16 planar deltas, shuffled;
/// mode 0: words.
void put_pixels(Writer& w, const std::vector<Vec2f>& v)
{
    const bool small_ints = std::ranges::all_of(v, [](const Vec2f& p)
    {
        for (int c = 0; c < 2; ++c)
        {
            const float f = p[c];
            if (not (f >= -32768.f and f <= 32767.f) or static_cast<float>(static_cast<std::int16_t>(f)) != f
                or std::signbit(f) != (f < 0.f))
                return false;
        }
        return true;
    });
    w.raw(static_cast<std::uint8_t>(small_ints ? 1 : 0));
    if (not small_ints) { put_words(w, v); return; }
    const std::size_t n = v.size();
    w.raw(static_cast<std::uint64_t>(n));
    std::vector<std::uint16_t> d(2 * n);
    for (int c = 0; c < 2; ++c)
    {
        std::uint16_t prev = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto u = static_cast<std::uint16_t>(static_cast<std::int16_t>(v[i][c]));
            d[c * n + i] = static_cast<std::uint16_t>(u - prev);
            prev = u;
        }
    }
    std::string raw(4 * n, '\0');
    const auto* b = reinterpret_cast<const unsigned char*>(d.data());
    for (std::size_t j = 0; j < 2; ++j)
        for (std::size_t i = 0; i < 2 * n; ++i) raw[j * 2 * n + i] = static_cast<char>(b[i * 2 + j]);
    put_blob(w, raw);
}
void get_pixels(Reader& r, std::vector<Vec2f>& v)
{
    std::uint8_t mode = 0;
    r.raw(mode);
    if (not r.ok) return;
    if (mode == 0) { get_words(r, v); return; }
    if (mode != 1) { r.ok = false; return; }
    std::uint64_t n = 0;
    r.raw(n);
    if (not r.ok or n > kMaxUnpacked / 4) { r.ok = false; return; }           // no 4·n overflow
    std::string raw;
    if (not get_blob(r, raw, 4 * n)) return;
    std::vector<std::uint16_t> d(2 * n);
    auto* b = reinterpret_cast<unsigned char*>(d.data());
    for (std::size_t j = 0; j < 2; ++j)
        for (std::size_t i = 0; i < 2 * n; ++i) b[i * 2 + j] = static_cast<unsigned char>(raw[j * 2 * n + i]);
    v.resize(static_cast<std::size_t>(n));
    for (int c = 0; c < 2; ++c)
    {
        std::uint16_t prev = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            prev = static_cast<std::uint16_t>(prev + d[c * n + i]);
            v[i][c] = static_cast<float>(static_cast<std::int16_t>(prev));
        }
    }
}

/// Patch bytes by format: DepthF32 as 32-bit words, anything else as plain bytes. u64 n | blob.
void put_patch_bytes(Writer& w, const std::vector<std::uint8_t>& v, PixelFormat f)
{
    w.raw(static_cast<std::uint64_t>(v.size()));
    if (f == PixelFormat::DepthF32 and v.size() % 4 == 0)
        put_blob(w, pack_words(reinterpret_cast<const std::uint32_t*>(v.data()), v.size() / 4, 1));
    else
        put_blob(w, std::string(reinterpret_cast<const char*>(v.data()), v.size()));
}
void get_patch_bytes(Reader& r, std::vector<std::uint8_t>& v, PixelFormat f)
{
    std::uint64_t n = 0;
    r.raw(n);
    if (not r.ok or n > kMaxUnpacked) { r.ok = false; return; }
    std::string raw;
    if (not get_blob(r, raw, n)) return;
    v.resize(static_cast<std::size_t>(n));
    if (f == PixelFormat::DepthF32 and n % 4 == 0)
    {
        std::vector<std::uint32_t> w(n / 4);
        unpack_words(raw, w.data(), w.size(), 1);
        std::memcpy(v.data(), w.data(), n);
    }
    else std::memcpy(v.data(), raw.data(), n);
}

// ─── the four packed structs: v1 = the original field order, unpacked; v2 = prediction inputs first, packed ──────
void fields(Writer& w, const MaskPacket& x)
{
    if (w.version < 2)
    { w(x.valid, x.frame_id, x.timestamp_ms, x.slices, x.support_points, x.support_points_cam, x.mask_pixels, x.cam_twist,
        x.frame_dt_s, x.room_T_zed_capture, x.room_T_robot_capture, x.room_zed_chain, x.ms_since_last_frame); return; }
    w(x.valid, x.frame_id, x.timestamp_ms, x.slices, x.cam_twist, x.frame_dt_s, x.room_T_zed_capture,
      x.room_T_robot_capture, x.room_zed_chain, x.ms_since_last_frame);
    put_words(w, x.support_points_cam);
    put_derived(w, x.support_points, x.support_points_cam, x.room_T_zed_capture);
    put_pixels(w, x.mask_pixels);
}
void fields(Reader& r, MaskPacket& x)
{
    if (r.version < 2)
    { r(x.valid, x.frame_id, x.timestamp_ms, x.slices, x.support_points, x.support_points_cam, x.mask_pixels, x.cam_twist,
        x.frame_dt_s, x.room_T_zed_capture, x.room_T_robot_capture, x.room_zed_chain, x.ms_since_last_frame); return; }
    r(x.valid, x.frame_id, x.timestamp_ms, x.slices, x.cam_twist, x.frame_dt_s, x.room_T_zed_capture,
      x.room_T_robot_capture, x.room_zed_chain, x.ms_since_last_frame);
    if (r.ok) get_words(r, x.support_points_cam);
    if (r.ok) get_derived(r, x.support_points, x.support_points_cam, x.room_T_zed_capture);
    if (r.ok) get_pixels(r, x.mask_pixels);
}
void fields(Writer& w, const BpearlSweep& x)
{
    if (w.version < 2)
    { w(x.room_fresh, x.pts_room, x.origin_room, x.robot_fresh, x.pts_robot, x.origin_robot, x.robot_stamp_ms,
        x.room_T_robot_capture); return; }
    w(x.room_fresh, x.origin_room, x.robot_fresh, x.origin_robot, x.robot_stamp_ms, x.room_T_robot_capture);
    put_words(w, x.pts_robot);
    put_derived(w, x.pts_room, x.pts_robot, x.room_T_robot_capture);
}
void fields(Reader& r, BpearlSweep& x)
{
    if (r.version < 2)
    { r(x.room_fresh, x.pts_room, x.origin_room, x.robot_fresh, x.pts_robot, x.origin_robot, x.robot_stamp_ms,
        x.room_T_robot_capture); return; }
    r(x.room_fresh, x.origin_room, x.robot_fresh, x.origin_robot, x.robot_stamp_ms, x.room_T_robot_capture);
    if (r.ok) get_words(r, x.pts_robot);
    if (r.ok) get_derived(r, x.pts_room, x.pts_robot, x.room_T_robot_capture);
}
void fields(Writer& w, const SemanticPlane& x)
{
    w(x.pw, x.ph, x.frame_w, x.frame_h, x.stamp_ms);
    if (w.version < 2) w(x.plane); else put_words(w, x.plane);
}
void fields(Reader& r, SemanticPlane& x)
{
    r(x.pw, x.ph, x.frame_w, x.frame_h, x.stamp_ms);
    if (not r.ok) return;
    if (r.version < 2) r(x.plane); else get_words(r, x.plane);
}
void fields(Writer& w, const ImageFrame& x)
{
    w(x.format, x.width, x.height, x.stamp_ms, x.frame_ref);
    if (w.version < 2) { w(x.patches); return; }
    w.raw(static_cast<std::uint64_t>(x.patches.size()));
    for (const auto& p : x.patches) { w(p.x0, p.y0, p.w, p.h); put_patch_bytes(w, p.bytes, x.format); }
}
void fields(Reader& r, ImageFrame& x)
{
    r(x.format, x.width, x.height, x.stamp_ms, x.frame_ref);
    if (not r.ok) return;
    if (r.version < 2) { r(x.patches); return; }
    std::uint64_t n = 0;
    r.raw(n);
    if (not r.ok or n > r.left()) { r.ok = false; return; }
    x.patches.clear();
    for (std::uint64_t i = 0; i < n and r.ok; ++i)
    {
        ImagePatch p;
        r(p.x0, p.y0, p.w, p.h);
        if (r.ok) get_patch_bytes(r, p.bytes, x.format);
        if (r.ok) x.patches.push_back(std::move(p));
    }
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

template <class T> Got read_record_ex(std::istream& is, T& out, std::uint32_t version = kCycleIoVersion)
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
    r.version = version;
    T x{};
    r.one(x);
    if (not r.ok or r.p != r.e) return Got::Corrupt;   // a record must be consumed EXACTLY: anything else is a mismatch
    out = std::move(x);
    return Got::Ok;
}

template <class T> std::optional<T> read_record(std::istream& is, std::uint32_t version = kCycleIoVersion)
{
    T x{};
    if (read_record_ex(is, x, version) != Got::Ok) return std::nullopt;
    return x;
}

const char* magic_of(RecordingKind k) { return k == RecordingKind::Input ? "DCIN" : "DCOU"; }

template <class Rec, class T> std::optional<Rec> read_file(const std::string& path, RecordingKind kind)
{
    std::ifstream f(path, std::ios::binary);
    if (not f.is_open()) return std::nullopt;
    std::uint32_t version = 0;
    auto h = read_header(f, kind, &version);
    if (not h) return std::nullopt;
    Rec rec;
    rec.header = std::move(*h);
    for (;;)
    {
        const auto at = static_cast<std::uint64_t>(f.tellg());
        T x{};
        const Got g = read_record_ex(f, x, version);
        if (g == Got::Ok) { rec.cycles.push_back(std::move(x)); continue; }
        // clean EOF, a truncated tail, or a corrupt record: stop, keep what is whole, and SAY which
        rec.stop = g == Got::Eof ? StopReason::CleanEof : g == Got::Truncated ? StopReason::Truncated : StopReason::Corrupt;
        rec.stop_offset = at;
        return rec;
    }
}

}  // namespace

// ═══ THE v2 PREDICTION — BUILD-INVARIANT BY CONSTRUCTION (review fix round 3, I-1) ═════════════════════════════════
// Only the residual against this function is stored, so encoder and decoder MUST compute the same bits whatever the
// build: an Eigen product here would not (its vectorised path uses FMA pmadd under __FMA__, its scalar path sums in
// another order, and -ffp-contract=fast / -ffast-math rewrite either). So: plain scalar IEEE double, a FIXED order
// ((m0·x + m1·y) + m2·z) + m3, each product forced through a volatile double so it is rounded before the add — a
// contraction into an FMA is then impossible whatever -ffp-contract / -mfma say (a volatile load cannot be fused).
// The pragma below turns contraction off for this function as well (belt and braces; GCC honours fp-contract in
// optimize attributes — tests/test_door_cycle_io.cpp checks golden bits under -O3 -march=native and
// -O2 -mfma -ffp-contract=fast). -ffast-math (reassociation) and x87 excess precision are refused at compile time.
// The order matches the producers' (MaskIngestor / LidarPlaneReader, Eigen, SSE2, no FMA) on 100 % of tourR1's mask
// coordinates, so the mask residual stays all-zero; a different order would cost only compression, never fidelity.
#if defined(__FAST_MATH__)
#error "door_cycle_io.cpp: -ffast-math makes the DCIN v2 prediction build-dependent; build this TU without it"
#endif
#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
#error "door_cycle_io.cpp: needs FLT_EVAL_METHOD == 0 (SSE2 doubles, no x87 excess precision) for the v2 prediction"
#endif
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
Vec3f derived_prediction(const Mat4d& T, const Vec3f& c)
{
    const double x = c.x(), y = c.y(), z = c.z();
    float out[3];
    for (int i = 0; i < 3; ++i)
    {
        volatile double p0 = T(i, 0) * x;
        volatile double p1 = T(i, 1) * y;
        volatile double p2 = T(i, 2) * z;
        volatile double s = p0 + p1;
        s = s + p2;
        s = s + T(i, 3);
        out[i] = static_cast<float>(s);
    }
    return {out[0], out[1], out[2]};
}
#pragma GCC pop_options

bool write_header(std::ostream& os, const DoorRecordingHeader& h, RecordingKind kind)
{
    os.write(magic_of(kind), 4);
    const std::uint32_t v = kCycleIoVersion;
    os.write(reinterpret_cast<const char*>(&v), sizeof v);
    return write_record(os, h);
}

std::optional<DoorRecordingHeader> read_header(std::istream& is, RecordingKind kind, std::uint32_t* version)
{
    char m[4]{};
    std::uint32_t v = 0;
    if (not is.read(m, 4) or not is.read(reinterpret_cast<char*>(&v), sizeof v)) return std::nullopt;
    if (std::memcmp(m, magic_of(kind), 4) != 0 or v < kCycleIoOldestReadable or v > kCycleIoVersion) return std::nullopt;
    if (version != nullptr) *version = v;
    return read_record<DoorRecordingHeader>(is, v);
}

bool write(std::ostream& os, const DoorCycleInput& x)  { return write_record(os, x); }
bool write(std::ostream& os, const DoorCycleOutput& x) { return write_record(os, x); }
std::optional<DoorCycleInput>  read_input(std::istream& is)  { return read_record<DoorCycleInput>(is); }
std::optional<DoorCycleInput>  read_input(std::istream& is, std::uint32_t version)
{ return read_record<DoorCycleInput>(is, version); }
std::optional<DoorCycleOutput> read_output(std::istream& is) { return read_record<DoorCycleOutput>(is); }

std::optional<InputRecording> read_input_file(const std::string& path)
{ return read_file<InputRecording, DoorCycleInput>(path, RecordingKind::Input); }
std::optional<OutputRecording> read_output_file(const std::string& path)
{ return read_file<OutputRecording, DoorCycleOutput>(path, RecordingKind::Output); }

}  // namespace rc::door::core
