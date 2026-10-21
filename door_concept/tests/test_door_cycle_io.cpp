// Offline harness for src/core/door_cycle_io.{h,cpp}: the DoorCore boundary types (DoorCycleInput / DoorCycleOutput)
// and their versioned binary recordings ("DCIN" v1 / "DCOU" v1). Spec: docs/superpowers/specs/2026-10-21-door-core-
// replay-design.md §3.1/§6. Checks:
//   1. ROUND TRIP: 200 random-filled inputs and outputs written and read back under the es_ES locale compare equal
//      (every field, every optional/variant branch exercised by the seeded generator).
//   2. BIT-EXACT: NaN payloads, -0.0, denormals and infinities in doubles AND floats come back with the same bits.
//   3. PARTIAL FILE: a recording truncated at EVERY byte offset of its last two records reads back exactly the records
//      that were complete — never a misread, never a throw.
//   4. REFUSALS: wrong magic, wrong version, and an output file handed to the input reader are all refused.
//   5. DSR-FREE CORE: every file under src/core/ (and every local header it pulls in, transitively) includes nothing
//      from DSR / Qt / Ice / the media plane / the DSR-bound ingestors.
// ★WHY es_ES: these agents run under LANG=es_ES.UTF-8 (CLAUDE.md "Parsing numbers"); a binary format must be immune,
// and a test that runs in "C" cannot show that it is.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_cycle_io.cpp src/core/door_cycle_io.cpp -lzstd -o /tmp/test_door_cycle_io && /tmp/test_door_cycle_io
#include <bit>
#include <clocale>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <print>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include "../src/core/door_cycle_io.h"

using namespace rc::door::core;
namespace fs = std::filesystem;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}  {}\n", __FILE__, __LINE__, #c); } } while (0)

// ─── a seeded random filler ──────────────────────────────────────────────────────────────────────────────────────
struct Rng
{
    std::mt19937_64 g;
    explicit Rng(std::uint64_t s) : g(s) {}
    bool   coin()            { return (g() & 1u) != 0; }
    int    upto(int n)       { return static_cast<int>(g() % static_cast<std::uint64_t>(n + 1)); }
    double d()               { return std::uniform_real_distribution<double>(-1e3, 1e3)(g); }
    float  f()               { return static_cast<float>(d()); }
    std::uint64_t u64()      { return g(); }
    std::int64_t  i64()      { return static_cast<std::int64_t>(g()); }
    int    i32()             { return static_cast<int>(g()); }
    std::string str()
    {
        // commas, newlines, NUL and UTF-8 on purpose: a recording is not a CSV
        static const std::string pool = std::string("door_,;\n\t é∑") + '\0' + "xyz0123.5";
        std::string s;
        for (int i = 0, n = upto(12); i < n; ++i) s.push_back(pool[g() % pool.size()]);
        return s;
    }
    Vec2f v2() { return {f(), f()}; }
    Vec3f v3() { return {f(), f(), f()}; }
    Mat4d m4() { Mat4d m; for (int i = 0; i < 16; ++i) m.data()[i] = d(); return m; }
    Mat6d m6() { Mat6d m; for (int i = 0; i < 36; ++i) m.data()[i] = d(); return m; }
    template <class T, class F> std::optional<T> opt(F&& make) { return coin() ? std::optional<T>(make()) : std::nullopt; }
    template <class F> auto vec(int n, F&& make) { std::vector<decltype(make())> v; for (int i = 0, k = upto(n); i < k; ++i) v.push_back(make()); return v; }
};

static MaskSlice rnd_slice(Rng& r)
{
    MaskSlice s;
    s.label = r.str(); s.class_id = r.f(); s.confidence = r.f();
    s.centroid = r.v3(); s.bbox_min = r.v3(); s.bbox_max = r.v3();
    s.support_begin = r.u64(); s.support_end = r.u64(); s.pixel_begin = r.u64(); s.pixel_end = r.u64();
    s.motion_var = r.f(); s.motion_bias = r.f(); s.motion_dotd = r.f(); s.trunc_frac = r.f();
    s.centroid_radius = r.f(); s.range = r.f(); s.has_depth = r.coin(); s.azimuth_room_rad = r.f();
    s.source = static_cast<std::uint8_t>(r.upto(3)); s.depth_var = r.f();
    s.color_chroma = r.v3(); s.color_var = r.v3(); s.color_neff = r.f();
    return s;
}

static ImageFrame rnd_image(Rng& r)
{
    ImageFrame im;
    im.format = static_cast<PixelFormat>(1 + r.upto(2));
    im.width = r.i32(); im.height = r.i32(); im.stamp_ms = r.u64(); im.frame_ref = r.f();
    im.patches = r.vec(3, [&] {
        ImagePatch p; p.x0 = r.i32(); p.y0 = r.i32(); p.w = r.i32(); p.h = r.i32();
        p.bytes = r.vec(300, [&] { return static_cast<std::uint8_t>(r.u64()); });
        return p; });
    return im;
}

static DoorEvent rnd_event(Rng& r)
{
    DoorEvent e;
    e.t_steady_ns = r.i64();
    switch (r.upto(4))
    {
        case 0: e.payload = NodeDeleted{r.u64(), r.coin()}; break;
        case 1:
        {
            DoorNodeUpserted u;
            u.node.id = r.u64(); u.node.name = r.str(); u.node.type = r.str();
            u.node.parent = r.opt<std::uint64_t>([&] { return r.u64(); });
            u.node.width_m = r.opt<float>([&] { return r.f(); });  u.node.depth_m = r.opt<float>([&] { return r.f(); });
            u.node.height_m = r.opt<float>([&] { return r.f(); });
            u.node.room_T_node = r.opt<Mat4d>([&] { return r.m4(); });
            u.node.is_anchor = r.coin(); u.node.epistemic_pending = r.opt<bool>([&] { return r.coin(); });
            u.node.door_open_prob = r.opt<float>([&] { return r.f(); }); u.node.door_phi_rad = r.opt<float>([&] { return r.f(); });
            u.rt_parent = r.opt<RtEdge>([&] { return RtEdge{r.u64(), r.v3(), r.v3()}; });
            e.payload = std::move(u); break;
        }
        case 2: e.payload = UiActuationRequest{r.coin(), r.str()}; break;
        case 3: e.payload = NodeIdAssigned{r.u64(), r.u64(), r.str()}; break;
        default: e.payload = OwnedNodesSwept{r.str(), r.vec(5, [&] { return r.u64(); })}; break;
    }
    return e;
}

static ActuatorPending rnd_pending(Rng& r) { return ActuatorPending{r.i32(), r.str(), r.coin(), r.str(), r.coin()}; }

static ActuatorCall rnd_call(Rng& r)
{
    ActuatorCall c;
    c.kind = static_cast<ActuatorCallKind>(r.upto(2));
    c.door_name = r.str(); c.xy_m = r.v2(); c.yaw = r.f(); c.width_m = r.f(); c.open = r.coin();
    c.world_T_room = r.m4(); c.purpose = r.str(); c.provider_id = r.str();
    c.poll_in = r.opt<ActuatorPending>([&] { return rnd_pending(r); });
    c.caps = r.opt<ActuatorCaps>([&] {
        ActuatorCaps k; k.available = r.coin(); k.continuous_angle = r.coin(); k.requires_human = r.coin();
        k.typical_latency_s = r.f(); k.n_doors = r.i32(); k.note = r.str();
        k.doors = r.vec(4, [&] { return ProviderDoor{r.str(), r.v2(), r.f()}; });
        return k; });
    c.pending = r.opt<ActuatorPending>([&] { return rnd_pending(r); });
    return c;
}

static DoorCycleInput rnd_input(Rng& r)
{
    DoorCycleInput in;
    in.cycle = r.u64(); in.t_steady_ns = r.i64(); in.t_wall_ms = r.i64();
    in.rooms.graph_current_room = r.opt<std::uint64_t>([&] { return r.u64(); });
    in.rooms.n_current_edges = r.i32();
    in.rooms.current_edge_room = r.opt<std::uint64_t>([&] { return r.u64(); });
    in.rooms.recall_names = r.vec(3, [&] { return r.str(); });
    in.rooms.rooms = r.vec(4, [&] {
        RoomInfo ri; ri.id = r.u64(); ri.name = r.str(); ri.is_proto = r.coin();
        ri.root_T_room = r.opt<Mat4d>([&] { return r.m4(); }); ri.polygon = r.vec(8, [&] { return r.v2(); });
        ri.level = r.opt<int>([&] { return r.i32(); });
        return ri; });
    in.walls = r.vec(6, [&] {
        WallInfo w; w.id = r.u64(); w.parent = r.opt<std::uint64_t>([&] { return r.u64(); });
        w.parent_parent = r.opt<std::uint64_t>([&] { return r.u64(); });
        w.rt = r.opt<RtEdge>([&] { return RtEdge{r.u64(), r.v3(), r.v3()}; });
        w.rt_from_room = r.opt<RtEdge>([&] { return RtEdge{r.u64(), r.v3(), r.v3()}; });
        w.width_m = r.opt<float>([&] { return r.f(); });
        return w; });
    in.objects = r.vec(6, [&] {
        ObjectInfo o; o.id = r.u64(); o.name = r.str(); o.type = r.str();
        o.parent = r.opt<std::uint64_t>([&] { return r.u64(); });
        o.width_m = r.opt<float>([&] { return r.f(); }); o.depth_m = r.opt<float>([&] { return r.f(); });
        o.height_m = r.opt<float>([&] { return r.f(); });
        o.room_T_node = r.opt<Mat4d>([&] { return r.m4(); });
        o.is_anchor = r.coin(); o.epistemic_pending = r.opt<bool>([&] { return r.coin(); });
        o.door_open_prob = r.opt<float>([&] { return r.f(); }); o.door_phi_rad = r.opt<float>([&] { return r.f(); });
        return o; });
    auto& rb = in.robot;
    rb.present = r.coin(); rb.name = r.str(); rb.id = r.u64();
    rb.room_T_robot = r.opt<Mat4d>([&] { return r.m4(); }); rb.room_T_body = r.opt<Mat4d>([&] { return r.m4(); });
    rb.room_T_zed = r.opt<Mat4d>([&] { return r.m4(); });   rb.robot_T_zed = r.opt<Mat4d>([&] { return r.m4(); });
    rb.body_T_zed = r.opt<Mat4d>([&] { return r.m4(); });   rb.root_T_body = r.opt<Mat4d>([&] { return r.m4(); });
    rb.rt_cov = r.opt<Mat6d>([&] { return r.m6(); }); rb.rt_cov_edge_robot_to_room = r.coin();
    rb.twist = r.opt<Twist>([&] { return Twist{r.v3(), r.v3(), r.u64()}; });
    rb.sim_stamps = r.opt<SimStamps>([&] { return SimStamps{r.i64(), r.i64()}; });
    rb.speed_variance = r.vec(4, [&] { return r.f(); });
    rb.gt = r.opt<Vec3f>([&] { return r.v3(); });
    auto& c = in.camera;
    c.present = r.coin(); c.width = static_cast<std::uint32_t>(r.u64()); c.height = static_cast<std::uint32_t>(r.u64());
    c.focal_x = r.f(); c.focal_y = r.f(); c.projection_model = static_cast<std::uint8_t>(r.upto(2));
    c.cam_rgb_width = r.opt<int>([&] { return r.i32(); }); c.cam_rgb_focalx = r.opt<int>([&] { return r.i32(); });
    c.cam_fov = r.opt<float>([&] { return r.f(); });
    in.masks = r.opt<MaskPacket>([&] {
        MaskPacket p; p.valid = r.coin(); p.frame_id = r.i32(); p.timestamp_ms = r.u64();
        p.slices = r.vec(4, [&] { return rnd_slice(r); });
        p.support_points = r.vec(40, [&] { return r.v3(); }); p.support_points_cam = r.vec(40, [&] { return r.v3(); });
        p.mask_pixels = r.vec(60, [&] { return r.v2(); });
        for (auto& x : p.cam_twist) x = r.f();
        p.frame_dt_s = r.f(); p.room_T_zed_capture = r.opt<Mat4d>([&] { return r.m4(); });
        p.room_T_robot_capture = r.opt<Mat4d>([&] { return r.m4(); });
        p.room_zed_chain = r.vec(3, [&] { return RtChainStep{r.str(), r.str(), r.m4(), r.opt<Mat6d>([&] { return r.m6(); }), r.coin()}; });
        p.ms_since_last_frame = r.i64();
        return p; });
    in.mask_stream_ms_since_last_frame = r.i64();
    in.bpearl = r.opt<BpearlSweep>([&] {
        BpearlSweep b; b.room_fresh = r.coin(); b.pts_room = r.vec(50, [&] { return r.v3(); }); b.origin_room = r.v3();
        b.robot_fresh = r.coin(); b.pts_robot = r.vec(50, [&] { return r.v3(); }); b.origin_robot = r.v3();
        b.robot_stamp_ms = r.i64(); b.room_T_robot_capture = r.opt<Mat4d>([&] { return r.m4(); });
        return b; });
    in.semantic_state = static_cast<SemanticState>(r.upto(2));
    in.semantic = r.opt<SemanticPlane>([&] {
        SemanticPlane s; s.pw = r.i32(); s.ph = r.i32(); s.frame_w = r.i32(); s.frame_h = r.i32();
        s.stamp_ms = r.opt<std::uint64_t>([&] { return r.u64(); }); s.plane = r.vec(80, [&] { return r.f(); });
        return s; });
    in.rgb_gray = r.opt<ImageFrame>([&] { return rnd_image(r); });
    in.depth    = r.opt<ImageFrame>([&] { return rnd_image(r); });
    in.affordances = r.vec(5, [&] {
        AffordanceFlags a; a.node_id = r.u64(); a.exists = r.coin(); a.name = r.str();
        a.active = r.opt<bool>([&] { return r.coin(); }); a.epistemic_pending = r.opt<bool>([&] { return r.coin(); });
        a.target_epoch = r.opt<int>([&] { return r.i32(); });
        a.target_x = r.opt<float>([&] { return r.f(); }); a.target_y = r.opt<float>([&] { return r.f(); });
        a.target_yaw = r.opt<float>([&] { return r.f(); });
        return a; });
    in.executing = r.vec(4, [&] {
        ExecutingEdge e; e.from = r.u64(); e.to = r.u64();
        e.epoch = r.opt<int>([&] { return r.i32(); }); e.x = r.opt<float>([&] { return r.f(); });
        e.y = r.opt<float>([&] { return r.f(); }); e.yaw = r.opt<float>([&] { return r.f(); });
        return e; });
    in.identity_load = r.opt<IdentityLoad>([&] {
        return IdentityLoad{r.coin(), r.str(), r.opt<std::uint64_t>([&] { return r.u64(); }), r.str(), r.coin(), r.str()}; });
    in.events = r.vec(6, [&] { return rnd_event(r); });
    in.actuator_calls = r.vec(3, [&] { return rnd_call(r); });
    return in;
}

static DoorCycleOutput rnd_output(Rng& r)
{
    DoorCycleOutput o;
    o.cycle = r.u64(); o.t_ms = r.i64(); o.room_id = r.u64();
    o.decisions = r.vec(8, [&] {
        rc::door::DoorDecision d; d.cycle = r.u64(); d.t_ms = r.i64(); d.kind = static_cast<std::uint16_t>(r.u64());
        d.door_id = r.u64(); d.name = r.str(); d.a = r.d(); d.b = r.d(); d.c = r.d(); d.d = r.d(); d.note = r.str();
        return d; });
    o.doors = r.vec(4, [&] {
        DoorStateOut d; d.node_id = r.u64(); d.name = r.str(); d.parent = r.u64(); d.pose = r.m4();
        d.width_m = r.f(); d.depth_m = r.f(); d.height_m = r.f(); d.open_prob = r.f(); d.phi = r.f(); d.phi_sigma = r.f();
        d.exist_logodds = r.f(); d.p_exists = r.f(); d.cov = r.m6(); d.flags = static_cast<std::uint32_t>(r.u64());
        return d; });
    o.creates = r.vec(3, [&] {
        NodeCreateOut c; c.seq = r.u64(); c.kind = static_cast<CreateKind>(r.upto(4)); c.parent = r.u64();
        c.name = r.str(); c.preferred_name = r.str(); c.pose = r.m4(); c.width_m = r.f(); c.depth_m = r.f();
        c.height_m = r.f(); c.open_prob = r.opt<float>([&] { return r.f(); }); c.phi = r.opt<float>([&] { return r.f(); });
        c.source_id = r.u64(); c.note = r.str();
        return c; });
    o.deletes = r.vec(3, [&] { return NodeDeleteOut{r.u64(), r.str()}; });
    o.affordances = r.vec(4, [&] {
        AffordanceOut a; a.create_seq = r.u64(); a.node_id = r.u64(); a.owner_id = r.u64(); a.kind = r.str(); a.action = static_cast<AffordanceAction>(r.upto(3));
        a.epoch = r.i32(); a.p = r.f(); a.value = r.f(); a.x = r.f(); a.y = r.f(); a.yaw = r.f(); a.why = r.str();
        return a; });
    o.identity_table_csv = r.opt<std::string>([&] { return r.str(); });
    o.anchor = r.opt<AnchorOut>([&] {
        AnchorOut a; a.door_id = r.u64(); a.x = r.v3(); a.P = Mat3f::Random(); a.drivable = r.coin();
        a.unseen_ms = r.i32(); a.epoch = r.i32(); a.clear_width = r.f(); a.note = r.str();
        return a; });
    o.diag = r.vec(4, [&] { return DiagRow{r.str(), r.str()}; });
    return o;
}

// ─── helpers ─────────────────────────────────────────────────────────────────────────────────────────────────────
static std::string bytes_of(const auto& write_fn) { std::ostringstream s(std::ios::binary); write_fn(s); return s.str(); }

static void write_file(const fs::path& p, const std::string& b)
{ std::ofstream f(p, std::ios::binary | std::ios::trunc); f.write(b.data(), static_cast<std::streamsize>(b.size())); }

// Every local header a file pulls in, transitively ("..." includes only — angle includes are leaves we inspect).
static void collect_includes(const fs::path& file, std::set<fs::path>& seen, std::vector<std::string>& angle)
{
    const auto canon = fs::weakly_canonical(file);
    if (not seen.insert(canon).second or not fs::exists(canon)) return;
    std::ifstream f(canon);
    static const std::regex inc(R"(^\s*#\s*include\s*([<"])([^>"]+)[>"])");
    for (std::string line; std::getline(f, line);)
        if (std::smatch m; std::regex_search(line, m, inc))
        {
            if (m[1] == "<") angle.push_back(m[2].str() + "   (from " + canon.filename().string() + ")");
            else             collect_includes(canon.parent_path() / m[2].str(), seen, angle);
        }
}

int main()
{
    // ★The agents' locale (CLAUDE.md): a harness left in "C" would answer a different question.
    // ★NO std::locale::global(classic()) here: it calls setlocale(LC_ALL, "C") and the test would silently run in "C".
    if (std::setlocale(LC_ALL, "es_ES.UTF-8") == nullptr) std::setlocale(LC_ALL, "");
    const char* loc = std::setlocale(LC_ALL, nullptr);
    const std::string lc_numeric = std::setlocale(LC_NUMERIC, nullptr);
    std::print("locale: {} (LC_NUMERIC {}, decimal point '{}')\n", loc ? loc : "(none)", lc_numeric,
               std::localeconv()->decimal_point);
    CHECK(lc_numeric.starts_with("es_ES"));                       // really running under the agents' locale
    CHECK(std::string(std::localeconv()->decimal_point) == ",");

    const fs::path dir = "/tmp/test_door_cycle_io_dir";
    fs::remove_all(dir);
    fs::create_directories(dir);

    // ── 1. round trip, random-filled ─────────────────────────────────────────────────────────────────────────────
    {
        Rng r(20261021);
        std::vector<DoorCycleInput> ins;
        std::vector<DoorCycleOutput> outs;
        for (int i = 0; i < 200; ++i) { ins.push_back(rnd_input(r)); outs.push_back(rnd_output(r)); }
        const DoorRecordingHeader h{.agent_id = 21, .agent_name = "door_concept", .config_text = "[A]\nx = 0.5\n",
                                    .start_steady_ns = 123456789, .start_wall_ms = 987654321, .note = "test"};
        {
            std::ofstream f(dir / "in.dcin", std::ios::binary);
            CHECK(write_header(f, h, RecordingKind::Input));
            for (const auto& x : ins) CHECK(write(f, x));
        }
        {
            std::ofstream f(dir / "out.dcou", std::ios::binary);
            CHECK(write_header(f, h, RecordingKind::Output));
            for (const auto& x : outs) CHECK(write(f, x));
        }
        const auto ri = read_input_file((dir / "in.dcin").string());
        CHECK(ri.has_value());
        if (ri)
        {
            CHECK(ri->header == h);
            CHECK(ri->cycles.size() == ins.size());
            CHECK(ri->stop == StopReason::CleanEof and ri->stop_offset == fs::file_size(dir / "in.dcin"));
            for (std::size_t i = 0; i < std::min(ins.size(), ri->cycles.size()); ++i)
                if (not (ri->cycles[i] == ins[i])) { CHECK(ri->cycles[i] == ins[i]); std::print("  input #{}\n", i); break; }
        }
        const auto ro = read_output_file((dir / "out.dcou").string());
        CHECK(ro.has_value());
        if (ro)
        {
            CHECK(ro->header == h);
            CHECK(ro->cycles.size() == outs.size());
            for (std::size_t i = 0; i < std::min(outs.size(), ro->cycles.size()); ++i)
                if (not (ro->cycles[i] == outs[i])) { CHECK(ro->cycles[i] == outs[i]); std::print("  output #{}\n", i); break; }
        }
        // the magic is the first four bytes, the version the next u32
        std::ifstream f(dir / "in.dcin", std::ios::binary);
        char m[4]{}; std::uint32_t v = 0;
        f.read(m, 4); f.read(reinterpret_cast<char*>(&v), 4);
        CHECK(std::memcmp(m, "DCIN", 4) == 0 and v == kCycleIoVersion);
        std::ifstream g(dir / "out.dcou", std::ios::binary);
        g.read(m, 4); g.read(reinterpret_cast<char*>(&v), 4);
        CHECK(std::memcmp(m, "DCOU", 4) == 0 and v == kCycleIoVersion);
        std::print("round trip: {} inputs ({} B), {} outputs ({} B)\n", ins.size(), fs::file_size(dir / "in.dcin"),
                   outs.size(), fs::file_size(dir / "out.dcou"));
    }

    // ── 2. bit-exact non-finite / signed-zero / denormal values ────────────────────────────────────────────────
    {
        const double qnan = std::bit_cast<double>(std::uint64_t{0x7ff8'0000'dead'beefULL});
        const float  fnan = std::bit_cast<float>(std::uint32_t{0x7fc0'1234u});
        DoorCycleOutput o;
        rc::door::DoorDecision d;
        d.a = qnan; d.b = -0.0; d.c = std::numeric_limits<double>::denorm_min(); d.d = -std::numeric_limits<double>::infinity();
        o.decisions.push_back(d);
        DoorCycleInput in;
        in.robot.room_T_robot = Mat4d::Identity();
        (*in.robot.room_T_robot)(0, 3) = qnan;
        in.robot.speed_variance = {fnan, -0.0f, std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::infinity()};
        std::stringstream so(std::ios::in | std::ios::out | std::ios::binary), si(std::ios::in | std::ios::out | std::ios::binary);
        CHECK(write(so, o)); CHECK(write(si, in));
        const auto o2 = read_output(so); const auto i2 = read_input(si);
        CHECK(o2.has_value() and i2.has_value());
        if (o2 and i2)
        {
            const auto& e = o2->decisions.at(0);
            CHECK(std::bit_cast<std::uint64_t>(e.a) == std::bit_cast<std::uint64_t>(qnan));
            CHECK(std::bit_cast<std::uint64_t>(e.b) == std::bit_cast<std::uint64_t>(-0.0));
            CHECK(e.c == std::numeric_limits<double>::denorm_min() and std::isinf(e.d) and e.d < 0);
            CHECK(std::bit_cast<std::uint64_t>((*i2->robot.room_T_robot)(0, 3)) == std::bit_cast<std::uint64_t>(qnan));
            CHECK(i2->robot.speed_variance.size() == 4);
            if (i2->robot.speed_variance.size() == 4)
                for (int k = 0; k < 4; ++k)
                    CHECK(std::bit_cast<std::uint32_t>(i2->robot.speed_variance[k])
                          == std::bit_cast<std::uint32_t>(in.robot.speed_variance[k]));
        }
    }

    // ── 3. partial file: truncated at every byte of the last two records ────────────────────────────────────────
    {
        Rng r(7);
        std::vector<DoorCycleInput> ins;
        for (int i = 0; i < 5; ++i) ins.push_back(rnd_input(r));
        const DoorRecordingHeader h{.agent_id = 1, .agent_name = "door_concept", .config_text = {}, .start_steady_ns = 0, .start_wall_ms = 0, .note = {}};
        const std::string head = bytes_of([&](std::ostream& s) { (void)write_header(s, h, RecordingKind::Input); });
        std::vector<std::string> recs;
        for (const auto& x : ins) recs.push_back(bytes_of([&](std::ostream& s) { (void)write(s, x); }));
        std::string all = head;
        std::vector<std::size_t> ends;   // byte offset where record i ends
        for (const auto& b : recs) { all += b; ends.push_back(all.size()); }
        const std::size_t from = ends[2];   // start of record 3
        int bad = 0, checked = 0;
        for (std::size_t cut = from; cut <= all.size(); ++cut)
        {
            write_file(dir / "cut.dcin", all.substr(0, cut));
            const auto got = read_input_file((dir / "cut.dcin").string());
            const std::size_t expect = static_cast<std::size_t>(std::ranges::count_if(ends, [&](std::size_t e) { return e <= cut; }));
            ++checked;
            if (not got or got->cycles.size() != expect) { ++bad; continue; }
            const bool on_boundary = std::ranges::contains(ends, cut);
            if (got->stop != (on_boundary ? StopReason::CleanEof : StopReason::Truncated)) { ++bad; continue; }
            if (got->stop_offset != (expect == 0 ? head.size() : ends[expect - 1])) { ++bad; continue; }
            for (std::size_t i = 0; i < expect; ++i) if (not (got->cycles[i] == ins[i])) { ++bad; break; }
        }
        CHECK(bad == 0);
        std::print("partial file: {} truncation points checked, {} wrong\n", checked, bad);
        // truncated INSIDE the header: refused (no header ⇒ nothing is trustworthy)
        write_file(dir / "cut.dcin", head.substr(0, head.size() - 1));
        CHECK(not read_input_file((dir / "cut.dcin").string()).has_value());
        // the stream-level reader: a short record is nullopt, never a half-filled struct
        std::stringstream s(recs[0].substr(0, recs[0].size() / 2), std::ios::in | std::ios::binary);
        CHECK(not read_input(s).has_value());
    }

    // ── 3b. a corrupt record (enum byte out of range) is refused and reported CORRUPT, not truncated ─────────────
    {
        DoorCycleOutput o;
        NodeCreateOut c; c.kind = CreateKind::Anchor; o.creates.push_back(c);
        const DoorRecordingHeader h{.agent_id = 1, .agent_name = {}, .config_text = {}, .start_steady_ns = 0, .start_wall_ms = 0, .note = {}};
        const std::string head = bytes_of([&](std::ostream& s) { (void)write_header(s, h, RecordingKind::Output); });
        std::string rec = bytes_of([&](std::ostream& s) { (void)write(s, o); });
        const std::string good = head + rec + rec;
        // the kind byte: header(12) + cycle(8) t_ms(8) room(8) decisions n(8) doors n(8) creates n(8) seq(8) = offset 68
        std::string bad = rec;
        CHECK(bad[12 + 56] == static_cast<char>(CreateKind::Anchor));
        bad[12 + 56] = 9;
        write_file(dir / "corrupt.dcou", head + rec + bad + rec);
        const auto got = read_output_file((dir / "corrupt.dcou").string());
        CHECK(got.has_value());
        if (got)
        {
            CHECK(got->cycles.size() == 1);
            CHECK(got->stop == StopReason::Corrupt);
            CHECK(got->stop_offset == head.size() + rec.size());
        }
        write_file(dir / "good.dcou", good);
        const auto g2 = read_output_file((dir / "good.dcou").string());
        CHECK(g2 and g2->cycles.size() == 2 and g2->cycles[0].creates.at(0).kind == CreateKind::Anchor);
        // a huge claimed vector count is refused without allocating it
        std::string huge = rec;
        const std::uint64_t n = ~std::uint64_t{0} >> 4;
        std::memcpy(huge.data() + 12 + 24, &n, 8);   // the decisions count
        write_file(dir / "huge.dcou", head + huge);
        const auto g3 = read_output_file((dir / "huge.dcou").string());
        CHECK(g3 and g3->cycles.empty() and g3->stop == StopReason::Corrupt);
    }

    // ── 3c. ownership: created_by reads the creator from the low 12 bits of a DSR id ───────────────────────────────
    CHECK(created_by((std::uint64_t{777} << 12) | 21u, 21u));
    CHECK(not created_by((std::uint64_t{777} << 12) | 22u, 21u));

    // ── 3d. v2 PACKING: the prediction paths, the pixel modes, the unchanged (nullopt) path, sizes ──────────────────
    {
        Rng r(99);
        auto cloud = [&](std::size_t n) { std::vector<Vec3f> v(n); for (auto& p : v) p = Vec3f(r.f(), r.f() * 5.f, r.f()); return v; };
        Mat4d T = Mat4d::Identity();
        T.block<3, 3>(0, 0) = Eigen::AngleAxisd(0.7, Eigen::Vector3d(0.1, 0.2, 1.0).normalized()).toRotationMatrix();
        T.block<3, 1>(0, 3) = Eigen::Vector3d(1.25, -3.5, 0.3);
        auto predicted = [&](const std::vector<Vec3f>& src)
        {
            std::vector<Vec3f> o;
            for (const auto& c : src)
            {
                const Eigen::Vector4d q = T * Eigen::Vector4d(c.x(), c.y(), c.z(), 1.0);
                o.emplace_back(float(q.x()), float(q.y()), float(q.z()));
            }
            return o;
        };
        DoorCycleInput a;
        a.cycle = 1;
        MaskPacket m;
        m.valid = true;
        m.support_points_cam = cloud(20000);
        m.support_points = predicted(m.support_points_cam);        // the ingestor's arithmetic: residual all zero
        m.room_T_zed_capture = T;
        for (int i = 0; i < 30000; ++i) m.mask_pixels.emplace_back(float(100 + i % 200), float(50 + i / 200));
        m.mask_pixels[7] = Vec2f(-0.0f, 3.f);                       // -0 must not take the int16 path
        a.masks = m;
        BpearlSweep b;
        b.robot_fresh = b.room_fresh = true;
        b.pts_robot = cloud(15000);
        b.pts_room = predicted(b.pts_robot);
        for (std::size_t i = 0; i < b.pts_room.size(); i += 3)     // NOT exactly predicted (a different path made it)
            b.pts_room[i].x() = std::nextafter(b.pts_room[i].x(), 1e9f);
        b.room_T_robot_capture = T;
        a.bpearl = b;
        ImageFrame d; d.format = PixelFormat::DepthF32; d.width = 1280; d.height = 720;
        ImagePatch pa; pa.x0 = 3; pa.y0 = 4; pa.w = 50; pa.h = 40;
        std::vector<float> depth(50 * 40);
        for (std::size_t i = 0; i < depth.size(); ++i) depth[i] = i % 17 == 0 ? std::numeric_limits<float>::quiet_NaN() : 2.f + 0.001f * float(i);
        pa.bytes.resize(depth.size() * 4); std::memcpy(pa.bytes.data(), depth.data(), pa.bytes.size());
        d.patches.push_back(pa);
        a.depth = d;
        DoorCycleInput unchanged;            // the next cycle: nothing new → every big stream nullopt
        unchanged.cycle = 2;
        DoorCycleInput no_T = a;             // a derived array whose prediction input is missing → stored explicitly
        no_T.cycle = 3;
        no_T.masks->room_T_zed_capture.reset();
        DoorCycleInput pix_float = a;        // non-integral pixels → the float path
        pix_float.cycle = 4;
        pix_float.masks->mask_pixels[0] = Vec2f(10.5f, 3.25f);
        std::ostringstream o(std::ios::binary);
        CHECK(write(o, a) and write(o, unchanged) and write(o, no_T) and write(o, pix_float));
        const std::string bytes = o.str();
        std::istringstream i(bytes, std::ios::binary);
        const auto ra = read_input(i), ru = read_input(i), rn = read_input(i), rp = read_input(i);
        CHECK(ra.has_value() and *ra == a);
        CHECK(ru.has_value() and *ru == unchanged and not ru->masks and not ru->bpearl and not ru->depth);
        CHECK(rn.has_value() and *rn == no_T);
        CHECK(rp.has_value() and *rp == pix_float);
        if (ra) CHECK(std::bit_cast<std::uint32_t>(ra->masks->mask_pixels[7].x()) == std::bit_cast<std::uint32_t>(-0.0f));
        const std::size_t raw = (m.support_points.size() + m.support_points_cam.size() + b.pts_robot.size() + b.pts_room.size()) * 12
                                + m.mask_pixels.size() * 8 + pa.bytes.size();
        std::ostringstream one(std::ios::binary);
        (void)write(one, a);
        std::print("packing: record {} B for {} B of raw arrays ({:.1f}x)\n", one.str().size(), raw, double(raw) / one.str().size());
        CHECK(one.str().size() * 2 < raw);
        // a corrupted packed byte is refused (zstd checksum / FNV), never decoded into wrong numbers
        std::string bad = one.str();
        bad[bad.size() / 2] = static_cast<char>(bad[bad.size() / 2] ^ 0x5a);
        std::istringstream bi(bad, std::ios::binary);
        CHECK(not read_input(bi).has_value());
    }

    // ── 4. refusals ──────────────────────────────────────────────────────────────────────────────────────────────
    {
        const DoorRecordingHeader h{.agent_id = 1, .agent_name = {}, .config_text = {}, .start_steady_ns = 0, .start_wall_ms = 0, .note = {}};
        std::string good = bytes_of([&](std::ostream& s) { (void)write_header(s, h, RecordingKind::Input); });
        std::string bad_magic = good;   bad_magic[0] = 'X';
        std::string bad_ver   = good;   bad_ver[4]   = static_cast<char>(kCycleIoVersion + 1);   // a future version
        std::string ver0      = good;   ver0[4]      = 0;
        write_file(dir / "m.dcin", bad_magic); CHECK(not read_input_file((dir / "m.dcin").string()).has_value());
        write_file(dir / "v.dcin", bad_ver);   CHECK(not read_input_file((dir / "v.dcin").string()).has_value());
        write_file(dir / "v0.dcin", ver0);     CHECK(not read_input_file((dir / "v0.dcin").string()).has_value());
        write_file(dir / "g.dcin", good);      CHECK(read_input_file((dir / "g.dcin").string()).has_value());
        const std::string out_head = bytes_of([&](std::ostream& s) { (void)write_header(s, h, RecordingKind::Output); });
        write_file(dir / "o.dcou", out_head);
        CHECK(not read_input_file((dir / "o.dcou").string()).has_value());   // an output file is not an input file
        CHECK(not read_output_file((dir / "g.dcin").string()).has_value());  // nor the reverse
        CHECK(not read_input_file((dir / "missing.dcin").string()).has_value());
    }

    // ── 5. the core is DSR / Qt / Ice / media free ───────────────────────────────────────────────────────────────
    {
        const fs::path core = fs::path(__FILE__).parent_path() / ".." / "src" / "core";
        CHECK(fs::exists(core));
        std::set<fs::path> seen;
        std::vector<std::string> angle;
        int n_files = 0;
        if (fs::exists(core))
            for (const auto& e : fs::recursive_directory_iterator(core))
                if (e.is_regular_file()) { ++n_files; collect_includes(e.path(), seen, angle); }
        // also scan the quoted files themselves for forbidden QUOTED includes (resolved paths)
        // ★Two patterns: library names case-INsensitively, and Qt's class headers (<QObject>, <QtCore/…>) case-
        //   SENSITIVELY — case-insensitive, "Q[A-Z]…" would also flag any lowercase header starting with q.
        static const std::regex forbidden(
            R"((^|/)(dsr/|qt[a-z0-9]*/|ice/|iceutil|media_transport|mask_ingestor|lidar_ingestor|rgb_ingestor|depth_ingestor|genericworker|specificworker|cortex|fastdds|fastrtps))",
            std::regex::icase);
        static const std::regex forbidden_qt(R"((^|/)(Q[A-Z][A-Za-z]*$|Qt[A-Za-z]*/))");
        const auto bad = [](const std::string& x) { return std::regex_search(x, forbidden) or std::regex_search(x, forbidden_qt); };
        CHECK(bad("QObject") and bad("QtCore/qobject.h") and bad("dsr/api/dsr_api.h") and bad("Ice/Ice.h"));
        CHECK(not bad("queue") and not bad("quaternion.h") and not bad("Eigen/Dense"));
        int hits = 0;
        for (const auto& a : angle)
            if (bad(a.substr(0, a.find(' ')))) { ++hits; std::print("  forbidden include <{}\n", a); }
        for (const auto& p : seen)
            if (bad(p.string())) { ++hits; std::print("  forbidden local include {}\n", p.string()); }
        CHECK(n_files >= 2);
        CHECK(hits == 0);
        std::print("core include scan: {} file(s) under src/core, {} reachable local header(s), {} system include(s), {} forbidden\n",
                   n_files, seen.size(), angle.size(), hits);
    }

    if (fails == 0) std::print("test_door_cycle_io: ALL PASS\n");
    else            std::print("test_door_cycle_io: {} FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
