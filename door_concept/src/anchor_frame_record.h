/*
 * anchor_frame_record.h — Task 8, fix 3: the interaction anchor's measured cycles, recorded LIVE and replayed OFFLINE.
 *
 * While an anchor exists, door_concept appends every MEASURED cycle (one update_door_anchor call) to
 * etc/anchor_frames.bin: everything the update read and everything it returned, so tests/replay_anchor_frames.cpp can
 * re-run the same update on the same inputs and print where the live numbers came from. It is how the next live
 * failure gets debugged without a re-run (try65: the anchor was right and its publication flapped; the cause had to be
 * reconstructed by hand from a CSV of outputs).
 *
 * ─── FORMAT (version 4, fixed layout, host byte order = little-endian on every machine this fleet runs) ─────────────
 *   header   8 bytes "DANCHORF", u32 version
 *   record   u32 record_bytes (the rest of the record, for skipping a version this reader does not know)
 *            i64 t_ms
 *            prior RelPose: f32 x[3], f32 P[9] (row-major), i64 stamp_ms
 *            f32 width, thickness, wall_u_min, wall_u_max, floor_below
 *            f32 lidar_origin[3]
 *            staged × 2 (at the PRIOR: the fit; at the POSTERIOR: door_view, what was published), each:
 *              u32 n_pre (rays after select_anchor_rays, before the budget), f64 az_step (the frame's measured azimuth
 *              step), u32 n, then n × f32[3] (the budgeted robot-frame rays) and n × f64 (their Horvitz-Thompson weights).
 *              (Fix round 1: v1 stored the whole pre-budget set, ~90 KB a frame; ≤ 2 × 400 rays is ≤ 16 KB. The update
 *              consumes exactly these, so the replay is still exact.)
 *            u32 n_mask, then n_mask × f32[3]    — the robot-frame door-mask points (0 on a cycle with no new mask)
 *            camera frustum: u8 known, f32 origin[2], f32 forward[2], f32 half_fov, f32 max_range
 *            leaf belief: u8 known, u8 phi_known, f32 hinge_u, dir_u, swing_y, phi, phi_sigma, phi_max, p_hinge,
 *                         p_swing, p_not_as_believed
 *            f32 pi_open (the visibility's P(open))
 *            posterior RelPose: f32 x[3], f32 P[9], i64 stamp_ms
 *            f32 p_lidar, p_cam, p_view          — AS PUBLISHED: p_cam / p_view include fix 1's held camera term
 *            f32 lidar_open, lidar_door, lidar_wall   — p_lidar's three factors
 *   Binary, so no text parsing (and no locale) is involved anywhere.
 *   VERSION 4 (Task 10) adds, after p_view's trio of factors, f32 bias_L[3 × kBiasCols] (row-major): the persistent-bias
 *   factor the update was handed with its prior (door_anchor_update.h AnchorBias), so a plain replay re-runs the same
 *   update; versions 1–3 read L = 0 (they predate it: their prior P is all data-space).
 *   VERSION 3 (Task 9 fix round 1) adds f32 leaf.len after p_not_as_believed (the leaf's own length: the clear width).
 *   VERSION 2 (dd1d525f–f8dc1356, e.g. try67) is still read: leaf.len = 0 (= the opening's width, as it then was).
 *   VERSION 1 (cc39a46c, e.g. try66) is still READ (Task 9): in place of the two staged sets it holds u32 n_rays then
 *   n_rays × f32[3] (the PRE-budget rays after select_anchor_rays) and u32 n_budget. read_all puts those in
 *   Frame::rays_v1 / n_budget_v1 and sets Frame::version = 1; a replay then stages them itself (the budget is
 *   deterministic), so the CURRENT model can be re-run on a v1 file.
 * ⚑ The file is capped at kMaxBytes (200 MB, a disk budget, not a model level): past it, recording stops for the run.
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <istream>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

#include "door_anchor_update.h"

namespace rc::door::rec
{

inline constexpr char          kMagic[8] = {'D', 'A', 'N', 'C', 'H', 'O', 'R', 'F'};
inline constexpr std::uint32_t kVersion  = 4;
inline constexpr std::uint64_t kMaxBytes = 200ull * 1024ull * 1024ull;   // ⚑ disk budget
// I/O batching, not a model level: the recorder (and door_concept's anchor_log.csv) flush every this many records and at
// close — never per record on the compute thread (final review I-4).
inline constexpr std::uint32_t kAnchorRecordFlushEvery = 64;

struct Frame
{
    std::int64_t t_ms = 0;
    rc::interaction::RelPose prior;
    float width = 0.f, thickness = 0.f, wall_u_min = 0.f, wall_u_max = 0.f, floor_below = 0.f;
    Eigen::Vector3f lidar_origin = Eigen::Vector3f::Zero();
    AnchorStaging staging;                 // the two budgeted ray sets the update consumed (replay = true to re-run it)
    std::vector<Eigen::Vector3f> mask;
    CameraFrustum cam;
    LeafPrior leaf;
    float pi_open = 0.5f;
    rc::interaction::RelPose post;
    float p_lidar = 0.f, p_cam = 0.f, p_view = 0.f;
    float lidar_open = 0.f, lidar_door = 0.f, lidar_wall = 0.f;
    AnchorBias bias;                       // v4: the persistent-bias factor handed to the update with `prior`
    // Version-1 records only (read, never written): the pre-budget rays and the budgeted count.
    std::uint32_t version = kVersion;
    std::vector<Eigen::Vector3f> rays_v1;
    std::uint32_t n_budget_v1 = 0;
};

namespace detail
{
template <class T> void put(std::string& b, const T& v)
{
    static_assert(std::is_trivially_copyable_v<T>);
    char c[sizeof(T)];
    std::memcpy(c, &v, sizeof(T));
    b.append(c, sizeof(T));
}
template <class T> bool get(const char*& p, const char* end, T& v)
{
    static_assert(std::is_trivially_copyable_v<T>);
    if (std::size_t(end - p) < sizeof(T)) return false;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return true;
}
inline void put_pose(std::string& b, const rc::interaction::RelPose& r)
{
    for (int i = 0; i < 3; ++i) put(b, float(r.x(i)));
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) put(b, float(r.P(i, j)));
    put(b, std::int64_t(r.stamp_ms));
}
inline bool get_pose(const char*& p, const char* e, rc::interaction::RelPose& r)
{
    bool ok = true;
    for (int i = 0; i < 3; ++i) { float v = 0; ok = ok and get(p, e, v); r.x(i) = v; }
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { float v = 0; ok = ok and get(p, e, v); r.P(i, j) = v; }
    std::int64_t s = 0;
    ok = ok and get(p, e, s);
    r.stamp_ms = s;
    return ok;
}
inline void put_pts(std::string& b, const std::vector<Eigen::Vector3f>& v)
{
    put(b, std::uint32_t(v.size()));
    for (const auto& q : v) { put(b, q.x()); put(b, q.y()); put(b, q.z()); }
}
inline bool get_pts(const char*& p, const char* e, std::vector<Eigen::Vector3f>& v)
{
    std::uint32_t n = 0;
    if (not get(p, e, n) or std::size_t(e - p) < std::size_t(n) * 3 * sizeof(float)) return false;
    v.resize(n);
    for (auto& q : v) { float x = 0, y = 0, z = 0; get(p, e, x); get(p, e, y); get(p, e, z); q = {x, y, z}; }
    return true;
}
inline void put_staged(std::string& b, const StagedRays& s)
{
    put(b, s.n_pre); put(b, s.az_step);
    put_pts(b, s.rays);
    for (const double w : s.wt) put(b, w);
}
inline bool get_staged(const char*& p, const char* e, StagedRays& s)
{
    if (not (get(p, e, s.n_pre) and get(p, e, s.az_step) and get_pts(p, e, s.rays))) return false;
    if (std::size_t(e - p) < s.rays.size() * sizeof(double)) return false;
    s.wt.resize(s.rays.size());
    for (double& w : s.wt) get(p, e, w);
    return true;
}
}  // namespace detail

// One record, serialised (without its leading size word).
inline std::string encode(const Frame& f)
{
    using namespace detail;
    std::string b;
    b.reserve(256 + 20 * (f.staging.prior.rays.size() + f.staging.post.rays.size()) + 12 * f.mask.size());
    put(b, f.t_ms);
    put_pose(b, f.prior);
    for (const float v : {f.width, f.thickness, f.wall_u_min, f.wall_u_max, f.floor_below}) put(b, v);
    put(b, f.lidar_origin.x()); put(b, f.lidar_origin.y()); put(b, f.lidar_origin.z());
    put_staged(b, f.staging.prior);
    put_staged(b, f.staging.post);
    put_pts(b, f.mask);
    put(b, std::uint8_t(f.cam.known ? 1 : 0));
    for (const float v : {f.cam.origin.x(), f.cam.origin.y(), f.cam.forward.x(), f.cam.forward.y(), f.cam.half_fov, f.cam.max_range}) put(b, v);
    put(b, std::uint8_t(f.leaf.known ? 1 : 0));
    put(b, std::uint8_t(f.leaf.phi_known ? 1 : 0));
    for (const float v : {f.leaf.hinge_u, f.leaf.dir_u, f.leaf.swing_y, f.leaf.phi, f.leaf.phi_sigma, f.leaf.phi_max, f.leaf.p_hinge,
                          f.leaf.p_swing, f.leaf.p_not_as_believed, f.leaf.len})
        put(b, v);
    put(b, f.pi_open);
    put_pose(b, f.post);
    for (const float v : {f.p_lidar, f.p_cam, f.p_view, f.lidar_open, f.lidar_door, f.lidar_wall}) put(b, v);
    for (int i = 0; i < 3; ++i) for (int j = 0; j < kBiasCols; ++j) put(b, f.bias.L(i, j));
    return b;
}

inline bool decode(const char* p, const char* e, Frame& f, std::uint32_t version = kVersion)
{
    using namespace detail;
    f.version = version;
    bool ok = get(p, e, f.t_ms) and get_pose(p, e, f.prior);
    for (float* v : {&f.width, &f.thickness, &f.wall_u_min, &f.wall_u_max, &f.floor_below}) ok = ok and get(p, e, *v);
    float ox = 0, oy = 0, oz = 0;
    ok = ok and get(p, e, ox) and get(p, e, oy) and get(p, e, oz);
    f.lidar_origin = {ox, oy, oz};
    if (version == 1) ok = ok and get_pts(p, e, f.rays_v1) and get(p, e, f.n_budget_v1) and get_pts(p, e, f.mask);
    else ok = ok and get_staged(p, e, f.staging.prior) and get_staged(p, e, f.staging.post) and get_pts(p, e, f.mask);
    std::uint8_t k = 0;
    ok = ok and get(p, e, k);
    f.cam.known = k != 0;
    float c[6] = {};
    for (float& v : c) ok = ok and get(p, e, v);
    f.cam.origin = {c[0], c[1]}; f.cam.forward = {c[2], c[3]}; f.cam.half_fov = c[4]; f.cam.max_range = c[5];
    std::uint8_t lk = 0, pk = 0;
    ok = ok and get(p, e, lk) and get(p, e, pk);
    f.leaf.known = lk != 0; f.leaf.phi_known = pk != 0;
    for (float* v : {&f.leaf.hinge_u, &f.leaf.dir_u, &f.leaf.swing_y, &f.leaf.phi, &f.leaf.phi_sigma, &f.leaf.phi_max, &f.leaf.p_hinge,
                     &f.leaf.p_swing, &f.leaf.p_not_as_believed})
        ok = ok and get(p, e, *v);
    if (version >= 3) ok = ok and get(p, e, f.leaf.len);
    ok = ok and get(p, e, f.pi_open) and get_pose(p, e, f.post);
    for (float* v : {&f.p_lidar, &f.p_cam, &f.p_view, &f.lidar_open, &f.lidar_door, &f.lidar_wall}) ok = ok and get(p, e, *v);
    f.bias = {};
    if (version >= 4)
        for (int i = 0; i < 3; ++i) for (int j = 0; j < kBiasCols; ++j) ok = ok and get(p, e, f.bias.L(i, j));
    return ok;
}

// The live side: truncate at start-up, append records until the cap, then stop (once, and say so).
class Recorder
{
public:
    // Truncates `path` and writes the header. False if the file cannot be opened (recording then stays off).
    bool open(const std::string& path)
    {
        out_.open(path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (not out_.is_open()) return false;
        out_.write(kMagic, sizeof(kMagic));
        out_.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
        bytes_ = sizeof(kMagic) + sizeof(kVersion);
        full_ = false;
        return bool(out_);
    }
    // Appends one record. Returns false when nothing was written: not open, or the cap was reached (`capped()` then
    // turns true for the rest of the run).
    bool append(const Frame& f)
    {
        if (not out_.is_open() or full_) return false;
        const std::string b = encode(f);
        const std::uint32_t n = std::uint32_t(b.size());
        if (bytes_ + sizeof(n) + b.size() > kMaxBytes) { full_ = true; out_.flush(); return false; }
        out_.write(reinterpret_cast<const char*>(&n), sizeof(n));
        out_.write(b.data(), std::streamsize(b.size()));
        if (++since_flush_ >= kAnchorRecordFlushEvery) { out_.flush(); since_flush_ = 0; }
        bytes_ += sizeof(n) + b.size();
        return bool(out_);
    }
    bool capped() const { return full_; }
    bool is_open() const { return out_.is_open(); }
    // Flush the tail and close. Called from request_shutdown: the agent exits through std::_Exit and the worker is
    // never deleted, so no destructor ever runs to do it.
    void close() { if (out_.is_open()) out_.close(); }
    std::uint64_t bytes() const { return bytes_; }

private:
    std::ofstream out_;              // closed (and so flushed) by close(), from request_shutdown
    std::uint32_t since_flush_ = 0;
    std::uint64_t bytes_ = 0;
    bool full_ = false;
};

// The offline side: every record of a file, in order (version 2, or version 1 — see the header). False if the header is
// not this format; a truncated last
// record (the agent killed mid-write) ends the read without failing it.
inline bool read_all(std::istream& in, std::vector<Frame>& frames, std::uint32_t* version = nullptr)
{
    char magic[8] = {};
    std::uint32_t ver = 0;
    if (not in.read(magic, sizeof(magic)) or std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) return false;
    if (not in.read(reinterpret_cast<char*>(&ver), sizeof(ver)) or (ver != kVersion and ver != 3 and ver != 2 and ver != 1)) return false;
    if (version != nullptr) *version = ver;
    // A record's length word is bounded by what the file still holds: a corrupt word cannot allocate gigabytes.
    const std::streampos here = in.tellg();
    in.seekg(0, std::ios::end);
    const std::streampos end = in.tellg();
    in.seekg(here);
    std::string buf;
    for (;;)
    {
        std::uint32_t n = 0;
        if (not in.read(reinterpret_cast<char*>(&n), sizeof(n))) break;
        if (std::streamoff(n) > std::streamoff(end - in.tellg())) break;   // truncated (or corrupt) last record
        buf.resize(n);
        if (not in.read(buf.data(), std::streamsize(n))) break;
        Frame f;
        if (not decode(buf.data(), buf.data() + buf.size(), f, ver)) break;
        frames.push_back(std::move(f));
    }
    return true;
}

}  // namespace rc::door::rec
