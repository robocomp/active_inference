// The SCAN TAPE: what the relocaliser sees, one record per relocalisation_step call, so an old and a new relocaliser can
// be replayed offline on a recorded run (spec 2026-11-01-relocaliser-redesign.md, step 0). Pure: no Qt, no graph.
//
// File = "SCTP" magic + u32 version, then records [u8 type][u32 payload bytes][payload], little-endian, floats raw.
//   type 1 ROOM  : name, polygon, apertures  — written once per room change, BEFORE the first frame of that room.
//   type 2 FRAME : t_ms, pose, cov, odo_delta, odo_cov, room, points (robot frame).
// A record cut short by a crash is dropped by the reader, never an error.
#pragma once

#include <Eigen/Dense>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace rc::tape
{
static_assert(std::endian::native == std::endian::little, "the tape is raw little-endian");

struct Frame
{
    std::int64_t t_ms = 0;
    Eigen::Vector3f pose = Eigen::Vector3f::Zero();       ///< (x, y, theta) the tracker committed this frame (room frame)
    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();        ///< its covariance
    Eigen::Vector3f odo_delta = Eigen::Vector3f::Zero();  ///< the PREDICT step's odometry increment (room frame)
    Eigen::Matrix3f odo_cov = Eigen::Matrix3f::Zero();    ///< and the covariance the predict step used for it
    std::string room;
    std::vector<Eigen::Vector2f> pts_robot;               ///< the scan, robot frame
};

struct Aperture
{
    Eigen::Vector2f a{0.f, 0.f}, b{0.f, 0.f}, n_out{0.f, 0.f};
    float p_open = -1.f, prior_closed = 1.f, half_depth = 0.02f, recess = 0.f, sigma_along = 0.f;
    std::string name;
    bool operator==(const Aperture&) const = default;
};

struct RoomRecord
{
    std::string room;
    std::vector<Eigen::Vector2f> polygon;
    std::vector<Aperture> apertures;
};

struct Tape
{
    std::vector<RoomRecord> rooms;
    std::vector<Frame> frames;
    std::vector<int> frame_room;   ///< per frame: index into rooms of the record in force (-1 = none yet)
};

/// Does a new RoomRecord have to be written? When the name, the polygon or the doorway GEOMETRY (a, b, name) differ from
/// the record in force. p_open etc. drift every frame and do not count. A name change alone (polygon unchanged) counts:
/// the label and the geometry are published by different code and can change in either order.
inline bool room_state_changed(const RoomRecord& last, const RoomRecord& now)
{
    if (last.room != now.room or last.polygon != now.polygon or last.apertures.size() != now.apertures.size()) return true;
    for (std::size_t i = 0; i < now.apertures.size(); ++i)
        if (last.apertures[i].a != now.apertures[i].a or last.apertures[i].b != now.apertures[i].b
            or last.apertures[i].name != now.apertures[i].name) return true;
    return false;
}

inline constexpr char kMagic[4] = {'S', 'C', 'T', 'P'};
inline constexpr std::uint32_t kVersion = 1;
inline constexpr std::uint8_t kTypeRoom = 1, kTypeFrame = 2;

namespace detail
{
struct Closer { void operator()(std::FILE* f) const { if (f) std::fclose(f); } };
class Out
{
public:
    std::vector<char> buf;
    template <class T> void put(const T& v) { const auto* p = reinterpret_cast<const char*>(&v); buf.insert(buf.end(), p, p + sizeof(T)); }
    void put_raw(const void* p, std::size_t n) { const auto* c = static_cast<const char*>(p); buf.insert(buf.end(), c, c + n); }
    void put_str(const std::string& s) { put<std::uint32_t>(static_cast<std::uint32_t>(s.size())); put_raw(s.data(), s.size()); }
    void put_pts(const std::vector<Eigen::Vector2f>& v)
    {
        put<std::uint32_t>(static_cast<std::uint32_t>(v.size()));
        if (not v.empty()) put_raw(v.data(), v.size() * sizeof(Eigen::Vector2f));
    }
};

class In
{
public:
    In(const char* p, std::size_t n) : p_(p), n_(n) {}
    bool ok() const { return ok_; }
    template <class T> T get() { T v{}; if (take(sizeof(T))) std::memcpy(static_cast<void*>(&v), p_ + off_ - sizeof(T), sizeof(T)); return v; }
    void get_raw(void* dst, std::size_t n) { if (take(n)) std::memcpy(dst, p_ + off_ - n, n); }
    std::string get_str()
    {
        const auto n = get<std::uint32_t>();
        std::string s;
        if (ok_ and take(n)) s.assign(p_ + off_ - n, n);
        return s;
    }
    std::vector<Eigen::Vector2f> get_pts()
    {
        const auto n = get<std::uint32_t>();
        std::vector<Eigen::Vector2f> v;
        if (ok_ and take(static_cast<std::size_t>(n) * sizeof(Eigen::Vector2f)))
        {
            v.resize(n);
            if (n) std::memcpy(static_cast<void*>(v.data()), p_ + off_ - n * sizeof(Eigen::Vector2f), n * sizeof(Eigen::Vector2f));
        }
        return v;
    }
private:
    bool take(std::size_t n) { if (not ok_ or n > n_ - off_) { ok_ = false; return false; } off_ += n; return true; }
    const char* p_; std::size_t n_, off_ = 0; bool ok_ = true;
};
}   // namespace detail

/// Appends records to `path` (truncating it). A buffered append: one fwrite per record; flushed every ~1 s of frames.
class Writer
{
public:
    explicit Writer(std::string path) : path_(std::move(path))
    {
        f_.reset(std::fopen(path_.c_str(), "wb"));
        if (not f_) return;
        std::setvbuf(f_.get(), nullptr, _IOFBF, 1 << 18);
        std::fwrite(kMagic, 1, 4, f_.get());
        const std::uint32_t v = kVersion;
        std::fwrite(&v, sizeof v, 1, f_.get());
    }
    bool is_open() const { return static_cast<bool>(f_); }
    const std::string& path() const { return path_; }

    void write_room(const RoomRecord& r)
    {
        detail::Out o;
        o.put_str(r.room);
        o.put_pts(r.polygon);
        o.put<std::uint32_t>(static_cast<std::uint32_t>(r.apertures.size()));
        for (const auto& a : r.apertures)
        {
            o.put(a.a); o.put(a.b); o.put(a.n_out);
            o.put(a.p_open); o.put(a.prior_closed); o.put(a.half_depth); o.put(a.recess); o.put(a.sigma_along);
            o.put_str(a.name);
        }
        put_record(kTypeRoom, o);
        if (f_) std::fflush(f_.get());
    }

    void write(const Frame& fr)
    {
        detail::Out o;
        o.buf.reserve(160 + fr.room.size() + fr.pts_robot.size() * sizeof(Eigen::Vector2f));
        o.put(fr.t_ms);
        o.put_raw(fr.pose.data(), 12); o.put_raw(fr.cov.data(), 36);
        o.put_raw(fr.odo_delta.data(), 12); o.put_raw(fr.odo_cov.data(), 36);
        o.put_str(fr.room);
        o.put_pts(fr.pts_robot);
        put_record(kTypeFrame, o);
        if (++since_flush_ >= 20) { since_flush_ = 0; if (f_) std::fflush(f_.get()); }
    }

private:
    void put_record(std::uint8_t type, const detail::Out& o)
    {
        if (not f_) return;
        const std::uint32_t n = static_cast<std::uint32_t>(o.buf.size());
        std::fwrite(&type, 1, 1, f_.get());
        std::fwrite(&n, sizeof n, 1, f_.get());
        std::fwrite(o.buf.data(), 1, o.buf.size(), f_.get());
    }
    std::string path_;
    std::unique_ptr<std::FILE, detail::Closer> f_;
    int since_flush_ = 0;
};

/// The whole tape; an unreadable file, a bad magic/version or a record cut short yields what was complete before it.
inline Tape read_tape(const std::string& path)
{
    Tape t;
    std::unique_ptr<std::FILE, detail::Closer> f(std::fopen(path.c_str(), "rb"));
    if (not f) return t;
    char magic[4]; std::uint32_t ver = 0;
    if (std::fread(magic, 1, 4, f.get()) != 4 or std::memcmp(magic, kMagic, 4) != 0
        or std::fread(&ver, sizeof ver, 1, f.get()) != 1 or ver != kVersion)
        return t;
    std::vector<char> payload;
    for (;;)
    {
        std::uint8_t type = 0; std::uint32_t n = 0;
        if (std::fread(&type, 1, 1, f.get()) != 1 or std::fread(&n, sizeof n, 1, f.get()) != 1) break;
        payload.resize(n);
        if (n and std::fread(payload.data(), 1, n, f.get()) != n) break;
        detail::In in(payload.data(), n);
        if (type == kTypeRoom)
        {
            RoomRecord r;
            r.room = in.get_str();
            r.polygon = in.get_pts();
            const auto na = in.get<std::uint32_t>();
            for (std::uint32_t i = 0; i < na and in.ok(); ++i)
            {
                Aperture a;
                a.a = in.get<Eigen::Vector2f>(); a.b = in.get<Eigen::Vector2f>(); a.n_out = in.get<Eigen::Vector2f>();
                a.p_open = in.get<float>(); a.prior_closed = in.get<float>(); a.half_depth = in.get<float>();
                a.recess = in.get<float>(); a.sigma_along = in.get<float>();
                a.name = in.get_str();
                r.apertures.push_back(std::move(a));
            }
            if (not in.ok()) break;
            t.rooms.push_back(std::move(r));
        }
        else if (type == kTypeFrame)
        {
            Frame fr;
            fr.t_ms = in.get<std::int64_t>();
            in.get_raw(fr.pose.data(), 12); in.get_raw(fr.cov.data(), 36);
            in.get_raw(fr.odo_delta.data(), 12); in.get_raw(fr.odo_cov.data(), 36);
            fr.room = in.get_str();
            fr.pts_robot = in.get_pts();
            if (not in.ok()) break;
            t.frames.push_back(std::move(fr));
            t.frame_room.push_back(static_cast<int>(t.rooms.size()) - 1);
        }
        // an unknown type is skipped by its length: a newer writer's extra records do not break an older reader
    }
    return t;
}

inline std::vector<Frame> read_all(const std::string& path) { return read_tape(path).frames; }
}   // namespace rc::tape
