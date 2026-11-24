// countable_replay.h — the replay tap (shared-engine plan, revision 2, Task R1).
//
// Every engine change (offer cache, volatility, birth covariance, wall latent, …) must be checked before and after on REAL
// live input. A live tour cannot be the baseline (the bed's σ depends on the birth view), so the fitter can record, per
// instance, every call that changes the persistent belief — the birth seed and its params, the birth position covariance,
// the room interior, each update's complete CountableFrame, predict, inflate_for_age, a step-bound restore and the front
// resolver — and an offline tool (tools/replay/replay.cpp) rebuilds the belief from the seed and re-runs the same calls
// through the CURRENT engine. With the engine unchanged the replay is bit-identical to the live run (tests/replay_test.cpp).
//
// File: one stream per instance birth, `etc/replay/<node>_<run>_<serial>.bin` (Model.ReplayDump; off by default).
//   header  : 8 bytes magic "RCRPLAY\0", u32 version
//   events  : u32 event type, u64 payload length, payload
//   payload : one NAMED record — u32 field count, then per field u16 name length, name, u64 value length, value.
// Values are fixed-width little-endian (float32 / float64 / int32 / int64 / u8 bool), raw IEEE bytes (no text, no locale);
// Eigen fixed matrices column-major; a vector is u64 count + elements; an optional is u8 + value; a struct is a nested
// named record. A reader looks fields up BY NAME: a field the current code has and the dump lacks keeps its default
// (reported as "missing"), a field the dump has and the code lacks is skipped (reported as "unknown"), so a dump stays
// replayable after the params or the frame gain or lose a member. A field whose stored size disagrees with its type is an
// error ("bad"). The tests check, with Boost.PFR, that every member of each recorded struct is listed below — a new member
// with no line here fails replay_test, not a live run.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <locale>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <Eigen/Dense>

#include "countable_belief.h"

namespace rc::replay
{
static_assert(std::endian::native == std::endian::little, "the replay format is little-endian; add byte swapping first");
static_assert(sizeof(int) == 4 and sizeof(long) == 8 and sizeof(float) == 4 and sizeof(double) == 8);

inline constexpr char          kMagic[8] = {'R', 'C', 'R', 'P', 'L', 'A', 'Y', '\0'};
// Bump kVersion when a field's MEANING or UNIT changes (or an event's semantics do). Adding or removing a field needs no
// bump: the by-name reader handles it (missing / unknown). A reader refuses a dump newer than the kVersion it was built with.
inline constexpr std::uint32_t kVersion  = 1;

// ── The recorded structs' fields (one line per member; the tests check the count against Boost.PFR) ─────────────────────
template <class V> void fields(V& v, CountableBeliefState& x)
{ v("cx", x.cx); v("cy", x.cy); v("H", x.H); v("w", x.w); v("h", x.h); v("yaw", x.yaw); }

template <class V> void fields(V& v, rc::ai::beam::StudentT& x)
{
    v("p0", x.p0); v("c", x.c); v("nu", x.nu); v("scale2", x.scale2); v("half_p0c2", x.half_p0c2);
    v("log_norm", x.log_norm); v("lgab", x.lgab);
}
template <class V> void fields(V& v, rc::ai::beam::GaussLegendre01& x) { v("u", x.u); v("w", x.w); }
template <class V> void fields(V& v, rc::ai::beam::BeamConfig& x)
{ v("st", x.st); v("gl", x.gl); v("gl_t", x.gl_t); v("z_levels", x.z_levels); v("e_levels", x.e_levels); }

template <class V> void fields(V& v, CountableBeliefParams& x)
{
    v("sigma_base_m", x.sigma_base_m); v("range_noise_lat_per_m", x.range_noise_lat_per_m);
    v("clutter_frac", x.clutter_frac); v("clutter_scale_m", x.clutter_scale_m); v("model_sigma_m", x.model_sigma_m);
    v("prior_pos_std", x.prior_pos_std); v("prior_size_std", x.prior_size_std); v("prior_yaw_std", x.prior_yaw_std);
    v("prior_footprint_m", x.prior_footprint_m); v("prior_footprint_std", x.prior_footprint_std);
    v("prior_depth_m", x.prior_depth_m); v("prior_height_m", x.prior_height_m); v("prior_height_std", x.prior_height_std);
    v("depth_unobs_precision", x.depth_unobs_precision); v("depth_obs_band_m", x.depth_obs_band_m);
    v("headboard_h_m", x.headboard_h_m); v("headboard_t_m", x.headboard_t_m);
    v("class_p_label_true", x.class_p_label_true);
    v("class_fit_H_m", x.class_fit_H_m); v("class_fit_w_m", x.class_fit_w_m); v("class_fit_h_m", x.class_fit_h_m);
    v("class_alt_std_m", x.class_alt_std_m); v("birth_height_noise_m", x.birth_height_noise_m);
    v("birth_height_outlier_frac", x.birth_height_outlier_frac);
    v("top_no_float_precision", x.top_no_float_precision); v("top_no_float_margin_m", x.top_no_float_margin_m);
    v("plaus_aspect_scale", x.plaus_aspect_scale); v("plaus_size_scale", x.plaus_size_scale);
    v("plaus_height_min", x.plaus_height_min); v("plaus_height_soft", x.plaus_height_soft);
    v("plaus_fe_ref", x.plaus_fe_ref); v("plaus_fe_scale", x.plaus_fe_scale);
    v("plaus_height_prior_gain", x.plaus_height_prior_gain);
    v("wall_precision", x.wall_precision); v("wall_reach_m", x.wall_reach_m); v("wall_flush_prior", x.wall_flush_prior);
    v("room_axis_std", x.room_axis_std); v("wall_parallel_precision", x.wall_parallel_precision);
    v("wall_no_cross_precision", x.wall_no_cross_precision); v("wall_no_cross_margin_m", x.wall_no_cross_margin_m);
    v("door_clearance_gain", x.door_clearance_gain);
    v("volatility_infer", x.volatility_infer); v("volatility_lr", x.volatility_lr);
    v("volatility_sigma", x.volatility_sigma); v("volatility_omega_min", x.volatility_omega_min);
    v("wall_explain_frac", x.wall_explain_frac); v("wall_explain_sigma_m", x.wall_explain_sigma_m);
    v("process_std_m", x.process_std_m); v("process_std_yaw", x.process_std_yaw);
    v("common_mode_pos_std", x.common_mode_pos_std); v("common_mode_size_std", x.common_mode_size_std);
    v("common_mode_yaw_std", x.common_mode_yaw_std);
    v("persistent", x.persistent); v("layer_spread_m", x.layer_spread_m); v("beam", x.beam);
    v("gn_iters", x.gn_iters); v("gn_iters_max", x.gn_iters_max); v("gn_tol_m", x.gn_tol_m); v("gn_tol_rad", x.gn_tol_rad);
    v("gn_tol_sigma", x.gn_tol_sigma); v("gn_reject_all_for_test", x.gn_reject_all_for_test); v("gn_trace", x.gn_trace);
    v("fd_eps", x.fd_eps); v("jac_slope_clamp", x.jac_slope_clamp);
    v("top_thickness", x.top_thickness); v("leg_radius", x.leg_radius); v("pixel_sigma_over_f", x.pixel_sigma_over_f);
    v("depth_sigma0_m", x.depth_sigma0_m); v("depth_sigma_range_coef", x.depth_sigma_range_coef);
    v("footprint_residual", x.footprint_residual); v("depth_bias_std", x.depth_bias_std);
    v("depth_scale_std", x.depth_scale_std); v("coverage_precision", x.coverage_precision);
    v("coverage_robust_c_m", x.coverage_robust_c_m); v("free_space_precision", x.free_space_precision);
    v("footprint_moment_precision", x.footprint_moment_precision);
    v("footprint_moment_completeness_gain", x.footprint_moment_completeness_gain);
    v("footprint_moment_min_completeness", x.footprint_moment_min_completeness);
}

template <class V> void fields(V& v, WallRef& x)
{ v("ok", x.ok); v("p", x.p); v("n", x.n); v("sigma_m", x.sigma_m); v("length_m", x.length_m); }

template <class V> void fields(V& v, rc::ai::LidarRays& x)
{
    v("origin", x.origin); v("endpoints", x.endpoints); v("weights", x.weights); v("precision", x.precision);
    v("robust_c_m", x.robust_c_m); v("surf_eps_m", x.surf_eps_m); v("max_range_m", x.max_range_m);
    v("max_steps", x.max_steps); v("first_crossing", x.first_crossing); v("surface_sigma_m", x.surface_sigma_m);
}

template <class V> void fields(V& v, rc::ai::beam::BeamRays& x)
{
    v("origin", x.origin); v("endpoints", x.endpoints); v("m", x.m); v("kappa", x.kappa); v("q_sum", x.q_sum);
    v("rho_max", x.rho_max); v("sigma_s", x.sigma_s); v("p_f", x.p_f);
}

template <class V> void fields(V& v, rc::ai::beam::ForeignBox& x)
{ v("c", x.c); v("H", x.H); v("w", x.w); v("h", x.h); v("yaw", x.yaw); v("std", x.std); v("z0", x.z0); }

template <class V> void fields(V& v, CountableFrame& x)
{
    v("points", x.points); v("R", x.R); v("cam_origin", x.cam_origin); v("has_rays", x.has_rays);
    v("point_azim", x.point_azim); v("chain_cov_xx", x.chain_cov_xx); v("chain_cov_yy", x.chain_cov_yy);
    v("chain_cov_yaw", x.chain_cov_yaw); v("chain_cov_size", x.chain_cov_size); v("moment_extra_var", x.moment_extra_var);
    v("z_top_obs", x.z_top_obs); v("wall", x.wall); v("room_axis_ok", x.room_axis_ok); v("room_axis", x.room_axis);
    v("lidar", x.lidar); v("lidar_extra", x.lidar_extra); v("lidar_only", x.lidar_only);
    v("rho_cam", x.rho_cam); v("rho_L", x.rho_L); v("rho_chain", x.rho_chain); v("sigma_pose", x.sigma_pose);
    v("chain_dr", x.chain_dr); v("robot_xy", x.robot_xy); v("camera_xy", x.camera_xy);
    v("sig2_cam", x.sig2_cam); v("sig2_L", x.sig2_L); v("fresh_cam", x.fresh_cam); v("fresh_L", x.fresh_L);
    v("beam", x.beam); v("foreign_boxes", x.foreign_boxes); v("mask_error_rate", x.mask_error_rate);
}

template <class V> void fields(V& v, FrontCue& x)
{ v("bearing_rad", x.bearing_rad); v("confidence", x.confidence); v("view_bearing_rad", x.view_bearing_rad); }

// ── Events: one per call that changes the belief ─────────────────────────────────────────────────────────────────────────
// Seed: CountableBelief(state, params) — the fitter's birth (run_inference's lazy init).
struct SeedEv        { std::string node; CountableBeliefState state; CountableBeliefParams params; };
struct BirthPosCovEv { Eigen::Matrix2f excess = Eigen::Matrix2f::Zero(); };   // set_birth_position_cov
struct RoomInteriorEv{ Eigen::Vector2f c = Eigen::Vector2f::Zero(); };        // set_room_interior (written when it changes)
// update(frame); cycle and gated are the fitter's (the ai2 log's `cycle` and `gated`), not read by the belief.
struct UpdateEv      { std::int64_t cycle = 0; bool gated = false; CountableFrame frame; };
// What the LIVE update left: θ, the joint P's diagonal, the GN step count and update()'s return. Not an input: the replay
// tool compares against it (a self-check that the stream holds every call), the test asserts on it.
struct ResultEv      { CountableBeliefState theta; std::vector<double> p_diag; std::int32_t gn_it = 0; float fe = 0.0f; };
struct PredictEv     {};                                                       // predict()
struct InflateEv     { float dt_s = 0.0f, dt_nominal_s = 0.0f; };             // inflate_for_age(dt_s, dt_nominal_s)
struct RestoreEv     {};   // the step bound rejected the last update: the belief goes back to its state before it
struct ResolveFrontEv{ FrontCue cue; float weight = 1.0f; };                   // resolve_front(cue, weight)

template <class V> void fields(V& v, SeedEv& x)         { v("node", x.node); v("state", x.state); v("params", x.params); }
template <class V> void fields(V& v, BirthPosCovEv& x)  { v("excess", x.excess); }
template <class V> void fields(V& v, RoomInteriorEv& x) { v("c", x.c); }
template <class V> void fields(V& v, UpdateEv& x)       { v("cycle", x.cycle); v("gated", x.gated); v("frame", x.frame); }
template <class V> void fields(V& v, ResultEv& x)       { v("theta", x.theta); v("p_diag", x.p_diag); v("gn_it", x.gn_it); v("fe", x.fe); }
template <class V> void fields(V&, PredictEv&)          {}
template <class V> void fields(V& v, InflateEv& x)      { v("dt_s", x.dt_s); v("dt_nominal_s", x.dt_nominal_s); }
template <class V> void fields(V&, RestoreEv&)          {}
template <class V> void fields(V& v, ResolveFrontEv& x) { v("cue", x.cue); v("weight", x.weight); }

using Event = std::variant<SeedEv, BirthPosCovEv, RoomInteriorEv, UpdateEv, ResultEv, PredictEv, InflateEv, RestoreEv,
                           ResolveFrontEv>;
// The on-disk event type is the variant index + 1 (0 is never written). Append new events at the END only.
template <class E, std::size_t I = 0> constexpr std::uint32_t event_type()
{
    if constexpr (std::is_same_v<std::variant_alternative_t<I, Event>, E>) return static_cast<std::uint32_t>(I) + 1u;
    else return event_type<E, I + 1>();
}

// ── Encoding ────────────────────────────────────────────────────────────────────────────────────────────────────────────
namespace detail
{
struct Counter { int n = 0; template <class T> void operator()(const char*, T&) { ++n; } };
template <class T> concept Record = requires(Counter& c, T& t) { fields(c, t); };
template <class T> concept Scalar = std::is_arithmetic_v<T>;
template <class T> struct is_fixed_eigen : std::false_type {};
template <class S, int R, int C, int O, int MR, int MC>
struct is_fixed_eigen<Eigen::Matrix<S, R, C, O, MR, MC>> : std::bool_constant<(R > 0 and C > 0 and std::is_arithmetic_v<S>)> {};
template <class T> concept FixedEigen = is_fixed_eigen<T>::value;
template <class T> concept Raw = Scalar<T> or FixedEigen<T>;     // stored as its bytes
template <class T> struct is_std_array : std::false_type {};
template <class T, std::size_t N> struct is_std_array<std::array<T, N>> : std::true_type {};

struct Out
{
    std::string b;
    void bytes(const void* p, std::size_t n) { b.append(static_cast<const char*>(p), n); }
    template <class T> void pod(const T& v) { bytes(&v, sizeof v); }
};

template <class T> void enc(Out& o, const T& v);

template <Record T> void enc_record(Out& o, const T& rec)
{
    T& r = const_cast<T&>(rec);   // fields() takes a non-const reference; the encoder only reads
    Counter c;
    fields(c, r);
    o.pod(static_cast<std::uint32_t>(c.n));
    const auto field = [&o]<class F>(const char* name, F& f)
    {
        const std::uint16_t nl = static_cast<std::uint16_t>(std::strlen(name));
        o.pod(nl);
        o.bytes(name, nl);
        const std::size_t at = o.b.size();
        o.pod(std::uint64_t{0});                             // the value's length, patched below
        enc(o, f);
        const std::uint64_t len = o.b.size() - at - sizeof(std::uint64_t);
        std::memcpy(o.b.data() + at, &len, sizeof len);
    };
    fields(field, r);
}

template <class T> void enc(Out& o, const T& v)
{
    if constexpr (std::is_same_v<T, bool>) o.pod(static_cast<std::uint8_t>(v ? 1 : 0));
    else if constexpr (Scalar<T>) o.pod(v);
    else if constexpr (FixedEigen<T>) o.bytes(v.data(), sizeof(typename T::Scalar) * T::RowsAtCompileTime * T::ColsAtCompileTime);
    else if constexpr (std::is_same_v<T, std::string>) { o.pod(static_cast<std::uint64_t>(v.size())); o.bytes(v.data(), v.size()); }
    else if constexpr (is_std_array<T>::value) { for (const auto& e : v) enc(o, e); }
    else if constexpr (Record<T>) enc_record(o, v);
    else
    {   // std::vector<E> or std::optional<E>
        if constexpr (requires { v.has_value(); })
        {
            o.pod(static_cast<std::uint8_t>(v.has_value() ? 1 : 0));
            if (v.has_value()) enc(o, *v);
        }
        else
        {
            o.pod(static_cast<std::uint64_t>(v.size()));
            using E = typename T::value_type;
            if constexpr (Raw<E> and not std::is_same_v<E, bool>)
                o.bytes(v.data(), v.size() * sizeof(E));
            else
                for (const auto& e : v) enc(o, e);
        }
    }
}

// What a reader noticed (dotted paths): fields the code has but the dump lacks (default kept), fields the dump has but
// the code lacks (skipped), and fields whose stored size or shape disagrees with the code's type (left at default).
struct Diag { std::set<std::string> missing, unknown, bad; };

struct In
{
    const char* p = nullptr;
    const char* end = nullptr;
    bool ok = true;
    bool take(void* dst, std::size_t n)
    {
        if (not ok or std::size_t(end - p) < n) { ok = false; return false; }
        std::memcpy(dst, p, n);
        p += n;
        return true;
    }
    template <class T> bool pod(T& v) { return take(&v, sizeof v); }
};

template <class T> void dec(In& in, T& v, Diag& d, const std::string& path);

template <Record T> void dec_record(In& in, T& r, Diag& d, const std::string& path)
{
    std::uint32_t n = 0;
    if (not in.pod(n)) return;
    struct Slot { std::string name; const char* p; std::uint64_t len; bool used = false; };
    std::vector<Slot> entries;
    entries.reserve(n);
    for (std::uint32_t i = 0; i < n and in.ok; ++i)
    {
        std::uint16_t nl = 0;
        if (not in.pod(nl)) return;
        std::string name(nl, '\0');
        if (not in.take(name.data(), nl)) return;
        std::uint64_t len = 0;
        if (not in.pod(len) or std::uint64_t(in.end - in.p) < len) { in.ok = false; return; }
        entries.push_back({std::move(name), in.p, len});
        in.p += len;
    }
    const auto field = [&]<class F>(const char* name, F& f)
    {
        const std::string sub = path.empty() ? std::string(name) : path + "." + name;
        for (auto& s : entries)
            if (s.name == name)
            {
                s.used = true;
                In fin{s.p, s.p + s.len};
                F tmp = f;
                dec(fin, tmp, d, sub);
                if (fin.ok and fin.p == fin.end) f = std::move(tmp);
                else d.bad.insert(sub);
                return;
            }
        d.missing.insert(sub);
    };
    fields(field, r);
    for (const auto& s : entries)
        if (not s.used) d.unknown.insert(path.empty() ? s.name : path + "." + s.name);
}

template <class T> void dec(In& in, T& v, Diag& d, const std::string& path)
{
    if constexpr (std::is_same_v<T, bool>) { std::uint8_t b = 0; in.pod(b); v = b != 0; }
    else if constexpr (Scalar<T>) in.pod(v);
    else if constexpr (FixedEigen<T>) in.take(v.data(), sizeof(typename T::Scalar) * T::RowsAtCompileTime * T::ColsAtCompileTime);
    else if constexpr (std::is_same_v<T, std::string>)
    {
        std::uint64_t n = 0;
        if (not in.pod(n) or std::uint64_t(in.end - in.p) < n) { in.ok = false; return; }
        v.assign(in.p, n);
        in.p += n;
    }
    else if constexpr (is_std_array<T>::value) { for (auto& e : v) dec(in, e, d, path); }
    else if constexpr (Record<T>) dec_record(in, v, d, path);
    else if constexpr (requires { v.has_value(); })
    {
        std::uint8_t has = 0;
        if (not in.pod(has)) return;
        if (has == 0) { v.reset(); return; }
        typename T::value_type e{};
        dec(in, e, d, path);
        v = std::move(e);
    }
    else
    {
        std::uint64_t n = 0;
        if (not in.pod(n)) return;
        using E = typename T::value_type;
        if constexpr (Raw<E> and not std::is_same_v<E, bool>)
        {
            if (std::uint64_t(in.end - in.p) / sizeof(E) < n) { in.ok = false; return; }
            v.resize(n);
            in.take(v.data(), n * sizeof(E));
        }
        else
        {
            v.clear();
            for (std::uint64_t i = 0; i < n and in.ok; ++i) { E e{}; dec(in, e, d, path); v.push_back(std::move(e)); }
        }
    }
}
}   // namespace detail

// The number of fields() lines of a recorded struct (the tests compare it with Boost.PFR's member count).
template <detail::Record T> int field_count() { T t{}; detail::Counter c; fields(c, t); return c.n; }

// One event's payload, encoded / decoded alone (the tests use these to exercise the by-name reader).
template <detail::Record T> std::string encode(const T& rec) { detail::Out o; detail::enc_record(o, rec); return std::move(o.b); }
template <detail::Record T> bool decode(std::string_view bytes, T& rec, detail::Diag& d)
{
    detail::In in{bytes.data(), bytes.data() + bytes.size()};
    detail::dec_record(in, rec, d, "");
    return in.ok and in.p == in.end;
}
using Diag = detail::Diag;

// ── Writer: the fitter's tap, one per instance birth ────────────────────────────────────────────────────────────────────
class Writer
{
public:
    // nullptr when the file cannot be created (the caller logs once and runs without a dump).
    static std::shared_ptr<Writer> open(const std::string& path)
    {
        auto w = std::shared_ptr<Writer>(new Writer);
        w->f_.open(path, std::ios::binary | std::ios::trunc);
        w->f_.imbue(std::locale::classic());   // binary, but never a locale on a data file
        if (not w->f_) return nullptr;
        w->path_ = path;
        w->f_.write(kMagic, sizeof kMagic);
        w->f_.write(reinterpret_cast<const char*>(&kVersion), sizeof kVersion);
        w->f_.flush();
        w->check();   // a full or read-only target fails here already: the writer is inert from the start
        return w;
    }
    void seed(const std::string& node, const CountableBeliefState& s, const CountableBeliefParams& p) { put(SeedEv{node, s, p}); }
    void birth_pos_cov(const Eigen::Matrix2f& excess) { put(BirthPosCovEv{excess}); }
    void room_interior(const Eigen::Vector2f& c)
    {
        if (room_ and *room_ == c) return;   // unchanged: the belief's state is the same with or without the call
        room_ = c;
        put(RoomInteriorEv{c});
    }
    void update(std::int64_t cycle, bool gated, const CountableFrame& f)
    {
        if (dead_) return;
        // Encoded in place (a bed's frame is ~35k points): the event is written without copying the frame.
        detail::Out o;
        o.b.reserve(64 + f.points.size() * 16 + f.beam.endpoints.size() * 40);
        UpdateEv head{cycle, gated, {}};
        const auto field = [&]<class F>(const char* name, const F& v)
        {
            const std::uint16_t nl = static_cast<std::uint16_t>(std::strlen(name));
            o.pod(nl); o.bytes(name, nl);
            const std::size_t at = o.b.size();
            o.pod(std::uint64_t{0});
            detail::enc(o, v);
            const std::uint64_t len = o.b.size() - at - sizeof(std::uint64_t);
            std::memcpy(o.b.data() + at, &len, sizeof len);
        };
        detail::Counter c;
        fields(c, head);
        o.pod(static_cast<std::uint32_t>(c.n));
        field("cycle", head.cycle);
        field("gated", head.gated);
        field("frame", f);
        write(event_type<UpdateEv>(), o.b);
    }
    void result(const CountableBelief& b, float fe)
    {
        ResultEv r;
        r.theta = b.state();
        const auto& P = b.joint_covariance();
        r.p_diag.resize(static_cast<std::size_t>(P.rows()));
        for (int i = 0; i < P.rows(); ++i) r.p_diag[static_cast<std::size_t>(i)] = P(i, i);
        r.gn_it = b.last_gn_iters();
        r.fe = fe;
        put(r);
    }
    void predict() { put(PredictEv{}); }
    void inflate(float dt_s, float dt_nominal_s) { put(InflateEv{dt_s, dt_nominal_s}); }
    void restore() { put(RestoreEv{}); }
    void resolve_front(const FrontCue& cue, float weight) { put(ResolveFrontEv{cue, weight}); }

    const std::string& path() const { return path_; }
    // A write failed (disk full, read-only, I/O error): the dump is TRUNCATED at the last whole event before failure(), every
    // later call is a no-op, and the failure was logged once. The belief is never affected.
    bool dead() const { return dead_; }
    const std::string& failure() const { return failure_; }
    int failure_logs() const { return failure_logs_; }
    std::uint64_t bytes_written() const { return bytes_; }
    std::uint64_t last_event_bytes() const { return last_; }

private:
    Writer() = default;
    template <class E> void put(const E& e) { write(event_type<E>(), encode(e)); }
    void write(std::uint32_t type, const std::string& payload)
    {
        if (dead_) return;
        const std::uint64_t len = payload.size();
        f_.write(reinterpret_cast<const char*>(&type), sizeof type);
        f_.write(reinterpret_cast<const char*>(&len), sizeof len);
        f_.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        f_.flush();   // a crashed agent leaves a dump that replays up to its last whole event
        if (not check()) return;
        last_ = sizeof type + sizeof len + len;
        bytes_ += last_;
    }
    // After every write + flush: on a stream failure go inert and log ONCE (path, errno), so a truncated dump is visible in
    // the agent's log and cannot pass for a complete one.
    bool check()
    {
        if (f_) return true;
        const int err = errno;
        dead_ = true;
        failure_ = "Model.ReplayDump: write to " + path_ + " FAILED (errno " + std::to_string(err) + ": " + std::strerror(err)
                   + ") after " + std::to_string(bytes_) + " bytes; the dump is TRUNCATED there and no further events are written";
        std::printf("%s\n", failure_.c_str());
        std::fflush(stdout);
        ++failure_logs_;
        return false;
    }
    std::ofstream f_;
    std::string path_;
    std::optional<Eigen::Vector2f> room_;
    std::uint64_t bytes_ = 0, last_ = 0;
    bool dead_ = false;
    std::string failure_;
    int failure_logs_ = 0;
};

// ── Reader: one event at a time (a tour's dump can be large) ─────────────────────────────────────────────────────────────
class Reader
{
public:
    // false (with error()) when the file is missing, not a replay dump, or of an unknown version (only 1..kVersion).
    bool open(const std::string& path)
    {
        f_.open(path, std::ios::binary | std::ios::ate);
        if (not f_) { err_ = "cannot open " + path; return false; }
        size_ = static_cast<std::uint64_t>(f_.tellg());
        f_.seekg(0);
        char magic[8]{};
        std::uint32_t ver = 0;
        f_.read(magic, sizeof magic);
        f_.read(reinterpret_cast<char*>(&ver), sizeof ver);
        if (not f_ or std::memcmp(magic, kMagic, sizeof kMagic) != 0) { err_ = path + " is not a replay dump"; return false; }
        if (ver < 1 or ver > kVersion) { err_ = path + " has format version " + std::to_string(ver) + ", this reader knows up to "
                                     + std::to_string(kVersion); return false; }
        version_ = ver;
        return true;
    }
    // The next whole event; false at the end (a truncated last event — a crashed agent — ends the stream, error() says so).
    bool next(Event& e)
    {
        std::uint32_t type = 0;
        for (;;)   // skip event types newer than this reader (iteratively)
        {
            std::uint64_t len = 0;
            if (not f_.read(reinterpret_cast<char*>(&type), sizeof type)) return false;   // a clean end
            if (not f_.read(reinterpret_cast<char*>(&len), sizeof len)) { err_ = "truncated event header"; return false; }
            const std::uint64_t at = static_cast<std::uint64_t>(f_.tellg());
            if (len > size_ - std::min(at, size_))
            {
                err_ = "truncated last event (" + std::to_string(len) + " bytes announced, "
                       + std::to_string(size_ - std::min(at, size_)) + " left in the file)";
                return false;
            }
            buf_.resize(len);
            if (not f_.read(buf_.data(), static_cast<std::streamsize>(len))) { err_ = "truncated last event"; return false; }
            if (type >= 1 and type <= std::variant_size_v<Event>) break;
            ++skipped_;
        }
        bool ok = false;
        const auto decode_as = [&]<std::size_t I>(std::integral_constant<std::size_t, I>)
        {
            if (type != I + 1) return;
            std::variant_alternative_t<I, Event> v{};
            ok = decode(buf_, v, diag_);
            e = std::move(v);
        };
        [&]<std::size_t... Is>(std::index_sequence<Is...>)
        { (decode_as(std::integral_constant<std::size_t, Is>{}), ...); }(std::make_index_sequence<std::variant_size_v<Event>>{});
        if (not ok) { err_ = "malformed event of type " + std::to_string(type); return false; }
        return true;
    }
    const Diag& diag() const { return diag_; }
    const std::string& error() const { return err_; }
    std::uint32_t version() const { return version_; }
    int skipped_events() const { return skipped_; }

private:
    std::ifstream f_;
    std::string buf_, err_;
    Diag diag_;
    std::uint32_t version_ = 0;
    std::uint64_t size_ = 0;
    int skipped_ = 0;
};

// ── Player: rebuilds the belief from the seed and re-runs every call through the CURRENT engine ──────────────────────────
// on_update(const UpdateEv&, const CountableBelief& after, float fe) runs after each replayed update; on_result(const
// ResultEv& live, const CountableBelief& replayed) after each recorded live result (the same update's). Returns false on
// a stream error (reader.error()) or an event before the seed. on_change(const CountableBelief&), when given, runs after
// EVERY event that changes the belief (predicts, ageing and restores included), e.g. to track σ between updates.
// on_frame(CountableFrame&), when given, may rewrite each recorded frame BEFORE it is replayed (a tool re-staging an input
// whose rule changed since the dump was recorded; the live result then measures that change).
struct NoChange { void operator()(const CountableBelief&) const {} };
struct NoFrame  { void operator()(CountableFrame&) const {} };
template <class OnUpdate, class OnResult, class OnChange = NoChange, class OnFrame = NoFrame>
bool play(Reader& reader, CountableBelief& b, OnUpdate&& on_update, OnResult&& on_result, OnChange&& on_change = {},
          OnFrame&& on_frame = {})
{
    bool seeded = false;
    CountableBelief pre;   // the belief before the last update (RestoreEv)
    Event e;
    while (reader.next(e))
    {
        if (const auto* s = std::get_if<SeedEv>(&e)) { b = CountableBelief(s->state, s->params); seeded = true; continue; }
        if (not seeded) return false;
        std::visit([&]<class E>(E& ev)
        {
            if constexpr (std::is_same_v<E, BirthPosCovEv>) b.set_birth_position_cov(ev.excess);
            else if constexpr (std::is_same_v<E, RoomInteriorEv>) b.set_room_interior(ev.c);
            else if constexpr (std::is_same_v<E, UpdateEv>)
            {
                pre = b;
                on_frame(ev.frame);
                const float fe = b.update(ev.frame);
                on_update(ev, std::as_const(b), fe);
            }
            else if constexpr (std::is_same_v<E, ResultEv>) on_result(ev, std::as_const(b));
            else if constexpr (std::is_same_v<E, PredictEv>) b.predict();
            else if constexpr (std::is_same_v<E, InflateEv>) b.inflate_for_age(ev.dt_s, ev.dt_nominal_s);
            else if constexpr (std::is_same_v<E, RestoreEv>) b = pre;
            else if constexpr (std::is_same_v<E, ResolveFrontEv>) b.resolve_front(ev.cue, ev.weight);
            if constexpr (not std::is_same_v<E, ResultEv>) on_change(std::as_const(b));
        }, e);
    }
    return seeded and reader.error().empty();
}

}   // namespace rc::replay
