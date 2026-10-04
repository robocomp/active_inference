// rc::StatusReporter — see status_reporter.h.
#include "status_reporter.h"

#include <QDateTime>

#include <algorithm>
#include <cmath>

namespace rc
{
namespace
{
long long now_ms() { return QDateTime::currentMSecsSinceEpoch(); }
float safe_sqrt(float v) { return v > 0.f ? std::sqrt(v) : 0.f; }
}  // namespace

void StatusReporter::waiting(bool peers_ok, const std::vector<std::string>& missing, bool lidar_ok,
                             std::string_view why)
{
    std::string key = std::string(peers_ok ? "1" : "0") + (lidar_ok ? "1" : "0") + std::string(why);
    for (const auto& m : missing) key += "|" + m;
    if (key == last_wait_key_) return;
    last_wait_key_ = std::move(key);
    status::Arr miss;
    for (const auto& m : missing) miss.s(m);
    status::event("wait", status::Obj{}
                              .b("peers_ok", peers_ok)
                              .raw("missing", miss.str())
                              .b("lidar_ok", lidar_ok)
                              .s("why", why));
}

void StatusReporter::observe(const RoomConcept::UpdateResult& r)
{
    auto& l = loc_;
    l.have = true;
    l.ok = r.ok;
    l.diverged = r.diverged;
    l.polished = r.sdf_polished;
    l.x = r.state[2];
    l.y = r.state[3];
    l.th = r.state[4];
    l.sx = safe_sqrt(r.covariance(0, 0));
    l.sy = safe_sqrt(r.covariance(1, 1));
    l.sth = safe_sqrt(r.covariance(2, 2));
    l.cond = r.condition_number;
    l.iters = r.iterations_used;
    l.ee = r.early_exit_metric;
    l.sdf = r.sdf_mse;          // median |SDF| in metres, despite the legacy name
    l.pred_sdf = r.pred_sdf_median;
    l.innov = r.innovation_norm;
    l.kl = r.surprise.kl;
    l.mismatch = r.surprise.mismatch;
    l.scored = r.surprise.scored;
    l.epoch = r.reloc_epoch;
    l.ts_ms = r.timestamp_ms;
    l.walls = static_cast<int>(r.wall_view.walls.size());
    l.verts = static_cast<int>(r.wall_view.polygon.verts.size());
    l.closed = r.wall_view.polygon.closed;
    l.map_ready_view = r.wall_view.map_ready;
    for (int i = 0; i < calib::P_COUNT; ++i)
    {
        l.cv[static_cast<std::size_t>(i)] = r.calib_value(i);
        l.cs[static_cast<std::size_t>(i)] = r.calib_sigma(i);
    }
    l.informed = r.calib_informed;
    l.applied = r.calib_applied;
    l.ccond = r.calib_condition;
    l.episodes = r.calib_episodes;
    l.carried = r.calib_carried;
    l.dropped = r.calib_dropped;
}

void StatusReporter::observe_compute(long long us)
{
    if (comp_t0_ms_ == 0) comp_t0_ms_ = now_ms();
    ++comp_n_;
    comp_sum_us_ += us;
    comp_max_us_ = std::max(comp_max_us_, us);
}

void StatusReporter::note_camcal(const std::string& cam, const std::array<float, 4>& value,
                                 const std::array<float, 4>& sigma, int informed, float cond, long pairs)
{
    auto& c = cams_[cam];
    c.v = value;
    c.s = sigma;
    c.informed = informed;
    c.cond = cond;
    c.pairs = pairs;
    c.t_ms = now_ms();
}

std::string StatusReporter::build_state()
{
    const long long now = now_ms();
    status::Obj st;

    // ── localisation ──
    {
        const auto& l = loc_;
        status::Obj o;
        o.b("have", l.have);
        if (l.have)
        {
            o.b("ok", l.ok).b("diverged", l.diverged).b("polished", l.polished)
             .f("x", l.x, 5).f("y", l.y, 5).f("theta", l.th, 5)
             .f("sx", l.sx, 4).f("sy", l.sy, 4).f("sth", l.sth, 4)
             .f("cond", l.cond, 4).i("iters", l.iters)
             .s("mode", l.iters > 0 ? "solve" : "predict")
             .f("early_exit_metric", l.ee, 4).f("sdf_med", l.sdf, 4).f("pred_sdf_med", l.pred_sdf, 4)
             .f("innov", l.innov, 4).f("kl", l.kl, 4).f("mismatch", l.mismatch, 4).b("scored", l.scored)
             .u("reloc_epoch", l.epoch).i("age_ms", l.ts_ms > 0 ? now - l.ts_ms : -1);
        }
        st.raw("loc", o.str());
    }

    // ── room ──
    {
        status::Obj o;
        if (src_.room) src_.room(o);
        o.i("walls", loc_.walls).i("verts", loc_.verts).b("closed", loc_.closed);
        st.raw("room", o.str());
    }

    // ── motion calibration (7 parameters) ──
    {
        status::Arr names, val, sig;
        for (int i = 0; i < calib::P_COUNT; ++i)
        {
            names.s(calib::param_name(i));
            val.f(loc_.cv[static_cast<std::size_t>(i)], 5);
            sig.f(loc_.cs[static_cast<std::size_t>(i)], 4);
        }
        st.raw("calib", status::Obj{}
                            .raw("names", names.str()).raw("value", val.str()).raw("sigma", sig.str())
                            .i("informed", loc_.informed).i("applied", loc_.applied)
                            .f("cond", loc_.ccond, 4).i("episodes", loc_.episodes)
                            .i("carried", loc_.carried).i("dropped", loc_.dropped)
                            .str());
    }

    // ── camera mounts (pitch deg, height mm, yaw deg, dt x) ──
    {
        status::Arr cams;
        for (const auto& [name, c] : cams_)
        {
            status::Arr v, s;
            for (int i = 0; i < 4; ++i) { v.f(c.v[static_cast<std::size_t>(i)], 5); s.f(c.s[static_cast<std::size_t>(i)], 4); }
            cams.raw(status::Obj{}
                         .s("cam", name).raw("value", v.str()).raw("sigma", s.str())
                         .i("informed", c.informed).f("cond", c.cond, 4).i("pairs", c.pairs)
                         .i("age_ms", now - c.t_ms)
                         .str());
        }
        st.raw("camcal", cams.str());
    }

    // ── streams: rate from the counters' deltas, smoothed over a few snapshots ──
    {
        status::Obj o;
        if (src_.streams)
            for (const auto& smp : src_.streams())
            {
                auto& r = rates_[smp.name];
                if (smp.count >= 0)
                {
                    if (r.count >= 0 and now > r.t_ms and smp.count >= r.count)
                    {
                        const double inst = 1000.0 * static_cast<double>(smp.count - r.count)
                                            / static_cast<double>(now - r.t_ms);
                        r.hz = r.hz <= 0.0 ? inst : 0.5 * r.hz + 0.5 * inst;
                    }
                    r.count = smp.count;
                    r.t_ms = now;
                }
                o.raw(smp.name, status::Obj{}
                                    .f("hz", smp.count >= 0 ? r.hz : -1.0, 4)
                                    .i("age_ms", smp.age_ms)
                                    .str());
            }
        st.raw("streams", o.str());
    }

    // ── compute ──
    {
        const double secs = comp_t0_ms_ > 0 ? 1e-3 * static_cast<double>(now - comp_t0_ms_) : 0.0;
        st.raw("compute", status::Obj{}
                              .f("hz", secs > 0.0 ? static_cast<double>(comp_n_) / secs : 0.0, 4)
                              .f("us", comp_n_ > 0 ? static_cast<double>(comp_sum_us_) / static_cast<double>(comp_n_) : 0.0, 5)
                              .i("us_max", comp_max_us_)
                              .str());
        comp_n_ = comp_sum_us_ = comp_max_us_ = 0;
        comp_t0_ms_ = now;
    }

    // ── presence ──
    {
        status::Arr miss;
        if (src_.missing)
            for (const auto& m : src_.missing()) miss.s(m);
        st.raw("presence", status::Obj{}.raw("missing", miss.str()).s("state", sm_state_).str());
    }
    if (src_.overlay_verbose) st.b("overlay_verbose", src_.overlay_verbose());

    status::event("state", st);
    last_state_ = st.str();
    return last_state_;
}

}  // namespace rc
