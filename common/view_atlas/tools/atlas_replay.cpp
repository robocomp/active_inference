// atlas_replay — run the view-atlas module over the labelled views dataset (VIEW_ATLAS.md §6 layer 3).
// usage: atlas_replay <dataset_dir> <constants.txt> <out_dir>
#include "../atlas_io.h"
#include "../view_atlas.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <clocale>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace rc::atlas;

struct Row { int id; std::string dataset, inst, cls; std::uint64_t stamp; int lap;
             float cam_x, cam_y, obj_x, obj_y, obj_yaw, range; int px; float trunc; };

static std::vector<Row> read_views(const std::string& path)
{
    std::vector<Row> out;
    std::ifstream f(path);
    std::string line;
    std::getline(f, line);   // header
    while (std::getline(f, line))
    {
        const auto c = detail::split(line, ',');
        Row r{};
        auto num = [&](std::size_t i, auto& v) { detail::parse(c[i], v); };
        num(0, r.id); r.dataset = std::string(c[1]); num(2, r.stamp); num(3, r.lap);
        r.inst = std::string(c[4]); r.cls = std::string(c[6]);
        num(7, r.cam_x); num(8, r.cam_y); num(9, r.obj_x); num(10, r.obj_y); num(11, r.obj_yaw);
        num(12, r.range); num(13, r.px); num(14, r.trunc);
        out.push_back(r);
    }
    return out;
}

static std::vector<float> read_f32(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    const auto n = std::size_t(f.tellg()); f.seekg(0);
    std::vector<float> v(n / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n));
    return v;
}

static float bearing_of(const Row& r) { return wrap_pi(std::atan2(r.cam_y - r.obj_y, r.cam_x - r.obj_x) - r.obj_yaw); }

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");
    if (argc < 4) { std::fprintf(stderr, "usage: %s dataset_dir constants.txt out_dir\n", argv[0]); return 2; }
    AtlasParams P; std::string why;
    if (not load_params(argv[2], P, &why)) { std::printf("constants refused: %s\n", why.c_str()); return 2; }
    const std::string d = argv[1];
    const auto views = read_views(d + "/views.csv");
    const auto S = read_f32(d + "/shape.f32"), C = read_f32(d + "/colour.f32");
    std::filesystem::create_directories(argv[3]);
    std::ofstream log(std::string(argv[3]) + "/replay_views.csv");
    log.imbue(std::locale::classic());
    log << "view_id,inst,lap,bearing,quality,learning_value,encoded,lr_own,lr_rival\n";

    auto obs_of = [&](const Row& r) {
        ViewObs o;
        o.shape = Eigen::Map<const Eigen::VectorXf>(S.data() + std::size_t(r.id) * 384, 384);
        o.colour = Eigen::Map<const Eigen::VectorXf>(C.data() + std::size_t(r.id) * 64, 64);
        o.bearing_rad = bearing_of(r); o.range_m = r.range;
        o.quality = view_quality(r.trunc, 1.f, r.px, P.quality_px_scale); o.stamp_ms = r.stamp;
        return o;
    };
    const auto known_yaw   = shift_log_weights(P.n_bins, 0.f);    // the object fit gives yaw: sigma 0
    const auto unknown_yaw = shift_log_weights(P.n_bins, 1e3f);   // robustness: yaw not known at all

    // ── A + C: lap 0, chronological, both datasets; atlases per instance ─────────────────────────
    std::map<std::string, ViewAtlas> atlas;
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> seen;   // first/last stamp
    std::map<std::string, int> n_encoded, n_views_f;   // n_views_f: frames-dataset views seen (info only)
    int chair_views = 0, swaps = 0;
    for (const auto& r : views)
    {
        if (r.lap != 0) continue;
        auto [it, fresh] = atlas.try_emplace(r.inst, P, 384, 64);
        const ViewObs o = obs_of(r);
        const float lv = it->second.learning_value(o.bearing_rad, o.quality);
        float lr_own = 0.f, lr_riv = 0.f;
        if (r.cls == "chair")
        {
            const std::string rival = r.inst == "Chair" ? "chair(1)" : "Chair";
            if (atlas.contains(rival) and it->second.state().n_c > 0.f)
            {
                // rival's bearing = the camera seen from the rival, in the rival's frame
                const Row* any_rival = nullptr;
                for (const auto& x : views) if (x.inst == rival and x.lap == 0) { any_rival = &x; break; }
                Row rr = r; rr.obj_x = any_rival->obj_x; rr.obj_y = any_rival->obj_y; rr.obj_yaw = any_rival->obj_yaw;
                lr_own = it->second.log_ratio_marginal(o, o.bearing_rad, known_yaw);
                lr_riv = atlas.at(rival).log_ratio_marginal(o, bearing_of(rr), known_yaw);
                ++chair_views; if (lr_riv > lr_own) ++swaps;
            }
        }
        const bool enc = lv > P.lambda_nats;
        if (enc) it->second.update(o);
        if (r.dataset == "frames")      // orbit stamps are synthetic (100 ms per frame): no real time there
        {
            ++n_views_f[r.inst];
            if (enc) ++n_encoded[r.inst];
            auto& s = seen[r.inst]; if (s.first == 0) s.first = r.stamp; s.second = r.stamp;
        }
        log << r.id << ',' << r.inst << ',' << r.lap << ',' << o.bearing_rad << ',' << o.quality << ','
            << lv << ',' << enc << ',' << lr_own << ',' << lr_riv << '\n';
    }
    const float swap_rate = chair_views ? float(swaps) / float(chair_views) : 1.f;
    std::printf("A. chair swaps: %d / %d = %.2f %% -> %s\n", swaps, chair_views, 100.f * swap_rate, swap_rate < 0.01f ? "PASS" : "FAIL");

    float worst_rate = 0.f;
    for (const auto& [inst, n] : n_encoded)
    {
        const float minutes = std::max(1e-3f, float(seen[inst].second - seen[inst].first) / 60000.f);
        worst_rate = std::max(worst_rate, float(n) / minutes);
        std::printf("   %-28s encoded %4d of %4d views over %6.1f min (first-to-last stamp) = %5.1f /min\n", inst.c_str(), n, n_views_f[inst], minutes, float(n) / minutes);
    }
    std::printf("C. worst encodings per object-minute (apartment tour, real time): %.1f -> %s\n", worst_rate, worst_rate <= 60.f ? "PASS" : "FAIL");

    // ── B + D: archive the chairs after lap 0; lap 1 windows of 5 views are birth candidates ──────
    // The bearing of a detection comes from ITS OWN geometric fit (position + yaw), whichever identity is
    // hypothesised; identity changes which atlas is asked, not where the object is.
    const std::vector<std::string> chairs = {"Chair", "chair(1)"};
    std::map<std::string, Eigen::Vector2f> last_xy;
    for (const auto& r : views) if (r.lap == 0 and (r.inst == "Chair" or r.inst == "chair(1)")) last_xy[r.inst] = {r.obj_x, r.obj_y};
    const float room_area = 60.f;                               // apartment free space (rooms.json id 0: 60.41 m^2)
    const Eigen::Matrix2f pose_cov = Eigen::Matrix2f::Identity() * 0.05f * 0.05f;
    const float log_prior_new = std::log(0.5f), log_prior_k = std::log(0.25f);
    struct Dec { float p; bool right; };
    std::vector<Dec> decisions;
    bool b_ok = true;
    for (const auto* yaw_w : {&known_yaw, &unknown_yaw})
    {
        const bool known = yaw_w == &known_yaw;
        std::map<std::string, std::pair<int, int>> correct;     // inst -> (right, total)
        int wrong_chair = 0, total_windows = 0;
        for (const auto& truth : chairs)
        {
            std::vector<const Row*> lap1;
            for (const auto& r : views) if (r.lap == 1 and r.inst == truth) lap1.push_back(&r);
            for (std::size_t w0 = 0; w0 + 5 <= lap1.size(); w0 += 5)
            {
                std::vector<ReidCandidate> cand;
                for (const auto& k : chairs)
                {
                    ReidCandidate c; c.log_prior = log_prior_k;
                    c.log_loc = location_log_lik({lap1[w0]->obj_x, lap1[w0]->obj_y}, last_xy[k], pose_cov, P.pi_move, room_area);
                    for (std::size_t i = w0; i < w0 + 5; ++i)
                        c.log_ratio += atlas.at(k).log_ratio_marginal(obs_of(*lap1[i]), bearing_of(*lap1[i]), *yaw_w);
                    cand.push_back(c);
                }
                const auto post = reid_posterior(cand, log_prior_new, std::log(1.f / room_area));
                const std::size_t map = std::size_t(std::max_element(post.begin(), post.end()) - post.begin());
                const std::size_t want = truth == "Chair" ? 0 : 1;
                auto& cr = correct[truth]; ++cr.second; if (map == want) ++cr.first;
                if (map < chairs.size() and map != want) ++wrong_chair;
                ++total_windows;
                if (known) decisions.push_back({post[map], map == want});
            }
        }
        bool ok = true;
        for (const auto& [inst, rt] : correct)
        {
            const float acc = rt.second ? float(rt.first) / float(rt.second) : 0.f;
            ok = ok and acc >= 0.95f;
            std::printf("   [yaw %s] %-10s re-identified in %d / %d windows = %.1f %%\n", known ? "known  " : "unknown",
                        inst.c_str(), rt.first, rt.second, 100.f * acc);
        }
        const float wrong_rate = total_windows ? float(wrong_chair) / float(total_windows) : 1.f;
        ok = ok and wrong_rate <= 0.02f;
        if (known) b_ok = ok;
        std::printf("%s moved/stayed chair re-ID, yaw %s (other chair chosen %.1f %%) -> %s\n",
                    known ? "B." : "  (info)", known ? "known" : "unknown", 100.f * wrong_rate, ok ? "PASS" : "FAIL");
    }

    bool d_ok = true;
    const std::pair<float, float> bands[] = {{0.5f, 0.7f}, {0.7f, 0.9f}, {0.9f, 1.0001f}};
    for (auto [lo, hi] : bands)
    {
        float ps = 0.f, rs = 0.f; int n = 0;
        for (const auto& dd : decisions) if (dd.p >= lo and dd.p < hi) { ps += dd.p; rs += dd.right; ++n; }
        if (n == 0) { std::printf("   posterior [%.1f,%.1f): no decisions\n", lo, hi); continue; }
        const float gap = std::fabs(ps / float(n) - rs / float(n));
        if (n >= 20) d_ok = d_ok and gap <= 0.1f;
        std::printf("   posterior [%.1f,%.1f): n=%d mean p %.3f, accuracy %.3f\n", lo, hi, n, ps / float(n), rs / float(n));
    }
    std::printf("D. calibration -> %s\n", d_ok ? "PASS" : "FAIL");
    return (swap_rate < 0.01f and worst_rate <= 60.f and b_ok and d_ok) ? 0 : 1;
}
