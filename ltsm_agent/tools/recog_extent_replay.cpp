// Offline replay of a live room-recognition episode's METRIC expert (room_chain.cpp) against a memory generation.
//   make -C build recog_extent_replay; run from the component root:
//   ./bin/recog_extent_replay <memory.json> <ltsm_agent.log> <first_line> <last_line> <candidates,comma,separated>
//                             [--supports <file: "wall_name value" per line>] [--no-doors]
// For every `[recog][metric] ref ...` line in [first_line, last_line] (1-based) the logged robot pose + covariance in the
// reference room is fed to build_metric_query (empty path, identity fresh seam, so robot_in_ref is the logged one), q.rooms is
// restricted to the episode's candidates, metric_evidence runs, and the recogniser's own filter is emulated (EMA of each
// hypothesis's log-evidence, tau 5 s, neutral start; uniform prior over candidates + new; argmax = what it would propose).
// Logged frames are 5 s apart (the log throttle); 4 poses are interpolated between consecutive frames so the filter sees
// the recogniser's 1 Hz. --supports overrides obj_exist_prob on the memory's wall nodes (the new room_concept model,
// computed offline by room_concept/tools/wall_support_test from the run's own raster); --no-doors switches the door-edge
// boundary off (the code before 2026-11-12, with the memory's own obj_exist_prob, is the "before" replay).
#include "../src/room_chain.h"
#include <QCoreApplication>
#include <charconv>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <print>
#include <regex>
#include <sstream>
#include <string>

namespace
{
double num(const std::string &s)
{
    double v = 0.0;
    const char *b = s.data(), *e = s.data() + s.size();
    if (*b == '+') ++b;
    std::from_chars(b, e, v);
    return v;
}
struct Frame { std::string ref; Eigen::Vector3d pose; Eigen::Matrix3d cov; std::map<std::string, std::string> logged; };
}   // namespace

int main(int argc, char **argv)
{
    std::setlocale(LC_ALL, "");
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 6) { std::println("usage: see the header"); return 2; }
    qputenv("QT_LOGGING_RULES", "*.info=false");
    QCoreApplication app(argc, argv);
    const std::string mem_path = argv[1], log_path = argv[2];
    const int first = int(num(argv[3])), last = int(num(argv[4]));
    std::vector<std::string> cands;
    { std::stringstream ss(argv[5]); for (std::string c; std::getline(ss, c, ',');) cands.push_back(c); }
    std::string supports_path;
    bool doors = true, dump = false;   // --dump: print each candidate's per-edge support and door flags once
    for (int i = 6; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--supports" and i + 1 < argc) supports_path = argv[++i];
        else if (a == "--no-doors") doors = false;
        else if (a == "--dump") dump = true;
    }
    auto mem = std::make_shared<DSR::DSRGraph>("replay_mem", 197, mem_path, true, 47);
    if (not supports_path.empty())
    {
        std::ifstream f(supports_path);
        std::string name, val;
        int n = 0;
        while (f >> name >> val)
            if (auto w = mem->get_node(name); w.has_value())
            {
                mem->add_or_modify_attrib_local<obj_exist_prob_att>(w.value(), float(num(val)));
                mem->update_node(w.value()); ++n;
            }
            else std::println("  (no wall {} in memory)", name);
        std::println("supports: {} walls overridden from {}", n, supports_path);
    }

    // ── the logged frames ──
    std::vector<Frame> frames;
    {
        std::ifstream f(log_path);
        const std::regex head(R"(\[recog\]\[metric\] ref (\S+) robot_ok=true robot_in_ref \(([-+\d.e]+),([-+\d.e]+),([-+\d.e]+)\) cov \[xx (\S+) xy (\S+) xt (\S+) yy (\S+) yt (\S+) tt (\S+)\])");
        const std::regex room(R"(\[recog\]\[metric\]   (\S+) id \d+: .* P_in (\S+) log_inside (\S+))");
        std::string line;
        std::smatch m;
        for (int ln = 1; std::getline(f, line); ++ln)
        {
            if (ln < first or ln > last) continue;
            if (std::regex_search(line, m, head))
            {
                Frame fr;
                fr.ref = m[1];
                fr.pose = {num(m[2]), num(m[3]), num(m[4])};
                fr.cov << num(m[5]), num(m[6]), num(m[7]), num(m[6]), num(m[8]), num(m[9]), num(m[7]), num(m[9]), num(m[10]);
                frames.push_back(fr);
            }
            else if (not frames.empty() and std::regex_search(line, m, room))
                frames.back().logged[m[1]] = std::string(m[2]) + "/" + std::string(m[3]);
        }
    }
    std::println("{} logged frames, candidates {}, doors {}", frames.size(), argv[5], doors ? "ON" : "off");

    std::map<std::string, double> filt;   // EMA per hypothesis ("" = new)
    std::map<std::string, double> max_post, max_inst;   // filtered / instantaneous (= the filter's fixed point at that pose)
    std::string proposals;
    constexpr double kTau = 5.0, kSub = 5.0;   // the recogniser's tau; 5 sub-frames per logged frame (1 Hz)
    const double keep = std::exp(-1.0 / kTau);
    for (std::size_t i = 0; i < frames.size(); ++i)
    {
        const int nsub = i + 1 < frames.size() ? int(kSub) : 1;
        for (int s = 0; s < nsub; ++s)
        {
            const double a = double(s) / kSub;
            Eigen::Vector3d p = frames[i].pose;
            if (s > 0)
            {
                const auto &n = frames[i + 1].pose;
                p.head<2>() = (1 - a) * p.head<2>() + a * n.head<2>();
                p.z() += a * std::remainder(n.z() - p.z(), 2 * M_PI);
            }
            const ltsm::ChainPose robot{p, frames[i].cov};
            auto q = ltsm::build_metric_query(*mem, {}, frames[i].ref, ltsm::ChainPose{}, robot);
            std::erase_if(q.rooms, [&](const ltsm::KnownRoomMetric &r) { return std::ranges::find(cands, r.mem_room) == cands.end(); });
            if (not doors) for (auto &r : q.rooms) r.edge_door.clear();
            if (dump and i == 0 and s == 0)
                for (const auto &r : q.rooms)
                {
                    std::print("  {} support:", r.mem_room);
                    for (std::size_t e = 0; e < r.edge_support.size(); ++e)
                        std::print(" {}:{:.4f}{}", e, r.edge_support[e], e < r.edge_door.size() and r.edge_door[e] ? "D" : "");
                    std::println("");
                }
            float log_new = 0.f;
            const auto ev = ltsm::metric_evidence(q, log_new);
            std::vector<std::pair<std::string, double>> hyp;
            for (std::size_t k = 0; k < q.rooms.size(); ++k)
                if (ev[k].available) hyp.push_back({q.rooms[k].mem_room, ev[k].log_inside});
            hyp.push_back({"", log_new});
            {
                double zi = -1e300, si = 0.0;
                for (const auto &h : hyp) zi = std::max(zi, h.second);
                for (const auto &h : hyp) si += std::exp(h.second - zi);
                for (const auto &h : hyp) max_inst[h.first] = std::max(max_inst[h.first], std::exp(h.second - zi) / si);
            }
            double z = -1e300;
            std::vector<double> lp;
            for (const auto &[name, x] : hyp)
            {
                const double y = keep * (filt.contains(name) ? filt[name] : 0.0) + (1 - keep) * x;
                filt[name] = y; lp.push_back(y);
                z = std::max(z, y);
            }
            double sum = 0; for (const double v : lp) sum += std::exp(v - z);
            std::size_t arg = 0;
            for (std::size_t h = 0; h < hyp.size(); ++h)
            {
                const double post = std::exp(lp[h] - z) / sum;
                max_post[hyp[h].first] = std::max(max_post[hyp[h].first], post);
                if (lp[h] > lp[arg]) arg = h;
            }
            if (s == 0)
            {
                std::print("  frame {:2d} ref {} robot ({:+.2f},{:+.2f}):", i, frames[i].ref, p.x(), p.y());
                for (std::size_t h = 0; h < hyp.size(); ++h)
                {
                    const auto &nm = hyp[h].first.empty() ? std::string("new") : hyp[h].first;
                    const auto it = frames[i].logged.find(hyp[h].first);
                    std::print(" {} {:+.2f} [post {:.3f}{}]", nm, hyp[h].second, std::exp(lp[h] - z) / sum,
                               it == frames[i].logged.end() ? "" : ", logged P_in/log_inside " + it->second);
                }
                std::println("");
            }
            if (not hyp[arg].first.empty()) proposals += std::format(" frame{}.{}:{}", i, s, hyp[arg].first);
        }
    }
    std::print("MAX posterior:");
    for (const auto &[k, v] : max_post) std::print(" {} {:.3f}", k.empty() ? "new" : k, v);
    std::print("\nMAX instantaneous posterior:");
    for (const auto &[k, v] : max_inst) std::print(" {} {:.3f}", k.empty() ? "new" : k, v);
    std::println("\nMAP not new at:{}", proposals.empty() ? " never" : proposals);
    std::fflush(stdout);
    std::_Exit(0);
}
