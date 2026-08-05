// replay_main.cpp
// Standalone replay harness: read body keypoints from a CSV and run the
// AInfLaplacePoseEstimator on each frame, printing the same summary / per-joint table
// as the Python main.py (minus ZED I/O and the OpenCV split-view).
//
// CSV format (comma-separated, lines starting with '#' ignored):
//   frame, kp0_x,kp0_y,kp0_z, ..., kp17_x,kp17_y,kp17_z [, c0, ..., c17]
//   -> 1 + 54 columns (no confidence) or 1 + 54 + 18 columns (with confidence).
//   Missing keypoints use "nan".
#include <algorithm>
#include <array>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "../core/body18.h"
#include "../core/csv_parse.h"
#include "../core/human_kinematic_model.h"
#include "../core/vfe_inference.h"

using namespace rc::human;

namespace
{
struct Frame
{
    long id = 0;
    KpArray kp;
    std::optional<std::array<float, NUM_KP>> conf;
};

// ★locale-independent (see core/csv_parse.h) — std::strtof truncated every value at the '.' here.
float parse_float(const std::string &s) { return rc::csv::parse_float(s); }

std::vector<Frame> load_csv(const std::string &path)
{
    std::vector<Frame> frames;
    std::ifstream f(path);
    if (!f.is_open())
    {
        std::fprintf(stderr, "[replay] cannot open %s\n", path.c_str());
        return frames;
    }
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> cols;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) cols.push_back(cell);

        if (cols.size() < 1 + 54)
        {
            std::fprintf(stderr, "[replay] skipping short line (%zu cols)\n", cols.size());
            continue;
        }
        Frame fr;
        fr.id = rc::csv::parse_long(cols[0]);
        fr.kp.resize(NUM_KP, 3);
        for (int i = 0; i < NUM_KP; ++i)
            for (int c = 0; c < 3; ++c)
                fr.kp(i, c) = parse_float(cols[1 + i * 3 + c]);

        if (cols.size() >= 1 + 54 + NUM_KP)
        {
            std::array<float, NUM_KP> cf{};
            for (int i = 0; i < NUM_KP; ++i)
                cf[i] = parse_float(cols[1 + 54 + i]);
            fr.conf = cf;
        }
        frames.push_back(std::move(fr));
    }
    return frames;
}

std::string fmt(float v)
{
    char buf[16];
    if (std::isfinite(v)) std::snprintf(buf, sizeof(buf), "%8.3f", v);
    else                  std::snprintf(buf, sizeof(buf), "   nan  ");
    return buf;
}

std::string fmte(float v)
{
    char buf[16];
    if (std::isfinite(v)) std::snprintf(buf, sizeof(buf), "%8.4f", v);
    else                  std::snprintf(buf, sizeof(buf), "   nan  ");
    return buf;
}

void print_summary(const InferenceResult &res)
{
    std::printf("\n%s\n", std::string(120, '=').c_str());
    std::string anchors;
    for (size_t i = 0; i < res.used_anchors.size(); ++i)
        anchors += (i ? "," : "") + std::to_string(res.used_anchors[i]);
    std::printf("conf=? | mean_L2=%.4f m | RMSE=%.4f m | valid=%d/18 | tr(cov)=%.6f | anchors=[%s]\n",
                res.mean_l2, res.rmse, res.valid_count, res.uncertainty_trace, anchors.c_str());
    std::printf("%s\n", std::string(120, '=').c_str());
}

void print_table(const KpArray &live, const InferenceResult &res)
{
    std::printf("%2s  %-12s | %8s %8s %8s | %8s %8s %8s | %8s %8s %8s | %8s\n",
                "kp", "name", "live_x", "live_y", "live_z",
                "pred_x", "pred_y", "pred_z", "dx", "dy", "dz", "L2(m)");
    std::printf("%s\n", std::string(125, '-').c_str());
    for (int i = 0; i < NUM_KP; ++i)
    {
        std::printf("%2d  %-12s | %s %s %s | %s %s %s | %s %s %s | %s\n",
                    i, std::string(KP18_NAMES[i]).c_str(),
                    fmt(live(i, 0)).c_str(), fmt(live(i, 1)).c_str(), fmt(live(i, 2)).c_str(),
                    fmt(res.kp_pred_aligned(i, 0)).c_str(), fmt(res.kp_pred_aligned(i, 1)).c_str(),
                    fmt(res.kp_pred_aligned(i, 2)).c_str(),
                    fmt(res.diff(i, 0)).c_str(), fmt(res.diff(i, 1)).c_str(), fmt(res.diff(i, 2)).c_str(),
                    fmte(res.per_l2[i]).c_str());
    }
}

std::vector<int> parse_anchors(const std::string &s)
{
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) out.push_back(rc::csv::parse_int(tok));
    return out;
}
}  // namespace

int main(int argc, char **argv)
{
    // Same locale the agent runs in (Qt does this) — with from_chars parsing the result is identical
    // either way, which is exactly the property being asserted. See ../core/csv_parse.h.
    std::setlocale(LC_ALL, "");

    std::string input;
    InferenceConfig cfg;
    int print_every = 1;
    int min_valid = 12;
    float uncertainty_thresh = 0.05f;

    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if      (a == "--input")              input = next();
        else if (a == "--anchors")            cfg.anchors = parse_anchors(next());
        // CLI numbers go through the same locale-independent parser: `--sigma_obs 0.06` used to
        // arrive as 0 under es_ES, silently running the sweep at a different config than requested.
        else if (a == "--sigma_obs")          cfg.sigma_obs = rc::csv::parse_number<float>(next(), cfg.sigma_obs);
        else if (a == "--sigma_dyn")          cfg.sigma_dyn = rc::csv::parse_number<float>(next(), cfg.sigma_dyn);
        else if (a == "--sigma_min")          cfg.sigma_min = rc::csv::parse_number<float>(next(), cfg.sigma_min);
        else if (a == "--sigma_max")          cfg.sigma_max = rc::csv::parse_number<float>(next(), cfg.sigma_max);
        else if (a == "--w_limits")           cfg.w_limits = rc::csv::parse_number<float>(next(), cfg.w_limits);
        else if (a == "--w_sym")              cfg.w_sym = rc::csv::parse_number<float>(next(), cfg.w_sym);
        else if (a == "--gn_steps")           cfg.gn_steps = rc::csv::parse_int(next(), cfg.gn_steps);
        else if (a == "--damping")            cfg.damping = rc::csv::parse_number<float>(next(), cfg.damping);
        else if (a == "--print_every")        print_every = rc::csv::parse_int(next(), print_every);
        else if (a == "--min_valid")          min_valid = rc::csv::parse_int(next(), min_valid);
        else if (a == "--uncertainty_thresh") uncertainty_thresh = rc::csv::parse_number<float>(next(), uncertainty_thresh);
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }

    if (input.empty())
    {
        std::fprintf(stderr, "usage: %s --input <csv> [--anchors a,b,c ...]\n", argv[0]);
        return 2;
    }

    const SegmentLengths lengths = lengths_from_standard(standard_template());
    const HumanKinematicModel model(lengths);
    AInfLaplacePoseEstimator estimator(model, cfg);

    const std::vector<Frame> frames = load_csv(input);
    std::printf("[replay] loaded %zu frames from %s\n", frames.size(), input.c_str());

    int idx = 0;
    for (const Frame &fr : frames)
    {
        const InferenceResult res = estimator.infer(fr.kp, fr.conf);
        ++idx;
        if ((idx % std::max(1, print_every)) != 0) continue;

        print_summary(res);
        print_table(fr.kp, res);
        if (res.valid_count < min_valid || res.uncertainty_trace > uncertainty_thresh)
            std::printf("\n[Action hint] High uncertainty / missing joints -> "
                        "change viewpoint, reduce occlusion, move closer.\n");
    }
    return 0;
}
