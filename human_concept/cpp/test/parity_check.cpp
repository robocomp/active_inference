// parity_check.cpp
// Replay inputs.csv through the C++ estimator and compare against the authoritative
// Python reference.csv (produced by dump_reference.py). Exits non-zero if any frame
// exceeds tolerance. Usage: parity_check inputs.csv reference.csv
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "../core/body18.h"
#include "../core/human_kinematic_model.h"
#include "../core/vfe_inference.h"

using namespace rc::human;

namespace
{
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

std::vector<std::string> split(const std::string &line)
{
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) cols.push_back(cell);
    return cols;
}

float pf(const std::string &s)
{
    std::string t = s;
    t.erase(0, t.find_first_not_of(" \t\r\n"));
    t.erase(t.find_last_not_of(" \t\r\n") + 1);
    if (t.empty()) return kNaN;
    std::string lo = t;
    std::transform(lo.begin(), lo.end(), lo.begin(), ::tolower);
    if (lo == "nan" || lo == "-nan") return kNaN;
    if (lo == "inf") return std::numeric_limits<float>::infinity();
    return std::strtof(t.c_str(), nullptr);
}

struct InRow { KpArray kp; std::optional<std::array<float, NUM_KP>> conf; };
struct RefRow { Vec11 mu; float mean_l2, rmse, utrace; };

std::vector<std::string> read_data_lines(const std::string &path)
{
    std::vector<std::string> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line))
        if (!line.empty() && line[0] != '#') out.push_back(line);
    return out;
}
}  // namespace

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: %s inputs.csv reference.csv\n", argv[0]);
        return 2;
    }
    const auto in_lines  = read_data_lines(argv[1]);
    const auto ref_lines = read_data_lines(argv[2]);
    if (in_lines.empty() || in_lines.size() != ref_lines.size())
    {
        std::fprintf(stderr, "[parity] row count mismatch: inputs=%zu reference=%zu\n",
                     in_lines.size(), ref_lines.size());
        return 2;
    }

    std::vector<InRow> inputs;
    for (const auto &l : in_lines)
    {
        const auto c = split(l);
        if (c.size() < 1 + 54) { std::fprintf(stderr, "[parity] bad input row\n"); return 2; }
        InRow r;
        r.kp.resize(NUM_KP, 3);
        for (int i = 0; i < NUM_KP; ++i)
            for (int k = 0; k < 3; ++k) r.kp(i, k) = pf(c[1 + i * 3 + k]);
        if (c.size() >= 1 + 54 + NUM_KP)
        {
            std::array<float, NUM_KP> cf{};
            for (int i = 0; i < NUM_KP; ++i) cf[i] = pf(c[1 + 54 + i]);
            r.conf = cf;
        }
        inputs.push_back(std::move(r));
    }

    std::vector<RefRow> refs;
    for (const auto &l : ref_lines)
    {
        const auto c = split(l);
        if (c.size() < 1 + 11 + 3) { std::fprintf(stderr, "[parity] bad ref row\n"); return 2; }
        RefRow r;
        for (int i = 0; i < 11; ++i) r.mu(i) = pf(c[1 + i]);
        r.mean_l2 = pf(c[1 + 11]);
        r.rmse    = pf(c[1 + 12]);
        r.utrace  = pf(c[1 + 13]);
        refs.push_back(r);
    }

    // Defaults must match dump_reference.py's InferenceConfig.
    const SegmentLengths lengths = lengths_from_standard(standard_template());
    const HumanKinematicModel model(lengths);
    InferenceConfig cfg;  // anchors {1,2,5,8,11}, defaults match Python
    AInfLaplacePoseEstimator est(model, cfg);

    // Tolerances.
    const float tol_mu   = 1.5e-2f;   // radians / meters
    const float tol_err  = 3.0e-3f;   // mean_l2 / rmse (meters)
    const float tol_tr   = 0.30f;     // relative tolerance on uncertainty_trace

    float max_dmu = 0.f, max_derr = 0.f, max_dtr = 0.f;
    int fail = 0;

    for (size_t f = 0; f < inputs.size(); ++f)
    {
        const InferenceResult res = est.infer(inputs[f].kp, inputs[f].conf);
        const RefRow &ref = refs[f];

        const float dmu = (res.mu - ref.mu).cwiseAbs().maxCoeff();
        max_dmu = std::max(max_dmu, dmu);

        auto derr = [](float a, float b) {
            if (!std::isfinite(a) && !std::isfinite(b)) return 0.f;
            return std::fabs(a - b);
        };
        const float dml = derr(res.mean_l2, ref.mean_l2);
        const float drm = derr(res.rmse, ref.rmse);
        max_derr = std::max({max_derr, dml, drm});

        float dtr = 0.f;
        if (std::isfinite(res.uncertainty_trace) && std::isfinite(ref.utrace))
            dtr = std::fabs(res.uncertainty_trace - ref.utrace) /
                  std::max(1e-6f, std::fabs(ref.utrace));
        else if (std::isfinite(res.uncertainty_trace) != std::isfinite(ref.utrace))
            dtr = 1e9f;  // one inf, one finite
        max_dtr = std::max(max_dtr, dtr);

        const bool ok = dmu <= tol_mu && dml <= tol_err && drm <= tol_err && dtr <= tol_tr;
        if (!ok)
        {
            ++fail;
            std::printf("[FAIL f=%zu] dmu=%.5f dmean_l2=%.5f drmse=%.5f dtr(rel)=%.4f\n",
                        f, dmu, dml, drm, dtr);
        }
    }

    std::printf("\n[parity] frames=%zu  max|dmu|=%.5f  max|derr|=%.5f  max dtr(rel)=%.4f\n",
                inputs.size(), max_dmu, max_derr, max_dtr);
    std::printf("[parity] tolerances: mu=%.4f err=%.4f tr(rel)=%.2f\n", tol_mu, tol_err, tol_tr);
    std::printf("%s\n", fail == 0 ? "[parity] PASS" : "[parity] FAIL");
    return fail == 0 ? 0 : 1;
}
