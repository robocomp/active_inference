// blind_spot_test — rc::detect::BlindSpotMap (common/detectability/blind_spot_map.h, BLIND_SPOT_MAP.md).
// Runs under the machine's locale (setlocale "" as the Qt agents do), so a comma-decimal locale would break
// the save/load round trip here exactly as it would live.
#include <clocale>
#include <cmath>
#include <cstdio>
#include <string>

#include "../../common/detectability/blind_spot_map.h"

using rc::detect::BlindSpotMap;
using rc::detect::Viewpoint;

static int failures = 0;
static void expect(bool ok, const char* what, float got)
{
    std::printf("%-62s %8.4f  %s\n", what, got, ok ? "OK" : "FAIL");
    failures += ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");
    const Viewpoint here{0.2f, 0.4f, std::log(1.5f)};
    const Viewpoint far_round{1.4f, 0.4f, std::log(1.5f)};   // ~4 az kernel lengths away
    const float p_env = 0.64f;

    BlindSpotMap m;
    expect(std::abs(m.query(1, here, p_env).p - p_env) < 1e-6f, "no data: answer is the geometric prior", m.query(1, here, p_env).p);

    for (int i = 0; i < 1114; ++i)                      // table_1: one pose, 1114 confident misses
        m.observe(1, here, false, 0.94f);
    const auto a = m.query(1, here, p_env);
    expect(a.n_own <= 1.0f + 1e-5f, "1114 misses at one pose count as <= ONE observation", a.n_own);
    expect(a.p <= 0.5f * p_env + 1e-4f, "...and halve P(detect) there (one obs vs a one-obs prior)", a.p);
    expect(std::abs(m.query(1, far_round, p_env).p - p_env) < 0.02f, "a viewpoint 4 kernel lengths away keeps the prior", m.query(1, far_round, p_env).p);

    for (int k = 0; k < 6; ++k)                         // five more visits from nearby viewpoints, all misses
    {
        const Viewpoint v{0.2f + 0.08f * k, 0.4f, std::log(1.5f)};
        for (int i = 0; i < 50; ++i) m.observe(1, v, false, 0.9f);
    }
    m.flush(1);
    const auto b = m.query(1, here, p_env);
    expect(b.p < a.p, "more visits in the blind spot: P(detect) falls further", b.p);

    const auto c = m.query(2, here, p_env);              // a NEW table of the same class
    expect(c.n_own == 0.0f and c.n_c > 1.0f, "a new table sees the class evidence (n_c)", c.n_c);
    expect(c.p < p_env, "...and starts with a lowered P(detect) there", c.p);

    for (int i = 0; i < 20; ++i) m.observe(3, far_round, true, 0.9f);   // hits elsewhere do not leak here
    m.flush(3);
    expect(std::abs(m.query(4, here, p_env).p - c.p) < 0.02f, "hits at a distant viewpoint do not leak into the blind spot", m.query(4, here, p_env).p);

    const std::string path = argc > 1 ? argv[1] : "/tmp/blind_spot_test.csv";
    const float before = m.query(7, here, p_env).p;
    expect(m.save(path), "save", static_cast<float>(m.size()));
    BlindSpotMap r;
    const std::size_t n = r.load(path);
    expect(n == m.size(), "load reads every sample (locale-proof)", static_cast<float>(n));
    expect(std::abs(r.query(7, here, p_env).p - before) < 1e-3f, "reloaded class map gives the same answer", r.query(7, here, p_env).p);

    std::printf("%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
