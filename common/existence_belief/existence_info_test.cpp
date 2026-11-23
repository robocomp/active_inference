// existence_info_test.cpp — common/existence_belief/existence_info.h (I(E;Y), the existence information of one look).
// Standalone:  g++ -std=c++23 -O1 existence_info_test.cpp -o t && ./t
#include <cmath>
#include <cstdio>
#include <numbers>

#include "existence_info.h"

static int fails = 0;
#define CHECK(cond, msg) do { const bool ok_ = (cond); std::printf("  %-60s %s\n", msg, ok_ ? "ok" : "FAIL"); if (not ok_) ++fails; } while (0)

int main()
{
    std::printf("existence_info_test\n");
    using rc::exist::existence_mutual_info;
    const float ln2 = std::numbers::ln2_v<float>;

    {   // a settled belief buys nothing, whatever the sensor
        bool ok0 = true, ok1 = true;
        for (float ph = 0.0f; ph <= 1.0001f; ph += 0.1f)
            for (float pf = 0.0f; pf <= 1.0001f; pf += 0.1f)
            {
                ok0 = ok0 and std::abs(existence_mutual_info(0.0f, ph, pf)) < 1e-6f;
                ok1 = ok1 and std::abs(existence_mutual_info(1.0f, ph, pf)) < 1e-6f;
            }
        CHECK(ok0, "MI(0, *, *) == 0");
        CHECK(ok1, "MI(1, *, *) == 0");
    }
    {   // bounded by H(E) <= ln 2
        bool ok = true;
        float mx = 0.0f;
        for (float p = 0.0f; p <= 1.0001f; p += 0.05f)
            for (float ph = 0.0f; ph <= 1.0001f; ph += 0.05f)
                for (float pf = 0.0f; pf <= 1.0001f; pf += 0.05f)
                {
                    const float mi = existence_mutual_info(p, ph, pf);
                    ok = ok and std::isfinite(mi) and mi >= 0.0f and mi <= ln2 + 1e-6f
                            and mi <= rc::exist::binary_entropy_nats(p) + 1e-6f;
                    mx = std::max(mx, mi);
                }
        CHECK(ok, "0 <= MI <= min(H(p), ln 2) over a 21^3 grid");
        std::printf("    grid max MI = %.6f (ln 2 = %.6f)\n", mx, ln2);
    }
    {   // a better detector (higher hit rate, fixed false rate) tells more
        bool ok = true;
        float prev = existence_mutual_info(0.5f, 0.05f, 0.05f);
        CHECK(std::abs(prev) < 1e-6f, "MI(0.5, p_hit == p_false) == 0 (uninformative detector)");
        for (float ph = 0.10f; ph <= 1.0001f; ph += 0.05f)
        {
            const float mi = existence_mutual_info(0.5f, ph, 0.05f);
            ok = ok and mi > prev;
            prev = mi;
        }
        CHECK(ok, "MI(0.5, p_hit, 0.05) strictly increasing in p_hit");
    }
    {   // the reference number for the default sensor model
        const float mi = existence_mutual_info(0.5f, 0.85f, 0.05f);
        std::printf("    MI(0.5, 0.85, 0.05) = %.6f nats\n", mi);
        CHECK(mi > 0.0f and mi < ln2, "MI(0.5, 0.85, 0.05) in (0, ln 2)");
    }
    std::printf(fails ? "existence_info_test: %d FAILED\n" : "existence_info_test: all passed\n", fails);
    return fails ? 1 : 0;
}
