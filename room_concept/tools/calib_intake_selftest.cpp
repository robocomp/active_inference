// The intake's job is to take what arrives and refuse only what is unusable — and to make
// "nothing arrived" distinguishable from "everything was refused".
#include "../src/calibration_intake.h"
#include <cstdio>
#include <cmath>

using namespace rc::calib;
namespace {
int failures = 0;
void check(bool c, const char* what)
{ std::printf("   %-58s %s\n", what, c ? "PASS" : "*** FAIL ***"); if (not c) ++failures; }

Episode good(float d_fwd, float d_th, float dur)
{
    Episode e; e.d_forward = d_fwd; e.d_theta = d_th; e.duration = dur;
    e.r_forward = -0.012f * d_fwd; e.r_lateral = -0.009f * d_fwd; e.r_theta = -0.029f * d_th;
    e.pos_var = 1.6e-5f; e.theta_var = 4e-6f;
    return e;
}
} // namespace

int main()
{
    IntakeParams ip; ip.window = 128;
    CalibrationIntake in; in.configure(ip, Prior{});

    // 1. ordinary driving is admitted
    for (int i = 0; i < 40; ++i) in.offer(good(1.0f, 0.5f, 2.0f), Source::Passive, 0.027f);
    check(in.count(Verdict::Accepted) == 40, "ordinary driving admitted");

    // 2. a corrupt frame is refused, and named
    Episode bad = good(1.f, 0.f, 2.f); bad.r_forward = std::nanf("");
    check(in.offer(bad, Source::Passive, 0.027f) == Verdict::NonFinite, "NaN refused as non-finite");

    // 3. the localiser not tracking is refused -- its correction is recovery, not model error
    check(in.offer(good(1.f, 0.5f, 2.f), Source::Passive, 0.22f) == Verdict::NotTracking,
          "poor localiser fit refused as not-tracking");

    // 4. a teleport charged to odometry is refused
    check(in.offer(good(9.0f, 0.f, 0.5f), Source::Passive, 0.027f) == Verdict::Implausible,
          "kinematically impossible episode refused");

    // 5. AN INTERRUPTED MANOEUVRE IS STILL DATA. The task failed; the robot still turned.
    const int before = in.count(Verdict::Accepted);
    for (int i = 0; i < 3; ++i) in.offer(good(0.f, 0.4f, 1.0f), Source::Manoeuvre, 0.03f);
    check(in.count(Verdict::Accepted) == before + 3, "partial manoeuvre data kept, not discarded");
    check(in.accepted_from(Source::Manoeuvre) == 3, "manoeuvre provenance recorded");
    check(in.accepted_from(Source::Passive) == 40,  "passive provenance recorded separately");

    // 6. provenance must NOT change the estimate. Same episodes, different tags, same answer.
    CalibrationIntake a, b; a.configure(ip, Prior{}); b.configure(ip, Prior{});
    for (int i = 0; i < 30; ++i)
    {
        const auto e = good(1.0f + 0.05f * i, 0.3f, 2.0f);
        a.offer(e, Source::Passive,   0.03f);
        b.offer(e, Source::Manoeuvre, 0.03f);
    }
    const float da = a.estimate().value[P_K_V], db = b.estimate().value[P_K_V];
    std::printf("   passive-tagged k_v %+.6f vs manoeuvre-tagged %+.6f\n", da, db);
    check(std::abs(da - db) < 1e-9f, "provenance does not change the estimate");

    // 7. nothing arrived vs everything refused
    CalibrationIntake idle;  idle.configure(ip, Prior{});
    CalibrationIntake broken; broken.configure(ip, Prior{});
    for (int i = 0; i < 20; ++i) broken.offer(good(1.f, 0.5f, 2.f), Source::Passive, 0.5f);
    std::printf("   idle: offered %d, pool %zu | broken: offered %d, pool %zu, refused-not-tracking %d\n",
                idle.offered(), idle.pool(), broken.offered(), broken.pool(),
                broken.count(Verdict::NotTracking));
    check(idle.offered() == 0 and broken.offered() == 20,
          "'nothing arrived' and 'all refused' are distinguishable");

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
