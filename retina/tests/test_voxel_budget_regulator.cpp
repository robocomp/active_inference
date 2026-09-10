// ─────────────────────────────────────────────────────────────────────────────
//  Standalone experiment for VoxelBudgetRegulator (no robot stack required).
//
//  Hypothesis: an AIMD controller that adjusts the voxel cap from measured FPS
//  will (1) settle the cap so FPS holds the target band, and (2) back off fast
//  when the scene gets heavier, then re-probe upward when it gets lighter.
//
//  Synthetic machine model
//  ------------------------
//  The grid is assumed to fill to its cap (scene demand >> cap), so the live
//  voxel count == cap. Per-frame compute time is modelled as:
//        t_frame = t0 + k * voxels          (k = per-voxel cost, varies with load)
//        fps     = 1 / t_frame
//  A "load spike" raises k mid-run (e.g. more objects / denser cloud), then it
//  relaxes — letting us watch back-off and recovery.
//
//  Build & run:
//    g++ -std=c++23 -O2 -I../src test_voxel_budget_regulator.cpp -o /tmp/vbr && /tmp/vbr
// ─────────────────────────────────────────────────────────────────────────────
#include "voxel_budget_regulator.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

namespace
{
    // Per-voxel cost (seconds/voxel). Tuned so that at 40k voxels fps≈8 in the
    // nominal regime: t = 0.02 + k*40000 = 0.125  ->  k ≈ 2.625e-6.
    constexpr double kT0          = 0.020;     // fixed per-frame overhead (s)
    constexpr double kNominalK    = 2.625e-6;  // nominal per-voxel cost (s)
    constexpr double kSpikeK      = 6.000e-6;  // heavy-scene per-voxel cost (s)

    double simulate_fps(int voxels, double k)
    {
        const double t_frame = kT0 + k * static_cast<double>(voxels);
        return 1.0 / t_frame;
    }
}

int main()
{
    VoxelBudgetRegulator reg;          // defaults: target 8 fps, [8k, 80k], start 40k
    reg.target_fps    = 8.0f;
    reg.floor         = 8'000;
    reg.ceiling       = 80'000;
    reg.current       = 40'000;

    constexpr int kTicks      = 60;
    constexpr int kSpikeStart = 20;    // scene gets heavy
    constexpr int kSpikeEnd   = 40;    // scene relaxes

    std::printf("tick  load    cap     fps    target  action\n");
    std::printf("----  ------  ------  -----  ------  ------\n");

    int   prev_cap   = reg.current;
    float last_fps   = 0.0f;
    int   in_band_streak = 0;

    for (int t = 0; t < kTicks; ++t)
    {
        const bool   spiking = (t >= kSpikeStart && t < kSpikeEnd);
        const double k       = spiking ? kSpikeK : kNominalK;

        // Measure FPS at the *current* cap, then let the controller react.
        const float fps = static_cast<float>(simulate_fps(reg.current, k));
        last_fps = fps;
        const int new_cap = reg.update(fps);

        const char* action = (new_cap < prev_cap) ? "DOWN"
                           : (new_cap > prev_cap) ? "UP"
                           : "hold";

        std::printf("%4d  %-6s  %6d  %5.2f  %6.1f  %s\n",
                    t, spiking ? "HEAVY" : "light",
                    reg.current, fps, reg.target_fps, action);

        // track stability in nominal regime (last 15 ticks, post-recovery)
        if (t >= 45)
        {
            const bool in_band = fps >= reg.target_fps * reg.low_band
                              && fps <= reg.target_fps * reg.high_band;
            if (in_band) ++in_band_streak;
        }

        prev_cap = new_cap;
    }

    // ── checks ────────────────────────────────────────────────────────────────
    std::printf("\n=== summary ===\n");
    std::printf("final cap   : %d voxels\n", reg.current);
    std::printf("final fps   : %.2f (target band %.2f .. %.2f)\n",
                last_fps, reg.target_fps * reg.low_band, reg.target_fps * reg.high_band);
    std::printf("in-band ticks (post-recovery, t>=45): %d / 15\n", in_band_streak);

    bool ok = true;
    if (!(last_fps >= reg.target_fps * reg.low_band - 1.0f))
    { std::printf("FAIL: final fps below band\n"); ok = false; }
    if (reg.current < reg.floor || reg.current > reg.ceiling)
    { std::printf("FAIL: cap left [floor,ceiling]\n"); ok = false; }
    if (in_band_streak < 8)
    { std::printf("FAIL: did not stabilise in band after recovery\n"); ok = false; }

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
