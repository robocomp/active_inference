#pragma once

#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
//  VoxelBudgetRegulator
//
//  Homeostatic controller for the voxel-grid population cap (`max_voxels`).
//  FPS is treated as the metabolic constraint: the budget is the resource we
//  spend to build a richer map, and we spend only as much as we can afford
//  while holding a target frame-rate.
//
//  Control law = AIMD (Additive-Increase / Multiplicative-Decrease), the same
//  scheme TCP uses for congestion control:
//    - FPS below the target band  → MULTIPLICATIVE decrease (fast back-off).
//    - FPS above the target band  → ADDITIVE increase (gentle upward probe).
//    - inside the band            → hold.
//  AIMD converges to a stable sawtooth around the largest budget the machine
//  can sustain at the target FPS, and backs off quickly under load spikes.
//
//  Pure / header-only so it can be unit-tested standalone (no robot stack).
// ─────────────────────────────────────────────────────────────────────────────
struct VoxelBudgetRegulator
{
    // ── tunables ──────────────────────────────────────────────────────────────
    int   floor          = 8'000;    // never starve the map below this
    int   ceiling        = 80'000;   // never exceed this even with headroom
    float target_fps     = 8.0f;     // frame-rate we want to hold
    float low_band       = 0.85f;    // act when fps < target_fps * low_band
    float high_band      = 1.15f;    // probe up when fps > target_fps * high_band
    int   increase_step  = 4'000;    // additive increase (voxels per adjust)
    float decrease_factor= 0.75f;    // multiplicative decrease

    // ── state ───────────────────────────────────────────────────────────────
    int   current        = 40'000;   // current cap (starts with headroom)

    // Feed the measured fps; returns the new cap. Call once per control tick
    // (e.g. once per FPS refresh period, not every frame).
    int update(float fps)
    {
        if (fps <= 0.0f)                       // no valid measurement yet → hold
            return current;

        if (fps < target_fps * low_band)
            current = std::max(floor, static_cast<int>(static_cast<float>(current) * decrease_factor));
        else if (fps > target_fps * high_band && current < ceiling)
            current = std::min(ceiling, current + increase_step);
        // else: inside the dead-band → hold (prevents oscillation)

        return current;
    }
};
