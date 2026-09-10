# EXP2 — Per-segment precision (partition of c) (2026-06-11)

Goal: improve execution time while PRESERVING reliability by partitioning the single
global confidence c into one precision per motion segment, each learning from its own
outcome. c_k = Π_m[k]/(Π_m[k]+Π_s); Π_m[k] accumulates from segment k's confirmed outcome
(quality-graded) and deflates on its surprise. Segments: APPROACH, INSERT, LIFT, PLACE,
RETREAT. Commit events (close/release) stay hard. cur_seg_ set per phase so every existing
skill_c()/skilled_speed() knob becomes segment-local.

Protocol: 3 fresh rounds (Π_m=0,0,0,0,0), learn_pick_place=true, precision_reweighting=true,
sensory_precision=2.0, evidence_unit=1.0, conf_decay=0.5, fixed pick/place, round_cycles=10.
Per-rep per-segment c logged to kinova_skill_metrics.csv (round{1,2,3}_metrics.csv).

## Result — the partition does what a global c cannot
Final Π_m per segment (c = Π_m/(Π_m+2)):
            approach  insert  lift   place  retreat   misses
  round1     6.35     8.00    4.47   6.78   7.83      1 (lift)
  round2     7.46    10.00    6.23   9.22   9.50      0
  round3     6.59     9.00    3.03   8.00   8.98      1
  mean c     0.77     0.82    0.70   0.80   0.81

1. SEGMENTS DIVERGE, REPRODUCIBLY. Ordering insert ≥ retreat ≥ place > approach > lift in
   ALL 3 rounds. insert is binary-clean (q=1) → highest; lift confirms only partially (rise
   fraction) AND takes the "bottle not held" misses → lowest + most variable (3.0–6.2).
2. CREDIT + CAUTION ARE LOCALIZED. In round1 a lift surprise deflated ONLY the lift
   (c 0.64→0.55) while approach/insert/place/retreat kept climbing. A global c would have
   deflated the whole skill on that one miss. This is the decoupling that lets the safe legs
   keep cruising while the risky leg stays careful.
3. TIME STILL IMPROVES: cycle ~16s (novice) → 6.7s as the safe legs speed up.
4. FREE DIAGNOSTIC: the per-segment profile auto-identifies LIFT as the reliability
   bottleneck — exactly where the grasp-slip ("bottle not held") fix should go.

Files: round{1,2,3}.log, round{1,2,3}_metrics.csv.
