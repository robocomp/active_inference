# SDD ledger — plan: docs/superpowers/plans/2026-10-31-perception-cards.md
Spec: docs/superpowers/specs/2026-10-31-perception-cards-design.md (approved 2026-10-31). Worktree active_inference.worktrees/perception-cards, branch feat/perception-cards, from feat/sensor-intrinsics 9afcacda; docs commit 5d1a02eb.

## Preflight scan
| pair / task | produces → consumes | finding |
|---|---|---|
| T1→T2 | kTermNames (card.h) → terms.h one fn per name | ok (plan fixed: list lives in card.h) |
| T2→T3 | terms.h → evaluator; T3 adds term_motion to terms.h | ok, sequential |
| T3→T4→T5 | evaluator.h types (World, Hypothesis, Viewpoint, ChannelResult, Result, Limit) | ok, same file, sequential |
| T2→T8 | moved door/cabinet terms → their cards | ok |
| T6→T7,T8,T10 | write_card_file → agents | ok |
| T6b | needs cortex perception_card_att (user reinstall) | see Ruling R3 |
| T5 test (d) | timing <200 ms | Ruling R2 |
| T9 test | "within 0.15 of old" bound | test-only guard, not a model threshold; ok |
| T12 | merges feat/perception-cards into feat/vlm-missions | Ruling R1 |
| each task self-consistent | files created == files touched later | ok |

Ruling R1: T12 merges feat/perception-cards into feat/vlm-missions locally (both our unpushed branches, never main) — the plan mandates it and chat_agent's tool infra lives only there — cost if wrong: one local merge to undo.
Ruling R2: perception_card tests are built with -O2 so the 200 ms where_to_stand budget measures real cost — a debug build would fail on speed, not logic — cost if wrong: a slow path hidden until live.
Ruling R3: Task 0 asked to the user at start; if cortex is not reinstalled when T6b is reached, T6b is parked after T12 and done when it lands — cost if wrong: cards only in files until then (chat reads files anyway).
Task 1: dispatched (BASE 5d1a02eb, implementer ac806f5078a8c41ab, sonnet)
Task 0: done — perception_card REGISTER_TYPE added to cortex source dsr_attr_name.h (after media_ice_port); user reinstalling
Task 1: minor (deferred): card.h two namespace blocks + unused flat alias; stranded locale comment in media_transport.cpp; flat_json.h missing <utility>/<cstddef>; card.h:500 count cap 4096 unjustified; bad "version" silently defaults; duplicate subtype/JSON keys overwrite silently; ';'-joined lists unescaped; q() escapes few control chars; extra ../media_transport include dir.
Task 1: complete (commits 5d1a02eb..b514cb42, review clean)
Task 2: dispatched (BASE b514cb42, implementer a004f1cd5d801f627, sonnet)
Task 2: minor (deferred → carried into Task 3): eval_term returns 1.0 for unknown names (must signal); eval_term leaf_obliquity inline clamp instead of a term_ fn; mixed include styles; envelope default-param fallback untested.
Task 2: complete (commits b514cb42..ed85a854, review clean). Note: TermSpec moved to term_spec.h (door core forbids media_transport includes).
Task 3: dispatched (BASE ed85a854, implementer a8f2399ac32998d9a, sonnet; carries T2 minors: term_motion + eval_term rejects unknown names)
Task 3: review 1 — spec ❌: mass not coverage-only; p>=0.5 cut on Limit hides occluder (threshold). GH nodes verified standard-normal (kGHx ±2.857/±1.356/0, weights sum 1) by controller.
Ruling R4: SensorModel moves to DDS-free common/media_transport/sensor_model.h so evaluator.h readers don't link FastDDS — spec says the module is header-only Eigen+std and chat_agent/agents read it — cost if wrong: one extra header.
Task 3: fix round 1 dispatched (resume a8f2399ac32998d9a; FIX_BASE f62a3d82)
Task 3: fix round 1/5 (7 addressed, 0 open; commits f62a3d82..02261692)
Task 3: minor (deferred): fmin non-finite with finite fmax not guarded; FP-noise limit with tiny gain (consumer filters); eval_term("motion") returns metres not a factor; std::abort on unknown name; redundant occluders → empty occluder (leave-one-out limit).
Task 3: complete (commits ed85a854..02261692, review clean after 1 fix round)
Task 4: dispatched (BASE 02261692, implementer aed739a78081b7856, sonnet)
Task 4: review 1 — Important: occluder block duplicated from camera. 
Ruling R5: LiDAR range factor is common-mode (all returns of a target share r) → multiplies P, not λ — per-return application gave p≈0.93 inside the dead zone; CLAUDE.md's common-mode principle — cost if wrong: slightly pessimistic P near range edges.
Task 4: minor (deferred): room_elevation 360-step trig loop per ring (perf; check in Task 5 budget); ring crossing only at target-centre range (crude for deep/small objects).
Task 4: fix round 1 dispatched (resume aed739a78081b7856; FIX_BASE 349c238d)
