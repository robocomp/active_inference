# The Vygotskian machine — handoff

**Read this first.** This folder holds the design and the paper draft for the *Vygotskian machine*, worked out in a
long design conversation (2026-11-18 → 20) held on `pbustos/active_inference` and moved here so the work continues
inside this repository, under its `CLAUDE.md`. This README is the handoff: what was decided, what is in the folder,
what was verified in the code, and what to do first. Nothing in the folder is built or committed yet.

## The idea in one paragraph

A robot whose higher functions arrive from outside — from the person who lives with it and from a vision-language
model (VLM) carrying human culture — and are internalised: first as an inner voice that questions, plans and
imagines, then as priors, counts and habits that need no language. The automatic level (the Active Inference agents:
concept agents, metaconcepts, controller, exploration drive) builds the belief hierarchy and drives the robot; the
VLM is recruited only when that level is out of its depth, and imagines when it is not. Every function appears twice,
first between people, then within the individual (Vygotsky).

## Contents

| File | What |
|---|---|
| `README.md` | this handoff |
| `LANGUAGE_OVER_BELIEF_HIERARCHY.md` | **the engineering design note**: Part A culture (level-3 `function_metaconcept`, VLM-elicited priors), B sequencing over the hierarchy, C imagination, D the three roles in the three modes, E phases P0–P6, F risks, G open questions |
| `paper/vygotskian_machine.tex` | paper §1, framing: the Vygotsky ↔ machine mapping, internalisation in Active Inference, the internalist contrast, prior work (Mirolli & Parisi 2011; Colas et al. 2022), the two bitter lessons (Sutton 2019; Verbelen 2026, blog) and where the machine's novelty lies |
| `paper/vygotskian_cortex.tex` | paper §2: the CORTEX architecture as the given starting point — shared belief graph, concept agents, metaconcepts, controller + exploration drive, memories, the language agent and its `imagine` tools; table of what the machine reuses; what is missing |
| `paper/vygotskian_internalisation.tex` | paper §3: the inner voice as epistemic action (question objects, grounding constraint, calibrated proposer prior), attention modes and mind-wandering, return to reality, chain-of-thought, learning, why a language model, where its knowledge comes from, the non-Vygotskian control (DreamCoder), predictions P1–P8 |
| `paper/vygotskian_cultural_plane.tex` | paper §4: the VLM bound to the belief hierarchy in three roles — culture, sequencing, imagination — with function evidence, plan surprisal, habit prior, imagination priority, predictions L0–L6 |
| `paper/vygotskian_experiments.tex` | paper §5: experimental evidence — STRUCTURE ONLY, nothing run yet; E1–E7 in build order (elicited priors, function evidence offline, live read-only function agent, expected belief outcomes, inner voice, imagination, scaffold withdrawal + non-Vygotskian control), each tied to predictions L0–L6 / P1–P8; every number a red `\result{}` placeholder |
| `paper/figures/cortex-architecture-overview.tex` | Fig. 1: TikZ architecture (from the Autonomous Robots paper), extended with a LANGUAGE LEVEL band (outer speech solid; inner voice + cultural plane dashed = this paper), the command gate, and a dashed level-4 function agent |
| `paper/vygotskian_machine.bib` | shared bibliography (all entries written from memory — **verify before submission**) |
| `paper/vygotskian_machine_build.tex` | standalone build: `pdflatex`, `biber`, `pdflatex` ×2 |
| `paper/vygotskian_machine.pdf` | the built draft, 27 pp. (authors: Pablo Bustos, Vicente Matellán) |
| `paper/figures/full_graph.png` | original render (dark background) of Fig. 2 |
| `paper/figures/full_graph_white.png` | Fig. 2 as used: recoloured to a white background with black text (grey pixels inverted, saturated colours kept); included as `figures/full_graph_white.png` (relative to `paper/`), cropped with `trim` |

To fold the sections into a larger document: `\input` the five files in the order above, add
`\addbibresource{vygotskian_machine.bib}`, `\usepackage{array}`, `graphicx`, `tikz` (+ libraries), `adjustbox`, and the colours and `\cortex`/`\result` macros from `vygotskian_machine_build.tex`. Labels kept their original prefixes
(`sec:inner_speech:*`, `sec:lang_hierarchy:*`).

## How the design got here (short)

1. Reading `language_agent`: the model proposes, C++ decides (`CommandGate`), missions need the person's yes.
2. An **inner voice**: C++ detectors raise self-addressed questions ("what is this?"); a question is a persistent
   object with hypotheses, predictions and a budget; the LLM proposes tests, and **only perception moves a posterior**.
   It stops by expected information gain, not by talking itself out.
3. **The driving example**: while the road is predictable the mind wanders; a surprise snaps it back. Formalised as a
   demand signal (precision-weighted prediction error) with asymmetric hysteresis and three modes: focused,
   monitoring, wandering. The return to reality helps the automatic layer by deciding, directing attention, and
   teaching (habits grow; the competence envelope expands).
4. **Chain of thought** is focused inner speech; think slowly when there is time (wandering), fast when there is not.
5. **Why a VLM at all**: inference evaluates but cannot generate; the VLM generates but cannot evaluate. Generate and
   test. **Where its knowledge comes from**: culture (outer speech of others), not an inner faculty — so the machine is
   Vygotskian in origin; the internalist view (Chomsky) is the contrast, tested by a culture-free DreamCoder control.
6. **The externalist turn** — the VLM as culture extending the robot's mind; a ladder of what it provides, mapped to
   the generative model: grounding (A), sequencing (π), norms (C), affordances, causality (B), other minds,
   metacognition. Culture enters as G and C and settles into counts and habits E.
7. **Representational dynamics**: the automatic level handles uncertainty *within* a model; the VLM handles
   uncertainty *about* the model. Then the discovery that **metaconcepts already exist** (`ring_metaconcept`,
   `kitchen_metaconcept`): level 2 already imagines geometrically (slots, down-priors, cavity). Culture belongs at
   **level 3: room function**.
8. Sequencing and imagination were added over the same hierarchy; the whole became the paper and the design note,
   and took the name.

## Principles that were agreed

- **The VLM proposes, the models dispose** — for beliefs, plans and imagined answers alike.
- **Observe before steering** (the kitchen metaconcept's milestone-2 pattern): every new piece first logs its
  predictions and is scored against reality before it may influence anything.
- **The cavity rule binds the VLM too**: a function hypothesis must not confirm itself through what it caused the
  robot to look at or to birth.
- **No new thresholds** (`CLAUDE.md`); evidence against an explicit null, as the chassis does.
- **The VLM is not in any runtime belief loop**: priors are reviewed data with provenance; inference is C++.
- **Imagination is done by the robot's models** (`imagine*` tools), never answered by the VLM itself.
- Imagined content lives in an imagined store, never in the live graph or factual episodic memory.

## Facts checked in this repository during the conversation

- `language_agent/src/mission_plan.h`: `WakeQueue` and `allowed_in_wake` (wake turns may look, query, stop, propose —
  never act). `question_policy.h`: ask-or-assume by Beta counts and costs.
- `language_agent/src/imagined_world.h`, `imagine_fit.h`, `imagine_see.h`, `imagine_view.h`, `imagine_succeed.h`:
  an imagined copy of the room that cannot write the graph, and the subjunctive models — the basis of Part C.
- `language_agent/etc/config.toml`: `gemma4:26b` on the Orin for chat and planner; thinking disabled because a
  thinking model spent 3 000–6 000 tokens (40–80 s) per tool call.
- `kitchen_metaconcept/etc/config.toml` lines 56–82: `MemberClassLogOdds` are **authored, "elicited, not measured"**,
  awaiting a corpus — target of step 1 below. The live fridge read membership 0.00 until the `Tall` tier existed.
- `ring_metaconcept/SCHEMA_GENERALITY_TODO.md` §2.5: member yaw is predicted but not scored (an outward chair raises
  no surprise).
- `common/mission/mission_ir.h:157`: `ref:region`, `speed`, `pred:battery_low` listed as unsupported.
- `controller/src/exploration_drive.h` (the automatic explorer) and `room_explored_p` read by
  `controller/src/mission_explored.h`.
- `graph_recorder` exists on `main` — recorded tours for offline evaluation.

## What to do first

The agreed order (design note Part E; each step changes no behaviour until its check passes):

1. **P0 — elicit the kitchen class priors from the VLM.** A script asks P(class | kitchen unit) and P(class | not)
   for the seven `MemberClasses`, closed vocabulary, temperature 0, three paraphrases, median, provenance
   (model tag, date, prompt hash, spread). Compare with the authored `MemberClassLogOdds` and the live membership
   logs. It becomes the elicitation tool for function priors.
2. **Function evidence offline.** `function_evidence.h`, a pure header with standalone tests (pattern:
   `common/structure_evidence/`), implementing the function evidence Λ_f with the expected-but-absent term and the
   Beta prior from VLM pseudo-counts (paper §3, Eqs. 10–11). Run it over `graph_recorder` tours: do the rooms get the
   right functions, and does Λ_f leave zero as exploration completes?
3. **P1 — read-only `function_metaconcept`** on the metaconcept chassis (recipe §6), CSV only. Webots check, including
   removing the fridge.
4. **Per-step expected belief outcomes logged by the mission interpreter** (derived from each op, not written by the
   planner).
5. **The demand signal as a display only** (status bar + CSV), to tune it on real tours.
6. **Rehearsal**: `imagine_succeed` on the running mission's next legs, logged and scored — imagination's calibration.
7. Then the couplings, one at a time, each gated on the previous check.

Steps 1 and 2 are independent and need no live system: a good first session.

## Open questions (for Pablo)

From the design note, Part G, plus two practical ones:

1. One `function_metaconcept`, or "function" as an instance type on a generalised metaconcept chassis?
2. Agent id, DDS domain, `REGISTER_TYPE` lines for new attributes.
3. Is a person-confirmed room function a `human_claim` kind (`ltom_agent`) or room identity (`ltsm_agent`)?
4. Per-step expected outcomes: in the mission IR, or derived by the interpreter? (The conversation recommended the
   interpreter.)
5. P0 before the kitchen frame is allowed to steer cabinets?
6. Where the code lives: suggested `common/function_evidence/` for the header and `language_agent/tools/` (or this
   folder's `tools/`) for the elicitation script.
7. Is this folder the canonical home of the paper sections from now on? (Copies remain on `pbustos/active_inference`
   `main`, commit `7e10e30`.)

## Starting the new session

```
cd ~/robocomp/components/active_inference
claude
```
and begin with: *"Read `vygotskian_machine/README.md` and `vygotskian_machine/LANGUAGE_OVER_BELIEF_HIERARCHY.md`,
then let's do steps 1 and 2."*
