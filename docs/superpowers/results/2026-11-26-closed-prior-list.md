# The closed list of allowed prior terms, and each class checked against it

**This list is written late.** Plan revision 2 (Task G1x) required it to be fixed *before K1*. The Fable order review proposed
it on 11-24 (`2026-11-24-fable-order-review.md`, line 96, and again in §4 risk 2: "decide the closed list ... before 10c"). It
was never fixed. K1 (fridge, hood), K2a/K2a-L (toilet parts), K2b (chair), K2c (table) and K3 (standpoint oracle) were all
built and committed without it (d258657a … cf65a471). It is written now, on 2026-11-26, at BASE cf65a471. So it could not
have steered any of those tasks. It is a check after the fact, not a design input.

The list is copied verbatim from the proposal. No term was added or removed because of what the classes ended up using.

## 1. The list (verbatim, fable-order-review.md:96–98)

> A closed list of allowed prior terms: class prior (once), attachment (once, through the wall latent), room
> axis (once), room containment (no-cross, world knowledge), floor support. Anything else passed through
> `accumulate_extra` or `accumulate_camera_extra` must be argued in the ledger, or the recount will not move.

So five terms are allowed:
- **L1 class prior (once)**
- **L2 attachment (once, through the wall latent)**
- **L3 room axis (once)**
- **L4 room containment (no-cross, world knowledge)**
- **L5 floor support**

## 2. How the list is applied (my reading, stated so a referee can reject it)

- **Scope.** A prior term is any factor of the fit that is not a data likelihood. That covers factors on the state θ, on
  its latents, on its discrete front mode, and on its process model. Data likelihoods are not prior terms, and neither are
  the latent marginals that describe a SENSOR (the camera and LiDAR common modes, the pose chain). This holds even when a
  manifest block is called `[prior.*]`; such blocks are listed separately in §5.
- **"once"** means entered once per instance (or per wall anchor or room-axis anchor), not on every frame or every predict.
  A class-prior quantity that is entered again per frame, or used again as a cap, is not L1.
- **Read as L1:**
  - the G4 shape-spread latent's birth marginal Δ²/12;
  - the K2a-L proportion latents' birth marginal, from the class's cited range;
  - the birth position covariance (class extent − seen extent)²/12.

  Each enters once, at birth, from class knowledge. This is a reading; if the referee does not accept it, these are three
  more violations.
- **The graded path.** For nightstand, bed, toilet, refrigerator, hood and chair the graded path is the countable binary's
  persistent path (`LidarMeasuresOnGated = true`; `concept_agents.sh` runs the K1/K2b classes through
  `<cls>_concept/countable/<cls>` by default). The table is cut from part 1 (owner, 11-26). The batch runs it on its legacy
  binary, and its K2c countable config is checked as built (§4.7).
- **Where to look.** The hooks the list names are `CountableEngineModel::accumulate_camera_extra`
  (`countable_concept/src/countable_belief.cpp:768`) and `CountableEngineModel::accumulate_extra` (`:778`). Beside them:
  - the wall-latent priors, `accumulate_wall_prior` (`:852`);
  - the front-mode prior, `door_clearance_logprior` (`:1185`);
  - the engine's predict, `persistent_engine.h:1510`;
  - the engine's process model, `persistent_engine.h:110–124`.

## 3. Terms ON the list, per class (graded path)

| term | where it enters | nightstand | bed | toilet | refrigerator | hood | chair | table (countable cfg, cut) |
|---|---|---|---|---|---|---|---|---|
| L1 class prior (once) | Σ₀ = `prior_cov_diag()` and the class mean at the first update, `persistent_engine.h:449` | yes | yes | yes | yes | yes | yes | yes |
| L2 attachment (once, wall latent) | `accumulate_wall_prior`, `countable_belief.cpp:852`: stand-off N(gap − δ; μ, σ) plus parallel, entered once per physical wall | yes (0.03 ± 0.02) | yes (0.02 ± 0.02) | yes (0.03 ± 0.02) | yes (0.05 ± 0.02) | yes (0.00 ± 0.01) | yes (0.05 ± 0.05) | no (c2v refuses it) |
| L3 room axis (once) | `accumulate_room_axis` behind `axis_prior_added_`, `countable_belief.cpp:781` | 3° | 3° | 3° | 3° | 3° | off (0) | off (0) |
| L4 room containment (no-cross) | **not on the persistent path**: deleted by the F1 spec amendment (the stand-off carries it); `WallNoCrossPrecision` is read only by the legacy path | — | — | — | — | — | — | — |
| L5 floor support | the θ convention (base on the floor, or hanging below its attachment: `support`, K1 vertical span) | floor | floor | floor | floor | hangs | floor | legs |

## 4. VIOLATIONS: prior terms on the graded path that are not on the list

"Argued" means the ledger (`.superpowers/sdd/2026-11-23-shared-engine/progress.md`) gives a recorded reason for keeping the
term. The list's own rule is that an unargued term keeps the recount from moving. An argued term is still not on the list.

### 4.1 The terms

| id | term | engine site | what it does | argued in the ledger? |
|---|---|---|---|---|
| V1 | top anchor (`[prior.top]`, two-sided) | `countable_belief.cpp:609` (`accumulate_top`), called on every camera frame from `accumulate_camera_extra` `:770` | pulls H to the observed robust cloud top + margin with precision 10000 m⁻², per frame | **no** (F1 did not touch it; spec G3 registered it for removal) |
| V2 | depth-unobserved footprint share (`[prior.depth_observability]`) | `countable_belief.cpp:544` (`accumulate_footprint`, anchor = false), called per camera frame `:772` | re-enters the class footprint mean on w and h with precision `AI2DepthUnobsPrecision`·(1 − two-sided) on every frame: a class prior counted per frame, not once | **yes**: F1 item 5 ruling (progress.md:106): dropping it made the bed's joint P non-PD and floated nightstand extents in replay. The K2c diagnosis shows it pinning the table's depth (progress.md:236) |
| V3 | short-height prior (`[prior.identity]` height_min / soft, `PlausHeightPriorGain`) | `countable_belief.cpp:630` (`accumulate_short_height`), called on every update from `accumulate_extra` `:783` | one-sided pull of H up to `HeightPlausibleMin` at fixed precision, every update | **no** |
| V4 | door-clearance front-mode prior (`[prior.clearance]`, `AI2DoorClearanceGain`) | `countable_belief.cpp:1185` (`door_clearance_logprior`), used by `resolve_front` `:1210`, `:1453`, `:1473` | log-prior on the discrete front mode: the front faces away from the wall it stands against | **no** (pre-existing, not on the list; the K2c minor notes it is live for the table, inert at flush 0) |
| V5 | class prior reused as a predict cap (Σ ≤ Σ₀) | `common/ai_belief/persistent_engine.h:1510–1520` (`predict_theta_on`) | on every predict, rescales any θ row whose variance exceeds the class prior back to it: L1 applied again, as a clamp, for ever | **no** (the Fable review §2.3 says "count it"; not argued) |
| V6 | permanence prior on the process noise (`[prior.permanence]`) | `persistent_engine.h:110–124` (spike-and-slab Q around q₀ = rate·dt·scale²) | a class prior on how often and how far the object moves; it sets Q, not θ | **yes, owner-ruled**: progress.md:115 (owner 11-25: Q = class permanence prior). Not one of the five terms: the list names no history-stage prior |

### 4.2 Per class, with file:line

Config lines are where the term's value is set. Manifest lines are where it is declared. A dash means the manifest has no
such block, but the config sets the term and the binary runs it.

| class | V1 top anchor | V2 depth-unobs share | V3 short-height | V4 door clearance | V5 Σ cap | V6 permanence |
|---|---|---|---|---|---|---|
| nightstand | `nightstand_concept/etc/config.toml:103-104`; manifest `nightstand.concept.toml:50` | `config.toml:97-98`; manifest `:43` | `config.toml:112-113,116`; manifest `:94` | `config.toml:137`; manifest `:88` | engine (all) | manifest `nightstand.concept.toml:74` |
| bed | `bed_concept/etc/config.toml:104-105`; manifest `bed.concept.toml:59` | `config.toml:98-99`; manifest `:52` | `config.toml:113-114,117`; manifest `:103` | `config.toml:138`; manifest `:97` | engine | manifest `bed.concept.toml:83` |
| toilet | `toilet_concept/etc/config.toml:104-105`; manifest `toilet.concept.toml:51` | `config.toml:98-99`; manifest `:44` | `config.toml:113-114,117`; manifest `:95` | `config.toml:138`; manifest `:89` | engine | manifest `toilet.concept.toml:75` |
| refrigerator | `refrigerator_concept/etc/countable.toml:118-119`; manifest `refrigerator.concept.toml:77` | `countable.toml:112-113`; manifest `:65` | `countable.toml:127-128,131`; manifest `:104` | `countable.toml:152`; manifest `:98` | engine | manifest `refrigerator.concept.toml:121` |
| hood | `hood_concept/etc/countable.toml:118-119`; — | `countable.toml:112-113`; — | `countable.toml:127-128,131` (legacy hood value: for a hanging class it pulls the MOUNTING height); — | `countable.toml:152`; — | engine | manifest `hood.concept.toml:100` |
| chair | `chair_concept/etc/countable.toml:115-116`; — | `countable.toml:109-110`; — | `countable.toml:124-125,128` ("uncited convention"); — | `countable.toml:149`; — | engine | manifest `chair.concept.toml:222` |
| table (countable cfg, cut) | `table_concept/etc/countable.toml:120-121`; — | `countable.toml:114-115`; — | `countable.toml:129-130,133` ("uncited convention"); — | `countable.toml:154` (inert: flush posterior 0 under c2v); — | engine | manifest `table.concept.toml:259` |

**Count: six violating terms (V1–V6). Every one of them is present in all six graded classes.** For the table's countable
config the count is five live terms plus V4, which is configured but inert. Two of the six are argued in the ledger (V2 by
replay evidence, V6 by owner ruling). Four are not argued (V1, V3, V4, V5).

### 4.3 The table as it runs in the batch (legacy binary, cut from part 1)

The legacy `table_concept/etc/config.toml` is unchanged since the baseline. Its prior-like terms against the list:
- **L1:** the class size prior, `AI2PriorSizeStd`, `table_concept/etc/config.toml:93`. Whether the legacy recursive-Laplace
  path enters it once or per frame was not checked here.
- **VIOLATION:** Σ capped at the prior on predict, `AI2ClampSigmaToPrior = true`, `:106`. This is the legacy form of V5.
- **VIOLATION:** constant process noise, `AI2ProcessStdM` / `AI2ProcessStdYaw` / `AI2ProcessStdExtentM`, `:94, :95, :102`.
  It is a history-stage prior that is not on the list (and that CONCEPT_AGENT_LIFECYCLE forbids as a constant).

## 5. Manifest `[prior.*]` blocks that are not fit priors (referee's call)

These blocks are called priors in the manifests, but they do not enter the fit as prior factors on the state. Under my
reading (§2) they are outside the list's scope. If the referee reads the list as covering every manifest prior, each one is
a further violation.

| block | enters | classes (manifest line) |
|---|---|---|
| `[prior.explaining_away]` (`AI2WallExplainFrac` 0.25, σ 0.05) | the camera-point mixture: the weight of the "known wall" component in the LIKELIHOOD (`point_energy`, `countable_belief.cpp:744`) | nightstand `:112`, bed `:121`, toilet `:113`, refrigerator `:113`; hood, chair, table: config only |
| `[prior.class]` (p_label_true, fit_error_*) | existence and identity (label reliability), not the fit | nightstand `:103`, bed `:112`, toilet `:104`, chair `:230`, table `:267` |
| `[prior.identity]` aspect/size scales | the plausibility filter (existence); its height_min/soft also drive V3 | nightstand `:94`, bed `:103`, toilet `:95`, refrigerator `:104` |
| `[prior.placement]` (chair p_against_wall 0.5) | the flush posterior's prior weight, which is L2 | chair `:52` (on the list as L2) |

## 6. What this means for the recount

- The list's own rule: "Anything else passed through `accumulate_extra` or `accumulate_camera_extra` must be argued in the
  ledger, or the recount will not move."
- V1, V2 and V3 are exactly the terms those two hooks pass. V2 is argued; V1 and V3 are not. Their config rows
  (`AI2TopNoFloat*`, `AI2DepthUnobs*`, `PlausHeightPriorGain`) are KEPT in the recount
  (`2026-11-23-patch-constants-baseline.md`, "After part 1").
- The spec registered all three of them for removal (G2, G3, G4). **Those registered removals fail.**
