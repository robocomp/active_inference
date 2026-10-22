# Blind-spot map — a learned P(detect | object, viewpoint)

Status: design + first implementation in table_concept (2026-10-21). Module: `blind_spot_map.h`.

## 1. The problem

`rc::detect::p_detect` (detectability.h) answers "could YOLO have seen this object from here?" from
GEOMETRY alone: how much of the frame the object fills, whether it is in the frustum, whether it is
centred. It gives the same answer for every table at every angle.

The detector does not behave like that. On 2026-10-21 the square `table_1` went **1114 cycles with no
YOLO mask** from one viewpoint while the image proved it was there (contour channel: RGB edge excess
+13, depth verdict +0.77). The geometric envelope scored every one of those misses at p_detect 0.64 —
a confident look that saw nothing. On 2026-10-19 the same table was deleted in ~6 cycles of a turn for
the same reason. The detector has **blind spots**: viewpoints where it fails on an object that is
present and well framed — edge-on tables, top-down views, a texture or lighting it dislikes.

## 2. What the map learns

For each object, the probability that YOLO fires given that the object is present and visible, as a
function of where the camera is **relative to the object**:

| coordinate | meaning | why |
|---|---|---|
| `az`  | bearing of the camera around the object, in the object's own frame, folded by the object's symmetry | edge-on vs face-on |
| `el`  | elevation of the camera above the object's top | top-down vs level |
| `lr`  | log of camera→object distance | far/near (the envelope's own axis, refined) |

The geometric envelope stays — it is the **prior**. The map corrects it with what the robot actually
saw. Where nothing has been seen, the answer is exactly the envelope, so the map can only change
behaviour where it has evidence.

## 3. Where the labels come from

From the **contour channel** (`common/contour_edge`), never from the existence belief.

A frame is a labelled sample when a mask frame arrived (YOLO had its chance) AND the contour channel
measured the object this cycle. The label is `hit = 1` if a mask was assigned to the object, else 0.
The sample's weight is how sure the IMAGE is that the object is there and in view:
`w = max(0, tanh(contour verdict)) × visible fraction`. A contour that refutes or abstains contributes
nothing — there is then no evidence the object was there to be detected.

★**No circularity.** Labelling from the existence belief would teach the map "objects we believe in get
detected" and it would then shield phantoms. The contour verdict is computed from pixels and depth,
independently of the YOLO masks.

## 4. Counting: by viewpoint, not by frame

The 1114 misses were one pose seen 1114 times — the same image again. Counted per frame they would make
the map certain after a single visit (the *fractional counts manufacture confidence* mistake). So
frames are pooled into **visits**: a visit stays open while the viewpoint stays within half a kernel
length of where it started, and accumulates the hit rate. It is worth at most ONE observation:
`weight = 1 − exp(−Σw)` (≈1 after a few confident frames, never more). When the viewpoint moves on, the
visit closes and becomes one stored sample. Open visits are queried too, so a long stare counts at once.

## 5. The model — kernel Beta-Bernoulli, two levels

Each sample `i` = (viewpoint vᵢ, hit rate rᵢ, weight wᵢ, owner). A Gaussian kernel
`k(v, vᵢ) = exp(−½ Σ_d ((v_d − v_{i,d}) / ℓ_d)²)` spreads it to nearby viewpoints. No bins, no edges.

- **Class level** (all objects of this kind, persisted across runs): pseudo-counts from every sample
  NOT owned by the querying object:
  `n_c = Σ k wᵢ`, `p_c = (κ₀·p_env + Σ k wᵢ rᵢ) / (κ₀ + n_c)`.
- **Instance level** (this object, from its own samples), with the class answer as its prior:
  `κ = κ₀ + β·n_c`, `p = (κ·p_c + Σ k wⱼ rⱼ) / (κ + n_own)`.

The instance's own samples are excluded from the class term so they are not counted twice. `β ∈ (0,1]`
is how much other objects of the class predict this one (between-instance similarity).
`p_env` is the geometric prior: `rc::detect::p_detect(...)` envelope term × the score grade `p_clear`.

The map's answer REPLACES that prior in p_detect; the frustum/occlusion fraction and the centring term
still multiply it, since the labels are weighted by visibility.

### Model constants (flagged — measurable, not thresholds)

| constant | default | meaning |
|---|---|---|
| `ℓ_az` | 0.30 rad | how far around the object a blind spot extends |
| `ℓ_el` | 0.12 rad | same for elevation |
| `ℓ_lr` | 0.25 | same for log range (~28 % in distance) |
| `κ₀`   | 1.0 | strength of the geometric prior, in observations |
| `β`    | 0.5 | how much the class predicts an individual |

All of them are fittable from the logged samples (held-out likelihood of visit outcomes).

## 6. Where it is used

1. **Removal** (table_concept, now): absence is weighted by the map's P(detect). A miss inside a known
   blind spot is nearly no evidence; this protects the object even where the contour channel cannot
   measure (clutter, darkness, depth holes).
2. **Where to look** (epistemic planner, NEXT): the NBV must score candidate viewpoints with the same
   map, so the robot drives to views its own removal test trusts — the one-envelope-both-directions rule
   detectability.h already states. Not wired yet; until it is, the planner may still pick a blind spot.

## 7. Persistence

The class map is saved to `etc/blind_spot_map_<class>.csv` on a graceful exit and loaded at start
(owner = 0 on load: all loaded samples count as "other objects"). Written with
`imbue(std::locale::classic())`, read with `std::from_chars` — see the locale rule in CLAUDE.md.
Per-instance samples live in the same file under their owner id while the agent runs; a later step can
move an object's own samples into its view-atlas archive so a re-identified object keeps its history.

## 8. Rollout

1. table_concept (it has the contour labels): removal side, logging `bsm_p`, `bsm_n_c`, `bsm_n_own`. ← now
2. Check against the 2026-10-21 log: table_1's blind spot must appear as a low-p region.
3. Epistemic planner reads the map.
4. chair / refrigerator / door (they already have contour channels).

## 9. Risks

- **A wrong contour makes a wrong label.** A contour that confirms over a textured background would
  teach "YOLO misses tables here". Mitigated by the weight (tanh of the verdict, the depth half refutes
  blank floor at −1.0), and bounded: the map can only lower p_detect to slow removal, the same
  direction the contour channel already pushes.
- **Environment drift.** Lighting changes between runs. No decay yet; if loaded class maps start
  disagreeing with fresh visits, add a per-run discount.
