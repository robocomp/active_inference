# The Ricoh wall-ceiling line as a room-layout measurement — an assessment

Written 2026-09-03 against `ROBOT_GEOMETRY.md`, `src/wall_map.h`, `WALL_SLAM_REVIEW_2026-09-02.md`
and `LITERATURE_MANHATTAN_LAYOUT.md`. Every paper and number was checked against a primary source;
every repo claim carries a file:line.

## 1. Verdict

**We already extract this line, and it already drives the pose.** Missing is (a) letting it constrain
the wall landmarks' (φ, d) rather than only the robot pose, and (b) any learned component. The line is
**not competitive with the LiDAR per column** — it loses beyond ~1–2.6 m — but a **pooled per-wall line
fit is** (≈2 cm at 5 m, from a σ we have measured). The DNN's real value is **topology, not metric**.
One blocking defect surfaced during the survey (§4).

## 2. What already exists, versus what would be new

| Piece | Where | State |
|---|---|---|
| Ricoh panorama **1920×960**, media plane domain 7, topic `rc/ricoh/rgb`, ~33 Hz | `robot_concept/p3bot.json:1300`; `ricoh_omni_dds/etc/config_webots.toml` | live |
| Subscriber factory `make_image360_subscriber_from_graph(G,"ricoh","rgb360")`, domain/topic from the descriptor | `common/media_transport/media_transport.h:528` | live |
| 360 camera model in cortex; column **W/2 = 960 is forward**, column 0 the seam | `cortex/api/dsr_camera_api.cpp:103-171` | live |
| **`WallCeiling` contour** from the current polygon at `room_height`; 0.10 m arc-length sampling; 1-D edge search along the normal; per-frame σ_i measured (Immerkaer) | `room_concept/src/image_edge_source.cpp:108-114,451` | live |
| Common-mode nuisances **[pitch, height, yaw, image/LiDAR dt, contour offset]**, marginalised per segment (Woodbury) | `image_edge_types.h:76-84`; `image_edge_accumulate.h:24-43` | live |
| Ceiling **triple points**; `useWallCeiling=true`, `camera="ricoh"`, `drive=true` — **corners only**, no dense contour in the loss | `room_concept/etc/config.toml:1400,1390,1442` | live |
| onnxruntime C++, TensorRT FP16 → CUDA, **RTX 5070 Ti 16 GB**; YOLO26-seg/pose/sem, SAM2-tiny, DINOv2 448×224 on the 2:1 panorama | `retina/src/onnx_providers.cpp:104-146` | live |
| Ricoh worker budget: stages **14.0 ms mean** in a 50 ms cycle (20.0 Hz measured) ⇒ **~36 ms headroom** | `retina/etc/viewer_perf_ricoh_worker.csv` | measured |
| ADE20K net already loaded, with `wall`/`floor`/`ceiling` classes; used offline by `DatasetEnricher` (σ_ceiling = 0.06 m), **absent from the live stage's `accepted_labels`** | `retina/src/depth_enrichment.cpp:61-72`; `retina/etc/config.toml:158` | half-live |
| Classical line detection (Hough / LSD / Canny) | — | **none, anywhere** |
| Anything that **measures** `room_height` | sole writer `room_scene_graph.cpp:547`, value from config | **nothing** |

So the image path corrects the **pose**, never the map — `wall_map` has never seen a camera
measurement — and a 512×1024 panoramic network is a model file and a session, not a project.

## 3. The DNN family, verified

All predict, from one gravity-aligned equirectangular image, the ceiling-wall and floor-wall boundary
**per image column**.

| Method | Input | World model | Output | Reported | Licence / weights |
|---|---|---|---|---|---|
| **HorizonNet** (Sun et al., CVPR 2019, arXiv 1901.03861) | 3×512×1024, ResNet-50 | Manhattan post-proc; `--force_cuboid` optional | **3 × 1-D vectors: ceiling-wall row, floor-wall row, corner existence, per column** | 82.17 / 84.23 % 3D IoU (PanoContext); 79.79 / 83.51 % (Stanford2D3D); post-proc "only 12 ms" | **MIT** code; 4 checkpoints (PanoContext+S2D3D, Structured3D 18 362 img, Matterport, ZInD 20 077 img); no official ONNX, but a third-party ONNX of the Structured3D weights runs under onnxruntime-web |
| **LGT-Net** (Jiang et al., CVPR 2022, arXiv 2203.01824) | 512×1024×3, SWG-Transformer | Atlanta | **horizon-depth per column + one room height** — the range, directly | 2D/3D IoU **83.52 / 81.11** (MatterportLayout), **91.77 / 89.95** (ZInD), 85.16 % (PanoContext) | **MIT**; per-dataset checkpoints; no ONNX; no FPS reported |
| **HoHoNet** (Sun et al., CVPR 2021, arXiv 2011.11498) | 512×1024 | latent horizontal features | layout + depth + semantics | 82.32 / 79.88 2D/3D IoU (MatterportLayout) | **52 FPS (ResNet-50) / 110 FPS (ResNet-34), RTX 2080 Ti** — the only clean speed number in the family |

Weaker for us, verified: **DuLa-Net** (CVPR 2019) returns a floor plan + height — a *shape*, not a
per-column boundary; **AtlantaNet** (ECCV 2020) drops Manhattan via planes above and below the camera;
**LED2-Net** (CVPR 2021 oral) predicts depth on the horizon line. Newer, with public code: DOPNet
(CVPR 2023), Seg2Reg (CVPR 2024), uLayout (WACV 2025), PanoTPS-Net (Oct 2025, arXiv 2510.11992).

**Pick HorizonNet-Structured3D or LGT-Net.** HorizonNet emits the ceiling row per column directly, and
its Structured3D checkpoint is the cleanest licence — the Matterport3D and ZInD checkpoints inherit
those datasets' research-only terms though the code is MIT. LGT-Net emits horizon-depth, skipping §4's
geometry at the cost of trusting its internal height model.

## 4. Geometry, error budget, and a blocking defect

`body ← ricoh` = +1.275 m, `Shadow ← body` = +0.0325 m, `Shadow` at the floor ⇒ **h_cam = 1.3075 m**.
With Δh = h_ceil − h_cam and α the elevation of the ceiling-wall pixel above the horizon:

    r = Δh / tan α ,   dr/dα = −(Δh² + r²)/Δh ≈ −r²/Δh  (r ≫ Δh) ,   dr/dh_ceil = r/Δh = cot α.

The first is the relation `room_concept/etc/config.toml:1362-1364` already states for the ZED floor
junction ("δd = θ_pitch·d²/h, 1 degree of pitch error is 14 cm at 3 m"): the ceiling line is that
measurement inverted, and inherits its pathology.

**Apartamento (RoomHeight = 3.0 m, Δh = 1.6925 m):**

| r | α | dr/dα | σ_α = 0.2° | 0.5° | 1.0° | dr/dh_ceil | δh = 2 cm | 5 cm |
|---|---|---|---|---|---|---|---|---|
| 2 m | 40.2° | 0.071 m/° | 0.014 m | 0.035 m | 0.071 m | 1.18 | 0.024 m | 0.059 m |
| 5 m | 18.7° | 0.287 m/° | 0.057 m | 0.144 m | 0.287 m | 2.95 | 0.059 m | 0.148 m |
| 8 m | 11.9° | 0.690 m/° | 0.138 m | 0.345 m | 0.690 m | 4.73 | 0.095 m | 0.236 m |

**A 2.4 m room (waf / beta, Δh = 1.0925 m)** — materially worse:

| r | α | dr/dα | 0.2° | 0.5° | 1.0° | dr/dh_ceil | 2 cm | 5 cm |
|---|---|---|---|---|---|---|---|---|
| 2 m | 28.6° | 0.083 m/° | 0.017 | 0.041 | 0.083 | 1.83 | 0.037 | 0.092 |
| 5 m | 12.3° | 0.418 m/° | 0.084 | 0.209 | 0.418 | 4.58 | 0.092 | 0.229 |
| 8 m | 7.8° | 1.042 m/° | 0.208 | 0.521 | 1.042 | 7.32 | 0.146 | 0.366 |

**Crossover: the range below which the ceiling line beats the LiDAR.** One panorama row is 0.1875°, so
σ_α = 0.2° is a **one-pixel** error — optimistic for any extractor.

| σ_α | vs LiDAR 2 cm, 3.0 m ceiling | vs 2 cm, 2.4 m | vs `obs_sigma` 5 cm, 3.0 m | vs 5 cm, 2.4 m |
|---|---|---|---|---|
| 0.2° | r < 2.6 m | r < 2.25 m | r < 4.6 m | r < 3.8 m |
| 0.5° | r < 1.0 m | r < 1.14 m | r < 2.6 m | r < 2.25 m |
| 1.0° | never | r < 0.24 m | r < 1.4 m | r < 1.4 m |

**Per column the line is an order worse than the LiDAR everywhere the robot looks. Pooled, it is not.**
Our one real measurement — P3Bot ceiling triple points, `room_concept/etc/config.toml:1444-1447` — is
**σ_v = 0.344 px** for the ceiling contour's fitted offset (floor: 0.456). At 960 rows that is 0.0645°,
giving σ_r = **1.9 cm at 5 m, 4.5 cm at 8 m**. That distinction should drive the design.

### ⚠ Blocking defect: the projection model is mis-declared in simulation

Webots renders the two Ricoh cameras with **`projection "cylindrical"`**
(`webots-p3bot/protos/P3Bot.proto:102,111`) — row linear in `tan(elevation)`. Both graph files declare
**`cam_projection = "equirectangular"`** (`robot_concept/shadow.json:520`, `p3bot.json:1320`), so
`CameraAPI` uses the `asin` map. Azimuth is identical in both — which is why the corner channel works
and nobody noticed — but elevation is not:

| r | true elevation | rendered row (cylindrical) | read as equirect | range error |
|---|---|---|---|---|
| 2 m | 40.24° | 221.4 | 48.49° | **+0.58 m** |
| 3 m | 29.43° | 307.6 | 32.32° | +0.35 m |
| 5 m | 18.70° | 376.6 | 19.39° | +0.20 m |
| 8 m | 11.95° | 415.4 | 12.12° | +0.12 m |

Larger than every other term at close range, and a bookkeeping error, not a sensor limit. It is
**simulation-only** — the real Theta's gstreamer path is genuinely equirectangular — which is worse, not
better: the bench would validate a channel that behaves differently on hardware. Settle it first.

## 5. Failure modes in this flat

A ratio to a ceiling height assumed flat and known. It is neither.

| Failure | Effect on r | Magnitude |
|---|---|---|
| **h_ceil is a hand-entered constant** — `config.toml:1719` "the LiDAR cannot reach 3 m, so it is **stated** here rather than detected"; beta's 2.4 m flagged "⚠ NOT MEASURED" | pure **scale**, δr/r = δh/Δh — adds no noise, resizes the room, the one error a Manhattan estimator cannot detect internally | 5 cm ⇒ 3.0 % (3.0 m ceiling), 4.6 % (2.4 m) |
| **Beam / soffit** — local drop of the ceiling | biases *toward* the robot by (r/Δh)·δh; spatially coherent, so it looks like a real wall — the worst mode | 20 cm soffit at 5 m ⇒ wall **0.59 m too near** |
| **Extractor hood** (`hood_concept` exists) | per column the strongest near-horizontal discontinuity in the kitchen is the hood, not the wall | classical extractor takes it; a layout DNN ignores it — the clearest argument for the DNN |
| **Sloped ceiling** | r = Δh/tan α no longer separable; a per-column h_ceil would be needed | hard stop; absent in the apartamento |
| **Curtain / wardrobe / bookcase to the ceiling** | boundary is the object's top, not the wall | the DNN beats both a geometric extractor and the LiDAR — the nets output the **envelope** behind clutter |

## 6. Calibration sensitivity

A 1° boresight error at 5 m costs **0.29 m** (3.0 m ceiling) or **0.42 m** (2.4 m) — 10–20× the LiDAR.
But *which* degree matters is not what our history suggests. **Yaw is harmless**: r does not depend on
azimuth, so the measured +0.814° ZED yaw would rotate the room, not resize it — and yaw is already a
modelled nuisance (`image_edge_types.h:76-84`, column [2]). **Pitch and roll are the killers, and worse
than a bias**: a tilt δ gives α' = α + δ·cos(ψ − ψ₀), so the range error is (r²/Δh)·δ·cos(ψ − ψ₀) —
sinusoidal in azimuth, quadratic in range. It does not translate the polygon; it stretches one side and
compresses the opposite, destroying the rectilinearity `enforce_manhattan` holds at 0.00° internal tilt.

Manageable, but only as a hard ~0.1° gravity-alignment prerequisite — which every one of these networks
assumes anyway. The procedure is half built: `[ImageEdge] shadow = true` logs `bias_const_px` (⇒ pitch)
and `bias_invd_m` (⇒ height) against a LiDAR-only pose that cannot depend on camera pitch. Per
`mount-solve-ignores-vertex-clustering`, report that fit's error per **azimuth cluster**, never pooled.

## 7. Where it should enter the estimator

**Not as per-column range factors into `WallPointFactor`.** A well-observed wall is already fitted to
its LiDAR data to 1–2 mm over 12k–151k points; a 6–35 cm channel cannot improve it, and every failure in
§5 is a *coherent bias*, not noise, so it would only move a converged wall. Three admissible entries:

1. **Corner bearings — already built.** Ceiling triple points are range-free, hence immune to the Δh
   scale and the soffit bias, and exploit the panorama's real strength (bearings over 359°, which
   collapsed corr(x, θ) from 0.98 to ~3 on P3Bot). Keep it.
2. **A per-wall (φ, d) line observation, occlusion fallback only.** Fit one polygon edge's ceiling
   samples to a single (φ, d) with §4's pooled covariance, admitted **only for order walls with no
   associated LiDAR points this window**. A global Δh scale and a two-parameter gravity tilt become
   states integrated out, never constants — per CLAUDE.md a covariance growing as r²/Δh, not a range
   gate, and `image_edge_accumulate.h` already has the Woodbury pattern for it.
3. **A DNN layout as a topology proposal, touching no factor.** Offer the predicted closed rectilinear
   polygon to `re_derive` as a candidate cycle, priced by the existing `edge_code_nats` and judged by
   the existing grid / forward-beam evidence. It cannot corrupt the metric estimate because it never
   becomes a measurement.

**Is the DNN needed?** For 1 and 2, no. Our extractor works *because it is predicted*: it searches ±L px
around where the polygon puts the line, so clutter degrades the weight rather than capturing the search.
An *unpredicted* classical extraction fails on white-on-white junctions and locks onto the hood — and we
have no line detector to build one from. Classical suffices for **tracking a polygon we have**; only a
DNN yields a layout **from nothing**, use 3. Try a cheap middle option first: the **ADE20K net is already
loaded and has a `ceiling` class**, and the lower boundary of its ceiling region per column *is* this
line (`DatasetEnricher` assigns it σ = 0.06 m) — a string in `accepted_labels`, not a dependency. The
DNN's costs are a domain gap (Webots renders are flat-shaded; Structured3D, itself synthetic, is the
nearest training domain), a dependency, and checkpoint licences looser than the MIT code.

## 8. What it buys that the LiDAR cannot

1. **L-shaped rooms and topology — the strongest case.** 19 of 50 population rooms have a reflex corner
   and average 0.859 IoU against 0.942 for rectangles; at saturation the incumbent adoption judge adopted
   **nothing on 12 of 12 seeds**, so the escape hatch never fires. A layout net emits a complete
   non-cuboid Manhattan polygon from a *single* view: the global proposal the greedy splicer cannot
   reach. The ceiling for a perfect selector was measured at +0.020 mean IoU and +0.077 on the floor; an
   independent proposal raises that ceiling rather than competing for it.
2. **Walls behind floor-to-ceiling occluders.** Narrower than it sounds — the high band already clears
   normal furniture — but wardrobes, bookcases and curtains block it, and a layout net outputs the
   envelope behind them.
3. **A measurement of `room_height`**, which nothing does today and which two of three scenarios admit
   is unmeasured.

**It does not buy spurs.** Free-standing interior walls are absent from every layout dataset's ground
truth — these nets output the *envelope*, and a spur is inside it. The review's diagnosis stands:
detection is grid-limited, measurement is beam-limited. Do not expect the 13 % spur rate to move.

## 9. Staged plan, with a one-day falsifier

**Stage 0 — one day, no DNN, no download, no dependency.** First settle §4: make the graph's
`cam_projection` agree with the renderer. Then park the robot where the polygon has converged (per-wall
LiDAR residual 1–2 mm) and log, per `WallCeiling` sample, azimuth ψ, model range r, predicted and
measured row; fit `Δrow(ψ) = a + b·cos ψ + c·sin ψ`. **Pre-registered falsifiers:**

- scatter σ_v > 2.7 px at 960 rows (0.5°) ⇒ ranging is dead beyond ~1 m and only uses 1 and 3 survive
  (P3Bot's pooled 0.344 px suggests it may pass);
- √(b² + c²) > 0.5°, or unstable across runs ⇒ gravity alignment is not controllable; ranging dead;
- `a` gives h_ceil = h_cam + r·tan α_meas — disagreement with the stated 3.0 m by >5 cm means a ≥3 %
  scale error is already present, worth knowing regardless.

**Stage 1** (if 0 passes): add `ceiling`/`wall` to the live ADE20K `accepted_labels` for a second free
extractor; then build entry point 2, graded on the bench with LiDAR returns synthetically removed from
one wall. Falsifier: it must not move a well-observed wall by more than that wall's own σ.

**Stage 2** (independent): run HorizonNet's Structured3D checkpoint offline on recorded frames and
compare its ceiling row against the polygon-predicted row, on the real flat *and* on a Webots render.
Falsifier: median |Δrow| > 5 px, or failure on the render — the domain gap then dominates.

**Stage 3** (only if 2 passes): ONNX export, a retina session beside the existing YOLO26 / SAM2 / DINOv2
ones (36 ms headroom, 5070 Ti), and the layout wired into `re_derive` as a priced candidate cycle.
Nothing enters the solver.
