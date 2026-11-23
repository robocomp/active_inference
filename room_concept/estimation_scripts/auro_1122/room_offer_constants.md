# Room offers A and B: constants, registered prediction, "delivered" (2026-11-23, before the live tour)

Branch: feat/room-offer-info.
- A: the room-building offer in nats (b1158bc9).
- B: change information instead of the neglect drive (71c5e903, revised here after auro's referee notes).

Labels follow the objects' definitions:
- **class knowledge**: a stated prior about homes, learnt where data allow;
- **sensor**: a property of the sensor or test, a prior and learnt;
- **numerical**: a discretisation or support that does not change the model;
- **patch**: a tuned constant with no model behind it.

## 1. Constants

### Introduced by A (room-building offer)
| constant | value | label | learnt? |
|---|---|---|---|
| `StartupProto.OccPriorA/B` | Beta(1,1) | class knowledge (uninformative) | yes: occupied/free cells of each finished room, discounted by cell²/ℓ² |
| `StartupProto.DoorPriorA/B` | Beta(1,1) | class knowledge (uninformative) | yes: doors / door-width segments of each finished room |
| `StartupProto.EllPriorM` | 0.3 m | class knowledge (occupancy correlation length) | yes: raster autocorrelation of each finished room |
| `StartupProto.EllPriorN` | 1 room | prior strength | n/a |
| countable band | 0.40 m | pre-existing: the outline snap's reach | no |
| autocorrelation threshold | 1/e | definition of a correlation length | n/a |
| max lag | 3 m | numerical | n/a |

### Introduced by B (known-room offer, `ChangeBelief = true`)
| constant | value | label | learnt? |
|---|---|---|---|
| `ChangeHazardMean` | 3e-6 /m²/s (about 5 changes per 20 m² per day) | class knowledge | yes: joint posterior |
| `ChangeHazardShape` | 1 | prior strength | n/a |
| `ChangeDetect` d | 0.85, strength 10 tests | sensor | **yes**: joint posterior (λ, d, f). Learnable only from real changes; in a static world it stays at its prior |
| `ChangeFalseAlarm` f | 0.05, strength 20 tests | sensor | **yes**: joint posterior; short-age re-observations pin it |
| episode state = majority of the episode's sweeps | ½ | MAP of a binary state | n/a |
| grid supports | λ 1e-10…1e-2, d 0.30…0.99, f 1e-3…0.4 | numerical | n/a |
| change cell | 0.5 m (`IorCellSize`) | numerical (pre-existing grid size; information is counted per ℓ² region) | n/a |
| visibility range | `FimMaxRange` 10 m | sensor (LiDAR range), pre-existing | no |

- **Removed by the revision**: the one-day prior age for unseen cells (auro note 2).
- A cell with no observation on record is now valued as unknown STATE: n·h(P(occ)) with A's learnt occupancy prior.
- With a record (this run, or the persisted `etc/room_change_memory.txt` from earlier runs), its real age is used.

### Patch constants in the room's offer, ON vs OFF
| | ChangeBelief OFF (today) | ChangeBelief ON |
|---|---|---|
| neglect weight `WIorDrive` 0.5 | patch | gone |
| neglect time constant `IorDecayTime` 120 s | patch | gone |
| never-visited sentinel age (about 11 days → 4.49 nats) | patch | gone |
| IoR suppressor exponent `WIor` | patch | gone |
| path-interest blend `WPathInterest` 0.7 | patch | gone |
| path marking radius `IorPathRadius` 1 m | patch | gone (the visit grid only draws the overlay) |
| pose-prior forgetting `BeliefForgetTime` 90 s | patch | **stays**: it inflates the pose prior of the pose-information term. Not touched by A/B; flagged |
| controller `AffordanceRoomGainScale` 0.35 | patch | to be removed if C0b passes at 1.0 |
| **count** | **8** | **1** (+1 until the scale goes) |

- `WTravelCost` stays in both. It is the room's own ranking cost of travel and is not part of the advertised gain.

## 2. Registered prediction for the live tour

**Setup**:
- one furnished warm tour, 4 legs, static Webots world;
- `AffordanceRoomGainScale = 1.0`;
- branch build with `ChangeBelief = true`;
- empty change memory, so this is the first run of the belief.

**Predictions**:
- **P1, λ.** It falls from 3.0e-6 and never rises more than 10% above it.
  - Size: the tour's exposure (Σ cell area × age over re-observations) is about 0.05–0.3 of the prior's β₀ = 3.3e5 m²·s.
  - So the fall is **5–30%** if flips are only false alarms.
- **P2, f.** It moves from 0.05 towards the observed flip rate of short-age re-observations. Predicted to end in [0.005, 0.15]. The direction is not predicted, because LiDAR flicker at cell edges is unmeasured.
- **P3, d.** It stays within ±0.02 of 0.85: no real changes, so no evidence.
- **P4, change offer.**
  - At a room's first entry it is dominated by unknown-state cells hidden from the robot (n·h(P(occ)) ≈ 7.7 nats per m² at Beta(1,1)).
  - After the room has been swept it falls **below 1 nat**.
  - On re-entries within the tour (ages of minutes) it is **below 0.5 nats per room**.
- **P5, arbitration at scale 1.0.** Once each room has been swept, object offers beat the room in **more than half** of object-vs-room contests (step 1 at 1.0: 2 of 10).
  - If not, the room's gains are reported split into pose / change / coverage from `room_offer_log.csv` (auro note 4).
- **P6, calibration of the change term.** Over all re-observation tests, Σ delivered / Σ promised is within **[0.5, 2]**.

## 3. "Delivered" for the room class (same per-visit window as the objects: arrival at the viewpoint → end of dwell)
- **Change tests** (`etc/room_change_tests.csv`, one row per test):
  - Promised = I(p_cell; d̄, f̄).
  - Delivered = KL(P(changed | observed flip or not) ‖ P(changed)). It is ≥ 0, and its expectation over the outcome is the promised I.
- **First observations** (same file, `first = 1`): the cell's unknown state is resolved. Promised = delivered = n·h(P(occ)): the LiDAR's free/occupied call is taken as certain at the raster's scale.
- **Pose information**: ½ ln det Σ_before / det Σ_after of the robot pose covariance over the window. Robot covariance is logged by room_concept.
- **Coverage information** (room building, A): unknown floor actually revealed in the window × (h(P(occ)) / ℓ²), from the free-space raster before and after.
- The room's delivered for a visit is the sum, graded against the promised offer the controller selected (`room_offer_log.csv`).
