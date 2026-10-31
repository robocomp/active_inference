# The crossing's door model is fitted in the robot frame — the anchor's leaf from its own posterior

2026-10-31 · status: DRAFT for the user's review · door_concept (+ the controller's reading of the anchor)

## Why

The user (10-31): "to keep the model, in door_concept, while the robot is crossing, it should run the model fitting against
the lidar points in the robot's frame". The anchor `door_N@robot` is meant to be exactly that: the ONE picture of the door
the controller crosses on, owned by door_concept, fitted to live data. Two live runs showed where it is not:

- pingpong30 leg 3: the controller's own gap tracker (a second picture of the door) latched onto the open leaf's tip 1 m
  short of the door; it is now switched off (a4206b5d), so the anchor must carry everything the crossing needs.
- pingpong31: the anchor's OPENING was right (s −1.80 m, matching ground truth), but the LEAF it carried was phi 83 deg with
  the hinge 8 cm inside the casing — the room-frame door belief, frozen on a crossed door by de094f9c ("hold both"), not a
  fit to the scan. The jamb harness (test_jamb_cross p31) shows the 83 deg leaf costs 5 s with the old servo law and makes the
  Active-Inference law's P(fit) fall to 0.49.

## What door_concept does today (read 2026-10-31)

- **Opening pose (x, y, yaw):** initialised ONCE from the room frame when the cross is claimed, then predicted by the body
  twist and corrected every frame by the bpearl rays and the door mask, in the robot frame (`door_anchor_update.h`,
  `update_door_anchor`). Correct; unchanged here.
- **Leaf inside the update:** `update_door_anchor` already MARGINALISES the leaf per frame against those rays — hinge side ×
  swing side × phi over the door's range, plus a "leaf not as believed" branch (`LeafPrior`, `LeafModel`). Its posterior
  weights over the configurations are computed (`LeafModel::active_w`) but NOT returned (`AnchorEvidence` has no leaf).
- **Leaf published to the controller:** `specificworker_anchor.cpp` (~line 1135) builds the anchor's leaf hypotheses from
  door_concept's ROOM-FRAME belief (`inst->interaction.phi`, `inst->leaf.hinge/swing`), which de094f9c holds fixed on a
  face backed by a crossing.

## The change

1. **Return the leaf posterior.** `update_door_anchor` returns, in `AnchorEvidence` (or a sibling struct), the posterior over
   leaf configurations it already computed: per hinge × swing block its weight, and within it the phi mean and sigma
   (moment-matched over the phi nodes), plus the weight of the "no leaf" branch. Nothing new is estimated.
2. **Publish that posterior on the anchor.** The anchor's leaf hypotheses (`anchor_leaf_hypotheses`, the controller's guards
   and the AIF law read them) are built from (1), not from the room-frame belief.
3. **Recursive during the passage.** The `LeafPrior` of the next frame is (1)'s posterior, carried (it is rigid in the anchor
   frame), with the same "not as believed" mass so a moved leaf is still found. The room-frame belief seeds it ONCE, at the
   anchor's creation, like the opening pose.
4. **Unseen = carried, not refitted.** When no ray reaches the doorway the leaf posterior is carried unchanged with the
   anchor's predicted pose (as the pose is today).
5. **de094f9c narrows to the room-frame door.** "Hold both" keeps the crossed face's ROOM-FRAME pose and leaf from being
   refitted from the far side (its reason, pingpong13). It no longer feeds the crossing: the crossing reads (2).

## What it removes / keeps

- Removes: the room-frame leaf as an input to the crossing; with a4206b5d, the last reason for the controller to measure
  the door itself.
- Keeps: the anchor update's model, its ray budget and cost (it already marginalises the leaf; only its output changes).

## Verification (offline first)

- door_concept's own anchor tests (the shut_door / leaf cases of `door_anchor_update`): the returned posterior puts its mass on
  the configuration the synthetic scan was made with, for hinge flipped, swing flipped, phi 60/83/90 deg.
- `test_jamb_cross` p31: feed the posterior a pingpong31-like scan of the corridor stub and check the published leaf lands at
  the true 90 deg within its sigma; the AIF law with that leaf (after its P(fit)/through defect is fixed) gets THROUGH.
- Live: the controller's "cross:" status line shows the anchor's leaf (phi, hinge) beside the anchor's s, e.

## Open questions for the user

- Should the controller's guards use the full posterior (all configurations, weighted) or only the MAP? Today they take
  every hypothesis weighted by its probability; the proposal keeps that.
- The AIF law's defect (with any leaf it never reports through, P(fit) 0.6 at the TRUE leaf) is separate and should be fixed
  in the law; this change only makes the leaf it is given correct.
