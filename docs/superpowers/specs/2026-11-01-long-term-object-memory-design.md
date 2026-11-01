# Long-term object memory (ltom_agent) — design

Date: 2026-11-01. Status: agreed in conversation with the user (11-01); not yet planned.
Related: `2026-11-01-instance-names-design.md` (given names), `common/view_atlas/VIEW_ATLAS.md` (DINO re-identification),
ltsm_agent (rooms and passages), episodic_agent (what happened), CONCEPT_AGENT_LIFECYCLE.md.

## 1. Intent

Each concept agent today rebuilds, in its own way, the same piece of memory: who its instances were after they
left the working memory (the live graph). Name high-water marks (only door_concept persists one), records of
deleted instances (one format per agent), `object_uid` and the atlas archive (VIEW_ATLAS.md §5), and now given
names. That is a long-term object memory scattered over six agents. This design makes it one agent.

The agent keeps a **repertory of re-identifiable objects**: one consolidated entry per physical object, built
from many sightings, and offers it back to the working memory so that a known object is recognised as itself.

## 2. Decided by the user (11-01)

- U1. A separate agent, not part of ltsm_agent.
- U2. It **observes the working memory** and captures instance information to build the repertory.
- U3. One object store. LTSM keeps rooms and passages; static furniture (bed, fridge) is not copied into LTSM.
  (The user asked; the answer agreed in conversation is §5: permanence is a learned belief, and LTSM refers to
  high-permanence objects by uid.)
- U4. Identity is decided by the concept agent's life cycle and the atlas; the memory never decides it.
- U5. Rooms will be nameable later ("this room is the bathroom"), owned by ltsm_agent — out of scope here.

## 3. Goals, non-goals, anti-goals

Goals
- G1. Every instance any concept agent ever had is in the repertory, with its identity, history and given name,
  surviving restarts of every agent.
- G2. Internal names and uids are never re-issued to a different object, across restarts.
- G3. On entering a room, the objects remembered there are offered to the working memory as priors, with a
  strength that follows how likely each is to still be there.
- G4. A concept agent creating a new instance can find the remembered candidates and resurrect one (old uid,
  internal name, given name) when the evidence says it is the same object.
- G5. language_agent can answer "where was grandma's chair yesterday" and "what is usually in the kitchen".
- G6. Relations acquired over time, learned or told, are kept as edges between object nodes (§6.4).

Non-goals
- Deciding identity (U4). The memory offers candidates and stores decisions; it does not make them.
- Storing raw images or crops (only the atlas state of VIEW_ATLAS.md §3.1).
- Events and steps (episodic_agent), room layouts and passages (ltsm_agent).
- Room naming (U5).

Anti-goals
- A birth that waits for this agent. If ltom_agent is down, concept agents run exactly as today.
- Two writers of any object node, or of identity.
- A second object store anywhere (the per-agent archives become caches or go away, §7).
- Hard cut-offs for "same object" or "still there" (CLAUDE.md: model terms, not thresholds).

## 4. Shape of the agent

- Process `ltom_agent`, a new `[Agent].id` (next free in CONCEPT_AGENT_RECIPE.md §1; 27 is believed taken by
  self_calibration — check), same layout as ltsm_agent/episodic_agent: one process, the LIVE graph on domain 0
  joined READ-mostly, and its own memory graph on a dedicated DDS domain (not 0, 2, 4 or 7), persisted to disk.
- Writes to the live graph: only its own nodes (its agent node and the recall nodes of §6.2). Never an object
  node.
- Presence: optional peer of the concept agents (they never require it). Debounced Degraded, SIGTERM only,
  `[Owns]` with its agent node and its recall-node glob, stale sweep at init and first Operating.

## 5. Permanence: one store, learned per instance

- Each entry carries **P(still where it was left)** — a Beta (or equivalent) per instance, updated every time the
  robot is in a position to see the remembered place: found there ⇒ success, expected-visible and absent ⇒
  failure, weighted by P(detect) from the agent's perception card (the same detectability the life cycle uses
  for absence), occluded ⇒ no update.
- The **prior** comes from the class, through the object-kind axes (CONCEPT_AGENT_LIFECYCLE.md §6.1): a bed,
  fridge, cabinet start high; a chair, bottle low; animate ~0. Builder fits the priors from logged tours where
  possible; first values flagged as guesses.
- **LTSM refers, it does not copy:** ltsm_agent's room record gets a list of `object_uid`s of the objects
  whose permanence is high, as landmarks for room recognition and relocalisation (and to explain tall furniture
  that would otherwise read as wall — memory note "tall furniture = wall to geometry"). The object data itself
  lives only here. How strongly ltsm uses a landmark is weighted by its permanence, not gated.

## 6. The two paths

### 6.1 Write path: observe and consolidate
- Polls the live graph each cycle (main thread; no DSR signal, or Queued if one is ever needed) for object
  instances of the known concept agents: class, internal name, `object_uid`, `given_name`, pose and covariance
  in its room frame, size, room, creating agent, born time; and notices deaths (node gone).
- One repertory entry per `object_uid` (agent-scoped uid, §7). Each sighting updates: last pose and covariance
  per room, a running pose history (sparse, keyframed by change, not by time), permanence, last seen time,
  given name.
- On death (node gone): the entry is closed for this life, with the death time; nothing is deleted.
- Data the graph does not carry (appearance): §6.3.

### 6.2 Read path: recall
- **Automatic, with the room:** when the robot's current room is known (the same room identity ltsm uses), the
  agent publishes, for each remembered object of that room, a **recall node** into the live graph: its own node
  type (NOT `object`, NOT named like a concept instance — concept agents' sweeps must never match it; same lesson
  as ltsm's `metaconcept` passage node), RT child of the room at the remembered pose, with attributes: class,
  `object_uid`, internal name, given name, pose covariance, size, permanence. Recall strength = permanence:
  every remembered object is published, and the consumer weights it; there is no publish cut-off.
- Recall nodes are removed when the robot leaves the room, and when the matching instance is alive (it is then
  represented by the real node).
- **Consumers:**
  - Concept agents at birth: compare the new instance with recall nodes of their class (pose likelihood under the
    recall covariance inflated by (1 − permanence) as a "moved" mixture, size, and the atlas when present) and,
    when they decide it is the same object, resurrect it: take its uid, internal name and given name. The
    decision is the agent's life cycle (U4); a shared helper in `common/` does the matching maths so each agent
    does not reinvent it.
  - room_concept / ltsm: high-permanence recall nodes as landmarks (later; v1 only publishes them).
  - language_agent: reads the memory graph read-only for history questions (G5), the way it reads ltsm's
    topology digest today, or via a digest on the live graph if a direct read is impractical.
- On request (no room known, or another room): language_agent can ask for one object or one class; v1 may
  serve this from the memory graph read directly rather than a request channel.

### 6.3 Appearance: the DINO atlas lives here once the object is dead (user, 11-01)
- **Alive:** the atlas stays with the owning agent (used every cycle; single writer). The owner publishes it at
  a low rate as an attribute of the instance node (< 10 KB, VIEW_ATLAS.md §3.1); ltom copies it by observation,
  like any other attribute. No files cross agents.
- **Dead:** ltom keeps the atlas in the object's entry. It is the store of record; the owner's
  `AtlasArchive` file stays only as its cache for when ltom is down (VIEW_ATLAS.md §13, on feat/view-atlas-2b).
- **Recall:** the recall node carries the remembered atlas; the owner scores a birth candidate against it
  (VIEW_ATLAS.md §3.4 maths unchanged).
- **Encoder id** on every atlas (model hash + fitted-constants version). Different ids are not compared:
  likelihood ratio 1, never a low similarity.
- **Later:** with every atlas in one place, cross-class re-identification (first a chair, later a table).

### 6.4 Acquired relations: edges between object nodes (user, 11-01)
In the memory graph each remembered object is a node; **edges are reserved for acquired relations**, both kinds
(user: "both"):

- **Learned from repeated observation** — habitual relations, e.g. `usually_on` (bottle → table),
  `usually_near` (chair → table), `member_of` (chair → dining set, from ring_metaconcept's arrangements),
  object → room `usually_in`. Each edge carries a belief that grows with evidence (a Beta of "held when both
  were observed" against "both observed, relation did not hold"), updated in the same visibility-aware way as
  permanence (§5): only co-observations count, deletions never do. No edge is "created at a threshold": an edge
  exists once the pair has been co-observed, its strength is the belief.
- **Told by the person** — e.g. `belongs_to` ("grandma's chair"), `goes_in` ("the remote goes in that
  drawer"), properties as attributes ("fragile"). Written by ltom on language_agent's request (same
  request/adopt pattern as given names: language_agent never writes the memory graph), carrying
  `source = told`, who said it and when. The robot can then answer "you told me".
- **Not stored:** relations computable from current poses ("40 cm from the table now") — the live graph has them.
  Memory edges hold only what accumulates over time or what was said.
- **Uses:** recall priors (the bottle is not where it was left ⇒ look on the table: the edge's target and strength
  go on the recall node), spatial descriptions (instance-names spec §5: a habitual relation is a better
  landmark), language_agent's history questions ("where does the remote usually go?").
- Edge types are registered in cortex (`REGISTER_EDGE_TYPE`, the user reinstalls). Relation vocabulary beyond the
  examples is the builder's, kept small; each learned relation's geometric likelihood reuses the relation model
  of the instance-names spec §5.1 so "near" means the same thing when learned, spoken and understood.
- Rollout: after step 3 of §8 (learned `usually_in` and `usually_near` first; told relations with the
  instance-names work).

## 7. Identity and names without waiting for the memory

- `object_uid` = (agent id, per-agent counter), allocated by the concept agent at birth from a counter it
  persists in its own small file — never from ltom_agent. Globally unique with no coordination.
- The **name high-water mark** is persisted the same way (the bug noted in the instance-names spec §11:
  chair/table/bottle rebuild it from the live graph; door_concept already persists it). Both counters live in
  one shared `common/` module (e.g. `common/instance_identity/`) used by all six agents.
- After ltom_agent runs, the per-agent records of deleted instances stop being the store: an agent may keep
  them as a cache for when ltom_agent is down, but resurrection reads the recall nodes first.
- Given names (instance-names spec): the owner still adopts them onto its node; ltom_agent then stores them by
  observation like any attribute, so they survive even an agent whose own record is lost. The instance-names
  handler's persistence hook may write to the agent cache only.

## 8. Rollout
1. `common/instance_identity/`: persisted uid and name counters in all six agents (fixes the reuse bug alone).
2. ltom_agent write path only (observe, consolidate, persist), with a viewer/CSV dump to check the repertory
   against a tour. No recall.
3. Recall nodes for chair only + the shared matching helper + chair resurrecting from them. Live tour: leave a
   room, come back, the same chair_N with the same given name.
4. Other agents; permanence feeding ltsm landmarks; language_agent history questions.
   Includes the memory graph on DDS domain 5 (user, 11-01: part of step 4, not a step of its own). Steps 2-3
   kept the repertory JSON as the store of record with no memory graph (no reader yet). In step 4 the graph is a
   ONE-WAY PROJECTION of the repertory: ltom owns domain 5 and rebuilds it from the JSON at start-up (no second
   persistence, no MemoryStore copy); one node per remembered object under per-room nodes, history and lives as
   attributes, the §6.4 relation edges between object nodes. language_agent joins it read-only and lazily while
   ltom is present (episodic's pattern for ltsm). Recall nodes stay on domain 0: concept agents never join domain 5.
   First consumers of the domain-5 graph (user, 11-01): language_agent AND an object inspector — click a
   remembered object (3-D graph viewer or belief-inspector dashboard) and see its node: identity, lives, keyframe
   trail on the room map, permanence, relations; its DINO atlas as structure (filled bins, directions, view counts,
   quality — embeddings are not decodable into a usable image); and an inverse sensor projection (the object's
   model rendered at its remembered pose into the ZED/LiDAR view from the current pose or from each atlas bin's
   direction — the imagined-observations renders, which the permanence visibility test needs anyway).
   OPEN (user): may ltom keep one thumbnail crop per atlas bin? It is what a person would want to see, but it
   contradicts the §3 non-goal "no raw images or crops".

## 9. Constraints
CLAUDE.md throughout: and/or/not, C++23, `std::from_chars` (es_ES locale) and `imbue(classic)` on every file
written, type-attributed DSR (new attributes/node type need `REGISTER_TYPE` lines and the user's cortex
reinstall), Queued connections only, no `runtime_checked_*`, SIGTERM only, `cbuild -j8`, presence protocol as
bottle_concept, a conformance-audit row in CONCEPT_AGENT_LIFECYCLE.md, optional `inner_eigen` results always
checked. Live runs coordinated with the room session.

## 10. Left to the builder
- Memory graph schema (node per object, edges per room), file format, consolidation cadence.
- Exact permanence update and its weights, from P(detect) and the occlusion model already in common/.
- The recall node's type name, attribute list beyond §6.2, and how the robot's current room is read.
- Whether language_agent reads the memory graph directly or through a digest.

## 11. My calls (not stated by the user)
- Name `ltom_agent`; own DDS domain like ltsm/episodic.
- Recall publishes every remembered object of the room, weighted by permanence; removed on leaving the room or
  when the real instance is alive.
- uid = (agent id, persisted counter), allocated by the agent; counters in one `common/` module.
- Atlas published by the owner at a change-driven low rate; ltom is the archive (VIEW_ATLAS.md §13).
- Rollout order of §8, starting with the counter fix.

## 12. Settled after the builder check (my calls)
- **Room key:** entries are keyed by ltsm's remembered room identity (the `room_id` ltsm recalls and archives,
  not the live `room_N` node name, which room_concept re-issues), and poses are stored in that remembered room's
  frame. When ltsm merges or re-anchors a room, ltom re-keys through ltsm's mapping; if ltsm offers none, that
  is a dependency to add to ltsm (one task), not a guess.
- **Death vs no information:** a node disappearing closes the entry's current life only. It is never evidence of
  absence. Permanence is updated **only** by the visibility test of §5 (expected visible and not found), never by
  deletions. Deletions while the owning agent's own node is gone or Degraded (graceful exit, flap) are recorded
  as "owner exit", not death. ltom's own downtime is a gap, nothing is inferred over it.
- **Scope:** chair, table, bottle, cabinet, refrigerator instances (generic `object` / `box` nodes filtered by
  each agent's name prefix). Doors stay in ltsm (passages); not in the repertory.
- **Attributes:** `object_uid` (VIEW_ATLAS.md §5) and `given_name` (instance-names spec) are not published yet.
  Both need `REGISTER_TYPE` lines and the user's cortex reinstall; step 1 of §8 adds `object_uid` publication
  alongside the counters. Rollout step 3 tests the given name only if the instance-names chair slice has landed;
  otherwise it tests uid and internal name.
- **Duplicates:** accepted in v1. A failed resurrection gives a second uid; the repertory keeps both. A later merge
  is a concept-agent decision (U4); ltom only flags pairs that look like one object (logged, and shown to the
  developer). No cross-class merge in v1.
- **Matching helper:** the "moved" component is uniform over the room's free area; mixture weight = 1 −
  permanence. It runs inside the agent's birth decision (birth stays tracker-only; resurrection only chooses
  which identity the newborn takes).
- **Visibility for permanence:** reuse the perception-card evaluator in common/perception_card (the one
  language_agent's can_see uses, with walls in line of sight and P(detect)). No new visibility code.
- **ids and domain:** builder checks CONCEPT_AGENT_RECIPE.md §1 and every config for the next free `[Agent].id`
  and DDS domain, records both in RECIPE §1. The memory graph is persisted the way ltsm's is (its memory_store).
- **Lost or corrupt counter file:** start from max(counter seen in the live graph, in ltom's memory graph when
  reachable, in the agent's own cache) + 1, log it loudly, and write the file back. Never from zero.
