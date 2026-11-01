# Instance names and spatial descriptions — design

Date: 2026-11-01. Status: agreed in conversation with the user (brainstorm 11-01); not yet planned.
Supersedes the open brainstorm in memory note `referent-resolution-next-project` (10-24): this is that project.

## 1. Intent

The robot, the person and the developer must be able to refer to the same instance (a chair, a table, the
fridge, a door) without ambiguity. People do not learn ids such as `chair_3`; they say "the chair next to the
table" or "the one by the door", and sometimes they give a thing a name of their own ("grandma's chair").
Proper names for every object are not an option: nobody calls furniture by human names.

So there are two layers:

1. **Stable internal name** — `chair_3`. For the developer (logs, grep) and for the robot (tool arguments,
   mission text). Never shown to the person unless asked.
2. **Words for the person** — either a **name the person gave** ("grandma's chair"), or a **spatial description**
   made on the fly, the shortest one that picks out the instance ("the chair", "the chair in the kitchen",
   "the chair next to the table", "the one by the door"). The robot uses these words when it speaks, and
   understands the same kind of words when the person speaks.

## 2. Goals, non-goals, anti-goals

Goals
- G1. A person can name an instance in the chat; the name sticks to that instance and comes back with it
  after a restart or a long absence, wherever the instance's identity comes back.
- G2. The robot never shows an internal id to the person in normal talk; it describes things spatially or by
  their given name.
- G3. The person's descriptions ("the chair by the door") resolve to one instance, or the robot asks a short
  clarifying question that names what tells the candidates apart.
- G4. Speaking and listening use one relation model, so the robot's own descriptions are ones it would itself
  resolve back to the right instance.

Non-goals
- Deciding whether an object is new or the same one again. That is the concept agent's life cycle
  (CONCEPT_AGENT_LIFECYCLE.md), extended by the DINO view atlas (common/view_atlas/VIEW_ATLAS.md, ongoing work).
  Names ride on whatever identity those produce.
- A long-term object memory. It comes later (user, 11-01); until then each concept agent's record of its
  deleted instances is where names persist.
- Room names ("the kitchen"): the 10-13 design reserved `room_label`; out of scope here except that the
  description generator may use a room's label when one exists.
- Appearance words (colour, material). v2: needs the atlas or the VLM.
- Deixis tied to gesture or gaze ("that one", pointing).

Anti-goals (failure even if it "works")
- Two instances with the same given name.
- Chat writing an instance node (breaks one-writer-per-object, intention_node.h, and the life-cycle
  ownership rule §5).
- Hard distance cut-offs for "next to" / "by" (CLAUDE.md: no thresholds where a model term works).
- The LLM inventing ids: the model copies the person's phrase into tool arguments and C++ resolves it.

## 3. Decided by the user

- D1. Internal name (`chair_3`) is already published and restored on recognition by the owning agent. Keep it.
- D2. The person-given name is an attribute **of the instance**, written **only by the owning concept agent**.
- D3. Chat proposes, the owner adopts: chat writes a request; the owning agent copies the name onto its node
  and into its record of deleted instances.
- D4. The handler lives in `common/` so every concept agent gets it once.
- D5. **Chat removes the request** (whoever writes something removes it). The owner only reads it.
- D6. Same given name already on another instance ⇒ chat **warns and refuses**; no request is written.
- D7. Descriptions are spatial, as humans do: "the chair next to the table", "the one by the door".
- D8. Both directions in this project: speaking (generate) and listening (resolve).

## 4. Naming protocol (layer 2a, given names)

### 4.1 Graph shape
- Request: a NON-RT edge `named`, `chat_intention → <instance node>` (chat's one owned node, intention_node.h),
  carrying attribute `given_name` (string). Written and deleted by language_agent only.
- Result: attribute `given_name` on the instance node, written by its owning concept agent only.
- New cortex registrations (the user reinstalls cortex; never `sudo cp`):
  `REGISTER_EDGE_TYPE(named)` in `dsr_edge_type.h` with a comment block in the style of `exit`;
  `REGISTER_TYPE(given_name, std::reference_wrapper<const std::string>, false)` in `dsr_attr_name.h`.
  Type-attributed access only (`given_name_att`), never `runtime_checked_*`.

### 4.2 Flow
1. Person: "call the chair by the door grandma's chair". The LLM calls a tool (§6) with the person's phrase
   for the referent and the name.
2. Chat resolves the phrase (§5). Ambiguous ⇒ ask, stop.
3. Chat checks every instance's `given_name` (normalised with `name_norm.h`): taken by another instance ⇒
   "grandma's chair is already the name of <description of that one>; pick another name", stop.
4. Chat writes/updates the `named` edge. If chat already has a `named` edge to that instance, it updates it.
5. Owner's handler (common/, §4.3) sees an incoming `named` edge, and if the node's `given_name` differs, writes
   it and updates its deleted-instance record entry for that instance. Idempotent.
6. Chat sees the instance's `given_name` equal to the requested name ⇒ deletes the edge, tells the person
   "done". While waiting, the request is "pending" and chat says so if asked. No timeout.

### 4.3 The common/ handler
- New module `common/instance_name/` (header-only is fine), e.g. `adopt_given_names(G, owned_nodes, record)`
  called from the agent's compute cycle on the main thread. It polls incoming `named` edges of the agent's
  own instance nodes; it does not connect DSR signals (if a later version does: Queued, never Direct).
- The persistence hook is a small interface the agent supplies ("store given_name for instance X in my
  deleted-instance record", "look it up on resurrection"), so the handler does not depend on how each agent
  keeps that record. Where the agent uses the view atlas, the record is `rc::atlas::ArchiveRecord`
  (common/view_atlas/atlas_io.h): add a `given_name` field and a column to its CSV (from_chars-safe; strings
  only, escape commas as the existing fields do).
- On resurrection/recognition the agent restores `given_name` together with the internal name (D1).
- An agent with no deleted-instance record yet keeps the name only for the instance's lifetime; the handler
  still works. The life-cycle doc gets a row in its conformance audit.
- Wire into each concept agent that owns nameable instances (chair, table, refrigerator, cabinet, bottle, door).
  Doors: the given name goes on the door node like any instance; the passage in LTSM already joins both faces,
  so a door seen from the other room is described by the same name via the passage (builder: copy the face's
  `given_name` onto the mirror face through door_concept, which owns both).

### 4.4 Clean-up cases
- Instance deleted before adoption: DSR drops its edges; chat reports "it is gone".
- Chat graceful exit: `chat_intention` is deleted ([Owns]), its edges with it.
- Chat SIGKILL: the existing stale sweep of `chat_intention` removes the leftover node and its edges.
- Owner not running: its instances are gone too; nothing to name.
- Renaming: a new name for the same instance replaces the old one (my call). Removing a name ("forget that
  name"): request with an empty `given_name`; the owner clears the attribute and the record field.

## 5. Spatial descriptions (layer 2b, speaking and listening)

### 5.1 One relation model
In `language_agent/src/referents.h` (C++, Eigen/plain, no DSR include), built on the facts chat already snapshots
(world_digest / `SeeScene`): instances with class, room, pose, size, σ; doors; walls; the robot.

Each relation is a likelihood `P(relation(target, landmark) | geometry)`, smooth, with scales taken from the
objects' own sizes and pose σ, never a fixed cut-off:
- `in <room>` — membership from the room the instance is in.
- `next to` / `by` / `near` — falls with the gap between the two footprints relative to their sizes; also
  competes with the other candidates (it is the closest such object to the landmark).
- `between A and B` — closeness to the segment joining A and B, relative to its length.
- `in front of` / `behind` — only for landmarks that have a front (a table side, a door's side, the robot);
  for the robot, the speaker's viewpoint. (Builder may defer front/behind for objects with no facing.)
- `the only <class>` — uniqueness in the scope (room, then house).

Landmarks: doors, tables, the fridge, walls/corners, the robot, and any named instance (my guess at the list;
the builder may add others with a stable size).

### 5.2 Listening (resolve)
Posterior over candidates of the stated class: salience prior × description likelihood.
- Salience prior: current room, in view, near the robot, recently mentioned (from the chat history),
  given name matches.
- Description likelihood: product of the parsed relations.
- Given names match first (exact after `name_norm`).
- Decision: act when one candidate dominates; otherwise ask a clarifying question built from the relation that
  best splits the leading candidates ("the one by the door or the one by the window?"). Framed as an epistemic
  action: ask when the expected cost of acting on the wrong instance exceeds the cost of a question. If a
  numeric decision rule is unavoidable it must be flagged and justified (CLAUDE.md).
- Parsing: the LLM passes the person's phrase verbatim; C++ parses class word + relation words + landmark
  words, EN and ES (reuse `canonical_concept` synonyms in can_see.h and `name_norm.h`). The LLM may also pass
  pre-split fields (class, relation, landmark) if that evaluates better on the Orin — builder's choice, decided
  by the eval.

### 5.3 Speaking (generate)
Choose the description a listener running §5.2 would resolve to the target with the highest probability,
with a cost per added word (shortest that works). Order of trial: given name → "the <class>" → "the <class> in
<room>" → one relation to a landmark → two relations. The landmark itself is described recursively but
never with a relation back to the target.

### 5.4 Where the words are used
- Every chat answer and tool result text that names an instance goes through the generator; ids stay in tool
  arguments, mission text and logs.
- `STATE` and digest text for the LLM can keep ids alongside the description (the model needs ids to call
  tools); the person-facing reply uses descriptions. Builder: decide by eval whether to give the model
  "chair_3 (the chair by the door)" pairs.

## 6. Chat tools
- `name_thing(what, name)` — §4.2. `what` is the person's phrase. Refusals use the REJECTED/ASK style of the
  other tools.
- Existing tools that take an instance (`approach_object`, `where_is`, `can_see`, `imagine`, …) accept a
  phrase in their object argument and resolve it with §5.2; an id still works.
- Eval cases in `tests/check_llm_tools.py`: naming (EN/ES), duplicate refused, "the chair by the door"
  passed verbatim into approach/can_see, an ambiguous phrase leading to a question. Run on the Orin
  (gemma4:26b, 3 repeats); old cases must not regress.

## 7. Constraints
- CLAUDE.md: and/or/not; C++23; `std::from_chars` only; type-attributed DSR; Queued connections, never
  Direct; SIGTERM only; `cbuild -j8`; no thresholds where a model term works.
- One writer per object: chat writes only `chat_intention` and its `named` edges; owners write only their nodes.
- The cortex registrations need the user's reinstall before any code using `named_edge_type` / `given_name_att`
  compiles.
- Concept-agent edits touch agents other sessions may run; coordinate live runs, and add the conformance-audit
  row to CONCEPT_AGENT_LIFECYCLE.md.

## 8. Left to the builder
- Exact likelihood forms and their scales, as long as they come from sizes and σ, not constants.
- Module and function names, the persistence interface's shape, file layout of tests.
- Order of wiring the concept agents (chair first is suggested: it has the most instances).
- Whether the clarifying question lists two or three candidates.

## 9. Calls I made (not stated by the user)
- Edge type `named` from `chat_intention`; attribute `given_name` on both edge and node.
- One given name per instance; a new name replaces the old; an empty name clears it.
- Doors: the name copied to the mirror face by door_concept.
- Landmark list (§5.1) and the speaking order (§5.3).
- No timeout on a pending request.

## 10. Settled after the builder check (my calls unless marked)
- **Finding requests:** the owner calls `G->get_edges_to_id(id)` (dsr_api.h) for each of its own instance nodes
  every compute cycle and keeps the `named` ones. "Own instances" = the nodes the agent's `[Owns]` globs match
  (the same set the presence coordinator cleans); the handler takes that list from the agent. Key = node id.
- **Deleted-instance records exist in every listed agent** (user, 11-01: "each concept agent keeps a record of its
  deleted instances"). The persistence interface (§4.3) adapts to each one; the builder locates them (chair:
  chair_scene_graph.cpp / specificworker; table: table_scene_graph.cpp; refrigerator and cabinet:
  specificworker_lifecycle.cpp; bottle: bottle_instance.h; door: door_core.cpp). G1 holds for all six.
- **Duplicate check scope:** names on live instances plus names in chat's own pending `named` edges. Names
  sitting only in a deleted-instance record are invisible to chat; if a resurrected instance brings back a name
  that is now taken, the owner restores it and chat, on seeing two equal names, tells the person and asks which
  one keeps it. Accepted residual risk.
- **Doors:** the request targets the face in the robot's current room, else the face the resolver picked.
  door_concept owns both faces and pairs them through the `exit` edges (dsr_edge_type.h); it copies `given_name`
  to the paired face. No LTSM write.
- **Snapshot data:** if world_digest / SeeScene lacks a footprint size, σ or room for some class, extending the
  snapshot is in scope (language_agent only).
- **front/behind:** v1 supports only the robot as landmark ("the chair in front of you"), in the robot frame.
  Front/behind of objects and of the person are deferred.
- **Ids in tool args:** an exact internal id (after `name_norm`) wins before any parsing. `name_thing` names only
  concept-agent instances, not rooms, walls or the robot (rooms keep the reserved `room_label`).
- **Eval bar for new cases:** each new case at least 2/3 on gemma4:26b; old cases no worse than the last run
  (362/378 on feat/imagine).
