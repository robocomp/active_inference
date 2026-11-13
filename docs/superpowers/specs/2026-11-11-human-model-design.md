# The human at the chat, remembered — design (2026-11-11)

Status: APPROVED by the owner 2026-11-11 (builder check done; the designer's calls listed below). Part 1 of a larger model (name and things); later parts listed at the end.

## Why

Asked "that is my chair", the robot can do nothing: the instance-names design
(`2026-11-01-instance-names-design.md`) copies a name STRING onto an instance, but "my chair" is a RELATION between a
person and an object, and it means something only if the robot represents who is speaking (owner, 2026-11-11).

Owner's intent: LTOM holds an element standing for THE HUMAN on the other side of the chat — one, always the same for
now. A human accumulates far more than a chair does, so the model is not a list of attributes but a hub that the
robot's other memories hang from: the person's name, their things, the missions they usually ask, the requests that
were not fulfilled. The long-term purpose is learning: failures and the person's corrections in the chat are kept so
the robot can later connect a new phrasing to a known request, compose a new tool from existing ones, or hand the
developer a well-formed request for what needs code (the robot cannot write code).

The owner chose the order: **name and things first** (this document). Everything else is a later part.

## Who and how the person talks to the robot

The person is the developer, at a desk, watching the robot's camera in the chat window and Webots (or the real
apartment). They are not standing next to the object. So "that" refers to what the person sees THROUGH the robot.

## Goals (part 1)

1. **Name.** "My name is Pablo" — said once, kept across restarts; the chat uses it when addressing the person and
   answers "what is my name?".
2. **Things.** "That is my chair" / "chair_34 is mine" links the person to that object. Afterwards "where is my
   chair?", "go to my chair", "is my chair still there?" work whether the object is perceived now or only remembered;
   if it is missing the robot can say so.
3. **Corrections.** "That is not my chair" retracts a link; the retraction is kept as history, never a deletion.
4. **Provenance.** Every fact about the person carries when it was said and the exact words.

## Anti-goal (owner)

**Linking the wrong object to the person without asking.** Therefore every ownership claim is read back and waits for
the person's yes before it is stored, even when the reference looks unambiguous ("chair_34, the magenta one ahead, is
yours?"). One word per claim is the accepted cost.

## Non-goals (part 1)

- Recognising who is speaking (face, voice): it is always the one human.
- Several humans.
- Objects the robot does not track (a cup): only objects with an LTOM entry (an `object_uid`) can be owned.
- The record of failed requests and corrections (part 2), usual missions (part 3), learning from them (later).
- Clicking the object in the chat's camera view (possible later aid for "that").
- Giving objects proper names ("call it the reading chair") — that stays the instance-names design; the two must not
  collide (see Constraints).

## Resolving "that" (owner agreed)

In order, from what the robot perceives and remembers:
1. objects in the camera's view, weighted by how centred they are (as the controller's lock-on does);
2. an object named in the last turn or two of the conversation;
3. the target of the last approach.
If more than one is plausible the robot asks, using its own words for them (name, colour, bearing: "chair_34, the
magenta one ahead, or chair_35 to the right?"). In every case the final read-back and yes (anti-goal) comes before
anything is stored. "My chair" with no "that" and no owned chair is answered honestly ("I don't know which chair is
yours yet").

## The structure (owner agreed)

### In LTOM (store of record)

```cpp
struct Said {                        // provenance of every fact about the human
    std::int64_t t_ms;               // when it was said
    std::string  words;              // the exact chat text ("that's my chair")
};

struct Ownership {                   // "my chair": a link, not a copy of the chair
    std::uint64_t object_uid;        // LTOM's stable id: survives restarts and chair_N renumbering
    std::string   cls;               // "chair"
    std::string   name_when_said;    // "chair_34", for the person's reading only
    Said          said;              // confirmed in the chat before it is stored
    std::optional<Said> retracted;   // "that's not my chair": kept as history, never deleted
};

struct Human {                       // ONE for now: whoever is at the chat
    std::string id = "human_1";
    std::string role = "developer";
    std::optional<std::pair<std::string, Said>> name;   // "Pablo", and when he said it
    std::vector<Ownership> things;                       // links into the repertory's entries
    // later parts hang here: request episodes (failures, corrections), habits read from episodic memory
};
```

### On disk

A `"humans"` section in the same repertory file (`ltom_agent/etc/ltom_repertory.json`): one store of record. An
older file without the section loads as "no human known yet". Written through the classic locale; any numeric parse
uses `std::from_chars` (CLAUDE.md).

```json
"humans": [{
  "id": "human_1", "role": "developer",
  "name": { "value": "Pablo", "t_ms": 1794420000000, "words": "my name is Pablo" },
  "things": [{ "object_uid": "21990232555554", "cls": "chair", "name_when_said": "chair_34",
               "t_ms": 1794420100000, "words": "that's my chair" }]
}]
```

### On LTOM's memory graph (domain 5, which language_agent already reads)

```
root ── human_1  (type person; person_name "Pablo", person_role "developer")
           │ owns
           ▼
room_1 ── chair_34  (the object's memory node, remembered or live)
```

`person`, `person_name`, `person_role` already exist in cortex. `owns` is a NEW edge type (`REGISTER_EDGE_TYPE(owns)`
in cortex's `dsr_edge_type.h`, with a comment block); the owner reinstalls cortex. The existing `has` is not reused:
its meaning is vague. A retracted link is not drawn.

## How the chat tells LTOM (agents talk only through memories)

language_agent never writes LTOM's state. After the person's yes it posts a one-off **claim** on the working graph,
under the node it already owns (`chat_intention`): its kind (`name`, `owns`, `disowns`), the object's `object_uid`
when there is one, the person's words, and a timestamp. LTOM reads the claim, records it in the hub, saves, and
reflects it on the memory graph. language_agent sees the result there and removes its claim (only the owner of a node
removes it — the instance-names pattern: requester writes, owner records, requester deletes on seeing the result).
A claim for an object LTOM does not know is not recorded; the chat says so.

New attributes the claim needs are registered in cortex (type-attributed access only). Their exact names and whether
the claim is a node or an edge are left to the builder, within these rules.

## What the chat says and does

- System prompt / STATE: "You are talking to Pablo (developer)." once a name is known; owned objects are marked in
  the object lines ("chair_34, magenta (p 0.69), Pablo's").
- `find_objects`, `where_is`, approach and remembered-goto accept "my chair" / "mine" and resolve it through the
  ownership links (live object first, else remembered).
- New chat abilities: tell the name; claim / disclaim a thing (with the read-back); answer "what is mine?".

## Constraints

- One human. The structure must not assume it forever (a list of humans on disk, `human_1` id), but no code path
  chooses between humans.
- Ownership is keyed on `object_uid`; chair_N names are display only (they are re-issued across restarts).
- No collision with instance names: "my chair" is a relation; a given name is a label. The chat must not turn
  "call it my chair" into a given name.
- Rates before features: the claim channel is event-driven and tiny; LTOM's cycle must not slow measurably.
- TDD; type-attributed DSR access; no `runtime_checked_*`; `and/or/not`; C++23 containers and algorithms.

## Later parts (recorded, not designed)

2. **Requests that were not fulfilled, and corrections**, each an episode linked to the human: what was asked, what the
   robot did, why it failed (missing ability / wrong tool / not understood), and how it was resolved. Uses: a
   developer backlog ranked by frequency; past cases as in-context examples for tool choice; phrasing analogies;
   composing new tools from existing ones (skills, procedural memory).
3. **Usual missions**: patterns read from the episodic memory's existing mission history (missions are already logged
   as "asked by a person (chat)"), offered back ("the usual?").

## Settled after the builder check (designer's calls, for the owner to check)

1. **The human node is not a perceived person.** language_agent already lists `person` nodes from human_concept as
   perceived objects; that reader is on the WORKING graph, the human node is on LTOM's MEMORY graph, and the memory
   reader takes only `object` nodes. Kept type `person`, and every chat listing excludes it explicitly (a test pins it),
   so "what do you see?" never answers "human_1".
2. **The yes is enforced in code, not left to the model.** A claim is built and shown as a card by a deterministic
   path, as remembered_goto does; the pending claim lives in the chat session; the next message binds to it only if
   it is a yes (the existing yes fast path), anything else cancels it. The model can PROPOSE a claim, never store one.
   This is the only reading of the anti-goal that cannot be talked past.
3. **One claim at a time, with an answer.** LTOM answers every claim on the human node of the memory graph
   (`recorded` or `rejected: <reason>`, matched by the claim's timestamp), and only then does the chat remove the
   claim and tell the person. If no answer comes (LTOM not running) the chat says so after a few seconds and removes
   it — an operational wait, not a modelling threshold (flagged).
4. **Ownership semantics.** Several things may be owned, of the same class too ("my chairs"); "my chair" with two
   owned chairs asks which. A re-claim after a retraction is a NEW `Ownership` entry; the history keeps both.
5. **Object ids are stable.** An `object_uid` is the owning agent's id in the high bits and its number below; a
   resurrected chair keeps it; LTOM never merges or rewrites entries. A chair re-born with a NEW uid (not resurrected)
   is a different object to the memory: the link stays on the old one, and when the robot sees a chair where the
   owned one was last seen it ASKS before moving the link (anti-goal), it never moves it by itself.
6. **Object lines:** ownership is said after the colour ("chair_34, magenta (p 0.69), Pablo's"); a given name, when
   the instance-names work lands, is separate and both may appear.
7. **A name change keeps history**, as a retraction does: the previous name and when it was replaced.

## Left to the builder

- Node vs edge for the claim, and the claim's attribute names (within the rules above).
- How the chat's tool schema exposes name / claim / disclaim (new tools or arguments of existing ones), checked with
  the tool-choice eval (`language_agent/tests/check_llm_tools.py`) on gemma4:26b.
- Where on LTOM's memory-graph canvas the human node is drawn.
