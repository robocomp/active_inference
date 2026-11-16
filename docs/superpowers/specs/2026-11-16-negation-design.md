# Negation: what the owner's "no" and "not" mean to the robot

Status: APPROVED by the owner 2026-11-16, with mission and fleet confirmations included in (a). Two corrections
were made while planning (see "Settled after the builder check"): the cortex float counts, and the wire carries the
verdict. Scope chosen by the owner: **(a)** replies to the robot's own
questions and **(b)** the owner's own statements. **(c)** negation inside mission conditions ("if you don't see the
bottle…") is later; it is noted, not designed here.

## Intent and why

The changed-object runs of 2026-11-16 showed three ways negation went wrong:

1. A keyword reader decided what a reply meant. "Yes, I didn't tell you" contains "I didn't", and only the order of
   the checks saved it. "Nobody moved it" contains no "no" at all and was not read as a no.
2. In run 6, the model was given a bare "yes" without the question it answered. It guessed an ownership claim
   ("that chair is mine") and replied as if it knew.
3. "I didn't see it there" became evidence only once it was weighed by how visible the spot was (the ghost and its
   P(detect)).

The owner asked for a theory instead of patches. The theory is: **a negation is evidence about one particular
statement, weighted by how reliable it is.** It is not a flag on a word. The same rule already covers perception
(item 3), and this spec applies it to words.

## The theory

1. **The statement under discussion.** Each question or read-back the robot shows puts one statement up for
   answer, for example "the owner moved chair_50 since 15 November", or "remember that chair_36 is not the owner's".
   A reply is read only against that statement, and the reading is one of:
   - **affirm**: the statement is true;
   - **deny**: the statement is false;
   - **unknown**: the owner says they don't know;
   - **none**: the reply doesn't answer it.

   The reading also carries a **remainder**: anything else the reply says or asks for. A "not" counts only when it
   applies to the statement itself. "Yes, I didn't tell you" is affirm. "No, go to the kitchen" is deny, with "go
   to the kitchen" as the remainder.
2. **The robot asks only positive questions.** It asks "Did you move it?" and "Shall I remember that chair_36 is not
   yours?". It never asks "You didn't move it, did you?" or "chair_36 is NOT yours?" (today's disowns read-back has
   that form). With a positive question, a bare yes or no means the same thing in English and in Spanish. Each
   question records its own form, and the reading is made against that form.
3. **A reading is a probability, and the update uses all of it.** The reader returns a distribution
   p(affirm), p(deny), p(unknown), p(none) that sums to 1, not one label. The update is Jeffrey's rule, applied to
   the one hypothesis the question is about:
   - **"Did you move it?"** The owner is the only one who moves things (changed-object spec), so the hypothesis is
     "this sighting is the remembered chair, moved". A "yes" makes it certain and a "no" makes it impossible. Then
     `P(same | reply) = p(affirm) + (p(unknown) + p(none)) · P(same)`. The belief "the owner moves things" gets
     fractional counts: `yes += p(affirm)`, `no += p(deny)`.
   - **A read-back of a statement:** the same rule, with the hypothesis "the owner meant this".
4. **What to do next is decided by cost, not by a cutoff.** After a reading, the robot weighs three actions for the
   question, using the costs the question policy already uses (CostAsk, CostWrong):
   - **act on it**: expected cost (1 − P(true)) · CostWrong. "Act" means confirm the link, or store the claim.
   - **act against it**: expected cost P(true) · CostWrong. This means cut the link (said_new), or drop the claim.
   - **ask again**: cost CostAsk. The robot says one short clarifying question, for example "Sorry, did you move
     the magenta chair or not?".

   The robot takes the cheapest. A clear reply acts at once. "I don't think so" acts at once only if the costs say
   so. "No, I moved it" (half affirm, half deny) asks again. A question can be raised at most twice (changed-object
   anti-goal: no nagging), and asking again counts as the second raise. After that, the robot takes the cheaper of
   the two acts.
5. **The model reads; the robot code decides.** The language model only produces the reading. It never stores a
   fact or calls a memory tool because of a reply. The robot code makes the update, the decision in item 4, and the
   store (LTOM, through the existing human-claim path).
6. **Words and perception follow one rule.** "Not seen there" counts as P(detect)-weighted evidence against
   "there". A spoken "no" counts as reading-weighted evidence against the statement asked about. Nothing new is
   needed for the perception side here; (c) will connect mission conditions to it later.

## Behaviour

### (a) Replies to the robot's questions

- While a question is open (the moved-chair question today, and any later question of the same kind), the owner's
  next message goes to the **reply reader** before anything else. The reader gets:
  - the statement;
  - the exact words the robot said;
  - the owner's reply.

  It returns the distribution and the remainder.
- The robot code updates and decides (item 4). If it acts, the claim is posted to LTOM as today
  (`moved:yes|no|unknown:<stamp>`) and also carries the reading's probabilities, so LTOM stores fractional evidence.
- The remainder goes on to the model as a normal user message. Before it, the history holds the robot's question
  and the reading ("the owner answered NO to: did you move the magenta chair?"). The model can never again face a
  bare "yes" without knowing what it answered (fixes run 6).
- If the reading is mostly **none**, nothing is stored, the whole message goes to the model, and the question
  follows its existing re-raise rule.
- **Exact one-word replies are read in code:** a message that is only "yes", "no", "sí", "si" or "no", ignoring
  case and punctuation, gets p = 1 for its reading. This is safe only because item 2 makes every question positive.
  It is an exact match on the whole reply, not a keyword search. *(My call: it saves a model call on the most
  common reply. Cost if wrong: none for meaning; the corpus covers these too.)*

### (b) The owner's own statements

- The main model still decides that a message is a claim: `human_claim` with kind owns, disowns, forget or name.
  Choosing the polarity (owns or disowns) is the model's reading of the owner's sentence. That is where "that's not
  my chair", "it's not that it isn't mine" and "don't forget that" are told apart. "Don't forget that" makes no
  claim at all.
- The read-back card becomes a positive question about the act of remembering:
  - "Shall I remember that chair_36 is yours?"
  - "Shall I remember that chair_36 is not yours?"
  - "Shall I forget that you move things?"
  - "Shall I remember that your name is Pablo?"
- The owner's reply to the card goes through the same reader and the same decision (items 3 and 4) as in (a),
  replacing today's exact `is_yes` check. "No, I meant the yellow one" is deny with a remainder: the card is dropped
  and the remainder goes to the model, which can make a new claim.
- **"Not mine" is kept even when the robot had no ownership to retract.** Today, disowns on an unowned object is
  rejected ("it was not yours") and the statement is lost. Under this theory, "not yours" is evidence about
  ownership whatever the robot believed before. Ownership becomes a list of told records with a polarity:
  "owns" and "not owns", each with words and a time. The `owns` edge follows the latest record. "Is chair_36
  mine?" can then answer "On 16 November you told me it is not yours." *(My call: ownership stays told, not
  inferred. The latest statement wins and the history stays. Cost if wrong: a later Beta over ownership is a
  change to the record type only.)*
- "Forget that" is not a negation. It retracts the records of a belief, as it does today. Only its read-back and
  the reading of the reply change.

## Measuring it (no hand checks)

- **Reply corpus** `language_agent/tests/reply_corpus.json`. Each item has the statement, the robot's question, the
  reply, and the expected reading. It starts with the 17 probe replies from tonight, then adds Spanish, negatives
  that contain no "no" ("nobody touched it", "it was like that already"), double negatives, hedges ("I don't think
  so", "probably"), mixed replies ("no, I moved it"), replies with a remainder ("no, go to the kitchen"), and
  non-answers ("what time is it?").
- **Statement corpus.** It extends `tests/check_llm_tools.py` with owner sentences and the human_claim kind expected
  for each, or no call at all: "that's not my chair", "that chair isn't mine anymore", "it's not that it isn't
  mine", "don't forget that", "forget that I move things", and Spanish forms.
- A script `tests/check_reply_reader.py` runs the reader on the reply corpus with the configured local model
  (`config_local.toml`, gemma4:12b). It reports:
  - how many readings are wrong at the most likely label;
  - the Brier score of the distributions;
  - for the mixed and hedged items, the action the decision rule picks.

  **Pass:** no wrong most-likely reading on the plain yes/no/unknown items, and every mixed item leads to "ask
  again" or to the right act. The numbers go into the commit message.

## Goals

- No reply to a robot question is read by keyword search any more (`classify_reply` goes away).
- The model never sees a bare yes or no without the question it answers.
- A hedged or mixed reply moves the beliefs only as far as it deserves, and a confusing one makes the robot ask once
  more instead of guessing.
- "That's not my chair" is remembered with its words and date, even when nothing was owned before.
- Run 6 repeated (yes arrives late) and Run C (no) pass with the reader in place.

## Non-goals

- (c) negation in mission conditions: later.
- Other speakers. Only the owner, as in the changed-object spec.
- Inferring ownership from behaviour. Ownership stays told.
- Re-reading old answers stored before this change. They count as p = 1.

## Anti-goals

- A word list growing one phrase at a time. The fix for a misread reply is a corpus item and a better reader
  prompt, never another `has("…")`.
- The model writing memory on its own reading.
- New cutoffs. The only constants are the question policy's existing CostAsk and CostWrong.
- Asking more often than today. Clarification counts against the existing two-raise limit.

## Constraints

- CLAUDE.md: no thresholds; and/or/not; `from_chars`; type-attributed attributes only. A new attribute for the
  reading's probabilities, if one is needed, goes into cortex's `dsr_attr_name.h`, and the owner reinstalls cortex.
- Agents talk only through memory: the reading reaches LTOM inside the claim on the graph, as today.
- The reader runs on the same Ollama endpoint as the chat (local GPU, gemma4:12b). It must not block the chat's main
  turn. While it runs, the message is held, exactly like the existing busy-hold.

## Settled after the builder check (my calls)

- **Raise budget.** The first raise is the question itself; the clarifying question is the second and last. If the
  reply to the clarifier is still mixed, the robot takes the cheaper of the two acts. If that reply is mostly
  **none**, the robot acts on nothing, the question goes quiet (today's rule), and the message goes to the model.
- **Cards.** A mostly-**none** reply to a claim card drops the card, as any other reply does today, and the message
  goes to the model. A card gets at most one clarifying question.
- **One cost pair** (CostAsk 0.4, CostWrong 1.0) for both kinds of question. A card's prior P(true) is 0.5, so the
  reply alone decides: the owner has only just said it, and the read-back exists to check the model's reading.
- **On the wire.** The claim value carries the chat's decision and the reading:
  `<yes|no|unknown>:<p_affirm>:<p_deny>:<stamp>`. The decision is in it because LTOM must not decide a second
  time. It is written through a classic locale and parsed with `from_chars`. The old form `<verdict>:<stamp>` still
  parses, as p = 1/0. `BeliefRecord` drops its `bool yes` for `double p_yes, p_no`, records already on disk load as
  1/0, and `learned_text` prints the summed weights.
- **One cortex change (corrected while planning).** The belief counts on the graph are `int` (`belief_yes`,
  `belief_no`), so they cannot carry fractions. Two new float attributes, `belief_p_yes` and `belief_p_no`, replace
  them in both agents. The owner reinstalls cortex once.
- **"Not yours" reaches the chat as a belief.** Each told ownership is also published as the belief `owns:<uid>`,
  with its records and words. "What have you learned" and "is chair_36 mine?" can then quote "you told me on 16
  November it is not yours" with no other new attribute.
- **Old ownership** loads as one "owns" record. A retracted ownership loads as "owns" plus a "not owns" record at
  its retraction time. The `owns` edge follows the latest record.
- **The reader** is its own non-streaming Ollama request, separate from the chat turn. The owner's message is held
  while it runs, the same way as the existing busy-hold.
- **The shortcut** is a whole-message match only. A message that is just "si" is a reply. "si" inside a sentence
  never is.

## Left to the builder

- How the reader gets its probabilities: token log-probabilities of a one-word label, if the installed Ollama
  (0.40.16) gives them for this model, or else a small JSON with stated probabilities, checked against the corpus
  by the Brier score. The corpus decides, not taste.
- The exact reader prompt, the clarifying-question wording (positive form, it names the object), and where the
  fractional counts sit in LTOM's BeliefRecord (a `weight` next to `yes` is the obvious shape).
- How old records without probabilities load: as p = 1.
