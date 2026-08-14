#!/usr/bin/env bash
#
# concept_audit.sh — is every <obj>_concept agent still aligned?
#
# WHY THIS EXISTS. Keeping the concept agents aligned is a PRIMARY goal: the fleet is the evidence base from
# which the minimal common pattern for generating new agents is derived, so a fix applied to one agent and
# not its siblings does not merely leave a bug — it corrupts the pattern. CONCEPT_AGENT_RECIPE.md is the
# prose form of that pattern, but a document cannot enforce itself: on 2026-08-09 the SAME defect (no
# room_polygon reaching plan_faces) was found independently in five agents. Copy-paste inheritance makes
# drift invisible. This script makes it a question with an answer.
#
# It is deliberately GREP-BASED and dependency-free: no build, no run, no graph. It answers "was this wired
# at all", not "does it work" — a structural audit, not a test. Cheap enough to run on every change.
#
#   usage:  tools/concept_audit.sh            # matrix + summary; exit 1 if anything FAILs
#           tools/concept_audit.sh --quiet    # summary only
#
# Adding a check: add a column below and a probe in audit_agent(). Keep every probe a single grep whose
# meaning is obvious from the pattern — an audit nobody trusts is worse than no audit.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
ROOT=$(pwd)
QUIET=0; [[ "${1:-}" == "--quiet" ]] && QUIET=1

PROTOCOL="common/affordance_protocol/affordance_protocol.h"
fails=0; warns=0

# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────
# Print a cell and count failures. mark <ok|no|na> <label>
mark() {
    case "$1" in
        ok) printf "%-13s" "ok" ;;
        no) printf "%-13s" "MISSING"; fails=$((fails+1)) ;;
        na) printf "%-13s" "n/a" ;;
        wr) printf "%-13s" "warn";   warns=$((warns+1)) ;;
    esac
}

audit_agent() {
    local agent="$1" src="$1/src" name="${1%_concept}"
    printf "  %-22s" "$name"

    # 1. NBV reachability: the room polygon must reach plan_faces, or rc::nbv::is_reachable imposes no
    #    constraint (it refuses to guess on an empty polygon) and a viewpoint OUTSIDE the room scores the
    #    same as the one inside — the raw information term is direction-blind and cannot break that tie.
    #    ★Require an actual CALL, not a mention: bottle_concept carries a comment reading "this path does
    #    NOT go through plan_faces", and a bare grep scored that explanation as use of the function.
    if grep -rqE "rc::nbv::plan_faces\\(|[^_a-z]plan_faces\\(" "$src" 2>/dev/null; then
        grep -rq "room_polygon" "$src"/epistemic_planner.cpp 2>/dev/null && mark ok || mark no
        # 2. …and it must REFUSE when no face is usable, rather than publishing the raw argmax as a hint:
        #    the controller REPAIRS an unroutable standpoint onto the object instead of rejecting it.
        grep -rq "any_usable" "$src"/epistemic_planner.cpp 2>/dev/null && mark ok || mark no
    else
        # Single-candidate planners (bottle's far-side arc, human's orbit) score their OWN geometry via
        # rc::nbv::score(), which checks stands_inside but NOT is_reachable — that lives in plan_faces. They
        # must therefore make the reachability check themselves, and refuse, or a viewpoint outside the room
        # is published and the controller repairs it onto the object.
        grep -rq "room_polygon" "$src"/epistemic_planner.cpp 2>/dev/null && mark ok || mark no
        grep -rq "is_reachable" "$src"/epistemic_planner.cpp 2>/dev/null && mark ok || mark no
    fi

    # 3. Affordance contract: every default_contract_for(<key>) this agent asks for must HAVE a case.
    #    ★The fallback returns Contract::reach() — a valid-LOOKING contract — so a missing case is
    #    invisible, and since wants_final_facing() honours the target yaw only for Servo/Orient the robot
    #    silently arrives facing the wrong way. This probe is the only thing that makes it visible.
    local keys missing=0
    # ★2026-08-11: the producer moved to common/object_affordance, so no agent calls default_contract_for
    #    directly any more — this probe read `n/a` for ALL EIGHT agents and stopped checking invariant 11
    #    entirely, silently, the moment the extraction landed. An audit that goes blind is worse than no
    #    audit, because the green column is read as evidence. The key now comes from the object_type the
    #    agent passes to ObjectAffordance::init(), which is the same string that reaches the contract.
    #    (grep the init LINE, then take its last quoted token — the arg list contains node.id(), so a
    #    [^)]* pattern stops at that inner ')' and matches nothing, which is how the first repair of this
    #    probe ALSO read n/a everywhere. Verified against a known-good agent before being trusted.)
    keys=$(grep -rh 'affordance\.init(' "$src" 2>/dev/null \
           | grep -oE '"[a-z_]+"' | tr -d '"' | sort -u)
    if [[ -z "$keys" ]]; then
        mark na
    else
        for k in $keys; do
            grep -q "object_type == \"$k\"" "$PROTOCOL" || missing=1
        done
        [[ $missing -eq 0 ]] && mark ok || mark no
    fi

    # 4. sigma* coverage in <obj>_dof.h. With NO DOF declaring one, any_sigma_star() is false and the
    #    adequacy gap silently degrades to 0.5*ln det Sigma — which has no meaningful zero, so the belief
    #    strip cannot show a fixation working. Partial coverage is fine (a nuisance DOF carries no demand).
    local dof="$src/${name}_dof.h" total nostar
    if [[ -f "$dof" ]]; then
        total=$(grep -cE '^\s*\{"' "$dof")
        nostar=$(grep -E '^\s*\{"' "$dof" | grep -c -- '-1\.0f')
        if   [[ "$total" -eq 0 ]]; then mark na
        elif [[ "$nostar" -eq "$total" ]]; then
            # Nothing published ⇒ the adequacy gap is unusable and the strip falls back to logdet. That is a
            # DEFECT when it was forgotten and a FACT when no consumer has stated a demand — and the two look
            # identical in the code. An agent may declare the second with a "SIGMA-STAR: none — <reason>"
            # line in its dof header; the audit then reports it without failing. An audit that scores a
            # reviewed decision as a failure is one people learn to ignore.
            if grep -q "SIGMA-STAR: none" "$dof"; then printf "%-13s" "none(decl)"; else mark no; fi
        else printf "%-13s" "$((total-nostar))/$total"; fi
    else
        mark na
    fi

    # 5. Detector envelope from config, not a hardcoded prior. The envelope is genuinely object-dependent
    #    (measured: fridge max_fill 1.32 vs table 0.677), so every agent needs the keys even before it has
    #    a calibration of its own.
    if grep -rq "DetectorEnvelope{}" "$src" 2>/dev/null; then mark wr; else
        grep -rq "DetectorEnvelope" "$src" 2>/dev/null && mark ok || mark na; fi

    # 6. Compact belief strip (one row per BELIEF UNIT, each row a time series) — the standing display.
    grep -rq "belief_strip" "$src" 2>/dev/null && mark ok || mark wr

    # 7. The controller-owned protocol flags must be POLLED, never taken from update_node_attr_signal:
    #    that signal fires for every attribute of every node in the shared graph.
    #    ★Match an actual connect(), not a MENTION: several agents carry a comment reading "NOT
    #    update_node_attr_signal" explaining why they poll, and a bare grep scores those as offenders.
    if grep -rE "connect\\(.*update_node_attr_signal" "$src" 2>/dev/null | grep -qv "^\\s*//"; then mark no
    else grep -rq "poll_affordance_protocol" "$src" 2>/dev/null && mark ok || mark na; fi

    # 8. ONE removal authority. Invariant 5: removal is a Bayesian decision on the existence log-odds, never
    #    a miss counter. An agent that has an existence channel AND still arms the tracker's death counter
    #    has TWO ways to delete an instance, and the counter's deletion is the one that carries no evidence
    #    and leaves no attributable record — so a phantom analysis cannot tell a reasoned removal from a
    #    timeout. Found 2026-08-10 in chair, cabinet and door, all three of which do remove on L as well.
    #    refrigerator/table/bottle disable it (`tp.death_frames = INT_MAX` / behind the existence flag).
    if grep -rqE "exist(ence)?[._]" "$src" 2>/dev/null; then
        if grep -rqE "tp\\.death_frames\\s*=\\s*(std::numeric_limits<int>::max|cfg_\\.[a-z_]*exist)" "$src" 2>/dev/null
        then mark ok
        elif grep -rq "tp\\.death_frames" "$src" 2>/dev/null; then mark wr
        else mark na; fi
    else
        mark na
    fi

    # 9. SHARED-MODULE ADOPTION, as a COUNT rather than one column per module. Extraction is not adoption:
    #    every shared module in this fleet was written because N agents had drifted, and every one of them
    #    then reached some agents and not others — decide_removal existed while door hand-wrote it in three
    #    places and chair never called it; rc::birth::evidence said "every concept agent must use it" while
    #    four ignored it; cell_key was defined once in common/ and re-inlined in two agents on a grid that
    #    only happened to agree. This counts the modules an object-concept agent is expected to use, so a
    #    new one shows up as a number falling rather than as a column nobody added.
    #    owned_nodes is UNIVERSAL — every agent owns instance + affordance nodes and must reap both on the
    #    startup stale-sweep and on shutdown, so it belongs on this list. birth_surprise and lidar_ingestor
    #    do NOT: they are legitimately used by a SUBSET (4 and 5 agents), and a universal list is only
    #    honest for a universal obligation.
    local must="existence_belief instance_tracker mask_ingestor rt_covariance object_affordance nbv detectability occlusion phantom_log graph_provenance concept_manifest support_bank footprint lidar_select owned_nodes"
    local have=0 want=0
    for m in $must; do
        want=$((want+1))
        grep -rq "common/$m/" "$src" 2>/dev/null && have=$((have+1))
    done
    if [[ $have -eq $want ]]; then mark ok; else printf "%-13s" "$have/$want"; fi

    # 10. THE MANIFEST IS AUTHORITATIVE, AND AN INHERITED WORLD FACT IS FATAL. A concept agent must declare
    #    what its object IS in common/concept_manifest/<concept>.concept.toml and READ it — not merely
    #    cross-check it, and not carry the values in code where a clone inherits them silently. The gate is
    #    rc::manifest::provenance_ok: a block declaring `from = "inherited"` stops the agent at startup.
    #    'warn' = no manifest, or one that is never read. hood_concept shipped TEN cloned defects in a week,
    #    several DECLARED inherited in writing, because the declaration was only ever a note.
    if [[ -f "common/concept_manifest/${name}.concept.toml" ]]; then
        grep -rq "provenance_ok" "$src" 2>/dev/null && mark ok || mark wr
    else
        mark wr
    fi

    # 11. BIRTH IS AN OBSERVATION, NOT A CYCLE. common/instance_tracker/birth_evidence.h says "every concept
    #    agent must use it": agents feed the tracker on EVERY compute cycle (a candidate with no matching
    #    detection expires), so leaving birth_evidence at its 1.0 default makes birth_frames count COMPUTE
    #    CYCLES. At ~10 Hz compute against a ~9.5 Hz mask stream, "8 frames" is well under a second of one
    #    unchanging view — which is how a YOLO false positive on a wall panel becomes furniture. Found
    #    2026-08-12 in bottle, chair and door (never set it) and in cabinet (worse: it applied a
    #    corroboration boost RAW, accruing >1 per stale inadmissible cycle).
    if grep -rq "DetectionView" "$src" 2>/dev/null; then
        grep -rq "rc::birth::evidence" "$src" 2>/dev/null && mark ok || mark no
    else
        mark na
    fi

    # 12. THE ABSENCE GUARD MUST READ A *FRESH* VERDICT. An agent that suppresses silhouette absence on
    #    `dbg_gated` ("this frame may not move the geometry, so it may not destroy the object either") must
    #    also check dbg_gate_fresh, because run_inference returns EARLY when no mask arrives — so on exactly
    #    the cycles the guard fires, the flag is whatever the last real look left behind. A phantom has no
    #    mask by definition, so its verdict freezes forever and it can never accrue absence. Measured on
    #    refrigerator_concept 2026-08-12: phantom refrigerator_2 ran 6578 cycles with 678 predicted-visible
    #    pixels and none lit, the robot centred on it, and admitted 0.000 absence evidence. Replayed through
    #    the freshness-aware guard it dies after 399 cycles.
    if grep -rq "dbg_gated" "$src" 2>/dev/null; then
        grep -rq "dbg_gate_fresh" "$src" 2>/dev/null && mark ok || mark no
    else
        mark na
    fi

    # 13. ONLY THE FRONT RGB-D CAMERA MAY CREATE OR UPDATE AN OBJECT. The voxelizer publishes `mask_source`
    #    (0 = zed, 1 = ricoh) precisely because `mask_has_depth` stopped answering that question: once ricoh
    #    masks were depth-filled from reprojected LiDAR they ship as full 3D slices with has_depth = 1. Every
    #    guard written as `if (has_depth)` then silently began accepting 360-degree detections from BEHIND the
    #    robot. Measured on bottle_concept 2026-08-12: of 130 births, 124 (95%) were at more than 55 degrees
    #    off the robot heading — median 141 degrees — i.e. outside the ZED cone entirely.
    #    An agent that builds a DetectionView must gate it on may_fit_geometry().
    if grep -rq "DetectionView" "$src" 2>/dev/null; then
        grep -rq "may_fit_geometry" "$src" 2>/dev/null && mark ok || mark no
    else
        mark na
    fi

    # 14. THE REMOVAL DECISION IS THE SHARED ONE. rc::exist::decide_removal owns the boundary, the debounce
    #    unit (IDEAL OBSERVATIONS, never cycles) and — since 2026-08-11 — the confidence-scaled requirement.
    #    Every hand-written copy of those six lines has drifted: three ways by 2026-08-10 (door paid with
    #    twelve deaths at fixated=0), and on 2026-08-11 door STILL had three hand-rolled sites, two of them
    #    counting cycles, while refrigerator/hood ran a second integer streak on the same L that fired first
    #    and masked the fix in the shared one.
    #    ★This probe is what would have caught chair: chair integrates through the shared BELIEF but decides
    #    with a bare `L < RemoveLogodds` and NO debounce at all — the only agent in the fleet that can delete
    #    an instance on a single frame. Check 8 above says nothing about it: it only asks whether the
    #    TRACKER's death counter is armed, and chair's is not.
    if grep -rqE "exist(ence)?[._]" "$src" 2>/dev/null; then
        grep -rq "rc::exist::decide_removal" "$src" 2>/dev/null && mark ok || mark wr
    else
        mark na
    fi

    echo
}

# ── run ──────────────────────────────────────────────────────────────────────────────────────────────
echo
echo "concept-agent alignment audit — $ROOT"
echo
printf "  %-22s%-13s%-13s%-13s%-13s%-13s%-13s%-13s%-13s%-13s%-13s%-13s%-13s%-13s%-13s\n" \
       agent room_poly any_usable contract "sigma*" envelope strip poll removal shared manifest birth absence mask_src decision
printf "  %s\n" "$(printf '%.0s-' {1..205})"

skipped=()
for d in *_concept; do
    [[ -d "$d/src" ]] || continue
    # An OBJECT-concept agent is one that owns instances of a class and publishes an affordance for them.
    # robot/room/residual are concept agents in NAME but model the robot, the room and the unexplained
    # remainder — not objects IN the room. They have no four-face viewpoint plan and no per-object contract,
    # so scoring them against this pattern would be noise, not drift. room is excluded by name rather than
    # by structure: it does own an epistemic_planner (it plans viewpoints to localise itself), which makes
    # it structurally indistinguishable from an object agent while being conceptually the container.
    if [[ "$d" == room_concept ]]; then skipped+=("${d%_concept}"); continue; fi
    if [[ -f "$d/src/epistemic_planner.h" ]] || compgen -G "$d/src/*_affordance.cpp" > /dev/null; then
        audit_agent "$d"
    else
        skipped+=("${d%_concept}")
    fi
done
[[ ${#skipped[@]} -gt 0 ]] && printf "\n  not object-concept agents (skipped): %s\n" "${skipped[*]}"

echo
if [[ $QUIET -eq 0 ]]; then
    cat <<'NOTE'
  room_poly / any_usable  the NBV must know the reachable region, and must refuse when no face is usable
  contract                every default_contract_for(key) this agent asks for has a case in the protocol
  sigma*                  DOFs with a consumer demand / total; 0 of N ⇒ the adequacy gap degrades to
                          logdet. 'none(decl)' = reviewed and deliberately absent (SIGMA-STAR: none)
  envelope                'warn' = hardcoded DetectorEnvelope{} prior instead of config keys
  strip                   'warn' = no compact belief strip
  poll                    protocol flags polled, not pushed by the update_node_attr_signal firehose
  removal                 ONE removal authority: an existence channel AND an armed miss counter is two
  shared                  how many of the modules an object-concept agent is expected to use it actually
                          uses. Extraction is not adoption — every gap this session began as a green matrix
  manifest                the agent declares what its object IS in a manifest AND reads it. An inherited
                          world fact ('from = \"inherited\"') stops the agent at startup — a note could not
  birth                   birth_evidence must come from rc::birth::evidence — an OBSERVATION, admitted by
                          the UPDATE rule, weighted by confidence x range. The 1.0 default counts CYCLES
  absence                 an admissibility guard on dbg_gated must also read dbg_gate_fresh — a verdict from
                          a cycle with no mask is STALE, and a phantom (which never has one) freezes it
  mask_src                only the front RGB-D camera may CREATE or UPDATE an object: a DetectionView must
                          be gated on may_fit_geometry(). has_depth does NOT answer this — a lidar-depth
                          ricoh mask has has_depth = 1
  decision                the removal DECISION goes through rc::exist::decide_removal (shared boundary +
                          debounce in ideal observations + confidence-scaled requirement). 'warn' = the
                          agent hand-writes it, which is how the debounce drifted three ways before

NOTE
fi
echo "  $fails failing, $warns warnings"
echo
[[ $fails -eq 0 ]]
