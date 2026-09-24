#!/usr/bin/env bash
# check_registry_complete.sh <agent> — does this agent's declare_complete() claim actually hold?
#
# declare_complete() ARMS the unread sweep. It is a CLAIM: every config key this agent reads goes
# through an rc::cfg::Reader. If the claim is false the sweep names keys as "read by nothing" that
# are in fact read through an old dialect — an instrument that lies confidently, which is worse than
# one that says INCONCLUSIVE. So the claim in code and this grep must agree, or the claim is withdrawn.
#
# Exit 0: no old dialect survives, declare_complete() is honest.
# Exit 1: sites remain; they are listed. Do NOT call declare_complete() yet.
set -u
root="$(cd "$(dirname "$0")/../.." && pwd)"
agent="${1:-}"
[ -n "$agent" ] || { echo "usage: $(basename "$0") <agent>   (e.g. table_concept)"; exit 2; }
src="$root/$agent/src"
[ -d "$src" ] || { echo "no such agent: $src"; exit 2; }

# What counts as a BYPASS is a read that touches the loader directly - not the NAME of the helper
# that wraps it. A migrated agent keeps its getf/geti/gets/getb call sites verbatim; the four lambdas
# behind them now forward to rc::cfg::Reader, so matching on those names would flag all 130 correct
# sites and report a lie of the opposite kind. Match the loader contact itself instead.
#
# generated/ is excluded on purpose: it is robocompdsl output, and the fleet-wide fix for it is the
# generator template, not a per-agent edit.
# Matches the loader by TYPE-ish name, not by one spelling: `configLoader`, `config_loader`, and
# `peer_loader` are all ConfigLoaders, and the last one is exactly the read that must NOT
# register (it is a crashed peer's file). Missing it is how a checker passes an agent that is
# still reading raw somewhere.
pattern='ConfigLoaderUtils::|load_optional_cast|\b\w*[Ll]oader\s*\.\s*(get|exists)\s*[<(]|\bcfg\s*\.\s*(get|exists)\s*[<(]'

# ★Strip COMMENTS first. controller carries two commented-out example reads and viewer3d documents
# the old dialect in a header comment; counting those as live bypasses is a checker that cannot be
# satisfied, which is the same as a checker nobody runs.
strip_comments() { grep -vE "^\s*(//|\*|/\*)" ; }

hits=$(grep -rnE "$pattern" "$src" --include='*.cpp' --include='*.h' 2>/dev/null \
       | grep -vE ':[0-9]+:\s*(//|\*|/\*)' | grep -v 'rc::cfg::' || true)

# The agent's own src/ is only half the question. Shared units under common/ read config on its
# behalf and are NOT under $agent/src, so a check that stopped here would print a clean bill of
# health while 16 [Presence.*] keys were still being read the old way — an instrument lying
# confidently, which is the one outcome this whole facility exists to prevent.
shared=$(grep -rnE "$pattern" "$root/common" --include='*.cpp' --include='*.h' 2>/dev/null \
         | grep -vE ':[0-9]+:\s*(//|\*|/\*)' | grep -v 'rc::cfg::' | grep -v '/config_report/' || true)

if [ -z "$hits" ]; then
    if [ -z "$shared" ]; then
        echo "$agent: no pre-migration config reads remain, here or in common/ — declare_complete() is honest."
        exit 0
    fi
    echo "$agent: its OWN src/ is clean, but $(echo "$shared" | wc -l) read(s) remain in shared common/ units"
    echo "that read config on this agent's behalf:"
    echo "$shared" | sed 's|^'"$root"'/|  |'
    echo
    echo "declare_complete(\"$agent\") would still be premature: those keys would be named as unread."
    echo "These files are SHARED, so migrating them arms the sweep for every agent that links them."
    exit 1
fi

echo "$agent: $(echo "$hits" | wc -l) config read site(s) still bypass rc::cfg::Reader:"
echo "$hits" | sed 's|^'"$root"'/|  |'
echo
echo "declare_complete(\"$agent\") would arm the unread sweep on an incomplete registry and name"
echo "keys as unread that these sites do read. Migrate them first, or leave the sweep INCONCLUSIVE."
echo
echo "NOTE: the [Presence.*] keys are read in common/agent_presence_monitor/agent_presence_monitor.cpp,"
echo "which is SHARED by 14 agents and is not under $agent/src - this check does not see it. Migrating"
echo "that one file is what lets every one of those agents arm its sweep."
exit 1
