#!/usr/bin/env bash
# Refactor guardrail (docs/refactor_proposal_2026-10-13.md §5): no source file over 800 lines, no function
# over 300, and nothing in the lower layers (core/solver/layout/localizer) may name DSR or Qt widgets.
#
# During the migration the current offenders are grandfathered in tools/lint_allow.txt, each with its
# size today. An allowed offender may SHRINK but never GROW; a new offender fails. The list must be
# empty when step 6 of the plan is done.
#
#   tools/lint_file_lengths.sh                 check (exit 1 on any violation)
#   tools/lint_file_lengths.sh --write-allow   rewrite the allow-list from the CURRENT tree — run it only
#                                              after a step that shrank files, never to admit growth
set -euo pipefail
cd "$(dirname "$0")/.."

FILE_MAX=800
FUNC_MAX=300
ALLOW=tools/lint_allow.txt

mapfile -t SOURCES < <(find src -name '*.cpp' -o -name '*.h' | grep -v '\.bak' | LC_ALL=C sort)

current() {   # prints "file <path> <n>" and "func <path>:<name> <n>" for every offender
    for f in "${SOURCES[@]}"; do
        n=$(wc -l < "$f")
        (( n > FILE_MAX )) && echo "file $f $n"
    done
    python3 -I tools/lint_function_lengths.py "$FUNC_MAX" "${SOURCES[@]}" | while read -r n key; do
        echo "func $key $n"
    done
}

if [[ "${1:-}" == "--write-allow" ]]; then
    {
        echo "# Grandfathered offenders: kind path[:function] lines-at-recording. May shrink, never grow."
        echo "# Regenerate with tools/lint_file_lengths.sh --write-allow ONLY after a step that shrank them."
        current | LC_ALL=C sort
    } > "$ALLOW"
    echo "wrote $ALLOW ($(grep -vc '^#' "$ALLOW") entries)"
    exit 0
fi

declare -A allowed
if [[ -f "$ALLOW" ]]; then
    while read -r kind key n; do
        [[ -z "$kind" || "$kind" == \#* ]] && continue
        allowed["$kind $key"]=$n
    done < "$ALLOW"
fi

bad=0
declare -A seen
while read -r kind key n; do
    seen["$kind $key"]=1
    limit=${allowed["$kind $key"]:-}
    if [[ -z "$limit" ]]; then
        echo "TOO LONG ($kind, new): $key = $n lines (max $([[ $kind == file ]] && echo $FILE_MAX || echo $FUNC_MAX))"
        bad=1
    elif (( n > limit )); then
        echo "GREW ($kind, allow-listed at $limit): $key = $n lines"
        bad=1
    fi
done < <(current)

for k in "${!allowed[@]}"; do
    [[ -z "${seen[$k]:-}" ]] && echo "note: '$k' is now within the limit — remove it from $ALLOW"
done

# Layering: once the library directories exist, nothing below agent/ may name DSR or widgets.
layer_dirs=()
for d in src/core src/solver src/layout src/localizer; do [[ -d $d ]] && layer_dirs+=("$d"); done
if (( ${#layer_dirs[@]} )); then
    if grep -rnE 'dsr/|DSR::|G->|QWidget|QGraphics' "${layer_dirs[@]}"; then
        echo "LAYERING: DSR / Qt-widget use below agent/ (see above)"
        bad=1
    fi
fi

(( bad == 0 )) && echo "lint: OK (${#allowed[@]} grandfathered offenders)"
exit $bad
