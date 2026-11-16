#!/usr/bin/env bash
# clean_memory.sh — wipe ltsm_agent's stored long-term memory so the next run starts fresh.
#
# Removes the memory file named by [Memory] persist_path in the config (default etc/config.toml) and its
# previous generation (<file>.prev). With --all, also the hand-made backups <file>.bak_*.
# Refuses while an ltsm_agent is running: it would write the memory back on its next save.
#
#   tools/clean_memory.sh                 # etc/ltsm_memory.json + .prev
#   tools/clean_memory.sh --all           # ... and etc/ltsm_memory.json.bak_*
#   tools/clean_memory.sh -c etc/x.toml   # use another config's persist_path
#   tools/clean_memory.sh -n              # dry run: list what would go
set -euo pipefail

cd "$(dirname "$0")/.."

config=etc/config.toml
all=false
dry=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config) config="$2"; shift 2 ;;
        -a|--all)    all=true; shift ;;
        -n|--dry-run) dry=true; shift ;;
        -h|--help)   sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if pids=$(pgrep -x ltsm_agent); then
    echo "ltsm_agent is running (pid $(echo $pids)); stop it first with SIGTERM (kill $pids), never kill -9." >&2
    exit 1
fi

# persist_path from the [Memory] section of the config; fall back to the default.
path=$(awk '/^\[/{s=$0} s=="[Memory]" && /^[[:space:]]*persist_path[[:space:]]*=/{
             sub(/^[^=]*=[[:space:]]*"?/, ""); sub(/"?[[:space:]]*(#.*)?$/, ""); print; exit}' "$config" 2>/dev/null || true)
path=${path:-etc/ltsm_memory.json}

files=()
for f in "$path" "$path.prev"; do [[ -e "$f" ]] && files+=("$f"); done
if $all; then
    for f in "$path".bak_*; do [[ -e "$f" ]] && files+=("$f"); done
fi

if [[ ${#files[@]} -eq 0 ]]; then
    echo "nothing to remove (memory: $path)"
    exit 0
fi

for f in "${files[@]}"; do
    if $dry; then echo "would remove $f"; else rm -- "$f"; echo "removed $f"; fi
done
