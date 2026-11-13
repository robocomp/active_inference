#!/bin/bash
# scaffold_from_refrigerator.sh <obj> <agent_id> — CONCEPT_AGENT_RECIPE.md "Scaffolding a new agent", steps 2-3, mechanised.
# Copies refrigerator_concept's TRACKED files (never build/, bin/, logs or instance_identity.csv) to <obj>_concept,
# renames the three token cases in file names AND contents (refrigerator/Refrigerator/REFRIGERATOR and the short form
# fridge/Fridge/FRIDGE), and renumbers [Agent] id. Everything object-specific (manifest, priors, mesh, contract case,
# σ*) is done by hand afterwards — this script only does what must be identical across clones.
# Run from the repo root. Refuses to overwrite an existing <obj>_concept.
set -euo pipefail
OBJ=${1:?usage: scaffold_from_refrigerator.sh <obj> <agent_id>}; ID=${2:?agent id}
Obj="${OBJ^}"; OBJU="${OBJ^^}"
SRC=refrigerator_concept; DST=${OBJ}_concept
[ -e "$DST" ] && { echo "$DST exists — refusing"; exit 1; }
grep -rqE "^\s*id\s*=\s*$ID\b" */etc/config.toml && { echo "id $ID already taken"; exit 1; }
git ls-files "$SRC" | while read -r f; do
  rel=${f#$SRC/}
  case "$rel" in meshes/*) continue;; esac          # the display mesh is the object's own, added by hand
  out="$DST/$(echo "$rel" | sed -e "s/refrigerator/$OBJ/g; s/fridge/$OBJ/g")"
  mkdir -p "$(dirname "$out")"
  if file -b --mime "$f" | grep -q "^text/"; then
    sed -e "s/REFRIGERATOR/$OBJU/g; s/Refrigerator/$Obj/g; s/refrigerator/$OBJ/g" \
        -e "s/FRIDGE/$OBJU/g; s/Fridge/$Obj/g; s/fridge/$OBJ/g" "$f" > "$out"
  else
    cp "$f" "$out"
  fi
done
sed -i -E "0,/^\s*id\s*=.*/s//id = $ID  # unique across the shared graph — see CONCEPT_AGENT_RECIPE.md agent-ID registry/" "$DST/etc/config.toml"
# [Owns] names the agent node by id too: a stale id there makes the presence cleanup miss the agent's own node.
sed -i -E "s/\"${OBJ}_concept [0-9]+\"/\"${OBJ}_concept $ID\"/" "$DST/etc/config.toml"
chmod +x "$DST"/*.sh 2>/dev/null || true
echo "$DST scaffolded (id $ID). Next: manifest, contract case, mesh, priors, σ* — CONCEPT_AGENT_RECIPE.md step 6."
