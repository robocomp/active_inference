#!/usr/bin/env bash
# Static A/B for the YOLO-score → covariance effect (Part A verification).
#
#   ►► HOLD THE ROBOT STILL, pointed at a STATIC bottle, for the whole test. ◄◄
#
# Runs bottle_concept twice back-to-back on the SAME frozen scene, differing ONLY in MaskConfWeight:
#   A_on  : MaskConfWeight=true   (YOLO score down-weights obs precision)  → etc/bottle_ab_on.csv
#   A_off : MaskConfWeight=false  (w≡1, byte-exact legacy)                 → etc/bottle_ab_off.csv
# Because nothing moves between the two runs, conf is ~identical, so any σ difference is the score.
#
# Prediction: with conf≈0.85 → w≈0.72 (Ref=0.95 in config.toml), the weighted run carries ~28% less
# observation precision, so   σ_on / σ_off ≈ 1/sqrt(w) ≈ 1.18×   (a ~18% wider belief).
#
# Needs the full stack up (robot/room/voxelizer + Webots + bridges) and the convergence fix so the fit
# latches (frames_converged → K_stable). Mirrors bottle_static_test.sh conventions.
#
#   ./bottle_static_ab.sh [run_secs]      (default 25s/condition; fit settles in ~10-12s, +margin for K_stable=30)
set -u
cd "$(dirname "$0")"

RUN_SECS="${1:-25}"
GAP=6                       # gap after clean SIGTERM exit (DDS participant removal propagates)
BASE=etc/config.toml
K_STABLE=30                 # must match BottleModel/K_stable; rows at this fconv are "converged"

pkill -TERM -x bottle_concept 2>/dev/null; sleep 4; pkill -9 -x bottle_concept 2>/dev/null

run_condition() {
    local name="$1" weight="$2"
    local cfg="etc/config_ab_${name}.toml"
    local csv="etc/bottle_ab_${name}.csv"
    # Derive a config from the live base: force MaskConfWeight + a dedicated fisher CSV (everything else
    # — Ref/Floor/Power, DetectionNoiseM, the scene — is inherited unchanged).
    sed -e "s/^MaskConfWeight *=.*/MaskConfWeight = ${weight}/" \
        -e "s#^FisherCsvPath *=.*#FisherCsvPath = \"${csv}\"#" \
        "$BASE" > "$cfg"
    rm -f "$csv"
    echo "=== A_${name}: MaskConfWeight=${weight} → ${csv}  (${RUN_SECS}s) ==="
    setsid ./bin/bottle_concept "$cfg" > "/tmp/bottle_ab_${name}.log" 2>&1 < /dev/null &
    local pid=$!
    sleep "$RUN_SECS"
    kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 10); do pgrep -x bottle_concept >/dev/null || break; sleep 1; done
    pkill -9 -x bottle_concept 2>/dev/null
    grep -q "already an agent connected" "/tmp/bottle_ab_${name}.log" 2>/dev/null && \
        echo "  WARN: id-collision (increase GAP)"
    sleep "$GAP"
}

echo "static A/B: ${RUN_SECS}s/condition — KEEP THE ROBOT STILL until both runs finish."
run_condition on  true
run_condition off false

# ── Steady-state σ over the most-converged rows (max frames_converged seen; ideally == K_stable). ──
metrics() {  # prints: fconv n conf w cx cy r h   (means in mm over rows at max frames_converged)
    awk -F, 'NR>1{ if($5>mx)mx=$5; L[NR]=$0 }
        END{ for(i in L){ split(L[i],a,",");
                 if(a[5]==mx){ n++; cf+=a[7]; w+=a[8]; cx+=a[29]; cy+=a[30]; r+=a[32]; h+=a[33] } }
             if(n==0){ print "0 0 0 0 0 0 0 0"; exit }
             printf "%d %d %.3f %.3f %.4f %.4f %.4f %.4f\n", mx,n,cf/n,w/n,cx/n,cy/n,r/n,h/n }' "$1"
}

echo
echo "=== steady-state σ (mm), mean over rows at max frames_converged ==="
read -r f_on  n_on  conf_on  w_on  cx_on  cy_on  r_on  h_on  < <(metrics etc/bottle_ab_on.csv)
read -r f_off n_off conf_off w_off cx_off cy_off r_off h_off < <(metrics etc/bottle_ab_off.csv)
printf "A_on  : fconv=%s n=%s conf=%s w=%s  σ(cx,cy,r,h)mm = %s %s %s %s\n" \
       "$f_on"  "$n_on"  "$conf_on"  "$w_on"  "$cx_on"  "$cy_on"  "$r_on"  "$h_on"
printf "A_off : fconv=%s n=%s conf=%s w=%s  σ(cx,cy,r,h)mm = %s %s %s %s\n" \
       "$f_off" "$n_off" "$conf_off" "$w_off" "$cx_off" "$cy_off" "$r_off" "$h_off"

[ "$f_on"  -lt "$K_STABLE" ] 2>/dev/null && echo "  ⚠ A_on never reached K_stable=$K_STABLE (fit not converged — robot moved? longer run?)"
[ "$f_off" -lt "$K_STABLE" ] 2>/dev/null && echo "  ⚠ A_off never reached K_stable=$K_STABLE (fit not converged — robot moved? longer run?)"

awk -v on="$cx_on" -v off="$cx_off" -v w="$w_on" 'BEGIN{
    if(off>0) printf "→ σcx ratio on/off = %.3f   (predicted 1/sqrt(w) = %.3f)\n", on/off, (w>0?1/sqrt(w):0)
    else      print  "→ no off baseline σ (off run did not converge)"
}'
echo "logs: /tmp/bottle_ab_on.log  /tmp/bottle_ab_off.log    configs: etc/config_ab_{on,off}.toml"
