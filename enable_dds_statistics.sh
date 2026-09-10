#!/usr/bin/env bash
# Arm both DDS-statistics gates for the dsr_gui "mind" node real-time traffic view.
#
# TWO ways to use it — pick whichever fits; both set the SAME two env vars:
#
#   1. As a LAUNCHER (recommended — no "did I source it?" foot-gun):
#          ./enable_dds_statistics.sh bin/robot_concept etc/config_beta.toml
#      It exports the vars and exec()s your command, so the launched process
#      (and the mind view it hosts) inherits them. Works by *executing* the file.
#
#   2. SOURCED into your current shell (then launch things yourself):
#          source ./enable_dds_statistics.sh
#          bin/robot_concept etc/config_beta.toml
#
# Two independent gates, both armed here (either alone = no labels):
#   FASTDDS_STATISTICS  — PRODUCER side. Each agent must have it BEFORE it creates
#     its DomainParticipant. No-op unless FastDDS was built -DFASTDDS_STATISTICS=ON.
#   RC_DDS_STATS=1      — CONSUMER side. Read by the mind widget's DdsStatsMonitor at
#     construction; without it the monitor stays disabled even though compiled in.
#     Must be in the environment of the process that HOSTS the graph viewer.

# ── the two gates ──────────────────────────────────────────────────────────
# Mind-node feed: throughput + discovery/topology. MONITOR_SERVICE_TOPIC gives the
# participant/endpoint proxy graph; the throughput topics carry the byte rates the
# in/out labels show. More topics = more overhead on the DDS bus.
export FASTDDS_STATISTICS="MONITOR_SERVICE_TOPIC;DISCOVERY_TOPIC;PUBLICATION_THROUGHPUT_TOPIC;SUBSCRIPTION_THROUGHPUT_TOPIC"
export RC_DDS_STATS=1

# ── source vs execute dispatch ─────────────────────────────────────────────
# (return works only in a sourced file) → detect how we were invoked.
(return 0 2>/dev/null) && _sourced=1 || _sourced=0

if [ "$_sourced" = "1" ]; then
    echo "[dds-stats] sourced — vars set in current shell:"
    echo "           FASTDDS_STATISTICS=${FASTDDS_STATISTICS}"
    echo "           RC_DDS_STATS=${RC_DDS_STATS}"
else
    if [ "$#" -gt 0 ]; then
        echo "[dds-stats] launching with stats env: $*"
        echo "           RC_DDS_STATS=${RC_DDS_STATS}  FASTDDS_STATISTICS set"
        exec "$@"                       # replace this process → child inherits the env
    else
        echo "[dds-stats] ERROR: executed with no command, so the exported vars die with"
        echo "            this subshell and reach nothing. Use one of:"
        echo "              ./$(basename "$0") <command...>   # e.g. bin/robot_concept etc/config_beta.toml"
        echo "              source ./$(basename "$0")         # then launch in this shell"
        exit 1
    fi
fi
