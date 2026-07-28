#!/usr/bin/env python3
"""RoboComp cognitive launcher + monitor v2.

Launches the concept agents in cognitive.toml, shows a rich console table, and publishes
their state to the shared /tmp registry (feeding the same web monitor as subcognitive_v2).
Cognitive agents talk over the DSR graph: each is grouped under a symbolic DSR sphere by
its [Agent] domain, on top of any ICE edges they have toward the sensorimotor layer.

Assumes the sensorimotor base is already up (subcognitive_v2.py sub.toml). Whichever
launcher grabs the monitor lock first serves the web; this one otherwise just updates.
"""

import argparse

from netmon.launcher import run_launcher

parser = argparse.ArgumentParser(description="RoboComp cognitive launcher + monitor v2")
parser.add_argument("file_name", nargs="?", default="cognitive.toml", help="Path to TOML components file")
parser.add_argument("--no-web", dest="web", action="store_false", help="Do not serve/host the web monitor")
parser.add_argument("--web-port", type=int, default=8080)
parser.add_argument("--web-host", default="0.0.0.0", help="Bind address for the web monitor (0.0.0.0 = all interfaces)")
parser.add_argument("--iface", default="lo", help="Interface to sniff for bandwidth")
parser.add_argument("--no-bw", action="store_true", help="Disable loopback bandwidth capture")
args = parser.parse_args()

run_launcher(args.file_name, launcher="cognitive", layer="cognitive",
             start_webots=False, web=args.web, web_port=args.web_port,
             web_host=args.web_host, iface=args.iface, no_bw=args.no_bw)
