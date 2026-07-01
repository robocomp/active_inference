#!/usr/bin/env python3
"""RoboComp sensorimotor launcher + monitor v2.

Launches the components in sub.toml, shows a rich console table, and publishes their
state to the shared /tmp registry. With --web (default) it also serves the unified
interconnection map (topology + per-connection bandwidth) if no other launcher already
runs the monitor. Run cognitive_v2.py alongside to add the cognitive agents + DSR sphere.
"""

import argparse

from netmon.launcher import run_launcher

parser = argparse.ArgumentParser(description="RoboComp sensorimotor launcher + monitor v2")
parser.add_argument("file_name", nargs="?", default="sub.toml", help="Path to TOML components file")
parser.add_argument("--no-web", dest="web", action="store_false", help="Do not serve/host the web monitor")
parser.add_argument("--web-port", type=int, default=8080)
parser.add_argument("--iface", default="lo", help="Interface to sniff for bandwidth")
parser.add_argument("--no-bw", action="store_true", help="Disable loopback bandwidth capture")
parser.add_argument("--no-webots", dest="webots", action="store_false", help="Do not autostart Webots")
args = parser.parse_args()

run_launcher(args.file_name, launcher="subcognitive", layer="sensorimotor",
             start_webots=args.webots, web=args.web, web_port=args.web_port,
             iface=args.iface, no_bw=args.no_bw)
