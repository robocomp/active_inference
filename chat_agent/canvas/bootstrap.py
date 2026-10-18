"""bootstrap.py — how chat_agent runs a canvas script (canvas_runner.cpp).

    python3 -I bootstrap.py <script.py> [args...]      cwd = the run dir (snapshot.json, zed.jpg, ...)

-I (isolated) ignores PYTHON* env vars, the user site and the script's own directory, so nothing planted next to
the script is imported. This file puts ONLY the canvas dir (rcanvas.py) on the path, closes the network at the
socket level (a best-effort guard against a model-written script reaching out; canvas_runner adds a network
namespace when the kernel allows one), then runs the script as __main__ with sys.argv = [script, args...].
"""
import os
import runpy
import socket
import sys

CANVAS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CANVAS_DIR)


def _no_network(*_a, **_k):
    raise PermissionError("canvas scripts have no network access")


for _name in ("connect", "connect_ex", "bind", "sendto"):
    setattr(socket.socket, _name, _no_network)
socket.create_connection = _no_network
socket.getaddrinfo = _no_network

if len(sys.argv) < 2:
    sys.exit("usage: bootstrap.py <script.py> [args...]")
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
