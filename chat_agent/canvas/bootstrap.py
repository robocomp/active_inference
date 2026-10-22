"""bootstrap.py — how chat_agent runs a canvas script (canvas_runner.cpp).

    python3 -I bootstrap.py <script.py> [args...]           one run       cwd = the run dir (snapshot.json, zed.jpg, ...)
    python3 -I bootstrap.py --serve <script.py> [args...]   live worker: one run per stdin line (see _serve)

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



def _serve(script, args):
    """--serve: a LIVE drawing's worker. Runs `script` once per line read on stdin and answers one line on the
    protocol fd: "ok" or "err <traceback, newlines escaped>". The interpreter, matplotlib and numpy stay loaded, so
    a redraw costs the drawing only (~45 ms for stock/layout.py against ~350 ms for a fresh process). One worker per
    live drawing: the session starts a new one when the drawing changes, so no state outlives it. Anything the
    script prints goes to stderr (fd 1 is pointed there), never into the protocol."""
    import traceback
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    for _ in sys.stdin:
        sys.argv = [script] + args
        try:
            runpy.run_path(script, run_name="__main__")
            reply = "ok"
        except BaseException:   # SystemExit included: a script that calls exit() must not end the worker
            reply = "err " + traceback.format_exc()[-2000:].replace("\\", "\\\\").replace("\n", "\\n")
        finally:
            if "matplotlib.pyplot" in sys.modules:
                sys.modules["matplotlib.pyplot"].close("all")
        proto.write(reply + "\n")


if len(sys.argv) < 2:
    sys.exit("usage: bootstrap.py [--serve] <script.py> [args...]")
if sys.argv[1] == "--serve":
    _serve(sys.argv[2], sys.argv[3:])
else:
    script = sys.argv[1]
    sys.argv = [script] + sys.argv[2:]
    runpy.run_path(script, run_name="__main__")
