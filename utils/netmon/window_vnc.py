"""On-demand, per-component X11 window capture bridged to the browser via VNC.

For a selected component, resolves its window id (x11_windows.find_window_for_pid)
and launches a PRIVATE `x11vnc -id <id>` scoped to just that window, plus a
`websockify` process bridging it to a WebSocket port noVNC (static/novnc/) can
connect to directly. Both are started lazily -- only when a window panel is
actually opened -- and stopped when the panel closes: no VNC server sits around
for a window nobody is looking at, same "cost only while open" rule as the tmux
terminal (see term_tmux.py).

No VNC password (explicit choice for this deployment: controlled LAN, ease of use
over defense in depth -- revisit if this ever runs somewhere less trusted).
"""

import socket
import subprocess
import time

from . import x11_windows

_DISPLAY = ":1"          # where retina/room_concept/etc actually open their windows
_BASE_VNC_PORT = 15900   # x11vnc <-> websockify, loopback only
_BASE_WS_PORT = 16900    # websockify <-> browser, needs to be reachable from the LAN


def _free_port(base):
    for port in range(base, base + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"no free port in [{base}, {base + 200})")


def _wait_for_port(port, timeout=2.0, host="127.0.0.1"):
    """Poll until something is actually listening, instead of a blind sleep(). Both
    x11vnc and websockify take a real (if usually brief) moment to bind their socket
    after Popen() returns -- returning "ok" to the browser before the LAST hop
    (websockify) is confirmed listening was a genuine race: the very first RFB
    connection attempt landed before websockify existed, failed with 1006, and only
    a second attempt (e.g. a double-click retry) succeeded because by then it was
    up. This closes that race for real instead of relying on the user to retry."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.05)
    return False


class _WindowSession:
    def __init__(self, name, pid, window_id):
        self.name = name
        self.pid = pid
        self.window_id = window_id
        self.ws_port = None
        self._vnc = None
        self._ws = None

    @property
    def alive(self):
        return (self._vnc is not None and self._vnc.poll() is None
                and self._ws is not None and self._ws.poll() is None)

    def start(self):
        if self.alive:
            return True
        wid = self.window_id
        vnc_port = _free_port(_BASE_VNC_PORT)
        self.ws_port = _free_port(_BASE_WS_PORT)
        self._vnc = subprocess.Popen(
            ["x11vnc", "-display", _DISPLAY, "-id", str(wid), "-rfbport", str(vnc_port),
             "-nopw", "-forever", "-shared", "-noxdamage", "-quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not _wait_for_port(vnc_port):
            self.stop()   # x11vnc never bound (stale/invalid window id, etc.)
            return False
        self._ws = subprocess.Popen(
            ["websockify", f"0.0.0.0:{self.ws_port}", f"localhost:{vnc_port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not _wait_for_port(self.ws_port):
            self.stop()
            return False
        return True

    def stop(self):
        # terminate() alone doesn't confirm the process actually died -- if either
        # one ignores/delays SIGTERM (seen in practice: a websockify outliving its
        # already-dead x11vnc after a close), it leaks as an orphan holding the port,
        # since dropping the Popen reference below means nothing will ever wait() on
        # it again. Give both a moment, then SIGKILL whatever's still alive.
        procs = [p for p in (self._vnc, self._ws) if p and p.poll() is None]
        for p in procs:
            p.terminate()
        deadline = time.time() + 2.0
        for p in procs:
            remaining = deadline - time.time()
            try:
                p.wait(timeout=max(0, remaining))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=1)
        self._vnc = self._ws = None


class WindowManager:
    """One _WindowSession per component name, created the first time it's opened."""

    def __init__(self):
        self._sessions = {}

    def list_windows(self, pid):
        """[(window_id, title), ...] -- a component can genuinely have more than one
        top-level window (e.g. retina: its main view + the Ricoh 360 panorama), so
        the caller (server_v2.py) offers a picker instead of silently guessing."""
        return x11_windows.windows_for_pid(pid, _DISPLAY)

    def open(self, name, pid, window_id=None):
        """Returns the websocket port to connect noVNC to, or None if the requested
        window couldn't be found (component not running, hasn't opened a window
        yet, or -- with no window_id given -- has none at all: all real,
        not-an-error conditions). window_id=None picks whichever window this
        component's PID owns first (fine for the common single-window case)."""
        if window_id is None:
            wins = self.list_windows(pid)
            if not wins:
                return None
            window_id = wins[0][0]
        sess = self._sessions.get(name)
        if sess is None or sess.pid != pid or sess.window_id != window_id:
            if sess:
                sess.stop()
            sess = self._sessions[name] = _WindowSession(name, pid, window_id)
        return sess.ws_port if sess.start() else None

    def close(self, name):
        sess = self._sessions.pop(name, None)
        if sess:
            sess.stop()

    def shutdown(self):
        for sess in self._sessions.values():
            sess.stop()
        self._sessions.clear()
