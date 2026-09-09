"""tmux-backed process launch + terminal I/O for netmon v2.

Each component runs inside its own detached tmux session ("nm-<name>") instead of a
plain Popen redirected to a log file. This buys a REAL terminal for free: cursor
addressing, redraws, ANSI color, progress bars -- all rendered correctly by tmux
itself, snapshot-able at any time with `capture-pane`, and steerable with
`send-keys` (including control keys, e.g. C-c). No PTY/WebSocket code of our own,
no new Python dependency -- tmux is an external, already-audited binary.

Deliberately NOT polled in the background: every function here is a one-shot tmux
CLI call, invoked only when the web client asks for it (open a terminal panel /
type into it). Idle cost is whatever tmux itself costs to keep a detached session
around (a few MB), same as leaving a terminal open.
"""

import os
import shlex
import subprocess
import time

import psutil

_PREFIX = "nm-"


def session_name(name: str) -> str:
    return _PREFIX + name


def _run(args, **kw):
    """Every caller in this module treats a nonzero returncode as "not available right
    now" and degrades gracefully (alive()->False, capture()->None, send_*()->False) --
    a missing tmux binary must fail the exact same soft way, not crash the request
    thread with an unhandled FileNotFoundError (see server_v2.term_snapshot: this runs
    on every open terminal panel, so a bare traceback would hit constantly until tmux
    is installed)."""
    try:
        return subprocess.run(["tmux", *args], capture_output=True, text=True, **kw)
    except FileNotFoundError:
        return subprocess.CompletedProcess(args, 127, stdout="", stderr="tmux not found")


def available() -> bool:
    """False if the tmux binary isn't installed -- callers fall back to v1 behavior."""
    try:
        return subprocess.run(["tmux", "-V"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def alive(name: str) -> bool:
    return _run(["has-session", "-t", session_name(name)]).returncode == 0


def pane_pid(name: str) -> int | None:
    r = _run(["list-panes", "-t", session_name(name), "-F", "#{pane_pid}"])
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return int(r.stdout.strip().splitlines()[0])
    except ValueError:
        return None


def kill(name: str) -> None:
    _run(["kill-session", "-t", session_name(name)])


def _foreground_child_pid(bash_pid: int) -> int | None:
    """The command currently running as this shell's foreground job, or None if the
    shell is sitting idle at its prompt (nothing to report -- not an error)."""
    try:
        kids = psutil.Process(bash_pid).children()
    except psutil.NoSuchProcess:
        return None
    return kids[0].pid if kids else None


def ensure_running(name: str, command: str, cwd: str | None, env: dict | None = None,
                   cols: int = 220, rows: int = 50) -> int | None:
    """Idempotent-start. The session's pane runs a PERSISTENT interactive shell (bash,
    sitting at a prompt in `cwd`) rather than `sh -c command` directly -- `command` is
    typed into it via send-keys, exactly like a user would. This is what makes Ctrl-C
    interrupt the command and drop back to a live prompt (still in the component's
    directory) instead of ending the whole session: with a bare `sh -c command`, SIGINT
    kills the command, `sh` has nothing left to do and exits, and tmux (remain-on-exit
    off by default) tears the pane/session down with it.

    Returns the launched command's PID (the actual binary, i.e. bash's foreground child --
    what _launch()'s CPU/mem tracking and stop/restart logic care about), or None on
    failure. If the session already exists with the shell sitting idle (previous run
    exited, or was Ctrl-C'd from the terminal), re-sends `command` into that SAME shell
    instead of recreating the session -- the terminal and its scrollback survive a
    relaunch, matching what a person would expect from a real terminal.
    """
    sess = session_name(name)
    env_prefix = "".join(f"{k}={shlex.quote(str(v))} " for k, v in (env or {}).items())
    shell_cmd = env_prefix + command

    if alive(name):
        bash_pid = pane_pid(name)
        if bash_pid is not None and psutil.pid_exists(bash_pid):
            child = _foreground_child_pid(bash_pid)
            if child is not None:
                return child   # already running this (or something) -- idempotent no-op
            # Shell is alive but idle: re-run the command in place, keep the same terminal.
            if not send_text(name, shell_cmd) or not send_key(name, "Enter"):
                return None
            time.sleep(0.3)
            return _foreground_child_pid(bash_pid)
    kill(name)   # drop any stale/dead session with this name before recreating

    args = ["new-session", "-d", "-s", sess, "-x", str(cols), "-y", str(rows)]
    if cwd:
        args += ["-c", cwd]
    args += [os.environ.get("SHELL", "/bin/bash")]
    r = _run(args)
    if r.returncode != 0:
        return None
    # Headless session: tmux normally sizes a window to the smallest ATTACHED client,
    # which is none here -- "manual" pins the size we asked for at creation so
    # capture-pane keeps returning a stable WxH instead of collapsing to 80x24.
    _run(["set-window-option", "-t", sess, "window-size", "manual"])

    bash_pid = pane_pid(name)
    if bash_pid is None or not send_text(name, shell_cmd) or not send_key(name, "Enter"):
        return None
    time.sleep(0.3)
    return _foreground_child_pid(bash_pid)


def capture(name: str, lines: int = 2000, color: bool = True) -> str | None:
    """Snapshot of the pane's current screen + up to `lines` of scrollback, ANSI
    color codes included (color=True) so the client renders it faithfully. None if
    the session doesn't exist."""
    if not alive(name):
        return None
    args = ["capture-pane", "-p", "-t", session_name(name), "-S", f"-{lines}"]
    if color:
        args.insert(1, "-e")
    r = _run(args)
    return r.stdout if r.returncode == 0 else None


def send_text(name: str, text: str) -> bool:
    """Literal keystrokes (no key-name parsing) -- what the user actually typed."""
    if not text or not alive(name):
        return False
    return _run(["send-keys", "-t", session_name(name), "-l", text]).returncode == 0


def send_key(name: str, key: str) -> bool:
    """A tmux key name: Enter, Tab, Up/Down/Left/Right, BSpace, C-c, C-d, Escape, ..."""
    if not key or not alive(name):
        return False
    return _run(["send-keys", "-t", session_name(name), key]).returncode == 0


def resize(name: str, cols: int, rows: int) -> bool:
    if cols <= 0 or rows <= 0 or not alive(name):
        return False
    return _run(["resize-window", "-t", session_name(name), "-x", str(cols), "-y", str(rows)]).returncode == 0
