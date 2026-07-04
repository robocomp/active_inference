"""Shared /tmp registry so several launchers feed one web monitor.

Each launcher writes its component state to /tmp/robocomp_netmon/agents/<launcher>.json
(atomically, once per tick). Whichever launcher grabs the monitor.lock first runs the
web server and reads *all* registries; the others just keep updating their file. If the
monitor holder dies, the next launcher to tick steals the stale lock and takes over.
"""

import json
import os
import tempfile
import time
import uuid

import psutil

REG_DIR = "/tmp/robocomp_netmon"
AGENTS_DIR = os.path.join(REG_DIR, "agents")
COMMANDS_DIR = os.path.join(REG_DIR, "commands")
LOCK_PATH = os.path.join(REG_DIR, "monitor.lock")
STALE_AFTER = 5.0  # a registry/lock older than this is considered dead


def _ensure_dirs():
    os.makedirs(AGENTS_DIR, exist_ok=True)
    os.makedirs(COMMANDS_DIR, exist_ok=True)


def _atomic_write(path, data):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass


class RegistryWriter:
    def __init__(self, launcher, layer):
        _ensure_dirs()
        self.launcher = launcher
        self.layer = layer
        self.path = os.path.join(AGENTS_DIR, f"{launcher}.json")

    def update(self, components):
        _atomic_write(self.path, {
            "launcher": self.launcher,
            "layer": self.layer,
            "ts": time.time(),
            "pid": os.getpid(),
            "components": components,
        })

    def remove(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass


def read_registries(stale_after=STALE_AFTER):
    out = {}
    if not os.path.isdir(AGENTS_DIR):
        return out
    now = time.time()
    for fn in os.listdir(AGENTS_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(AGENTS_DIR, fn)) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if now - d.get("ts", 0) > stale_after:
            continue
        out[d.get("launcher", fn)] = d
    return out


def merged_components(stale_after=STALE_AFTER):
    """Flat list of every component across all live launchers."""
    comps = []
    for d in read_registries(stale_after).values():
        for c in d.get("components", []):
            c = dict(c)
            c.setdefault("layer", d.get("layer"))
            c.setdefault("launcher", d.get("launcher"))
            comps.append(c)
    return comps


# ---- command mailbox: the web monitor pushes, the owning launcher claims & executes ----

def push_command(action, name):
    _ensure_dirs()
    path = os.path.join(COMMANDS_DIR, f"{uuid.uuid4().hex}.json")
    _atomic_write(path, {"action": action, "name": name, "ts": time.time()})


def claim_commands(names, stale_after=30.0):
    """Return (and remove) pending commands targeting any of `names` (this launcher's)."""
    out = []
    if not os.path.isdir(COMMANDS_DIR):
        return out
    now = time.time()
    for fn in os.listdir(COMMANDS_DIR):
        p = os.path.join(COMMANDS_DIR, fn)
        try:
            with open(p) as f:
                c = json.load(f)
        except (OSError, ValueError):
            try: os.unlink(p)
            except OSError: pass
            continue
        if now - c.get("ts", 0) > stale_after:          # drop orphan commands
            try: os.unlink(p)
            except OSError: pass
            continue
        if c.get("name") in names:
            try:
                os.unlink(p)                             # claim it (atomic)
            except OSError:
                continue                                 # someone else took it
            out.append(c)
    return out


class MonitorLock:
    """Single-holder lock for the web server role, with dead-holder takeover."""

    def __init__(self):
        _ensure_dirs()
        self.owned = False

    def holder(self):
        try:
            with open(LOCK_PATH) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def alive(self):
        h = self.holder()
        if not h:
            return False
        pid = h.get("pid")
        if pid == os.getpid():
            return True
        if not psutil.pid_exists(pid):
            return False
        return (time.time() - h.get("ts", 0)) <= STALE_AFTER

    def try_acquire(self, port, _retry=True):
        if self.alive():
            return False
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if _retry and not self.alive():          # stale lock -> steal it
                try:
                    os.unlink(LOCK_PATH)
                except OSError:
                    pass
                return self.try_acquire(port, _retry=False)
            return False
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "port": port, "ts": time.time()}, f)
        self.owned = True
        return True

    def refresh(self, port):
        if self.owned:
            _atomic_write(LOCK_PATH, {"pid": os.getpid(), "port": port, "ts": time.time()})

    def release(self):
        if self.owned:
            try:
                os.unlink(LOCK_PATH)
            except OSError:
                pass
            self.owned = False
