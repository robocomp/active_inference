"""FastDDS shared-memory hygiene, shared by every launcher.

Why this exists (2026-07-21): an unclean shutdown leaves orphaned /dev/shm/fastdds_port* segments
behind. New participants attach to the dead ring buffer and discovery messages go nowhere —
SILENTLY, with no error on either side. The whole zero-copy media plane then reads 0.0 Hz while
anything carried over Ice/TCP keeps working, so it looks exactly like a code regression.

Two layers, both meant to run at launch time while our own processes are known to be down:
  clean_orphan_shm()  removes the known cause
  preflight_dds()     verifies SHM discovery really works, so anything else fails LOUDLY

Kept free of heavy imports (no Ice, no scapy, no netmon package) so a standalone launcher can load
this module by file path without dragging in the rest of netmon.
"""

import glob
import os
import subprocess

# utils/netmon/shm_guard.py -> utils/dds_preflight/dds_preflight
_PREFLIGHT_EXE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dds_preflight", "dds_preflight")


def shm_segments_in_use():
    """Paths of /dev/shm FastDDS segments some live process still has mapped or open.

    Read from /proc rather than from a process list: a segment is "in use" if and only if a running
    process references it, which is exactly the condition that makes deleting it unsafe.
    """
    in_use = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/maps", "rb") as fh:
                for line in fh:
                    idx = line.find(b"/dev/shm/fastdds")
                    if idx != -1:
                        in_use.add(line[idx:].strip().decode("utf-8", "replace"))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
        try:
            fd_dir = f"/proc/{pid}/fd"
            for fd in os.listdir(fd_dir):
                try:
                    target = os.readlink(os.path.join(fd_dir, fd))
                except OSError:
                    continue
                if target.startswith("/dev/shm/fastdds"):
                    in_use.add(target)
        except (FileNotFoundError, PermissionError, NotADirectoryError, ProcessLookupError):
            pass
    return in_use


def clean_orphan_shm(console):
    """Delete FastDDS shared-memory segments left behind by an unclean shutdown.

    Only segments NO live process references are removed. That check is not optional: the fleet
    spans several launchers, and a blind wipe here would tear down a peer fleet that is already
    running.
    """
    paths = sorted(glob.glob("/dev/shm/fastdds_*"))
    if not paths:
        return
    in_use = shm_segments_in_use()
    removed, kept = 0, 0
    for path in paths:
        if path in in_use:
            kept += 1
            continue
        try:
            os.unlink(path)
            removed += 1
        except OSError:
            kept += 1
    if removed:
        msg = f"[yellow][shm] removed {removed} orphaned FastDDS segment(s)"
        msg += f"; {kept} still in use, kept[/yellow]" if kept else "[/yellow]"
        console.print(msg)
    elif kept:
        console.print(f"[dim][shm] {kept} FastDDS segment(s) in use by live processes, none orphaned[/dim]")


def preflight_dds(console, domain=7):
    """Verify SHM discovery works on the media domain before any producer is started.

    clean_orphan_shm() removes the known cause; this catches everything else, turning a silent
    0.0 Hz fleet into a loud refusal to start. Returns True when it is safe to launch.

    Skipped (never failed) when the tool is not built, so a missing binary can never block a launch.
    """
    if not os.path.exists(_PREFLIGHT_EXE):
        console.print(f"[dim][shm] preflight skipped (not built: {_PREFLIGHT_EXE})[/dim]")
        return True
    try:
        # Belt and braces: the tool self-bounds with an internal watchdog, but never let a wedged
        # SHM stack hang the launcher.
        res = subprocess.run([_PREFLIGHT_EXE, str(domain), "2000"],
                             capture_output=True, text=True, timeout=20)
    except (subprocess.TimeoutExpired, OSError) as e:
        console.print(f"[red][shm] preflight could not run ({e}) — continuing without the check[/red]")
        return True
    if res.returncode == 0:
        console.print(f"[green][shm] DDS preflight OK on domain {domain}[/green]")
        return True
    for line in (res.stdout or "").strip().splitlines():
        console.print(f"[red]{line}[/red]")
    console.print(f"[bold red][shm] PREFLIGHT FAILED on domain {domain} — refusing to start: "
                  f"the media plane would silently read 0.0 Hz.[/bold red]")
    return False
