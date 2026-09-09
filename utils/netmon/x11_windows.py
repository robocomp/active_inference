"""Resolve a component's X11 window id from its PID.

Reads the standard EWMH properties (_NET_CLIENT_LIST on the root window,
_NET_WM_PID on each top-level window) -- read-only, no window is touched, no
window manager cooperation beyond what any EWMH-compliant WM (xfce, in this
deployment) already publishes. Used to target `x11vnc -id <id>` at exactly the
GUI window a given component (or one of its child processes -- Qt apps launched
via `vglrun`/`chrt` wrappers report a different top-level PID than the window's
actual owner) opened, instead of serving the whole virtual desktop.
"""

import psutil
from Xlib import X, display


def _descendant_pids(pid):
    pids = {pid}
    try:
        pids.update(p.pid for p in psutil.Process(pid).children(recursive=True))
    except psutil.NoSuchProcess:
        pass
    return pids


def _client_list(root, net_client_list):
    prop = root.get_full_property(net_client_list, X.AnyPropertyType)
    return list(prop.value) if prop else []


def _geometry(win):
    """(width, height) of the window's actual content area, or (None, None) if the
    query fails (window closed mid-call, etc.) -- used so the browser panel can be
    sized to the real thing instead of an arbitrary fixed box, and so x11vnc's
    capture isn't scaled/cropped relative to what's actually there."""
    try:
        geom = win.get_geometry()
        return geom.width, geom.height
    except Exception:
        return None, None


def windows_for_pid(pid, display_name=":1"):
    """[(window_id, title, width, height), ...] for every top-level window owned by
    `pid` or one of its descendants (Qt apps launched via `vglrun`/`chrt` wrappers
    report a different top-level PID than the window's actual owner, and some
    components genuinely open more than one window -- e.g. retina's main view +
    its Ricoh 360 panorama viewer share one PID). Empty list, not None, if nothing
    matches."""
    pids = _descendant_pids(pid)
    d = display.Display(display_name)
    try:
        root = d.screen().root
        net_client_list = d.intern_atom("_NET_CLIENT_LIST")
        net_wm_pid = d.intern_atom("_NET_WM_PID")
        net_wm_name = d.intern_atom("_NET_WM_NAME")
        utf8 = d.intern_atom("UTF8_STRING")
        out = []
        for wid in _client_list(root, net_client_list):
            win = d.create_resource_object("window", wid)
            wpid = win.get_full_property(net_wm_pid, X.AnyPropertyType)
            if wpid and wpid.value and wpid.value[0] in pids:
                name = win.get_full_property(net_wm_name, utf8)
                title = bytes(name.value).decode("utf-8", "replace") if name else f"0x{wid:x}"
                width, height = _geometry(win)
                out.append((wid, title, width, height))
        return out
    finally:
        d.close()


def find_window_for_pid(pid, display_name=":1"):
    """First top-level window owned by `pid` or one of its descendants, or None.
    Convenience wrapper for the common single-window case -- see windows_for_pid()
    for components that open more than one."""
    wins = windows_for_pid(pid, display_name)
    return wins[0][0] if wins else None


def list_windows(display_name=":1"):
    """[(window_id, pid, title), ...] for every top-level window -- diagnostic use
    (e.g. confirming a component's window is even visible to the WM) from a shell,
    not called by the running server."""
    d = display.Display(display_name)
    try:
        root = d.screen().root
        net_client_list = d.intern_atom("_NET_CLIENT_LIST")
        net_wm_pid = d.intern_atom("_NET_WM_PID")
        net_wm_name = d.intern_atom("_NET_WM_NAME")
        utf8 = d.intern_atom("UTF8_STRING")
        out = []
        for wid in _client_list(root, net_client_list):
            win = d.create_resource_object("window", wid)
            wpid = win.get_full_property(net_wm_pid, X.AnyPropertyType)
            name = win.get_full_property(net_wm_name, utf8)
            out.append((
                wid,
                wpid.value[0] if wpid and wpid.value else None,
                bytes(name.value).decode("utf-8", "replace") if name else None,
            ))
        return out
    finally:
        d.close()
