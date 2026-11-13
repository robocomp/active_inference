#!/usr/bin/env python3
"""Type one message into language_agent's "Robot chat" window (X11 XTEST) and press Enter.

Safety: the window is found by language_agent's PID, activated, and checked to be the ACTIVE window before every key;
if focus moves anywhere else the script stops at once (it never types into another window).
Then it waits for the chat log (logs/chat_*.jsonl, newest) to record the message as a "user" event.
usage: chat_say.py "<text>" <chat_log_dir>
"""
import glob, json, os, subprocess, sys, time
from Xlib import X, XK, display, protocol
from Xlib.ext import xtest

text, logdir = sys.argv[1], sys.argv[2]
d = display.Display(":0"); root = d.screen().root
pid = int(subprocess.check_output(["pgrep", "-x", "language_agent"]).split()[0])
A = lambda n: d.intern_atom(n)
win = None
for wid in root.get_full_property(A("_NET_CLIENT_LIST"), X.AnyPropertyType).value:
    w = d.create_resource_object("window", wid)
    p = w.get_full_property(A("_NET_WM_PID"), X.AnyPropertyType)
    if p and p.value[0] == pid:
        win = w
        break
if win is None:
    sys.exit("no language_agent window")

def active():
    return root.get_full_property(A("_NET_ACTIVE_WINDOW"), X.AnyPropertyType).value[0]

ev = protocol.event.ClientMessage(window=win, client_type=A("_NET_ACTIVE_WINDOW"), data=(32, [2, X.CurrentTime, 0, 0, 0]))
root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
d.flush()
for _ in range(40):
    if active() == win.id:
        break
    time.sleep(0.05)
else:
    sys.exit("could not activate the chat window; typed nothing")
time.sleep(0.3)

# Click into the input line (bottom of the window, ~45 px up from the bottom edge, centre) so keys reach it.
g = win.get_geometry(); t = win.translate_coords(root, 0, 0)
cx, cy = -t.x + g.width // 2, -t.y + g.height - 45
xtest.fake_input(d, X.MotionNotify, x=cx, y=cy); d.sync(); time.sleep(0.1)
if active() != win.id:
    sys.exit("focus left the chat window; clicked nothing")
xtest.fake_input(d, X.ButtonPress, 1); xtest.fake_input(d, X.ButtonRelease, 1); d.sync(); time.sleep(0.2)
logs = sorted(glob.glob(os.path.join(logdir, "chat_*.jsonl")), key=os.path.getmtime)
log = logs[-1]
start = os.path.getsize(log)

def key(sym_name, ch=None):
    ks = XK.string_to_keysym(sym_name) if ch is None else ord(ch) if ch.isalnum() else XK.string_to_keysym(sym_name)
    kc = d.keysym_to_keycode(ks)
    if kc == 0:
        sys.exit(f"no keycode for {sym_name!r}")
    shift = d.keycode_to_keysym(kc, 0) != ks
    if active() != win.id:
        sys.exit("focus left the chat window; stopped")
    if shift:
        xtest.fake_input(d, X.KeyPress, d.keysym_to_keycode(XK.XK_Shift_L))
    xtest.fake_input(d, X.KeyPress, kc); xtest.fake_input(d, X.KeyRelease, kc)
    if shift:
        xtest.fake_input(d, X.KeyRelease, d.keysym_to_keycode(XK.XK_Shift_L))
    d.sync(); time.sleep(0.012)

NAMES = {" ": "space", "_": "underscore", ".": "period", ",": "comma", "?": "question", "!": "exclam",
         "'": "apostrophe", "-": "minus", "(": "parenleft", ")": "parenright", ":": "colon"}
for ch in text:
    if ch.isalnum():
        key(ch, ch)
    elif ch in NAMES:
        key(NAMES[ch])
    else:
        sys.exit(f"cannot type {ch!r}")
key("Return")

for _ in range(100):   # the chat log records the message as a "user" event
    with open(log) as f:
        f.seek(start)
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("ev") == "user":
                print("SENT:", r.get("text"))
                sys.exit(0)
    time.sleep(0.1)
sys.exit("typed, but the chat log shows no user message (the input field may not have had focus)")
