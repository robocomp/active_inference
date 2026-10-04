#!/usr/bin/env python3
"""
room_tui.py — terminal dashboard for room_concept's live status stream.

    tools/room_run.sh                       # starts the agent in the background and opens this
    python3 tools/room_tui.py               # (re)attach to a running room_concept
    python3 tools/room_tui.py --log tmp/agent_<ts>.log --pid <pid>

ONE SCREEN, everything at once: a top bar (agent, pid, uptime, state, map mode, config, solve/predict,
compute Hz); peers, streams and the non-default config keys on the left; localisation (with σθ and
surprise sparklines) and the room in the centre; motion calibration and camera mounts on the right;
the latest warnings and the latest event along the bottom. Narrow terminals collapse to two columns,
then one (scrollable).

Full-screen detail views (Esc closes):  F1/s startup timeline · F2/w warnings history · F3/l raw log
· F4/g config gates (searchable) · F5/c commands.   / searches in the raw log and the gates view.

The agent (rc::StatusStream, common/status_stream/) serves newline-delimited JSON on
$XDG_RUNTIME_DIR/rc_status/<agent>_<id>.sock and appends every event to tmp/agent_events_<start>.jsonl.
This viewer auto-discovers the socket, replays what the agent kept (hello, startup phases, loaded
files, state-machine transitions, the last 200 log lines, the last state snapshot) and reconnects on
its own when the agent restarts.

QUITTING: `q` asks — stop the agent (SIGTERM, so its graph cleanup runs; NEVER SIGKILL) and quit, or
detach and leave it running. If this viewer dies, the agent keeps running; run this file again.

Requires Textual (python3 -m pip install textual). Written against Textual 8.2.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import datetime as _dt
import glob
import json
import os
import signal
import time

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (Button, DataTable, Footer, Input, Label, RichLog, Sparkline, Static)

MAX_LOG_LINES = 20000
MAX_WARN = 5000
DEG = 57.29577951308232
OK, AMBER, RED, DIM = "green", "yellow", "bold red", "dim"

LEVEL_STYLE = {"debug": DIM, "info": "", "warning": AMBER, "critical": RED, "fatal": "bold white on red"}

# Motion calibration: the Calib window's own labels and display scales (calibration_viewer.cpp).
CALIB_DISPLAY = {
    "k_v":       ("translation scale", 100.0, "%"),
    "eps_yaw":   ("mount yaw",         DEG,   "deg"),
    "k_omega":   ("gyro scale",        100.0, "%"),
    "b_omega":   ("gyro bias",         DEG,   "deg/s"),
    "k_lat":     ("lateral scale",     100.0, "%"),
    "dk_wheel":  ("wheel mismatch",    1000.0, "mrad/m"),
    "k_omega_w": ("wheel rot. scale",  100.0, "%"),
}
# Camera mount parameters, internal SI -> display (calibration_viewer.cpp again).
CAM_DISPLAY = [("pitch", DEG, "°"), ("height", 1000.0, "mm"), ("yaw", DEG, "°"), ("dt", 1.0, "x")]


# ── helpers ───────────────────────────────────────────────────────────────────────────────────────────
def socket_dir() -> str:
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    return os.path.join(xdg, "rc_status") if xdg else f"/tmp/rc_status_{os.getuid()}"


def find_socket(agent: str) -> str | None:
    """The newest <agent>_*.sock in the status directory (the agent removes a stale one at start)."""
    cands = glob.glob(os.path.join(socket_dir(), f"{agent}_*.sock"))
    cands.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
    return cands[0] if cands else None


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def pid_is_agent(pid: int, agent: str) -> bool:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return agent.encode() in f.read()
    except OSError:
        return False


def hms(t_ms) -> str:
    if not t_ms:
        return ""
    return _dt.datetime.fromtimestamp(t_ms / 1000.0).strftime("%H:%M:%S.%f")[:-3]


def fmt(v, nd=3, scale=1.0) -> str:
    if v is None:
        return "—"
    try:
        return f"{v * scale:.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def dur(sec: float) -> str:
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def compact_table() -> Table:
    return Table(box=None, show_header=True, header_style="bold dim", pad_edge=False, padding=(0, 1), expand=True)


def gate_shown(r: dict) -> bool:
    """The full gates view's default filter: non-default, A/B arms, every bool/string key."""
    if r.get("effective") == "(absent)":
        return False
    o = r.get("origin", "")
    return (o in ("overlay", "manifest", "runtime", "shadowed", "typeerr", "unread")
            or r.get("effective") != r.get("default")
            or r.get("kind") == "experiment"
            or r.get("type") in ("bool", "str"))


# ── modals ────────────────────────────────────────────────────────────────────────────────────────────
class QuitScreen(ModalScreen[str]):
    DEFAULT_CSS = """
    QuitScreen { align: center middle; }
    #dlg { width: 72; max-width: 95%; height: auto; border: thick $warning; background: $surface; padding: 1 2; }
    #dlg Button { width: 100%; margin: 1 0 0 0; }
    """

    def __init__(self, pid: int | None, alive: bool):
        super().__init__()
        self.pid, self.alive = pid, alive

    def compose(self) -> ComposeResult:
        with Vertical(id="dlg"):
            if self.alive:
                yield Label(f"room_concept is running (pid {self.pid}).")
                yield Button("Stop agent (SIGTERM, graceful cleanup) and quit", id="stop", variant="error")
                yield Button("Detach — quit the viewer, leave the agent running", id="detach", variant="primary")
            else:
                yield Label("The agent is not running (or its pid is unknown).")
                yield Button("Quit", id="detach", variant="primary")
            yield Button("Cancel", id="cancel")

    @on(Button.Pressed)
    def _pressed(self, ev: Button.Pressed) -> None:
        self.dismiss(ev.button.id or "cancel")

    def key_escape(self) -> None:
        self.dismiss("cancel")


class ConfirmScreen(ModalScreen[bool]):
    DEFAULT_CSS = """
    ConfirmScreen { align: center middle; }
    #cdlg { width: 80; max-width: 95%; height: auto; border: thick $accent; background: $surface; padding: 1 2; }
    #cdlg Horizontal { height: auto; margin: 1 0 0 0; }
    #cdlg Button { margin: 0 2 0 0; }
    """

    def __init__(self, name: str, description: str, question: str):
        super().__init__()
        self.cmd_name, self.description, self.question = name, description, question

    def compose(self) -> ComposeResult:
        with Vertical(id="cdlg"):
            yield Label(f"[b]{self.cmd_name}[/b] — {self.description}")
            yield Label(self.question or f"Send '{self.cmd_name}' to the running agent?")
            with Horizontal():
                yield Button("Yes, send it", id="yes", variant="warning")
                yield Button("Cancel", id="no")

    @on(Button.Pressed)
    def _pressed(self, ev: Button.Pressed) -> None:
        self.dismiss(ev.button.id == "yes")

    def key_escape(self) -> None:
        self.dismiss(False)


# ── full-screen detail views ──────────────────────────────────────────────────────────────────────────
class DetailScreen(Screen):
    """Base: a title line, the body, Esc closes. Subclasses implement rebuild() and feed_event()."""
    BINDINGS = [Binding("escape", "close", "Back"), Binding("q", "app.quit_dialog", "Quit")]
    DEFAULT_CSS = """
    DetailScreen #dtitle { height: 1; background: $boost; padding: 0 1; }
    DetailScreen #dsearch { display: none; }
    DetailScreen #dsearch.visible { display: block; }
    DetailScreen .cap { height: 1; color: $text-muted; padding: 0 1; }
    """
    TITLE_TEXT = ""

    def compose_body(self) -> ComposeResult:
        yield from ()

    def compose(self) -> ComposeResult:
        yield Static(self.TITLE_TEXT, id="dtitle")
        yield from self.compose_body()
        yield Footer()

    def on_mount(self) -> None:
        self.rebuild()

    def rebuild(self) -> None:
        pass

    def feed_event(self, ev: dict) -> None:
        pass

    def action_close(self) -> None:
        self.app.pop_screen()


class StartupScreen(DetailScreen):
    TITLE_TEXT = "Startup timeline — phases, loaded files, state machine, peers, stalls  (Esc: back)"

    def compose_body(self) -> ComposeResult:
        yield DataTable(id="startup", zebra_stripes=True, cursor_type="row")

    def rebuild(self) -> None:
        t = self.query_one("#startup", DataTable)
        t.clear(columns=True)
        t.add_columns("time", "kind", "what", "ms", "detail")
        for row in self.app.startup_rows:
            t.add_row(*row)
        t.move_cursor(row=max(0, t.row_count - 1))

    def feed_event(self, ev: dict) -> None:
        if ev.get("_startup_row"):
            t = self.query_one("#startup", DataTable)
            t.add_row(*ev["_startup_row"])
            t.move_cursor(row=t.row_count - 1)


class WarningsScreen(DetailScreen):
    TITLE_TEXT = "Warnings history — every warning/critical/fatal line, on the terminal or not  (Esc: back)"

    def compose_body(self) -> ComposeResult:
        yield DataTable(id="warnfull", zebra_stripes=True, cursor_type="row")

    def rebuild(self) -> None:
        t = self.query_one("#warnfull", DataTable)
        t.clear(columns=True)
        t.add_columns("time", "level", "tag", "message")
        for row in self.app.warnings:
            t.add_row(*row)
        t.move_cursor(row=max(0, t.row_count - 1))

    def feed_event(self, ev: dict) -> None:
        if ev.get("_warn_row"):
            t = self.query_one("#warnfull", DataTable)
            t.add_row(*ev["_warn_row"])
            t.move_cursor(row=t.row_count - 1)


class RawLogScreen(DetailScreen):
    TITLE_TEXT = "Raw log  (t: events log ⇄ stdout file · /: search · f: follow · Esc: back)"
    BINDINGS = [Binding("slash", "search", "Search"), Binding("t", "toggle_source", "Source"),
                Binding("f", "toggle_follow", "Follow")]

    def compose_body(self) -> ComposeResult:
        yield Static("", id="rawbar", classes="cap")
        yield Input(placeholder="search (case-insensitive substring; Esc clears)", id="dsearch")
        yield RichLog(id="raw", max_lines=MAX_LOG_LINES, wrap=False, highlight=False, markup=False)

    def rebuild(self) -> None:
        a = self.app
        src = (f"stdout file {os.path.basename(a.log_path or '') or '(unknown)'}" if a.raw_source == "stdout"
               else f"events log {os.path.basename(a.hello.get('events') or '') or '(socket replay)'}"
                    "  (⟂ = not on the terminal)")
        flt = f"  ·  filter '{a.search}'" if a.search else ""
        self.query_one("#rawbar", Static).update(f"source: {src}{flt}  ·  follow {'on' if a.follow else 'off'}")
        log = self.query_one("#raw", RichLog)
        log.clear()
        lines = (Text(s) for s in a.stdout_lines) if a.raw_source == "stdout" else iter(a.event_lines)
        for line in lines:
            if a.matches(line.plain):
                log.write(line, scroll_end=False)
        if a.follow:
            log.scroll_end(animate=False)

    def feed_event(self, ev: dict) -> None:
        line = ev.get("_raw_line")
        if line is not None and ev.get("_raw_source") == self.app.raw_source and self.app.matches(line.plain):
            self.query_one("#raw", RichLog).write(line, scroll_end=self.app.follow)

    def action_search(self) -> None:
        inp = self.query_one("#dsearch", Input)
        inp.add_class("visible")
        inp.value = self.app.search
        inp.focus()

    @on(Input.Submitted, "#dsearch")
    def _submitted(self, ev: Input.Submitted) -> None:
        self.app.search = ev.value.strip()
        ev.input.remove_class("visible")
        self.rebuild()
        self.query_one("#raw", RichLog).focus()

    def action_close(self) -> None:
        inp = self.query_one("#dsearch", Input)
        if inp.has_class("visible") or self.app.search:
            inp.remove_class("visible")
            if self.app.search:
                self.app.search = ""
                self.rebuild()
            return
        self.app.pop_screen()

    def action_toggle_source(self) -> None:
        a = self.app
        a.raw_source = "events" if a.raw_source == "stdout" else "stdout"
        if a.raw_source == "stdout" and not a.log_path:
            self.notify("no stdout log known (pass --log, or the agent's stdout is a terminal)", severity="warning")
            a.raw_source = "events"
        self.rebuild()

    def action_toggle_follow(self) -> None:
        self.app.follow = not self.app.follow
        self.rebuild()


class GatesScreen(DetailScreen):
    TITLE_TEXT = "Config gates — non-default keys, A/B arms, every bool/string key  (/: search · a: all · Esc: back)"
    BINDINGS = [Binding("slash", "search", "Search"), Binding("a", "toggle_all", "All keys")]
    ORIGIN_STYLE = {"default": DIM, "file": "", "overlay": "cyan", "manifest": "blue", "runtime": "magenta",
                    "shadowed": RED, "typeerr": RED, "unread": AMBER}

    def __init__(self):
        super().__init__()
        self.show_all = False

    def compose_body(self) -> ComposeResult:
        yield Static("", id="gates_head", classes="cap")
        yield Input(placeholder="search keys/descriptions (Esc clears)", id="dsearch")
        yield DataTable(id="gates", zebra_stripes=True, cursor_type="row")

    def rebuild(self) -> None:
        a = self.app
        ev = a.config_ev
        recs = ev.get("records", [])
        t = self.query_one("#gates", DataTable)
        t.clear(columns=True)
        t.add_columns("key", "origin", "kind", "type", "effective", "default", "description")
        q = a.gsearch.lower()
        shown = bad = 0
        for r in recs:
            o = r.get("origin", "")
            if o in ("shadowed", "typeerr"):
                bad += 1
            if not (self.show_all or gate_shown(r)):
                continue
            if q and q not in (r.get("key", "") + " " + r.get("description", "")).lower():
                continue
            shown += 1
            changed = r.get("effective") != r.get("default")
            kind = r.get("kind", "")
            org = o + (f" ({r['overlay']})" if r.get("overlay") else "")
            t.add_row(Text(r.get("key", ""), style=RED if o in ("shadowed", "typeerr") else ""),
                      Text(org, style=self.ORIGIN_STYLE.get(o, "")),
                      Text("A/B" if kind == "experiment" else kind,
                           style="magenta bold" if kind == "experiment" else DIM),
                      Text(r.get("type", ""), style=DIM),
                      Text(r.get("effective", ""), style="bold" if changed else ""),
                      Text(r.get("default", "") if changed else "", style=DIM),
                      Text(r.get("description", ""), style=DIM))
        head = (f"{shown} of {len(recs)} keys ({'all' if self.show_all else 'non-default, A/B, bool/string'})"
                f"  ·  fingerprint {ev.get('fingerprint', '?')}  ·  {ev.get('undocumented', 0)} undocumented  ·  "
                f"unread {ev.get('unread', '?') if ev.get('sweep_armed') else 'INCONCLUSIVE'}")
        if bad:
            head += f"  ·  [b red]{bad} key(s) present but NOT in force[/]"
        if a.gsearch:
            head += f"  ·  filter '{a.gsearch}'"
        if not recs:
            head = "no config event yet — it is published at the END of the agent's startup"
        self.query_one("#gates_head", Static).update(head)

    def feed_event(self, ev: dict) -> None:
        if ev.get("kind") == "config":
            self.rebuild()

    def action_toggle_all(self) -> None:
        self.show_all = not self.show_all
        self.rebuild()

    def action_search(self) -> None:
        inp = self.query_one("#dsearch", Input)
        inp.add_class("visible")
        inp.value = self.app.gsearch
        inp.focus()

    @on(Input.Submitted, "#dsearch")
    def _submitted(self, ev: Input.Submitted) -> None:
        self.app.gsearch = ev.value.strip()
        ev.input.remove_class("visible")
        self.rebuild()
        self.query_one("#gates", DataTable).focus()

    def action_close(self) -> None:
        inp = self.query_one("#dsearch", Input)
        if inp.has_class("visible") or self.app.gsearch:
            inp.remove_class("visible")
            if self.app.gsearch:
                self.app.gsearch = ""
                self.rebuild()
            return
        self.app.pop_screen()


class CommandsScreen(DetailScreen):
    TITLE_TEXT = "Commands — the agent's whitelist; each asks for confirmation  (Esc: back)"
    DEFAULT_CSS = """
    CommandsScreen #cmd_buttons { height: auto; padding: 0 1; }
    CommandsScreen #cmd_buttons Horizontal { height: 3; }
    CommandsScreen #cmd_buttons Button { min-width: 30; }
    CommandsScreen #cmd_buttons Label { padding: 1 2; color: $text-muted; }
    """

    def compose_body(self) -> ComposeResult:
        yield Vertical(id="cmd_buttons")
        yield Static("replies", classes="cap")
        yield DataTable(id="replies", cursor_type="none")

    def rebuild(self) -> None:
        box = self.query_one("#cmd_buttons", Vertical)
        box.remove_children()
        if not self.app.commands:
            box.mount(Label("the agent has registered no commands (yet)"))
        for c in self.app.commands:
            box.mount(Horizontal(Button(c["name"], id=f"cmd-{c['name']}", variant="primary"),
                                 Label(c.get("description", ""))))
        t = self.query_one("#replies", DataTable)
        t.clear(columns=True)
        t.add_columns("time", "command", "ok", "message")
        for row in self.app.replies:
            t.add_row(*row)

    def feed_event(self, ev: dict) -> None:
        if ev.get("kind") == "commands":
            self.rebuild()
        elif ev.get("_reply_row"):
            self.query_one("#replies", DataTable).add_row(*ev["_reply_row"])

    @on(Button.Pressed, "#cmd_buttons Button")
    def _cmd_pressed(self, ev: Button.Pressed) -> None:
        self.app.ask_command((ev.button.id or "")[4:])


# ── the app: the dashboard is its default screen ─────────────────────────────────────────────────────
class RoomTUI(App):
    TITLE = "room_concept"
    CSS = """
    #topbar { height: auto; max-height: 2; background: $boost; padding: 0 1; }
    #grid { grid-size: 3; grid-columns: 1fr 1fr 1fr; grid-gutter: 0 1; height: 1fr; padding: 0 1; }
    #grid.cols2 { grid-size: 2; grid-columns: 1fr 1fr; grid-rows: auto; }
    #grid.cols1 { grid-size: 1; grid-columns: 1fr; grid-rows: auto; }
    .col { height: 100%; }
    #grid.cols2 .col, #grid.cols1 .col { height: auto; }
    .panel { border: round $panel-lighten-2; border-title-color: $text-muted; padding: 0 1; height: auto; }
    .spark { height: 1; }
    .sparklabel { height: 1; color: $text-muted; }
    #bottom { height: 9; padding: 0 1; }
    #warnlog { height: 1fr; border: round $panel-lighten-2; border-title-color: $text-muted; }
    #lastev { height: 1; color: $text-muted; }
    """
    BINDINGS = [
        Binding("q", "quit_dialog", "Quit"),
        Binding("f1,s", "detail('startup')", "Startup", key_display="F1"),
        Binding("f2,w", "detail('warnings')", "Warnings", key_display="F2"),
        Binding("f3,l", "detail('raw')", "Raw log", key_display="F3"),
        Binding("f4,g", "detail('gates')", "Config", key_display="F4"),
        Binding("f5,c", "detail('commands')", "Commands", key_display="F5"),
    ]

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.agent = args.agent
        self.pid: int | None = args.pid
        self.log_path: str | None = args.log
        self.hello: dict = {}
        self.connected = False
        self.writer: asyncio.StreamWriter | None = None
        self.sm_state = "?"
        self.state: dict = {}
        self.config_ev: dict = {}
        self.commands: list[dict] = []
        self.next_cmd_id = 1
        # history the detail screens render from
        self.startup_rows: list[tuple] = []
        self.warnings: collections.deque[tuple] = collections.deque(maxlen=MAX_WARN)
        self.replies: list[tuple] = []
        self.peers: dict[str, tuple[str, int]] = {}        # name -> (last event, t_ms)
        self.sth_hist: collections.deque[float] = collections.deque(maxlen=240)
        self.kl_hist: collections.deque[float] = collections.deque(maxlen=240)
        # raw log
        self.raw_source = "events"   # the events log is the complete, timestamped record
        self.stdout_lines: collections.deque[str] = collections.deque(maxlen=MAX_LOG_LINES)
        self.event_lines: collections.deque[Text] = collections.deque(maxlen=MAX_LOG_LINES)
        self.max_file_seq = 0
        self.last_seq = 0
        self.search = ""
        self.gsearch = ""
        self.follow = True
        self.n_warn = 0
        self.tail_task_for: str | None = None

    # ── layout ──
    def compose(self) -> ComposeResult:
        yield Static("", id="topbar")
        with Grid(id="grid"):
            with VerticalScroll(classes="col", id="col-left"):
                yield Static("", id="p-peers", classes="panel")
                yield Static("", id="p-streams", classes="panel")
                yield Static("", id="p-gates", classes="panel")
            with VerticalScroll(classes="col", id="col-centre"):
                with Vertical(id="p-loc", classes="panel"):
                    yield Static("", id="loc-body")
                    yield Static("σθ (deg)", classes="sparklabel", id="sth-label")
                    yield Sparkline([], id="sth-spark", classes="spark")
                    yield Static("surprise KL (nats)", classes="sparklabel", id="kl-label")
                    yield Sparkline([], id="kl-spark", classes="spark")
                yield Static("", id="p-room", classes="panel")
            with VerticalScroll(classes="col", id="col-right"):
                yield Static("", id="p-calib", classes="panel")
                yield Static("", id="p-cams", classes="panel")
        with Vertical(id="bottom"):
            yield RichLog(id="warnlog", max_lines=500, wrap=False, markup=False, highlight=False)
            yield Static("", id="lastev")
        yield Footer()

    def on_mount(self) -> None:
        titles = {"p-peers": "peers / presence", "p-streams": "streams",
                  "p-gates": "config: non-default + A/B  (F4)", "p-loc": "localisation", "p-room": "room",
                  "p-calib": "motion calibration", "p-cams": "camera mounts (total correction in force)"}
        for wid, title in titles.items():
            self.query_one(f"#{wid}").border_title = title
        self.query_one("#warnlog").border_title = "warnings  (F2: history)"
        self.relayout(self.size.width, self.size.height)
        for fn in (self.render_peers, self.render_streams, self.render_gates, self.render_loc, self.render_room,
                   self.render_calib, self.render_cams, self.update_topbar):
            fn()
        self.run_worker(self.socket_loop(), exclusive=False, group="sock")
        if self.log_path:
            self.start_tail()
        self.set_interval(1.0, self.update_topbar)

    # ── responsive: 3 columns, then 2, then 1 (scrollable) ──
    def on_resize(self, ev) -> None:
        self.relayout(ev.size.width, ev.size.height)

    def relayout(self, width: int, height: int) -> None:
        try:
            g = self.query_one("#grid", Grid)
        except Exception:
            return
        n = 3 if width >= 150 else (2 if width >= 100 else 1)
        for k in (1, 2, 3):
            g.set_class(k == n, f"cols{k}")
        # Fewer columns ⇒ the panels stack at their natural height and the grid itself scrolls.
        g.styles.overflow_y = "hidden" if n == 3 else "auto"
        self.query_one("#bottom").styles.height = 13 if height >= 55 else (9 if height >= 40 else 6)

    # ── the dashboard panels ──
    def panel(self, wid: str, renderable) -> None:
        try:
            self.query_one(f"#{wid}", Static).update(renderable)
        except Exception:
            pass   # tearing down, or a detail screen is on top of a not-yet-mounted dashboard

    def update_topbar(self) -> None:
        h = self.hello
        alive = pid_alive(self.pid)
        pid = (f"pid {self.pid} " + (f"[{OK}]alive[/]" if alive else f"[{RED}]not running[/]")) if self.pid else "pid ?"
        up = dur(time.time() - h["start_ms"] / 1000.0) if h.get("start_ms") and alive else "—"
        sm_col = {"Operating": OK, "Waiting": AMBER, "Degraded": RED, "stopping": AMBER,
                  "exited": RED}.get(self.sm_state, "default")
        st = self.state
        room, loc, comp = st.get("room", {}), st.get("loc", {}), st.get("compute", {})
        cfg = "—"
        argv = h.get("argv", [])
        if len(argv) > 1:
            path = argv[1] if os.path.isabs(argv[1]) else os.path.join(h.get("cwd", ""), argv[1])
            try:
                mt = _dt.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%m-%d %H:%M")
                cfg = f"{os.path.basename(path)} ({mt})"
            except OSError:
                cfg = os.path.basename(path)
        conn = "" if self.connected else f" · [{AMBER}]socket reconnecting…[/]"
        miss = st.get("presence", {}).get("missing", [])
        pres = f"[{RED}]missing {' '.join(miss)}[/]" if miss else (f"[{OK}]peers ok[/]" if st else "peers ?")
        warn = f" · [{AMBER}]⚠ {self.n_warn}[/]" if self.n_warn else ""
        self.panel("topbar", f"[b]{h.get('agent', self.agent)}[/] id {h.get('id', '?')} · {pid} · up {up} · "
                             f"[{sm_col}]{self.sm_state}[/] · {pres} · map {room.get('map_mode', '—')} · "
                             f"cfg {cfg} · {loc.get('mode', '—')} · compute {fmt(comp.get('hz'), 1)} Hz{warn}{conn}")

    def render_peers(self) -> None:
        t = compact_table()
        t.show_header = False
        t.add_column("peer", ratio=1, no_wrap=True, overflow="ellipsis")
        t.add_column("state", no_wrap=True)
        t.add_column("at", style=DIM, no_wrap=True)
        miss = self.state.get("presence", {}).get("missing", [])
        for m in miss:
            t.add_row(m, Text("MISSING (required)", style=RED), "")
        for name, (evname, t_ms) in sorted(self.peers.items()):
            if name in miss:
                continue
            style = RED if "lost" in evname else (OK if "ready" in evname else AMBER)
            t.add_row(name, Text(evname, style=style), hms(t_ms)[:8])
        footer = (Text("required peers: all present", style=OK) if self.state and not miss
                  else Text("no state yet", style=DIM) if not self.state else Text(""))
        self.panel("p-peers", Group(t, footer) if t.row_count else footer)

    def render_streams(self) -> None:
        t = compact_table()
        t.add_column("stream", no_wrap=True, ratio=1)
        t.add_column("Hz", justify="right", no_wrap=True)
        t.add_column("age", justify="right", no_wrap=True)
        t.add_column("", no_wrap=True)
        for name, s in self.state.get("streams", {}).items():
            hz, age = s.get("hz", -1), s.get("age_ms", -1)
            if age is None or age < 0:
                flag = Text("—" if name == "imu" else "no frame", style=DIM if name == "imu" else AMBER)
            elif age > 1000:
                flag = Text("STALE", style=RED)
            elif age > 250:
                flag = Text("slow", style=AMBER)
            else:
                flag = Text("ok", style=OK)
            t.add_row(name, fmt(hz, 1) if hz is not None and hz >= 0 else "—",
                      "—" if age is None or age < 0 else f"{age} ms", flag)
        comp = self.state.get("compute", {})
        if comp:
            t.add_row("compute()", fmt(comp.get("hz"), 1),
                      f"{fmt(comp.get('us', 0) / 1000.0, 1)}/{fmt(comp.get('us_max', 0) / 1000.0, 0)}ms", "")
        self.panel("p-streams", t if t.row_count else Text("no state yet", style=DIM))

    def render_gates(self) -> None:
        recs = self.config_ev.get("records", [])
        if not recs:
            self.panel("p-gates", Text("published at the end of startup", style=DIM))
            return
        bad = [r for r in recs if r.get("origin") in ("shadowed", "typeerr")]
        rows = [r for r in recs if r not in bad and r.get("effective") != "(absent)" and (
            r.get("effective") != r.get("default") or r.get("kind") == "experiment" or r.get("origin") == "overlay")]
        # Experiment arms and overlays first: they are what makes THIS run different from the last.
        rows.sort(key=lambda r: (r.get("kind") != "experiment", r.get("origin") != "overlay", r["key"]))
        t = compact_table()
        t.show_header = False
        t.add_column("key", ratio=1, no_wrap=True, overflow="ellipsis")
        t.add_column("value", no_wrap=True, max_width=16, overflow="ellipsis")
        for r in bad:
            t.add_row(Text(r["key"], style=RED), Text(r.get("origin", ""), style=RED))
        limit = 12
        for r in rows[:limit]:
            o = r.get("origin", "")
            style = "magenta" if r.get("kind") == "experiment" else ("cyan" if o == "overlay" else "")
            t.add_row(Text(r["key"], style=style), Text(r.get("effective", ""), style="bold"))
        if len(rows) > limit:
            t.add_row(Text(f"+{len(rows) - limit} more  (F4)", style=DIM), "")
        self.panel("p-gates", t)

    def render_loc(self) -> None:
        l = self.state.get("loc", {})
        if not l.get("have"):
            self.panel("loc-body", Text("no localisation result yet", style=DIM))
            return
        ok = Text("ok", style=OK) if l.get("ok") else Text("NOT OK", style=RED)
        if l.get("diverged"):
            ok = Text("DIVERGED", style=RED)
        age = l.get("age_ms", -1)
        t = compact_table()
        t.show_header = False
        t.add_column("k", style=DIM, no_wrap=True)
        t.add_column("v", ratio=1, no_wrap=True, overflow="ellipsis")
        t.add_row("pose", f"x {fmt(l.get('x'), 3)}  y {fmt(l.get('y'), 3)} m  θ {fmt(l.get('theta'), 1, DEG)}°")
        t.add_row("σ", f"{fmt(l.get('sx'), 0, 1000)} / {fmt(l.get('sy'), 0, 1000)} mm   {fmt(l.get('sth'), 2, DEG)}°")
        t.add_row("result", Text.assemble(ok, f"  {l.get('mode', '')} · {l.get('iters')} it · "
                                              f"cond {fmt(l.get('cond'), 1)}"))
        t.add_row("early-exit", f"{fmt(l.get('early_exit_metric'), 1, 1000)} mm · |SDF| "
                                f"{fmt(l.get('sdf_med'), 1, 1000)} · innov {fmt(l.get('innov'), 1, 1000)} mm")
        t.add_row("surprise", f"KL {fmt(l.get('kl'), 2)} · mismatch {fmt(l.get('mismatch'), 2)} nats"
                              + ("" if l.get("scored") else " (unscored)"))
        t.add_row("age", Text.assemble(Text(f"{age} ms", style=RED if age > 1000 else (AMBER if age > 250 else "")),
                                       f" · reloc epoch {l.get('reloc_epoch')}"))
        self.panel("loc-body", t)
        try:
            self.query_one("#sth-spark", Sparkline).data = list(self.sth_hist)
            self.query_one("#kl-spark", Sparkline).data = list(self.kl_hist)
            if self.sth_hist:
                self.query_one("#sth-label", Static).update(
                    f"σθ {fmt(self.sth_hist[-1], 3)}°  (max {fmt(max(self.sth_hist), 3)}, last {len(self.sth_hist) // 2} s)")
            if self.kl_hist:
                self.query_one("#kl-label", Static).update(
                    f"surprise KL {fmt(self.kl_hist[-1], 2)}  (max {fmt(max(self.kl_hist), 2)})")
        except Exception:
            pass

    def render_room(self) -> None:
        r = self.state.get("room", {})
        if not r:
            self.panel("p-room", Text("no state yet", style=DIM))
            return
        t = compact_table()
        t.show_header = False
        t.add_column("k", style=DIM, no_wrap=True)
        t.add_column("v", ratio=1, no_wrap=True, overflow="ellipsis")
        stable = f"{r.get('stable_frames', 0)}/{r.get('needed', '?')}"
        t.add_row("room node", Text("created", style=OK) if r.get("node_created")
                  else Text(f"not yet · stable {stable}", style=AMBER))
        t.add_row("map", Text.assemble(f"{r.get('map_mode', '?')} · ",
                                       Text("ready", style=OK) if r.get("map_ready") else Text("not ready", style=AMBER),
                                       Text("  RELOCALISING", style=AMBER) if r.get("grid_searching") else ""))
        if r.get("map_mode") == "estimate" or r.get("walls"):
            t.add_row("layout", f"{r.get('walls', 0)} walls · {r.get('verts', 0)} verts · "
                                f"{'closed' if r.get('closed') else 'open'}")
        hm, hs = r.get("height_measured"), r.get("height_stated")
        t.add_row("height", (f"{fmt(hm, 2)} m measured" if hm and hm > 0 else "not measured yet")
                  + (f" · {fmt(hs, 2)} stated" if hs else ""))
        self.panel("p-room", t)

    def render_calib(self) -> None:
        c = self.state.get("calib", {})
        names = c.get("names", [])
        if not names:
            self.panel("p-calib", Text("no state yet", style=DIM))
            return
        val, sig = c.get("value", []), c.get("sigma", [])
        inf, app = c.get("informed", 0), c.get("applied", 0)
        t = compact_table()
        t.add_column("param", no_wrap=True, overflow="ellipsis", ratio=1)
        t.add_column("value ± σ", justify="right", no_wrap=True)
        t.add_column("inf", no_wrap=True)
        t.add_column("", no_wrap=True)
        for i, n in enumerate(names):
            label, scale, unit = CALIB_DISPLAY.get(n, (n, 1.0, ""))
            v = val[i] if i < len(val) else None
            s = sig[i] if i < len(sig) else None
            informed, applied = bool(inf >> i & 1), bool(app >> i & 1)
            t.add_row(label, f"{fmt(v, 2, scale)} ± {fmt(s, 2, scale)} {unit}",
                      Text("●", style=OK) if informed else Text("○", style=DIM),
                      Text("APPLIED", style="bold green") if applied else Text("·", style=DIM))
        t.caption = (f"● informed · APPLIED = correcting odometry · {c.get('episodes', 0)} episodes · "
                     f"cond {fmt(c.get('cond'), 1)}")
        t.caption_style = DIM
        self.panel("p-calib", t)

    def render_cams(self) -> None:
        cams = self.state.get("camcal", [])
        if not cams:
            self.panel("p-cams", Text("no mount solve yet (first after ~5 s of paired corners)", style=DIM))
            return
        # Transposed — one COLUMN per camera — so two or three cameras fit a third of the screen.
        t = compact_table()
        t.add_column("", style=DIM, no_wrap=True)
        for c in cams:
            t.add_column(c.get("cam", "?"), justify="right", no_wrap=True, ratio=1)
        for i, (name, scale, unit) in enumerate(CAM_DISPLAY[:3]):
            nd = 1 if i == 1 else 2
            cells = []
            for c in cams:
                v, sg = c.get("value", [None] * 4), c.get("sigma", [None] * 4)
                informed = bool(c.get("informed", 0) >> i & 1)
                cells.append(Text(f"{fmt(v[i], nd, scale)} ± {fmt(sg[i], nd, scale)}", style="" if informed else DIM))
            t.add_row(f"{name} {unit}", *cells)
        t.add_row("pairs", *[Text(str(c.get("pairs", 0)), style=AMBER if c.get("age_ms", 0) > 30000 else "")
                             for c in cams])
        t.caption = "dim = axis not informed by the data"
        t.caption_style = DIM
        self.panel("p-cams", t)

    # ── socket ──
    async def socket_loop(self) -> None:
        while True:
            path = self.args.sock or find_socket(self.agent)
            if not path:
                await asyncio.sleep(1.0)
                continue
            try:
                reader, writer = await asyncio.open_unix_connection(path, limit=16 * 1024 * 1024)
            except OSError:
                await asyncio.sleep(1.0)
                continue
            self.writer, self.connected = writer, True
            self.update_topbar()
            try:
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.handle(ev)
            except (OSError, asyncio.IncompleteReadError):
                pass
            finally:
                self.connected, self.writer = False, None
                try:
                    writer.close()
                except Exception:
                    pass
                self.update_topbar()
            await asyncio.sleep(1.0)

    def send(self, obj: dict) -> bool:
        if not self.writer:
            return False
        self.writer.write((json.dumps(obj) + "\n").encode())
        return True

    # ── event dispatch ──
    def handle(self, ev: dict) -> None:
        kind = ev.get("kind")
        if kind != "hello":
            # A reconnect to the SAME run replays what this viewer already has: skip it by seq.
            seq = ev.get("seq", 0)
            if seq and seq <= self.last_seq:
                return
            if seq:
                self.last_seq = seq
        fn = getattr(self, f"on_ev_{kind}", None)
        if fn:
            fn(ev)
        self.forward(ev)

    def forward(self, ev: dict) -> None:
        """Hand an event to the detail screen on top, if one is open."""
        scr = self.screen
        if isinstance(scr, DetailScreen):
            try:
                scr.feed_event(ev)
            except Exception:
                pass

    def on_ev_hello(self, ev: dict) -> None:
        new_run = (ev.get("pid"), ev.get("start_ms")) != (self.hello.get("pid"), self.hello.get("start_ms"))
        self.hello = ev
        if new_run:
            self.reset_run()
            if ev.get("pid"):
                self.pid = ev["pid"]
            if not self.log_path and ev.get("stdout", "").startswith("/") and os.path.isfile(ev["stdout"]):
                self.log_path = ev["stdout"]
                self.start_tail()
            self.load_events_file()
            self.startup_row(ev, "hello", f"pid {ev.get('pid')}", "", f"cwd {ev.get('cwd', '')}")
        self.update_topbar()

    def reset_run(self) -> None:
        self.startup_rows.clear()
        self.warnings.clear()
        self.event_lines.clear()
        self.peers.clear()
        self.sth_hist.clear()
        self.kl_hist.clear()
        self.state, self.config_ev = {}, {}
        self.max_file_seq = self.last_seq = self.n_warn = 0
        self.sm_state = "?"
        try:
            self.query_one("#warnlog", RichLog).clear()
        except Exception:
            pass

    def startup_row(self, ev: dict, kind: str, what: str, ms, detail: str, style: str = "") -> None:
        cells = [hms(ev.get("t")), kind, what, "" if ms in (None, "") else str(ms), detail]
        row = tuple(Text(str(c), style=style) for c in cells)
        self.startup_rows.append(row)
        ev["_startup_row"] = row
        self.panel("lastev", Text.assemble((hms(ev.get("t"))[:8] + "  ", DIM), (kind + "  ", style or "bold"),
                                           (what + "  ", style), (detail, style or DIM)))

    def on_ev_phase(self, ev: dict) -> None:
        ms = ev.get("ms", 0)
        style = RED if ms > 1000 else (AMBER if ms > 200 else "")
        cum = "" if ev.get("sub") else f"cumulative {ev.get('cum_ms', '')} ms"
        self.startup_row(ev, "phase", ev.get("name", ""), ms, cum, style)

    def on_ev_loaded(self, ev: dict) -> None:
        detail = ev.get("detail", "")
        self.startup_row(ev, "loaded", ev.get("what", ""), "", f"{ev.get('path', '')}  {detail}".strip(),
                         RED if "FAILED" in detail else "")

    def on_ev_sm(self, ev: dict) -> None:
        self.sm_state = ev.get("state", "?")
        style = {"Operating": OK, "Degraded": RED, "Waiting": AMBER}.get(self.sm_state, "")
        self.startup_row(ev, "sm", self.sm_state, "", ev.get("detail", ""), style)
        self.update_topbar()

    def on_ev_wait(self, ev: dict) -> None:
        peers = "peers OK" if ev.get("peers_ok") else "peers MISSING: " + " ".join(ev.get("missing", []))
        lidar = (("lidar OK (" + ev.get("why", "") + ")") if ev.get("lidar_ok") else ("lidar: " + ev.get("why", "")))
        self.startup_row(ev, "wait", "Waiting", "", f"{peers} | {lidar}", AMBER)

    def on_ev_peer(self, ev: dict) -> None:
        name = ev.get("name", "") or f"id {ev.get('id', '?')}"
        self.peers[name] = (ev.get("event", ""), ev.get("t", 0))
        self.startup_row(ev, "peer", ev.get("event", ""), "", f"{ev.get('name', '')} id {ev.get('id', '')}")
        self.render_peers()

    def on_ev_stall(self, ev: dict) -> None:
        age = ev.get("age_ms", -1)
        self.startup_row(ev, "stall", ev.get("stream", ""), "",
                         "no sweep ever arrived" if age < 0 else f"last sweep {age} ms ago", RED)
        self.add_warning(ev, "critical", ev.get("stream", ""), f"stream STALLED ({age} ms)")

    def on_ev_fatal(self, ev: dict) -> None:
        self.startup_row(ev, "FATAL", "", "", ev.get("msg", ""), "bold white on red")
        self.add_warning(ev, "fatal", "fatal", ev.get("msg", ""))

    def on_ev_lifecycle(self, ev: dict) -> None:
        self.startup_row(ev, "lifecycle", ev.get("state", ""), "", ev.get("reason", ""), "cyan")
        if ev.get("state") in ("stopping", "exited"):
            self.sm_state = ev["state"]
            self.update_topbar()

    def on_ev_state(self, ev: dict) -> None:
        self.state = ev
        loc = ev.get("loc", {})
        if loc.get("have"):
            self.sth_hist.append((loc.get("sth") or 0.0) * DEG)
            self.kl_hist.append(loc.get("kl") or 0.0)
        for fn in (self.render_peers, self.render_streams, self.render_loc, self.render_room,
                   self.render_calib, self.render_cams, self.update_topbar):
            fn()

    def on_ev_config(self, ev: dict) -> None:
        self.config_ev = ev
        self.render_gates()

    def on_ev_commands(self, ev: dict) -> None:
        self.commands = ev.get("list", [])

    def on_ev_reply(self, ev: dict) -> None:
        ok = ev.get("ok")
        msg = ev.get("msg", "") + (f"  →  {ev['path']}" if ev.get("path") else "")
        row = (Text(hms(ev.get("t"))), Text(ev.get("cmd", "")), Text("ok" if ok else "FAILED", style=OK if ok else RED),
               Text(msg))
        self.replies.append(row)
        ev["_reply_row"] = row
        self.notify(f"{ev.get('cmd')}: {msg}", severity="information" if ok else "error")

    def on_ev_log(self, ev: dict, from_file: bool = False) -> None:
        if not from_file and ev.get("seq", 0) and ev["seq"] <= self.max_file_seq:
            return   # already read from the events file
        lvl = ev.get("level", "info")
        line = Text(f"{hms(ev.get('t'))} ", style=DIM)
        line.append(ev.get("msg", ""), style=LEVEL_STYLE.get(lvl, ""))
        if not ev.get("term", True):
            line.append("  ⟂", style=DIM)   # not on the terminal: stream/file only
        self.event_lines.append(line)
        ev["_raw_line"], ev["_raw_source"] = line, "events"
        if lvl in ("warning", "critical", "fatal"):
            self.add_warning(ev, lvl, ev.get("tag", ""), ev.get("msg", ""))

    def add_warning(self, ev: dict, lvl: str, tag: str, msg: str) -> None:
        self.n_warn += 1
        style = LEVEL_STYLE.get(lvl, "")
        row = (Text(hms(ev.get("t"))), Text(lvl, style=style), Text(tag), Text(msg, style=style))
        self.warnings.append(row)
        ev["_warn_row"] = row
        try:
            self.query_one("#warnlog", RichLog).write(
                Text.assemble((hms(ev.get("t"))[:8] + "  ", DIM), (msg, style)), scroll_end=True)
        except Exception:
            pass

    # ── the events file: the WHOLE history, not just the replay ring ──
    def load_events_file(self) -> None:
        path = self.hello.get("events")
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, "r", errors="replace") as f:
                for raw in f:
                    try:
                        ev = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("kind") == "log":
                        self.on_ev_log(ev, from_file=True)
                        self.max_file_seq = max(self.max_file_seq, ev.get("seq", 0))
        except OSError:
            pass

    # ── the stdout file ──
    def start_tail(self) -> None:
        if self.tail_task_for != self.log_path:
            self.tail_task_for = self.log_path
            self.run_worker(self.tail_stdout(self.log_path), exclusive=False, group="tail")

    async def tail_stdout(self, path: str) -> None:
        pos = 0
        while path == self.log_path:
            try:
                size = os.path.getsize(path)
                if size < pos:
                    pos = 0   # truncated / rotated
                if size > pos:
                    with open(path, "r", errors="replace") as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos = f.tell()
                    for ln in chunk.splitlines():
                        self.stdout_lines.append(ln)
                        self.forward({"_raw_line": Text(ln), "_raw_source": "stdout"})
            except OSError:
                pass
            await asyncio.sleep(0.3)

    def matches(self, text: str) -> bool:
        return not self.search or self.search.lower() in text.lower()

    # ── actions ──
    def action_detail(self, which: str) -> None:
        cls = {"startup": StartupScreen, "warnings": WarningsScreen, "raw": RawLogScreen,
               "gates": GatesScreen, "commands": CommandsScreen}[which]
        if isinstance(self.screen, cls):
            return
        if isinstance(self.screen, DetailScreen):
            self.pop_screen()
        self.push_screen(cls())

    def ask_command(self, name: str) -> None:
        c = next((c for c in self.commands if c["name"] == name), None)
        if not c:
            return

        def go(yes: bool | None) -> None:
            if not yes:
                return
            cid = self.next_cmd_id
            self.next_cmd_id += 1
            if not self.send({"cmd": name, "id": cid, "args": {}}):
                self.notify("not connected — command not sent", severity="error")

        self.push_screen(ConfirmScreen(name, c.get("description", ""), c.get("confirm", "")), go)

    def action_quit_dialog(self) -> None:
        alive = pid_alive(self.pid) and pid_is_agent(self.pid, self.agent)

        def done(choice: str | None) -> None:
            if choice == "detach":
                self.exit(message=f"detached — {self.agent} keeps running"
                          + (f" (pid {self.pid}); reattach with tools/room_tui.py" if alive else ""))
            elif choice == "stop":
                self.run_worker(self.stop_agent(), exclusive=True, group="stop")

        self.push_screen(QuitScreen(self.pid, alive), done)

    async def stop_agent(self) -> None:
        pid = self.pid
        if not (pid and pid_alive(pid) and pid_is_agent(pid, self.agent)):
            self.exit(message="agent not running")
            return
        os.kill(pid, signal.SIGTERM)   # graceful: the agent deletes its own graph nodes. Never SIGKILL.
        self.notify(f"SIGTERM sent to {pid}; waiting for a clean exit…", timeout=20)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if not pid_alive(pid):
                self.exit(message=f"{self.agent} (pid {pid}) stopped cleanly")
                return
            await asyncio.sleep(0.25)
        self.notify(f"pid {pid} is still running 30 s after SIGTERM. NOT escalating to SIGKILL (it would leak "
                    f"its graph nodes). Check the raw log (F3); press q again to detach.", severity="error", timeout=30)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="room_concept", help="agent name in the socket file name")
    ap.add_argument("--sock", help="explicit socket path (default: auto-discover the newest)")
    ap.add_argument("--log", help="the agent's raw stdout/stderr log file (the launcher passes it)")
    ap.add_argument("--pid", type=int, help="the agent's pid (the launcher passes it; else taken from hello)")
    RoomTUI(ap.parse_args()).run()


if __name__ == "__main__":
    main()
