#!/usr/bin/env python3
"""
room_tui.py — terminal viewer for room_concept's live status stream.

    tools/room_run.sh                       # starts the agent in the background and opens this
    python3 tools/room_tui.py               # (re)attach to a running room_concept
    python3 tools/room_tui.py --log tmp/agent_<ts>.log --pid <pid>

The agent (rc::StatusStream, common/status_stream/) serves newline-delimited JSON on
$XDG_RUNTIME_DIR/rc_status/<agent>_<id>.sock and appends every event to tmp/agent_events_<start>.jsonl.
This viewer auto-discovers the socket, replays what the agent kept (hello, startup phases, loaded
files, state-machine transitions, the last 200 log lines, the last state snapshot), and reconnects on
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
import re
import signal
import time

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (Button, DataTable, Footer, Input, Label, RichLog, Static,
                             TabbedContent, TabPane)

LEVEL_STYLE = {"debug": "dim", "info": "", "warning": "yellow", "critical": "bold red", "fatal": "bold white on red"}
MAX_LOG_LINES = 20000


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


def hms(t_ms: int | float | None) -> str:
    if not t_ms:
        return ""
    return _dt.datetime.fromtimestamp(t_ms / 1000.0).strftime("%H:%M:%S.%f")[:-3]


# ── modal: quit ───────────────────────────────────────────────────────────────────────────────────────
class QuitScreen(ModalScreen[str]):
    DEFAULT_CSS = """
    QuitScreen { align: center middle; }
    #dlg { width: 72; height: auto; border: thick $warning; background: $surface; padding: 1 2; }
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


# ── the app ───────────────────────────────────────────────────────────────────────────────────────────
class RoomTUI(App):
    TITLE = "room_concept"
    CSS = """
    #status { height: 1; background: $boost; padding: 0 1; }
    #rawbar { height: 1; padding: 0 1; color: $text-muted; }
    #filter { display: none; }
    #filter.visible { display: block; }
    DataTable { height: 1fr; }
    RichLog { height: 1fr; }
    """
    BINDINGS = [
        Binding("q", "quit_dialog", "Quit"),
        Binding("slash", "search", "Search"),
        Binding("t", "toggle_source", "Raw: stdout/events"),
        Binding("f", "toggle_follow", "Follow"),
        Binding("escape", "clear_search", "Clear search", show=False),
    ]

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.agent = args.agent
        self.pid: int | None = args.pid
        self.log_path: str | None = args.log
        self.hello: dict = {}
        self.connected = False
        self.sock_path: str | None = None
        self.sm_state = "?"
        self.writer: asyncio.StreamWriter | None = None
        # raw log
        self.raw_source = "stdout" if args.log else "events"
        self.stdout_lines: collections.deque[str] = collections.deque(maxlen=MAX_LOG_LINES)
        self.event_lines: collections.deque[Text] = collections.deque(maxlen=MAX_LOG_LINES)
        self.max_file_seq = 0
        self.last_seq = 0
        self.search = ""
        self.follow = True
        self.n_warn = 0

    # ── layout ──
    def compose(self) -> ComposeResult:
        yield Static("", id="status")
        with TabbedContent(id="tabs"):
            with TabPane("Startup", id="tab-startup"):
                yield DataTable(id="startup", zebra_stripes=True, cursor_type="row")
            with TabPane("Warnings", id="tab-warn"):
                yield DataTable(id="warn", zebra_stripes=True, cursor_type="row")
            with TabPane("Raw log", id="tab-raw"):
                yield Static("", id="rawbar")
                yield Input(placeholder="search (case-insensitive substring; Esc clears)", id="filter")
                yield RichLog(id="raw", max_lines=MAX_LOG_LINES, wrap=False, highlight=False, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one("#startup", DataTable)
        t.add_columns("time", "kind", "what", "ms", "detail")
        w = self.query_one("#warn", DataTable)
        w.add_columns("time", "level", "tag", "message")
        self.update_status()
        self.update_rawbar()
        self.run_worker(self.socket_loop(), exclusive=False, group="sock")
        if self.log_path:
            self.run_worker(self.tail_stdout(), exclusive=False, group="tail")
        self.set_interval(1.0, self.update_status)

    # ── status line ──
    def update_status(self) -> None:
        if not self.is_running or not self.screen_stack:
            return
        try:
            self.query_one("#status", Static)
        except Exception:
            return   # tearing down
        alive = pid_alive(self.pid)
        conn = "[green]connected[/]" if self.connected else "[yellow]reconnecting…[/]"
        proc = (f"pid {self.pid} " + ("[green]alive[/]" if alive else "[red]not running[/]")) if self.pid else "pid ?"
        agent = f"{self.hello.get('agent', self.agent)} id {self.hello.get('id', '?')}"
        warn = f"  ·  [yellow]{self.n_warn} warning(s)[/]" if self.n_warn else ""
        self.query_one("#status", Static).update(
            f"[b]{agent}[/]  ·  {proc}  ·  socket {conn}  ·  SM [b]{self.sm_state}[/]{warn}")

    def update_rawbar(self) -> None:
        src = (f"stdout file {os.path.basename(self.log_path or '')}" if self.raw_source == "stdout"
               else f"events log {os.path.basename(self.hello.get('events') or '') or '(socket replay)'}"
                    "  (⟂ = not printed on the terminal)")
        flt = f"  ·  filter: '{self.search}'" if self.search else ""
        self.query_one("#rawbar", Static).update(f"source: {src}  (t toggles){flt}  ·  follow {'on' if self.follow else 'off'}")

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
            self.sock_path, self.writer, self.connected = path, writer, True
            self.update_status()
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
                self.update_status()
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

    def on_ev_hello(self, ev: dict) -> None:
        new_run = (ev.get("pid"), ev.get("start_ms")) != (self.hello.get("pid"), self.hello.get("start_ms"))
        self.hello = ev
        if new_run:
            self.reset_run()
            if ev.get("pid"):
                self.pid = ev["pid"]
            if not self.log_path and ev.get("stdout", "").startswith("/") and os.path.isfile(ev["stdout"]):
                self.log_path = ev["stdout"]
                self.run_worker(self.tail_stdout(), exclusive=False, group="tail")
            self.load_events_file()
            self.startup_row(ev, "hello", f"pid {ev.get('pid')}", "", f"cwd {ev.get('cwd', '')}")
        self.update_status()
        self.update_rawbar()

    def reset_run(self) -> None:
        self.query_one("#startup", DataTable).clear()
        self.query_one("#warn", DataTable).clear()
        self.event_lines.clear()
        self.max_file_seq = 0
        self.last_seq = 0
        self.n_warn = 0
        self.sm_state = "?"

    def startup_row(self, ev: dict, kind: str, what: str, ms, detail: str, style: str = "") -> None:
        t = self.query_one("#startup", DataTable)
        cells = [hms(ev.get("t")), kind, what, "" if ms in (None, "") else str(ms), detail]
        t.add_row(*[Text(str(c), style=style) for c in cells])
        if self.follow:
            t.move_cursor(row=t.row_count - 1)

    def on_ev_phase(self, ev: dict) -> None:
        ms = ev.get("ms", 0)
        style = "bold red" if ms > 1000 else ("yellow" if ms > 200 else "")
        cum = "" if ev.get("sub") else f"cumulative {ev.get('cum_ms', '')} ms"
        self.startup_row(ev, "phase", ev.get("name", ""), ms, cum, style)

    def on_ev_loaded(self, ev: dict) -> None:
        detail = ev.get("detail", "")
        style = "red" if "FAILED" in detail else ""
        self.startup_row(ev, "loaded", ev.get("what", ""), "", f"{ev.get('path', '')}  {detail}".strip(), style)

    def on_ev_sm(self, ev: dict) -> None:
        self.sm_state = ev.get("state", "?")
        style = {"Operating": "green", "Degraded": "bold red", "Waiting": "yellow"}.get(self.sm_state, "")
        self.startup_row(ev, "sm", self.sm_state, "", ev.get("detail", ""), style)
        self.update_status()

    def on_ev_wait(self, ev: dict) -> None:
        peers = "peers OK" if ev.get("peers_ok") else "peers MISSING: " + " ".join(ev.get("missing", []))
        lidar = ("lidar OK (" if ev.get("lidar_ok") else "lidar: ") + ev.get("why", "") + (")" if ev.get("lidar_ok") else "")
        self.startup_row(ev, "wait", "Waiting", "", f"{peers} | {lidar}", "yellow")

    def on_ev_peer(self, ev: dict) -> None:
        self.startup_row(ev, "peer", ev.get("event", ""), "", f"{ev.get('name', '')} id {ev.get('id', '')}")

    def on_ev_stall(self, ev: dict) -> None:
        age = ev.get("age_ms", -1)
        self.startup_row(ev, "stall", ev.get("stream", ""), "",
                         "no sweep ever arrived" if age < 0 else f"last sweep {age} ms ago", "bold red")
        self.add_warning(ev, "critical", ev.get("stream", ""), f"stream STALLED ({age} ms)")

    def on_ev_fatal(self, ev: dict) -> None:
        self.startup_row(ev, "FATAL", "", "", ev.get("msg", ""), "bold white on red")
        self.add_warning(ev, "fatal", "fatal", ev.get("msg", ""))

    def on_ev_lifecycle(self, ev: dict) -> None:
        self.startup_row(ev, "lifecycle", ev.get("state", ""), "", ev.get("reason", ""), "cyan")
        if ev.get("state") in ("stopping", "exited"):
            self.sm_state = ev["state"]
            self.update_status()

    def on_ev_log(self, ev: dict, from_file: bool = False) -> None:
        if not from_file and ev.get("seq", 0) and ev["seq"] <= self.max_file_seq:
            return   # already read from the events file
        lvl = ev.get("level", "info")
        line = Text(f"{hms(ev.get('t'))} ", style="dim")
        line.append(ev.get("msg", ""), style=LEVEL_STYLE.get(lvl, ""))
        if not ev.get("term", True):
            line.append("  ⟂", style="dim")   # not on the terminal: stream/file only
        self.event_lines.append(line)
        if self.raw_source == "events":
            self.write_raw(line)
        if lvl in ("warning", "critical", "fatal"):
            self.add_warning(ev, lvl, ev.get("tag", ""), ev.get("msg", ""))

    def add_warning(self, ev: dict, lvl: str, tag: str, msg: str) -> None:
        self.n_warn += 1
        w = self.query_one("#warn", DataTable)
        style = LEVEL_STYLE.get(lvl, "")
        w.add_row(Text(hms(ev.get("t"))), Text(lvl, style=style), Text(tag), Text(msg, style=style))
        if self.follow:
            w.move_cursor(row=w.row_count - 1)

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
        if self.raw_source == "events":
            self.redraw_raw()

    # ── raw log ──
    async def tail_stdout(self) -> None:
        path = self.log_path
        pos = 0
        first = True
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
                    lines = chunk.splitlines()
                    for ln in lines:
                        self.stdout_lines.append(ln)
                    if self.raw_source == "stdout":
                        if first:
                            self.redraw_raw()
                        else:
                            for ln in lines:
                                self.write_raw(Text(ln))
                    first = False
            except OSError:
                pass
            await asyncio.sleep(0.3)

    def matches(self, text: str) -> bool:
        return not self.search or self.search.lower() in text.lower()

    def write_raw(self, line: Text) -> None:
        if self.matches(line.plain):
            log = self.query_one("#raw", RichLog)
            log.write(line, scroll_end=self.follow)

    def redraw_raw(self) -> None:
        log = self.query_one("#raw", RichLog)
        log.clear()
        src = (Text(s) for s in self.stdout_lines) if self.raw_source == "stdout" else iter(self.event_lines)
        for line in src:
            if self.matches(line.plain):
                log.write(line, scroll_end=False)
        if self.follow:
            log.scroll_end(animate=False)

    # ── actions ──
    def action_search(self) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab-raw"
        inp = self.query_one("#filter", Input)
        inp.add_class("visible")
        inp.value = self.search
        inp.focus()

    @on(Input.Submitted, "#filter")
    def _search_submitted(self, ev: Input.Submitted) -> None:
        self.search = ev.value.strip()
        self.update_rawbar()
        self.redraw_raw()
        self.query_one("#raw", RichLog).focus()

    def action_clear_search(self) -> None:
        inp = self.query_one("#filter", Input)
        inp.remove_class("visible")
        if self.search:
            self.search = ""
            self.update_rawbar()
            self.redraw_raw()

    def action_toggle_source(self) -> None:
        self.raw_source = "events" if self.raw_source == "stdout" else "stdout"
        if self.raw_source == "stdout" and not self.log_path:
            self.notify("no stdout log known (pass --log, or the agent's stdout is a terminal)", severity="warning")
            self.raw_source = "events"
        self.update_rawbar()
        self.redraw_raw()

    def action_toggle_follow(self) -> None:
        self.follow = not self.follow
        self.update_rawbar()

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
                    f"its graph nodes). Check the raw log; press q again to detach.", severity="error", timeout=30)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="room_concept", help="agent name in the socket file name")
    ap.add_argument("--sock", help="explicit socket path (default: auto-discover the newest)")
    ap.add_argument("--log", help="the agent's raw stdout/stderr log file (the launcher passes it)")
    ap.add_argument("--pid", type=int, help="the agent's pid (the launcher passes it; else taken from hello)")
    args = ap.parse_args()
    app = RoomTUI(args)
    app.run()


if __name__ == "__main__":
    main()
