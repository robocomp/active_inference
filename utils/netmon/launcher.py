"""Shared launcher+monitor used by subcognitive_v2 and cognitive_v2.

Launches the components of a TOML file, keeps a rich console table, and publishes
their live state to the shared /tmp registry. Whichever launcher grabs the monitor
lock first also runs the web server (reading every launcher's registry); the others
just keep updating. If the monitor holder dies, the next launcher takes over.
"""

import os
import signal
import subprocess
import threading
import time

import Ice
import psutil
import toml
from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table

from .bandwidth import BandwidthMonitor
from .registry import MonitorLock, RegistryWriter, claim_commands
from .server import MonitorServer
from .topology import agent_domain, parse_endpoint

_STATUS_RENDER = {
    "alive": "[green]✅ Alive[/green]",
    "running": "[green]✅ Running[/green]",
    "up": "[yellow]🟢 Up (ICE ✗)[/yellow]",
    "stopped": "[red]❌ Stopped[/red]",
    "unknown": "[yellow]⏳ Checking...[/yellow]",
}


def _cpu_bar(pct, width=10):
    filled = int((pct / 100) * width)
    color = "green" if pct < 50 else "yellow" if pct < 80 else "red"
    return f"[{color}]" + ("█" * filled + "░" * (width - filled)) + "[/]"


def _expand(p):
    return os.path.expanduser(p) if p else None


def _fmt_uptime(sec):
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h:02}:{m:02}:{s:02}"


def _ping(ice_string):
    try:
        with Ice.initialize() as comm:
            comm.stringToProxy(ice_string).ice_ping()
            return True
    except Exception:
        return False


def _check_webots(console):
    for p in psutil.process_iter(["name"]):
        try:
            if "webots" in (p.info["name"] or "").lower():
                console.print("[green]✓ Webots is already running[/green]")
                return p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    console.print("[yellow]Webots not detected. Starting Webots...[/yellow]")
    try:
        subprocess.Popen(["/usr/local/bin/webots"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        time.sleep(2)
        for p in psutil.process_iter(["name"]):
            try:
                if "webots" in (p.info["name"] or "").lower():
                    return p
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        console.print(f"[red]Failed to start Webots: {e}[/red]")
    return None


def _launch(command, cwd, name):
    log = os.path.expanduser(f"~/.local/logs/{name}")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    proc = subprocess.Popen(command, cwd=cwd, shell=True,
                            stdout=open(log + ".out", "w"), stderr=open(log + ".err", "w"))
    time.sleep(0.3)
    try:
        kids = psutil.Process(proc.pid).children()
        ps = kids[0] if kids else psutil.Process(proc.pid)
    except Exception:
        ps = psutil.Process(proc.pid)
    return proc, ps


def _remove_existing(components, console):
    for comp in components:
        cmd = comp.get("cmd", "")
        if not cmd:
            continue
        exe = cmd.split()[0].split("/")[-1]
        cname = comp.get("name", "")
        for p in psutil.process_iter(["name", "cmdline"]):
            try:
                pn = p.info.get("name", "") or ""
                cs = " ".join(p.info.get("cmdline", []) or [])
                if not cs:
                    continue
                # precise match: exact process name (comm is truncated to 15) or the full cmd
                if pn == exe[:15] or cmd in cs:
                    if "cognitive" not in cs and "webots" not in pn.lower() and p.pid != os.getpid():
                        console.print(f"[yellow]Killing existing '{cname}' (PID {p.pid})[/yellow]")
                        p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                pass
    time.sleep(1)


def _kill_tree(proc, timeout=5):
    """Kill a shell-launched process AND its children (the real binary), SIGKILL stragglers.

    shell=True means proc.pid is /bin/sh; terminating only that orphans the real process,
    so we terminate the whole tree.
    """
    if proc is None:
        return
    try:
        parent = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True)
    procs.append(parent)
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def _status(info):
    if info.get("is_webots"):
        try:
            return "running" if info["psutil_proc"].is_running() else "stopped"
        except Exception:
            return "stopped"
    if info["process"].poll() is not None:
        return "stopped"
    ice = info.get("ice_name")
    if not ice:
        return "alive"
    return "alive" if _ping(ice) else "up"


def run_launcher(toml_path, launcher, layer, start_webots=False,
                 web=True, web_port=8080, iface="lo", no_bw=False):
    console = Console()

    webots = _check_webots(console) if start_webots else None

    path = os.path.expanduser(toml_path)
    if not os.path.exists(path):
        console.print(f"[red]Missing components file at {path}[/red]")
        raise SystemExit(1)
    components = toml.load(path)["components"]
    for c in components:
        c["_domain"] = agent_domain(c).get("domain") if layer == "cognitive" else None

    _remove_existing(components, console)

    tbl = Table(title=f"🧠 Loaded Components — {launcher} [{layer}]", box=box.SIMPLE_HEAVY)
    for col, st in [("Name", "bold cyan"), ("Endpoint", "bold blue"), ("Port", "bold magenta"),
                    ("CWD", "dim"), ("Command", "magenta")]:
        tbl.add_column(col, style=st)
    for c in components:
        ep = parse_endpoint(c.get("ice_name"))
        tbl.add_row(c["name"], ep["identity"] if ep else "-",
                    str(ep["port"]) if ep and ep.get("port") else "-",
                    c.get("cwd", "-"), c.get("cmd", "-"))
    console.print(tbl)

    processes = {}
    if webots:
        try:
            webots.cpu_percent(interval=None)
            processes["Webots"] = {"process": None, "psutil_proc": webots, "ice_name": None,
                                   "start_time": time.time(), "is_webots": True,
                                   "status": "running", "in_registry": False}
        except Exception:
            pass
    for c in components:
        console.print(f"Starting {c['name']}...")
        proc, ps = _launch(c["cmd"], _expand(c.get("cwd")), c["name"])
        ps.cpu_percent(interval=None)
        processes[c["name"]] = {"process": proc, "psutil_proc": ps, "ice_name": c.get("ice_name"),
                                "start_time": time.time(), "status": "unknown",
                                "in_registry": True, "comp": c}

    writer = RegistryWriter(launcher, layer)
    mlock = MonitorLock()
    srv = {"server": None, "bw": None}

    def take_monitor_role():
        bw = None
        if not no_bw:
            bw = BandwidthMonitor([], iface=iface)
            bw.start()
            if bw.available:
                console.print(f"[green]✓ Bandwidth capture on {iface}[/green]")
            else:
                console.print(f"[yellow]⚠ Bandwidth off: {bw.error}[/yellow]")
        server = MonitorServer(bw, port=web_port)
        try:
            url = server.start()
        except OSError as e:
            console.print(f"[yellow]⚠ Puerto :{web_port} ocupado ({e}); otro proceso sirve la web. "
                          f"Sigo actualizando el registro.[/yellow]")
            if bw:
                bw.stop()
            mlock.release()
            return False
        console.print(f"[bold green]🌐 Monitor at {url}  (this launcher is serving)[/bold green]")
        srv["server"], srv["bw"] = server, bw
        return True

    def run_command(cmd):
        name, action = cmd.get("name"), cmd.get("action")
        info = processes.get(name)
        if not info or info.get("is_webots"):
            return
        p = info["process"]
        running = p is not None and p.poll() is None
        console.print(f"[yellow]{action} '{name}' (from web)[/yellow]")
        if action in ("stop", "restart") and running:
            _kill_tree(p)
            running = False
        if action in ("start", "restart") and not running:
            c = info["comp"]
            proc, ps = _launch(c["cmd"], _expand(c.get("cwd")), name)
            ps.cpu_percent(interval=None)
            info.update({"process": proc, "psutil_proc": ps,
                         "start_time": time.time(), "status": "unknown"})

    def collector():
        next_serve_try = 0.0
        while True:
            try:
                for cmd in claim_commands(set(processes.keys())):
                    run_command(cmd)
                reg = []
                for name, info in processes.items():
                    try:
                        p = info["psutil_proc"]
                        if p.is_running():
                            info["mem_last"] = p.memory_info().rss / (1024 ** 2)
                            info["cpu_last"] = p.cpu_percent(interval=0.0)
                        else:
                            info["mem_last"] = info["cpu_last"] = 0
                    except Exception:
                        info["mem_last"] = info["cpu_last"] = 0
                    info["status"] = _status(info)
                    if info.get("in_registry"):
                        c = info["comp"]
                        reg.append({"name": name, "status": info["status"],
                                    "cpu": info.get("cpu_last", 0.0), "mem": info.get("mem_last", 0.0),
                                    "pid": info["psutil_proc"].pid, "ice_name": c.get("ice_name"),
                                    "cwd": c.get("cwd"), "cmd": c.get("cmd"),
                                    "layer": layer, "domain": c.get("_domain")})
                writer.update(reg)
                if web:
                    if not mlock.owned:
                        if time.time() >= next_serve_try and mlock.try_acquire(web_port):
                            if take_monitor_role() is False:
                                next_serve_try = time.time() + 10
                    else:
                        mlock.refresh(web_port)
            except Exception as e:
                console.print(f"[red]collector: {e}[/red]")
            time.sleep(1)

    threading.Thread(target=collector, daemon=True).start()

    def build_table():
        t = Table(title=f"🧠 {launcher} monitor v2 [{layer}]", box=box.SIMPLE_HEAVY)
        for col, kw in [("Name", {"style": "bold cyan"}), ("Endpoint", {"style": "bold blue"}),
                        ("Port", {"style": "bold magenta"}), ("Status", {"style": "bold"}),
                        ("Uptime", {"justify": "right"}), ("Memory", {"justify": "right"}),
                        ("CPU", {"justify": "right"})]:
            t.add_column(col, **kw)
        for name, info in processes.items():
            ep = parse_endpoint(info.get("ice_name"))
            cpu = info.get("cpu_last", 0.0)
            t.add_row(name, ep["identity"] if ep else "-",
                      str(ep["port"]) if ep and ep.get("port") else "-",
                      _STATUS_RENDER.get(info.get("status", "unknown"), _STATUS_RENDER["unknown"]),
                      _fmt_uptime(time.time() - info["start_time"]),
                      f"{info.get('mem_last', 0.0):6.1f} MB",
                      f"{cpu:5.1f}% {_cpu_bar(cpu)}")
        return t

    try:
        with Live(build_table(), refresh_per_second=1, console=console, screen=False) as live:
            while True:
                time.sleep(1)
                live.update(build_table())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting. Terminating all processes...[/yellow]")
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore further Ctrl+C during cleanup
        if srv["bw"]:
            srv["bw"].stop()
        if srv["server"]:
            srv["server"].stop()
        mlock.release()
        writer.remove()
        for info in processes.values():
            if info["process"]:
                _kill_tree(info["process"])
