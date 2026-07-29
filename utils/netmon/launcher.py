"""Shared launcher+monitor used by subcognitive_v2 and cognitive_v2.

Launches the components of a TOML file, keeps a rich console table, and publishes
their live state to the shared /tmp registry. Whichever launcher grabs the monitor
lock first also runs the web server (reading every launcher's registry); the others
just keep updating. If the monitor holder dies, the next launcher takes over.
"""

import json
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
from .battery import BatteryMonitor
from .registry import MonitorLock, RegistryWriter, claim_commands
from .shm_guard import clean_orphan_shm, preflight_dds
from .server import MonitorServer
from .topology import agent_domain, component_dds, parse_endpoint

_FASTDDS_STATISTICS_TOPIC = "_fastdds_statistics_publication_throughput"
_DDS_BRIDGE_BIN = os.path.join(os.path.dirname(__file__), "dds_stats_bridge", "build", "dds_stats_bridge")
_BATTERY_JSON = "/tmp/robocomp_netmon/battery.json"

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


def _make_ice_comm():
    """One long-lived Communicator for all status pings, with bounded connect/invoke
    timeouts. Ice.initialize() defaults to no timeout at all: a single slow/unresponsive
    component would otherwise stall the whole collector loop (and its cpu_percent()
    sampling for every other component) for however long ICE takes to give up."""
    props = Ice.createProperties()
    props.setProperty("Ice.Override.ConnectTimeout", "500")
    props.setProperty("Ice.Override.Timeout", "500")
    init_data = Ice.InitializationData()
    init_data.properties = props
    return Ice.initialize(init_data)


def _ping(comm, ice_string):
    try:
        comm.stringToProxy(ice_string).ice_ping()
        return True
    except Exception:
        return False


def _check_rcnode(console):
    for p in psutil.process_iter(["name"]):
        try:
            if (p.info["name"] or "").lower() == "icebox":
                console.print("[green]✓ rcnode (icebox) is already running[/green]")
                return p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    console.print("[yellow]rcnode not detected. Starting rcnode...[/yellow]")
    script = os.path.expanduser("~/robocomp/tools/rcnode/rcnode.sh")
    env = dict(os.environ)
    env.setdefault("ROBOCOMP", os.path.expanduser("~/robocomp"))
    try:
        subprocess.Popen(["bash", script], env=env, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        time.sleep(2)
        for p in psutil.process_iter(["name"]):
            try:
                if (p.info["name"] or "").lower() == "icebox":
                    return p
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        console.print(f"[red]Failed to start rcnode: {e}[/red]")
    return None


def _check_dds_bridge(console, domain):
    """Idempotent-start a dds_stats_bridge for one DDS domain (measures real publish
    bandwidth via Fast DDS's statistics module; see netmon/dds_stats_bridge/main.cpp)."""
    out_path = f"/tmp/robocomp_netmon/dds_stats_d{domain}.json"
    marker = f"--domain {domain}"
    for p in psutil.process_iter(["cmdline"]):
        try:
            cs = " ".join(p.info.get("cmdline") or [])
            if "dds_stats_bridge" in cs and marker in cs:
                console.print(f"[green]✓ dds_stats_bridge (domain {domain}) already running[/green]")
                return p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if not os.path.exists(_DDS_BRIDGE_BIN):
        console.print(f"[yellow]dds_stats_bridge binary missing at {_DDS_BRIDGE_BIN} "
                      f"(build netmon/dds_stats_bridge/); DDS bandwidth won't be measured[/yellow]")
        return None
    console.print(f"[yellow]Starting dds_stats_bridge for domain {domain}...[/yellow]")
    try:
        proc = subprocess.Popen([_DDS_BRIDGE_BIN, "--domain", str(domain), "--out", out_path],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        time.sleep(1)
        return psutil.Process(proc.pid)
    except Exception as e:
        console.print(f"[red]Failed to start dds_stats_bridge: {e}[/red]")
        return None


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


def _check_shm(console):
    """Remove zombie Fast DDS SHM segments (left in /dev/shm by crashed/killed publishers)
    before components start, so a stuck writer can't reuse a stale/undersized segment."""
    before = [f for f in os.listdir("/dev/shm") if f.startswith(("fastdds_", "sem.fastdds_"))]
    if not before:
        console.print("[green]✓ /dev/shm has no Fast DDS segments[/green]")
        return
    console.print(f"[yellow]{len(before)} Fast DDS SHM file(s) in /dev/shm, running 'fastdds shm clean'...[/yellow]")
    try:
        result = subprocess.run(["fastdds", "shm", "clean"], capture_output=True, text=True, timeout=15)
        after = [f for f in os.listdir("/dev/shm") if f.startswith(("fastdds_", "sem.fastdds_"))]
        if result.returncode == 0:
            console.print(f"[green]✓ fastdds shm clean: {len(before) - len(after)} zombie file(s) removed, "
                          f"{len(after)} still in use[/green]")
        else:
            console.print(f"[red]fastdds shm clean failed: {result.stderr.strip()}[/red]")
    except FileNotFoundError:
        console.print("[red]'fastdds' CLI not found in PATH — skipping SHM cleanup[/red]")
    except Exception as e:
        console.print(f"[red]SHM cleanup error: {e}[/red]")


def _launch(command, cwd, name, extra_env=None):
    log = os.path.expanduser(f"~/.local/logs/{name}")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    env = dict(os.environ, **extra_env) if extra_env else None
    proc = subprocess.Popen(command, cwd=cwd, shell=True, env=env,
                            stdout=open(log + ".out", "w"), stderr=open(log + ".err", "w"))
    time.sleep(0.3)
    try:
        kids = psutil.Process(proc.pid).children()
        ps = kids[0] if kids else psutil.Process(proc.pid)
    except Exception:
        ps = psutil.Process(proc.pid)
    return proc, ps


_CBUILD_CMD = "cmake -B build && make -C build -j32"  # matches the `cbuild` shell alias


def _build(name, cwd):
    """Fire-and-forget cbuild, appended to the same out/err logs the terminal viewer tails
    (so a build triggered from the web shows up where the user is already looking)."""
    log = os.path.expanduser(f"~/.local/logs/{name}")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    marker = f"\n=== cbuild {time.strftime('%H:%M:%S')} ===\n"
    with open(log + ".out", "a") as f:
        f.write(marker)
    subprocess.Popen(_CBUILD_CMD, cwd=cwd, shell=True,
                     stdout=open(log + ".out", "a"), stderr=open(log + ".err", "a"))


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


def _status(info, ice_comm):
    if info.get("is_external"):
        try:
            return "running" if info["psutil_proc"].is_running() else "stopped"
        except Exception:
            return "stopped"
    if info["process"].poll() is not None:
        return "stopped"
    ice = info.get("ice_name")
    if not ice:
        return "alive"
    return "alive" if _ping(ice_comm, ice) else "up"


def run_launcher(toml_path, launcher, layer, start_webots=False, start_rcnode=False,
                 web=True, web_port=8080, web_host="0.0.0.0", iface="lo", no_bw=False,
                 battery_port=None):
    console = Console()

    path = os.path.expanduser(toml_path)
    if not os.path.exists(path):
        console.print(f"[red]Missing components file at {path}[/red]")
        raise SystemExit(1)
    data = toml.load(path)
    components = data["components"]
    general = data.get("general", {})
    start_webots = general.get("start_webots", start_webots)
    start_rcnode = general.get("start_rcnode", start_rcnode)
    battery_port = general.get("battery_port", battery_port)

    rcnode = _check_rcnode(console) if start_rcnode else None
    webots = _check_webots(console) if start_webots else None

    for c in components:
        c["_domain"] = agent_domain(c).get("domain") if layer == "cognitive" else None

    # Components with a [DDS] config get FASTDDS_STATISTICS so they publish real throughput
    # samples; one dds_stats_bridge per distinct domain then turns those into bytes/s the
    # web monitor can read (see netmon/dds_stats_bridge/).
    dds_domains = set()
    dds_component_names = set()
    for c in components:
        info = component_dds(c)
        if info and info.get("domain") is not None:
            dds_domains.add(info["domain"])
            dds_component_names.add(c["name"])
    dds_bridges = {dom: _check_dds_bridge(console, dom) for dom in dds_domains}

    _remove_existing(components, console)
    _check_shm(console)

    # With this launcher's components just killed, garbage-collect FastDDS segments nothing
    # references any more, then confirm shared-memory discovery really works before starting the
    # producers. Both run here because this is the one moment we know our own processes are down.
    clean_orphan_shm(console)
    if not preflight_dds(console):
        raise SystemExit(1)

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
                                   "start_time": time.time(), "is_external": True,
                                   "status": "running", "in_registry": False}
        except Exception:
            pass
    if rcnode:
        try:
            rcnode.cpu_percent(interval=None)
            processes["rcnode"] = {"process": None, "psutil_proc": rcnode, "ice_name": None,
                                   "start_time": time.time(), "is_external": True,
                                   "status": "running", "in_registry": False}
        except Exception:
            pass
    for dom, p in dds_bridges.items():
        if not p:
            continue
        try:
            p.cpu_percent(interval=None)
            processes[f"dds_stats·d{dom}"] = {"process": None, "psutil_proc": p, "ice_name": None,
                                              "start_time": time.time(), "is_external": True,
                                              "status": "running", "in_registry": False}
        except Exception:
            pass
    for c in components:
        console.print(f"Starting {c['name']}...")
        extra_env = {"FASTDDS_STATISTICS": _FASTDDS_STATISTICS_TOPIC} if c["name"] in dds_component_names else None
        proc, ps = _launch(c["cmd"], _expand(c.get("cwd")), c["name"], extra_env)
        ps.cpu_percent(interval=None)
        processes[c["name"]] = {"process": proc, "psutil_proc": ps, "ice_name": c.get("ice_name"),
                                "start_time": time.time(), "status": "unknown",
                                "in_registry": True, "comp": c}

    # Victron VE.Direct battery is hardware on this machine; the launcher that owns the
    # serial (subcognitive passes battery_port) publishes its state to a shared json so the
    # web monitor picks it up regardless of which launcher ends up serving (as DDS/media do).
    battery = BatteryMonitor(port=battery_port) if battery_port else None
    if battery:
        console.print(f"[green]🔋 Battery monitor on {battery_port}[/green]")

    writer = RegistryWriter(launcher, layer)
    mlock = MonitorLock()
    ice_comm = _make_ice_comm()
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
        server = MonitorServer(bw, port=web_port, host=web_host)
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
        if not info or info.get("is_external"):
            return
        console.print(f"[yellow]{action} '{name}' (from web)[/yellow]")
        if action == "build":
            _build(name, _expand(info["comp"].get("cwd")))
            return
        p = info["process"]
        running = p is not None and p.poll() is None
        if action in ("stop", "restart") and running:
            _kill_tree(p)
            running = False
        if action in ("start", "restart") and not running:
            c = info["comp"]
            extra_env = {"FASTDDS_STATISTICS": _FASTDDS_STATISTICS_TOPIC} if name in dds_component_names else None
            proc, ps = _launch(c["cmd"], _expand(c.get("cwd")), name, extra_env)
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
                    info["status"] = _status(info, ice_comm)
                    if info.get("in_registry"):
                        c = info["comp"]
                        reg.append({"name": name, "status": info["status"],
                                    "cpu": info.get("cpu_last", 0.0), "mem": info.get("mem_last", 0.0),
                                    "pid": info["psutil_proc"].pid, "ice_name": c.get("ice_name"),
                                    "cwd": c.get("cwd"), "cmd": c.get("cmd"),
                                    "layer": layer, "domain": c.get("_domain")})
                writer.update(reg)
                if battery:
                    tmp = _BATTERY_JSON + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(battery.snapshot(), f)
                    os.replace(tmp, _BATTERY_JSON)  # atomic: server never reads a half-written file
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
        ice_comm.destroy()
        if battery:
            try:
                os.remove(_BATTERY_JSON)
            except OSError:
                pass
        for info in processes.values():
            if info["process"]:
                _kill_tree(info["process"])
