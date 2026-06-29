#!/usr/bin/env python3

import subprocess
import time
import os
import sys
import termios
import tty
import select as kbselect
from pathlib import Path
import Ice
import psutil
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich import box
import toml
import argparse
import threading
import pprint

parser = argparse.ArgumentParser(description="RoboComp cognitive monitor")
parser.add_argument("file_name", type=str, nargs="?", default="cognitive.toml", help="Path to TOML components file")
args = parser.parse_args()
console = Console()

# Check if Webots is running, start it if not
def check_and_start_webots():
    webots_running = False
    webots_proc = None

    # First check if Webots is already running
    for proc in psutil.process_iter(['name', 'pid']):
        try:
            if 'webots' in proc.info['name'].lower():
                webots_running = True
                webots_proc = proc
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if not webots_running:
        console.print("[yellow]Webots not detected. Starting Webots...[/yellow]")
        try:
            # Start Webots detached (won't block the script)
            proc = subprocess.Popen(
                ["/usr/local/bin/webots"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True  # Detach from parent process
            )
            console.print("[green]✓ Webots started successfully[/green]")
            time.sleep(2)  # Give Webots a moment to initialize

            # Find the actual Webots process
            for p in psutil.process_iter(['name', 'pid']):
                try:
                    if 'webots' in p.info['name'].lower():
                        webots_proc = p
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            console.print(f"[red]Failed to start Webots: {e}[/red]")
            return None
    else:
        console.print("[green]✓ Webots is already running[/green]")

    return webots_proc

webots_process = check_and_start_webots()

def cpu_usage_bar(cpu_percent, width=10):
    """Return a colored CPU usage bar."""
    cpu_percent = min(cpu_percent, 100)  # clamp so multi-core >100% doesn't overflow the bar
    filled = int((cpu_percent / 100) * width)
    empty = width - filled
    if cpu_percent < 50:
        color = "green"
    elif cpu_percent < 80:
        color = "yellow"
    else:
        color = "red"
    bar = f"[{color}]" + ("█" * filled + "░" * empty) + "[/]"
    return bar

# Load components from TOML file
def toml_loader():
    config_path = os.path.expanduser(args.file_name)
    if not os.path.exists(config_path):
        console.print(f"[red]Missing file with components at {config_path}[/red]")
        exit(1)

    components = toml.load(args.file_name)["components"]
    return components

def ping_proxy(ice_string):
    try:
        # Initialize Ice communicator if not already done
        with Ice.initialize() as communicator:
            proxy = communicator.stringToProxy(ice_string)
            # Narrow to Ice::Object (the base interface)
            obj = proxy.ice_ping()
            return True
    except Exception as e:
        return False

def green(text):
    return f"\033[92m{text}\033[0m"

def red(text):
    return f"\033[91m{text}\033[0m"

def format_uptime(seconds):
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"

processes = {}

# Add Webots to monitoring if it's running
if webots_process:
    try:
        webots_process.cpu_percent(interval=None)  # Initialize CPU tracking
        processes["Webots"] = {
            "process": None,  # We don't have a subprocess.Popen object for Webots
            "psutil_proc": webots_process,
            "ice_name": None,
            "start_time": time.time(),
            "is_webots": True
        }
    except Exception:
        pass

def expand_path(p):
    return os.path.expanduser(p) if p else None

# Start all components
def launch_process(command, cwd=None, name=None):
    stdout_path = os.path.expanduser(f"~/.local/logs/{name}.out") if name else os.devnull
    stderr_path = os.path.expanduser(f"~/.local/logs/{name}.err") if name else os.devnull
    os.makedirs(os.path.dirname(stdout_path), exist_ok=True)

    stdout = open(stdout_path, "w")
    stderr = open(stderr_path, "w")

    proc = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdout=stdout,
        stderr=stderr
    )
    time.sleep(0.3)  # wait for child process to start
    try:
        children = psutil.Process(proc.pid).children()
        if children:
            ps_proc = children[0]  # use real process, not shell
        else:
            ps_proc = psutil.Process(proc.pid)
    except Exception:
        ps_proc = psutil.Process(proc.pid)
    return proc, ps_proc

components = toml_loader()

def remove_existing_processes(components):
    for comp in components:
        cmd = comp.get("cmd", "")
        if not cmd:
            continue
        # Extract the executable name from the command
        exe_name = cmd.split()[0].split("/")[-1]
        comp_name = comp.get("name", "")

        for p in psutil.process_iter(['name', 'cmdline']):
            try:
                p_name = p.info.get('name', '') or ''
                p_cmdline = p.info.get('cmdline', []) or []
                cmd_str = " ".join(p_cmdline)

                # Identify if process belongs to the component (match exe_name or comp_name)
                if (exe_name and exe_name in p_name) or (exe_name and exe_name in cmd_str) or (comp_name and comp_name in p_name):
                    # Do not kill this script or webots
                    if "cognitive.py" not in cmd_str and "subcognitive.py" not in cmd_str and "webots" not in p_name.lower() and p.pid != os.getpid():
                        console.print(f"[yellow]Killing existing process meant for '{comp_name}' (PID: {p.pid})[/yellow]")
                        p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                pass
    time.sleep(1)

remove_existing_processes(components)

def print_components_table(components):
    table = Table(title="🧠 Loaded Cognitive Components", box=box.SIMPLE_HEAVY)
    table.add_column("Name", style="bold cyan")
    table.add_column("Endpoint", style="bold blue")
    table.add_column("Port", style="bold magenta")
    table.add_column("CWD", style="dim")
    table.add_column("Command", style="magenta")

    for comp in components:
        ice = comp.get("ice_name", "") or ""
        ep = ice.split(":")[0] if ":" in ice else "-"
        port = ice.split("-p ")[1].split()[0] if "-p " in ice else "-"

        table.add_row(
            comp["name"],
            ep,
            port,
            comp.get("cwd", "-"),
            comp.get("cmd", "-")
        )

    console.print(table)

print_components_table(components)

for comp in components:
    cwd = expand_path(comp.get("cwd"))
    command = comp["cmd"]
    print(f"Starting {comp['name']}...")
    proc, ps_proc = launch_process(command, cwd=cwd, name=comp["name"])
    ps_proc.cpu_percent(interval=None)  # <<< Add this here
    processes[comp["name"]] = {
        "process": proc,
        "psutil_proc": ps_proc,
        "ice_name": comp.get("ice_name"),
        "start_time": time.time()
    }
    if comp["name"] == "room_concept":
        time.sleep(5)    # cover room_concept's slow first compute (~3.5s) before launching dependents
    else:
        time.sleep(1)    # stagger the remaining agents one second apart, sequentially

#Monitor loop
selected_idx = 0

def build_table(selected=0):
    table = Table(title="🧠 RoboComp Cognitive Monitor", box=box.SIMPLE_HEAVY)
    table.add_column("", style="bold")
    table.add_column("Name", style="bold cyan")
    table.add_column("Endpoint", style="bold blue")
    table.add_column("Port", style="bold magenta")
    table.add_column("Status", style="bold")
    table.add_column("Uptime", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("CPU", justify="right")

    for idx, (name, info) in enumerate(processes.items()):
        ice_name = info.get("ice_name")
        ice_str = ice_name or ""
        ep = ice_str.split(":")[0] if ":" in ice_str else "-"
        port = ice_str.split("-p ")[1].split()[0] if "-p " in ice_str else "-"

        status = "[yellow]⏳ Checking...[/yellow]"

        # Special handling for Webots
        if info.get("is_webots", False):
            try:
                if info["psutil_proc"].is_running():
                    status = "[green]✅ Running[/green]"
                else:
                    status = "[red]❌ Stopped[/red]"
            except Exception:
                status = "[red]❌ Stopped[/red]"
        else:
            # Regular component handling
            proc = info["process"]
            running = proc.poll() is None

            if running:
                # Ice ping
                try:
                    if ice_name:
                        ping_proxy(ice_name)
                        status = "[green]✅ Alive[/green]"
                    else:
                        status = "[blue]⚠️ No ICE[/blue]"
                except Exception:
                    status = "[red]❌ Down[/red]"
            else:
                status = "[red]❌ Stopped[/red]"

        # Uptime
        uptime = time.time() - info["start_time"]
        uptime_str = format_uptime(uptime)

        # Resource usage
        try:
            mem = info.get("mem_last", 0.0)
            cpu = info.get("cpu_last", 0.0)
        except Exception:
            mem, cpu = 0, 0

        is_selected = idx == selected
        table.add_row(
            "▶" if is_selected else " ",
            name,
            ep,
            port,
            status,
            uptime_str,
            f"{mem:6.1f} MB",
            #f"{cpu:5.1f}%"
            f"{cpu:5.1f}% {cpu_usage_bar(cpu)}",
            style="reverse" if is_selected else None
        )

    return table

def update_cpu_mem():
    while True:
        for info in processes.values():
            try:
                proc = info["psutil_proc"]
                if proc.is_running():
                    info["mem_last"] = proc.memory_info().rss / (1024 ** 2)
                    info["cpu_last"] = proc.cpu_percent(interval=0.0)
            except Exception:
                info["mem_last"] = 0
                info["cpu_last"] = 0
        time.sleep(1)

threading.Thread(target=update_cpu_mem, daemon=True).start()

def tail_file(path, n=20):
    try:
        with open(path, errors="replace") as f:
            return f.readlines()[-n:]
    except Exception:
        return []

def build_view():
    table = build_table(selected_idx)
    names = list(processes.keys())
    sel_name = names[selected_idx] if names else None

    # líneas disponibles para el panel = alto terminal - filas tabla - overhead (título/
    # cabecera/separadores de la tabla + bordes del panel). Conservador para evitar scroll.
    n = max(3, console.size.height - len(processes) - 8)

    out = tail_file(os.path.expanduser(f"~/.local/logs/{sel_name}.out"), n)
    err = tail_file(os.path.expanduser(f"~/.local/logs/{sel_name}.err"), n)

    lines = [(l.rstrip("\n"), None) for l in out]
    if any(l.strip() for l in err):
        lines.append(("── stderr ──", "bold red"))
        lines += [(l.rstrip("\n"), "red") for l in err]
    lines = lines[-n:]  # recorta al alto disponible

    body = Text()
    if not lines:
        body.append("(no output)", style="dim")
    for i, (txt, style) in enumerate(lines):
        if i:
            body.append("\n")
        body.append(txt, style=style)

    panel = Panel(body, title=f"📜 {sel_name} — stdout/stderr", border_style="blue")
    help_line = Text("  ↑/↓ o j/k: seleccionar agente   ·   Ctrl+C: salir", style="dim italic")
    return Group(help_line, table, panel)

_tty_fd = None
_tty_old = None

def key_listener():
    global selected_idx
    while True:
        if not kbselect.select([_tty_fd], [], [], 0.1)[0]:
            continue
        data = os.read(_tty_fd, 16)  # read at the fd level so escape sequences arrive whole
        n = len(processes)
        if data in (b"\x1b[A", b"k", b"\x1bOA"):      # up / k
            selected_idx = (selected_idx - 1) % n
        elif data in (b"\x1b[B", b"j", b"\x1bOB"):    # down / j
            selected_idx = (selected_idx + 1) % n

def restore_terminal():
    if _tty_old is not None:
        termios.tcsetattr(_tty_fd, termios.TCSADRAIN, _tty_old)

if sys.stdin.isatty():
    _tty_fd = sys.stdin.fileno()
    _tty_old = termios.tcgetattr(_tty_fd)
    tty.setcbreak(_tty_fd)  # keeps ISIG so Ctrl+C still works
    threading.Thread(target=key_listener, daemon=True).start()

try:
    with Live(build_view(), refresh_per_second=1, console=console, screen=False) as live:
        while True:
            time.sleep(0.2)
            live.update(build_view())
except KeyboardInterrupt:
    console.print("\n[yellow]Exiting. Terminating all processes...[/yellow]")
    for name, info in processes.items():
        if info["process"]:  # Only terminate if we have a subprocess object
            info["process"].terminate()
    for name, info in processes.items():
        if info["process"]:  # Only wait if we have a subprocess object
            info["process"].wait()
finally:
    restore_terminal()
