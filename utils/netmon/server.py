"""Local web dashboard. Reads the shared /tmp registry (all launchers) and serves
the unified interconnection map + live state (stdlib http.server, no framework).

The frontend polls /api/state every second. Topology is rebuilt from the merged
registry with a short TTL (parsing configs is the expensive part).
"""

import glob
import json
import os
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import psutil

from .bandwidth import connection_edges
from .registry import merged_components, push_command
from .topology import build_full_topology

_STATIC = os.path.join(os.path.dirname(__file__), "static")
_LOGS_DIR = os.path.expanduser("~/.local/logs")
_TOPO_TTL = 2.0
_DDS_STATS_GLOB = "/tmp/robocomp_netmon/dds_stats_d*.json"
_MEDIA_STATS_GLOB = "/tmp/robocomp_netmon/media_stats_*.json"
_BATTERY_JSON = "/tmp/robocomp_netmon/battery.json"


def _config_path(comp):
    """The component's config file: the one `cmd` token that points into its etc/ dir
    (works regardless of wrapper prefixes like `vglrun` shifting the binary's position)."""
    cwd = os.path.expanduser(comp.get("cwd") or "")
    for tok in (comp.get("cmd") or "").split():
        if "etc/" in tok:
            return os.path.normpath(os.path.join(cwd, tok))
    return None


def _dds_bw():
    """Merge every dds_stats_bridge output (one file per DDS domain) into {topic: bytes/s}."""
    out = {}
    for path in glob.glob(_DDS_STATS_GLOB):
        try:
            with open(path) as f:
                out.update(json.load(f))
        except (OSError, ValueError):
            pass
    return out


def _battery():
    """Latest Victron battery snapshot written by the launcher that owns the serial, or None."""
    try:
        with open(_BATTERY_JSON) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _media_stats():
    """Merge every media_transport StreamStats dump (one file per component/stream, written by
    MediaSubscriber::combined_stats()/write_media_stats_json -- see media_transport.h) into
    {label: {frames,drops,sample_lost,fps,latency_ms}}. Labels already encode component + role +
    stream (e.g. "retina:zed:rgb" vs "robot_concept:ingest:zed_camera:rgb") so an ingest-side
    and a final-consumer view of the same physical stream never collide here."""
    out = {}
    for path in glob.glob(_MEDIA_STATS_GLOB):
        try:
            with open(path) as f:
                out.update(json.load(f))
        except (OSError, ValueError):
            pass
    return out


class MonitorServer:
    def __init__(self, bw_monitor, port=8080, host="127.0.0.1"):
        self.bw = bw_monitor
        self.port = port
        self.host = host
        self._httpd = None
        self._topo = None
        self._topo_ts = 0.0

    def topology(self):
        now = time.time()
        if self._topo is None or now - self._topo_ts > _TOPO_TTL:
            self._topo = build_full_topology(merged_components())
            self._topo_ts = now
            if self.bw:
                self.bw.set_ports(self._topo.get("server_ports", []))
        return self._topo

    def _pid_to_name(self, comps):
        mapping = {}
        for c in comps:
            pid = c.get("pid")
            if not pid:
                continue
            mapping[pid] = c["name"]
            try:
                for ch in psutil.Process(pid).children(recursive=True):
                    mapping[ch.pid] = c["name"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return mapping

    def state(self):
        comps = merged_components()
        topo = self.topology()
        server_ports = set(topo.get("server_ports", []))
        dds_bw = _dds_bw()

        # "up" means the process is alive but the ICE ping failed -- for DDS-only publishers
        # that's a false alarm if they're visibly moving real bytes/s, so real DDS traffic
        # is treated as a stronger liveness signal than the ICE ping and promotes to "alive".
        dds_topics_by_name = {n["id"]: (n.get("dds") or {}).get("topics") or []
                              for n in topo.get("nodes", [])}

        nodes = []
        for c in comps:
            status = c.get("status", "unknown")
            if status == "up" and any(dds_bw.get(t, 0.0) > 1 for t in dds_topics_by_name.get(c["name"], [])):
                status = "alive"
            nodes.append({
                "name": c["name"],
                "status": status,
                "cpu": round(c.get("cpu", 0.0), 1),
                "mem": round(c.get("mem", 0.0), 1),
                "layer": c.get("layer"),
            })

        edges_bw = {}
        bw = self.bw.sample() if (self.bw and self.bw.available) else {}
        for key, meta in connection_edges(server_ports, self._pid_to_name(comps)).items():
            src, dst = meta["src"], meta["dst"]
            if not src or not dst or src == dst:
                continue
            eid = f"{src}->{dst}:{meta['server_port']}"
            edges_bw[eid] = edges_bw.get(eid, 0.0) + bw.get(key, 0.0)

        return {
            "nodes": nodes,
            "edges_bw": edges_bw,
            "dds_bw": dds_bw,
            "media": _media_stats(),
            "battery": _battery(),
            "bw_available": bool(self.bw and self.bw.available),
            "bw_error": self.bw.error if self.bw else "captura deshabilitada",
        }

    def _known_names(self):
        return {c["name"] for c in merged_components()}

    def _component(self, name):
        for c in merged_components():
            if c.get("name") == name:
                return c
        return None

    def config_get(self, name):
        c = self._component(name)
        if not c:
            return {"ok": False, "error": "componente desconocido", "text": "", "path": ""}
        path = _config_path(c)
        if not path or not os.path.isfile(path):
            return {"ok": False, "error": "sin archivo de configuración detectable", "text": "", "path": path or ""}
        try:
            with open(path, "r", errors="replace") as f:
                text = f.read()
        except OSError as e:
            return {"ok": False, "error": str(e), "text": "", "path": path}
        return {"ok": True, "name": name, "path": path, "text": text}

    def config_set(self, name, text):
        c = self._component(name)
        if not c:
            return {"ok": False, "error": "componente desconocido"}
        path = _config_path(c)
        if not path or not os.path.isfile(path):
            return {"ok": False, "error": "sin archivo de configuración detectable"}
        try:
            shutil.copy2(path, path + ".bak")  # single-generation safety net before overwrite
            with open(path, "w") as f:
                f.write(text)
        except OSError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "path": path}

    def action(self, action, name):
        if action not in ("stop", "start", "restart", "build") or name not in self._known_names():
            return {"ok": False, "error": "acción o componente inválido"}
        push_command(action, name)
        return {"ok": True, "action": action, "name": name}

    def logs(self, name, stream, lines):
        if name not in self._known_names() or stream not in ("out", "err"):
            return {"ok": False, "error": "componente/stream inválido", "text": ""}
        path = os.path.join(_LOGS_DIR, f"{os.path.basename(name)}.{stream}")
        if not os.path.exists(path):
            return {"ok": True, "name": name, "stream": stream, "text": "(sin log)"}
        try:
            with open(path, "r", errors="replace") as f:
                text = "".join(f.readlines()[-lines:])
        except OSError as e:
            return {"ok": False, "error": str(e), "text": ""}
        return {"ok": True, "name": name, "stream": stream, "text": text}

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body, ctype):
                data = body.encode() if isinstance(body, str) else body
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _file(self, name, ctype):
                path = os.path.join(_STATIC, name)
                if not os.path.exists(path):
                    self._send(404, "not found", "text/plain")
                    return
                with open(path, "rb") as f:
                    self._send(200, f.read(), ctype)

            def do_GET(self):
                route = self.path.split("?", 1)[0]
                if route in ("/", "/index.html"):
                    self._file("index.html", "text/html; charset=utf-8")
                elif route == "/app.js":
                    self._file("app.js", "application/javascript")
                elif route == "/api/topology":
                    self._send(200, json.dumps(server.topology()), "application/json")
                elif route == "/api/state":
                    self._send(200, json.dumps(server.state()), "application/json")
                elif route == "/api/logs":
                    q = parse_qs(urlparse(self.path).query)
                    name = (q.get("name") or [""])[0]
                    stream = (q.get("stream") or ["err"])[0]
                    lines = int((q.get("lines") or ["200"])[0])
                    self._send(200, json.dumps(server.logs(name, stream, lines)), "application/json")
                elif route == "/api/config":
                    q = parse_qs(urlparse(self.path).query)
                    name = (q.get("name") or [""])[0]
                    self._send(200, json.dumps(server.config_get(name)), "application/json")
                else:
                    self._send(404, "not found", "text/plain")

            def _json_body(self):
                n = int(self.headers.get("Content-Length", 0))
                return json.loads(self.rfile.read(n) or b"{}")

            def do_POST(self):
                route = self.path.split("?", 1)[0]
                try:
                    body = self._json_body()
                except (ValueError, TypeError):
                    self._send(400, json.dumps({"ok": False, "error": "json inválido"}), "application/json")
                    return
                if route == "/api/action":
                    res = server.action(body.get("action"), body.get("name"))
                elif route == "/api/config":
                    res = server.config_set(body.get("name"), body.get("text", ""))
                else:
                    self._send(404, "not found", "text/plain")
                    return
                self._send(200 if res.get("ok") else 400, json.dumps(res), "application/json")

        return Handler

    def start(self):
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._handler())
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return f"http://{self.host}:{self.port}"

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()
