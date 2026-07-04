"""Local web dashboard. Reads the shared /tmp registry (all launchers) and serves
the unified interconnection map + live state (stdlib http.server, no framework).

The frontend polls /api/state every second. Topology is rebuilt from the merged
registry with a short TTL (parsing configs is the expensive part).
"""

import json
import os
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

        nodes = [{
            "name": c["name"],
            "status": c.get("status", "unknown"),
            "cpu": round(c.get("cpu", 0.0), 1),
            "mem": round(c.get("mem", 0.0), 1),
            "layer": c.get("layer"),
        } for c in comps]

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
            "bw_available": bool(self.bw and self.bw.available),
            "bw_error": self.bw.error if self.bw else "captura deshabilitada",
        }

    def _known_names(self):
        return {c["name"] for c in merged_components()}

    def action(self, action, name):
        if action not in ("stop", "start", "restart") or name not in self._known_names():
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
                else:
                    self._send(404, "not found", "text/plain")

            def do_POST(self):
                route = self.path.split("?", 1)[0]
                if route != "/api/action":
                    self._send(404, "not found", "text/plain")
                    return
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(n) or b"{}")
                except (ValueError, TypeError):
                    self._send(400, json.dumps({"ok": False, "error": "json inválido"}), "application/json")
                    return
                res = server.action(body.get("action"), body.get("name"))
                self._send(200 if res.get("ok") else 400, json.dumps(res), "application/json")

        return Handler

    def start(self):
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._handler())
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return f"http://{self.host}:{self.port}"

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()
