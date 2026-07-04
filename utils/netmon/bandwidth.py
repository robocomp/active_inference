"""Per-connection bandwidth on loopback via a raw AF_PACKET socket (stdlib only).

All components talk over localhost, so every proxy connection crosses `lo`.
We sniff `lo`, count bytes per (portA, portB) pair filtered to the known server
ports, and psutil.net_connections maps each pair back to (client, server) components.

On loopback every packet is delivered to AF_PACKET twice (once PACKET_OUTGOING,
once PACKET_HOST), so we drop the OUTGOING copy to avoid counting each byte 2x.

Note: counts include TCP/IP headers (raw frame length), so figures are wire bytes,
a small over-estimate of ICE payload. Requires cap_net_raw (setcap) or root.
"""

import socket
import struct
import threading
import time

ETH_P_ALL = 3
_ETHERTYPE_IPV4 = 0x0800
_PROTO_TCP = 6
_PACKET_OUTGOING = 4  # sll_pkttype: on lo every packet is seen twice (HOST + OUTGOING); drop one
_RCVBUF = 16 * 1024 * 1024  # big capture buffer so bursts don't overflow the ring and drop packets


class BandwidthMonitor:
    def __init__(self, server_ports, iface="lo"):
        self.server_ports = set(int(p) for p in server_ports)
        self.iface = iface
        self.available = False
        self.error = None
        self._sock = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._counters = {}          # (loport, hiport) -> cumulative bytes
        self._prev = {}
        self._prev_t = time.time()

    def set_ports(self, ports):
        """Update the set of server ports to attribute traffic to (topology may change)."""
        self.server_ports = set(int(p) for p in ports)

    def start(self):
        try:
            s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_ALL))
            s.bind((self.iface, 0))
            s.settimeout(0.5)
            try:  # FORCE bypasses rmem_max (needs cap_net_admin); fall back to the capped setsockopt
                s.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_RCVBUFFORCE", 33), _RCVBUF)
            except OSError:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RCVBUF)
        except PermissionError:
            self.error = "sin permisos de captura (setcap cap_net_raw+ep en python3, o sudo)"
            return
        except Exception as e:  # interfaz inexistente, etc.
            self.error = str(e)
            return
        self._sock = s
        self.available = True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass

    def _loop(self):
        while self._running:
            try:
                frame, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            if addr[2] == _PACKET_OUTGOING:  # avoid loopback double-counting
                continue
            n = len(frame)
            if n < 38:  # 14 eth + 20 ip + 4 tcp ports
                continue
            if struct.unpack("!H", frame[12:14])[0] != _ETHERTYPE_IPV4:
                continue
            ihl = (frame[14] & 0x0F) * 4
            if frame[14 + 9] != _PROTO_TCP:
                continue
            off = 14 + ihl
            if n < off + 4:
                continue
            sport, dport = struct.unpack("!HH", frame[off:off + 4])
            if sport not in self.server_ports and dport not in self.server_ports:
                continue
            key = (min(sport, dport), max(sport, dport))
            with self._lock:
                self._counters[key] = self._counters.get(key, 0) + n

    def sample(self):
        """bytes/s per (loport, hiport) since the previous call."""
        now = time.time()
        dt = max(1e-3, now - self._prev_t)
        with self._lock:
            snapshot = dict(self._counters)
        out = {}
        for key, total in snapshot.items():
            out[key] = max(0.0, (total - self._prev.get(key, total)) / dt)
        self._prev = snapshot
        self._prev_t = now
        return out


def connection_edges(server_ports, pid_to_name):
    """Map live loopback connections to component edges.

    Returns {(loport, hiport): {"server_port", "src", "dst"}}.
    On loopback each connection shows up from both ends in net_connections,
    so we can fill client (src) and server (dst) names from either sighting.
    """
    import psutil

    server_ports = set(int(p) for p in server_ports)
    res = {}
    for c in psutil.net_connections(kind="tcp"):
        if c.status != psutil.CONN_ESTABLISHED or not c.laddr or not c.raddr:
            continue
        lport, rport = c.laddr.port, c.raddr.port
        if lport in server_ports:
            sp, cp, is_server = lport, rport, True
        elif rport in server_ports:
            sp, cp, is_server = rport, lport, False
        else:
            continue
        key = (min(cp, sp), max(cp, sp))
        entry = res.setdefault(key, {"server_port": sp, "src": None, "dst": None})
        name = pid_to_name.get(c.pid)
        if is_server:
            entry["dst"] = entry["dst"] or name
        else:
            entry["src"] = entry["src"] or name
    return res
