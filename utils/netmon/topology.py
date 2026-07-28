"""Interconnection graph derived from each component's etc/config.

For every component we read from its config:
  - implements (RPC):  Endpoints.<Iface> = "tcp -p N"
  - requires   (RPC):  Proxies.<Name>  = "identity:tcp -h host -p N"
  - publishes  (topic): Proxies.<Topic>Prefix / <Topic>PubPrefix
  - subscribes (topic): Endpoints.<Topic>Topic / <Topic>Prefix

RPC edges are resolved by matching a required port to the component that
implements it (identity name is only a fallback). Pub/sub edges are routed
through an IceStorm broker node. Handles both ICE-property and TOML configs.
"""

import os
import re

_ENDPOINT_HOST = re.compile(r"-h\s+(\S+)")
_ENDPOINT_PORT = re.compile(r"-p\s+(\d+)")
_SECTION = re.compile(r"^\s*\[(\w+)\]")
_KV = re.compile(r"^\s*([\w.]+)\s*=\s*(.*?)\s*$")


def _expand(p):
    return os.path.expanduser(p) if p else p


def parse_endpoint(ice_string):
    """"identity:tcp -h host -p 10097" -> dict, or None."""
    if not ice_string:
        return None
    identity, _, rest = ice_string.partition(":")
    identity = identity.strip()
    if not identity:
        return None
    transport = rest.strip().split()[0] if rest.strip() else "tcp"
    if transport not in ("tcp", "ssl", "udp", "default"):
        transport = "tcp"
    host = _ENDPOINT_HOST.search(rest)
    port = _ENDPOINT_PORT.search(rest)
    return {
        "identity": identity,
        "transport": transport,
        "host": host.group(1) if host else None,
        "port": int(port.group(1)) if port else None,
    }


def _port(value):
    m = _ENDPOINT_PORT.search(value or "")
    return int(m.group(1)) if m else None


def _config_path(comp):
    cwd = _expand(comp.get("cwd"))
    cand = None
    for tok in comp.get("cmd", "").split():
        if tok.startswith("--Ice.Config="):
            tok = tok.split("=", 1)[1]
        if "etc/" in tok or tok.endswith("config") or ".toml" in tok or ".conf" in tok:
            cand = tok
    if not cand:
        return None
    if not os.path.isabs(cand) and cwd:
        cand = os.path.join(cwd, cand)
    return cand


def _parse_config(path):
    """Return (implements, requires, publishes, subscribes, dds) for one config.

    dds is {"domain": int, "topics": [str, ...]} from a [DDS] block (Domain plus any
    key ending in "Topic" — a component can publish more than one, e.g. RGBTopic/DepthTopic),
    or None.
    """
    impl, req, pub, sub, dds = [], [], set(), set(), None
    if not path or not os.path.exists(path):
        return impl, req, [], [], None
    section = None
    try:
        with open(path, "r", errors="ignore") as f:
            for raw in f:
                line = raw.split("#", 1)[0]
                sec = _SECTION.match(line)
                if sec:
                    section = sec.group(1)
                    continue
                kv = _KV.match(line)
                if not kv:
                    continue
                key, val = kv.group(1), kv.group(2).strip().strip('"')
                if section == "DDS":
                    if key == "Domain":
                        dds = dds or {}
                        dds["domain"] = int(val) if val.lstrip("-").isdigit() else val
                    elif key.endswith("Topic"):
                        dds = dds or {}
                        dds.setdefault("topics", []).append(val)
                    continue
                if key.startswith("Proxies."):
                    group, name = "Proxies", key[len("Proxies."):]
                elif key.startswith("Endpoints."):
                    group, name = "Endpoints", key[len("Endpoints."):]
                elif section in ("Proxies", "Endpoints"):
                    group, name = section, key
                else:
                    continue

                if group == "Endpoints":
                    if name.endswith("Topic"):
                        sub.add(name[:-len("Topic")])
                    elif name.endswith("Prefix"):
                        sub.add(name[:-len("Prefix")])
                    else:
                        p = _port(val)
                        if p:
                            impl.append({"iface": name, "port": p})
                else:  # Proxies
                    if name == "TopicManager":
                        continue
                    if name.endswith("Prefix"):
                        pub.add(name[:-len("Prefix")])
                    else:
                        ep = parse_endpoint(val)
                        if ep:
                            req.append({"name": name, "identity": ep["identity"],
                                        "host": ep["host"], "port": ep["port"]})
    except OSError:
        pass
    return impl, req, sorted(pub), sorted(sub), dds


def _dedup(edges):
    seen, out = set(), []
    for e in edges:
        sig = (e["src"], e["dst"], e.get("port"), e.get("iface"), e.get("topic"), e["kind"])
        if sig not in seen:
            seen.add(sig)
            out.append(e)
    return out


def component_dds(comp):
    """[DDS] domain/topics for one component's config, or None. Used by the launcher to
    know which components to start with FASTDDS_STATISTICS and which DDS domains need a
    dds_stats_bridge instance for real publish-bandwidth measurement."""
    return _parse_config(_config_path(comp))[4]


def build_topology(components):
    data = {}
    for c in components:
        impl, req, pub, sub, dds = _parse_config(_config_path(c))
        data[c["name"]] = {"impl": impl, "req": req, "pub": pub, "sub": sub, "dds": dds}

    port_owner = {}          # port -> component that implements it
    ident_owner = {}         # iface(lower) -> [(component, port)]
    for name, d in data.items():
        for e in d["impl"]:
            port_owner[e["port"]] = name
            ident_owner.setdefault(e["iface"].lower(), []).append((name, e["port"]))

    nodes = {}
    for name, d in data.items():
        nodes[name] = {
            "id": name, "role": "component",
            "implements": d["impl"],
            "requires": [{"identity": r["identity"], "port": r["port"]} for r in d["req"]],
            "publishes": d["pub"], "subscribes": d["sub"], "dds": d["dds"],
        }

    edges, externals = [], {}
    # RPC edges: match required port -> implementing component
    for name, d in data.items():
        for r in d["req"]:
            target, tport = None, r["port"]
            if tport and tport in port_owner:
                target = port_owner[tport]
            else:
                cand = ident_owner.get((r["identity"] or "").lower(), [])
                if len(cand) == 1:
                    target, tport = cand[0]
            if target and target != name:
                edges.append({"src": name, "dst": target, "port": tport,
                              "iface": r["identity"], "kind": "rpc"})
            elif not target:
                ext = f"{r['identity']}:{r['port']}"
                externals[ext] = {"id": ext, "role": "external",
                                  "implements": [{"iface": r["identity"], "port": r["port"]}],
                                  "requires": [], "publishes": [], "subscribes": []}
                edges.append({"src": name, "dst": ext, "port": r["port"],
                              "iface": r["identity"], "kind": "rpc"})

    # Pub/sub edges routed through the IceStorm broker
    any_ps = False
    subs_by_topic = {}
    for name, d in data.items():
        for t in d["sub"]:
            subs_by_topic.setdefault(t, []).append(name)
    for name, d in data.items():
        for t in d["pub"]:
            any_ps = True
            edges.append({"src": name, "dst": "IceStorm", "topic": t, "kind": "pub"})
    for t, subs in subs_by_topic.items():
        for s in subs:
            any_ps = True
            edges.append({"src": "IceStorm", "dst": s, "topic": t, "kind": "sub"})

    # DDS edges: one broker node per domain, components publish straight into it
    # (no discovery/matching modeled here — RTPS pub/sub is host-wide per domain).
    dds_domains = set()
    for name, d in data.items():
        if not d["dds"] or d["dds"].get("domain") is None:
            continue
        dom = d["dds"]["domain"]
        broker = f"DDS·d{dom}"
        dds_domains.add((broker, dom))
        for topic in d["dds"].get("topics") or [None]:
            edges.append({"src": name, "dst": broker, "topic": topic, "kind": "dds"})

    nodes.update(externals)
    if any_ps:
        nodes["IceStorm"] = {"id": "IceStorm", "role": "broker",
                             "implements": [{"iface": "TopicManager", "port": 9999}],
                             "requires": [], "publishes": [], "subscribes": []}
    for broker, dom in dds_domains:
        nodes[broker] = {"id": broker, "role": "ddsbroker", "domain": dom,
                         "implements": [], "requires": [], "publishes": [], "subscribes": []}

    return {
        "nodes": list(nodes.values()),
        "edges": _dedup(edges),
        "server_ports": sorted(port_owner.keys()),
    }


def agent_domain(comp):
    """Read [Agent] domain / domain_id from a cognitive agent's TOML config."""
    path = _config_path(comp)
    if not path or not os.path.exists(path):
        return {"domain": None, "domain_id": None}
    try:
        import toml
        ag = toml.load(path).get("Agent", {})
        return {"domain": ag.get("domain"), "domain_id": ag.get("domain_id")}
    except Exception:
        return {"domain": None, "domain_id": None}


def build_full_topology(components):
    """ICE graph for everyone + symbolic DSR spheres grouping cognitive agents by domain."""
    t = build_topology(components)
    nodes, edges = t["nodes"], t["edges"]
    by_id = {n["id"]: n for n in nodes}

    domains = {}
    for c in components:
        if c.get("layer") != "cognitive":
            continue
        name = c["name"]
        dom = c.get("domain")
        if dom is None:
            dom = agent_domain(c).get("domain")
        dom = 0 if dom is None else dom
        n = by_id.get(name)
        if n:
            n["role"], n["domain"] = "agent", dom
        else:
            n = {"id": name, "role": "agent", "domain": dom, "implements": [],
                 "requires": [], "publishes": [], "subscribes": []}
            nodes.append(n)
            by_id[name] = n
        domains.setdefault(dom, []).append(name)

    for dom, members in sorted(domains.items()):
        sphere = f"DSR·d{dom}"
        nodes.append({"id": sphere, "role": "dsr", "domain": dom, "implements": [],
                      "requires": [], "publishes": [], "subscribes": []})
        for m in members:
            edges.append({"src": m, "dst": sphere, "kind": "dsr"})

    return {"nodes": nodes, "edges": edges, "server_ports": t["server_ports"]}
