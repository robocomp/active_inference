import RFB from "/novnc/core/rfb.js";

// vis-network falls back to its OWN default highlight/hover colors (a light-blue
// background, #D2E5FF) for any node color that doesn't explicitly set them -- combined
// with this dashboard's fixed light node font (#e6e6e6, see buildNodes' `base.font`),
// a selected node became light-on-light and unreadable, and it also silently masked the
// node's actual status color (alive/up/stopped) with that generic blue. Every entry here
// defines its own highlight/hover explicitly: same dark background (status stays legible
// at a glance even while selected), just a brighter border for the "this one's selected"
// emphasis.
function _withEmphasis(background, border, brightBorder) {
  const emphasis = { background, border: brightBorder };
  return { background, border, highlight: emphasis, hover: emphasis };
}
const STATUS_COLOR = {
  alive:   _withEmphasis("#16351c", "#3fb950", "#5fe377"),
  running: _withEmphasis("#16351c", "#3fb950", "#5fe377"),
  up:      _withEmphasis("#3a3410", "#d6b528", "#f0d24e"),
  stopped: _withEmphasis("#3d1618", "#f85149", "#ff8078"),
  unknown: _withEmphasis("#26292f", "#6b7280", "#98a3b3"),
};
const ROLE_COLOR = {
  broker:    _withEmphasis("#2b1d3d", "#b072e0", "#d5a6f5"),
  ddsbroker: _withEmphasis("#3d2a14", "#e0952b", "#f5b95a"),
  external:  _withEmphasis("#22252b", "#55606e", "#7f8b9c"),
  dsr:       _withEmphasis("#10303a", "#2ec5d3", "#6fe3ee"),
};
const LABEL_STROKE = { strokeWidth: 3, strokeColor: "#14161a" };

const HIER_LAYOUT = {
  hierarchical: {
    enabled: true, direction: "DU", sortMethod: "directed",     // bottom -> up
    levelSeparation: 160, nodeSpacing: 240, treeSpacing: 260,
    blockShifting: true, edgeMinimization: true, parentCentralization: true,
  },
};
const HIER_PHYSICS = {
  enabled: true, solver: "hierarchicalRepulsion",
  hierarchicalRepulsion: { nodeDistance: 240, avoidOverlap: 1, springLength: 150,
                           springConstant: 0.01, damping: 0.5 },
  stabilization: { iterations: 400 }, minVelocity: 0.6,
};
const ORG_LAYOUT = { hierarchical: { enabled: false }, improvedLayout: true };
const ORG_PHYSICS = {
  enabled: true, solver: "forceAtlas2Based",
  forceAtlas2Based: { gravitationalConstant: -140, centralGravity: 0.008, springLength: 200,
                      springConstant: 0.05, damping: 0.6, avoidOverlap: 1 },
  stabilization: { iterations: 500 }, minVelocity: 0.6,
};
// "level": organic (force-directed) but each node pinned to its level band (fixed Y, free X)
const LEVEL_GAP = 130;
const LEVEL_PHYSICS = {
  enabled: true, solver: "barnesHut",
  barnesHut: { gravitationalConstant: -12000, centralGravity: 0.12, springLength: 110,
               springConstant: 0.05, damping: 0.6, avoidOverlap: 1 },
  stabilization: { iterations: 300 }, minVelocity: 0.6,
};
const OPTIONS = {
  layout: ORG_LAYOUT,
  physics: LEVEL_PHYSICS,
  nodes: { margin: 10, widthConstraint: { maximum: 190 } },
  interaction: { hover: true, tooltipDelay: 100, dragNodes: true },
};

const MODES = ["level", "hier", "free"];
const MODE_LABEL = { level: "Orgánico por nivel", hier: "Jerárquico", free: "Orgánico libre" };
let layoutMode = "level";   // default view

// which edge groups are shown; toggled from the toolbar, persists across topology rebuilds
const edgeVisibility = { rpc: true, icestorm: true, dds: true };

function setLevelPositions(on) {
  if (!nodes) return;
  nodes.update(nodes.getIds().map(id => {
    const lv = nodeLevels[id];
    return (on && lv !== undefined)
      ? { id, y: (maxLevel - lv) * LEVEL_GAP, fixed: { x: false, y: true } }
      : { id, fixed: false };
  }));
}

function applyLayout(mode) {
  layoutMode = mode;
  if (mode === "hier") {
    setLevelPositions(false);
    network.setOptions({ layout: HIER_LAYOUT, physics: HIER_PHYSICS });
  } else if (mode === "level") {
    network.setOptions({ layout: ORG_LAYOUT, physics: LEVEL_PHYSICS });
    setLevelPositions(true);
  } else {
    setLevelPositions(false);
    network.setOptions({ layout: ORG_LAYOUT, physics: ORG_PHYSICS });
  }
  network.stabilize();   // then the stabilization listener freezes physics again
  const btn = document.getElementById("layoutBtn");
  if (btn) btn.textContent = "Vista: " + MODE_LABEL[mode];
}

function cycleLayout() {
  applyLayout(MODES[(MODES.indexOf(layoutMode) + 1) % MODES.length]);
}

function fmtBps(bps) {
  if (!bps || bps < 1) return "";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0, v = bps;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${u[i]}/s`;
}

// hierarchy levels (bottom -> top). Within a layer, stack nodes by RPC-dependency depth so
// connected nodes never share a level (no horizontal edges); leave gaps between layers.
function computeLevels(topo) {
  const roleOf = {}, reqOf = {};
  topo.nodes.forEach(n => { roleOf[n.id] = n.role; reqOf[n.id] = []; });
  topo.edges.forEach(e => { if (e.kind === "rpc" && reqOf[e.src]) reqOf[e.src].push(e.dst); });

  const memo = {};
  function depth(id, seen) {                       // longest same-layer dependency chain
    if (memo[id] !== undefined) return memo[id];
    if (seen.has(id)) return 0;
    seen.add(id);
    let d = 0;
    for (const t of reqOf[id] || [])
      if (roleOf[t] === roleOf[id]) d = Math.max(d, depth(t, seen) + 1);
    seen.delete(id);
    return (memo[id] = d);
  }

  let compMax = 0;
  topo.nodes.forEach(n => {
    if (n.role === "component" || n.role === "external") compMax = Math.max(compMax, depth(n.id, new Set()));
  });
  const base = { external: 0, component: 0, broker: compMax + 2, ddsbroker: compMax + 2,
                 agent: compMax + 3, dsr: compMax + 5 };

  const levels = {};
  topo.nodes.forEach(n => {
    const d = (n.role === "component" || n.role === "external") ? depth(n.id, new Set()) : 0;
    levels[n.id] = (base[n.role] ?? 0) + d;
  });
  return levels;
}

function nodeTooltip(n) {
  const L = [];
  if (n.implements && n.implements.length)
    L.push("implementa: " + n.implements.map(e => `${e.iface}:${e.port}`).join(", "));
  if (n.requires && n.requires.length)
    L.push("requiere: " + n.requires.map(r => `${r.identity}:${r.port}`).join(", "));
  if (n.publishes && n.publishes.length) L.push("publica: " + n.publishes.join(", "));
  if (n.subscribes && n.subscribes.length) L.push("suscribe: " + n.subscribes.join(", "));
  if (n.dds && n.dds.domain !== undefined)
    L.push(`DDS: dominio ${n.dds.domain}, topics ${(n.dds.topics || []).join(", ") || "?"}`);
  return `${n.id}\n${L.join("\n") || "(sin interfaces ICE)"}`;
}

let network, nodes, edges;
let rpcKey = {};        // "src->dst:port" -> vis edge id
let ddsEdgeMeta = {};    // vis edge id -> [topic, ...] it aggregates (from dds_stats_bridge)
let nodeInfo = {};      // id -> { serves, title }
let nodeLevels = {};    // id -> hierarchy level
let maxLevel = 0;
let expectedProc = new Set();  // process node ids we expect in /api/state

function buildNodes(topo) {
  rpcKey = {}; ddsEdgeMeta = {}; nodeInfo = {};
  const levels = nodeLevels = computeLevels(topo);
  maxLevel = Math.max(0, ...Object.values(levels));
  return topo.nodes.map(n => {
    const impl = n.implements.map(e => `${e.iface}:${e.port}`);
    const serves = impl.length > 4
      ? impl.slice(0, 3).join("\n") + `\n…(+${impl.length - 3})`
      : impl.join("\n");
    nodeInfo[n.id] = { serves, title: nodeTooltip(n) };
    const base = {
      id: n.id, level: levels[n.id], title: nodeTooltip(n),
      font: { color: "#e6e6e6", face: "monospace", size: 13, multi: false },
    };
    if (n.role === "broker")
      return { ...base, shape: "diamond", size: 22, color: ROLE_COLOR.broker, label: n.id };
    if (n.role === "ddsbroker")
      return { ...base, shape: "hexagon", size: 24, color: ROLE_COLOR.ddsbroker, label: n.id };
    if (n.role === "dsr")
      return { ...base, shape: "dot", size: 32, color: ROLE_COLOR.dsr,
               label: `${n.id}\n(grafo DSR)`, font: { ...base.font, size: 15 } };
    if (n.role === "external")
      return { ...base, shape: "box", shapeProperties: { borderDashes: [4, 4] },
               color: ROLE_COLOR.external, label: n.id };
    if (n.role === "agent")
      return { ...base, shape: "ellipse", color: STATUS_COLOR.unknown, label: n.id };
    return { ...base, shape: "box", color: STATUS_COLOR.unknown,
             label: serves ? `${n.id}\n${serves}` : n.id };
  });
}

function buildEdges(topo) {
  const rpcEdges = [], dsrEdges = [], psGroups = {}, ddsGroups = {};
  topo.edges.forEach(e => {
    if (e.kind === "pub" || e.kind === "sub") {
      const k = `${e.src}|${e.dst}|${e.kind}`;
      (psGroups[k] || (psGroups[k] = { src: e.src, dst: e.dst, kind: e.kind, topics: [] })).topics.push(e.topic);
    } else if (e.kind === "dds") {
      const k = `${e.src}|${e.dst}`;
      (ddsGroups[k] || (ddsGroups[k] = { src: e.src, dst: e.dst, topics: [] })).topics.push(e.topic);
    } else if (e.kind === "dsr") dsrEdges.push(e);
    else rpcEdges.push(e);
  });

  const pairCount = {}, pairSeen = {};
  rpcEdges.forEach(e => { const p = `${e.src}->${e.dst}`; pairCount[p] = (pairCount[p] || 0) + 1; });

  const visEdges = [];
  let ei = 0;
  rpcEdges.forEach(e => {
    const id = "e" + (ei++);
    rpcKey[`${e.src}->${e.dst}:${e.port}`] = id;
    const p = `${e.src}->${e.dst}`;
    const idx = pairSeen[p] = (pairSeen[p] || 0), n = pairCount[p];
    pairSeen[p]++;
    const round = n > 1 ? 0.2 + 0.28 * (idx - (n - 1) / 2) : 0.25;
    visEdges.push({
      id, from: e.src, to: e.dst, arrows: "to", label: `:${e.port} ${e.iface}`,
      color: { color: "#4a5568", highlight: "#7fd1ff" },
      font: { color: "#9aa4b2", size: 11, align: "horizontal", ...LABEL_STROKE },
      width: 1, smooth: { type: "cubicBezier", forceDirection: "vertical", roundness: round },
      group: "rpc", hidden: !edgeVisibility.rpc,
    });
  });
  dsrEdges.forEach(e => visEdges.push({
    id: "e" + (ei++), from: e.src, to: e.dst, arrows: "",
    color: { color: "#2ec5d3", highlight: "#7fe3ec" }, width: 2, smooth: { type: "continuous" },
    group: "dsr",
  }));
  Object.values(psGroups).forEach(g => {
    const head = g.kind === "pub" ? "pub" : "sub";
    const label = g.topics.length > 1
      ? `${head} (${g.topics.length}):\n` + g.topics.join("\n")
      : `${head} ${g.topics[0]}`;
    visEdges.push({
      id: "e" + (ei++), from: g.src, to: g.dst, arrows: "to", label, dashes: [6, 4],
      color: { color: "#b072e0", highlight: "#d9b3ff" },
      font: { color: "#c39be6", size: 10, align: "middle", ...LABEL_STROKE },
      width: 1.5, smooth: { type: "cubicBezier" },
      group: "icestorm", hidden: !edgeVisibility.icestorm,
    });
  });
  Object.values(ddsGroups).forEach(g => {
    const label = g.topics.length > 1
      ? `dds (${g.topics.length}):\n` + g.topics.join("\n")
      : `dds ${g.topics[0] || ""}`;
    const id = "e" + (ei++);
    ddsEdgeMeta[id] = { topics: g.topics, label };
    visEdges.push({
      id, from: g.src, to: g.dst, arrows: "to", label, dashes: [2, 3],
      color: { color: "#e0952b", highlight: "#f5b95a" },
      font: { color: "#e0952b", size: 10, align: "middle", ...LABEL_STROKE },
      width: 1.5, smooth: { type: "cubicBezier" },
      group: "dds", hidden: !edgeVisibility.dds,
    });
  });
  return visEdges;
}

function setEdgeGroupVisible(group, visible) {
  edgeVisibility[group] = visible;
  if (!edges) return;
  const ids = edges.getIds().filter(id => edges.get(id).group === group);
  edges.update(ids.map(id => ({ id, hidden: !visible })));
}

async function loadTopology() {
  const topo = await (await fetch("/api/topology")).json();
  const nodeArr = buildNodes(topo), edgeArr = buildEdges(topo);
  expectedProc = new Set(topo.nodes.filter(n => n.role === "component" || n.role === "agent").map(n => n.id));

  if (!network) {
    nodes = new vis.DataSet(nodeArr);
    edges = new vis.DataSet(edgeArr);
    network = new vis.Network(document.getElementById("graph"), { nodes, edges }, OPTIONS);
    // freeze physics once settled so 1 Hz state updates don't jiggle the layout
    network.on("stabilizationIterationsDone", () => network.setOptions({ physics: { enabled: false } }));
  } else {
    nodes.clear(); nodes.add(nodeArr);
    edges.clear(); edges.add(edgeArr);
  }
  applyLayout(layoutMode);   // enforce current view (positions/physics) + re-stabilize once
}

function renderTable(stNodes) {
  const order = { sensorimotor: 0, cognitive: 1 };
  const rows = [...stNodes].sort((a, b) =>
    (order[a.layer] ?? 9) - (order[b.layer] ?? 9) || a.name.localeCompare(b.name));
  document.getElementById("tbody").innerHTML = rows.map(n => {
    const col = (STATUS_COLOR[n.status] || STATUS_COLOR.unknown).border;
    const lay = n.layer || "";
    const short = lay === "sensorimotor" ? "SM" : lay === "cognitive" ? "COG" : lay;
    const sel = n.name === selected ? " selrow" : "";
    return `<tr class="${sel}" data-name="${n.name}"><td>${n.name}</td>`
      + `<td><span class="lay ${lay}">${short}</span></td>`
      + `<td class="st" style="color:${col}">${n.status}</td>`
      + `<td class="num">${n.cpu}%</td>`
      + `<td class="num">${n.mem}</td></tr>`;
  }).join("");
}

// Per-stream fps/drops/sample_lost/latency from media_transport's StreamStats (see
// write_media_stats_json in media_transport.h). Labels already encode component + role +
// stream (e.g. "retina:zed:rgb" vs "robot_concept:ingest:zed_camera:rgb"), so an
// ingest-side row and a final-consumer row for the same physical stream sit side by side
// here instead of overwriting each other -- that's what pinpoints WHERE fps drop.
function renderMediaTable(media) {
  const rows = Object.entries(media || {}).sort(([a], [b]) => a.localeCompare(b));
  document.getElementById("mediaTbody").innerHTML = rows.map(([label, s]) => {
    const dropCol = s.drops > 0 ? "#f85149" : "#cfd6e0";
    const lostCol = s.sample_lost > 0 ? "#f85149" : "#cfd6e0";
    // min fps well under the window average flags a brief stall the average alone hides.
    const minFpsCol = (s.min_fps > 0 && s.min_fps < s.fps * 0.5) ? "#f85149" : "#cfd6e0";
    const maxLatCol = (s.max_latency_ms > s.latency_ms * 2) ? "#e0952b" : "#cfd6e0";
    return `<tr><td>${label}</td>`
      + `<td class="num">${s.fps.toFixed(1)}</td>`
      + `<td class="num" style="color:${minFpsCol}">${s.min_fps.toFixed(1)}</td>`
      + `<td class="num" style="color:${dropCol}">${s.drops}</td>`
      + `<td class="num" style="color:${lostCol}">${s.sample_lost}</td>`
      + `<td class="num">${s.latency_ms.toFixed(1)}</td>`
      + `<td class="num" style="color:${maxLatCol}">${s.max_latency_ms.toFixed(1)}</td></tr>`;
  }).join("");
}

// --- selection + process control ---
let selected = null;

// A colored glow around the selected node, in ITS OWN status/role color (green if
// alive, red if stopped, purple for a broker, ...) instead of vis-network's generic
// selection outline -- easier to spot which node is selected at a glance on a busy
// graph. size/x/y are vis-network's node.shadow fields (a soft drop-shadow, not a
// literal outline) -- {enabled:false} for "no aura".
function nodeAura(color) {
  return { enabled: true, color, size: 22, x: 0, y: 0 };
}

function selectComponent(name) {
  if (!expectedProc.has(name)) { clearSelection(); return; }   // only controllable nodes
  if (selected && selected !== name && nodes && nodes.get(selected))
    nodes.update({ id: selected, shadow: { enabled: false } });
  selected = name;
  document.getElementById("selname").textContent = name;
  document.getElementById("sel").classList.remove("hidden");
  document.querySelectorAll("#tbody tr").forEach(tr =>
    tr.classList.toggle("selrow", tr.dataset.name === name));
  if (network) network.selectNodes([name]);
  if (nodes) {
    const n = nodes.get(name);
    const color = (n && n.color && n.color.border) || "#7fd1ff";
    nodes.update({ id: name, shadow: nodeAura(color) });
  }
  openWindowView(name);                                        // jump to its display
}

function clearSelection() {
  if (selected && nodes && nodes.get(selected)) nodes.update({ id: selected, shadow: { enabled: false } });
  selected = null;
  document.getElementById("sel").classList.add("hidden");
  document.querySelectorAll("#tbody tr.selrow").forEach(tr => tr.classList.remove("selrow"));
  if (network) network.unselectAll();
}

// small non-blocking status pill, replaces alert() for routine action feedback
function toast(msg, isError) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = "show" + (isError ? " err" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.className = ""; }, 2500);
}

// after an action, the launcher's collector only claims commands once per second (see
// run_launcher's `collector()` loop), so a single immediate poll can still land just before
// the change is applied -- keep polling faster for a few seconds to catch the transition
// as soon as it happens instead of waiting for the next lazy 1Hz tick.
let burstTimer = null;
function burstPoll(durationMs = 4000, intervalMs = 350) {
  if (burstTimer) clearInterval(burstTimer);
  const deadline = Date.now() + durationMs;
  poll();
  burstTimer = setInterval(() => {
    poll();
    if (Date.now() > deadline) { clearInterval(burstTimer); burstTimer = null; }
  }, intervalMs);
}

async function sendAction(action, name) {
  try {
    const r = await (await fetch("/api/action", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, name }),
    })).json();
    if (!r.ok) { toast("Error: " + (r.error || "acción rechazada"), true); return; }
    toast(`${action} → ${name}`);
    burstPoll();
  } catch { toast("No se pudo enviar la acción", true); }
}

// disables a button for a short cooldown after click, so a slow action (build, restart)
// can't be fired twice in a row while the first request is still in flight
function withCooldown(btn, fn) {
  return async (...args) => {
    if (btn.disabled) return;
    btn.disabled = true;
    try { await fn(...args); } finally { setTimeout(() => { btn.disabled = false; }, 1200); }
  };
}

// --- config file editor: reads the etc/ file the component's own `cmd` points at ---
let cfgName = null, cfgPath = null, cfgOriginal = null;

async function openConfig(name) {
  closeTty();   // mutually exclusive, both dock at the bottom
  cfgName = name;
  cfgPath = cfgOriginal = null;
  document.getElementById("cfgtitle").textContent = "config · " + name;
  document.getElementById("cfgpath").textContent = "";
  document.getElementById("cfgtext").value = "cargando...";
  document.getElementById("cfg").classList.remove("hidden");
  try {
    const r = await (await fetch(`/api/config?name=${encodeURIComponent(name)}`)).json();
    if (!r.ok) {
      document.getElementById("cfgtext").value = "";
      document.getElementById("cfgpath").textContent = r.error || "sin archivo de configuración";
      return;
    }
    cfgPath = r.path; cfgOriginal = r.text;
    document.getElementById("cfgpath").textContent = r.path;
    document.getElementById("cfgtext").value = r.text;
  } catch {
    document.getElementById("cfgtext").value = "";
    document.getElementById("cfgpath").textContent = "error de red";
  }
}

async function saveConfig() {
  if (!cfgName || !cfgPath) return;
  const text = document.getElementById("cfgtext").value;
  if (text === cfgOriginal) return;
  if (!confirm(`¿Guardar cambios en ${cfgPath}?\nSe guarda una copia .bak del contenido anterior. `
              + `El componente en marcha no recarga solo — usa 🔄 relaunch después.`)) return;
  try {
    const r = await (await fetch("/api/config", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: cfgName, text }),
    })).json();
    if (!r.ok) { toast("Error al guardar: " + (r.error || "desconocido"), true); return; }
    cfgOriginal = text;
    toast(`config guardada · ${cfgName}`);
  } catch { toast("No se pudo guardar (red)", true); }
}

// --- real terminal (tmux-backed, see netmon/term_tmux.py) ---------------------
// Interactive: you can type into the actual running process (and send C-c/C-d), not just
// read its output. Polling only runs while the panel is open (started in openTty, stopped
// in closeTty) -- an idle/closed terminal costs nothing beyond the tmux session itself.
let ttyName = null, ttyPollTimer = null;

// Minimal SGR (color/bold/underline) -> HTML span converter. Handles the common codes
// tmux's `capture-pane -e` actually emits (reset/bold/dim/underline, 30-37/90-97 fg,
// 40-47/100-107 bg, 39/49 defaults) -- not a full ANSI/VT100 parser, just enough for
// typical console output (rich tables, cmake/make progress, colored log levels).
const _ANSI_FG = { 30:"#000",31:"#f85149",32:"#3fb950",33:"#d6b528",34:"#58a6ff",35:"#c79bea",36:"#39c5cf",37:"#cbd3de",
                   90:"#6b7280",91:"#ff8f88",92:"#7ee2a8",93:"#f0d379",94:"#8fc4ff",95:"#dcb8f5",96:"#7fe3ec",97:"#eef2f7" };
const _ANSI_BG = { 40:"#000",41:"#3d1618",42:"#16351c",43:"#3a3410",44:"#0d2a4a",45:"#2b1d3d",46:"#0d3a3f",47:"#cbd3de",
                   100:"#262b34",101:"#5a2320",102:"#1f4a29",103:"#4a3f14",104:"#123657",105:"#3a2650",106:"#134d52",107:"#e6e6e6" };
function ansiToHtml(text) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  let out = "", open = false, style = {};
  const styleAttr = () => {
    const decl = [];
    if (style.bold) decl.push("font-weight:bold");
    if (style.dim) decl.push("opacity:.6");
    if (style.underline) decl.push("text-decoration:underline");
    if (style.fg) decl.push(`color:${style.fg}`);
    if (style.bg) decl.push(`background:${style.bg}`);
    return decl.join(";");
  };
  const parts = text.split(/\x1b\[([0-9;]*)m/);
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 0) { out += esc(parts[i]); continue; }
    for (const code of (parts[i] || "0").split(";").map(Number)) {
      if (code === 0 || Number.isNaN(code)) style = {};
      else if (code === 1) style.bold = true;
      else if (code === 2) style.dim = true;
      else if (code === 4) style.underline = true;
      else if (code === 22) { style.bold = false; style.dim = false; }
      else if (code === 24) style.underline = false;
      else if (code === 39) delete style.fg;
      else if (code === 49) delete style.bg;
      else if (_ANSI_FG[code]) style.fg = _ANSI_FG[code];
      else if (_ANSI_BG[code]) style.bg = _ANSI_BG[code];
    }
    if (open) { out += "</span>"; open = false; }
    const attr = styleAttr();
    if (attr) { out += `<span style="${attr}">`; open = true; }
  }
  if (open) out += "</span>";
  return out;
}

function openTty(name) {
  document.getElementById("cfg").classList.add("hidden");   // mutually exclusive, both dock at the bottom
  ttyName = name;
  document.getElementById("ttytitle").textContent = "tty · " + name;
  document.getElementById("ttystatus").textContent = "";
  document.getElementById("ttyscreen").textContent = "cargando…";
  document.getElementById("tty").classList.remove("hidden");
  document.getElementById("ttyline").value = "";
  document.getElementById("ttyline").focus();
  refreshTty();
  if (ttyPollTimer) clearInterval(ttyPollTimer);
  ttyPollTimer = setInterval(refreshTty, 400);
}

function closeTty() {
  document.getElementById("tty").classList.add("hidden");
  if (ttyPollTimer) { clearInterval(ttyPollTimer); ttyPollTimer = null; }
  ttyName = null;
}

async function refreshTty() {
  if (!ttyName || document.getElementById("tty").classList.contains("hidden")) return;
  try {
    const r = await (await fetch(`/api/term/snapshot?name=${encodeURIComponent(ttyName)}&lines=2000`)).json();
    const status = document.getElementById("ttystatus");
    if (!r.ok) { status.textContent = "error: " + (r.error || "?"); return; }
    status.textContent = r.alive ? "" : "(sin sesión — arranca/relanza el componente)";
    const pre = document.getElementById("ttyscreen");
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30;
    pre.innerHTML = ansiToHtml(r.text || "");
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  } catch { /* ignore a dropped poll, next tick retries */ }
}

async function ttySendKey(key) {
  if (!ttyName) return;
  try {
    const r = await (await fetch("/api/term/key", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: ttyName, key }),
    })).json();
    if (!r.ok) toast("Error: " + (r.error || key), true);
    refreshTty();
  } catch { toast("No se pudo enviar a la terminal", true); }
}

async function ttySendChar(ch) {
  if (!ttyName) return;
  try {
    const r = await (await fetch("/api/term/input", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: ttyName, text: ch }),
    })).json();
    if (!r.ok) toast("Error: " + (r.error || ch), true);
    refreshTty();
  } catch { toast("No se pudo enviar a la terminal", true); }
}

// Real per-keystroke forwarding -- no local line buffer. #ttyline's own value never
// accumulates (every handled key calls preventDefault); the ONLY thing that ever shows
// what you typed is the next capture-pane poll echoing it back from the remote shell,
// same as a real terminal. This is what makes Ctrl-C/Ctrl-D, arrow-key bash history,
// Tab-completion and full-screen programs (less/htop/vim) actually work, at the cost of
// one request per keystroke and up to one poll interval of latency before you see it.
const _TTY_KEY_MAP = {
  Enter: "Enter", Backspace: "BSpace", Tab: "Tab", Escape: "Escape",
  ArrowUp: "Up", ArrowDown: "Down", ArrowLeft: "Left", ArrowRight: "Right",
  Home: "Home", End: "End", PageUp: "PPage", PageDown: "NPage",
  Delete: "DC", Insert: "IC",
};
function ttyKeydown(e) {
  if (e.ctrlKey && !e.altKey && !e.metaKey && /^[a-zA-Z]$/.test(e.key)) {
    e.preventDefault();
    ttySendKey("C-" + e.key.toLowerCase());
    return;
  }
  if (e.altKey || e.metaKey) return;   // let browser/OS shortcuts through (e.g. Alt-Tab)
  const mapped = _TTY_KEY_MAP[e.key];
  if (mapped) { e.preventDefault(); ttySendKey(mapped); return; }
  if (e.key.length === 1) { e.preventDefault(); ttySendChar(e.key); }
}

// --- per-component window view (x11vnc -id <window> + websockify, see window_vnc.py) --
// Real interaction, not just a picture: noVNC forwards mouse/keyboard to the actual X11
// window. Session is opened on demand (POST /api/window/open spins up x11vnc+websockify
// scoped to just that component's window) and torn down on close -- nothing runs in the
// background for a window nobody has open.
let winRfb = null, winName = null, winCurrentId = null;
let winKnownIds = new Set(), winWatchTimer = null;
const WIN_WATCH_MS = 1500;

function winStatus(text) { document.getElementById("winviewstatus").textContent = text; }

function winFillPicker(windows) {
  const picker = document.getElementById("winviewPicker");
  picker.innerHTML = "";
  picker.classList.toggle("hidden", windows.length <= 1);
  for (const w of windows) {
    const opt = document.createElement("option");
    opt.value = w.id; opt.textContent = w.title;
    if (w.id === winCurrentId) opt.selected = true;
    picker.appendChild(opt);
  }
}

let winSizes = new Map();   // window id -> {width, height}, from the last /api/window/list

// Size the floating panel to the window's OWN pixel dimensions (from X11 geometry) so
// x11vnc's capture shows at native resolution instead of being scaled to fit an
// arbitrary fixed box -- capped to the viewport so a huge window doesn't run off-screen
// (still fully resizable by hand afterward, this is just the opening size).
function resizeWinPanel(width, height) {
  if (!width || !height) return;
  const HEAD_H = 33;   // approx #winviewhead height
  const el = document.getElementById("winview");
  el.style.width = Math.min(width, window.innerWidth - 32) + "px";
  el.style.height = Math.min(height + HEAD_H, window.innerHeight - 32) + "px";
}

async function openWindowView(name) {
  // Check first, open nothing if there's nothing to show -- a headless component
  // (imu_dds, python_xbox_controller, ...) has no window at all, and popping an
  // empty black panel just to immediately say "no window found" in it is worse
  // than not opening it and saying so as a toast instead.
  let windows;
  try {
    const lr = await (await fetch(`/api/window/list?name=${encodeURIComponent(name)}`)).json();
    if (!lr.ok) { toast("⚠ " + (lr.error || "error"), true); return; }
    windows = lr.windows;
  } catch { toast("⚠ no se pudo conectar con netmon", true); return; }
  if (windows.length === 0) { toast(`${name}: sin ventana gráfica`); return; }

  document.getElementById("cfg").classList.add("hidden");
  closeTty();
  winName = name;
  document.getElementById("winviewtitle").textContent = "ventana · " + name;
  document.getElementById("winview").classList.remove("hidden");
  document.getElementById("winviewbody").innerHTML = "";

  // A component can genuinely own more than one top-level window (e.g. retina:
  // its main view + the Ricoh 360 panorama share one PID) -- offer a picker instead
  // of silently grabbing whichever one the WM happens to report first.
  winKnownIds = new Set(windows.map((w) => w.id));
  winSizes = new Map(windows.map((w) => [w.id, { width: w.width, height: w.height }]));
  winFillPicker(windows);
  await connectWindowView(name, windows[0].id);

  if (winWatchTimer) clearInterval(winWatchTimer);
  winWatchTimer = setInterval(() => watchWindowList(name), WIN_WATCH_MS);
}

// While the panel is open, keep polling this component's window list. If a window
// that wasn't there before shows up (e.g. the user's own click/interaction inside
// the VNC session popped a new dialog or viewer), follow it automatically instead
// of continuing to stare at the now-stale one. If the window currently being viewed
// disappears (closed), fall back to whatever else is still open. If it's just been
// resized by whatever is running inside it, follow that too (no reconnect needed --
// x11vnc keeps capturing the same window fine across a resize, only the panel/
// canvas need to catch up to the new geometry).
async function watchWindowList(name) {
  if (name !== winName) return;   // panel switched to a different component/closed
  let windows;
  try {
    const lr = await (await fetch(`/api/window/list?name=${encodeURIComponent(name)}`)).json();
    if (!lr.ok) return;
    windows = lr.windows;
  } catch { return; }

  const prevSize = winSizes.get(winCurrentId);
  const currentIds = new Set(windows.map((w) => w.id));
  const freshlyOpened = windows.find((w) => !winKnownIds.has(w.id));
  winKnownIds = currentIds;
  winSizes = new Map(windows.map((w) => [w.id, { width: w.width, height: w.height }]));
  winFillPicker(windows);

  if (freshlyOpened && freshlyOpened.id !== winCurrentId) {
    await connectWindowView(name, freshlyOpened.id);
  } else if (winCurrentId !== null && !currentIds.has(winCurrentId)) {
    if (windows.length > 0) await connectWindowView(name, windows[0].id);
    else { winStatus("⚠ la ventana se cerró"); if (winRfb) { winRfb.disconnect(); winRfb = null; } }
  } else if (winCurrentId !== null) {
    const size = winSizes.get(winCurrentId);
    if (size && prevSize && (size.width !== prevSize.width || size.height !== prevSize.height)) {
      resizeWinPanel(size.width, size.height);
    }
  }
}

async function connectWindowView(name, windowId) {
  winStatus("conectando…");
  if (winRfb) { winRfb.disconnect(); winRfb = null; }
  winCurrentId = windowId;
  const size = winSizes.get(windowId);
  if (size) resizeWinPanel(size.width, size.height);
  const body = document.getElementById("winviewbody");
  body.innerHTML = "";
  try {
    const r = await (await fetch("/api/window/open", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, window_id: windowId }),
    })).json();
    if (!r.ok) { winStatus("⚠ " + (r.error || "error")); return; }
    winRfb = new RFB(body, `ws://${location.hostname}:${r.ws_port}/`);
    winRfb.addEventListener("connect", () => winStatus(""));
    winRfb.addEventListener("disconnect", (e) =>
      winStatus(e.detail.clean ? "sesión cerrada" : "⚠ conexión perdida"));
    // Panel is already sized to the window's real geometry -- show it 1:1 instead of
    // scaled to fit, per what was asked ("resolución original de la ventana").
    winRfb.scaleViewport = false;
    winRfb.resizeSession = false;
  } catch {
    winStatus("⚠ no se pudo conectar con netmon");
  }
}

function closeWindowView() {
  if (winWatchTimer) { clearInterval(winWatchTimer); winWatchTimer = null; }
  if (winRfb) { winRfb.disconnect(); winRfb = null; }
  document.getElementById("winviewbody").innerHTML = "";
  document.getElementById("winview").classList.add("hidden");
  if (winName) {
    fetch("/api/window/close", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: winName }),
    }).catch(() => {});
  }
  winName = null;
  winCurrentId = null;
  winKnownIds = new Set();
}

function bindEdgeToggle(btnId, group) {
  const btn = document.getElementById(btnId);
  btn.classList.toggle("off", !edgeVisibility[group]);
  btn.addEventListener("click", () => {
    setEdgeGroupVisible(group, !edgeVisibility[group]);
    btn.classList.toggle("off", !edgeVisibility[group]);
  });
}

// --- live view: WHEP (WebRTC-HTTP Egress Protocol) client straight to mediamtx --------
// Used to be a plain <iframe> onto mediamtx's own player page. That was cross-origin
// (different port), so the browser sandboxes it completely -- no way for this page's JS
// to see whether the video inside is actually moving or frozen. This is used to pilot
// the robot, so "silently frozen, looks fine" is a real hazard worth losing the
// "paste any URL" genericness for. Embedding the stream ourselves (same origin, real
// <video> element) gives two independent, real signals instead of none:
//   1. RTCPeerConnection.connectionState -- "conexión perdida" on disconnected/failed.
//   2. video.currentTime not advancing for a few seconds despite connectionState still
//      saying "connected" -- "sin frames nuevos" (a stalled encoder/pipeline on the
//      producer side, e.g. the known ricoh_omni_dds GStreamer issues, doesn't always
//      drop the ICE connection -- the RTP session can stay up with nothing flowing).
let livePc = null, liveWhepUrl = null, liveWatchdog = null, liveLastTime = -1, liveStallTicks = 0;
let liveBaseUrl = null, liveReconnectTimer = null;
const LIVE_STALL_CHECK_MS = 500;      // watchdog tick
const LIVE_STALL_TICKS = 2;           // ~1s of no advancement before warning -- was 3s, too slow to pilot by
const LIVE_RECONNECT_MS = 2000;       // retry cadence once the connection drops

function liveOverlay(msg) {
  const el = document.getElementById("liveOverlay");
  if (!msg) { el.classList.add("hidden"); el.textContent = ""; return; }
  el.textContent = msg;
  el.classList.remove("hidden");
}

async function connectWhep(baseUrl) {
  disconnectWhep();
  liveBaseUrl = baseUrl;
  liveOverlay("conectando…");
  // No STUN: this is a same-LAN robot stream (mediamtx.yml even pins
  // webrtcAdditionalHosts to its own local IP) -- an unreachable public STUN server
  // on a network with no internet egress just stalls ICE gathering, which is exactly
  // what left this stuck on "conectando…" before. Host candidates alone are enough here.
  const pc = new RTCPeerConnection({ iceServers: [] });
  livePc = pc;
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("audio", { direction: "recvonly" });
  // Build the MediaStream ourselves from individual tracks instead of trusting
  // e.streams[0] -- some WHEP answers don't carry an a=msid grouping the track into a
  // stream, so e.streams comes back empty even though the track itself is fine, and
  // srcObject silently never gets set (connectionState still reaches "connected",
  // exactly the "stuck on sin frames nuevos" symptom).
  const inboundStream = new MediaStream();
  pc.ontrack = (e) => {
    console.log("[netmon-live] ontrack", e.track.kind, "streams:", e.streams.length);
    inboundStream.addTrack(e.track);
    const v = document.getElementById("liveVideo");
    if (v.srcObject !== inboundStream) v.srcObject = inboundStream;
    v.play().catch((err) => console.warn("[netmon-live] video.play() failed:", err.message));
  };
  pc.onconnectionstatechange = () => {
    console.log("[netmon-live] connectionState:", pc.connectionState);
    if (pc !== livePc) return;   // stale handler from a superseded connection
    if (pc.connectionState === "connected") {
      liveOverlay(null);
    } else if (pc.connectionState === "disconnected" || pc.connectionState === "failed") {
      liveOverlay("⚠ conexión perdida (" + pc.connectionState + ") — reintentando…");
      scheduleLiveReconnect();
    }
  };

  try {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    // Non-trickle: wait for full ICE gathering so the offer already carries every
    // candidate -- one POST, no separate PATCH-based trickle exchange to implement.
    // Capped at 2s: host candidates on a LAN gather almost instantly, so hitting this
    // cap means gathering is stuck for some other reason -- proceed with whatever
    // candidates exist rather than hang "conectando…" forever again.
    await new Promise((resolve) => {
      if (pc.iceGatheringState === "complete") return resolve();
      const timer = setTimeout(resolve, 2000);
      pc.addEventListener("icegatheringstatechange", function onchg() {
        if (pc.iceGatheringState === "complete") {
          clearTimeout(timer);
          pc.removeEventListener("icegatheringstatechange", onchg);
          resolve();
        }
      });
    });
    const whepUrl = baseUrl.replace(/\/+$/, "") + "/whep";
    const res = await fetch(whepUrl, {
      method: "POST", headers: { "Content-Type": "application/sdp" }, body: pc.localDescription.sdp,
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const loc = res.headers.get("Location");
    liveWhepUrl = loc ? new URL(loc, whepUrl).href : null;   // session resource, for DELETE on close
    const answer = await res.text();
    console.log("[netmon-live] answer media lines:",
      answer.split("\r\n").filter((l) => l.startsWith("m=")).join(" | "));
    await pc.setRemoteDescription({ type: "answer", sdp: answer });
    console.log("[netmon-live] setRemoteDescription ok, signalingState:", pc.signalingState);
  } catch (e) {
    liveOverlay("⚠ no se pudo conectar: " + e.message);
    return;
  }

  liveLastTime = -1; liveStallTicks = 0;
  liveWatchdog = setInterval(() => {
    const v = document.getElementById("liveVideo");
    if (pc.connectionState !== "connected") return;   // connection-state handler already covers this
    if (v.currentTime === liveLastTime) {
      liveStallTicks++;
      if (liveStallTicks >= LIVE_STALL_TICKS)
        liveOverlay("⚠ sin frames nuevos (" + (liveStallTicks * LIVE_STALL_CHECK_MS / 1000).toFixed(1) + "s)");
    } else {
      liveStallTicks = 0;
      liveLastTime = v.currentTime;
      liveOverlay(null);
    }
  }, LIVE_STALL_CHECK_MS);
}

function scheduleLiveReconnect() {
  if (liveReconnectTimer) return;   // already scheduled -- don't stack retries
  liveReconnectTimer = setTimeout(() => {
    liveReconnectTimer = null;
    if (!document.getElementById("live").classList.contains("hidden") && liveBaseUrl)
      connectWhep(liveBaseUrl);
  }, LIVE_RECONNECT_MS);
}

function disconnectWhep() {
  if (liveReconnectTimer) { clearTimeout(liveReconnectTimer); liveReconnectTimer = null; }
  if (liveWatchdog) { clearInterval(liveWatchdog); liveWatchdog = null; }
  if (livePc) { livePc.close(); livePc = null; }
  document.getElementById("liveVideo").srcObject = null;
  if (liveWhepUrl) { fetch(liveWhepUrl, { method: "DELETE" }).catch(() => {}); liveWhepUrl = null; }
  liveOverlay(null);
}

function setLiveVisible(show) {
  document.getElementById("live").classList.toggle("hidden", !show);
  if (show) {
    const url = document.getElementById("liveUrl").value.trim();
    if (url) connectWhep(url);
  } else {
    liveBaseUrl = null;   // explicit close: no reconnect should fire after this
    disconnectWhep();     // stop decoding/bandwidth while closed
  }
}

function loadLiveUrl() {
  const url = document.getElementById("liveUrl").value.trim();
  if (!url) return;
  localStorage.setItem("netmon_live_url", url);
  connectWhep(url);
}

// --- web joystick: on-screen virtual stick -> /api/joystick -> JoystickAdapter -------
// Pure client-side input capture + a fixed-rate tick loop. ALL safety semantics (armed
// gating, motor disable) live server-side (joystick_bridge.py forwarding to
// SVD48VBase.JoystickAdapter_sendData, which already ignores axes while disarmed) --
// this only reports what the stick/buttons currently show, and renders back whatever
// the server says the REAL armed state is rather than assuming a click "worked".
// Axis sign convention (side/rotate direction, "up = forward") is my best guess from
// reading the config, not verified against the physical robot -- flip in
// joySendTick()/the pad callbacks below if a first real test shows it backwards.
let joyAxes = { advance: 0, side: 0, rotate: 0 };
let joyArmed = false, joyTickTimer = null;
let joyPendingArm = null, joyPendingStop = false, joyPendingBlock = false;
const JOY_TICK_MS = 150;
// Matches python_xbox_controller's etc/config_shadow Axis_0 (advance) range -750..750;
// "side"/"rotate" there are already -1..1, same as this stick's native output. Exact
// scale doesn't affect safety -- SVD48VBase clamps to maxLinSpeed/maxRotSpeed regardless,
// this just avoids a barely-moving stick if the scale were left at 1.
const JOY_ADVANCE_SCALE = 750;

function setupPad2D(padEl, knobEl, onChange) {
  let dragging = false;
  function apply(nx, ny) {
    const r = padEl.clientWidth / 2, kr = knobEl.clientWidth / 2;
    knobEl.style.left = (r - kr + nx * (r - kr)) + "px";
    knobEl.style.top = (r - kr + ny * (r - kr)) + "px";
    onChange(nx, ny);
  }
  function fromEvent(e) {
    const rect = padEl.getBoundingClientRect();
    const px = e.clientX - (rect.left + rect.width / 2);
    const py = e.clientY - (rect.top + rect.height / 2);
    const r = rect.width / 2;
    const mag = Math.hypot(px, py);
    const scale = mag > r ? r / mag : 1;
    apply((px * scale) / r, (py * scale) / r);
  }
  padEl.addEventListener("pointerdown", (e) => { dragging = true; padEl.setPointerCapture(e.pointerId); fromEvent(e); });
  padEl.addEventListener("pointermove", (e) => { if (dragging) fromEvent(e); });
  const release = () => { dragging = false; apply(0, 0); };
  padEl.addEventListener("pointerup", release);
  padEl.addEventListener("pointercancel", release);
  // NOT called here: the panel is display:none at setup time (init() runs once at page
  // load, before the joystick is ever opened), so padEl.clientWidth is 0 and this would
  // center the knob against a zero-size box -- i.e. plant it at literal (0,0) instead of
  // the pad's middle. Caller re-centers via the returned function once the panel is
  // actually visible (see joyOpen()).
  return () => apply(0, 0);
}

function setupSlider1D(trackEl, knobEl, onChange) {
  let dragging = false;
  function apply(n) {
    const w = trackEl.clientWidth, kw = knobEl.clientWidth;
    knobEl.style.left = ((w - kw) / 2 + n * (w - kw) / 2) + "px";
    onChange(n);
  }
  function fromEvent(e) {
    const rect = trackEl.getBoundingClientRect();
    const px = e.clientX - (rect.left + rect.width / 2);
    apply(Math.max(-1, Math.min(1, px / (rect.width / 2))));
  }
  trackEl.addEventListener("pointerdown", (e) => { dragging = true; trackEl.setPointerCapture(e.pointerId); fromEvent(e); });
  trackEl.addEventListener("pointermove", (e) => { if (dragging) fromEvent(e); });
  const release = () => { dragging = false; apply(0); };
  trackEl.addEventListener("pointerup", release);
  trackEl.addEventListener("pointercancel", release);
  return () => apply(0);   // see setupPad2D's comment -- same "hidden at setup" issue
}

function joyRenderArmed() {
  const btn = document.getElementById("joyArmBtn");
  btn.className = joyArmed ? "armed" : "disarmed";
  btn.textContent = joyArmed ? "🔒 ARMADO — pulsa para desarmar" : "🔓 DESARMADO — pulsa para armar";
}

async function joySendTick() {
  const body = {
    axes: { advance: joyAxes.advance * JOY_ADVANCE_SCALE, side: joyAxes.side, rotate: joyAxes.rotate },
    arm: joyPendingArm, stop: joyPendingStop, block: joyPendingBlock,
  };
  joyPendingArm = null; joyPendingStop = false; joyPendingBlock = false;
  try {
    const r = await (await fetch("/api/joystick", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    })).json();
    const status = document.getElementById("joystatus");
    if (!r.ok) { status.textContent = "⚠ " + (r.error || "error"); return; }
    if (joyArmed !== r.armed) { joyArmed = r.armed; joyRenderArmed(); }
    status.textContent = r.available ? "" : "⚠ sin conexión ICE (¿rcnode caído?)";
  } catch { document.getElementById("joystatus").textContent = "⚠ sin conexión con netmon"; }
}

let joyPadReset = null, joySliderReset = null;

function joyOpen() {
  document.getElementById("joy").classList.remove("hidden");
  // Re-center now that the panel actually has layout (see setupPad2D/setupSlider1D's
  // comments) -- without this the knobs stay wherever init()'s zero-size computation
  // left them, which visually is stuck at the pad's corner instead of its middle.
  if (joyPadReset) joyPadReset();
  if (joySliderReset) joySliderReset();
  if (joyTickTimer) clearInterval(joyTickTimer);
  joyTickTimer = setInterval(joySendTick, JOY_TICK_MS);
  joySendTick();
}

function joyClose() {
  // Best-effort disarm on close via sendBeacon (works even during unload); the
  // server-side deadman watchdog is the real backstop if this never lands.
  if (joyArmed) {
    navigator.sendBeacon("/api/joystick", new Blob(
      [JSON.stringify({ axes: { advance: 0, side: 0, rotate: 0 }, arm: false })],
      { type: "application/json" }));
  }
  if (joyTickTimer) { clearInterval(joyTickTimer); joyTickTimer = null; }
  document.getElementById("joy").classList.add("hidden");
  joyArmed = false; joyRenderArmed();
}

function joyToggleArm() { joyPendingArm = !joyArmed; joySendTick(); }
function joyStop() { joyPendingStop = true; joySendTick(); }

async function init() {
  await loadTopology();
  document.getElementById("layoutBtn").addEventListener("click", cycleLayout);
  document.getElementById("fitBtn").addEventListener("click",
    () => { applyLayout(layoutMode); network.fit(); });
  document.getElementById("tableBtn").addEventListener("click", () => {
    const hidden = document.getElementById("panel").classList.toggle("hidden");
    document.getElementById("resizer").classList.toggle("hidden", hidden);
    if (network) setTimeout(() => network.redraw(), 0);
  });
  bindEdgeToggle("toggleRpc", "rpc");
  bindEdgeToggle("toggleIcestorm", "icestorm");
  bindEdgeToggle("toggleDds", "dds");

  const liveUrlInput = document.getElementById("liveUrl");
  liveUrlInput.value = localStorage.getItem("netmon_live_url") || `http://${location.hostname}:8889/theta`;
  document.getElementById("liveBtn").addEventListener("click",
    () => setLiveVisible(document.getElementById("live").classList.contains("hidden")));
  document.getElementById("liveClose").addEventListener("click", () => setLiveVisible(false));
  document.getElementById("liveLoad").addEventListener("click", loadLiveUrl);
  liveUrlInput.addEventListener("keydown", (e) => { if (e.key === "Enter") loadLiveUrl(); });

  // Main pad: vertical = advance, horizontal = rotate (tank/differential-drive style --
  // one stick to drive forward/back and turn). Side-stepping (omni) is the separate
  // slider, since a differential base mostly won't use it. Every movement (including
  // the release-to-zero) sends immediately instead of waiting for the next heartbeat
  // tick, so letting go of the stick stops the robot right away, not up to 150ms later.
  joyPadReset = setupPad2D(document.getElementById("joypad"), document.getElementById("joyknob"),
    (nx, ny) => { joyAxes.advance = -ny; joyAxes.rotate = -nx; joySendTick(); });
  joySliderReset = setupSlider1D(document.getElementById("joyrot"), document.getElementById("joyrotknob"),
    (n) => { joyAxes.side = n; joySendTick(); });
  document.getElementById("joyBtn").addEventListener("click",
    () => { document.getElementById("joy").classList.contains("hidden") ? joyOpen() : joyClose(); });
  document.getElementById("joyClose").addEventListener("click", joyClose);
  document.getElementById("joyArmBtn").addEventListener("click", joyToggleArm);
  document.getElementById("joyStopBtn").addEventListener("click", joyStop);
  // Extra client-side safety net on top of the server-side deadman watchdog: don't
  // wait 750ms of silence if we already KNOW the tab went to the background.
  document.addEventListener("visibilitychange", () => { if (document.hidden && joyArmed) joyClose(); });
  window.addEventListener("beforeunload", () => {
    if (joyArmed) navigator.sendBeacon("/api/joystick", new Blob(
      [JSON.stringify({ axes: { advance: 0, side: 0, rotate: 0 }, arm: false })],
      { type: "application/json" }));
  });
  joyRenderArmed();

  // manual drag frees that node from the level layout's y-lock, so it can be
  // repositioned anywhere afterwards to declutter a crowded graph
  network.on("dragEnd", (params) => {
    if (!params.nodes || !params.nodes.length) return;
    nodes.update(params.nodes.map(id => ({ id, fixed: false })));
  });

  // draggable divider to resize the side panel
  const resizer = document.getElementById("resizer"), panel = document.getElementById("panel");
  let resizing = false;
  resizer.addEventListener("mousedown", (e) => {
    resizing = true; resizer.classList.add("drag");
    document.body.style.userSelect = "none"; e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!resizing) return;
    const w = Math.max(260, Math.min(window.innerWidth - e.clientX, window.innerWidth - 320));
    panel.style.width = w + "px";
    if (network) network.redraw();
  });
  window.addEventListener("mouseup", () => {
    if (!resizing) return;
    resizing = false; resizer.classList.remove("drag");
    document.body.style.userSelect = "";
    if (network) network.redraw();
  });

  // select a component from the table row or from a graph node
  document.getElementById("tbody").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-name]");
    if (tr) selectComponent(tr.dataset.name);
  });
  network.on("click", (params) => {
    if (params.nodes && params.nodes.length) selectComponent(params.nodes[0]);
    else clearSelection();
  });

  // contextual actions in the top bar (act on the selected component)
  const btnStart = document.getElementById("selStart");
  const btnRestart = document.getElementById("selRestart");
  const btnStop = document.getElementById("selStop");
  const btnBuild = document.getElementById("selBuild");
  btnStart.addEventListener("click", withCooldown(btnStart, () => selected && sendAction("start", selected)));
  btnRestart.addEventListener("click", withCooldown(btnRestart, () => selected && sendAction("restart", selected)));
  btnStop.addEventListener("click", withCooldown(btnStop,
    () => { if (selected && confirm(`¿Parar ${selected}?`)) sendAction("stop", selected); }));
  btnBuild.addEventListener("click", withCooldown(btnBuild, () => {
    if (!selected) return;
    sendAction("build", selected);
    // NOTE: cbuild still runs as a plain background process writing to
    // ~/.local/logs/<name>.out (see launcher.py::_build) -- it is NOT inside the tmux
    // session, so 🖥 tty won't show its output. No viewer for build output yet.
    toast(`build → ${selected} (sin panel de salida todavía)`);
  }));

  document.getElementById("selConfig").addEventListener("click", () => selected && openConfig(selected));
  document.getElementById("cfgSave").addEventListener("click", saveConfig);
  document.getElementById("cfgClose").addEventListener("click",
    () => { document.getElementById("cfg").classList.add("hidden"); cfgName = null; });

  document.getElementById("selTty").addEventListener("click", () => selected && openTty(selected));
  document.getElementById("ttyClose").addEventListener("click", closeTty);
  document.getElementById("selWindow").addEventListener("click", () => selected && openWindowView(selected));
  document.getElementById("winviewClose").addEventListener("click", closeWindowView);
  document.getElementById("winviewPicker").addEventListener("change", (e) =>
    winName && connectWindowView(winName, parseInt(e.target.value, 10)));
  document.getElementById("ttyCtrlC").addEventListener("click", () => ttySendKey("C-c"));
  document.getElementById("ttyCtrlD").addEventListener("click", () => ttySendKey("C-d"));
  document.getElementById("ttyline").addEventListener("keydown", ttyKeydown);
  // Keystrokes only reach ttyKeydown while #ttyline has focus -- clicking the output
  // screen (the natural place to click before typing) would otherwise silently eat
  // every keypress with the browser's default focus (nowhere in particular / body),
  // which looks exactly like "Ctrl-C does nothing" from the outside.
  document.getElementById("ttyscreen").addEventListener("click",
    () => document.getElementById("ttyline").focus());

  poll();
  setInterval(poll, 1000);
}

function renderBattery(b) {
  const el = document.getElementById("battery");
  if (!b) { el.style.display = "none"; return; }
  el.style.display = "";
  if (!b.available) {
    el.className = "pill warn";
    el.textContent = "🔋 sin datos";
    el.title = "batería: " + (b.error || "n/d");
    return;
  }
  const pct = b.percentage;
  const icon = b.state === "Charging" ? "⚡" : "🔋";
  el.className = "pill " + (pct <= 20 ? "warn" : "ok");
  el.textContent = `${icon} ${pct}%  ${b.voltage}V ${b.current > 0 ? "+" : ""}${b.current}A`;
  el.title = `batería ${b.state} — ${b.voltage} V · ${b.current} A · ${pct}%`;
}

async function poll() {
  let st;
  try { st = await (await fetch("/api/state")).json(); }
  catch { return; }

  // rebuild only when the set of process nodes actually changes (launcher started/stopped)
  const cur = new Set(st.nodes.map(n => n.name));
  if (cur.size !== expectedProc.size || [...cur].some(x => !expectedProc.has(x))) {
    await loadTopology();
    return;
  }

  renderTable(st.nodes);
  renderMediaTable(st.media);
  if (selected && !cur.has(selected)) clearSelection();

  st.nodes.forEach(n => {
    const info = nodeInfo[n.name];
    if (!info) return;
    const c = STATUS_COLOR[n.status] || STATUS_COLOR.unknown;
    nodes.update({
      id: n.name,
      color: c,
      title: `${n.name} — ${n.status}\ncpu ${n.cpu}%  mem ${n.mem} MB\n${info.title}`,
      label: `${n.name}\ncpu ${n.cpu}%${info.serves ? "\n" + info.serves : ""}`,
      // Keep the selection aura's color in sync with the node's OWN status color as it
      // changes (e.g. goes from up/yellow to alive/green while selected) -- DataSet.update
      // merges by field, so leaving `shadow` out entirely for every other node doesn't
      // touch/clear whatever aura state they already have.
      ...(n.name === selected ? { shadow: nodeAura(c.border) } : {}),
    });
  });

  for (const [key, id] of Object.entries(rpcKey)) {
    const bps = st.edges_bw[key] || 0;
    const base = edges.get(id);
    if (!base) continue;
    const baseLabel = (base.label || "").split("\n")[0];
    const active = bps > 1;
    edges.update({
      id,
      label: baseLabel + (active ? "\n" + fmtBps(bps) : ""),
      width: active ? Math.min(8, 1 + Math.log2(1 + bps / 512)) : 1,
      color: { color: active ? "#3fb950" : "#4a5568", highlight: "#7fd1ff" },
    });
  }

  const ddsBw = st.dds_bw || {};
  for (const [id, meta] of Object.entries(ddsEdgeMeta)) {
    const base = edges.get(id);
    if (!base) continue;
    const bps = meta.topics.reduce((sum, t) => sum + (ddsBw[t] || 0), 0);
    const active = bps > 1;
    edges.update({
      id,
      label: meta.label + (active ? "\n" + fmtBps(bps) : ""),
      width: active ? Math.min(8, 1 + Math.log2(1 + bps / 512)) : 1.5,
      color: { color: active ? "#3fb950" : "#e0952b", highlight: active ? "#7fd1ff" : "#f5b95a" },
    });
  }

  const bw = document.getElementById("bw");
  if (st.bw_available) { bw.className = "pill ok"; bw.textContent = "captura activa"; }
  else { bw.className = "pill warn"; bw.textContent = "bw: " + (st.bw_error || "n/d"); }

  renderBattery(st.battery);
  document.getElementById("ts").textContent = new Date().toLocaleTimeString();
}

init();
