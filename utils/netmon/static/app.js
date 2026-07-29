const STATUS_COLOR = {
  alive:   { background: "#16351c", border: "#3fb950" },
  running: { background: "#16351c", border: "#3fb950" },
  up:      { background: "#3a3410", border: "#d6b528" },
  stopped: { background: "#3d1618", border: "#f85149" },
  unknown: { background: "#26292f", border: "#6b7280" },
};
const ROLE_COLOR = {
  broker:    { background: "#2b1d3d", border: "#b072e0" },
  ddsbroker: { background: "#3d2a14", border: "#e0952b" },
  external:  { background: "#22252b", border: "#55606e" },
  dsr:       { background: "#10303a", border: "#2ec5d3" },
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
// stream (e.g. "voxelizer:zed:rgb" vs "robot_concept:ingest:zed_camera:rgb"), so an
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

// --- selection + process control + log viewer ---
let selected = null, logName = null, logStream = "err";

function selectComponent(name) {
  if (!expectedProc.has(name)) { clearSelection(); return; }   // only controllable nodes
  selected = name;
  document.getElementById("selname").textContent = name;
  document.getElementById("sel").classList.remove("hidden");
  document.querySelectorAll("#tbody tr").forEach(tr =>
    tr.classList.toggle("selrow", tr.dataset.name === name));
  if (network) network.selectNodes([name]);
  openLogs(name);                                              // jump to its terminal
}

function clearSelection() {
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

function openLogs(name) {
  document.getElementById("cfg").classList.add("hidden");   // mutually exclusive, both dock at the bottom
  logName = name;
  document.getElementById("logtitle").textContent = "logs · " + name;
  document.getElementById("logs").classList.remove("hidden");
  refreshLogs();
}

// --- config file editor: reads the etc/ file the component's own `cmd` points at ---
let cfgName = null, cfgPath = null, cfgOriginal = null;

async function openConfig(name) {
  document.getElementById("logs").classList.add("hidden");
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

async function refreshLogs() {
  if (!logName || document.getElementById("logs").classList.contains("hidden")) return;
  document.getElementById("logstream").textContent = logStream === "err" ? "stderr" : "stdout";
  try {
    const r = await (await fetch(`/api/logs?name=${encodeURIComponent(logName)}&stream=${logStream}&lines=400`)).json();
    const pre = document.getElementById("logtext");
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30;
    pre.textContent = r.text || "(sin log)";
    if (atBottom) pre.scrollTop = pre.scrollHeight;   // follow tail unless user scrolled up
  } catch { /* ignore */ }
}

function bindEdgeToggle(btnId, group) {
  const btn = document.getElementById(btnId);
  btn.classList.toggle("off", !edgeVisibility[group]);
  btn.addEventListener("click", () => {
    setEdgeGroupVisible(group, !edgeVisibility[group]);
    btn.classList.toggle("off", !edgeVisibility[group]);
  });
}

// --- live view: generic iframe embed (works with mediamtx's built-in WebRTC player
// page at http://host:8889/<path>, or any other URL the user pastes in) ---
function setLiveVisible(show) {
  document.getElementById("live").classList.toggle("hidden", !show);
  const frame = document.getElementById("liveFrame");
  if (show) {
    const url = document.getElementById("liveUrl").value.trim();
    if (url) frame.src = url;
  } else {
    frame.src = "about:blank";   // stop decoding/bandwidth while closed
  }
}

function loadLiveUrl() {
  const url = document.getElementById("liveUrl").value.trim();
  if (!url) return;
  localStorage.setItem("netmon_live_url", url);
  document.getElementById("liveFrame").src = url;
}

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
  document.getElementById("selTerm").addEventListener("click", () => selected && openLogs(selected));
  btnBuild.addEventListener("click", withCooldown(btnBuild, () => {
    if (!selected) return;
    sendAction("build", selected);
    logStream = "out";
    openLogs(selected);           // jump to stdout so the cbuild output is visible right away
  }));

  document.getElementById("logOut").addEventListener("click", () => { logStream = "out"; refreshLogs(); });
  document.getElementById("logErr").addEventListener("click", () => { logStream = "err"; refreshLogs(); });
  document.getElementById("logClose").addEventListener("click",
    () => { document.getElementById("logs").classList.add("hidden"); logName = null; });
  setInterval(refreshLogs, 1500);

  document.getElementById("selConfig").addEventListener("click", () => selected && openConfig(selected));
  document.getElementById("cfgSave").addEventListener("click", saveConfig);
  document.getElementById("cfgClose").addEventListener("click",
    () => { document.getElementById("cfg").classList.add("hidden"); cfgName = null; });

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
