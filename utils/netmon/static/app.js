const STATUS_COLOR = {
  alive:   { background: "#16351c", border: "#3fb950" },
  running: { background: "#16351c", border: "#3fb950" },
  up:      { background: "#3a3410", border: "#d6b528" },
  stopped: { background: "#3d1618", border: "#f85149" },
  unknown: { background: "#26292f", border: "#6b7280" },
};
const ROLE_COLOR = {
  broker:   { background: "#2b1d3d", border: "#b072e0" },
  external: { background: "#22252b", border: "#55606e" },
  dsr:      { background: "#10303a", border: "#2ec5d3" },
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
  const base = { external: 0, component: 0, broker: compMax + 2, agent: compMax + 3, dsr: compMax + 5 };

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
  return `${n.id}\n${L.join("\n") || "(sin interfaces ICE)"}`;
}

let network, nodes, edges;
let rpcKey = {};        // "src->dst:port" -> vis edge id
let nodeInfo = {};      // id -> { serves, title }
let nodeLevels = {};    // id -> hierarchy level
let maxLevel = 0;
let expectedProc = new Set();  // process node ids we expect in /api/state

function buildNodes(topo) {
  rpcKey = {}; nodeInfo = {};
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
  const rpcEdges = [], dsrEdges = [], psGroups = {};
  topo.edges.forEach(e => {
    if (e.kind === "pub" || e.kind === "sub") {
      const k = `${e.src}|${e.dst}|${e.kind}`;
      (psGroups[k] || (psGroups[k] = { src: e.src, dst: e.dst, kind: e.kind, topics: [] })).topics.push(e.topic);
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
    });
  });
  dsrEdges.forEach(e => visEdges.push({
    id: "e" + (ei++), from: e.src, to: e.dst, arrows: "",
    color: { color: "#2ec5d3", highlight: "#7fe3ec" }, width: 2, smooth: { type: "continuous" },
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
    });
  });
  return visEdges;
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
    return `<tr><td>${n.name}</td>`
      + `<td><span class="lay ${lay}">${short}</span></td>`
      + `<td class="st" style="color:${col}">${n.status}</td>`
      + `<td class="num">${n.cpu}%</td>`
      + `<td class="num">${n.mem}</td></tr>`;
  }).join("");
}

async function init() {
  await loadTopology();
  document.getElementById("layoutBtn").addEventListener("click", cycleLayout);
  document.getElementById("fitBtn").addEventListener("click",
    () => { applyLayout(layoutMode); network.fit(); });
  document.getElementById("tableBtn").addEventListener("click",
    () => document.getElementById("panel").classList.toggle("hidden"));
  poll();
  setInterval(poll, 1000);
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

  const bw = document.getElementById("bw");
  if (st.bw_available) { bw.className = "pill ok"; bw.textContent = "captura activa"; }
  else { bw.className = "pill warn"; bw.textContent = "bw: " + (st.bw_error || "n/d"); }
  document.getElementById("ts").textContent = new Date().toLocaleTimeString();
}

init();
