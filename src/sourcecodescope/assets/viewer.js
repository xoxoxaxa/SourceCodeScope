/* SourceCodeScope viewer. Reads the embedded IR and draws it with D3 + ELK. */
(() => {
  "use strict";

  const DATA = JSON.parse(document.getElementById("scs-data").textContent);
  const CONFIG = JSON.parse(document.getElementById("scs-config").textContent);
  const rootEl = document.documentElement;
  const XLINK = "http://www.w3.org/1999/xlink";

  const nodes = new Map(DATA.nodes.map((n) => [n.id, n]));
  const tensors = new Map(DATA.tensors.map((t) => [t.id, t]));
  const childMap = new Map();
  for (const n of DATA.nodes) {
    const key = n.parent ?? -1;
    if (!childMap.has(key)) childMap.set(key, []);
    childMap.get(key).push(n);
  }
  const kidsOf = (id) => childMap.get(id) || [];
  const isContainer = (n) => kidsOf(n.id).length > 0;
  const ancestors = (n) => {
    const out = [];
    for (let p = n.parent; p != null; p = nodes.get(p).parent) out.push(nodes.get(p));
    return out;
  };

  const state = {
    mode: CONFIG.mode,
    depth: CONFIG.depth,
    expanded: new Set(),
    selected: null,
    playing: !window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    head: 0,
  };
  let model = null; // last layout result
  let anchor = null; // keeps a toggled node fixed on screen

  // ---------------------------------------------------------------- helpers
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const clip = (s, n) => (s.length > n ? s.slice(0, Math.max(1, n - 1)) + "…" : s);
  const color = (cat) => `var(--c-${cat in CATS ? cat : "other"})`;
  const CATS = {
    input: "入力", output: "出力", module: "モジュール", repeat: "繰り返し", conv: "Conv", linear: "Linear",
    attention: "Attention", norm: "Norm", activation: "活性化", rnn: "RNN", embedding: "Embedding",
    shape: "形状操作", math: "演算", pool: "Pool", pad: "Pad", dropout: "Dropout", resample: "Resample", other: "その他",
  };

  function fmtShape(shape, dims) {
    if (!shape) return "";
    if (!shape.length) return "scalar";
    if (dims && dims.length === shape.length) return shape.map((v, i) => `${dims[i]}=${v}`).join(" × ");
    return shape.join("×");
  }
  function fmtCount(v) {
    if (v >= 1e9) return (v / 1e9).toFixed(2) + "B";
    if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
    return String(v);
  }
  function fmtNum(v) {
    if (v === 0) return "0";
    const a = Math.abs(v);
    return a >= 1e4 || a < 1e-3 ? v.toExponential(2) : String(+v.toPrecision(4));
  }
  const shapeOf = (tid) => (tensors.has(tid) ? tensors.get(tid).shape : null);
  function ioText(n) {
    const dims = n.attrs.dims ? n.attrs.dims.split(",") : null;
    const side = (ids) => (ids.length ? fmtShape(shapeOf(ids[0]), dims) + (ids.length > 1 ? ` +${ids.length - 1}` : "") : "");
    return { inp: side(n.inputs), out: side(n.outputs) };
  }

  // ---------------------------------------------------------------- heatmaps
  const lutCache = new Map();
  const imgCache = new Map();
  function lut(preview, kind) {
    const diverging = preview.lo < 0 && preview.hi > 0;
    const name = kind === "attention" ? "magma" : preview.label === "phase" ? "cyclic" : diverging ? `div:${preview.lo}:${preview.hi}` : "seq";
    if (lutCache.has(name)) return lutCache.get(name);
    const table = new Uint8ClampedArray(1024);
    const span = Math.max(-preview.lo, preview.hi) || 1;
    for (let q = 0; q < 256; q++) {
      let c;
      if (name === "magma") c = d3.interpolateMagma(q / 255);
      else if (name === "cyclic") c = d3.interpolateSinebow(q / 255);
      else if (name === "seq") c = d3.interpolateViridis(q / 255);
      else c = d3.interpolateRdBu(0.5 - (preview.lo + (q / 255) * (preview.hi - preview.lo)) / (2 * span));
      const rgb = d3.rgb(c);
      table.set([rgb.r, rgb.g, rgb.b, 255], q * 4);
    }
    lutCache.set(name, table);
    return table;
  }
  function bytesOf(preview) {
    if (!preview.bytes) {
      const raw = atob(preview.data);
      preview.bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) preview.bytes[i] = raw.charCodeAt(i);
    }
    return preview.bytes;
  }
  function dimsOf(preview) {
    if (preview.layout === "maps") return preview.shape;
    if (preview.layout === "grid") return [1, preview.shape[0], preview.shape[1]];
    return [1, 1, preview.shape[0]];
  }
  /** Data URL of one preview map, or null when the tensor has no preview. */
  function heat(tid, pi = 0, mi = 0, kind = "") {
    const key = `${tid}:${pi}:${mi}:${kind}`;
    if (imgCache.has(key)) return imgCache.get(key);
    const preview = tensors.get(tid)?.previews[pi];
    if (!preview) return null;
    const [count, h, w] = dimsOf(preview);
    const index = clamp(mi, 0, count - 1);
    const bytes = bytesOf(preview);
    const table = lut(preview, kind);
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    const image = ctx.createImageData(w, h);
    for (let i = 0, off = index * h * w; i < h * w; i++) image.data.set(table.subarray(bytes[off + i] * 4, bytes[off + i] * 4 + 4), i * 4);
    ctx.putImageData(image, 0, 0);
    const url = canvas.toDataURL("image/png");
    imgCache.set(key, url);
    return url;
  }
  function thumbOf(n) {
    if (n.kind === "op" && n.attention == null) return null;
    if (n.attention != null && tensors.get(n.attention)?.previews.length) return { tid: n.attention, kind: "attention" };
    const tid = n.outputs.length ? n.outputs[0] : n.inputs[0];
    return tid != null && tensors.get(tid)?.previews.length ? { tid, kind: "" } : null;
  }

  // ---------------------------------------------------------------- layout
  const elk = new ELK();
  const HEADER = 30;

  function nodeSize(n) {
    const io = ioText(n);
    if (state.mode === "flow") {
      const chars = Math.max(n.name.length + n.type.length + 3, io.inp.length + 5, io.out.length + 5);
      const lines = (io.inp ? 1 : 0) + (io.out ? 1 : 0);
      return [clamp(chars * 6.5 + 30, 130, 320), 30 + lines * 15 + 4];
    }
    const big = !!thumbOf(n);
    const chars = Math.max(n.name.length * 1.12, n.type.length * 0.95, (io.out || io.inp).length * 0.95);
    return [clamp(chars * 6.6 + 34, big ? 156 : 92, 236), big ? 126 : 46];
  }

  function applyDepth(depth) {
    state.depth = Math.max(0, depth);
    state.expanded = new Set(
      DATA.nodes
        .filter((n) => isContainer(n) && n.kind !== "repeat")
        .filter((n) => { const up = ancestors(n); return up.length < state.depth && !up.some((a) => a.kind === "repeat"); })
        .map((n) => n.id),
    );
  }

  function visibleRep(cache, id) {
    if (cache.has(id)) return cache.get(id);
    const chain = [nodes.get(id), ...ancestors(nodes.get(id))];
    let rep = id;
    for (let i = chain.length - 1; i >= 0; i--) {
      if (!state.expanded.has(chain[i].id)) { rep = chain[i].id; break; }
    }
    cache.set(id, rep);
    return rep;
  }

  async function computeLayout() {
    const flow = state.mode === "flow";
    const toElk = (n) => {
      if (state.expanded.has(n.id) && isContainer(n)) {
        const label = (n.name.length + n.type.length) * 7 + 70;
        return {
          id: "n" + n.id,
          children: kidsOf(n.id).map(toElk),
          layoutOptions: {
            "elk.padding": `[top=${HEADER + 12},left=16,bottom=16,right=16]`,
            "elk.nodeSize.constraints": "MINIMUM_SIZE",
            "elk.nodeSize.minimum": `(${label},${HEADER + 30})`,
          },
        };
      }
      const [width, height] = nodeSize(n);
      return { id: "n" + n.id, width, height };
    };
    const cache = new Map();
    const merged = new Map();
    for (const e of DATA.edges) {
      if (!nodes.has(e.src) || !nodes.has(e.dst)) continue;
      const s = visibleRep(cache, e.src);
      const t = visibleRep(cache, e.dst);
      if (s === t) continue;
      const key = `${s}>${t}`;
      if (!merged.has(key)) merged.set(key, { key, s, t, tids: [] });
      if (!merged.get(key).tids.includes(e.tensor)) merged.get(key).tids.push(e.tensor);
    }
    const links = [...merged.values()];
    for (const link of links) {
      let elems = 0;
      for (const tid of link.tids) {
        const t = tensors.get(tid);
        elems += t ? t.numel / (t.shape.length > 1 ? Math.max(1, t.shape[0]) : 1) : 1;
      }
      link.width = flow ? 1.4 : clamp(2 + 2 * Math.log10(Math.max(1, elems)), 2, 16);
    }
    const graph = {
      id: "root",
      layoutOptions: {
        "elk.algorithm": "layered",
        "elk.direction": flow ? "DOWN" : "RIGHT",
        "elk.hierarchyHandling": "INCLUDE_CHILDREN",
        "elk.edgeRouting": flow ? "ORTHOGONAL" : "SPLINES",
        "elk.layered.spacing.nodeNodeBetweenLayers": flow ? "46" : "54",
        "elk.spacing.nodeNode": flow ? "26" : "24",
        "elk.spacing.edgeNode": "18",
        "elk.spacing.edgeEdge": flow ? "10" : "14",
        "elk.layered.spacing.edgeNodeBetweenLayers": "18",
        "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
        "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
        "elk.padding": "[top=10,left=10,bottom=10,right=10]",
      },
      children: kidsOf(-1).map(toElk),
      edges: links.map((l, i) => ({ id: "e" + i, sources: ["n" + l.s], targets: ["n" + l.t], layoutOptions: { "elk.edge.thickness": String(l.width) } })),
    };
    const result = await elk.layout(graph);
    const boxes = [];
    const origin = new Map([["root", [0, 0]]]);
    const routed = [];
    (function walk(parent, ox, oy) {
      for (const e of parent.edges || []) routed.push(e);
      for (const c of parent.children || []) {
        const x = ox + c.x;
        const y = oy + c.y;
        origin.set(c.id, [x, y]);
        boxes.push({ id: +c.id.slice(1), x, y, w: c.width, h: c.height, group: !!c.children });
        walk(c, x, y);
      }
    })(result, 0, 0);
    for (const e of routed) {
      const link = links[+e.id.slice(1)];
      const [ox, oy] = origin.get(e.container) || [0, 0];
      link.points = [];
      for (const s of e.sections || []) {
        for (const p of [s.startPoint, ...(s.bendPoints || []), s.endPoint]) link.points.push([p.x + ox, p.y + oy]);
      }
    }
    return { boxes, links: links.filter((l) => l.points && l.points.length > 1), w: result.width, h: result.height };
  }

  // ---------------------------------------------------------------- drawing
  const SVG_CSS = `
text{font-family:"Segoe UI","Yu Gothic UI","Hiragino Sans",system-ui,sans-serif;fill:var(--text);font-size:12px}
.mono{font-family:"Cascadia Code",Consolas,ui-monospace,Menlo,monospace;font-size:10.5px}
.muted{fill:var(--muted)}
.name{font-weight:600;font-size:12.5px}
.type{font-size:10.5px;fill:var(--muted)}
.head text{paint-order:stroke;stroke:var(--bg);stroke-width:5px;stroke-linejoin:round}
.card{fill:var(--surface);stroke:var(--line)}
.stack{fill:var(--surface2);stroke:var(--line)}
.frame{fill:none;stroke:var(--line)}
.box{fill:var(--group);stroke:var(--group-line)}
.badge{fill:var(--surface2);stroke:var(--line)}
.badge-t{font-size:11px;font-weight:700;fill:var(--muted);text-anchor:middle}
.times{font-size:11px;font-weight:700;fill:var(--accent);text-anchor:end}
.node,.head{cursor:pointer}
.node:hover .card{stroke:var(--accent)}
.sel .card,.sel>.box{stroke:var(--accent);stroke-width:2}
.edge{fill:none}
.flowline{stroke:var(--edge)}
.arrow{fill:var(--edge)}
.elabel{font-size:9.5px;fill:var(--muted);text-anchor:middle;paint-order:stroke;stroke:var(--bg);stroke-width:4px;stroke-linejoin:round}
.flowdash{fill:none;stroke:var(--flow-dash);stroke-dasharray:1.5 16.5;stroke-linecap:round;opacity:0}
.playing .flowdash{opacity:.8;animation:scsflow 1s linear infinite}
@keyframes scsflow{to{stroke-dashoffset:-18}}
.dim .node:not(.hl),.dim .link:not(.hl){opacity:.2}
.node,.link{transition:opacity .15s}
`;
  const svg = d3.select("#canvas");
  const defs = svg.append("defs");
  defs.append("style").text(SVG_CSS);
  defs.append("marker").attr("id", "arrow").attr("viewBox", "0 0 8 8").attr("refX", 7).attr("refY", 4)
    .attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto-start-reverse")
    .append("path").attr("class", "arrow").attr("d", "M0,0.5 L8,4 L0,7.5 z");
  const viewport = svg.append("g").attr("class", "viewport");
  const gLinks = viewport.append("g");
  const gGroups = viewport.append("g");
  const gNodes = viewport.append("g");
  const zoom = d3.zoom().scaleExtent([0.05, 4]).on("zoom", (ev) => viewport.attr("transform", ev.transform));
  svg.call(zoom).on("dblclick.zoom", null);
  const f1 = (v) => +v.toFixed(1);
  /** ELK spline routes: smooth B-spline over the routed control points. */
  const splinePath = d3.line().curve(d3.curveBasis);
  /** Orthogonal routes drawn with rounded corners. */
  function roundedPath(points, radius = 7) {
    let d = `M${f1(points[0][0])},${f1(points[0][1])}`;
    for (let i = 1; i < points.length - 1; i++) {
      const [a, b, c] = [points[i - 1], points[i], points[i + 1]];
      const r = Math.min(radius, Math.hypot(b[0] - a[0], b[1] - a[1]) / 2, Math.hypot(c[0] - b[0], c[1] - b[1]) / 2);
      const toward = (p, q) => { const l = Math.hypot(q[0] - p[0], q[1] - p[1]) || 1; return [p[0] + ((q[0] - p[0]) * r) / l, p[1] + ((q[1] - p[1]) * r) / l]; };
      const p1 = toward(b, a), p2 = toward(b, c);
      d += `L${f1(p1[0])},${f1(p1[1])}Q${f1(b[0])},${f1(b[1])} ${f1(p2[0])},${f1(p2[1])}`;
    }
    const last = points[points.length - 1];
    return d + `L${f1(last[0])},${f1(last[1])}`;
  }
  const line = (points) => (state.mode === "flow" ? roundedPath(points) : splinePath(points));

  function badge(g, x, y, symbol) {
    g.append("circle").attr("class", "badge").attr("cx", x).attr("cy", y).attr("r", 8);
    g.append("text").attr("class", "badge-t").attr("x", x).attr("y", y + 4).text(symbol);
  }

  function buildCard(g, d) {
    const n = nodes.get(d.id);
    const io = ioText(n);
    const container = isContainer(n);
    const flow = state.mode === "flow";
    if (n.kind === "repeat") {
      for (const o of [8, 4]) g.append("rect").attr("class", "stack").attr("x", o).attr("y", o).attr("width", d.w).attr("height", d.h).attr("rx", 10);
    }
    g.append("rect").attr("class", "card").attr("width", d.w).attr("height", d.h).attr("rx", 10);
    g.append("rect").attr("x", 0).attr("y", 9).attr("width", 3.5).attr("height", d.h - 18).attr("rx", 1.75).attr("fill", color(n.category));
    const room = Math.floor((d.w - (container ? 44 : 24)) / 6.9);
    const typeText = n.kind === "op" || n.kind === "input" || n.kind === "output" ? "" : n.type;
    if (flow) {
      const title = g.append("text").attr("x", 13).attr("y", 20);
      title.append("tspan").attr("class", "name").text(clip(n.name, room));
      if (typeText) title.append("tspan").attr("class", "type").attr("dx", 6).text(clip(typeText, Math.max(3, room - n.name.length - 1)));
      let y = 37;
      for (const [tag, text] of [["in ", io.inp], ["out", io.out]]) {
        if (!text) continue;
        const row = g.append("text").attr("class", "mono").attr("x", 13).attr("y", y);
        row.append("tspan").attr("class", "muted").text(tag + " ");
        row.append("tspan").text(text);
        y += 15;
      }
    } else {
      const thumb = thumbOf(n);
      g.append("text").attr("class", "name").attr("x", 13).attr("y", 20).text(clip(n.name, room));
      if (thumb) {
        if (typeText) g.append("text").attr("class", "type").attr("x", 13).attr("y", 34).text(clip(typeText, room + 4));
        const url = heat(thumb.tid, 0, 0, thumb.kind);
        const square = thumb.kind === "attention";
        const iw = square ? 60 : d.w - 22;
        const ix = square ? (d.w - iw) / 2 : 11;
        const image = g.append("image").attr("x", ix).attr("y", 42).attr("width", iw).attr("height", 60)
          .attr("preserveAspectRatio", "none").attr("style", "image-rendering:pixelated").attr("href", url);
        image.node().setAttributeNS(XLINK, "xlink:href", url);
        g.append("rect").attr("class", "frame").attr("x", ix).attr("y", 42).attr("width", iw).attr("height", 60).attr("rx", 2);
      }
      const label = g.append("text").attr("class", "mono").attr("x", 13).attr("y", d.h - 10).text(io.out || io.inp);
      if (!thumb) label.attr("class", "mono muted");
      if (n.params && thumb) g.append("text").attr("class", "mono muted").attr("x", d.w - 10).attr("y", d.h - 10).attr("text-anchor", "end").text(fmtCount(n.params));
    }
    if (n.kind === "repeat") g.append("text").attr("class", "times").attr("x", d.w - 30).attr("y", 20).text("×" + n.count);
    if (container) badge(g, d.w - 15, 16, "+");
    g.append("title").text(`${n.path}\n${n.type}`);
  }

  function buildGroup(g, d) {
    const n = nodes.get(d.id);
    const head = g.append("g").attr("class", "head");
    g.append("rect").attr("class", "box").attr("rx", 13).lower();
    head.append("rect").attr("class", "hit").attr("height", HEADER + 4).attr("fill", "transparent");
    head.append("rect").attr("x", 12).attr("y", 11).attr("width", 9).attr("height", 9).attr("rx", 2.5).attr("fill", color(n.category));
    const title = head.append("text").attr("x", 27).attr("y", 20);
    title.append("tspan").attr("class", "name").text(n.name);
    title.append("tspan").attr("class", "type").attr("dx", 7).text(n.kind === "repeat" ? `${n.type} ×${n.count}` : n.type);
    head.append("g").attr("class", "minus");
    badge(head.select(".minus"), 0, 16, "−");
    head.on("click", (ev) => { ev.stopPropagation(); toggle(d.id); });
    g.select(".box").on("click", (ev) => { ev.stopPropagation(); select(d.id); });
  }

  function draw(animate) {
    const t = d3.transition().duration(animate ? 420 : 0).ease(d3.easeCubicInOut);
    const place = (d) => `translate(${d.x},${d.y})`;

    const groups = gGroups.selectAll("g.group").data(model.boxes.filter((b) => b.group), (d) => d.id);
    groups.exit().transition(t).attr("opacity", 0).remove();
    const groupsIn = groups.enter().append("g").attr("class", "group").attr("transform", place).attr("opacity", 0);
    groupsIn.each(function (d) { buildGroup(d3.select(this), d); });
    groupsIn.select(".box").attr("width", (d) => d.w).attr("height", (d) => d.h);
    groupsIn.select(".hit").attr("width", (d) => d.w);
    groupsIn.select(".minus").attr("transform", (d) => `translate(${d.w - 16},0)`);
    const groupsAll = groupsIn.merge(groups).order();
    groupsAll.transition(t).attr("transform", place).attr("opacity", 1);
    groupsAll.select(".box").transition(t).attr("width", (d) => d.w).attr("height", (d) => d.h);
    groupsAll.select(".hit").attr("width", (d) => d.w);
    groupsAll.select(".minus").transition(t).attr("transform", (d) => `translate(${d.w - 16},0)`);

    const cards = gNodes.selectAll("g.node").data(model.boxes.filter((b) => !b.group), (d) => `${d.id}:${state.mode}`);
    cards.exit().transition(t).attr("opacity", 0).remove();
    const cardsIn = cards.enter().append("g").attr("class", "node").attr("transform", place).attr("opacity", 0);
    cardsIn.each(function (d) { buildCard(d3.select(this), d); });
    cardsIn
      .on("click", (ev, d) => { ev.stopPropagation(); isContainer(nodes.get(d.id)) ? toggle(d.id) : select(d.id); })
      .on("mouseenter", (ev, d) => highlight(d.id))
      .on("mouseleave", () => highlight(null));
    cardsIn.merge(cards).transition(t).attr("transform", place).attr("opacity", 1);

    const flow = state.mode === "flow";
    gLinks.selectAll("g.link").remove();
    const links = gLinks.selectAll("g.link").data(model.links, (d) => d.key).enter().append("g").attr("class", "link").attr("opacity", 0);
    links.append("path").attr("class", flow ? "edge flowline" : "edge").attr("d", (d) => line(d.points))
      .attr("stroke-width", (d) => d.width)
      .attr("stroke", (d) => (flow ? null : color(nodes.get(d.s).category)))
      .attr("stroke-opacity", flow ? null : 0.42)
      .attr("marker-end", flow ? "url(#arrow)" : null);
    if (flow) {
      links.append("text").attr("class", "mono elabel")
        .attr("x", (d) => along(d.points, 15)[0]).attr("y", (d) => along(d.points, 15)[1] + 3)
        .text((d) => fmtShape(shapeOf(d.tids[0])) + (d.tids.length > 1 ? ` +${d.tids.length - 1}` : ""));
    } else {
      links.append("path").attr("class", "flowdash").attr("d", (d) => line(d.points)).attr("stroke-width", (d) => clamp(d.width * 0.3, 1.2, 3));
    }
    links.transition(t).delay(animate ? 260 : 0).attr("opacity", 1);
    refreshSelection();
  }

  /** Point at ``distance`` from the start of a polyline (clamped to its middle). */
  function along(points, distance) {
    let total = 0;
    const lengths = points.slice(1).map((p, i) => { const l = Math.hypot(p[0] - points[i][0], p[1] - points[i][1]); total += l; return l; });
    let walked = Math.min(distance, total / 2);
    for (let i = 0; i < lengths.length; i++) {
      if (walked <= lengths[i]) {
        const f = lengths[i] ? walked / lengths[i] : 0;
        return [points[i][0] + (points[i + 1][0] - points[i][0]) * f, points[i][1] + (points[i + 1][1] - points[i][1]) * f];
      }
      walked -= lengths[i];
    }
    return points[0];
  }

  function highlight(id) {
    svg.classed("dim", id != null);
    if (id == null) return;
    const near = new Set([id]);
    gLinks.selectAll("g.link").classed("hl", (d) => { const hit = d.s === id || d.t === id; if (hit) { near.add(d.s); near.add(d.t); } return hit; });
    gNodes.selectAll("g.node").classed("hl", (d) => near.has(d.id));
  }

  function refreshSelection() {
    gNodes.selectAll("g.node").classed("sel", (d) => d.id === state.selected);
    gGroups.selectAll("g.group").classed("sel", (d) => d.id === state.selected);
  }

  function fit(animate) {
    if (!model) return;
    const { width, height } = svg.node().getBoundingClientRect();
    const k = clamp(Math.min((width - 60) / model.w, (height - 90) / model.h), 0.05, 1.15);
    const tf = d3.zoomIdentity.translate((width - model.w * k) / 2, (height - model.h * k) / 2 - 10).scale(k);
    (animate ? svg.transition().duration(420) : svg).call(zoom.transform, tf);
  }

  async function relayout({ refit = false, animate = true } = {}) {
    const before = anchor != null && model ? model.boxes.find((b) => b.id === anchor) : null;
    model = await computeLayout();
    const after = before ? model.boxes.find((b) => b.id === anchor) : null;
    draw(animate);
    if (refit || !before || !after) {
      fit(animate);
    } else {
      const tf = d3.zoomTransform(svg.node());
      const moved = tf.translate((before.x - after.x), (before.y - after.y));
      svg.transition().duration(animate ? 420 : 0).ease(d3.easeCubicInOut).call(zoom.transform, moved);
    }
    anchor = null;
    syncBar();
  }

  function focus(id, scale = 1) {
    const box = model && model.boxes.find((b) => b.id === id);
    if (!box) return;
    const { width, height } = svg.node().getBoundingClientRect();
    const tf = d3.zoomIdentity.translate(width / 2 - (box.x + box.w / 2) * scale, height / 2 - (box.y + box.h / 2) * scale).scale(scale);
    svg.transition().duration(420).call(zoom.transform, tf);
  }

  function toggle(id) {
    if (state.expanded.has(id)) state.expanded.delete(id);
    else state.expanded.add(id);
    anchor = id;
    state.selected = id;
    renderDetail();
    relayout();
  }
  function select(id) {
    state.selected = id;
    state.head = 0;
    refreshSelection();
    renderDetail();
  }

  // ---------------------------------------------------------------- detail panel
  const detail = document.getElementById("detail");

  function histSvg(t, title, W = 160) {
    if (!t || !t.hist.length) return "";
    const H = 58, max = Math.max(...t.hist, 1), bw = W / t.hist.length;
    const bars = t.hist.map((c, i) => `<rect class="bar" x="${(i * bw + 0.5).toFixed(1)}" y="${(H - 12 - (c / max) * (H - 14)).toFixed(1)}" width="${(bw - 1).toFixed(1)}" height="${((c / max) * (H - 14)).toFixed(1)}"/>`).join("");
    return `<div><svg class="plot" viewBox="0 0 ${W} ${H}">${bars}<line class="axis" x1="0" x2="${W}" y1="${H - 12}" y2="${H - 12}"/>` +
      `<text x="0" y="${H - 2}">${fmtNum(t.stats.min)}</text><text x="${W}" y="${H - 2}" text-anchor="end">${fmtNum(t.stats.max)}</text></svg>` +
      (title ? `<small>${title}</small>` : "") + `</div>`;
  }

  function tensorCard(tid, kind = "") {
    const t = tensors.get(tid);
    if (!t) return "";
    let html = `<div class="tcard"><span class="shape">${esc(fmtShape(t.shape))}</span><span class="dtype">${esc(t.dtype)} · ${fmtCount(t.numel)} 要素</span>`;
    if (t.stats) html += `<div class="cap"><span>平均 ${fmtNum(t.stats.mean)} ± ${fmtNum(t.stats.std)}</span><span>${fmtNum(t.stats.min)} … ${fmtNum(t.stats.max)}</span></div>`;
    t.previews.forEach((p, pi) => {
      const tag = p.label === "value" ? "" : `<div class="cap"><span>${p.label === "abs" ? "振幅" : "位相"}</span></div>`;
      const [count, h, w] = dimsOf(p);
      if (p.layout === "maps" && count > 1) {
        html += tag + `<div class="maps">` + p.channels.map((c, mi) => `<figure><img src="${heat(tid, pi, mi, kind)}" alt=""><figcaption>ch ${c}</figcaption></figure>`).join("") + `</div>`;
      } else {
        const ratio = clamp(h / w, 0.06, 0.6);
        html += tag + `<img class="heat" style="aspect-ratio:${(1 / ratio).toFixed(3)}" src="${heat(tid, pi, 0, kind)}" alt="">`;
      }
      html += `<div class="cap"><span>${p.layout === "strip" ? `${w} 点に縮小` : `${h}×${w} に縮小`}・先頭バッチ</span><span>${fmtNum(p.lo)} … ${fmtNum(p.hi)}</span></div>`;
    });
    if (t.stats) html += histSvg(t, "", 330);
    return html + `</div>`;
  }

  const ACTIVATIONS = {
    relu: (x) => Math.max(0, x), relu6: (x) => clamp(x, 0, 6), sigmoid: (x) => 1 / (1 + Math.exp(-x)), tanh: Math.tanh,
    gelu: (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))),
    silu: (x) => x / (1 + Math.exp(-x)), softplus: (x) => Math.log1p(Math.exp(x)), elu: (x) => (x > 0 ? x : Math.exp(x) - 1),
    hardtanh: (x) => clamp(x, -1, 1), mish: (x) => x * Math.tanh(Math.log1p(Math.exp(x))),
    hardswish: (x) => (x * clamp(x + 3, 0, 6)) / 6, softsign: (x) => x / (1 + Math.abs(x)),
  };
  function activationFn(n) {
    const key = n.type.toLowerCase().replace(/_/g, "");
    if (key === "leakyrelu" || key === "prelu") {
      const found = /(?:negative_slope=|slope=)([-\d.e]+)/.exec(`${n.attrs.config || ""} ${n.attrs.args || ""}`);
      const slope = n.attrs.slope ? +n.attrs.slope : found ? +found[1] : key === "prelu" ? 0.25 : 0.01;
      return (x) => (x > 0 ? x : slope * x);
    }
    return ACTIVATIONS[key] || null;
  }
  function curveSvg(n) {
    const fn = activationFn(n);
    const t = tensors.get(n.inputs[0]);
    if (!fn || !t || !t.stats) return "";
    const W = 330, H = 150, pad = 16;
    const lo = Math.min(t.stats.min, -0.5), hi = Math.max(t.stats.max, 0.5);
    const xs = d3.range(0, 121).map((i) => lo + ((hi - lo) * i) / 120);
    const ys = xs.map(fn);
    const sx = d3.scaleLinear([lo, hi], [pad, W - pad]);
    const sy = d3.scaleLinear([Math.min(0, d3.min(ys)), Math.max(d3.max(ys), 0.1)], [H - pad, pad]);
    const max = Math.max(...t.hist, 1), bw = (sx(t.stats.max) - sx(t.stats.min)) / t.hist.length;
    const bars = t.hist.map((c, i) => `<rect class="bar" opacity=".3" x="${(sx(t.stats.min) + i * bw).toFixed(1)}" y="${(H - pad - (c / max) * (H - 2 * pad) * 0.8).toFixed(1)}" width="${Math.max(0.5, bw - 0.6).toFixed(1)}" height="${((c / max) * (H - 2 * pad) * 0.8).toFixed(1)}"/>`).join("");
    const path = d3.line()(xs.map((x, i) => [sx(x), sy(ys[i])]));
    return `<svg class="plot" viewBox="0 0 ${W} ${H}">${bars}<line class="axis" x1="${pad}" x2="${W - pad}" y1="${sy(0)}" y2="${sy(0)}"/>` +
      `<line class="axis" x1="${sx(0)}" x2="${sx(0)}" y1="${pad}" y2="${H - pad}"/><path class="curve" d="${path}"/>` +
      `<text x="${pad}" y="${H - 3}">${fmtNum(lo)}</text><text x="${W - pad}" y="${H - 3}" text-anchor="end">${fmtNum(hi)}</text></svg>` +
      `<div class="cap"><span>曲線: 関数の形　棒: 入力値の分布</span></div>`;
  }

  function axisLabels(size) {
    for (const labels of Object.values(DATA.meta.axis_labels || {})) if (labels.length === size) return labels;
    return null;
  }
  function attentionHtml(n) {
    const t = tensors.get(n.attention);
    const p = t?.previews[0];
    if (!p) return "";
    const [count, h, w] = dimsOf(p);
    const head = clamp(state.head, 0, count - 1);
    let html = `<h3>Attention の重み</h3><div class="mono">${esc(fmtShape(t.shape))}　行 = 注目する側 / 列 = 参照される側</div>`;
    if (count > 1) {
      html += `<div class="maps">` + p.channels.map((c, mi) => `<figure class="${mi === head ? "on" : ""}" data-head="${mi}"><img src="${heat(n.attention, 0, mi, "attention")}" alt=""><figcaption>head ${c}</figcaption></figure>`).join("") + `</div>`;
    }
    html += `<img class="heat" style="aspect-ratio:${(w / h).toFixed(3)}" src="${heat(n.attention, 0, head, "attention")}" alt="">`;
    html += `<div class="cap"><span>${h}×${w} に縮小・先頭バッチ</span><span>${fmtNum(p.lo)} … ${fmtNum(p.hi)}</span></div>`;
    // Strongest links between positions: line width = weight.
    const bytes = bytesOf(p), off = head * h * w, rows = Math.min(h, 24), cols = Math.min(w, 24);
    const W = 330, H = 190, top = 14, y0 = (i, m) => top + ((H - top - 6) * (i + 0.5)) / m;
    const labels = axisLabels(t.shape[t.shape.length - 1]);
    let lines = "";
    for (let r = 0; r < rows; r++) {
      const rr = Math.floor((r * h) / rows);
      const row = d3.range(cols).map((c) => bytes[off + rr * w + Math.floor((c * w) / cols)] / 255);
      const cut = [...row].sort((a, b) => b - a)[Math.min(2, cols - 1)];
      for (let c = 0; c < cols; c++) {
        const v = row[c];
        if (v > cut && v > 0.05) lines += `<line x1="70" y1="${y0(r, rows).toFixed(1)}" x2="${W - 70}" y2="${y0(c, cols).toFixed(1)}" stroke-width="${(v * 2.4).toFixed(2)}" opacity="${(v * 0.75).toFixed(2)}"/>`;
      }
    }
    const dots = (x, m, anchorSide) => d3.range(m).map((i) => `<circle cx="${x}" cy="${y0(i, m).toFixed(1)}" r="1.8"/>` +
      (labels && m <= 24 ? `<text x="${x + (anchorSide === "end" ? -6 : 6)}" y="${(y0(i, m) + 3).toFixed(1)}" text-anchor="${anchorSide}">${esc(clip(String(labels[Math.floor((i * labels.length) / m)]), 9))}</text>` : "")).join("");
    html += `<svg class="plot links" viewBox="0 0 ${W} ${H}"><text x="70" y="9" text-anchor="middle">注目する側</text><text x="${W - 70}" y="9" text-anchor="middle">参照される側</text>${lines}${dots(70, rows, "end")}${dots(W - 70, cols, "start")}</svg>`;
    return html + `<div class="cap"><span>線の太さ = 重み (強い結びつきのみ表示)</span></div>`;
  }

  function specialHtml(n) {
    const tin = n.inputs[0], tout = n.outputs[0];
    let html = "";
    if (n.attention != null) html += attentionHtml(n);
    if (n.weight != null && heat(tin) && heat(tout)) {
      const wt = tensors.get(n.weight);
      html += `<h3>入力 → 重み → 出力</h3><div class="row3"><div><img src="${heat(tin)}" alt=""><small>入力</small></div><span>→</span>` +
        `<div><img src="${heat(n.weight)}" alt=""><small>重み ${esc(fmtShape(wt.shape))}</small></div><span>→</span><div><img src="${heat(tout)}" alt=""><small>出力</small></div></div>`;
    }
    if (n.category === "activation") html += `<h3>活性化関数</h3>` + curveSvg(n);
    if ((n.category === "norm" || n.category === "activation") && tensors.get(tin)?.stats && tensors.get(tout)?.stats) {
      html += `<h3>分布の変化</h3><div class="pair">${histSvg(tensors.get(tin), "入力")}${histSvg(tensors.get(tout), "出力")}</div>`;
    }
    if (n.category === "shape" && tin != null && tout != null) {
      html += `<h3>形状の変化</h3><div class="bigshape">${esc(fmtShape(shapeOf(tin)))}<em>→</em>${esc(fmtShape(shapeOf(tout)))}</div>`;
    }
    return html;
  }

  function renderDetail() {
    const n = nodes.get(state.selected);
    if (!n) {
      detail.innerHTML = `<div class="empty">ノードをクリックすると、<br>形状・統計量・値の中身をここに表示します。</div>`;
      return;
    }
    const kids = kidsOf(n.id).length;
    const rows = [["型", esc(n.type)]];
    if (n.params) rows.push(["パラメータ", `${n.params.toLocaleString()} (${fmtCount(n.params)})`]);
    if (n.kind === "repeat") rows.push(["繰り返し", `${n.count} 回`]);
    if (kids) rows.push(["内部", `${kids} 要素`]);
    if (n.attrs.config) rows.push(["設定", esc(n.attrs.config)]);
    if (n.attrs.args) rows.push(["引数", esc(n.attrs.args)]);
    if (n.attrs.dims) rows.push(["軸", esc(n.attrs.dims)]);
    let html = `<h2>${esc(n.name)}</h2><div class="sub">${esc(n.path)}</div>` +
      `<span class="tag" style="background:${color(n.category)}">${CATS[n.category] || esc(n.category)}</span>` +
      `<h3>概要</h3><table>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}</table>`;
    html += specialHtml(n);
    const list = (title, ids) => (ids.length ? `<h3>${title} (${ids.length})</h3>` + ids.slice(0, 6).map((tid) => tensorCard(tid)).join("") : "");
    html += list("出力", n.outputs) + list("入力", n.inputs);
    detail.innerHTML = html;
    detail.querySelectorAll("[data-head]").forEach((el) => el.addEventListener("click", () => { state.head = +el.dataset.head; renderDetail(); }));
    detail.parentElement.scrollTop = 0;
  }

  // ---------------------------------------------------------------- toolbar
  function syncBar() {
    document.querySelectorAll("#mode button").forEach((b) => b.classList.toggle("on", b.dataset.mode === state.mode));
    document.getElementById("depth").textContent = `深さ ${state.depth}`;
    document.getElementById("play").textContent = state.playing ? "⏸ 流れ" : "▶ 流れ";
    document.getElementById("play").style.visibility = state.mode === "flow" ? "hidden" : "visible";
    svg.classed("playing", state.playing && state.mode !== "flow");
  }

  function saveSvg() {
    if (!model) return;
    const clone = svg.node().cloneNode(true);
    clone.querySelectorAll(".flowdash, .hit").forEach((el) => el.remove());
    clone.querySelector(".viewport").removeAttribute("transform");
    const pad = 24, W = Math.ceil(model.w + pad * 2), H = Math.ceil(model.h + pad * 2);
    clone.removeAttribute("class");
    clone.removeAttribute("id");
    clone.setAttribute("xmlns:xlink", XLINK);
    clone.setAttribute("viewBox", `${-pad} ${-pad} ${W} ${H}`);
    clone.setAttribute("width", W);
    clone.setAttribute("height", H);
    const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    for (const [k, v] of Object.entries({ x: -pad, y: -pad, width: W, height: H, fill: "var(--bg)" })) bg.setAttribute(k, v);
    clone.insertBefore(bg, clone.querySelector(".viewport"));
    const styles = getComputedStyle(rootEl);
    const text = new XMLSerializer().serializeToString(clone).replace(/var\((--[\w-]+)\)/g, (_, name) => styles.getPropertyValue(name).trim());
    const blob = new Blob(['<?xml version="1.0" encoding="UTF-8"?>\n', text], { type: "image/svg+xml" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${DATA.meta.model || "model"}_${state.mode}.svg`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 2000);
  }

  function setTheme(theme) {
    rootEl.dataset.theme = theme;
    try { localStorage.setItem("scs-theme", theme); } catch (err) { /* storage may be unavailable */ }
  }

  function init() {
    let theme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    try { theme = localStorage.getItem("scs-theme") || theme; } catch (err) { /* ignore */ }
    rootEl.dataset.theme = theme;
    const top = kidsOf(-1).filter((n) => n.kind === "module");
    const params = top.reduce((sum, n) => sum + n.params, 0);
    document.getElementById("summary").textContent = `${DATA.nodes.length} nodes · ${fmtCount(params)} params · torch ${DATA.meta.torch || "?"}`;
    const used = new Set(DATA.nodes.map((n) => n.category));
    document.getElementById("legend").innerHTML = Object.keys(CATS).filter((c) => used.has(c)).map((c) => `<span><i style="background:${color(c)}"></i>${CATS[c]}</span>`).join("");
    document.querySelectorAll("#mode button").forEach((b) => b.addEventListener("click", () => { state.mode = b.dataset.mode; relayout({ refit: true, animate: false }); }));
    document.getElementById("deeper").addEventListener("click", () => { applyDepth(state.depth + 1); relayout({ refit: true }); });
    document.getElementById("shallower").addEventListener("click", () => { applyDepth(state.depth - 1); relayout({ refit: true }); });
    document.getElementById("play").addEventListener("click", () => { state.playing = !state.playing; syncBar(); });
    document.getElementById("fit").addEventListener("click", () => fit(true));
    document.getElementById("theme").addEventListener("click", () => setTheme(rootEl.dataset.theme === "dark" ? "light" : "dark"));
    document.getElementById("save").addEventListener("click", saveSvg);
    svg.on("click", () => select(null));
    window.addEventListener("resize", () => fit(false));
    applyDepth(state.depth);
    renderDetail();
    return relayout({ refit: true, animate: false });
  }

  window.scs = { state, saveSvg, relayout, toggle, select, focus, applyDepth, ready: null };
  window.scs.ready = Promise.resolve().then(init);
})();
