"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const getJSON = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
const sci = (x) => (x == null ? "—" : x === 0 ? "0" : Number(x).toExponential(1));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------- tabs ----------
const started = {};
$$(".tab").forEach((b) => b.addEventListener("click", () => {
  $$(".tab").forEach((x) => x.classList.toggle("active", x === b));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${b.dataset.view}`));
  start(b.dataset.view);
}));
function start(view) {
  if (started[view]) { if (view === "neo") Plotly.Plots.resize("neo-plot"); return; }
  started[view] = true;
  ({ globe: initGlobe, neo: initNeo, sky: initSky })[view]();
}

let simTime = Date.now();
setInterval(() => { $("#clock").textContent = new Date(simTime).toISOString().slice(0, 19).replace("T", " ") + " UTC"; }, 250);

// =====================================================================
// A. Satellites & debris globe
// =====================================================================
const KIND_COLORS = ["#5cc8ff", "#ffb347", "#ff5c7a"];
let viewer, points, catalog, prevPos, nextPos, frameAt = 0, ws, trackEntity, assetIds = new Set();

async function initGlobe() {
  viewer = new Cesium.Viewer("globe", {
    baseLayer: Cesium.ImageryLayer.fromProviderAsync(
      Cesium.TileMapServiceImageryProvider.fromUrl(Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII"))),
    baseLayerPicker: false, geocoder: false, timeline: false, animation: false, sceneModePicker: false,
    navigationHelpButton: false, homeButton: false, infoBox: false, selectionIndicator: false,
    fullscreenButton: false,
  });
  viewer.scene.globe.enableLighting = false;
  viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(-40, 20, 2.4e7) });
  points = viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());

  catalog = await getJSON("/api/catalog");
  assetIds = new Set(catalog.assets);
  const colors = KIND_COLORS.map((c) => Cesium.Color.fromCssColorString(c).withAlpha(0.85));
  const assetColor = Cesium.Color.fromCssColorString("#ffe45c");
  catalog.ids.forEach((id, i) => {
    const isAsset = assetIds.has(id);
    points.add({
      id: { norad: id, i }, position: Cesium.Cartesian3.ZERO,
      pixelSize: isAsset ? 9 : 2.2, color: isAsset ? assetColor : colors[catalog.kinds[i]],
      outlineColor: Cesium.Color.BLACK, outlineWidth: isAsset ? 1 : 0,
    });
  });
  const counts = [0, 0, 0]; catalog.kinds.forEach((k) => counts[k]++);
  $("#cat-status").textContent = `${catalog.ids.length.toLocaleString()} objects · ${counts[0].toLocaleString()} payloads · ${counts[1].toLocaleString()} rocket bodies · ${counts[2].toLocaleString()} debris`;

  $$("input[data-kind]").forEach((cb) => cb.addEventListener("change", applyFilter));
  connectWS();
  viewer.scene.preRender.addEventListener(interpolate);

  const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
  handler.setInputAction((e) => {
    const p = viewer.scene.pick(e.position);
    if (p && p.id && p.id.norad) showObject(p.id.norad);
  }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

  loadSpaceWeather(); setInterval(loadSpaceWeather, 10 * 60 * 1000);
  loadAssets();
}

function applyFilter() {
  const on = $$("input[data-kind]").map((cb) => cb.checked);
  for (let i = 0; i < points.length; i++) {
    const p = points.get(i);
    p.show = assetIds.has(p.id.norad) || on[catalog.kinds[p.id.i]];
  }
}

function connectWS() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/positions`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (ev) => {
    simTime = new Float64Array(ev.data, 0, 1)[0] * 1000;
    prevPos = nextPos || new Float32Array(ev.data, 8);
    nextPos = new Float32Array(ev.data, 8);
    frameAt = performance.now();
  };
  ws.onclose = () => setTimeout(connectWS, 2000);
  $$(".rate").forEach((b) => b.onclick = () => {
    $$(".rate").forEach((x) => x.classList.toggle("active", x === b));
    ws.send(JSON.stringify({ rate: Number(b.dataset.rate) }));
  });
  $("#reset").onclick = () => {
    $$(".rate").forEach((x) => x.classList.toggle("active", x.dataset.rate === "1"));
    ws.send(JSON.stringify({ reset: true, rate: 1 }));
  };
}

const scratch = new Cesium.Cartesian3();
function interpolate() {
  if (!nextPos) return;
  // Frames arrive once a second; blend the last two so motion is smooth (one-second display lag).
  const a = Math.min((performance.now() - frameAt) / 1000, 1);
  for (let i = 0, n = points.length; i < n; i++) {
    const p = points.get(i);
    if (!p.show) continue;
    const k = 3 * i;
    const x0 = prevPos[k], x1 = nextPos[k];
    if (x1 === 0 && nextPos[k + 1] === 0) { p.show = false; continue; }
    scratch.x = x0 + (x1 - x0) * a;
    scratch.y = prevPos[k + 1] + (nextPos[k + 1] - prevPos[k + 1]) * a;
    scratch.z = prevPos[k + 2] + (nextPos[k + 2] - prevPos[k + 2]) * a;
    p.position = scratch;
  }
}

async function showObject(id) {
  const o = await getJSON(`/api/object/${id}?t=${simTime / 1000}`);
  if (trackEntity) viewer.entities.remove(trackEntity);
  trackEntity = viewer.entities.add({
    polyline: {
      positions: o.track_ecef_m.filter((p) => p[0] || p[1]).map((p) => new Cesium.Cartesian3(p[0], p[1], p[2])),
      width: 1.5, material: Cesium.Color.fromCssColorString(assetIds.has(id) ? "#ffe45c" : "#ffffff").withAlpha(0.7),
    },
  });
  $("#obj-card").hidden = false;
  $("#obj-name").textContent = `${o.name} · ${o.id}`;
  let html = `<table>
    <tr><th>Type</th><td>${o.kind}</td></tr>
    <tr><th>Altitude</th><td>${fmt(o.perigee_km, 0)} – ${fmt(o.apogee_km, 0)} km</td></tr>
    <tr><th>Inclination</th><td>${fmt(o.inclination_deg, 2)}°</td></tr>
    <tr><th>Period</th><td>${fmt(o.period_min, 1)} min</td></tr></table>`;
  if (o.asset) html += eventsTable(o.asset);
  $("#obj-body").innerHTML = html;
}

function pcClass(pc) { return pc >= 1e-4 ? "hi" : pc >= 1e-6 ? "mid" : ""; }

function eventsTable(a) {
  if (!a.events || !a.events.length) return `<p class="muted">No approaches under 10 km in the next 3 days.</p>`;
  return `<table><tr><th>TCA (UTC)</th><th>Object</th><th class="num">Miss km</th><th class="num">Pc</th><th class="num">d<sub>crit</sub></th></tr>` +
    a.events.slice(0, 10).map((e) => `<tr>
      <td>${e.tca_utc.slice(5, 16).replace("T", " ")}</td>
      <td>${esc(e.secondary)}${e.secondary_is_debris ? ' <span class="pill hi">DEB</span>' : ""}</td>
      <td class="num">${fmt(e.miss_km, 2)}</td>
      <td class="num"><span class="pill ${pcClass(e.pc)}">${sci(e.pc)}</span></td>
      <td class="num" title="Critical diameter for a default Whipple shield at this relative speed">${fmt(e.d_crit_cm, 2)} cm</td></tr>`).join("") +
    `</table><p class="muted small">${a.covariance === "spacebook"
      ? "Pc uses Spacebook synthetic covariance (per-object TLE error statistics)."
      : "Pc uses assumed TLE covariance for now; Spacebook covariance is loading."} d<sub>crit</sub>: smallest Al particle that perforates a 1.6 mm / 3.2 mm Whipple shield at this speed.</p>`;
}

let assetPolls = 0;
async function loadAssets() {
  const list = await getJSON("/api/assets");
  const screening = !list.length || list.some((a) => a.status === "screening");
  // Keep refreshing until every asset has been re-scored with Spacebook covariance (or ~5 min).
  if ((screening || list.some((a) => a.status === "ok" && a.covariance !== "spacebook")) && assetPolls++ < 100) {
    setTimeout(loadAssets, 3000);
  }
  if (screening) {
    $("#assets").textContent = `Screening ${list.length ? list.filter((a) => a.status !== "screening").length + "/" + list.length : ""}…`;
    if (!list.length) return;
  }
  $("#assets").innerHTML = list.map((a) => `<div class="asset" data-id="${a.id}">
      <b>${esc(a.name)}</b> <span class="muted">${a.id}</span><br>
      ${a.status !== "ok" ? `<span class="muted">${esc(a.status)}</span>` :
        `${a.events.length} approach${a.events.length === 1 ? "" : "es"} &lt; 10 km · max Pc <span class="pill ${pcClass(a.max_pc)}">${sci(a.max_pc)}</span>
         <span class="muted small">${a.covariance === "spacebook" ? "Spacebook cov." : "assumed cov."}</span>`}
    </div>`).join("");
  $$("#assets .asset").forEach((el) => el.onclick = () => {
    const id = Number(el.dataset.id);
    showObject(id);
    const i = catalog.ids.indexOf(id);
    if (i >= 0 && nextPos) {
      const c = new Cesium.Cartesian3(nextPos[3 * i], nextPos[3 * i + 1], nextPos[3 * i + 2]);
      const carto = Cesium.Cartographic.fromCartesian(c);
      viewer.camera.flyTo({ destination: Cesium.Cartesian3.fromRadians(carto.longitude, carto.latitude, 8e6) });
    }
  });
}

async function loadSpaceWeather() {
  try {
    const sw = await getJSON("/api/spaceweather");
    const kp = sw.kp && sw.kp.value, f = sw.f107 && sw.f107.value;
    const storm = kp >= 5 ? `<span class="pill hi">G${Math.min(5, Math.floor(kp) - 4)} storm</span>` : `<span class="pill">quiet</span>`;
    $("#sw").innerHTML = `<div class="row"><div><div class="big">${fmt(f, 0)}</div><span class="muted small">F10.7 (sfu)</span></div>
      <div style="margin-left:24px"><div class="big">${fmt(kp, 2)}</div><span class="muted small">Kp</span></div>
      <div style="margin-left:auto">${storm}</div></div>
      <p class="muted small">Higher F10.7/Kp heats the thermosphere, raising drag on LEO objects.</p>`;
  } catch (e) { $("#sw").textContent = "Unavailable"; }
}

// =====================================================================
// B. Near-Earth asteroids
// =====================================================================
let neoData, nextCA;
const PLANET_COLORS = { Mercury: "#a0a0a0", Venus: "#e8c27a", Earth: "#5cc8ff", Mars: "#ff7a4d", Jupiter: "#d9b38c" };

async function initNeo() {
  try { neoData = await getJSON("/api/neo"); } catch (e) { neoData = { status: "Server error; retrying…" }; }
  if (neoData.status !== "ok") {
    $("#neo-next").textContent = neoData.status === "loading" ? "Loading JPL data…" : neoData.status;
    started.neo = false; setTimeout(() => start("neo"), 3000); return;
  }
  drawNeo();
  renderNeoTables();
  setInterval(async () => { neoData = await getJSON("/api/neo"); drawNeo(); }, 60 * 1000);
  setInterval(tickCountdown, 1000);
}

function drawNeo(extra = []) {
  const traces = [{ type: "scatter3d", mode: "markers", x: [0], y: [0], z: [0], name: "Sun",
    marker: { size: 8, color: "#ffd84d" }, hoverinfo: "name" }];
  for (const [name, path] of Object.entries(neoData.planet_paths)) {
    traces.push({ type: "scatter3d", mode: "lines", x: path.map((p) => p[0]), y: path.map((p) => p[1]), z: path.map((p) => p[2]),
      line: { color: PLANET_COLORS[name], width: 2 }, name: `${name} orbit`, hoverinfo: "skip", showlegend: false });
    const p = neoData.planets[name];
    traces.push({ type: "scatter3d", mode: "markers+text", x: [p[0]], y: [p[1]], z: [p[2]], text: [name], textposition: "top center",
      textfont: { color: "#dbe4f3", size: 11 }, marker: { size: name === "Jupiter" ? 7 : 5, color: PLANET_COLORS[name] }, name, hoverinfo: "name" });
  }
  const o = neoData.objects;
  const cadNames = new Set(neoData.close_approaches.map((c) => c.name));
  const idx = o.names.map((_, i) => i);
  const sel = (f) => idx.filter(f);
  const pha = sel((i) => !cadNames.has(o.names[i])), cad = sel((i) => cadNames.has(o.names[i]));
  traces.push({ type: "scatter3d", mode: "markers", name: `PHAs (${pha.length})`,
    x: pha.map((i) => o.xyz[i][0]), y: pha.map((i) => o.xyz[i][1]), z: pha.map((i) => o.xyz[i][2]),
    text: pha.map((i) => `${o.names[i]}<br>JPL MOID ${fmt(o.jpl_moid[i], 4)} au`), hoverinfo: "text",
    marker: { size: 1.6, color: "#8a94ab", opacity: 0.6 } });
  traces.push({ type: "scatter3d", mode: "markers", name: "Upcoming close approaches",
    x: cad.map((i) => o.xyz[i][0]), y: cad.map((i) => o.xyz[i][1]), z: cad.map((i) => o.xyz[i][2]),
    text: cad.map((i) => o.names[i]), hoverinfo: "text", marker: { size: 4, color: "#ff5c7a" } });
  traces.push(...extra);
  const axis = { showgrid: false, zeroline: false, showticklabels: false, title: "", range: [-2.2, 2.2], showbackground: false };
  Plotly.react("neo-plot", traces, {
    paper_bgcolor: "#05070d", font: { color: "#dbe4f3" }, margin: { l: 0, r: 0, t: 0, b: 0 },
    legend: { x: 0.01, y: 0.99, bgcolor: "rgba(0,0,0,0)" },
    scene: { xaxis: axis, yaxis: axis, zaxis: { ...axis, range: [-1, 1] }, aspectmode: "manual",
      aspectratio: { x: 1, y: 1, z: 0.45 }, camera: { eye: { x: 0.9, y: -0.9, z: 0.6 } } },
  }, { displaylogo: false, responsive: true });
}

function renderNeoTables() {
  const now = Date.now() / 86400000 + 2440587.5;
  const upcoming = neoData.close_approaches.filter((c) => c.jd > now);
  nextCA = upcoming[0];
  $("#neo-cad").innerHTML = `<table><tr><th>Object</th><th>Date (TDB)</th><th class="num">LD</th><th class="num">km/s</th><th class="num" title="Our MOID vs JPL's">MOID au</th></tr>` +
    upcoming.slice(0, 20).map((c) => `<tr class="click" data-name="${esc(c.name)}">
      <td>${esc(c.name)}</td><td>${c.date.slice(5)}</td><td class="num">${fmt(c.dist_ld, 1)}</td>
      <td class="num">${fmt(c.v_rel_kms, 1)}</td>
      <td class="num" title="JPL: ${fmt(c.jpl_moid_au, 4)}">${fmt(c.moid_au, 4)}</td></tr>`).join("") + "</table>" +
    `<p class="muted small">LD = lunar distances. MOID computed here from JPL elements (hover for JPL's value). Click a row to draw its orbit.</p>`;
  $$("#neo-cad tr.click").forEach((tr) => tr.onclick = () => showNeoOrbit(tr.dataset.name));
  $("#neo-sentry").innerHTML = `<table><tr><th>Object</th><th>Years</th><th class="num">Impact prob.</th><th class="num">Palermo</th><th class="num">Ø km</th></tr>` +
    neoData.sentry.slice(0, 12).map((s) => `<tr><td>${esc(s.name)}</td><td>${esc(s.years)}</td>
      <td class="num">${sci(s.impact_prob)}</td><td class="num">${fmt(s.palermo_cum, 2)}</td><td class="num">${fmt(s.diameter_km, 3)}</td></tr>`).join("") +
    `</table><p class="muted small">Torino scale max: ${Math.max(...neoData.sentry.map((s) => s.torino_max))} (0 = no hazard). Palermo &lt; −2: no concern.</p>`;
}

function tickCountdown() {
  if (!nextCA) return;
  const ms = (nextCA.jd - 2440587.5) * 86400000 - Date.now();
  if (ms < 0) { renderNeoTables(); return; }
  const d = Math.floor(ms / 86400000), h = Math.floor(ms / 3600000) % 24, m = Math.floor(ms / 60000) % 60, s = Math.floor(ms / 1000) % 60;
  $("#neo-next").innerHTML = `<div class="big">${d}d ${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}</div>
    <b>${esc(nextCA.name)}</b> passes at ${fmt(nextCA.dist_ld, 1)} lunar distances (${Math.round(nextCA.dist_au * 149597870.7).toLocaleString()} km), ${fmt(nextCA.v_rel_kms, 1)} km/s.`;
}

async function showNeoOrbit(name) {
  const o = await getJSON(`/api/neo/orbit?name=${encodeURIComponent(name)}`);
  drawNeo([
    { type: "scatter3d", mode: "lines", name: o.name, x: o.path.map((p) => p[0]), y: o.path.map((p) => p[1]), z: o.path.map((p) => p[2]),
      line: { color: "#ff5c7a", width: 4 }, hoverinfo: "name" },
    { type: "scatter3d", mode: "markers", name: `${o.name} now`, x: [o.position[0]], y: [o.position[1]], z: [o.position[2]],
      marker: { size: 6, color: "#ff5c7a" }, hoverinfo: "name", showlegend: false },
  ]);
}

// =====================================================================
// C. Sky tonight
// =====================================================================
let here = { lat: 28.573, lon: -80.649, label: "Cape Canaveral (default)" }, skyData;

function initSky() {
  $("#geo").onclick = () => navigator.geolocation && navigator.geolocation.getCurrentPosition((p) => {
    here = { lat: p.coords.latitude, lon: p.coords.longitude, label: `${p.coords.latitude.toFixed(2)}°, ${p.coords.longitude.toFixed(2)}°` };
    loadSky();
  }, () => { $("#loc").textContent = "Location denied; using default."; });
  window.addEventListener("resize", drawSky);
  loadSky(); setInterval(loadSky, 15000);
}

async function loadSky() {
  skyData = await getJSON(`/api/sky?lat=${here.lat}&lon=${here.lon}`);
  $("#loc").textContent = `${here.label} · ${skyData.is_night ? "night" : "Sun up (" + fmt(skyData.sun_alt, 0) + "°)"}`;
  $("#sky-bodies").innerHTML = `<table><tr><th>Body</th><th class="num">Alt</th><th class="num">Az</th></tr>` +
    skyData.bodies.map((b) => `<tr${b.alt < 0 ? ' class="muted"' : ""}><td>${b.name}</td><td class="num">${fmt(b.alt, 1)}°</td><td class="num">${fmt(b.az, 1)}°</td></tr>`).join("") + "</table>";
  $("#sky-passes").innerHTML = skyData.iss_passes.length ? `<table><tr><th>Rise (UTC)</th><th class="num">Max alt</th><th>Visible</th></tr>` +
    skyData.iss_passes.map((p) => `<tr><td>${new Date(p.rise_utc).toLocaleString("en-GB", { timeZone: "UTC", weekday: "short", hour: "2-digit", minute: "2-digit" })}</td>
      <td class="num">${fmt(p.max_alt, 0)}°</td><td>${p.visible ? '<span class="pill mid">naked eye</span>' : `<span class="muted small">${p.sunlit ? "daylight" : "in shadow"}</span>`}</td></tr>`).join("") + "</table>"
    : `<p class="muted">No passes above 10°.</p>`;
  drawSky();
}

function drawSky() {
  if (!skyData) return;
  const cv = $("#sky"), dpr = window.devicePixelRatio || 1;
  const size = cv.clientWidth; cv.width = size * dpr; cv.height = size * dpr;
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
  const R = size / 2 - 24, cx = size / 2, cy = size / 2;
  // Zenith at the centre, horizon at the rim, north up, east to the left (as seen looking up).
  const proj = (alt, az) => { const r = (90 - alt) / 90 * R, a = az * Math.PI / 180; return [cx - r * Math.sin(a), cy - r * Math.cos(a)]; };
  const night = skyData.is_night;
  const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R);
  g.addColorStop(0, night ? "#050a18" : "#1d3a66"); g.addColorStop(1, night ? "#0d1730" : "#3a6aa8");
  ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.fill();
  ctx.strokeStyle = "rgba(160,180,220,0.18)"; ctx.lineWidth = 1;
  for (const alt of [30, 60]) { ctx.beginPath(); ctx.arc(cx, cy, (90 - alt) / 90 * R, 0, 2 * Math.PI); ctx.stroke(); }
  ctx.fillStyle = "#7f8ba3"; ctx.font = "12px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  [["N", 0], ["E", 90], ["S", 180], ["W", 270]].forEach(([l, az]) => { const [x, y] = proj(-8, az); ctx.fillText(l, x, y); });

  for (const s of skyData.stars) {
    if (s.alt < 0) continue;
    const [x, y] = proj(s.alt, s.az), r = Math.max(0.6, 3.2 - 0.55 * s.mag);
    ctx.fillStyle = night ? `rgba(255,255,255,${Math.min(1, 1.1 - s.mag / 6)})` : "rgba(255,255,255,0.25)";
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill();
    if (s.name && s.mag < 1.6) { ctx.fillStyle = "rgba(200,210,235,0.75)"; ctx.font = "10px system-ui"; ctx.fillText(s.name, x, y + 10); }
  }
  for (const p of skyData.iss_passes.slice(0, 3)) {
    ctx.strokeStyle = p.visible ? "rgba(255,228,92,0.9)" : "rgba(255,228,92,0.35)"; ctx.setLineDash([4, 4]); ctx.beginPath();
    p.track.forEach(([alt, az], k) => { if (alt < 0) return; const [x, y] = proj(alt, az); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke(); ctx.setLineDash([]);
  }
  const BC = { Sun: "#ffd84d", Moon: "#e9edf5", Mercury: "#c8c8c8", Venus: "#f5dfa0", Mars: "#ff7a4d", Jupiter: "#e6c9a0", Saturn: "#e8d9a8", Uranus: "#9fe0e6", Neptune: "#7fa6ff" };
  for (const b of skyData.bodies) {
    if (b.alt < 0) continue;
    const [x, y] = proj(b.alt, b.az), r = b.name === "Sun" || b.name === "Moon" ? 9 : 5;
    ctx.fillStyle = BC[b.name]; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = "#dbe4f3"; ctx.font = "12px system-ui"; ctx.fillText(b.name, x, y - r - 8);
  }
}

start("globe");
