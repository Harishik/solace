import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const getJSON = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
const sci = (x) => (x == null ? "—" : x === 0 ? "0" : Number(x).toExponential(1));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (n) => Number(n).toLocaleString("en-US");

const RE_M = 6371000;
const KIND_HEX = ["#7dd3fc", "#fbbf24", "#fb7185"];
const KIND_NAME = ["Payload", "Rocket body", "Debris"];
const TEX = "/static/textures/";

// ---------------------------------------------------------------------------------------------
// Shared renderer: one WebGL canvas, one label layer; each 3-D view brings its scene and camera.
// ---------------------------------------------------------------------------------------------
const canvas = $("#scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
const loader = new THREE.TextureLoader();
const tex = (f, srgb = true) => { const t = loader.load(TEX + f); if (srgb) t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t; };

const views = {};
let active = "globe";
const tooltip = $("#tooltip");

// Each 3-D view gets its own label layer so labels never leak between views.
function makeLabels() {
  const el = document.createElement("div"); el.style.position = "absolute"; el.style.inset = "0";
  $("#labels").appendChild(el);
  const r = new CSS2DRenderer({ element: el }); r.setSize(innerWidth, innerHeight); return r;
}

function makeComposer(scene, camera, bloom) {
  const c = new EffectComposer(renderer);
  c.addPass(new RenderPass(scene, camera));
  c.addPass(new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), ...bloom));
  c.addPass(new OutputPass());
  return c;
}

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  for (const v of Object.values(views)) {
    v.labels.setSize(innerWidth, innerHeight);
    v.camera.aspect = innerWidth / innerHeight; v.camera.updateProjectionMatrix();
    v.composer.setSize(innerWidth, innerHeight);
  }
  if (active === "sky") drawSky();
}
addEventListener("resize", resize);

function loop(t) {
  requestAnimationFrame(loop);
  const v = views[active];
  if (!v) return;
  v.update?.(t);
  v.controls.update();
  v.composer.render();
  v.labels.render(v.scene, v.camera);
}

// Soft circular sprite for glows.
function glowTexture(inner = "rgba(255,255,255,1)", outer = "rgba(255,255,255,0)") {
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const g = c.getContext("2d"), grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, inner); grd.addColorStop(0.25, inner.replace(/[\d.]+\)$/, "0.45)")); grd.addColorStop(1, outer);
  g.fillStyle = grd; g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

// Screen-space picking over a point cloud; `occluder` is a sphere radius at the origin.
function pickPoints(v, ev, positions, count, visibleFn, occluder = 0, maxPx = 10) {
  const rect = canvas.getBoundingClientRect(), mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  const cam = v.camera, p = new THREE.Vector3(), cp = cam.position;
  let best = -1, bestD = maxPx * maxPx;
  for (let i = 0; i < count; i++) {
    if (!visibleFn(i)) continue;
    p.set(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]);
    if (occluder) {  // hidden if the camera->point segment enters the sphere first
      const dx = p.x - cp.x, dy = p.y - cp.y, dz = p.z - cp.z, L = Math.hypot(dx, dy, dz);
      const b = (cp.x * dx + cp.y * dy + cp.z * dz) / L, c = cp.lengthSq() - occluder * occluder;
      const disc = b * b - c;
      if (disc > 0 && -b - Math.sqrt(disc) < L && -b - Math.sqrt(disc) > 0) continue;
    }
    p.project(cam);
    if (p.z > 1) continue;
    const sx = (p.x + 1) / 2 * rect.width, sy = (1 - p.y) / 2 * rect.height;
    const d = (sx - mx) ** 2 + (sy - my) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

function showTip(ev, html) {
  if (!html) { tooltip.hidden = true; return; }
  tooltip.innerHTML = html; tooltip.hidden = false;
  tooltip.style.left = `${ev.clientX + 14}px`; tooltip.style.top = `${ev.clientY + 14}px`;
}

// ---------- tabs & clock ----------
$$(".tab").forEach((b) => b.addEventListener("click", () => {
  $$(".tab").forEach((x) => x.classList.toggle("active", x === b));
  $$(".view").forEach((s) => s.classList.toggle("active", s.id === `view-${b.dataset.view}`));
  active = b.dataset.view;
  for (const [k, v] of Object.entries(views)) { v.controls.enabled = k === active; v.labels.domElement.style.display = k === active ? "" : "none"; }
  canvas.style.display = $("#labels").style.display = active === "sky" ? "none" : "block";
  tooltip.hidden = true;
  start(active);
}));
const started = {};
function start(view) {
  if (started[view]) { if (view === "sky") drawSky(); return; }
  started[view] = true;
  ({ globe: initGlobe, neo: initNeo, sky: initSky })[view]();
}

let simTime = Date.now(), simRate = 1;
setInterval(() => {
  $("#clock").textContent = new Date(simTime).toISOString().slice(0, 19).replace("T", "  ") + " UTC";
  const m = $("#mode");
  m.className = `chip ${simRate === 1 ? "live" : "sim"}`;
  m.innerHTML = `<i></i>${simRate === 1 ? "LIVE" : `SIM ${simRate}×`}`;
}, 250);

// Sun direction in Earth-fixed coordinates (low-precision solar ephemeris, ~0.01 deg).
function sunEcef(ms) {
  const n = ms / 86400000 + 2440587.5 - 2451545.0, rad = Math.PI / 180;
  const L = (280.46 + 0.9856474 * n) % 360, g = (357.528 + 0.9856003 * n) % 360 * rad;
  const lam = (L + 1.915 * Math.sin(g) + 0.02 * Math.sin(2 * g)) * rad, eps = (23.439 - 4e-7 * n) * rad;
  const ra = Math.atan2(Math.cos(eps) * Math.sin(lam), Math.cos(lam)), dec = Math.asin(Math.sin(eps) * Math.sin(lam));
  const gmst = (280.46061837 + 360.98564736629 * n) % 360 * rad, lon = ra - gmst;
  return [Math.cos(dec) * Math.cos(lon), Math.cos(dec) * Math.sin(lon), Math.sin(dec)];
}
// Scene axes: ECEF (x, y, z) -> (x, z, -y), so scene y is the Earth's axis.

// =============================================================================================
// A. Satellites & debris
// =============================================================================================
let G, catalog, prevPos, nextPos, frameAt = 0, ws, assetIds = new Set(), kindOn = [1, 1, 1];

async function initGlobe() {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, innerWidth / innerHeight, 0.01, 2000);
  camera.position.set(-2.2, 1.4, 3.6).multiplyScalar(1.05);
  const controls = new OrbitControls(camera, canvas);
  Object.assign(controls, { enableDamping: true, dampingFactor: 0.06, minDistance: 1.25, maxDistance: 22,
    autoRotate: true, autoRotateSpeed: 0.25, rotateSpeed: 0.5, zoomSpeed: 0.8 });
  controls.addEventListener("start", () => { controls.autoRotate = false; });

  // Stars
  const sky = new THREE.Mesh(new THREE.SphereGeometry(900, 48, 32),
    new THREE.MeshBasicMaterial({ map: tex("night_sky.png"), side: THREE.BackSide, color: 0x6b7280 }));
  scene.add(sky);

  // Earth: day/night blend with city lights, ocean glint and a limb glow.
  const sun = new THREE.Vector3(1, 0, 0);
  const earthMat = new THREE.ShaderMaterial({
    uniforms: { dayMap: { value: tex("earth_atmos_2048.jpg") }, nightMap: { value: tex("earth_lights_2048.png") },
      specMap: { value: tex("earth_specular_2048.jpg", false) }, sunDir: { value: sun } },
    vertexShader: `varying vec2 vUv; varying vec3 vN; varying vec3 vW;
      void main() { vUv = uv; vN = normalize(mat3(modelMatrix) * normal); vec4 w = modelMatrix * vec4(position, 1.0);
        vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `uniform sampler2D dayMap, nightMap, specMap; uniform vec3 sunDir;
      varying vec2 vUv; varying vec3 vN; varying vec3 vW;
      void main() {
        vec3 n = normalize(vN), v = normalize(cameraPosition - vW);
        float c = dot(n, sunDir), day = smoothstep(-0.12, 0.22, c);
        vec3 dayCol = texture2D(dayMap, vUv).rgb * (0.08 + 1.05 * max(c, 0.0));
        vec3 nightCol = texture2D(nightMap, vUv).rgb * vec3(1.0, 0.78, 0.5) * 1.7;
        vec3 col = mix(nightCol, dayCol, day);
        vec3 h = normalize(sunDir + v);
        col += vec3(1.0, 0.92, 0.8) * pow(max(dot(n, h), 0.0), 120.0) * texture2D(specMap, vUv).r * 0.28 * day;
        float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
        col += vec3(0.32, 0.6, 1.0) * rim * (0.08 + 0.75 * smoothstep(-0.3, 0.4, c));
        float twilight = exp(-pow(c / 0.12, 2.0)) * 0.25;
        col += vec3(1.0, 0.45, 0.2) * twilight * (1.0 - day * 0.6);
        gl_FragColor = vec4(col, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  });
  const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 128, 96), earthMat);
  scene.add(earth);

  const sunLight = new THREE.DirectionalLight(0xffffff, 2.2);
  scene.add(sunLight, new THREE.AmbientLight(0x334155, 0.25));
  const clouds = new THREE.Mesh(new THREE.SphereGeometry(1.006, 96, 64), new THREE.MeshLambertMaterial({
    map: tex("earth_clouds_1024.png"), transparent: true, opacity: 0.85, depthWrite: false }));
  scene.add(clouds);

  const atmo = new THREE.Mesh(new THREE.SphereGeometry(1.045, 96, 64), new THREE.ShaderMaterial({
    uniforms: { sunDir: { value: sun } }, side: THREE.BackSide, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    vertexShader: `varying vec3 vN; varying vec3 vW; void main() { vN = normalize(mat3(modelMatrix) * normal);
      vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `uniform vec3 sunDir; varying vec3 vN; varying vec3 vW;
      void main() { vec3 v = normalize(cameraPosition - vW); float f = pow(clamp(-dot(vN, v) / 0.3, 0.0, 1.0), 2.2);
        float lit = 0.15 + 0.85 * smoothstep(-0.35, 0.3, dot(normalize(vW), sunDir));
        gl_FragColor = vec4(vec3(0.3, 0.62, 1.0) * f * 1.6 * lit, 1.0); }`,
  }));
  scene.add(atmo);

  // GEO belt guide.
  const geo = new THREE.Mesh(new THREE.RingGeometry(6.6, 6.63, 256), new THREE.MeshBasicMaterial({
    color: 0x7dd3fc, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false }));
  geo.rotation.x = -Math.PI / 2;
  scene.add(geo);

  const composer = makeComposer(scene, camera, [0.45, 0.45, 0.9]);
  G = views.globe = { scene, camera, controls, composer, labels: makeLabels(), sun, sunLight, clouds, update: updateGlobe };
  resize();
  requestAnimationFrame(loop);

  catalog = await getJSON("/api/catalog");
  assetIds = new Set(catalog.assets);
  const n = catalog.ids.length, counts = [0, 0, 0];
  catalog.kinds.forEach((k) => counts[k]++);
  $("#cat-total").textContent = num(n);
  $("#cat-status").textContent = "Public catalog · SGP4 · live positions";
  counts.forEach((c, k) => { $(`#n${k}`).textContent = num(c); });

  const geom = new THREE.BufferGeometry();
  G.pos = new Float32Array(3 * n);
  geom.setAttribute("position", new THREE.BufferAttribute(G.pos, 3).setUsage(THREE.DynamicDrawUsage));
  geom.setAttribute("kind", new THREE.BufferAttribute(new Float32Array(catalog.kinds), 1));
  geom.setAttribute("asset", new THREE.BufferAttribute(new Float32Array(catalog.ids.map((id) => (assetIds.has(id) ? 1 : 0))), 1));
  geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e4);
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { colors: { value: KIND_HEX.map((h) => new THREE.Color(h)) }, show: { value: new THREE.Vector3(1, 1, 1) },
      px: { value: renderer.getPixelRatio() }, time: { value: 0 }, selected: { value: -1 }, hovered: { value: -1 } },
    vertexShader: `attribute float kind; attribute float asset; uniform vec3 show; uniform float px, time, selected, hovered;
      varying vec3 vC; varying float vA, vSel;
      uniform vec3 colors[3];
      void main() {
        int k = int(kind + 0.5);
        float on = k == 0 ? show.x : (k == 1 ? show.y : show.z);
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        float id = float(gl_VertexID);
        vSel = (id == selected || id == hovered) ? 1.0 : 0.0;
        vA = asset;
        vC = asset > 0.5 ? vec3(1.0, 0.9, 0.55) : colors[k];
        float base = k == 0 ? 1.9 : (k == 1 ? 2.3 : 1.6);
        float s = base * clamp(4.5 / -mv.z, 0.6, 3.2);
        if (asset > 0.5) s = 9.0 + 2.0 * sin(time * 3.0);
        if (vSel > 0.5) s = max(s * 2.6, 10.0);
        bool dead = dot(position, position) < 0.01 || (on < 0.5 && asset < 0.5);
        gl_PointSize = dead ? 0.0 : s * px;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `varying vec3 vC; varying float vA, vSel;
      void main() {
        float d = length(gl_PointCoord - 0.5) * 2.0;
        if (d > 1.0) discard;
        float core = smoothstep(1.0, 0.35, d);
        float a = core;
        vec3 c = vC * (0.85 + 0.35 * smoothstep(0.5, 0.0, d));
        if (vA > 0.5 || vSel > 0.5) { float ring = smoothstep(0.12, 0.0, abs(d - 0.8)); a = max(smoothstep(0.45, 0.2, d), ring); c = vC * 1.6; }
        gl_FragColor = vec4(c, a * 0.85);
      }`,
  });
  G.points = new THREE.Points(geom, mat);
  G.points.renderOrder = 2;
  scene.add(G.points);

  // Labels for protected assets.
  G.assetLabels = catalog.assets.map((id) => {
    const i = catalog.ids.indexOf(id), el = document.createElement("div");
    el.className = "lbl asset-lbl"; el.textContent = catalog.names[i] ? catalog.names[i].replace(/ \(.*\)$/, "") : String(id);
    const o = new CSS2DObject(el); o.center.set(0, 0.5); scene.add(o);
    return { i, o, el };
  }).filter((a) => a.i >= 0);

  G.track = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({
    color: 0xffffff, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false }));
  G.track.visible = false;
  scene.add(G.track);

  $$(".kind").forEach((b) => b.addEventListener("click", () => {
    const k = Number(b.dataset.kind); kindOn[k] = kindOn[k] ? 0 : 1; b.classList.toggle("on", !!kindOn[k]);
    mat.uniforms.show.value.set(...kindOn);
  }));
  canvas.addEventListener("pointermove", (ev) => { if (active === "globe") hoverGlobe(ev); });
  canvas.addEventListener("pointerdown", (ev) => { G.down = [ev.clientX, ev.clientY]; });
  canvas.addEventListener("pointerup", (ev) => {
    if (active !== "globe" || !G.down || Math.hypot(ev.clientX - G.down[0], ev.clientY - G.down[1]) > 4) return;
    const i = pickGlobe(ev);
    if (i >= 0) showObject(catalog.ids[i]);
  });
  $("#obj-close").onclick = () => { $("#obj-card").hidden = true; G.track.visible = false; mat.uniforms.selected.value = -1; };

  connectWS();
  loadSpaceWeather(); setInterval(loadSpaceWeather, 10 * 60 * 1000);
  loadAssets();
}

const visibleSat = (i) => (kindOn[catalog.kinds[i]] || assetIds.has(catalog.ids[i])) && (G.pos[3 * i] || G.pos[3 * i + 1]);
const pickGlobe = (ev) => pickPoints(G, ev, G.pos, catalog.ids.length, visibleSat, 1.0, 9);
let lastHover = 0;
function hoverGlobe(ev) {
  if (!catalog || !nextPos || performance.now() - lastHover < 40) return;
  lastHover = performance.now();
  const i = pickGlobe(ev);
  G.points.material.uniforms.hovered.value = i;
  canvas.style.cursor = i >= 0 ? "pointer" : "grab";
  if (i < 0) return showTip(ev, null);
  const r = Math.hypot(G.pos[3 * i], G.pos[3 * i + 1], G.pos[3 * i + 2]);
  showTip(ev, `<b>${esc(catalog.names[i] || catalog.ids[i])}</b><span class="muted">${KIND_NAME[catalog.kinds[i]]} · NORAD ${catalog.ids[i]} · ${num(Math.round((r - 1) * RE_M / 1000))} km</span>`);
}

function updateGlobe(t) {
  const s = sunEcef(simTime);
  G.sun.set(s[0], s[2], -s[1]);
  G.sunLight.position.copy(G.sun).multiplyScalar(10);
  G.clouds.rotation.y += 0.00002;
  if (!G.points) return;
  G.points.material.uniforms.time.value = t / 1000;
  if (nextPos) {
    // Frames arrive once a second: blend the last two along the arc (direction and radius).
    const a = Math.min((performance.now() - frameAt) / 1000, 1), n = catalog.ids.length, P = G.pos;
    for (let i = 0; i < n; i++) {
      const k = 3 * i, x1 = nextPos[k], y1 = nextPos[k + 1], z1 = nextPos[k + 2];
      if (x1 === 0 && y1 === 0) { P[k] = P[k + 1] = P[k + 2] = 0; continue; }
      const x0 = prevPos[k], y0 = prevPos[k + 1], z0 = prevPos[k + 2];
      const r0 = Math.hypot(x0, y0, z0) || Math.hypot(x1, y1, z1), r1 = Math.hypot(x1, y1, z1);
      let x = x0 + (x1 - x0) * a, y = y0 + (y1 - y0) * a, z = z0 + (z1 - z0) * a;
      const f = (r0 + (r1 - r0) * a) / (Math.hypot(x, y, z) || 1) / RE_M;
      P[k] = x * f; P[k + 1] = z * f; P[k + 2] = -y * f;
    }
    G.points.geometry.attributes.position.needsUpdate = true;
    const camDir = G.camera.position.clone().normalize();
    for (const L of G.assetLabels) {
      const k = 3 * L.i;
      L.o.position.set(P[k], P[k + 1], P[k + 2]);
      const behind = L.o.position.dot(camDir) < 0 && L.o.position.length() < 1.2;
      L.el.style.opacity = behind ? 0 : 1;
    }
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
    simRate = Number(b.dataset.rate);
    ws.send(JSON.stringify({ rate: simRate }));
  });
  $("#reset").onclick = () => {
    $$(".rate").forEach((x) => x.classList.toggle("active", x.dataset.rate === "1"));
    simRate = 1;
    ws.send(JSON.stringify({ reset: true, rate: 1 }));
  };
}

async function showObject(id) {
  const o = await getJSON(`/api/object/${id}?t=${simTime / 1000}`);
  const i = catalog.ids.indexOf(id), color = assetIds.has(id) ? "#fde68a" : KIND_HEX[catalog.kinds[i]];
  const pts = o.track_ecef_m.filter((p) => p[0] || p[1]).map((p) => new THREE.Vector3(p[0] / RE_M, p[2] / RE_M, -p[1] / RE_M));
  G.track.geometry.dispose();
  const r = 0.0028 * Math.max(1, pts[0].length() / 1.1);
  G.track.geometry = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts, true), 360, r, 8, true);
  G.track.material.color.set(color).multiplyScalar(1.6);
  G.track.visible = true;
  G.points.material.uniforms.selected.value = i;
  $("#obj-card").hidden = false;
  $("#obj-kind").innerHTML = `<i class="dot" style="display:inline-block;background:${color};box-shadow:0 0 8px ${color};margin-right:6px"></i>${o.kind} · NORAD ${o.id}`;
  $("#obj-name").textContent = o.name;
  let html = `<div class="stats">
    <div class="stat"><div class="eyebrow">Perigee</div><div class="v">${num(Math.round(o.perigee_km))}<span class="u">km</span></div></div>
    <div class="stat"><div class="eyebrow">Apogee</div><div class="v">${num(Math.round(o.apogee_km))}<span class="u">km</span></div></div>
    <div class="stat"><div class="eyebrow">Inclination</div><div class="v">${fmt(o.inclination_deg, 2)}<span class="u">°</span></div></div>
    <div class="stat"><div class="eyebrow">Period</div><div class="v">${fmt(o.period_min, 1)}<span class="u">min</span></div></div></div>`;
  if (o.asset) html += eventsTable(o.asset);
  $("#obj-body").innerHTML = html;
}

function pcClass(pc) { return pc >= 1e-4 ? "hi" : pc >= 1e-6 ? "mid" : "ok"; }

function eventsTable(a) {
  if (!a.events || !a.events.length) return `<p class="note">No approaches under 10 km in the next 3 days.</p>`;
  return `<table><tr><th>TCA (UTC)</th><th>Object</th><th class="num">Miss km</th><th class="num">Pc</th></tr>` +
    a.events.slice(0, 8).map((e) => `<tr>
      <td class="mono">${e.tca_utc.slice(5, 16).replace("T", " ")}</td>
      <td>${esc(e.secondary)}${e.secondary_is_debris ? ' <span class="pill hi">DEB</span>' : ""}</td>
      <td class="num">${fmt(e.miss_km, 2)}</td>
      <td class="num"><span class="pill ${pcClass(e.pc)}">${sci(e.pc)}</span></td></tr>`).join("") +
    `</table><p class="note">${a.covariance === "spacebook"
      ? "Collision probability from Spacebook synthetic covariance."
      : "Collision probability from assumed TLE covariance (Spacebook covariance loading)."}</p>`;
}

let assetPolls = 0;
async function loadAssets() {
  const list = await getJSON("/api/assets");
  const screening = !list.length || list.some((a) => a.status === "screening");
  if ((screening || list.some((a) => a.status === "ok" && a.covariance !== "spacebook")) && assetPolls++ < 100) setTimeout(loadAssets, 3000);
  if (!list.length) { $("#assets").textContent = "Screening…"; return; }
  $("#assets").innerHTML = list.map((a) => `<div class="asset" data-id="${a.id}">
      <div><div class="an">${esc(a.name)}</div><div class="sub mono">NORAD ${a.id}</div></div>
      <div class="right">${a.status !== "ok" ? `<span class="muted small">${esc(a.status)}…</span>` :
        `<span class="pill ${pcClass(a.max_pc)}">Pc ${sci(a.max_pc)}</span>
         <div class="sub">${a.events.length} approach${a.events.length === 1 ? "" : "es"} &lt; 10 km</div>`}</div>
    </div>`).join("");
  $$("#assets .asset").forEach((el) => el.onclick = () => {
    const id = Number(el.dataset.id), i = catalog.ids.indexOf(id);
    showObject(id);
    if (i >= 0) flyTo(new THREE.Vector3(G.pos[3 * i], G.pos[3 * i + 1], G.pos[3 * i + 2]));
  });
}

function flyTo(target) {
  if (!target.lengthSq()) return;
  const from = G.camera.position.clone(), to = target.clone().normalize().multiplyScalar(Math.max(target.length() * 1.9, 2.4));
  G.controls.autoRotate = false;
  const t0 = performance.now();
  (function step() {
    const a = Math.min((performance.now() - t0) / 1400, 1), e = a < 0.5 ? 2 * a * a : 1 - (-2 * a + 2) ** 2 / 2;
    G.camera.position.lerpVectors(from, to, e).setLength(from.length() + (to.length() - from.length()) * e);
    if (a < 1) requestAnimationFrame(step);
  })();
}

async function loadSpaceWeather() {
  try {
    const sw = await getJSON("/api/spaceweather");
    const kp = sw.kp && sw.kp.value, f = sw.f107 && sw.f107.value;
    const storm = kp >= 5 ? `<span class="pill hi">G${Math.min(5, Math.floor(kp) - 4)} storm</span>` : kp >= 4 ? `<span class="pill mid">active</span>` : `<span class="pill ok">quiet</span>`;
    const fpct = Math.max(0, Math.min(1, (f - 65) / (250 - 65))) * 100;
    const kpCells = Array.from({ length: 9 }, (_, j) => `<i style="background:${j < Math.round(kp) ? (j >= 5 ? "#fb7185" : j >= 4 ? "#fbbf24" : "#34d399") : ""}"></i>`).join("");
    $("#sw").innerHTML = `<div class="sw-grid">
      <div><div class="big">${fmt(f, 0)}</div><div class="muted small">F10.7 solar flux, sfu</div>
        <div class="meter"><i style="width:${fpct}%;background:linear-gradient(90deg,#7dd3fc,#fbbf24)"></i></div></div>
      <div><div class="big">${fmt(kp, 2)} ${storm}</div><div class="muted small">Kp geomagnetic index</div><div class="kpbar">${kpCells}</div></div></div>
      <p class="note">Solar flux heats the thermosphere and raises drag on low orbits. It is the main driver of the re-entry forecasts.</p>`;
  } catch (e) { $("#sw").textContent = "Unavailable"; }
}

// =============================================================================================
// B. Near-Earth asteroids: inner Solar System, 1 au = 10 scene units, ecliptic (x, y, z) -> (x, z, -y)
// =============================================================================================
const AU = 10;
const PLANETS = { Mercury: ["#b8b8b8", 0.09], Venus: ["#f2d29b", 0.14], Earth: ["#5ea8ff", 0.15], Mars: ["#ff7a4d", 0.11], Jupiter: ["#e3c08f", 0.32] };
let N, neoData, nextCA;
const eclip = (p) => new THREE.Vector3(p[0] * AU, p[2] * AU, -p[1] * AU);

async function initNeo() {
  if (!N) {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.01, 3000);
    camera.position.set(6, 19, 30);
    const controls = new OrbitControls(camera, canvas);
    Object.assign(controls, { enableDamping: true, dampingFactor: 0.06, minDistance: 2, maxDistance: 140, autoRotate: true, autoRotateSpeed: 0.18 });
    controls.addEventListener("start", () => { controls.autoRotate = false; });
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1500, 48, 32),
      new THREE.MeshBasicMaterial({ map: tex("night_sky.png"), side: THREE.BackSide, color: 0x3a4150 })));
    const sun = new THREE.Mesh(new THREE.SphereGeometry(0.42, 48, 32), new THREE.MeshBasicMaterial({ color: new THREE.Color(2.0, 1.6, 0.95) }));
    const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture("rgba(255,200,120,1)"), blending: THREE.AdditiveBlending, depthWrite: false }));
    glow.scale.setScalar(2.6);
    scene.add(sun, glow, new THREE.PointLight(0xfff1dc, 900, 0, 2), new THREE.AmbientLight(0x404a60, 0.6));
    for (const r of [1, 2, 3, 4, 5]) {
      const ring = new THREE.Mesh(new THREE.RingGeometry(r * AU - 0.012, r * AU + 0.012, 256),
        new THREE.MeshBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.06, side: THREE.DoubleSide, depthWrite: false }));
      ring.rotation.x = -Math.PI / 2; scene.add(ring);
    }
    const composer = makeComposer(scene, camera, [0.5, 0.4, 0.85]);
    N = views.neo = { scene, camera, controls, composer, labels: makeLabels(), planets: {}, extra: [] };
    resize();
    canvas.addEventListener("pointermove", (ev) => { if (active === "neo") hoverNeo(ev); });
  }
  try { neoData = await getJSON("/api/neo"); } catch (e) { neoData = { status: "Server error; retrying…" }; }
  if (neoData.status !== "ok") {
    $("#neo-next").textContent = neoData.status === "loading" ? "Loading JPL data…" : neoData.status;
    started.neo = false; setTimeout(() => { if (active === "neo") start("neo"); }, 3000); return;
  }
  buildNeo();
  renderNeoTables();
  setInterval(tickCountdown, 1000);
}

function buildNeo() {
  const { scene } = N;
  for (const [name, path] of Object.entries(neoData.planet_paths)) {
    const [hex, r] = PLANETS[name];
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(path.map(eclip)),
      new THREE.LineBasicMaterial({ color: hex, transparent: true, opacity: 0.45 }));
    const body = new THREE.Mesh(new THREE.SphereGeometry(r, 32, 24), new THREE.MeshStandardMaterial({ color: hex, roughness: 0.85, emissive: hex, emissiveIntensity: 0.08 }));
    body.position.copy(eclip(neoData.planets[name]));
    const el = document.createElement("div"); el.className = "lbl planet"; el.textContent = name;
    const lbl = new CSS2DObject(el); lbl.position.copy(body.position); lbl.center.set(-0.15, 0.5);
    scene.add(line, body, lbl);
  }
  const o = neoData.objects, cad = new Set(neoData.close_approaches.map((c) => c.name));
  const n = o.names.length, pos = new Float32Array(3 * n), col = new Float32Array(3 * n), size = new Float32Array(n);
  const grey = new THREE.Color("#9aa9c4"), rose = new THREE.Color("#fb7185");
  o.xyz.forEach((p, i) => {
    const v = eclip(p); pos.set([v.x, v.y, v.z], 3 * i);
    const c = cad.has(o.names[i]) ? rose : grey; col.set([c.r, c.g, c.b], 3 * i);
    size[i] = cad.has(o.names[i]) ? 7 : 3.4;
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.setAttribute("size", new THREE.BufferAttribute(size, 1));
  N.pha = new THREE.Points(g, new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, uniforms: { px: { value: renderer.getPixelRatio() } },
    vertexShader: `attribute vec3 color; attribute float size; uniform float px; varying vec3 vC; varying float vBig;
      void main() { vC = color; vBig = size > 5.0 ? 1.0 : 0.0; vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * px * clamp(25.0 / -mv.z, 0.6, 2.5); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `varying vec3 vC; varying float vBig; void main() { float d = length(gl_PointCoord - 0.5) * 2.0; if (d > 1.0) discard;
      gl_FragColor = vec4(vC * (1.0 + vBig * 0.8), smoothstep(1.0, 0.3, d) * (vBig > 0.5 ? 1.0 : 0.7)); }`,
  }));
  N.phaPos = pos; N.phaNames = o.names; N.phaMoid = o.jpl_moid;
  scene.add(N.pha);
}

let lastNeoHover = 0;
function hoverNeo(ev) {
  if (!N.pha || performance.now() - lastNeoHover < 40) return;
  lastNeoHover = performance.now();
  const i = pickPoints(N, ev, N.phaPos, N.phaNames.length, () => true, 0, 8);
  canvas.style.cursor = i >= 0 ? "pointer" : "grab";
  showTip(ev, i < 0 ? null : `<b>${esc(N.phaNames[i])}</b><span class="muted">Potentially hazardous asteroid · MOID ${fmt(N.phaMoid[i], 4)} au</span>`);
}

function renderNeoTables() {
  const now = Date.now() / 86400000 + 2440587.5;
  const upcoming = neoData.close_approaches.filter((c) => c.jd > now);
  nextCA = upcoming[0];
  $("#neo-cad").innerHTML = `<table><tr><th>Object</th><th>Date (TDB)</th><th class="num">LD</th><th class="num">km/s</th><th class="num" title="Our MOID vs JPL's">MOID au</th></tr>` +
    upcoming.slice(0, 20).map((c) => `<tr class="click" data-name="${esc(c.name)}">
      <td>${esc(c.name)}</td><td class="mono">${c.date.slice(5)}</td><td class="num">${fmt(c.dist_ld, 1)}</td>
      <td class="num">${fmt(c.v_rel_kms, 1)}</td>
      <td class="num" title="JPL: ${fmt(c.jpl_moid_au, 4)}">${fmt(c.moid_au, 4)}</td></tr>`).join("") + "</table>" +
    `<p class="note">LD = lunar distances. MOID is computed here from JPL elements (hover for JPL's value).</p>`;
  $$("#neo-cad tr.click").forEach((tr) => tr.onclick = () => showNeoOrbit(tr.dataset.name));
  $("#neo-sentry").innerHTML = `<table><tr><th>Object</th><th>Years</th><th class="num">Impact prob.</th><th class="num">Palermo</th></tr>` +
    neoData.sentry.slice(0, 10).map((s) => `<tr><td>${esc(s.name)}</td><td class="mono">${esc(s.years)}</td>
      <td class="num">${sci(s.impact_prob)}</td><td class="num">${fmt(s.palermo_cum, 2)}</td></tr>`).join("") +
    `</table><p class="note">Torino scale maximum: ${Math.max(...neoData.sentry.map((s) => s.torino_max))} (0 = no hazard). Palermo below −2: no concern.</p>`;
}

function tickCountdown() {
  if (!nextCA) return;
  const ms = (nextCA.jd - 2440587.5) * 86400000 - Date.now();
  if (ms < 0) { renderNeoTables(); return; }
  const d = Math.floor(ms / 86400000), h = Math.floor(ms / 3600000) % 24, m = Math.floor(ms / 60000) % 60, s = Math.floor(ms / 1000) % 60;
  $("#neo-next").innerHTML = `<div class="hero mono">${d}d ${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}</div>
    <div style="margin-top:6px"><b>${esc(nextCA.name)}</b> <span class="muted">passes at</span> ${fmt(nextCA.dist_ld, 1)} LD
    <span class="muted">(${num(Math.round(nextCA.dist_au * 149597870.7))} km) at</span> ${fmt(nextCA.v_rel_kms, 1)} km/s</div>`;
}

async function showNeoOrbit(name) {
  const o = await getJSON(`/api/neo/orbit?name=${encodeURIComponent(name)}`);
  N.extra.forEach((x) => N.scene.remove(x));
  const line = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(o.path.map(eclip), true), 400, 0.035, 8, true),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(2.0, 0.55, 0.75), transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false }));
  const el = document.createElement("div"); el.className = "lbl cad"; el.textContent = o.name;
  const lbl = new CSS2DObject(el); lbl.position.copy(eclip(o.position));
  const dot = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture("rgba(251,113,133,1)"), blending: THREE.AdditiveBlending, depthWrite: false }));
  dot.position.copy(lbl.position); dot.scale.setScalar(0.9);
  N.extra = [line, lbl, dot];
  N.scene.add(...N.extra);
}

// =============================================================================================
// C. Sky tonight (2-D all-sky chart)
// =============================================================================================
let here = { lat: 28.573, lon: -80.649, label: "Cape Canaveral (default)" }, skyData;

function initSky() {
  $("#geo").onclick = () => navigator.geolocation && navigator.geolocation.getCurrentPosition((p) => {
    here = { lat: p.coords.latitude, lon: p.coords.longitude, label: `${p.coords.latitude.toFixed(2)}°, ${p.coords.longitude.toFixed(2)}°` };
    loadSky();
  }, () => { $("#loc").textContent = "Location denied; using default."; });
  loadSky(); setInterval(loadSky, 15000);
}

async function loadSky() {
  skyData = await getJSON(`/api/sky?lat=${here.lat}&lon=${here.lon}`);
  $("#loc").innerHTML = `${esc(here.label)} · ${skyData.is_night ? '<span class="pill ok">night</span>' : `<span class="pill mid">Sun up ${fmt(skyData.sun_alt, 0)}°</span>`}`;
  $("#sky-bodies").innerHTML = `<table><tr><th>Body</th><th class="num">Altitude</th><th class="num">Azimuth</th></tr>` +
    skyData.bodies.map((b) => `<tr${b.alt < 0 ? ' class="muted"' : ""}><td>${b.name}</td><td class="num">${fmt(b.alt, 1)}°</td><td class="num">${fmt(b.az, 1)}°</td></tr>`).join("") + "</table>";
  $("#sky-passes").innerHTML = skyData.iss_passes.length ? `<table><tr><th>Rise (UTC)</th><th class="num">Max alt</th><th></th></tr>` +
    skyData.iss_passes.map((p) => `<tr><td class="mono">${new Date(p.rise_utc).toLocaleString("en-GB", { timeZone: "UTC", weekday: "short", hour: "2-digit", minute: "2-digit" })}</td>
      <td class="num">${fmt(p.max_alt, 0)}°</td><td>${p.visible ? '<span class="pill mid">naked eye</span>' : `<span class="muted small">${p.sunlit ? "daylight" : "in shadow"}</span>`}</td></tr>`).join("") + "</table>"
    : `<p class="note">No passes above 10°.</p>`;
  drawSky();
}

function drawSky() {
  if (!skyData) return;
  const cv = $("#sky"), dpr = window.devicePixelRatio || 1;
  const size = cv.clientWidth; cv.width = size * dpr; cv.height = size * dpr;
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
  const R = size / 2 - 30, cx = size / 2, cy = size / 2;
  // Zenith at the centre, horizon at the rim, north up, east to the left (as seen looking up).
  const proj = (alt, az) => { const r = (90 - alt) / 90 * R, a = az * Math.PI / 180; return [cx - r * Math.sin(a), cy - r * Math.cos(a)]; };
  const night = skyData.is_night, sunAlt = skyData.sun_alt;
  const twi = Math.max(0, Math.min(1, (sunAlt + 18) / 24));  // 0 = full night, 1 = day
  const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R);
  const mix = (a, b) => a.map((x, i) => Math.round(x + (b[i] - x) * twi));
  const c0 = mix([4, 8, 22], [44, 98, 170]), c1 = mix([12, 22, 48], [98, 156, 214]);
  g.addColorStop(0, `rgb(${c0})`); g.addColorStop(1, `rgb(${c1})`);
  ctx.save(); ctx.shadowColor = "rgba(125,211,252,0.25)"; ctx.shadowBlur = 40;
  ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.fill(); ctx.restore();
  ctx.strokeStyle = "rgba(186,200,230,0.12)"; ctx.lineWidth = 1;
  for (const alt of [30, 60]) { ctx.beginPath(); ctx.arc(cx, cy, (90 - alt) / 90 * R, 0, 2 * Math.PI); ctx.stroke(); }
  for (let az = 0; az < 360; az += 45) { const [x, y] = proj(0, az); ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke(); }
  ctx.strokeStyle = "rgba(186,200,230,0.35)"; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
  ctx.fillStyle = "#94a3b8"; ctx.font = "600 12px Inter, system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  [["N", 0], ["E", 90], ["S", 180], ["W", 270]].forEach(([l, az]) => { const [x, y] = proj(-5.5, az); ctx.fillText(l, x, y); });
  ctx.font = "10px JetBrains Mono, monospace"; ctx.fillStyle = "rgba(148,163,184,0.6)";
  for (const alt of [30, 60]) { const [x, y] = proj(alt, 22); ctx.fillText(`${alt}°`, x, y); }

  for (const s of skyData.stars) {
    if (s.alt < 0) continue;
    const [x, y] = proj(s.alt, s.az), r = Math.max(0.7, 3.4 - 0.6 * s.mag);
    const alpha = (1.05 - s.mag / 6) * (1 - 0.8 * twi);
    if (alpha <= 0.02) continue;
    const sg = ctx.createRadialGradient(x, y, 0, x, y, r * 3);
    sg.addColorStop(0, `rgba(255,255,255,${Math.min(1, alpha)})`); sg.addColorStop(0.35, `rgba(200,220,255,${alpha * 0.35})`); sg.addColorStop(1, "rgba(200,220,255,0)");
    ctx.fillStyle = sg; ctx.beginPath(); ctx.arc(x, y, r * 3, 0, 2 * Math.PI); ctx.fill();
    if (s.name && s.mag < 1.6) { ctx.fillStyle = `rgba(203,213,225,${0.75 * (1 - 0.5 * twi)})`; ctx.font = "10.5px Inter, system-ui"; ctx.fillText(s.name, x, y + 12); }
  }
  for (const p of skyData.iss_passes.slice(0, 3)) {
    ctx.strokeStyle = p.visible ? "rgba(253,230,138,0.95)" : "rgba(253,230,138,0.35)"; ctx.lineWidth = p.visible ? 2 : 1.2;
    ctx.setLineDash([5, 5]); ctx.beginPath();
    let first = true;
    p.track.forEach(([alt, az]) => { if (alt < 0) return; const [x, y] = proj(alt, az); first ? ctx.moveTo(x, y) : ctx.lineTo(x, y); first = false; });
    ctx.stroke(); ctx.setLineDash([]);
  }
  const BC = { Sun: "#ffd27a", Moon: "#eef2f8", Mercury: "#c8c8c8", Venus: "#f7e3ae", Mars: "#ff7a4d", Jupiter: "#e9cfa6", Saturn: "#eadbaa", Uranus: "#a5e6ec", Neptune: "#86a9ff" };
  for (const b of skyData.bodies) {
    if (b.alt < 0) continue;
    const [x, y] = proj(b.alt, b.az), r = b.name === "Sun" ? 11 : b.name === "Moon" ? 9 : 5;
    const bg = ctx.createRadialGradient(x, y, 0, x, y, r * 3.2);
    bg.addColorStop(0, BC[b.name]); bg.addColorStop(0.3, BC[b.name] + "88"); bg.addColorStop(1, BC[b.name] + "00");
    ctx.fillStyle = bg; ctx.beginPath(); ctx.arc(x, y, r * 3.2, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = BC[b.name]; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = "#f1f5f9"; ctx.font = "600 12px Inter, system-ui"; ctx.fillText(b.name, x, y - r - 10);
  }
}

start("globe");
