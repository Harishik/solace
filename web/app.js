import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { TYPES, geometries, detailFor, material as modelMaterial } from "./models.js";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const getJSON = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
const sci = (x) => (x == null ? "—" : x === 0 ? "0" : Number(x).toExponential(1));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (n) => Number(n).toLocaleString("en-US");

const RE_M = 6371000;
const OMEGA_E = 7.2921159e-5;  // Earth rotation, rad/s
const KIND_HEX = ["#7dd3fc", "#fbbf24", "#fb7185"];
const KIND_NAME = ["Payload", "Rocket body", "Debris"];
const TEX = "/static/textures/";
const D2R = Math.PI / 180;

// Custom shaders share the renderer's logarithmic depth buffer (needed to fly from orbit-scale views
// down to a 100 m spacecraft without depth artefacts).
const LOG_VS = "#include <common>\n#include <logdepthbuf_pars_vertex>\n";
const LOG_VS_END = "\n#include <logdepthbuf_vertex>\n";
const LOG_FS = "#include <logdepthbuf_pars_fragment>\n";
const LOG_FS_START = "\n#include <logdepthbuf_fragment>\n";

// ---------------------------------------------------------------------------------------------
// Shared renderer: one WebGL canvas; each view brings its scene, camera and label layer.
// ---------------------------------------------------------------------------------------------
const canvas = $("#scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance", logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
const loader = new THREE.TextureLoader();
const tex = (f, srgb = true) => { const t = loader.load(TEX + f); if (srgb) t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t; };

const views = {};
let active = "globe";
const tooltip = $("#tooltip");

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

function glowTexture(inner = "rgba(255,255,255,1)", outer = "rgba(255,255,255,0)") {
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const g = c.getContext("2d"), grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, inner); grd.addColorStop(0.25, inner.replace(/[\d.]+\)$/, "0.45)")); grd.addColorStop(1, outer);
  g.fillStyle = grd; g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

// Screen-space picking over a point set; `occluder` is a sphere radius at the origin.
function pickPoints(v, ev, positions, count, visibleFn, occluder = 0, maxPx = 10) {
  const rect = canvas.getBoundingClientRect(), mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  const cam = v.camera, p = new THREE.Vector3(), cp = cam.position;
  let best = -1, bestD = maxPx * maxPx;
  for (let i = 0; i < count; i++) {
    if (!visibleFn(i)) continue;
    p.set(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]);
    if (occluder) {
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

// ---------- tabs ----------
function setView(name) {
  $$(".tab").forEach((x) => x.classList.toggle("active", x.dataset.view === name));
  $$(".view").forEach((s) => s.classList.toggle("active", s.id === `view-${name}`));
  active = name;
  for (const [k, v] of Object.entries(views)) { v.controls.enabled = k === active; v.labels.domElement.style.display = k === active ? "" : "none"; }
  tooltip.hidden = true;
  start(active);
}
$$(".tab").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
const started = {};
function start(view) {
  if (started[view]) return;
  started[view] = true;
  ({ globe: initGlobe, neo: initNeo, sky: initSky })[view]();
}

// =============================================================================================
// Simulation clock and positions. The server sends, once a second of wall time, the simulation
// time, GMST and every object's inertial (TEME) position and velocity. Between frames we draw
// a cubic Hermite curve through both endpoints, which follows the orbital arc even at 600x.
// =============================================================================================
const F = { prev: null, next: null, at: 0, gap: 1000, n: 0 };
let simRate = 1, ws;

function onFrame(buf) {
  const head = new Float64Array(buf, 0, 2), n = (buf.byteLength - 16) / 24;
  const f = { t: head[0], g: head[1], P: new Float32Array(buf, 16, 3 * n), V: new Float32Array(buf, 16 + 12 * n, 3 * n) };
  const now = performance.now();
  if (F.next) F.gap = Math.min(Math.max(now - F.at, 400), 2500);
  const jump = F.next && (f.t < F.next.t || f.t - F.next.t > simRate * 3 + 5);
  F.prev = !F.next || jump ? f : F.next;
  F.next = f; F.at = now; F.n = n;
}

function frameAlpha() { return F.next ? Math.min((performance.now() - F.at) / F.gap, 1) : 0; }
function simSeconds() {
  if (!F.next) return Date.now() / 1000;
  return F.prev.t + (F.next.t - F.prev.t) * frameAlpha();
}
function gmstAt(ts) { return F.next ? F.next.g + OMEGA_E * (ts - F.next.t) : 0; }

// Fill out (scene units, Earth radii; scene = (x, z, -y) of TEME) and vel (m/s, scene axes).
function interpolate(out, vel) {
  const a = frameAlpha(), { prev: A, next: B } = F, dt = B.t - A.t, n = F.n;
  const h00 = 2 * a ** 3 - 3 * a * a + 1, h10 = (a ** 3 - 2 * a * a + a) * dt, h01 = -2 * a ** 3 + 3 * a * a, h11 = (a ** 3 - a * a) * dt;
  const P0 = A.P, P1 = B.P, V0 = A.V, V1 = B.V, s = 1 / RE_M;
  for (let i = 0; i < n; i++) {
    const k = 3 * i;
    if (P1[k] === 0 && P1[k + 1] === 0) { out[k] = out[k + 1] = out[k + 2] = 0; continue; }
    const ok = P0[k] !== 0 || P0[k + 1] !== 0;
    const x = ok ? h00 * P0[k] + h10 * V0[k] + h01 * P1[k] + h11 * V1[k] : P1[k];
    const y = ok ? h00 * P0[k + 1] + h10 * V0[k + 1] + h01 * P1[k + 1] + h11 * V1[k + 1] : P1[k + 1];
    const z = ok ? h00 * P0[k + 2] + h10 * V0[k + 2] + h01 * P1[k + 2] + h11 * V1[k + 2] : P1[k + 2];
    out[k] = x * s; out[k + 1] = z * s; out[k + 2] = -y * s;
    if (vel) { vel[k] = V0[k] + (V1[k] - V0[k]) * a; vel[k + 1] = V0[k + 2] + (V1[k + 2] - V0[k + 2]) * a; vel[k + 2] = -(V0[k + 1] + (V1[k + 1] - V0[k + 1]) * a); }
  }
}

// Sun direction in the inertial frame (low-precision ephemeris, ~0.01 deg), scene axes.
function sunScene(ms) {
  const n = ms / 86400000 + 2440587.5 - 2451545.0;
  const L = (280.46 + 0.9856474 * n) % 360, g = (357.528 + 0.9856003 * n) % 360 * D2R;
  const lam = (L + 1.915 * Math.sin(g) + 0.02 * Math.sin(2 * g)) * D2R, eps = (23.439 - 4e-7 * n) * D2R;
  const x = Math.cos(lam), y = Math.cos(eps) * Math.sin(lam), z = Math.sin(eps) * Math.sin(lam);
  return new THREE.Vector3(x, z, -y);
}
const inShadow = (x, y, z, s) => { const d = x * s.x + y * s.y + z * s.z; return d < 0 && x * x + y * y + z * z - d * d < 1; };

setInterval(() => {
  $("#clock").textContent = new Date(simSeconds() * 1000).toISOString().slice(0, 19).replace("T", "  ") + " UTC";
  const m = $("#mode");
  m.className = `chip ${simRate === 1 ? "live" : "sim"}`;
  m.innerHTML = `<i></i>${simRate === 1 ? "LIVE" : `SIM ${simRate}×`}`;
}, 250);

function connectWS() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/positions`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (ev) => onFrame(ev.data);
  ws.onclose = () => setTimeout(connectWS, 2000);
}
function setRate(r, reset = false) {
  simRate = r;
  $$(".rate").forEach((x) => x.classList.toggle("active", Number(x.dataset.rate) === r));
  if (ws && ws.readyState === 1) ws.send(JSON.stringify(reset ? { reset: true, rate: r } : { rate: r }));
}
$$(".rate").forEach((b) => b.onclick = () => setRate(Number(b.dataset.rate)));
$$(".reset").forEach((b) => b.onclick = () => setRate(1, true));

// =============================================================================================
// A. Satellites & debris
// =============================================================================================
let G, catalog, assetIds = new Set(), kindOn = [1, 1, 1];

async function initGlobe() {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, innerWidth / innerHeight, 1e-8, 2000);
  camera.position.set(-2.2, 1.4, 3.6).multiplyScalar(1.05);
  const controls = new OrbitControls(camera, canvas);
  Object.assign(controls, { enableDamping: true, dampingFactor: 0.07, minDistance: 1.03, maxDistance: 22,
    autoRotate: false, rotateSpeed: 0.5, zoomSpeed: 1.1, enablePan: false });

  scene.add(new THREE.Mesh(new THREE.SphereGeometry(900, 48, 32),
    new THREE.MeshBasicMaterial({ map: tex("night_sky.png"), side: THREE.BackSide, color: 0x6b7280 })));

  // Everything fixed to the Earth turns with GMST; satellites and the Sun stay inertial.
  const earthFrame = new THREE.Group();
  scene.add(earthFrame);
  const sun = new THREE.Vector3(1, 0, 0);
  const earthMat = new THREE.ShaderMaterial({
    uniforms: { dayMap: { value: tex("earth_day_4096.jpg") }, nightMap: { value: tex("earth_night_4096.jpg") },
      specMap: { value: tex("earth_specular_2048.jpg", false) }, sunDir: { value: sun } },
    vertexShader: LOG_VS + `varying vec2 vUv; varying vec3 vN; varying vec3 vW;
      void main() { vUv = uv; vN = normalize(mat3(modelMatrix) * normal); vec4 w = modelMatrix * vec4(position, 1.0);
        vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; ${LOG_VS_END} }`,
    fragmentShader: LOG_FS + `uniform sampler2D dayMap, nightMap, specMap; uniform vec3 sunDir;
      varying vec2 vUv; varying vec3 vN; varying vec3 vW;
      void main() {
        ${LOG_FS_START}
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
  earthFrame.add(new THREE.Mesh(new THREE.SphereGeometry(1, 160, 120), earthMat));
  const clouds = new THREE.Mesh(new THREE.SphereGeometry(1.006, 96, 64), new THREE.MeshLambertMaterial({
    map: tex("earth_clouds_1024.png"), transparent: true, opacity: 0.85, depthWrite: false }));
  earthFrame.add(clouds);

  const sunLight = new THREE.DirectionalLight(0xffffff, 2.6);
  scene.add(sunLight, new THREE.AmbientLight(0x3b4558, 0.35));
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(1.045, 96, 64), new THREE.ShaderMaterial({
    uniforms: { sunDir: { value: sun } }, side: THREE.BackSide, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    vertexShader: LOG_VS + `varying vec3 vN; varying vec3 vW; void main() { vN = normalize(mat3(modelMatrix) * normal);
      vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; ${LOG_VS_END} }`,
    fragmentShader: LOG_FS + `uniform vec3 sunDir; varying vec3 vN; varying vec3 vW;
      void main() { ${LOG_FS_START} vec3 v = normalize(cameraPosition - vW); float f = pow(clamp(-dot(vN, v) / 0.3, 0.0, 1.0), 2.2);
        float lit = 0.15 + 0.85 * smoothstep(-0.35, 0.3, dot(normalize(vW), sunDir));
        gl_FragColor = vec4(vec3(0.3, 0.62, 1.0) * f * 1.6 * lit, 1.0); }`,
  })));

  const composer = makeComposer(scene, camera, [0.45, 0.45, 0.9]);
  G = views.globe = { scene, camera, controls, composer, labels: makeLabels(), sun, sunLight, clouds, earthFrame, update: updateGlobe, follow: -1, flight: 0 };
  window.solace = { views };  // for debugging from the console
  resize();
  requestAnimationFrame(loop);
  connectWS();

  catalog = await getJSON("/api/catalog");
  assetIds = new Set(catalog.assets);
  const n = catalog.ids.length, counts = [0, 0, 0];
  catalog.kinds.forEach((k) => counts[k]++);
  $("#cat-total").textContent = num(n);
  $("#cat-status").textContent = "Public catalog · SGP4 · live, inertial frame";
  counts.forEach((c, k) => { $(`#n${k}`).textContent = num(c); });

  // Point sprites for the whole catalogue (far view); per-object flags for "drawn as a 3-D model"
  // and "in Earth's shadow".
  G.pos = new Float32Array(3 * n); G.vel = new Float32Array(3 * n);
  G.model = new Float32Array(n); G.lit = new Float32Array(n).fill(1);
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(G.pos, 3).setUsage(THREE.DynamicDrawUsage));
  geom.setAttribute("kind", new THREE.BufferAttribute(new Float32Array(catalog.kinds), 1));
  geom.setAttribute("asset", new THREE.BufferAttribute(new Float32Array(catalog.ids.map((id) => (assetIds.has(id) ? 1 : 0))), 1));
  geom.setAttribute("model", new THREE.BufferAttribute(G.model, 1).setUsage(THREE.DynamicDrawUsage));
  geom.setAttribute("lit", new THREE.BufferAttribute(G.lit, 1).setUsage(THREE.DynamicDrawUsage));
  geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e4);
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { colors: { value: KIND_HEX.map((h) => new THREE.Color(h)) }, show: { value: new THREE.Vector3(1, 1, 1) },
      px: { value: renderer.getPixelRatio() }, time: { value: 0 }, selected: { value: -1 }, hovered: { value: -1 }, following: { value: 0 } },
    vertexShader: LOG_VS + `attribute float kind, asset, model, lit; uniform vec3 show; uniform float px, time, selected, hovered, following;
      uniform vec3 colors[3]; varying vec3 vC; varying float vA, vSel;
      void main() {
        int k = int(kind + 0.5);
        float on = k == 0 ? show.x : (k == 1 ? show.y : show.z);
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        float id = float(gl_VertexID);
        vSel = (id == selected || id == hovered) ? 1.0 : 0.0;
        vA = asset;
        vC = (asset > 0.5 ? vec3(1.0, 0.9, 0.55) : colors[k]) * mix(0.3, 1.0, lit);
        float base = k == 0 ? 1.9 : (k == 1 ? 2.3 : 1.6);
        float s = base * clamp(4.5 / -mv.z, 0.6, 3.2);
        if (asset > 0.5) s = 9.0 + 2.0 * sin(time * 3.0);
        if (vSel > 0.5) s = max(s * 2.6, 10.0);
        bool dead = dot(position, position) < 0.01 || (on < 0.5 && asset < 0.5) || (model > 0.5 && ((vSel < 0.5 && asset < 0.5) || following > 0.5));
        gl_PointSize = dead ? 0.0 : s * px;
        gl_Position = projectionMatrix * mv;
        ${LOG_VS_END}
      }`,
    fragmentShader: LOG_FS + `varying vec3 vC; varying float vA, vSel;
      void main() {
        ${LOG_FS_START}
        float d = length(gl_PointCoord - 0.5) * 2.0;
        if (d > 1.0) discard;
        float a = smoothstep(1.0, 0.35, d);
        vec3 c = vC * (0.85 + 0.35 * smoothstep(0.5, 0.0, d));
        if (vA > 0.5 || vSel > 0.5) { float ring = smoothstep(0.12, 0.0, abs(d - 0.8)); a = max(smoothstep(0.45, 0.2, d), ring); c = vC * 1.6; }
        gl_FragColor = vec4(c, a * 0.85);
      }`,
  });
  G.points = new THREE.Points(geom, mat);
  G.points.renderOrder = 2;
  scene.add(G.points);

  // 3-D spacecraft for objects near the camera (instanced; sized to stay visible at any zoom).
  G.type = new Uint8Array(n);
  catalog.kinds.forEach((k, i) => { G.type[i] = k === 1 ? 2 : k === 2 ? 3 : /STARLINK/.test(catalog.names[i] || "") ? 1 : 0; });
  G.spin = new Float32Array(3 * n).map(() => (Math.random() - 0.5) * 0.6);
  const geoms = geometries();
  G.inst = TYPES.map((t) => {
    const [g, size] = geoms[t], m = new THREE.InstancedMesh(g, modelMaterial(), 3000);
    m.instanceMatrix.setUsage(THREE.DynamicDrawUsage); m.count = 0; m.frustumCulled = false;
    m.setColorAt(0, new THREE.Color(1, 1, 1)); m.size = size; scene.add(m); return m;
  });
  G.geoms = geoms;

  G.assetLabels = catalog.assets.map((id) => {
    const i = catalog.ids.indexOf(id), el = document.createElement("div");
    el.className = "lbl asset-lbl"; el.textContent = catalog.names[i] ? catalog.names[i].replace(/ \(.*\)$/, "") : String(id);
    const o = new CSS2DObject(el); o.center.set(0, 0.5); scene.add(o);
    return { i, o, el };
  }).filter((a) => a.i >= 0);

  G.track = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({
    color: 0xffffff, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false }));
  G.track.visible = false; scene.add(G.track);
  G.ground = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xfde68a, transparent: true, opacity: 0.55 }));
  G.ground.visible = false; earthFrame.add(G.ground);

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
  canvas.addEventListener("dblclick", () => { if (active === "globe" && G.selected >= 0) followObject(G.selected); });
  addEventListener("keydown", (e) => { if (e.key === "Escape" && active === "globe") unfollow(); });
  $("#obj-close").onclick = () => { $("#obj-card").hidden = true; G.track.visible = G.ground.visible = false; mat.uniforms.selected.value = -1; G.selected = -1; unfollow(); };
  $("#view-reset").onclick = () => { unfollow(); flyTo(new THREE.Vector3(-2.2, 1.4, 3.6).multiplyScalar(1.05), true); };

  loadSpaceWeather(); setInterval(loadSpaceWeather, 10 * 60 * 1000);
  loadAssets();
  loadReentryWatch();
}

const visibleSat = (i) => (kindOn[catalog.kinds[i]] || assetIds.has(catalog.ids[i])) && (G.pos[3 * i] || G.pos[3 * i + 1]);
const pickGlobe = (ev) => pickPoints(G, ev, G.pos, catalog.ids.length, visibleSat, 1.0, 10);
let lastHover = 0;
function hoverGlobe(ev) {
  if (!catalog || !F.next || performance.now() - lastHover < 40) return;
  lastHover = performance.now();
  const i = pickGlobe(ev);
  G.points.material.uniforms.hovered.value = i;
  canvas.style.cursor = i >= 0 ? "pointer" : "grab";
  if (i < 0) return showTip(ev, null);
  const k = 3 * i, r = Math.hypot(G.pos[k], G.pos[k + 1], G.pos[k + 2]);
  showTip(ev, `<b>${esc(catalog.names[i] || catalog.ids[i])}</b><span class="muted">${KIND_NAME[catalog.kinds[i]]} · NORAD ${catalog.ids[i]} · ${num(Math.round((r - 1) * RE_M / 1000))} km · ${G.lit[i] ? "sunlit" : "in Earth's shadow"}</span>`);
}

const _m = new THREE.Matrix4(), _tumble = new THREE.Matrix4(), _x = new THREE.Vector3(), _y = new THREE.Vector3(), _z = new THREE.Vector3(), _q = new THREE.Quaternion(), _e = new THREE.Euler();
// Orientation: +X along velocity, +Z away from Earth; rocket bodies and debris tumble.
function attitude(i, t) {
  const k = 3 * i;
  _z.set(G.pos[k], G.pos[k + 1], G.pos[k + 2]).normalize();
  _x.set(G.vel[k], G.vel[k + 1], G.vel[k + 2]); _x.addScaledVector(_z, -_x.dot(_z)).normalize();
  _y.crossVectors(_z, _x);
  _m.makeBasis(_x, _y, _z);
  if (G.type[i] >= 2) { _e.set(t * G.spin[k], t * G.spin[k + 1], t * G.spin[k + 2]); _m.multiply(_tumble.makeRotationFromEuler(_e)); }
  return _m;
}

const DIM = new THREE.Color(0.1, 0.11, 0.14), FULL = new THREE.Color(1, 1, 1);
function updateGlobe(t) {
  const ts = simSeconds(), ms = ts * 1000;
  G.sun.copy(sunScene(ms));
  G.sunLight.position.copy(G.sun).multiplyScalar(10);
  G.earthFrame.rotation.y = gmstAt(ts);
  G.clouds.rotation.y += 0.00002;
  if (!G.points || !F.next || F.n !== catalog.ids.length) return;
  G.points.material.uniforms.time.value = t / 1000;
  interpolate(G.pos, G.vel);
  const n = F.n, P = G.pos, s = G.sun, cam = G.camera.position;
  // Models for objects close to the camera; points for the rest.
  const camAlt = Math.max(cam.length() - 1, 0.002), lim = Math.min(Math.max(camAlt * 1.4, 0.02), 1.2), lim2 = lim * lim;
  const counts = [0, 0, 0, 0], useModels = cam.length() < 4, tsec = t / 1000;
  // While riding along, objects docked to (or within 2 km of) the followed one are part of its model.
  const fk = 3 * G.follow, fx = P[fk], fy = P[fk + 1], fz = P[fk + 2], near2 = (2000 / RE_M) ** 2;
  for (let i = 0; i < n; i++) {
    const k = 3 * i, x = P[k], y = P[k + 1], z = P[k + 2];
    G.lit[i] = inShadow(x, y, z, s) ? 0 : 1;
    G.model[i] = 0;
    if (G.follow >= 0 && (x - fx) ** 2 + (y - fy) ** 2 + (z - fz) ** 2 < near2) { G.model[i] = 1; continue; }
    if (!useModels || (x === 0 && y === 0)) continue;
    if (!kindOn[catalog.kinds[i]] && !assetIds.has(catalog.ids[i])) continue;
    const dx = x - cam.x, dy = y - cam.y, dz = z - cam.z, d2 = dx * dx + dy * dy + dz * dz;
    const ty = G.type[i], m = G.inst[ty];
    if (d2 > lim2 || counts[ty] >= 3000) continue;
    const d = Math.sqrt(d2), sf = Math.max(1 / RE_M, 0.013 * d / m.size);
    attitude(i, tsec).scale(_y.set(sf, sf, sf)).setPosition(x, y, z);
    m.setMatrixAt(counts[ty], _m); m.setColorAt(counts[ty], G.lit[i] ? FULL : DIM);
    counts[ty]++; G.model[i] = 1;
  }
  G.inst.forEach((m, j) => { m.count = counts[j]; m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; });
  const at = G.points.geometry.attributes;
  at.position.needsUpdate = at.model.needsUpdate = at.lit.needsUpdate = true;

  if (G.follow >= 0) updateFollow(tsec);
  const camDir = cam.clone().normalize();
  for (const L of G.assetLabels) {
    const k = 3 * L.i;
    L.o.position.set(P[k], P[k + 1], P[k + 2]);
    const behind = L.o.position.dot(camDir) < 0 && L.o.position.length() < 1.2;
    L.el.style.opacity = behind || L.i === G.follow ? 0 : 1;
  }
}

// ---------- follow camera: ride along with one object, drawn at true scale ----------
function followObject(i) {
  G.flight++;
  unfollow(false);
  const name = catalog.names[i] || "", ty = G.type[i];
  const custom = detailFor(name), [geom, size] = custom || G.geoms[TYPES[ty]];
  G.detail = new THREE.Mesh(geom, modelMaterial()); G.detailOwned = !!custom;
  G.detail.scale.setScalar(1 / RE_M);
  G.scene.add(G.detail);
  G.detailSize = size / RE_M;
  G.follow = i; G.points.material.uniforms.following.value = 1;
  const k = 3 * i, p = new THREE.Vector3(G.pos[k], G.pos[k + 1], G.pos[k + 2]);
  // Start behind and above the spacecraft, looking along its path with the Earth below.
  const back = new THREE.Vector3(G.vel[k], G.vel[k + 1], G.vel[k + 2]).normalize().multiplyScalar(-2.2);
  const up = p.clone().normalize().multiplyScalar(0.9);
  G.controls.target.copy(p);
  G.camera.position.copy(p).addScaledVector(back.add(up).normalize(), G.detailSize * 3.2);
  attitude(i, 0); G.followBasis = new THREE.Matrix4().extractRotation(_m);
  Object.assign(G.controls, { minDistance: G.detailSize * 0.8, maxDistance: 2.5, zoomSpeed: 1.4 });
  $("#follow").textContent = "Stop following"; $("#follow-hint").hidden = false;
}

function unfollow(reset = true) {
  if (G.detail) { G.scene.remove(G.detail); if (G.detailOwned) G.detail.geometry.dispose(); G.detail.material.dispose(); G.detail = null; }
  if (G.follow < 0) return;
  G.follow = -1; G.points.material.uniforms.following.value = 0; G.followBasis = null;
  $("#follow").textContent = "Follow"; $("#follow-hint").hidden = true;
  if (reset) {
    G.controls.target.set(0, 0, 0);
    Object.assign(G.controls, { minDistance: 1.03, maxDistance: 22, zoomSpeed: 1.1 });
    const d = Math.max(G.camera.position.length(), 1.6);
    G.camera.position.setLength(d);
  }
}

let lastCard = 0;
function updateFollow(tsec) {
  const i = G.follow, k = 3 * i, p = new THREE.Vector3(G.pos[k], G.pos[k + 1], G.pos[k + 2]);
  if (!p.lengthSq()) return unfollow();
  // Keep the camera fixed in the spacecraft's local orbital frame (along-track, normal, radial), so
  // the Earth stays below as it orbits; the user's own drags and zooms are carried along.
  const off = G.camera.position.clone().sub(G.controls.target);
  if (G.followBasis) off.applyMatrix4(G.followBasis.clone().transpose());
  attitude(i, 0);
  G.followBasis = new THREE.Matrix4().extractRotation(_m);
  G.camera.position.copy(p).add(off.applyMatrix4(G.followBasis));
  G.controls.target.copy(p);
  attitude(i, tsec);
  G.detail.position.copy(p);
  G.detail.quaternion.setFromRotationMatrix(_m);
  G.detail.material.color.copy(G.lit[i] ? FULL : DIM);
  G.track.visible = G.camera.position.distanceTo(p) > 0.04;
  if (performance.now() - lastCard > 250) {
    lastCard = performance.now();
    const v = Math.hypot(G.vel[k], G.vel[k + 1], G.vel[k + 2]) / 1000, alt = (p.length() - 1) * RE_M / 1000;
    const g = gmstAt(simSeconds()), lon = ((Math.atan2(-p.z, p.x) - g) / D2R + 540) % 360 - 180, lat = Math.asin(p.y / p.length()) / D2R;
    $("#live").innerHTML = `<div class="stat"><div class="eyebrow">Altitude</div><div class="v">${num(Math.round(alt))}<span class="u">km</span></div></div>
      <div class="stat"><div class="eyebrow">Speed</div><div class="v">${fmt(v, 2)}<span class="u">km/s</span></div></div>
      <div class="stat"><div class="eyebrow">Over</div><div class="v">${fmt(Math.abs(lat), 1)}°${lat >= 0 ? "N" : "S"} ${fmt(Math.abs(lon), 1)}°${lon >= 0 ? "E" : "W"}</div></div>
      <div class="stat"><div class="eyebrow">Lighting</div><div class="v">${G.lit[i] ? "Sunlit" : "Eclipse"}</div></div>`;
  }
}

async function showObject(id) {
  const o = await getJSON(`/api/object/${id}?t=${simSeconds()}`);
  const i = catalog.ids.indexOf(id), color = assetIds.has(id) ? "#fde68a" : KIND_HEX[catalog.kinds[i]];
  const pts = o.track_teme_m.filter((p) => p[0] || p[1]).map((p) => new THREE.Vector3(p[0] / RE_M, p[2] / RE_M, -p[1] / RE_M));
  G.track.geometry.dispose();
  G.track.geometry = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts, true), 360, 0.0026 * Math.max(1, pts[0].length() / 1.1), 8, true);
  G.track.material.color.set(color).multiplyScalar(1.6);
  G.track.visible = true;
  const gp = o.ground_ecef_m.filter((p) => p[0] || p[1]).map((p) => new THREE.Vector3(p[0], p[2], -p[1]).normalize().multiplyScalar(1.0025));
  G.ground.geometry.dispose(); G.ground.geometry = new THREE.BufferGeometry().setFromPoints(gp);
  G.ground.material.color.set(color); G.ground.visible = true;
  G.points.material.uniforms.selected.value = i; G.selected = i;
  $("#obj-card").hidden = false;
  $("#obj-kind").innerHTML = `<i class="dot" style="display:inline-block;background:${color};box-shadow:0 0 8px ${color};margin-right:6px"></i>${o.kind} · NORAD ${o.id}`;
  $("#obj-name").textContent = o.name;
  let html = `<div class="stats">
    <div class="stat"><div class="eyebrow">Perigee</div><div class="v">${num(Math.round(o.perigee_km))}<span class="u">km</span></div></div>
    <div class="stat"><div class="eyebrow">Apogee</div><div class="v">${num(Math.round(o.apogee_km))}<span class="u">km</span></div></div>
    <div class="stat"><div class="eyebrow">Inclination</div><div class="v">${fmt(o.inclination_deg, 2)}<span class="u">°</span></div></div>
    <div class="stat"><div class="eyebrow">Period</div><div class="v">${fmt(o.period_min, 1)}<span class="u">min</span></div></div></div>
    <div class="stats" id="live"></div>
    <div class="row"><button id="follow" class="primary">Follow</button><span id="follow-hint" class="muted small" hidden>Drag to look around · scroll to zoom · Esc to leave</span></div>`;
  if (o.reentry) html += reentryBlock(o.reentry);
  if (o.asset) html += eventsTable(o.asset);
  $("#obj-body").innerHTML = html;
  $("#follow").onclick = () => (G.follow === i ? unfollow() : followObject(i));
  if (G.follow === i) { $("#follow").textContent = "Stop following"; $("#follow-hint").hidden = false; }
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
  $$("#assets .asset").forEach((el) => el.onclick = () => focusObject(Number(el.dataset.id)));
}

function focusObject(id) {
  const i = catalog.ids.indexOf(id);
  showObject(id);
  if (i >= 0) flyTo(new THREE.Vector3(G.pos[3 * i], G.pos[3 * i + 1], G.pos[3 * i + 2]));
}

function flyTo(target, exact = false) {
  if (!target.lengthSq()) return;
  unfollow(false); G.controls.target.set(0, 0, 0);
  const from = G.camera.position.clone(), to = exact ? target.clone() : target.clone().normalize().multiplyScalar(Math.max(target.length() * 2.2, 3.2));
  const t0 = performance.now(), id = ++G.flight;
  (function step() {
    if (id !== G.flight) return;  // superseded by another flight or by following an object
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

// ---------- re-entry watch: the paper's forecast, run on today's catalogue ----------
function reentryBlock(r) {
  const d = (s) => new Date(s).toISOString().slice(0, 10);
  return `<div class="reentry"><div class="eyebrow">Re-entry forecast</div>
    <div class="v mono">${d(r.median_utc)} <span class="muted">(${fmt(r.days, 0)} days)</span></div>
    <div class="muted small">80 % window ${d(r.lo_utc)} – ${d(r.hi_utc)}</div></div>`;
}

async function loadReentryWatch() {
  let w;
  try { w = await getJSON("/api/reentry"); } catch (e) { $("#reentry").textContent = "Unavailable"; return; }
  if (w.status !== "ok") { $("#reentry").textContent = w.status; if (w.status.startsWith("computing")) setTimeout(loadReentryWatch, 15000); return; }
  $("#reentry").innerHTML = `<table><tr><th>Object</th><th class="num">Median</th><th class="num">80 % window</th></tr>` +
    w.objects.slice(0, 12).map((r) => `<tr class="click" data-id="${r.id}"><td>${esc(r.name)}<div class="sub muted small">${r.perigee_km.toFixed(0)} km perigee</div></td>
      <td class="num">${fmt(r.days, 0)} d</td><td class="num muted">${fmt(r.lo_days, 0)}–${fmt(r.hi_days, 0)} d</td></tr>`).join("") +
    `</table><p class="note">${esc(w.method)} Forecast issued ${w.issued.slice(0, 10)}.</p>`;
  $$("#reentry tr.click").forEach((tr) => tr.onclick = () => focusObject(Number(tr.dataset.id)));
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
    vertexShader: LOG_VS + `attribute vec3 color; attribute float size; uniform float px; varying vec3 vC; varying float vBig;
      void main() { vC = color; vBig = size > 5.0 ? 1.0 : 0.0; vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * px * clamp(25.0 / -mv.z, 0.6, 2.5); gl_Position = projectionMatrix * mv; ${LOG_VS_END} }`,
    fragmentShader: LOG_FS + `varying vec3 vC; varying float vBig; void main() { ${LOG_FS_START} float d = length(gl_PointCoord - 0.5) * 2.0; if (d > 1.0) discard;
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
// C. Sky tonight: the sky as you would see it standing at your location. Drag to look around,
// scroll to zoom. Stars, constellations, planets and the Moon turn with the simulation clock;
// every tracked satellite above your horizon is drawn live, bright if sunlit.
// Horizon frame: +X east, +Y up, -Z north.
// =============================================================================================
const SKY_R = 100, CONST_URL = "https://cdn.jsdelivr.net/npm/d3-celestial@0.7.35/data/";
let S, here = { lat: 28.573, lon: -80.649, label: "Cape Canaveral (default)" }, skyData;

const eqVec = (ra, dec, r = SKY_R) => new THREE.Vector3(Math.cos(dec * D2R) * Math.cos(ra * D2R), Math.cos(dec * D2R) * Math.sin(ra * D2R), Math.sin(dec * D2R)).multiplyScalar(r);
const altazVec = (alt, az, r = SKY_R) => new THREE.Vector3(Math.cos(alt * D2R) * Math.sin(az * D2R), Math.sin(alt * D2R), -Math.cos(alt * D2R) * Math.cos(az * D2R)).multiplyScalar(r);

// Equatorial (of date) -> horizon frame for local sidereal time `lst` (rad) and latitude.
function eqToHorizon(lst, lat) {
  const sL = Math.sin(lst), cL = Math.cos(lst), sP = Math.sin(lat * D2R), cP = Math.cos(lat * D2R);
  const E = [-sL, cL, 0], N = [-sP * cL, -sP * sL, cP], U = [cP * cL, cP * sL, sP];
  return new THREE.Matrix4().set(E[0], E[1], E[2], 0, U[0], U[1], U[2], 0, -N[0], -N[1], -N[2], 0, 0, 0, 0, 1);
}

const ABOVE_HORIZON_VS = LOG_VS + `varying float vUp; void main() { vec4 w = modelMatrix * vec4(position, 1.0); vUp = w.y;
  gl_Position = projectionMatrix * viewMatrix * w; ${LOG_VS_END} }`;

function initSky() {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.1, 1000);
  camera.position.copy(altazVec(22, 180, -0.001));  // looking south, a little above the horizon
  const controls = new OrbitControls(camera, canvas);
  Object.assign(controls, { enableZoom: false, enablePan: false, enableDamping: true, dampingFactor: 0.08, rotateSpeed: -0.35, minPolarAngle: Math.PI * 0.39, maxPolarAngle: Math.PI - 0.01 });
  canvas.addEventListener("wheel", (ev) => {
    if (active !== "sky") return;
    camera.fov = Math.min(100, Math.max(12, camera.fov * (ev.deltaY > 0 ? 1.08 : 0.93))); camera.updateProjectionMatrix();
  }, { passive: true });

  // Sky and ground: one shader on a large sphere, coloured by the Sun's altitude.
  const sunDir = new THREE.Vector3(0, -1, 0);
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(400, 64, 48), new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, uniforms: { sunDir: { value: sunDir } },
    vertexShader: LOG_VS + `varying vec3 vD; void main() { vD = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); ${LOG_VS_END} }`,
    fragmentShader: LOG_FS + `uniform vec3 sunDir; varying vec3 vD;
      void main() { ${LOG_FS_START}
        float sa = sunDir.y, day = smoothstep(-0.12, 0.1, sa), twi = smoothstep(-0.32, -0.05, sa) * (1.0 - day);
        float toward = pow(max(dot(normalize(vec3(vD.x, 0.0, vD.z)), normalize(vec3(sunDir.x, 0.0, sunDir.z))), 0.0), 3.0);
        float h = max(vD.y, 0.0);
        vec3 night = mix(vec3(0.03, 0.05, 0.11), vec3(0.008, 0.012, 0.03), pow(h, 0.5));
        vec3 dayc = mix(vec3(0.55, 0.72, 0.92), vec3(0.18, 0.4, 0.78), pow(h, 0.6));
        vec3 twc = mix(vec3(0.95, 0.5, 0.25) * (0.35 + 0.65 * toward), vec3(0.06, 0.1, 0.25), pow(h, 0.35));
        vec3 col = night * (1.0 - day - twi) + dayc * day + twc * twi;
        col += vec3(1.0, 0.85, 0.6) * pow(max(dot(vD, sunDir), 0.0), 400.0) * day;
        if (vD.y < 0.0) {  // ground, faintly lit by the sky, with a soft horizon line
          vec3 g = mix(vec3(0.02, 0.025, 0.03), vec3(0.16, 0.17, 0.15), day) + vec3(0.08, 0.04, 0.02) * twi;
          col = mix(g, col * 0.7, exp(vD.y * 60.0) * 0.6);
        }
        gl_FragColor = vec4(col, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  })));

  // Everything fixed on the celestial sphere lives in `eq`, turned each frame by sidereal time.
  const eq = new THREE.Group(); eq.matrixAutoUpdate = false; scene.add(eq);
  const fade = { value: 1 };
  const starMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, uniforms: { fade, px: { value: renderer.getPixelRatio() }, fov: { value: 70 } },
    vertexShader: LOG_VS + `attribute float mag; uniform float px, fov; varying float vA, vUp;
      void main() { vec4 w = modelMatrix * vec4(position, 1.0); vUp = w.y;
        float zoom = 70.0 / fov;
        gl_PointSize = px * clamp((5.2 - mag) * 1.25, 1.2, 8.0) * sqrt(zoom);
        vA = clamp(1.25 - mag / 5.0, 0.18, 1.0);
        gl_Position = projectionMatrix * viewMatrix * w; ${LOG_VS_END} }`,
    fragmentShader: LOG_FS + `uniform float fade; varying float vA, vUp;
      void main() { ${LOG_FS_START} if (vUp < 0.0) discard; float d = length(gl_PointCoord - 0.5) * 2.0; if (d > 1.0) discard;
        float a = exp(-d * d * 4.0) * vA * fade; gl_FragColor = vec4(vec3(0.92, 0.95, 1.0) * (1.0 + 0.6 * vA), a); }`,
  });
  const lineMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, uniforms: { fade },
    vertexShader: ABOVE_HORIZON_VS,
    fragmentShader: LOG_FS + `uniform float fade; varying float vUp; void main() { ${LOG_FS_START} if (vUp < 0.0) discard;
      gl_FragColor = vec4(0.45, 0.62, 0.95, 0.32 * fade); }`,
  });

  // Live satellites above the horizon.
  const satMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, uniforms: { px: { value: renderer.getPixelRatio() }, hovered: { value: -1 }, on: { value: 1 } },
    vertexShader: LOG_VS + `attribute float lit; attribute float show; uniform float px, hovered, on; varying float vL, vH;
      void main() { vL = lit; vH = float(gl_VertexID) == hovered ? 1.0 : 0.0;
        gl_PointSize = show < 0.01 || on < 0.5 ? 0.0 : px * show * (vH > 0.5 ? 2.2 : 1.0);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); ${LOG_VS_END} }`,
    fragmentShader: LOG_FS + `varying float vL, vH; void main() { ${LOG_FS_START} float d = length(gl_PointCoord - 0.5) * 2.0; if (d > 1.0) discard;
      vec3 c = vL > 0.5 ? vec3(0.62, 0.95, 0.85) : vec3(0.3, 0.38, 0.52); if (vH > 0.5) c = vec3(1.0, 0.85, 0.4);
      gl_FragColor = vec4(c * (vL > 0.5 ? 1.3 : 1.0), smoothstep(1.0, 0.2, d) * (vL > 0.5 ? 0.95 : 0.45)); }`,
  });

  const labels = makeLabels();
  for (const [t, az] of [["N", 0], ["NE", 45], ["E", 90], ["SE", 135], ["S", 180], ["SW", 225], ["W", 270], ["NW", 315]]) {
    const el = document.createElement("div"); el.className = `lbl compass${t.length === 1 ? " major" : ""}`; el.textContent = t;
    const o = new CSS2DObject(el); o.position.copy(altazVec(1.2, az, SKY_R * 0.95)); scene.add(o);
  }
  // Moon: a lit sphere, so its phase comes out of the Sun direction.
  const moonLight = new THREE.DirectionalLight(0xffffff, 3.2); scene.add(moonLight, moonLight.target);
  const moon = new THREE.Mesh(new THREE.SphereGeometry(1.6, 48, 32), new THREE.MeshStandardMaterial({ color: 0xd8dde6, roughness: 1, metalness: 0 }));
  scene.add(moon);
  const sunSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture("rgba(255,226,170,1)"), blending: THREE.AdditiveBlending, depthWrite: false }));
  sunSprite.scale.setScalar(18); scene.add(sunSprite);

  const composer = makeComposer(scene, camera, [0.6, 0.5, 0.75]);
  S = views.sky = { scene, camera, controls, composer, labels, eq, sunDir, fade, starMat, lineMat, satMat, moon, moonLight, sunSprite, bodyLbl: {}, update: updateSky };
  resize();

  // Satellites: one point per catalogue object, placed each frame.
  S.satGeom = new THREE.BufferGeometry();
  S.pos = new Float32Array(1); S.view = null;
  canvas.addEventListener("pointermove", (ev) => { if (active === "sky") hoverSky(ev); });
  canvas.addEventListener("click", (ev) => {
    if (active !== "sky" || !S.view || !S.satMat.uniforms.on.value) return;
    const i = pickPoints(S, ev, S.view, catalog.ids.length, (j) => S.show[j] > 0, 0, 10);
    if (i >= 0) { setView("globe"); focusObject(catalog.ids[i]); }
  });

  $("#geo").onclick = () => navigator.geolocation && navigator.geolocation.getCurrentPosition((p) => {
    here = { lat: p.coords.latitude, lon: p.coords.longitude, label: `${p.coords.latitude.toFixed(2)}°, ${p.coords.longitude.toFixed(2)}°` };
    loadSky();
  }, () => { $("#loc").textContent = "Location denied; using default."; });
  loadConstellations();
  loadSky(); setInterval(loadSky, 60000);
  if (!started.globe) start("globe");  // the sky needs the live catalogue stream
}

async function loadConstellations() {
  try {
    const [lines, names] = await Promise.all([getJSON(CONST_URL + "constellations.lines.json"), getJSON(CONST_URL + "constellations.json")]);
    const seg = [];
    for (const f of lines.features) for (const l of f.geometry.coordinates) for (let k = 1; k < l.length; k++) {
      seg.push(eqVec(l[k - 1][0], l[k - 1][1], SKY_R * 1.01), eqVec(l[k][0], l[k][1], SKY_R * 1.01));
    }
    S.eq.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(seg), S.lineMat));
    S.constLbl = names.features.filter((f) => Number(f.properties.rank) <= 2).map((f) => {
      const el = document.createElement("div"); el.className = "lbl const"; el.textContent = f.properties.name;
      const o = new CSS2DObject(el); o.position.copy(eqVec(f.geometry.coordinates[0], f.geometry.coordinates[1], SKY_R * 1.01)); S.eq.add(o);
      return o;
    });
  } catch (e) { /* the sky still works without constellation figures */ }
}

async function loadSky() {
  skyData = await getJSON(`/api/sky?lat=${here.lat}&lon=${here.lon}`);
  if (!S.stars) {
    const pos = [], mag = [];
    for (const s of skyData.stars) { const v = eqVec(s.ra, s.dec); pos.push(v.x, v.y, v.z); mag.push(s.mag); }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute("mag", new THREE.Float32BufferAttribute(mag, 1));
    S.stars = new THREE.Points(g, S.starMat); S.eq.add(S.stars);
    S.starLbl = skyData.stars.filter((s) => s.name && s.mag < 1.6).map((s) => {
      const el = document.createElement("div"); el.className = "lbl star"; el.textContent = s.name;
      const o = new CSS2DObject(el); o.position.copy(eqVec(s.ra, s.dec)); o.center.set(-0.1, 0.5); S.eq.add(o); return o;
    });
  }
  const BC = { Sun: "#ffd27a", Moon: "#eef2f8", Mercury: "#c8c8c8", Venus: "#f7e3ae", Mars: "#ff7a4d", Jupiter: "#e9cfa6", Saturn: "#eadbaa", Uranus: "#a5e6ec", Neptune: "#86a9ff" };
  for (const b of skyData.bodies) {
    let o = S.bodyLbl[b.name];
    if (!o) {
      const el = document.createElement("div"); el.className = "lbl body"; el.innerHTML = `<i style="background:${BC[b.name]};box-shadow:0 0 10px ${BC[b.name]}"></i>${b.name}`;
      o = S.bodyLbl[b.name] = new CSS2DObject(el); o.center.set(0, 0.5); S.eq.add(o);
    }
    o.position.copy(eqVec(b.ra, b.dec, SKY_R * 0.9)); o.body = b;
  }
  // ISS passes in the next 48 h, drawn on the sky where they will happen.
  S.passes?.forEach((l) => S.scene.remove(l));
  S.passes = skyData.iss_passes.slice(0, 3).map((p) => {
    const pts = p.track.filter(([alt]) => alt > 0).map(([alt, az]) => altazVec(alt, az, SKY_R * 0.97));
    const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({ color: p.visible ? 0xfde68a : 0x8b97ad, dashSize: 1.2, gapSize: 0.8, transparent: true, opacity: p.visible ? 0.9 : 0.4 }));
    l.computeLineDistances(); S.scene.add(l); return l;
  });
  $("#loc").innerHTML = `${esc(here.label)} · ${skyData.is_night ? '<span class="pill ok">night</span>' : `<span class="pill mid">Sun up ${fmt(skyData.sun_alt, 0)}°</span>`}`;
  $("#sky-bodies").innerHTML = `<table><tr><th>Body</th><th class="num">Altitude</th><th class="num">Azimuth</th><th></th></tr>` +
    skyData.bodies.map((b) => `<tr class="click${b.alt < 0 ? " muted" : ""}" data-name="${b.name}"><td>${b.name}</td><td class="num">${fmt(b.alt, 1)}°</td><td class="num">${fmt(b.az, 1)}°</td><td class="muted small">${b.alt < 0 ? "below horizon" : "look"}</td></tr>`).join("") + "</table>";
  $$("#sky-bodies tr.click").forEach((tr) => tr.onclick = () => { const b = skyData.bodies.find((x) => x.name === tr.dataset.name); lookAt(b.alt, b.az); });
  $("#sky-passes").innerHTML = skyData.iss_passes.length ? `<table><tr><th>Rise (UTC)</th><th class="num">Max alt</th><th></th></tr>` +
    skyData.iss_passes.map((p) => `<tr><td class="mono">${new Date(p.rise_utc).toLocaleString("en-GB", { timeZone: "UTC", weekday: "short", hour: "2-digit", minute: "2-digit" })}</td>
      <td class="num">${fmt(p.max_alt, 0)}°</td><td>${p.visible ? '<span class="pill mid">naked eye</span>' : `<span class="muted small">${p.sunlit ? "daylight" : "in shadow"}</span>`}</td></tr>`).join("") + "</table>"
    : `<p class="note">No passes above 10°.</p>`;
}

function lookAt(alt, az) {
  const from = S.camera.position.clone().normalize(), to = altazVec(Math.max(alt, 5), az, -1).normalize(), t0 = performance.now();
  (function step() {
    const a = Math.min((performance.now() - t0) / 900, 1), e = a < 0.5 ? 2 * a * a : 1 - (-2 * a + 2) ** 2 / 2;
    S.camera.position.copy(from.clone().lerp(to, e).normalize().multiplyScalar(0.001));
    if (a < 1) requestAnimationFrame(step);
  })();
}

let skyFrame = 0;
function updateSky() {
  const ts = simSeconds(), lst = gmstAt(ts) + here.lon * D2R;
  if (!F.next) return;
  const M = eqToHorizon(lst, here.lat);
  S.eq.matrix.copy(M); S.eq.matrixWorldNeedsUpdate = true;
  S.starMat.uniforms.fov.value = S.camera.fov;

  // Sun in the horizon frame (from the inertial Sun, via sidereal time), sets sky colour and star fade.
  const sun = sunScene(ts * 1000), sunEq = new THREE.Vector3(sun.x, -sun.z, sun.y);  // back to (x, y, z) equatorial
  S.sunDir.copy(sunEq.clone().applyMatrix4(M)).normalize();
  S.fade.value = 1 - Math.min(1, Math.max(0, (S.sunDir.y + 0.2) / 0.22)) * 0.97;
  S.sunSprite.position.copy(S.sunDir).multiplyScalar(SKY_R * 0.9); S.sunSprite.visible = S.sunDir.y > -0.02;
  const mb = S.bodyLbl.Moon?.body;
  if (mb) {
    const mp = eqVec(mb.ra, mb.dec, SKY_R * 0.88).applyMatrix4(M);
    S.moon.position.copy(mp); S.moon.visible = mp.y > 0;
    S.moonLight.position.copy(mp).addScaledVector(S.sunDir, 50); S.moonLight.target.position.copy(mp);
  }
  if (skyFrame++ % 10 === 0) {  // hide labels below the horizon
    const w = new THREE.Vector3();
    for (const o of [...(S.constLbl || []), ...(S.starLbl || []), ...Object.values(S.bodyLbl)]) {
      w.copy(o.position).applyMatrix4(M); o.element.style.visibility = w.y > 0.5 ? "visible" : "hidden";
    }
  }
  updateSkySats(ts, lst);
}

// Satellites from the observer: topocentric direction, Earth-shadow test, observer darkness.
function updateSkySats(ts, lst) {
  if (!catalog || F.n !== catalog.ids.length) return;
  const n = F.n;
  if (!S.view) {
    S.view = new Float32Array(3 * n); S.show = new Float32Array(n); S.lit = new Float32Array(n); S.inert = new Float32Array(3 * n);
    S.satGeom.setAttribute("position", new THREE.BufferAttribute(S.view, 3).setUsage(THREE.DynamicDrawUsage));
    S.satGeom.setAttribute("show", new THREE.BufferAttribute(S.show, 1).setUsage(THREE.DynamicDrawUsage));
    S.satGeom.setAttribute("lit", new THREE.BufferAttribute(S.lit, 1).setUsage(THREE.DynamicDrawUsage));
    S.satGeom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e3);
    S.sats = new THREE.Points(S.satGeom, S.satMat); S.sats.renderOrder = 3; S.scene.add(S.sats);
  }
  interpolate(S.inert, null);
  const sun = sunScene(ts * 1000), P = S.inert;
  const sP = Math.sin(here.lat * D2R), cP = Math.cos(here.lat * D2R), sL = Math.sin(lst), cL = Math.cos(lst);
  // Observer and local axes in scene-inertial coordinates (x, z, -y of TEME), Earth radii.
  const toS = (x, y, z) => [x, z, -y];
  const U = toS(cP * cL, cP * sL, sP), E = toS(-sL, cL, 0), N = toS(-sP * cL, -sP * sL, cP);
  const dark = S.sunDir.y < -0.1;
  let up = 0, bright = 0;
  const list = [];
  for (let i = 0; i < n; i++) {
    const k = 3 * i, x = P[k], y = P[k + 1], z = P[k + 2];
    S.show[i] = 0;
    if (x === 0 && y === 0) continue;
    const rx = x - U[0], ry = y - U[1], rz = z - U[2];
    const u = rx * U[0] + ry * U[1] + rz * U[2];
    if (u <= 0) continue;
    const e = rx * E[0] + ry * E[1] + rz * E[2], nn = rx * N[0] + ry * N[1] + rz * N[2], rho = Math.hypot(e, u, nn);
    S.view[k] = e / rho * SKY_R * 0.96; S.view[k + 1] = u / rho * SKY_R * 0.96; S.view[k + 2] = -nn / rho * SKY_R * 0.96;
    const lit = !inShadow(x, y, z, sun);
    // Point size falls with range; a naked-eye candidate is a sunlit payload or rocket body within
    // ~1,200 km of a dark-sky observer (typical intrinsic magnitudes put the rest beyond the eye).
    const km = rho * 6371;
    S.lit[i] = lit ? 1 : 0; S.show[i] = lit ? Math.min(3.6, Math.max(1.1, 4.2 - 2.2 * Math.log10(km / 400))) : 1.1; up++;
    if (lit && dark && km < 1200 && catalog.kinds[i] < 2) { bright++; list.push([i, km, Math.asin(u / rho) / D2R]); }
  }
  const at = S.satGeom.attributes; at.position.needsUpdate = at.show.needsUpdate = at.lit.needsUpdate = true;
  if (skyFrame % 30 === 1) {
    list.sort((a, b) => a[1] - b[1]);
    const named = (i) => esc((catalog.names[i] || String(catalog.ids[i])).replace(/ \(.*\)$/, ""));
    $("#sky-now").innerHTML = `<div class="sw-grid"><div><div class="big">${num(up)}</div><div class="muted small">tracked objects above your horizon</div></div>
      <div><div class="big">${num(bright)}</div><div class="muted small">${dark ? "naked-eye candidates: sunlit, close, satellite or rocket body" : "naked-eye passes need a dark sky"}</div></div></div>
      <label class="toggle"><input type="checkbox" id="sky-sats" ${S.satMat.uniforms.on.value ? "checked" : ""}> Show satellites</label>` +
      (list.length ? `<table><tr><th>Nearest candidates</th><th class="num">Range</th><th class="num">Alt</th></tr>` +
        list.slice(0, 6).map(([i, r, a]) => `<tr class="click" data-i="${i}"><td>${named(i)}</td><td class="num">${num(Math.round(r))} km</td><td class="num">${fmt(a, 0)}°</td></tr>`).join("") + "</table>" : "");
    $("#sky-sats").onchange = (e) => { S.satMat.uniforms.on.value = e.target.checked ? 1 : 0; };
    $$("#sky-now tr.click").forEach((tr) => tr.onclick = () => {
      const i = Number(tr.dataset.i), k = 3 * i, v = new THREE.Vector3(S.view[k], S.view[k + 1], S.view[k + 2]).normalize();
      lookAt(Math.asin(v.y) / D2R, (Math.atan2(v.x, -v.z) / D2R + 360) % 360);
    });
  }
}

let lastSkyHover = 0;
function hoverSky(ev) {
  if (!S.view || performance.now() - lastSkyHover < 50) return;
  lastSkyHover = performance.now();
  const i = S.satMat.uniforms.on.value ? pickPoints(S, ev, S.view, catalog.ids.length, (j) => S.show[j] > 0, 0, 9) : -1;
  S.satMat.uniforms.hovered.value = i;
  canvas.style.cursor = i >= 0 ? "pointer" : "grab";
  if (i < 0) return showTip(ev, null);
  const k = 3 * i, v = new THREE.Vector3(S.view[k], S.view[k + 1], S.view[k + 2]).normalize();
  const alt = Math.asin(v.y) / D2R, az = (Math.atan2(v.x, -v.z) / D2R + 360) % 360;
  showTip(ev, `<b>${esc(catalog.names[i] || catalog.ids[i])}</b><span class="muted">${KIND_NAME[catalog.kinds[i]]} · alt ${fmt(alt, 0)}° az ${fmt(az, 0)}° · ${S.lit[i] ? "sunlit" : "in Earth's shadow"} · click to open on the globe</span>`);
}

start("globe");
