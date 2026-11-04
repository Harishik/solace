// Procedural spacecraft models, built in metres. Model frame: +X along-track (velocity),
// +Y orbit normal (solar-array axis), +Z zenith (away from Earth).
import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";

const GOLD = "#b8913f", ARRAY = "#1f3b8c", WHITE = "#e5e7eb", GREY = "#9ca3af", DARK = "#3f4652";

function part(geom, hex, pos = [0, 0, 0], rot = [0, 0, 0]) {
  let g = geom.index ? geom.toNonIndexed() : geom;
  g.deleteAttribute("uv");
  g.applyMatrix4(new THREE.Matrix4().compose(new THREE.Vector3(...pos),
    new THREE.Quaternion().setFromEuler(new THREE.Euler(...rot)), new THREE.Vector3(1, 1, 1)));
  const c = new THREE.Color(hex), n = g.attributes.position.count, col = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) col.set([c.r, c.g, c.b], 3 * i);
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  return g;
}
const box = (x, y, z) => new THREE.BoxGeometry(x, y, z);
const cyl = (r1, r2, h, s = 20) => new THREE.CylinderGeometry(r1, r2, h, s);  // axis along Y
const ALONG_X = [0, 0, Math.PI / 2], ALONG_Z = [Math.PI / 2, 0, 0];

function payload() {
  const parts = [part(box(2.2, 2.0, 2.4), GOLD), part(cyl(0.9, 0.15, 0.35), WHITE, [0, 0, -1.4], ALONG_Z),
    part(cyl(0.03, 0.03, 1.4, 6), WHITE, [0.5, 0, 1.9], ALONG_Z)];
  for (const s of [-1, 1]) {
    parts.push(part(cyl(0.05, 0.05, 0.6, 6), GREY, [0, s * 1.3, 0]));
    for (const k of [0, 1, 2]) parts.push(part(box(1.9, 2.05, 0.05), ARRAY, [0, s * (1.65 + 1.05 + 2.1 * k), 0]));
  }
  return mergeGeometries(parts);
}

function starlink() {
  return mergeGeometries([part(box(2.8, 1.4, 0.3), GREY), part(box(0.15, 0.4, 0.2), DARK, [0, 0.9, 0]),
    part(box(2.6, 7.6, 0.04), "#1e2a4a", [0, 4.9, 0]), part(box(0.6, 0.6, 0.08), WHITE, [0.8, 0, -0.2])]);
}

function rocketBody() {
  return mergeGeometries([part(cyl(1.4, 1.4, 8, 28), "#d4d8de", [0, 0, 0], ALONG_X),
    part(cyl(1.42, 1.42, 0.7, 28), DARK, [2.4, 0, 0], ALONG_X),
    part(cyl(0.9, 1.4, 1.2, 28), "#c7ccd3", [4.6, 0, 0], ALONG_X),
    part(cyl(1.1, 0.35, 1.5, 20), "#4b5563", [-4.75, 0, 0], ALONG_X)]);
}

function debris() {
  const g = new THREE.IcosahedronGeometry(0.55, 0), p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {  // displacement is a function of position, so shared corners stay joined
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i), f = 1 + 0.38 * Math.sin(13 * x + 7 * y + 5 * z);
    p.setXYZ(i, x * f, y * f * 0.7, z * f);
  }
  g.computeVertexNormals();
  return mergeGeometries([part(g, "#a8a29e"), part(box(1.3, 0.55, 0.04), "#c9b48a", [0.5, 0.2, 0.3], [0.4, 0.3, 0.2])]);
}

function iss() {
  const parts = [part(box(1.6, 100, 1.6), "#aeb6c2")];  // integrated truss, along the orbit normal
  for (const y of [-49, -34, 34, 49]) for (const z of [-1, 1]) {
    parts.push(part(box(0.12, 11.6, 34), "#b5782c", [0, y, z * 18.2]));
    parts.push(part(box(0.3, 0.3, 34.5), GREY, [0, y, z * 18.2]));
  }
  for (const y of [-22, 22]) parts.push(part(box(13, 3.2, 0.15), WHITE, [0, y, -7]));
  parts.push(part(cyl(2.1, 2.1, 52, 24), WHITE, [-6, 0, -3.4], ALONG_X));       // US and Russian segments
  parts.push(part(cyl(2.2, 2.2, 15, 24), WHITE, [14, 9, -3.4]));                 // Kibo
  parts.push(part(cyl(2.2, 2.2, 7, 24), WHITE, [14, -6.5, -3.4]));               // Columbus
  parts.push(part(cyl(1.4, 1.4, 6, 20), "#d1d5db", [22, 0, -3.4], ALONG_X));     // node and docked vehicle
  for (const s of [-1, 1]) parts.push(part(box(10.6, 0.1, 3.4), ARRAY, [-30, s * 9, -3.4]));  // Russian arrays
  return mergeGeometries(parts);
}

function hubble() {
  const parts = [part(cyl(2.1, 2.1, 13.2, 32), "#d9dde3", [0, 0, 0], ALONG_X),
    part(cyl(2.12, 2.12, 0.4, 32), DARK, [6.7, 0, 0], ALONG_X), part(box(0.1, 2.6, 2.4), "#cfd5dd", [7.2, 0, 1.6])];
  for (const s of [-1, 1]) parts.push(part(box(2.5, 7.1, 0.05), ARRAY, [-1.5, s * 5.6, 0]));
  return mergeGeometries(parts);
}

// type -> [geometry, characteristic size in metres]
export const TYPES = ["payload", "starlink", "rocket", "debris"];
export function geometries() {
  return { payload: [payload(), 15], starlink: [starlink(), 10], rocket: [rocketBody(), 10], debris: [debris(), 1.4] };
}
export function detailFor(name, type) {
  const n = (name || "").toUpperCase();
  if (n.startsWith("ISS")) return [iss(), 109];
  if (n.startsWith("HST") || n.includes("HUBBLE")) return [hubble(), 16];
  return null;
}
export const material = () => new THREE.MeshStandardMaterial({ vertexColors: true, metalness: 0.25, roughness: 0.55 });
