// Canal manual (port de python/classifier_hands.py): dos manos (lateralidad y estado por mano), filtro One-Euro,
// rasgos 3D, clasificación del alfabeto LSM estático y trazos (J, K, Ñ, Q, X, Z, BORRAR, ?).
// Mismos umbrales y mismo orden de decisión que Python: los rangos medidos están comentados allá (no se repiten).
// Un punto es [x, y, z]; una mano, 21 puntos. Los arreglos de NumPy son arreglos de arreglos.

export const FRAMES_ESTABLES = 4;
export const V_QUIETO = 1.2;
export const MAX_MANOS = 2;
export const AREA_MIN_MANO = 0.001;
export const LADOS = ["Izq", "Der"];
const MP_A_LADO = { Left: "Izq", Right: "Der" };
const DUPLICADO_DIST = 0.35;
const CONTINUIDAD_DIST = 1.2;
const CONTINUIDAD_SEG = 0.25;
const LATERALIDAD_SUAVIZADO = 0.15;
const LATERALIDAD_CAMBIO = 0.85;
const ONE_EURO_CORTE_MIN = 1.5;
const ONE_EURO_BETA = 0.4;
const ONE_EURO_CORTE_D = 1.0;
const EXT_RECTO = [1.40, 1.54, 1.45, 1.25];
const EXT_CERRADO = 1.15;
const FLEX_MAX_RECTO = 75;
const ARTICULACION_RECTA = 55;
const EXT_TOLERANCIA = 0.15;
const MENIQUE_RECTO = 50;
const FACTOR_ANCHO_PALMA = 1.55;
const ORIENT_ARRIBA = 55;
const ORIENT_LADO = [55, 115];
const ORIENT_ABAJO = 115;
const PULGAR_FUERA = -0.45;
const F_CONTACTO = 0.55;
const F_FLOJA = [0.75, 1.30];
const O_CONTACTO = 0.32;
const O_PULGAR_LAT = -0.15;
const C_GAP = [0.32, 1.10];
const C_PULGAR_LAT = -0.40;
const C_FLEX = [60, 115];
const C_INDICE_MIN = 1.05;
const T_ALTURA = 0.0;
const E_MCP_MAX = 33;
const ESPACIO_PULGAR = 1.00;
const B_ABERTURA = 20;
const V_CRUCE = 0.40;
const R_CRUCE = -0.05;
const U_ANG_MAX = 11;
const P_PROFUNDIDAD = 0.05;
const P_NORMAL_Z = -0.88;
const Y_DIST = 0.75;
const IY_RESTO_MAX = 1.35;
const D_DIR_MAX = 50;
const X_DIR_MIN = 60;
const X_ORIENT_MIN = 48;
const X_EXT_MIN = 1.25;
const Q_DIR_INDICE_MIN = 105;
const Q_ORIENT_MIN = 80;
const TRAYECTORIA_MUESTRAS = 64;
const DIN_MIN_MUESTRAS = 6;
const REARME_VEL = 0.8;
const REARME_SEGUNDOS = 0.3;
const DIN_FRACCION_POSE = 0.60;
const DIN_ENFRIAMIENTO = 1.0;
export const DIN_PERDIDOS_MAX = 5;
const VEL_SUAVIZADO = 0.25;
const J_DESCENSO_MIN = 0.55;
const J_GANCHO_MIN = 0.25;
const Z_BARRA_MIN = 0.40;
const Z_DIAGONAL_MIN = 0.45;
const INTERROG_ALTO_MIN = 0.80;
const VAIVEN_REVERSOS = 2;
const VAIVEN_RECTITUD = 0.5;
const VAIVEN_PROPORCION = 2.5;
const VAIVEN_AMP = { "Ñ": 0.07, QX: 0.07, K: 0.15 };
const BORRAR_RECORRIDO_MIN = 1.10;
const BORRAR_VEL_MIN = 3.0;
const BORRAR_DT = 0.15;

// ===================================================================== geometría
export const mod = (a, n) => ((a % n) + n) % n;           // el % de Python (siempre del signo del divisor)
const grados = (r) => (r * 180) / Math.PI;
const recortar = (x) => Math.max(-1, Math.min(1, x));
export const sub = (a, b) => a.map((v, i) => v - b[i]);
const suma = (a, b) => a.map((v, i) => v + b[i]);
const por = (a, k) => a.map((v) => v * k);
export const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
export const norma = (a) => Math.sqrt(dot(a, a));
const cruz = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

export function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
export function puntoMedio(a, b) { return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]; }
export function restar(a, b) { return [a[0] - b[0], a[1] - b[1]]; }
export function unitario(v) { const n = Math.hypot(v[0], v[1]) || 1.0; return [v[0] / n, v[1] / n]; }
function unitario3(v) { const n = norma(v); return n > 1e-12 ? por(v, 1 / n) : v; }

function anguloVectores(v1, v2) {
  const n = norma(v1) * norma(v2);
  if (n < 1e-12) return 0.0;
  return grados(Math.acos(recortar(dot(v1, v2) / n)));
}
function angulo3(a, b, c) {
  if (norma(sub(a, b)) * norma(sub(c, b)) < 1e-12) return 180.0;
  return anguloVectores(sub(a, b), sub(c, b));
}
function desdeArriba(v) { return grados(Math.acos(recortar(-v[1] / (norma(v) || 1.0)))); }

export function mediana(xs) {                              // np.median: promedio de los dos centrales si es par
  const s = [...xs].sort((a, b) => a - b), k = s.length >> 1;
  return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2;
}

// Filtro One-Euro sobre los 21 puntos (corte que sube con la velocidad, en tamaños de mano/s)
export class FiltroOneEuro {
  constructor(corteMin = ONE_EURO_CORTE_MIN, beta = ONE_EURO_BETA, corteD = ONE_EURO_CORTE_D) {
    this.corteMin = corteMin; this.beta = beta; this.corteD = corteD;
    this.reiniciar();
  }
  reiniciar() { this.x = this.dx = this.t = null; }
  static alfa(corte, dt) { return 1.0 / (1.0 + 1.0 / (2 * Math.PI * corte * dt)); }
  filtrar(x, t, escala = 1.0) {
    if (this.x === null) {
      this.x = x.map((p) => [...p]); this.dx = x.map((p) => p.map(() => 0)); this.t = t;
      return this.x;
    }
    if (t <= this.t) return this.x;
    const dt = t - this.t;
    this.t = t;
    const ad = FiltroOneEuro.alfa(this.corteD, dt), e = Math.max(escala, 1e-9);
    this.dx = this.dx.map((p, i) => p.map((d, j) => d + ad * ((x[i][j] - this.x[i][j]) / dt - d)));
    this.x = this.x.map((p, i) => p.map((v, j) => {
      const corte = this.corteMin + (this.beta * Math.abs(this.dx[i][j])) / e;
      return v + FiltroOneEuro.alfa(corte, dt) * (x[i][j] - v);
    }));
    return this.x;
  }
}

// ===================================================================== rasgos 3D de la mano
const WRIST = 0, THUMB_TIP = 4, INDEX_MCP = 5, INDEX_PIP = 6, INDEX_TIP = 8;
const MIDDLE_MCP = 9, MIDDLE_PIP = 10, MIDDLE_TIP = 12, PINKY_MCP = 17, PINKY_TIP = 20;
const DEDOS = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
export const PALMA = [0, 5, 9, 13, 17];
const EXT = "EXT", DOBLADO = "DOB", CERRADO = "CER";

export function aCanonica(puntos, lado) {                  // refleja el eje x de la mano izquierda
  return lado === "Izq" ? puntos.map((p) => p.map((v, j) => (j === 0 ? -v : v))) : puntos;
}
export function escalaMano(p) {
  return Math.max(dist(p[WRIST], p[MIDDLE_MCP]), FACTOR_ANCHO_PALMA * dist(p[INDEX_MCP], p[PINKY_MCP])) || 1.0;
}
function centroPalma(p) {
  return [0, 1].map((j) => PALMA.reduce((s, i) => s + p[i][j], 0) / PALMA.length);
}

export function analizarMano(img, mundo, lado) {
  const ic = aCanonica(img, lado), P = aCanonica(mundo, lado), w = P[WRIST];
  const u = unitario3(sub(P[MIDDLE_MCP], w));
  let s = sub(P[PINKY_MCP], P[INDEX_MCP]);
  s = unitario3(sub(s, por(u, dot(s, u))));
  const n = cruz(s, u);
  const palma = norma(sub(P[MIDDLE_MCP], w)) || 1e-6;
  const extension = [], flexion = [], flexionMcp = [], estados = [];
  DEDOS.forEach(([mcp, pip, dip, tip], k) => {
    const ext = norma(sub(P[tip], w)) / (norma(sub(P[mcp], w)) || 1e-6);
    const flex = (180 - angulo3(P[mcp], P[pip], P[dip])) + (180 - angulo3(P[pip], P[dip], P[tip]));
    extension.push(ext);
    flexion.push(flex);
    flexionMcp.push(180 - angulo3(w, P[mcp], P[pip]));
    const recto = (ext >= EXT_RECTO[k] && flex <= FLEX_MAX_RECTO)
      || (flex <= ARTICULACION_RECTA && ext >= EXT_RECTO[k] - EXT_TOLERANCIA)
      || (k === 3 && flex <= MENIQUE_RECTO && ext >= EXT_CERRADO);
    estados.push(recto ? EXT : (ext < EXT_CERRADO ? CERRADO : DOBLADO));
  });
  const p = img.map((q) => [q[0], q[1]]), pc = ic.map((q) => [q[0], q[1]]);
  const pips = por(suma(P[INDEX_PIP], P[MIDDLE_PIP]), 0.5);
  const palmaImg = norma(sub(ic[MIDDLE_MCP], ic[WRIST])) || 1e-6;
  return {
    lado, p, pc, escala: escalaMano(p), centro: centroPalma(pc), extension, flexion, flexionMcp, estados, normal: n,
    orient: desdeArriba(u), dirIndice: desdeArriba(sub(P[INDEX_TIP], P[INDEX_MCP])),
    pulgarLat: dot(sub(P[THUMB_TIP], P[INDEX_MCP]), s) / palma,
    pulgarAlt: dot(sub(P[THUMB_TIP], pips), u) / palma,
    dTi: norma(sub(ic[THUMB_TIP], ic[INDEX_TIP])) / palmaImg,
    cruce: dot(sub(P[MIDDLE_TIP], P[INDEX_TIP]), s) / palma,
    angIm: anguloVectores(sub(P[INDEX_TIP], P[INDEX_MCP]), sub(P[MIDDLE_TIP], P[MIDDLE_MCP])),
    profMedio: dot(sub(P[MIDDLE_TIP], P[INDEX_TIP]), n) / palma,
    abertura: anguloVectores(sub(P[INDEX_TIP], P[INDEX_MCP]), sub(P[PINKY_TIP], P[PINKY_MCP])),
  };
}

// ===================================================================== clasificación de señas
export const LETRAS_TIPEABLES = new Set(["A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "M", "N", "O", "P", "R",
  "S", "T", "U", "V", "W", "Y", "ESPACIO"]);
export const POSES_SILENCIOSAS = new Set(["ABIERTA"]);

function decidirQoX(dirIndice) { return dirIndice >= Q_DIR_INDICE_MIN ? "Q" : "X"; }

export function clasificarPose(r) {
  const [i, m, a, p] = r.estados.map((e) => e === EXT);
  const nExt = i + m + a + p;
  const arriba = r.orient <= ORIENT_ARRIBA;
  const lado = ORIENT_LADO[0] < r.orient && r.orient < ORIENT_LADO[1];
  const abajo = r.orient >= ORIENT_ABAJO;
  const ext = r.extension;

  const pinza = r.dTi < F_CONTACTO || (r.dTi < F_FLOJA[0] && ext[0] < F_FLOJA[1]);
  if (pinza && a && p && r.estados[1] !== CERRADO) return "F";
  if (r.flexion.slice(0, 2).every((f) => C_FLEX[0] <= f && f <= C_FLEX[1]) && ext[3] < 1.60) {
    if (r.dTi < O_CONTACTO && r.pulgarLat < O_PULGAR_LAT) return "O";
    if (C_GAP[0] <= r.dTi && r.dTi <= C_GAP[1] && r.pulgarLat < C_PULGAR_LAT && ext[0] >= C_INDICE_MIN) return "C";
  }
  if (nExt === 4) {
    if (abajo) return "M";
    if (r.dTi >= ESPACIO_PULGAR && r.pulgarLat < PULGAR_FUERA) return "ESPACIO";
    if (r.pulgarLat > 0 && r.abertura < B_ABERTURA) return "B";
    return "ABIERTA";
  }
  if (nExt === 3) {
    if (i && m && a) return abajo ? "M" : (arriba ? "W" : null);
    return null;
  }
  if (nExt === 2 && i && m) {
    if (abajo) return "N";
    if (lado) return "H";
    if (!arriba) return null;
    if (r.cruce >= V_CRUCE) return "V";
    if (r.cruce <= R_CRUCE) return "R";
    if (r.angIm < U_ANG_MAX && r.profMedio < P_PROFUNDIDAD && r.normal[2] < P_NORMAL_Z) return "U";
    return "P";
  }
  if (nExt === 1) {
    if (i) {
      if (r.pulgarLat < PULGAR_FUERA) return arriba ? "L" : (lado ? "G" : (abajo ? "Q" : null));
      if (r.dirIndice >= Q_DIR_INDICE_MIN && r.orient >= X_ORIENT_MIN) return "Q";
      if (r.dirIndice >= X_DIR_MIN && r.orient >= X_ORIENT_MIN) return "X";
      return "D";
    }
    if (p && ext[0] < IY_RESTO_MAX && ext[1] < IY_RESTO_MAX) return r.dTi >= Y_DIST ? "Y" : "I";
    return null;
  }
  if (nExt === 0) {
    if (r.orient >= Q_ORIENT_MIN && r.dirIndice >= Q_DIR_INDICE_MIN) return "Q";
    if (ext[0] >= X_EXT_MIN && r.estados.slice(1).every((e) => e === CERRADO)
        && D_DIR_MAX <= r.dirIndice && r.dirIndice < Q_DIR_INDICE_MIN) return "X";
    if (r.pulgarLat < PULGAR_FUERA) {
      if (ext[0] >= EXT_RECTO[0]) return arriba ? "L" : (lado ? "G" : null);
      return "A";
    }
    if (r.pulgarAlt > T_ALTURA) return "T";
    if (r.flexionMcp.slice(1).reduce((x, y) => x + y, 0) / 3 < E_MCP_MAX) return "E";
    return "S";
  }
  return null;
}

// ===================================================================== trayectoria de la mano
function puntosRastreados(p, centro = null) {
  return {
    muneca: p[WRIST], indice: p[INDEX_TIP], medio: p[MIDDLE_TIP], menique: p[PINKY_TIP],
    centro: centro ?? centroPalma(p), puntas: puntoMedio(p[INDEX_TIP], p[MIDDLE_TIP]),
  };
}

export class Trayectoria {
  constructor(n = TRAYECTORIA_MUESTRAS) { this.max = n; this.muestras = []; this.perdidos = 0; this.vel = 0.0; }
  agregar(t, r, pose) {
    const puntos = puntosRastreados(r.pc, r.centro);
    if (this.muestras.length) {
      const [t0, puntos0] = this.muestras[this.muestras.length - 1];
      if (t - t0 > 1e-3) {
        const v = dist(puntos.centro, puntos0.centro) / (r.escala || 1.0) / (t - t0);
        this.vel += (v - this.vel) * VEL_SUAVIZADO;
      }
    }
    this.muestras.push([t, puntos, r.escala, pose, r.dirIndice, puntosRastreados(r.p)]);
    if (this.muestras.length > this.max) this.muestras.shift();
    this.perdidos = 0;
  }
  perder() {
    this.perdidos += 1;
    this.vel *= 0.5;
    if (this.perdidos > DIN_PERDIDOS_MAX) this.muestras = [];
  }
  limpiar() { this.muestras = []; }
  recientes(segundos) {
    if (!this.muestras.length) return 0;
    const limite = this.muestras[this.muestras.length - 1][0] - segundos;
    return this.muestras.filter((m) => m[0] >= limite).length;
  }
  ultimas(n = null) { return n ? this.muestras.slice(-n) : this.muestras.slice(); }
  serie(nombre, n = null) {
    const ms = this.ultimas(n);
    const escala = mediana(ms.map((m) => m[2])) || 1.0;
    return ms.map((m) => [m[1][nombre][0] / escala, m[1][nombre][1] / escala]);
  }
  seriePx(nombre) { return this.muestras.map((m) => m[5][nombre]); }
  tiempos(n = null) { return this.ultimas(n).map((m) => m[0]); }
  fraccionPose(permitidas, n = null) {
    const ms = this.ultimas(n);
    return ms.filter((m) => permitidas.has(m[3])).length / Math.max(ms.length, 1);
  }
}

// ===================================================================== reconocedores de trazos (mano canónica)
function simplificarRdp(pts, eps) {
  const mantener = new Set([0, pts.length - 1]);
  const pila = [[0, pts.length - 1]];
  while (pila.length) {
    const [i, j] = pila.pop();
    const a = pts[i], ab = restar(pts[j], pts[i]);
    const largo = Math.hypot(ab[0], ab[1]);
    let lejos = 0.0, idx = -1;
    for (let k = i + 1; k < j; k++) {
      const v = restar(pts[k], a);
      const d = largo > 1e-9 ? Math.abs(ab[0] * v[1] - ab[1] * v[0]) / largo : Math.hypot(v[0], v[1]);
      if (d > lejos) { lejos = d; idx = k; }
    }
    if (lejos > eps) { mantener.add(idx); pila.push([i, idx], [idx, j]); }
  }
  return [...mantener].sort((x, y) => x - y);
}

function segmentos(pts, eps = 0.10, largoMin = 0.12, fusion = 35) {
  const idx = simplificarRdp(pts, eps);
  const segs = [];
  for (let k = 0; k + 1 < idx.length; k++) {
    const v = restar(pts[idx[k + 1]], pts[idx[k]]);
    segs.push([mod(grados(Math.atan2(-v[1], v[0])), 360), Math.hypot(v[0], v[1]), v[0], v[1]]);
  }
  const fusionado = [];
  for (const s of segs) {
    if (fusionado.length) {
      const ant = fusionado[fusionado.length - 1];
      const dif = Math.abs(mod(s[0] - ant[0] + 180, 360) - 180);
      if (s[1] < largoMin || dif <= fusion) {
        ant[2] += s[2]; ant[3] += s[3];
        ant[0] = mod(grados(Math.atan2(-ant[3], ant[2])), 360);
        ant[1] = Math.hypot(ant[2], ant[3]);
        continue;
      }
    }
    fusionado.push(s);
  }
  return fusionado.map((s) => [s[0], s[1]]);
}

const enArco = (rumbo, centro, semiancho) => Math.abs(mod(rumbo - centro + 180, 360) - 180) <= semiancho;
const simetrico = (fn) => (p, ts) => fn(p, ts) || fn(p.map((q) => [-q[0], q[1]]), ts);
const columna = (p, j) => p.map((q) => q[j]);
function argmax(xs) { let k = 0; for (let i = 1; i < xs.length; i++) if (xs[i] > xs[k]) k = i; return k; }
function argmin(xs) { let k = 0; for (let i = 1; i < xs.length; i++) if (xs[i] < xs[k]) k = i; return k; }

function trazoJ(p) {
  const n = p.length, y = columna(p, 1);
  const k = argmax(y);
  if (k < 3 || n - 1 - k < 2) return false;
  const i0 = argmin(y.slice(0, k + 1));
  const descenso = p[k][1] - p[i0][1];
  const deriva = Math.abs(p[k][0] - p[i0][0]);
  const gancho = p[k][0] - Math.min(...columna(p.slice(k), 0));
  return descenso >= J_DESCENSO_MIN && deriva <= 0.6 * descenso && gancho >= J_GANCHO_MIN;
}

function trazoZ(p) {
  const segs = segmentos(p);
  if (segs.length < 3) return false;
  const [[h1, l1], [h2, l2], [h3, l3]] = segs.slice(-3);
  return enArco(h1, 0, 40) && l1 >= Z_BARRA_MIN && enArco(h2, 225, 35) && l2 >= Z_DIAGONAL_MIN
    && enArco(h3, 0, 40) && l3 >= Z_BARRA_MIN;
}

function trazoInterrogacion(p) {
  const n = p.length, x = columna(p, 0), y = columna(p, 1);
  if (Math.max(...y) - Math.min(...y) < INTERROG_ALTO_MIN) return false;
  const k = argmax(x);
  if (!(0.15 * n <= k && k <= 0.75 * n) || argmin(y) > 0.6 * n) return false;
  if (p[k][0] - p[0][0] < 0.20 || p[k][0] - p[n - 1][0] < 0.20 || p[n - 1][1] - p[k][1] < 0.40) return false;
  const dxFinal = p[n - 1][0] - p[n - 4][0], dyFinal = p[n - 1][1] - p[n - 4][1];
  return dyFinal >= 0.25 && Math.abs(dxFinal) <= 0.6 * dyFinal;
}

function puntosDeGiro(xs, amp) {
  let ultimo = xs[0], iUltimo = 0, sentido = 0;
  const giros = [];
  for (let i = 1; i < xs.length; i++) {
    const x = xs[i];
    if (sentido === 0) {
      if (Math.abs(x - ultimo) > amp) { sentido = x > ultimo ? 1 : -1; ultimo = x; iUltimo = i; }
    } else if ((x - ultimo) * sentido > 0) {
      ultimo = x; iUltimo = i;
    } else if ((ultimo - x) * sentido > amp) {
      giros.push(iUltimo);
      sentido = -sentido; ultimo = x; iUltimo = i;
    }
  }
  return giros;
}

function rectitud(p, a, b) {
  let camino = 0;
  for (let i = a; i < b; i++) camino += Math.hypot(p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1]);
  return camino > 1e-9 ? Math.abs(p[b][0] - p[a][0]) / camino : 0.0;
}

function vaiven(amp, horizontal = true) {
  return (p) => {
    const x = columna(p, 0), y = columna(p, 1);
    const ancho = Math.max(...x) - Math.min(...x), alto = Math.max(...y) - Math.min(...y);
    if ((horizontal && alto > 0.8 * ancho) || ancho < 1.8 * amp) return false;
    const giros = puntosDeGiro(x, amp);
    if (giros.length < VAIVEN_REVERSOS) return false;
    const hitos = [0, ...giros].slice(-(VAIVEN_REVERSOS + 1));
    const tramos = hitos.slice(0, -1).map((a, k) => [a, hitos[k + 1]]);
    const amplitudes = tramos.map(([a, b]) => Math.abs(p[b][0] - p[a][0]));
    return tramos.every(([a, b]) => rectitud(p, a, b) >= VAIVEN_RECTITUD)
      && Math.max(...amplitudes) <= VAIVEN_PROPORCION * Math.min(...amplitudes);
  };
}

function buscarOrdenado(ts, valor) {                      // np.searchsorted (izquierda)
  let lo = 0, hi = ts.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (ts[mid] < valor) lo = mid + 1; else hi = mid; }
  return lo;
}

function trazoBorrar(p, ts) {
  const n = p.length;
  const dx = p[n - 1][0] - p[0][0], dy = p[n - 1][1] - p[0][1];
  if (dx > -BORRAR_RECORRIDO_MIN || Math.abs(dy) > 0.6 * Math.abs(dx)) return false;
  let total = 0, negativos = 0;
  for (let i = 1; i < n; i++) {
    const paso = p[i][0] - p[i - 1][0];
    total += Math.abs(paso);
    if (paso < 0) negativos += -paso;
  }
  if (total <= 0 || negativos / total < 0.85) return false;
  let velMax = 0.0;
  for (let i = 0; i < n; i++) {
    const j = buscarOrdenado(ts, ts[i] + BORRAR_DT);
    if (j < n) velMax = Math.max(velMax, (p[i][0] - p[j][0]) / Math.max(ts[j] - ts[i], 1e-3));
  }
  return velMax >= BORRAR_VEL_MIN;
}

// (gesto, configuraciones válidas, punto rastreado, reconocedor, fracción mínima con esa configuración, ventana en s)
export const GESTOS_DINAMICOS = [
  ["J", new Set(["I", "Y"]), "menique", simetrico(trazoJ), DIN_FRACCION_POSE, 1.2],
  ["Z", new Set(["D"]), "indice", simetrico(trazoZ), DIN_FRACCION_POSE, 1.8],
  ["?", new Set(["D"]), "indice", simetrico(trazoInterrogacion), DIN_FRACCION_POSE, 1.8],
  ["K", new Set(["P", "R", "U"]), "puntas", vaiven(VAIVEN_AMP.K), DIN_FRACCION_POSE, 1.8],
  ["Ñ", new Set(["N", "H"]), "centro", vaiven(VAIVEN_AMP["Ñ"]), DIN_FRACCION_POSE, 1.2],
  ["QX", new Set(["Q", "X"]), "indice", vaiven(VAIVEN_AMP.QX, false), DIN_FRACCION_POSE, 1.0],
  ["BORRAR", new Set(["B", "ESPACIO", "ABIERTA"]), "centro", trazoBorrar, DIN_FRACCION_POSE, 0.6],
];
const GESTOS_OSCILANTES = new Set(["K", "Ñ", "QX"]);
export const PUNTO_DE_POSE = {};
for (const [, poses, punto] of GESTOS_DINAMICOS) for (const pose of poses) PUNTO_DE_POSE[pose] ??= punto;

export class ReconocedorDinamico {
  constructor() { this.libreDesde = 0.0; this.pausaPendiente = false; this.tQuieto = null; }
  actualizarPausa(tray, ahora) {
    if (!this.pausaPendiente) return;
    if (tray.vel < REARME_VEL) {
      this.tQuieto = this.tQuieto ?? ahora;
      if (ahora - this.tQuieto >= REARME_SEGUNDOS) this.pausaPendiente = false;
    } else this.tQuieto = null;
  }
  evaluar(tray, ahora) {
    this.actualizarPausa(tray, ahora);
    if (ahora < this.libreDesde || tray.muestras.length < DIN_MIN_MUESTRAS) return null;
    const candidatos = [];
    for (const [gesto, poses, punto, reconocer, fraccion, ventana] of GESTOS_DINAMICOS) {
      const n = tray.recientes(ventana);
      if (n < DIN_MIN_MUESTRAS || (GESTOS_OSCILANTES.has(gesto) && this.pausaPendiente)) continue;
      const dominio = tray.fraccionPose(poses, n);
      if (dominio >= fraccion && reconocer(tray.serie(punto, n), tray.tiempos(n))) candidatos.push([dominio, gesto, n]);
    }
    if (!candidatos.length) return null;
    // max() de tuplas de Python: dominio, luego el nombre del gesto, luego n
    candidatos.sort((x, y) => (x[0] - y[0]) || (x[1] < y[1] ? -1 : x[1] > y[1] ? 1 : 0) || (x[2] - y[2]));
    let [, gesto, n] = candidatos[candidatos.length - 1];
    if (gesto === "QX") {
      const dirs = tray.ultimas(n).map((m) => m[4]);
      gesto = decidirQoX(dirs.reduce((s, v) => s + v, 0) / dirs.length);
    }
    if (gesto === "Q" || gesto === "X" || GESTOS_OSCILANTES.has(gesto)) { this.pausaPendiente = true; this.tQuieto = null; }
    this.libreDesde = ahora + DIN_ENFRIAMIENTO;
    tray.limpiar();
    return gesto;
  }
}

// ===================================================================== dos manos
// Una mano que MediaPipe vio en este fotograma. img: 21 [x, y, z] en píxeles (z en la escala de x); mundo: metros
export class DeteccionMano {
  constructor(ladoMp, confianza, img, mundo) {
    this.ladoMp = ladoMp; this.confianza = confianza; this.img = img; this.mundo = mundo; this.nueva = false;
  }
  get centro() { return centroPalma(this.img); }
  get escala() { return escalaMano(this.img); }
}

// Salida de HandLandmarker (MediaPipe Tasks JS) -> [DeteccionMano] (como leer_manos de Python)
export function leerManos(resultado, ancho, alto) {
  const salida = [];
  const marcas = resultado?.landmarks ?? [];
  const mundos = resultado?.worldLandmarks ?? [];
  const lateralidad = resultado?.handedness ?? resultado?.handednesses ?? [];
  marcas.forEach((lm, k) => {
    const img = lm.map((l) => [l.x * ancho, l.y * alto, l.z * ancho]);
    const xs = img.map((q) => q[0]), ys = img.map((q) => q[1]);
    if ((Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys)) < AREA_MIN_MANO * ancho * alto) return;
    const mundo = k < mundos.length ? mundos[k].map((l) => [l.x, l.y, l.z]) : img.map((q) => q.map((v) => v / Math.max(ancho, 1)));
    const etiqueta = lateralidad[k]?.[0] ?? { categoryName: "Right", score: 0.5 };
    salida.push(new DeteccionMano(MP_A_LADO[etiqueta.categoryName] ?? "Der", etiqueta.score, img, mundo));
  });
  return salida;
}

export const otroLado = (lado) => (lado === "Izq" ? "Der" : "Izq");

export class SeguidorManos {
  constructor() {
    this.ranuras = Object.fromEntries(LADOS.map((l) => [l, { centro: null, escala: 1.0, t: -10.0, evidDer: 0.5 }]));
  }
  static sinDuplicados(dets) {
    if (dets.length < 2) return [...dets];
    const [a, b] = dets;
    if (dist(a.centro, b.centro) < DUPLICADO_DIST * Math.max(a.escala, b.escala)) return [b.confianza > a.confianza ? b : a];
    return dets.slice(0, 2);
  }
  asignar(detecciones, ahora) {
    const dets = SeguidorManos.sinDuplicados(detecciones);
    let asignadas = {};
    const usadas = new Set();
    const pares = [];
    for (const lado of LADOS) {
      const r = this.ranuras[lado];
      if (r.centro === null || ahora - r.t > CONTINUIDAD_SEG) continue;
      dets.forEach((d, k) => {
        const distancia = dist(d.centro, r.centro) / r.escala;
        if (distancia < CONTINUIDAD_DIST) pares.push([distancia, lado, k]);
      });
    }
    pares.sort((x, y) => (x[0] - y[0]) || (x[1] < y[1] ? -1 : x[1] > y[1] ? 1 : 0) || (x[2] - y[2]));
    for (const [, lado, k] of pares) {
      if (!(lado in asignadas) && !usadas.has(k)) { asignadas[lado] = dets[k]; usadas.add(k); }
    }
    dets.forEach((d, k) => {
      if (usadas.has(k)) return;
      const lado = !(d.ladoMp in asignadas) ? d.ladoMp : otroLado(d.ladoMp);
      if (lado in asignadas) return;
      asignadas[lado] = d;
      usadas.add(k);
      d.nueva = true;
      this.ranuras[lado].evidDer = d.ladoMp === "Der" ? d.confianza : 1 - d.confianza;
    });
    const contrarias = [];
    for (const [lado, d] of Object.entries(asignadas)) {
      const r = this.ranuras[lado];
      const pDer = d.ladoMp === "Der" ? d.confianza : 1 - d.confianza;
      r.evidDer += (pDer - r.evidDer) * LATERALIDAD_SUAVIZADO;
      const evidencia = lado === "Der" ? r.evidDer : 1 - r.evidDer;
      if (evidencia < 1 - LATERALIDAD_CAMBIO) contrarias.push(lado);
    }
    if (contrarias.length === 2 || (contrarias.length === 1 && !(otroLado(contrarias[0]) in asignadas))) {
      const evid = Object.fromEntries(LADOS.map((l) => [l, this.ranuras[l].evidDer]));
      const nuevas = {};
      for (const [lado, d] of Object.entries(asignadas)) {
        if (contrarias.includes(lado)) {
          d.nueva = true;
          nuevas[otroLado(lado)] = d;
          this.ranuras[otroLado(lado)].evidDer = evid[lado];
        } else nuevas[lado] = d;
      }
      asignadas = nuevas;
    }
    for (const [lado, d] of Object.entries(asignadas)) {
      const r = this.ranuras[lado];
      r.centro = d.centro; r.escala = d.escala; r.t = ahora;
    }
    return asignadas;
  }
}

// ===================================================================== estado de cada mano
export class Estabilizador {
  constructor(n) { this.n = n; this.estable = null; this.candidato = null; this.conteo = 0; }
  actualizar(sena) {
    if (sena === this.estable) { this.candidato = sena; this.conteo = 0; }
    else if (sena === this.candidato) {
      this.conteo += 1;
      if (this.conteo >= this.n) { this.estable = sena; this.conteo = 0; }
    } else { this.candidato = sena; this.conteo = 1; }
    return this.estable;
  }
}

export class ProgresoSuave {
  constructor(suavizado = 0.35) { this.valor = 0.0; this.suavizado = suavizado; }
  actualizar(objetivo) {
    if (objetivo < this.valor - 1e-3) this.valor = objetivo;
    else this.valor += (objetivo - this.valor) * this.suavizado;
    return this.valor;
  }
}

export class CanalMano {
  constructor(lado) {
    this.lado = lado;
    this.filtroImg = new FiltroOneEuro();
    this.filtroMundo = new FiltroOneEuro();
    this.estabilizador = new Estabilizador(FRAMES_ESTABLES);
    this.tray = new Trayectoria();
    this.reconocedor = new ReconocedorDinamico();
    this.barra = new ProgresoSuave();
    this.perdidos = 0;
  }
  procesar(det, ahora) {
    const vacio = { lado: this.lado, rasgos: null, poseCruda: null, pose: null, gesto: null, moviendo: false, progreso: 0.0, img: null };
    if (!det) {
      this.perdidos += 1;
      if (this.perdidos > DIN_PERDIDOS_MAX) { this.filtroImg.reiniciar(); this.filtroMundo.reiniciar(); }
      this.estabilizador.actualizar(null);
      this.tray.perder();
      return vacio;
    }
    if (det.nueva) { this.filtroImg.reiniciar(); this.filtroMundo.reiniciar(); this.tray.limpiar(); }
    this.perdidos = 0;
    const img = this.filtroImg.filtrar(det.img, ahora, det.escala);
    const mundo = this.filtroMundo.filtrar(det.mundo, ahora, norma(sub(det.mundo[MIDDLE_MCP], det.mundo[WRIST])));
    const r = analizarMano(img, mundo, this.lado);
    const poseCruda = clasificarPose(r);
    const pose = this.estabilizador.actualizar(poseCruda);
    this.tray.agregar(ahora, r, poseCruda);
    const gesto = this.reconocedor.evaluar(this.tray, ahora);
    return { ...vacio, rasgos: r, poseCruda, pose, gesto, moviendo: this.tray.vel > V_QUIETO, img };
  }
}
