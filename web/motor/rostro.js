// Canal no manual (port de python/classifier_face.py): pose 3D de la cabeza, cejas compensadas por la pose
// (pregunta), cara tapada, autocalibración de la cara neutra y gestos de cabeza (negar / asentir).
// En el navegador no hay OpenCV: solvePnP, Rodrigues y projectPoints van aquí (Levenberg-Marquardt sobre el mismo
// error de reproyección que cv2.SOLVEPNP_ITERATIVE, partiendo de la pose anterior como en Python).
import { dist, puntoMedio, restar, unitario, mediana } from "./manos.js";

const FRAMES_CALIBRACION = 45;
const CARA_ERROR_MAX = 0.018;
const CALIB_PERCENTIL = 35;
export const CEJAS_UMBRAL = 1.07;
const CEJAS_UMBRAL_SALIDA = 1.04;
const CEJAS_SEGUNDOS = 1.5;
const CEJAS_TOLERANCIA = 0.30;
const CEJAS_REARME = 0.30;
const CEJAS_MAX_SEGUNDOS = 4.0;
const CEJAS_ADAPTACION = 0.005;
const CEJAS_ZONA_ADAPT = 0.03;
const CEJAS_PAUSA_TRAS_CABEZA = 0.8;
export const NEG_AMPLITUD = 8.0;
const NEG_ENTRADAS_MIN = 3;
export const AFIRM_AMPLITUD = 5.0;
const AFIRM_ENTRADAS_MIN = 3;
const OSC_VENTANA = 1.4;
const OSC_TAU_CENTRO = 2.5;
const ENFRIAMIENTO_CABEZA = 1.2;
const DURACION_AVISO = 1.2;

const CEJA_SUP_DER = [70, 63, 105, 66, 107];
const CEJA_SUP_IZQ = [300, 293, 334, 296, 336];
const CANTO_DER = [33, 133];
const CANTO_IZQ = [362, 263];
const PARPADO_DER = [159, 145];
const PARPADO_IZQ = [386, 374];
const FRENTE = 10, MENTON = 152;
const CARA_PNP = [1, 4, 5, 6, 168, 197, 195, 33, 133, 362, 263, 10, 152, 234, 454];
const PUNTOS_CEJAS = [...CEJA_SUP_DER, ...CEJA_SUP_IZQ, ...CANTO_DER, ...CANTO_IZQ, FRENTE, MENTON];
const MODELO_CARA = {
  1: [0.0, 1.126865, -7.475604], 4: [0.0, 0.46317, -7.58658], 5: [0.0, -0.365669, -7.24287],
  6: [0.0, -2.473255, -5.788627], 168: [0.0, -3.271027, -5.236015], 197: [0.0, -1.728369, -6.31675],
  195: [0.0, -1.059413, -6.774605], 10: [0.0, -8.261778, -4.481535], 152: [0.0, 9.403378, -4.264492],
  33: [-4.445859, -2.663991, -3.173422], 133: [-1.856432, -2.585245, -3.757904],
  362: [1.856432, -2.585245, -3.757904], 263: [4.445859, -2.663991, -3.173422],
  234: [-7.664182, -0.673132, 2.435867], 454: [7.664182, -0.673132, 2.435867],
  70: [-5.720968, -4.254584, -2.830852], 63: [-4.985894, -4.802461, -3.751977],
  105: [-3.986562, -5.109487, -4.466315], 66: [-2.760292, -5.100971, -5.01599],
  107: [-1.395634, -5.011963, -5.316032], 300: [5.720968, -4.254584, -2.830852],
  293: [4.985894, -4.802461, -3.751977], 334: [3.986562, -5.109487, -4.466315],
  296: [2.760292, -5.100971, -5.01599], 336: [1.395634, -5.011963, -5.316032],
};
const MODELO_PNP = CARA_PNP.map((i) => MODELO_CARA[i]);
const MODELO_CEJAS = PUNTOS_CEJAS.map((i) => MODELO_CARA[i]);
export const INDICES_ROSTRO = [...new Set([...CARA_PNP, ...PUNTOS_CEJAS, ...PARPADO_DER, ...PARPADO_IZQ])].sort((a, b) => a - b);

const grados = (r) => (r * 180) / Math.PI;

// ===================================================================== geometría de cámara (lo que hacía OpenCV)
export function rodrigues([rx, ry, rz]) {
  const th = Math.hypot(rx, ry, rz);
  if (th < 1e-12) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const [x, y, z] = [rx / th, ry / th, rz / th], c = Math.cos(th), s = Math.sin(th), C = 1 - c;
  return [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
          [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
          [z * x * C - y * s, z * y * C + x * s, c + z * z * C]];
}
export function proyectar(modelo, rvec, tvec, ancho, alto) {   // cv2.projectPoints con la cámara de matriz_camara
  const R = rodrigues(rvec), f = ancho, cx = ancho / 2, cy = alto / 2;
  return modelo.map(([X, Y, Z]) => {
    const x = R[0][0] * X + R[0][1] * Y + R[0][2] * Z + tvec[0];
    const y = R[1][0] * X + R[1][1] * Y + R[1][2] * Z + tvec[1];
    const z = R[2][0] * X + R[2][1] * Y + R[2][2] * Z + tvec[2];
    return [f * x / z + cx, f * y / z + cy];
  });
}

function residuos(params, img, ancho, alto) {
  const pr = proyectar(MODELO_PNP, params.slice(0, 3), params.slice(3), ancho, alto);
  const r = [];
  pr.forEach((q, i) => r.push(q[0] - img[i][0], q[1] - img[i][1]));
  return r;
}

function resolver(A, b) {                                  // eliminación gaussiana con pivoteo (6 x 6)
  const n = b.length, M = A.map((fila, i) => [...fila, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let i = c + 1; i < n; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i;
    if (Math.abs(M[p][c]) < 1e-15) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let i = c + 1; i < n; i++) {
      const k = M[i][c] / M[c][c];
      for (let j = c; j <= n; j++) M[i][j] -= k * M[c][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

// Levenberg-Marquardt: minimiza la suma de errores de reproyección al cuadrado (como SOLVEPNP_ITERATIVE)
function refinar(params, img, ancho, alto, iteraciones = 50) {
  let p = [...params], r = residuos(p, img, ancho, alto), costo = r.reduce((s, v) => s + v * v, 0), lambda = 1e-3;
  for (let it = 0; it < iteraciones; it++) {
    const J = r.map(() => new Array(6).fill(0));
    for (let k = 0; k < 6; k++) {                          // jacobiano por diferencias centrales
      const h = k < 3 ? 1e-6 : 1e-6 * Math.max(1, Math.abs(p[k]));
      const mas = [...p], menos = [...p];
      mas[k] += h; menos[k] -= h;
      const rm = residuos(mas, img, ancho, alto), rn = residuos(menos, img, ancho, alto);
      for (let i = 0; i < r.length; i++) J[i][k] = (rm[i] - rn[i]) / (2 * h);
    }
    const JtJ = Array.from({ length: 6 }, (_, a) => Array.from({ length: 6 }, (_, b) => J.reduce((s, f) => s + f[a] * f[b], 0)));
    const Jtr = Array.from({ length: 6 }, (_, a) => J.reduce((s, f, i) => s + f[a] * r[i], 0));
    let mejoro = false;
    for (let intento = 0; intento < 10 && !mejoro; intento++) {
      const A = JtJ.map((fila, a) => fila.map((v, b) => (a === b ? v * (1 + lambda) : v)));
      const d = resolver(A, Jtr.map((v) => -v));
      if (!d) break;
      const q = p.map((v, k) => v + d[k]);
      const rq = residuos(q, img, ancho, alto), cq = rq.reduce((s, v) => s + v * v, 0);
      if (Number.isFinite(cq) && cq < costo) {
        const paso = Math.max(...d.map(Math.abs));
        p = q; r = rq; mejoro = true;
        const relativo = (costo - cq) / Math.max(costo, 1e-12);
        costo = cq; lambda = Math.max(lambda / 10, 1e-12);
        if (paso < 1e-10 || relativo < 1e-14) return p;
      } else lambda *= 10;
    }
    if (!mejoro) break;
  }
  return p;
}

// Pose inicial sin pose anterior (OpenCV usaba EPnP): cara de frente con el giro de la línea de los ojos y la
// distancia que da el alto de la cara; el refinamiento hace el resto
function poseInicial(img, ancho, alto) {
  const ojoD = img[CARA_PNP.indexOf(33)], ojoI = img[CARA_PNP.indexOf(263)];
  const frente = img[CARA_PNP.indexOf(10)], menton = img[CARA_PNP.indexOf(152)], nariz = img[0];
  const giro = Math.atan2(ojoI[1] - ojoD[1], ojoI[0] - ojoD[0]);
  const altoModelo = MODELO_CARA[152][1] - MODELO_CARA[10][1];
  const z = ancho * altoModelo / Math.max(dist(frente, menton), 1);
  const m = MODELO_CARA[1], zn = z + m[2];
  return [0, 0, giro, (nariz[0] - ancho / 2) * zn / ancho - m[0], (nariz[1] - alto / 2) * zn / ancho - m[1], z];
}

export class EstimadorPoseCabeza {
  constructor() { this.rvec = this.tvec = null; }
  reiniciar() { this.rvec = this.tvec = null; }
  estimar(f, ancho, alto) {
    const img = CARA_PNP.map((i) => [f[i][0], f[i][1]]);
    const inicio = this.rvec === null ? poseInicial(img, ancho, alto) : [...this.rvec, ...this.tvec];
    const p = refinar(inicio, img, ancho, alto);
    const rvec = p.slice(0, 3), tvec = p.slice(3);
    if (!rvec.every(Number.isFinite) || !(tvec[2] > 0)) { this.reiniciar(); return null; }
    this.rvec = rvec; this.tvec = tvec;
    const R = rodrigues(rvec);
    const frente = [-R[0][2], -R[1][2], -R[2][2]];
    const arriba = [-R[0][1], -R[1][1], -R[2][1]];
    const pr = proyectar(MODELO_PNP, rvec, tvec, ancho, alto);
    const error = pr.reduce((s, q, i) => s + Math.hypot(q[0] - img[i][0], q[1] - img[i][1]), 0) / pr.length
      / Math.max(dist(f[FRENTE], f[MENTON]), 1.0);
    return {
      pitch: grados(Math.asin(Math.max(-1, Math.min(1, frente[1])))),
      yaw: grados(Math.atan2(frente[0], -frente[2])),
      roll: grados(Math.atan2(arriba[0], -arriba[1])),
      distancia: tvec[2], error, rvec, tvec,
    };
  }
}

// ===================================================================== cejas y ojos
function alturaCejas(f) {
  const altoCara = dist(f[FRENTE], f[MENTON]);
  if (altoCara < 1e-6) return null;
  const eje = unitario(restar(f[FRENTE], f[MENTON]));
  const altura = (ref, ids) => ids.reduce((s, i) => s + (f[i][0] - ref[0]) * eje[0] + (f[i][1] - ref[1]) * eje[1], 0) / ids.length;
  const refDer = puntoMedio(f[CANTO_DER[0]], f[CANTO_DER[1]]);
  const refIzq = puntoMedio(f[CANTO_IZQ[0]], f[CANTO_IZQ[1]]);
  return (altura(refDer, CEJA_SUP_DER) + altura(refIzq, CEJA_SUP_IZQ)) / 2 / altoCara;
}

function cejasEsperadas(pose, ancho, alto) {
  const aMapa = (pr) => Object.fromEntries(PUNTOS_CEJAS.map((i, k) => [i, pr[k]]));
  const a = alturaCejas(aMapa(proyectar(MODELO_CEJAS, pose.rvec, pose.tvec, ancho, alto)));
  const b = alturaCejas(aMapa(proyectar(MODELO_CEJAS, [0, 0, 0], pose.tvec, ancho, alto)));
  return a && b ? a / b : 1.0;
}

function calcularRasgosFaciales(f, pose, ancho, alto) {
  const medida = alturaCejas(f);
  if (medida === null || medida <= 0) return null;
  const esperada = pose !== null ? cejasEsperadas(pose, ancho, alto) : 1.0;
  const anchoOjos = dist(f[CANTO_DER[0]], f[CANTO_DER[1]]) + dist(f[CANTO_IZQ[0]], f[CANTO_IZQ[1]]);
  const ojos = (dist(f[PARPADO_DER[0]], f[PARPADO_DER[1]]) + dist(f[PARPADO_IZQ[0]], f[PARPADO_IZQ[1]])) / Math.max(anchoOjos, 1e-6);
  return { cejas: medida / Math.max(esperada, 1e-6), cejas2d: medida, ojos };
}

function percentil(xs, q) {                                // np.percentile (interpolación lineal)
  const s = [...xs].sort((a, b) => a - b), pos = (s.length - 1) * q / 100, i = Math.floor(pos);
  return i + 1 < s.length ? s[i] + (s[i + 1] - s[i]) * (pos - i) : s[i];
}

// ===================================================================== negar / asentir
export class DetectorOscilacion {
  constructor(amplitud, entradasMin, requiereAmbosLados) {
    this.amplitud = amplitud; this.entradasMin = entradasMin; this.requiereAmbosLados = requiereAmbosLados;
    this.reiniciar();
  }
  reiniciar() { this.centro = null; this.tAnterior = null; this.zona = 0; this.entradas = []; this.desviacion = 0.0; }
  actualizar(t, valor) {
    if (this.centro === null) this.centro = valor;
    else {
      const dt = Math.max(0.0, t - this.tAnterior);
      this.centro += (valor - this.centro) * Math.min(1.0, dt / OSC_TAU_CENTRO);
    }
    this.tAnterior = t;
    const d = valor - this.centro;
    this.desviacion = d;
    let nueva;
    if (this.zona === 0) nueva = d > this.amplitud ? 1 : (d < -this.amplitud ? -1 : 0);
    else if (this.zona * d < this.amplitud / 2) nueva = Math.abs(d) > this.amplitud ? -this.zona : 0;
    else nueva = this.zona;
    if (nueva !== 0 && nueva !== this.zona) this.entradas.push([t, nueva]);
    this.zona = nueva;
    while (this.entradas.length && t - this.entradas[0][0] > OSC_VENTANA) this.entradas.shift();
    const lados = new Set(this.entradas.map((e) => e[1]));
    if (this.entradas.length >= this.entradasMin && (!this.requiereAmbosLados || lados.size === 2)) {
      this.entradas = [];
      return true;
    }
    return false;
  }
}

export class AnalizadorFacial {
  constructor() {
    this.pose = new EstimadorPoseCabeza();
    this.negacion = new DetectorOscilacion(NEG_AMPLITUD, NEG_ENTRADAS_MIN, true);
    this.afirmacion = new DetectorOscilacion(AFIRM_AMPLITUD, AFIRM_ENTRADAS_MIN, false);
    this.ultimoEvento = null; this.tEvento = -10.0;
    this.enfriamientoHasta = 0.0;
    this.cejasPausaHasta = 0.0;
    this.rasgos = null;
    this.poseActual = null;
    this.modo = "SIN_ROSTRO";
    this.recalibrar();
  }
  recalibrar() {
    this.muestrasCalibracion = [];
    this.baseCejas = null; this.baseOjos = null; this.poseNeutra = null;
    this.ratioCejas = 1.0; this.ratioOjos = 1.0;
    this.preguntaHecha = false;
    this.reiniciarCejas();
  }
  reiniciarCejas() { this.cejasInicio = null; this.ultimaArriba = -10.0; this.progresoPregunta = 0.0; }
  get progresoCalibracion() {
    return this.baseCejas !== null ? 1.0 : this.muestrasCalibracion.length / FRAMES_CALIBRACION;
  }
  perderRostro() {
    this.rasgos = null; this.poseActual = null;
    this.pose.reiniciar(); this.negacion.reiniciar(); this.afirmacion.reiniciar();
    this.reiniciarCejas();
  }
  // puntos: {índice: [x, y, z]} en píxeles (o null sin cara). Devuelve "PREGUNTA" | "NEGACION" | "AFIRMACION" | null
  actualizar(puntos, t, [ancho, alto]) {
    const pose = puntos !== null ? this.pose.estimar(puntos, ancho, alto) : null;
    if (pose !== null && pose.error > CARA_ERROR_MAX) {
      this.perderRostro();
      this.poseActual = pose;
      this.modo = "TAPADO";
      return null;
    }
    const rasgos = pose !== null ? calcularRasgosFaciales(puntos, pose, ancho, alto) : null;
    if (rasgos === null) { this.perderRostro(); this.modo = "SIN_ROSTRO"; return null; }
    this.rasgos = rasgos; this.poseActual = pose;

    if (this.baseCejas === null) {
      this.muestrasCalibracion.push([rasgos.cejas, rasgos.ojos, pose.pitch, pose.yaw]);
      if (this.muestrasCalibracion.length >= FRAMES_CALIBRACION) {
        const col = (j) => this.muestrasCalibracion.map((m) => m[j]);
        this.baseCejas = percentil(col(0), CALIB_PERCENTIL);
        this.baseOjos = mediana(col(1));
        this.poseNeutra = [mediana(col(2)), mediana(col(3))];
      }
      this.modo = "CALIBRANDO";
      return null;
    }

    let evento = null;
    const negar = this.negacion.actualizar(t, pose.yaw);
    const asentir = this.afirmacion.actualizar(t, pose.pitch);
    if (t >= this.enfriamientoHasta && (negar || asentir)) {
      evento = negar ? "NEGACION" : "AFIRMACION";
      this.enfriamientoHasta = t + ENFRIAMIENTO_CABEZA;
      this.negacion.entradas = []; this.afirmacion.entradas = [];
      this.cejasPausaHasta = t + CEJAS_PAUSA_TRAS_CABEZA;
      this.reiniciarCejas();
    }

    this.ratioCejas = rasgos.cejas / this.baseCejas;
    this.ratioOjos = rasgos.ojos / Math.max(this.baseOjos, 1e-6);
    if (t >= this.cejasPausaHasta) {
      if (this.ratioCejas >= CEJAS_UMBRAL) {
        this.ultimaArriba = t;
        if (this.cejasInicio === null) this.cejasInicio = t;
      } else if (this.ratioCejas >= CEJAS_UMBRAL_SALIDA && this.cejasInicio !== null) this.ultimaArriba = t;
      if (this.cejasInicio !== null && t - this.ultimaArriba > CEJAS_TOLERANCIA) this.reiniciarCejas();
      if (this.cejasInicio !== null && t - this.cejasInicio > CEJAS_MAX_SEGUNDOS) {
        this.baseCejas = rasgos.cejas;
        this.ratioCejas = 1.0;
        this.reiniciarCejas();
      }
      if (this.cejasInicio !== null) {
        this.progresoPregunta = Math.min(1.0, (t - this.cejasInicio) / CEJAS_SEGUNDOS);
        if (this.progresoPregunta >= 1.0 && !this.preguntaHecha && evento === null) {
          evento = "PREGUNTA";
          this.preguntaHecha = true;
        }
      } else if (this.preguntaHecha && this.ratioCejas < CEJAS_UMBRAL_SALIDA && t - this.ultimaArriba > CEJAS_REARME) {
        this.preguntaHecha = false;
      }
      if (this.cejasInicio === null && Math.abs(this.ratioCejas - 1.0) < CEJAS_ZONA_ADAPT) {
        this.baseCejas += (rasgos.cejas - this.baseCejas) * CEJAS_ADAPTACION;
      }
    }

    if (evento) { this.ultimoEvento = evento; this.tEvento = t; }
    if (this.ultimoEvento && t - this.tEvento < DURACION_AVISO) this.modo = this.ultimoEvento;
    else if (this.cejasInicio !== null && !this.preguntaHecha) this.modo = "CEJAS";
    else this.modo = "NEUTRO";
    return evento;
  }
}

// Salida de FaceLandmarker (MediaPipe Tasks JS) -> {índice: [x, y, z]} en píxeles, solo los puntos que se usan
export function leerRostro(resultado, ancho, alto) {
  const lm = resultado?.faceLandmarks?.[0];
  if (!lm) return null;
  const puntos = {};
  for (const i of INDICES_ROSTRO) puntos[i] = [lm[i].x * ancho, lm[i].y * alto, lm[i].z * ancho];
  return puntos;
}
