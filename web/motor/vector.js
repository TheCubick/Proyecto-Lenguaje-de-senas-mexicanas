// Vector por fotograma y remuestreo (port de las funciones de python/video_extractor.py que usa gestos.py).
// Formato v1, F = 254: presente(2) + por lado Izq, Der: img (21 x 3: x/ancho, y/alto, z/ancho) + mundo (21 x 3 m).
// Los vectores son Float32Array, igual que en NumPy (float32): mismas roundings al guardar.
import { LADOS } from "./manos.js";

export const VENTANA = 30;
export const FPS_MAX = 30.0;
const HUECO_MAX = 6;
export const MIN_FRAMES = 10;
export const REPOSO_Y = 0.8;
export const REPOSO_MIN = 12;
export const EJECUCION_MIN = 30;
export const F = 2 + LADOS.length * 2 * 63;

export const inicioMano = (k) => 2 + k * 126;               // columnas_mano(k) = [inicio, inicio + 126)

export function vectorFotograma(asignadas, ancho, alto) {
  const v = new Float32Array(F);
  LADOS.forEach((lado, k) => {
    const d = asignadas[lado];
    if (!d) return;
    v[k] = 1.0;
    const base = inicioMano(k);
    d.img.forEach((q, i) => { v[base + 3 * i] = q[0] / ancho; v[base + 3 * i + 1] = q[1] / alto; v[base + 3 * i + 2] = q[2] / ancho; });
    d.mundo.forEach((q, i) => { for (let j = 0; j < 3; j++) v[base + 63 + 3 * i + j] = q[j]; });
  });
  return v;
}

// Quita los fotogramas sin manos de los extremos y rellena huecos cortos de cada mano interpolando entre vecinos
export function limpiar(secuencia, huecoMax = HUECO_MAX) {
  const hay = [];
  secuencia.forEach((f, i) => { if (Math.max(f[0], f[1]) > 0) hay.push(i); });
  if (!hay.length) return [];
  const s = secuencia.slice(hay[0], hay[hay.length - 1] + 1).map((f) => Float32Array.from(f));
  LADOS.forEach((_, k) => {
    const presentes = [];
    s.forEach((f, i) => { if (f[k] > 0) presentes.push(i); });
    const ini = inicioMano(k);
    for (let n = 0; n + 1 < presentes.length; n++) {
      const i = presentes[n], j = presentes[n + 1];
      if (1 < j - i && j - i <= huecoMax + 1) {
        for (let t = 1; t < j - i; t++) {
          const w = t / (j - i);
          for (let c = ini; c < ini + 126; c++) s[i + t][c] = (1 - w) * s[i][c] + w * s[j][c];
          s[i + t][k] = 1.0;
        }
      }
    }
  });
  return s;
}

// T fotogramas -> exactamente `largo` con interpolación lineal (np.linspace + interpolar)
export function aLargoFijo(secuencia, largo = VENTANA) {
  const T = secuencia.length, paso = (T - 1) / (largo - 1);
  const pos = Array.from({ length: largo }, (_, i) => i * paso);
  pos[largo - 1] = T - 1;
  return interpolar(secuencia, pos);
}

function interpolar(secuencia, posiciones) {
  const T = secuencia.length;
  return posiciones.map((p0) => {
    const pos = Math.min(Math.max(p0, 0), T - 1);
    const i0 = Math.floor(pos), i1 = Math.min(i0 + 1, T - 1), w = pos - i0;
    const a = secuencia[i0], b = secuencia[i1];
    const salida = new Float32Array(F);
    for (let c = 0; c < F; c++) salida[c] = (1 - w) * a[c] + w * b[c];
    const cercano = w < 0.5 ? a : b;
    LADOS.forEach((_, k) => {                              // una mano solo se interpola si está en ambos
      if ((a[k] > 0) !== (b[k] > 0)) {
        salida[k] = cercano[k];
        const ini = inicioMano(k);
        for (let c = ini; c < ini + 126; c++) salida[c] = cercano[c];
      }
    });
    return salida;
  });
}
