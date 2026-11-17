// Gestos dinámicos: palabras y frases del dataset (port de python/gestos.py). Mismos umbrales (los rangos medidos
// están comentados allá). Diferencias del navegador: las plantillas llegan ya como descriptor (web/datos/,
// python/exportar_web.py), el DTW corre al cerrar el gesto en una tarea aparte (no frena ese fotograma) y las
// muestras propias (botón Guardar / tecla G) se guardan en localStorage en vez de dataset/.
import { LADOS } from "./manos.js";
import { EJECUCION_MIN, FPS_MAX, F, MIN_FRAMES, REPOSO_MIN, REPOSO_Y, VENTANA, aLargoFijo, inicioMano, limpiar,
  vectorFotograma } from "./vector.js";

const REPOSO_S = REPOSO_MIN / FPS_MAX;
const GESTO_MIN_S = EJECUCION_MIN / FPS_MAX;
const GESTO_MAX_S = 10.0;
const PESO_TRAYECTORIA = 0.5;
const CASTIGO_MANO = 1.0;
const BANDA = 8;
const MARGEN = 1.10;
const DIST_MAX = 2.2;
const DIST_MAX_TRADUCIR = 1.0;
const SEPARACION_MIN = 0.2;
const SOLO_DELETREO = /^(?:[a-zñ]|ll|rr|\d.*)$/u;
const SEPARACION_MIN_LARGA = 0.08;
const DIST_CLARA = 0.55;
const SEPARACION_MIN_CLARA = 0.03;
const V_PAUSA = 2.5;
const PAUSA_TRAS_S = 0.8;
const V_LENTO = 1.0;
const PAUSA_CIERRE_S = 0.5;
const CUADROS_MAX = Math.trunc(GESTO_MAX_S * FPS_MAX);
const ALMACEN = "lsm-muestras";                            // localStorage: {palabra: [base64 de 30 x F float32]}
const VALORES = VENTANA * 2 + VENTANA * 2 * 63 + VENTANA * 2 * 2;
const RAIZ21 = Math.sqrt(21);

const pyNum = (x) => (Number.isInteger(x) ? x.toFixed(1) : String(x));    // 1.0 como lo imprime Python
const redondear = (x, d) => Math.round(x * 10 ** d) / 10 ** d;
export const textoFrase = (etiqueta) => etiqueta.replaceAll("_", " ").toUpperCase();
function mediana(xs) {
  const s = Float64Array.from(xs).sort(), k = s.length >> 1;
  return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2;
}

// Muestra 30 x F -> {pres (T x 2), forma (T x 2 x 63), tray (T x 2 x 2)} en arreglos planos
export function descriptor(m) {
  const T = m.length, K = LADOS.length;
  const pres = new Uint8Array(T * K), forma = new Float64Array(T * K * 63), tray = new Float64Array(T * K * 2);
  for (let t = 0; t < T; t++) for (let k = 0; k < K; k++) pres[t * K + k] = m[t][k] > 0 ? 1 : 0;
  const img = (t, k, i, j) => m[t][inicioMano(k) + 3 * i + j];
  const mundo = (t, k, i, j) => m[t][inicioMano(k) + 63 + 3 * i + j];
  const tams = [], munecas = [];
  for (let k = 0; k < K; k++) for (let t = 0; t < T; t++) {
    if (pres[t * K + k]) tams.push(Math.hypot(img(t, k, 9, 0) - img(t, k, 0, 0), img(t, k, 9, 1) - img(t, k, 0, 1)));
  }
  for (let k = 0; k < K; k++) for (let t = 0; t < T; t++) if (pres[t * K + k]) munecas.push([img(t, k, 0, 0), img(t, k, 0, 1)]);
  const tamImg = tams.length ? mediana(tams) : 1.0;
  const centro = munecas.length ? [0, 1].map((j) => munecas.reduce((s, q) => s + q[j], 0) / munecas.length) : [0, 0];
  for (let k = 0; k < K; k++) {
    const ts = [];
    for (let t = 0; t < T; t++) if (pres[t * K + k]) ts.push(t);
    if (!ts.length) continue;
    const tam = mediana(ts.map((t) => Math.hypot(...[0, 1, 2].map((j) => mundo(t, k, 9, j) - mundo(t, k, 0, j))))) || 1.0;
    for (const t of ts) {
      for (let i = 0; i < 21; i++) for (let j = 0; j < 3; j++) {
        forma[(t * K + k) * 63 + 3 * i + j] = (mundo(t, k, i, j) - mundo(t, k, 0, j)) / tam;
      }
      for (let j = 0; j < 2; j++) tray[(t * K + k) * 2 + j] = (img(t, k, 0, j) - centro[j]) / Math.max(tamImg, 1e-6);
    }
  }
  return { pres, forma, tray, cuad: cuadrados(forma, T * K) };
}
function cuadrados(forma, n) {                              // |forma|² de cada fotograma y mano (para el DTW)
  const c = new Float64Array(n);
  for (let r = 0; r < n; r++) { let s = 0; for (let f = 0; f < 63; f++) s += forma[r * 63 + f] ** 2; c[r] = s; }
  return c;
}

// DTW (banda de BANDA fotogramas) de un descriptor contra una plantilla: costo medio por paso del camino
export function distancia(a, b) {
  const n = a.pres.length / 2, m = b.pres.length / 2, K = 2;
  let ant = new Float64Array(m + 1).fill(Infinity), act = new Float64Array(m + 1);
  ant[0] = 0.0;
  for (let i = 1; i <= n; i++) {
    act.fill(Infinity);
    for (let j = Math.max(1, i - BANDA); j <= Math.min(m, i + BANDA); j++) {
      let D = 0.0;
      for (let k = 0; k < K; k++) {
        const ra = (i - 1) * K + k, rb = (j - 1) * K + k, pa = a.pres[ra], pb = b.pres[rb];
        let termino = 0.0;
        if (pa && pb) {
          let prod = 0;
          for (let f = 0; f < 63; f++) prod += a.forma[ra * 63 + f] * b.forma[rb * 63 + f];
          const df = Math.sqrt(Math.max(a.cuad[ra] + b.cuad[rb] - 2 * prod, 0.0)) / RAIZ21;
          const dt = Math.hypot(a.tray[ra * 2] - b.tray[rb * 2], a.tray[ra * 2 + 1] - b.tray[rb * 2 + 1]);
          termino = df + PESO_TRAYECTORIA * dt;
        }
        D += termino + (pa !== pb ? CASTIGO_MANO : 0.0);
      }
      act[j] = D + Math.min(ant[j], act[j - 1], ant[j - 1]);
    }
    [ant, act] = [act, ant];
  }
  return ant[m] / (n + m);
}

// Plantillas exportadas (plantillas.bin / .json) -> [{etiqueta, descriptor}]
export function leerPlantillas(buffer, meta) {
  const datos = new Float32Array(buffer), T = meta.ventana, K = 2;
  return meta.etiquetas.map((etiqueta, n) => {
    const base = n * VALORES;
    const pres = Uint8Array.from(datos.subarray(base, base + T * K), (v) => (v > 0 ? 1 : 0));
    const forma = Float64Array.from(datos.subarray(base + T * K, base + T * K + T * K * 63));
    const tray = Float64Array.from(datos.subarray(base + T * K + T * K * 63, base + VALORES));
    return { etiqueta, d: { pres, forma, tray, cuad: cuadrados(forma, T * K) } };
  });
}

function leerAlmacen() {
  try { return JSON.parse(localStorage.getItem(ALMACEN) || "{}"); } catch (_) { return {}; }
}
const aBase64 = (muestra) => {
  const plano = new Float32Array(muestra.length * F);
  muestra.forEach((f, t) => plano.set(f, t * F));
  return btoa(String.fromCharCode(...new Uint8Array(plano.buffer)));
};
const deBase64 = (texto) => {
  const bytes = Uint8Array.from(atob(texto), (c) => c.charCodeAt(0));
  const plano = new Float32Array(bytes.buffer);
  return Array.from({ length: plano.length / F }, (_, t) => plano.subarray(t * F, (t + 1) * F));
};

export class ReconocedorGestos {
  // plantillas: de leerPlantillas. asincrono: el DTW corre en una tarea aparte y el resultado se recoge en un
  // fotograma siguiente (en vivo); false lo evalúa en el mismo fotograma (pruebas reproducibles)
  constructor(plantillas, { asincrono = true, propias = true, registrar = null } = {}) {
    this.plantillas = [...plantillas];
    this.usarAlmacen = propias;
    if (propias) {
      for (const [palabra, muestras] of Object.entries(leerAlmacen())) {
        for (const texto of muestras) this.plantillas.push({ etiqueta: palabra, d: descriptor(deBase64(texto)) });
      }
    }
    this.etiquetas = [...new Set(this.plantillas.map((p) => p.etiqueta))].sort();
    this.asincrono = asincrono;
    this.registrar = registrar;                             // registrar(r): cada evaluación (consola)
    this.pendiente = null;
    this.ultima = null;
    this.aviso = null;
    this.propias = 0;
    this.objetivo = null;
    this.cuadros = [];
    this.tAbajo = null;
    this.resultado = null;
    this.munecas = {};
    this.rapidoHasta = -1.0;
    this.primerMovimiento = false;
    this.huboMovimiento = false;
    this.tLento = null;
  }
  get grabando() { return this.cuadros.length > 0; }
  pausaLetras(ahora) { return ahora < this.rapidoHasta || (this.grabando && this.huboMovimiento); }

  ponerObjetivo(objetivo) {
    this.objetivo = this.etiquetas.includes(objetivo) ? objetivo : null;
    this.cuadros = [];
    this.tAbajo = null; this.resultado = null; this.tLento = null; this.huboMovimiento = false;
    this.pendiente = null; this.ultima = null;
    this.propias = this.objetivo ? this.contarPropias(this.objetivo) : 0;
  }
  contarPropias(palabra) { return this.usarAlmacen ? (leerAlmacen()[palabra] ?? []).length : 0; }

  guardarMuestra() {
    if (this.ultima === null || this.ultima[0] === null) {
      this.aviso = this.objetivo === null ? "Elige una frase en Práctica > Frases, hazla completa y luego guarda"
        : "Primero haz la frase completa (hasta que se evalúe) y luego guarda";
      return;
    }
    const [palabra, muestra] = this.ultima;
    const almacen = leerAlmacen();
    (almacen[palabra] ??= []).push(aBase64(muestra));
    try { localStorage.setItem(ALMACEN, JSON.stringify(almacen)); }
    catch (_) { this.aviso = "No se pudo guardar: el navegador no deja usar su almacenamiento"; return; }
    this.plantillas.push({ etiqueta: palabra, d: descriptor(muestra) });
    this.ultima = null;
    this.propias = almacen[palabra].length;
    this.aviso = `Guardada la muestra ${this.propias} en ${textoFrase(palabra)}: ${this.propias} muestras tuyas (en este navegador)`;
  }

  velocidad(v, ahora) {
    let maxima = null;
    LADOS.forEach((_, k) => {
      if (v[k] === 0) { delete this.munecas[k]; return; }
      const b = inicioMano(k);
      const tam = Math.max(Math.hypot(v[b + 27] - v[b], v[b + 28] - v[b + 1]), 1e-6);
      const previa = this.munecas[k];
      this.munecas[k] = [ahora, v[b], v[b + 1]];
      if (previa === undefined || !(0 < ahora - previa[0] && ahora - previa[0] < 0.3)) return;
      const rapidez = Math.hypot(v[b] - previa[1], v[b + 1] - previa[2]) / tam / (ahora - previa[0]);
      maxima = maxima === null ? rapidez : Math.max(maxima, rapidez);
      if (rapidez > V_PAUSA) {
        if (this.cuadros.length && ahora >= this.rapidoHasta) this.primerMovimiento = true;
        if (this.cuadros.length) this.huboMovimiento = true;
        this.rapidoHasta = ahora + PAUSA_TRAS_S;
      }
    });
    if (maxima !== null) this.tLento = maxima < V_LENTO ? (this.tLento || ahora) : null;
  }
  tomarPrimerMovimiento() { const a = this.primerMovimiento; this.primerMovimiento = false; return a; }

  clasificar(muestra) {
    const d = descriptor(muestra), mejor = new Map();
    for (const e of this.etiquetas) mejor.set(e, Infinity);
    for (const p of this.plantillas) mejor.set(p.etiqueta, Math.min(mejor.get(p.etiqueta), distancia(d, p.d)));
    return [...mejor.entries()].sort((x, y) => x[1] - y[1]);
  }

  paso(asignadas, ahora, [ancho, alto]) {
    if (!this.plantillas.length) return;
    if (this.pendiente?.listo) { const { base, ranking } = this.pendiente; this.pendiente = null; this.evaluar(base, ranking); }
    const v = vectorFotograma(asignadas, ancho, alto);
    this.velocidad(v, ahora);
    const y = LADOS.map((_, k) => k).filter((k) => v[k] > 0).map((k) => v[inicioMano(k) + 1]);
    const arriba = y.length > 0 && Math.min(...y) < REPOSO_Y;
    if (arriba) {
      this.agregarCuadro([ahora, v, true]);
      this.tAbajo = null;
      if (this.huboMovimiento && this.tLento !== null && ahora - this.tLento >= PAUSA_CIERRE_S) this.cerrar("pausa");
    } else if (this.cuadros.length) {
      this.agregarCuadro([ahora, v, false]);
      this.tAbajo = this.tAbajo || ahora;
      if (ahora - this.tAbajo >= REPOSO_S) this.cerrar("manos abajo");
    }
    if (this.cuadros.length && ahora - this.cuadros[0][0] > GESTO_MAX_S) this.cerrar("tiempo máximo");
  }
  agregarCuadro(c) { this.cuadros.push(c); if (this.cuadros.length > CUADROS_MAX) this.cuadros.shift(); }

  cerrar(cierre) {
    let cuadros = this.cuadros;
    let ultimo = null;
    cuadros.forEach((c, i) => { if (c[2]) ultimo = i; });
    this.cuadros = [];
    const movimiento = this.huboMovimiento;
    this.tAbajo = null; this.tLento = null; this.huboMovimiento = false;
    if (ultimo === null) return;
    cuadros = cuadros.slice(0, ultimo + 1);
    const segundos = cuadros[cuadros.length - 1][0] - cuadros[0][0];
    const base = { objetivo: this.objetivo, cierre, segundos: redondear(segundos, 1), mejor: null, distancia: null,
      separacion: null, aceptado: false, acierto: null, escrito: null, distancia_objetivo: null, motivo: "", top: [],
      movimiento };
    if (segundos < GESTO_MIN_S) return this.reportar({ ...base, motivo: `duró ${segundos.toFixed(1)} s (mínimo ${GESTO_MIN_S.toFixed(0)} s)` });
    const limpia = limpiar(cuadros.map((c) => c[1]));
    if (limpia.length < MIN_FRAMES) return this.reportar({ ...base, motivo: `solo ${limpia.length} fotogramas con manos` });
    const muestra = aLargoFijo(limpia, VENTANA);
    this.ultima = [this.objetivo, muestra];
    if (!this.asincrono) return this.evaluar(base, this.clasificar(muestra));
    const tarea = { base, ranking: null, listo: false };   // el DTW no frena este fotograma: se recoge en paso()
    this.pendiente = tarea;
    setTimeout(() => { tarea.ranking = this.clasificar(muestra); tarea.listo = true; }, 0);
  }

  evaluar(base, ranking) {
    const [mejor, d1] = ranking[0], d2 = ranking.length > 1 ? ranking[1][1] : Infinity;
    const separacion = d2 > 0 ? 1 - d1 / d2 : 0.0;
    let sepMin = (mejor.split("_").length - 1) >= 2 ? SEPARACION_MIN_LARGA : SEPARACION_MIN;
    if (d1 <= DIST_CLARA && mejor.includes("_")) sepMin = Math.min(sepMin, SEPARACION_MIN_CLARA);
    let aceptado = d1 <= DIST_MAX_TRADUCIR && separacion >= sepMin;
    let motivo = aceptado ? "" : d1 > DIST_MAX_TRADUCIR ? `distancia ${d1.toFixed(2)} > ${pyNum(DIST_MAX_TRADUCIR)}`
      : `separación ${separacion.toFixed(2)} < ${pyNum(sepMin)}`;
    if (aceptado && SOLO_DELETREO.test(mejor)) { aceptado = false; motivo = "letra o número: lo escribe el deletreo"; }
    else if (aceptado && !base.movimiento) { aceptado = false; motivo = "sin movimiento de la muñeca (seña quieta: la escribe el deletreo)"; }
    let acierto = null, dObjetivo = null;
    const objetivo = base.objetivo;
    if (objetivo !== null) {
      dObjetivo = new Map(ranking).get(objetivo);
      acierto = dObjetivo <= DIST_MAX && dObjetivo <= d1 * MARGEN;
      motivo = acierto ? "" : dObjetivo > DIST_MAX ? `distancia ${dObjetivo.toFixed(2)} > ${pyNum(DIST_MAX)}`
        : `más cerca de ${mejor} (${d1.toFixed(2)})`;
    }
    this.reportar({ ...base, mejor, distancia: redondear(d1, 3), separacion: redondear(separacion, 2), aceptado, acierto,
      motivo, distancia_objetivo: dObjetivo === null ? null : redondear(dObjetivo, 3),
      top: ranking.slice(0, 3).map(([e, d]) => [e, redondear(d, 3)]) });
  }
  reportar(r) { this.resultado = r; this.registrar?.(r); }
  sacarResultado() { const r = this.resultado; this.resultado = null; return r; }

  web(ahora) {
    if (!this.plantillas.length) return null;
    const aviso = this.aviso;
    this.aviso = null;
    return { objetivo: this.objetivo, fase: this.grabando ? "grabando" : "espera",
      segundos: this.grabando ? redondear(ahora - this.cuadros[0][0], 1) : 0.0, pausa: this.pausaLetras(ahora),
      resultado: this.sacarResultado(), muestra: this.ultima !== null && this.ultima[0] !== null,
      propias: this.propias, aviso };
  }
}
