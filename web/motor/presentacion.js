// Qué se muestra de cada mano y el estado JSON por fotograma que consume static/app.js (port de python/ui_panel.py).
// La configuración (guía del alfabeto, huesos, bases de los trazos, frases) llega hecha en web/datos/config.json.
import { LADOS, LETRAS_TIPEABLES, POSES_SILENCIOSAS, PUNTO_DE_POSE } from "./manos.js";

const DURACION_DESTELLO = 1.2;
const ESTELA_MAX = 32;
const PISTA_TRAZO = { I: "o traza una J", Y: "o traza una J", D: "o traza Z / ?", P: "o agita para K",
  R: "o agita para K", U: "o agita para K", N: "o agita para Ñ", Q: "balancea para Q", X: "mueve para X",
  B: "o barre al centro", ESPACIO: "o barre al centro" };

const r4 = (x) => Math.round(x * 1e4) / 1e4;
const r1 = (x) => Math.round(x * 10) / 10;
export const limpiarTexto = (texto) => String(texto).replaceAll("_", " ");

export function vistaMano(em, sostener, ahora) {
  const [gesto, tGesto] = sostener.destello;
  if (gesto && ahora - tGesto < DURACION_DESTELLO) return { sena: gesto, estado: "registrada", progreso: 1.0 };
  if (em.pose !== null && em.rasgos !== null && !POSES_SILENCIOSAS.has(em.pose)) {
    if (!LETRAS_TIPEABLES.has(em.pose)) return { sena: em.pose, estado: "pose", progreso: em.progreso };
    if (em.moviendo) return { sena: em.pose, estado: "mover", progreso: em.progreso };
    return { sena: em.pose, estado: sostener.registrada ? "registrada" : "sostener", progreso: em.progreso };
  }
  return { sena: null, estado: "ninguna", progreso: 0.0 };
}

// La mano que se muestra en grande: la que acaba de registrar o la que más avanzó; en empate, la anterior
export function manoPrincipal(vistas, anterior) {
  const candidatas = LADOS.filter((l) => vistas[l].sena !== null);
  if (!candidatas.length) return null;
  const clave = (l) => [vistas[l].estado === "registrada" ? 1 : 0, vistas[l].progreso, l === anterior ? 1 : 0];
  return candidatas.reduce((mejor, l) => {
    const a = clave(l), b = clave(mejor);
    for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] > b[i] ? l : mejor;
    return mejor;                                           // max() de Python: en empate total, el primero
  });
}

function manoWeb(em, vista, tray, ancho, alto) {
  let caja = null, puntos = null, estela = null;
  if (em.rasgos !== null) {
    const xs = em.rasgos.p.map((q) => q[0]), ys = em.rasgos.p.map((q) => q[1]);
    caja = [r4(Math.min(...xs) / ancho), r4(Math.min(...ys) / alto), r4(Math.max(...xs) / ancho), r4(Math.max(...ys) / alto)];
    if (em.img !== null) puntos = em.img.map((q) => [r4(q[0] / ancho), r4(q[1] / alto), r4(q[2] / ancho)]);
    if (vista.sena in PUNTO_DE_POSE && tray.vel > 0.6) {
      const serie = tray.seriePx(PUNTO_DE_POSE[vista.sena]);
      const paso = Math.max(1, Math.floor(serie.length / ESTELA_MAX));
      estela = serie.filter((_, i) => i % paso === 0).map((q) => [r4(q[0] / ancho), r4(q[1] / alto)]);
    }
  }
  const pista = ["sostener", "registrada", "mover", "pose"].includes(vista.estado) ? (PISTA_TRAZO[vista.sena] ?? null) : null;
  return { sena: vista.sena, estado: vista.estado, progreso: r1(vista.progreso * 100), caja, pista, puntos, estela };
}

function lineasDiagnostico(manos, canales, analizador) {
  const lineas = [], f2 = (x) => x.toFixed(2), f0 = (x) => x.toFixed(0);
  const s2 = (x) => (x >= 0 ? "+" : "") + x.toFixed(2), s1 = (x) => (x >= 0 ? "+" : "") + x.toFixed(1);
  for (const lado of LADOS) {
    const r = manos[lado].rasgos;
    if (!r) continue;
    lineas.push(`${lado}: ` + r.estados.map((e, i) => `${e}${f2(r.extension[i])}`).join(" "));
    lineas.push(`  flex ${r.flexion.map(f0).join(" ")}  orient=${f0(r.orient)} dirInd=${f0(r.dirIndice)}`);
    lineas.push(`  pulgar lat=${s2(r.pulgarLat)} alt=${s2(r.pulgarAlt)} dTI=${f2(r.dTi)} normalZ=${s2(r.normal[2])}`);
    lineas.push(`  cruce=${s2(r.cruce)} angIM=${f0(r.angIm)} prof=${s2(r.profMedio)} vel=${canales[lado].tray.vel.toFixed(1)}`);
  }
  if (analizador.poseActual) {
    lineas.push(`Yaw ${s1(analizador.negacion.desviacion)} (+-8)  Pitch ${s1(analizador.afirmacion.desviacion)} (+-5)  `
      + `${f0(analizador.poseActual.distancia)} cm  ojos x${f2(analizador.ratioOjos)}`);
  }
  return lineas;
}

export function estadoWeb(traductor, manos, vistas, principal, progreso, fps, voz, [ancho, alto], diagnostico, tiempos) {
  const vp = principal ? vistas[principal] : { sena: null, estado: "ninguna", progreso: 0.0 };
  const an = traductor.analizador, pose = an.poseActual;
  return {
    tipo: "estado",
    sena: vp.sena, estado: vp.estado, progreso: r1(progreso * 100), lado: principal,
    manos: Object.fromEntries(LADOS.map((l) => [l, manoWeb(manos[l], vistas[l], traductor.canales[l].tray, ancho, alto)])),
    texto: limpiarTexto(traductor.escritor.texto),
    rostro: { modo: an.modo, pregunta: Math.round(an.progresoPregunta * 100), calibracion: Math.round(an.progresoCalibracion * 100),
      cejas: Math.round(an.ratioCejas * 1000) / 1000,
      cabeza: pose === null ? null : [Math.round(pose.pitch), Math.round(pose.yaw), Math.round(pose.roll)] },
    voz, fps: r1(fps), tamano: [ancho, alto], tiempos,
    diagnostico: diagnostico ? [`Tiempo por fotograma: manos (MediaPipe) ${tiempos.manos_ms.toFixed(0)} ms · rostro `
      + `${tiempos.rostro_ms.toFixed(0)} ms · motor ${tiempos.motor_ms.toFixed(0)} ms · total ${tiempos.proceso_ms.toFixed(0)} ms · `
      + `${fps.toFixed(0)} FPS (motor en el navegador)`, ...lineasDiagnostico(manos, traductor.canales, an)] : null,
  };
}
