// Prueba de humo del motor del navegador (web/motor/motor.js) en Node: cámara, reloj, requestAnimationFrame y
// MediaPipe simulados (mp_falso/); los landmarks salen de los fixtures de Python. Se maneja como lo hace
// static/app.js: carga, config, fotogramas, Diagnóstico, Pausar y cámara sin permiso.
// Uso (desde la raíz): node web/pruebas/humo.mjs
import { readFileSync } from "node:fs";

const aqui = new URL(".", import.meta.url);
globalThis.LSM_MEDIAPIPE = new URL("mp_falso", aqui).href;

let reloj = 0;                                              // ms; lo avanza el conductor con el tiempo del fixture
Object.defineProperty(globalThis, "performance", { value: { now: () => reloj }, configurable: true });
let marcos = [];
globalThis.requestAnimationFrame = (cb) => marcos.push(cb);
let pistasDetenidas = 0, permiso = true;
const video = { muted: false, playsInline: false, srcObject: null, readyState: 4, videoWidth: 0, videoHeight: 0, currentTime: 0,
  play: async () => {} };
let canvas = null;
globalThis.document = { createElement: (tipo) => (tipo === "video" ? video
  : (canvas = { width: 0, height: 0, getContext: () => ({ setTransform() {}, drawImage() {} }) })) };
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { mediaDevices: { getUserMedia: async () => {
  if (!permiso) throw Object.assign(new Error("denegado"), { name: "NotAllowedError" });
  return { getTracks: () => [{ stop: () => pistasDetenidas++ }] };
} } } });
globalThis.fetch = async (url) => {
  const datos = readFileSync(new URL(url));
  return { json: async () => JSON.parse(datos.toString("utf8")), arrayBuffer: async () => datos.buffer.slice(datos.byteOffset, datos.byteOffset + datos.byteLength) };
};
const tick = () => new Promise((r) => setTimeout(r, 0));

const { MotorLocal } = await import(new URL("../motor/motor.js", aqui));
let fallas = 0;

for (const nombre of ["alfabeto", "frases"]) {
  const fx = JSON.parse(readFileSync(new URL(`fixtures/fixture_${nombre}.json`, aqui), "utf8"));
  globalThis.FIXTURE_TAMANO = fx.tamano;
  [video.videoWidth, video.videoHeight] = fx.tamano;
  const mensajes = [];
  let abierto = false;
  const motor = new MotorLocal();
  motor.onopen = () => { abierto = true; motor.send(JSON.stringify({ accion: "VOZ", activa: true, anunciar: false }));
    motor.send(JSON.stringify({ accion: "VELOCIDAD", segundos: 1 })); };
  motor.onmessage = (ev) => mensajes.push(JSON.parse(ev.data));
  globalThis.FIXTURE_CUADRO = fx.cuadros[0];
  for (let i = 0; i < 20 && !motor.activo; i++) await tick();          // carga (fetch, plantillas, MediaPipe) y cámara
  const config = mensajes.find((m) => m.tipo === "config");
  let estados = 0, ultimo = null, malTexto = 0;
  for (const c of fx.cuadros) {
    globalThis.FIXTURE_CUADRO = c;
    reloj = c.entrada.t * 1000;
    video.currentTime = c.entrada.t;
    const pendientes = marcos; marcos = [];
    pendientes.forEach((cb) => cb());
    await tick();                                               // deja correr el DTW (tarea aparte)
    const e = mensajes.filter((m) => m.tipo === "estado").at(-1);
    if (e && e !== ultimo) { estados++; ultimo = e; }
  }
  const finalPy = fx.cuadros.at(-1).salida.tx;
  motor.send(JSON.stringify({ accion: "DIAGNOSTICO" }));
  globalThis.FIXTURE_CUADRO = fx.cuadros.at(-1); reloj += 50; video.currentTime += 0.05;
  marcos.splice(0).forEach((cb) => cb());
  const conDiag = mensajes.at(-1);
  motor.send(JSON.stringify({ accion: "PAUSAR" }));
  const pausa = mensajes.at(-1);
  console.log(`[${nombre}] abierto=${abierto} config=${!!config} (${config?.gestos?.length} frases, ${config?.guia?.length} letras guía) ` +
    `| estados=${estados}/${fx.cuadros.length} | opciones manos: numHands=${globalThis.OPCIONES_MP?.[0]?.numHands} ` +
    `det=${globalThis.OPCIONES_MP?.[0]?.minHandDetectionConfidence}`);
  if (!abierto || !config || ultimo.texto !== finalPy || pausa.tipo !== "pausa") fallas++;
  console.log(`  texto final JS ${ultimo.texto === finalPy ? "== Python" : "DISTINTO"}: ${JSON.stringify(ultimo.texto)}`);
  if (ultimo.texto !== finalPy) console.log(`  Python: ${JSON.stringify(finalPy)}`);
  console.log(`  estado: claves=${Object.keys(ultimo).join(",")} | voz=${ultimo.voz} fps=${ultimo.fps} | ` +
    `diagnóstico con D: ${conDiag.diagnostico?.length} líneas | PAUSAR -> ${pausa.tipo}, pistas detenidas ${pistasDetenidas}`);
  globalThis.OPCIONES_MP = [];
}

// Sin permiso de cámara: debe avisar con un error (app.js muestra Reintentar)
permiso = false;
const m2 = new MotorLocal(), msgs = [];
m2.onmessage = (ev) => msgs.push(JSON.parse(ev.data));
for (let i = 0; i < 20 && !msgs.some((m) => m.tipo === "error"); i++) await tick();
const error = msgs.find((m) => m.tipo === "error");
if (!error) fallas++;
console.log("sin permiso ->", JSON.stringify(error));
console.log(fallas ? "HUMO: FALLÓ" : "HUMO: el motor funciona de punta a punta");
process.exit(fallas ? 1 : 0);
