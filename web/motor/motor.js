// Motor en el navegador: hace lo que en local hacen server.py + main.capturar, sin servidor (para Vercel o cualquier
// hosting estático). Se presenta a static/app.js como si fuera el WebSocket /ws: misma interfaz (readyState, send,
// onopen, onmessage, onclose) y mismos mensajes ("config", "estado" por fotograma, "pausa", "error").
// Cámara: getUserMedia (aquí no hay Python que la ocupe). Manos y rostro: MediaPipe Tasks (WASM + GPU), la misma
// versión 0.10.18 con la que se extrajo el dataset. El fotograma se espeja ANTES de MediaPipe, como en Python.
import { LADOS, MAX_MANOS, ProgresoSuave, leerManos } from "./manos.js";
import { leerRostro } from "./rostro.js";
import { ReconocedorGestos, leerPlantillas } from "./gestos.js";
import { Traductor, VELOCIDAD_MAX, VELOCIDAD_MIN } from "./traductor.js";
import { estadoWeb, manoPrincipal, vistaMano } from "./presentacion.js";
import { PoliticaVoz, VozNavegador } from "./voz.js";

const MEDIAPIPE = globalThis.LSM_MEDIAPIPE ?? "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";   // las pruebas en Node lo sustituyen
const MODELO_MANOS = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const MODELO_ROSTRO = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const DATOS = new URL("../datos/", import.meta.url);
const ACCIONES_TEXTO = new Set(["ESPACIO", "BORRAR", "LIMPIAR"]);

async function crear(Tarea, archivos, opciones) {          // GPU si se puede; si el navegador no la da, CPU
  try { return await Tarea.createFromOptions(archivos, { ...opciones, baseOptions: { ...opciones.baseOptions, delegate: "GPU" } }); }
  catch (_) { return Tarea.createFromOptions(archivos, { ...opciones, baseOptions: { ...opciones.baseOptions, delegate: "CPU" } }); }
}

function mensajeCamara(e) {
  if (e?.name === "NotAllowedError") return "No hay permiso para usar la cámara. Permítela en el candado de la barra de direcciones y pulsa Reintentar.";
  if (e?.name === "NotFoundError") return "No se encontró ninguna cámara. Conecta una y pulsa Reintentar.";
  if (e?.name === "NotReadableError") return "Otra aplicación está usando la cámara (¿el traductor de Python?). Ciérrala y pulsa Reintentar.";
  return `No se pudo abrir la cámara (${e?.name ?? "error"}). Pulsa Reintentar.`;
}

function registrarEnConsola(r) {                           // como la línea [gesto] de la terminal en Python
  if (r.mejor === null) console.info(`[gesto] ${r.cierre}, ${r.segundos} s: descartado (${r.motivo})`);
  else console.info(`[gesto] ${r.cierre}, ${r.segundos} s: ${r.mejor} dist ${r.distancia.toFixed(2)} sep ${r.separacion.toFixed(2)} -> `
    + `${r.aceptado ? "ACEPTADO" : "rechazado: " + r.motivo} | top: ${r.top.map(([e, d]) => `${e} ${d.toFixed(2)}`).join(", ")}`);
}

export class MotorLocal {
  constructor() {
    this.readyState = 1;          // WebSocket.OPEN: app.js puede mandar acciones desde ya (se guardan hasta cargar)
    this.onopen = this.onmessage = this.onclose = null;
    this.listo = false;
    this.cargando = false;
    this.pendientes = [];
    this.activo = false;
    this.stream = null;
    this.diagnostico = false;
    setTimeout(() => this.preparar(), 0);                   // primero app.js engancha onopen / onmessage
  }

  emitir(mensaje) { this.onmessage?.({ data: JSON.stringify(mensaje) }); }

  async preparar() {
    if (this.cargando || this.listo) return;
    this.cargando = true;
    try {
      this.emitir({ tipo: "cargando", mensaje: "Cargando el reconocedor en tu navegador (la primera vez descarga unos 25 MB)…" });
      const [config, meta, binario, vision] = await Promise.all([
        fetch(new URL("config.json", DATOS)).then((r) => r.json()),
        fetch(new URL("plantillas.json", DATOS)).then((r) => r.json()),
        fetch(new URL("plantillas.bin", DATOS)).then((r) => r.arrayBuffer()),
        import(`${MEDIAPIPE}/vision_bundle.mjs`),
      ]);
      const archivos = await vision.FilesetResolver.forVisionTasks(`${MEDIAPIPE}/wasm`);
      this.manos = await crear(vision.HandLandmarker, archivos, {
        baseOptions: { modelAssetPath: MODELO_MANOS }, runningMode: "VIDEO", numHands: MAX_MANOS,
        minHandDetectionConfidence: 0.7, minHandPresenceConfidence: 0.6, minTrackingConfidence: 0.6 });
      this.rostro = await crear(vision.FaceLandmarker, archivos, {
        baseOptions: { modelAssetPath: MODELO_ROSTRO }, runningMode: "VIDEO", numFaces: 1,
        minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5 });
      this.gestos = new ReconocedorGestos(leerPlantillas(binario, meta), { registrar: registrarEnConsola });
      this.traductor = new Traductor(this.gestos);
      this.voz = new VozNavegador();
      this.politica = new PoliticaVoz(this.voz);
      this.config = config;
    } catch (e) {
      this.cargando = false;
      this.emitir({ tipo: "error", mensaje: `No se pudo cargar el reconocedor (${e?.message ?? e}). Revisa la conexión y pulsa Reintentar.` });
      return;
    }
    this.cargando = false;
    this.listo = true;
    this.onopen?.();
    this.emitir(this.config);
    for (const m of this.pendientes.splice(0)) this.atender(m);
    await this.iniciar();
  }

  send(texto) {
    let mensaje;
    try { mensaje = JSON.parse(texto); } catch (_) { return; }
    if (!this.listo) {
      if (mensaje.accion === "INICIAR") this.preparar();    // reintento tras un error de carga
      else this.pendientes.push(mensaje);
      return;
    }
    this.atender(mensaje);
  }

  // Lo que server.py._recibir + la cola de main.capturar hacen con cada acción (aquí se aplica al instante)
  atender({ accion, ...m }) {
    const { escritor, analizador, gestos } = this.traductor;
    if (accion === "SALIR") { this.pausar(false); this.readyState = 3; this.onclose?.(); }
    else if (accion === "INICIAR") this.iniciar();
    else if (accion === "PAUSAR") this.pausar();
    else if (accion === "VOZ") {
      this.voz.silenciar(!m.activa);
      if (m.activa && m.anunciar) this.voz.decir("Voz activada", "palabra");
    } else if (accion === "VELOCIDAD") {
      if (typeof m.segundos === "number" && VELOCIDAD_MIN <= m.segundos && m.segundos <= VELOCIDAD_MAX) escritor.segundos = m.segundos;
    } else if (accion === "GESTO") gestos.ponerObjetivo(typeof m.objetivo === "string" ? m.objetivo : null);
    else if (ACCIONES_TEXTO.has(accion)) escritor.ejecutar(accion, performance.now() / 1000);
    else if (accion === "DIAGNOSTICO") this.diagnostico = !this.diagnostico;
    else if (accion === "RECALIBRAR") analizador.recalibrar();
    else if (accion === "GUARDAR_MUESTRA") gestos.guardarMuestra();
    else if (accion === "LEER") this.voz.decir(escritor.texto.toLowerCase(), "palabra", true);
  }

  async iniciar() {
    if (this.activo || this.abriendo) return;
    this.abriendo = true;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false });
      this.video = Object.assign(document.createElement("video"), { muted: true, playsInline: true, srcObject: this.stream });
      await this.video.play();
    } catch (e) {
      this.cerrarCamara();
      this.emitir({ tipo: "error", mensaje: mensajeCamara(e) });
      return;
    } finally {
      this.abriendo = false;
    }
    this.lienzo = document.createElement("canvas");
    this.pincel = this.lienzo.getContext("2d");
    this.barra = new ProgresoSuave();
    this.principal = null;
    this.fps = 0;
    this.tAnterior = null;
    this.n = 0;
    this.marca = 0;
    this.activo = true;
    this.programar();
  }

  // Un proceso por fotograma nuevo de la cámara (patrón de los ejemplos de MediaPipe: requestAnimationFrame y solo
  // si cambió currentTime; requestVideoFrameCallback no siempre avisa con un <video> que no está en la página)
  programar() {
    if (this.activo) requestAnimationFrame(() => this.fotograma());
  }

  fotograma() {
    if (!this.activo) return;
    try {
      if (this.video.readyState >= 2 && this.video.currentTime !== this.tVideo) {
        this.tVideo = this.video.currentTime;
        this.procesar();
      }
    } catch (e) {
      console.error("[motor] fotograma omitido:", e);       // un fotograma defectuoso no detiene el motor
    }
    this.programar();
  }

  procesar() {
    const tInicio = performance.now(), capturaMs = Date.now();
    const ancho = this.video.videoWidth, alto = this.video.videoHeight;
    if (this.lienzo.width !== ancho || this.lienzo.height !== alto) { this.lienzo.width = ancho; this.lienzo.height = alto; }
    this.pincel.setTransform(-1, 0, 0, 1, ancho, 0);        // espejo: MediaPipe espera la imagen espejada
    this.pincel.drawImage(this.video, 0, 0, ancho, alto);
    this.marca = Math.max(this.marca + 1, Math.round(tInicio));   // detectForVideo exige marcas crecientes (ms)
    const tManos = performance.now();
    const detecciones = leerManos(this.manos.detectForVideo(this.lienzo, this.marca), ancho, alto);
    const tRostro = performance.now();
    const cara = leerRostro(this.rostro.detectForVideo(this.lienzo, this.marca), ancho, alto);
    const tMotor = performance.now();
    const ahora = tInicio / 1000;
    const { escritor } = this.traductor;
    const [manos] = this.traductor.paso(detecciones, cara, ahora, [ancho, alto]);
    const tFin = performance.now();

    for (const ev of escritor.sacarEventos()) this.politica.evento(ev, escritor.texto, ahora);
    this.politica.tick(escritor.texto, ahora);

    const vistas = {};
    for (const lado of LADOS) {
      const v = vistaMano(manos[lado], escritor.manos[lado], ahora);
      v.progreso = this.traductor.canales[lado].barra.actualizar(v.progreso);
      vistas[lado] = v;
    }
    this.principal = manoPrincipal(vistas, this.principal);
    const progreso = this.barra.actualizar(this.principal ? vistas[this.principal].progreso : 0.0);
    if (this.tAnterior !== null && ahora > this.tAnterior) {
      const instantaneo = 1 / (ahora - this.tAnterior);
      this.fps = this.fps === 0 ? instantaneo : this.fps * 0.9 + instantaneo * 0.1;
    }
    this.tAnterior = ahora;
    const r1 = (x) => Math.round(x * 10) / 10;
    const tiempos = { n: ++this.n, captura_ms: capturaMs, proceso_ms: r1(performance.now() - tInicio),
      manos_ms: r1(tRostro - tManos), rostro_ms: r1(tMotor - tRostro), motor_ms: r1(tFin - tMotor), descartados: 0 };
    const estado = estadoWeb(this.traductor, manos, vistas, this.principal, progreso, this.fps, this.voz.estadoWeb,
      [ancho, alto], this.diagnostico, tiempos);
    estado.gesto = this.gestos.web(ahora);
    this.emitir(estado);
  }

  cerrarCamara() {
    this.activo = false;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    if (this.video) this.video.srcObject = null;
  }

  pausar(avisar = true) {                                  // pausar_motor: la cámara se apaga; texto y práctica siguen
    this.cerrarCamara();
    this.gestos?.ponerObjetivo(this.gestos.objetivo);       // sin un gesto a medias al reanudar
    if (avisar) this.emitir({ tipo: "pausa" });
  }

  close() { this.pausar(false); this.readyState = 3; }
}
