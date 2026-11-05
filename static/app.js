// Traductor LSM: cliente web. Muestra el video de la cámara que transmite el motor (/video_feed, MJPEG), recibe
// por WebSocket el estado JSON de server.py (señas, texto, rostro y coordenadas de las manos), dibuja el esqueleto
// en un <canvas> encima del video y manda las acciones de botones y teclado. El reconocimiento vive en Python; aquí solo se presenta y,
// en el Modo Práctica, se compara lo que llega con la seña elegida.
"use strict";

const $ = (id) => document.getElementById(id);
const LADOS = ["Izq", "Der"];
const NOMBRE_LADO = { Izq: "Mano izquierda", Der: "Mano derecha" };
const CIRCUNFERENCIA = 2 * Math.PI * 104;
const CEJAS_MIN = 0.95, CEJAS_MAX = 1.20;     // rango del medidor de cejas
const PALABRAS = ["HOLA", "GRACIAS", "AMIGO", "CASA", "AGUA", "NIÑO", "FELIZ", "JUEGO", "QUESO", "KIWI"];

const ICONOS = {
  neutro: '<circle cx="12" cy="12" r="10"/><path d="M8 15h8"/><path d="M9 9h.01M15 9h.01"/>',
  pregunta: '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
  negacion: '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6M9 9l6 6"/>',
  afirmacion: '<circle cx="12" cy="12" r="10"/><path d="m8.5 12 2.5 2.5 4.5-5"/>',
  calibrando: '<path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="3"/>',
  sinrostro: '<path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/><path d="m8 8 8 8"/>',
  tapado: '<path d="M9.88 9.88a3 3 0 1 0 4.24 4.24"/><path d="M10.73 5.08A10.4 10.4 0 0 1 12 5c7 0 10 7 10 7a13.2 13.2 0 0 1-1.67 2.68"/><path d="M6.61 6.61A13.5 13.5 0 0 0 2 12s3 7 10 7a9.7 9.7 0 0 0 5.39-1.61"/><path d="m2 2 20 20"/>',
};

// modo facial -> [título, explicación, icono]
const MODOS = {
  SIN_ROSTRO: ["Sin rostro", "Colócate de frente a la cámara", "sinrostro"],
  TAPADO: ["Cara tapada", "Una mano cubre la cara: gestos en pausa", "tapado"],
  CALIBRANDO: ["Calibrando", "Mantén la cara neutra un momento", "calibrando"],
  NEUTRO: ["Neutro", "Cejas = ? · negar = borrar · asentir = confirmar", "neutro"],
  CEJAS: ["Pregunta", "Mantén las cejas arriba…", "pregunta"],
  PREGUNTA: ["Pregunta", "Se agregó «?» al texto", "pregunta"],
  NEGACION: ["Negación", "Se borró la última letra", "negacion"],
  AFIRMACION: ["Afirmación", "Palabra confirmada", "afirmacion"],
};

const VOZ = { lista: "Voz lista", iniciando: "Iniciando voz", desactivada: "Voz apagada", "no disponible": "Sin voz" };

let socket = null;
let detenido = false;
let guia = {};                 // clave -> {clave, desc, mov, li}
let claves = [];               // orden del alfabeto (para la tira de práctica)
let bases = {};                // seña con movimiento -> configuraciones base (J <- I, Y)
let segundos = 1.5;
let umbralCejas = 1.08;
let textoPrevio = null;
let estadoPrevio = null;
let motorListo = false;        // ya llegó un estado del motor en esta conexión
let errorMotor = false;        // el motor reportó un error (sin cámara, señal perdida)
let estadoVideo = "pendiente"; // "pendiente" | "lista" (llega el MJPEG) | "esqueleto" (sin video)
let conexiones = [];           // huesos de la mano: pares de índices (llegan en config)
let coloresMano = {};          // colores del esqueleto según el tema
let ultimo = null;             // último estado recibido (lo dibuja requestAnimationFrame)
let dibujado = null;           // último estado ya dibujado
let latencia = null;           // ms desde la captura hasta el dibujo (media móvil)
let tLatencia = 0;
let nMedido = null;            // número de fotograma de la última medición de latencia

// ---------- utilidades ----------
function poner(el, texto) {
  if (el.textContent !== texto) el.textContent = texto;
}
function atributo(el, nombre, valor) {
  if (el.getAttribute(nombre) !== valor) el.setAttribute(nombre, valor);
}
function guardar(clave, valor) {
  try { localStorage.setItem(clave, valor); } catch (_) { /* almacenamiento bloqueado: solo esta sesión */ }
}
function leer(clave) {
  try { return localStorage.getItem(clave); } catch (_) { return null; }
}
function reiniciarAnimacion(el, clase) {
  el.classList.remove(clase);
  void el.offsetWidth;
  el.classList.add(clase);
}
function enviar(accion) {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ accion }));
}
function aviso(texto, error = false) {
  const el = $("aviso");
  el.hidden = texto === null;
  el.classList.toggle("error", error);
  if (texto !== null) poner($("aviso-texto"), texto);
}
function nombreSena(s) {
  return s === "ESPACIO" ? "Espacio" : s === "BORRAR" ? "Borrar" : s;
}
// Fase de color del hold-to-type: 0-50 % neutro, 50-99 % procesando, 100 % éxito
function fase(estado, progreso) {
  if (estado === "registrada") return "listo";
  if (estado === "sostener") return progreso >= 100 ? "listo" : progreso >= 50 ? "medio" : "bajo";
  if (estado === "pose" || estado === "mover") return "pose";
  return "nada";
}

// ---------- tema claro / oscuro ----------
function aplicarTema(tema) {
  document.documentElement.dataset.theme = tema;
  const b = $("btn-tema");
  b.setAttribute("aria-checked", String(tema === "dark"));
  b.setAttribute("aria-label", tema === "dark" ? "Tema oscuro" : "Tema claro");
  guardar("lsm-tema", tema);
  const css = getComputedStyle(document.documentElement);
  coloresMano = { Izq: css.getPropertyValue("--izq").trim(), Der: css.getPropertyValue("--der").trim(),
                  exito: css.getPropertyValue("--verde").trim() };
  dibujado = null;                                    // redibujar el esqueleto con los colores nuevos
}
$("btn-tema").addEventListener("click", () =>
  aplicarTema(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
aplicarTema(document.documentElement.dataset.theme === "light" ? "light" : "dark");

// ---------- conexión ----------
function actualizarAviso() {
  if (errorMotor) return;                             // el mensaje de error del motor se queda a la vista
  if (!motorListo) aviso("Conectando con el motor de reconocimiento…");
  else if (estadoVideo === "pendiente") aviso("Cargando el video de la cámara…");
  else aviso(null);
}

function conectar() {
  socket = new WebSocket(`ws://${location.host}/ws`);
  socket.onopen = () => {
    errorMotor = false;
    atributo($("conexion"), "data-conexion", "vivo");
    poner($("conexion-texto"), "En vivo");
    actualizarAviso();
  };
  socket.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.tipo === "estado") return mostrarEstado(msg);
    if (msg.tipo === "config") return configurar(msg);
    if (msg.tipo === "error") {
      errorMotor = true;
      atributo($("conexion"), "data-conexion", "error");
      poner($("conexion-texto"), "Sin cámara");
      aviso(msg.mensaje, true);
    }
  };
  socket.onclose = () => {
    motorListo = false;
    ultimo = null;
    limpiarEsqueleto();
    detenerVideo();                     // se reabre al reconectar (el servidor pudo reiniciarse)
    if (detenido) {
      atributo($("conexion"), "data-conexion", "error");
      poner($("conexion-texto"), "Detenido");
      aviso("Traductor detenido. La cámara está apagada; ya puedes cerrar esta pestaña.", true);
      return;
    }
    atributo($("conexion"), "data-conexion", "conectando");
    poner($("conexion-texto"), "Reconectando");
    aviso("Conexión perdida con el servidor. Reintentando…");
    setTimeout(conectar, 1000);
  };
}

// ---------- video del motor (MJPEG por /video_feed) ----------
// Python es el único dueño de la cámara (en Windows el navegador no puede abrirla a la vez). El <img> consume el
// flujo multipart que manda server.py; el esqueleto se dibuja encima con las coordenadas del WebSocket.
const video = $("video");

function iniciarVideo() {
  estadoVideo = "pendiente";
  actualizarAviso();
  video.src = `/video_feed?t=${Date.now()}`;       // parámetro nuevo: nunca reutilizar un flujo cerrado de caché
}

// Un flujo MJPEG no dispara "load" de forma fiable en todos los navegadores: el primer fotograma se detecta en
// el bucle de dibujo (naturalWidth > 0). Si la petición falla (503, servidor caído), salta "error".
video.addEventListener("error", () => {
  if (video.getAttribute("src")) soloEsqueleto("No llega el video del motor: solo esqueleto");
});

function revisarVideo() {
  if (estadoVideo === "pendiente" && video.naturalWidth > 0) {
    estadoVideo = "lista";
    $("lienzo").classList.remove("solo-esqueleto");
    $("lienzo").classList.add("con-video");
    $("modo-esqueleto").hidden = true;
    actualizarAviso();
  }
}

function soloEsqueleto(motivo) {
  estadoVideo = "esqueleto";
  $("lienzo").classList.remove("con-video");
  $("lienzo").classList.add("solo-esqueleto");
  poner($("modo-esqueleto-texto"), motivo);
  $("modo-esqueleto").hidden = false;
  actualizarAviso();
}

function detenerVideo() {
  video.removeAttribute("src");                     // cierra la conexión del flujo
  $("lienzo").classList.remove("con-video", "solo-esqueleto");
  $("modo-esqueleto").hidden = true;
  estadoVideo = "pendiente";
}

// ---------- esqueleto en el canvas ----------
const esqueleto = $("esqueleto");
const pincel = esqueleto.getContext("2d");
const PUNTAS = new Set([4, 8, 12, 16, 20]);

function ajustarLienzo() {
  const r = esqueleto.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (esqueleto.width !== w || esqueleto.height !== h) {
    esqueleto.width = w;
    esqueleto.height = h;
    dibujado = null;
  }
}
new ResizeObserver(ajustarLienzo).observe($("lienzo"));

function limpiarEsqueleto() {
  pincel.clearRect(0, 0, esqueleto.width, esqueleto.height);
  dibujado = null;
}

// Profundidad relativa a la muñeca (negativa = más cerca de la cámara) -> tamaño de las articulaciones
function factorProfundidad(z) {
  return Math.min(1.5, Math.max(0.75, 1 - z * 3));
}

function dibujarEsqueleto(e) {
  const W = esqueleto.width, H = esqueleto.height;
  const k = Math.max(1, W / 640);                    // grosores proporcionales al tamaño en pantalla
  pincel.clearRect(0, 0, W, H);
  pincel.lineCap = "round";
  pincel.lineJoin = "round";
  for (const l of LADOS) {
    const m = e.manos[l];
    if (!m.puntos) continue;
    const color = m.estado === "registrada" ? coloresMano.exito : coloresMano[l];
    const P = m.puntos.map(([x, y, z]) => [x * W, y * H, z]);

    // Estela del trazo (J, Z, ?, vaivenes): tenue y fina al inicio, intensa al final
    if (m.estela && m.estela.length > 1) {
      pincel.strokeStyle = color;
      for (let i = 1; i < m.estela.length; i++) {
        const f = i / m.estela.length;
        pincel.globalAlpha = 0.12 + 0.68 * f;
        pincel.lineWidth = (1 + 3 * f) * k;
        pincel.beginPath();
        pincel.moveTo(m.estela[i - 1][0] * W, m.estela[i - 1][1] * H);
        pincel.lineTo(m.estela[i][0] * W, m.estela[i][1] * H);
        pincel.stroke();
      }
      pincel.globalAlpha = 1;
    }

    // Huesos: trazo claro translúcido con un halo del color de la mano
    pincel.save();
    pincel.shadowColor = color;
    pincel.shadowBlur = 10 * k;
    pincel.strokeStyle = "rgba(238, 242, 250, 0.72)";
    pincel.lineWidth = 2.2 * k;
    pincel.beginPath();
    for (const [a, b] of conexiones) {
      pincel.moveTo(P[a][0], P[a][1]);
      pincel.lineTo(P[b][0], P[b][1]);
    }
    pincel.stroke();
    pincel.restore();

    // Articulaciones: puntos del color de la mano; las yemas, más grandes y con un aro
    pincel.fillStyle = color;
    pincel.strokeStyle = color;
    for (let i = 0; i < P.length; i++) {
      const [x, y, z] = P[i];
      const r = (PUNTAS.has(i) ? 4.2 : 2.8) * k * factorProfundidad(z);
      pincel.beginPath();
      pincel.arc(x, y, r, 0, Math.PI * 2);
      pincel.fill();
      if (PUNTAS.has(i)) {
        pincel.globalAlpha = 0.45;
        pincel.lineWidth = 1.2 * k;
        pincel.beginPath();
        pincel.arc(x, y, r + 3.2 * k, 0, Math.PI * 2);
        pincel.stroke();
        pincel.globalAlpha = 1;
      }
    }
  }
}

// Bucle de dibujo: solo redibuja cuando llegó un estado nuevo (o cambió el tamaño o el tema)
function cuadro() {
  requestAnimationFrame(cuadro);
  revisarVideo();
  if (!ultimo || ultimo === dibujado) return;
  dibujarEsqueleto(ultimo);
  if (ultimo.tiempos && ultimo.tiempos.n !== nMedido) {   // una medición por estado, no por redibujo
    nMedido = ultimo.tiempos.n;
    const l = Date.now() - ultimo.tiempos.captura_ms;   // captura -> dibujo (mismo reloj: misma computadora)
    if (l >= 0 && l < 5000) latencia = latencia === null ? l : latencia * 0.9 + l * 0.1;
    const ahora = performance.now();
    if (ahora - tLatencia > 250 && latencia !== null) {   // texto 4 veces por segundo: legible, sin parpadeo
      tLatencia = ahora;
      poner($("latencia"), String(Math.round(latencia)));
      $("rendimiento").title = `Latencia captura → dibujo: ${Math.round(latencia)} ms · proceso del motor: ` +
        `${ultimo.tiempos.proceso_ms} ms · fotogramas viejos descartados: ${ultimo.tiempos.descartados}`;
    }
  }
  dibujado = ultimo;
}
requestAnimationFrame(cuadro);

// ---------- configuración (una vez por conexión) ----------
function configurar(cfg) {
  umbralCejas = cfg.umbral_cejas;
  segundos = cfg.segundos;
  bases = cfg.bases;
  conexiones = cfg.conexiones;
  $("cejas-umbral").style.left = `${fraccionCejas(umbralCejas) * 100}%`;
  const lista = $("guia-lista");
  lista.textContent = "";
  guia = {};
  claves = cfg.guia.map((g) => g.clave);
  for (const g of cfg.guia) {
    const li = document.createElement("li");
    if (g.mov) li.className = "dinamica";
    const clave = document.createElement("span");
    clave.className = "guia-clave";
    clave.textContent = nombreSena(g.clave);
    const desc = document.createElement("span");
    desc.className = "guia-desc";
    desc.textContent = g.desc;
    if (g.mov) {
      const mov = document.createElement("small");
      mov.textContent = g.mov;
      desc.append(mov);
    }
    li.append(clave, desc);
    li.addEventListener("click", () => { abrirGuia(false); practicarLetra(g.clave); });
    lista.append(li);
    guia[g.clave] = { ...g, li };
  }
  construirTira();
  actualizarReferencia();
}

function fraccionCejas(r) {
  return Math.min(1, Math.max(0, (r - CEJAS_MIN) / (CEJAS_MAX - CEJAS_MIN)));
}

// ---------- estado de cada frame ----------
function textoEstado(e) {
  switch (e.estado) {
    case "registrada": return "Registrada";
    case "sostener": return `Mantén la seña quieta · ${Math.round(e.progreso)} %`;
    case "mover": return "En movimiento: haz el trazo o sostenla quieta";
    case "pose": return `Mueve la mano para escribir ${e.sena}`;
    default: return "Muestra una seña a la cámara con cualquier mano";
  }
}

function mostrarEstado(e) {
  // --- Primer estado de esta conexión: el motor ya tiene la cámara; se pide su video ---
  ultimo = e;
  if (!motorListo) {
    motorListo = true;
    $("lienzo").style.setProperty("--ar", (e.tamano[0] / e.tamano[1]).toFixed(4));
    if (estadoVideo === "pendiente") iniciarVideo();
    actualizarAviso();
  }

  // --- Seña principal y anillo ---
  const tarjeta = $("sena");
  const sena = e.sena ?? "–";
  atributo(tarjeta, "data-estado", e.estado);
  atributo(tarjeta, "data-fase", fase(e.estado, e.progreso));
  poner($("letra"), nombreSena(sena));
  $("letra").classList.toggle("larga", sena.length > 1);
  const pct = e.estado === "ninguna" ? 0 : e.progreso;
  $("anillo").style.strokeDashoffset = (CIRCUNFERENCIA * (1 - pct / 100)).toFixed(2);
  if (e.estado === "registrada" && estadoPrevio !== "registrada") reiniciarAnimacion(tarjeta, "pulso");
  estadoPrevio = e.estado;
  const lado = $("lado");
  atributo(lado, "data-lado", e.lado ?? "");
  poner(lado, e.lado ? NOMBRE_LADO[e.lado] : "Sin mano");
  poner($("estado-texto"), textoEstado(e));

  const g = guia[e.sena];
  $("como").hidden = !g;
  if (g) {
    poner($("como-desc"), g.desc);
    const pista = e.lado ? e.manos[e.lado].pista : null;
    poner($("como-mov"), g.mov ? `Movimiento: ${g.mov}` : pista ? `También: ${pista}` : "");
  }
  for (const clave in guia) guia[clave].li.classList.toggle("activa", clave === e.sena);

  // --- Cada mano: tarjeta mini y etiqueta flotante sobre el video ---
  for (const l of LADOS) {
    const m = e.manos[l];
    const mini = $(`mini-${l}`);
    atributo(mini, "data-estado", m.estado);
    mini.classList.toggle("ausente", !m.caja);
    poner(mini.querySelector(".mini-letra"), m.sena ? nombreSena(m.sena) : "–");
    poner(mini.querySelector(".mini-estado"), !m.caja ? "Sin detectar" : {
      registrada: "Registrada", sostener: "Sosteniendo", mover: "En movimiento", pose: "Haz el trazo",
    }[m.estado] ?? "Detectada");
    mini.querySelector(".mini-barra span").style.width = `${m.estado === "ninguna" ? 0 : m.progreso}%`;

    const chip = $(`chip-${l}`);
    chip.hidden = !m.caja;
    if (m.caja) {
      const [x0, y0, , y1] = m.caja;
      const abajo = y0 < 0.12;                // sin espacio arriba: la etiqueta va debajo de la mano
      chip.classList.toggle("abajo", abajo);
      chip.style.left = `${Math.min(Math.max(x0, 0), 0.82) * 100}%`;
      chip.style.top = `${(abajo ? Math.min(y1, 0.92) : y0) * 100}%`;
      atributo(chip, "data-estado", m.estado);
      atributo(chip, "data-fase", fase(m.estado, m.progreso));
      poner(chip.querySelector(".chip-letra"), m.sena ? nombreSena(m.sena) : "·");
      chip.querySelector(".chip-barra span").style.width = `${m.progreso}%`;
    }
  }

  // --- Rostro ---
  const r = e.rostro;
  const [titulo, sub, icono] = MODOS[r.modo] ?? MODOS.SIN_ROSTRO;
  const rostro = $("rostro");
  if (rostro.getAttribute("data-modo") !== r.modo) {
    rostro.setAttribute("data-modo", r.modo);
    $("rostro-svg").innerHTML = ICONOS[icono];
  }
  poner($("rostro-titulo"), titulo);
  poner($("rostro-sub"), sub);
  const carga = r.modo === "CALIBRANDO" ? r.calibracion : r.modo === "CEJAS" ? r.pregunta : null;
  $("rostro-carga").hidden = carga === null;
  if (carga !== null) $("rostro-carga").firstElementChild.style.width = `${carga}%`;
  const conCara = r.cabeza !== null && r.calibracion >= 100 && r.modo !== "TAPADO";
  $("cejas").hidden = !conCara;
  if (conCara) {
    poner($("cejas-valor"), `x${r.cejas.toFixed(2)}`);
    const barra = $("cejas-barra");
    barra.style.width = `${fraccionCejas(r.cejas) * 100}%`;
    barra.classList.toggle("arriba", r.cejas >= umbralCejas);
    const [p, y, ro] = r.cabeza;
    const s = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v);
    poner($("cabeza"), `Cabeceo ${s(p)}° · Giro ${s(y)}° · Inclinación ${s(ro)}°`);
  }

  // --- Subtítulos: lo nuevo entra con una animación breve ---
  if (e.texto !== textoPrevio) {
    const caja = $("texto");
    const cont = $("texto-contenido");
    const crecio = textoPrevio !== null && e.texto.length > textoPrevio.length && e.texto.startsWith(textoPrevio);
    cont.textContent = "";
    if (crecio) {
      const nuevo = document.createElement("span");
      nuevo.className = "nuevo";
      nuevo.textContent = e.texto.slice(textoPrevio.length);
      cont.append(textoPrevio, nuevo);
    } else {
      cont.textContent = e.texto;
    }
    caja.classList.toggle("vacio", e.texto.length === 0);
    caja.scrollTop = caja.scrollHeight;
    textoPrevio = e.texto;
  }

  // --- Sistema ---
  poner($("fps"), e.fps.toFixed(0));
  atributo($("voz"), "data-voz", e.voz);
  poner($("voz-texto"), VOZ[e.voz] ?? e.voz);
  const diag = $("diagnostico");
  diag.hidden = !e.diagnostico;
  if (e.diagnostico) poner(diag, e.diagnostico.join("\n"));
  document.querySelector('[data-accion="DIAGNOSTICO"]').classList.toggle("activo", !!e.diagnostico);

  evaluarPractica(e);
}

// =====================================================================================
// Modo Práctica: elige una letra o una palabra; se valida con lo que manda MediaPipe
// =====================================================================================
const practica = {
  activa: false,
  modo: "letras",              // "letras" | "palabras"
  letra: "A",
  palabra: PALABRAS[0],
  indice: 0,
  aciertos: 0,
  racha: 0,
  mejor: Number(leer("lsm-mejor-racha")) || 0,
  ocupada: false,              // durante la celebración no se cuentan registros nuevos
  previo: { Izq: { estado: "ninguna", sena: null }, Der: { estado: "ninguna", sena: null } },
  dominadas: new Set(),
};

function objetivo() {
  return practica.modo === "letras" ? practica.letra : [...practica.palabra][practica.indice];
}

function normalizarPalabra(t) {
  return [...t.toUpperCase()]
    .map((c) => (c === "Ñ" ? c : c.normalize("NFD").replace(/[̀-ͯ]/g, "")))
    .join("").replace(/[^A-ZÑ]/g, "").slice(0, 12);
}

// --- pestañas Traducir / Práctica ---
function cambiarVista(vista) {
  practica.activa = vista === "practica";
  $("tab-traducir").setAttribute("aria-selected", String(!practica.activa));
  $("tab-practica").setAttribute("aria-selected", String(practica.activa));
  document.querySelector(".panel").dataset.vista = vista;
  $("sena").hidden = practica.activa;
  $("rostro").hidden = practica.activa;
  $("practica").hidden = !practica.activa;
  $("tarjeta-video").classList.remove("coincide");
  guardar("lsm-vista", vista);
  if (practica.activa) { actualizarReferencia(); centrarLetra(); }
}
for (const b of document.querySelectorAll("[data-vista]")) b.addEventListener("click", () => cambiarVista(b.dataset.vista));

// --- letras / palabras ---
function cambiarModo(modo) {
  practica.modo = modo;
  for (const b of document.querySelectorAll("[data-modo-practica]")) {
    b.setAttribute("aria-selected", String(b.dataset.modoPractica === modo));
  }
  $("selector-letras").hidden = modo !== "letras";
  $("selector-palabras").hidden = modo !== "palabras";
  practica.indice = 0;
  actualizarReferencia();
  if (modo === "letras") centrarLetra();
}
for (const b of document.querySelectorAll("[data-modo-practica]")) {
  b.addEventListener("click", () => cambiarModo(b.dataset.modoPractica));
}

function construirTira() {
  const tira = $("tira");
  tira.textContent = "";
  for (const c of claves) {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.clave = c;
    b.textContent = c === "ESPACIO" ? "Esp" : c === "BORRAR" ? "Borrar" : c;
    b.title = guia[c].desc;
    b.setAttribute("role", "option");
    if (guia[c].mov) b.classList.add("dinamica");
    b.addEventListener("click", () => elegirLetra(c));
    tira.append(b);
  }
  marcarTira();
}

function marcarTira() {
  for (const b of $("tira").children) {
    b.setAttribute("aria-selected", String(b.dataset.clave === practica.letra));
    b.classList.toggle("dominada", practica.dominadas.has(b.dataset.clave));
  }
}

function centrarLetra() {
  const tira = $("tira");
  const b = tira.querySelector(`[data-clave="${CSS.escape(practica.letra)}"]`);
  if (b) tira.scrollTo({ left: b.offsetLeft - tira.clientWidth / 2 + b.offsetWidth / 2 });
}

function elegirLetra(c) {
  practica.letra = c;
  marcarTira();
  centrarLetra();
  actualizarReferencia();
}

function practicarLetra(c) {
  if (!practica.activa) cambiarVista("practica");
  if (practica.modo !== "letras") cambiarModo("letras");
  elegirLetra(c);
}

function moverLetra(paso) {
  if (!claves.length) return;
  const i = (claves.indexOf(practica.letra) + paso + claves.length) % claves.length;
  elegirLetra(claves[i]);
}
$("letra-anterior").addEventListener("click", () => moverLetra(-1));
$("letra-siguiente").addEventListener("click", () => moverLetra(1));
$("letra-azar").addEventListener("click", () => {
  const otras = claves.filter((c) => c !== practica.letra);
  if (otras.length) elegirLetra(otras[Math.floor(Math.random() * otras.length)]);
});

function construirChips() {
  const chips = $("chips-palabras");
  chips.textContent = "";
  for (const p of PALABRAS) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = p;
    b.addEventListener("click", () => elegirPalabra(p));
    chips.append(b);
  }
  marcarChips();
}
function marcarChips() {
  for (const b of $("chips-palabras").children) b.setAttribute("aria-selected", String(b.textContent === practica.palabra));
}
function elegirPalabra(p) {
  practica.palabra = p;
  practica.indice = 0;
  marcarChips();
  actualizarReferencia();
}
$("form-palabra").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const p = normalizarPalabra($("input-palabra").value);
  if (!p) return;
  $("input-palabra").value = "";
  $("input-palabra").blur();
  elegirPalabra(p);
});

function pintarFichas() {
  const fichas = $("fichas");
  fichas.textContent = "";
  [...practica.palabra].forEach((c, i) => {
    const s = document.createElement("span");
    s.textContent = c;
    if (i < practica.indice) s.className = "hecha";
    else if (i === practica.indice) s.className = "actual";
    fichas.append(s);
  });
}

// --- tarjeta de referencia ---
function actualizarReferencia() {
  const obj = objetivo();
  const g = guia[obj];
  if (practica.modo === "palabras") pintarFichas();
  if (!g) return;
  const conMovimiento = !!g.mov;
  poner($("ref-letra"), nombreSena(obj));
  poner($("ref-nombre"), obj.length > 1 ? nombreSena(obj) : obj === "?" ? "Pregunta (?)" : `Letra ${obj}`);
  atributo($("ref-tipo"), "data-tipo", conMovimiento ? "movimiento" : "estatica");
  poner($("ref-tipo"), conMovimiento ? "Con movimiento" : "Estática");
  poner($("ref-desc"), g.desc);
  poner($("ref-mov"), conMovimiento ? g.mov : `Sostenla quieta ${segundos.toLocaleString("es-MX")} s`);
}

// --- validación con los datos del WebSocket ---
function evaluarPractica(e) {
  // Registros nuevos de este frame: una mano que acaba de pasar a "registrada"
  const registros = [];
  for (const l of LADOS) {
    const m = e.manos[l], p = practica.previo[l];
    if (m.estado === "registrada" && m.sena && (p.estado !== "registrada" || p.sena !== m.sena)) registros.push(m.sena);
    practica.previo[l] = { estado: m.estado, sena: m.sena };
  }
  if (!practica.activa) return;

  const obj = objetivo();
  if (!practica.ocupada) {
    if (registros.includes(obj)) return acierto(obj);
    const fallida = registros.find((s) => s.length === 1 && s !== obj);   // ESPACIO / BORRAR no castigan
    if (fallida) fallo(fallida, obj);
  }

  // Retroalimentación en vivo: ¿alguna mano hace la seña (o su base)?
  const manos = LADOS.map((l) => e.manos[l]).filter((m) => m.caja);
  const exacta = manos.find((m) => m.sena === obj);
  const base = manos.find((m) => m.sena && (bases[obj] ?? []).includes(m.sena));
  const retro = $("retro");
  let tono = "nada", texto = "Muestra tu mano a la cámara", pct = 0, f = "nada";
  if (exacta && exacta.estado !== "pose") {
    tono = "bien";
    pct = exacta.progreso;
    f = fase(exacta.estado, exacta.progreso);
    texto = exacta.estado === "mover" ? "¡Esa es! Quédate quieto para registrarla" : "¡Esa es! Mantenla quieta…";
  } else if (base || exacta) {
    tono = "casi";
    texto = `Bien, ${nombreSena((base || exacta).sena)} es la base: ahora haz el movimiento`;
  } else if (manos.length) {
    const vista = manos.find((m) => m.sena);
    tono = vista ? "otra" : "nada";
    texto = vista ? `Detecto «${nombreSena(vista.sena)}». Compárala con la referencia` : "Mano detectada: forma la seña";
  }
  atributo(retro, "data-tono", tono);
  atributo(retro, "data-fase", f);
  poner($("retro-texto"), texto);
  $("retro-barra").style.width = `${pct}%`;
  $("tarjeta-video").classList.toggle("coincide", tono === "bien");
}

function sumar(id, valor) {
  const el = $(id);
  el.textContent = valor;
  reiniciarAnimacion(el, "sube");
}

function acierto(obj) {
  practica.aciertos += 1;
  practica.racha += 1;
  if (practica.racha > practica.mejor) {
    practica.mejor = practica.racha;
    guardar("lsm-mejor-racha", String(practica.mejor));
    sumar("mejor", practica.mejor);
  }
  sumar("aciertos", practica.aciertos);
  sumar("racha", practica.racha);
  practica.dominadas.add(obj);
  marcarTira();

  if (practica.modo === "letras") {
    celebrar("¡Correcto!", `${nombreSena(obj)} · racha ${practica.racha}`, 1800);
    return;
  }
  practica.indice += 1;
  if (practica.indice >= [...practica.palabra].length) {
    pintarFichas();
    celebrar("¡Palabra completa!", `${practica.palabra} · racha ${practica.racha}`, 2300, () => {
      practica.indice = 0;
      actualizarReferencia();
    });
  } else {
    destelloVideo(900);
    actualizarReferencia();
    bloquear(500);
  }
}

function fallo(hecha, buscada) {
  practica.racha = 0;
  poner($("racha"), "0");
  reiniciarAnimacion($("ref"), "sacudida");
  poner($("retro-texto"), `Esa fue «${nombreSena(hecha)}»; buscamos «${nombreSena(buscada)}»`);
  bloquear(700);
}

function bloquear(ms) {
  practica.ocupada = true;
  setTimeout(() => { practica.ocupada = false; }, ms);
}

function destelloVideo(ms) {
  const v = $("tarjeta-video");
  v.classList.add("exito");
  reiniciarAnimacion($("sello"), "activo");
  setTimeout(() => v.classList.remove("exito"), ms);
}

function celebrar(titulo, sub, ms, alTerminar) {
  poner($("exito-titulo"), titulo);
  poner($("exito-sub"), sub);
  const ref = $("ref");
  ref.classList.add("volteada");
  lanzarConfeti();
  destelloVideo(1300);
  practica.ocupada = true;
  setTimeout(() => {
    ref.classList.remove("volteada");
    practica.ocupada = false;
    if (alTerminar) alTerminar();
  }, ms);
}

function lanzarConfeti() {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const caja = $("confeti");
  const estilo = getComputedStyle(document.documentElement);
  const colores = ["--verde", "--azul", "--ambar", "--azul-claro", "--der"].map((v) => estilo.getPropertyValue(v).trim());
  for (let i = 0; i < 28; i++) {
    const p = document.createElement("i");
    const ang = Math.random() * Math.PI * 2;
    const dist = 60 + Math.random() * 90;
    p.style.setProperty("--dx", `${Math.cos(ang) * dist}px`);
    p.style.setProperty("--dy", `${Math.sin(ang) * dist * 0.7 - 30}px`);
    p.style.setProperty("--giro", `${Math.round(Math.random() * 720 - 360)}deg`);
    p.style.background = colores[i % colores.length];
    p.style.animationDelay = `${Math.random() * 120}ms`;
    caja.append(p);
    setTimeout(() => p.remove(), 1400);
  }
}

// ---------- guía ----------
function abrirGuia(abrir) {
  const panel = $("guia");
  abrir = abrir ?? !panel.classList.contains("abierta");
  panel.classList.toggle("abierta", abrir);
  panel.setAttribute("aria-hidden", String(!abrir));
  $("guia-fondo").hidden = !abrir;
}
$("btn-guia").addEventListener("click", () => abrirGuia());
$("guia-cerrar").addEventListener("click", () => abrirGuia(false));
$("guia-fondo").addEventListener("click", () => abrirGuia(false));

// ---------- botones y teclado ----------
function destellar(accion) {
  const b = document.querySelector(`[data-accion="${accion}"]`);
  if (!b) return;
  b.classList.add("presionado");
  setTimeout(() => b.classList.remove("presionado"), 140);
}
for (const b of document.querySelectorAll("[data-accion]")) {
  b.addEventListener("click", () => { enviar(b.dataset.accion); b.blur(); });
}

let tDetener = null;
$("btn-detener").addEventListener("click", () => {
  const b = $("btn-detener");
  if (!b.classList.contains("confirmar")) {          // dos pasos: evita apagar por accidente
    b.classList.add("confirmar");
    poner($("detener-texto"), "¿Detener?");
    tDetener = setTimeout(() => { b.classList.remove("confirmar"); poner($("detener-texto"), "Detener"); }, 3000);
    return;
  }
  clearTimeout(tDetener);
  detenido = true;
  detenerVideo();
  enviar("SALIR");
  b.disabled = true;
  poner($("detener-texto"), "Detenido");
});

const TECLAS = { " ": "ESPACIO", Backspace: "BORRAR", c: "LIMPIAR", d: "DIAGNOSTICO", n: "RECALIBRAR", r: "RECALIBRAR" };
document.addEventListener("keydown", (ev) => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  if (ev.target instanceof HTMLInputElement) return;     // escribiendo una palabra de práctica
  const tecla = ev.key.length === 1 ? ev.key.toLowerCase() : ev.key;
  if (tecla === "Escape") { abrirGuia(false); return; }
  if (tecla === "g" || tecla === "t") {
    if (!ev.repeat) {
      if (tecla === "g") abrirGuia();
      else aplicarTema(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
    }
    ev.preventDefault();
    return;
  }
  const accion = TECLAS[tecla];
  if (!accion) return;
  ev.preventDefault();
  if (ev.repeat && accion !== "BORRAR" && accion !== "ESPACIO") return;
  enviar(accion);
  destellar(accion);
});

// ---------- arranque ----------
$("rostro-svg").innerHTML = ICONOS.sinrostro;
poner($("mejor"), String(practica.mejor));
construirChips();
cambiarModo("letras");
cambiarVista(leer("lsm-vista") === "practica" ? "practica" : "traducir");
conectar();
