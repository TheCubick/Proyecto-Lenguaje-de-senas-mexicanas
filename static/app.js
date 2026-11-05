// Traductor LSM: cliente web. Recibe por WebSocket el video (JPEG binario) y el estado (JSON) de server.py y
// manda las acciones de botones y teclado. El reconocimiento vive en Python; aquí solo se presenta y, en el Modo
// Práctica, se compara lo que llega con la seña elegida.
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
let hayVideo = false;

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
}
$("btn-tema").addEventListener("click", () =>
  aplicarTema(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
aplicarTema(document.documentElement.dataset.theme === "light" ? "light" : "dark");

// ---------- conexión ----------
function conectar() {
  socket = new WebSocket(`ws://${location.host}/ws`);
  socket.binaryType = "blob";
  socket.onopen = () => {
    atributo($("conexion"), "data-conexion", "vivo");
    poner($("conexion-texto"), "En vivo");
  };
  socket.onmessage = (ev) => {
    if (typeof ev.data !== "string") return pintarFrame(ev.data);
    const msg = JSON.parse(ev.data);
    if (msg.tipo === "estado") return mostrarEstado(msg);
    if (msg.tipo === "config") return configurar(msg);
    if (msg.tipo === "error") {
      atributo($("conexion"), "data-conexion", "error");
      poner($("conexion-texto"), "Sin cámara");
      hayVideo = false;
      aviso(msg.mensaje, true);
    }
  };
  socket.onclose = () => {
    hayVideo = false;
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

// ---------- video ----------
const lienzo = $("camara");
const ctx = lienzo.getContext("2d");
let decodificando = false;
let pendiente = null;

async function pintarFrame(blob) {
  if (decodificando) { pendiente = blob; return; }        // siempre el más reciente: nunca se acumula retraso
  decodificando = true;
  try {
    const bmp = await createImageBitmap(blob);
    if (lienzo.width !== bmp.width || lienzo.height !== bmp.height) {
      lienzo.width = bmp.width;
      lienzo.height = bmp.height;
      $("lienzo").style.setProperty("--ar", (bmp.width / bmp.height).toFixed(4));
    }
    ctx.drawImage(bmp, 0, 0);
    bmp.close();
    if (!hayVideo) { hayVideo = true; aviso(null); }
  } catch (_) { /* frame dañado: se ignora */ }
  decodificando = false;
  if (pendiente) { const b = pendiente; pendiente = null; pintarFrame(b); }
}

// ---------- configuración (una vez por conexión) ----------
function configurar(cfg) {
  umbralCejas = cfg.umbral_cejas;
  segundos = cfg.segundos;
  bases = cfg.bases;
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
    chip.hidden = !m.caja || !hayVideo;
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
