// Voz en el navegador (Web Speech API) con la misma cola y la misma política que python/voice_engine.py:
// letras sueltas (como mucho 2 en espera), palabras que reemplazan a las letras pendientes y lectura de la palabra
// tras 4 s sin letras nuevas. Prefiere una voz es-MX; si no hay, cualquier español.
import { limpiarTexto } from "./presentacion.js";

const VOZ_INACTIVIDAD = 4.0;
const VOZ_MAX_LETRAS_EN_COLA = 2;
const NOMBRES_VOZ = { A: "a", B: "be", C: "ce", D: "de", E: "e", F: "efe", G: "ge", H: "hache", I: "i", J: "jota",
  K: "ka", L: "ele", M: "eme", N: "ene", "Ñ": "eñe", O: "o", P: "pe", Q: "cu", R: "erre", S: "ese", T: "te", U: "u",
  V: "uve", W: "doble ve", X: "equis", Y: "ye", Z: "zeta" };

const nombreHablado = (letra) => NOMBRES_VOZ[String(letra).toUpperCase()] ?? String(letra).toLowerCase();
function ultimaPalabra(texto) {
  const partes = limpiarTexto(texto).split(/\s+/).filter(Boolean);
  return partes.length ? partes[partes.length - 1] : "";
}

export class VozNavegador {
  constructor() {
    this.cola = [];                       // [texto, tipo]
    this.hablando = false;
    this.silencio = false;
    this.voz = null;
    this.sintesis = typeof speechSynthesis !== "undefined" ? speechSynthesis : null;
    this.estado = this.sintesis ? "iniciando" : "no disponible";
    if (this.sintesis) {
      this.elegirVoz();
      this.sintesis.addEventListener?.("voiceschanged", () => this.elegirVoz());
    }
  }
  elegirVoz() {
    const voces = this.sintesis.getVoices();
    this.voz = voces.find((v) => /es[-_]MX/i.test(v.lang)) ?? voces.find((v) => /^es/i.test(v.lang)) ?? null;
    this.estado = "lista";                // sin voces en español se usa la predeterminada con lang es-MX
  }
  decir(texto, tipo = "letra", forzar = false) {
    texto = limpiarTexto(texto).trim();
    if (!texto || (this.silencio && !forzar) || this.estado === "desactivada" || this.estado === "no disponible") return;
    if (tipo === "palabra") this.cola = this.cola.filter((x) => x[1] !== "letra");
    else {
      const letras = this.cola.filter((x) => x[1] === "letra");
      while (letras.length >= VOZ_MAX_LETRAS_EN_COLA) this.cola.splice(this.cola.indexOf(letras.shift()), 1);
    }
    this.cola.push([texto, tipo]);
    this.siguiente();
  }
  siguiente() {
    if (this.hablando || !this.cola.length) return;
    const [texto] = this.cola.shift();
    const frase = new SpeechSynthesisUtterance(texto);
    frase.lang = this.voz?.lang ?? "es-MX";
    if (this.voz) frase.voice = this.voz;
    const fin = () => { this.hablando = false; this.siguiente(); };
    frase.onend = fin;
    frase.onerror = fin;
    this.hablando = true;
    this.sintesis.speak(frase);
  }
  silenciar(si) {                         // la frase en curso termina; lo que esperaba se descarta
    this.silencio = si;
    if (si) this.cola = [];
  }
  get estadoWeb() { return this.silencio && this.estado === "lista" ? "silenciada" : this.estado; }
}

// Qué se dice y cuándo (PoliticaVoz de Python)
export class PoliticaVoz {
  constructor(voz, inactividad = VOZ_INACTIVIDAD) {
    this.voz = voz; this.inactividad = inactividad; this.pendiente = false; this.tUltimo = 0.0;
  }
  evento(ev, texto, ahora) {
    const tipo = ev[0];
    if (tipo === "letra") { this.voz.decir(nombreHablado(ev[1]), "letra"); this.pendiente = true; this.tUltimo = ahora; }
    else if (tipo === "confirmar") this.leer(texto);
    else if (tipo === "frase") { this.voz.decir(ev[1].toLowerCase(), "palabra"); this.pendiente = false; this.tUltimo = ahora; }
    else if (tipo === "limpiar") this.pendiente = false;
    else { this.tUltimo = ahora; if (!texto.trim()) this.pendiente = false; }
  }
  tick(texto, ahora) { if (this.pendiente && ahora - this.tUltimo >= this.inactividad) this.leer(texto); }
  leer(texto) {
    if (this.pendiente) {
      const palabra = ultimaPalabra(texto);
      if (palabra) this.voz.decir(palabra.toLowerCase(), "palabra");
    }
    this.pendiente = false;
  }
}
