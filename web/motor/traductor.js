// Escritura (hold-to-type por mano) y orquestación de un fotograma (port de python/traductor.py): manos -> señas ->
// texto, rostro -> gestos. No dibuja ni habla: lo mismo procesa la cámara en vivo y los fixtures de las pruebas.
import { AnalizadorFacial } from "./rostro.js";
import { LADOS, LETRAS_TIPEABLES, CanalMano, SeguidorManos, otroLado } from "./manos.js";
import { textoFrase } from "./gestos.js";

export const SEGUNDOS_PARA_ESCRIBIR = 1.0;
export const VELOCIDAD_MIN = 0.5, VELOCIDAD_MAX = 2.0;
const ACCION_NEGACION = "LETRA";
const ACCIONES_FACIALES = { PREGUNTA: "PREGUNTA", NEGACION: "NEGACION", AFIRMACION: "CONFIRMAR" };

const sostenerVacio = () => ({ sena: null, inicio: 0.0, registrada: false, bloqueo: null, destello: [null, -10.0] });

export class Escritor {
  constructor(segundos) {
    this.segundos = segundos;
    this.texto = "";
    this.manos = Object.fromEntries(LADOS.map((l) => [l, sostenerVacio()]));
    this.ultimoRegistro = -10.0;
    this.eventos = [];
  }
  sacarEventos() { const e = this.eventos; this.eventos = []; return e; }

  ejecutar(accion, ahora, lado = null) {
    if (accion === "ESPACIO") { this.texto += " "; this.eventos.push(["espacio", lado]); }
    else if (accion === "BORRAR") { this.texto = this.texto.slice(0, -1); this.eventos.push(["borrar"]); }
    else if (accion === "LIMPIAR") { this.texto = ""; this.eventos.push(["limpiar"]); }
    else if (accion === "CONFIRMAR") {
      if (this.texto.trim()) {
        if (!this.texto.endsWith(" ")) this.texto += " ";
        this.eventos.push(["confirmar"]);
      }
    } else if (accion === "PREGUNTA") {
      const base = this.texto.trimEnd();
      if (base && !base.endsWith("?")) { this.texto = base + "?"; this.eventos.push(["pregunta"]); }
    } else if (accion === "NEGACION") {
      if (ACCION_NEGACION === "TODO") this.texto = "";
      else if (ACCION_NEGACION === "PALABRA") {
        const base = this.texto.trimEnd();
        this.texto = base.includes(" ") ? base.slice(0, base.lastIndexOf(" ") + 1) : "";
      } else this.texto = this.texto.slice(0, -1);
      this.eventos.push(["negacion"]);
    } else { this.texto += accion; this.eventos.push(["letra", accion, lado]); }
    this.ultimoRegistro = ahora;
  }

  escribirFrase(frase, desde, ahora) {
    let base = this.texto.slice(0, Math.min(desde, this.texto.length));
    if (base && !base.endsWith(" ")) base += " ";
    this.texto = base + frase + " ";
    this.eventos.push(["frase", frase]);
    this.ultimoRegistro = ahora;
  }

  registrarDinamico(lado, gesto, ahora, pose) {
    this.ejecutar(gesto === "?" ? "PREGUNTA" : gesto, ahora, lado);
    const s = this.manos[lado];
    s.sena = gesto; s.inicio = ahora; s.registrada = true;
    s.bloqueo = LETRAS_TIPEABLES.has(pose) ? pose : null;
    s.destello = [gesto, ahora];
  }

  actualizar(lado, sena, ahora) {
    const s = this.manos[lado];
    if (s.bloqueo !== null) {
      if (sena === s.bloqueo) return 1.0;
      s.bloqueo = null;
    }
    if (sena !== s.sena) { s.sena = sena; s.inicio = ahora; s.registrada = false; }
    if (sena === null) return 0.0;
    if (s.registrada) return 1.0;
    const progreso = (ahora - s.inicio) / this.segundos;
    if (progreso >= 1.0) {
      const otra = this.manos[otroLado(lado)];
      if (!(otra.registrada && otra.sena === sena)) this.ejecutar(sena, ahora, lado);
      s.registrada = true;
      return 1.0;
    }
    return progreso;
  }
}

export class Traductor {
  constructor(gestos) {
    this.seguidor = new SeguidorManos();
    this.canales = Object.fromEntries(LADOS.map((l) => [l, new CanalMano(l)]));
    this.escritor = new Escritor(SEGUNDOS_PARA_ESCRIBIR);
    this.analizador = new AnalizadorFacial();
    this.gestos = gestos;                                  // ReconocedorGestos (plantillas ya cargadas)
    this.marcaTexto = 0;
    this.marcaCierre = 0;
  }

  // Un fotograma -> [{lado: estado de la mano}, gesto facial o null]
  paso(detecciones, rostro, ahora, tamano, procesarRostro = true) {
    const asignadas = this.seguidor.asignar(detecciones, ahora);
    const grababa = this.gestos.grabando;
    this.gestos.paso(asignadas, ahora, tamano);
    if (grababa && !this.gestos.grabando) this.marcaCierre = this.marcaTexto;
    if ((this.gestos.grabando && !grababa) || this.gestos.tomarPrimerMovimiento()) this.marcaTexto = this.escritor.texto.length;
    const frases = this.gestos.objetivo !== null;
    const pausa = frases || this.gestos.pausaLetras(ahora);
    const manos = {};
    for (const lado of LADOS) {
      const em = this.canales[lado].procesar(asignadas[lado], ahora);
      if (em.gesto && !frases) this.escritor.registrarDinamico(lado, em.gesto, ahora, em.pose);
      const letra = LETRAS_TIPEABLES.has(em.pose) && !em.moviendo && !pausa ? em.pose : null;
      em.progreso = this.escritor.actualizar(lado, letra, ahora);
      manos[lado] = em;
    }
    const r = this.gestos.resultado;
    if (r && r.mejor) {
      const frase = frases ? (r.acierto ? r.objetivo : null) : (r.aceptado ? r.mejor : null);
      if (frase) {
        r.escrito = textoFrase(frase);
        this.escritor.escribirFrase(r.escrito, this.marcaCierre, ahora);
        if (this.gestos.grabando) this.marcaTexto = this.escritor.texto.length;
      }
    }
    const evento = procesarRostro ? this.analizador.actualizar(rostro, ahora, tamano) : null;
    if (evento) this.escritor.ejecutar(ACCIONES_FACIALES[evento], ahora);
    return [manos, evento];
  }
}
