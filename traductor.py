"""Escritura (hold-to-type por mano) y orquestación de un frame: manos -> señas -> texto, rostro -> gestos.
No dibuja ni habla: el mismo código procesa la cámara en vivo y las grabaciones de las pruebas."""
from dataclasses import dataclass
from typing import Optional

from classifier_face import AnalizadorFacial
from classifier_hands import LADOS, LETRAS_TIPEABLES, CanalMano, SeguidorManos, otro_lado

SEGUNDOS_PARA_ESCRIBIR = 1.5   # Tiempo que hay que sostener una seña estática QUIETA para registrarla


ACCION_NEGACION = "LETRA" # Qué borra "No" con la cabeza: "LETRA", "PALABRA" o "TODO"


# Gesto facial -> acción del escritor
ACCIONES_FACIALES = {"PREGUNTA": "PREGUNTA", "NEGACION": "NEGACION", "AFIRMACION": "CONFIRMAR"}


@dataclass
class SostenerMano:
    """Hold-to-type de UNA mano."""
    sena: Optional[str] = None
    inicio: float = 0.0
    registrada: bool = False
    bloqueo: Optional[str] = None           # configuración con la que se hizo un trazo (no se escribe al seguir)
    destello: tuple = (None, -10.0)         # (seña con movimiento, instante) para mostrarla en grande


class Escritor:
    """Mantiene el texto, el hold-to-type de cada mano y la lista de eventos (letra, espacio, borrar...).

    Cualquiera de las dos manos escribe. Una seña estática se registra al sostenerla `segundos` quieta; para
    repetir la misma letra hay que cambiar de seña (o bajar la mano) y volver a hacerla. Una seña con
    movimiento se registra al instante y "bloquea" la configuración con la que se hizo (p. ej. la I tras
    dibujar una J) hasta que esa mano cambie.
    """

    def __init__(self, segundos):
        self.segundos = segundos
        self.texto = ""
        self.manos = {lado: SostenerMano() for lado in LADOS}
        self.ultimo_registro = -10.0
        self.eventos = []

    def sacar_eventos(self):
        eventos, self.eventos = self.eventos, []
        return eventos

    def ejecutar(self, accion, ahora, lado=None):
        """Aplica una letra o una acción de edición al texto y emite el evento correspondiente."""
        if accion == "ESPACIO":
            self.texto += " "
            self.eventos.append(("espacio", lado))
        elif accion == "BORRAR":
            self.texto = self.texto[:-1]
            self.eventos.append(("borrar",))
        elif accion == "LIMPIAR":
            self.texto = ""
            self.eventos.append(("limpiar",))
        elif accion == "CONFIRMAR":            # Asentir: cierra la palabra con un espacio
            if self.texto.strip():
                if not self.texto.endswith(" "):
                    self.texto += " "
                self.eventos.append(("confirmar",))
        elif accion == "PREGUNTA":             # Cejas arriba o "?" en el aire: "?" al final de la palabra
            base = self.texto.rstrip()
            if base and not base.endswith("?"):
                self.texto = base + "?"
                self.eventos.append(("pregunta",))
        elif accion == "NEGACION":             # Negar con la cabeza
            if ACCION_NEGACION == "TODO":
                self.texto = ""
            elif ACCION_NEGACION == "PALABRA":
                base = self.texto.rstrip()
                self.texto = base[:base.rfind(" ") + 1] if " " in base else ""
            else:
                self.texto = self.texto[:-1]
            self.eventos.append(("negacion",))
        else:                                  # Letra
            self.texto += accion
            self.eventos.append(("letra", accion, lado))
        self.ultimo_registro = ahora

    def registrar_dinamico(self, lado, gesto, ahora, pose):
        """Registra una seña con movimiento (J, K, Ñ, Q, X, Z, BORRAR, ?) en cuanto se completa el trazo."""
        self.ejecutar({"?": "PREGUNTA"}.get(gesto, gesto), ahora, lado)
        s = self.manos[lado]
        s.sena, s.inicio, s.registrada = gesto, ahora, True
        s.bloqueo = pose if pose in LETRAS_TIPEABLES else None
        s.destello = (gesto, ahora)

    def actualizar(self, lado, sena, ahora):
        """Avanza el hold-to-type de una mano. Devuelve su progreso real de 0.0 a 1.0."""
        s = self.manos[lado]
        if s.bloqueo is not None:
            if sena == s.bloqueo:              # sigue la misma mano con la que se hizo el trazo: no escribir
                return 1.0
            s.bloqueo = None
        if sena != s.sena:                     # Cambió la seña: el progreso se reinicia
            s.sena, s.inicio, s.registrada = sena, ahora, False
        if sena is None:
            return 0.0
        if s.registrada:
            return 1.0
        progreso = (ahora - s.inicio) / self.segundos
        if progreso >= 1.0:
            otra = self.manos[otro_lado(lado)]
            # La misma letra hecha con las dos manos a la vez se escribe una sola vez: si la otra mano ya la
            # escribió y la sigue sosteniendo, esta no la repite (con datos reales a veces se usan ambas manos)
            if not (otra.registrada and otra.sena == sena):
                self.ejecutar(sena, ahora, lado)
            s.registrada = True
            return 1.0
        return progreso


class Traductor:
    """Orquesta un frame completo: manos -> señas -> texto, y rostro -> gestos.

    No dibuja ni habla: el mismo código procesa la cámara en vivo y las grabaciones de las pruebas."""

    def __init__(self):
        self.seguidor = SeguidorManos()
        self.canales = {lado: CanalMano(lado) for lado in LADOS}
        self.escritor = Escritor(SEGUNDOS_PARA_ESCRIBIR)
        self.analizador = AnalizadorFacial()

    def paso(self, detecciones, rostro, ahora, tamano, procesar_rostro=True):
        """Procesa un frame. Devuelve ({lado: EstadoMano}, gesto facial o None)."""
        asignadas = self.seguidor.asignar(detecciones, ahora)
        manos = {}
        for lado in LADOS:
            em = self.canales[lado].procesar(asignadas.get(lado), ahora)
            if em.gesto:
                self.escritor.registrar_dinamico(lado, em.gesto, ahora, em.pose)
            # Las letras estáticas solo avanzan con esa mano quieta
            letra = em.pose if (em.pose in LETRAS_TIPEABLES and not em.moviendo) else None
            em.progreso = self.escritor.actualizar(lado, letra, ahora)
            manos[lado] = em
        evento = self.analizador.actualizar(rostro, ahora, tamano) if procesar_rostro else None
        if evento:
            self.escritor.ejecutar(ACCIONES_FACIALES[evento], ahora)
        return manos, evento
