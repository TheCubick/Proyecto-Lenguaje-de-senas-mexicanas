"""Voz asíncrona (pyttsx3 en un hilo propio), utilidades de texto y puente al hardware (mock serial)."""
import threading
from collections import deque

# --- Voz ---


VOZ_HABILITADA = True
VOZ_VELOCIDAD = 170       # Palabras por minuto
VOZ_VOLUMEN = 1.0
VOZ_INACTIVIDAD = 4.0     # Segundos sin nuevas letras para leer en voz alta la palabra pendiente
VOZ_MAX_LETRAS_EN_COLA = 2


# Nombre en español de cada letra: así la voz dice "jota" y no un sonido suelto
NOMBRES_VOZ = {
    "A": "a", "B": "be", "C": "ce", "D": "de", "E": "e", "F": "efe", "G": "ge", "H": "hache", "I": "i",
    "J": "jota", "K": "ka", "L": "ele", "M": "eme", "N": "ene", "Ñ": "eñe", "O": "o", "P": "pe", "Q": "cu",
    "R": "erre", "S": "ese", "T": "te", "U": "u", "V": "uve", "W": "doble ve", "X": "equis", "Y": "ye",
    "Z": "zeta",
}


def limpiar_texto(texto):
    """Reemplaza cualquier guion bajo por un espacio. Se aplica a TODO lo que se dibuja, se habla o se envía."""
    return str(texto).replace("_", " ")


def nombre_hablado(letra):
    return NOMBRES_VOZ.get(str(letra).upper(), str(letra).lower())


def ultima_palabra(texto):
    """Última palabra del texto (sin espacios ni guiones bajos alrededor); '' si no hay."""
    partes = limpiar_texto(texto).split()
    return partes[-1] if partes else ""


_avisos_mostrados = set()


def avisar_una_vez(mensaje):
    """Imprime un aviso solo la primera vez (evita llenar la consola en cada frame)."""
    if mensaje not in _avisos_mostrados:
        _avisos_mostrados.add(mensaje)
        print(f"[AVISO] {mensaje}")


# =============================================================================
# VOZ ASÍNCRONA Y CAPA DE HARDWARE
# =============================================================================
def _elegir_voz(motor):
    """Prefiere español de México; si no hay, cualquier voz en español; si no, None (voz por defecto)."""
    mejor, puntaje = None, 0
    for v in motor.getProperty("voices"):
        idiomas = [str(i).lower() for i in (v.languages or [])]
        ficha = f"{v.name} {v.id}".lower()
        if any("es-mx" in i or "es_mx" in i for i in idiomas) or "es-mx" in ficha or "mexico" in ficha:
            s = 3
        elif any("es" == i[:2] or "es" == i[1:3] for i in idiomas) or "spanish" in ficha or "es-es" in ficha:
            s = 2
        else:
            s = 0
        if s > puntaje:
            mejor, puntaje = v, s
    return mejor


class VozAsincrona:
    """Motor de voz (pyttsx3) en un hilo propio: hablar NUNCA bloquea el bucle de la cámara.

    El motor se crea DENTRO del hilo (en Windows, SAPI5 queda ligado al hilo que lo crea) y las frases
    esperan en una cola. Las palabras sustituyen a las letras pendientes; si se acumulan letras se
    descartan las más viejas, para que la voz nunca se quede atrasada respecto a lo que se ve.
    """

    def __init__(self, habilitada=True):
        self.estado = "desactivada"
        self._cola = deque()              # (texto, tipo)
        self._cond = threading.Condition()
        self._cerrando = False
        self._hilo = None
        if habilitada:
            self.estado = "iniciando"
            self._hilo = threading.Thread(target=self._bucle, name="voz-lsm", daemon=True)
            self._hilo.start()

    def decir(self, texto, tipo="letra"):
        """Encola una frase y regresa al instante. `tipo`: "letra" o "palabra"."""
        texto = limpiar_texto(texto).strip()
        if not texto or self.estado in ("desactivada", "no disponible"):
            return
        with self._cond:
            if tipo == "palabra":
                self._cola = deque(x for x in self._cola if x[1] != "letra")
            else:
                letras = [x for x in self._cola if x[1] == "letra"]
                while len(letras) >= VOZ_MAX_LETRAS_EN_COLA:
                    self._cola.remove(letras.pop(0))
            self._cola.append((texto, tipo))
            self._cond.notify()

    def cerrar(self):
        with self._cond:
            self._cerrando = True
            self._cola.clear()
            self._cond.notify()
        if self._hilo is not None:
            self._hilo.join(timeout=1.5)

    def _bucle(self):
        motor = None
        try:
            try:                                  # Windows: cada hilo debe inicializar COM
                import pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                pass
            import pyttsx3
            motor = pyttsx3.init()
            voz = _elegir_voz(motor)
            if voz is not None:
                motor.setProperty("voice", voz.id)
            else:
                avisar_una_vez("No hay una voz en español instalada; se usará la voz por defecto de Windows.")
            motor.setProperty("rate", VOZ_VELOCIDAD)
            motor.setProperty("volume", 0.0)      # calentamiento en silencio: la primera frase real no se retrasa
            motor.say("a")
            motor.runAndWait()
            motor.setProperty("volume", VOZ_VOLUMEN)
            self.estado = "lista"
        except Exception as e:
            self.estado = "no disponible"
            avisar_una_vez(f"Voz no disponible ({type(e).__name__}: {e}). Instala con: pip install pyttsx3")
            return
        while True:
            with self._cond:
                while not self._cola and not self._cerrando:
                    self._cond.wait()
                if self._cerrando:
                    break
                texto, _ = self._cola.popleft()
            try:
                motor.say(texto)
                motor.runAndWait()
            except Exception as e:                # Un fallo al hablar no debe matar el hilo
                avisar_una_vez(f"Error al hablar ({type(e).__name__}: {e}).")
        try:
            motor.stop()
        except Exception:
            pass


class PoliticaVoz:
    """Decide QUÉ se dice y CUÁNDO:
      · al registrarse una letra (barra al 100 % o trazo completo)  -> dice el nombre de la letra
      · al confirmar la palabra (asentir)                           -> lee la palabra completa
      · tras `inactividad` segundos sin nuevas letras               -> lee la palabra pendiente
    """

    def __init__(self, voz, inactividad=VOZ_INACTIVIDAD):
        self.voz = voz
        self.inactividad = inactividad
        self.pendiente = False            # hay letras nuevas que aún no se han leído como palabra
        self.t_ultimo = 0.0

    def evento(self, ev, texto, ahora):
        tipo = ev[0]
        if tipo == "letra":
            self.voz.decir(nombre_hablado(ev[1]), "letra")
            self.pendiente = True
            self.t_ultimo = ahora
        elif tipo == "confirmar":
            self.leer(texto)
        elif tipo == "limpiar":
            self.pendiente = False
        else:                             # espacio, borrar, negación, pregunta: reinician el contador
            self.t_ultimo = ahora
            if not texto.strip():
                self.pendiente = False

    def tick(self, texto, ahora):
        if self.pendiente and ahora - self.t_ultimo >= self.inactividad:
            self.leer(texto)

    def leer(self, texto):
        if self.pendiente:
            palabra = ultima_palabra(texto)
            if palabra:
                self.voz.decir(palabra.lower(), "palabra")
        self.pendiente = False


def enviar_a_hardware(texto_actual):
    """Envía el texto actual al microcontrolador.

    Por ahora es un MOCK. En la etapa de hardware se reemplaza por algo como:
        puerto = serial.Serial("COM3", 115200)          # pip install pyserial
        puerto.write(f"T:{texto_actual}\\n".encode("ascii", "replace"))
    Se llama solo cuando el texto cambia, no en cada frame.
    """
    print(f"[SERIAL MOCK] Enviando al ESP32: {limpiar_texto(texto_actual)}")
