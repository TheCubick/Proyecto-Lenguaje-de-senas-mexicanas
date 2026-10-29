"""Sistema Inteligente de Traducción de Lengua de Señas Mexicana (LSM) — Dos manos + Rostro + Voz.

Canal manual    (MediaPipe Hands, hasta 2 manos): alfabeto LSM completo con CUALQUIERA de las dos manos.
                · Estáticas:  A B C D E F G H I L M N O P R S T U V W Y + ESPACIO
                · Con movimiento: J K Ñ Q X Z + BORRAR + "?"  (buffer de trayectoria por mano)
                Cada mano tiene su propio filtro antivibración (One-Euro), estabilizador, trayectoria y
                hold-to-type. La mano izquierda se refleja al sistema de coordenadas de la derecha antes de
                clasificar, y la clasificación usa geometría 3D: extensión y ángulos articulares de cada dedo,
                normal de la palma (producto cruz) y la posición del pulgar en un marco propio de la mano.
Canal no manual (MediaPipe Face Mesh + cv2.solvePnP): pose 3D de la cabeza en grados (pitch, yaw, roll).
                Cejas arriba = "?", negar (yaw) = borrar, asentir (pitch) = confirmar la palabra.
                Al iniciar se calibra la cara neutra (tecla N para recalibrar).
Voz             (pyttsx3, hilo propio): dice cada letra al registrarla y lee la palabra al confirmar
                o tras un tiempo de inactividad. Nunca bloquea la cámara.

Escritura "hold-to-type": una seña estática se registra al sostenerla quieta 1.5 s; las señas con
movimiento se registran al completar el trazo.
Teclas: Espacio, Backspace, C (limpiar), D (diagnóstico), N (recalibrar rostro neutro), ESC (salir).

Estructura del archivo:
    1. Configuración                         7. Canal no manual (pose de cabeza y cejas)
    2. Utilidades de texto/voz               8. Escritura y orquestación (Traductor)
    3. Geometría 2D/3D y filtro One-Euro     9. Voz asíncrona y capa de hardware
    4. Canal manual (rasgos 3D de la mano)  10. Interfaz gráfica
    5. Clasificación de señas               11. Programa principal
       (estáticas y con movimiento)
    6. Dos manos (lateralidad y estado por mano)
"""
import math
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================
VENTANA_CAMARA = "Camara IA - Deteccion LSM"
VENTANA_PANEL = "Panel Traductor LSM"
ANCHO_PANEL = 640
ALTO_PANEL = 740
MARGEN_VENTANAS = 30

# --- Escritura ---
SEGUNDOS_PARA_ESCRIBIR = 1.5   # Tiempo que hay que sostener una seña estática QUIETA para registrarla
FRAMES_ESTABLES = 4            # Frames seguidos con la misma seña antes de aceptarla (anti-parpadeo)
V_QUIETO = 1.2                 # Velocidad de la palma (escalas/s) por encima de la cual la mano "se mueve".
                               # Sostener una seña produce 0.5-1.1 de temblor natural; un trazo, más de 2.
DURACION_DESTELLO = 1.2        # Segundos que se muestra en grande una seña con movimiento recién registrada

# --- Dos manos ---
MAX_MANOS = 2             # MediaPipe Hands (max_num_hands). Con 2, mientras solo hay una mano a la vista MediaPipe
                          # busca la segunda en cada frame: medido, el tiempo de las manos pasa de 37 a 76 ms en
                          # esta laptop en batería. En una PC lenta, 1 = más FPS (pero solo una mano a la vez)
LADOS = ("Izq", "Der")
MP_A_LADO = {"Left": "Izq", "Right": "Der"}  # La imagen se espeja ANTES de procesarla, así que la etiqueta de
                                             # MediaPipe (multi_handedness) es la mano real de la persona
DUPLICADO_DIST = 0.35       # Dos detecciones a menos de esto (en tamaños de mano) son la misma mano repetida
CONTINUIDAD_DIST = 1.2      # Una mano sigue en su ranura si aparece a menos de esto de donde estaba...
CONTINUIDAD_SEG = 0.25      # ... y la ranura la vio hace menos de esto (aunque la etiqueta parpadee)
LATERALIDAD_SUAVIZADO = 0.15  # Promedio (EMA) de la evidencia "es la derecha" de cada ranura
LATERALIDAD_CAMBIO = 0.85     # Evidencia sostenida de la otra mano necesaria para cambiar de ranura

# --- Filtro antivibración One-Euro (por mano, sobre los 21 puntos) ---
ONE_EURO_CORTE_MIN = 1.5    # Hz con la mano quieta: quita el temblor de la cámara
ONE_EURO_BETA = 0.4         # Cuánto sube el corte con la velocidad (por tamaño de mano/s): sin retraso al moverse
ONE_EURO_CORTE_D = 1.0      # Hz del filtro de la velocidad

# --- Mano: umbrales geométricos 3D (adimensionales => invariantes a escala, rotación y mano) ---
# Medidos con grabaciones reales de ambas manos (la izquierda reflejada); ver DOCUMENTACION.md, sección 8.
# Extensión de un dedo = |punta - muñeca| / |nudillo - muñeca| en 3D: recto 1.45-2.1 · curvo 1.0-1.5 · puño 0.6-1.15
EXT_RECTO = (1.40, 1.54, 1.45, 1.25)  # Extensión mínima de un dedo recto: índice, medio, anular, meñique
EXT_CERRADO = 1.15        # Por debajo: dedo cerrado (puño)
FLEX_MAX_RECTO = 75       # Flexión PIP + DIP (grados) máxima de un dedo recto (rectos 26-66°; el índice de la E, 90-121°)
ARTICULACION_RECTA = 55   # Con PIP y DIP así de rectas, el dedo cuenta como recto aunque se vea un poco corto...
EXT_TOLERANCIA = 0.15     # ... hasta 0.15 por debajo de EXT_RECTO (N con el índice a 1.39 y 51° de flexión)
MENIQUE_RECTO = 50        # Meñique con PIP y DIP rectos aunque el nudillo lo incline (Y de lado)
FACTOR_ANCHO_PALMA = 1.55 # Largo de palma ≈ 1.55 × ancho de nudillos (tamaño de la mano en píxeles)
ORIENT_ARRIBA = 55        # Orientación de la palma: 0° = dedos arriba, 90° = de lado, 180° = abajo
ORIENT_LADO = (55, 115)
ORIENT_ABAJO = 115
PULGAR_FUERA = -0.45      # Punta del pulgar por fuera del nudillo del índice (A, L, G, Y, ESPACIO), en largos de palma
F_CONTACTO = 0.55         # F: pinza pulgar-índice (distancia entre puntas / largo de palma; la B mide ≥ 0.65)
F_FLOJA = (0.75, 1.30)    # F con la pinza floja: vale si el índice está claramente doblado (extensión < 1.30)
O_CONTACTO = 0.32         # O: puntas de pulgar e índice juntas (medido 0.12-0.24)
O_PULGAR_LAT = -0.15      # O: el pulgar queda del lado de afuera (en la S cruza los dedos: > +0.2)
C_GAP = (0.32, 1.10)      # C: hueco entre las puntas de pulgar e índice (medido 0.38-0.71; por debajo es la O)
C_PULGAR_LAT = -0.40      # C: pulgar por fuera
C_FLEX = (60, 115)        # C y O: flexión PIP + DIP de índice y medio (un dedo recto mide < 60)
C_INDICE_MIN = 1.05       # C: índice curvo pero no metido en la palma (C 1.14-1.44; en el puño el pulgar va adentro)
T_ALTURA = 0.0            # T: punta del pulgar por ENCIMA de los PIP de índice y medio (T +0.03..+0.23; S -0.40..-0.03)
E_MCP_MAX = 33            # E: nudillos (MCP) casi rectos y dedos doblados en PIP/DIP (E 12-25°, S 44-67°)
ESPACIO_PULGAR = 1.00     # ESPACIO: pulgar bien separado del índice (ESPACIO 0.97-1.24; mano relajada 0.79-1.07)
B_ABERTURA = 20           # B: dedos juntos (ángulo 3D índice-meñique)
V_CRUCE = 0.40            # V: puntas de índice y medio separadas en el eje lateral (V 0.63-0.73)
R_CRUCE = -0.05           # R: puntas cruzadas (orden invertido; R -0.20..-0.07)
U_ANG_MAX = 11            # U: índice y medio paralelos (ángulo 3D U 4-8°, P 15-40°)
P_PROFUNDIDAD = 0.05      # P: el medio se adelanta al índice hacia el lado de la palma
P_NORMAL_Z = -0.88        # P: palma girada (componente z de la normal; U de frente -0.96..-0.91)
Y_DIST = 0.75             # Y: pulgar lejos del índice (Y 0.90-1.30; en la I el pulgar lo toca, 0.09-0.48)
IY_RESTO_MAX = 1.35       # I/Y: índice y medio sin estirar (I y Y miden 0.64-1.27; un dedo recto, ≥ 1.40)
D_DIR_MAX = 50            # D: el índice apunta arriba (D 20-31°); de lado es la X (62-91°)
X_DIR_MIN = 60            # X con el índice recto: apunta de lado (X 62-82°)...
X_ORIENT_MIN = 48         # ... y la palma también se inclina (X 52-61°). Al dibujar la Z o el "?" el índice
                          # se ladea hasta 90° pero la palma sigue derecha (16-50°): sigue siendo la D
X_EXT_MIN = 1.25          # X: índice a medio estirar con el resto cerrado (en el puño el índice mide < 1.15)
Q_DIR_INDICE_MIN = 105    # Q: el índice cuelga (Q 115-140°, X 62-91°)
Q_ORIENT_MIN = 80         # Q: la mano entera apunta hacia abajo (las letras de puño se sostienen verticales, < 45°)

# --- Señas con movimiento (buffer de trayectoria) ---
TRAYECTORIA_MUESTRAS = 64 # Muestras recordadas por mano (≈ 1.8 s aun a 35 FPS). Cada gesto mira su propia ventana EN
                          # SEGUNDOS (GESTOS_DINAMICOS), no en frames: así reconoce igual a 16 que a 30 FPS. Con datos
                          # reales, una Z a ritmo natural dura 1.4-2 s; un vaivén, ~1 s; el barrido de BORRAR, ~0.5 s
DIN_MIN_MUESTRAS = 6      # Muestras mínimas para evaluar un trazo
REARME_VEL = 0.8          # Tras un gesto de vaivén, la palma debe quedar por debajo de esta velocidad (escalas/s)...
REARME_SEGUNDOS = 0.3     # ... durante este tiempo antes de aceptar otro (si no, agitar seguido escribiría Q Q Q)
DIN_FRACCION_POSE = 0.60  # Fracción de la ventana en que debe verse la configuración de la mano
DIN_ENFRIAMIENTO = 1.0    # Segundos sin evaluar trazos tras reconocer uno (evita duplicados)
DIN_PERDIDOS_MAX = 5      # Frames sin mano antes de vaciar el buffer
VEL_SUAVIZADO = 0.25      # Suavizado (EMA) de la velocidad de la mano: ignora picos de 1-2 frames
J_DESCENSO_MIN = 0.55     # J: cuánto baja el meñique (en escalas)
J_GANCHO_MIN = 0.25       # J: cuánto se curva hacia un lado después de bajar
Z_BARRA_MIN = 0.40        # Z: largo mínimo de las dos barras
Z_DIAGONAL_MIN = 0.45     # Z: largo mínimo de la diagonal
INTERROG_ALTO_MIN = 0.80  # "?": altura mínima del trazo
VAIVEN_REVERSOS = 2       # Ñ, Q, X: cambios de sentido necesarios (2 = ida, vuelta e ida)
VAIVEN_RECTITUD = 0.5     # Cada ida o vuelta debe ser directa (desplazamiento / camino). Medido: K y Ñ reales
                          # 0.6-0.99; la K falsa al dibujar una Z con dos manos en escena, 0.34
VAIVEN_PROPORCION = 2.5   # y de tamaño parecido (mayor / menor). Medido: K reales 1.0-1.4; esa K falsa, 3.8
VAIVEN_AMP = {"Ñ": 0.07, "QX": 0.07, "K": 0.15}  # Amplitud mínima de cada oscilación (en escalas). Medido: una Ñ
                                                 # suave oscila 0.08-0.13; sostener la N quieta tiembla < 0.05
BORRAR_RECORRIDO_MIN = 1.10  # BORRAR: barrido hacia el centro del cuerpo, recorrido mínimo (en escalas)
BORRAR_VEL_MIN = 3.0         # BORRAR: velocidad pico mínima (escalas/s); evita borrar al reacomodar la mano
BORRAR_DT = 0.15             # ... medida sobre tramos de este tiempo

# --- Rostro ---
FRAMES_CALIBRACION = 45   # Frames con cara neutra para medir la línea base personal (cejas, ojos y pose)
CARA_ERROR_MAX = 0.018    # Error de reproyección de solvePnP / alto de la cara. Medido: cara visible 0.8-1.4 %;
                          # con las manos encima de la cara 1.7-3.8 %. Por encima, el frame no cuenta (cara tapada)
CALIB_PERCENTIL = 35      # Percentil de la calibración de cejas (ignora frames con cejas ya levantadas)
CEJAS_UMBRAL = 1.07       # Cejas "arriba" si su altura (compensada por la pose) supera en 7 % la línea base
CEJAS_UMBRAL_SALIDA = 1.04  # Histéresis: deben bajar de esto para volver a neutro
CEJAS_SEGUNDOS = 1.5      # Tiempo con las cejas arriba para agregar "?"
CEJAS_TOLERANCIA = 0.30   # Caídas más cortas que esto (parpadeo, ruido) no reinician el conteo
CEJAS_REARME = 0.30       # Tiempo con las cejas abajo para poder volver a preguntar
CEJAS_MAX_SEGUNDOS = 4.0  # Cejas arriba más tiempo que esto no es una pregunta (postura, mirada): ese nivel
                          # pasa a ser la nueva cara neutra. Medido: una pregunta dura 1.5-3 s
CEJAS_ADAPTACION = 0.005  # Qué tan rápido se reajusta la línea base mientras la cara está neutra
CEJAS_ZONA_ADAPT = 0.03   # Solo se reajusta si la razón está a menos de 3 % de la línea base
CEJAS_PAUSA_TRAS_CABEZA = 0.8  # Segundos sin evaluar cejas tras negar o asentir
NEG_AMPLITUD = 8.0        # Negar: desviación mínima del giro (yaw, grados). Medido: un "no" real oscila ±19°
NEG_ENTRADAS_MIN = 3      # izquierda-derecha-izquierda (o al revés)
AFIRM_AMPLITUD = 5.0      # Asentir: desviación mínima de la inclinación (pitch, grados). Un "sí" real: ±5-6°
AFIRM_ENTRADAS_MIN = 3    # tres cabeceos: con dos, mover la cabeza al leer la pantalla contaba como "sí"
OSC_VENTANA = 1.4         # Segundos en los que debe ocurrir el movimiento de cabeza
OSC_TAU_CENTRO = 2.5      # Constante de tiempo (s) de la posición neutra de la cabeza
ENFRIAMIENTO_CABEZA = 1.2 # Segundos sin detectar otro gesto de cabeza tras uno reconocido
DURACION_AVISO = 1.2      # Segundos que se muestra en pantalla el último gesto facial
ACCION_NEGACION = "LETRA" # Qué borra "No" con la cabeza: "LETRA", "PALABRA" o "TODO"
PROCESAR_ROSTRO_CADA = 1  # 1 = cada frame; 2 = frames alternos (si la PC va lenta)

# --- Voz ---
VOZ_HABILITADA = True
VOZ_VELOCIDAD = 170       # Palabras por minuto
VOZ_VOLUMEN = 1.0
VOZ_INACTIVIDAD = 4.0     # Segundos sin nuevas letras para leer en voz alta la palabra pendiente
VOZ_MAX_LETRAS_EN_COLA = 2

# --- Colores (BGR) ---
FONDO = (28, 24, 22)
TARJETA = (48, 42, 38)
BORDE = (80, 72, 66)
TEXTO = (235, 235, 235)
TEXTO_SUAVE = (150, 145, 140)
ACENTO = (255, 190, 60)     # Azul claro
EXITO = (110, 220, 80)      # Verde
ALERTA = (80, 90, 255)      # Rojo
AMBAR = (60, 190, 255)      # Ámbar
COLOR_LADO = {"Izq": (255, 150, 80), "Der": (90, 220, 120)}   # Puntos de cada mano: azul / verde

FUENTE = cv2.FONT_HERSHEY_SIMPLEX
FUENTE_TITULO = cv2.FONT_HERSHEY_DUPLEX

# Alfabeto: (clave, descripción corta, movimiento). Sin acentos: OpenCV no los dibuja (la Ñ se compone a mano).
GUIA = [
    ("A", "Puno, pulgar estirado al lado", None),
    ("B", "4 dedos juntos, pulgar cruzado", None),
    ("C", "Mano curva en forma de C", None),
    ("D", "Indice arriba, otros cerrados", None),
    ("E", "Dedos doblados sobre el pulgar", None),
    ("F", "Indice-pulgar, 3 dedos arriba", None),
    ("G", "Pistola: indice de lado", None),
    ("H", "Pistola: indice y medio de lado", None),
    ("I", "Solo el menique arriba", None),
    ("J", "I + gancho (baja y curva)", "Mueve el menique: baja y haz un gancho"),
    ("K", "P + agitar de lado a lado", "Agita la mano (grande) de lado a lado"),
    ("L", "Indice arriba y pulgar al lado", None),
    ("M", "3 dedos hacia abajo", None),
    ("N", "2 dedos hacia abajo", None),
    ("Ñ", "N + vaiven de lado a lado", "Agita la N de lado a lado"),
    ("O", "Circulo indice-pulgar", None),
    ("P", "Indice y medio cruzados en X", None),
    ("Q", "Pinza abajo + balanceo lateral", "Balancea la pinza de lado a lado"),
    ("R", "Indice y medio entrelazados", None),
    ("S", "Puno, pulgar sobre los dedos", None),
    ("T", "Puno, pulgar entre los dedos", None),
    ("U", "Indice y medio juntos", None),
    ("V", "Indice y medio en V", None),
    ("W", "Indice, medio y anular abiertos", None),
    ("X", "Indice de lado + vaiven", "Mueve el indice de lado a lado"),
    ("Y", "Pulgar y menique estirados", None),
    ("Z", "D + trazo de Z en el aire", "Dibuja una Z en el aire con el indice"),
    ("ESPACIO", "Mano abierta, 5 dedos", None),
    ("BORRAR", "Mano abierta barre al centro", "Barre la mano abierta hacia el centro del cuerpo"),
    ("?", "Indice dibuja ? en el aire", "Dibuja un ? en el aire con el indice"),
]
DESCRIPCIONES = {clave: (desc, mov) for clave, desc, mov in GUIA}

# Pistas para las configuraciones que sirven de base a un trazo
PISTA_TRAZO = {"I": "o traza una J", "Y": "o traza una J", "D": "o traza Z / ?", "P": "o agita para K",
               "R": "o agita para K", "U": "o agita para K", "N": "o agita para Ñ", "Q": "balancea para Q",
               "X": "mueve para X", "B": "o barre al centro", "ESPACIO": "o barre al centro"}

# Texto y color de cada estado facial
MODOS_FACIALES = {
    "SIN_ROSTRO": ("Sin rostro", TEXTO_SUAVE),
    "TAPADO": ("Cara tapada", TEXTO_SUAVE),  # una mano tapa la cara: no se interpretan gestos faciales
    "CALIBRANDO": ("Calibrando...", AMBAR),
    "NEUTRO": ("Neutro", TEXTO),
    "CEJAS": ("Pregunta ?", ACENTO),       # Cejas arriba, cargando
    "PREGUNTA": ("Pregunta ?", EXITO),     # "?" agregado
    "NEGACION": ("Negacion X", ALERTA),
    "AFIRMACION": ("Afirmacion OK", EXITO),
}

# Gesto facial -> acción del escritor
ACCIONES_FACIALES = {"PREGUNTA": "PREGUNTA", "NEGACION": "NEGACION", "AFIRMACION": "CONFIRMAR"}


# =============================================================================
# 2. UTILIDADES DE TEXTO Y VOZ
# =============================================================================
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


def _a_ascii(texto):
    """Quita acentos y signos que las fuentes de OpenCV no dibujan; conserva Ñ/ñ (se componen aparte)."""
    salida = []
    for ch in texto.replace("¿", "?").replace("¡", "!"):
        if ch in "Ññ":
            salida.append(ch)
            continue
        base = "".join(c for c in unicodedata.normalize("NFD", ch) if unicodedata.category(c) != "Mn")
        salida.append(base if base.isascii() else "?")
    return "".join(salida)


_avisos_mostrados = set()


def avisar_una_vez(mensaje):
    """Imprime un aviso solo la primera vez (evita llenar la consola en cada frame)."""
    if mensaje not in _avisos_mostrados:
        _avisos_mostrados.add(mensaje)
        print(f"[AVISO] {mensaje}")


# =============================================================================
# 3. GEOMETRÍA 2D / 3D Y FILTRO ONE-EURO
# =============================================================================
def dist(a, b):
    """Distancia euclidiana 2D."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def punto_medio(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def restar(a, b):
    return (a[0] - b[0], a[1] - b[1])


def unitario(v):
    n = math.hypot(v[0], v[1]) or 1.0
    return (v[0] / n, v[1] / n)


def unitario3(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def angulo_vectores(v1, v2):
    """Ángulo en grados entre dos vectores (2D o 3D), por producto punto."""
    n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if n < 1e-12:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / n))))


def angulo3(a, b, c):
    """Ángulo interior en el vértice b de los puntos 3D a-b-c (180 = articulación recta)."""
    if float(np.linalg.norm(a - b) * np.linalg.norm(c - b)) < 1e-12:
        return 180.0
    return angulo_vectores(a - b, c - b)


def desde_arriba(v):
    """Ángulo (grados) entre un vector y "arriba" de la imagen (y crece hacia abajo): 0 arriba, 90 de lado, 180 abajo."""
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(v[1]) / (float(np.linalg.norm(v)) or 1.0)))))


class FiltroOneEuro:
    """Filtro One-Euro (Casiez, Roussel y Vogel, 2012) aplicado a un arreglo de coordenadas.

    Es un promedio exponencial cuya frecuencia de corte sube con la velocidad: con la mano quieta corta
    bajo (quita el temblor de la cámara) y al hacer un trazo corta alto (casi sin retraso). La velocidad se
    mide en tamaños de mano por segundo, así que los parámetros no dependen de la distancia a la cámara.
    """

    def __init__(self, corte_min=ONE_EURO_CORTE_MIN, beta=ONE_EURO_BETA, corte_d=ONE_EURO_CORTE_D):
        self.corte_min, self.beta, self.corte_d = corte_min, beta, corte_d
        self.reiniciar()

    def reiniciar(self):
        self.x = self.dx = self.t = None

    @staticmethod
    def _alfa(corte, dt):
        return 1.0 / (1.0 + 1.0 / (2 * math.pi * corte * dt))

    def __call__(self, x, t, escala=1.0):
        x = np.asarray(x, dtype=float)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x
        if t <= self.t:
            return self.x
        dt, self.t = t - self.t, t
        self.dx = self.dx + self._alfa(self.corte_d, dt) * ((x - self.x) / dt - self.dx)
        corte = self.corte_min + self.beta * np.abs(self.dx) / max(escala, 1e-9)
        self.x = self.x + self._alfa(corte, dt) * (x - self.x)
        return self.x


# =============================================================================
# 4. CANAL MANUAL — rasgos 3D de la mano
# =============================================================================
WRIST, THUMB_TIP = 0, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
PINKY_MCP, PINKY_TIP = 17, 20
# (MCP, PIP, DIP, TIP) de índice, medio, anular y meñique
DEDOS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
PALMA = (0, 5, 9, 13, 17)
EXT, DOBLADO, CERRADO = "EXT", "DOB", "CER"
ESPEJO = np.array([-1.0, 1.0, 1.0])


def a_canonica(puntos, lado):
    """Normalización de simetría: refleja el eje lateral (x) de la mano izquierda, así cualquier mano se
    analiza en el sistema de coordenadas de una mano derecha."""
    return puntos * ESPEJO[:puntos.shape[1]] if lado == "Izq" else puntos


def escala_mano(p):
    """Tamaño de la mano en píxeles, robusto a la rotación: el máximo entre el largo de la palma y el ancho de
    nudillos (escalado), porque al rotar la mano uno de los dos se acorta en la imagen pero el otro no."""
    return max(dist(p[WRIST], p[MIDDLE_MCP]), FACTOR_ANCHO_PALMA * dist(p[INDEX_MCP], p[PINKY_MCP])) or 1.0


@dataclass
class RasgosMano:
    """Resumen geométrico de una mano. Los rasgos 3D se calculan sobre la mano CANÓNICA (la izquierda reflejada)
    en un marco propio de la mano: u (muñeca -> nudillo del medio), s (nudillo del índice -> nudillo del meñique)
    y la normal de la palma n = s × u. Todo es adimensional: no depende de la distancia ni de qué mano sea."""
    lado: str
    p: np.ndarray            # 21x2 en píxeles de la imagen (para dibujar)
    pc: np.ndarray           # 21x2 en píxeles de la mano canónica (para las trayectorias)
    escala: float            # tamaño de la mano en píxeles
    centro: tuple            # centro de la palma (canónico)
    extension: list          # por dedo: |punta - muñeca| / |nudillo - muñeca| en 3D
    flexion: list            # por dedo: flexión en PIP + DIP en grados (0 = recto)
    flexion_mcp: list        # por dedo: flexión en el nudillo (MCP)
    estados: list            # [índice, medio, anular, meñique] -> EXT / DOB / CER
    normal: np.ndarray       # vector que sale de la palma (cámara: x der, y abajo, z hacia adentro; z < 0 = de frente)
    orient: float            # eje de la palma: 0° dedos arriba · 90° de lado · 180° abajo
    dir_indice: float        # hacia dónde apunta el índice (nudillo -> punta), mismos grados
    pulgar_lat: float        # punta del pulgar respecto al nudillo del índice en el eje lateral (< 0 = por fuera)
    pulgar_alt: float        # punta del pulgar respecto a los PIP de índice y medio en el eje de la mano (> 0 = encima)
    d_ti: float              # distancia punta del pulgar - punta del índice / largo de palma
    cruce: float             # (punta del medio - punta del índice) en el eje lateral: < 0 = cruzados
    ang_im: float            # ángulo 3D entre índice y medio
    prof_medio: float        # cuánto se adelanta la punta del medio a la del índice hacia el lado de la palma
    abertura: float          # ángulo 3D entre índice y meñique


def analizar_mano(img, mundo, lado):
    """Rasgos de una mano. `img`: 21x3 en píxeles (x, y; z en la escala de x). `mundo`: 21x3 en metros
    (multi_hand_world_landmarks de MediaPipe). Las distancias de contacto se miden en la imagen (más precisa
    en x-y); los ángulos, la normal y el marco de la mano, en las coordenadas 3D de mundo."""
    ic = a_canonica(img, lado)
    P = a_canonica(mundo, lado)
    w = P[WRIST]
    u = unitario3(P[MIDDLE_MCP] - w)
    s = P[PINKY_MCP] - P[INDEX_MCP]
    s = unitario3(s - np.dot(s, u) * u)          # eje lateral, perpendicular al de la mano
    n = np.cross(s, u)                            # producto cruz: sale de la palma
    palma = float(np.linalg.norm(P[MIDDLE_MCP] - w)) or 1e-6

    # Estado de cada dedo: extensión 3D + ángulos articulares (producto punto en MCP->PIP->DIP->TIP)
    extension, flexion, flexion_mcp, estados = [], [], [], []
    for k, (mcp, pip, dip, tip) in enumerate(DEDOS):
        ext = float(np.linalg.norm(P[tip] - w)) / (float(np.linalg.norm(P[mcp] - w)) or 1e-6)
        flex = (180 - angulo3(P[mcp], P[pip], P[dip])) + (180 - angulo3(P[pip], P[dip], P[tip]))
        extension.append(ext)
        flexion.append(flex)
        flexion_mcp.append(180 - angulo3(w, P[mcp], P[pip]))
        recto = ((ext >= EXT_RECTO[k] and flex <= FLEX_MAX_RECTO)
                 or (flex <= ARTICULACION_RECTA and ext >= EXT_RECTO[k] - EXT_TOLERANCIA)
                 or (k == 3 and flex <= MENIQUE_RECTO and ext >= EXT_CERRADO))
        estados.append(EXT if recto else (CERRADO if ext < EXT_CERRADO else DOBLADO))

    p, pc = img[:, :2], ic[:, :2]
    centro = pc[list(PALMA)].mean(axis=0)
    pips = (P[INDEX_PIP] + P[MIDDLE_PIP]) / 2
    palma_img = float(np.linalg.norm(ic[MIDDLE_MCP] - ic[WRIST])) or 1e-6
    return RasgosMano(
        lado=lado, p=p, pc=pc, escala=escala_mano(p), centro=(float(centro[0]), float(centro[1])),
        extension=extension, flexion=flexion, flexion_mcp=flexion_mcp, estados=estados, normal=n,
        orient=desde_arriba(u), dir_indice=desde_arriba(P[INDEX_TIP] - P[INDEX_MCP]),
        pulgar_lat=float(np.dot(P[THUMB_TIP] - P[INDEX_MCP], s)) / palma,
        pulgar_alt=float(np.dot(P[THUMB_TIP] - pips, u)) / palma,
        d_ti=float(np.linalg.norm(ic[THUMB_TIP] - ic[INDEX_TIP])) / palma_img,
        cruce=float(np.dot(P[MIDDLE_TIP] - P[INDEX_TIP], s)) / palma,
        ang_im=angulo_vectores(P[INDEX_TIP] - P[INDEX_MCP], P[MIDDLE_TIP] - P[MIDDLE_MCP]),
        prof_medio=float(np.dot(P[MIDDLE_TIP] - P[INDEX_TIP], n)) / palma,
        abertura=angulo_vectores(P[INDEX_TIP] - P[INDEX_MCP], P[PINKY_TIP] - P[PINKY_MCP]),
    )


# =============================================================================
# 5. CLASIFICACIÓN DE SEÑAS
# =============================================================================
# Letras que se escriben sosteniendo la mano quieta
LETRAS_TIPEABLES = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "M", "N", "O", "P", "R", "S", "T",
                    "U", "V", "W", "Y", "ESPACIO"}
# Configuraciones que no se muestran ni escriben pero sirven de base a un trazo (mano abierta -> BORRAR)
POSES_SILENCIOSAS = {"ABIERTA"}


def decidir_q_o_x(dir_indice):
    """Q si el dedo índice apunta hacia abajo; si no, X."""
    return "Q" if dir_indice >= Q_DIR_INDICE_MIN else "X"


def clasificar_pose(r):
    """Devuelve la configuración de la mano: una letra estática, "ESPACIO", "Q" o "X" (configuraciones que
    solo valen con movimiento), "ABIERTA" (mano abierta sin seña) o None si no hay certeza.
    Se basa en el alfabeto oficial de la LSM (SEP) y está calibrada con grabaciones reales de ambas manos.

    Orden de decisión: pinzas y manos curvas (F, O, C) -> número de dedos rectos -> orientación ->
    posición del pulgar y relación entre índice y medio.
    """
    i, m, a, p = (e == EXT for e in r.estados)
    n_ext = i + m + a + p
    arriba = r.orient <= ORIENT_ARRIBA
    lado = ORIENT_LADO[0] < r.orient < ORIENT_LADO[1]
    abajo = r.orient >= ORIENT_ABAJO
    ext = r.extension

    # --- F: pinza pulgar-índice con anular y meñique rectos (el medio puede quedar a medio doblar) ---
    pinza = r.d_ti < F_CONTACTO or (r.d_ti < F_FLOJA[0] and ext[0] < F_FLOJA[1])
    if pinza and a and p and r.estados[1] != CERRADO:
        return "F"
    # --- Índice y medio curvos con el pulgar por fuera: O (se tocan) o C (queda un hueco) ---
    if all(C_FLEX[0] <= f <= C_FLEX[1] for f in r.flexion[:2]) and ext[3] < 1.60:
        if r.d_ti < O_CONTACTO and r.pulgar_lat < O_PULGAR_LAT:
            return "O"
        if C_GAP[0] <= r.d_ti <= C_GAP[1] and r.pulgar_lat < C_PULGAR_LAT and ext[0] >= C_INDICE_MIN:
            return "C"

    # --- Cuatro dedos rectos: M (hacia abajo), ESPACIO, B o mano abierta ---
    if n_ext == 4:
        if abajo:
            return "M"
        if r.d_ti >= ESPACIO_PULGAR and r.pulgar_lat < PULGAR_FUERA:
            return "ESPACIO"
        if r.pulgar_lat > 0 and r.abertura < B_ABERTURA:
            return "B"
        return "ABIERTA"                            # mano relajada: no escribe, pero vale para barrer (BORRAR)

    # --- Tres dedos (índice, medio, anular): W arriba, M abajo ---
    if n_ext == 3:
        if i and m and a:
            return "M" if abajo else ("W" if arriba else None)
        return None

    # --- Índice y medio: N, H, V, R, U o P según orientación, cruce y ángulo entre ellos ---
    if n_ext == 2 and i and m:
        if abajo:
            return "N"
        if lado:
            return "H"
        if not arriba:
            return None
        if r.cruce >= V_CRUCE:                      # puntas separadas
            return "V"
        if r.cruce <= R_CRUCE:                      # puntas cruzadas (orden lateral invertido)
            return "R"
        if r.ang_im < U_ANG_MAX and r.prof_medio < P_PROFUNDIDAD and r.normal[2] < P_NORMAL_Z:
            return "U"                              # paralelos, en el mismo plano y con la palma de frente
        return "P"                                  # el medio cruza por delante en diagonal o la palma gira

    # --- Un solo dedo ---
    if n_ext == 1:
        if i:
            if r.pulgar_lat < PULGAR_FUERA:         # pulgar separado: L (arriba), G (de lado) o Q (colgando)
                return "L" if arriba else ("G" if lado else ("Q" if abajo else None))
            if r.dir_indice >= Q_DIR_INDICE_MIN and r.orient >= X_ORIENT_MIN:
                return "Q"                          # índice colgando: con movimiento
            if r.dir_indice >= X_DIR_MIN and r.orient >= X_ORIENT_MIN:
                return "X"                          # índice de lado con la palma inclinada: con movimiento
            return "D"
        if p and ext[0] < IY_RESTO_MAX and ext[1] < IY_RESTO_MAX:
            return "Y" if r.d_ti >= Y_DIST else "I"
        return None

    # --- Ningún dedo recto: Q, X, A, T, E o S ---
    if n_ext == 0:
        if r.orient >= Q_ORIENT_MIN and r.dir_indice >= Q_DIR_INDICE_MIN:
            return "Q"                              # puño o pinza con la mano colgando hacia abajo
        if (ext[0] >= X_EXT_MIN and all(e == CERRADO for e in r.estados[1:])
                and D_DIR_MAX <= r.dir_indice < Q_DIR_INDICE_MIN):
            return "X"                              # índice a medio estirar apuntando de lado
        if r.pulgar_lat < PULGAR_FUERA:
            if ext[0] >= EXT_RECTO[0]:              # índice largo aunque doblado + pulgar afuera: "pistola"
                return "L" if arriba else ("G" if lado else None)
            return "A"                              # pulgar al lado del puño, índice metido (A: ≤ 1.21)
        if r.pulgar_alt > T_ALTURA:                 # punta del pulgar asomada entre índice y medio
            return "T"
        if float(np.mean(r.flexion_mcp[1:])) < E_MCP_MAX:
            return "E"                              # nudillos rectos y dedos doblados sobre el pulgar
        return "S"                                  # puño con el pulgar cruzado por delante

    return None


# --- Trayectoria de la mano -------------------------------------------------
def puntos_rastreados(p, centro=None):
    """Puntos que siguen los trazos, a partir de 21 puntos 2D."""
    if centro is None:
        centro = p[list(PALMA)].mean(axis=0)
    return {"muneca": p[WRIST], "indice": p[INDEX_TIP], "medio": p[MIDDLE_TIP], "menique": p[PINKY_TIP],
            "centro": np.asarray(centro, dtype=float),
            "puntas": (p[INDEX_TIP] + p[MIDDLE_TIP]) / 2}      # no se mueve al separar los dedos


class Trayectoria:
    """Buffer circular (deque) con las últimas posiciones de la muñeca / los dedos de UNA mano.

    Cada muestra guarda: tiempo, puntos rastreados en la mano canónica (para reconocer: la izquierda
    reflejada), escala, configuración vista, dirección del índice y los puntos reales (para dibujar la estela).
    También calcula la velocidad de la palma (EMA).
    """

    def __init__(self, n=TRAYECTORIA_MUESTRAS):
        self.muestras = deque(maxlen=n)
        self.perdidos = 0
        self.vel = 0.0                 # escalas por segundo

    def agregar(self, t, r, pose):
        puntos = puntos_rastreados(r.pc, r.centro)
        if self.muestras:
            t0, puntos0 = self.muestras[-1][:2]
            if t - t0 > 1e-3:
                v = dist(puntos["centro"], puntos0["centro"]) / (r.escala or 1.0) / (t - t0)
                self.vel += (v - self.vel) * VEL_SUAVIZADO
        self.muestras.append((t, puntos, r.escala, pose, r.dir_indice, puntos_rastreados(r.p)))
        self.perdidos = 0

    def perder(self):
        """Un frame sin mano: tras unos cuantos seguidos se vacía el buffer."""
        self.perdidos += 1
        self.vel *= 0.5
        if self.perdidos > DIN_PERDIDOS_MAX:
            self.muestras.clear()

    def limpiar(self):
        self.muestras.clear()

    def recientes(self, segundos):
        """Cuántas muestras caen en los últimos `segundos` (la ventana de un gesto no depende de los FPS)."""
        if not self.muestras:
            return 0
        limite = self.muestras[-1][0] - segundos
        return sum(1 for m in self.muestras if m[0] >= limite)

    def _ultimas(self, n=None):
        """Las últimas `n` muestras (todas si n es None)."""
        muestras = list(self.muestras)
        return muestras[-n:] if n else muestras

    def serie(self, nombre, n=None):
        """Trayectoria canónica de las últimas `n` muestras normalizada por el tamaño de la mano (y hacia abajo)."""
        ms = self._ultimas(n)
        escala = float(np.median([m[2] for m in ms])) or 1.0
        return np.array([m[1][nombre] for m in ms], dtype=float) / escala

    def serie_px(self, nombre):
        """Recorrido real en la imagen (para dibujar la estela)."""
        return [m[5][nombre] for m in self.muestras]

    def tiempos(self, n=None):
        return np.array([m[0] for m in self._ultimas(n)], dtype=float)

    def fraccion_pose(self, permitidas, n=None):
        ms = self._ultimas(n)
        return sum(m[3] in permitidas for m in ms) / max(len(ms), 1)


# --- Reconocedores de trazos (sobre la mano canónica: la izquierda llega reflejada) ---------
def simplificar_rdp(pts, eps):
    """Ramer-Douglas-Peucker: índices de los vértices que describen el trazo con tolerancia `eps`."""
    mantener = {0, len(pts) - 1}
    pila = [(0, len(pts) - 1)]
    while pila:
        i, j = pila.pop()
        a, ab = pts[i], pts[j] - pts[i]
        largo = float(np.hypot(*ab))
        lejos, idx = 0.0, -1
        for k in range(i + 1, j):
            v = pts[k] - a
            d = abs(ab[0] * v[1] - ab[1] * v[0]) / largo if largo > 1e-9 else float(np.hypot(*v))
            if d > lejos:
                lejos, idx = d, k
        if lejos > eps:
            mantener.add(idx)
            pila += [(i, idx), (idx, j)]
    return sorted(mantener)


def segmentos(pts, eps=0.10, largo_min=0.12, fusion=35):
    """Trazo -> lista de (rumbo en grados, largo). Rumbo: 0 = derecha, 90 = arriba, 270 = abajo (y hacia arriba)."""
    idx = simplificar_rdp(pts, eps)
    segs = []
    for i, j in zip(idx[:-1], idx[1:]):
        v = pts[j] - pts[i]
        segs.append([(math.degrees(math.atan2(-v[1], v[0]))) % 360, float(np.hypot(*v)), v[0], v[1]])
    fusionado = []
    for s in segs:
        if fusionado:
            ant = fusionado[-1]
            dif = abs((s[0] - ant[0] + 180) % 360 - 180)
            if s[1] < largo_min or dif <= fusion:   # tramo diminuto o casi en la misma dirección: se une
                ant[2] += s[2]
                ant[3] += s[3]
                ant[0] = math.degrees(math.atan2(-ant[3], ant[2])) % 360
                ant[1] = float(np.hypot(ant[2], ant[3]))
                continue
        fusionado.append(s)
    return [(s[0], s[1]) for s in fusionado]


def en_arco(rumbo, centro, semiancho):
    return abs((rumbo - centro + 180) % 360 - 180) <= semiancho


def _simetrico(fn):
    """Acepta el trazo y su imagen en espejo: la J, la Z y el "?" se pueden dibujar como letra escrita o reflejados."""
    def g(p, ts):
        return fn(p, ts) or fn(p * np.array([-1.0, 1.0]), ts)
    return g


def _trazo_j(p, ts):
    """J: el meñique baja en vertical y se curva hacia un lado al final (gancho)."""
    n = len(p)
    k = int(np.argmax(p[:, 1]))                       # punto más bajo
    if k < 3 or n - 1 - k < 2:
        return False
    i0 = int(np.argmin(p[:k + 1, 1]))                 # donde empezó a bajar
    descenso = p[k, 1] - p[i0, 1]
    deriva = abs(p[k, 0] - p[i0, 0])
    gancho = p[k, 0] - p[k:, 0].min()                 # cuánto se fue a la izquierda después de bajar
    return descenso >= J_DESCENSO_MIN and deriva <= 0.6 * descenso and gancho >= J_GANCHO_MIN


def _trazo_z(p, ts):
    """Z: barra a la derecha, diagonal hacia abajo-izquierda y otra barra a la derecha."""
    segs = segmentos(p)
    if len(segs) < 3:
        return False
    (h1, l1), (h2, l2), (h3, l3) = segs[-3:]
    return (en_arco(h1, 0, 40) and l1 >= Z_BARRA_MIN and en_arco(h2, 225, 35) and l2 >= Z_DIAGONAL_MIN
            and en_arco(h3, 0, 40) and l3 >= Z_BARRA_MIN)


def _trazo_interrogacion(p, ts):
    """"?": arco que abulta a la derecha, regresa hacia el centro y termina en un tallo vertical."""
    n = len(p)
    if p[:, 1].max() - p[:, 1].min() < INTERROG_ALTO_MIN:
        return False
    k = int(np.argmax(p[:, 0]))                       # punto más a la derecha (el abultamiento)
    if not (0.15 * n <= k <= 0.75 * n) or int(np.argmin(p[:, 1])) > 0.6 * n:
        return False
    if p[k, 0] - p[0, 0] < 0.20 or p[k, 0] - p[-1, 0] < 0.20 or p[-1, 1] - p[k, 1] < 0.40:
        return False
    dx_final, dy_final = p[-1, 0] - p[-4, 0], p[-1, 1] - p[-4, 1]   # el tallo termina casi vertical
    return dy_final >= 0.25 and abs(dx_final) <= 0.6 * dy_final


def puntos_de_giro(xs, amp):
    """Índices donde una señal 1D cambia de sentido, con histéresis `amp` para ignorar el temblor."""
    ultimo, i_ultimo, sentido, giros = xs[0], 0, 0, []
    for i, x in enumerate(xs[1:], 1):
        if sentido == 0:
            if abs(x - ultimo) > amp:
                sentido, ultimo, i_ultimo = (1 if x > ultimo else -1), x, i
        elif (x - ultimo) * sentido > 0:          # sigue en el mismo sentido: nuevo extremo
            ultimo, i_ultimo = x, i
        elif (ultimo - x) * sentido > amp:        # regresó más que `amp`: el extremo anterior fue un giro
            giros.append(i_ultimo)
            sentido, ultimo, i_ultimo = -sentido, x, i
    return giros


def rectitud(p, a, b):
    """Qué tan directo es el tramo p[a..b] en x: desplazamiento lateral / camino recorrido (1 = línea recta)."""
    camino = float(np.sum(np.hypot(*np.diff(p[a:b + 1], axis=0).T)))
    return abs(float(p[b, 0] - p[a, 0])) / camino if camino > 1e-9 else 0.0


def _vaiven(amp, horizontal=True):
    """Ñ, K, Q, X: oscilación de al menos `amp` con VAIVEN_REVERSOS cambios de sentido en x.
    Con `horizontal` el movimiento debe ser casi lateral; Q y X (balanceo del índice curvo) no lo exigen.
    Movimiento coherente: las últimas idas y vueltas completas deben ser casi rectas y de tamaño parecido
    (acomodar la mano o dibujar una Z da tramos erráticos o desiguales)."""
    def f(p, ts):
        ancho = p[:, 0].max() - p[:, 0].min()
        alto = p[:, 1].max() - p[:, 1].min()
        if (horizontal and alto > 0.8 * ancho) or ancho < 1.8 * amp:
            return False
        giros = puntos_de_giro(p[:, 0], amp)
        if len(giros) < VAIVEN_REVERSOS:
            return False
        hitos = ([0] + giros)[-(VAIVEN_REVERSOS + 1):]
        tramos = list(zip(hitos[:-1], hitos[1:]))
        amplitudes = [abs(float(p[b, 0] - p[a, 0])) for a, b in tramos]
        return (all(rectitud(p, a, b) >= VAIVEN_RECTITUD for a, b in tramos)
                and max(amplitudes) <= VAIVEN_PROPORCION * min(amplitudes))
    return f


def _trazo_borrar(p, ts):
    """BORRAR: barrido rápido de la mano abierta hacia el centro del cuerpo (en la mano canónica, a la
    izquierda: con la mano derecha se barre a la izquierda y con la izquierda, a la derecha)."""
    dx, dy = p[-1, 0] - p[0, 0], p[-1, 1] - p[0, 1]
    if dx > -BORRAR_RECORRIDO_MIN or abs(dy) > 0.6 * abs(dx):
        return False
    pasos = np.diff(p[:, 0])
    total = np.abs(pasos).sum()
    if total <= 0 or -pasos[pasos < 0].sum() / total < 0.85:
        return False
    vel_max = 0.0                                   # velocidad pico, medida sobre tramos de ~0.15 s (no de k frames)
    for i in range(len(p)):
        j = int(np.searchsorted(ts, ts[i] + BORRAR_DT))
        if j < len(p):
            vel_max = max(vel_max, (p[i, 0] - p[j, 0]) / max(ts[j] - ts[i], 1e-3))
    return vel_max >= BORRAR_VEL_MIN


# (gesto, configuraciones válidas, punto que se rastrea, reconocedor, fracción mínima con esa configuración, ventana)
# La ventana son los últimos segundos que mira cada gesto: un trazo largo (Z, ?) necesita más que un vaivén.
GESTOS_DINAMICOS = [
    ("J", {"I", "Y"}, "menique", _simetrico(_trazo_j), DIN_FRACCION_POSE, 1.2),      # el pulgar parpadea entre I e Y
    ("Z", {"D"}, "indice", _simetrico(_trazo_z), DIN_FRACCION_POSE, 1.8),
    ("?", {"D"}, "indice", _simetrico(_trazo_interrogacion), DIN_FRACCION_POSE, 1.8),
    ("K", {"P", "R", "U"}, "puntas", _vaiven(VAIVEN_AMP["K"]), DIN_FRACCION_POSE, 1.8),   # índice y medio que se agitan
    ("Ñ", {"N", "H"}, "centro", _vaiven(VAIVEN_AMP["Ñ"]), DIN_FRACCION_POSE, 1.2),       # una N de lado se lee como H
    # Q y X son el mismo vaivén: se decide Q o X con la dirección media del índice en la ventana
    ("QX", {"Q", "X"}, "indice", _vaiven(VAIVEN_AMP["QX"], horizontal=False), DIN_FRACCION_POSE, 1.0),
    ("BORRAR", {"B", "ESPACIO", "ABIERTA"}, "centro", _trazo_borrar, DIN_FRACCION_POSE, 0.6),
]
GESTOS_OSCILANTES = {"K", "Ñ", "QX"}       # tras uno de estos hace falta una pausa antes de aceptar otro
PUNTO_DE_POSE = {}
for _g, _poses, _punto, _, _, _ in GESTOS_DINAMICOS:
    for _pose in _poses:
        PUNTO_DE_POSE.setdefault(_pose, _punto)


class ReconocedorDinamico:
    """Evalúa en cada frame si el buffer de UNA mano contiene un trazo completo con la configuración adecuada.

    Un gesto se evalúa cuando su configuración DOMINA su ventana (no basta el último frame: al moverse rápido la
    mano a veces se lee como otra letra, p. ej. la D como X al dibujar). Si varios trazos coinciden, gana el de
    configuración más dominante."""

    def __init__(self):
        self.libre_desde = 0.0
        self.pausa_pendiente = False    # tras un vaivén hay que soltar la mano un momento
        self.t_quieto = None

    def _actualizar_pausa(self, tray, ahora):
        if not self.pausa_pendiente:
            return
        if tray.vel < REARME_VEL:
            self.t_quieto = ahora if self.t_quieto is None else self.t_quieto
            if ahora - self.t_quieto >= REARME_SEGUNDOS:
                self.pausa_pendiente = False
        else:
            self.t_quieto = None

    def evaluar(self, tray, ahora):
        self._actualizar_pausa(tray, ahora)
        if ahora < self.libre_desde or len(tray.muestras) < DIN_MIN_MUESTRAS:
            return None
        candidatos = []
        for gesto, poses, punto, reconocer, fraccion, ventana in GESTOS_DINAMICOS:
            n = tray.recientes(ventana)
            if n < DIN_MIN_MUESTRAS or (gesto in GESTOS_OSCILANTES and self.pausa_pendiente):
                continue
            dominio = tray.fraccion_pose(poses, n)
            if dominio >= fraccion and reconocer(tray.serie(punto, n), tray.tiempos(n)):
                candidatos.append((dominio, gesto, n))
        if not candidatos:
            return None
        _, gesto, n = max(candidatos)
        if gesto == "QX":                           # Q y X son el mismo vaivén: se decide con el promedio de la ventana
            gesto = decidir_q_o_x(float(np.mean([m[4] for m in tray._ultimas(n)])))
        if gesto in ("Q", "X") or gesto in GESTOS_OSCILANTES:
            self.pausa_pendiente, self.t_quieto = True, None
        self.libre_desde = ahora + DIN_ENFRIAMIENTO
        tray.limpiar()
        return gesto


# =============================================================================
# 6. DOS MANOS — lateralidad, seguimiento y estado independiente por mano
# =============================================================================
@dataclass
class DeteccionMano:
    """Una mano que MediaPipe vio en este frame."""
    lado_mp: str             # "Izq" / "Der" según multi_handedness
    confianza: float         # confianza de esa etiqueta (0-1)
    img: np.ndarray          # 21x3: x, y en píxeles; z (profundidad relativa) en la escala de x
    mundo: np.ndarray        # 21x3 en metros, con origen en el centro de la mano
    nueva: bool = False      # la marca el seguidor: no continúa una mano anterior (se reinicia su estado)

    @property
    def centro(self):
        return self.img[list(PALMA), :2].mean(axis=0)

    @property
    def escala(self):
        return escala_mano(self.img[:, :2])


def leer_manos(resultado, ancho, alto):
    """Convierte la salida de MediaPipe Hands (landmarks, world landmarks y handedness) en DeteccionMano."""
    if not resultado.multi_hand_landmarks:
        return []
    mundos = resultado.multi_hand_world_landmarks or []
    salida = []
    for k, lm in enumerate(resultado.multi_hand_landmarks):
        img = np.array([(l.x * ancho, l.y * alto, l.z * ancho) for l in lm.landmark], dtype=float)
        mundo = (np.array([(l.x, l.y, l.z) for l in mundos[k].landmark], dtype=float) if k < len(mundos)
                 else img / max(ancho, 1))
        etiqueta = resultado.multi_handedness[k].classification[0]
        salida.append(DeteccionMano(MP_A_LADO.get(etiqueta.label, "Der"), float(etiqueta.score), img, mundo))
    return salida


def otro_lado(lado):
    return "Der" if lado == "Izq" else "Izq"


class SeguidorManos:
    """Asigna cada mano detectada a su ranura ("Izq" o "Der") de forma estable entre frames.

    MediaPipe da la lateralidad en cada frame, pero con datos reales a veces repite la misma mano dos veces
    o la etiqueta parpadea unos frames (puño, mano de lado). Reglas:
      1. Dos detecciones casi en el mismo lugar son la misma mano: se queda la más confiable.
      2. Continuidad: una mano que aparece donde estaba una ranura activa sigue en esa ranura aunque la
         etiqueta parpadee; así sus trayectorias y su barra no se mezclan con las de la otra mano.
      3. Las manos nuevas van a la ranura que indica su etiqueta (o a la que quede libre).
      4. Si la evidencia acumulada (EMA) dice de forma sostenida que la mano es la otra, se corrige.
    """

    def __init__(self):
        self.ranuras = {lado: {"centro": None, "escala": 1.0, "t": -10.0, "evid_der": 0.5} for lado in LADOS}

    @staticmethod
    def _sin_duplicados(dets):
        if len(dets) < 2:
            return list(dets)
        a, b = dets[0], dets[1]
        if float(np.linalg.norm(a.centro - b.centro)) < DUPLICADO_DIST * max(a.escala, b.escala):
            return [max((a, b), key=lambda d: d.confianza)]
        return list(dets[:2])

    def asignar(self, detecciones, ahora):
        """Devuelve {lado: DeteccionMano} con a lo más una mano por lado."""
        dets = self._sin_duplicados(detecciones)
        asignadas, usadas = {}, set()
        # 1) Continuidad: cada ranura activa conserva la detección más cercana a donde estaba
        pares = []
        for lado in LADOS:
            r = self.ranuras[lado]
            if r["centro"] is None or ahora - r["t"] > CONTINUIDAD_SEG:
                continue
            for k, d in enumerate(dets):
                distancia = float(np.linalg.norm(d.centro - r["centro"])) / r["escala"]
                if distancia < CONTINUIDAD_DIST:
                    pares.append((distancia, lado, k))
        for _, lado, k in sorted(pares):
            if lado not in asignadas and k not in usadas:
                asignadas[lado] = dets[k]
                usadas.add(k)
        # 2) Manos nuevas: a la ranura de su etiqueta, o a la libre
        for k, d in enumerate(dets):
            if k in usadas:
                continue
            lado = d.lado_mp if d.lado_mp not in asignadas else otro_lado(d.lado_mp)
            if lado in asignadas:
                continue
            asignadas[lado] = d
            usadas.add(k)
            d.nueva = True
            self.ranuras[lado]["evid_der"] = d.confianza if d.lado_mp == "Der" else 1 - d.confianza
        # 3) Evidencia acumulada de lateralidad: corrige una mano que quedó en la ranura equivocada
        contrarias = []
        for lado, d in asignadas.items():
            r = self.ranuras[lado]
            p_der = d.confianza if d.lado_mp == "Der" else 1 - d.confianza
            r["evid_der"] += (p_der - r["evid_der"]) * LATERALIDAD_SUAVIZADO
            evidencia = r["evid_der"] if lado == "Der" else 1 - r["evid_der"]
            if evidencia < 1 - LATERALIDAD_CAMBIO:
                contrarias.append(lado)
        if len(contrarias) == 2 or (len(contrarias) == 1 and otro_lado(contrarias[0]) not in asignadas):
            evid = {lado: self.ranuras[lado]["evid_der"] for lado in LADOS}
            nuevas = {}
            for lado, d in asignadas.items():
                if lado in contrarias:
                    d.nueva = True
                    nuevas[otro_lado(lado)] = d
                    self.ranuras[otro_lado(lado)]["evid_der"] = evid[lado]
                else:
                    nuevas[lado] = d
            asignadas = nuevas
        # 4) Posición de cada ranura para el siguiente frame
        for lado, d in asignadas.items():
            r = self.ranuras[lado]
            r["centro"], r["escala"], r["t"] = d.centro, d.escala, ahora
        return asignadas


@dataclass
class EstadoMano:
    """Lo que pasó con UNA mano en este frame."""
    lado: str
    rasgos: Optional[RasgosMano] = None
    pose_cruda: Optional[str] = None        # configuración del frame
    pose: Optional[str] = None              # configuración estable (estabilizador)
    gesto: Optional[str] = None             # seña con movimiento completada en este frame
    moviendo: bool = False
    progreso: float = 0.0                   # avance del hold-to-type (0 a 1)


class CanalMano:
    """Todo el estado de UNA mano: filtro antivibración, estabilizador, trayectoria y reconocedor de trazos.
    Hay un canal por lado, así que las trayectorias de las dos manos nunca se cruzan."""

    def __init__(self, lado):
        self.lado = lado
        self.filtro_img = FiltroOneEuro()
        self.filtro_mundo = FiltroOneEuro()
        self.estabilizador = Estabilizador(FRAMES_ESTABLES)
        self.tray = Trayectoria()
        self.reconocedor = ReconocedorDinamico()
        self.barra = ProgresoSuave()
        self.perdidos = 0

    def procesar(self, det, ahora):
        if det is None:
            self.perdidos += 1
            if self.perdidos > DIN_PERDIDOS_MAX:
                self.filtro_img.reiniciar()
                self.filtro_mundo.reiniciar()
            self.estabilizador.actualizar(None)
            self.tray.perder()
            return EstadoMano(self.lado)
        if det.nueva:                              # otra mano entró a esta ranura: no mezclar su historia
            self.filtro_img.reiniciar()
            self.filtro_mundo.reiniciar()
            self.tray.limpiar()
        self.perdidos = 0
        img = self.filtro_img(det.img, ahora, det.escala)
        mundo = self.filtro_mundo(det.mundo, ahora, float(np.linalg.norm(det.mundo[MIDDLE_MCP] - det.mundo[WRIST])))
        r = analizar_mano(img, mundo, self.lado)
        pose_cruda = clasificar_pose(r)
        pose = self.estabilizador.actualizar(pose_cruda)
        self.tray.agregar(ahora, r, pose_cruda)
        gesto = self.reconocedor.evaluar(self.tray, ahora)
        return EstadoMano(self.lado, r, pose_cruda, pose, gesto, self.tray.vel > V_QUIETO)


# =============================================================================
# 7. CANAL NO MANUAL — pose 3D de la cabeza (solvePnP) y cejas
# =============================================================================
# Índices de Face Mesh (468 puntos). "Der./izq." son del sujeto.
CEJA_SUP_DER = (70, 63, 105, 66, 107)       # Contorno superior de la ceja derecha
CEJA_SUP_IZQ = (300, 293, 334, 296, 336)    # Contorno superior de la ceja izquierda
CANTO_DER = (33, 133)                       # Comisuras del ojo derecho (óseas: no se mueven al parpadear)
CANTO_IZQ = (362, 263)
PARPADO_DER = (159, 145)                    # Párpado superior e inferior
PARPADO_IZQ = (386, 374)
FRENTE, MENTON, NARIZ = 10, 152, 1
POMULO_DER, POMULO_IZQ = 234, 454
# Puntos rígidos para la pose: nariz, comisuras de los ojos, frente, mentón y pómulos (sin cejas ni boca,
# que se mueven con la expresión)
CARA_PNP = (1, 4, 5, 6, 168, 197, 195, 33, 133, 362, 263, 10, 152, 234, 454)
PUNTOS_CEJAS = (*CEJA_SUP_DER, *CEJA_SUP_IZQ, *CANTO_DER, *CANTO_IZQ, FRENTE, MENTON)
# Modelo canónico de la cara de MediaPipe (canonical_face_model.obj, en cm), pasado a los ejes de la cámara
# de OpenCV: x a la derecha de la imagen, y hacia abajo, z hacia adentro (la nariz es lo más cercano).
MODELO_CARA = {
    1: (0.0, 1.126865, -7.475604), 4: (0.0, 0.46317, -7.58658), 5: (0.0, -0.365669, -7.24287),
    6: (0.0, -2.473255, -5.788627), 168: (0.0, -3.271027, -5.236015), 197: (0.0, -1.728369, -6.31675),
    195: (0.0, -1.059413, -6.774605), 10: (0.0, -8.261778, -4.481535), 152: (0.0, 9.403378, -4.264492),
    33: (-4.445859, -2.663991, -3.173422), 133: (-1.856432, -2.585245, -3.757904),
    362: (1.856432, -2.585245, -3.757904), 263: (4.445859, -2.663991, -3.173422),
    234: (-7.664182, -0.673132, 2.435867), 454: (7.664182, -0.673132, 2.435867),
    70: (-5.720968, -4.254584, -2.830852), 63: (-4.985894, -4.802461, -3.751977),
    105: (-3.986562, -5.109487, -4.466315), 66: (-2.760292, -5.100971, -5.01599),
    107: (-1.395634, -5.011963, -5.316032), 300: (5.720968, -4.254584, -2.830852),
    293: (4.985894, -4.802461, -3.751977), 334: (3.986562, -5.109487, -4.466315),
    296: (2.760292, -5.100971, -5.01599), 336: (1.395634, -5.011963, -5.316032),
}
MODELO_PNP = np.array([MODELO_CARA[i] for i in CARA_PNP], dtype=float)
MODELO_CEJAS = np.array([MODELO_CARA[i] for i in PUNTOS_CEJAS], dtype=float)
INDICES_ROSTRO = tuple(sorted(set(CARA_PNP) | set(PUNTOS_CEJAS) | set(PARPADO_DER) | set(PARPADO_IZQ)))


@dataclass
class PoseCabeza:
    pitch: float             # + = mira hacia abajo (asentir)
    yaw: float               # + = gira hacia la derecha de la imagen (negar)
    roll: float              # + = inclina la cabeza hacia la derecha de la imagen
    distancia: float         # cm a la cámara (aproximada)
    error: float             # error de reproyección / alto de la cara (alto = puntos tapados o deformados)
    rvec: np.ndarray
    tvec: np.ndarray


def matriz_camara(ancho, alto):
    """Cámara aproximada: foco = ancho de la imagen (≈ 53° de campo horizontal), centro óptico al centro."""
    return np.array([[ancho, 0, ancho / 2], [0, ancho, alto / 2], [0, 0, 1]], dtype=float)


class EstimadorPoseCabeza:
    """Pose 3D de la cabeza con cv2.solvePnP: empareja puntos rígidos de Face Mesh con su posición en el
    modelo canónico y obtiene la rotación real en grados. Parte de la pose anterior (más estable y rápido)."""

    def __init__(self):
        self.rvec = self.tvec = None

    def reiniciar(self):
        self.rvec = self.tvec = None

    def estimar(self, f, ancho, alto):
        K = matriz_camara(ancho, alto)
        img = np.ascontiguousarray(f[list(CARA_PNP), :2], dtype=float)
        try:
            if self.rvec is None:
                ok, rvec, tvec = cv2.solvePnP(MODELO_PNP, img, K, None, flags=cv2.SOLVEPNP_EPNP)
                if ok:
                    ok, rvec, tvec = cv2.solvePnP(MODELO_PNP, img, K, None, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
            else:
                ok, rvec, tvec = cv2.solvePnP(MODELO_PNP, img, K, None, self.rvec.copy(), self.tvec.copy(), True,
                                              cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            ok = False
        if not ok or not np.all(np.isfinite(rvec)) or float(tvec[2]) <= 0:
            self.reiniciar()
            return None
        self.rvec, self.tvec = rvec, tvec
        R, _ = cv2.Rodrigues(rvec)
        frente = -R[:, 2]          # hacia dónde mira la cara (hacia la cámara cuando está de frente)
        arriba = -R[:, 1]
        proyectados, _ = cv2.projectPoints(MODELO_PNP, rvec, tvec, K, None)
        error = float(np.mean(np.linalg.norm(proyectados[:, 0] - img, axis=1))) / max(dist(f[FRENTE], f[MENTON]), 1.0)
        return PoseCabeza(pitch=math.degrees(math.asin(max(-1.0, min(1.0, float(frente[1]))))),
                          yaw=math.degrees(math.atan2(float(frente[0]), -float(frente[2]))),
                          roll=math.degrees(math.atan2(float(arriba[0]), -float(arriba[1]))),
                          distancia=float(tvec[2]), error=error, rvec=rvec, tvec=tvec)


def altura_cejas(f):
    """Altura media de las cejas sobre la línea de las comisuras de los ojos, medida a lo largo del eje de la
    cara (mentón -> frente) y dividida entre el alto del rostro. `f` se indexa por número de punto -> (x, y).
    Se usa la comisura y no el párpado: el párpado sube y baja al parpadear o mirar abajo."""
    alto_cara = dist(f[FRENTE], f[MENTON])
    if alto_cara < 1e-6:
        return None
    eje = unitario(restar(f[FRENTE], f[MENTON]))

    def altura(ref, ids):
        return sum((f[i][0] - ref[0]) * eje[0] + (f[i][1] - ref[1]) * eje[1] for i in ids) / len(ids)

    ref_der = punto_medio(f[CANTO_DER[0]], f[CANTO_DER[1]])
    ref_izq = punto_medio(f[CANTO_IZQ[0]], f[CANTO_IZQ[1]])
    return (altura(ref_der, CEJA_SUP_DER) + altura(ref_izq, CEJA_SUP_IZQ)) / 2 / alto_cara


def cejas_esperadas(pose, ancho, alto):
    """Altura de cejas que tendría el modelo canónico (cara neutra) con esta pose, dividida entre la que tiene de
    frente. Es lo que cambia la medida 2D SOLO por inclinar o girar la cabeza (escorzo)."""
    K = matriz_camara(ancho, alto)
    con_pose, _ = cv2.projectPoints(MODELO_CEJAS, pose.rvec, pose.tvec, K, None)
    de_frente, _ = cv2.projectPoints(MODELO_CEJAS, np.zeros(3), pose.tvec, K, None)
    a = altura_cejas(dict(zip(PUNTOS_CEJAS, con_pose[:, 0])))
    b = altura_cejas(dict(zip(PUNTOS_CEJAS, de_frente[:, 0])))
    return (a / b) if (a and b) else 1.0


def calcular_rasgos_faciales(f, pose, ancho, alto):
    """Rasgos faciales a partir de los puntos de Face Mesh (`f`: arreglo 468x3 en píxeles) y la pose 3D.

    - cejas: altura de cejas compensada por la pose. La medida 2D se divide entre la que tendría el modelo
             canónico con la misma pose: lo que cambia solo por inclinar (pitch) o girar la cabeza se cancela
             y queda el movimiento real de las cejas.
    - ojos:  apertura de los ojos (párpados / ancho del ojo).
    Devuelve None si la geometría es degenerada.
    """
    medida = altura_cejas(f)
    if medida is None or medida <= 0:
        return None
    esperada = cejas_esperadas(pose, ancho, alto) if pose is not None else 1.0
    ancho_ojos = dist(f[CANTO_DER[0]], f[CANTO_DER[1]]) + dist(f[CANTO_IZQ[0]], f[CANTO_IZQ[1]])
    ojos = (dist(f[PARPADO_DER[0]], f[PARPADO_DER[1]]) + dist(f[PARPADO_IZQ[0]], f[PARPADO_IZQ[1]])) / max(ancho_ojos, 1e-6)
    return {"cejas": medida / max(esperada, 1e-6), "cejas_2d": medida, "ojos": ojos}


class DetectorOscilacion:
    """Detecta un movimiento de vaivén (negar o asentir) en una señal 1D (ángulo de la cabeza en grados).

    Mantiene una "posición neutra" que se adapta lentamente (EMA), así que mirar a un lado y
    quedarse ahí no cuenta. Cuenta las veces que la señal ENTRA en la zona positiva o negativa
    (con histéresis para ignorar el temblor) dentro de una ventana de tiempo.
    """

    def __init__(self, amplitud, entradas_min, requiere_ambos_lados):
        self.amplitud = amplitud
        self.entradas_min = entradas_min
        self.requiere_ambos_lados = requiere_ambos_lados
        self.reiniciar()

    def reiniciar(self):
        self.centro = None
        self.t_anterior = None
        self.zona = 0
        self.entradas = deque()       # (tiempo, lado) de cada entrada a una zona
        self.desviacion = 0.0

    def actualizar(self, t, valor):
        """Agrega una muestra. Devuelve True si se completó el movimiento."""
        if self.centro is None:
            self.centro = valor
        else:
            dt = max(0.0, t - self.t_anterior)
            self.centro += (valor - self.centro) * min(1.0, dt / OSC_TAU_CENTRO)
        self.t_anterior = t
        d = valor - self.centro
        self.desviacion = d

        # Máquina de zonas con histéresis: se entra con |d| > amplitud y se sale con |d| < amplitud/2
        if self.zona == 0:
            nueva = 1 if d > self.amplitud else (-1 if d < -self.amplitud else 0)
        elif self.zona * d < self.amplitud / 2:
            nueva = -self.zona if abs(d) > self.amplitud else 0
        else:
            nueva = self.zona
        if nueva != 0 and nueva != self.zona:
            self.entradas.append((t, nueva))
        self.zona = nueva

        while self.entradas and t - self.entradas[0][0] > OSC_VENTANA:
            self.entradas.popleft()

        lados = {lado for _, lado in self.entradas}
        if len(self.entradas) >= self.entradas_min and (not self.requiere_ambos_lados or len(lados) == 2):
            self.entradas.clear()
            return True
        return False


class AnalizadorFacial:
    """Interpreta los componentes no manuales: cejas (pregunta), negar y asentir con la cabeza.

    Al iniciar (y con la tecla N) calibra la cara neutra: guarda la línea base personal de las cejas y de
    la apertura de los ojos, y la pose neutra de la cabeza."""

    def __init__(self):
        self.pose = EstimadorPoseCabeza()
        self.negacion = DetectorOscilacion(NEG_AMPLITUD, NEG_ENTRADAS_MIN, requiere_ambos_lados=True)
        self.afirmacion = DetectorOscilacion(AFIRM_AMPLITUD, AFIRM_ENTRADAS_MIN, requiere_ambos_lados=False)
        self.ultimo_evento, self.t_evento = None, -10.0
        self.enfriamiento_hasta = 0.0
        self.cejas_pausa_hasta = 0.0
        self.rasgos = None
        self.pose_actual = None
        self.modo = "SIN_ROSTRO"
        self.recalibrar()

    def recalibrar(self):
        """Vuelve a medir la cara neutra (tecla N)."""
        self.muestras_calibracion = []
        self.base_cejas = None
        self.base_ojos = None
        self.pose_neutra = None
        self.ratio_cejas = 1.0
        self.ratio_ojos = 1.0
        self.pregunta_hecha = False
        self._reiniciar_cejas()

    def _reiniciar_cejas(self):
        self.cejas_inicio = None
        self.ultima_arriba = -10.0
        self.progreso_pregunta = 0.0

    @property
    def progreso_calibracion(self):
        return 1.0 if self.base_cejas is not None else len(self.muestras_calibracion) / FRAMES_CALIBRACION

    def _perder_rostro(self):
        self.rasgos = None
        self.pose_actual = None
        self.pose.reiniciar()
        self.negacion.reiniciar()
        self.afirmacion.reiniciar()
        self._reiniciar_cejas()

    def actualizar(self, puntos, t, tamano):
        """Procesa un frame. `puntos`: arreglo 468x3 en píxeles (o None si no hay cara); `tamano`: (ancho, alto).

        Devuelve el gesto reconocido en este frame ("PREGUNTA", "NEGACION", "AFIRMACION") o None.
        """
        ancho, alto = tamano
        pose = self.pose.estimar(puntos, ancho, alto) if puntos is not None else None
        if pose is not None and pose.error > CARA_ERROR_MAX:
            self._perder_rostro()            # una mano tapa la cara: sus puntos no son confiables
            self.pose_actual = pose
            self.modo = "TAPADO"
            return None
        rasgos = calcular_rasgos_faciales(puntos, pose, ancho, alto) if pose is not None else None
        if rasgos is None:
            self._perder_rostro()
            self.modo = "SIN_ROSTRO"
            return None
        self.rasgos, self.pose_actual = rasgos, pose

        # 1) Calibración de la cara neutra: cejas (percentil bajo), ojos y pose de la cabeza (medianas)
        if self.base_cejas is None:
            self.muestras_calibracion.append((rasgos["cejas"], rasgos["ojos"], pose.pitch, pose.yaw))
            if len(self.muestras_calibracion) >= FRAMES_CALIBRACION:
                m = np.array(self.muestras_calibracion)
                self.base_cejas = float(np.percentile(m[:, 0], CALIB_PERCENTIL))
                self.base_ojos = float(np.median(m[:, 1]))
                self.pose_neutra = (float(np.median(m[:, 2])), float(np.median(m[:, 3])))
            self.modo = "CALIBRANDO"
            return None

        evento = None

        # 2) Movimientos de cabeza con los ángulos reales: negar = yaw, asentir = pitch
        negar = self.negacion.actualizar(t, pose.yaw)
        asentir = self.afirmacion.actualizar(t, pose.pitch)
        if t >= self.enfriamiento_hasta and (negar or asentir):
            evento = "NEGACION" if negar else "AFIRMACION"
            self.enfriamiento_hasta = t + ENFRIAMIENTO_CABEZA
            self.negacion.entradas.clear()
            self.afirmacion.entradas.clear()
            self.cejas_pausa_hasta = t + CEJAS_PAUSA_TRAS_CABEZA   # negar/asentir no deben contar como pregunta
            self._reiniciar_cejas()

        # 3) Cejas (pregunta): razón respecto a la línea base, con histéresis y tolerancia a caídas breves
        self.ratio_cejas = rasgos["cejas"] / self.base_cejas
        self.ratio_ojos = rasgos["ojos"] / max(self.base_ojos, 1e-6)
        if t >= self.cejas_pausa_hasta:
            if self.ratio_cejas >= CEJAS_UMBRAL:
                self.ultima_arriba = t
                if self.cejas_inicio is None:
                    self.cejas_inicio = t
            elif self.ratio_cejas >= CEJAS_UMBRAL_SALIDA and self.cejas_inicio is not None:
                self.ultima_arriba = t       # banda de histéresis: sigue contando
            if self.cejas_inicio is not None and t - self.ultima_arriba > CEJAS_TOLERANCIA:
                self._reiniciar_cejas()      # las cejas bajaron de verdad
            if self.cejas_inicio is not None and t - self.cejas_inicio > CEJAS_MAX_SEGUNDOS:
                self.base_cejas = rasgos["cejas"]   # arriba demasiado tiempo: es la nueva cara neutra
                self.ratio_cejas = 1.0
                self._reiniciar_cejas()
            if self.cejas_inicio is not None:
                self.progreso_pregunta = min(1.0, (t - self.cejas_inicio) / CEJAS_SEGUNDOS)
                if self.progreso_pregunta >= 1.0 and not self.pregunta_hecha and evento is None:
                    evento = "PREGUNTA"
                    self.pregunta_hecha = True   # Una sola "?" por cada vez que se levantan las cejas
            elif self.pregunta_hecha and self.ratio_cejas < CEJAS_UMBRAL_SALIDA and t - self.ultima_arriba > CEJAS_REARME:
                self.pregunta_hecha = False
            # Reajuste lento de la línea base mientras la cara está neutra (luz, postura...)
            if self.cejas_inicio is None and abs(self.ratio_cejas - 1.0) < CEJAS_ZONA_ADAPT:
                self.base_cejas += (rasgos["cejas"] - self.base_cejas) * CEJAS_ADAPTACION

        # 4) Estado que se muestra en pantalla
        if evento:
            self.ultimo_evento, self.t_evento = evento, t
        if self.ultimo_evento and t - self.t_evento < DURACION_AVISO:
            self.modo = self.ultimo_evento
        elif self.cejas_inicio is not None and not self.pregunta_hecha:
            self.modo = "CEJAS"
        else:
            self.modo = "NEUTRO"
        return evento


# =============================================================================
# 8. ESCRITURA (hold-to-type por mano) Y ORQUESTACIÓN
# =============================================================================
class Estabilizador:
    """Solo acepta una seña nueva cuando se repite N frames seguidos (filtro anti-parpadeo)."""

    def __init__(self, n):
        self.n = n
        self.estable = None
        self.candidato = None
        self.conteo = 0

    def actualizar(self, sena):
        if sena == self.estable:
            self.candidato, self.conteo = sena, 0
        elif sena == self.candidato:
            self.conteo += 1
            if self.conteo >= self.n:
                self.estable, self.conteo = sena, 0
        else:
            self.candidato, self.conteo = sena, 1
        return self.estable


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


class ProgresoSuave:
    """Suaviza la barra al subir y la reinicia al instante si la seña cambia."""

    def __init__(self, suavizado=0.35):
        self.valor = 0.0
        self.suavizado = suavizado

    def actualizar(self, objetivo):
        if objetivo < self.valor - 1e-3:
            self.valor = objetivo
        else:
            self.valor += (objetivo - self.valor) * self.suavizado
        return self.valor


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


# =============================================================================
# 9. VOZ ASÍNCRONA Y CAPA DE HARDWARE
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


# =============================================================================
# 10. INTERFAZ GRÁFICA
# =============================================================================
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
CONEXIONES_MANO = tuple(mp_hands.HAND_CONNECTIONS)


@dataclass
class VistaMano:
    """Cómo se muestra una mano: seña visible, estado ("sostener", "mover", "registrada", "pose") y progreso."""
    sena: Optional[str]
    estado: str
    progreso: float


@dataclass
class EstadoUI:
    """Todo lo que necesitan las ventanas para dibujarse en un frame."""
    sena: Optional[str]         # seña de la mano principal (la que se muestra en grande)
    estado_sena: str            # "ninguna" | "sostener" | "mover" | "registrada" | "pose"
    progreso: float
    lado: Optional[str]         # mano principal ("Izq" / "Der")
    vistas: dict                # lado -> VistaMano de cada mano
    texto: str
    ahora: float
    modo_facial: str
    progreso_pregunta: float
    ratio_cejas: float
    calib: float                # 0..1: progreso de la calibración del rostro (1 = lista)
    pose_cabeza: Optional[PoseCabeza]
    fps: float
    voz: str


def vista_mano(em, sostener, ahora):
    """Qué se muestra de una mano: el trazo recién hecho, la seña que se sostiene, o nada."""
    gesto, t_gesto = sostener.destello
    if gesto and ahora - t_gesto < DURACION_DESTELLO:
        return VistaMano(gesto, "registrada", 1.0)
    if em.pose is not None and em.rasgos is not None and em.pose not in POSES_SILENCIOSAS:
        if em.pose not in LETRAS_TIPEABLES:
            return VistaMano(em.pose, "pose", em.progreso)
        if em.moviendo:
            return VistaMano(em.pose, "mover", em.progreso)
        return VistaMano(em.pose, "registrada" if sostener.registrada else "sostener", em.progreso)
    return VistaMano(None, "ninguna", 0.0)


def mano_principal(vistas, anterior):
    """La mano que se muestra en grande: la que acaba de registrar o la que más avanzó; en empate, la anterior."""
    candidatas = [lado for lado in LADOS if vistas[lado].sena is not None]
    if not candidatas:
        return None
    return max(candidatas, key=lambda lado: (vistas[lado].estado == "registrada", vistas[lado].progreso,
                                             lado == anterior))


# --- Texto (con soporte para la Ñ y sin guiones bajos) ---------------------------
def medir_texto(texto, escala, grosor=1, fuente=FUENTE):
    limpio = _a_ascii(limpiar_texto(texto)).replace("Ñ", "N").replace("ñ", "n")
    (w, h), base = cv2.getTextSize(limpio, fuente, escala, grosor)
    return w, h


def poner_texto(img, texto, org, escala=0.5, color=TEXTO, grosor=1, fuente=FUENTE):
    """cv2.putText que reemplaza "_" por " " y dibuja la Ñ (las fuentes Hershey no la incluyen)."""
    texto = _a_ascii(limpiar_texto(texto))
    x, y = org
    trozo = ""

    def volcar():
        nonlocal x, trozo
        if trozo:
            cv2.putText(img, trozo, (int(x), int(y)), fuente, escala, color, grosor, cv2.LINE_AA)
            x += cv2.getTextSize(trozo, fuente, escala, grosor)[0][0]
            trozo = ""

    for ch in texto:
        if ch in "Ññ":
            volcar()
            base = "N" if ch == "Ñ" else "n"
            (w, h), _ = cv2.getTextSize(base, fuente, escala, grosor)
            cv2.putText(img, base, (int(x), int(y)), fuente, escala, color, grosor, cv2.LINE_AA)
            ancho_t = max(int(w * 0.72), 4)       # tilde: un periodo de onda seno, centrado sobre la letra
            cx, cy = int(x + w / 2), int(y - h - max(4, int(h * 0.20)))
            amplitud = max(int(h * 0.10), 1)
            u = np.linspace(0.0, 1.0, 16)
            onda = np.stack([cx - ancho_t / 2 + ancho_t * u, cy - amplitud * np.sin(2 * np.pi * u)], axis=1)
            cv2.polylines(img, [onda.astype(np.int32).reshape(-1, 1, 2)], False, color,
                          max(1, int(grosor * 0.7)), cv2.LINE_AA)
            x += w
        else:
            trozo += ch
    volcar()


def texto_centrado(img, texto, cx, cy, escala, color, grosor, fuente=FUENTE):
    w, h = medir_texto(texto, escala, grosor, fuente)
    poner_texto(img, texto, (cx - w / 2, cy + h / 2), escala, color, grosor, fuente)


def escala_para_ancho(texto, ancho_max, escala_max, grosor, fuente=FUENTE):
    w, _ = medir_texto(texto, escala_max, grosor, fuente)
    return escala_max if w <= ancho_max else escala_max * ancho_max / w


def ajustar_lineas(texto, ancho_max, escala, fuente=FUENTE):
    """Parte un texto en líneas que quepan en `ancho_max` píxeles."""
    lineas, actual = [], ""
    for palabra in limpiar_texto(texto).split():
        prueba = (actual + " " + palabra).strip()
        if medir_texto(prueba, escala, 1, fuente)[0] > ancho_max and actual:
            lineas.append(actual)
            actual = palabra
        else:
            actual = prueba
    return lineas + [actual] if actual else lineas


# --- Formas ---------------------------------------------------------------------
def rect_redondeado(img, x0, y0, x1, y1, radio, color):
    """Rectángulo relleno con esquinas redondeadas."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    radio = int(max(0, min(radio, (x1 - x0) // 2, (y1 - y0) // 2)))
    if x1 <= x0 or y1 <= y0:
        return
    cv2.rectangle(img, (x0 + radio, y0), (x1 - radio, y1), color, -1)
    cv2.rectangle(img, (x0, y0 + radio), (x1, y1 - radio), color, -1)
    for cx, cy in ((x0 + radio, y0 + radio), (x1 - radio, y0 + radio),
                   (x0 + radio, y1 - radio), (x1 - radio, y1 - radio)):
        cv2.circle(img, (cx, cy), radio, color, -1, cv2.LINE_AA)


def aclarar(color, factor=0.45):
    return tuple(int(c + (255 - c) * factor) for c in color)


def barra_progreso(img, x0, y0, x1, y1, progreso, color):
    """Barra moderna: pista oscura redondeada, relleno de color con brillo superior."""
    alto = y1 - y0
    rect_redondeado(img, x0, y0, x1, y1, alto // 2, (55, 52, 50))
    progreso = max(0.0, min(1.0, progreso))
    x_fin = x0 + (x1 - x0) * progreso
    if x_fin - x0 >= 2:
        rect_redondeado(img, x0, y0, x_fin, y1, alto // 2, color)
        rect_redondeado(img, x0 + 3, y0 + 2, x_fin - 3, y0 + alto * 0.4, alto // 4, aclarar(color))


def tarjeta(img, x0, y0, x1, y1, color=TARJETA, borde=BORDE):
    rect_redondeado(img, x0, y0, x1, y1, 10, borde)
    rect_redondeado(img, x0 + 1, y0 + 1, x1 - 1, y1 - 1, 9, color)


def esquinas(img, x0, y0, x1, y1, color, largo=22, grosor=3):
    """Recuadro elegante: solo las esquinas marcadas."""
    for (x, y, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (x, y), (x + dx * largo, y), color, grosor, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + dy * largo), color, grosor, cv2.LINE_AA)


def pastilla_modo(img, x_derecha, y, modo, progreso=0.0):
    """Indicador "Modo: ..." alineado a la derecha, con punto de color y mini-barra de carga."""
    nombre, color = MODOS_FACIALES[modo]
    etiqueta = f"Modo: {nombre}"
    w, h = medir_texto(etiqueta, 0.5)
    x0, x1 = x_derecha - w - 44, x_derecha
    rect_redondeado(img, x0, y, x1, y + 28, 14, (35, 32, 30))
    cv2.circle(img, (x0 + 16, y + 14), 6, color, -1, cv2.LINE_AA)
    poner_texto(img, etiqueta, (x0 + 30, y + 14 + h // 2), 0.5, color, 1)
    if modo == "CEJAS":
        barra_progreso(img, x0 + 10, y + 32, x1 - 10, y + 38, progreso, ACENTO)


def resumen_manos(vistas):
    """Texto "Izq: B | Der: O" con lo que hace cada mano en este momento."""
    return "  |  ".join(f"{lado}: {vistas[lado].sena or '-'}" for lado in LADOS)


def medidor_cejas(img, x, y, ui):
    """Medidor SIEMPRE visible de las cejas: cuánto se levantan respecto a tu línea base (ya compensado por la
    pose de la cabeza), y debajo los ángulos reales de la cabeza."""
    ancho = 150
    rect_redondeado(img, x, y, x + ancho + 20, y + 62, 8, (35, 32, 30))
    poner_texto(img, "CEJAS", (x + 10, y + 16), 0.42, TEXTO_SUAVE, 1)
    if ui.pose_cabeza is not None:
        pc = ui.pose_cabeza
        poner_texto(img, f"P{pc.pitch:+3.0f} Y{pc.yaw:+3.0f} R{pc.roll:+3.0f}", (x + 10, y + 56), 0.38,
                    TEXTO_SUAVE, 1)
    else:
        poner_texto(img, "cabeza: -", (x + 10, y + 56), 0.38, TEXTO_SUAVE, 1)
    if ui.calib < 1.0:
        poner_texto(img, f"calibrando {int(ui.calib * 100)}%", (x + 66, y + 16), 0.42, AMBAR, 1)
        barra_progreso(img, x + 10, y + 26, x + 10 + ancho, y + 36, ui.calib, AMBAR)
        return
    minimo, maximo = 0.95, 1.20
    frac = (ui.ratio_cejas - minimo) / (maximo - minimo)
    umbral = (CEJAS_UMBRAL - minimo) / (maximo - minimo)
    color = EXITO if ui.modo_facial == "PREGUNTA" else (ACENTO if ui.ratio_cejas >= CEJAS_UMBRAL else TEXTO_SUAVE)
    poner_texto(img, f"x{ui.ratio_cejas:.2f}", (x + 66, y + 16), 0.42, color, 1)
    barra_progreso(img, x + 10, y + 26, x + 10 + ancho, y + 36, frac, color)
    tx = int(x + 10 + ancho * max(0.0, min(1.0, umbral)))
    cv2.line(img, (tx, y + 22), (tx, y + 40), (255, 255, 255), 1, cv2.LINE_AA)   # marca del umbral


def dibujar_estela(frame, tray, punto):
    """Dibuja el recorrido reciente del punto rastreado (se ve cómo se traza la J, la Z, el ?...)."""
    pts = tray.serie_px(punto)
    for i in range(1, len(pts)):
        k = i / len(pts)
        color = (int(60 + 195 * k), int(180 * k + 40), int(255 * (1 - k * 0.4)))
        cv2.line(frame, (int(pts[i - 1][0]), int(pts[i - 1][1])), (int(pts[i][0]), int(pts[i][1])), color,
                 2 + int(3 * k), cv2.LINE_AA)
    if pts:
        cv2.circle(frame, (int(pts[-1][0]), int(pts[-1][1])), 6, (255, 255, 255), -1, cv2.LINE_AA)


def dibujar_mano(frame, em, vista, tray):
    """Esqueleto de una mano (color según el lado), recuadro, etiqueta "Izq: B", mini-barra, pista y estela."""
    alto, ancho = frame.shape[:2]
    p = em.rasgos.p
    color_estado = {"registrada": EXITO, "sostener": ACENTO, "mover": TEXTO_SUAVE, "pose": AMBAR}.get(
        vista.estado, TEXTO_SUAVE)
    for a, b in CONEXIONES_MANO:
        cv2.line(frame, (int(p[a][0]), int(p[a][1])), (int(p[b][0]), int(p[b][1])), (230, 230, 230), 2, cv2.LINE_AA)
    for k, (x, y) in enumerate(p):
        cv2.circle(frame, (int(x), int(y)), 5 if k in (4, 8, 12, 16, 20) else 3, COLOR_LADO[em.lado], -1, cv2.LINE_AA)

    pad = 20
    x0, y0 = max(int(p[:, 0].min()) - pad, 0), max(int(p[:, 1].min()) - pad, 0)
    x1, y1 = min(int(p[:, 0].max()) + pad, ancho - 1), min(int(p[:, 1].max()) + pad, alto - 1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color_estado, 1)
    esquinas(frame, x0, y0, x1, y1, color_estado)

    etiqueta = f"{em.lado}: {vista.sena if vista.sena else '?'}"
    if vista.estado == "pose" and vista.sena in PISTA_TRAZO:
        etiqueta += " (mueve)"
    tw, th = medir_texto(etiqueta, 0.7, 2, FUENTE_TITULO)
    ey = max(y0 - 14, th + 14)
    rect_redondeado(frame, x0, ey - th - 12, x0 + tw + 20, ey + 4, 6, color_estado)
    cv2.circle(frame, (x0 + 1, ey - th // 2 - 4), 5, COLOR_LADO[em.lado], -1, cv2.LINE_AA)
    poner_texto(frame, etiqueta, (x0 + 10, ey - 4), 0.7, (20, 20, 20), 2, FUENTE_TITULO)
    if vista.estado in ("sostener", "registrada"):
        barra_progreso(frame, x0, ey + 6, x0 + tw + 20, ey + 12, vista.progreso, EXITO)
    pista = PISTA_TRAZO.get(vista.sena) if vista.estado in ("sostener", "registrada", "mover") else None
    if pista:
        poner_texto(frame, pista, (x0 + 4, min(y1 + 18, alto - 60)), 0.45, AMBAR, 1)
    if vista.sena in PUNTO_DE_POSE and tray.vel > 0.6:   # la estela solo se dibuja mientras la mano se mueve
        dibujar_estela(frame, tray, PUNTO_DE_POSE[vista.sena])


def dibujar_camara(frame, ui, manos, canales, rostro, analizador, diagnostico):
    """Ventana de cámara: las dos manos con su etiqueta, medidor de cejas y pose de la cabeza, modo facial,
    resumen "Izq: B | Der: O" y barra de la mano principal."""
    alto, ancho = frame.shape[:2]

    # --- Puntos faciales clave (cejas y comisuras) ---
    if rostro is not None:
        for i in (*CEJA_SUP_DER, *CEJA_SUP_IZQ, *CANTO_DER, *CANTO_IZQ):
            cv2.circle(frame, (int(rostro[i][0]), int(rostro[i][1])), 2, (255, 220, 120), -1, cv2.LINE_AA)

    # --- Manos ---
    for lado in LADOS:
        if manos[lado].rasgos is not None:
            dibujar_mano(frame, manos[lado], ui.vistas[lado], canales[lado].tray)

    # --- Medidor de cejas (arriba a la izquierda), resumen de manos (centro) y estado facial (derecha) ---
    medidor_cejas(frame, 12, 12, ui)
    resumen = resumen_manos(ui.vistas)
    rw, rh = medir_texto(resumen, 0.5, 1)
    rx = 200 + max(0, (ancho - 420 - rw) // 2)
    rect_redondeado(frame, rx - 12, 12, rx + rw + 12, 40, 14, (35, 32, 30))
    poner_texto(frame, resumen, (rx, 26 + rh // 2), 0.5, TEXTO, 1)
    pastilla_modo(frame, ancho - 12, 12, ui.modo_facial, ui.progreso_pregunta)

    # --- Diagnóstico (tecla D): rasgos 3D de cada mano y del rostro ---
    if diagnostico:
        lineas = []
        for lado in LADOS:
            r = manos[lado].rasgos
            if r is None:
                continue
            lineas.append(f"{lado}: " + " ".join(f"{e}{x:.2f}" for e, x in zip(r.estados, r.extension)))
            lineas.append("  flex " + " ".join(f"{f:.0f}" for f in r.flexion) + f"  orient={r.orient:.0f}"
                          f" dirInd={r.dir_indice:.0f}")
            lineas.append(f"  pulgar lat={r.pulgar_lat:+.2f} alt={r.pulgar_alt:+.2f} dTI={r.d_ti:.2f}"
                          f" normalZ={r.normal[2]:+.2f}")
            lineas.append(f"  cruce={r.cruce:+.2f} angIM={r.ang_im:.0f} prof={r.prof_medio:+.2f}"
                          f" vel={canales[lado].tray.vel:.1f}")
        if analizador.pose_actual is not None:
            lineas.append(f"Yaw {analizador.negacion.desviacion:+.1f} (+-{NEG_AMPLITUD:.0f})  Pitch "
                          f"{analizador.afirmacion.desviacion:+.1f} (+-{AFIRM_AMPLITUD:.0f})  "
                          f"{analizador.pose_actual.distancia:.0f} cm  ojos x{analizador.ratio_ojos:.2f}")
        for i, linea in enumerate(lineas):
            poner_texto(frame, linea, (10, 98 + i * 18), 0.44, (0, 255, 255), 1)

    # --- Barra inferior de progreso (mano principal) ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, alto - 52), (ancho, alto), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    quien = f" ({ui.lado})" if ui.lado else ""
    if ui.estado_sena == "registrada":
        estado = f"REGISTRADO{quien}"
    elif ui.estado_sena == "sostener":
        estado = f"Sosten la sena {ui.sena}{quien} quieta... {int(ui.progreso * 100)}%"
    elif ui.estado_sena == "mover":
        estado = "Mano en movimiento: haz el trazo o sostenla quieta"
    elif ui.estado_sena == "pose":
        estado = f"Mueve la mano para escribir {ui.sena}"
    else:
        estado = "Muestra una sena a la camara (cualquier mano)"
    poner_texto(frame, estado, (16, alto - 30), 0.5, TEXTO if ui.estado_sena != "ninguna" else TEXTO_SUAVE, 1)
    poner_texto(frame, f"{ui.fps:4.1f} FPS", (ancho - 86, alto - 30), 0.45, TEXTO_SUAVE, 1)
    barra_progreso(frame, 16, alto - 20, ancho - 16, alto - 8, ui.progreso, EXITO)


Y_TOP, Y_FILA = 60, 250            # tarjetas "seña actual" y "cómo se hace"
Y_GUIA0, Y_GUIA1 = 258, 584        # tarjeta de la guía del alfabeto
_panel_base = None                 # fondo del panel (lo que no cambia), se dibuja una sola vez
_cache_lineas = {}                 # descripciones ya partidas en líneas
_cache_texto = ("", "")            # (texto completo, parte visible en la caja)


def _dibujar_entrada_guia(panel, i, activa):
    """Una fila de la guía del alfabeto, con o sin resaltado."""
    clave_g, desc, mov = GUIA[i]
    col, fila = divmod(i, (len(GUIA) + 1) // 2)
    x = 22 + col * 304
    y = Y_GUIA0 + 50 + fila * 19
    if activa:
        rect_redondeado(panel, x - 4, y - 15, x + 296, y + 5, 4, (70, 60, 40))
    color_clave = EXITO if activa else (AMBAR if mov else ACENTO)
    etiqueta = clave_g if len(clave_g) == 1 or clave_g == "?" else clave_g[:3] + "."
    poner_texto(panel, etiqueta, (x + 4, y), 0.45, color_clave, 1)
    poner_texto(panel, desc, (x + 46, y), 0.4, TEXTO if activa else (200, 200, 200), 1)


def _panel_estatico():
    """Todo lo que no cambia entre fotogramas: encabezado, tarjetas, etiquetas, guía y pie."""
    panel = np.full((ALTO_PANEL, ANCHO_PANEL, 3), FONDO, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (ANCHO_PANEL, 50), (40, 34, 30), -1)
    poner_texto(panel, "TRADUCTOR LSM", (16, 33), 0.8, ACENTO, 1, FUENTE_TITULO)
    tarjeta(panel, 14, Y_TOP, 300, Y_FILA)
    poner_texto(panel, "SENA ACTUAL", (28, Y_TOP + 26), 0.5, TEXTO_SUAVE, 1)
    tarjeta(panel, 308, Y_TOP, 626, Y_FILA)
    poner_texto(panel, "COMO SE HACE", (322, Y_TOP + 26), 0.5, TEXTO_SUAVE, 1)
    tarjeta(panel, 14, Y_GUIA0, 626, Y_GUIA1)
    poner_texto(panel, "GUIA RAPIDA  (naranja = lleva movimiento; sirve cualquier mano)", (28, Y_GUIA0 + 24),
                0.5, TEXTO_SUAVE, 1)
    for i in range(len(GUIA)):
        _dibujar_entrada_guia(panel, i, False)
    tarjeta(panel, 14, 590, 626, 620)
    poner_texto(panel, "ROSTRO:  cejas arriba 1.5 s = ?   |   negar = borrar   |   asentir = confirmar y leer",
                (24, 610), 0.42, AMBAR, 1)
    tarjeta(panel, 14, 628, 626, 704, (18, 16, 14), ACENTO)
    poner_texto(panel, "TEXTO", (28, 650), 0.45, TEXTO_SUAVE, 1)
    poner_texto(panel, "Espacio | Backspace | C: limpiar | D: diagnostico | N: rostro neutro | ESC", (14, 726),
                0.38, TEXTO_SUAVE, 1)
    return panel


def _lineas(texto, ancho, escala):
    clave = (texto, ancho, escala)
    if clave not in _cache_lineas:
        _cache_lineas[clave] = ajustar_lineas(texto, ancho, escala)
    return _cache_lineas[clave]


def _texto_visible(texto):
    """La parte final del texto que cabe en la caja (los espacios se ven como espacios)."""
    global _cache_texto
    if _cache_texto[0] != texto:
        visible = limpiar_texto(texto)
        while visible and medir_texto(visible, 1.3, 2, FUENTE_TITULO)[0] > ANCHO_PANEL - 70:
            visible = visible[1:]  # Si no cabe, se muestra el final
        _cache_texto = (texto, visible)
    return _cache_texto[1]


def dibujar_panel(ui):
    """Panel traductor generado con NumPy: seña actual (y de qué mano), cómo se hace, guía y texto.

    El fondo (guía, tarjetas, etiquetas) se dibuja una sola vez y se copia en cada fotograma; solo se pinta
    lo que cambia. Así el panel cuesta ≈ 3 ms en lugar de ≈ 9 ms."""
    global _panel_base
    if _panel_base is None:
        _panel_base = _panel_estatico()
    panel = _panel_base.copy()
    pastilla_modo(panel, ANCHO_PANEL - 14, 10, ui.modo_facial, ui.progreso_pregunta)
    x_resumen = 16 + medir_texto("TRADUCTOR LSM", 0.8, 1, FUENTE_TITULO)[0] + 24
    poner_texto(panel, resumen_manos(ui.vistas), (x_resumen, 31), 0.45, TEXTO_SUAVE, 1)

    # --- Seña actual (izquierda) ---
    cx, cy = 157, 140
    if ui.lado:
        poner_texto(panel, f"MANO {ui.lado.upper()}", (200, Y_TOP + 26), 0.45, COLOR_LADO[ui.lado], 1)
    if ui.sena:
        color = {"registrada": EXITO, "pose": AMBAR, "mover": TEXTO_SUAVE}.get(ui.estado_sena, TEXTO)
        largo = len(ui.sena)
        grosor = 7 if largo == 1 else 3
        # La letra gigante debe caber entre el título de la tarjeta y la barra; la Ñ lleva tilde y necesita más aire
        escala_max = (3.3 if ui.sena == "Ñ" else 4.4) if largo == 1 else 1.6
        escala = escala_para_ancho(ui.sena, 240, escala_max, grosor, FUENTE_TITULO)
        texto_centrado(panel, ui.sena, cx, cy, escala, color, grosor, FUENTE_TITULO)
    else:
        texto_centrado(panel, "-", cx, cy, 4.0, BORDE, 6, FUENTE_TITULO)
    barra_progreso(panel, 30, Y_FILA - 36, 284, Y_FILA - 24, ui.progreso, EXITO)
    porcentaje = {"registrada": "LISTO", "pose": "MUEVE LA MANO", "mover": "EN MOVIMIENTO"}.get(
        ui.estado_sena, f"{int(ui.progreso * 100)}%")
    texto_centrado(panel, porcentaje, cx, Y_FILA - 48, 0.45, TEXTO_SUAVE, 1)

    # --- Cómo se hace (derecha) ---
    clave = ui.sena if ui.sena in DESCRIPCIONES else None
    if clave:
        desc, mov = DESCRIPCIONES[clave]
        poner_texto(panel, clave, (322, Y_TOP + 62), 0.9, ACENTO, 2, FUENTE_TITULO)
        y = Y_TOP + 90
        for linea in _lineas(desc, 290, 0.55):
            poner_texto(panel, linea, (322, y), 0.55, TEXTO, 1)
            y += 22
        if mov:
            for linea in _lineas("Movimiento: " + mov, 290, 0.5):
                poner_texto(panel, linea, (322, y + 6), 0.5, AMBAR, 1)
                y += 20
    else:
        for i, linea in enumerate(_lineas("Muestra una sena a la camara con cualquier mano. La guia de abajo "
                                          "explica cada letra.", 290, 0.5)):
            poner_texto(panel, linea, (322, Y_TOP + 60 + i * 22), 0.5, TEXTO_SUAVE, 1)

    # --- Fila resaltada de la guía ---
    for i, (clave_g, _, _) in enumerate(GUIA):
        if clave_g == ui.sena:
            _dibujar_entrada_guia(panel, i, True)
            break

    # --- Caja de texto acumulado (los espacios se ven como espacios) ---
    cursor = "|" if int(ui.ahora * 2) % 2 == 0 else " "
    poner_texto(panel, _texto_visible(ui.texto) + cursor, (28, 692), 1.3, TEXTO, 2, FUENTE_TITULO)

    # --- Estado de la voz (pie) ---
    color_voz = EXITO if ui.voz == "lista" else (AMBAR if ui.voz == "iniciando" else TEXTO_SUAVE)
    ancho_voz = medir_texto(f"Voz: {ui.voz}", 0.4)[0]
    poner_texto(panel, f"Voz: {ui.voz}", (ANCHO_PANEL - ancho_voz - 14, 726), 0.4, color_voz, 1)
    return panel


def colocar_ventanas(ancho_camara):
    """Cámara a la izquierda y panel a la derecha, sin encimarse."""
    cv2.namedWindow(VENTANA_CAMARA, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(VENTANA_PANEL, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(VENTANA_CAMARA, MARGEN_VENTANAS, MARGEN_VENTANAS)
    cv2.moveWindow(VENTANA_PANEL, MARGEN_VENTANAS * 2 + ancho_camara, MARGEN_VENTANAS)


def ventana_cerrada(nombre):
    """True si el usuario cerró la ventana con la X."""
    try:
        return cv2.getWindowProperty(nombre, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


# =============================================================================
# 11. PROGRAMA PRINCIPAL
# =============================================================================
def detectar_manos(hands, rgb, ancho, alto):
    """Hasta dos manos de MediaPipe como lista de DeteccionMano ([] si no hay o hay error)."""
    try:
        return leer_manos(hands.process(rgb), ancho, alto)
    except Exception as e:  # Un frame defectuoso no debe tirar el programa
        avisar_una_vez(f"Error procesando las manos ({type(e).__name__}: {e}); se omite el frame.")
        return []


def detectar_rostro(face_mesh, rgb, ancho, alto):
    """Arreglo 468x3 en píxeles con los puntos faciales que se usan (el resto en cero), o None."""
    try:
        resultado = face_mesh.process(rgb)
        if not resultado.multi_face_landmarks:
            return None
        lm = resultado.multi_face_landmarks[0].landmark
        puntos = np.zeros((468, 3), dtype=float)
        for i in INDICES_ROSTRO:
            puntos[i] = (lm[i].x * ancho, lm[i].y * alto, lm[i].z * ancho)
        return puntos
    except Exception as e:
        avisar_una_vez(f"Error procesando el rostro ({type(e).__name__}: {e}); se omite el frame.")
        return None


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara. Verifica que esté conectada y disponible.")

    traductor = Traductor()
    escritor, analizador = traductor.escritor, traductor.analizador
    voz = VozAsincrona(VOZ_HABILITADA)
    politica = PoliticaVoz(voz)
    barra = ProgresoSuave()
    texto_enviado = escritor.texto
    diagnostico = False
    ventanas_listas = False
    rostro = None
    principal = None
    fps, t_frame_anterior = 0.0, time.monotonic()
    n_frame = 0

    print("Traductor LSM iniciado. Mantén la cara neutra un momento para calibrar. ESC para salir.")

    with mp_hands.Hands(static_image_mode=False, max_num_hands=MAX_MANOS,
                        min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands, \
         mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    print("[AVISO] No se pudo leer un frame de la cámara.")
                    break

                frame = cv2.flip(frame, 1)  # Espejo (MediaPipe espera la imagen espejada para la lateralidad)
                alto, ancho = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                ahora = time.monotonic()
                n_frame += 1

                # --- Percepción (cada canal tolera fallos por separado) ---
                detecciones = detectar_manos(hands, rgb, ancho, alto)
                procesar_rostro = n_frame % PROCESAR_ROSTRO_CADA == 0
                if procesar_rostro:
                    rostro = detectar_rostro(face_mesh, rgb, ancho, alto)

                # --- Manos -> señas -> texto, y rostro -> gestos (un canal independiente por mano) ---
                manos, _ = traductor.paso(detecciones, rostro, ahora, (ancho, alto), procesar_rostro)

                # --- Voz y hardware: reaccionan a lo que ocurrió en este frame ---
                for ev in escritor.sacar_eventos():
                    politica.evento(ev, escritor.texto, ahora)
                politica.tick(escritor.texto, ahora)
                if escritor.texto != texto_enviado:
                    enviar_a_hardware(escritor.texto)
                    texto_enviado = escritor.texto

                # --- Qué se muestra: cada mano con su etiqueta y, en grande, la mano principal ---
                vistas = {}
                for lado in LADOS:
                    v = vista_mano(manos[lado], escritor.manos[lado], ahora)
                    v.progreso = traductor.canales[lado].barra.actualizar(v.progreso)
                    vistas[lado] = v
                principal = mano_principal(vistas, principal)
                vp = vistas[principal] if principal else VistaMano(None, "ninguna", 0.0)

                # --- Interfaz ---
                dt = ahora - t_frame_anterior
                t_frame_anterior = ahora
                if dt > 0:
                    fps = 1.0 / dt if fps == 0 else fps * 0.9 + (1.0 / dt) * 0.1
                ui = EstadoUI(vp.sena, vp.estado, barra.actualizar(vp.progreso), principal, vistas, escritor.texto,
                              ahora, analizador.modo, analizador.progreso_pregunta, analizador.ratio_cejas,
                              analizador.progreso_calibracion, analizador.pose_actual, fps, voz.estado)
                dibujar_camara(frame, ui, manos, traductor.canales, rostro, analizador, diagnostico)
                panel = dibujar_panel(ui)

                if not ventanas_listas:
                    colocar_ventanas(ancho)
                    ventanas_listas = True
                cv2.imshow(VENTANA_CAMARA, frame)
                cv2.imshow(VENTANA_PANEL, panel)

                # --- Teclado ---
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == 27:                          # ESC
                    break
                elif tecla == 32:                        # Espacio
                    escritor.ejecutar("ESPACIO", ahora)
                elif tecla == 8:                         # Backspace
                    escritor.ejecutar("BORRAR", ahora)
                elif tecla in (ord("c"), ord("C")):
                    escritor.ejecutar("LIMPIAR", ahora)
                elif tecla in (ord("d"), ord("D")):
                    diagnostico = not diagnostico
                elif tecla in (ord("n"), ord("N"), ord("r"), ord("R")):   # rostro neutro (R se conserva)
                    analizador.recalibrar()

                if ventana_cerrada(VENTANA_CAMARA) or ventana_cerrada(VENTANA_PANEL):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            voz.cerrar()

    print(f"Texto final: {limpiar_texto(escritor.texto)!r}")
    print("Programa cerrado correctamente.")


if __name__ == "__main__":
    main()
