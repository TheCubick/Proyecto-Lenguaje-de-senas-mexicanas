"""Canal manual: dos manos (lateralidad y estado por mano), filtro One-Euro, rasgos 3D, clasificación del
alfabeto LSM estático y reconocimiento de trazos (J, K, Ñ, Q, X, Z, BORRAR, ?)."""
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from voice_engine import avisar_una_vez

# --- Estabilidad de la seña ---


FRAMES_ESTABLES = 4            # Frames seguidos con la misma seña antes de aceptarla (anti-parpadeo)
V_QUIETO = 1.2                 # Velocidad de la palma (escalas/s) por encima de la cual la mano "se mueve".
                               # Sostener una seña produce 0.5-1.1 de temblor natural; un trazo, más de 2.


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


# =============================================================================
# GEOMETRÍA 2D / 3D Y FILTRO ONE-EURO
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
# CANAL MANUAL — rasgos 3D de la mano
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
# CLASIFICACIÓN DE SEÑAS
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
# DOS MANOS — lateralidad, seguimiento y estado independiente por mano
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


# --- Estabilización temporal ---------------------------------------------------
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


def detectar_manos(hands, rgb, ancho, alto):
    """Hasta dos manos de MediaPipe como lista de DeteccionMano ([] si no hay o hay error)."""
    try:
        return leer_manos(hands.process(rgb), ancho, alto)
    except Exception as e:  # Un frame defectuoso no debe tirar el programa
        avisar_una_vez(f"Error procesando las manos ({type(e).__name__}: {e}); se omite el frame.")
        return []
