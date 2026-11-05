"""Canal no manual: pose 3D de la cabeza con cv2.solvePnP, cejas compensadas por la pose (pregunta),
cara tapada, autocalibración de la cara neutra y gestos de cabeza (negar / asentir)."""
import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from classifier_hands import dist, punto_medio, restar, unitario
from voice_engine import avisar_una_vez

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


PROCESAR_ROSTRO_CADA = 1  # 1 = cada frame; 2 = frames alternos (si la PC va lenta)


# =============================================================================
# CANAL NO MANUAL — pose 3D de la cabeza (solvePnP) y cejas
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
