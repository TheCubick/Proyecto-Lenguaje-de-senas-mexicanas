"""Gestos dinámicos (palabras y frases del dataset) en vivo, mientras no haya una LSTM entrenada.

Segmenta igual que video_extractor: el gesto empieza cuando una mano sube (muñeca más alta con y < REPOSO_Y) y
termina cuando las manos quedan abajo o fuera de cuadro REPOSO_S segundos. El búfer guarda la ejecución COMPLETA
(hasta GESTO_MAX_S) y al cerrarse se limpia y se remuestrea a 30 fotogramas con las MISMAS funciones del extractor:
una ventana fija de 30 fotogramas en vivo (1-1.5 s) vería solo un pedazo de una seña que en el dataset dura 1.5-8.4 s.
Se compara por DTW contra cada muestra de dataset/ (vecino más cercano). Los tiempos van en segundos (nada depende
de los FPS).

Se acepta una frase (y traductor.py la escribe completa) solo si su distancia es <= DIST_MAX y se separa de la
segunda palabra más cercana al menos SEPARACION_MIN. Medido (2026-11-05): las frases del dataset contra las demás
muestras quedan a 0.12-0.98 y el deletreo de los videos Alfabeto/Abecedario (33 tramos) a 1.39-2.72 con separación
<= 0.13; con estos umbrales se aceptan 44/46 frases (todas correctas) y 0/33 tramos de deletreo. Mientras las
muñecas se mueven rápido (frases: >= 2.75 tamaños de mano/s), las letras estáticas se pausan (pausa_letras).

El DTW (vectorizado contra todas las plantillas) corre solo al cerrarse un gesto y en un hilo aparte: el bucle de la
cámara no lo espera. Muestras propias: en Práctica > Frases, la tecla G guarda la última ejecución evaluada como
dataset/<frase>/pablo_seq_XX.npy y entra de inmediato a las plantillas (referencias con tus manos y tu estilo).

Descriptor por fotograma y mano: la FORMA (puntos de mundo relativos a la muñeca, divididos entre el tamaño de la
mano) y la TRAYECTORIA de la muñeca en la imagen (relativa a su posición media en el gesto, en tamaños de mano: no
importa dónde esté parado ni qué tan lejos).
"""
import csv
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from video_extractor import (EJECUCION_MIN, FPS_MAX, LADOS, MIN_FRAMES, REPOSO_MIN, REPOSO_Y, SALIDA, VENTANA,
                             a_largo_fijo, cargar_dataset, columnas_mano, limpiar, vector_fotograma)

REPOSO_S = REPOSO_MIN / FPS_MAX        # 0.4 s de manos abajo cierran el gesto (los mismos umbrales del dataset)
GESTO_MIN_S = EJECUCION_MIN / FPS_MAX  # 1 s: menos es subir o bajar las manos, no una seña
GESTO_MAX_S = 10.0                     # el más largo del dataset dura 8.4 s (versión lenta); después se evalúa igual
PESO_TRAYECTORIA = 0.5                 # trayectoria (tamaños de mano) frente a forma (fracción del tamaño de mano)
CASTIGO_MANO = 1.0                     # una mano presente en un gesto y ausente en el otro
BANDA = 8                              # DTW: desfase máximo en fotogramas (las dos van del inicio al fin de la seña)
MARGEN = 1.10                          # práctica: el objetivo es el más cercano o está a <= 10 % del más cercano
DIST_MAX = 2.2                         # TEMPORAL para calibrar en vivo (2026-11-05): con 1.2 las frases del usuario no
                               # pasaban. Medido en videos: frases <= 0.98, deletreo 1.39-2.72 (ahí lo detiene la
                               # separación, que llegó a 0.19). Ajustar con gestos_vivo.csv
SEPARACION_MIN = 0.2                   # aceptar: 1 - d1/d2; deletreo <= 0.13. Rechaza empates (sordo / eres sordo)
SEPARACION_MIN_LARGA = 0.08            # frases de 3+ palabras (pedido del usuario). Medido: deletreo sigue 0/33; en
                               # gestos_vivo.csv solo 1/49 pasaría y era errónea (TENGO UN DUDA haciendo MI NOMBRE ES)
V_PAUSA = 2.5                          # muñeca, tamaños de mano/s: más rápido es un gesto y se pausan las letras
PAUSA_TRAS_S = 0.8                     # las letras siguen en pausa este tiempo después del último movimiento rápido
V_LENTO = 1.0                          # muñecas más lentas que esto (tamaños de mano/s) = pausa dentro del gesto
PAUSA_CIERRE_S = 0.5                   # pausa así de larga tras un movimiento rápido cierra el gesto sin bajar manos
CUADROS_MAX = int(GESTO_MAX_S * FPS_MAX)
REGISTRO = SALIDA.parent / "gestos_vivo.csv"   # cada evaluación en vivo, para calibrar los umbrales
PREFIJO_USUARIO = "pablo"              # muestras grabadas en vivo (tecla G): dataset/<palabra>/pablo_seq_XX.npy


def descriptor(m):
    """Muestra 30 x F -> (presencia (T, 2), forma (T, 2, 63), trayectoria (T, 2, 2))."""
    T = len(m)
    presencia = m[:, :len(LADOS)] > 0
    forma, trayectoria = np.zeros((T, len(LADOS), 63)), np.zeros((T, len(LADOS), 2))
    img = [m[:, columnas_mano(k)][:, :63].reshape(T, 21, 3) for k in range(len(LADOS))]
    mundo = [m[:, columnas_mano(k)][:, 63:].reshape(T, 21, 3) for k in range(len(LADOS))]
    tam_img = [np.linalg.norm(img[k][presencia[:, k], 9, :2] - img[k][presencia[:, k], 0, :2], axis=1)
               for k in range(len(LADOS))]
    tam_img = np.median(np.concatenate(tam_img)) if any(len(t) for t in tam_img) else 1.0
    munecas = np.concatenate([img[k][presencia[:, k], 0, :2] for k in range(len(LADOS))])
    centro = munecas.mean(axis=0) if len(munecas) else np.zeros(2)
    for k in range(len(LADOS)):
        p = presencia[:, k]
        if not p.any():
            continue
        relativa = mundo[k][p] - mundo[k][p][:, :1]
        tam = np.median(np.linalg.norm(relativa[:, 9], axis=1)) or 1.0
        forma[p, k] = (relativa / tam).reshape(-1, 63)
        trayectoria[p, k] = (img[k][p, 0, :2] - centro) / max(tam_img, 1e-6)
    return presencia, forma, trayectoria


def apilar(descriptores):
    """[descriptor] -> (presencia (N, T, 2), forma (N, T, 2, 63), trayectoria (N, T, 2, 2)): el banco de plantillas."""
    return tuple(np.stack(x) for x in zip(*descriptores))


def distancias(a, banco):
    """DTW de un descriptor contra TODAS las plantillas del banco a la vez (costo medio por paso del camino) -> (N,).
    Vectorizado sobre las plantillas: medido 70 ms -> ~5 ms con 57 (el bucle es por celda, no por plantilla)."""
    pa, fa, ta = a
    pb, fb, tb = banco
    n, m = len(pa), pb.shape[1]
    D = np.zeros((n, m, len(pb)))                                      # costo de cada par de fotogramas
    for k in range(len(LADOS)):
        ambos = pa[:, None, None, k] & pb[:, :, k].T[None]
        una = pa[:, None, None, k] ^ pb[:, :, k].T[None]
        cuadrados = ((fa[:, k] ** 2).sum(-1)[:, None, None] + (fb[:, :, k] ** 2).sum(-1).T[None]
                     - 2 * np.einsum("if,njf->ijn", fa[:, k], fb[:, :, k]))
        df = np.sqrt(np.maximum(cuadrados, 0.0)) / np.sqrt(21)
        dt = np.linalg.norm(ta[:, None, None, k] - tb[:, :, k].transpose(1, 0, 2)[None], axis=-1)
        D += np.where(ambos, df + PESO_TRAYECTORIA * dt, 0.0) + np.where(una, CASTIGO_MANO, 0.0)
    C = np.full((n + 1, m + 1, len(pb)), np.inf)
    C[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - BANDA), min(m, i + BANDA) + 1):
            C[i, j] = D[i - 1, j - 1] + np.minimum(np.minimum(C[i - 1, j], C[i, j - 1]), C[i - 1, j - 1])
    return C[n, m] / (n + m)


def etiquetas_dataset(carpeta=SALIDA):
    """Palabras con muestras en dataset/ (para la interfaz), sin cargar los archivos."""
    if not carpeta.is_dir():
        return []
    return sorted(p.name for p in carpeta.iterdir() if p.is_dir() and any(p.glob("*seq_*.npy")))


def texto_frase(etiqueta):
    """'mi_nombre_es' -> 'MI NOMBRE ES' (como se escribe en el texto)."""
    return etiqueta.replace("_", " ").upper()


class ReconocedorGestos:
    def __init__(self, carpeta=SALIDA, asincrono=True):
        """asincrono: el DTW corre en un hilo aparte y el resultado se recoge en un fotograma siguiente (en vivo);
        False lo evalúa en el mismo fotograma (pruebas reproducibles con grabaciones)."""
        self.carpeta = carpeta
        X, y = cargar_dataset(carpeta)
        self.plantillas = [(str(e), descriptor(m)) for m, e in zip(X, y)]
        self._apilar()
        self.etiquetas = sorted(set(y.tolist()))
        self._ejecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dtw") if asincrono else None
        self._pendiente = None             # (resultado base, futuro del DTW) de la ejecución que se está evaluando
        self.ultima = None                 # (palabra, muestra 30 x F) de la última ejecución evaluada: tecla G
        self.aviso = None                  # mensaje de guardar_muestra para el navegador (una vez)
        self.propias = 0                   # muestras del usuario de la palabra que se practica
        self.objetivo = None               # palabra que se practica (None = solo traducir)
        self.cuadros = deque(maxlen=CUADROS_MAX)   # [(instante, vector F, manos arriba)] del gesto en curso
        self.t_abajo = None
        self.resultado = None              # se entrega una vez (sacar_resultado)
        self.munecas = {}                  # k -> (instante, x, y) de la muñeca en el fotograma anterior
        self.rapido_hasta = -1.0           # las letras estáticas siguen en pausa hasta este instante
        self.primer_movimiento = False     # aviso (una vez por gesto) del primer movimiento rápido
        self.hubo_movimiento = False       # el gesto en curso ya tuvo un movimiento rápido
        self.t_lento = None                # desde cuándo las muñecas van lento (pausa dentro del gesto)

    @property
    def grabando(self):
        return bool(self.cuadros)

    def pausa_letras(self, ahora):
        """Letras estáticas en pausa: justo después de un movimiento rápido y, mientras dure, todo gesto que ya se
        movió (sus pausas intermedias no deben escribir letras: así se colaban "Q I" en MI NOMBRE ES)."""
        return ahora < self.rapido_hasta or (self.grabando and self.hubo_movimiento)

    def _apilar(self):
        self.nombres = np.array([e for e, _ in self.plantillas])
        self.banco = apilar([d for _, d in self.plantillas]) if self.plantillas else None

    def poner_objetivo(self, objetivo):
        self.objetivo = objetivo if objetivo in self.etiquetas else None
        self.cuadros.clear()
        self.t_abajo, self.resultado, self.t_lento, self.hubo_movimiento = None, None, None, False
        self._pendiente, self.ultima = None, None
        self.propias = len(self._propias(self.objetivo)) if self.objetivo else 0

    def _propias(self, palabra):
        return sorted((self.carpeta / palabra).glob(f"{PREFIJO_USUARIO}_seq_*.npy"))

    def guardar_muestra(self):
        """Tecla G / botón: guarda la última ejecución evaluada (ya remuestreada a 30 x F, igual que el dataset) como
        dataset/<palabra>/<usuario>_seq_XX.npy y la suma al instante a las plantillas del DTW. La palabra es la que
        se practicaba cuando se hizo el movimiento."""
        if self.ultima is None or self.ultima[0] is None:
            self.aviso = ("Elige una frase en Práctica > Frases, hazla completa y luego guarda" if self.objetivo is None
                          else "Primero haz la frase completa (hasta que se evalúe) y luego guarda")
            return
        palabra, muestra = self.ultima
        n = 1
        while (self.carpeta / palabra / f"{PREFIJO_USUARIO}_seq_{n:02d}.npy").exists():
            n += 1
        ruta = self.carpeta / palabra / f"{PREFIJO_USUARIO}_seq_{n:02d}.npy"
        np.save(ruta, muestra)
        self.plantillas.append((palabra, descriptor(muestra)))
        self._apilar()
        self.ultima = None                 # la misma ejecución no se guarda dos veces
        self.propias = len(self._propias(palabra))
        self.aviso = f"Guardada {ruta.name} en {texto_frase(palabra)}: {self.propias} muestras tuyas"
        print(f"[gesto] muestra guardada: {ruta}", flush=True)

    def _velocidad(self, v, ahora):
        """Velocidad de cada muñeca en tamaños de mano por segundo; si pasa de V_PAUSA, pausa las letras. También
        lleva desde cuándo las muñecas van lento (para cerrar el gesto con una pausa)."""
        maxima = None
        for k in range(len(LADOS)):
            if v[k] == 0:
                self.munecas.pop(k, None)
                continue
            img = v[columnas_mano(k)][:63].reshape(21, 3)
            tam = max(float(np.linalg.norm(img[9, :2] - img[0, :2])), 1e-6)
            previa = self.munecas.get(k)
            self.munecas[k] = (ahora, img[0, 0], img[0, 1])
            if previa is None or not 0 < ahora - previa[0] < 0.3:
                continue
            rapidez = np.hypot(img[0, 0] - previa[1], img[0, 1] - previa[2]) / tam / (ahora - previa[0])
            maxima = rapidez if maxima is None else max(maxima, rapidez)
            if rapidez > V_PAUSA:
                if self.cuadros and ahora >= self.rapido_hasta:
                    self.primer_movimiento = True
                if self.cuadros:
                    self.hubo_movimiento = True
                self.rapido_hasta = ahora + PAUSA_TRAS_S
        if maxima is not None:
            self.t_lento = (self.t_lento or ahora) if maxima < V_LENTO else None

    def tomar_primer_movimiento(self):
        aviso, self.primer_movimiento = self.primer_movimiento, False
        return aviso

    def clasificar(self, muestra):
        """Muestra 30 x F -> [(palabra, distancia)] de la más cercana a la más lejana."""
        nombres, banco = self.nombres, self.banco      # guardar_muestra puede cambiarlos desde el otro hilo
        d = distancias(descriptor(muestra), banco)
        mejor = {e: float(d[nombres == e].min()) for e in np.unique(nombres)}
        return sorted(mejor.items(), key=lambda par: par[1])

    def paso(self, asignadas, ahora, tamano):
        """Un fotograma en vivo ({lado: DeteccionMano} del SeguidorManos)."""
        if not self.plantillas:
            return
        if self._pendiente is not None and self._pendiente[1].done():   # el DTW del hilo terminó
            (base, futuro), self._pendiente = self._pendiente, None
            self._evaluar(base, futuro.result())
        v = vector_fotograma(asignadas, *tamano)
        self._velocidad(v, ahora)
        y = [v[columnas_mano(k).start + 1] for k in range(len(LADOS)) if v[k] > 0]
        arriba = bool(y) and min(y) < REPOSO_Y
        if arriba:
            self.cuadros.append((ahora, v, True))
            self.t_abajo = None
            # Señas encadenadas: tras un movimiento rápido, una pausa de PAUSA_CIERRE_S cierra el gesto sin bajar
            # las manos (el deletreo casi no mueve la muñeca y no llega aquí)
            if self.hubo_movimiento and self.t_lento is not None and ahora - self.t_lento >= PAUSA_CIERRE_S:
                self._cerrar("pausa")
        elif self.cuadros:
            self.cuadros.append((ahora, v, False))
            self.t_abajo = self.t_abajo or ahora
            if ahora - self.t_abajo >= REPOSO_S:
                self._cerrar("manos abajo")
        if self.cuadros and ahora - self.cuadros[0][0] > GESTO_MAX_S:
            self._cerrar("tiempo máximo")

    def _cerrar(self, cierre):
        cuadros = list(self.cuadros)
        ultimo = max((i for i, c in enumerate(cuadros) if c[2]), default=None)   # sin el reposo del final
        self.cuadros.clear()
        self.t_abajo, self.t_lento, self.hubo_movimiento = None, None, False
        if ultimo is None:
            return
        cuadros = cuadros[:ultimo + 1]
        segundos = cuadros[-1][0] - cuadros[0][0]
        base = {"objetivo": self.objetivo, "cierre": cierre, "segundos": round(segundos, 1), "mejor": None,
                "distancia": None, "separacion": None, "aceptado": False, "acierto": None, "escrito": None,
                "distancia_objetivo": None, "motivo": "", "top": []}
        if segundos < GESTO_MIN_S:                                       # manos que solo subieron y bajaron
            return self._reportar({**base, "motivo": f"duró {segundos:.1f} s (mínimo {GESTO_MIN_S:.0f} s)"})
        limpia, _ = limpiar(np.stack([c[1] for c in cuadros]))
        if len(limpia) < MIN_FRAMES:
            return self._reportar({**base, "motivo": f"solo {len(limpia)} fotogramas con manos"})
        muestra = a_largo_fijo(limpia, VENTANA)
        self.ultima = (self.objetivo, muestra)
        if self._ejecutor is None:
            self._evaluar(base, self.clasificar(muestra))
        else:                              # el DTW no frena el bucle de la cámara: se recoge en paso()
            self._pendiente = (base, self._ejecutor.submit(self.clasificar, muestra))

    def _evaluar(self, base, ranking):
        (mejor, d1), d2 = ranking[0], ranking[1][1] if len(ranking) > 1 else np.inf
        separacion = 1 - d1 / d2 if d2 > 0 else 0.0
        sep_min = SEPARACION_MIN_LARGA if mejor.count("_") >= 2 else SEPARACION_MIN
        aceptado = bool(d1 <= DIST_MAX and separacion >= sep_min)
        motivo = ("" if aceptado else f"distancia {d1:.2f} > {DIST_MAX}" if d1 > DIST_MAX
                  else f"separación {separacion:.2f} < {sep_min}")
        acierto = d_objetivo = None
        objetivo = base["objetivo"]
        if objetivo is not None:           # práctica: el candidato más cercano con margen relativo (sin separación)
            d_objetivo = dict(ranking)[objetivo]
            acierto = bool(d_objetivo <= DIST_MAX and d_objetivo <= d1 * MARGEN)
            motivo = ("" if acierto else f"distancia {d_objetivo:.2f} > {DIST_MAX}" if d_objetivo > DIST_MAX
                      else f"más cerca de {mejor} ({d1:.2f})")
        self._reportar({**base, "mejor": mejor, "distancia": round(d1, 3), "separacion": round(separacion, 2),
                        "aceptado": aceptado, "acierto": acierto, "motivo": motivo,
                        "distancia_objetivo": None if d_objetivo is None else round(d_objetivo, 3),
                        "top": [[e, round(d, 3)] for e, d in ranking[:3]]})

    def _reportar(self, r):
        """Cada evaluación: al navegador (una vez), a la terminal y a gestos_vivo.csv para calibrar."""
        self.resultado = r
        if r["mejor"] is None:
            print(f"[gesto] {r['cierre']}, {r['segundos']} s: descartado ({r['motivo']})", flush=True)
        else:
            top = ", ".join(f"{e} {d:.2f}" for e, d in r["top"])
            veredicto = "ACEPTADO" if r["aceptado"] else "rechazado: " + (r["motivo"] if r["objetivo"] is None
                                                                        else f"separación {r['separacion']:.2f}")
            practica = ("" if r["objetivo"] is None else f" | objetivo {r['objetivo']} dist "
                        f"{r['distancia_objetivo']:.2f}: {'ACIERTO' if r['acierto'] else 'no: ' + r['motivo']}")
            print(f"[gesto] {r['cierre']}, {r['segundos']} s: {r['mejor']} dist {r['distancia']:.2f} "
                  f"sep {r['separacion']:.2f} -> {veredicto}{practica} | top: {top}", flush=True)
        nuevo = not REGISTRO.is_file()
        try:
            with REGISTRO.open("a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                if nuevo:
                    w.writerow(["fecha", "cierre", "segundos", "objetivo", "mejor", "distancia", "separacion",
                                "aceptado", "acierto", "motivo", "top3"])
                w.writerow([datetime.now().isoformat(timespec="seconds"), r["cierre"], r["segundos"], r["objetivo"],
                            r["mejor"], r["distancia"], r["separacion"], r["aceptado"], r["acierto"], r["motivo"],
                            " | ".join(f"{e} {d}" for e, d in r["top"])])
        except OSError:
            pass                           # p. ej. el archivo abierto en Excel: el registro no detiene al motor

    def sacar_resultado(self):
        r, self.resultado = self.resultado, None
        return r

    def web(self, ahora):
        """Estado para el navegador: fase del gesto en curso, si las letras están en pausa y, una sola vez, el
        resultado de la evaluación (con "escrito" si traductor.py escribió la frase)."""
        if not self.plantillas:
            return None
        aviso, self.aviso = self.aviso, None
        return {"objetivo": self.objetivo, "fase": "grabando" if self.grabando else "espera",
                "segundos": round(ahora - self.cuadros[0][0], 1) if self.grabando else 0.0,
                "pausa": self.pausa_letras(ahora), "resultado": self.sacar_resultado(),
                "muestra": self.ultima is not None and self.ultima[0] is not None,   # hay qué guardar (tecla G)
                "propias": self.propias, "aviso": aviso}
