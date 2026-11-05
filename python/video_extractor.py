"""Extracción masiva de muestras para la LSTM desde videos de referencia (compilaciones de señas).

Lee los cortes de un archivo de configuración (JSON o CSV), salta a cada fragmento, extrae los landmarks 3D de las
manos con el mismo pipeline que el traductor en vivo (classifier_hands) y guarda cada EJECUCIÓN de la seña como una
secuencia de EXACTAMENTE 30 fotogramas en dataset/<palabra>/seq_XX_N.npy. Al terminar sin fallas todos los cortes
de un video, lo mueve a videos/listo/. Herramienta independiente: no toca el servidor ni la interfaz.

Uso:
  python python/video_extractor.py                         # procesa dataset_config.json (solo cortes nuevos o cambiados)
  python python/video_extractor.py --importar "datos/videos/minutos exactos de videos.txt"   # notas a mano -> config, y procesa
  python python/video_extractor.py --revisar               # valida la configuración y muestra el plan, sin MediaPipe
  python python/video_extractor.py --config cortes.csv     # CSV con encabezado video,palabra,inicio,fin[,espejo]
  python python/video_extractor.py --rehacer               # vuelve a extraer todo
  python python/video_extractor.py --no-mover              # no mover los videos terminados a videos/listo/
  python python/video_extractor.py --listar                # muestras por palabra en dataset/

Cada corte de la configuración:
  {"video": "Frutas.mp4", "palabra": "manzana", "inicio": "00:15", "fin": "00:18"}
  video    nombre del archivo dentro de videos/
  palabra  etiqueta; la carpeta se llama igual, en minúsculas, sin signos y con "_" en lugar de espacios
  inicio   "SS", "MM:SS" o "HH:MM:SS", con décimas opcionales ("00:15.4") o un número de segundos (0 si falta)
  fin      igual que inicio (si falta: hasta el final del video)
  espejo   opcional (true): false si el video ya viene espejado, p. ej. grabado como selfie

Proceso de cada corte:
  1. Salto a `inicio` con cap.set(POS_MSEC) (medido en los videos de videos/: cae en el fotograma exacto) y lectura
     hasta `fin` con la marca de tiempo real de cada fotograma; los videos a 60 FPS se leen a 30.
  2. Espejo + MediaPipe Hands + SeguidorManos, igual que en vivo: Izq/Der son las manos reales de quien seña. Cada
     corte usa un detector nuevo para que el seguimiento no arrastre manos de otro fragmento.
  3. Ejecuciones: muchos videos hacen la seña a velocidad normal, bajan las manos y la repiten lento. El corte se
     parte en los REPOSOS (manos abajo o fuera de cuadro durante REPOSO_MIN fotogramas) y cada ejecución es una
     muestra: seq_XX_1 la normal, seq_XX_2 la lenta, etc. Sin reposos, el corte entero es una sola ejecución.
  4. Limpieza de cada ejecución: se quitan los fotogramas sin manos de los extremos y se rellenan huecos de hasta
     HUECO_MAX fotogramas de una mano interpolando entre sus vecinos (MediaPipe la pierde con el desenfoque de
     movimiento). Ejecuciones con menos de MIN_FRAMES fotogramas con manos se descartan.
  5. Remuestreo a exactamente 30 fotogramas con interpolación lineal en el tiempo: una ejecución de 40 se comprime y
     una de 20 se estira. Toda muestra va del inicio al final de la seña, sin padding (la lenta queda como otra
     muestra de la misma forma).

Notas a mano (--importar): una línea con el video ("1. video.mp4") y debajo una por seña: "sordo: 11 s - 17 s",
"si: 58s - 1:03", "Mi nombre es: 1:08 - 1:21". Reemplaza en la configuración los cortes de esos videos.

Vector por fotograma (F = 254 float32), en este orden:
  [presente_izq, presente_der,
   izq_img (21 x [x/ancho, y/alto, z/ancho]), izq_mundo (21 x [x, y, z] en metros, centrado en la mano),
   der_img (21 x 3),                         der_mundo (21 x 3)]
  Una mano ausente vale 0 en sus 126 columnas. "img" conserva la posición en la imagen (el MOVIMIENTO); "mundo" es
  la forma 3D de la mano. Solo se guardan vectores: ningún píxel del video.

Re-ejecutar es incremental: dataset/indice.json recuerda de qué corte salió cada archivo. Un corte sin cambios no se
vuelve a procesar; si se corrige su tiempo se extrae de nuevo, y los archivos de cortes que ya no están en la
configuración se borran. seq_XX es el XX-ésimo corte de esa palabra en el orden del archivo; _N, su ejecución.
"""
import argparse
import csv
import io
import json
import os
import re
import shutil
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from classifier_hands import LADOS, MAX_MANOS, SeguidorManos, detectar_manos

RAIZ = Path(__file__).resolve().parent.parent   # el código vive en python/; los datos, en datos/
DATOS = RAIZ / "datos"
CONFIG = DATOS / "dataset_config.json"
VIDEOS = DATOS / "videos"
LISTO = "listo"                # subcarpeta de videos/ adonde pasan los videos ya procesados
SALIDA = DATOS / "dataset"
INDICE = "indice.json"
VENTANA = 30                   # Fotogramas por muestra (la LSTM recibe secuencias de este largo)
FPS_MAX = 30.0                 # Videos más rápidos se leen a 30 FPS: más fotogramas no aportan al remuestrear a 30
HUECO_MAX = 6                  # Fotogramas seguidos sin una mano que se rellenan interpolando (0.2 s a 30 FPS)
MIN_FRAMES = 10                # Cortes con menos fotogramas con manos se descartan (estirarlos a 30 inventa datos)
CONFIANZA = 0.5                # min_detection_confidence: más bajo que en vivo (0.7) porque los videos de
                               # internet suelen estar comprimidos; el seguimiento usa el mismo 0.6
DURACION_AVISO = 20.0          # Cortes más largos (s) se avisan: casi siempre un tiempo mal escrito. Medido: rótulos
                               # de "Tutorial Dilo en señas" 3.7-5.5 s; normal + lenta en "25 palabras" 3-15 s
REPOSO_Y = 0.8                 # Manos en reposo: la muñeca más alta queda por debajo de esto (y/alto, 0 = arriba).
                               # Medido en "25 palabras": señando 0.33-0.67; en reposo 0.85-0.96 o fuera de cuadro
REPOSO_MIN = 12                # Fotogramas (0.4 s) de reposo que separan dos ejecuciones. Medido: los reposos entre
                               # normal y lenta duran 0.6-1.2 s; un salto suelto de 0.2 s ("mi nombre es") no corta
EJECUCION_MIN = 30             # Fotogramas (1 s). Medido en "25 palabras": la ejecución real más corta dura 1.53 s;
                               # los restos de la seña anterior al inicio del corte, 0.07-0.7 s
FORMATO = "v1: presente(2) + por lado Izq,Der: img(21x3 x/ancho,y/alto,z/ancho) + mundo(21x3 m); F=254"
F = 2 + len(LADOS) * 2 * 63


@dataclass
class Corte:
    n: int                     # posición en la configuración (1 = primera entrada), para los mensajes
    video: str
    palabra: str               # ya normalizada: es el nombre de la carpeta
    inicio: float
    fin: float
    espejo: bool
    destino: str = ""          # "palabra/seq_XX" (sus archivos: seq_XX_1.npy, ...), relativo a la salida

    def clave(self):
        """Lo que determina los datos. La palabra no: corregir una etiqueta reutiliza la muestra sin MediaPipe."""
        return [self.video, round(self.inicio, 3), round(self.fin, 3), self.espejo]


# --- Configuración -----------------------------------------------------------------------------------------
def a_segundos(valor):
    """'SS', 'MM:SS' o 'HH:MM:SS' (décimas opcionales: '00:15.4' o '00:15,4') o un número -> segundos."""
    if isinstance(valor, bool):
        raise ValueError(valor)
    if isinstance(valor, (int, float)):
        if valor < 0:
            raise ValueError(valor)
        return float(valor)
    texto = str(valor).strip().replace(",", ".")
    if not re.fullmatch(r"(\d+:){0,2}\d+(\.\d+)?", texto):
        raise ValueError(valor)
    segundos = 0.0
    for parte in texto.split(":"):
        segundos = segundos * 60 + float(parte)
    return segundos


def a_bool(valor):
    if isinstance(valor, bool):
        return valor
    texto = str(valor).strip().lower()
    if texto in ("", "true", "si", "sí", "1"):
        return True
    if texto in ("false", "no", "0"):
        return False
    raise ValueError(valor)


def mmss(segundos):
    m, s = divmod(round(segundos, 1), 60)
    return f"{int(m):02d}:{s:04.1f}"


def nombre_carpeta(palabra):
    """'¿Cómo estás?' -> 'cómo_estás': minúsculas, sin signos, '_' en lugar de espacios (válido en Windows).
    voice_engine.limpiar_texto ya convierte '_' en espacio al hablar."""
    texto = unicodedata.normalize("NFC", str(palabra)).strip().lower()
    texto = re.sub(r"[^\w\s-]", "", texto)              # \w incluye acentos y ñ
    return re.sub(r"\s+", "_", texto).strip("_")


def leer_config(ruta):
    """Lista de entradas del JSON (una lista de objetos) o del CSV (encabezado video,palabra,inicio,fin[,espejo])."""
    if not ruta.is_file():
        raise SystemExit(f"[ERROR] No existe el archivo de configuración: {ruta}")
    texto = ruta.read_text(encoding="utf-8-sig")        # -sig: Excel y el Bloc de notas agregan BOM
    if ruta.suffix.lower() == ".csv":
        primera = texto.split("\n", 1)[0]
        separador = ";" if primera.count(";") > primera.count(",") else ","   # Excel en español usa ";"
        return [{(k or "").strip().lower(): (v or "").strip() for k, v in fila.items()}
                for fila in csv.DictReader(io.StringIO(texto), delimiter=separador)]
    try:
        entradas = json.loads(texto)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] {ruta.name}: JSON inválido en la línea {e.lineno}, columna {e.colno} ({e.msg}). "
                         "Revisa comas (la última entrada no lleva coma) y comillas dobles.")
    if not isinstance(entradas, list):
        raise SystemExit(f"[ERROR] {ruta.name} debe ser una lista: [ {{...}}, {{...}} ]")
    return entradas


def escribir_config(ruta, entradas):
    """Un corte por renglón. Antes se respalda el archivo anterior en .json.bak (el usuario lo edita a mano)."""
    if ruta.is_file():
        ruta.with_suffix(".json.bak").write_bytes(ruta.read_bytes())
    filas = [json.dumps(e, ensure_ascii=False) for e in entradas]
    ruta.write_text("[\n" + ",\n".join("  " + f for f in filas) + "\n]\n", encoding="utf-8")


TIEMPO = r"\d+(?::\d{1,2}){0,2}(?:[.,]\d+)?"
LINEA_VIDEO = re.compile(r"^\s*(?:\d+\s*[.)-]\s*)?(?P<video>.+\.(?:mp4|mkv|webm|mov))\s*$", re.IGNORECASE)
LINEA_CORTE = re.compile(rf"^\s*[-*]?\s*(?P<palabra>.+?)\s*:\s*(?P<inicio>{TIEMPO})\s*s?\s*-\s*"
                         rf"(?P<fin>{TIEMPO})\s*s?\s*$", re.IGNORECASE)


def importar_anotaciones(ruta):
    """Notas a mano -> (entradas para la configuración, avisos). Una línea "1. video.mp4" fija el video de las
    siguientes; cada "palabra: inicio - fin" (11 s, 58s, 1:03, 1:08.5) es un corte en segundos."""
    entradas, avisos, video = [], [], None
    for n, linea in enumerate(ruta.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not linea.strip() or linea.strip().lower().rstrip(":") == "video":
            continue
        if m := LINEA_VIDEO.match(linea):
            video = m["video"].strip()
        elif (m := LINEA_CORTE.match(linea)) and video:
            entradas.append({"video": video, "palabra": m["palabra"].strip(), "inicio": a_segundos(m["inicio"]),
                             "fin": a_segundos(m["fin"]), "origen": ruta.name})
        else:
            avisos.append(f"línea {n}: no se entiende \"{linea.strip()}\" (se espera \"palabra: 11 s - 17 s\" "
                          "debajo de una línea con el nombre del video)")
    return entradas, avisos


def ruta_video(carpeta, nombre):
    """El video en videos/ o, si ya se procesó, en videos/listo/."""
    ruta = carpeta / nombre
    return ruta if ruta.is_file() else carpeta / LISTO / nombre


def duracion(ruta):
    cap = cv2.VideoCapture(str(ruta))
    fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps if fps > 0 and n > 0 else None


def validar(entradas, carpeta_videos):
    """Entradas crudas -> (cortes con destino asignado, errores, avisos). Abre cada video una vez (duración)."""
    cortes, errores, avisos, duraciones = [], [], [], {}
    for n, e in enumerate(entradas, 1):
        if not isinstance(e, dict):
            errores.append(f"entrada {n}: debe ser un objeto {{...}}")
            continue
        e = {str(k).strip().lower(): v for k, v in e.items()}
        video, palabra = str(e.get("video") or "").strip(), nombre_carpeta(e.get("palabra") or "")
        if not video or not palabra:
            errores.append(f"entrada {n}: faltan \"video\" o \"palabra\"")
            continue
        donde = f"entrada {n} ({palabra})"
        try:
            inicio = a_segundos(e.get("inicio") or 0)
            fin = a_segundos(e["fin"]) if e.get("fin") not in (None, "") else None
            espejo = a_bool(e.get("espejo", True))
        except ValueError as x:
            errores.append(f"{donde}: valor inválido {x} (tiempos como \"01:05.5\"; espejo true/false)")
            continue
        if video not in duraciones:
            ruta = ruta_video(carpeta_videos, video)
            duraciones[video] = duracion(ruta) if ruta.is_file() else None
        total = duraciones[video]
        if total is None:
            nombres = [p.name for p in [*carpeta_videos.glob("*"), *(carpeta_videos / LISTO).glob("*")]]
            parecidos = get_close_matches(video, nombres, n=1, cutoff=0.6)
            sugerencia = f'; ¿quisiste decir "{parecidos[0]}"?' if parecidos else ""
            errores.append(f"{donde}: no existe o no se puede abrir {carpeta_videos.name}/{video}{sugerencia}")
            continue
        fin = total if fin is None else fin
        if inicio >= total:
            errores.append(f"{donde}: inicio {mmss(inicio)} pasa del final del video ({mmss(total)})")
            continue
        if fin <= inicio:
            errores.append(f"{donde}: fin {mmss(fin)} debe ser mayor que inicio {mmss(inicio)}")
            continue
        if fin > total:
            if fin > total + 0.5:
                avisos.append(f"{donde}: fin {mmss(fin)} pasa del final del video ({mmss(total)}); se corta ahí")
            fin = total
        if fin - inicio > DURACION_AVISO:
            avisos.append(f"{donde}: el corte dura {fin - inicio:.1f} s; ¿trae más de una seña?")
        cortes.append(Corte(n, video, palabra, inicio, fin, espejo))

    unicos, vistos = [], {}
    for c in cortes:
        otro = vistos.setdefault(json.dumps(c.clave()), c)
        if otro is not c and otro.palabra == c.palabra:
            avisos.append(f"entrada {c.n}: repite la entrada {otro.n}; se omite (duplicaría la muestra)")
            continue
        if otro is not c:
            avisos.append(f"entrada {c.n}: mismo corte que la entrada {otro.n} pero con otra palabra ({otro.palabra})")
        unicos.append(c)
    orden = sorted(unicos, key=lambda c: (c.video, c.inicio))
    for a, b in zip(orden, orden[1:]):                   # casi siempre un tiempo mal escrito
        if a.video == b.video and a.palabra != b.palabra and b.inicio < a.fin - 0.05:
            avisos.append(f"entradas {a.n} ({a.palabra}) y {b.n} ({b.palabra}) se traslapan en {a.video}")
    cuenta = {}
    for c in unicos:
        cuenta[c.palabra] = cuenta.get(c.palabra, 0) + 1
        c.destino = f"{c.palabra}/seq_{cuenta[c.palabra]:02d}"
    return unicos, errores, avisos


# --- Extracción y normalización ----------------------------------------------------------------------------
def vector_fotograma(asignadas, ancho, alto):
    """{lado: DeteccionMano} -> vector de F float32 (ver el formato en el docstring del módulo)."""
    v = np.zeros(F, np.float32)
    for k, lado in enumerate(LADOS):
        d = asignadas.get(lado)
        if d is None:
            continue
        v[k] = 1.0
        base = 2 + k * 126
        v[base:base + 63] = (d.img / np.array([ancho, alto, ancho])).ravel()
        v[base + 63:base + 126] = d.mundo.ravel()
    return v


def extraer_corte(cap, inicio, fin, espejo=True, confianza=CONFIANZA):
    """Fotogramas de [inicio, fin] (s) del video ya abierto -> secuencia T x F, a lo más FPS_MAX por segundo."""
    cap.set(cv2.CAP_PROP_POS_MSEC, inicio * 1000)
    seguidor, filas, t_siguiente = SeguidorManos(), [], inicio
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=MAX_MANOS,
                                  min_detection_confidence=confianza, min_tracking_confidence=0.6) as hands:
        while cap.grab():
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000     # marca de tiempo real del fotograma
            if t > fin + 1e-3:
                break
            if t + 1e-3 < t_siguiente:                    # antes de inicio, o sobrante de un video a 60 FPS
                continue
            t_siguiente += 1.0 / FPS_MAX
            ok, frame = cap.retrieve()
            if not ok:
                break
            if espejo:
                frame = cv2.flip(frame, 1)
            alto, ancho = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            asignadas = seguidor.asignar(detectar_manos(hands, rgb, ancho, alto), t)
            filas.append(vector_fotograma(asignadas, ancho, alto))
    return np.array(filas, np.float32).reshape(-1, F)


def columnas_mano(k):
    return slice(2 + k * 126, 2 + (k + 1) * 126)


def limpiar(secuencia, hueco_max=HUECO_MAX):
    """Quita los fotogramas sin manos de los extremos y rellena huecos cortos de cada mano interpolando entre sus
    vecinos. -> (secuencia, (a, b)) con [a, b) el tramo conservado de la original."""
    hay = np.flatnonzero(secuencia[:, :2].max(axis=1) > 0)
    if len(hay) == 0:
        return secuencia[:0], (0, 0)
    a, b = int(hay[0]), int(hay[-1]) + 1
    s = secuencia[a:b].copy()
    for k in range(len(LADOS)):
        cols = columnas_mano(k)
        presentes = np.flatnonzero(s[:, k] > 0)
        for i, j in zip(presentes[:-1], presentes[1:]):
            if 1 < j - i <= hueco_max + 1:
                w = (np.arange(1, j - i) / (j - i))[:, None]
                s[i + 1:j, cols] = (1 - w) * s[i, cols] + w * s[j, cols]
                s[i + 1:j, k] = 1.0
    return s, (a, b)


def ejecuciones(secuencia):
    """[(a, b)] de los tramos con las manos arriba (señando), separados por reposos de al menos REPOSO_MIN
    fotogramas: manos abajo (muñeca más alta con y >= REPOSO_Y) o fuera de cuadro."""
    y = np.full(len(secuencia), np.inf)
    for k in range(len(LADOS)):
        muneca_y = secuencia[:, columnas_mano(k).start + 1]
        y = np.where(secuencia[:, k] > 0, np.minimum(y, muneca_y), y)
    arriba = np.flatnonzero(y < REPOSO_Y)
    if len(arriba) == 0:
        return []
    tramos, a = [], arriba[0]
    for i, j in zip(arriba[:-1], arriba[1:]):
        if j - i - 1 >= REPOSO_MIN:
            tramos.append((int(a), int(i) + 1))
            a = j
    tramos.append((int(a), int(arriba[-1]) + 1))
    return [(a, b) for a, b in tramos if b - a >= EJECUCION_MIN]


def a_largo_fijo(secuencia, largo=VENTANA):
    """T fotogramas -> exactamente `largo` con interpolación lineal en el tiempo (comprime o estira)."""
    return interpolar(secuencia, np.linspace(0, len(secuencia) - 1, largo))


def interpolar(secuencia, pos):
    """Muestrea la secuencia en las posiciones `pos` (índices fraccionarios crecientes). Una mano solo se interpola
    entre dos fotogramas donde está presente; si falta en uno, se copia el fotograma más cercano."""
    T = len(secuencia)
    pos = np.clip(pos, 0, T - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (pos - i0)[:, None]
    salida = ((1 - w) * secuencia[i0] + w * secuencia[i1]).astype(np.float32)
    cercano = np.where(w[:, 0] < 0.5, i0, i1)
    for k in range(len(LADOS)):
        mezcla = (secuencia[i0, k] > 0) != (secuencia[i1, k] > 0)
        cols = np.r_[k, np.arange(F)[columnas_mano(k)]]
        salida[np.ix_(mezcla, cols)] = secuencia[np.ix_(cercano[mezcla], cols)]
    return salida


# --- Dataset en disco --------------------------------------------------------------------------------------
def leer_indice(salida):
    ruta = salida / INDICE
    if ruta.is_file():
        try:
            return json.loads(ruta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[AVISO] {ruta} dañado: se extrae todo de nuevo")
    return {"parametros": {}, "muestras": {}}


def escribir_indice(salida, parametros, muestras):
    salida.mkdir(parents=True, exist_ok=True)
    temporal = salida / (INDICE + ".tmp")
    temporal.write_text(json.dumps({"parametros": parametros, "muestras": dict(sorted(muestras.items()))},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temporal, salida / INDICE)               # nunca queda un índice a medias


def planear(cortes, indice, parametros, rehacer, salida):
    """-> (al_dia, reusar {destino: origen}, pendientes). Con los mismos parámetros, un corte cuya clave ya está en
    disco no vuelve a pasar por MediaPipe (aunque cambie su número o su palabra)."""
    previas = {} if rehacer or indice.get("parametros") != parametros else indice.get("muestras", {})
    previas = {d: m for d, m in previas.items()
               if m.get("archivos") and all((salida / r).is_file() for r in m["archivos"])}
    por_clave = {json.dumps(m["clave"]): r for r, m in previas.items()}
    al_dia, reusar, pendientes = [], {}, []
    for c in cortes:
        if previas.get(c.destino, {}).get("clave") == c.clave():
            al_dia.append(c)
        elif json.dumps(c.clave()) in por_clave:
            reusar[c.destino] = por_clave[json.dumps(c.clave())]
        else:
            pendientes.append(c)
    return al_dia, reusar, pendientes


def guardar_muestras(salida, destino, muestras):
    """'palabra/seq_03' + [m1, m2] -> palabra/seq_03_1.npy, palabra/seq_03_2.npy. Devuelve las rutas relativas."""
    (salida / destino).parent.mkdir(parents=True, exist_ok=True)
    archivos = [f"{destino}_{k}.npy" for k in range(1, len(muestras) + 1)]
    for ruta, muestra in zip(archivos, muestras):
        np.save(salida / ruta, muestra)
    return archivos


def procesar(cortes, args, parametros):
    salida = args.salida
    indice = leer_indice(salida)
    viejas = indice.get("muestras", {})
    al_dia, reusar, pendientes = planear(cortes, indice, parametros, args.rehacer, salida)
    nuevas = {c.destino: viejas[c.destino] for c in al_dia}
    copias = {d: (viejas[o], [np.load(salida / r) for r in viejas[o]["archivos"]])   # en memoria antes de escribir
              for d, o in reusar.items()}
    for destino, (meta, muestras) in copias.items():
        nuevas[destino] = {**meta, "archivos": guardar_muestras(salida, destino, muestras)}
    print(f"{len(cortes)} cortes: {len(al_dia)} al día, {len(reusar)} reutilizados, {len(pendientes)} por extraer")

    grupos = {}
    for c in sorted(pendientes, key=lambda c: (c.video, c.inicio)):   # cada video se abre una vez
        grupos.setdefault(c.video, []).append(c)
    descartados, hecho, t_total, i = [], 0, time.perf_counter(), 0
    try:
        for video, grupo in grupos.items():
            print(f"\n{video}")
            cap = cv2.VideoCapture(str(ruta_video(args.videos, video)))
            fps = min(cap.get(cv2.CAP_PROP_FPS) or FPS_MAX, FPS_MAX)     # fotogramas por segundo muestreados
            for c in grupo:
                i += 1
                t0 = time.perf_counter()
                cruda = extraer_corte(cap, c.inicio, c.fin, c.espejo, args.confianza)
                con_manos = int((cruda[:, :2].max(axis=1) > 0).sum())
                tramos = ejecuciones(cruda) or [(0, len(cruda))]   # sin reposos: el corte es una sola ejecución
                muestras, segundos = [], []
                for a, b in tramos:
                    limpia, _ = limpiar(cruda[a:b])
                    if len(limpia) >= MIN_FRAMES:
                        muestras.append(a_largo_fijo(limpia, args.ventana))
                        segundos.append(round(len(limpia) / fps, 2))   # duración real: el remuestreo la borra
                linea = (f"  [{i:>3}/{len(pendientes)}] {c.destino:<28} {mmss(c.inicio)}-{mmss(c.fin)}  "
                         f"{len(cruda)} fot., {con_manos} con manos")
                if not muestras:
                    print(f"{linea}  -> DESCARTADO (menos de {MIN_FRAMES} fotogramas con manos)")
                    descartados.append(c)
                    continue
                anteriores = set(viejas.get(c.destino, {}).get("archivos", []))
                archivos = guardar_muestras(salida, c.destino, muestras)
                for r in anteriores - set(archivos):     # antes salieron más ejecuciones de este corte
                    (salida / r).unlink(missing_ok=True)
                nuevas[c.destino] = {"clave": c.clave(), "frames": len(cruda), "frames_con_manos": con_manos,
                                     "ejecuciones": [list(t) for t in tramos], "segundos": segundos,
                                     "archivos": archivos, "creado": datetime.now().isoformat(timespec="seconds")}
                aviso = "  [pocas manos: revisa el corte]" if con_manos < 0.6 * len(cruda) else ""
                print(f"{linea}; {len(muestras)} ejecuc. de {' + '.join(f'{s:.1f}' for s in segundos)} s "
                      f"-> {len(muestras)} x {args.ventana}  ({time.perf_counter() - t0:.1f} s){aviso}", flush=True)
            cap.release()
        hecho = True
    finally:
        if not hecho:                                    # Ctrl+C: lo viejo sigue en el índice para limpiarlo luego
            print("\n[INTERRUMPIDO] Se guarda lo procesado; vuelve a correr para continuar.")
            nuevas = {**{d: m for d, m in viejas.items()
                         if all((salida / r).is_file() for r in m.get("archivos", []))}, **nuevas}
        escribir_indice(salida, parametros, nuevas)

    vigentes = {r for m in nuevas.values() for r in m["archivos"]}
    borrados = {r for m in viejas.values() for r in m.get("archivos", [])} - vigentes   # cortes quitados o corregidos
    for r in borrados:
        (salida / r).unlink(missing_ok=True)
    for carpeta in {(salida / r).parent for r in borrados}:
        if carpeta.is_dir() and not any(carpeta.iterdir()):
            carpeta.rmdir()
    print(f"\nListo en {time.perf_counter() - t_total:.0f} s: {len(vigentes)} muestras de {len(nuevas)} cortes, "
          f"{len(descartados)} cortes descartados, {len(borrados)} archivos viejos borrados.")
    for c in descartados:
        print(f"  [DESCARTADO] entrada {c.n}: {c.palabra} {mmss(c.inicio)}-{mmss(c.fin)} en {c.video}")

    if hecho and not args.no_mover:                      # video con TODOS sus cortes extraídos -> videos/listo/
        for video in dict.fromkeys(c.video for c in cortes):
            origen = args.videos / video
            if origen.is_file() and all(c.destino in nuevas for c in cortes if c.video == video):
                (args.videos / LISTO).mkdir(exist_ok=True)
                shutil.move(str(origen), str(args.videos / LISTO / video))
                print(f"  Movido a {args.videos.name}/{LISTO}/: {video}")


def cargar_dataset(salida=SALIDA):
    """Para entrenar: X (N, 30, F) float32 y y (N,) con la palabra (nombre de la carpeta) de cada muestra."""
    rutas = sorted(Path(salida).glob("*/*seq_*.npy"))   # del extractor (seq_) y propias (pablo_seq_, gestos.py)
    X = np.stack([np.load(r) for r in rutas]) if rutas else np.zeros((0, VENTANA, F), np.float32)
    return X, np.array([r.parent.name for r in rutas])


def listar(salida):
    carpetas = sorted(p for p in salida.iterdir() if p.is_dir()) if salida.is_dir() else []
    filas = [(p.name, len(list(p.glob("*seq_*.npy")))) for p in carpetas]
    filas = [f for f in filas if f[1]]
    if not filas:
        print(f"{salida} está vacío.")
        return
    for palabra, n in filas:
        print(f"  {palabra:<24}{n:>4}")
    print(f"  {len(filas)} palabras, {sum(n for _, n in filas)} muestras en {salida}")


def main():
    sys.stdout.reconfigure(errors="replace")             # nombres con ñ/acentos en consolas que no son UTF-8
    p = argparse.ArgumentParser(description="Extrae muestras de 30 fotogramas de los cortes de un archivo JSON/CSV.")
    p.add_argument("--config", type=Path, default=CONFIG, help=f"cortes (default {CONFIG.name})")
    p.add_argument("--videos", type=Path, default=VIDEOS, help=f"carpeta de los videos (default {VIDEOS.name}/)")
    p.add_argument("--salida", type=Path, default=SALIDA, help=f"carpeta del dataset (default {SALIDA.name}/)")
    p.add_argument("--ventana", type=int, default=VENTANA, help=f"fotogramas por muestra (default {VENTANA})")
    p.add_argument("--confianza", type=float, default=CONFIANZA, help=f"detección de MediaPipe (default {CONFIANZA})")
    p.add_argument("--revisar", action="store_true", help="solo validar y mostrar el plan (no corre MediaPipe)")
    p.add_argument("--rehacer", action="store_true", help="extraer todo de nuevo aunque ya exista")
    p.add_argument("--listar", action="store_true", help="mostrar cuántas muestras hay por palabra")
    p.add_argument("--importar", type=Path, help="notas a mano (.txt) que reemplazan los cortes de sus videos")
    p.add_argument("--no-mover", action="store_true", help=f"no mover los videos terminados a videos/{LISTO}/")
    args = p.parse_args()
    if args.ventana < 2:
        p.error("--ventana debe ser al menos 2")
    if args.listar:
        listar(args.salida)
        return
    if args.importar:
        if args.config.suffix.lower() != ".json":
            p.error("--importar escribe en un .json")
        importadas, avisos = importar_anotaciones(args.importar)
        for a in avisos:
            print(f"[AVISO] {args.importar.name}, {a}")
        if not importadas:
            raise SystemExit(f"[ERROR] {args.importar.name} no tiene cortes reconocibles.")
        videos = {e["video"] for e in importadas}
        previas = leer_config(args.config) if args.config.is_file() else []
        conservadas = [e for e in previas if not (isinstance(e, dict) and e.get("video") in videos)]
        escribir_config(args.config, conservadas + importadas)
        print(f"Importados {len(importadas)} cortes de {len(videos)} video(s) a {args.config.name} "
              f"(reemplazan {len(previas) - len(conservadas)} que había de esos videos; respaldo en .json.bak).\n")

    cortes, errores, avisos = validar(leer_config(args.config), args.videos)
    for a in avisos:
        print(f"[AVISO] {a}")
    for e in errores:
        print(f"[ERROR] {e}")
    if errores:
        raise SystemExit(f"\n{len(errores)} errores en {args.config.name}: corrígelos y vuelve a correr "
                         "(no se procesó nada).")
    if not cortes:
        raise SystemExit(f"{args.config.name} no tiene cortes.")
    parametros = {"formato": FORMATO, "ventana": args.ventana, "fps_max": FPS_MAX, "confianza": args.confianza,
                  "hueco_max": HUECO_MAX, "min_frames": MIN_FRAMES, "reposo_y": REPOSO_Y, "reposo_min": REPOSO_MIN,
                  "ejecucion_min": EJECUCION_MIN}

    if args.revisar:
        al_dia, reusar, pendientes = planear(cortes, leer_indice(args.salida), parametros, args.rehacer, args.salida)
        por_palabra = {}
        for c in cortes:
            n, s = por_palabra.get(c.palabra, (0, 0.0))
            por_palabra[c.palabra] = (n + 1, s + c.fin - c.inicio)
        for palabra, (n, s) in sorted(por_palabra.items()):
            print(f"  {palabra:<24}{n:>4} cortes {s:>6.1f} s")
        print(f"\nConfiguración válida: {len(cortes)} cortes, {len(por_palabra)} palabras. {len(al_dia)} al día, "
              f"{len(reusar)} reutilizables, {len(pendientes)} por extraer "
              f"({sum(c.fin - c.inicio for c in pendientes):.0f} s de video).")
        return
    procesar(cortes, args, parametros)
    print()
    listar(args.salida)


if __name__ == "__main__":
    sys.exit(main())
