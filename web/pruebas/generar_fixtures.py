"""Fixtures para comparar el motor del navegador (web/motor/) con Python: por fotograma, las detecciones crudas de
MediaPipe (manos y rostro) y lo que produce Traductor.paso + la presentación de main.py. Necesita los videos en
datos/videos/ (o listo/). Salida: web/pruebas/fixtures/fixture_<tramo>.json (no se versionan, ~27 MB).

Uso (desde la raíz):  .\\venv\\Scripts\\python.exe web\\pruebas\\generar_fixtures.py
Después:              node web/pruebas/paridad.mjs   y   node web/pruebas/humo.mjs
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "python"))
import gestos  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "fixtures"
SALIDA.mkdir(exist_ok=True)
gestos.REGISTRO = SALIDA / "log_fixtures.csv"             # no ensucia datos/gestos_vivo.csv
from classifier_face import INDICES_ROSTRO, detectar_rostro  # noqa: E402
from classifier_hands import LADOS, MAX_MANOS, MODELO_MANOS, ProgresoSuave, detectar_manos  # noqa: E402
from traductor import Traductor  # noqa: E402
from ui_panel import mano_principal, vista_mano  # noqa: E402
from video_extractor import VIDEOS, ruta_video  # noqa: E402

# (nombre, video, inicio s, fin s): alfabeto completo (letras, J, Z) y frases con movimiento + rostro
TRAMOS = [("alfabeto", "Alfabeto de Lengua de Señas Mexicana.mp4", 0, 99),
          ("frases", "25 palabras y frases en LSM para principiantes. Aprende Lengua de Señas Mexicana.mp4", 10, 110)]
FPS = 20


def r6(x):
    return round(float(x), 6)


def rasgos(r):
    if r is None:
        return None
    return {"ext": [r6(e) for e in r.extension], "orient": r6(r.orient), "dti": r6(r.d_ti)}


rec = gestos.ReconocedorGestos(asincrono=False)              # reproducible: el DTW se evalúa en el mismo fotograma
for nombre, video, a, b in TRAMOS:
    tr = Traductor()
    tr.gestos = rec
    rec.poner_objetivo(None)
    barra, principal = ProgresoSuave(), None
    cap = cv2.VideoCapture(str(ruta_video(VIDEOS, video)))
    cap.set(cv2.CAP_PROP_POS_MSEC, a * 1000)
    t_sig, cuadros = a, []
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=MAX_MANOS, model_complexity=MODELO_MANOS,
                                  min_detection_confidence=0.7, min_tracking_confidence=0.6) as h, \
            mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                                            min_detection_confidence=0.5, min_tracking_confidence=0.5) as fm, \
            contextlib.redirect_stdout(io.StringIO()):
        while cap.grab():
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if t > b:
                break
            if t + 1e-3 < t_sig:
                continue
            t_sig += 1 / FPS
            f = cv2.flip(cap.retrieve()[1], 1)
            alto, ancho = f.shape[:2]
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            dets = detectar_manos(h, rgb, ancho, alto)
            cara = detectar_rostro(fm, rgb, ancho, alto)
            entrada = {"t": t, "dets": [{"l": d.lado_mp, "c": d.confianza, "img": d.img.ravel().tolist(),
                                         "mundo": d.mundo.ravel().tolist()} for d in dets],
                       "cara": None if cara is None else {str(i): cara[i].tolist() for i in INDICES_ROSTRO}}
            manos, evento = tr.paso(dets, cara, t, (ancho, alto))
            vistas = {}
            for lado in LADOS:
                v = vista_mano(manos[lado], tr.escritor.manos[lado], t)
                v.progreso = tr.canales[lado].barra.actualizar(v.progreso)
                vistas[lado] = v
            principal = mano_principal(vistas, principal)
            prog = barra.actualizar(vistas[principal].progreso if principal else 0.0)
            an, pose = tr.analizador, tr.analizador.pose_actual
            g = tr.gestos.resultado
            tr.gestos.resultado = None
            cuadros.append({"entrada": entrada, "salida": {
                "m": {l: [manos[l].pose_cruda, manos[l].pose, manos[l].gesto, manos[l].moviendo, r6(manos[l].progreso),
                          rasgos(manos[l].rasgos)] for l in LADOS},
                "v": {l: [vistas[l].sena, vistas[l].estado, r6(vistas[l].progreso)] for l in LADOS},
                "p": [principal, r6(prog)], "tx": tr.escritor.texto, "ev": evento,
                "f": [an.modo, None if pose is None else [r6(pose.pitch), r6(pose.yaw), r6(pose.roll), r6(pose.error)],
                      r6(an.ratio_cejas)],
                "g": None if g is None else {k: g[k] for k in ("mejor", "distancia", "separacion", "aceptado", "motivo",
                                                               "escrito")}}})
    (SALIDA / f"fixture_{nombre}.json").write_text(json.dumps({"tamano": [ancho, alto], "cuadros": cuadros}),
                                                    encoding="utf-8")
    print(f"[{nombre}] {len(cuadros)} fotogramas {ancho}x{alto}; texto final: {cuadros[-1]['salida']['tx']!r}",
          file=sys.stderr)
