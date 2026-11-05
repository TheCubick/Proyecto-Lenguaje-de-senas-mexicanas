"""Traductor de Lengua de Señas Mexicana (LSM): dos manos + rostro + voz, en tiempo real con una cámara web.

Motor: procesa siempre el fotograma más reciente de la cámara (camara.py, hilo propio) y publica el estado JSON:
señas, texto, rostro y coordenadas de las manos. El video para la vista previa lo transmite server.py por
/video_feed (MJPEG) desde la misma cámara; el navegador dibuja el esqueleto encima. Ejecutar con `python server.py`.

Módulos:  camara.py            captura en un hilo propio (único dueño de la cámara) y JPEG de la vista previa
            classifier_hands.py  manos: lateralidad, One-Euro, rasgos 3D, alfabeto estático y trazos
            classifier_face.py   rostro: pose 3D (solvePnP), cejas (pregunta), negar / asentir
            traductor.py         hold-to-type por mano y orquestación de un frame (Traductor.paso)
            voice_engine.py      voz asíncrona (pyttsx3) y puente al hardware (mock serial)
            ui_panel.py          estado JSON para el navegador (incluye las coordenadas de las manos)
            server.py            servidor web local (FastAPI: WebSocket + video MJPEG) y apertura del navegador
"""
import json
import queue
import time

import cv2
import mediapipe as mp

from classifier_face import PROCESAR_ROSTRO_CADA, detectar_rostro
from classifier_hands import LADOS, MAX_MANOS, ProgresoSuave, detectar_manos
from traductor import Traductor
from ui_panel import estado_web, mano_principal, vista_mano
from voice_engine import VOZ_HABILITADA, PoliticaVoz, VozAsincrona, enviar_a_hardware, limpiar_texto

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

ACCIONES_TEXTO = ("ESPACIO", "BORRAR", "LIMPIAR")   # Botones / teclas de la interfaz que editan el texto


def _error(publicar, mensaje):
    print(f"[ERROR] {mensaje}")
    publicar(json.dumps({"tipo": "error", "mensaje": mensaje}, ensure_ascii=False))


def capturar(cam, publicar, comandos, detener):
    """Bucle de procesamiento (en su propio hilo; la lectura de la cámara va en otro, ver camara.py).

    cam: CamaraReciente (la abre y la cierra server.py, que también la usa para el video); publicar(estado_json):
    entrega el estado de cada frame; comandos: cola con acciones de la interfaz ("ESPACIO", "BORRAR", "LIMPIAR",
    "DIAGNOSTICO", "RECALIBRAR"); detener: threading.Event que termina el bucle."""
    if not cam.activa:
        _error(publicar, "El motor no pudo abrir la cámara. Verifica que esté conectada y que otra app no la use.")
        return

    traductor = Traductor()
    escritor, analizador = traductor.escritor, traductor.analizador
    voz = VozAsincrona(VOZ_HABILITADA)
    politica = PoliticaVoz(voz)
    barra = ProgresoSuave()
    texto_enviado = escritor.texto
    diagnostico = False
    rostro = None
    principal = None
    fps, t_frame_anterior = 0.0, time.monotonic()
    n_frame, visto = 0, 0

    print("Traductor LSM iniciado. Mantén la cara neutra un momento para calibrar.")

    with mp_hands.Hands(static_image_mode=False, max_num_hands=MAX_MANOS,
                        min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands, \
            mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        try:
            while not detener.is_set():
                paquete = cam.leer(visto)              # el más nuevo; los que llegaron mientras procesábamos, no
                if paquete is None:
                    if cam.activa:
                        continue
                    _error(publicar, "Se perdió la señal de la cámara.")
                    break
                visto, frame, t_captura = paquete
                t_inicio = time.perf_counter()         # monotonic() en Windows tiene ~16 ms de resolución

                frame = cv2.flip(frame, 1)  # Espejo (MediaPipe espera la imagen espejada para la lateralidad)
                alto, ancho = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                ahora = time.monotonic()
                n_frame += 1

                # --- Acciones que llegan de la interfaz web (botones y teclado del navegador) ---
                while True:
                    try:
                        accion = comandos.get_nowait()
                    except queue.Empty:
                        break
                    if accion in ACCIONES_TEXTO:
                        escritor.ejecutar(accion, ahora)
                    elif accion == "DIAGNOSTICO":
                        diagnostico = not diagnostico
                    elif accion == "RECALIBRAR":
                        analizador.recalibrar()

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
                progreso = barra.actualizar(vistas[principal].progreso if principal else 0.0)

                dt = ahora - t_frame_anterior
                t_frame_anterior = ahora
                if dt > 0:
                    fps = 1.0 / dt if fps == 0 else fps * 0.9 + (1.0 / dt) * 0.1

                # --- Salida: solo el estado (coordenadas incluidas); el navegador pone el video ---
                tiempos = {"n": visto, "captura_ms": round(t_captura * 1000),
                           "proceso_ms": round((time.perf_counter() - t_inicio) * 1000, 1),
                           "descartados": cam.descartados}
                estado = estado_web(traductor, manos, vistas, principal, progreso, fps, voz.estado, (ancho, alto),
                                    diagnostico, tiempos)
                publicar(json.dumps(estado, ensure_ascii=False, separators=(",", ":")))
        finally:
            voz.cerrar()

    print(f"Texto final: {limpiar_texto(escritor.texto)!r}")


if __name__ == "__main__":
    import server
    server.main()
