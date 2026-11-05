"""Captura en un hilo propio: el procesamiento siempre toma el fotograma MÁS RECIENTE y descarta los viejos.

Sin este hilo, mientras MediaPipe procesa un fotograma (25-60 ms) la cámara deja los siguientes en su búfer y el
bucle procesa imágenes atrasadas. Con él, cada vuelta del bucle toma la imagen más nueva y su hora de captura.

Python es el ÚNICO dueño de la cámara (en Windows el controlador puede no compartirla con el navegador). El video
que ve la persona sale de aquí mismo: `a_jpeg` lo comprime y server.py lo transmite por /video_feed (MJPEG)."""
import threading
import time

import cv2

VIDEO_CALIDAD = 70             # JPEG de la vista previa: medido 1.0 ms y ~16-18 KB por fotograma a 640x480
VIDEO_ANCHO_MAX = 640          # Una cámara más ancha se reduce a esto antes de comprimir


class CamaraReciente:
    """Lee la cámara sin parar y guarda solo el último fotograma (número, imagen BGR SIN espejar, hora en s)."""

    def __init__(self, indice=0):
        self.cap = cv2.VideoCapture(indice)
        self.activa = self.cap.isOpened()
        self.descartados = 0                 # fotogramas que el proceso no tomó porque llegó uno más nuevo
        self._cond = threading.Condition()
        self._n, self._consumido = 0, 0
        self._frame, self._t = None, 0.0
        self._hilo = threading.Thread(target=self._leer, name="camara", daemon=True)
        if self.activa:
            self._hilo.start()

    def _leer(self):
        while self.activa:
            ok, frame = self.cap.read()
            t = time.time()                  # reloj de pared: el navegador lo compara con Date.now() (latencia)
            with self._cond:
                if not ok:
                    self.activa = False      # señal perdida: quien espere en leer() se entera
                else:
                    if self._n > self._consumido:
                        self.descartados += 1
                    self._n, self._frame, self._t = self._n + 1, frame, t
                self._cond.notify_all()

    def leer(self, visto, espera=1.0, consumir=True):
        """(n, frame, t_captura) del fotograma más nuevo posterior a `visto`. None si en `espera` segundos no
        llegó ninguno (revisar `activa` para saber si la cámara se cerró). El proceso consume; la vista previa de
        video lee con consumir=False para no alterar el conteo de descartados. El frame es de solo lectura."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._n != visto or not self.activa, espera) or self._n == visto:
                return None
            if consumir:
                self._consumido = self._n
            return self._n, self._frame, self._t

    def cerrar(self):
        self.activa = False
        if self._hilo.is_alive():
            self._hilo.join(timeout=1.0)     # que termine cap.read() antes de liberar la cámara
        self.cap.release()


def a_jpeg(frame):
    """Fotograma -> bytes JPEG para la vista previa (sin espejar: el navegador lo espeja con CSS)."""
    alto, ancho = frame.shape[:2]
    if ancho > VIDEO_ANCHO_MAX:
        frame = cv2.resize(frame, (VIDEO_ANCHO_MAX, round(alto * VIDEO_ANCHO_MAX / ancho)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, VIDEO_CALIDAD])
    return jpeg.tobytes() if ok else None
