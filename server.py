"""Servidor web local del Traductor LSM: sirve la interfaz (static/), el estado por WebSocket y el video por MJPEG.

Ejecutar:  .\\venv\\Scripts\\python.exe server.py   -> abre http://localhost:5000 en el navegador predeterminado.
- /ws: por cada frame procesado, UN mensaje de texto (JSON: señas, texto, rostro y coordenadas de las manos,
  ~0.5-1.6 KB). El navegador manda acciones {"accion": "ESPACIO" | "BORRAR" | "LIMPIAR" | "DIAGNOSTICO" |
  "RECALIBRAR" | "SALIR"}.
- /video_feed: vista previa de la cámara (multipart/x-mixed-replace, JPEG calidad 70) para un <img>. Sale del
  fotograma más nuevo SIN esperar a MediaPipe; el navegador dibuja el esqueleto encima con los datos de /ws.
Python es el único dueño de la cámara (en Windows el navegador no siempre puede compartirla). Solo 127.0.0.1.
"""
import asyncio
import json
import queue
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from camara import CamaraReciente, a_jpeg
from main import ACCIONES_TEXTO, capturar
from ui_panel import config_web

HOST, PUERTO = "127.0.0.1", 5000
URL = f"http://localhost:{PUERTO}"
ESTATICOS = Path(__file__).with_name("static")
ACCIONES = set(ACCIONES_TEXTO) | {"DIAGNOSTICO", "RECALIBRAR"}


class Difusor:
    """Guarda solo el último estado (JSON). Cada cliente toma siempre lo más reciente: si el navegador va lento se
    saltan estados en lugar de acumular retraso."""

    def __init__(self):
        self._cond = threading.Condition()
        self._n = 0
        self._estado = None

    def publicar(self, estado):
        with self._cond:
            self._n += 1
            self._estado = estado
            self._cond.notify_all()

    def esperar(self, visto, espera):
        """(n, estado) en cuanto haya uno posterior a `visto`; None si pasa `espera` sin novedades."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._n != visto, espera):
                return None
            return self._n, self._estado


difusor = Difusor()
comandos = queue.Queue()
detener = threading.Event()
servidor = None
camara = None                  # CamaraReciente compartida: la consume el proceso y la lee (sin consumir) el video


def _cerrando():
    return detener.is_set() or (servidor is not None and servidor.should_exit)


@asynccontextmanager
async def ciclo_de_vida(app):
    global camara
    camara = await asyncio.to_thread(CamaraReciente, 0)
    hilo = threading.Thread(target=capturar, args=(camara, difusor.publicar, comandos, detener), name="proceso",
                            daemon=True)
    hilo.start()
    threading.Timer(1.0, webbrowser.open, args=(URL,)).start()
    yield
    detener.set()
    await asyncio.to_thread(hilo.join, 5)
    camara.cerrar()


app = FastAPI(lifespan=ciclo_de_vida, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=ESTATICOS), name="static")


@app.get("/")
async def inicio():
    return FileResponse(ESTATICOS / "index.html")


async def _fotogramas_mjpeg():
    """Cada fotograma nuevo de la cámara como una parte JPEG del flujo multipart. Si el navegador va lento, se
    salta fotogramas (siempre el más nuevo). Termina al cerrar el servidor o si la cámara se pierde."""
    visto = 0
    while not _cerrando():
        paquete = await asyncio.to_thread(camara.leer, visto, 0.5, False)
        if paquete is None:
            if not camara.activa:
                return
            continue
        visto, frame, _ = paquete
        jpeg = await asyncio.to_thread(a_jpeg, frame)
        if jpeg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + \
                b"\r\n\r\n" + jpeg + b"\r\n"


@app.get("/video_feed")
async def video_feed():
    if camara is None or not camara.activa:
        return Response(status_code=503)            # el <img> dispara onerror: la interfaz pasa a solo esqueleto
    return StreamingResponse(_fotogramas_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


async def _recibir(ws):
    """Acciones del navegador -> cola del hilo de proceso (el único que toca el texto)."""
    while True:
        try:
            accion = json.loads(await ws.receive_text()).get("accion")
        except (ValueError, AttributeError):
            continue
        except (WebSocketDisconnect, RuntimeError):
            return
        if accion == "SALIR" and servidor is not None:
            servidor.should_exit = True
        elif accion in ACCIONES:
            comandos.put(accion)


@app.websocket("/ws")
async def transmitir(ws: WebSocket):
    await ws.accept()
    recibir = asyncio.create_task(_recibir(ws))
    visto = 0
    try:
        await ws.send_text(json.dumps(config_web(), ensure_ascii=False))
        while not recibir.done() and not _cerrando():
            paquete = await asyncio.to_thread(difusor.esperar, visto, 0.5)
            if paquete is None:
                continue
            visto, estado = paquete
            await ws.send_text(estado)
    except (WebSocketDisconnect, RuntimeError):
        pass                                 # el navegador cerró la pestaña
    finally:
        recibir.cancel()


def main():
    global servidor
    servidor = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PUERTO, log_level="warning",
                                             timeout_graceful_shutdown=2))
    print(f"Traductor LSM en {URL}  (Ctrl+C o el botón Detener para salir)")
    servidor.run()
    print("Programa cerrado correctamente.")


if __name__ == "__main__":
    main()
