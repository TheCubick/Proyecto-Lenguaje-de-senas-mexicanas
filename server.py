"""Servidor web local del Traductor LSM: sirve la interfaz (static/) y transmite por WebSocket el video y el estado.

Ejecutar:  .\\venv\\Scripts\\python.exe server.py   -> abre http://localhost:5000 en el navegador predeterminado.
Por /ws viajan, por cada frame, un mensaje binario (JPEG) y uno de texto (JSON del estado); el navegador manda
acciones {"accion": "ESPACIO" | "BORRAR" | "LIMPIAR" | "DIAGNOSTICO" | "RECALIBRAR" | "SALIR"}.
Solo escucha en 127.0.0.1: el video de la cámara no sale de la computadora.
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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from main import ACCIONES_TEXTO, capturar
from ui_panel import config_web

HOST, PUERTO = "127.0.0.1", 5000
URL = f"http://localhost:{PUERTO}"
ESTATICOS = Path(__file__).with_name("static")
ACCIONES = set(ACCIONES_TEXTO) | {"DIAGNOSTICO", "RECALIBRAR"}


class Difusor:
    """Guarda solo el último (JPEG, JSON). Cada cliente toma siempre lo más reciente: si el navegador va lento se
    saltan frames en lugar de acumular retraso."""

    def __init__(self):
        self._cond = threading.Condition()
        self._n = 0
        self._paquete = (None, None)

    def publicar(self, jpeg, estado):
        with self._cond:
            self._n += 1
            self._paquete = (jpeg, estado)
            self._cond.notify_all()

    def esperar(self, visto, espera):
        """(n, jpeg, estado) en cuanto haya un paquete posterior a `visto`; None si pasa `espera` sin novedades."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._n != visto, espera):
                return None
            return (self._n, *self._paquete)


difusor = Difusor()
comandos = queue.Queue()
detener = threading.Event()
servidor = None


@asynccontextmanager
async def ciclo_de_vida(app):
    hilo = threading.Thread(target=capturar, args=(difusor.publicar, comandos, detener), name="captura",
                            daemon=True)
    hilo.start()
    threading.Timer(1.0, webbrowser.open, args=(URL,)).start()
    yield
    detener.set()
    await asyncio.to_thread(hilo.join, 5)


app = FastAPI(lifespan=ciclo_de_vida, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=ESTATICOS), name="static")


@app.get("/")
async def inicio():
    return FileResponse(ESTATICOS / "index.html")


async def _recibir(ws):
    """Acciones del navegador -> cola del hilo de captura (el único que toca el texto)."""
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
        while not recibir.done() and not detener.is_set():
            paquete = await asyncio.to_thread(difusor.esperar, visto, 0.5)
            if paquete is None:
                continue
            visto, jpeg, estado = paquete
            if jpeg is not None:
                await ws.send_bytes(jpeg)
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
