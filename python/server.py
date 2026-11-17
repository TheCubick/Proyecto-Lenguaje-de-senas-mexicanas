"""Servidor web local del Traductor LSM: sirve la interfaz (static/), el estado por WebSocket y el video por MJPEG.

Ejecutar:  .\\venv\\Scripts\\python.exe python\\server.py   -> abre http://localhost:5000 en el navegador predeterminado.
- /ws: por cada frame procesado, UN mensaje de texto (JSON: señas, texto, rostro y coordenadas de las manos,
  ~0.5-1.6 KB). El navegador manda acciones {"accion": "ESPACIO" | "BORRAR" | "LIMPIAR" | "DIAGNOSTICO" |
  "RECALIBRAR" | "GUARDAR_MUESTRA" | "LEER" | "INICIAR" | "PAUSAR" | "SALIR"}, {"accion": "GESTO", "objetivo": frase | null},
  {"accion": "VOZ", "activa": bool, "anunciar": bool} y {"accion": "VELOCIDAD", "segundos": 0.5-2.0}.
  INICIAR / PAUSAR abren y cierran la cámara sin cerrar el servidor; en pausa se publica {"tipo": "pausa"}.
- /videos/<nombre>: clips de referencia del Modo Práctica (solo los videos de ui_panel.demostraciones).
- /video_feed: vista previa de la cámara (multipart/x-mixed-replace, JPEG calidad 70) para un <img>. Sale del
  fotograma más nuevo SIN esperar a MediaPipe; el navegador dibuja el esqueleto encima con los datos de /ws.
Python es el único dueño de la cámara (en Windows el navegador no siempre puede compartirla). Solo 127.0.0.1.
"""
import asyncio
import json
import queue
import sys
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
from traductor import VELOCIDAD_MAX, VELOCIDAD_MIN, Traductor
from ui_panel import config_web, demostraciones
from voice_engine import VOZ_HABILITADA, VozAsincrona

HOST, PUERTO = "127.0.0.1", 5000
URL = f"http://localhost:{PUERTO}"
ESTATICOS = Path(__file__).resolve().parent.parent / "static"   # python/ y static/ son carpetas hermanas
ACCIONES = set(ACCIONES_TEXTO) | {"DIAGNOSTICO", "RECALIBRAR", "GUARDAR_MUESTRA", "LEER"}


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
apagar = threading.Event()     # cierre del servidor
servidor = None
camara = None                  # CamaraReciente compartida: la consume el proceso y la lee (sin consumir) el video
traductor = None               # uno para toda la sesión: el texto y las muestras sobreviven a las pausas
voz = None                     # igual: la voz silenciada sigue silenciada al reanudar
motor = {"hilo": None, "detener": None}
cerrojo_motor = threading.Lock()


def _cerrando():
    return apagar.is_set() or (servidor is not None and servidor.should_exit)


def iniciar_motor():
    """Abre la cámara y arranca el hilo de proceso (botón Iniciar). No hace nada si ya corre."""
    global camara
    with cerrojo_motor:
        if motor["hilo"] is not None and motor["hilo"].is_alive():
            return
        camara = CamaraReciente(0)
        motor["detener"] = threading.Event()
        motor["hilo"] = threading.Thread(target=capturar, name="proceso", daemon=True,
                                         args=(camara, difusor.publicar, comandos, motor["detener"], traductor, voz))
        motor["hilo"].start()


def pausar_motor(avisar=True):
    """Detiene el hilo de proceso y apaga la cámara (botón Pausar); el servidor y el texto siguen."""
    global camara
    with cerrojo_motor:
        if motor["hilo"] is None:
            return
        motor["detener"].set()
        motor["hilo"].join(5)
        motor["hilo"] = None
        if camara is not None:
            camara.cerrar()
            camara = None
        traductor.gestos.poner_objetivo(traductor.gestos.objetivo)   # sin un gesto a medias al reanudar
    if avisar:
        difusor.publicar(json.dumps({"tipo": "pausa"}))


@asynccontextmanager
async def ciclo_de_vida(app):
    global traductor, voz
    traductor = await asyncio.to_thread(Traductor)
    voz = VozAsincrona(VOZ_HABILITADA)
    await asyncio.to_thread(iniciar_motor)
    threading.Timer(1.0, webbrowser.open, args=(URL,)).start()
    yield
    apagar.set()
    await asyncio.to_thread(pausar_motor, False)
    voz.cerrar()


app = FastAPI(lifespan=ciclo_de_vida, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=ESTATICOS), name="static")


@app.middleware("http")
async def sin_cache(request, siguiente):
    """La interfaz (/ y /static/) se revalida en cada carga: sin esto el navegador reusaba su copia en caché y,
    tras actualizar los archivos, mostraba la página vieja con el motor nuevo (2026-11-10). Con ETag la respuesta
    es un 304 sin cuerpo si nada cambió."""
    respuesta = await siguiente(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        respuesta.headers["Cache-Control"] = "no-cache"
    return respuesta


@app.get("/")
async def inicio():
    return FileResponse(ESTATICOS / "index.html")


async def _fotogramas_mjpeg():
    """Cada fotograma nuevo de la cámara como una parte JPEG del flujo multipart. Si el navegador va lento, se
    salta fotogramas (siempre el más nuevo). Termina al cerrar el servidor o si la cámara se pierde."""
    visto, cam = 0, camara                   # la de este flujo: al pausar se cierra y el flujo termina
    while not _cerrando():
        paquete = await asyncio.to_thread(cam.leer, visto, 0.5, False)
        if paquete is None:
            if not cam.activa:
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


@app.get("/videos/{nombre}")
async def video_demo(nombre: str):
    """Video original de una demostración (el <video> pide rangos para saltar al corte). Lista blanca: nunca se
    sirve un archivo que no esté en demostraciones()."""
    ruta = demostraciones()[1].get(nombre)
    return FileResponse(ruta, media_type="video/mp4") if ruta else Response(status_code=404)


async def _recibir(ws):
    """Acciones del navegador -> cola del hilo de proceso (el único que toca el texto)."""
    while True:
        try:
            mensaje = json.loads(await ws.receive_text())
            accion = mensaje.get("accion")
        except (ValueError, AttributeError):
            continue
        except (WebSocketDisconnect, RuntimeError):
            return
        if accion == "SALIR" and servidor is not None:
            servidor.should_exit = True
        elif accion == "INICIAR":
            await asyncio.to_thread(iniciar_motor)
        elif accion == "PAUSAR":
            await asyncio.to_thread(pausar_motor)
        elif accion == "VOZ":                # interruptor de voz; "anunciar" = decir "Voz activada"
            comandos.put(("VOZ", bool(mensaje.get("activa")), bool(mensaje.get("anunciar"))))
        elif accion == "VELOCIDAD":          # segundos del hold-to-type (Lenta / Normal / Rápida)
            s = mensaje.get("segundos")
            if isinstance(s, (int, float)) and VELOCIDAD_MIN <= s <= VELOCIDAD_MAX:
                comandos.put(("VELOCIDAD", float(s)))
        elif accion == "GESTO":              # objetivo del Modo Práctica (None = ninguno); se valida en gestos.py
            objetivo = mensaje.get("objetivo")
            comandos.put(("GESTO", objetivo if isinstance(objetivo, str) else None))
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


def sin_ahorro_de_energia():
    """Windows: saca al proceso del estrangulamiento de energía (EcoQoS). En batería y con la consola en segundo
    plano (el navegador al frente), Windows lo manda a núcleos de bajo consumo. Medido (2026-11-05, en batería):
    MediaPipe Hands 47 ms por fotograma por defecto, 84 ms estrangulado y 23 ms con esto."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class Estado(ctypes.Structure):              # PROCESS_POWER_THROTTLING_STATE
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    # ControlMask EXECUTION_SPEED (0x1) | IGNORE_TIMER_RESOLUTION (0x4) con StateMask 0 = nunca estrangular
    estado = Estado(1, 0x1 | 0x4, 0)
    if not k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(estado), ctypes.sizeof(estado)):
        print(f"[AVISO] No se pudo quitar el ahorro de energía (error {ctypes.get_last_error()}): "
              "conecta la laptop a la corriente si va lenta.")


def main():
    global servidor
    sin_ahorro_de_energia()
    servidor = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PUERTO, log_level="warning",
                                             timeout_graceful_shutdown=2))
    print(f"Traductor LSM en {URL}  (Ctrl+C o el botón Apagar para salir)")
    servidor.run()
    print("Programa cerrado correctamente.")


if __name__ == "__main__":
    main()
