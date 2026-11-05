# CLAUDE.md — Traductor LSM

Traductor en tiempo real del alfabeto de la Lengua de Señas Mexicana (LSM) con cámara web: dos manos + rostro + voz.
Python 3.12 (MediaPipe 0.10.18 no corre en 3.13) · MediaPipe Hands/Face Mesh · OpenCV · NumPy · pyttsx3 · FastAPI + uvicorn (WebSocket). Windows.
Interfaz web local (sin ventanas de OpenCV): tema claro/oscuro, Modo Práctica (letras y palabras).

## Comandos (siempre con el Python del venv)
- Ejecutar: `.\venv\Scripts\python.exe server.py` → abre solo `http://localhost:5000` en el navegador predeterminado (`python main.py` hace lo mismo). Salir: botón Detener o Ctrl+C.
- Revisar: `.\venv\Scripts\python.exe -m pyflakes *.py` y `node --check static/app.js`
- Importa todo: `.\venv\Scripts\python.exe -c "import main, server"`
- Entorno nuevo: `py -3.12 -m venv venv` y `pip install opencv-python mediapipe==0.10.18 numpy pyttsx3 fastapi uvicorn websockets` (+ `faster-whisper` solo para `auto_etiquetador.py`; no sube numpy ni protobuf)
- NO lanzar navegadores headless ni pruebas automáticas de UI (el sistema las bloquea y la sesión se cuelga): solo escribir archivos; el usuario prueba en su navegador.

## Mapa
Flujo por frame: `CamaraReciente` (hilo, solo el último frame) → espejo → `detectar_manos` + `detectar_rostro` → `Traductor.paso()` → voz/hardware → JSON (estado + coordenadas de las manos) → WebSocket `/ws` → navegador (esqueleto en canvas). En paralelo, `/video_feed` (MJPEG) transmite el fotograma más nuevo SIN esperar a MediaPipe → `<img>` debajo del canvas.
Python es el ÚNICO dueño de la cámara: en la laptop del usuario Windows no la comparte con Chrome (`NotReadableError`); por eso no se usa `getUserMedia`.

| Archivo | Contenido |
|---|---|
| `server.py` | FastAPI: sirve `static/`, `/ws` manda por frame UN JSON (~0.5 KB sin manos, ~1.6 KB con dos); recibe `{"accion": ESPACIO/BORRAR/LIMPIAR/DIAGNOSTICO/RECALIBRAR/SALIR}`. `Difusor` = último estado (salta, no acumula retraso). `/video_feed`: MJPEG (`multipart/x-mixed-replace`) del frame más nuevo. El `lifespan` abre y cierra la cámara. Solo 127.0.0.1 |
| `main.py` | `capturar(cam, publicar, comandos, detener)`: bucle de proceso en un hilo, acciones de la cola, FPS, voz/hardware, `tiempos` (n, captura_ms, proceso_ms, descartados) |
| `camara.py` | `CamaraReciente`: lee la cámara en su propio hilo y guarda solo el último frame con su hora (`time.time()`, comparable con `Date.now()`). `leer(visto, espera, consumir)`: el proceso consume, el video lee con `consumir=False`. `a_jpeg`: calidad 70, ~1 ms |
| `classifier_hands.py` | Umbrales de mano; geometría 2D/3D; `FiltroOneEuro`; `a_canonica` (refleja la izquierda); `analizar_mano` → `RasgosMano`; `clasificar_pose`; `Trayectoria` + reconocedores (J, Z, ?, vaivenes K/Ñ/QX, BORRAR) + `ReconocedorDinamico`; `DeteccionMano`, `leer_manos`, `SeguidorManos` (ranuras Izq/Der), `CanalMano` (estado por mano); `Estabilizador`, `ProgresoSuave` |
| `classifier_face.py` | Umbrales de rostro; `EstimadorPoseCabeza` (solvePnP, modelo canónico); cejas compensadas por la pose; cara tapada (error de reproyección); `DetectorOscilacion`; `AnalizadorFacial` (calibración neutra, ?, negar, asentir) |
| `traductor.py` | `Escritor` (texto, hold-to-type por mano, eventos) y `Traductor.paso()`: une manos y rostro. No dibuja ni habla |
| `voice_engine.py` | `limpiar_texto` (`_` → espacio), `VozAsincrona` (hilo), `PoliticaVoz`, `enviar_a_hardware` (mock serial) |
| `ui_panel.py` | Presentación: `GUIA` del alfabeto, `vista_mano`, `mano_principal`, `config_web` (guía, umbral cejas, segundos, `conexiones` de los huesos, bases de los trazos), `estado_web` (JSON por frame, progresos 0-100; por mano `puntos` 21×[x,y,z] normalizados a la imagen espejada y `estela`) |
| `static/index.html` | Encabezado (voz, interruptor sol/luna, guía, Detener); video + subtítulos centrados; panel con pestañas Traducir / Práctica; manos; rostro; guía lateral |
| `static/estilos.css` | Variables por tema (`:root`/`[data-theme="dark"]` y `[data-theme="light"]`); fases del anillo (0-50 % azul, 50-99 % ámbar, 100 % verde + brillo); tarjeta con giro, confeti, sello sobre el video |
| `static/app.js` | `<img src="/video_feed">` (espejo con CSS; primer fotograma = `naturalWidth > 0`; si falla → "solo esqueleto"), esqueleto en `<canvas>` con requestAnimationFrame, latencia = `Date.now() - captura_ms`, estado en el DOM, tema en `localStorage` (`lsm-tema`), Modo Práctica: acierto = una mano pasa a "registrada" con la seña objetivo; fallo = otra letra registrada (reinicia racha) |

| `video_extractor.py` | Herramienta independiente (no la usa el servidor): cortes de `dataset_config.json` (o CSV) → salto con `POS_MSEC` (exacto en los videos de `videos/`) → mismo pipeline de manos (espejo + `detectar_manos` + `SeguidorManos`, ≤ 30 FPS) → `ejecuciones` (parte en reposos: muñeca más alta y ≥ 0.8 o sin manos ≥ 0.4 s; tramos ≥ 1 s; normal y lenta) → `limpiar` (quita extremos sin manos, rellena huecos ≤ 6) → `a_largo_fijo` (interpolación a 30) → `dataset/<palabra>/seq_XX_N.npy`. F = 254 por fotograma. `dataset/indice.json`: incremental por corte (clave = video, inicio, fin, espejo; `archivos`). `--importar notas.txt` ("palabra: 11 s - 1:03") reemplaza los cortes de esos videos. Video con todos sus cortes extraídos → `videos/listo/` (`ruta_video` busca en ambos). `cargar_dataset()` → X (N, 30, 254), y |
| `auto_etiquetador.py` | Independiente, opcional (`faster-whisper`): audio con PyAV propio (`faster_whisper.decode_audio` falla con PyAV ≥ 15) → Whisper small (es, VAD, palabras) → emisiones de 1-3 palabras entre pausas → corte [palabra − 0.5 s, + 4 s] → agrega a `dataset_config.json` con `"origen": "whisper"`; sin voz → `requiere_anotacion_manual.txt`. Nunca toca videos con cortes manuales. Vocabulario solo filtra (como hotwords cambió la segmentación) |

Dependencias: `voice_engine` ← `classifier_hands` ← `classifier_face` ← `traductor` ← `ui_panel` ← `main` (+ `camara`) ← `server`. El navegador no reconoce nada: solo presenta el JSON.

## Reglas del dominio (no romper)
- Imagen espejada ANTES de MediaPipe: la etiqueta Left/Right es la mano real. La izquierda se refleja (`a_canonica`) y las reglas se escriben una sola vez.
- Rasgos de forma en coordenadas de mundo 3D; contactos (pulgar-índice) en la imagen.
- Ventanas de trazo en SEGUNDOS y hold-to-type con `time.monotonic()`: nada depende de los FPS.
- `Traductor.paso()` se usa igual en vivo y al reproducir grabaciones: toda la lógica nueva va ahí o debajo, no en `main.py`.
- Solo el hilo de proceso toca el texto: el servidor pasa las acciones por una cola. Python no dibuja ni manda video; las etiquetas son HTML y el esqueleto lo dibuja el navegador.
- Coordenadas en la imagen ESPEJADA: el `<img>` del video (frame sin espejar) se espeja con CSS (`scaleX(-1)`); el canvas del esqueleto, no.
- Cada umbral está junto a su comentario con el rango medido. No mover umbrales a ciegas: medir con sesión guiada (grabar solo landmarks), separar calibración de prueba y congelar el código antes de medir.
- Alfabeto: lámina oficial SEP. Cifras y decisiones: `DOCUMENTACION.md` secciones 8 y 10.

## Estilo de código
- Simple y modular; sin código muerto ni abstracciones de más. Nombres en español, como el código existente.
- Editar solo las líneas necesarias; igualar densidad de comentarios y estilo del entorno.
- Sin dependencias nuevas salvo necesidad real.

## Comunicación
- Responder en español, directo, sin relleno. Reportar resultados con cifras reales y decir qué falló.
- Git: commit/push solo si se pide. Documentación sin emojis en títulos. Leer un archivo del usuario antes de sobrescribirlo (los edita a mano).
