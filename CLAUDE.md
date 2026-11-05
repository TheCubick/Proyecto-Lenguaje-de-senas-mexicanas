# CLAUDE.md — Traductor LSM

Traductor en tiempo real del alfabeto de la Lengua de Señas Mexicana (LSM) con cámara web: dos manos + rostro + voz.
Python 3.12 (MediaPipe 0.10.18 no corre en 3.13) · MediaPipe Hands/Face Mesh · OpenCV · NumPy · pyttsx3 · FastAPI + uvicorn (WebSocket). Windows.
Interfaz web local (sin ventanas de OpenCV): tema claro/oscuro, Modo Práctica (letras y palabras).

## Comandos (siempre con el Python del venv)
- Ejecutar: `.\venv\Scripts\python.exe server.py` → abre solo `http://localhost:5000` en el navegador predeterminado (`python main.py` hace lo mismo). Salir: botón Detener o Ctrl+C.
- Revisar: `.\venv\Scripts\python.exe -m pyflakes *.py` y `node --check static/app.js`
- Importa todo: `.\venv\Scripts\python.exe -c "import main, server"`
- Entorno nuevo: `py -3.12 -m venv venv` y `pip install opencv-python mediapipe==0.10.18 numpy pyttsx3 fastapi uvicorn websockets`
- NO lanzar navegadores headless ni pruebas automáticas de UI (el sistema las bloquea y la sesión se cuelga): solo escribir archivos; el usuario prueba en su navegador.

## Mapa
Flujo por frame: cámara → espejo → `detectar_manos` + `detectar_rostro` → `Traductor.paso()` → voz/hardware → JPEG (puntos de las manos) + JSON → WebSocket `/ws` → navegador.

| Archivo | Contenido |
|---|---|
| `server.py` | FastAPI: sirve `static/`, `/ws` manda por frame un JPEG binario + un JSON; recibe `{"accion": ESPACIO/BORRAR/LIMPIAR/DIAGNOSTICO/RECALIBRAR/SALIR}`. `Difusor` = último paquete (salta frames, no acumula retraso). Solo 127.0.0.1 |
| `main.py` | `capturar(publicar, comandos, detener)`: bucle de captura en un hilo, acciones de la cola, FPS, voz/hardware |
| `classifier_hands.py` | Umbrales de mano; geometría 2D/3D; `FiltroOneEuro`; `a_canonica` (refleja la izquierda); `analizar_mano` → `RasgosMano`; `clasificar_pose`; `Trayectoria` + reconocedores (J, Z, ?, vaivenes K/Ñ/QX, BORRAR) + `ReconocedorDinamico`; `DeteccionMano`, `leer_manos`, `SeguidorManos` (ranuras Izq/Der), `CanalMano` (estado por mano); `Estabilizador`, `ProgresoSuave` |
| `classifier_face.py` | Umbrales de rostro; `EstimadorPoseCabeza` (solvePnP, modelo canónico); cejas compensadas por la pose; cara tapada (error de reproyección); `DetectorOscilacion`; `AnalizadorFacial` (calibración neutra, ?, negar, asentir) |
| `traductor.py` | `Escritor` (texto, hold-to-type por mano, eventos) y `Traductor.paso()`: une manos y rostro. No dibuja ni habla |
| `voice_engine.py` | `limpiar_texto` (`_` → espacio), `VozAsincrona` (hilo), `PoliticaVoz`, `enviar_a_hardware` (mock serial) |
| `ui_panel.py` | Presentación: `GUIA` del alfabeto, `vista_mano`, `mano_principal`, `dibujar_manos` (solo esqueleto, sin texto), `config_web` (guía, umbral cejas, segundos, bases de los trazos), `estado_web` (JSON por frame, progresos 0-100) |
| `static/index.html` | Encabezado (voz, interruptor sol/luna, guía, Detener); video + subtítulos centrados; panel con pestañas Traducir / Práctica; manos; rostro; guía lateral |
| `static/estilos.css` | Variables por tema (`:root`/`[data-theme="dark"]` y `[data-theme="light"]`); fases del anillo (0-50 % azul, 50-99 % ámbar, 100 % verde + brillo); tarjeta con giro, confeti, sello sobre el video |
| `static/app.js` | WebSocket (video en `<canvas>`, estado en el DOM), tema en `localStorage` (`lsm-tema`), Modo Práctica: acierto = una mano pasa a "registrada" con la seña objetivo; fallo = otra letra registrada (reinicia racha) |

Dependencias: `voice_engine` ← `classifier_hands` ← `classifier_face` ← `traductor` ← `ui_panel` ← `main` ← `server`. El navegador no reconoce nada: solo presenta el JSON.

## Reglas del dominio (no romper)
- Imagen espejada ANTES de MediaPipe: la etiqueta Left/Right es la mano real. La izquierda se refleja (`a_canonica`) y las reglas se escriben una sola vez.
- Rasgos de forma en coordenadas de mundo 3D; contactos (pulgar-índice) en la imagen.
- Ventanas de trazo en SEGUNDOS y hold-to-type con `time.monotonic()`: nada depende de los FPS.
- `Traductor.paso()` se usa igual en vivo y al reproducir grabaciones: toda la lógica nueva va ahí o debajo, no en `main.py`.
- Solo el hilo de captura toca el texto: el servidor pasa las acciones por una cola. El video no lleva textos dibujados; las etiquetas son HTML.
- Cada umbral está junto a su comentario con el rango medido. No mover umbrales a ciegas: medir con sesión guiada (grabar solo landmarks), separar calibración de prueba y congelar el código antes de medir.
- Alfabeto: lámina oficial SEP. Cifras y decisiones: `DOCUMENTACION.md` secciones 8 y 10.

## Estilo de código
- Simple y modular; sin código muerto ni abstracciones de más. Nombres en español, como el código existente.
- Editar solo las líneas necesarias; igualar densidad de comentarios y estilo del entorno.
- Sin dependencias nuevas salvo necesidad real.

## Comunicación
- Responder en español, directo, sin relleno. Reportar resultados con cifras reales y decir qué falló.
- Git: commit/push solo si se pide. Documentación sin emojis en títulos. Leer un archivo del usuario antes de sobrescribirlo (los edita a mano).
