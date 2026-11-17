# Traductor de Lengua de Señas Mexicana (LSM)

<p align="center">
  <img src="https://cdn.simpleicons.org/python" alt="Python" title="Python" width="44" height="44">
  <img src="https://cdn.simpleicons.org/mediapipe" alt="MediaPipe" title="MediaPipe" width="44" height="44">
  <img src="https://cdn.simpleicons.org/opencv" alt="OpenCV" title="OpenCV" width="44" height="44">
  <img src="https://cdn.simpleicons.org/numpy" alt="NumPy" title="NumPy" width="44" height="44">
  <img src="https://cdn.simpleicons.org/fastapi" alt="FastAPI" title="FastAPI" width="44" height="44">
  <img src="https://cdn.simpleicons.org/javascript" alt="JavaScript" title="JavaScript" width="44" height="44">
  <img src="https://cdn.simpleicons.org/html5" alt="HTML5" title="HTML5" width="44" height="44">
  <img src="https://cdn.simpleicons.org/css" alt="CSS" title="CSS" width="44" height="44">
  <img src="https://cdn.simpleicons.org/pwa" alt="PWA" title="PWA" width="44" height="44">
  <img src="https://cdn.simpleicons.org/nodedotjs" alt="Node.js" title="Node.js" width="44" height="44">
  <picture><source media="(prefers-color-scheme: dark)" srcset="https://cdn.simpleicons.org/vercel/white"><img src="https://cdn.simpleicons.org/vercel" alt="Vercel" title="Vercel" width="44" height="44"></picture>
  <img src="https://cdn.simpleicons.org/git" alt="Git" title="Git" width="44" height="44">
  <picture><source media="(prefers-color-scheme: dark)" srcset="https://cdn.simpleicons.org/github/white"><img src="https://cdn.simpleicons.org/github" alt="GitHub" title="GitHub" width="44" height="44"></picture>
</p>

<p align="center">
  <img src="docs/img/gatito.gif" alt="Gatito bailando" width="160">
</p>

Proyecto de la materia Inteligencia Artificial. Con una cámara web reconoce en tiempo real el **alfabeto de la LSM** (con cualquiera de las dos manos o con ambas), **frases con movimiento** y **gestos del rostro**. Lo convierte en texto, lo **dice en voz alta** y lo muestra en una **app web** instalable (PWA) con tema claro y oscuro y un **Modo Práctica** para aprender las señas.

Funciona de dos maneras con la misma interfaz:
- **Local:** un servidor en Python hace el reconocimiento y transmite el video.
- **En el navegador (Vercel):** el motor está portado a JavaScript y corre en el dispositivo de quien lo abre. El video no sale de ahí.

## Qué hace

| Función | Descripción |
|---|---|
| Alfabeto | Letras estáticas (se sostienen 1.5 s) y letras con movimiento (J, K, Ñ, Q, X, Z) con la lámina oficial de la SEP como referencia |
| Dos manos | Cada mano tiene su propio filtro, trayectoria y progreso; la izquierda se refleja para usar las mismas reglas |
| Frases | Reconoce palabras y frases con movimiento (*gracias*, *mi nombre es*, *sordo*…) comparando contra muestras sacadas de videos |
| Rostro | Cejas levantadas agregan `?`, negar con la cabeza borra y asentir confirma la palabra |
| Voz | Dice cada letra y lee la palabra o la frase completa |
| Modo Práctica | Eliges una letra, palabra o frase, ves la demostración y recibes retroalimentación en vivo, con aciertos y rachas |

## Herramientas utilizadas

| Logo | Herramienta | Para qué se usa |
|:---:|---|---|
| <img src="https://cdn.simpleicons.org/python" alt="Python" title="Python" width="28" height="28"> | **Python 3.12** | Motor de reconocimiento y servidor local |
| <img src="https://cdn.simpleicons.org/mediapipe" alt="MediaPipe" title="MediaPipe" width="28" height="28"> | **MediaPipe Hands y Face Mesh 0.10.18** | Detecta 21 puntos 3D por mano y 478 del rostro en cada fotograma |
| <img src="https://cdn.simpleicons.org/opencv" alt="OpenCV" title="OpenCV" width="28" height="28"> | **OpenCV** | Captura de cámara, JPEG del video y `solvePnP` para la pose real de la cabeza en grados |
| <img src="https://cdn.simpleicons.org/numpy" alt="NumPy" title="NumPy" width="28" height="28"> | **NumPy** | Geometría 3D, remuestreo de secuencias y DTW vectorizado |
|  | **DTW (Dynamic Time Warping)** | Compara un movimiento completo contra las muestras del dataset para reconocer frases |
| <img src="https://cdn.simpleicons.org/fastapi" alt="FastAPI" title="FastAPI" width="28" height="28"> | **FastAPI + uvicorn + WebSocket** | Servidor local: sirve la interfaz y manda el estado por fotograma como JSON; el video sale por MJPEG |
|  | **SAPI de Windows (pywin32) / pyttsx3** | Voz en español en un hilo propio, sin congelar la cámara |
| <img src="https://cdn.simpleicons.org/html5" alt="HTML5" title="HTML5" width="28" height="28"> <img src="https://cdn.simpleicons.org/css" alt="CSS" title="CSS" width="28" height="28"> <img src="https://cdn.simpleicons.org/javascript" alt="JavaScript" title="JavaScript" width="28" height="28"> | **HTML, CSS y JavaScript** | Interfaz sin dependencias: esqueleto de las manos en `<canvas>`, temas, Modo Práctica |
| <img src="https://cdn.simpleicons.org/pwa" alt="PWA" title="PWA" width="28" height="28"> | **PWA (manifest + metas móviles)** | La interfaz se instala como app y se adapta a celular (App Shell de pantalla completa) |
| <img src="https://cdn.simpleicons.org/mediapipe" alt="MediaPipe" title="MediaPipe" width="28" height="28"> <img src="https://cdn.simpleicons.org/webassembly" alt="WebAssembly" title="WebAssembly" width="28" height="28"> | **MediaPipe Tasks JS + `getUserMedia`** | El mismo reconocimiento dentro del navegador, sin Python |
|  | **Web Speech API** | Voz en la versión del navegador |
|  | **Levenberg-Marquardt propio** | `solvePnP` reescrito en JavaScript, porque en el navegador no hay OpenCV |
| <img src="https://cdn.simpleicons.org/nodedotjs" alt="Node.js" title="Node.js" width="28" height="28"> | **Node.js** | Pruebas de paridad entre el motor de Python y el de JavaScript |
| <picture><source media="(prefers-color-scheme: dark)" srcset="https://cdn.simpleicons.org/vercel/white"><img src="https://cdn.simpleicons.org/vercel" alt="Vercel" title="Vercel" width="28" height="28"></picture> | **Vercel** | Publicación de la versión web (`vercel.json` y `.vercelignore`) |
| <img src="https://cdn.simpleicons.org/git" alt="Git" title="Git" width="28" height="28"> <picture><source media="(prefers-color-scheme: dark)" srcset="https://cdn.simpleicons.org/github/white"><img src="https://cdn.simpleicons.org/github" alt="GitHub" title="GitHub" width="28" height="28"></picture> | **Git y GitHub** | Control de versiones y despliegue continuo |
|  | **faster-whisper (opcional)** | Proponer cortes automáticos en los videos del dataset a partir de la voz del maestro |

## Cómo funciona

1. **Captura:** un hilo lee la cámara y guarda solo el último fotograma, así nunca se acumula retraso. La imagen se espeja antes de MediaPipe, por lo que la etiqueta Izquierda/Derecha es la mano real.
2. **Detección:** MediaPipe da los puntos de las manos y del rostro. Un seguidor mantiene la identidad de cada mano entre fotogramas y un filtro One Euro suaviza el temblor.
3. **Rasgos 3D:** de las manos salen ángulos de articulaciones, normal de la palma y un marco propio; del rostro, la pose de la cabeza y las cejas compensadas por esa pose.
4. **Decisión:** reglas medidas sobre datos reales clasifican las letras estáticas. Los trazos (J, Z, `?`…) usan la trayectoria del dedo, y las frases se comparan por DTW contra el dataset al cerrar el gesto.
5. **Salida:** el texto se escribe al sostener la seña, se dice en voz alta y la interfaz recibe un JSON por fotograma para dibujar el esqueleto, el anillo de progreso y los subtítulos.

El dataset de frases sale de videos de YouTube: `video_extractor.py` corta cada seña, extrae solo vectores de landmarks (ningún píxel) y la remuestrea a 30 fotogramas. Detalles en [docs/DATASET.md](docs/DATASET.md).

## Qué tan bien funciona

Calibrado y probado con **nueve sesiones guiadas** grabadas con la cámara real, de **una sola persona**. Las sesiones de prueba se midieron primero con el código congelado (**independiente**); después se ajustaron algunas reglas con esos datos (**versión final**).

| Parte | Independiente | Versión final |
|---|---|---|
| Letras estáticas, mano derecha (por fotograma) | 94.4 % y 85.6 % | 96.0 % y 93.0 % |
| Letras estáticas, mano izquierda | 79.8 % y 96.2 % | 91.6 % y 98.4 % |
| Letras escritas de punta a punta | 35 de 44 y 41 de 44 | 40 de 44 y **43 de 44** |
| Dos manos con letras distintas a la vez | — | **15 de 15** |
| Rostro, eventos falsos en 19.4 min | — | **11** (versión anterior: 100) |
| Frases del dataset en repetición de video | — | 46 aceptadas, 0 falsas |
| Motor de JavaScript contra Python | — | **0 diferencias** en 3 976 fotogramas |
| Velocidad | — | 25-26 FPS conectada; 17 FPS en batería con dos manos |

**Limitaciones:**
- La **P y la R** dependen de cómo cruces los dedos y en las pruebas se confundieron.
- Hay preguntas falsas con las cejas mientras haces letras, al mirar la mano o la pantalla.
- Todo se calibró con una sola persona.
- La versión del navegador aún no se midió con una cámara real; sus landmarks pueden diferir un poco de los de Python.
- En la versión publicada no hay videos de demostración (los videos de terceros no se suben) y las muestras propias se guardan solo en ese navegador.

## Estructura

```
Proyecto/
├── python/        # Motor de reconocimiento, servidor FastAPI y herramientas del dataset
├── static/        # Interfaz web (HTML, CSS, JS) y archivos de la PWA; es la misma para los dos motores
├── web/           # Motor portado a JavaScript (web/motor), plantillas (web/datos) y pruebas de paridad
├── datos/         # Cortes de los videos (dataset_config.json); videos y dataset no se suben
├── docs/          # Documentación técnica y del dataset
├── vercel.json    # Publicación en Vercel (solo static/ y web/)
└── CLAUDE.md      # Mapa compacto del código para el asistente de programación
```

## Más documentación

- [Documentación técnica](docs/DOCUMENTACION.md): arquitectura, alfabeto, decisiones de diseño, cifras y metodología completa.
- [Dataset y entrenamiento](docs/DATASET.md): extracción desde videos, aumento de datos, auto-etiquetado y calibración.
