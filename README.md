# Proyecto Lenguaje de señas mexicanas

## Sistema Inteligente de Traducción de Lengua de Señas Mexicana (LSM): dos manos, rostro y voz

Proyecto de la materia Inteligencia Artificial. Reconoce en tiempo real, con una cámara web, el **alfabeto de la Lengua de Señas Mexicana** hecho con **cualquiera de las dos manos o con ambas a la vez**. Lo convierte en texto, lo **dice en voz alta** y lo muestra en una **interfaz web** con tema claro y oscuro y un **Modo Práctica** para aprender las señas.

| Capa | Tecnología |
|---|---|
| Visión e inferencia | MediaPipe Hands y Face Mesh, OpenCV (`solvePnP`), NumPy |
| Voz | `pyttsx3` (SAPI5 de Windows) en un hilo propio |
| Servidor | FastAPI + uvicorn, WebSocket local |
| Interfaz | HTML, CSS y JavaScript sin dependencias |

## Características

- **Letras estáticas:** se escriben **sosteniendo la mano quieta 1.5 s**; cada mano tiene su propio progreso.
- **Letras con movimiento (J, K, Ñ, Q, X, Z):** se registran al completar el trazo.
- **Dos manos:** cada una tiene su propio filtro, trayectoria y progreso, y su etiqueta sobre el video (`Izq · B`, `Der · O`). La misma letra hecha con las dos se escribe una sola vez.
- **Geometría 3D:** ángulos de las articulaciones, normal de la palma y un marco propio de cada mano. La mano izquierda se refleja para que las mismas reglas valgan para ambas.
- **Rostro:** pose real de la cabeza en grados (`solvePnP`). Las cejas levantadas agregan `?`, negar con la cabeza borra y asentir confirma la palabra.
- **Voz:** dice cada letra al registrarla y lee la palabra completa al confirmarla o tras 4 s sin escribir.
- **Interfaz web:**
  - anillo de progreso que cambia de color (azul → ámbar → verde);
  - subtítulos centrados bajo el video;
  - guía del alfabeto;
  - tema claro/oscuro que se recuerda.
- **Modo Práctica:** eliges una letra o una palabra, ves la tarjeta de referencia y recibes retroalimentación en vivo. Al acertar, la tarjeta gira, sale confeti y el borde de la cámara se ilumina; se lleva un marcador de aciertos y rachas.

Para repetir la misma letra, cambia de seña o baja la mano y vuelve a hacerla.

## Inicio rápido

Requisitos: **Windows 10/11**, **Python 3.12** (MediaPipe 0.10.18 no es compatible con 3.13), una cámara web y un navegador reciente (Chrome, Edge o Firefox).

```powershell
# 1. Clonar el repositorio
git clone https://github.com/TheCubick/Proyecto-Lenguaje-de-se-a.git
cd Proyecto-Lenguaje-de-se-a

# 2. Crear el entorno virtual
py -3.12 -m venv venv

# 3. Instalar dependencias
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install opencv-python mediapipe==0.10.18 numpy pyttsx3 fastapi uvicorn websockets

# 4. Ejecutar
.\venv\Scripts\python.exe python\server.py
```

Al arrancar, la terminal muestra `Traductor LSM en http://localhost:5000` y el **navegador predeterminado se abre solo** en esa dirección; si no se abre, escríbela a mano. El navegador no pide permiso de cámara: el video lo transmite el propio programa. Para terminar, pulsa **Detener** dos veces en la interfaz o **Ctrl+C** en la terminal.

**Con VS Code:** abre la carpeta y pulsa **F5** ("Traductor LSM (python/server.py)"). `.vscode/launch.json` ya usa el Python de `venv`.

> La voz usa las voces SAPI5 de Windows (por ejemplo *Microsoft Sabina*, español de México). Si no hay ninguna en español, el programa funciona igual y el encabezado muestra **Sin voz**.

## Arquitectura

```
 Cámara ─▶ HILO DE CÁMARA (camara.py): solo el fotograma más reciente
               │                                     SERVIDOR (server.py)              NAVEGADOR (static/)
               │                                     FastAPI · 127.0.0.1:5000
               ├──────── sin esperar a MediaPipe ──▶ /video_feed: MJPEG (~16 KB) ───▶  <img> en espejo
               ▼
           HILO DE PROCESO (main.py)
           MediaPipe manos + rostro
           Traductor.paso() → texto, voz
           estado + coordenadas → JSON ────────────▶ /ws: JSON (~0.5-1.6 KB) ──────▶  <canvas>: esqueleto encima
                                       ◀── cola ───  {"accion": ...}         ◀──────  anillo, manos, rostro,
                                                                                      subtítulos, Práctica
```

- **Python es el único dueño de la cámara:** en Windows el navegador no siempre puede abrir una cámara que ya usa otro programa, así que el video lo transmite el mismo servidor.
- **Video sin esperar al reconocimiento:** cada fotograma nuevo se comprime en ~1 ms y sale por `/video_feed` al instante. El estado y las coordenadas 3D de las manos llegan aparte por WebSocket, y el esqueleto se dibuja en un `<canvas>` sobre el video con los colores del tema. Como ambos salen de la misma cámara, el esqueleto siempre coincide con la imagen.
- **Siempre lo más reciente:** un hilo lee la cámara y el proceso toma solo el último fotograma. El servidor y el dibujo también usan siempre el último estado, así que nunca se acumula retraso.
- **El backend hace todo el reconocimiento.** El navegador presenta el estado y manda acciones (Espacio, Borrar, Limpiar, Diagnóstico, Recalibrar, Detener).
- **Latencia a la vista:** junto a los FPS aparece la latencia desde la captura hasta el dibujo; al pasar el cursor se ve el desglose.
- **Privado:** el servidor solo escucha en `127.0.0.1` y no se guarda ningún fotograma.
- **Desempeño medido sin manos a la vista:**
  - video a 30 FPS (~16 KB por fotograma);
  - estado a 30 por segundo, 0.5 KB cada uno (1.6 KB con dos manos);
  - 16 ms desde la captura hasta que el estado llega al cliente.

El protocolo, el modelo de hilos y las decisiones de diseño están en la [documentación técnica](docs/DOCUMENTACION.md#25-arquitectura-cliente-servidor).

## Interfaz

| Zona | Contenido |
|---|---|
| **Encabezado** | Estado de la voz, interruptor de tema (sol/luna), guía del alfabeto, botón Detener |
| **Cámara** | Video del motor en espejo con el esqueleto de cada mano dibujado encima (azul la izquierda, verde la derecha, verde de éxito al registrar), etiqueta flotante por mano, estela del dedo al trazar, FPS y latencia, y barra de herramientas |
| **Subtítulos** | Texto acumulado centrado bajo el video; la última letra entra con una animación breve |
| **Traducir** | La seña en grande dentro del anillo de progreso, qué mano la hace y cómo se hace |
| **Práctica** | Selector de letras o palabras, tarjeta de referencia, retroalimentación en vivo y marcador |
| **Manos y rostro** | Tarjeta por mano; estado facial con icono, medidor de cejas y ángulos de la cabeza |

El anillo de progreso cambia de color según cuánto falta para registrar la seña:

| Progreso | Color |
|---|---|
| 0-49 % | Azul claro |
| 50-99 % | Ámbar ("procesando") |
| 100 % | Verde, con pulso y brillo |

## Modo Práctica

1. Abre la pestaña **Práctica** del panel derecho. También puedes tocar una letra en la guía (`G`).
2. Elige **Letras**: cualquiera de las 30 señas, con flechas o al azar. O **Deletreo**: una palabra sugerida o la que escribas, letra por letra. O **Frases**: una palabra o frase con movimiento del dataset (*gracias*, *mi nombre es*, *sordo*…).
3. Haz la seña frente a la cámara. La retroalimentación te dice si es la correcta, si ya tienes la base de una seña con movimiento (*"Bien, I es la base: ahora haz el movimiento"*) o qué seña está detectando.
4. Al registrarse la seña correcta, la tarjeta gira, sale confeti y suben los aciertos y la racha. Si registras otra letra, la racha vuelve a cero.

**Frases (tarjeta "Dinámica").** Empieza con las manos abajo, haz el movimiento completo y continuo, y bájalas: al bajarlas, `gestos.py` compara el movimiento (remuestreado a 30 fotogramas, como el dataset) contra las ejecuciones extraídas de los videos y te dice si coincide o a qué frase se pareció. Es un comparador por plantillas (DTW) mientras se entrena la LSTM; la lista de frases sale de las carpetas de `dataset/`.

**Muestras propias.** Después de hacer la frase (cuando el cuadro de diagnóstico muestre la evaluación), pulsa **Guardar como muestra** o la tecla **G**: tu movimiento se guarda como `datos/dataset/<frase>/pablo_seq_XX.npy` y se usa como referencia desde ese momento. Con 3-4 repeticiones propias por frase el comparador reconoce tus manos y tu ritmo. En Práctica > Frases la tecla G guarda; en los demás modos abre la guía.

> Lo que aciertas en práctica también se escribe en los subtítulos; usa **Limpiar** al terminar.

## Teclado

| Tecla | Acción |
|---|---|
| `Espacio` | Agregar espacio |
| `Backspace` | Borrar la última letra |
| `C` | Limpiar todo el texto |
| `D` | Mostrar u ocultar el diagnóstico (estado 3D de cada dedo y ángulos de la cabeza) |
| `N` (o `R`) | Recalibrar la cara neutra |
| `G` | Abrir o cerrar la guía del alfabeto (`Esc` la cierra) |
| `T` | Cambiar entre tema claro y oscuro |

Los atajos funcionan con la pestaña del navegador activa; no se activan mientras escribes una palabra de práctica.

## Alfabeto (según la lámina oficial de la SEP)

| Seña | Cómo hacerla |
|---|---|
| **A** | Puño con el pulgar estirado al lado |
| **B** | 4 dedos juntos arriba, pulgar sobre la palma |
| **C** | Mano curva en forma de C |
| **D** | Índice arriba, los demás dedos cerrados |
| **E** | Dedos doblados sobre el pulgar |
| **F** | Índice tocando el pulgar y los otros 3 dedos arriba |
| **G** | "Pistola": índice apuntando de lado, pulgar arriba |
| **H** | "Pistola" con índice y medio de lado |
| **I** | Solo el meñique arriba |
| **J** | La I, y el meñique baja haciendo un gancho (*con movimiento*) |
| **K** | Índice y medio juntos o cruzados, agitando la mano de lado a lado (*con movimiento*) |
| **L** | Índice arriba y pulgar al lado |
| **M** | 3 dedos hacia abajo |
| **N** | 2 dedos hacia abajo |
| **Ñ** | La N agitada de lado a lado (*con movimiento*) |
| **O** | Círculo con el índice y el pulgar |
| **P** | Índice y medio cruzados en "X" |
| **Q** | Pinza o puño colgando hacia abajo, balanceándolo (*con movimiento*) |
| **R** | Índice y medio entrelazados |
| **S** | Puño con el pulgar sobre los dedos |
| **T** | Puño con el pulgar entre los dedos |
| **U** | Índice y medio juntos arriba |
| **V** | Índice y medio en V |
| **W** | Índice, medio y anular abiertos |
| **X** | Índice de lado con la palma inclinada, moviéndolo (*con movimiento*) |
| **Y** | Pulgar y meñique estirados |
| **Z** | La D dibujando una Z en el aire (*con movimiento*) |
| **ESPACIO** | Mano abierta con los 5 dedos separados y el pulgar bien abierto |
| **BORRAR** | Mano abierta que **barre rápido hacia el centro del cuerpo** (a la izquierda con la mano derecha, a la derecha con la izquierda) |
| **?** | Índice dibujando un signo de interrogación en el aire (respaldo de las cejas) |

> **Con la mano izquierda los movimientos van en espejo** (el gancho de la J, el barrido de BORRAR). La J, la Z y el `?` también se aceptan dibujados como letra escrita.
>
> **Los trazos deben ser pequeños, del tamaño de la mano**, y sin salirse del cuadro de la cámara.
>
> **CH, LL y RR** no están incluidas.

## Dos manos

- Ten las manos **a la altura del pecho, separadas y sin taparte la cara**: con las manos juntas delante de la cara la cámara puede dejar de verlas.
- Cada mano escribe por su cuenta. Si las dos completan letras distintas se escriben ambas; **la misma letra con las dos manos se escribe una vez**.
- Una mano relajada a la vista no escribe.

## Gestos del rostro (MediaPipe Face Mesh + `cv2.solvePnP`)

Al iniciar, mantén la cara neutra un momento mientras el sistema calibra (la tarjeta *Rostro* muestra **Calibrando**); la tecla **`N`** vuelve a calibrar. La tarjeta incluye un **medidor de cejas** con la marca del umbral y los **ángulos de la cabeza** (cabeceo, giro e inclinación, en grados).

| Gesto | Efecto |
|---|---|
| **Cejas levantadas 1.5 s** | Agrega `?` al final de la palabra |
| **Negar con la cabeza** (izquierda-derecha-izquierda, ≥ 8°) | Borra la última letra |
| **Asentir con la cabeza** (tres cabeceos, ≥ 5°) | Confirma la palabra (agrega un espacio) y la lee en voz alta |

Si una mano tapa la cara, la tarjeta muestra **Cara tapada** y no se interpreta ningún gesto facial.

## Voz

Un hilo aparte (`pyttsx3`, con una voz en español) habla **sin congelar la cámara**. Dice el nombre de cada letra (`jota`, `eñe`, `doble ve`…) y lee la palabra completa al asentir o tras 4 s sin escribir.

## Qué tan bien funciona

Se calibró y se probó con **nueve sesiones guiadas** grabadas con la cámara real (solo coordenadas de manos y rostro, sin imágenes), de **una sola persona** y con ambas manos. Las sesiones de **prueba** se midieron primero con el código congelado, antes de verlas (**independiente**); después se ajustaron algunas reglas con esos datos (**versión final**, ya no independiente).

| Parte | Independiente | Versión final |
|---|---|---|
| Letras estáticas, mano derecha (exactitud por fotograma) | 94.4 % y 85.6 % (dos pruebas) | 96.0 % y 93.0 % |
| Letras estáticas, mano izquierda | 79.8 % y 96.2 % | 91.6 % y 98.4 % |
| Letras escritas de punta a punta | 35 de 44 y 41 de 44 | 40 de 44 y **43 de 44** |
| Versión anterior (2D) en las mismas pruebas | 80.3 % y 79.6 % | — |
| Dos manos con letras distintas a la vez | — | **15 de 15** escritas, cada una por su mano |
| Trazos con movimiento (fases reconocidas) | 11 de 16 | 13 de 16 (calibración: 16 de 16) |
| Rostro, eventos falsos en 19.4 min con cara visible | — | **11** (la versión anterior: 100), con los mismos aciertos |
| Misma mano reflejada como la otra (espejo) | — | **100 %** igual (8 336 fotogramas) |
| Velocidad del reconocimiento | — | 25-26 FPS conectada; 17 FPS en batería con dos manos |
| Transmisión a la interfaz web (sin manos a la vista) | — | Video 30 FPS (~16 KB); estado 30/s, 0.5 KB (1.6 KB con dos manos), 16 ms captura → cliente |

**Limitaciones:**
- La **P y la R** dependen de cómo cruces los dedos y en las pruebas se confundieron.
- Aparecen algunas preguntas falsas con las cejas mientras haces letras, al mirar la mano o la pantalla.
- Con dos manos y la laptop en batería, el reconocimiento baja a ~17 FPS.
- El servidor es solo local.
- El esqueleto va unos milisegundos detrás del video, porque espera al reconocimiento.
- La práctica solo guarda la mejor racha.

Detalles, cifras y la metodología completa en la [documentación técnica](docs/DOCUMENTACION.md).

## Minería de datos desde videos (para entrenar la LSTM)

`video_extractor.py` es una herramienta independiente del traductor. Toma los cortes de `dataset_config.json` (qué palabra hay en qué segundos de qué video de `datos/videos/`), extrae los landmarks 3D de las manos con el mismo pipeline que el modo en vivo y guarda cada seña como una secuencia de exactamente 30 fotogramas en `datos/dataset/<palabra>/seq_XX.npy`.

```json
[
  {"video": "Tutorial Dilo en señas - Frutas.mp4", "palabra": "manzana", "inicio": "00:04.8", "fin": "00:09.1"},
  {"video": "Tutorial Dilo en señas - Frutas.mp4", "palabra": "plátano", "inicio": "00:09.1", "fin": "00:13.2"}
]
```

```powershell
.\venv\Scripts\python.exe python\video_extractor.py --revisar   # valida tiempos y nombres sin correr MediaPipe
.\venv\Scripts\python.exe python\video_extractor.py             # extrae solo los cortes nuevos o corregidos
.\venv\Scripts\python.exe python\video_extractor.py --listar    # muestras por palabra
```

- **Tiempos:** `"MM:SS"`, `"HH:MM:SS"` o con décimas (`"00:04.8"`). Conviene dejar una holgura de unas décimas: los fotogramas sin manos de los extremos se quitan solos.
- **Ejecuciones:** si dentro del corte el maestro hace la seña normal, baja las manos y la repite lento, cada ejecución se guarda como muestra aparte (`seq_01_1.npy` normal, `seq_01_2.npy` lenta). El corte se parte donde las manos quedan abajo o fuera de cuadro al menos 0.4 s.
- **Normalización:** cada ejecución se remuestrea a 30 fotogramas por interpolación lineal (una de 40 se comprime, una de 20 se estira), así cada muestra va del inicio al final de la seña sin padding. Los huecos cortos donde MediaPipe pierde una mano se rellenan interpolando.
- **Notas a mano:** `--importar "datos/videos/minutos exactos de videos.txt"` convierte líneas como `sordo: 11 s - 17 s` o `Mi nombre es: 1:08 - 1:21` (debajo de una línea con el nombre del video) en cortes de `dataset_config.json` y luego procesa.
- **Videos terminados:** cuando todos los cortes de un video se extraen sin fallas, el video se mueve a `datos/videos/listo/` (desactivar con `--no-mover`); el extractor los sigue encontrando ahí.
- **Vector por fotograma (254 números):** presencia de cada mano, y por mano 21 puntos en la imagen (movimiento) más 21 puntos 3D de mundo (forma). Solo vectores, ningún píxel.
- **Incremental:** `datos/dataset/indice.json` recuerda de qué corte salió cada archivo. Al volver a correr solo se procesa lo nuevo o lo cambiado, y se borran las muestras de cortes que ya no están en la configuración.
- **CSV:** también acepta `--config cortes.csv` con columnas `video,palabra,inicio,fin` (Excel con `;` también).
- **Videos ya espejados** (grabados como selfie): agrega `"espejo": false` a esos cortes.
- Para entrenar: `X, y = video_extractor.cargar_dataset()` da `X` de forma (N, 30, 254) y las palabras en `y`. Más opciones: `--help`.

### Aumento de datos

`data_augmentation.py` multiplica las muestras con variantes sintéticas. Cada variante combina ruido leve en las coordenadas (~2 % del tamaño de la mano), ritmo variable por tramos (entre 25/30 y 35/30 de la velocidad original, regresando a 30 fotogramas) y escala (distancia a la cámara ±15 %, tamaño de mano ±10 %). Las manos ausentes siguen en cero.

```python
from video_extractor import cargar_dataset
from data_augmentation import aumentar
X, y = cargar_dataset()
X5, y5, origen = aumentar(X, y, factor=5)   # o factor=10
```

Separa entrenamiento y prueba **antes** de aumentar (o agrupa por `origen`): si variantes de una misma muestra quedan en ambos lados, la exactitud de prueba sale inflada. Para revisar o subir a Colab: `.\venv\Scripts\python.exe python\data_augmentation.py --factor 10` escribe `datos/dataset_aumentado.npz`.

### Auto-etiquetado por voz (opcional)

`auto_etiquetador.py` propone los cortes de `dataset_config.json` en los videos donde el maestro dice el nombre de la seña. Transcribe el audio con Whisper (`faster-whisper`, modelo `small`, en la CPU), toma las palabras dichas sueltas entre pausas y crea un corte que va de 0.5 s antes de cada palabra hasta 4 s después, sin llegar a la siguiente. Los videos sin voz utilizable quedan en `datos/requiere_anotacion_manual.txt` con el motivo.

```powershell
.\venv\Scripts\python.exe -m pip install faster-whisper            # una vez (~120 MB; el modelo, ~480 MB, se baja en la primera corrida)
.\venv\Scripts\python.exe python\auto_etiquetador.py --simular            # ver qué propondría, sin escribir
.\venv\Scripts\python.exe python\auto_etiquetador.py --vocabulario vocabulario.txt   # recomendado: solo esas palabras
```

- Solo procesa videos que aún no están en `dataset_config.json` ni en la lista manual; los cortes hechos a mano nunca se tocan. `--rehacer` vuelve a etiquetar lo que Whisper propuso antes.
- Cada corte lleva `"origen": "whisper"` y `"oido"` (lo que se escuchó). **Revísalos antes de extraer**: en la prueba con los rótulos de "Tutorial Dilo en señas", los 9 cortes contenían la seña, pero Whisper oyó "fresa" donde se ve CEREZA.
- `vocabulario.txt` (una palabra o frase por línea) descarta muletillas y letras sueltas, y corrige la ortografía ("durazmo" → "durazno").
- Antes de escribir se respalda el archivo en `datos/dataset_config.json.bak`.

## Calibración

Los umbrales están agrupados al inicio de cada módulo (`python/classifier_hands.py` para las manos, `python/classifier_face.py` para el rostro), cada uno con el rango medido que lo justifica. Si una seña cuesta trabajo, activa el diagnóstico con la tecla **D**. Verás sobre el video, por cada mano:

- el estado de cada dedo (`EXT` estirado, `DOB` doblado, `CER` cerrado) con su extensión y flexión;
- la posición del pulgar y la normal de la palma;
- los ángulos de la cabeza.

Con eso puedes ajustar el umbral correspondiente.

## Estructura

```
Proyecto/
├── python/                  # Todo el código Python
│   ├── server.py            # Punto de entrada: FastAPI, WebSocket /ws, video MJPEG, abre el navegador
│   ├── main.py              # Motor: bucle de proceso en su propio hilo
│   ├── camara.py            # Captura en un hilo propio (fotograma más reciente) y JPEG del video
│   ├── classifier_hands.py  # Manos: rasgos 3D, alfabeto y trazos
│   ├── classifier_face.py   # Rostro: solvePnP, cejas, negar y asentir
│   ├── traductor.py         # Hold-to-type por mano y orquestación de un fotograma
│   ├── gestos.py            # Frases con movimiento: DTW contra el dataset y muestras propias (tecla G)
│   ├── voice_engine.py      # Voz (SAPI de Windows) y puente al hardware
│   ├── ui_panel.py          # Estado JSON con las coordenadas de las manos
│   ├── video_extractor.py   # Herramienta: dataset para la LSTM desde videos
│   ├── auto_etiquetador.py  # Herramienta: propone cortes a partir de la voz de los videos (Whisper)
│   └── data_augmentation.py # Herramienta: variantes sintéticas (ruido, ritmo, escala) para entrenar
├── static/                  # Interfaz web (la sirve python/server.py)
│   ├── index.html           # Estructura
│   ├── estilos.css          # Temas claro/oscuro, maquetación y animaciones
│   └── app.js               # Cliente WebSocket, video MJPEG, esqueleto, tema y Modo Práctica
├── datos/
│   ├── dataset_config.json  # Cortes de los videos: video, palabra, inicio, fin (se versiona)
│   ├── videos/              # Videos de entrada y notas de tiempos; los terminados pasan a videos/listo/ (no se suben)
│   ├── dataset/             # Muestras de 30 fotogramas por palabra: seq_XX_N.npy y pablo_seq_XX.npy (no se sube)
│   └── gestos_vivo.csv      # Cada evaluación de frases en vivo, para calibrar umbrales (no se sube)
├── docs/
│   ├── DOCUMENTACION.md     # Documentación técnica completa
│   └── errores_detectado.txt # Pendientes anotados a mano
├── .vscode/                 # F5 ejecuta python/server.py con el Python de venv
├── venv/                    # Entorno virtual (no se sube a Git)
├── .gitignore
├── CLAUDE.md                # Mapa compacto para el asistente de código
└── README.md
```

Los comandos se corren desde la carpeta `Proyecto/` (`.\venv\Scripts\python.exe python\<archivo>.py`). Las rutas a `datos/` y `static/` salen de la ubicación del código, no de la carpeta actual.
