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
.\venv\Scripts\python.exe server.py
```

Al arrancar, la terminal muestra `Traductor LSM en http://localhost:5000` y el **navegador predeterminado se abre solo** en esa dirección. Si no se abre, escríbela a mano. Para terminar, pulsa **Detener** dos veces en la interfaz o **Ctrl+C** en la terminal.

**Con VS Code:** abre la carpeta y pulsa **F5** ("Traductor LSM (server.py)"). `.vscode/launch.json` ya usa el Python de `venv`.

> La voz usa las voces SAPI5 de Windows (por ejemplo *Microsoft Sabina*, español de México). Si no hay ninguna en español, el programa funciona igual y el encabezado muestra **Sin voz**.

## Arquitectura

```
 Cámara ─▶ HILO DE CAPTURA (main.py)                  SERVIDOR (server.py)              NAVEGADOR (static/)
           MediaPipe manos + rostro                   FastAPI · 127.0.0.1:5000
           Traductor.paso() → texto                   WebSocket /ws
           voz · hardware              ──Difusor──▶   binario: JPEG          ──────▶   video en <canvas>
           esqueleto → JPEG                           texto: JSON del estado ──────▶   anillo, manos, rostro,
           estado → JSON               ◀──cola────    {"accion": ...}        ◀──────   subtítulos, Práctica
```

- **El backend hace todo el reconocimiento.** El navegador solo presenta el estado y manda acciones (Espacio, Borrar, Limpiar, Diagnóstico, Recalibrar, Detener).
- **Latencia acotada:** el servidor guarda solo el último fotograma; si el navegador va lento, se salta fotogramas en lugar de acumular retraso.
- **Privado:** el servidor solo escucha en `127.0.0.1` y no guarda ningún fotograma.
- **Desempeño medido:** 30 FPS recibidos en el navegador, unos 11 KB por fotograma.

El protocolo, el modelo de hilos y las decisiones de diseño están en la [documentación técnica](DOCUMENTACION.md#25-arquitectura-cliente-servidor).

## Interfaz

| Zona | Contenido |
|---|---|
| **Encabezado** | Estado de la voz, interruptor de tema (sol/luna), guía del alfabeto, botón Detener |
| **Cámara** | Video en espejo con el esqueleto de cada mano (azul la izquierda, verde la derecha), etiqueta flotante por mano, estela del dedo al trazar y barra de herramientas |
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
2. Elige **Letras**: cualquiera de las 30 señas, con flechas o al azar. O elige **Palabras**: una sugerida o la que escribas, para deletrearla.
3. Haz la seña frente a la cámara. La retroalimentación te dice si es la correcta, si ya tienes la base de una seña con movimiento (*"Bien, I es la base: ahora haz el movimiento"*) o qué seña está detectando.
4. Al registrarse la seña correcta, la tarjeta gira, sale confeti y suben los aciertos y la racha. Si registras otra letra, la racha vuelve a cero.

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
| Transmisión a la interfaz web | — | 30 FPS recibidos, ~11 KB por fotograma |

**Limitaciones:**
- La **P y la R** dependen de cómo cruces los dedos y en las pruebas se confundieron.
- Aparecen algunas preguntas falsas con las cejas mientras haces letras, al mirar la mano o la pantalla.
- Con dos manos y la laptop en batería, el reconocimiento baja a ~17 FPS.
- El servidor es solo local.
- La práctica solo guarda la mejor racha.

Detalles, cifras y la metodología completa en la [documentación técnica](DOCUMENTACION.md).

## Calibración

Los umbrales están agrupados al inicio de cada módulo (`classifier_hands.py` para las manos, `classifier_face.py` para el rostro), cada uno con el rango medido que lo justifica. Si una seña cuesta trabajo, activa el diagnóstico con la tecla **D**. Verás sobre el video, por cada mano:

- el estado de cada dedo (`EXT` estirado, `DOB` doblado, `CER` cerrado) con su extensión y flexión;
- la posición del pulgar y la normal de la palma;
- los ángulos de la cabeza.

Con eso puedes ajustar el umbral correspondiente.

## Estructura

```
Proyecto/
├── .vscode/            # Ejecutar con F5 (server.py con el Python de venv)
├── venv/               # Entorno virtual (no se sube a Git)
├── .gitignore
├── server.py           # Servidor web: FastAPI, WebSocket /ws, abre el navegador
├── main.py             # Motor: bucle de captura en su propio hilo
├── classifier_hands.py # Manos: rasgos 3D, alfabeto y trazos
├── classifier_face.py  # Rostro: solvePnP, cejas, negar y asentir
├── traductor.py        # Hold-to-type por mano y orquestación
├── voice_engine.py     # Voz y puente al hardware
├── ui_panel.py         # Esqueleto de las manos sobre el video y estado JSON
├── static/
│   ├── index.html      # Interfaz
│   ├── estilos.css     # Temas claro/oscuro, maquetación y animaciones
│   └── app.js          # Cliente WebSocket, tema y Modo Práctica
├── CLAUDE.md           # Mapa compacto para el asistente de código
├── README.md
└── DOCUMENTACION.md    # Documentación técnica completa
```
