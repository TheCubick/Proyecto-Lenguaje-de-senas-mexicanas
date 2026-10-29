# Proyecto-Lenguaje-de-se-a

## Sistema Inteligente de Traducción de Lengua de Señas Mexicana (LSM): dos manos, rostro y voz

Proyecto de la materia Inteligencia Artificial. Con una cámara web reconoce en tiempo real el **alfabeto de la Lengua de Señas Mexicana** hecho con **cualquiera de las dos manos, o con ambas a la vez**, lo convierte en texto y lo **dice en voz alta**. Usa MediaPipe (manos y rostro), OpenCV y `pyttsx3`.

- **Letras estáticas:** se escriben **sosteniendo la mano quieta 1.5 s**; cada mano tiene su propia barra de progreso.
- **Letras con movimiento (J, K, Ñ, Q, X, Z):** se registran al completar el trazo.
- **Dos manos:** la cámara etiqueta cada una (`Izq: B | Der: O`) y cada una tiene su propio filtro, trayectoria y barra; si haces la misma letra con las dos, se escribe una sola vez.
- **Geometría 3D:** ángulos de las articulaciones, normal de la palma y un marco propio de cada mano; la mano izquierda se refleja para que las mismas reglas valgan para ambas.
- **Rostro:** pose real de la cabeza en grados (`solvePnP`); las cejas levantadas agregan `?`, negar con la cabeza borra y asentir confirma la palabra.
- **Voz:** dice cada letra al registrarla y lee la palabra completa al confirmar o tras 4 s sin escribir.

Para repetir la misma letra, cambia de seña o baja la mano y vuelve a hacerla.

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

Al iniciar, mantén la cara neutra un momento mientras el sistema calibra (`Modo: Calibrando...`); la tecla **`N`** vuelve a calibrar. En la ventana de cámara hay un **medidor de cejas** (`CEJAS x1.11`) y debajo los **ángulos de la cabeza** (`P` pitch, `Y` yaw, `R` roll, en grados).

| Gesto | Efecto |
|---|---|
| **Cejas levantadas 1.5 s** | Agrega `?` al final de la palabra |
| **Negar con la cabeza** (izquierda-derecha-izquierda, ≥ 8°) | Borra la última letra |
| **Asentir con la cabeza** (tres cabeceos, ≥ 5°) | Confirma la palabra (agrega un espacio) y la lee en voz alta |

Si una mano tapa la cara, el indicador muestra `Cara tapada` y no se interpreta ningún gesto facial.

## Voz

Un hilo aparte (`pyttsx3`, con una voz en español) habla **sin congelar la cámara**. Dice el nombre de cada letra (`jota`, `eñe`, `doble ve`…) y lee la palabra completa al asentir o tras 4 s sin escribir. Si no hay voz en español instalada o falta `pyttsx3`, el programa funciona igual y el panel indica `Voz: no disponible`.

## Interfaz

- **Camara IA - Deteccion LSM:** video en espejo con cada mano (puntos azules la izquierda, verdes la derecha), su recuadro, la etiqueta `Izq: B` / `Der: O` con su mini barra, la estela del dedo al dibujar, el resumen `Izq: B | Der: O`, el medidor de cejas con la pose de la cabeza, el `Modo:` facial y la barra de la mano principal.
- **Panel Traductor LSM:** la seña en grande (con `MANO IZQ` o `MANO DER`), cómo se hace, la **guía del alfabeto** (en naranja las que llevan movimiento), el texto acumulado y el estado de la voz. Los espacios se ven como espacios.

## Teclas

| Tecla | Acción |
|---|---|
| `Espacio` | Agregar espacio |
| `Backspace` | Borrar la última letra |
| `C` | Limpiar todo el texto |
| `D` | Mostrar u ocultar el diagnóstico (estado 3D de cada dedo de cada mano y ángulos de la cabeza) |
| `N` (o `R`) | Recalibrar la cara neutra |
| `ESC` o la `X` de una ventana | Salir |

## Cómo ejecutar

1. Abre la carpeta del proyecto en VS Code.
2. Pulsa **F5** (o el botón en *Ejecutar y depurar* → "Traductor LSM (main.py)").

`.vscode/launch.json` ya usa el Python de `venv`. También puedes ejecutarlo desde la terminal:

```powershell
.\venv\Scripts\python.exe main.py
```

### Crear el entorno desde cero

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install opencv-python mediapipe==0.10.18 numpy pyttsx3
```

> MediaPipe 0.10.18 requiere Python 3.12 (no es compatible con 3.13). La voz usa las voces SAPI5 de Windows (por ejemplo *Microsoft Sabina*, español de México). Esta versión **no agrega dependencias**: el filtro One-Euro está escrito en `main.py` y la pose de la cabeza usa `cv2.solvePnP`.

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
| Velocidad | — | 25-26 FPS conectada; 17 FPS en batería con dos manos |

**Limitaciones:** la **P y la R** dependen de cómo cruces los dedos y en las pruebas se confundieron; algunas preguntas falsas con las cejas aparecen mientras haces letras (al mirar la mano o la pantalla); con dos manos y la laptop en batería baja a ~17 FPS. Detalles, cifras y la metodología completa en la [documentación técnica](DOCUMENTACION.md).

## Calibración

Los umbrales están agrupados al inicio de `main.py`, cada uno con el rango medido que lo justifica. Si una seña cuesta trabajo, activa el diagnóstico con la tecla **D**: verás por cada mano el estado de cada dedo (`EXT` estirado, `DOB` doblado, `CER` cerrado) con su extensión y flexión, la posición del pulgar, la normal de la palma y los ángulos de la cabeza, y con eso puedes ajustar el umbral correspondiente.

## Estructura

```
Proyecto/
├── .vscode/          # Configuración para ejecutar con F5
├── venv/             # Entorno virtual (no se sube a Git)
├── .gitignore
├── main.py           # Programa principal
├── README.md
└── DOCUMENTACION.md  # Documentación técnica completa
```
