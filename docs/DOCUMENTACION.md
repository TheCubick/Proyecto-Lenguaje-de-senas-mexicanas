# Sistema Inteligente de Traducción de Lengua de Señas con Visión Artificial y Componentes Multimodales

> **Documentación técnica oficial**
> **Materia:** Inteligencia Artificial
> **Stack del backend:** Python 3.12 · MediaPipe 0.10.18 · OpenCV 4.11 (captura y `solvePnP`) · NumPy · pyttsx3 · FastAPI + uvicorn (HTTP y WebSocket)
> **Stack del frontend:** HTML5 · CSS3 (variables, container queries, `color-mix`) · JavaScript sin dependencias (WebSocket nativo, video MJPEG en `<img>`, Canvas 2D)
> **Arquitectura:** cliente-servidor local (`127.0.0.1:5000`)
> **Repositorio:** https://github.com/TheCubick/Proyecto-Lenguaje-de-se-a

### Leyenda de estado

En este documento cada componente lleva una etiqueta que indica su grado de avance real:

| Etiqueta | Significado |
|---|---|
| **Implementado** | Funciona en el código actual (`main.py` y sus módulos) y se validó con grabaciones reales (sección 8) |
| **Diseñado** | Arquitectura y algoritmo definidos; pendiente de construir |
| **Planeado** | Forma parte del roadmap a mediano o largo plazo |

---

## Índice

1. [Descripción general](#1-descripción-general)
2. [Arquitectura del sistema (enfoque multimodal)](#2-arquitectura-del-sistema-enfoque-multimodal)
   - 2.1 [Módulo de manos: dos manos y geometría 3D](#21-módulo-de-manos-dos-manos-y-geometría-3d-mediapipe-hands)
   - 2.2 [Señas con movimiento](#22-señas-con-movimiento-buffer-de-trayectoria-por-mano)
   - 2.3 [Módulo facial](#23-módulo-facial-pose-3d-de-la-cabeza-y-cejas)
   - 2.4 [Voz asíncrona](#24-voz-asíncrona-text-to-speech)
   - 2.5 [Arquitectura cliente-servidor](#25-arquitectura-cliente-servidor)
   - 2.6 [Frontend web](#26-frontend-web)
3. [Flujo de trabajo y experiencia de usuario](#3-flujo-de-trabajo-y-experiencia-de-usuario-ux--ui)
4. [Integración de hardware](#4-integración-de-hardware)
5. [Roadmap y futuro del proyecto](#5-roadmap-y-futuro-del-proyecto)
6. [Guía de instalación y ejecución](#6-guía-de-instalación-y-ejecución)
7. [Calibración](#7-calibración)
8. [Validación con datos reales](#8-validación-con-datos-reales)
9. [Limitaciones conocidas](#9-limitaciones-conocidas)
10. [Decisiones de diseño respaldadas por datos](#10-decisiones-de-diseño-respaldadas-por-datos)
11. [Referencia del código](#11-referencia-del-código)

---

## 1. Descripción general

### 1.1 Propósito

El sistema traduce **en tiempo real** el **alfabeto dactilológico completo de la Lengua de Señas Mexicana (LSM)** a texto y a voz, usando solamente una cámara web. Combina tres canales:

| Canal | Qué aporta | Estado |
|---|---|---|
| **Manual** (manos) | Qué letra o seña se hace: 21 letras estáticas, 6 letras con movimiento y señas de control, **con cualquiera de las dos manos o con ambas a la vez** | Implementado |
| **No manual** (rostro y cabeza) | Si la frase es pregunta, negación o afirmación, a partir de la **pose 3D real de la cabeza** | Implementado |
| **Voz** | Dice cada letra al registrarla y lee la palabra completa | Implementado |

La LSM no se comunica solo con las manos: los **componentes no manuales** (expresión facial, movimientos de cabeza) tienen función gramatical. Por eso la arquitectura es **multimodal**.

El sistema se entrega como una **aplicación web local**. Un backend en Python captura la cámara, ejecuta toda la inferencia y transmite por dos canales: el **video** de la cámara (MJPEG) y, por WebSocket, el **estado del reconocimiento con las coordenadas de las manos**. El navegador muestra el video y dibuja el esqueleto encima. Un frontend en el navegador los presenta y añade un **Modo Práctica** para aprender el alfabeto con retroalimentación inmediata. La interfaz no reconoce nada por sí misma: solo presenta lo que calcula el backend (sección 2.5).

### 1.2 Problema que atiende

La comunidad sorda enfrenta barreras de comunicación cotidianas con personas oyentes que no conocen la lengua de señas. Las soluciones comerciales suelen requerir guantes con sensores o hardware costoso. Este proyecto explora una alternativa **de bajo costo, sin contacto y basada en visión artificial**, que funciona en cualquier computadora con cámara.

### 1.3 Enfoque técnico

1. **Estimación de pose con aprendizaje profundo** (modelos preentrenados de MediaPipe): 21 puntos por mano, en coordenadas de imagen y en **coordenadas 3D de mundo** (metros), y 468 puntos por rostro.
2. **Geometría vectorial 3D**: ángulos articulares por producto punto, normal de la palma por producto cruz, un marco de referencia propio de cada mano y la pose de la cabeza con `cv2.solvePnP`.
3. **Clasificadores geométricos basados en reglas**, diseñados por el equipo y **calibrados con grabaciones reales** de ambas manos.
4. **Filtrado temporal**: filtro antivibración One-Euro por mano, estabilizador de etiquetas y análisis de trayectorias en ventanas de tiempo.

| Ventaja del enfoque | Explicación |
|---|---|
| **Explicable** | Cada decisión se justifica con una regla geométrica concreta |
| **Sin dataset previo** | No requiere recolectar ni etiquetar miles de imágenes |
| **Tiempo real** | 25-26 FPS de captura con la laptop conectada; 11-17 FPS en batería con dos manos (sección 8.9) |
| **Invariante** | No depende de la distancia a la cámara, de la posición, de la rotación de la mano ni de si se usa la izquierda o la derecha |

### 1.4 Alfabeto de referencia

Las señas se definieron a partir de la **lámina oficial "Abecedario en Lengua de Señas Mexicana"** de la SEP y Aprende.mx, que tiene 26 letras y usa flechas rojas para las que llevan movimiento. La **Ñ** se forma como la N con un vaivén lateral. **CH, LL y RR** no están incluidas.

---

## 2. Arquitectura del sistema (enfoque multimodal)

```
                              ┌────────────────────────┐
                              │       Cámara web       │
                              └───────────┬────────────┘
                                          │ frame BGR
                              ┌───────────▼────────────┐
                              │ Espejo · BGR → RGB     │
                              └──────┬───────────┬─────┘
                                     │           │
   ┌─────────────────────────────────▼──┐   ┌────▼───────────────────────────────┐
   │ MANOS (MediaPipe Hands, 2 manos)   │   │ ROSTRO (MediaPipe Face Mesh)       │
   │ imagen + mundo 3D + lateralidad    │   │ 468 puntos                         │
   ├────────────────────────────────────┤   ├────────────────────────────────────┤
   │ SeguidorManos → ranura Izq / Der   │   │ EstimadorPoseCabeza (solvePnP)     │
   │ ┌──────────────┐ ┌──────────────┐  │   │  pitch · yaw · roll · error        │
   │ │ CanalMano Izq│ │ CanalMano Der│  │   │ calcular_rasgos_faciales           │
   │ │ One-Euro     │ │ One-Euro     │  │   │  cejas compensadas por la pose     │
   │ │ espejo → 3D  │ │ 3D           │  │   │ AnalizadorFacial                   │
   │ │ clasificar   │ │ clasificar   │  │   │  · pregunta · negar · asentir      │
   │ │ Estabilizador│ │ Estabilizador│  │   │  · cara neutra (tecla N)           │
   │ │ Trayectoria  │ │ Trayectoria  │  │   └───────────────┬────────────────────┘
   │ │ Reconocedor  │ │ Reconocedor  │  │                   │ ?  ·  borrar  ·  confirmar
   │ └──────┬───────┘ └──────┬───────┘  │                   │
   └────────┼────────────────┼──────────┘                   │
            └──────────┬─────┘                              │
                ┌──────▼────────────────────────────────────▼──┐
                │ Escritor: hold-to-type 1.5 s POR MANO, texto  │   (Traductor.paso orquesta todo:
                └──────┬───────────────────────────────┬────────┘    el mismo código en vivo y en pruebas)
       ┌───────────────▼──────────────────┐   ┌────────▼────────────────┐
       │ ui_panel.estado_web() → JSON     │   │ Hilo de voz (pyttsx3)   │
       │ señas · texto · rostro ·         │   │ + enviar_a_hardware()   │
       │ coordenadas 21×[x,y,z] por mano  │   └─────────────────────────┘
       └───────────────┬──────────────────┘
                       │ Difusor (último estado)            HILO DE PROCESO (main.capturar)
 ══════════════════════╪══════════════════════════════════════════════════════════════════
                       │                                    SERVIDOR (server.py, asyncio)
       ┌───────────────▼──────────────────┐
       │ FastAPI · WebSocket /ws          │◀── acciones {"accion": ...} ── cola → hilo de proceso
       │ texto: JSON (~0.5-1.6 KB)        │
       └───────────────┬──────────────────┘
                       │ ws://127.0.0.1:5000/ws
       ┌───────────────▼──────────────────────────────────────────────────────┐
       │ NAVEGADOR (static/): <img> con el video + <canvas> con el esqueleto, │
       │ anillo hold-to-type, manos, rostro, subtítulos, guía,                │
       │ tema claro/oscuro y Modo Práctica                                    │
       └──────────────────────────────────────────────────────────────────────┘
   (En paralelo, GET /video_feed transmite en MJPEG el fotograma más nuevo de CamaraReciente sin esperar a
    MediaPipe. Python es el único dueño de la cámara; ver sección 2.5.1.)
```

---

### 2.1 Módulo de manos: dos manos y geometría 3D (MediaPipe Hands)

#### 2.1.1 Detección

| Parámetro | Valor | Justificación |
|---|---|---|
| `static_image_mode` | `False` | Seguimiento temporal: más rápido y estable |
| `max_num_hands` | `2` (`MAX_MANOS`) | Cualquiera de las dos manos escribe, o ambas a la vez. Costo medido: con una sola mano a la vista, MediaPipe busca la segunda en cada frame y el tiempo de las manos se duplica (37 → 76 ms en batería) |
| `min_detection_confidence` | `0.7` | Reduce detecciones falsas |
| `min_tracking_confidence` | `0.6` | Mantiene el seguimiento con movimiento |
| Salidas que se usan | `multi_hand_landmarks`, `multi_hand_world_landmarks`, `multi_handedness` | Imagen (x, y, z), mundo 3D en metros y etiqueta Left/Right con su confianza |

```
          8    12   16   20      ← TIP  (puntas)
          7    11   15   19      ← DIP
          6    10   14   18      ← PIP  (articulación media)
    4     5     9   13   17      ← MCP  (nudillos)
     3
      2                          Pulgar: 1 CMC · 2 MCP · 3 IP · 4 TIP
       1
            0                    ← WRIST (muñeca)
```

#### 2.1.2 Lateralidad y estado independiente por mano

La imagen se espeja **antes** de procesarla, así que la etiqueta de `multi_handedness` corresponde a la mano real de la persona. Con datos reales la etiqueta fue muy confiable (0.94-1.00), pero se observaron tres problemas que resuelve `SeguidorManos`:

| Problema observado en las grabaciones | Regla |
|---|---|
| MediaPipe devolvió **la misma mano dos veces** (mismo lugar, misma etiqueta) | Dos detecciones a menos de 0.35 tamaños de mano son la misma: se queda la más confiable |
| La etiqueta **parpadeó** unos frames en letras como C o F | **Continuidad:** una mano que aparece donde estaba una ranura activa (< 1.2 tamaños de mano, < 0.25 s) sigue en esa ranura aunque la etiqueta parpadee |
| El orden de las manos en la lista de MediaPipe cambia entre frames | Se asignan por posición, no por orden |
| Una mano quedó en la ranura equivocada | Se acumula la evidencia (EMA) de la etiqueta; si la otra lateralidad se sostiene (> 0.85), se corrige |

Cada ranura tiene su propio `CanalMano`: **filtro One-Euro, estabilizador, buffer de trayectoria, reconocedor de trazos y barra de progreso**. Por eso las trayectorias de las dos manos nunca se cruzan, aunque ambas estén en escena o se crucen físicamente.

#### 2.1.3 Normalización de simetría (espejo)

Antes de clasificar, **la mano izquierda se refleja en el eje lateral** (`x → −x`) en sus coordenadas de imagen y de mundo (`a_canonica`). Así todas las reglas se escriben una sola vez, para una mano derecha canónica, y valen igual para la izquierda. Sin el espejo, la normal de la palma de una mano izquierda apuntaría al revés y las reglas de orientación se invertirían.

En los trazos esto significa que **la mano izquierda hace los movimientos en espejo**: por ejemplo BORRAR es un barrido hacia el centro del cuerpo (a la izquierda con la mano derecha y a la derecha con la izquierda). La J, la Z y el `?` aceptan además el trazo sin reflejar, porque se pueden dibujar como letra escrita.

#### 2.1.4 Filtro antivibración One-Euro

`FiltroOneEuro` (Casiez, Roussel y Vogel, 2012) suaviza los 21 puntos de cada mano (imagen y mundo) con un promedio exponencial cuya frecuencia de corte **sube con la velocidad**:

```
α(f_c) = 1 / (1 + 1 / (2π · f_c · Δt))
dx̂ ← dx̂ + α(f_d) · ((x − x̂)/Δt − dx̂)                     velocidad filtrada
f_c = f_min + β · |dx̂| / tamaño_de_mano                     corte adaptativo
x̂ ← x̂ + α(f_c) · (x − x̂)
```

Con la mano quieta corta bajo (f_min = 1.5 Hz) y quita el temblor; al hacer un trazo el corte sube (β = 0.4) y casi no mete retraso. La velocidad se mide en tamaños de mano por segundo, así que no depende de la distancia. Con datos reales mejoró la escritura de extremo a extremo y eliminó trazos falsos causados por el temblor (sección 8.7).

#### 2.1.5 Rasgos 3D (`RasgosMano`)

Todo se calcula sobre la mano canónica en un **marco propio de la mano**: `u` (muñeca → nudillo del medio), `s` (nudillo del índice → nudillo del meñique, ortogonalizado) y la **normal de la palma** `n = s × u`. Las medidas se dividen entre el largo de la palma, así que son adimensionales.

| Rasgo | Cómo se mide | Para qué sirve |
|---|---|---|
| **Extensión de cada dedo** | ‖punta − muñeca‖ / ‖nudillo − muñeca‖ en 3D | Recto (1.45-2.1), curvo (1.0-1.5) o cerrado (0.6-1.15) |
| **Flexión articular** | ángulos en MCP, PIP y DIP por **producto punto** de los vectores MCP→PIP→DIP→TIP | Distingue un dedo largo pero doblado; la E dobla PIP/DIP con el MCP recto |
| **Estado del dedo** (`EXT` / `DOB` / `CER`) | recto si está estirado (extensión) **y** sus articulaciones casi no se doblan (PIP+DIP ≤ 75°), o si las articulaciones están rectas (≤ 55°) aunque se vea hasta 0.15 más corto; el meñique, con PIP+DIP ≤ 50° aunque el nudillo lo incline | Base de casi todas las letras. Con los ángulos se separa la N (índice a 1.39 pero 51° de flexión: recto) de la E (índice a 1.39 con 95°: doblado) |
| **Normal de la palma** | producto cruz `s × u` (cámara: z < 0 = palma hacia la cámara) | P (palma girada), orientación |
| **Orientación** | ángulo 3D del eje `u` respecto a "arriba": 0° arriba · 90° de lado · 180° abajo | G y H (de lado), M y N (abajo), Q (colgando) |
| **Dirección del índice** | ángulo 3D nudillo → punta del índice | D (arriba) contra X (de lado) y Q (abajo) |
| **Pulgar respecto a las falanges** | punta del pulgar contra el nudillo del índice (eje lateral) y contra los PIP de índice y medio (eje de la mano) | A (por fuera), T (asoma entre índice y medio), S (cruzado por debajo) |
| **Cruce índice-medio** | (punta del medio − punta del índice) en el eje lateral, con signo | R (orden invertido), V (separadas) |
| **Ángulo 3D índice-medio** y **profundidad del medio** | ángulo entre ambos dedos y cuánto se adelanta el medio hacia el lado de la palma | U (paralelos) contra P (el medio cruza por delante) |
| **Contacto del pulgar** | distancia punta del pulgar ↔ punta del índice **en la imagen** (más precisa en x-y) | O, F, C, Y, ESPACIO |

#### 2.1.6 Clasificador (`clasificar_pose`)

Orden de decisión: **pinzas y manos curvas (F, O, C) → número de dedos rectos → orientación → pulgar y relación índice-medio**. Si ninguna regla coincide devuelve `None`: el sistema prefiere no responder antes que equivocarse.

| Seña | Cómo se hace (lámina SEP) | Regla implementada |
|---|---|---|
| **A** | Puño con el pulgar estirado al lado | ningún dedo recto, pulgar por fuera del nudillo del índice (< −0.45) |
| **B** | 4 dedos juntos arriba, pulgar sobre la palma | 4 rectos, pulgar sobre la palma (> 0), apertura índice-meñique < 20° |
| **C** | Mano curva en forma de C | índice y medio curvos (flexión PIP+DIP 60-115°), pulgar por fuera, hueco pulgar-índice 0.32-1.10 (más cerca es la O), índice no metido en la palma (≥ 1.05) |
| **D** | Índice arriba, los demás cerrados | solo el índice recto, pulgar adentro, sin la inclinación de la X |
| **E** | Dedos doblados sobre el pulgar | ningún dedo recto, pulgar por debajo de los PIP, nudillos (MCP) casi rectos (< 33°) |
| **F** | Índice toca el pulgar, 3 dedos arriba | anular y meñique rectos, medio no cerrado, pinza < 0.55 (o < 0.75 si el índice está claramente doblado) |
| **G** | "Pistola": índice de lado, pulgar arriba | índice recto (o largo aunque algo doblado, ≥ 1.40), pulgar afuera, mano de lado |
| **H** | "Pistola" con dos dedos | índice y medio rectos, mano de lado |
| **I** | Solo el meñique arriba | meñique recto, índice y medio sin estirar, pulgar cerca del índice (< 0.75) |
| **L** | Índice arriba y pulgar al lado | índice recto, pulgar afuera, mano arriba |
| **M** | 3 dedos hacia abajo | índice, medio y anular rectos, mano hacia abajo |
| **N** | 2 dedos hacia abajo | índice y medio rectos, mano hacia abajo |
| **O** | Círculo índice-pulgar | índice y medio curvos, pulgar por fuera, puntas juntas (< 0.32) |
| **P** | Índice y medio cruzados en "X" | índice y medio arriba sin cruce de puntas, pero con ángulo ≥ 11°, el medio adelantado o la palma girada |
| **R** | Índice y medio entrelazados | índice y medio arriba con las puntas **cruzadas** (cruce ≤ −0.05) |
| **S** | Puño, pulgar sobre los dedos | ningún dedo recto, pulgar cruzado por debajo de los PIP |
| **T** | Puño, pulgar entre los dedos | ningún dedo recto, punta del pulgar **por encima** de los PIP de índice y medio (> 0) |
| **U** | Índice y medio juntos | paralelos (< 11°), en el mismo plano y con la palma de frente |
| **V** | Índice y medio en V | puntas separadas en el eje lateral (≥ 0.40) |
| **W** | Índice, medio y anular abiertos | 3 rectos, mano arriba |
| **Y** | Pulgar y meñique estirados | meñique recto, índice y medio sin estirar, pulgar lejos del índice (≥ 0.75) |
| **ESPACIO** | Mano abierta, 5 dedos separados | 4 rectos, pulgar lejos del índice (≥ 1.00) y por fuera |
| (mano abierta sin seña) | Mano relajada | 4 rectos sin cumplir B ni ESPACIO: no escribe, pero sirve para barrer (BORRAR) |

Las configuraciones de **Q** (puño o pinza con la mano colgando: orientación ≥ 80° e índice ≥ 105°) y **X** (índice recto o a medio estirar de lado con la palma inclinada) solo escriben con su movimiento.

```
                 ¿Pinza con anular y meñique rectos? ── sí ──▶ F
                 ¿Índice y medio curvos y pulgar fuera? ─ sí ─▶ O (se tocan) · C (hueco)
                              ¿Cuántos dedos rectos?
     ┌──────────┬──────────┬──────────────────┬─────────────────┬────────────────┐
     4          3          2 (índice-medio)   1                 0
  M (abajo)   M · W     N (abajo) · H (lado)  índice: L·G·Q·X·D  Q (colgando) · X
  ESPACIO              V (separadas)           meñique: I · Y     pulgar fuera: L · G (índice
  B · abierta          R (cruzadas) · U · P                        largo) o A (índice metido)
                                                                   T (pulgar arriba)
                                                                   E (MCP rectos) · S
```

---

### 2.2 Señas con movimiento (buffer de trayectoria por mano)

Ocho gestos necesitan **movimiento**: seis letras (J, K, Ñ, Q, X, Z) y dos señas de control (BORRAR y el `?` dibujado en el aire). Cada mano tiene su propio buffer y su propio reconocedor.

#### 2.2.1 Buffer

`Trayectoria` guarda en una `collections.deque` (hasta 64 muestras) la posición de la muñeca, el centro de la palma y las puntas de los dedos **en la mano canónica** (la izquierda reflejada), normalizadas por el tamaño de la mano, junto con la configuración vista en cada frame. Guarda también las posiciones reales para dibujar la estela.

> **Las ventanas se miden en segundos, no en fotogramas.** Con 45 fotogramas, a 25 FPS una ventana dura 1.8 s y a 16 FPS 2.8 s. En la sesión de prueba (grabada a ~17 FPS) eso cambiaba los resultados; con ventanas en segundos el reconocimiento es el mismo en una computadora rápida o lenta. Una Z a ritmo natural dura 1.4-2 s; un vaivén, ~1 s; el barrido de BORRAR, ~0.5 s.

#### 2.2.2 Reconocedores

Cada gesto exige que **su configuración domine su ventana** (≥ 60 % de las muestras) y **un patrón de movimiento**. No basta con el último fotograma: al moverse rápido la mano a veces se lee como otra letra (por ejemplo la D se ve como X al dibujar una Z). Si varios trazos coinciden, gana el de configuración más dominante.

| Gesto | Configuración | Punto rastreado | Ventana | Patrón |
|---|---|---|---|---|
| **J** | I o Y (el pulgar parpadea) | punta del meñique | 1.2 s | baja ≥ 0.55 manos y se curva ≥ 0.25 hacia un lado (gancho) |
| **Z** | D | punta del índice | 1.8 s | 3 trazos: barra →, diagonal ↙ (225° ± 35°), barra →; barras ≥ 0.40, diagonal ≥ 0.45 |
| **?** | D | punta del índice | 1.8 s | arco que abulta a un lado, regresa al centro y termina en un tallo vertical; altura ≥ 0.80 |
| **K** | P, R o U | punto medio de las puntas índice-medio | 1.8 s | **vaivén** lateral: ≥ 2 cambios de sentido de ≥ 0.15 manos |
| **Ñ** | N (o H si la N queda de lado) | centro de la palma | 1.2 s | vaivén lateral: ≥ 2 cambios de sentido de ≥ 0.07 manos |
| **Q / X** | Q o X | punta del índice | 1.0 s | vaivén de ≥ 0.07 manos; Q si el índice cuelga en promedio (≥ 105°), X si no |
| **BORRAR** | mano abierta (B, ESPACIO o abierta) | centro de la palma | 0.6 s | barrido hacia el centro del cuerpo ≥ 1.10 manos, ≥ 3.0 manos/s (medido en tramos de 0.15 s), ≥ 85 % monótono |

**Técnicas usadas:** simplificación de trazos con **Ramer-Douglas-Peucker** y análisis por segmentos (rumbo y largo) para la Z; detección de **puntos de giro** con histéresis para los vaivenes; velocidad pico para BORRAR.

#### 2.2.3 Movimiento coherente (vaivenes)

Para que acomodar la mano no cuente como K, Ñ, Q o X, las **últimas dos idas y vueltas completas** (entre puntos de giro) deben ser:

| Condición | Valor | Medido con datos reales |
|---|---|---|
| **Directas**: desplazamiento lateral / camino recorrido | ≥ 0.5 | K y Ñ reales: 0.6-0.99; la K falsa al dibujar una Z: 0.34 |
| **De tamaño parecido**: la mayor / la menor | ≤ 2.5 | K reales: 1.0-1.4; esa K falsa: 3.8 |

#### 2.2.4 Reglas que evitan errores

| Regla | Qué evita |
|---|---|
| Una letra **estática** solo avanza si **esa mano** se mueve a menos de 1.2 manos/s | que moverse mientras se forma la seña escriba letras |
| Tras un trazo, la configuración base de esa mano queda **bloqueada** hasta que cambie o baje (p. ej. la I tras dibujar una J) | que la I se escriba 1.5 s después de la J |
| Tras un vaivén hace falta una **pausa** de 0.3 s con la palma casi quieta | que agitar seguido escriba `Q Q Q` |
| Enfriamiento de 1.0 s entre trazos de la misma mano | duplicados |
| El buffer se vacía si la mano desaparece más de 5 fotogramas, o si otra mano ocupa la ranura | trazos que mezclan dos intentos o dos manos |

---

### 2.3 Módulo facial: pose 3D de la cabeza y cejas

#### 2.3.1 Fundamento lingüístico

| Componente no manual | Función gramatical | Acción en el sistema |
|---|---|---|
| Cejas levantadas | Pregunta cerrada (sí/no) | Agrega `?` |
| Giro lateral repetido de la cabeza | Negación | Borra la última letra |
| Asentir con la cabeza | Afirmación | Confirma la palabra, agrega un espacio y la lee en voz alta |
| Cejas fruncidas | Pregunta abierta (qué, dónde, quién…) | Planeado |

> **¿Existe una seña manual de "pregunta" en la LSM?** No se encontró ninguna fuente que describa un gesto manual con el dedo índice. Lo documentado es la marca facial. El **"?" dibujado en el aire con el índice** es un **respaldo propio del sistema**, no una seña oficial de la LSM.

#### 2.3.2 Pose real de la cabeza con `cv2.solvePnP`

`EstimadorPoseCabeza` empareja **15 puntos rígidos** de Face Mesh (nariz, comisuras de los ojos, frente, mentón y pómulos; **sin cejas ni boca**, que se mueven con la expresión) con su posición en el **modelo canónico de la cara de MediaPipe** (`canonical_face_model.obj`, en cm, convertido a los ejes de la cámara de OpenCV) y resuelve la rotación:

```
K = [[ancho, 0, ancho/2], [0, ancho, alto/2], [0, 0, 1]]     cámara aproximada (foco = ancho)
solvePnP(modelo_3D, puntos_2D, K)  →  R (Rodrigues), t
frente = −R·ẑ      pitch = asin(frente_y)            (+ = mira abajo)   → asentir
                   yaw   = atan2(frente_x, −frente_z)                    → negar
arriba = −R·ŷ      roll  = atan2(arriba_x, −arriba_y)                    → inclinación
```

La primera vez usa EPnP y después parte de la pose anterior (más estable). Con datos reales el **error de reproyección** fue de 1.6 px de mediana y la distancia estimada a la cámara (~55-90 cm) es coherente con una laptop. Los ángulos se muestran en vivo bajo el medidor de cejas.

#### 2.3.3 Cejas compensadas por la pose

La medida base es la de la versión anterior: altura media de las cejas (5 puntos del contorno superior de cada una) sobre la línea de las **comisuras de los ojos**, proyectada sobre el eje de la cara (mentón → frente) y dividida entre el alto del rostro. Al inclinar o girar la cabeza esa medida 2D cambia por escorzo, aunque las cejas no se muevan. Para compensarlo se **proyecta el modelo canónico con la misma pose** que dio `solvePnP`:

```
cejas = medida_2D / ( medida del modelo canónico con esta pose / medida del modelo de frente )
```

Lo que cambia solo por mover la cabeza se cancela, y lo que queda es el movimiento real de las cejas.

#### 2.3.4 Calibración de la cara neutra (tecla `N`)

Durante los primeros 45 fotogramas con rostro se guarda la **línea base personal**: cejas (percentil 35, que ignora fotogramas con las cejas ya levantadas), apertura de los ojos (mediana) y pose neutra de la cabeza. La tecla **`N`** la repite en cualquier momento (`R` sigue funcionando).

#### 2.3.5 Detección de la pregunta

```
razón = cejas / cejas_base
   razón ≥ 1.07                    → empieza a contar (cejas arriba)
   1.04 ≤ razón < 1.07             → histéresis: sigue contando
   razón < 1.04 por más de 0.30 s  → se reinicia (un parpadeo no cuenta)
   1.5 s seguidos arriba           → "?"  (una sola por cada vez que se levantan las cejas)
   más de 4 s arriba               → no es una pregunta: ese nivel pasa a ser la nueva cara neutra
```

La última regla viene de los datos: en una sesión la razón se quedó en ~1.11 durante 50 s seguidos (cambio de postura y de mirada) y, sin ella, cada parpadeo rearmaba una pregunta falsa. Una pregunta real dura 1.5-3 s.

#### 2.3.6 Cara tapada por una mano

Con dos manos en escena es común taparse la cara. Cuando eso pasa, Face Mesh sigue dando puntos pero deformados, y la pose y las cejas dejan de ser confiables. El **error de reproyección de `solvePnP`** lo detecta:

| Situación (datos reales) | Error / alto de la cara |
|---|---|
| Cara visible | 0.8-1.4 % |
| Manos encima de la cara | 1.7-3.8 % |

Por encima de **1.8 %** el fotograma no cuenta para ningún gesto facial y la tarjeta *Rostro* de la interfaz muestra **Cara tapada**.

#### 2.3.7 Negar y asentir con ángulos reales (`DetectorOscilacion`)

Negar y asentir son **patrones temporales** sobre el yaw y el pitch en grados. Una media móvil exponencial (τ = 2.5 s) sigue la postura neutra, así que mirar a un lado y quedarse ahí no cuenta. Se cuentan las **entradas** a zonas con histéresis dentro de una ventana de 1.4 s:

| Gesto | Señal | Amplitud | Condición | Medido |
|---|---|---|---|---|
| Negación | yaw | 8° | ≥ 3 entradas **en ambos lados** | un "no" real oscila ±19° |
| Afirmación | pitch | 5° | ≥ 3 entradas | un "sí" real oscila ±5-6°; con 2 entradas, mover la cabeza al leer la pantalla contaba como "sí" |

Tras un gesto de cabeza hay un enfriamiento de 1.2 s y las cejas se ignoran 0.8 s.

---

### 2.4 Voz asíncrona (text-to-speech)

`pyttsx3` (SAPI5 en Windows) corre en **un hilo propio** (`VozAsincrona`), así que hablar **nunca bloquea la cámara**.

| Aspecto | Diseño |
|---|---|
| **Hilo** | El motor se crea **dentro** del hilo (en Windows, SAPI5 queda ligado al hilo que lo crea) y se inicializa COM en él |
| **Voz** | Prefiere español de México (*Microsoft Sabina*); si no, cualquier voz en español; si no, la voz por defecto |
| **Arranque** | Una frase de calentamiento en silencio, para que la primera letra real no se retrase |
| **Cola** | `decir()` regresa al instante. Una **palabra** sustituye a las letras pendientes; si se acumulan más de 2 letras, se descartan las más viejas |
| **Nombres de letras** | Se dice el nombre en español (`jota`, `eñe`, `doble ve`, `equis`…) y no un sonido suelto |
| **Tolerancia a fallos** | Si falta `pyttsx3` o una voz, el programa sigue sin voz y lo indica en el encabezado de la interfaz (**Sin voz**) |

**Cuándo habla** (`PoliticaVoz`):

| Evento | Qué dice |
|---|---|
| Se registra una letra con cualquier mano (barra al 100 % o trazo completo) | el nombre de la letra |
| Se **confirma** la palabra (asentir) | la palabra completa |
| Pasan **4 s sin nuevas letras** | la palabra pendiente (una sola vez) |
| Espacio, borrar, `?` | nada |

---

### 2.5 Arquitectura cliente-servidor

**Implementado.** La percepción y la presentación están separadas en dos procesos lógicos que se comunican por HTTP y WebSocket locales:

| Capa | Archivos | Responsabilidad |
|---|---|---|
| **Motor** (hilos de cámara y de proceso) | `camara.py`, `main.py` → `classifier_hands`, `classifier_face`, `traductor`, `voice_engine`, `ui_panel` | Leer la cámara, reconocer manos y rostro, escribir el texto, hablar y serializar el estado con las coordenadas de las manos |
| **Servidor** | `server.py` (FastAPI + uvicorn) | Abrir y cerrar la cámara, servir los archivos estáticos, transmitir el **estado** por WebSocket (`/ws`) y el **video** por MJPEG (`/video_feed`), y recibir las acciones |
| **Cliente** | `static/index.html`, `estilos.css`, `app.js` | Mostrar el video, dibujar el esqueleto encima, presentar el estado, ofrecer controles, tema visual y Modo Práctica |

**Dos canales separados:**
- **Estado** por WebSocket: JSON de unos 0.5 KB por fotograma sin manos y 1.6 KB con dos manos.
- **Video** por un flujo MJPEG (`multipart/x-mixed-replace`) que un `<img>` muestra de forma nativa.

El video sale del fotograma más nuevo **sin esperar a MediaPipe**, así que la imagen nunca se retrasa por el reconocimiento; el esqueleto llega unos milisegundos después por el otro canal.

#### 2.5.1 Modelo de concurrencia

| Hilo | Qué ejecuta | Por qué está separado |
|---|---|---|
| **Cámara** (`CamaraReciente`) | `cap.read()` sin parar; guarda solo el último fotograma con su hora de captura | Que el proceso siempre tome la imagen más nueva (ver abajo) |
| **Proceso** (`main.capturar`) | Fotograma más reciente → MediaPipe → `Traductor.paso()` → voz y hardware → JSON | MediaPipe es bloqueante; no debe detener el servidor ni la lectura de la cámara |
| **Voz** (`VozAsincrona`) | Motor SAPI5 de `pyttsx3` | Hablar tarda segundos; nunca bloquea la captura (sección 2.4) |
| **Bucle de eventos** (uvicorn) | Conexiones HTTP y WebSocket, una corrutina por cliente | E/S de red asíncrona |
| **Grupo de hilos de asyncio** | `Difusor.esperar()`, `CamaraReciente.leer(..., consumir=False)` y `a_jpeg` vía `asyncio.to_thread` | Esperas bloqueantes y compresión JPEG sin congelar el bucle de eventos |

**Escritor único.** Solo el hilo de proceso modifica el texto y el estado del reconocimiento. Las acciones del navegador (Espacio, Borrar, Limpiar, Diagnóstico, Recalibrar) entran a una `queue.Queue` y el hilo de proceso las aplica al **inicio del siguiente fotograma**. Así no hace falta ningún candado sobre `Escritor` ni sobre `AnalizadorFacial`.

**Fotograma más reciente (`camara.py`).** Si la lectura y el proceso están en el mismo bucle, la cámara acumula imágenes en su búfer mientras MediaPipe trabaja, y cada vuelta procesa una imagen más vieja que la anterior. `CamaraReciente` lee en su propio hilo y conserva solo la última imagen; el proceso la pide con `leer(visto)` y espera si todavía no hay una nueva. Las que se reemplazaron sin procesarse se cuentan en `descartados`. Cada imagen lleva su hora de captura en reloj de pared (`time.time()`), que el navegador compara con `Date.now()` para medir la latencia, porque ambos corren en la misma computadora.

**Python es el único dueño de la cámara.** Una versión intermedia hacía que el navegador abriera la misma cámara con `getUserMedia` solo para mostrarla. En la laptop de pruebas, Windows no permitió compartirla: Chrome recibía `NotReadableError` porque Python ya la tenía abierta. Se evaluaron dos alternativas:

| Alternativa | Ventajas | Desventajas | Decisión |
|---|---|---|---|
| **El navegador es el dueño** y sube fotogramas a Python por WebSocket | El video se ve nativo; se parece a un despliegue en la nube | Chrome frena o pausa las pestañas en segundo plano, así que el **reconocimiento, la voz y el hardware se detendrían** al cambiar de ventana. Exige subir y comprimir fotogramas en el navegador, controlar el flujo y depender del permiso de cámara | Descartada |
| **Python es el dueño** y transmite video MJPEG | El motor no depende de que el navegador esté al frente (sirve también para la pantalla OLED del roadmap). El video y el esqueleto salen de la misma cámara, así que siempre coinciden. Reusa `CamaraReciente`; ~30 líneas | Comprimir JPEG cuesta ~1 ms por fotograma (medido) | **Elegida** |

**Una cámara, dos lectores.** El proceso lee con `leer(visto)` y **consume** el fotograma, lo que alimenta el conteo de `descartados`. El flujo de video lee con `leer(visto, consumir=False)`: toma la misma imagen más reciente sin alterar ese conteo y la comprime con `a_jpeg` (calidad 70, ancho máximo 640 px). Si el `<img>` no recibe video (por ejemplo, un `503` porque la cámara no abrió), la interfaz pasa a modo **solo esqueleto**: dibuja las manos sobre un fondo oscuro con cuadrícula y lo indica con una etiqueta.

#### 2.5.2 Difusión con semántica de "último valor" (`Difusor`)

El `Difusor` guarda **un solo estado**, el más reciente (JSON), con un número de secuencia y una `threading.Condition`. Cada cliente recuerda el último número que envió y pide el siguiente:

```
proceso ──publicar(json)──▶ [ n = 1042 | estado ] ──esperar(visto=1040)──▶ cliente A (recibe 1042)
                                                  ──esperar(visto=1042)──▶ cliente B (espera al 1043)
```

La misma política de "último valor" se aplica en todas las etapas: cámara (`CamaraReciente`), red (`Difusor` para el estado; el flujo MJPEG también salta a la imagen más nueva) y dibujo (`requestAnimationFrame` solo pinta el estado más nuevo).

- **Sin colas por cliente:** si un navegador va lento, **se salta estados** en lugar de acumular retraso. La latencia queda acotada por un solo fotograma.
- **Varios clientes** pueden ver la misma sesión; todos comparten el mismo texto.
- La espera tiene un tiempo límite de 0.5 s para que el servidor pueda cerrarse aunque la cámara se detenga.

#### 2.5.3 Protocolo: WebSocket (`/ws`) y video (`/video_feed`)

**Video:** `GET /video_feed` responde `multipart/x-mixed-replace; boundary=frame`. Cada parte es un JPEG con sus encabezados `Content-Type: image/jpeg` y `Content-Length`. Los fotogramas van **sin espejar** (el navegador los espeja con CSS) y la respuesta lleva `Cache-Control: no-store`. Si la cámara no está disponible, responde `503`. El flujo termina solo cuando el servidor se cierra o la cámara se pierde.

**WebSocket:**

| Dirección | Tipo de mensaje | Cuándo | Contenido |
|---|---|---|---|
| Servidor → cliente | Texto `{"tipo": "config"}` | Una vez, al conectar | `guia` (30 señas con descripción y movimiento), `umbral_cejas`, `segundos` (hold-to-type), `conexiones` (los 21 huesos de la mano como pares de índices de MediaPipe) y `bases` (configuración base de cada seña con movimiento, p. ej. `J ← I, Y`) |
| Servidor → cliente | Texto `{"tipo": "estado"}` | Cada fotograma procesado | Estado completo con coordenadas (tabla siguiente). Es el **único** mensaje por fotograma |
| Servidor → cliente | Texto `{"tipo": "error"}` | Si no hay cámara o se pierde la señal | `mensaje` legible; se conserva para los clientes que se conecten después |
| Cliente → servidor | Texto `{"accion": ...}` | Botón o tecla | `ESPACIO`, `BORRAR`, `LIMPIAR`, `DIAGNOSTICO`, `RECALIBRAR` o `SALIR`; cualquier otro valor o JSON inválido se ignora |

Campos del mensaje `estado` (`ui_panel.estado_web`):

| Campo | Tipo | Significado |
|---|---|---|
| `sena`, `estado`, `progreso`, `lado` | texto · texto · 0-100 · `"Izq"`/`"Der"` | Seña de la **mano principal** (la que acaba de registrar o la que más avanzó). `estado`: `ninguna`, `sostener`, `mover`, `pose` (necesita trazo) o `registrada` |
| `manos.Izq`, `manos.Der` | objeto | Por mano: `sena`, `estado`, `progreso`, `caja` (recuadro normalizado 0-1 para colocar la etiqueta HTML), `pista` (`"o traza una J"`…), `puntos` y `estela` (abajo) |
| `manos.*.puntos` | 21 × `[x, y, z]` o `null` | Coordenadas **filtradas con One-Euro** de los 21 puntos, en la imagen **espejada**: `x`, `y` en 0-1 respecto al ancho y alto; `z` = profundidad relativa a la muñeca en la escala del ancho (negativa hacia la cámara). 4 decimales (0.06 px a 640) |
| `manos.*.estela` | lista de `[x, y]` o `null` | Recorrido reciente del punto que sigue el trazo (máx. 32 puntos), solo mientras la mano se mueve |
| `tiempos` | objeto | `n` (número de fotograma), `captura_ms` (hora de captura, ms de reloj de pared), `proceso_ms` (MediaPipe + `Traductor.paso`) y `descartados` (fotogramas viejos que no se procesaron) |
| `texto` | texto | Texto acumulado, ya con `limpiar_texto` aplicado |
| `rostro` | objeto | `modo` (`SIN_ROSTRO`, `TAPADO`, `CALIBRANDO`, `NEUTRO`, `CEJAS`, `PREGUNTA`, `NEGACION`, `AFIRMACION`), `pregunta` y `calibracion` (0-100), `cejas` (razón contra la base) y `cabeza` (`[pitch, yaw, roll]` en grados) |
| `voz`, `fps`, `tamano` | texto · número · `[ancho, alto]` | Estado de la voz, FPS del motor y resolución del video |
| `diagnostico` | lista de textos o `null` | Rasgos 3D por mano y desviaciones de la cabeza, solo con el diagnóstico activo |

> **Por qué el video va por un canal aparte.** La primera versión web mandaba por el WebSocket un JPEG con el esqueleto ya dibujado, **después** de MediaPipe. Así, la imagen esperaba al reconocimiento. Separar los canales tiene cuatro ventajas:
> - el video sale en cuanto llega de la cámara;
> - el JSON es 7 a 20 veces más ligero que aquel mensaje;
> - el `<img>` decodifica el MJPEG de forma nativa, sin JavaScript;
> - el esqueleto se dibuja con los colores del tema.
>
> El costo es que el esqueleto va unos milisegundos detrás de la imagen (sección 2.5.6).

#### 2.5.4 Ciclo de vida

1. `python python/server.py` crea el servidor uvicorn. Al arrancar (`lifespan`), abre la cámara con `CamaraReciente` y lanza el hilo de proceso. Un segundo después abre `http://localhost:5000` con `webbrowser`.
2. El navegador se conecta, recibe `config` y, con el **primer estado**, apunta el `<img>` a `/video_feed`. El primer fotograma se detecta en el bucle de dibujo (`naturalWidth > 0`), porque un flujo MJPEG no dispara `load` de forma fiable en todos los navegadores.
3. Si el motor no puede abrir la cámara, publica un mensaje `error` y termina; el servidor sigue arriba para que la interfaz lo muestre.
4. **Cierre:** el botón *Detener* (con confirmación en dos pasos) suelta el `<img>` y envía `SALIR`; también se puede pulsar Ctrl+C en la terminal. El flujo MJPEG y el WebSocket revisan `should_exit` y terminan solos; uvicorn cierra las conexiones (`timeout_graceful_shutdown = 2 s`). Después, el `lifespan` activa `detener`, espera al hilo de proceso (que cierra la voz) y libera la cámara.

#### 2.5.5 Seguridad y privacidad

- El servidor escucha **solo en `127.0.0.1`**: ni el video ni el texto son accesibles desde otra computadora de la red.
- No se guarda ningún fotograma: Python solo conserva en memoria la última imagen, y el video sale únicamente por `127.0.0.1`.
- Están desactivadas las rutas automáticas de documentación de FastAPI (`/docs`, `/redoc`, `/openapi.json`).
- No hay autenticación porque el alcance es local. Exponerlo en una red exigiría autenticación y TLS (`wss://`).

#### 2.5.6 Desempeño medido

Prueba en vivo con la cámara real y dos clientes en Python durante 8 s: uno de WebSocket y uno que leía `/video_feed` al mismo tiempo. La cámara estaba encendida pero **sin manos a la vista**:

| Medida | Primera versión (JPEG por WebSocket) | Versión actual (JSON + MJPEG) |
|---|---|---|
| Estados recibidos por el cliente | 30.1 por segundo | **30.2 por segundo**, sin saltos en la numeración |
| Bytes de estado por fotograma | ~11.3 KB (JPEG) + estado | **0.54 KB** sin manos; **1.57 KB** con dos manos (medido con manos sintéticas) |
| Video | después de MediaPipe | **30.1 FPS** por `/video_feed`, 227 de 227 JPEG válidos, ~16 KB cada uno, sin esperar a MediaPipe |
| Compresión JPEG (calidad 70, 640×480) | — | **1.0 ms** de mediana (máx. 1.8 ms), en 40 fotogramas reales |
| Captura → recepción del estado en el cliente | no medida | **16.1 ms** de mediana (p95 22.3 ms) |
| Proceso del motor (MediaPipe + `Traductor.paso`) | no medida | **15.6 ms** de mediana, sin manos |
| Fotogramas descartados por el proceso | — | 0: el proceso tarda menos que los 33 ms entre fotogramas, y leer el video no consume fotogramas |
| Cierre con `SALIR` con el video abierto | 0.5 s | **0.51 s**; el flujo MJPEG termina limpio 0.03 s después de `SALIR` |

Lo que falta medir:
- **Con manos en escena:** MediaPipe Hands tarda más (25-76 ms según la carga y la batería, sección 8.9). Cuando el proceso supera los 33 ms entre fotogramas, `CamaraReciente` empieza a descartar imágenes viejas en lugar de procesarlas con retraso.
- **Latencia hasta la pantalla:** la de extremo a extremo hasta el dibujo en pantalla la calcula el navegador (`Date.now() - captura_ms`) y aparece junto a los FPS en la cabecera de la cámara; pasa el cursor sobre ella para ver el desglose. Todavía no se ha medido en el navegador.

---

### 2.6 Frontend web

**Implementado.** Son tres archivos estáticos, sin frameworks ni paso de compilación. Las únicas dependencias externas son las fuentes *Inter* y *JetBrains Mono* de Google Fonts; sin conexión a internet se usa la fuente del sistema (*Segoe UI*).

| Archivo | Contenido |
|---|---|
| `index.html` | Estructura semántica: encabezado, tarjeta de cámara con subtítulos, panel lateral con pestañas *Traducir* / *Práctica*, tarjetas de manos y rostro, guía lateral del alfabeto |
| `estilos.css` | Variables de diseño por tema, maquetación con CSS Grid y *container queries*, animaciones y diseño adaptable |
| `app.js` | Conexión y reconexión del WebSocket, video MJPEG, dibujo del esqueleto, medición de latencia, actualización del DOM, tema, teclado y Modo Práctica |

#### 2.6.1 Video y esqueleto

El área de la cámara apila tres capas:

| Capa | Elemento | Detalle |
|---|---|---|
| 1 | `<img>` | Flujo MJPEG de `/video_feed`, **espejado con CSS** (`transform: scaleX(-1)`) para coincidir con la imagen que procesa el motor; `object-fit: cover` |
| 2 | `<canvas>` | Esqueleto dibujado con los `puntos` del estado. **No se espeja**, porque las coordenadas ya vienen de la imagen espejada |
| 3 | HTML | Etiquetas de cada mano (`Izq · D`, `Der · B`, con su mini barra) colocadas con la `caja` normalizada, sello de acierto, diagnóstico y avisos |

- **Alineación:** el video y las coordenadas salen de la misma cámara y del mismo tamaño de imagen, así que el esqueleto coincide sin ajustes. La proporción del área se toma del `tamano` del motor (`--ar`) y se ajusta al espacio disponible con unidades `cqw`/`cqh`.
- **Dibujo:** un bucle `requestAnimationFrame` redibuja solo cuando llega un estado nuevo (o cambia el tamaño o el tema). El lienzo usa la densidad real de la pantalla (`devicePixelRatio`), así que las líneas se ven nítidas.
- **Estilo del esqueleto:**
  - huesos en blanco translúcido con un halo (`shadowBlur`) del color de la mano: azul eléctrico la izquierda, verde esmeralda la derecha, tomados de `--izq` / `--der`;
  - articulaciones del mismo color, con las yemas más grandes y rodeadas por un aro;
  - el tamaño de cada punto crece cuando el punto está más cerca de la cámara (`z`);
  - al registrarse una seña, la mano se pinta en el verde de éxito;
  - la estela de un trazo se dibuja fina y tenue al inicio e intensa al final.
- **Latencia:** al dibujar cada estado nuevo se calcula `Date.now() - tiempos.captura_ms` (media móvil) y se muestra junto a los FPS, actualizada 4 veces por segundo.

#### 2.6.2 Actualización eficiente del DOM

El estado llega unas 30 veces por segundo. Las funciones `poner()` y `atributo()` solo escriben cuando el valor cambia, para no provocar recálculos de estilo innecesarios. Los estados visuales se expresan con atributos `data-*` (`data-fase`, `data-estado`, `data-modo`, `data-tono`) y el CSS decide colores y animaciones.

#### 2.6.3 Sistema de temas

- Todos los colores son **variables CSS**: `:root, [data-theme="dark"]` define el tema oscuro y `[data-theme="light"]` el claro.
- El **tema oscuro** es el predeterminado: fondo `#121212`, superficies `#1B1B1F`, acento verde esmeralda `#2EE59D` y azul eléctrico `#3B82F6`.
- El **tema claro** usa fondo `#F5F7FA`, superficies blancas y acentos más oscuros (verde `#059669`, azul `#2563EB`) para mantener el contraste.
- Un script en el `<head>` aplica el tema guardado **antes del primer pintado**, así que no hay destello del tema equivocado.
- El cambio se hace con el interruptor sol/luna del encabezado (`role="switch"`) o con la tecla `T`.
- El video y las etiquetas sobre él mantienen colores fijos en ambos temas.

#### 2.6.4 Persistencia en el navegador

| Clave de `localStorage` | Guarda |
|---|---|
| `lsm-tema` | `dark` o `light` |
| `lsm-vista` | Última pestaña (`traducir` o `practica`) |
| `lsm-mejor-racha` | Mejor racha del Modo Práctica |

Todas las lecturas y escrituras están protegidas con `try/catch`: si el navegador bloquea el almacenamiento (modo privado), la interfaz funciona igual y solo pierde la preferencia al recargar.

#### 2.6.5 Accesibilidad y diseño adaptable

- Los subtítulos y la retroalimentación de práctica usan `aria-live="polite"`, para que un lector de pantalla los anuncie.
- Pestañas con `role="tab"`, interruptor con `role="switch"` y foco visible en todos los controles.
- Con `prefers-reduced-motion` se desactivan animaciones, confeti y transiciones.
- Los atajos de teclado se ignoran mientras se escribe en un campo de texto.
- **Diseño adaptable** según el tamaño de la pantalla:

| Pantalla | Distribución |
|---|---|
| Ancho > 1000 px | Dos columnas: video a la izquierda y panel a la derecha |
| Altura ≤ 860 px | Panel compacto: anillo más chico y menos espacio entre tarjetas |
| Ancho ≤ 1000 px | Una sola columna |
| Ancho ≤ 620 px | Botones solo con icono |

---

## 3. Flujo de trabajo y experiencia de usuario (UX / UI)

### 3.1 Estabilización

`Estabilizador` (uno por mano): una configuración nueva solo se acepta si aparece en **4 fotogramas consecutivos**, lo que elimina el parpadeo de la etiqueta.

### 3.2 Constructor de palabras *Hold-to-Type*

| Característica | Detalle |
|---|---|
| Por mano | Cada mano tiene su propio contador: **cualquiera de las dos escribe** |
| Retención | Una seña **estática** se registra al sostenerla **quieta 1.5 s** (`time.monotonic()`, no depende de los FPS) |
| Dos manos, letras distintas | Se escriben ambas, en el orden en que se completan (`Izq: B` y `Der: O` → `BO`) |
| Dos manos, **la misma letra** | Se escribe **una sola vez**: mientras una mano la siga sosteniendo, la otra no la repite (con datos reales a veces se hacía la letra con ambas manos) |
| Anti-repetición | No se repite hasta cambiar de seña o bajar la mano |
| Señas con movimiento | Se registran **al completar el trazo** (la barra salta al 100 %) |
| Texto | Los espacios se ven como espacios y **cualquier guion bajo se reemplaza por un espacio** en pantalla, en la voz y en lo que se envía al hardware (`limpiar_texto`) |

### 3.3 Interfaz web

Una sola página con cuatro zonas:

| Zona | Contenido |
|---|---|
| **Encabezado** | Marca, estado de la voz (*Voz lista*, *Iniciando voz*, *Sin voz*), interruptor de tema sol/luna, botón de la guía del alfabeto y botón *Detener* |
| **Cámara** | Indicador *En vivo* y FPS; barra de herramientas (Espacio, Borrar, Limpiar, Recalibrar rostro, Diagnóstico); video en espejo con el esqueleto de cada mano, la **estela** del dedo mientras traza y una **etiqueta flotante por mano** (`Izq · D`, `Der · B`) con su mini barra; diagnóstico superpuesto (tecla `D`) |
| **Subtítulos** | Texto acumulado **centrado debajo del video**, en tipografía grande y con cursor parpadeante. Lo último que se agrega entra con un desplazamiento y un fundido breve en verde |
| **Panel lateral** | Pestañas *Traducir* / *Práctica*; tarjetas de cada mano (seña, estado y progreso); tarjeta *Rostro* (icono y estado, barra de calibración o de pregunta, medidor de cejas contra el umbral y ángulos de cabeceo, giro e inclinación) |

**Pestaña Traducir:** la seña de la **mano principal** dentro de un **anillo de progreso SVG** que representa el hold-to-type, con la mano que la hace, un mensaje de estado (*Mantén la seña quieta · 64 %*, *Registrada*, *Mueve la mano para escribir J*) y la tarjeta *Cómo se hace*, con la descripción, el movimiento o la pista de trazo.

El anillo cambia de color según la fase de retención:

| Progreso | Fase | Color | Efecto |
|---|---|---|---|
| 0-49 % | Inicio | Azul claro | — |
| 50-99 % | Procesando | Ámbar | La letra se tiñe de ámbar |
| 100 % (registrada) | Éxito | Verde | Pulso de escala y brillo (`box-shadow`) alrededor del anillo |
| Configuración que necesita trazo | Esperando movimiento | Gris, trazo punteado | — |

**Guía del alfabeto** (botón del libro o tecla `G`): panel lateral con las 30 señas; las que llevan movimiento se marcan en ámbar. Se resalta la que se está haciendo, y al tocar una se abre directamente en el Modo Práctica.

Los botones, tarjetas, colores y estados usan `transition: all 0.3s ease`. Las excepciones son las etiquetas que siguen a la mano y las barras de progreso: tienen transiciones de 90-150 ms, porque con 0.3 s quedarían visiblemente atrasadas respecto a la mano.

### 3.4 Modo Práctica

**Implementado.** Herramienta de aprendizaje: la persona elige qué practicar y la interfaz valida sus intentos con el mismo reconocimiento que usa el traductor.

| Elemento | Detalle |
|---|---|
| **Letras** | Tira desplazable con las 30 señas (las de movimiento en ámbar), flechas anterior/siguiente y letra al azar. Las acertadas en la sesión llevan un punto verde |
| **Palabras** | 10 palabras sugeridas (`HOLA`, `GRACIAS`, `AMIGO`, `CASA`, `AGUA`, `NIÑO`, `FELIZ`, `JUEGO`, `QUESO`, `KIWI`) o una propia de hasta 12 letras (se quitan acentos y se conserva la Ñ). Se deletrea letra por letra; unas fichas muestran las letras hechas, la actual y las pendientes |
| **Tarjeta de referencia** | Espacio reservado para la imagen de la seña (aún sin imágenes), letra, tipo (*Estática* o *Con movimiento*), descripción y movimiento o tiempo de retención |
| **Marcador** | Aciertos y racha de la sesión, y mejor racha (guardada en el navegador) |

**Validación** (`evaluarPractica` en `app.js`, sobre cada mensaje `estado`):

- **Acierto:** una mano **pasa a `registrada`** con la seña objetivo. Se detecta el flanco (de otro estado a `registrada`), así que sostener la seña no cuenta varias veces; para repetirla hay que deshacerla y volver a hacerla, igual que en el traductor.
- **Fallo:** se registra **otra letra**. La racha vuelve a cero y la tarjeta se sacude. `ESPACIO` y `BORRAR` no cuentan como fallo, porque la mano abierta es una postura de reposo frecuente.
- Durante una celebración (1.8 s; 2.3 s al completar una palabra) no se evalúan registros nuevos.

**Retroalimentación en vivo**, en cada fotograma:

| Situación | Mensaje | Indicador |
|---|---|---|
| Ninguna mano a la vista | *Muestra tu mano a la cámara* | Gris |
| Alguna mano hace la seña objetivo | *¡Esa es! Mantenla quieta…* y barra con las fases del anillo | Verde; el borde de la cámara se tiñe de verde |
| Alguna mano hace la **base** de una seña con movimiento (p. ej. la I cuando se practica la J; viene de `bases` en el mensaje `config`) | *Bien, I es la base: ahora haz el movimiento* | Ámbar |
| Otra seña | *Detecto «B». Compárala con la referencia* | Rojo |

**Celebración de un acierto:**

- La tarjeta de referencia **gira** (`rotateY`, 0.6 s) y muestra una palomita que se dibuja.
- Salen **28 partículas de confeti** con los colores del tema.
- El borde de la cámara se ilumina en verde y aparece un **sello** con palomita sobre el video.
- En una palabra, cada letra correcta marca su ficha y se pasa a la siguiente; al completarla gira la tarjeta.

> El modo práctica comparte el registro con el traductor: **cada letra acertada también se escribe en los subtítulos** (y se dice en voz alta). Con Limpiar (`C`) se vacía el texto.

### 3.5 Teclado y controles

Los atajos funcionan con la ventana del navegador activa, salvo mientras se escribe en un campo de texto.

| Tecla | Botón | Acción |
|---|---|---|
| `Espacio` | Espacio | Agregar un espacio |
| `Backspace` | Borrar | Borrar la última letra |
| `C` | Limpiar | Limpiar todo el texto |
| `D` | Diagnóstico | Mostrar u ocultar el diagnóstico sobre el video |
| `N` (o `R`) | Recalibrar | Recalibrar la cara neutra |
| `G` | Libro | Abrir o cerrar la guía del alfabeto (`Esc` la cierra) |
| `T` | Sol/luna | Cambiar entre tema claro y oscuro |
| — | Detener (dos clics) | Cerrar el servidor, liberar la cámara y terminar el programa (también Ctrl+C en la terminal) |

---

## 4. Integración de hardware

> **Estado:** el **puente de software está implementado**: `enviar_a_hardware(texto_actual)` se llama **solo cuando el texto cambia** (por una seña de cualquier mano, por el rostro o por el teclado) e imprime `[SERIAL MOCK] Enviando al ESP32: <texto>`. El **hardware físico** (ESP32 y pantalla) está en fase de **diseño**. Para conectarlo basta con reemplazar el cuerpo de esa función por el envío serial de la sección 4.2.

### 4.1 Arquitectura propuesta

```
┌─────────────────────────────┐                 ┌───────────────────────────────────┐
│     PC (visión artificial)  │                 │    Protoboard                     │
│                             │   USB / UART    │  ┌─────────────┐   I²C   ┌───────┐ │
│  main.py                    │ ──────────────▶ │  │ ESP32 /     │ ──────▶ │ OLED  │ │
│   └─ enviar_a_hardware()    │   115200 baud   │  │ Arduino Uno │         │128×64 │ │
│                             │   ── o bien ──  │  └─────────────┘         └───────┘ │
│   └─ WebSocket (ESP32)      │ ~~~ Wi-Fi ~~~~▶ │        ▲ (solo ESP32)              │
└─────────────────────────────┘                 └───────────────────────────────────┘
```

| Componente | Opción recomendada | Alternativa | Función |
|---|---|---|---|
| Microcontrolador | **ESP32 DevKit** (Wi-Fi + Bluetooth, 240 MHz) | Arduino Uno / Nano (solo serial) | Recibe el texto y controla la pantalla |
| Pantalla | **OLED SSD1306 de 0.96" (128×64, I²C)** | LCD 16×2 con adaptador I²C (PCF8574) | Muestra la letra y la palabra |
| Conexión | Cable USB (alimentación + datos) | Wi-Fi (solo ESP32) | Canal PC → dispositivo |
| Prototipado | Protoboard + cables Dupont | — | Montaje sin soldadura |
| Autonomía (fase posterior) | Batería LiPo / 18650 + módulo de carga TP4056 | Power bank USB | Portabilidad |

**Conexiones (ESP32 ↔ OLED SSD1306):**

| OLED | ESP32 | Arduino Uno |
|---|---|---|
| VCC | 3V3 | 5V |
| GND | GND | GND |
| SDA | GPIO 21 | A4 |
| SCL | GPIO 22 | A5 |

### 4.2 Protocolo de comunicación

Protocolo de texto línea por línea, sencillo de depurar con cualquier monitor serial:

| Mensaje | Significado | Ejemplo |
|---|---|---|
| `T:<texto>\n` | Texto acumulado completo | `T:HOLA LSM` |
| `C\n` | Limpiar la pantalla | `C` |

**Reglas de diseño:**
- Se envía **solo cuando algo cambia**, no en cada frame.
- Se envía el **texto completo** en lugar de diferencias: si se pierde un mensaje, el siguiente corrige el estado.
- Se usa `\n` como delimitador, compatible con `Serial.readStringUntil('\n')` en Arduino.

**Envío desde la PC (propuesta con `pyserial`):**

```python
import serial  # pip install pyserial

puerto = serial.Serial("COM3", 115200, timeout=0)
puerto.write(f"T:{limpiar_texto(texto_actual)}\n".encode("utf-8"))
```

**Recepción en el ESP32 (propuesta en Arduino/C++):**

```cpp
#include <Wire.h>
#include <Adafruit_SSD1306.h>

Adafruit_SSD1306 oled(128, 64, &Wire, -1);

void setup() {
  Serial.begin(115200);
  oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  oled.clearDisplay(); oled.display();
}

void loop() {
  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
    if (msg.startsWith("T:")) {
      oled.clearDisplay();
      oled.setTextSize(2); oled.setTextColor(SSD1306_WHITE);
      oled.setCursor(0, 0);
      oled.print(msg.substring(2));
      oled.display();
    } else if (msg == "C") {
      oled.clearDisplay(); oled.display();
    }
  }
}
```

> **Dos notas para la etapa de hardware:**
> 1. La fuente por defecto de Adafruit GFX **no tiene la Ñ**. Con la librería `U8g2` y una fuente Latin-1 (por ejemplo `u8g2_font_6x12_tf`) sí se puede mostrar.
> 2. Un ESP32 **no puede ejecutar MediaPipe**: la visión sigue en la PC (o en una Raspberry Pi o un teléfono) y el microcontrolador solo muestra el texto.

### 4.3 Variante inalámbrica (Wi-Fi, solo ESP32)

| Aspecto | Propuesta |
|---|---|
| Rol del ESP32 | Servidor WebSocket en la red local (o *access point* propio) |
| Rol de la PC | Cliente que envía los mismos mensajes |
| Ventaja | El dispositivo no necesita cable de datos: solo alimentación |
| Riesgo | Latencia variable de la red; conviene usar un *timeout* y reintentos |

### 4.4 Camino hacia la autonomía total

```
 Etapa 1 (actual)         Etapa 2 (próxima)             Etapa 3 (portátil)
 ┌──────────────┐         ┌──────────────────────┐      ┌──────────────────────────┐
 │ PC: visión + │   ──▶   │ PC: visión           │ ──▶  │ SBC (Raspberry Pi 5)     │
 │ pantalla     │         │ ESP32: pantalla OLED │      │ o teléfono: visión       │
 └──────────────┘         └──────────────────────┘      │ ESP32 + batería: pantalla│
                                                        └──────────────────────────┘
```

---

## 5. Roadmap y futuro del proyecto

| Fase | Entregable | Estado |
|---|---|---|
| **1. Prototipo manual** | Letras estáticas, hold-to-type, interfaz dual | Completado |
| **2. Componentes no manuales** | Pregunta, negación y afirmación por rostro y cabeza | Completado |
| **3. Puente de hardware** | `enviar_a_hardware()` (mock) | Completado |
| **4. Alfabeto completo** | 21 letras estáticas + J, K, Ñ, Q, X, Z con movimiento | Completado |
| **5. Voz** | Cada letra y la palabra completa, en un hilo aparte | Completado |
| **6. Dos manos y geometría 3D** | Lateralidad, estado por mano, espejo canónico, One-Euro, rasgos 3D, `solvePnP`, cara tapada | Completado (P/R y algunos trazos con fiabilidad media, ver sección 8) |
| **6.1 Modularización** | `main.py` monolítico dividido en módulos, con equivalencia verificada (sección 11.2) | Completado |
| **6.2 Arquitectura web** | Servidor FastAPI + WebSocket, interfaz web con tema claro/oscuro y Modo Práctica (secciones 2.5, 2.6 y 3.4) | Completado |
| **7. Hardware físico** | Pantalla OLED en protoboard por USB o Wi-Fi | Diseñado |
| **8. Machine Learning** | Clasificador entrenado con landmarks propios (k-NN, Random Forest, red densa) para los pares ambiguos (P/R, D/X, ESPACIO/mano relajada) | Planeado |
| **9. Señas dinámicas con LSTM / GRU** | Reemplazar las reglas de trazo; añadir CH, LL y RR | Planeado |
| **10. Frases continuas** | Modelos secuenciales / Transformers | Planeado |
| **11. Impacto social** | App móvil o dispositivo cerrado; validación con personas usuarias de LSM | Planeado |

### 5.1 Hacia el aprendizaje automático

El clasificador de reglas es explicable, pero escala mal: cada letra nueva exige diseñar reglas a mano, y la sección 8 muestra que la forma de hacer algunas letras varía entre sesiones más que la diferencia entre ellas. La evolución natural es **aprender las fronteras de decisión a partir de datos**:

```
 Captura de datos        Vector de rasgos                 Modelo                Salida
 ┌──────────────┐       ┌───────────────────────┐       ┌───────────────┐      ┌─────────┐
 │ sesiones     │  ──▶  │ rasgos 3D de la mano  │ ──▶   │ k-NN / RF     │ ──▶  │ letra   │
 │ guiadas      │       │ canónica (sección 2.1)│       │ (estáticas)   │      │         │
 └──────────────┘       └───────────────────────┘       ├───────────────┤      ├─────────┤
                                                        │ LSTM sobre    │ ──▶  │ seña    │
                        secuencia de T frames ──▶       │ T × rasgos    │      │ dinámica│
                                                        └───────────────┘      └─────────┘
```

Las **sesiones guiadas** que se usaron para calibrar (sección 8.1) ya son una herramienta de **captura de datos**: graban solo coordenadas de ambas manos y del rostro. Con varias personas se podría entrenar un clasificador sobre los rasgos 3D, mucho más ligero que una CNN sobre imágenes.

### 5.2 Impacto social

- **Aplicación móvil:** MediaPipe tiene soporte nativo para Android e iOS, así que el mismo pipeline puede llevarse al teléfono.
- **Herramienta educativa:** el Modo Práctica (sección 3.4) ya permite practicar letras y deletrear palabras con retroalimentación inmediata. Faltan imágenes de referencia de cada seña y estadísticas persistentes por persona.
- **Validación con la comunidad:** pruebas con personas usuarias de LSM para corregir la forma de las señas y priorizar el vocabulario.

---

## 6. Guía de instalación y ejecución

### 6.1 Requisitos

| Requisito | Versión / detalle |
|---|---|
| Sistema operativo | Windows 10/11 |
| Python | **3.12** (MediaPipe 0.10.18 **no** es compatible con 3.13) |
| Hardware | Cámara web integrada o USB |
| Navegador | Chrome, Edge o Firefox recientes (usa *container queries* y `color-mix()`). No pide permiso de cámara: el video lo transmite el servidor |
| Puerto | `5000` libre en `127.0.0.1` |
| Voz | Una voz SAPI5 en español (en Windows: *Microsoft Sabina* o *Helena*). Si falta, el programa funciona sin voz |
| Editor | VS Code con la extensión **Python** (opcional) |

### 6.2 Dependencias

| Librería | Capa | Uso |
|---|---|---|
| `mediapipe==0.10.18` | Motor | Detección de las manos (Hands, con coordenadas de mundo y lateralidad) y del rostro (Face Mesh) |
| `opencv-python` | Motor | Captura de video (hilo `CamaraReciente`), espejo, conversión de color y **`solvePnP`** |
| `numpy` | Motor | Geometría 3D, filtro One-Euro y trayectorias |
| `pyttsx3` | Motor | Voz (en Windows instala también `comtypes` y `pywin32`) |
| `fastapi` | Servidor | Rutas HTTP, archivos estáticos y WebSocket |
| `uvicorn` | Servidor | Servidor ASGI |
| `websockets` | Servidor | Implementación del protocolo WebSocket que usa uvicorn |

> El frontend no requiere instalar nada: HTML, CSS y JavaScript sin dependencias. El filtro One-Euro y la pose de la cabeza (`cv2.solvePnP`) tampoco requieren librerías extra; las coordenadas del modelo canónico de la cara están escritas en `classifier_face.py`.

### 6.3 Instalación

```powershell
# 1. Clonar el repositorio
git clone https://github.com/TheCubick/Proyecto-Lenguaje-de-se-a.git
cd Proyecto-Lenguaje-de-se-a

# 2. Crear el entorno virtual con Python 3.12
py -3.12 -m venv venv

# 3. Instalar dependencias
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install opencv-python mediapipe==0.10.18 numpy pyttsx3 fastapi uvicorn websockets

# 4. Verificar que todo importa
.\venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'python'); import main, server; print('OK')"
```

> La carpeta `venv/` está en `.gitignore`: cada integrante crea la suya localmente y no se sube al repositorio.

### 6.4 Ejecución

```powershell
.\venv\Scripts\python.exe python\server.py
```

1. La terminal muestra `Traductor LSM en http://localhost:5000` y, un segundo después, se abre esa dirección en el **navegador predeterminado**. Si no se abre, escríbela a mano.
2. Permite que la cámara se encienda; el indicador cambia a **En vivo**.
3. Mantén la cara neutra unos segundos mientras la tarjeta *Rostro* dice **Calibrando**.
4. Para terminar, pulsa **Detener** dos veces en la interfaz o **Ctrl+C** en la terminal.

**Con VS Code:** pulsa **`F5`** y elige **"Traductor LSM (server.py)"**.

| Archivo | Qué configura |
|---|---|
| `.vscode/launch.json` | Ejecuta `python/server.py` con `venv\Scripts\python.exe` directamente, sin depender de activar el entorno |
| `.vscode/settings.json` | Fija `venv` como intérprete del proyecto y crea un perfil de terminal **"PowerShell (venv)"** con `ExecutionPolicy Bypass`, para que Windows no bloquee `Activate.ps1` |

`python python/main.py` también funciona: delega en `server.main()`.

**Solución de problemas:**

| Síntoma | Causa probable | Solución |
|---|---|---|
| La interfaz dice *El motor no pudo abrir la cámara* | Otra aplicación (Teams, Zoom, otra instancia) tiene la cámara | Cerrarla y volver a ejecutar |
| Etiqueta *solo esqueleto* sobre un fondo oscuro | No llega el video de `/video_feed` (la cámara no abrió o el servidor se reinició) | Recargar la página; si persiste, revisar la terminal |
| `error while attempting to bind on address ('127.0.0.1', 5000)` | Otra instancia del servidor sigue abierta | Cerrarla, o cambiar `PUERTO` en `server.py` |
| *Reconectando* en el indicador | El servidor se detuvo o se reinició | El navegador reintenta cada segundo; volver a ejecutar `server.py` |
| La tipografía se ve distinta | Sin conexión a Google Fonts | No afecta el funcionamiento; se usa la fuente del sistema |

### 6.5 Recomendaciones de uso

- Iluminación frontal y uniforme; evita tener una ventana detrás.
- Manos completas dentro del cuadro, a 40-80 cm de la cámara.
- **Con dos manos:** tenlas a la altura del pecho, separadas y **sin tapar la cara**. Con las manos juntas delante de la cara, MediaPipe llegó a no detectar ninguna (sección 8.3).
- Al iniciar, **mantén la cara neutra** unos 1.5 s (la tarjeta *Rostro* muestra **Calibrando**); `N` recalibra.
- **Los trazos (Z, ?, J…) deben ser pequeños, del tamaño de la mano**, sin sacarla del cuadro. Los movimientos muy rápidos pueden hacer que la cámara pierda la mano.
- Sostén cada letra **quieta** hasta que la barra llegue al 100 %.

> Los mensajes `W0000 … inference_feedback_manager` que aparecen en la consola son avisos internos de MediaPipe, no errores.

---

## 7. Calibración

Todos los umbrales están al inicio de su módulo (`classifier_hands.py`, `classifier_face.py`, `traductor.py`, `voice_engine.py`), cada uno con el rango medido que lo justifica. El **modo diagnóstico** (tecla `D`) muestra los valores en vivo.

| Constante | Valor | Controla |
|---|---|---|
| `SEGUNDOS_PARA_ESCRIBIR` | 1.5 s | Retención de una seña estática |
| `FRAMES_ESTABLES` | 4 | Consenso para aceptar una seña |
| `V_QUIETO` | 1.2 manos/s | Por encima, la mano "se mueve" y no escribe letras estáticas |
| `ONE_EURO_CORTE_MIN` / `ONE_EURO_BETA` | 1.5 Hz / 0.4 | Filtro antivibración |
| `CONTINUIDAD_DIST` / `LATERALIDAD_CAMBIO` | 1.2 / 0.85 | Seguimiento de cada mano en su ranura |
| `MAX_MANOS` | 2 | Manos que busca MediaPipe (1 = más FPS en una PC lenta) |
| `EXT_RECTO` / `EXT_CERRADO` | (1.40, 1.54, 1.45, 1.25) / 1.15 | Extensión 3D de un dedo recto (índice, medio, anular, meñique) / cerrado |
| `FLEX_MAX_RECTO` / `ARTICULACION_RECTA` / `EXT_TOLERANCIA` | 75° / 55° / 0.15 | Ángulos articulares de un dedo recto |
| `MENIQUE_RECTO` | 50° | Meñique recto aunque el nudillo lo incline (Y de lado) |
| `PULGAR_FUERA` | −0.45 | Pulgar por fuera del nudillo del índice (A, L, G, Y, ESPACIO) |
| `T_ALTURA` / `E_MCP_MAX` | 0 / 33° | T contra S, y E contra S |
| `F_CONTACTO` / `F_FLOJA` | 0.55 / (0.75, 1.30) | Pinza de la F |
| `O_CONTACTO` / `C_GAP` / `C_INDICE_MIN` | 0.32 / 0.32-1.10 / 1.05 | O y C |
| `V_CRUCE` / `R_CRUCE` / `U_ANG_MAX` | 0.40 / −0.05 / 11° | V, R, U (el resto, P) |
| `ESPACIO_PULGAR` / `Y_DIST` | 1.00 / 0.75 | ESPACIO y Y |
| `X_DIR_MIN` / `X_ORIENT_MIN` / `Q_DIR_INDICE_MIN` | 60° / 48° / 105° | D, X y Q |
| `VAIVEN_AMP` | Ñ 0.07 · Q/X 0.07 · K 0.15 | Amplitud mínima de cada vaivén |
| `VAIVEN_RECTITUD` / `VAIVEN_PROPORCION` | 0.5 / 2.5 | Movimiento coherente de los vaivenes |
| `BORRAR_RECORRIDO_MIN` / `BORRAR_VEL_MIN` | 1.10 / 3.0 | Barrido de BORRAR |
| `CARA_ERROR_MAX` | 1.8 % | Cara tapada |
| `CEJAS_UMBRAL` / `CEJAS_UMBRAL_SALIDA` / `CEJAS_MAX_SEGUNDOS` | 1.07 / 1.04 / 4 s | Cejas arriba / de vuelta / nueva cara neutra |
| `NEG_AMPLITUD` / `AFIRM_AMPLITUD` | 8° / 5° | Negar (yaw) y asentir (pitch) |
| `ACCION_NEGACION` | `"LETRA"` | Qué borra negar: `"LETRA"`, `"PALABRA"` o `"TODO"` |
| `VOZ_INACTIVIDAD` | 4.0 s | Tiempo sin letras para leer la palabra pendiente |

**Procedimiento:** pulsa `D`, haz la seña que falla, compara los valores de esa mano con la tabla de la sección 2.1.6 y ajusta la constante correspondiente. Lo más fiable es grabar una sesión guiada y medir (sección 8.1).

---

## 8. Validación con datos reales

### 8.1 Metodología

Se grabaron **nueve sesiones guiadas** con la cámara real del autor. Una ventana mostraba la **lámina oficial** de cada letra (espejada para la mano izquierda), qué mano usar, una animación del trazo y un aviso en vivo si la mano detectada no era la pedida o si no se veían las dos manos. Se guardaron **solo coordenadas** (manos en imagen y en 3D de mundo, lateralidad, 468 puntos del rostro), nunca imágenes; se borraron al terminar.

Cada grabación se **reprodujo con el `Traductor` real (`traductor.py`)** (seguidor de manos, One-Euro, clasificador, estabilizador, trayectorias, hold-to-type y rostro), así que los resultados son los del programa, no de una copia.

| # | Sesión | Uso | Duración | Fotogramas |
|---|---|---|---|---|
| 1 | Letras estáticas (pedida con la derecha; de la F en adelante se hizo con la izquierda) | calibración | 104 s | 2 700 |
| 2 | Letras estáticas, mano izquierda | calibración | 104 s | 2 645 |
| 3 | Letras estáticas, mano derecha | calibración | 104 s | 2 694 |
| 4 | Trazos con cada mano + acomodar la mano + movimiento libre | calibración | 175 s | 4 376 |
| 5 | Una mano escribe y la otra a la vista, cruzar manos; rostro (cejas, asentir, negar, mirar arriba/abajo, girar, lejos) | calibración | 154 s | 3 240 |
| 6 | Dos manos a la vez (letras distintas, trazo con una mientras la otra sostiene) | calibración | 59 s | 961 |
| 7 | **Prueba:** letras estáticas, derecha e izquierda | prueba | 190 s | 2 986 |
| 8 | **Prueba:** trazos con cada mano + rostro | prueba | 181 s | 3 093 |
| 9 | **Prueba final:** letras estáticas, derecha e izquierda | prueba independiente | 190 s | 2 919 |

> **Cómo leer las cifras.** Las sesiones 1-6 se usaron para diseñar y calibrar, así que sus resultados son "de entrenamiento". Las sesiones 7, 8 y 9 se midieron primero con el **código congelado** antes de verlas: esas son las **mediciones independientes**. Después de analizar sus fallos se hicieron ajustes con motivo físico; las cifras posteriores sobre esas sesiones ya **no** son independientes y se reportan aparte, marcadas como "versión final". Son datos de **una sola persona**, grabados el mismo día.

### 8.2 Lateralidad

| Medida | Resultado |
|---|---|
| Etiqueta de MediaPipe con la imagen espejada | coincide con la mano real; confianza 0.94-1.00 |
| Duplicados de la misma mano y parpadeos de etiqueta | ocurrieron en las sesiones 1 y 6; los resuelve `SeguidorManos` (28 pruebas de lógica, sección 8.10) |
| Dos manos a la vez **con la guía** (manos al pecho, separadas, sin tapar la cara) | **ambas detectadas el 100 %** de los fotogramas en todas las fases |
| Dos manos juntas **delante de la cara** (sesión 5) | MediaPipe no detectó **ninguna** mano durante ~25 s |

### 8.3 Dos manos en escena

| Prueba | Resultado |
|---|---|
| Letras distintas con cada mano a la vez (B-O, L-V, Y-D, W-A, F-R, I-U) | **15 de 15 letras escritas correctamente**, cada una por su mano |
| Izquierda sostiene B mientras la derecha dibuja Z; derecha sostiene A mientras la izquierda hace la J | B y A escritas; Z y J reconocidas en su mano |
| Ambas manos relajadas a la vista | **no escribieron nada** |
| La misma letra hecha con ambas manos (A, B, L, V) | **una sola letra** por seña (antes de la regla se escribía dos veces) |
| BORRAR con la derecha mientras la izquierda sostiene L | **no reconocido**: en el barrido rápido MediaPipe perdió la mano |
| Precisión por fotograma con ambas manos a la vez | izquierda 99.7 %; derecha 80.0 % (R y U bailaban), aun así escritas bien |

### 8.4 Letras estáticas

**Calibración** (sesiones 2 y 3, versión final; exactitud por fotograma en los últimos 2.5 s de cada fase):

| Mano | Exactitud media | Letras ≥ 95 % | Escritas de punta a punta |
|---|---|---|---|
| Derecha | 99.5 % | 21 de 22 | **22 de 22** |
| Izquierda | 95.3 % | 18 de 22 | **22 de 22** |

**Mediciones independientes** (cada sesión de prueba medida con el código congelado antes de verla):

| Sesión | Mano | Exactitud media | Letras ≥ 95 % | Por debajo | Escritas de punta a punta | Versión anterior (2D) |
|---|---|---|---|---|---|---|
| 7 | Derecha | **94.4 %** | 17 de 22 | P 71 (sale R), S 74 (sale T), G 82, R 87, ESPACIO 62 | **35 de 44** | 80.3 % |
| 7 | Izquierda | **79.8 %** | 14 de 22 | Y 0, P 24 y R 26 (intercambiadas), F 44, G 41, M 59, S 81, ESPACIO 81 | | |
| 9 | Derecha | **85.6 %** | 17 de 22 | C 6 (sale I), R 10 (sale P), L 39 (seguía la I anterior), N 53, E 74 | **41 de 44** | 79.6 % |
| 9 | Izquierda | **96.2 %** | 20 de 22 | G 48 (sale A), ESPACIO 73 | | |

**Análisis de los fallos.** Casi todos fueron **variación de ejecución entre sesiones**: en la sesión 7 la P izquierda se hizo con las puntas cruzadas y la palma de frente (como la R de la calibración) y la R casi sin cruzar; la Y con índice y medio menos cerrados; la F con la pinza floja; la S con el pulgar más alto; el ESPACIO con el pulgar menos abierto. En la 9, la C derecha fue más cerrada (casi una O), la N y la E quedaron con el índice justo en la frontera de "recto" y la G izquierda con el índice doblado. Con esa variación se hicieron ajustes con motivo físico: I/Y con índice y medio "no rectos", F con pinza floja si el índice está claramente doblado, frontera S/T en el punto medio, ESPACIO con el pulgar a ≥ 1.00, **dedo recto por ángulos articulares** (separa la N de la E), C con el índice más cerrado y la "pistola" con el índice largo aunque doblado (G).

**Versión final** (ajustada con todos los datos, **ya no independiente**):

| Sesión | Mano | Exactitud media | Letras ≥ 95 % | Escritas de punta a punta |
|---|---|---|---|---|
| 3 (calibración) | Derecha | 99.5 % | 21 de 22 | 22 de 22 |
| 2 (calibración) | Izquierda | 95.3 % | 18 de 22 | 22 de 22 |
| 7 | Derecha / Izquierda | 96.0 % / 91.6 % | 18 / 18 de 22 | 40 de 44 (fallan P y R) |
| 9 | Derecha / Izquierda | 93.0 % / 98.4 % | 20 / 20 de 22 | **43 de 44** (falla la R derecha, que sale P) |

Ninguna fase de mano relajada escribió letras ni espacios.

### 8.5 Señas con movimiento

| Medida | Calibración (sesión 4), versión final | Prueba (sesión 8), independiente | Prueba (sesión 8), versión final |
|---|---|---|---|
| Trazos reconocidos (fases con al menos un acierto) | **16 de 16** | **11 de 16** | 13 de 16 |
| Fallos | — | Q y X derecha; K, Z y BORRAR izquierda | Q derecha; K y Z izquierda |
| Trazos falsos | 1 (`?` dentro de una Z izquierda) | 5 | 5: Z dentro de la X derecha, `?` en la Z izquierda (2), K al acomodar la izquierda (2) |
| Acomodar la mano entre U, V, R, P y H | **0 K falsas** con cada mano | | 2 K falsas (izquierda) |

En la prueba, la Q derecha casi no se vio (25 fotogramas con mano); la K izquierda fue un vaivén más chico (< 0.27 manos) o vertical; la Z izquierda se dibujó con forma de `?`. BORRAR izquierda y X derecha se recuperaron con las ventanas en segundos y la regla de D/X.

### 8.6 Rostro

Se corrió el `AnalizadorFacial` real sobre **todo** el tiempo con cara visible, incluido el que se pasó haciendo letras con las manos (que es cuando más falsos positivos aparecen):

| Medida (las 9 sesiones, 19.4 min con cara visible) | Versión nueva | Versión anterior (2D) |
|---|---|---|
| Pregunta con cejas | 3 de 7 (las demás subidas fueron mínimas o de < 1.5 s) | 3 de 7 |
| Negar | 2 de 2 | 2 de 2 |
| Asentir | 2 de 2 | 2 de 2 |
| **Eventos falsos** | **11** (8 preguntas, 3 asentimientos, 0 negaciones) | **100** (74 asentimientos, 5 negaciones, 21 preguntas) |

Las sesiones de prueba por separado (6-9, sin ajustar el rostro después de verlas): 5 falsos en la sesión 9 y 2 en las 6-8.

Lo que más redujo los falsos: ángulos reales de `solvePnP`, tres cabeceos para asentir, el filtro de cara tapada (error de reproyección) y la regla de "más de 4 s arriba = nueva cara neutra". **Lo que queda son preguntas falsas mientras se hacen letras** (≈ 1 cada 2.5 min): al mirar la mano o la pantalla, las cejas suben de verdad más de 10 % sin que la cabeza se mueva. Subir el umbral a 1.10 solo las bajaría de 8 a 5 y perdería una pregunta real, así que se dejó en 1.07. En una prueba en vivo de 25 s también aparecieron 2 `?` así.

### 8.7 Filtro One-Euro (con y sin filtro, mismas grabaciones)

| Sesión (versión final) | Sin filtro | Con filtro |
|---|---|---|
| 2 (izquierda) | 21 de 22 escritas; una Ñ falsa al sostener la H | **22 de 22**; sin trazos falsos |
| 3 (derecha) | 22 de 22; 98.8 % por fotograma | **22 de 22**; 99.5 % |
| 1 (derecha A-E, izquierda F-Y) | 20 de 22; un `?` falso al sostener la E; derecha 78.7 % | **21 de 22**; sin trazos falsos; derecha 85.2 % |
| 9 (prueba final) | 41 de 44; una J falsa al sostener la Y; izquierda 96.5 %, derecha 92.6 % | **43 de 44**; sin trazos falsos; izquierda 98.4 %, derecha 93.0 % |
| 4 (trazos) | 15 de 16 trazos | **16 de 16** |

### 8.8 Normalización de simetría

Al reflejar la mano izquierda, **las mismas reglas** clasifican ambas manos sin reglas separadas. Prueba de espejo con datos reales: cada mano de las sesiones 2, 3 y 9 se reflejó (imagen y mundo) y se le dio la etiqueta de la otra mano:

| Medida | Resultado |
|---|---|
| Clasificación por fotograma igual entre la mano y su espejo | **8 336 de 8 336 (100 %)** |
| Texto escrito de punta a punta por el `Traductor` completo (seguidor, filtro, hold-to-type) | **idéntico** en las tres sesiones |

### 8.9 Rendimiento

| Medida | Resultado |
|---|---|
| `Traductor.paso` (dos canales de mano, seguidor, clasificación 3D, trazos y rostro con `solvePnP`) | **3.3 ms** de mediana (p95 6.8 ms) |
| MediaPipe (2 manos + rostro), laptop **conectada** (sesiones 1-4) | ≈ 25 ms por fotograma → **25-26 FPS** de captura |
| MediaPipe solo, laptop **en batería** (40 %) | manos 37 ms con `max_num_hands=1` y **76 ms con 2** (con una mano a la vista busca la segunda en cada frame); rostro 7-11 ms |
| Motor completo en vivo, en batería (versión con ventanas de OpenCV) | **17 FPS** (60 ms de mediana por fotograma), 400 fotogramas, teclas simuladas, cierre limpio |
| Arquitectura web en vivo, laptop conectada, sin manos a la vista | **30.2 estados/s**, 0.54 KB por estado, captura → cliente 15.3 ms de mediana, proceso 14.9 ms (sección 2.5.6) |
| Voz | no bloquea el bucle (`decir()` regresa al instante; hilo propio) |

En la arquitectura web el motor no dibuja ni codifica imágenes: solo procesa y manda coordenadas. Toda la presentación la hace el navegador. El costo principal es MediaPipe, no el código del proyecto. En una PC lenta, `PROCESAR_ROSTRO_CADA = 2` o `MAX_MANOS = 1` recuperan FPS. Los trazos se miden en segundos y el hold-to-type con `time.monotonic()`, así que el reconocimiento no cambia con los FPS.

### 8.10 Pruebas automáticas

| Batería | Resultado |
|---|---|
| **28 pruebas de lógica**: hold-to-type por mano, misma letra con ambas manos, bloqueo tras un trazo, seguidor de manos (duplicados, parpadeo, corrección, orden), espejo canónico, One-Euro (temblor y retraso), voz, guiones bajos, coherencia de vaivenes | **28 de 28** |
| Reproducción completa de las 9 sesiones con el `Traductor` real | resultados de las secciones 8.2-8.8 |
| Servidor web en vivo con la cámara real, un cliente WebSocket y un cliente de `/video_feed` a la vez | 243 estados en 8 s sin saltos; 227 JPEG válidos a 30.1 FPS; 0 fotogramas descartados; cierre limpio con `SALIR` en 0.51 s con el video abierto |
| Coordenadas con dos manos sintéticas por `Traductor.paso` → `estado_web` | 21 puntos por mano en la posición esperada; estela de 20 puntos con la mano en movimiento; 1.57 KB por estado |
| Revisión estática | `pyflakes` sin avisos en todos los módulos y `node --check` sin errores en `app.js` |

> La interfaz web (tema, Modo Práctica, animaciones) se revisa manualmente en el navegador; todavía no tiene pruebas automáticas de interfaz.

---

## 9. Limitaciones conocidas

| Limitación | Detalle | Mitigación |
|---|---|---|
| **P y R** | En la lámina ambas cruzan índice y medio; entre sesiones se hicieron de forma distinta (en la prueba quedaron intercambiadas en la izquierda) | Hacerlas siempre igual: R con las puntas entrelazadas, P con el medio cruzando por delante. Candidatas a un clasificador entrenado |
| **D / X y la Q** | La X se separa de la D por la inclinación de la palma y del índice; si la palma queda vertical, se lee D | Inclinar la palma al hacer la X |
| **ESPACIO y mano relajada** | Se separan por cuánto se abre el pulgar; si se abre poco, es mano relajada y no escribe | Abrir bien el pulgar |
| **Trazos con movimiento muy rápido** | MediaPipe puede perder la mano (pasó con BORRAR y dos manos) | Trazos del tamaño de la mano y a ritmo moderado |
| **Dos manos delante de la cara** | MediaPipe dejó de detectarlas | Manos a la altura del pecho, separadas |
| **Asentir** | Un "sí" pequeño (±5°) se parece al movimiento natural de la cabeza; quedaron 3 asentimientos falsos en 19.4 min | Asentir de forma marcada; tres cabeceos |
| **Preguntas falsas con las cejas al hacer letras** | Al mirar la mano o la pantalla, las cejas suben de verdad más de 10 % sin mover la cabeza: 8 `?` falsos en 19.4 min (la versión anterior, 21) | Mirar a la cámara; `Backspace` o negar con la cabeza borran el `?`. Seguir la mirada con los puntos del iris (`refine_landmarks`) es una mejora posible |
| **FPS con dos manos** | `max_num_hands=2` duplica el tiempo de MediaPipe Hands mientras solo hay una mano a la vista; en batería, 17 FPS | Conectar la laptop; `MAX_MANOS = 1` o `PROCESAR_ROSTRO_CADA = 2` en una PC lenta |
| **Una sola persona** | Todos los umbrales se calibraron con la mano y la cara del autor | Repetir las sesiones guiadas con más personas |
| **CH, LL y RR** | No incluidas | Roadmap |
| **Voz del sistema** | Requiere una voz SAPI5 en español | Si falta, el programa sigue sin voz |
| **Una sesión por computadora** | La cámara y el texto son únicos: varias pestañas ven y controlan la misma sesión | Usar una sola pestaña |
| **Esqueleto detrás del video** | El video sale sin esperar a MediaPipe; el esqueleto llega después del proceso (~15-60 ms) | Inherente a separar la vista del reconocimiento; se mide en la cabecera |
| **Cámara fija** | Se usa la cámara de índice 0 de OpenCV | Cambiar el índice en `CamaraReciente(0)` (`server.py`) |
| **Modo Práctica y texto** | Cada acierto también se escribe en los subtítulos | Limpiar (`C`) al terminar de practicar |
| **Estadísticas de práctica** | Solo se guarda la mejor racha, en el navegador; los aciertos se pierden al recargar | Persistencia en base de datos (roadmap) |
| **Imágenes de referencia** | La tarjeta de práctica muestra un espacio reservado, sin foto de la seña | Agregar las imágenes de la lámina SEP |
| **Solo local** | El servidor escucha en `127.0.0.1`, sin autenticación ni TLS | Requisito previo a cualquier despliegue en red |

---

## 10. Decisiones de diseño respaldadas por datos

| Decisión | Evidencia |
|---|---|
| **Rasgos 3D en un marco propio de la mano** | Separan los pares confundibles con un solo umbral en datos de ambas manos (p. ej. S/T, S/A, E/S, R/U al 100 % en calibración); la versión 2D quedaba en 69-78 % por letra con datos nuevos |
| **Extensión 3D en lugar de la suma de ángulos para "dedo recto"** | En la G izquierda de lado, el anular cerrado medía "recto" por ángulos (desde 56°) y 0.72-0.98 por extensión |
| **Pulgar medido contra las falanges del índice y el medio** | T: punta del pulgar +0.03 a +0.23 sobre los PIP; S: −0.40 a −0.03 |
| **Dedo recto por ángulos articulares, no solo por extensión** | La N con el índice a 1.39 tiene 51° de flexión (recto) y la E a 1.39 tiene 95° (doblado); la derecha subió de 85.6 % a 93.0 % en la sesión 9 |
| **"Pistola" con el índice largo aunque doblado** | La G izquierda de la sesión 9 tenía el índice a 1.71 con ~96° de flexión; la A siempre tiene el índice metido (≤ 1.21) |
| **Prueba de espejo** | 8 336 de 8 336 fotogramas reales clasificados igual al reflejarlos como la otra mano |
| **Ventanas de trazo en segundos** | A 17 FPS una ventana de 45 fotogramas dura 2.8 s; con segundos, BORRAR izquierda pasó de fallar a funcionar en la prueba |
| **Configuración dominante en la ventana, no la del último frame** | La Z izquierda tenía una Z perfecta, pero el último frame se leía como X |
| **Movimiento coherente en vaivenes** | La única K falsa en dos manos tenía rectitud 0.34 y tramos desiguales (3.8×) |
| **Ñ con amplitud 0.07** | La Ñ suave real oscila 0.08-0.13; sostener la N tiembla < 0.05 y no dio Ñ falsas |
| **D/X por la inclinación de la palma** | Al dibujar Z o `?` el índice se ladea hasta 90° pero la palma sigue en 16-50°; la X real tiene la palma a 52-61° |
| **One-Euro por mano** | 22 de 22 escritas con filtro frente a 21 de 22 sin él, y sin trazos falsos por temblor |
| **Seguimiento de manos por continuidad** | MediaPipe repitió la misma mano y la etiqueta parpadeó en C y F |
| **Misma letra con ambas manos = una letra** | En la sesión 5 se hicieron A, B, L y V con ambas manos y se escribían dos veces |
| **Asentir con 5° y tres cabeceos** | Con 5° y dos, 24 asentimientos falsos; con tres, 4; con 6° no se detectaba el "sí" real |
| **Cara tapada por error de reproyección > 1.8 %** | Cara visible 0.8-1.4 %; manos encima 1.7-3.8 % |
| **Más de 4 s con cejas arriba = nueva cara neutra** | Una sesión mantuvo la razón en ~1.11 durante 50 s y producía preguntas repetidas |
| **BORRAR como barrido de mano abierta** | El puño es S, T o A; borrar con un puño borraría letras al escribirlas |

**Decisiones de la arquitectura web**

| Decisión | Motivo |
|---|---|
| **El navegador no reconoce nada** | El reconocimiento sigue siendo el `Traductor.paso()` validado con las 9 sesiones; la interfaz no puede alterar los resultados |
| **Último valor en lugar de cola** (cámara, `Difusor` y dibujo) | Con una cola, un cliente lento o un proceso lento acumularía segundos de retraso; con el último valor, el retraso máximo es un fotograma |
| **Python dueño de la cámara; video MJPEG aparte del estado** | `getUserMedia` falló con `NotReadableError` en la laptop de pruebas (Windows no compartió la cámara). Subir fotogramas desde el navegador detendría el reconocimiento con la pestaña en segundo plano. MJPEG: 30 FPS, 1 ms de compresión, sin esperar a MediaPipe |
| **Estado por WebSocket sin imagen** | 0.54-1.57 KB por estado frente a ~11 KB del JPEG anterior; el esqueleto se dibuja en el navegador con los colores del tema |
| **Hora de captura en reloj de pared** | El navegador y Python están en la misma computadora: `Date.now() - captura_ms` mide la latencia real sin sincronizar relojes |
| **`perf_counter` para medir el proceso** | En Windows, `time.monotonic()` tiene ~16 ms de resolución: el proceso salía siempre 16.0 ms |
| **Un solo hilo escribe el texto** | Las acciones del navegador pasan por una cola; no hace falta sincronizar `Escritor` |
| **Etiquetas de las manos en HTML, no dibujadas en el frame** | Tipografía con acentos y Ñ reales, temas y animaciones, sin costo de dibujo en Python |
| **`127.0.0.1` y sin persistencia de video** | La cámara es un dato sensible; nada sale de la computadora |
| **Fallo de práctica solo con letras** | La mano abierta (ESPACIO) es una postura de reposo y castigaría la racha sin razón |

---

## 11. Referencia del código

### 11.1 Estructura del repositorio

```
Proyecto/
├── python/                  # Todo el código Python (módulos planos que se importan por nombre)
│   ├── server.py            # Punto de entrada: FastAPI, WebSocket /ws, Difusor, video MJPEG, apertura del navegador
│   ├── main.py              # Motor: bucle de proceso en su propio hilo
│   ├── camara.py            # Captura en un hilo propio: siempre el fotograma más reciente
│   ├── classifier_hands.py  # Manos: dos manos, One-Euro, rasgos 3D, alfabeto y trazos
│   ├── classifier_face.py   # Rostro: solvePnP, cejas, negar y asentir
│   ├── traductor.py         # Hold-to-type por mano y orquestación de un frame
│   ├── gestos.py            # Frases con movimiento: DTW contra el dataset y muestras propias
│   ├── voice_engine.py      # Voz asíncrona (SAPI) y puente al hardware
│   ├── ui_panel.py          # Presentación: estado JSON con las coordenadas de las manos
│   ├── video_extractor.py   # Herramienta: cortes de video -> dataset de 30 fotogramas
│   ├── auto_etiquetador.py  # Herramienta: cortes propuestos a partir de la voz (Whisper)
│   └── data_augmentation.py # Herramienta: variantes sintéticas para entrenar
├── static/                  # Interfaz web (la sirve python/server.py)
│   ├── index.html           # Estructura de la interfaz
│   ├── estilos.css          # Temas, maquetación y animaciones
│   └── app.js               # Cliente WebSocket, video MJPEG, esqueleto, tema y Modo Práctica
├── datos/
│   ├── dataset_config.json  # Cortes de los videos (versionado)
│   ├── videos/              # Videos de entrada; los terminados pasan a videos/listo/ (ignorado por Git)
│   ├── dataset/             # Muestras .npy por palabra e indice.json (ignorado por Git)
│   └── gestos_vivo.csv      # Evaluaciones de frases en vivo (ignorado por Git)
├── docs/
│   ├── DOCUMENTACION.md     # Este documento
│   └── errores_detectado.txt # Pendientes anotados a mano
├── .vscode/
│   ├── launch.json          # F5: python/server.py con el Python de venv
│   └── settings.json        # Intérprete, python/ para Pylance y perfil de PowerShell
├── venv/                    # Entorno virtual (ignorado por Git)
├── .gitignore
├── CLAUDE.md                # Mapa compacto del proyecto para el asistente de código
└── README.md                # Resumen e inicio rápido
```

### 11.2 Módulos

Dependencias: `voice_engine` ← `classifier_hands` ← `classifier_face` ← `traductor` ← `ui_panel` ← `main` (+ `camara`) ← `server`. Cada módulo de reconocimiento empieza con sus umbrales, cada uno con el rango medido que lo justifica.

| Módulo | Elementos |
|---|---|
| **`classifier_hands.py`** | geometría (`dist`, `unitario3`, `angulo3`, `angulo_vectores`, `desde_arriba`), `FiltroOneEuro`, `a_canonica` (espejo), `escala_mano`, `RasgosMano`, `analizar_mano`, `clasificar_pose`, `decidir_q_o_x`, `Trayectoria`, `simplificar_rdp`, `segmentos`, `_trazo_j`, `_trazo_z`, `_trazo_interrogacion`, `puntos_de_giro`, `rectitud`, `_vaiven`, `_trazo_borrar`, `ReconocedorDinamico`, `DeteccionMano`, `leer_manos`, `SeguidorManos`, `EstadoMano`, `CanalMano`, `Estabilizador`, `ProgresoSuave`, `detectar_manos` |
| **`classifier_face.py`** | `EstimadorPoseCabeza` (`solvePnP`), `PoseCabeza`, `altura_cejas`, `cejas_esperadas`, `calcular_rasgos_faciales`, `DetectorOscilacion`, `AnalizadorFacial`, `detectar_rostro` |
| **`traductor.py`** | `SostenerMano`, `Escritor`, `Traductor` |
| **`voice_engine.py`** | `limpiar_texto`, `nombre_hablado`, `ultima_palabra`, `avisar_una_vez`, `VozAsincrona`, `PoliticaVoz`, `_elegir_voz`, `enviar_a_hardware` |
| **`ui_panel.py`** | `GUIA`, `PISTA_TRAZO`, `CONEXIONES_MANO`, `VistaMano`, `vista_mano`, `mano_principal`, `config_web`, `estado_web` (con `puntos`, `estela` y `tiempos`), `lineas_diagnostico` |
| **`main.py`** | `capturar(cam, publicar, comandos, detener)`: fotograma más reciente, acciones de la cola, voz, hardware, tiempos y estado |
| **`camara.py`** | `CamaraReciente` (`leer` con `consumir`, `cerrar`, `descartados`), `a_jpeg`, `VIDEO_CALIDAD`, `VIDEO_ANCHO_MAX` |
| **`server.py`** | `Difusor`, `ciclo_de_vida` (lifespan: abre la cámara y lanza el proceso), `inicio` (`/`), `video_feed` y `_fotogramas_mjpeg` (`/video_feed`), `transmitir` (`/ws`), `_recibir`, `_cerrando`, `main` |
| **`static/app.js`** | `conectar`, `iniciarVideo`, `revisarVideo`, `soloEsqueleto`, `detenerVideo`, `dibujarEsqueleto`, `cuadro` (bucle de dibujo y latencia), `configurar`, `mostrarEstado`, `fase`, `aplicarTema`, `cambiarVista`, `evaluarPractica`, `acierto`, `fallo`, `celebrar`, `lanzarConfeti` |

La división en módulos movió el código sin cambiarlo: 235 de las 236 definiciones son idénticas en su árbol de sintaxis (la otra solo cambia cómo se escribe la misma constante), y con 6 000 fotogramas sintéticos de dos manos y rostro la versión dividida produjo exactamente las mismas salidas que el `main.py` anterior. La migración a la web después reemplazó las ventanas de OpenCV de `ui_panel.py` por la interfaz de `static/`; el reconocimiento (`classifier_*`, `traductor`) no cambió.

### 11.3 Referencias

- SEP y Aprende.mx. *Abecedario en Lengua de Señas Mexicana* (lámina). https://nuevaescuelamexicana.sep.gob.mx/contenido/recurso/abecedario-en-lengua-de-senas-mexicana/
- Zhang, F. et al. (2020). *MediaPipe Hands: On-device Real-time Hand Tracking*. arXiv:2006.10214.
- Kartynnik, Y. et al. (2019). *Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs* (MediaPipe Face Mesh). arXiv:1907.06724.
- MediaPipe. *Canonical face model* (`mediapipe/modules/face_geometry/data/canonical_face_model.obj`). https://github.com/google-ai-edge/mediapipe
- Casiez, G., Roussel, N. y Vogel, D. (2012). *1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems*. CHI 2012.
- OpenCV. *Perspective-n-Point (PnP) pose computation* (`cv2.solvePnP`). https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
- Ramer, U. (1972). *An iterative procedure for the polygonal approximation of plane curves*. Computer Graphics and Image Processing, 1(3).
- Hochreiter, S. y Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8).
- Documentación de MediaPipe Solutions: https://ai.google.dev/edge/mediapipe/solutions/guide
- Documentación de pyttsx3: https://pyttsx3.readthedocs.io/
- FastAPI. *WebSockets*. https://fastapi.tiangolo.com/advanced/websockets/
- IETF (2011). *RFC 6455: The WebSocket Protocol*. https://www.rfc-editor.org/rfc/rfc6455
