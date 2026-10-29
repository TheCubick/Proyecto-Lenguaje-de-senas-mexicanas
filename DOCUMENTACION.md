# Sistema Inteligente de Traducción de Lengua de Señas con Visión Artificial y Componentes Multimodales

> **Documentación técnica oficial**
> **Materia:** Inteligencia Artificial
> **Stack:** Python 3.12 · MediaPipe 0.10.18 · OpenCV 4.11 (incluye `solvePnP`) · NumPy · pyttsx3
> **Repositorio:** https://github.com/TheCubick/Proyecto-Lenguaje-de-se-a

### Leyenda de estado

En este documento cada componente lleva una etiqueta que indica su grado de avance real:

| Etiqueta | Significado |
|---|---|
| **Implementado** | Funciona en el código actual (`main.py`) y se validó con grabaciones reales (sección 8) |
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
                └──────┬───────────────┬───────────────┬────────┘    el mismo código en vivo y en pruebas)
          ┌────────────▼───┐   ┌───────▼────────┐   ┌──▼──────────────────────┐
          │ Ventana cámara │   │ Panel          │   │ Hilo de voz (pyttsx3)   │
          │ Izq: B | Der: O│   │ traductor      │   │ + enviar_a_hardware()   │
          └────────────────┘   └────────────────┘   └─────────────────────────┘
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

Por encima de **1.8 %** el fotograma no cuenta para ningún gesto facial y el indicador muestra `Modo: Cara tapada`.

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
| **Tolerancia a fallos** | Si falta `pyttsx3` o una voz, el programa sigue sin voz y lo indica en el panel (`Voz: no disponible`) |

**Cuándo habla** (`PoliticaVoz`):

| Evento | Qué dice |
|---|---|
| Se registra una letra con cualquier mano (barra al 100 % o trazo completo) | el nombre de la letra |
| Se **confirma** la palabra (asentir) | la palabra completa |
| Pasan **4 s sin nuevas letras** | la palabra pendiente (una sola vez) |
| Espacio, borrar, `?` | nada |

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

### 3.3 Interfaz dual

**Ventana "Camara IA - Deteccion LSM"**
- Video en espejo con **el esqueleto de cada mano** (puntos azules la izquierda, verdes la derecha) y los puntos de cejas y ojos.
- Por cada mano: recuadro con esquinas resaltadas (gris: sin seña · azul: cargando · verde: registrada · ámbar: configuración que necesita movimiento), **etiqueta `Izq: B` / `Der: O`**, mini barra de progreso propia, pista (`o traza una J`, `o agita para K`…) y **estela** del dedo mientras dibuja.
- **Resumen arriba al centro:** `Izq: B  |  Der: O`.
- **Medidor de cejas** (`CEJAS x1.11` con la marca del umbral) y debajo los **ángulos reales de la cabeza** (`P+3 Y-12 R-1`).
- Indicador `Modo: Neutro / Pregunta ? / Negacion X / Afirmacion OK / Calibrando... / Cara tapada`.
- Barra inferior de la mano principal (`Sosten la sena B (Der) quieta... 60%`) y FPS.
- Modo diagnóstico (`D`): por mano, estado y extensión de cada dedo, flexiones, orientación, pulgar, normal de la palma, cruce y velocidad; de la cabeza, desviaciones de yaw y pitch, distancia y apertura de ojos.

**Ventana "Panel Traductor LSM"** (lienzo NumPy): la seña de la **mano principal** en grande con la etiqueta `MANO IZQ` / `MANO DER` (la principal es la que acaba de registrar o la que más avanzó su barra), el resumen de ambas manos en el encabezado, cómo se hace, la **guía del alfabeto**, el recordatorio de los gestos del rostro, la caja de texto y el estado de la voz.

### 3.4 Teclado

| Tecla | Acción |
|---|---|
| `Espacio` | Agregar un espacio |
| `Backspace` | Borrar la última letra |
| `C` | Limpiar todo el texto |
| `D` | Modo diagnóstico |
| `N` (o `R`) | Recalibrar la cara neutra |
| `ESC` o la `X` de cualquier ventana | Salir limpiamente |

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
- **Herramienta educativa:** practicar el alfabeto dactilológico con retroalimentación inmediata.
- **Validación con la comunidad:** pruebas con personas usuarias de LSM para corregir la forma de las señas y priorizar el vocabulario.

---

## 6. Guía de instalación y ejecución

### 6.1 Requisitos

| Requisito | Versión / detalle |
|---|---|
| Sistema operativo | Windows 10/11 |
| Python | **3.12** (MediaPipe 0.10.18 **no** es compatible con 3.13) |
| Hardware | Cámara web integrada o USB |
| Voz | Una voz SAPI5 en español (en Windows: *Microsoft Sabina* o *Helena*). Si falta, el programa funciona sin voz |
| Editor | VS Code con la extensión **Python** |

### 6.2 Dependencias

| Librería | Uso |
|---|---|
| `mediapipe==0.10.18` | Detección de las manos (Hands, con coordenadas de mundo y lateralidad) y del rostro (Face Mesh) |
| `opencv-python` | Captura de video, dibujo, ventanas y **`solvePnP`** |
| `numpy` | Geometría 3D, filtro One-Euro, panel y trayectorias |
| `pyttsx3` | Voz (en Windows instala también `comtypes` y `pywin32`) |

> **No se agregaron dependencias para esta versión.** El filtro One-Euro (20 líneas) y la pose de la cabeza (`cv2.solvePnP`) no necesitan librerías extra; las coordenadas del modelo canónico de la cara están escritas en `main.py`.

### 6.3 Crear el entorno virtual (`venv`)

```powershell
# 1. Crear el entorno con Python 3.12
py -3.12 -m venv venv

# 2. Instalar dependencias
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install opencv-python mediapipe==0.10.18 numpy pyttsx3
```

> La carpeta `venv/` está en `.gitignore`: cada integrante crea la suya localmente y no se sube al repositorio.

### 6.4 Ejecución rápida con F5

1. Abre la carpeta `Proyecto/` en VS Code.
2. Pulsa **`F5`**, o el botón en *Ejecutar y depurar*, y elige **"Traductor LSM (main.py)"**.

| Archivo | Qué configura |
|---|---|
| `.vscode/launch.json` | Ejecuta `main.py` con `venv\Scripts\python.exe` directamente, sin depender de activar el entorno |
| `.vscode/settings.json` | Fija `venv` como intérprete del proyecto y crea un perfil de terminal **"PowerShell (venv)"** con `ExecutionPolicy Bypass`, para que Windows no bloquee `Activate.ps1` |

**Alternativa desde la terminal:** `.\venv\Scripts\python.exe main.py`

### 6.5 Recomendaciones de uso

- Iluminación frontal y uniforme; evita tener una ventana detrás.
- Manos completas dentro del cuadro, a 40-80 cm de la cámara.
- **Con dos manos:** tenlas a la altura del pecho, separadas y **sin tapar la cara**. Con las manos juntas delante de la cara, MediaPipe llegó a no detectar ninguna (sección 8.3).
- Al iniciar, **mantén la cara neutra** unos 1.5 s (`Modo: Calibrando...`); `N` recalibra.
- **Los trazos (Z, ?, J…) deben ser pequeños, del tamaño de la mano**, sin sacarla del cuadro. Los movimientos muy rápidos pueden hacer que la cámara pierda la mano.
- Sostén cada letra **quieta** hasta que la barra llegue al 100 %.

> Los mensajes `W0000 … inference_feedback_manager` que aparecen en la consola son avisos internos de MediaPipe, no errores.

---

## 7. Calibración

Todos los umbrales están al inicio de `main.py`, cada uno con el rango medido que lo justifica. El **modo diagnóstico** (tecla `D`) muestra los valores en vivo.

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

Cada grabación se **reprodujo con el `Traductor` real de `main.py`** (seguidor de manos, One-Euro, clasificador, estabilizador, trayectorias, hold-to-type y rostro), así que los resultados son los del programa, no de una copia.

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
| Dibujo de ambas ventanas | 7.3 ms de mediana (p95 16.6 ms) |
| MediaPipe (2 manos + rostro), laptop **conectada** (sesiones 1-4) | ≈ 25 ms por fotograma → **25-26 FPS** de captura |
| MediaPipe solo, laptop **en batería** (40 %) | manos 37 ms con `max_num_hands=1` y **76 ms con 2** (con una mano a la vista busca la segunda en cada frame); rostro 7-11 ms |
| `main.py` completo en vivo, en batería | **17 FPS** (60 ms de mediana por fotograma), 400 fotogramas, teclas simuladas, cierre limpio |
| Voz | no bloquea el bucle (`decir()` regresa al instante; hilo propio) |

El costo principal es MediaPipe, no el código del proyecto. En una PC lenta, `PROCESAR_ROSTRO_CADA = 2` o `MAX_MANOS = 1` recuperan FPS. Los trazos se miden en segundos y el hold-to-type con `time.monotonic()`, así que el reconocimiento no cambia con los FPS.

### 8.10 Pruebas automáticas

| Batería | Resultado |
|---|---|
| **28 pruebas de lógica**: hold-to-type por mano, misma letra con ambas manos, bloqueo tras un trazo, seguidor de manos (duplicados, parpadeo, corrección, orden), espejo canónico, One-Euro (temblor y retraso), voz, guiones bajos, coherencia de vaivenes | **28 de 28** |
| Dibujo de las dos ventanas con 520 fotogramas reales de dos manos y rostro (con y sin diagnóstico) | sin errores |
| Reproducción completa de las 9 sesiones con el `Traductor` real | resultados de las secciones 8.2-8.8 |
| `main.py` en vivo con la cámara (sección 8.9) | ventanas, teclas `N`, `D`, `Espacio`, `Backspace`, envíos al hardware y cierre limpio |

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
| **Fuentes de OpenCV** | No dibujan acentos; la Ñ se compone a mano | Texto sin acentos en pantalla y Ñ propia |
| **Voz del sistema** | Requiere una voz SAPI5 en español | Si falta, el programa sigue sin voz |

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

---

## 11. Referencia del código

### 11.1 Estructura del repositorio

```
Proyecto/
├── .vscode/
│   ├── launch.json       # Ejecución con F5 usando el Python de venv
│   └── settings.json     # Intérprete del proyecto y perfil de PowerShell
├── venv/                 # Entorno virtual (ignorado por Git)
├── .gitignore
├── main.py               # Código fuente del sistema (11 secciones)
├── README.md             # Resumen e inicio rápido
└── DOCUMENTACION.md      # Este documento
```

### 11.2 `main.py` por secciones

| Sección | Elementos |
|---|---|
| **1. Configuración** | constantes (con el rango medido que justifica cada umbral), `GUIA` del alfabeto, estados faciales |
| **2. Texto y voz** | `limpiar_texto`, `nombre_hablado`, `ultima_palabra`, `_a_ascii` |
| **3. Geometría y filtro** | `dist`, `unitario3`, `angulo3`, `angulo_vectores`, `desde_arriba`, `FiltroOneEuro` |
| **4. Manos (rasgos 3D)** | `a_canonica` (espejo), `escala_mano`, `RasgosMano`, `analizar_mano` |
| **5. Clasificación** | `clasificar_pose`, `decidir_q_o_x`, `Trayectoria`, `simplificar_rdp`, `segmentos`, `_trazo_j`, `_trazo_z`, `_trazo_interrogacion`, `puntos_de_giro`, `rectitud`, `_vaiven`, `_trazo_borrar`, `ReconocedorDinamico` |
| **6. Dos manos** | `DeteccionMano`, `leer_manos`, `SeguidorManos`, `EstadoMano`, `CanalMano` |
| **7. Rostro** | `EstimadorPoseCabeza` (`solvePnP`), `PoseCabeza`, `altura_cejas`, `cejas_esperadas`, `calcular_rasgos_faciales`, `DetectorOscilacion`, `AnalizadorFacial` |
| **8. Escritura y orquestación** | `Estabilizador`, `SostenerMano`, `Escritor`, `ProgresoSuave`, `Traductor` |
| **9. Voz y hardware** | `VozAsincrona`, `PoliticaVoz`, `_elegir_voz`, `enviar_a_hardware` |
| **10. Interfaz** | `VistaMano`, `EstadoUI`, `vista_mano`, `mano_principal`, `poner_texto` (con Ñ), `dibujar_mano`, `dibujar_camara`, `dibujar_panel`, `medidor_cejas`, `dibujar_estela` |
| **11. Principal** | `detectar_manos`, `detectar_rostro`, `main` |

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
