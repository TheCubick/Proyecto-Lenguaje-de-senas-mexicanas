# Minería de datos desde videos (para entrenar la LSTM)

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

## Aumento de datos

`data_augmentation.py` multiplica las muestras con variantes sintéticas. Cada variante combina ruido leve en las coordenadas (~2 % del tamaño de la mano), ritmo variable por tramos (entre 25/30 y 35/30 de la velocidad original, regresando a 30 fotogramas) y escala (distancia a la cámara ±15 %, tamaño de mano ±10 %). Las manos ausentes siguen en cero.

```python
from video_extractor import cargar_dataset
from data_augmentation import aumentar
X, y = cargar_dataset()
X5, y5, origen = aumentar(X, y, factor=5)   # o factor=10
```

Separa entrenamiento y prueba **antes** de aumentar (o agrupa por `origen`): si variantes de una misma muestra quedan en ambos lados, la exactitud de prueba sale inflada. Para revisar o subir a Colab: `.\venv\Scripts\python.exe python\data_augmentation.py --factor 10` escribe `datos/dataset_aumentado.npz`.

## Auto-etiquetado por voz (opcional)

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
