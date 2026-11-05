"""Auto-etiquetado por voz: propone los cortes de dataset_config.json a partir de lo que el maestro DICE en el video.

Para cada video de videos/ que aún no está en dataset_config.json:
  audio (PyAV) -> faster-whisper (español, VAD, marca de tiempo por palabra) -> "palabras sueltas" -> un corte por
  palabra, agregado a dataset_config.json con "origen": "whisper" y "oido" (lo que se escuchó). Los videos sin voz
  utilizable van a requiere_anotacion_manual.txt con el motivo. Después: revisar y correr video_extractor.py.

Uso:
  python auto_etiquetador.py                               # etiqueta los videos nuevos
  python auto_etiquetador.py --simular                     # solo muestra lo que agregaría, no escribe nada
  python auto_etiquetador.py --vocabulario vocabulario.txt # solo esas palabras (una por línea), con ortografía corregida
  python auto_etiquetador.py --modelo base                 # más rápido y menos preciso (default small)
  python auto_etiquetador.py --rehacer                     # vuelve a etiquetar lo que no tiene cortes manuales

Medido en los 12 videos de videos/ (2026-11-04), en los que se basan las reglas:
  - 7 no tienen voz (música o silencio): el VAD encuentra 0-1 s de habla. Whisper inventa "No, no, no, no." en uno,
    con avg_logprob -1.33: se descarta por confianza.
  - Con voz, el maestro explica con frases largas y dice el nombre de la seña SUELTO, entre pausas, mientras la hace
    ("Día." "Día." "Semana."). Una palabra suelta = una emisión de 1 a 3 palabras separada por >= PAUSA o por punto.
  - La seña empieza con la palabra (voz en off de "Tutorial Dilo en señas": las manos aparecen +-0.4 s de la palabra)
    o hasta 0.9 s después ("lunes": la palabra a los 52.1 s, la seña de 53.0 a 55.5 s y luego una repetición en
    cámara lenta). Corte = de ANTES s antes de la palabra hasta MAX_CORTE s después, sin llegar a la siguiente.
  - Contra los cortes hechos a mano con los rótulos de "Tutorial Dilo en señas" (9 frutas): los 9 cortes contienen la
    seña (bordes a <= 0.5 s), pero 1 de 9 palabras está mal: oyó "fresa" (probabilidad 1.00) donde el video muestra
    CEREZA, con y sin vocabulario. Por eso cada corte queda marcado "origen": "whisper" para revisarlo.
"""
import argparse
import os
import sys
import unicodedata
from difflib import get_close_matches
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # Windows sin modo desarrollador: copia en vez de
                                                                 # enlazar; el aviso no afecta en nada
import av
import numpy as np
from faster_whisper import WhisperModel

from video_extractor import CONFIG, VIDEOS, escribir_config, leer_config, mmss, nombre_carpeta

RAIZ = Path(__file__).resolve().parent
MANUAL = RAIZ / "requiere_anotacion_manual.txt"
EXTENSIONES = {".mp4", ".mkv", ".webm", ".mov"}
MODELO = "small"               # base oyó lo mismo pero con menos certeza ("plátano" 0.81 contra 0.98)
VOZ_MIN = 1.0                  # s de habla (VAD) por debajo de los cuales el video se considera mudo
PAUSA = 0.3                    # s de silencio que separan dos emisiones
MAX_PALABRAS = 3               # una emisión más larga es explicación ("con la configuración de la mano en D")
PROB_MIN = 0.4                 # probabilidad media de las palabras: "Mes." salió con 0.47 y es correcta
ANTES = 0.5                    # el corte empieza esto antes de la palabra
MAX_CORTE = 4.0                # y termina a lo más esto después: las manos terminan 3.4-4.0 s después de la palabra
HUECO = 0.2                    # separación con el corte siguiente (no se enciman)
MIN_CORTE = 0.8                # cortes más cortos no alcanzan a contener una seña
ABREN = {"iniciamos", "empezamos", "comenzamos", "luego", "entonces", "otra", "donde", "es", "son", "esta", "este",
         "esto", "así", "ahora", "bien", "listo", "siguiente", "también", "vamos", "y"}   # muletillas al inicio
CIERRAN = {"es", "la", "el", "los", "las", "de", "del", "con", "que", "y", "en", "un", "una", "a", "al"}  # frase abierta
ALUCINACIONES = ("amara", "subtítulos", "suscríbete", "suscribete", "gracias por ver")   # típicas sobre música


def audio_16k(ruta):
    """Pista de audio -> float32 mono a 16 kHz (lo que espera Whisper), o None si el video no tiene audio. Se
    decodifica aquí y no con faster_whisper.decode_audio: esa función falla con PyAV >= 15 (metadata_errors)."""
    with av.open(str(ruta)) as contenedor:
        if not contenedor.streams.audio:
            return None
        remuestreo = av.AudioResampler(format="s16", layout="mono", rate=16000)
        trozos = []
        for cuadro in contenedor.decode(audio=0):
            trozos += [c.to_ndarray() for c in remuestreo.resample(cuadro)]
        trozos += [c.to_ndarray() for c in remuestreo.resample(None)]
    return np.concatenate(trozos, axis=1).ravel().astype(np.float32) / 32768.0 if trozos else np.zeros(0, np.float32)


def transcribir(modelo, ruta):
    """-> (palabras confiables de Whisper, duración en s) o (None, motivo para la anotación manual). El vocabulario
    NO se le pasa a Whisper (hotwords): medido, cambia cómo segmenta y perdió "hola" x3 y un "día" a cambio de
    "cómo estás" x2. Solo se usa después, para filtrar y corregir la ortografía."""
    audio = audio_16k(ruta)
    if audio is None or len(audio) == 0:
        return None, "sin pista de audio"
    segmentos, info = modelo.transcribe(audio, language="es", vad_filter=True, word_timestamps=True,
                                        condition_on_previous_text=False)   # evita bucles de texto inventado
    segmentos = list(segmentos)                          # la transcripción ocurre al recorrer el generador
    if info.duration_after_vad < VOZ_MIN:
        return None, f"sin voz ({info.duration_after_vad:.1f} s de habla; música o silencio)"
    palabras = [w for s in segmentos
                if s.avg_logprob > -1.0 and s.no_speech_prob < 0.6 and s.compression_ratio < 2.4
                and not any(a in s.text.lower() for a in ALUCINACIONES)
                for w in (s.words or [])]
    if not palabras:
        return None, "sin voz clara (Whisper con baja confianza)"
    return palabras, info.duration


def emisiones(palabras):
    """Agrupa las palabras separadas por menos de PAUSA s y sin punto final entre ellas."""
    grupos, actual = [], []
    for w in palabras:
        if actual and (w.start - actual[-1].end >= PAUSA or actual[-1].word.strip()[-1:] in ".!?"):
            grupos.append(actual)
            actual = []
        actual.append(w)
    return grupos + ([actual] if actual else [])


def sin_acentos(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def etiqueta(grupo, vocabulario):
    """Emisión -> palabra para el dataset, o None si no parece el nombre de una seña."""
    partes = [p for p in (nombre_carpeta(w.word) for w in grupo) if p]
    partes = [p for i, p in enumerate(partes) if i == 0 or p != partes[i - 1]]   # "buenos, buenos días"
    if not 1 <= len(partes) <= MAX_PALABRAS or partes[0] in ABREN or partes[-1] in CIERRAN:
        return None
    if np.mean([w.probability for w in grupo]) < PROB_MIN:
        return None
    texto = "_".join(partes)
    if vocabulario:                                      # "durazmo" -> "durazno"; fuera del vocabulario, nada
        claves = {sin_acentos(v): v for v in vocabulario}
        parecida = get_close_matches(sin_acentos(texto), list(claves), n=1, cutoff=0.75)
        return claves[parecida[0]] if parecida else None
    return None if len(texto) == 1 else texto            # letras sueltas del alfabeto: ya las reconocen las reglas


def cortes(candidatos, duracion):
    """[(inicio de la palabra, etiqueta, oído)] en orden -> [(inicio, fin, etiqueta, oído)] sin encimarse. Los
    tiempos se redondean a décimas hacia adentro del corte."""
    salida, fin_previo = [], 0.0
    for i, (t, palabra, oido) in enumerate(candidatos):
        siguiente = candidatos[i + 1][0] - HUECO if i + 1 < len(candidatos) else duracion
        inicio = np.ceil(max(t - ANTES, fin_previo, 0.0) * 10) / 10
        fin = np.floor(min(siguiente, t + MAX_CORTE, duracion) * 10) / 10
        if fin - inicio >= MIN_CORTE:
            salida.append((float(inicio), float(fin), palabra, oido))
            fin_previo = fin
    return salida


def leer_manual(ruta):
    """requiere_anotacion_manual.txt: una línea por video, "nombre<TAB>motivo"."""
    if not ruta.is_file():
        return {}
    filas = [linea.split("\t", 1) for linea in ruta.read_text(encoding="utf-8").splitlines() if linea.strip()]
    return {f[0].strip(): (f[1].strip() if len(f) > 1 else "") for f in filas}


def main():
    sys.stdout.reconfigure(errors="replace")
    p = argparse.ArgumentParser(description="Propone cortes para dataset_config.json a partir de la voz de los videos.")
    p.add_argument("--videos", type=Path, default=VIDEOS, help=f"carpeta de los videos (default {VIDEOS.name}/)")
    p.add_argument("--config", type=Path, default=CONFIG, help=f"archivo de cortes (default {CONFIG.name})")
    p.add_argument("--manual", type=Path, default=MANUAL, help=f"lista de videos sin voz (default {MANUAL.name})")
    p.add_argument("--modelo", default=MODELO, help=f"tamaño de Whisper: tiny, base, small, medium (default {MODELO})")
    p.add_argument("--vocabulario", type=Path, help="archivo con las palabras a buscar, una por línea")
    p.add_argument("--simular", action="store_true", help="mostrar lo que agregaría sin escribir nada")
    p.add_argument("--rehacer", action="store_true", help="re-etiquetar videos sin cortes manuales")
    args = p.parse_args()
    if args.config.suffix.lower() != ".json":
        p.error("--config debe ser un .json (el CSV se llena a mano)")

    entradas = leer_config(args.config) if args.config.is_file() else []
    manual_previo = leer_manual(args.manual)
    con_cortes = {e.get("video") for e in entradas if isinstance(e, dict)}
    a_mano = {e.get("video") for e in entradas if isinstance(e, dict) and e.get("origen") != "whisper"}
    videos = sorted(v for v in args.videos.glob("*") if v.suffix.lower() in EXTENSIONES)
    pendientes = [v for v in videos if v.name not in a_mano
                  and (args.rehacer or (v.name not in con_cortes and v.name not in manual_previo))]
    if not pendientes:
        print("No hay videos nuevos: todos están en la configuración o en la lista manual (usa --rehacer).")
        return
    vocabulario = []
    if args.vocabulario:
        lineas = args.vocabulario.read_text(encoding="utf-8-sig").splitlines()
        vocabulario = list(dict.fromkeys(v for v in map(nombre_carpeta, lineas) if v))
    print(f"{len(pendientes)} videos por etiquetar ({len(videos) - len(pendientes)} ya revisados). Cargando Whisper "
          f"'{args.modelo}' (la primera vez se descarga)...", flush=True)
    modelo = WhisperModel(args.modelo, device="cpu", compute_type="int8")

    nuevas, manual, hechos = [], dict(manual_previo), []
    try:
        for ruta in pendientes:
            print(f"\n{ruta.name}", flush=True)
            try:
                palabras, extra = transcribir(modelo, ruta)
            except (av.FFmpegError, OSError) as e:
                palabras, extra = None, f"no se pudo leer el audio ({e})"
            hechos.append(ruta.name)
            manual.pop(ruta.name, None)
            if palabras is None:
                print(f"  -> anotación manual: {extra}")
                manual[ruta.name] = extra
                continue
            candidatos = [(g[0].start, etq, " ".join(w.word.strip() for w in g))
                          for g in emisiones(palabras) if (etq := etiqueta(g, vocabulario))]
            propuestos = cortes(candidatos, extra)
            if not propuestos:
                motivo = f"solo explicación, sin palabras sueltas ({len(palabras)} palabras oídas)"
                print(f"  -> anotación manual: {motivo}")
                manual[ruta.name] = motivo
                continue
            for inicio, fin, palabra, oido in propuestos:
                print(f"  {mmss(inicio)}-{mmss(fin)}  {palabra:<20} (oído: {oido})")
                nuevas.append({"video": ruta.name, "palabra": palabra, "inicio": mmss(inicio), "fin": mmss(fin),
                               "origen": "whisper", "oido": oido})
    except KeyboardInterrupt:
        print("\n[INTERRUMPIDO] Se guarda lo etiquetado hasta ahora; vuelve a correr para seguir.")

    if args.rehacer:                                     # lo nuevo reemplaza lo que Whisper propuso antes
        entradas = [e for e in entradas if not (isinstance(e, dict) and e.get("origen") == "whisper"
                                                and e.get("video") in hechos)]
    final = entradas + nuevas
    con_cortes = {e.get("video") for e in final if isinstance(e, dict)}
    manual = {v: m for v, m in sorted(manual.items()) if v not in con_cortes}
    print(f"\n{len(nuevas)} cortes propuestos en {len({e['video'] for e in nuevas})} videos; "
          f"{len(manual)} videos requieren anotación manual.")
    if args.simular:
        print("(--simular: no se escribió nada)")
        return
    escribir_config(args.config, final)
    args.manual.write_text("".join(f"{v}\t{m}\n" for v, m in manual.items()), encoding="utf-8")
    print(f"Escrito {args.config.name} (respaldo en {args.config.with_suffix('.json.bak').name}) y {args.manual.name}.")
    print("Revisa los cortes con \"origen\": \"whisper\" y luego: python video_extractor.py --revisar")


if __name__ == "__main__":
    sys.exit(main())
