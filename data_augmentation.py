"""Aumento de datos para la LSTM: variantes sintéticas de las muestras de dataset/ (30 x 254, ver video_extractor).

Cada variante combina, con parámetros al azar:
  - Jittering: ruido gaussiano en las coordenadas de cada mano presente (no en las ausentes ni en las banderas).
  - Time warping: la seña se recorre con velocidad variable, entre 25/30 y 35/30 de la original por tramos, y se
    vuelve a muestrear a 30 fotogramas. Ojo: estirar a 25 o 35 y regresar a 30 de forma UNIFORME devuelve casi la
    misma secuencia (las muestras ya van del inicio al fin de la seña); lo que cambia la seña es acelerar unas partes
    y frenar otras.
  - Escalado: "distancia a la cámara" escala las coordenadas de imagen alrededor del centro de las manos (también
    el recorrido); "tamaño de mano" escala las coordenadas de mundo (en metros, no dependen de la distancia).

Uso al entrenar (en memoria, sin escribir nada):
  from data_augmentation import aumentar
  from video_extractor import cargar_dataset
  X, y = cargar_dataset()
  X5, y5, origen = aumentar(X, y, factor=5)      # x5: cada original + 4 variantes

  IMPORTANTE: separar entrenamiento y prueba ANTES de aumentar (o agrupar por `origen`). Si variantes de una misma
  muestra caen en ambos lados, la exactitud de prueba sale inflada.

Línea de comandos (exporta para revisar o para Colab):
  python data_augmentation.py --factor 5           # -> dataset_aumentado.npz con X, y, origen
  python data_augmentation.py --factor 10 --semilla 7 --salida aumentado_x10.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from video_extractor import F, LADOS, SALIDA, cargar_dataset, columnas_mano, interpolar

# Medido en dataset/ (57 muestras, 2026-11-05): la mano (muñeca -> nudillo medio) mide 0.068-0.086 del cuadro en la
# imagen (p10-p90: 0.037-0.115) y 8.2-9.4 cm en el mundo. El ruido es ~2 % de esa medida.
RUIDO_IMG = 0.0015             # desviación del ruido en coordenadas de imagen (fracción del ancho/alto)
RUIDO_MUNDO = 0.0018           # desviación del ruido en coordenadas de mundo (m)
VELOCIDAD = (25 / 30, 35 / 30) # rango de velocidad local del time warping
NUDOS = 4                      # tramos con velocidad propia dentro de la seña
DISTANCIA = (0.85, 1.15)       # escala de la imagen: manos más lejos (< 1) o más cerca (> 1). El dataset ya varía
                               # 2.7x entre p10 y p90, así que +-15 % es leve
TAMANO_MANO = (0.9, 1.1)       # escala del mundo: manos más chicas o más grandes


def bloques(k):
    """Columnas de imagen y de mundo de la mano k dentro del vector de F."""
    base = columnas_mano(k).start
    return slice(base, base + 63), slice(base + 63, base + 126)


def jitter(m, rng):
    m = m.copy()
    for k in range(len(LADOS)):
        presente = m[:, k] > 0
        img, mundo = bloques(k)
        m[presente, img] += rng.normal(0, RUIDO_IMG, (presente.sum(), 63))
        m[presente, mundo] += rng.normal(0, RUIDO_MUNDO, (presente.sum(), 63))
    return m


def time_warp(m, rng):
    """Velocidad al azar en NUDOS tramos -> posiciones crecientes de 0 a T-1 -> nueva muestra de T fotogramas."""
    T = len(m)
    velocidad = np.interp(np.linspace(0, NUDOS - 1, T - 1), np.arange(NUDOS), rng.uniform(*VELOCIDAD, NUDOS))
    pos = np.concatenate([[0.0], np.cumsum(velocidad)])
    return interpolar(m, pos * (T - 1) / pos[-1])         # mismos extremos: la seña completa, a otro ritmo


def escalar(m, rng):
    m = m.copy()
    distancia, tamano = rng.uniform(*DISTANCIA), rng.uniform(*TAMANO_MANO)
    puntos = [m[m[:, k] > 0, bloques(k)[0]].reshape(-1, 3) for k in range(len(LADOS))]
    puntos = np.concatenate(puntos) if any(len(p) for p in puntos) else np.zeros((1, 3))
    centro = np.tile(puntos.mean(axis=0) * [1, 1, 0], 21)      # x, y del centro; z ya es relativa a la muñeca
    for k in range(len(LADOS)):
        presente = m[:, k] > 0
        img, mundo = bloques(k)
        m[presente, img] = centro + (m[presente, img] - centro) * distancia
        m[presente, mundo] *= tamano
    return m


def variante(m, rng):
    return escalar(jitter(time_warp(m, rng), rng), rng).astype(np.float32)


def aumentar(X, y, factor=5, semilla=0):
    """(N, T, F) -> (N * factor, T, F): cada original seguido de factor-1 variantes. Devuelve también `origen` (índice
    de la muestra original de cada fila) para separar entrenamiento y prueba sin fugas."""
    if factor < 1:
        raise ValueError("factor debe ser >= 1")
    rng = np.random.default_rng(semilla)
    filas, etiquetas, origen = [], [], []
    for i, (m, e) in enumerate(zip(X, y)):
        filas += [m] + [variante(m, rng) for _ in range(factor - 1)]
        etiquetas += [e] * factor
        origen += [i] * factor
    if not filas:
        return np.zeros((0, *X.shape[1:]), np.float32), np.array(etiquetas), np.array(origen, int)
    return np.stack(filas), np.array(etiquetas), np.array(origen)


def main():
    sys.stdout.reconfigure(errors="replace")
    p = argparse.ArgumentParser(description="Multiplica las muestras de dataset/ con variantes sintéticas.")
    p.add_argument("--factor", type=int, default=5, help="multiplicador total, p. ej. 5 o 10 (default 5)")
    p.add_argument("--semilla", type=int, default=0, help="semilla del azar, para repetir el mismo resultado")
    p.add_argument("--dataset", type=Path, default=SALIDA, help=f"carpeta de muestras (default {SALIDA.name}/)")
    p.add_argument("--salida", type=Path, default=Path("dataset_aumentado.npz"), help="archivo .npz de salida")
    args = p.parse_args()
    X, y = cargar_dataset(args.dataset)
    if len(X) == 0:
        raise SystemExit(f"[ERROR] {args.dataset} no tiene muestras: corre video_extractor.py primero.")
    if X.shape[2] != F:
        raise SystemExit(f"[ERROR] las muestras tienen {X.shape[2]} columnas y se esperaban {F}.")
    Xa, ya, origen = aumentar(X, y, args.factor, args.semilla)
    np.savez_compressed(args.salida, X=Xa, y=ya, origen=origen)
    print(f"{len(X)} muestras -> {len(Xa)} (x{args.factor}) de {len(set(y.tolist()))} palabras en {args.salida}")


if __name__ == "__main__":
    sys.exit(main())
