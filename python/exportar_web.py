"""Exporta lo que necesita la versión web (motor en el navegador, web/motor/) a web/datos/:

- plantillas.bin: descriptor de cada muestra de dataset/ (el mismo de gestos.descriptor), float32 little-endian.
  Por plantilla: presencia (30 x 2, 0/1), forma (30 x 2 x 63) y trayectoria (30 x 2 x 2) = 3960 valores. Pesa la
  mitad que las muestras crudas (30 x 254) y el navegador no tiene que recalcularlo.
- plantillas.json: etiqueta de cada plantilla (en el orden del .bin) y las medidas.
- config.json: lo que server.py manda al conectar (ui_panel.config_web) sin las demostraciones (los videos no se
  publican).

Correr después de cambiar el dataset:  .\\venv\\Scripts\\python.exe python\\exportar_web.py
"""
import json
from pathlib import Path

import numpy as np

from gestos import descriptor
from ui_panel import config_web
from video_extractor import RAIZ, VENTANA, cargar_dataset

DESTINO = RAIZ / "web" / "datos"
VALORES = VENTANA * 2 + VENTANA * 2 * 63 + VENTANA * 2 * 2          # 3960 por plantilla


def exportar(destino=DESTINO):
    destino.mkdir(parents=True, exist_ok=True)
    X, y = cargar_dataset()
    filas = []
    for m in X:
        presencia, forma, trayectoria = descriptor(m)
        filas.append(np.concatenate([presencia.astype(np.float32).ravel(), forma.astype(np.float32).ravel(),
                                     trayectoria.astype(np.float32).ravel()]))
    banco = np.stack(filas).astype("<f4") if filas else np.zeros((0, VALORES), "<f4")
    assert banco.shape[1] == VALORES
    (destino / "plantillas.bin").write_bytes(banco.tobytes())
    meta = {"formato": "presencia(30x2) + forma(30x2x63) + trayectoria(30x2x2), float32 LE", "ventana": VENTANA,
            "valores": VALORES, "etiquetas": [str(e) for e in y]}
    (destino / "plantillas.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    cfg = config_web()
    cfg["demos"] = {}
    (destino / "config.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    print(f"{len(banco)} plantillas ({len(set(y.tolist()))} etiquetas) -> {destino} "
          f"({(destino / 'plantillas.bin').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    exportar(Path(DESTINO))
