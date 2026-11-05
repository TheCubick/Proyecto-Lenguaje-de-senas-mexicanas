"""Capa de presentación de la interfaz web: qué se muestra de cada mano y el estado JSON que consume static/app.js
(seña, progreso, coordenadas de las manos, estela del trazo, rostro). Python no dibuja nada: el navegador muestra
su propio video de la cámara y dibuja el esqueleto encima con estas coordenadas."""
from dataclasses import dataclass
from typing import Optional

import mediapipe as mp

from classifier_face import AFIRM_AMPLITUD, CEJAS_UMBRAL, NEG_AMPLITUD
from classifier_hands import GESTOS_DINAMICOS, LADOS, LETRAS_TIPEABLES, POSES_SILENCIOSAS, PUNTO_DE_POSE
from traductor import SEGUNDOS_PARA_ESCRIBIR
from voice_engine import limpiar_texto

CONEXIONES_MANO = sorted(tuple(c) for c in mp.solutions.hands.HAND_CONNECTIONS)   # pares de índices (huesos)

DURACION_DESTELLO = 1.2        # Segundos que se muestra en grande una seña con movimiento recién registrada
DECIMALES = 4                  # 1e-4 de la imagen = 0.06 px a 640: de sobra para dibujar; JSON más ligero
ESTELA_MAX = 32                # Puntos máximos de la estela que se mandan (se submuestrea)

# Alfabeto: (clave, cómo se hace, movimiento). Lámina oficial SEP
GUIA = [
    ("A", "Puño, pulgar estirado al lado", None),
    ("B", "4 dedos juntos, pulgar cruzado", None),
    ("C", "Mano curva en forma de C", None),
    ("D", "Índice arriba, otros cerrados", None),
    ("E", "Dedos doblados sobre el pulgar", None),
    ("F", "Índice y pulgar unidos, 3 dedos arriba", None),
    ("G", "Pistola: índice de lado", None),
    ("H", "Pistola: índice y medio de lado", None),
    ("I", "Solo el meñique arriba", None),
    ("J", "I + gancho (baja y curva)", "Mueve el meñique: baja y haz un gancho"),
    ("K", "P + agitar de lado a lado", "Agita la mano (grande) de lado a lado"),
    ("L", "Índice arriba y pulgar al lado", None),
    ("M", "3 dedos hacia abajo", None),
    ("N", "2 dedos hacia abajo", None),
    ("Ñ", "N + vaivén de lado a lado", "Agita la N de lado a lado"),
    ("O", "Círculo índice-pulgar", None),
    ("P", "Índice y medio cruzados en X", None),
    ("Q", "Pinza abajo + balanceo lateral", "Balancea la pinza de lado a lado"),
    ("R", "Índice y medio entrelazados", None),
    ("S", "Puño, pulgar sobre los dedos", None),
    ("T", "Puño, pulgar entre los dedos", None),
    ("U", "Índice y medio juntos", None),
    ("V", "Índice y medio en V", None),
    ("W", "Índice, medio y anular abiertos", None),
    ("X", "Índice de lado + vaivén", "Mueve el índice de lado a lado"),
    ("Y", "Pulgar y meñique estirados", None),
    ("Z", "D + trazo de Z en el aire", "Dibuja una Z en el aire con el índice"),
    ("ESPACIO", "Mano abierta, 5 dedos", None),
    ("BORRAR", "Mano abierta barre al centro", "Barre la mano abierta hacia el centro del cuerpo"),
    ("?", "Índice dibuja ? en el aire", "Dibuja un ? en el aire con el índice"),
]

# Pistas para las configuraciones que sirven de base a un trazo
PISTA_TRAZO = {"I": "o traza una J", "Y": "o traza una J", "D": "o traza Z / ?", "P": "o agita para K",
               "R": "o agita para K", "U": "o agita para K", "N": "o agita para Ñ", "Q": "balancea para Q",
               "X": "mueve para X", "B": "o barre al centro", "ESPACIO": "o barre al centro"}


@dataclass
class VistaMano:
    """Cómo se muestra una mano: seña visible, estado ("sostener", "mover", "registrada", "pose") y progreso."""
    sena: Optional[str]
    estado: str
    progreso: float


def vista_mano(em, sostener, ahora):
    """Qué se muestra de una mano: el trazo recién hecho, la seña que se sostiene, o nada."""
    gesto, t_gesto = sostener.destello
    if gesto and ahora - t_gesto < DURACION_DESTELLO:
        return VistaMano(gesto, "registrada", 1.0)
    if em.pose is not None and em.rasgos is not None and em.pose not in POSES_SILENCIOSAS:
        if em.pose not in LETRAS_TIPEABLES:
            return VistaMano(em.pose, "pose", em.progreso)
        if em.moviendo:
            return VistaMano(em.pose, "mover", em.progreso)
        return VistaMano(em.pose, "registrada" if sostener.registrada else "sostener", em.progreso)
    return VistaMano(None, "ninguna", 0.0)


def mano_principal(vistas, anterior):
    """La mano que se muestra en grande: la que acaba de registrar o la que más avanzó; en empate, la anterior."""
    candidatas = [lado for lado in LADOS if vistas[lado].sena is not None]
    if not candidatas:
        return None
    return max(candidatas, key=lambda lado: (vistas[lado].estado == "registrada", vistas[lado].progreso,
                                             lado == anterior))


# --- Estado para el navegador -----------------------------------------------------
def config_web():
    """Lo que no cambia (se manda una vez al conectar): guía del alfabeto, umbral de las cejas, segundos del
    hold-to-type, huesos de la mano (pares de puntos) y, para el Modo Práctica, la configuración base de cada
    seña con movimiento (J <- I/Y...)."""
    bases = {}
    for gesto, poses, *_ in GESTOS_DINAMICOS:
        for g in (sorted(poses) if gesto == "QX" else [gesto]):      # Q y X comparten trazo; base = sí mismas
            bases.setdefault(g, set()).update({g} if gesto == "QX" else poses - POSES_SILENCIOSAS)
    return {"tipo": "config", "umbral_cejas": CEJAS_UMBRAL, "segundos": SEGUNDOS_PARA_ESCRIBIR,
            "conexiones": CONEXIONES_MANO, "bases": {g: sorted(p) for g, p in bases.items()},
            "guia": [{"clave": c, "desc": d, "mov": m} for c, d, m in GUIA]}


def _normalizar(x, y, ancho, alto):
    return [round(float(x) / ancho, DECIMALES), round(float(y) / alto, DECIMALES)]


def _mano_web(em, vista, tray, ancho, alto):
    """Una mano para el navegador. `puntos`: 21 x [x, y, z] normalizados a la imagen espejada (x, y en 0-1;
    z = profundidad relativa a la muñeca en la escala del ancho, negativa hacia la cámara)."""
    caja = puntos = estela = None
    if em.rasgos is not None:
        p = em.rasgos.p
        caja = _normalizar(p[:, 0].min(), p[:, 1].min(), ancho, alto) + _normalizar(p[:, 0].max(), p[:, 1].max(),
                                                                                       ancho, alto)
        if em.img is not None:
            puntos = [_normalizar(q[0], q[1], ancho, alto) + [round(float(q[2]) / ancho, DECIMALES)] for q in em.img]
        if vista.sena in PUNTO_DE_POSE and tray.vel > 0.6:          # estela solo mientras la mano se mueve
            serie = tray.serie_px(PUNTO_DE_POSE[vista.sena])
            paso = max(1, len(serie) // ESTELA_MAX)
            estela = [_normalizar(q[0], q[1], ancho, alto) for q in serie[::paso]]
    pista = PISTA_TRAZO.get(vista.sena) if vista.estado in ("sostener", "registrada", "mover", "pose") else None
    return {"sena": vista.sena, "estado": vista.estado, "progreso": round(vista.progreso * 100, 1),
            "caja": caja, "pista": pista, "puntos": puntos, "estela": estela}


def lineas_diagnostico(manos, canales, analizador):
    """Rasgos 3D de cada mano y del rostro (tecla D)."""
    lineas = []
    for lado in LADOS:
        r = manos[lado].rasgos
        if r is None:
            continue
        lineas.append(f"{lado}: " + " ".join(f"{e}{x:.2f}" for e, x in zip(r.estados, r.extension)))
        lineas.append("  flex " + " ".join(f"{f:.0f}" for f in r.flexion) + f"  orient={r.orient:.0f}"
                      f" dirInd={r.dir_indice:.0f}")
        lineas.append(f"  pulgar lat={r.pulgar_lat:+.2f} alt={r.pulgar_alt:+.2f} dTI={r.d_ti:.2f}"
                      f" normalZ={r.normal[2]:+.2f}")
        lineas.append(f"  cruce={r.cruce:+.2f} angIM={r.ang_im:.0f} prof={r.prof_medio:+.2f}"
                      f" vel={canales[lado].tray.vel:.1f}")
    if analizador.pose_actual is not None:
        lineas.append(f"Yaw {analizador.negacion.desviacion:+.1f} (+-{NEG_AMPLITUD:.0f})  Pitch "
                      f"{analizador.afirmacion.desviacion:+.1f} (+-{AFIRM_AMPLITUD:.0f})  "
                      f"{analizador.pose_actual.distancia:.0f} cm  ojos x{analizador.ratio_ojos:.2f}")
    return lineas


def estado_web(traductor, manos, vistas, principal, progreso, fps, voz, tamano, diagnostico, tiempos):
    """Todo lo que la interfaz necesita en un frame. Progresos en 0-100. `tiempos`: número de fotograma, hora de
    captura (ms de reloj de pared, para que el navegador mida la latencia), ms de proceso y descartados."""
    ancho, alto = tamano
    vp = vistas[principal] if principal else VistaMano(None, "ninguna", 0.0)
    an = traductor.analizador
    pose = an.pose_actual
    return {
        "tipo": "estado",
        "sena": vp.sena, "estado": vp.estado, "progreso": round(progreso * 100, 1), "lado": principal,
        "manos": {lado: _mano_web(manos[lado], vistas[lado], traductor.canales[lado].tray, ancho, alto)
                  for lado in LADOS},
        "texto": limpiar_texto(traductor.escritor.texto),
        "rostro": {"modo": an.modo, "pregunta": round(an.progreso_pregunta * 100),
                   "calibracion": round(an.progreso_calibracion * 100), "cejas": round(an.ratio_cejas, 3),
                   "cabeza": None if pose is None else [round(pose.pitch), round(pose.yaw), round(pose.roll)]},
        "voz": voz, "fps": round(fps, 1), "tamano": [ancho, alto], "tiempos": tiempos,
        "diagnostico": lineas_diagnostico(manos, traductor.canales, an) if diagnostico else None,
    }
