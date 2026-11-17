// MediaPipe Tasks falso para la prueba de humo: devuelve los landmarks del fixture del fotograma en curso
// (globalThis.FIXTURE_CUADRO lo pone el conductor de la prueba).
export const FilesetResolver = { forVisionTasks: async () => ({}) };

function manos() {
  const c = globalThis.FIXTURE_CUADRO, [w, h] = globalThis.FIXTURE_TAMANO;
  const tri = (plano, f) => Array.from({ length: 21 }, (_, i) => f(plano.slice(3 * i, 3 * i + 3)));
  return {
    landmarks: c.entrada.dets.map((d) => tri(d.img, ([x, y, z]) => ({ x: x / w, y: y / h, z: z / w }))),
    worldLandmarks: c.entrada.dets.map((d) => tri(d.mundo, ([x, y, z]) => ({ x, y, z }))),
    handedness: c.entrada.dets.map((d) => [{ categoryName: d.l === "Izq" ? "Left" : "Right", score: d.c, index: 0, displayName: "" }]),
  };
}
function rostro() {
  const c = globalThis.FIXTURE_CUADRO, [w, h] = globalThis.FIXTURE_TAMANO;
  if (!c.entrada.cara) return { faceLandmarks: [] };
  const lm = new Array(478);
  for (const [i, [x, y, z]] of Object.entries(c.entrada.cara)) lm[Number(i)] = { x: x / w, y: y / h, z: z / w };
  return { faceLandmarks: [lm] };
}
const tarea = (detectar) => ({ createFromOptions: async (_, opciones) => {
  globalThis.OPCIONES_MP = [...(globalThis.OPCIONES_MP ?? []), opciones];
  return { detectForVideo: (imagen, marca) => { globalThis.MARCAS_MP = [...(globalThis.MARCAS_MP ?? []), marca]; return detectar(); } };
} });
export const HandLandmarker = tarea(manos);
export const FaceLandmarker = tarea(rostro);
