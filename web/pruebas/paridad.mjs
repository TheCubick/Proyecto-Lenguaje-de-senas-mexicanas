// Paridad del motor del navegador (web/motor/) contra Python sobre los fixtures de generar_fixtures.py: señas por
// mano, vistas, mano principal, texto fotograma a fotograma, rostro (modo, eventos, ángulos del solvePnP) y
// decisiones del DTW. Sale con código 1 si algo difiere.   Uso (desde la raíz): node web/pruebas/paridad.mjs
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const aqui = new URL(".", import.meta.url);
const M = await import(new URL("../motor/manos.js", aqui));
const G = await import(new URL("../motor/gestos.js", aqui));
const { Traductor } = await import(new URL("../motor/traductor.js", aqui));
const P = await import(new URL("../motor/presentacion.js", aqui));

const meta = JSON.parse(readFileSync(new URL("../datos/plantillas.json", aqui), "utf8"));
const bin = readFileSync(new URL("../datos/plantillas.bin", aqui));
const plantillas = G.leerPlantillas(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), meta);
const a3 = (plano) => Array.from({ length: 21 }, (_, i) => plano.slice(3 * i, 3 * i + 3));
let fallas = 0;

for (const nombre of ["alfabeto", "frases"]) {
  const ruta = new URL(`fixtures/fixture_${nombre}.json`, aqui);
  if (!existsSync(ruta)) { console.log(`[${nombre}] falta ${fileURLToPath(ruta)}: corre generar_fixtures.py`); fallas++; continue; }
  const fx = JSON.parse(readFileSync(ruta, "utf8"));
  const rec = new G.ReconocedorGestos(plantillas, { asincrono: false, propias: false });
  const tr = new Traductor(rec);
  const barra = new M.ProgresoSuave();
  let principal = null, mal = { senas: 0, vistas: 0, principal: 0, texto: 0, rostro: 0, dtw: 0 };
  let maxRasgo = 0, maxAngulo = 0, evaluaciones = 0, lento = 0, primero = null;
  for (const [k, c] of fx.cuadros.entries()) {
    const t = c.entrada.t;
    const dets = c.entrada.dets.map((d) => new M.DeteccionMano(d.l, d.c, a3(d.img), a3(d.mundo)));
    const t0 = performance.now();
    const [manos, evento] = tr.paso(dets, c.entrada.cara, t, fx.tamano);
    lento = Math.max(lento, performance.now() - t0);
    const vistas = {};
    for (const l of M.LADOS) {
      const em = manos[l], [cruda, pose, gesto, mov, , rp] = c.salida.m[l];
      if (em.poseCruda !== cruda || em.pose !== pose || em.gesto !== gesto || em.moviendo !== mov) mal.senas++;
      if (rp && em.rasgos) maxRasgo = Math.max(maxRasgo, Math.abs(em.rasgos.orient - rp.orient), Math.abs(em.rasgos.dTi - rp.dti),
        ...em.rasgos.extension.map((e, i) => Math.abs(e - rp.ext[i])));
      const v = P.vistaMano(em, tr.escritor.manos[l], t);
      v.progreso = tr.canales[l].barra.actualizar(v.progreso);
      vistas[l] = v;
      const [sena, estado, prog] = c.salida.v[l];
      if (v.sena !== sena || v.estado !== estado || Math.abs(v.progreso - prog) > 1e-5) mal.vistas++;
    }
    principal = P.manoPrincipal(vistas, principal);
    const prog = barra.actualizar(principal ? vistas[principal].progreso : 0.0);
    if (principal !== c.salida.p[0] || Math.abs(prog - c.salida.p[1]) > 1e-5) mal.principal++;
    if (tr.escritor.texto !== c.salida.tx) { mal.texto++; primero ??= { k, js: tr.escritor.texto, py: c.salida.tx }; }
    const [modo, pose] = c.salida.f, pj = tr.analizador.poseActual;
    if (tr.analizador.modo !== modo || (evento ?? null) !== c.salida.ev) mal.rostro++;
    if (pose && pj) maxAngulo = Math.max(maxAngulo, Math.abs(pj.pitch - pose[0]), Math.abs(pj.yaw - pose[1]), Math.abs(pj.roll - pose[2]));
    const gj = tr.gestos.sacarResultado(), gp = c.salida.g;
    if (gp) evaluaciones++;
    if (!!gj !== !!gp || (gp && (gj.mejor !== gp.mejor || gj.aceptado !== gp.aceptado || gj.motivo !== gp.motivo
        || gj.escrito !== gp.escrito || (gp.mejor && Math.abs(gj.distancia - gp.distancia) > 1e-3)))) mal.dtw++;
  }
  const distinto = Object.values(mal).reduce((a, b) => a + b, 0);
  fallas += distinto + (maxRasgo > 1e-5) + (maxAngulo > 1e-3);
  console.log(`[${nombre}] ${fx.cuadros.length} fotogramas, ${evaluaciones} evaluaciones DTW | diferencias: ${JSON.stringify(mal)} | ` +
    `máx dif rasgos ${maxRasgo.toExponential(1)}, ángulos ${maxAngulo.toExponential(1)}° | fotograma más lento ${lento.toFixed(0)} ms`);
  if (primero) console.log("  primera diferencia de texto:", JSON.stringify(primero));
}
console.log(fallas ? "PARIDAD: HAY DIFERENCIAS" : "PARIDAD: idéntico a Python");
process.exit(fallas ? 1 : 0);
