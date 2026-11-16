import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { evaluar, MLP, HGB } from '../web/inferencia.mjs';
import { evaluarCsv, resultadosCsv } from '../web/csv.mjs';

const base = new URL('../web/', import.meta.url);
const artefactos = JSON.parse(await readFile(new URL('modelos.json', base), 'utf8'));
const ejemplo = { fecha: '2019-06-17', hora: '12:30', monto: 850, modalidad: 'Chip Transaction', estado: 'CA', mcc: 5411 };
for (const modelo of [MLP, HGB]) {
  const resultado = evaluar(modelo, ejemplo, artefactos);
  assert.ok(Number.isFinite(resultado.puntuacion) && resultado.puntuacion >= 0 && resultado.puntuacion <= 1);
  assert.ok(Number.isFinite(resultado.umbral));
  assert.equal(evaluar(modelo, {...ejemplo, monto: -1}, artefactos).prioridad, 'Fuera del alcance');
  const filas = evaluarCsv(await readFile(new URL('plantilla_transacciones.csv', base), 'utf8'), modelo, artefactos);
  assert.ok(filas.length > 0 && filas.every(f => !f.error));
  assert.match(resultadosCsv(filas, modelo), /puntuacion_0_a_100/);
}
console.log('Modelos, excepción y procesamiento CSV verificados.');
