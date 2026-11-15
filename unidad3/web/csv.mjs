import { evaluar } from "./inferencia.mjs";

export const MAX_TRANSACCIONES = 10_000;
export const MAX_BYTES = 5_000_000;

const columnas = ["fecha", "hora", "monto", "modalidad", "estado", "mcc"];
const columnasTabFormer = ["year", "month", "day", "time", "amount", "use chip", "merchant state", "mcc"];

function leerRegistros(contenido) {
  const texto = contenido.replace(/^\uFEFF/, "");
  const cabecera = texto.trimStart().split(/\r?\n/, 1)[0];
  const separador = cabecera.includes(";") && !cabecera.includes(",") ? ";" : ",";
  const registros = [];
  let fila = [];
  let campo = "";
  let estado = "inicio";

  function agregarCampo() {
    fila.push(campo.trim());
    campo = "";
    estado = "inicio";
  }

  function agregarFila() {
    agregarCampo();
    if (fila.length > 1 || fila[0] !== "") registros.push(fila);
    if (registros.length > MAX_TRANSACCIONES + 1) {
      throw new Error("El CSV supera el máximo de 10 000 transacciones.");
    }
    fila = [];
  }

  for (let i = 0; i < texto.length; i++) {
    const caracter = texto[i];
    if (estado === "entrecomillado") {
      if (caracter !== '"') {
        campo += caracter;
      } else if (texto[i + 1] === '"') {
        campo += '"';
        i++;
      } else {
        estado = "cerrado";
      }
    } else if (caracter === separador) {
      agregarCampo();
    } else if (caracter === "\r" || caracter === "\n") {
      agregarFila();
      if (caracter === "\r" && texto[i + 1] === "\n") i++;
    } else if (caracter === '"' && estado === "inicio" && campo.trim() === "") {
      campo = "";
      estado = "entrecomillado";
    } else if (caracter === '"' || (estado === "cerrado" && caracter.trim() !== "")) {
      throw new Error("El CSV contiene comillas inválidas cerca del registro " + registros.length + ".");
    } else {
      campo += caracter;
    }
  }
  if (estado === "entrecomillado") throw new Error("El CSV contiene un campo con comillas sin cerrar.");
  agregarFila();
  return { registros, separador };
}

function fechaTabFormer(year, month, day) {
  if (!/^\d{4}$/.test(year) || !/^\d{1,2}$/.test(month) || !/^\d{1,2}$/.test(day)) return "";
  return [year, month.padStart(2, "0"), day.padStart(2, "0")].join("-");
}

function importeNumerico(valor, separador) {
  let limpio = valor.replace(/^([+-]?)\s*\$\s*/, "$1");
  if (separador === ";" && !limpio.includes(".")) limpio = limpio.replace(",", ".");
  return /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(limpio) ? Number(limpio) : NaN;
}

export function evaluarCsv(contenido, modelo, artefactos) {
  const { registros, separador } = leerRegistros(contenido);
  if (registros.length < 2) throw new Error("El CSV debe tener encabezados y al menos una transacción.");
  const [encabezados, ...filas] = registros;
  const indices = new Map(encabezados.map((columna, i) => [columna.toLowerCase(), i]));
  const esPlantilla = columnas.every(columna => indices.has(columna));
  if (!esPlantilla && !columnasTabFormer.every(columna => indices.has(columna))) {
    throw new Error("Faltan columnas requeridas. Usa la plantilla CSV o los encabezados originales de TabFormer.");
  }
  const requeridas = esPlantilla ? columnas : columnasTabFormer;
  for (const columna of requeridas) {
    if (encabezados.filter(nombre => nombre.toLowerCase() === columna).length > 1) {
      throw new Error("Hay un encabezado repetido: " + columna + ".");
    }
  }

  return filas.map((fila, indice) => {
    const registro = indice + 1;
    if (fila.length !== encabezados.length) {
      return { registro, error: "La cantidad de columnas difiere del encabezado." };
    }
    const valores = requeridas.map(nombre => fila[indices.get(nombre)]);
    let [fecha, hora, monto, modalidad, estado, mcc] = valores;
    if (!esPlantilla) {
      fecha = fechaTabFormer(...valores.slice(0, 3));
      [hora, monto, modalidad, estado, mcc] = valores.slice(3);
    }
    const datos = {
      fecha, hora: hora.padStart(5, "0"), monto: importeNumerico(monto, separador),
      modalidad, estado, mcc: /^\d{1,4}$/.test(mcc) ? Number(mcc) : NaN,
    };
    try {
      return { registro, fecha: datos.fecha, monto: datos.monto, ...evaluar(modelo, datos, artefactos) };
    } catch (error) {
      return { registro, error: error.message };
    }
  });
}

export function resultadosCsv(filas, modelo) {
  const campos = ["registro", "modelo", "fecha", "importe_usd", "puntuacion_0_a_100",
    "umbral_0_a_100", "prioridad", "accion", "error"];
  const valorCsv = valor => typeof valor === "number" ? String(valor) :
    '"' + String(valor ?? "").replaceAll('"', '""') + '"';
  const filasCsv = filas.map(fila => {
    const [puntos, umbral] = [fila.puntuacion, fila.umbral].map(valor =>
      valor == null ? "" : Number((valor * 100).toFixed(8)),
    );
    return [fila.registro, modelo, fila.fecha, fila.monto, puntos, umbral, fila.prioridad, fila.accion, fila.error]
      .map(valorCsv).join(",");
  });
  return "\uFEFF" + [campos.join(","), ...filasCsv].join("\r\n") + "\r\n";
}
