import { evaluar, MLP, HGB } from "./inferencia.mjs";
import { evaluarCsv, resultadosCsv, MAX_BYTES } from "./csv.mjs";
import { conectarResultados } from "./resultados.mjs";

const porId = id => document.getElementById(id);
const formulario = porId("transaccion");
const boton = porId("evaluar");
const error = porId("error");
const prioridad = porId("prioridad");
const puntuacion = porId("puntuacion");
const barra = porId("barra");
const marca = porId("marca");
const umbral = porId("umbral");
const accion = porId("accion");
const modos = document.querySelectorAll("[data-modo]");
const formularioLote = porId("lote");
const botonLote = porId("evaluar-lote");
const errorLote = porId("lote-error");
const resultadoLote = porId("resultado-lote");

for (const modo of modos) {
  modo.addEventListener("click", () => {
    for (const opcion of modos) {
      const activo = modo === opcion;
      opcion.setAttribute("aria-pressed", String(activo));
      porId(opcion.dataset.modo).hidden = !activo;
    }
  });
}

const casos = {
  prioritaria: ["2019-06-17", "12:30", 850, "Chip Transaction", "CA", 5411],
  online: ["2019-06-15", "02:15", 850, "Online Transaction", "ONLINE", 5812],
  presencial: ["2019-06-17", "14:30", 35.50, "Chip Transaction", "CA", 5411],
  reembolso: ["2019-06-17", "14:30", -35.50, "Chip Transaction", "CA", 5411],
};
const nombres = ["fecha", "hora", "monto", "modalidad", "estado", "mcc"];
const control = nombre => formulario.elements.namedItem(nombre);
const botonesCaso = document.querySelectorAll("[data-caso]");
const decimal = new Intl.NumberFormat("es-PE", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const entero = new Intl.NumberFormat("es-PE");
const promedio = new Intl.NumberFormat("es-PE", { minimumFractionDigits: 4, maximumFractionDigits: 4 });
const porcentaje = new Intl.NumberFormat("es-PE", {
  style: "percent", minimumFractionDigits: 2, maximumFractionDigits: 2,
});

function seleccionarCaso(caso) {
  for (const opcion of botonesCaso) {
    opcion.setAttribute("aria-pressed", String(opcion.dataset.caso === caso));
  }
  if (caso) nombres.forEach((nombre, i) => { control(nombre).value = casos[caso][i]; });
}

function mostrar(resultado) {
  const score = resultado?.puntuacion;
  const corte = resultado?.umbral;
  prioridad.textContent = resultado?.prioridad ?? "Pendiente";
  prioridad.dataset.tone = score == null ? "neutral" : score >= corte ? "priority" : "routine";
  puntuacion.textContent = score == null ? "—" : decimal.format(score * 100);
  barra.style.width = (score ?? 0) * 100 + "%";
  marca.hidden = corte == null;
  marca.style.left = (corte ?? 0) * 100 + "%";
  umbral.textContent = corte == null ? "" : "Umbral de revisión: " + decimal.format(corte * 100) + " / 100";
  accion.textContent = resultado?.accion ?? "Evalúa una transacción para obtener una recomendación.";
}

function llenarMetricas(evaluacion) {
  for (const modelo of [MLP, HGB]) {
    const metricas = evaluacion.modelos[modelo];
    const operativo = metricas.test_operativo;
    const fila = porId("metricas").insertRow();
    const etiqueta = document.createElement("th");
    etiqueta.scope = "row";
    etiqueta.textContent = modelo;
    fila.append(etiqueta);
    const datos = [
      promedio.format(metricas.test_average_precision), promedio.format(metricas.test_roc_auc),
      entero.format(operativo.confusion.tp), porcentaje.format(operativo.recall), porcentaje.format(operativo.precision),
    ];
    for (const valor of datos) fila.insertCell().textContent = valor;
  }
}

function conectarLote(artefactos) {
  const campoArchivo = formularioLote.elements.namedItem("archivo");
  const campoModelo = formularioLote.elements.namedItem("modelo");
  const mostrarLote = conectarResultados(resultadoLote);
  let evaluacion = null;

  formularioLote.addEventListener("change", () => {
    evaluacion = null;
    resultadoLote.hidden = true;
    errorLote.hidden = true;
  });
  formularioLote.addEventListener("submit", async evento => {
    evento.preventDefault();
    const archivo = campoArchivo.files[0];
    if (!archivo) return;
    const modelo = campoModelo.value;
    evaluacion = null;
    resultadoLote.hidden = true;
    errorLote.hidden = true;
    botonLote.disabled = campoArchivo.disabled = campoModelo.disabled = true;
    botonLote.textContent = "Evaluando archivo…";
    try {
      if (archivo.size > MAX_BYTES) throw new Error("El archivo supera 5 MB. Divide las transacciones en varios CSV.");
      const filas = evaluarCsv(await archivo.text(), modelo, artefactos);
      evaluacion = { filas, modelo };
      mostrarLote(filas, modelo, artefactos.evaluacion.modelos[modelo].umbral_operativo);
    } catch (error) {
      errorLote.textContent = error.message;
      errorLote.hidden = false;
    } finally {
      botonLote.textContent = "Evaluar archivo";
      botonLote.disabled = campoArchivo.disabled = campoModelo.disabled = false;
    }
  });
  porId("descargar-lote").addEventListener("click", () => {
    if (!evaluacion) return;
    const { filas, modelo } = evaluacion;
    const blob = new Blob([resultadosCsv(filas, modelo)], { type: "text/csv;charset=utf-8" });
    const enlace = document.createElement("a");
    enlace.href = URL.createObjectURL(blob);
    enlace.download = "resultados_transacciones.csv";
    enlace.click();
    setTimeout(() => URL.revokeObjectURL(enlace.href), 1000);
  });
  botonLote.textContent = "Evaluar archivo";
  botonLote.disabled = false;
}

async function iniciar() {
  const respuesta = await fetch("./modelos.json");
  if (!respuesta.ok) throw new Error("No se pudieron cargar los modelos.");
  const artefactos = await respuesta.json();
  llenarMetricas(artefactos.evaluacion);
  conectarLote(artefactos);

  function calcular() {
    error.hidden = true;
    const datos = Object.fromEntries(nombres.map(nombre => [nombre, control(nombre).value]));
    datos.monto = control("monto").valueAsNumber;
    datos.mcc = control("mcc").valueAsNumber;
    try {
      mostrar(evaluar(control("modelo").value, datos, artefactos));
    } catch (fallo) {
      mostrar(null);
      error.textContent = fallo.message;
      error.hidden = false;
    }
  }

  formulario.addEventListener("submit", evento => {
    evento.preventDefault();
    calcular();
  });
  formulario.addEventListener("input", evento => {
    if (evento.target.name === "modelo") {
      if (formulario.checkValidity()) calcular();
    } else {
      seleccionarCaso(null);
      error.hidden = true;
      mostrar(null);
    }
  });
  formulario.addEventListener("reset", () => {
    seleccionarCaso(null);
    error.hidden = true;
    mostrar(null);
  });
  for (const opcion of botonesCaso) {
    opcion.addEventListener("click", () => {
      seleccionarCaso(opcion.dataset.caso);
      calcular();
    });
  }

  boton.textContent = "Evaluar transacción";
  boton.disabled = false;
  seleccionarCaso(null);
}

iniciar().catch(() => {
  boton.textContent = "Modelos no disponibles";
  error.textContent = "No se pudieron cargar los modelos. Recarga la página para volver a intentarlo.";
  error.hidden = false;
  botonLote.textContent = boton.textContent;
  errorLote.textContent = error.textContent;
  errorLote.hidden = false;
});
