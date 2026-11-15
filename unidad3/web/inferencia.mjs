export const MLP = "MLP";
export const HGB = "HistGradientBoosting";
const modalidades = ["Chip Transaction", "Swipe Transaction", "Online Transaction"];

const rangosMcc = [
  [3000, 3299, "Airlines"],
  [3300, 3499, "Car_Rental"],
  [3500, 3999, "Lodging_Hotels"],
  [4000, 4799, "Transportation"],
  [4800, 4999, "Utilities_Telecom"],
  [5200, 5499, "Supermarkets_Groceries"],
  [5500, 5599, "Automotive_Fuel"],
  [5600, 5699, "Apparel_Clothing"],
  [5800, 5899, "Restaurants_Dining"],
  [5900, 5999, "Misc_Retail"],
  [6000, 6999, "Financial_Services"],
  [7000, 7999, "Entertainment_Recreation"],
  [8000, 8999, "Professional_Health"],
];

export function preparar(datos, { escalado, categorias }) {
  const fecha = new Date(datos.fecha + "T00:00:00Z");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(datos.fecha) || datos.fecha.startsWith("0000") ||
      Number.isNaN(+fecha) || fecha.toISOString().slice(0, 10) !== datos.fecha) {
    throw new Error("Ingresa una fecha válida.");
  }
  if (!/^([01]\d|2[0-3]):[0-5]\d$/.test(datos.hora)) {
    throw new Error("Ingresa la hora local en formato HH:MM.");
  }
  if (!Number.isFinite(datos.monto)) throw new Error("Ingresa un importe válido.");
  if (!modalidades.includes(datos.modalidad)) throw new Error("Selecciona una modalidad válida.");
  if (!Number.isInteger(datos.mcc) || datos.mcc < 0 || datos.mcc > 9999) {
    throw new Error("Ingresa un MCC entero entre 0000 y 9999.");
  }

  const [hora, minuto] = datos.hora.split(":").map(Number);
  const dia = fecha.getUTCDay() || 7;
  const angulo = hora * 2 * Math.PI / 24;
  const numericas = [
    Math.log(Math.max(datos.monto, 0) + 1), Math.sin(angulo), Math.cos(angulo),
    minuto, dia, Number(dia >= 6), Number(hora <= 5),
  ];
  const categoria = rangosMcc.find(([desde, hasta]) => datos.mcc >= desde && datos.mcc <= hasta)?.[2] ?? "Other";
  const online = datos.estado.trim().toUpperCase() === "ONLINE" || datos.modalidad === modalidades[2];
  const categoricas = [datos.modalidad, categoria, online ? "ONLINE" : "IN_PERSON"];

  return Float32Array.from([
    ...numericas.map((valor, indice) => (valor - escalado.media[indice]) / escalado.escala[indice]),
    ...categorias.flatMap((columna, indice) => columna.map(valor => Number(valor === categoricas[indice]))),
  ]);
}

const sigmoide = valor => 1 / (1 + Math.exp(-valor));
const relu = valor => Math.max(0, valor);

function densa(entrada, capa, activar) {
  return Float32Array.from(capa.pesos, (pesos, indice) => {
    let valor = capa.sesgos[indice];
    for (let i = 0; i < entrada.length; i++) valor += entrada[i] * pesos[i];
    return activar(valor);
  });
}

function redNeuronal(entrada, red) {
  const primera = densa(entrada, red.densa1, relu);
  const { media, varianza, gamma, beta, epsilon } = red.normalizacion;
  const normalizada = primera.map((valor, i) =>
    (valor - media[i]) * gamma[i] / Math.sqrt(varianza[i] + epsilon) + beta[i],
  );
  return densa(densa(normalizada, red.densa2, relu), red.salida, sigmoide)[0];
}

function gradientBoosting(entrada, modelo) {
  let total = modelo.base;
  for (const arbol of modelo.arboles) {
    let nodo = arbol[0];
    while (Array.isArray(nodo)) {
      const [variable, umbral, izquierda, derecha] = nodo;
      nodo = arbol[entrada[variable] <= umbral ? izquierda : derecha];
    }
    total += nodo;
  }
  return sigmoide(total);
}

export function evaluar(modelo, datos, artefactos) {
  const entrada = preparar(datos, artefactos);
  if (datos.monto <= 0) {
    return {
      puntuacion: null,
      prioridad: "Fuera del alcance",
      accion: "Derivar a revisión de reembolsos, contracargos u operaciones de importe cero.",
    };
  }
  const umbral = artefactos.evaluacion.modelos[modelo].umbral_operativo;
  const puntuacion = modelo === MLP ? redNeuronal(entrada, artefactos.mlp) : gradientBoosting(entrada, artefactos.hgb);
  const revision = puntuacion >= umbral;
  return {
    puntuacion,
    umbral,
    prioridad: revision ? "Revisión prioritaria" : "Seguimiento habitual",
    accion: revision ? "Enviar a un analista para verificar la operación." :
      "Continuar el flujo habitual y conservar el registro.",
  };
}
