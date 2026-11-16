const entero = new Intl.NumberFormat("es-PE");
const decimal = new Intl.NumberFormat("es-PE", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const compacto = new Intl.NumberFormat("es-PE", { notation: "compact", maximumFractionDigits: 1 });
const tamanoPagina = 6;
const grupos = new Map([
  ["todas", ["Todas", ""]],
  ["prioritaria", ["Prioritarias", "Prioritaria"]],
  ["habitual", ["Habituales", "Habitual"]],
  ["fuera", ["Fuera de alcance", "Fuera de alcance"]],
  ["error", ["Con errores", "Dato inválido"]],
]);

function grupoDe(fila) {
  if (fila.error) return "error";
  if (fila.puntuacion == null) return "fuera";
  return fila.puntuacion >= fila.umbral ? "prioritaria" : "habitual";
}

function importe(monto) {
  return monto == null ? "—" : "$" + decimal.format(monto);
}

function etiquetaImporte(valor) {
  if (valor < 0.01 || valor >= 1e9) return valor.toExponential(0);
  if (valor >= 1000) return compacto.format(valor);
  return Number(valor.toPrecision(2)).toLocaleString("es-PE");
}

function marca(ctx, { x, y, grupo }, activa, visible) {
  const prioritaria = grupo === "prioritaria";
  const radio = activa ? 6 : prioritaria ? 4.5 : 3.5;
  ctx.globalAlpha = visible ? activa ? 1 : 0.8 : 0.18;
  ctx.fillStyle = prioritaria ? "#a6382c" : "#3e6789";
  ctx.beginPath();
  if (prioritaria) {
    ctx.moveTo(x, y - radio);
    ctx.lineTo(x + radio, y);
    ctx.lineTo(x, y + radio);
    ctx.lineTo(x - radio, y);
    ctx.closePath();
  } else {
    ctx.arc(x, y, radio, 0, 2 * Math.PI);
  }
  ctx.fill();
  if (activa) {
    ctx.strokeStyle = "#273342";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

function dibujar(canvas, puntuadas, umbral, filtro, seleccion) {
  const ancho = canvas.clientWidth;
  const alto = canvas.clientHeight;
  if (ancho < 100 || !alto) return [];
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(ancho * dpr);
  canvas.height = Math.round(alto * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return [];
  ctx.scale(dpr, dpr);
  ctx.font = "12px system-ui";
  ctx.fillStyle = "#58616e";
  if (!puntuadas.length) {
    ctx.textAlign = "center";
    ctx.fillText("Este lote no tiene compras con puntaje.", ancho / 2, alto / 2, ancho - 12);
    return [];
  }

  const izquierda = 34, derecha = ancho - 18, arriba = 20, abajo = alto - 27;
  const valores = puntuadas.map(fila => Math.log10(fila.monto));
  const minimo = Math.min(...valores);
  const maximo = Math.max(...valores);
  const margen = Math.max((maximo - minimo) * 0.04, 0.12);
  const desde = minimo - margen, hasta = maximo + margen;
  const anchoGrafico = derecha - izquierda;
  const y = score => abajo - score * (abajo - arriba);
  const corte = y(umbral);

  ctx.fillStyle = "#f7ebe6";
  ctx.fillRect(izquierda, arriba, anchoGrafico, corte - arriba);
  ctx.fillStyle = "#f2f5f7";
  ctx.fillRect(izquierda, corte, anchoGrafico, abajo - corte);
  for (const nivel of [0, 25, 50, 75, 100]) {
    const altura = y(nivel / 100);
    ctx.fillStyle = "#58616e";
    ctx.textAlign = "right";
    ctx.fillText(nivel, izquierda - 7, altura + 4);
    ctx.fillStyle = "#dce3e7";
    ctx.fillRect(izquierda, altura, anchoGrafico, 1);
  }
  const intervalo = ancho < 360 ? 3 : 4;
  ctx.fillStyle = "#58616e";
  ctx.textAlign = "center";
  for (let i = 0; i <= intervalo; i++) {
    const valor = 10 ** (minimo + (maximo - minimo) * i / intervalo);
    if (minimo === maximo && i !== Math.floor(intervalo / 2)) continue;
    if (!Number.isFinite(valor) || valor === 0) continue;
    const x = izquierda + ((Math.log10(valor) - desde) / (hasta - desde)) * anchoGrafico;
    ctx.fillText(etiquetaImporte(valor), Math.max(21, Math.min(ancho - 23, x)), abajo + 18, 48);
  }
  ctx.strokeStyle = "#a6382c";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(izquierda, corte);
  ctx.lineTo(derecha, corte);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#8d3329";
  ctx.textAlign = "left";
  ctx.fillText("Umbral " + decimal.format(umbral * 100), izquierda + 7, Math.max(arriba + 13, corte - 6));

  const puntos = puntuadas.map((fila, i) => ({
    fila,
    grupo: grupoDe(fila),
    x: izquierda + ((valores[i] - desde) / (hasta - desde)) * anchoGrafico,
    y: y(fila.puntuacion),
  }));
  for (const punto of puntos) {
    if (punto.fila !== seleccion) marca(ctx, punto, false, filtro === "todas" || filtro === punto.grupo);
  }
  const elegido = puntos.find(punto => punto.fila === seleccion);
  if (elegido) marca(ctx, elegido, true, true);
  ctx.globalAlpha = 1;
  return puntos;
}

export function conectarResultados(panel) {
  const buscar = id => panel.querySelector("#" + id);
  const canvas = buscar("mapa-lote");
  const detalle = buscar("detalle-lote");
  const tabla = buscar("filas-lote");
  const filtros = buscar("grupos-lote");
  const anterior = buscar("lote-anterior");
  const siguiente = buscar("lote-siguiente");
  let ordenadas = [], puntuadas = [], visibles = [], puntos = [];
  let pagina = 0, filtro = "todas", umbral, seleccion;
  const botones = new Map();

  function redibujar() {
    if (!panel.hidden) puntos = dibujar(canvas, puntuadas, umbral, filtro, seleccion);
  }

  for (const [grupo, [nombre]] of grupos) {
    const boton = document.createElement("button");
    boton.type = "button";
    boton.dataset.grupo = grupo;
    boton.addEventListener("click", () => {
      filtro = grupo;
      visibles = grupo === "todas" ? ordenadas : ordenadas.filter(fila => grupoDe(fila) === grupo);
      pagina = 0;
      seleccion = visibles[0];
      renderizar();
    });
    botones.set(grupo, boton);
    filtros.append(boton);
  }

  function seleccionar(fila, enfocar = false) {
    seleccion = fila;
    pagina = Math.floor(visibles.indexOf(fila) / tamanoPagina);
    renderizar();
    if (enfocar) tabla.querySelector('[aria-pressed="true"]').focus();
  }

  function renderizar() {
    for (const [grupo, boton] of botones) boton.setAttribute("aria-pressed", String(grupo === filtro));
    const inicio = pagina * tamanoPagina;
    tabla.replaceChildren();
    for (const fila of visibles.slice(inicio, inicio + tamanoPagina)) {
      const renglon = tabla.insertRow();
      renglon.dataset.seleccionado = String(fila === seleccion);
      const boton = document.createElement("button");
      boton.type = "button";
      boton.className = "batch-record";
      boton.setAttribute("aria-pressed", String(fila === seleccion));
      boton.textContent = "#" + entero.format(fila.registro);
      boton.addEventListener("click", () => seleccionar(fila, true));
      renglon.insertCell().append(boton);
      renglon.insertCell().textContent = importe(fila.monto);
      renglon.insertCell().textContent = fila.puntuacion == null ? "—" : decimal.format(fila.puntuacion * 100);
      const grupo = grupoDe(fila);
      const estado = document.createElement("span");
      estado.className = "batch-tag";
      estado.dataset.grupo = grupo;
      estado.textContent = grupos.get(grupo)[1];
      renglon.insertCell().append(estado);
    }
    buscar("limite-vista").textContent = entero.format(inicio + 1) + "–" +
      entero.format(Math.min(inicio + tamanoPagina, visibles.length)) + " de " + entero.format(visibles.length);
    anterior.disabled = inicio === 0;
    siguiente.disabled = inicio + tamanoPagina >= visibles.length;
    detalle.replaceChildren();
    if (seleccion) {
      const encabezado = document.createElement("strong");
      encabezado.textContent = ["Registro #" + entero.format(seleccion.registro), seleccion.fecha, seleccion.prioridad ?? "Dato inválido"]
        .filter(Boolean).join(" · ");
      const accion = document.createElement("span");
      accion.textContent = seleccion.error ?? seleccion.accion;
      detalle.append(encabezado, accion);
    }
    redibujar();
  }

  function cambiarPagina(direccion) {
    pagina += direccion;
    seleccion = visibles[pagina * tamanoPagina];
    renderizar();
  }

  anterior.addEventListener("click", () => cambiarPagina(-1));
  siguiente.addEventListener("click", () => cambiarPagina(1));
  canvas.addEventListener("click", evento => {
    let objetivo;
    let distancia = 9;
    for (const punto of puntos) {
      if (filtro !== "todas" && filtro !== punto.grupo) continue;
      const cercana = Math.hypot(punto.x - evento.offsetX, punto.y - evento.offsetY);
      if (cercana <= distancia) {
        objetivo = punto.fila;
        distancia = cercana;
      }
    }
    if (objetivo) seleccionar(objetivo);
  });
  new ResizeObserver(redibujar).observe(canvas);

  return (filas, modelo, corte) => {
    ordenadas = [...filas].sort((a, b) => (b.puntuacion ?? -1) - (a.puntuacion ?? -1));
    puntuadas = ordenadas.filter(fila => fila.puntuacion != null && fila.monto > 0);
    umbral = corte;
    filtro = "todas";
    visibles = ordenadas;
    pagina = 0;
    seleccion = ordenadas[0];
    const conteos = new Map([["todas", filas.length]]);
    for (const fila of filas) {
      const grupo = grupoDe(fila);
      conteos.set(grupo, (conteos.get(grupo) ?? 0) + 1);
    }
    buscar("resumen-lote").textContent = modelo;
    for (const [grupo, boton] of botones) {
      const cantidad = conteos.get(grupo) ?? 0;
      boton.hidden = cantidad === 0;
      boton.textContent = grupos.get(grupo)[0] + " " + entero.format(cantidad);
    }
    panel.hidden = false;
    renderizar();
  };
}
