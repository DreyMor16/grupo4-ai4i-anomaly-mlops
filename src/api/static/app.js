"use strict";

const columnasRequeridas = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
];

const muestraNormal = {
    Type: "L",
    "Air temperature": 298.1,
    "Process temperature": 308.6,
    "Rotational speed": 1551,
    Torque: 42.8,
    "Tool wear": 0,
};

const muestraAnomala = {
    Type: "L",
    "Air temperature": 298.9,
    "Process temperature": 309.1,
    "Rotational speed": 2861,
    Torque: 4.6,
    "Tool wear": 143,
};

let registrosBatch = [];
let anomaliasBatch = [];
let paginaBatch = 1;

const REGISTROS_POR_PAGINA = 10;
const MAXIMO_REGISTROS = 1000;

const formularioIndividual = document.getElementById(
    "predict-form"
);

const formularioBatch = document.getElementById(
    "batch-form"
);

const campoTipo = document.getElementById("type");

const campoTemperaturaAire = document.getElementById(
    "air-temperature"
);

const campoTemperaturaProceso = document.getElementById(
    "process-temperature"
);

const campoVelocidad = document.getElementById(
    "rotational-speed"
);

const campoTorque = document.getElementById("torque");

const campoDesgaste = document.getElementById(
    "tool-wear"
);

const botonNormal = document.getElementById(
    "load-normal"
);

const botonAnomalia = document.getElementById(
    "load-anomaly"
);

const botonPredecir = document.getElementById(
    "predict-button"
);

const resultadoIndividual = document.getElementById(
    "single-result"
);

const campoArchivo = document.getElementById(
    "batch-file"
);

const zonaArchivo = document.getElementById(
    "drop-zone"
);

const seleccionArchivo = document.getElementById(
    "file-selection"
);

const nombreArchivo = document.getElementById(
    "file-name"
);

const informacionArchivo = document.getElementById(
    "file-meta"
);

const botonQuitarArchivo = document.getElementById(
    "clear-batch"
);

const botonDescargarPlantilla = document.getElementById(
    "download-template"
);

const botonBatch = document.getElementById(
    "batch-button"
);

const batchVacio = document.getElementById(
    "batch-empty"
);

const salidaBatch = document.getElementById(
    "batch-output"
);

const resumenBatch = document.getElementById(
    "batch-summary"
);

const tablaBatch = document.getElementById(
    "batch-table-body"
);

const paginacionBatch = document.getElementById(
    "batch-pagination"
);

const informacionPaginaBatch = document.getElementById(
    "batch-page-info"
);

const botonPaginaAnterior = document.getElementById(
    "batch-previous"
);

const botonPaginaSiguiente = document.getElementById(
    "batch-next"
);

const notificacion = document.getElementById(
    "toast"
);

function escaparHtml(valor) {
    return String(valor)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function formatearScore(valor) {
    const numero = Number(valor);

    return Number.isFinite(numero)
        ? numero.toFixed(6)
        : "No disponible";
}

function mostrarMensaje(mensaje, tipo = "error") {
    notificacion.textContent = mensaje;
    notificacion.className =
        `toast ${tipo} visible`;

    window.clearTimeout(
        mostrarMensaje.temporizador
    );

    mostrarMensaje.temporizador =
        window.setTimeout(() => {
            notificacion.classList.remove(
                "visible"
            );
        }, 4000);
}

function establecerCarga(boton, cargando) {
    boton.classList.toggle(
        "is-loading",
        cargando
    );

    boton.disabled = cargando;

    boton.setAttribute(
        "aria-busy",
        String(cargando)
    );
}

async function solicitarJson(ruta, opciones = {}) {
    const respuesta = await fetch(
        ruta,
        {
            ...opciones,
            headers: {
                "Content-Type": "application/json",
                ...(opciones.headers || {}),
            },
        }
    );

    let contenido;

    try {
        contenido = await respuesta.json();
    } catch (error) {
        contenido = null;
    }

    if (!respuesta.ok) {
        let mensaje =
            `La solicitud falló con código ${respuesta.status}.`;

        if (contenido?.detail) {
            if (typeof contenido.detail === "string") {
                mensaje = contenido.detail;
            } else {
                mensaje = JSON.stringify(
                    contenido.detail
                );
            }
        }

        throw new Error(mensaje);
    }

    return contenido;
}

async function verificarEstado() {
    const indicador = document.getElementById(
        "service-status"
    );

    const textoEstado = document.getElementById(
        "status-text"
    );

    indicador.classList.remove(
        "online",
        "offline"
    );

    indicador.classList.add("checking");
    textoEstado.textContent = "Verificando API";

    try {
        const estado = await solicitarJson(
            "/health",
            {
                method: "GET",
                headers: {},
            }
        );

        indicador.classList.remove(
            "checking",
            "offline"
        );

        indicador.classList.add("online");

        textoEstado.textContent =
            "API disponible";

        document.getElementById(
            "hero-model-name"
        ).textContent = estado.model_name;

        document.getElementById(
            "hero-model-version"
        ).textContent = estado.model_version;
    } catch (error) {
        indicador.classList.remove(
            "checking",
            "online"
        );

        indicador.classList.add("offline");

        textoEstado.textContent =
            "API no disponible";

        console.error(error);
    }
}

document
    .querySelectorAll(".tab-button")
    .forEach((boton) => {
        boton.addEventListener(
            "click",
            () => {
                const panelSeleccionado =
                    boton.dataset.tab;

                document
                    .querySelectorAll(
                        ".tab-button"
                    )
                    .forEach((otroBoton) => {
                        const activo =
                            otroBoton === boton;

                        otroBoton.classList.toggle(
                            "active",
                            activo
                        );

                        otroBoton.setAttribute(
                            "aria-selected",
                            String(activo)
                        );
                    });

                document
                    .querySelectorAll(".tab-panel")
                    .forEach((panel) => {
                        panel.hidden = (
                            panel.id
                            !== panelSeleccionado
                        );
                    });
            }
        );
    });

function cargarMuestra(muestra) {
    campoTipo.value = muestra.Type;

    campoTemperaturaAire.value =
        muestra["Air temperature"];

    campoTemperaturaProceso.value =
        muestra["Process temperature"];

    campoVelocidad.value =
        muestra["Rotational speed"];

    campoTorque.value =
        muestra.Torque;

    campoDesgaste.value =
        muestra["Tool wear"];
}

function obtenerEntradaIndividual() {
    return {
        Type: campoTipo.value,

        "Air temperature":
            Number(campoTemperaturaAire.value),

        "Process temperature":
            Number(campoTemperaturaProceso.value),

        "Rotational speed":
            Number(campoVelocidad.value),

        Torque:
            Number(campoTorque.value),

        "Tool wear":
            Number(campoDesgaste.value),
    };
}

function mostrarPrediccion(resultado) {
    const esAnomalia = Boolean(
        resultado.anomaly
    );

    resultadoIndividual.className =
        `card result-card ${
            esAnomalia
                ? "anomaly"
                : "normal"
        }`;

    resultadoIndividual.innerHTML = `
        <div class="result-content">
            <div class="result-icon">
                ${esAnomalia ? "!" : "✓"}
            </div>

            <p class="card-kicker">
                Resultado
            </p>

            <h2 class="result-state">
                ${
                    esAnomalia
                        ? "Comportamiento anómalo"
                        : "Funcionamiento normal"
                }
            </h2>

            <p class="result-description">
                ${
                    esAnomalia
                        ? "Las condiciones se alejaron del comportamiento aprendido por el modelo."
                        : "Las condiciones se encuentran dentro del comportamiento esperado."
                }
            </p>

            <div class="result-score">
                <span class="score-label">
                    Anomaly score
                </span>

                <strong class="score-value">
                    ${formatearScore(
                        resultado.anomaly_score
                    )}
                </strong>
            </div>

            <div class="result-meta">
                <div class="meta-row">
                    <span>Predicción</span>

                    <strong>
                        ${
                            esAnomalia
                                ? "Anomalía (1)"
                                : "Normal (0)"
                        }
                    </strong>
                </div>

                <div class="meta-row">
                    <span>Modelo</span>

                    <strong>
                        ${escaparHtml(
                            resultado.model_name
                        )}
                    </strong>
                </div>

                <div class="meta-row">
                    <span>Versión</span>

                    <strong>
                        ${escaparHtml(
                            resultado.model_version
                        )}
                    </strong>
                </div>
            </div>

            <p class="recommendation">
                ${
                    esAnomalia
                        ? "Se recomienda revisar la máquina. Esta alerta no reemplaza una inspección técnica."
                        : "Continúa con el monitoreo y mantenimiento preventivo habitual."
                }
            </p>
        </div>
    `;
}

botonNormal.addEventListener(
    "click",
    () => {
        cargarMuestra(muestraNormal);
    }
);

botonAnomalia.addEventListener(
    "click",
    () => {
        cargarMuestra(muestraAnomala);
    }
);

formularioIndividual.addEventListener(
    "submit",
    async (evento) => {
        evento.preventDefault();

        if (!formularioIndividual.reportValidity()) {
            return;
        }

        establecerCarga(
            botonPredecir,
            true
        );

        try {
            const resultado = await solicitarJson(
                "/predict",
                {
                    method: "POST",
                    body: JSON.stringify(
                        obtenerEntradaIndividual()
                    ),
                }
            );

            mostrarPrediccion(resultado);

            mostrarMensaje(
                "Predicción realizada correctamente.",
                "success"
            );
        } catch (error) {
            mostrarMensaje(error.message);
        } finally {
            establecerCarga(
                botonPredecir,
                false
            );
        }
    }
);

function detectarSeparador(texto) {
    const primeraLinea = (
        texto.split(/\r?\n/)[0] || ""
    );

    const cantidadComas = (
        primeraLinea.match(/,/g) || []
    ).length;

    const cantidadPuntoComas = (
        primeraLinea.match(/;/g) || []
    ).length;

    return cantidadPuntoComas > cantidadComas
        ? ";"
        : ",";
}

function dividirCSV(texto, separador) {
    const filas = [];

    let fila = [];
    let campo = "";
    let dentroComillas = false;

    for (
        let indice = 0;
        indice < texto.length;
        indice += 1
    ) {
        const caracter = texto[indice];
        const siguiente = texto[indice + 1];

        if (caracter === '"') {
            if (
                dentroComillas
                && siguiente === '"'
            ) {
                campo += '"';
                indice += 1;
            } else {
                dentroComillas =
                    !dentroComillas;
            }
        } else if (
            caracter === separador
            && !dentroComillas
        ) {
            fila.push(campo.trim());
            campo = "";
        } else if (
            (caracter === "\n")
            && !dentroComillas
        ) {
            fila.push(
                campo.replace(/\r$/, "").trim()
            );

            if (
                fila.some(
                    (valor) => valor !== ""
                )
            ) {
                filas.push(fila);
            }

            fila = [];
            campo = "";
        } else {
            campo += caracter;
        }
    }

    fila.push(
        campo.replace(/\r$/, "").trim()
    );

    if (
        fila.some(
            (valor) => valor !== ""
        )
    ) {
        filas.push(fila);
    }

    return filas;
}

function convertirNumero(valor, fila, columna) {
    const numero = Number(
        String(valor)
            .trim()
            .replace(",", ".")
    );

    if (!Number.isFinite(numero)) {
        throw new Error(
            `La fila ${fila} contiene un valor `
            + `inválido en "${columna}".`
        );
    }

    return numero;
}

function convertirCSV(texto) {
    const contenido = texto
        .replace(/^\uFEFF/, "")
        .trim();

    if (!contenido) {
        throw new Error(
            "El archivo CSV está vacío."
        );
    }

    const separador = detectarSeparador(
        contenido
    );

    const filas = dividirCSV(
        contenido,
        separador
    );

    if (filas.length < 2) {
        throw new Error(
            "El archivo no contiene registros."
        );
    }

    const encabezados = filas
        .shift()
        .map((encabezado) => {
            return encabezado.trim();
        });

    const columnasFaltantes =
        columnasRequeridas.filter(
            (columna) => {
                return !encabezados.includes(
                    columna
                );
            }
        );

    if (columnasFaltantes.length > 0) {
        throw new Error(
            "Faltan las columnas: "
            + columnasFaltantes.join(", ")
        );
    }

    if (filas.length > MAXIMO_REGISTROS) {
        throw new Error(
            `El archivo contiene ${filas.length} registros. `
            + `El máximo permitido es ${MAXIMO_REGISTROS}.`
        );
    }

    return filas.map(
        (valores, indice) => {
            const filaOriginal = indice + 2;
            const registro = {};

            encabezados.forEach(
                (encabezado, posicion) => {
                    registro[encabezado] =
                        valores[posicion] ?? "";
                }
            );

            const tipo = String(
                registro.Type
            ).trim().toUpperCase();

            if (!["L", "M", "H"].includes(tipo)) {
                throw new Error(
                    `La fila ${filaOriginal} contiene `
                    + `un Type inválido: "${tipo}".`
                );
            }

            return {
                Type: tipo,

                "Air temperature":
                    convertirNumero(
                        registro["Air temperature"],
                        filaOriginal,
                        "Air temperature"
                    ),

                "Process temperature":
                    convertirNumero(
                        registro["Process temperature"],
                        filaOriginal,
                        "Process temperature"
                    ),

                "Rotational speed":
                    convertirNumero(
                        registro["Rotational speed"],
                        filaOriginal,
                        "Rotational speed"
                    ),

                Torque:
                    convertirNumero(
                        registro.Torque,
                        filaOriginal,
                        "Torque"
                    ),

                "Tool wear":
                    convertirNumero(
                        registro["Tool wear"],
                        filaOriginal,
                        "Tool wear"
                    ),
            };
        }
    );
}

function limpiarArchivo() {
    registrosBatch = [];
    anomaliasBatch = [];
    paginaBatch = 1;

    campoArchivo.value = "";

    seleccionArchivo.hidden = true;
    salidaBatch.hidden = true;
    batchVacio.hidden = false;
    paginacionBatch.hidden = true;

    botonBatch.disabled = true;

    nombreArchivo.textContent = "";
    informacionArchivo.textContent = "";
    tablaBatch.innerHTML = "";
}

async function cargarArchivo(archivo) {
    if (!archivo) {
        return;
    }

    if (
        !archivo.name
            .toLowerCase()
            .endsWith(".csv")
    ) {
        mostrarMensaje(
            "Debes seleccionar un archivo CSV."
        );
        return;
    }

    try {
        const texto = await archivo.text();

        registrosBatch = convertirCSV(texto);
        anomaliasBatch = [];
        paginaBatch = 1;

        nombreArchivo.textContent =
            archivo.name;

        informacionArchivo.textContent =
            `${registrosBatch.length} registros `
            + "listos para procesar";

        seleccionArchivo.hidden = false;
        salidaBatch.hidden = true;
        batchVacio.hidden = false;
        paginacionBatch.hidden = true;

        botonBatch.disabled = false;

        mostrarMensaje(
            "Archivo CSV cargado correctamente.",
            "success"
        );
    } catch (error) {
        limpiarArchivo();
        mostrarMensaje(error.message);
    }
}

campoArchivo.addEventListener(
    "change",
    () => {
        cargarArchivo(
            campoArchivo.files[0]
        );
    }
);

botonQuitarArchivo.addEventListener(
    "click",
    limpiarArchivo
);

[
    "dragenter",
    "dragover",
].forEach((nombreEvento) => {
    zonaArchivo.addEventListener(
        nombreEvento,
        (evento) => {
            evento.preventDefault();

            zonaArchivo.classList.add(
                "dragging"
            );
        }
    );
});

[
    "dragleave",
    "drop",
].forEach((nombreEvento) => {
    zonaArchivo.addEventListener(
        nombreEvento,
        (evento) => {
            evento.preventDefault();

            zonaArchivo.classList.remove(
                "dragging"
            );
        }
    );
});

zonaArchivo.addEventListener(
    "drop",
    (evento) => {
        const archivo =
            evento.dataTransfer.files[0];

        cargarArchivo(archivo);
    }
);

botonDescargarPlantilla.addEventListener(
    "click",
    () => {
        const contenido = [
            columnasRequeridas.join(","),
            [
                "L",
                "298.1",
                "308.6",
                "1551",
                "42.8",
                "0",
            ].join(","),
            [
                "L",
                "298.9",
                "309.1",
                "2861",
                "4.6",
                "143",
            ].join(","),
        ].join("\n");

        const archivo = new Blob(
            [contenido],
            {
                type: "text/csv;charset=utf-8",
            }
        );

        const url = URL.createObjectURL(
            archivo
        );

        const enlace = document.createElement(
            "a"
        );

        enlace.href = url;
        enlace.download =
            "plantilla_ai4i.csv";

        enlace.click();

        URL.revokeObjectURL(url);
    }
);

function renderizarPaginaBatch() {
    if (anomaliasBatch.length === 0) {
        tablaBatch.innerHTML = `
            <tr>
                <td
                    colspan="4"
                    class="empty-table"
                >
                    No se detectaron anomalías en este archivo.
                </td>
            </tr>
        `;

        paginacionBatch.hidden = true;
        return;
    }

    const totalPaginas = Math.ceil(
        anomaliasBatch.length
        / REGISTROS_POR_PAGINA
    );

    paginaBatch = Math.min(
        Math.max(paginaBatch, 1),
        totalPaginas
    );

    const indiceInicial = (
        paginaBatch - 1
    ) * REGISTROS_POR_PAGINA;

    const indiceFinal =
        indiceInicial
        + REGISTROS_POR_PAGINA;

    const anomaliasPagina =
        anomaliasBatch.slice(
            indiceInicial,
            indiceFinal
        );

    tablaBatch.innerHTML =
        anomaliasPagina
            .map((prediccion) => {
                return `
                    <tr>
                        <td>
                            ${prediccion.filaOriginal}
                        </td>

                        <td>
                            <span class="badge badge-anomaly">
                                Anomalía
                            </span>
                        </td>

                        <td>
                            ${prediccion.prediction}
                        </td>

                        <td>
                            ${formatearScore(
                                prediccion.anomaly_score
                            )}
                        </td>
                    </tr>
                `;
            })
            .join("");

    informacionPaginaBatch.textContent =
        `Página ${paginaBatch} de ${totalPaginas} · `
        + `${anomaliasBatch.length} anomalías`;

    botonPaginaAnterior.disabled = (
        paginaBatch === 1
    );

    botonPaginaSiguiente.disabled = (
        paginaBatch === totalPaginas
    );

    paginacionBatch.hidden = false;
}

function mostrarResultadoBatch(resultado) {
    batchVacio.hidden = true;
    salidaBatch.hidden = false;

    resumenBatch.innerHTML = `
        <div class="summary-item">
            <span>Registros</span>

            <strong>
                ${resultado.total_instances}
            </strong>
        </div>

        <div class="summary-item">
            <span>Anomalías</span>

            <strong>
                ${resultado.total_anomalies}
            </strong>
        </div>

        <div class="summary-item">
            <span>Normales</span>

            <strong>
                ${
                    resultado.total_instances
                    - resultado.total_anomalies
                }
            </strong>
        </div>
    `;

    anomaliasBatch = resultado.predictions
        .map((prediccion, indice) => {
            return {
                ...prediccion,
                filaOriginal: indice + 1,
            };
        })
        .filter((prediccion) => {
            return Boolean(
                prediccion.anomaly
            );
        });

    paginaBatch = 1;

    renderizarPaginaBatch();
}

botonPaginaAnterior.addEventListener(
    "click",
    () => {
        if (paginaBatch > 1) {
            paginaBatch -= 1;
            renderizarPaginaBatch();
        }
    }
);

botonPaginaSiguiente.addEventListener(
    "click",
    () => {
        const totalPaginas = Math.ceil(
            anomaliasBatch.length
            / REGISTROS_POR_PAGINA
        );

        if (paginaBatch < totalPaginas) {
            paginaBatch += 1;
            renderizarPaginaBatch();
        }
    }
);

formularioBatch.addEventListener(
    "submit",
    async (evento) => {
        evento.preventDefault();

        if (registrosBatch.length === 0) {
            mostrarMensaje(
                "Primero debes cargar un archivo CSV válido."
            );
            return;
        }

        establecerCarga(
            botonBatch,
            true
        );

        try {
            const resultado = await solicitarJson(
                "/predict/batch",
                {
                    method: "POST",
                    body: JSON.stringify({
                        instances: registrosBatch,
                    }),
                }
            );

            mostrarResultadoBatch(resultado);

            mostrarMensaje(
                "El archivo se procesó correctamente.",
                "success"
            );
        } catch (error) {
            mostrarMensaje(error.message);
        } finally {
            establecerCarga(
                botonBatch,
                false
            );
        }
    }
);

cargarMuestra(muestraNormal);
verificarEstado();