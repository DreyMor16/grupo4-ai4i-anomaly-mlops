"use strict";


const STATUS_LABELS = {
    stable: "Estable",
    warning: "Advertencia",
    critical: "Crítico",
    insufficient_data: "Muestra insuficiente",
    no_data: "Sin datos",
    neutral: "Sin evaluar",
};


function getElement(id) {
    return document.getElementById(id);
}


function formatNumber(value, digits = 3) {
    if (
        value === null
        || value === undefined
        || Number.isNaN(Number(value))
    ) {
        return "—";
    }

    return Number(value).toLocaleString(
        "es-CO",
        {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits,
        }
    );
}


function formatInteger(value) {
    if (
        value === null
        || value === undefined
        || Number.isNaN(Number(value))
    ) {
        return "—";
    }

    return Number(value).toLocaleString(
        "es-CO",
        {
            maximumFractionDigits: 0,
        }
    );
}


function formatPercent(value) {
    if (
        value === null
        || value === undefined
        || Number.isNaN(Number(value))
    ) {
        return "—";
    }

    return Number(value).toLocaleString(
        "es-CO",
        {
            style: "percent",
            minimumFractionDigits: 1,
            maximumFractionDigits: 2,
        }
    );
}


function formatDate(value) {
    if (!value) {
        return "Fecha no disponible";
    }

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return value;
    }

    return date.toLocaleString(
        "es-CO",
        {
            dateStyle: "medium",
            timeStyle: "short",
        }
    );
}


function setStatus(element, status) {
    const normalizedStatus = (
        STATUS_LABELS[status]
        ? status
        : "neutral"
    );

    element.classList.remove(
        "stable",
        "warning",
        "critical",
        "insufficient_data",
        "no_data",
        "neutral"
    );

    element.classList.add(
        normalizedStatus
    );

    element.textContent = (
        STATUS_LABELS[normalizedStatus]
    );
}


function createStateBadge(status) {
    const badge = document.createElement(
        "span"
    );

    badge.className = (
        `metric-state ${status}`
    );

    badge.textContent = (
        STATUS_LABELS[status]
        || STATUS_LABELS.neutral
    );

    return badge;
}


function renderSystem(system) {
    setStatus(
        getElement("system-status"),
        system.status
    );

    getElement("latency-p95").textContent = (
        formatNumber(
            system.latency_p95_ms,
            2
        )
    );

    getElement("throughput").textContent = (
        formatNumber(
            system.throughput_requests_per_minute,
            2
        )
    );

    getElement("error-rate").textContent = (
        formatPercent(
            system.error_rate
        )
    );

    getElement("availability").textContent = (
        formatPercent(
            system.availability
        )
    );

    if (
        system.status === "no_data"
    ) {
        getElement("system-summary").textContent = (
            system.message
            || "No hay solicitudes para analizar."
        );

        return;
    }

    getElement("system-summary").textContent = (
        `${formatInteger(system.request_count)} solicitudes `
        + `y ${formatInteger(system.processed_instances)} `
        + "instancias procesadas durante la ventana."
    );
}


function appendDriftRow(
    tableBody,
    variable,
    technique,
    value,
    status
) {
    const row = document.createElement(
        "tr"
    );

    const variableCell = document.createElement(
        "td"
    );
    variableCell.textContent = variable;

    const techniqueCell = document.createElement(
        "td"
    );
    techniqueCell.textContent = technique;

    const valueCell = document.createElement(
        "td"
    );
    valueCell.textContent = formatNumber(
        value,
        4
    );

    const statusCell = document.createElement(
        "td"
    );
    statusCell.appendChild(
        createStateBadge(
            status
        )
    );

    row.append(
        variableCell,
        techniqueCell,
        valueCell,
        statusCell
    );

    tableBody.appendChild(
        row
    );
}


function appendEmptyDriftRow(
    tableBody,
    message
) {
    const row = document.createElement(
        "tr"
    );

    const cell = document.createElement(
        "td"
    );

    cell.colSpan = 4;
    cell.textContent = message;

    row.appendChild(
        cell
    );

    tableBody.appendChild(
        row
    );
}


function renderData(data) {
    setStatus(
        getElement("data-status"),
        data.status
    );

    const tableBody = getElement(
        "drift-table-body"
    );

    tableBody.replaceChildren();

    if (
        data.status === "no_data"
        || data.status === "insufficient_data"
    ) {
        appendEmptyDriftRow(
            tableBody,
            data.message
            || "No hay datos suficientes."
        );

        getElement("data-summary").textContent = (
            `${formatInteger(data.production_rows)} registros disponibles. `
            + (
                data.minimum_required_rows
                ? `Se requieren ${formatInteger(data.minimum_required_rows)}.`
                : ""
            )
        );

        return;
    }

    Object.entries(
        data.numeric_drift || {}
    ).forEach(
        ([variable, metric]) => {
            appendDriftRow(
                tableBody,
                variable,
                "PSI",
                metric.psi,
                metric.status
            );
        }
    );

    Object.entries(
        data.categorical_drift || {}
    ).forEach(
        ([variable, metric]) => {
            appendDriftRow(
                tableBody,
                variable,
                "Jensen-Shannon",
                metric.js_divergence,
                metric.status
            );
        }
    );

    getElement("data-summary").textContent = (
        `${formatInteger(data.production_rows)} registros de producción `
        + `comparados con ${formatInteger(data.reference_rows)} `
        + "registros de referencia."
    );
}


function clearModelValues() {
    [
        "anomaly-rate",
        "reference-anomaly-rate",
        "score-mean",
        "reference-score-mean",
        "score-median",
        "score-p95",
        "score-minimum",
        "score-maximum",
    ].forEach(
        (id) => {
            getElement(id).textContent = "—";
        }
    );
}


function renderModel(model) {
    setStatus(
        getElement("model-status"),
        model.status
    );

    if (
        model.status === "no_data"
        || model.status === "insufficient_data"
    ) {
        clearModelValues();

        getElement(
            "false-positive-note"
        ).textContent = (
            model.message
            || "No hay resultados suficientes para evaluar el modelo."
        );

        return;
    }

    getElement("anomaly-rate").textContent = (
        formatPercent(
            model.anomaly_rate
        )
    );

    getElement(
        "reference-anomaly-rate"
    ).textContent = (
        formatPercent(
            model.reference_anomaly_rate
        )
    );

    const scores = (
        model.score_distribution
        || {}
    );

    getElement("score-mean").textContent = (
        formatNumber(
            scores.mean,
            4
        )
    );

    getElement(
        "reference-score-mean"
    ).textContent = (
        formatNumber(
            scores.reference_mean,
            4
        )
    );

    getElement("score-median").textContent = (
        formatNumber(
            scores.median,
            4
        )
    );

    getElement("score-p95").textContent = (
        formatNumber(
            scores.p95,
            4
        )
    );

    getElement("score-minimum").textContent = (
        formatNumber(
            scores.minimum,
            4
        )
    );

    getElement("score-maximum").textContent = (
        formatNumber(
            scores.maximum,
            4
        )
    );

    const falsePositive = (
        model.false_positive_monitoring
        || {}
    );

    getElement(
        "false-positive-note"
    ).textContent = (
        falsePositive.message
        || "No existe ground truth productivo disponible."
    );
}


function renderAlerts(alerts) {
    const list = getElement(
        "alerts-list"
    );

    const count = getElement(
        "alert-count"
    );

    count.textContent = formatInteger(
        alerts.length
    );

    list.replaceChildren();

    if (alerts.length === 0) {
        const empty = document.createElement(
            "p"
        );

        empty.className = "empty-state";
        empty.textContent = (
            "No hay alertas activas en la ventana analizada."
        );

        list.appendChild(
            empty
        );

        return;
    }

    alerts.forEach(
        (alert) => {
            const item = document.createElement(
                "article"
            );

            item.className = (
                `alert-item ${alert.severity}`
            );

            const marker = document.createElement(
                "span"
            );
            marker.className = "alert-marker";
            marker.setAttribute(
                "aria-hidden",
                "true"
            );

            const content = document.createElement(
                "div"
            );

            const title = document.createElement(
                "strong"
            );
            title.textContent = (
                `${alert.category.toUpperCase()} · ${alert.metric}`
            );

            const message = document.createElement(
                "p"
            );
            message.textContent = alert.message;

            const action = document.createElement(
                "small"
            );
            action.textContent = (
                `Acción: ${alert.recommended_action}`
            );

            content.append(
                title,
                message,
                action
            );

            item.append(
                marker,
                content
            );

            list.appendChild(
                item
            );
        }
    );
}


function formatDecisionPercentage(value) {
    if (
        value === null
        || value === undefined
        || !Number.isFinite(Number(value))
    ) {
        return null;
    }

    return new Intl.NumberFormat(
        "es-CR",
        {
            style: "percent",
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        }
    ).format(
        Number(value)
    );
}


function renderDecision(recommendation = {}) {
    const panel = getElement(
        "decision-panel"
    );

    const badge = getElement(
        "decision-badge"
    );

    const message = getElement(
        "decision-message"
    );

    const decision = (
        recommendation.decision
        || (
            recommendation.recommended
                ? "evaluate_retraining"
                : "continue_monitoring"
        )
    );

    const decisionConfig = {
        continue_monitoring: {
            className: "monitoring",
            badge: "Continuar monitoreo",
            fallback: (
                "No existe evidencia suficiente para "
                + "evaluar reentrenamiento."
            ),
        },

        investigate_drift: {
            className: "investigating",
            badge: "Investigar drift",
            fallback: (
                "Existe drift crítico, pero todavía "
                + "no hay ground truth para confirmar "
                + "una degradación del modelo."
            ),
        },

        evaluate_retraining: {
            className: "retraining",
            badge: "Evaluar reentrenamiento",
            fallback: (
                "Existe drift crítico y el desempeño "
                + "cayó 10% o más con respecto "
                + "a la referencia."
            ),
        },
    };

    const selected = (
        decisionConfig[decision]
        || decisionConfig.continue_monitoring
    );

    panel.classList.remove(
        "monitoring",
        "investigating",
        "retraining"
    );

    panel.classList.add(
        selected.className
    );

    badge.textContent = (
        selected.badge
    );

    const messageParts = [];

    if (
        decision === "evaluate_retraining"
    ) {
        messageParts.push(
            "Existe drift crítico y el desempeño "
            + "cayó 10% o más con respecto "
            + "a la referencia."
        );
    } else if (
        Array.isArray(recommendation.reasons)
        && recommendation.reasons.length > 0
    ) {
        messageParts.push(
            recommendation.reasons.join(" ")
        );
    } else {
        messageParts.push(
            selected.fallback
        );
    }

    const referencePerformance = (
        formatDecisionPercentage(
            recommendation.reference_performance
        )
    );

    const currentPerformance = (
        formatDecisionPercentage(
            recommendation.current_performance
        )
    );

    if (
        decision !== "evaluate_retraining"
        && referencePerformance
    ) {
        messageParts.push(
            `Recall de referencia: ${referencePerformance}.`
        );
    }

    if (
        decision !== "evaluate_retraining"
        && currentPerformance
    ) {
        messageParts.push(
            `Recall actual: ${currentPerformance}.`
        );
    } else if (
        decision === "investigate_drift"
    ) {
        messageParts.push(
            "Recall actual: no disponible "
            + "por falta de ground truth."
        );
    }

    if (
        decision === "evaluate_retraining"
    ) {
        messageParts.push(
            "La decisión requiere validación humana, "
            + "comparación de candidatos en MLflow "
            + "y promoción controlada."
        );
    }

    if (
        decision === "continue_monitoring"
    ) {
        messageParts.push(
            "El modelo continúa bajo monitoreo."
        );
    }

    if (
        decision === "investigate_drift"
    ) {
        messageParts.push(
            "Se debe investigar la causa del drift "
            + "antes de tomar una decisión."
        );
    }

    message.textContent = (
        messageParts.join(" ")
    );
}


function renderReport(report) {
    setStatus(
        getElement("overall-status"),
        report.overall_status
    );

    getElement("generated-at").textContent = (
        `Actualizado: ${formatDate(report.generated_at_utc)}`
    );

    getElement("model-name").textContent = (
        report.model?.name
        || "—"
    );

    getElement("model-version").textContent = (
        report.model?.version
        || "—"
    );

    getElement("window-hours").textContent = (
        `${formatNumber(report.window_hours, 0)} h`
    );

    getElement("production-rows").textContent = (
        formatInteger(
            report.model_monitoring?.production_rows
            ?? report.data_monitoring?.production_rows
        )
    );

    renderSystem(
        report.system_monitoring
    );

    renderData(
        report.data_monitoring
    );

    renderModel(
        report.model_monitoring
    );

    renderAlerts(
        report.alerts || []
    );

    renderDecision(
        report.retraining_recommendation
        || {
            recommended: false,
        }
    );
}


async function loadReport() {
    const refreshButton = getElement(
        "refresh-button"
    );

    const loadingPanel = getElement(
        "loading-panel"
    );

    const errorPanel = getElement(
        "error-panel"
    );

    const dashboard = getElement(
        "dashboard"
    );

    refreshButton.disabled = true;
    refreshButton.textContent = "Actualizando…";

    loadingPanel.hidden = false;
    errorPanel.hidden = true;

    try {
        const response = await fetch(
            `/monitoring/report?t=${Date.now()}`,
            {
                cache: "no-store",
                headers: {
                    Accept: "application/json",
                },
            }
        );

        if (!response.ok) {
            let detail = (
                "Ejecuta primero "
                + "python src/monitoring/run_monitoring.py"
            );

            try {
                const errorBody = await response.json();

                if (errorBody.detail) {
                    detail = errorBody.detail;
                }
            } catch {
                // Se conserva el mensaje predeterminado.
            }

            throw new Error(
                detail
            );
        }

        const report = await response.json();

        renderReport(
            report
        );

        dashboard.hidden = false;
        loadingPanel.hidden = true;

    } catch (error) {
        dashboard.hidden = true;
        loadingPanel.hidden = true;
        errorPanel.hidden = false;

        getElement(
            "error-message"
        ).textContent = (
            error.message
        );

        setStatus(
            getElement("overall-status"),
            "no_data"
        );

        getElement("generated-at").textContent = (
            "Reporte no disponible"
        );

    } finally {
        refreshButton.disabled = false;
        refreshButton.textContent = "Actualizar";
    }
}


getElement(
    "refresh-button"
).addEventListener(
    "click",
    loadReport
);


loadReport();