/**
 * Dashboard charts.
 *
 * Counters are rendered server-side; this file draws the Chart.js views from
 * the same data (embedded as JSON) plus the live /api/statistics endpoint for
 * the time series. Nothing here invents numbers.
 */
(function () {
    "use strict";

    var PALETTE = {
        safe: "#34d399",
        suspicious: "#fbbf24",
        phishing: "#fb7185",
        critical: "#f43f5e",
        accent: "#22d3ee",
        accent2: "#818cf8",
        grid: "rgba(255,255,255,0.06)",
        text: "#94a3b8"
    };

    var dataNode = document.getElementById("dashboard-data");
    if (!dataNode || typeof window.Chart === "undefined") {
        showChartFallback();
        return;
    }

    var payload = {};
    try {
        payload = JSON.parse(dataNode.textContent) || {};
    } catch (error) {
        showChartFallback();
        return;
    }

    /** Replace every empty canvas with an explanatory message. */
    function showChartFallback() {
        Array.prototype.forEach.call(document.querySelectorAll(".chart-box"), function (box) {
            if (!box.querySelector(".chart-empty")) {
                box.innerHTML = '<div class="chart-empty">Charts could not be loaded.<br>' +
                    "The Chart.js CDN is unreachable from this machine.</div>";
            }
        });
    }

    Chart.defaults.color = PALETTE.text;
    Chart.defaults.font.family = "Inter, system-ui, sans-serif";
    Chart.defaults.font.size = 11;
    Chart.defaults.plugins.legend.labels.boxWidth = 10;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;

    var stats = payload.statistics || {};
    var metrics = payload.metrics || null;
    var importances = payload.importances || {};

    function canvas(id) { return document.getElementById(id); }

    function markEmpty(id, message) {
        var element = canvas(id);
        if (element && element.parentNode) {
            element.parentNode.innerHTML = '<div class="chart-empty">' + message + "</div>";
        }
    }

    var baseScales = {
        x: { grid: { color: PALETTE.grid }, ticks: { color: PALETTE.text } },
        y: { grid: { color: PALETTE.grid }, ticks: { color: PALETTE.text }, beginAtZero: true }
    };

    /* ---------------------------------------------------- verdict split -- */
    var verdictTotal = (stats.safe || 0) + (stats.suspicious || 0) + (stats.phishing || 0);
    if (canvas("chart-verdicts")) {
        if (verdictTotal === 0) {
            markEmpty("chart-verdicts", "No scans recorded yet.");
        } else {
            new Chart(canvas("chart-verdicts"), {
                type: "doughnut",
                data: {
                    labels: ["Safe", "Suspicious", "Phishing"],
                    datasets: [{
                        data: [stats.safe || 0, stats.suspicious || 0, stats.phishing || 0],
                        backgroundColor: [PALETTE.safe, PALETTE.suspicious, PALETTE.phishing],
                        borderColor: "rgba(5,7,15,0.9)",
                        borderWidth: 3,
                        hoverOffset: 6
                    }]
                },
                options: {
                    maintainAspectRatio: false,
                    cutout: "62%",
                    plugins: { legend: { position: "bottom" } }
                }
            });
        }
    }

    /* ----------------------------------------------- risk distribution -- */
    var riskLevels = stats.risk_levels || {};
    var riskValues = [riskLevels.Low || 0, riskLevels.Medium || 0, riskLevels.High || 0, riskLevels.Critical || 0];
    if (canvas("chart-risk")) {
        if (riskValues.reduce(function (a, b) { return a + b; }, 0) === 0) {
            markEmpty("chart-risk", "No scans recorded yet.");
        } else {
            new Chart(canvas("chart-risk"), {
                type: "bar",
                data: {
                    labels: ["Low", "Medium", "High", "Critical"],
                    datasets: [{
                        label: "URLs",
                        data: riskValues,
                        backgroundColor: [PALETTE.safe, PALETTE.suspicious, "#fb923c", PALETTE.critical],
                        borderRadius: 6,
                        maxBarThickness: 52
                    }]
                },
                options: {
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: baseScales
                }
            });
        }
    }

    /* ------------------------------------------------ model comparison -- */
    if (canvas("chart-models") && metrics && metrics.models) {
        new Chart(canvas("chart-models"), {
            type: "bar",
            data: {
                labels: metrics.models.map(function (m) { return m.name; }),
                datasets: [
                    {
                        label: "Accuracy",
                        data: metrics.models.map(function (m) { return +(m.accuracy * 100).toFixed(2); }),
                        backgroundColor: PALETTE.accent, borderRadius: 5
                    },
                    {
                        label: "F1",
                        data: metrics.models.map(function (m) { return +(m.f1 * 100).toFixed(2); }),
                        backgroundColor: PALETTE.accent2, borderRadius: 5
                    },
                    {
                        label: "ROC-AUC",
                        data: metrics.models.map(function (m) { return +(m.roc_auc * 100).toFixed(2); }),
                        backgroundColor: PALETTE.safe, borderRadius: 5
                    }
                ]
            },
            options: {
                maintainAspectRatio: false,
                plugins: { legend: { position: "bottom" } },
                scales: {
                    x: baseScales.x,
                    y: {
                        grid: { color: PALETTE.grid },
                        ticks: { color: PALETTE.text, callback: function (v) { return v + "%"; } },
                        suggestedMin: 50, suggestedMax: 100
                    }
                }
            }
        });
    }

    /* ---------------------------------------------- feature importance -- */
    var importanceEntries = Object.keys(importances)
        .map(function (key) { return [key, importances[key]]; })
        .sort(function (a, b) { return b[1] - a[1]; })
        .slice(0, 12);

    if (canvas("chart-importance")) {
        if (importanceEntries.length === 0) {
            markEmpty("chart-importance", "Train the model to see feature importances.");
        } else {
            new Chart(canvas("chart-importance"), {
                type: "bar",
                data: {
                    labels: importanceEntries.map(function (e) { return e[0].replace(/_/g, " "); }),
                    datasets: [{
                        label: "Importance",
                        data: importanceEntries.map(function (e) { return +(e[1] * 100).toFixed(2); }),
                        backgroundColor: PALETTE.accent,
                        borderRadius: 5
                    }]
                },
                options: {
                    indexAxis: "y",
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: {
                            grid: { color: PALETTE.grid },
                            ticks: { color: PALETTE.text, callback: function (v) { return v + "%"; } }
                        },
                        y: { grid: { display: false }, ticks: { color: PALETTE.text } }
                    }
                }
            });
        }
    }

    /* ---------------------------------------------------- trend series -- */
    if (canvas("chart-trend")) {
        fetch("/api/statistics", { headers: { Accept: "application/json" } })
            .then(function (response) {
                if (!response.ok) { throw new Error("statistics unavailable"); }
                return response.json();
            })
            .then(function (data) {
                var trend = data.trend || [];
                var total = trend.reduce(function (sum, day) {
                    return sum + day.safe + day.suspicious + day.phishing;
                }, 0);
                if (total === 0) {
                    markEmpty("chart-trend", "No scans in the last 14 days.");
                    return;
                }
                new Chart(canvas("chart-trend"), {
                    type: "line",
                    data: {
                        labels: trend.map(function (d) { return d.date.slice(5); }),
                        datasets: [
                            series("Safe", trend.map(function (d) { return d.safe; }), PALETTE.safe),
                            series("Suspicious", trend.map(function (d) { return d.suspicious; }), PALETTE.suspicious),
                            series("Phishing", trend.map(function (d) { return d.phishing; }), PALETTE.phishing)
                        ]
                    },
                    options: {
                        maintainAspectRatio: false,
                        interaction: { mode: "index", intersect: false },
                        plugins: { legend: { position: "bottom" } },
                        scales: baseScales
                    }
                });
            })
            .catch(function () {
                markEmpty("chart-trend", "Trend data could not be loaded.");
            });
    }

    /** Build a consistent line-series configuration. */
    function series(label, data, color) {
        return {
            label: label,
            data: data,
            borderColor: color,
            backgroundColor: color + "22",
            borderWidth: 2,
            tension: 0.35,
            fill: true,
            pointRadius: 2,
            pointHoverRadius: 4
        };
    }
}());
