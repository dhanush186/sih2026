// ============================================================
// REAL ML DASHBOARD
// ============================================================

const HISTORY_STORAGE_KEY =
    "deepfakeAnalysisHistory";


// ============================================================
// STORAGE
// ============================================================

function getAnalysisHistory() {

    try {

        const stored =
            localStorage.getItem(
                HISTORY_STORAGE_KEY
            );

        if (!stored) {
            return [];
        }

        const parsed =
            JSON.parse(stored);

        return Array.isArray(parsed)
            ? parsed
            : [];

    } catch (error) {

        console.error(
            "Could not read analysis history:",
            error
        );

        return [];
    }
}


// ============================================================
// HELPERS
// ============================================================

function setText(
    element,
    value
) {

    if (!element) {
        return;
    }

    element.textContent =
        String(value);
}


function formatPercentage(
    value
) {

    return (
        Number(value || 0) * 100
    ).toFixed(2) + "%";
}


// ============================================================
// CURRENT ANALYSIS
// ============================================================

function renderCurrentAnalysis(
    latest
) {

    const container =
        document.getElementById(
            "currentAnalysisMeters"
        );

    if (!container) {
        return;
    }


    if (!latest) {

        container.innerHTML =
            meterRowMarkup(
                "AI detection",
                0,
                100,
                "%"
            )
            +
            meterRowMarkup(
                "Risk score",
                0,
                100,
                "/100"
            );

        animateMeters(
            container
        );

        return;
    }


    const aiDetection =
        Number(
            latest.peakFakeProbability
            || latest.rollingFakeProbability
            || 0
        ) * 100;


    const riskScore =
        Number(
            latest.riskScore
            || 0
        );


    container.innerHTML =
        meterRowMarkup(
            "AI detection",
            aiDetection,
            100,
            "%"
        )
        +
        meterRowMarkup(
            "Risk score",
            riskScore,
            100,
            "/100"
        );


    animateMeters(
        container
    );
}


// ============================================================
// TOP STATISTICS
// ============================================================

function renderStatistics(
    history
) {

    const statisticTiles =
        document.querySelectorAll(
            ".statistics .stat-tile"
        );


    if (statisticTiles.length < 4) {

        console.warn(
            "Expected four dashboard statistic tiles."
        );

        return;
    }


    const totalAnalyses =
        history.length;


    const suspiciousAnalyses =
        history.filter(
            function (record) {

                const risk =
                    String(
                        record.overallRisk
                        || ""
                    ).toUpperCase();

                return (
                    risk === "MEDIUM"
                    || risk === "HIGH"
                );
            }
        ).length;


    const criticalAlerts =
        history.filter(
            function (record) {

                return Boolean(
                    record.sessionAlert
                );
            }
        ).length;


    const peakFakeProbability =
        history.length > 0
            ? Math.max(
                ...history.map(
                    function (record) {

                        return (
                            Number(
                                record.peakFakeProbability
                                || record.rollingFakeProbability
                                || 0
                            ) * 100
                        );
                    }
                )
            )
            : 0;


    // --------------------------------------------------------
    // Tile 1: Total analyses
    // --------------------------------------------------------

    const totalValue =
        statisticTiles[0].querySelector(
            ".stat-value"
        );

    setText(
        totalValue,
        totalAnalyses
    );


    const totalTitle =
        statisticTiles[0].querySelector(
            "h3"
        );

    setText(
        totalTitle,
        "Total analyses"
    );


    // --------------------------------------------------------
    // Tile 2: Suspicious analyses
    // --------------------------------------------------------

    const suspiciousValue =
        statisticTiles[1].querySelector(
            ".stat-value"
        );

    setText(
        suspiciousValue,
        suspiciousAnalyses
    );


    const suspiciousTitle =
        statisticTiles[1].querySelector(
            "h3"
        );

    setText(
        suspiciousTitle,
        "Suspicious analyses"
    );


    // --------------------------------------------------------
    // Tile 3: Session alerts
    // --------------------------------------------------------

    const alertValue =
        statisticTiles[2].querySelector(
            ".stat-value"
        );

    setText(
        alertValue,
        criticalAlerts
    );


    const alertTitle =
        statisticTiles[2].querySelector(
            "h3"
        );

    setText(
        alertTitle,
        "Session alerts"
    );


    // --------------------------------------------------------
    // Tile 4: Peak fake probability
    // --------------------------------------------------------

    const probabilityValue =
        statisticTiles[3].querySelector(
            ".stat-value"
        );

    setText(
        probabilityValue,
        peakFakeProbability.toFixed(2) + "%"
    );


    const probabilityTitle =
        statisticTiles[3].querySelector(
            "h3"
        );

    setText(
        probabilityTitle,
        "Peak fake probability"
    );
}


// ============================================================
// RISK RESULT PANEL
// ============================================================

function renderRiskResult(
    latest
) {

    const riskSection =
        document.querySelector(
            ".risk-result"
        );

    if (!riskSection) {
        return;
    }


    const riskHeading =
        riskSection.querySelector(
            "h2"
        );

    const riskBadge =
        riskSection.querySelector(
            ".badge"
        );

    const riskDescription =
        riskSection.querySelector(
            "p"
        );


    if (!latest) {

        if (riskHeading) {
            riskHeading.textContent =
                "Risk level";
        }

        if (riskBadge) {

            riskBadge.textContent =
                "NO DATA";

            riskBadge.className =
                "badge";
        }

        if (riskDescription) {

            riskDescription.textContent =
                "No voice analysis has been performed yet.";
        }

        return;
    }


    const risk =
        String(
            latest.overallRisk
            || "LOW"
        ).toUpperCase();


    const riskScore =
        Number(
            latest.riskScore
            || 0
        );


    // --------------------------------------------------------
    // Section risk class
    // --------------------------------------------------------

    riskSection.classList.remove(
        "risk-high",
        "risk-medium",
        "risk-low"
    );


    if (risk === "HIGH") {

        riskSection.classList.add(
            "risk-high"
        );

    } else if (risk === "MEDIUM") {

        riskSection.classList.add(
            "risk-medium"
        );

    } else {

        riskSection.classList.add(
            "risk-low"
        );
    }


    // --------------------------------------------------------
    // Heading
    // --------------------------------------------------------

    if (riskHeading) {

        riskHeading.textContent =
            "Risk level";
    }


    // --------------------------------------------------------
    // Badge
    // --------------------------------------------------------

    if (riskBadge) {

        riskBadge.classList.remove(
            "badge-high",
            "badge-medium",
            "badge-low"
        );


        if (risk === "HIGH") {

            riskBadge.classList.add(
                "badge-high"
            );

        } else if (risk === "MEDIUM") {

            riskBadge.classList.add(
                "badge-medium"
            );

        } else {

            riskBadge.classList.add(
                "badge-low"
            );
        }


        riskBadge.textContent =
            `${risk} RISK`;
    }


    // --------------------------------------------------------
    // Description
    // --------------------------------------------------------

    if (riskDescription) {

        if (risk === "HIGH") {

            riskDescription.textContent =
                `The analyzed voice shows highly suspicious `
                + `characteristics. Risk score: `
                + `${riskScore.toFixed(2)}/100. `
                + `Independent verification is recommended.`;

        } else if (risk === "MEDIUM") {

            riskDescription.textContent =
                `The analyzed voice shows moderately `
                + `suspicious characteristics. Risk score: `
                + `${riskScore.toFixed(2)}/100. `
                + `Additional verification is recommended.`;

        } else {

            riskDescription.textContent =
                `The latest analysis shows low synthetic-voice `
                + `risk. Risk score: `
                + `${riskScore.toFixed(2)}/100.`;
        }
    }
}


// ============================================================
// INITIALIZE DASHBOARD
// ============================================================

function renderDashboard() {

    const history =
        getAnalysisHistory();


    // The newest analysis is stored first.
    const latest =
        history.length > 0
            ? history[0]
            : null;


    renderStatistics(
        history
    );


    renderCurrentAnalysis(
        latest
    );


    renderRiskResult(
        latest
    );
}


// ============================================================
// AUTO-REFRESH WHEN ANOTHER TAB CHANGES HISTORY
// ============================================================

window.addEventListener(
    "storage",
    function (event) {

        if (
            event.key ===
            HISTORY_STORAGE_KEY
        ) {

            renderDashboard();
        }
    }
);


// ============================================================
// PAGE LOAD
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        renderDashboard();

    }
);