// ============================================================
// REAL-TIME ANALYSIS HISTORY
// ============================================================

const HISTORY_STORAGE_KEY =
    "deepfakeAnalysisHistory";


// ============================================================
// HELPERS
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


function formatDate(timestamp) {

    if (!timestamp) {
        return "Unknown";
    }

    const date =
        new Date(timestamp);

    if (Number.isNaN(date.getTime())) {
        return "Unknown";
    }

    return date.toLocaleString();
}


function formatPercent(value) {

    return (
        Number(value || 0) * 100
    ).toFixed(2) + "%";
}


function escapeHtml(value) {

    return String(
        value ?? ""
    )
        .replace(
            /&/g,
            "&amp;"
        )
        .replace(
            /</g,
            "&lt;"
        )
        .replace(
            />/g,
            "&gt;"
        )
        .replace(
            /"/g,
            "&quot;"
        )
        .replace(
            /'/g,
            "&#039;"
        );
}


// ============================================================
// RISK CLASS
// ============================================================

function getRiskClass(risk) {

    const value =
        String(
            risk || "LOW"
        ).toUpperCase();

    if (value === "HIGH") {
        return "risk-high";
    }

    if (value === "MEDIUM") {
        return "risk-medium";
    }

    return "risk-low";
}


// ============================================================
// RENDER HISTORY
// ============================================================

function renderAnalysisHistory() {

    const table =
        document.getElementById(
            "historyTable"
        );

    const emptyState =
        document.getElementById(
            "tableEmpty"
        );


    if (!table) {
        return;
    }


    const tbody =
        table.querySelector(
            "tbody"
        );


    if (!tbody) {
        return;
    }


    const history =
        getAnalysisHistory();


    // Remove old static/demo rows
    tbody.innerHTML = "";


    // --------------------------------------------------------
    // No history
    // --------------------------------------------------------

    if (history.length === 0) {

        if (emptyState) {
            emptyState.hidden = false;
        }

        return;
    }


    if (emptyState) {
        emptyState.hidden = true;
    }


    // --------------------------------------------------------
    // Create real ML history rows
    // --------------------------------------------------------

    history.forEach(
        function (record) {

            const row =
                document.createElement(
                    "tr"
                );


            const risk =
                String(
                    record.overallRisk
                    || "LOW"
                ).toUpperCase();


            const riskClass =
                getRiskClass(
                    risk
                );


            row.innerHTML = `

                <td>
                    ${escapeHtml(
                        record.filename
                        || "Unknown audio"
                    )}
                </td>


                <td>
                    ${escapeHtml(
                        formatDate(
                            record.timestamp
                        )
                    )}
                </td>


                <td>

                    <span
                        class="${riskClass}"
                    >
                        ${escapeHtml(
                            risk
                        )}
                    </span>

                </td>


                <td>
                    ${Number(
                        record.riskScore
                        || 0
                    ).toFixed(2)}/100
                </td>


                <td>
                    ${formatPercent(
                        record.peakFakeProbability
                    )}
                </td>


                <td>
                    ${Number(
                        record.totalChunks
                        || 0
                    )}
                </td>

            `;


            tbody.appendChild(
                row
            );
        }
    );
}


// ============================================================
// SEARCH FILTER
// ============================================================

function setupHistorySearch() {

    const search =
        document.getElementById(
            "historySearch"
        );

    const table =
        document.getElementById(
            "historyTable"
        );

    const emptyState =
        document.getElementById(
            "tableEmpty"
        );


    if (!search || !table) {
        return;
    }


    search.addEventListener(
        "input",
        function () {

            const query =
                search.value
                    .trim()
                    .toLowerCase();


            const rows =
                Array.from(
                    table.querySelectorAll(
                        "tbody tr"
                    )
                );


            let visibleCount = 0;


            rows.forEach(
                function (row) {

                    const filename =
                        row
                            .firstElementChild
                            ?.textContent
                            ?.toLowerCase()
                            || "";


                    const matches =
                        filename.includes(
                            query
                        );


                    row.hidden =
                        !matches;


                    if (matches) {
                        visibleCount++;
                    }
                }
            );


            if (emptyState) {

                emptyState.hidden =
                    visibleCount !== 0;
            }

        }
    );
}


// ============================================================
// AUTO REFRESH
// ============================================================

function setupHistoryRefresh() {

    window.addEventListener(
        "storage",
        function (event) {

            if (
                event.key ===
                HISTORY_STORAGE_KEY
            ) {

                renderAnalysisHistory();
            }
        }
    );

}


// ============================================================
// INITIALIZE
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        renderAnalysisHistory();

        setupHistorySearch();

        setupHistoryRefresh();

    }
);