// ============================================================
// SECURITY ALERTS
// ============================================================

const ALERT_STORAGE_KEY = "deepfakeAlerts";


// ============================================================
// STORAGE HELPERS
// ============================================================

function getStoredAlerts() {

    try {

        const stored =
            localStorage.getItem(
                ALERT_STORAGE_KEY
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
            "Failed to read alerts:",
            error
        );

        return [];
    }
}


function saveStoredAlerts(alerts) {

    try {

        localStorage.setItem(
            ALERT_STORAGE_KEY,
            JSON.stringify(alerts)
        );

    } catch (error) {

        console.error(
            "Failed to save alerts:",
            error
        );
    }
}


// ============================================================
// DISPLAY HELPERS
// ============================================================

function formatPercent(value) {

    return (
        Number(value || 0) * 100
    ).toFixed(2) + "%";
}


function formatDate(timestamp) {

    if (!timestamp) {
        return "Unknown time";
    }

    const date =
        new Date(timestamp);

    if (Number.isNaN(date.getTime())) {
        return "Unknown time";
    }

    return date.toLocaleString();
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
// CREATE ALERT FROM REAL-TIME ML RESULT
// ============================================================

function createDeepfakeAlert(result) {

    if (!result) {
        return null;
    }

    if (!result.session_alert) {
        return null;
    }


    const alerts =
        getStoredAlerts();


    const chunks =
        Array.isArray(result.chunks)
            ? result.chunks
            : [];


    let peakFakeProbability =
        Number(
            result.rolling_fake_probability || 0
        );


    if (chunks.length > 0) {

        peakFakeProbability =
            Math.max(
                ...chunks.map(
                    function (chunk) {

                        return Number(
                            chunk.fake_probability
                            || 0
                        );
                    }
                )
            );
    }


    const alert = {

        id:
            Date.now().toString(),

        title:
            "High-Risk Voice Alert",

        severity:
            String(
                result.overall_risk
                || "HIGH"
            ).toUpperCase(),

        filename:
            result.filename
            || "Unknown audio",

        riskScore:
            Number(
                result.overall_risk_score
                || 0
            ),

        peakFakeProbability:
            peakFakeProbability,

        rollingFakeProbability:
            Number(
                result.rolling_fake_probability
                || 0
            ),

        totalChunks:
            Number(
                result.total_chunks
                || chunks.length
                || 0
            ),

        message:
            "Suspicious synthetic-voice evidence detected.",

        verificationRequired:
            Boolean(
                result.requires_verification
            ),

        timestamp:
            new Date().toISOString()
    };


    // --------------------------------------------------------
    // Add newest alert at the beginning
    // --------------------------------------------------------

    alerts.unshift(
        alert
    );


    // --------------------------------------------------------
    // Keep latest 50 alerts
    // --------------------------------------------------------

    saveStoredAlerts(
        alerts.slice(
            0,
            50
        )
    );


    return alert;
}


// ============================================================
// RENDER REAL ML ALERTS
// ============================================================

function renderDeepfakeAlerts() {

    const list =
        document.getElementById(
            "alertList"
        );

    const emptyState =
        document.getElementById(
            "alertEmpty"
        );


    if (!list) {
        return;
    }


    // --------------------------------------------------------
    // Remove previously rendered ML alerts
    // --------------------------------------------------------

    const existingCards =
        list.querySelectorAll(
            "[data-deepfake-alert='true']"
        );


    existingCards.forEach(
        function (card) {

            card.remove();
        }
    );


    const alerts =
        getStoredAlerts();


    // --------------------------------------------------------
    // Empty state
    // --------------------------------------------------------

    if (alerts.length === 0) {

        if (emptyState) {

            emptyState.hidden =
                false;
        }

        return;
    }


    if (emptyState) {

        emptyState.hidden =
            true;
    }


    // --------------------------------------------------------
    // Render each alert
    // --------------------------------------------------------

    alerts.forEach(
        function (alert) {

            const card =
                document.createElement(
                    "article"
                );


            card.className =
                "alert";


            card.dataset.deepfakeAlert =
                "true";


            card.dataset.alertId =
                alert.id;


            const severity =
                String(
                    alert.severity
                    || "HIGH"
                ).toUpperCase();


            card.innerHTML = `

                <div class="alert-header">

                    <div>

                        <h3>
                            ${escapeHtml(
                                alert.title
                            )}
                        </h3>

                        <p>
                            ${escapeHtml(
                                alert.filename
                            )}
                        </p>

                    </div>


                    <span class="alert-severity">

                        ${escapeHtml(
                            severity
                        )}

                    </span>

                </div>


                <div class="alert-content">

                    <p>
                        ${escapeHtml(
                            alert.message
                        )}
                    </p>


                    <div class="alert-metrics">

                        <div>

                            <strong>
                                ${Number(
                                    alert.riskScore
                                    || 0
                                ).toFixed(2)}
                            </strong>

                            <span>
                                Risk Score
                            </span>

                        </div>


                        <div>

                            <strong>
                                ${formatPercent(
                                    alert.peakFakeProbability
                                )}
                            </strong>

                            <span>
                                Peak Fake Probability
                            </span>

                        </div>


                        <div>

                            <strong>
                                ${Number(
                                    alert.totalChunks
                                    || 0
                                )}
                            </strong>

                            <span>
                                Chunks Analyzed
                            </span>

                        </div>

                    </div>


                    <p>

                        ${
                            alert.verificationRequired
                                ? "Independent verification recommended."
                                : "No additional verification required."
                        }

                    </p>


                    <p class="alert-time">

                        ${escapeHtml(
                            formatDate(
                                alert.timestamp
                            )
                        )}

                    </p>

                </div>


                <button
                    type="button"
                    class="alert-dismiss"
                    data-alert-id="${escapeHtml(
                        alert.id
                    )}"
                >
                    Dismiss
                </button>

            `;


            list.appendChild(
                card
            );
        }
    );
}


// ============================================================
// DISMISS ALERT
// ============================================================

function dismissStoredAlert(
    alertId,
    card
) {

    const alerts =
        getStoredAlerts();


    const remaining =
        alerts.filter(
            function (alert) {

                return String(
                    alert.id
                ) !== String(
                    alertId
                );
            }
        );


    saveStoredAlerts(
        remaining
    );


    if (!card) {

        renderDeepfakeAlerts();

        return;
    }


    card.classList.add(
        "is-leaving"
    );


    let removed = false;


    function removeCard() {

        if (removed) {
            return;
        }

        removed = true;


        if (card.parentNode) {

            card.parentNode.removeChild(
                card
            );
        }


        updateEmptyState();
    }


    card.addEventListener(
        "transitionend",
        removeCard,
        {
            once: true
        }
    );


    // Fallback in case CSS has no transition
    setTimeout(
        removeCard,
        500
    );
}


// ============================================================
// EMPTY STATE
// ============================================================

function updateEmptyState() {

    const list =
        document.getElementById(
            "alertList"
        );

    const emptyState =
        document.getElementById(
            "alertEmpty"
        );


    if (!list || !emptyState) {
        return;
    }


    const cards =
        list.querySelectorAll(
            "[data-deepfake-alert='true']"
        );


    emptyState.hidden =
        cards.length !== 0;
}


// ============================================================
// CLICK HANDLER
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        const list =
            document.getElementById(
                "alertList"
            );


        if (!list) {
            return;
        }


        // Render stored ML alerts
        renderDeepfakeAlerts();


        // Handle dismiss
        list.addEventListener(
            "click",
            function (event) {

                const button =
                    event.target.closest(
                        ".alert-dismiss"
                    );


                if (!button) {
                    return;
                }


                const alertId =
                    button.dataset.alertId;


                const card =
                    button.closest(
                        ".alert"
                    );


                dismissStoredAlert(
                    alertId,
                    card
                );
            }
        );
    }
);