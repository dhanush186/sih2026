// ============================================================
// REAL-TIME DEEPFAKE DETECTION FRONTEND
// ============================================================


// ============================================================
// DOM ELEMENTS
// ============================================================

const audioFile =
    document.getElementById("audioFile");

const analyzeButton =
    document.getElementById("analyzeButton");

const result =
    document.getElementById("result");

const dropzone =
    document.getElementById("dropzone");

let isAnalyzing = false;


// ============================================================
// DEBUG
// ============================================================

console.log("Frontend elements:", {
    audioFile,
    analyzeButton,
    result,
    dropzone
});


// ============================================================
// ELEMENT CHECK
// ============================================================

if (!audioFile) {
    console.error("Missing #audioFile element.");
}

if (!analyzeButton) {
    console.error("Missing #analyzeButton element.");
}

if (!result) {
    console.error("Missing #result element.");
}

if (!dropzone) {
    console.error("Missing #dropzone element.");
}


// ============================================================
// INITIAL BUTTON STATE
// ============================================================

if (analyzeButton) {
    analyzeButton.disabled = true;
}


// ============================================================
// FILE INPUT CHANGE
// ============================================================

if (audioFile) {

    audioFile.addEventListener(
        "change",
        function () {

            console.log(
                "File input changed."
            );

            updateDropzone();
        }
    );
}


// ============================================================
// DRAG AND DROP
// ============================================================

if (dropzone && audioFile) {

    dropzone.addEventListener(
        "dragover",
        function (event) {

            event.preventDefault();

            dropzone.classList.add(
                "dragover"
            );
        }
    );


    dropzone.addEventListener(
        "dragleave",
        function () {

            dropzone.classList.remove(
                "dragover"
            );
        }
    );


    dropzone.addEventListener(
        "drop",
        function (event) {

            event.preventDefault();

            dropzone.classList.remove(
                "dragover"
            );


            const files =
                event.dataTransfer.files;


            if (
                !files
                || files.length === 0
            ) {

                return;
            }


            // ------------------------------------------------
            // Put dropped file into the input
            // ------------------------------------------------

            const dataTransfer =
                new DataTransfer();


            dataTransfer.items.add(
                files[0]
            );


            audioFile.files =
                dataTransfer.files;


            updateDropzone();
        }
    );
}


// ============================================================
// UPDATE DROPZONE + ANALYZE BUTTON
// ============================================================

function updateDropzone() {

    if (
        !audioFile
        || !dropzone
        || !analyzeButton
    ) {

        return;
    }


    const hasFile =
        audioFile.files
        && audioFile.files.length > 0;


    if (hasFile) {

        const selectedFile =
            audioFile.files[0];


        // ----------------------------------------------------
        // Update dropzone
        // ----------------------------------------------------

        dropzone.classList.add(
            "has-file"
        );


        dropzone.dataset.filename =
            selectedFile.name;


        // ----------------------------------------------------
        // ENABLE ANALYZE BUTTON
        // ----------------------------------------------------

        analyzeButton.disabled = false;


        console.log(
            "Selected audio:",
            selectedFile.name
        );


        console.log(
            "File size:",
            selectedFile.size,
            "bytes"
        );


        console.log(
            "Analyze button enabled."
        );


    } else {

        // ----------------------------------------------------
        // No file
        // ----------------------------------------------------

        dropzone.classList.remove(
            "has-file"
        );


        delete dropzone.dataset.filename;


        analyzeButton.disabled = true;
    }
}


// ============================================================
// FILE VALIDATION
// ============================================================

function validateSelectedFile(
    file
) {

    if (!file) {

        return "No file selected.";
    }


    const fileName =
        String(
            file.name || ""
        ).toLowerCase();


    const extensionAllowed =
        fileName.endsWith(".wav")
        || fileName.endsWith(".mp3");


    if (!extensionAllowed) {

        return (
            "Only WAV and MP3 audio files are supported."
        );
    }


    if (file.size === 0) {

        return (
            "The selected audio file is empty."
        );
    }


    // 25 MB limit, matching the frontend text
    const maxFileSize =
        25 * 1024 * 1024;


    if (file.size > maxFileSize) {

        return (
            "The selected file is larger than 25 MB."
        );
    }


    return null;
}


// ============================================================
// ANALYZE BUTTON
// ============================================================

if (analyzeButton) {

    analyzeButton.addEventListener(
        "click",
        async function () {

            if (isAnalyzing) {

                return;
            }


            if (
                !audioFile
                || !result
            ) {

                console.error(
                    "Required frontend elements are missing."
                );

                return;
            }


            // ------------------------------------------------
            // Check file
            // ------------------------------------------------

            if (
                !audioFile.files
                || audioFile.files.length === 0
            ) {

                result.innerHTML = `
                    <div class="result-inner">

                        <h3>
                            Please select an audio file
                        </h3>

                        <p>
                            Select a voice recording
                            before starting the analysis.
                        </p>

                    </div>
                `;

                analyzeButton.disabled = true;

                return;
            }


            const selectedFile =
                audioFile.files[0];


            // ------------------------------------------------
            // Validate
            // ------------------------------------------------

            const validationError =
                validateSelectedFile(
                    selectedFile
                );


            if (validationError) {

                result.innerHTML = `
                    <div class="result-inner">

                        <h3>
                            File not accepted
                        </h3>

                        <p>
                            ${escapeHtml(
                                validationError
                            )}
                        </p>

                    </div>
                `;

                return;
            }


            // ------------------------------------------------
            // Start analysis
            // ------------------------------------------------

            isAnalyzing = true;

            analyzeButton.disabled = true;


            result.innerHTML = `
                <div class="result-inner">

                    <div class="result-status">

                        <span
                            class="scan-icon"
                            aria-hidden="true"
                        ></span>

                        <h3>
                            Analyzing voice...
                        </h3>

                    </div>

                    <p class="result-file">
                        ${escapeHtml(
                            selectedFile.name
                        )}
                    </p>

                    <p>
                        Sending audio to the
                        real-time detection engine...
                    </p>

                </div>
            `;


            // ------------------------------------------------
            // Prepare form data
            // ------------------------------------------------

            const formData =
                new FormData();


            formData.append(
                "file",
                selectedFile
            );


            try {

                console.log(
                    "Uploading:",
                    selectedFile.name
                );


                console.log(
                    "API endpoint:",
                    API_BASE_URL
                    + "/predict/realtime"
                );


                // ------------------------------------------------
                // API request
                // ------------------------------------------------

                const response =
                    await fetch(
                        API_BASE_URL
                        + "/predict/realtime",
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                // ------------------------------------------------
                // Parse server response
                // ------------------------------------------------

                const rawText =
                    await response.text();


                let rawData;


                try {

                    rawData =
                        JSON.parse(
                            rawText
                        );

                } catch (parseError) {

                    console.error(
                        "Invalid JSON response:",
                        rawText
                    );

                    throw new Error(
                        "Server returned an invalid response."
                    );
                }


                console.log(
                    "Real-time ML API response:",
                    rawData
                );


                // ------------------------------------------------
                // HTTP error
                // ------------------------------------------------

                if (!response.ok) {

                    throw new Error(
                        rawData.detail
                        || "Prediction request failed."
                    );
                }


                // ------------------------------------------------
                // Display
                // ------------------------------------------------

                displayRealtimeResult(
                    rawData
                );


                // ------------------------------------------------
                // Save alert
                // ------------------------------------------------

                if (
                    rawData.session_alert === true
                ) {

                    createDeepfakeAlert(
                        rawData
                    );
                }


                // ------------------------------------------------
                // Save history
                // ------------------------------------------------

                saveAnalysisHistory(
                    rawData
                );


            } catch (error) {

                console.error(
                    "ML API error:",
                    error
                );


                result.innerHTML = `
                    <div class="result-inner">

                        <h3>
                            Analysis failed
                        </h3>

                        <p>
                            ${escapeHtml(
                                error.message
                            )}
                        </p>

                        <p>
                            Make sure the real-time ML API
                            is running on port 8000.
                        </p>

                    </div>
                `;


            } finally {

                isAnalyzing = false;


                // Re-enable only if file still exists
                analyzeButton.disabled =
                    !(
                        audioFile.files
                        && audioFile.files.length > 0
                    );
            }
        }
    );
}


// ============================================================
// SAVE HIGH-RISK ALERT
// ============================================================

function createDeepfakeAlert(
    resultData
) {

    if (
        !resultData
        || !resultData.session_alert
    ) {

        return;
    }


    let alerts = [];


    try {

        alerts = JSON.parse(
            localStorage.getItem(
                ALERT_STORAGE_KEY
            ) || "[]"
        );


        if (!Array.isArray(alerts)) {

            alerts = [];
        }


    } catch (error) {

        console.error(
            "Could not read stored alerts:",
            error
        );

        alerts = [];
    }


    const chunks =
        Array.isArray(
            resultData.chunks
        )
            ? resultData.chunks
            : [];


    // --------------------------------------------------------
    // Peak fake probability
    // --------------------------------------------------------

    const peakFakeProbability =
        chunks.length > 0
            ? Math.max(
                ...chunks.map(
                    function (chunk) {

                        return Number(
                            chunk.fake_probability
                            || 0
                        );
                    }
                )
            )
            : Number(
                resultData.rolling_fake_probability
                || 0
            );


    // --------------------------------------------------------
    // Speaker verification
    // --------------------------------------------------------

    const speakerVerification =
        resultData.speaker_verification
        || {};


    const speakerSimilarity =
        speakerVerification.similarity_score !== null
        && speakerVerification.similarity_score !== undefined
            ? Number(
                speakerVerification.similarity_score
            )
            : null;


    const speakerMismatch =
        speakerVerification.mismatch_score !== null
        && speakerVerification.mismatch_score !== undefined
            ? Number(
                speakerVerification.mismatch_score
            )
            : null;


    const sameSpeaker =
        speakerVerification.same_speaker !== null
        && speakerVerification.same_speaker !== undefined
            ? Boolean(
                speakerVerification.same_speaker
            )
            : null;


    // --------------------------------------------------------
    // Alert object
    // --------------------------------------------------------

    const alert = {

        id:
            Date.now().toString(),

        type:
            "DEEPFAKE_VOICE",

        title:
            "High-Risk Voice Alert",

        severity:
            String(
                resultData.overall_risk
                || "HIGH"
            ).toUpperCase(),

        filename:
            resultData.filename
            || "Unknown audio",

        riskScore:
            Number(
                resultData.overall_risk_score
                || 0
            ),

        peakFakeProbability:
            peakFakeProbability,

        rollingFakeProbability:
            Number(
                resultData.rolling_fake_probability
                || 0
            ),

        totalChunks:
            Number(
                resultData.total_chunks
                || chunks.length
                || 0
            ),

        sessionAlert:
            Boolean(
                resultData.session_alert
            ),

        message:
            "Suspicious synthetic-voice evidence detected.",

        verificationRequired:
            Boolean(
                resultData.requires_verification
            ),

        speakerVerificationEnabled:
            Boolean(
                speakerVerification.enabled
            ),

        speakerReferenceFile:
            speakerVerification.reference_file
            || null,

        speakerSimilarity:
            speakerSimilarity,

        speakerMismatchScore:
            speakerMismatch,

        sameSpeaker:
            sameSpeaker,

        timestamp:
            new Date().toISOString()
    };


    // --------------------------------------------------------
    // Avoid duplicate alert for 10 seconds
    // --------------------------------------------------------

    const recentlyExists =
        alerts.some(
            function (existingAlert) {

                if (
                    existingAlert.filename
                    !== alert.filename
                ) {

                    return false;
                }


                if (
                    existingAlert.severity
                    !== alert.severity
                ) {

                    return false;
                }


                if (!existingAlert.timestamp) {

                    return false;
                }


                const existingTime =
                    new Date(
                        existingAlert.timestamp
                    ).getTime();


                return (
                    Date.now()
                    - existingTime
                    < 10000
                );
            }
        );


    if (recentlyExists) {

        console.log(
            "Recent deepfake alert already exists."
        );

        return;
    }


    alerts.unshift(
        alert
    );


    alerts =
        alerts.slice(
            0,
            50
        );


    try {

        localStorage.setItem(
            ALERT_STORAGE_KEY,
            JSON.stringify(
                alerts
            )
        );


        console.log(
            "Deepfake alert saved:",
            alert
        );


    } catch (error) {

        console.error(
            "Could not save deepfake alert:",
            error
        );
    }
}


// ============================================================
// SAVE ANALYSIS HISTORY
// ============================================================

function saveAnalysisHistory(
    resultData
) {

    if (!resultData) {

        return;
    }


    let history = [];


    try {

        history = JSON.parse(
            localStorage.getItem(
                HISTORY_STORAGE_KEY
            ) || "[]"
        );


        if (!Array.isArray(history)) {

            history = [];
        }


    } catch (error) {

        console.error(
            "Could not read analysis history:",
            error
        );

        history = [];
    }


    const chunks =
        Array.isArray(
            resultData.chunks
        )
            ? resultData.chunks
            : [];


    const peakFakeProbability =
        chunks.length > 0
            ? Math.max(
                ...chunks.map(
                    function (chunk) {

                        return Number(
                            chunk.fake_probability
                            || 0
                        );
                    }
                )
            )
            : Number(
                resultData.rolling_fake_probability
                || 0
            );


    const speakerVerification =
        resultData.speaker_verification
        || {};


    const speakerSimilarity =
        speakerVerification.similarity_score !== null
        && speakerVerification.similarity_score !== undefined
            ? Number(
                speakerVerification.similarity_score
            )
            : null;


    const speakerMismatch =
        speakerVerification.mismatch_score !== null
        && speakerVerification.mismatch_score !== undefined
            ? Number(
                speakerVerification.mismatch_score
            )
            : null;


    const sameSpeaker =
        speakerVerification.same_speaker !== null
        && speakerVerification.same_speaker !== undefined
            ? Boolean(
                speakerVerification.same_speaker
            )
            : null;


    const record = {

        id:
            Date.now().toString(),

        filename:
            resultData.filename
            || "Unknown audio",

        durationSeconds:
            Number(
                resultData.duration_seconds
                || 0
            ),

        overallRisk:
            String(
                resultData.overall_risk
                || "LOW"
            ).toUpperCase(),

        riskScore:
            Number(
                resultData.overall_risk_score
                || 0
            ),

        rollingFakeProbability:
            Number(
                resultData.rolling_fake_probability
                || 0
            ),

        peakFakeProbability:
            peakFakeProbability,

        totalChunks:
            Number(
                resultData.total_chunks
                || chunks.length
                || 0
            ),

        sessionAlert:
            Boolean(
                resultData.session_alert
            ),

        requiresVerification:
            Boolean(
                resultData.requires_verification
            ),

        speakerVerificationEnabled:
            Boolean(
                speakerVerification.enabled
            ),

        speakerReferenceFile:
            speakerVerification.reference_file
            || null,

        speakerSimilarity:
            speakerSimilarity,

        speakerMismatchScore:
            speakerMismatch,

        sameSpeaker:
            sameSpeaker,

        timestamp:
            new Date().toISOString()
    };


    history.unshift(
        record
    );


    history =
        history.slice(
            0,
            100
        );


    try {

        localStorage.setItem(
            HISTORY_STORAGE_KEY,
            JSON.stringify(
                history
            )
        );


        console.log(
            "Analysis history saved:",
            record
        );


    } catch (error) {

        console.error(
            "Could not save analysis history:",
            error
        );
    }
}


// ============================================================
// DISPLAY REAL-TIME RESULT
// ============================================================

function displayRealtimeResult(
    data
) {

    const overallRisk =
        String(
            data.overall_risk
            || "LOW"
        ).toUpperCase();


    const riskScore =
        Number(
            data.overall_risk_score
            || 0
        );


    const rollingFakeProbability =
        Number(
            data.rolling_fake_probability
            || 0
        );


    const sessionAlert =
        Boolean(
            data.session_alert
        );


    const requiresVerification =
        Boolean(
            data.requires_verification
        );


    const chunks =
        Array.isArray(
            data.chunks
        )
            ? data.chunks
            : [];


    // --------------------------------------------------------
    // Speaker verification
    // --------------------------------------------------------

    const speakerVerification =
        data.speaker_verification
        || {};


    const speakerEnabled =
        Boolean(
            speakerVerification.enabled
        );


    const speakerSimilarity =
        speakerVerification.similarity_score !== null
        && speakerVerification.similarity_score !== undefined
            ? Number(
                speakerVerification.similarity_score
            )
            : null;


    const speakerMismatch =
        speakerVerification.mismatch_score !== null
        && speakerVerification.mismatch_score !== undefined
            ? Number(
                speakerVerification.mismatch_score
            )
            : null;


    const sameSpeaker =
        speakerVerification.same_speaker !== null
        && speakerVerification.same_speaker !== undefined
            ? Boolean(
                speakerVerification.same_speaker
            )
            : null;


    // --------------------------------------------------------
    // Risk badge
    // --------------------------------------------------------

    let riskBadgeClass =
        "risk-badge low";


    if (
        overallRisk === "HIGH"
    ) {

        riskBadgeClass =
            "risk-badge high";

    } else if (
        overallRisk === "MEDIUM"
    ) {

        riskBadgeClass =
            "risk-badge medium";
    }


    // --------------------------------------------------------
    // Chunk rows
    // --------------------------------------------------------

    const chunkRows =
        chunks
            .map(
                function (chunk) {

                    const start =
                        Number(
                            chunk.start_seconds
                            || 0
                        );


                    const end =
                        Number(
                            chunk.end_seconds
                            || 0
                        );


                    const fakeProbability =
                        Number(
                            chunk.fake_probability
                            || 0
                        );


                    const realProbability =
                        Number(
                            chunk.real_probability
                            || 0
                        );


                    const chunkRisk =
                        String(
                            chunk.risk_level
                            || "LOW"
                        ).toUpperCase();


                    const chunkScore =
                        Number(
                            chunk.risk_score
                            || 0
                        );


                    let chunkRiskClass =
                        "chunk-risk-low";


                    if (
                        chunkRisk === "HIGH"
                    ) {

                        chunkRiskClass =
                            "chunk-risk-high";

                    } else if (
                        chunkRisk === "MEDIUM"
                    ) {

                        chunkRiskClass =
                            "chunk-risk-medium";
                    }


                    return `
                        <div class="chunk-row">

                            <div class="chunk-time">
                                ${start.toFixed(2)}s
                                -
                                ${end.toFixed(2)}s
                            </div>

                            <div class="chunk-probability">
                                Fake:
                                ${(fakeProbability * 100).toFixed(2)}%
                            </div>

                            <div class="chunk-probability">
                                Real:
                                ${(realProbability * 100).toFixed(2)}%
                            </div>

                            <div
                                class="${chunkRiskClass}"
                            >
                                ${escapeHtml(
                                    chunkRisk
                                )}
                            </div>

                            <div class="chunk-score">
                                ${chunkScore.toFixed(2)}
                            </div>

                        </div>
                    `;
                }
            )
            .join("");


    // --------------------------------------------------------
    // Verification warning
    // --------------------------------------------------------

    const verificationMessage =
        requiresVerification
            ? `
                <div class="verification-warning">
                    Independent verification is recommended
                    before taking any sensitive action.
                </div>
            `
            : "";


    // --------------------------------------------------------
    // Session alert
    // --------------------------------------------------------

    const sessionAlertMessage =
        sessionAlert
            ? `
                <div class="session-alert">
                    Session alert triggered:
                    suspicious synthetic-voice evidence
                    was detected.
                </div>
            `
            : `
                <div class="session-normal">
                    No persistent session alert triggered.
                </div>
            `;


    // --------------------------------------------------------
    // Speaker panel
    // --------------------------------------------------------

    let speakerVerificationPanel = "";


    if (speakerEnabled) {

        let speakerStatus =
            "NOT AVAILABLE";


        if (
            sameSpeaker === true
        ) {

            speakerStatus =
                "SAME SPEAKER";

        } else if (
            sameSpeaker === false
        ) {

            speakerStatus =
                "DIFFERENT SPEAKER";
        }


        speakerVerificationPanel = `
            <h3>
                Speaker Verification
            </h3>

            <div class="realtime-summary">

                <div class="result-row">

                    <span>
                        Verification
                    </span>

                    <strong>
                        ENABLED
                    </strong>

                </div>


                <div class="result-row">

                    <span>
                        Reference
                    </span>

                    <strong>
                        ${escapeHtml(
                            speakerVerification.reference_file
                            || "Trusted reference"
                        )}
                    </strong>

                </div>


                <div class="result-row">

                    <span>
                        Similarity Score
                    </span>

                    <strong>
                        ${
                            speakerSimilarity !== null
                                ? speakerSimilarity.toFixed(4)
                                : "N/A"
                        }
                    </strong>

                </div>


                <div class="result-row">

                    <span>
                        Speaker Status
                    </span>

                    <strong>
                        ${escapeHtml(
                            speakerStatus
                        )}
                    </strong>

                </div>


                <div class="result-row">

                    <span>
                        Mismatch Score
                    </span>

                    <strong>
                        ${
                            speakerMismatch !== null
                                ? speakerMismatch.toFixed(2)
                                : "N/A"
                        }${
                            speakerMismatch !== null
                                ? "%"
                                : ""
                        }
                    </strong>

                </div>

            </div>
        `;

    } else {

        speakerVerificationPanel = `
            <h3>
                Speaker Verification
            </h3>

            <div class="session-normal">
                Trusted speaker verification was not used
                for this analysis.
            </div>
        `;
    }


    // --------------------------------------------------------
    // Render
    // --------------------------------------------------------

    result.innerHTML = `

        <div class="result-inner">

            <div class="result-status">

                <h3>
                    Real-time analysis
                </h3>

                <span
                    class="${riskBadgeClass}"
                >
                    ${escapeHtml(
                        overallRisk
                    )}
                    RISK
                </span>

            </div>


            <p class="result-file">
                ${escapeHtml(
                    data.filename
                    || ""
                )}
            </p>


            <div class="realtime-summary">

                <div class="result-row">

                    <span>
                        Overall Risk
                    </span>

                    <strong>
                        ${escapeHtml(
                            overallRisk
                        )}
                    </strong>

                </div>


                <div class="result-row">

                    <span>
                        Risk Score
                    </span>

                    <strong>
                        ${riskScore.toFixed(2)}/100
                    </strong>

                </div>


                <div class="result-row">

                    <span>
                        Rolling Fake Probability
                    </span>

                    <strong>
                        ${(rollingFakeProbability * 100).toFixed(2)}%
                    </strong>

                </div>


                <div class="result-row">

                    <span>
                        Chunks Analyzed
                    </span>

                    <strong>
                        ${chunks.length}
                    </strong>

                </div>


                <div class="result-row">

                    <span>
                        Session Alert
                    </span>

                    <strong>
                        ${
                            sessionAlert
                                ? "TRIGGERED"
                                : "NOT TRIGGERED"
                        }
                    </strong>

                </div>

            </div>


            ${speakerVerificationPanel}


            ${sessionAlertMessage}


            <h3>
                Chunk Analysis
            </h3>


            <div class="chunk-list">

                <div class="chunk-header">

                    <span>
                        Time
                    </span>

                    <span>
                        Fake
                    </span>

                    <span>
                        Real
                    </span>

                    <span>
                        Risk
                    </span>

                    <span>
                        Score
                    </span>

                </div>

                ${chunkRows}

            </div>


            ${verificationMessage}


            <p class="result-footnote">
                ⚠ Detection is a risk signal, not definitive proof
                of synthetic audio.
            </p>

        </div>
    `;
}


// ============================================================
// HTML ESCAPING
// ============================================================

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