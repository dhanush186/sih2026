const audioFile = document.getElementById("audioFile");
const analyzeButton = document.getElementById("analyzeButton");
const result = document.getElementById("result");
const dropzone = document.getElementById("dropzone");
const dropzoneTitle = document.getElementById("dropzoneTitle");
const dropzoneSub = document.getElementById("dropzoneSub");

// --- Upload restrictions (checklist #14 Validate input / #16 Restrict file uploads) ---
// The accept="audio/*" attribute on the <input> is only a UI hint — browsers
// don't enforce it, so the real check happens here before anything is sent.
const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024; // 25 MB
const ALLOWED_AUDIO_TYPES = [
  "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3",
  "audio/mp4", "audio/x-m4a", "audio/aac", "audio/ogg", "audio/webm",
];
const ALLOWED_EXTENSIONS = [".wav", ".mp3", ".m4a", ".aac", ".ogg", ".webm"];

let isAnalyzing = false;

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function getExtension(name) {
  const match = /\.[^.]+$/.exec(name || "");
  return match ? match[0].toLowerCase() : "";
}

/**
 * Client-side validation only. This is a UX convenience, not a security
 * boundary — the backend must re-validate type, size and content itself
 * once it exists, since anything sent from the browser can be forged.
 */
function validateSelectedFile(file) {
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return `File is too large (${formatFileSize(file.size)}). Maximum size is ${formatFileSize(MAX_FILE_SIZE_BYTES)}.`;
  }
  const typeOk = ALLOWED_AUDIO_TYPES.includes(file.type);
  const extensionOk = ALLOWED_EXTENSIONS.includes(getExtension(file.name));
  if (!typeOk && !extensionOk) {
    return "Unsupported file type. Please choose a WAV, MP3, M4A, AAC, OGG or WEBM recording.";
  }
  return null;
}

function updateDropzone() {
  if (audioFile.files.length === 0) {
    dropzone.classList.remove("has-file");
    dropzoneTitle.textContent = "Drop an audio file here, or click to browse";
    dropzoneSub.textContent = "WAV, MP3, M4A and other common audio formats — up to " + formatFileSize(MAX_FILE_SIZE_BYTES);
    analyzeButton.disabled = true;
    return;
  }

  const file = audioFile.files[0];
  const validationError = validateSelectedFile(file);

  if (validationError) {
    dropzone.classList.remove("has-file");
    dropzoneTitle.textContent = escapeHtml(file.name);
    dropzoneSub.textContent = validationError;
    analyzeButton.disabled = true;
    audioFile.value = "";
    return;
  }

  dropzone.classList.add("has-file");
  dropzoneTitle.textContent = file.name;
  dropzoneSub.textContent = formatFileSize(file.size) + " — click to choose a different file";
  analyzeButton.disabled = false;
}

audioFile.addEventListener("change", updateDropzone);

// Drag-and-drop support layered on top of the native file input.
["dragenter", "dragover"].forEach(function (eventName) {
  dropzone.addEventListener(eventName, function (event) {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach(function (eventName) {
  dropzone.addEventListener(eventName, function (event) {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", function (event) {
  const files = event.dataTransfer.files;
  if (files.length > 0) {
    audioFile.files = files;
    updateDropzone();
  }
});


analyzeButton.addEventListener("click", async function () {

    // Guard against double submits / button-spamming while a request
    // is already in flight (checklist #12 — lightweight bot/abuse guard;
    // real rate limiting still has to live on the backend, see #11/#12).
    if (isAnalyzing) {
        return;
    }

    if (audioFile.files.length === 0) {

        result.innerHTML = `
            <div class="result-inner">
                <h3>Please select an audio file</h3>
                <p>Select a voice recording before starting the analysis.</p>
            </div>
        `;

        return;
    }


    const selectedFile = audioFile.files[0];

    const validationError = validateSelectedFile(selectedFile);
    if (validationError) {
        result.innerHTML = `
            <div class="result-inner">
                <h3>File not accepted</h3>
                <p>${escapeHtml(validationError)}</p>
            </div>
        `;
        return;
    }

    isAnalyzing = true;
    analyzeButton.disabled = true;

    result.innerHTML = `
        <div class="result-inner">
            <div class="result-status">
                <span class="scan-icon" aria-hidden="true"></span>
                <h3>Preparing analysis…</h3>
            </div>
            <p class="result-file">${escapeHtml(selectedFile.name)}</p>
        </div>
    `;


    const formData = new FormData();

    formData.append("audio", selectedFile);


    try {

        result.innerHTML = `
            <div class="result-inner">
                <div class="result-status">
                    <span class="scan-icon" aria-hidden="true"></span>
                    <h3>Analyzing voice…</h3>
                </div>
                <p class="result-file">Please wait while the recording is being processed.</p>
            </div>
        `;


        /*
         * REAL BACKEND REQUEST WILL BE ADDED HERE
         *
         * Example:
         *
         * const response = await fetch(
         *     API_BASE_URL + "/analyze",
         *     {
         *         method: "POST",
         *         body: formData
         *     }
         * );
         *
         * const rawData = await response.json();
         * const data = pickResultFields(rawData);
         *
         * API_BASE_URL should point at an https:// origin in production
         * (checklist #19 Force HTTPS) — it is only http://localhost here
         * because there is no backend to talk to yet.
         */


        // Temporary mock result for frontend testing

        await new Promise(function (resolve) {

            setTimeout(resolve, 2000);

        });


        const mockResult = {

            fileName: selectedFile.name,

            aiProbability: 87,

            speakerMatch: 92,

            riskScore: 78,

            riskLevel: "HIGH"

        };


        displayResult(pickResultFields(mockResult));


    } catch (error) {

        console.error(error);

        result.innerHTML = `
            <div class="result-inner">
                <h3>Analysis failed</h3>
                <p>Something went wrong while analyzing the voice.</p>
                <p>Please try again.</p>
            </div>
        `;

    } finally {

        isAnalyzing = false;
        analyzeButton.disabled = audioFile.files.length === 0;

    }

});


/**
 * Whitelists exactly the fields the UI knows how to render and coerces
 * their types (checklist #17 Trim API responses). Once a real backend
 * is wired up, its response should be passed through this before it
 * ever reaches displayResult() — never render an API payload directly.
 */
function pickResultFields(raw) {
  const allowedRiskLevels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];
  const riskLevel = allowedRiskLevels.includes(String(raw.riskLevel).toUpperCase())
    ? String(raw.riskLevel).toUpperCase()
    : "LOW";

  return {
    fileName: String(raw.fileName ?? "").slice(0, 255),
    aiProbability: Math.min(Math.max(Number(raw.aiProbability) || 0, 0), 100),
    speakerMatch: Math.min(Math.max(Number(raw.speakerMatch) || 0, 0), 100),
    riskScore: Math.min(Math.max(Number(raw.riskScore) || 0, 0), 100),
    riskLevel: riskLevel,
  };
}

function displayResult(data) {

    result.innerHTML = `
        <div class="result-inner">
            <div class="result-status">
                <h3>Analysis result</h3>
                <span class="${riskBadgeClass(data.riskLevel)}">${escapeHtml(data.riskLevel)} RISK</span>
            </div>
            <p class="result-file">${escapeHtml(data.fileName)}</p>

            <div class="result-gauges">
                <div>${gaugeMarkup("gaugeAiProbability", "AI probability")}</div>
                <div>${gaugeMarkup("gaugeSpeakerMatch", "Speaker match")}</div>
            </div>

            <div class="meters">
                ${meterRowMarkup("Risk score", data.riskScore, 100, "/100")}
            </div>

            <p class="result-footnote">⚠ This is currently a mock result — live analysis will replace it once the backend is connected.</p>
        </div>
    `;

    animateGauge("gaugeAiProbability", data.aiProbability);
    animateGauge("gaugeSpeakerMatch", data.speakerMatch);
    animateMeters(result);

}