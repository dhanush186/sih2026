/* =========================================================
   Shared UI helpers — radial gauges, meter bars, risk badges.
   Used by the analysis result panel and the dashboard.
   ========================================================= */

const RISK_COLOR = {
  LOW: "var(--signal)",
  MEDIUM: "var(--warn)",
  HIGH: "var(--danger)",
  CRITICAL: "var(--danger)",
};

/**
 * Escapes a string before it is interpolated into innerHTML.
 * Anything that reaches the page from outside our own code —
 * a selected file name, a future API response field — must go
 * through this first so it can never be interpreted as markup.
 */
function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = String(value ?? "");
  return div.innerHTML;
}

function riskBadgeClass(level) {
  const key = String(level || "").toUpperCase();
  if (key === "LOW") return "badge badge-low";
  if (key === "MEDIUM") return "badge badge-medium";
  if (key === "CRITICAL") return "badge badge-critical";
  return "badge badge-high";
}

function colorForPercent(percent) {
  if (percent >= 75) return "var(--danger)";
  if (percent >= 45) return "var(--warn)";
  return "var(--signal)";
}

/**
 * Builds the static markup for a radial gauge. Call animateGauge()
 * afterwards (e.g. on next animation frame) to fill it in.
 */
function gaugeMarkup(id, label) {
  return `
    <div class="gauge" id="${id}">
      <svg viewBox="0 0 128 128">
        <circle class="gauge-track" cx="64" cy="64" r="54"></circle>
        <circle class="gauge-value" cx="64" cy="64" r="54" stroke="var(--signal)"></circle>
      </svg>
      <div class="gauge-readout">
        <strong data-gauge-number>0%</strong>
      </div>
    </div>
    <p class="gauge-caption">${label}</p>
  `;
}

function animateGauge(id, percent) {
  const wrap = document.getElementById(id);
  if (!wrap) return;
  const circle = wrap.querySelector(".gauge-value");
  const number = wrap.querySelector("[data-gauge-number]");
  const circumference = 339.292;
  const color = colorForPercent(percent);
  circle.style.stroke = color;

  requestAnimationFrame(function () {
    circle.style.strokeDashoffset = String(
      circumference - (Math.min(percent, 100) / 100) * circumference
    );
  });

  const duration = 900;
  const start = performance.now();

  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    number.textContent = Math.round(eased * percent) + "%";
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function meterRowMarkup(label, value, max, displaySuffix) {
  const percent = Math.min((value / max) * 100, 100);
  return `
    <div class="meter-row">
      <div class="meter-label">${label}</div>
      <div class="meter-track">
        <div class="meter-fill" data-target="${percent}" style="background:${colorForPercent(percent)}"></div>
      </div>
      <div class="meter-value">${value}${displaySuffix || ""}</div>
    </div>
  `;
}

function animateMeters(root) {
  const fills = (root || document).querySelectorAll(".meter-fill[data-target]");
  fills.forEach(function (fill) {
    const target = fill.getAttribute("data-target");
    requestAnimationFrame(function () {
      fill.style.width = target + "%";
    });
  });
}
