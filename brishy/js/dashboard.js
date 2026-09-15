/* =========================================================
   Renders the "Current analysis" meters using the same
   meter component used on the Voice Analysis page, then
   animates them in on load.
   ========================================================= */

document.addEventListener("DOMContentLoaded", function () {
  const container = document.getElementById("currentAnalysisMeters");
  if (!container) return;

  container.innerHTML =
    meterRowMarkup("Speaker match", 92, 100, "%") +
    meterRowMarkup("AI detection", 87, 100, "%") +
    meterRowMarkup("Risk score", 78, 100, "/100");

  animateMeters(container);
});
