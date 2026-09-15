/* =========================================================
   Client-side filter for the analysis history table.
   Purely cosmetic/UX — no data is added or removed, rows are
   just shown or hidden based on the file name search.
   ========================================================= */

document.addEventListener("DOMContentLoaded", function () {
  const search = document.getElementById("historySearch");
  const table = document.getElementById("historyTable");
  const emptyState = document.getElementById("tableEmpty");
  if (!search || !table) return;

  const rows = Array.from(table.querySelectorAll("tbody tr"));

  search.addEventListener("input", function () {
    const query = search.value.trim().toLowerCase();
    let visibleCount = 0;

    rows.forEach(function (row) {
      const fileName = row.firstElementChild.textContent.toLowerCase();
      const matches = fileName.includes(query);
      row.hidden = !matches;
      if (matches) visibleCount++;
    });

    emptyState.hidden = visibleCount !== 0;
  });
});
