/* =========================================================
   Lets a security analyst dismiss an alert from the list.
   Client-side only — purely a UI affordance for this static
   demo view, no data is deleted from any backend.
   ========================================================= */

document.addEventListener("DOMContentLoaded", function () {
  const list = document.getElementById("alertList");
  const emptyState = document.getElementById("alertEmpty");
  if (!list) return;

  list.addEventListener("click", function (event) {
    const button = event.target.closest(".alert-dismiss");
    if (!button) return;

    const alertCard = button.closest(".alert");
    alertCard.classList.add("is-leaving");

    alertCard.addEventListener("transitionend", function () {
      alertCard.remove();
      if (list.children.length === 0) {
        emptyState.hidden = false;
      }
    }, { once: true });
  });
});
