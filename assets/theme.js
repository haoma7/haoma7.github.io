// Theme slider: light <-> dark.
// With no stored choice the page follows the operating system; the slider
// shows whichever theme is actually in effect. Clicking stores an explicit
// choice, which then wins over the system setting.
(function () {
  function systemDark() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function stored() {
    try {
      var v = localStorage.getItem("theme");
      return v === "light" || v === "dark" ? v : null;
    } catch (e) { return null; }
  }

  function current() {
    return stored() || (systemDark() ? "dark" : "light");
  }

  function apply(mode, btn) {
    document.documentElement.setAttribute("data-theme", mode);
    try { localStorage.setItem("theme", mode); } catch (e) {}
    if (btn) btn.setAttribute("aria-checked", mode === "dark" ? "true" : "false");
  }

  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.getElementById("theme-btn");
    if (!btn) return;

    var mode = current();
    btn.setAttribute("aria-checked", mode === "dark" ? "true" : "false");

    btn.addEventListener("click", function () {
      mode = mode === "dark" ? "light" : "dark";
      apply(mode, btn);
    });

    // while no explicit choice has been made, keep following the system
    if (!stored() && window.matchMedia) {
      var mq = window.matchMedia("(prefers-color-scheme: dark)");
      var onChange = function (e) {
        if (stored()) return;
        mode = e.matches ? "dark" : "light";
        btn.setAttribute("aria-checked", e.matches ? "true" : "false");
      };
      if (mq.addEventListener) mq.addEventListener("change", onChange);
      else if (mq.addListener) mq.addListener(onChange);
    }
  });
})();
