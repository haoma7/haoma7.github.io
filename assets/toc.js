// Highlights the contents entry for the section currently in view.
document.addEventListener("DOMContentLoaded", function () {
  var links = Array.prototype.slice.call(document.querySelectorAll(".toc a"));
  if (!links.length || !("IntersectionObserver" in window)) return;

  var targets = links
    .map(function (a) { return document.getElementById(a.getAttribute("href").split("#")[1]); })
    .filter(Boolean);

  var obs = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      links.forEach(function (a) {
        a.classList.toggle("on", a.getAttribute("href").split("#")[1] === e.target.id);
      });
    });
  }, { rootMargin: "0px 0px -70% 0px", threshold: 0 });

  targets.forEach(function (t) { obs.observe(t); });
});
