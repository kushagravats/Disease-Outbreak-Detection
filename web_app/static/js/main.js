document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener("click", function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute("href"));
    if (target) {
      target.scrollIntoView({
        behavior: "smooth",
      });
    }
  });
});

const currentLocation = window.location.pathname;
const navLinks = document.querySelectorAll(".nav-links a");

navLinks.forEach((link) => {
  if (link.getAttribute("href") === currentLocation) {
    link.classList.add("active");
  }
});

window.addEventListener("load", () => {
  const confidenceBars = document.querySelectorAll(".confidence-fill");
  confidenceBars.forEach((bar) => {
    const width = bar.style.width;
    if (width) {
      bar.style.width = "0%";
      setTimeout(() => {
        bar.style.width = width;
      }, 300);
    }
  });
});

const predictForm = document.querySelector("form");
if (predictForm) {
  predictForm.addEventListener("submit", function (e) {
    const textarea = this.querySelector("textarea");
    if (textarea && textarea.value.trim().length < 10) {
      e.preventDefault();
      alert("Please enter at least 10 characters for analysis.");
      textarea.focus();
      return false;
    }

    const submitButton = this.querySelector(".btn-submit");
    if (submitButton) {
      submitButton.innerHTML = "<span>🔄 Analyzing...</span>";
    }

    return true;
  });
}

console.log("🏥 Disease Outbreak Detection Dashboard Loaded");
