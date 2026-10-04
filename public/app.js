/**
 * News Category Predictor - Client Application
 */

const CATEGORIES = {
  "World": {
    icon: "🌍",
    color: "#3B82F6",
    title: "World News",
    sub: "Global affairs, diplomacy, international politics, and conflicts"
  },
  "Sports": {
    icon: "⚽",
    color: "#10B981",
    title: "Sports",
    sub: "Athletics, tournaments, teams, championship games, and scores"
  },
  "Business": {
    icon: "💼",
    color: "#F59E0B",
    title: "Business & Finance",
    sub: "Stock markets, corporate earnings, trade, economics, and investing"
  },
  "Sci/Tech": {
    icon: "🔬",
    color: "#8B5CF6",
    title: "Science & Technology",
    sub: "Space exploration, software, artificial intelligence, and research"
  }
};

const EXAMPLES = {
  world: "Diplomats from 15 member nations reached consensus in Geneva regarding immediate peacekeeping deployment and humanitarian corridors in conflict zones.",
  sports: "Real Madrid clinches Champions League quarterfinal spot with dramatic stoppage-time header in the 94th minute.",
  business: "Wall Street indices rallied after the Federal Reserve held interest rates steady and signaled potential cuts as inflation cooled.",
  scitech: "NASA James Webb Space Telescope discovers earliest galaxy candidates in deep infrared spectrometry survey."
};

// Elements
const form = document.getElementById("predictForm");
const textarea = document.getElementById("articleText");
const charCounter = document.getElementById("charCounter");
const clearBtn = document.getElementById("clearTextBtn");
const submitBtn = document.getElementById("submitBtn");
const spinner = document.getElementById("spinner");
const btnLabel = document.getElementById("btnLabel");

const resultCard = document.getElementById("resultCard");
const categoryBanner = document.getElementById("categoryBanner");
const bannerIcon = document.getElementById("bannerIcon");
const bannerTitle = document.getElementById("bannerTitle");

// =====================================================================
// Initialization
// =====================================================================
document.addEventListener("DOMContentLoaded", () => {
  setupTextarea();
  setupExamples();
  setupFormSubmit();
});

// Character counter
function setupTextarea() {
  function updateCount() {
    const len = textarea.value.length;
    charCounter.textContent = `${len} character${len === 1 ? '' : 's'}`;
  }
  textarea.addEventListener("input", updateCount);

  // Clear button
  clearBtn.addEventListener("click", () => {
    textarea.value = "";
    updateCount();
    resultCard.style.display = "none";
    textarea.focus();
  });

  // Ctrl + Enter shortcut
  textarea.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      form.requestSubmit();
    }
  });
}

// Example chips
function setupExamples() {
  document.querySelectorAll(".example-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      const type = pill.dataset.type;
      if (EXAMPLES[type]) {
        textarea.value = EXAMPLES[type];
        textarea.dispatchEvent(new Event("input"));
        textarea.focus();
      }
    });
  });
}



// =====================================================================
// Form Submission & API Call
// =====================================================================
function setupFormSubmit() {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = textarea.value.trim();

    if (!text) {
      textarea.focus();
      return;
    }

    setLoading(true);

    try {
      const payload = { text };
      let res = null;

      // Robust serverless endpoint fallback
      const candidateEndpoints = ["/api/predict", "/predict", "/api", "/api/index.py"];
      let lastStatus = 0;

      for (const endpoint of candidateEndpoints) {
        try {
          res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          });
          if (res.ok) {
            break;
          }
          lastStatus = res.status;
          if (res.status === 404) {
            continue;
          }
          break;
        } catch (fetchErr) {
          // Continue to next endpoint on network issue
        }
      }

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned ${res.status}`);
      }

      const data = await res.json();
      displayResult(data);
    } catch (err) {
      alert(`Error predicting category: ${err.message}`);
    } finally {
      setLoading(false);
    }
  });
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  spinner.style.display = isLoading ? "inline-block" : "none";
  btnLabel.textContent = isLoading ? "Predicting..." : "Predict Category";
}

// =====================================================================
// Display Prediction Result
// =====================================================================
function displayResult(data) {
  const cat = data.category_name;
  const config = CATEGORIES[cat] || {
    icon: "📰",
    color: "#6366F1",
    title: cat
  };

  // Show result card
  resultCard.style.display = "block";

  // Category banner
  categoryBanner.style.setProperty("--banner-accent", config.color);
  bannerIcon.textContent = config.icon;
  bannerTitle.textContent = config.title;

  // Smooth scroll down to result
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
