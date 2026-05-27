const predictionForm = document.getElementById("predictionForm");
const formError = document.getElementById("formError");
const resultGrid = document.getElementById("resultGrid");
const emptyState = document.getElementById("emptyState");
const loadingPill = document.getElementById("loadingPill");
const audioInput = document.getElementById("audioInput");
const fileName = document.getElementById("fileName");
const uploadZone = document.getElementById("uploadZone");
const textInput = document.getElementById("textInput");
const audioSection = document.getElementById("audioSection");
const textSection = document.getElementById("textSection");
const modeHelp = document.getElementById("modeHelp");
const resetButton = document.getElementById("resetButton");
const backendStatus = document.getElementById("backendStatus");
const resultHint = document.getElementById("resultHint");

const modeCopy = {
  speech: {
    help: "Speech mode uses only the uploaded audio file.",
    empty: "Upload an audio file to run the speech-only model.",
  },
  text: {
    help: "Text mode uses only the transcript field.",
    empty: "Enter transcript text to run the text-only model.",
  },
  fusion: {
    help: "Fusion mode combines the uploaded audio file with transcript text.",
    empty: "Upload audio and enter transcript text to run fusion inference.",
  },
};

const resultTone = {
  angry: "tone-angry",
  disgust: "tone-disgust",
  fear: "tone-fear",
  happy: "tone-happy",
  neutral: "tone-neutral",
  pleasant_surprise: "tone-surprise",
  sad: "tone-sad",
};

function currentMode() {
  return document.querySelector("input[name='mode']:checked").value;
}

function prettyName(value) {
  return String(value).replaceAll("_", " ");
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with status ${response.status}`);
  }
  return payload;
}

function setBackendStatus(ok) {
  backendStatus.innerHTML = `<span class="status-dot ${ok ? "ok" : "bad"}"></span>${
    ok ? "Models ready" : "Backend unavailable"
  }`;
}

function updatePredictionMode() {
  const mode = currentMode();
  const needsAudio = mode === "speech" || mode === "fusion";
  const needsText = mode === "text" || mode === "fusion";

  audioSection.classList.toggle("hidden", !needsAudio);
  textSection.classList.toggle("hidden", !needsText);
  modeHelp.textContent = modeCopy[mode].help;
  emptyState.querySelector("span").textContent = modeCopy[mode].empty;
  resultHint.textContent = modeCopy[mode].empty;
  formError.textContent = "";
  clearResults();
}

function validatePrediction() {
  const mode = currentMode();
  const hasAudio = audioInput.files.length > 0;
  const hasText = textInput.value.trim().length > 0;

  if ((mode === "speech" || mode === "fusion") && !hasAudio) {
    return "Upload a WAV or AIFF audio file first.";
  }
  if ((mode === "text" || mode === "fusion") && !hasText) {
    return "Enter transcript text first.";
  }
  return "";
}

function clearResults() {
  resultGrid.innerHTML = "";
  emptyState.classList.remove("hidden");
}

function renderPrediction(payload) {
  const cards = [];
  if (payload.speech_prediction) cards.push(payload.speech_prediction);
  if (payload.text_prediction) cards.push(payload.text_prediction);
  if (payload.fusion_prediction) cards.push(payload.fusion_prediction);
  if (payload.model_variant) cards.push(payload);

  emptyState.classList.toggle("hidden", cards.length > 0);
  resultHint.textContent = cards.length
    ? "Prediction completed successfully."
    : "No prediction returned.";

  resultGrid.innerHTML = cards
    .map((card) => {
      const tone = resultTone[card.emotion] || "tone-neutral";
      return `
        <article class="result-card ${tone}">
          <span class="result-model">${prettyName(card.model_variant)}</span>
          <strong>${prettyName(card.emotion)}</strong>
          <span class="result-caption">${
            card.confidence ? `Confidence ${(card.confidence * 100).toFixed(1)}%` : "Detected emotion"
          }</span>
        </article>
      `;
    })
    .join("");
}

function buildFormData() {
  const mode = currentMode();
  const formData = new FormData();
  formData.append("mode", mode);
  if (mode === "text" || mode === "fusion") {
    formData.append("text", textInput.value.trim());
  }
  if ((mode === "speech" || mode === "fusion") && audioInput.files[0]) {
    formData.append("audio", audioInput.files[0]);
  }
  return formData;
}

predictionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.textContent = "";

  const validation = validatePrediction();
  if (validation) {
    formError.textContent = validation;
    return;
  }

  loadingPill.classList.remove("hidden");
  predictionForm.querySelector("button[type='submit']").disabled = true;
  try {
    const payload = await fetchJson("/api/predict/upload", {
      method: "POST",
      body: buildFormData(),
    });
    renderPrediction(payload);
  } catch (error) {
    formError.textContent = error.message;
  } finally {
    loadingPill.classList.add("hidden");
    predictionForm.querySelector("button[type='submit']").disabled = false;
  }
});

document.querySelectorAll("input[name='mode']").forEach((input) => {
  input.addEventListener("change", updatePredictionMode);
});

audioInput.addEventListener("change", () => {
  fileName.textContent = audioInput.files[0]?.name || "Choose or drop a WAV/AIFF file";
  formError.textContent = "";
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.add("drag-over");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.remove("drag-over");
  });
});

uploadZone.addEventListener("drop", (event) => {
  if (event.dataTransfer.files.length) {
    audioInput.files = event.dataTransfer.files;
    fileName.textContent = audioInput.files[0].name;
    formError.textContent = "";
  }
});

resetButton.addEventListener("click", () => {
  predictionForm.reset();
  fileName.textContent = "Choose or drop a WAV/AIFF file";
  formError.textContent = "";
  updatePredictionMode();
});

fetchJson("/health")
  .then((payload) => {
    const ready = payload.models?.speech && payload.models?.text && payload.models?.fusion;
    setBackendStatus(Boolean(ready));
  })
  .catch(() => setBackendStatus(false));

updatePredictionMode();
