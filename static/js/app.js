const API = {
  health: "/api/health",
  drugs: "/api/drugs",
  cellLines: "/api/cell-lines",
  predict: "/api/predict",
  explain: "/api/explain",
  moleculePair: "/api/molecule-pair",
  batch: "/api/batch-predict",
  chat: "/api/chat",
  demoCases: "/api/demo-cases",
  modelPerformance: "/api/model-performance-summary",
  download: "/api/download/"
};

const state = {
  shapChart: null,
  batchBlob: null,
  batchDownloadUrl: "",
  drugs: [],
  drugIds: new Set(),
  cellLines: [],
  cellLinesByModel: {
    classical: [],
    quantum: []
  },
  drugsLoaded: false,
  cellLinesLoaded: false,
  drugLoadError: "",
  cellLineLoadError: "",
  demos: [],
  lastPrediction: null,
  lastExplanation: null,
  predictionHistory: [],
  selectedBatchFile: null,
  lastBatchRendered: false,
  modelPerformance: null,
  projectChatBusy: false,
  projectChatTopic: "basics",
  projectChatExpanded: false,
  predictionChatBusy: false,
  projectChatDrag: null,
  projectChatResize: null,
  projectChatSuppressClick: false,
  inlineLoaders: {},
  gaugeDisplayScore: 0,
  gaugeInitialized: false,
  gaugeAnimationFrame: null,
  gaugeTicksRendered: false
};

const autocompleteTimers = {};
const THEME_STORAGE_KEY = "synergylens-theme";
const PREDICTION_HISTORY_KEY = "synergylens-recent-predictions";
const PROJECT_ASSISTANT_POSITION_KEY = "synergylens-project-assistant-position";
const PROJECT_ASSISTANT_SIZE_KEY = "synergylens-project-assistant-size";
const PROJECT_ASSISTANT_VIEWPORT_MARGIN = 16;
const MAX_PREDICTION_HISTORY = 5;
const SAFETY_NOTE_TEXT = "This is a machine-learning screening prediction. It is not biological proof and not clinical advice. Promising synergy predictions should be validated experimentally.";
const SAMPLE_BATCH_CSV_FILENAME = "batch_prediction_sample.csv";
const SAMPLE_BATCH_CSV_CONTENT = "NSC1,NSC2,CELLNAME\n740,750,786-0\n740,752,A498\n750,755,A549/ATCC";
const MODEL_LABELS = {
  classical: "Classical model",
  quantum: "Quantum model"
};
const PROJECT_CHAT_TOPICS = {
  basics: {
    label: "Basics",
    questions: [
      "What is SynergyLens?",
      "What is the purpose of this project?",
      "What dataset is used?",
      "What is NCI ALMANAC?",
      "What is ComboScore?",
      "What is NSC?",
      "How many drugs are available?",
      "How many cell lines are available?",
      "What are the main project limitations?"
    ]
  },
  prediction: {
    label: "Prediction",
    questions: [
      "How does prediction work?",
      "What inputs are required?",
      "What happens after I enter NSC1, NSC2, and CELLNAME?",
      "What does synergistic mean?",
      "What does antagonistic mean?",
      "What does neutral mean?",
      "Why predict both directions?",
      "What thresholds are used for labels?",
      "What model was used?",
      "How is the model selected?",
      "Can I trust the prediction?",
      "What should I check next after prediction?",
      "What should be validated experimentally?"
    ]
  },
  xai: {
    label: "XAI",
    questions: [
      "What is Explain AI?",
      "What is XAI?",
      "What is SHAP?",
      "How do I read XAI results?",
      "What do positive XAI values mean?",
      "What do negative XAI values mean?",
      "Can XAI prove the biological mechanism?",
      "Why are features named fingerprint feature or physicochemical feature?"
    ]
  },
  batch: {
    label: "Batch",
    questions: [
      "What is batch prediction?",
      "What CSV format is required?",
      "What columns are required in the batch CSV?",
      "How do I download batch results?",
      "Why did a batch row fail?",
      "What does successful/failed mean in batch results?"
    ]
  },
  molecules: {
    label: "Molecules",
    questions: [
      "What is Molecule Lookup?",
      "What molecule information is shown?",
      "What does direct molecule match mean?",
      "Why might molecule structure be unavailable?"
    ]
  },
  backend: {
    label: "Backend",
    questions: [
      "What files does the backend use?",
      "What feature files are used?",
      "What model files are used?",
      "What is the model registry?",
      "What is Step 6 model?",
      "Are models cell-line specific?",
      "What are 263 and 526 features?",
      "What is the data flow from frontend to backend?"
    ]
  },
  safety: {
    label: "Safety",
    questions: [
      "Is this clinical advice?",
      "Can this be used for treatment decisions?",
      "Can I trust the prediction?",
      "What should be validated experimentally?",
      "What does screening confidence mean?",
      "What are the limitations of ML predictions?"
    ]
  }
};
const PREDICTION_FLOW_MESSAGES = [
  "Validating inputs",
  "Building 526-feature vector",
  "Selecting best cell-line model",
  "Running NSC1 \u2192 NSC2 prediction",
  "Running NSC2 \u2192 NSC1 prediction",
  "Averaging ComboScore",
  "Preparing result"
];
const EXPLAIN_FLOW_MESSAGES = [
  "Validating prediction input",
  "Loading selected cell-line model",
  "Building 526-feature vector",
  "Computing XAI explanation",
  "Ranking top molecular contributors",
  "Preparing explainable AI story"
];
const BATCH_FLOW_MESSAGES = [
  "Reading uploaded CSV",
  "Validating required columns",
  "Checking drug and cell-line availability",
  "Running predictions row by row",
  "Saving batch output CSV",
  "Preparing download link"
];
const DRUG_INFO_FLOW_MESSAGES = [
  "Validating drug NSC values",
  "Resolving molecule aliases",
  "Loading molecule structures",
  "Generating RDKit SVG structures",
  "Preparing compound profile cards"
];
const DEMO_ORDER = ["synergistic", "neutral", "antagonistic", "strong_synergy", "antagonism"];
const DEMO_TITLES = {
  synergistic: "Synergistic",
  antagonistic: "Antagonistic",
  strong_synergy: "Strong Synergy",
  neutral: "Neutral / weak",
  antagonism: "Antagonism"
};
const VALIDATION_GROUPS = {
  predict: {
    buttonId: "predict-btn",
    drugFields: [
      { inputId: "drug1-input", hiddenId: "drug1-id", messageId: "drug1-validation" },
      { inputId: "drug2-input", hiddenId: "drug2-id", messageId: "drug2-validation" }
    ],
    cellField: { selectId: "cell-line", messageId: "cell-line-validation" }
  },
  explain: {
    buttonId: "explain-btn",
    drugFields: [
      { inputId: "edrug1-input", hiddenId: "edrug1-id", messageId: "edrug1-validation" },
      { inputId: "edrug2-input", hiddenId: "edrug2-id", messageId: "edrug2-validation" }
    ],
    cellField: { selectId: "ecell-line", messageId: "ecell-line-validation" }
  },
  drugs: {
    buttonId: "drug-info-btn",
    drugFields: [
      { inputId: "ddrug1-input", hiddenId: "ddrug1-id", messageId: "ddrug1-validation" },
      { inputId: "ddrug2-input", hiddenId: "ddrug2-id", messageId: "ddrug2-validation" }
    ]
  }
};
const NEUTRAL_THRESHOLD = 4;
const STRONG_EFFECT_THRESHOLD = 20;
const SCORE_DISPLAY_MIN = -60;
const SCORE_DISPLAY_MAX = 60;
const GAUGE_MAJOR_TICKS = [-60, -20, 0, 20, 60];
const GAUGE_MEDIUM_TICKS = [-40, -4, 4, 40];
const GAUGE_MINOR_TICKS = [-50, -30, -10, 10, 30, 50];
const GAUGE_GEOMETRY = {
  cx: 130,
  cy: 122,
  radius: 95,
  tickOuterRadius: 82,
  labelRadius: 50,
  startAngle: -180,
  endAngle: 0
};
const GAUGE_ANIMATION_DURATION = 1080;
const DEFAULT_VIEW = "home";
const VIEW_HASHES = {
  home: "home",
  predict: "predict",
  drugs: "molecules",
  explain: "explain-ai",
  batch: "batch",
  "prediction-chat": "prediction-assistant",
  about: "about"
};
const VIEW_ALIASES = {
  home: "home",
  predict: "predict",
  prediction: "predict",
  drugs: "drugs",
  drug: "drugs",
  molecule: "drugs",
  molecules: "drugs",
  "drug-info": "drugs",
  explain: "explain",
  explainai: "explain",
  "explain-ai": "explain",
  batch: "batch",
  assistant: "prediction-chat",
  "prediction-assistant": "prediction-chat",
  "prediction-chat": "prediction-chat",
  about: "about"
};

document.addEventListener("DOMContentLoaded", () => {
  bindThemeToggle();
  initializeGauge();
  document.body.classList.add("is-ready");
  bindViewNavigation();
  bindAutocomplete();
  bindLiveValidation();
  bindActions();
  bindBatchUpload();
  bindHistoryActions();
  bindProjectChat();
  bindPredictionChat();
  loadPredictionHistory();
  renderPredictionHistory();
  refreshAllValidation();
  bootstrapBackendData();
});

function bindViewNavigation() {
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.preventDefault();
      switchView(button.dataset.view);
    });
  });

  document.querySelectorAll("[data-view-trigger]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.preventDefault();
      switchView(button.dataset.viewTrigger);
    });
  });

  window.addEventListener("popstate", () => {
    switchView(viewFromHash(), { updateHash: false, scroll: true });
  });
  window.addEventListener("hashchange", () => {
    switchView(viewFromHash(), { updateHash: false, scroll: true });
  });

  switchView(viewFromHash(), { updateHash: false, scroll: false });
}

function bindThemeToggle() {
  const toggle = document.getElementById("switch");
  if (!toggle) {
    return;
  }

  const initialTheme = getPreferredTheme();
  applyTheme(initialTheme);
  toggle.checked = initialTheme === "dark";

  toggle.addEventListener("change", () => {
    applyTheme(toggle.checked ? "dark" : "light");
  });
}

function getPreferredTheme() {
  try {
    const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (storedTheme === "dark" || storedTheme === "light") {
      return storedTheme;
    }
  } catch (error) {
    // Local storage can be unavailable in private or restricted browser modes.
  }

  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme) {
  document.body.dataset.theme = theme;

  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch (error) {
    // Keep the current session theme even if persistence fails.
  }

  refreshChartTheme();
}

function switchView(viewName, options = {}) {
  const settings = {
    updateHash: true,
    scroll: true,
    ...options
  };
  const normalizedView = normalizeViewName(viewName) || DEFAULT_VIEW;

  document.querySelectorAll(".view").forEach((view) => {
    const isActive = view.id === `view-${normalizedView}`;
    view.classList.toggle("is-active", isActive);
    view.setAttribute("aria-hidden", String(!isActive));
  });

  document.querySelectorAll("[data-view]").forEach((button) => {
    const isActive = normalizeViewName(button.dataset.view) === normalizedView;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });

  document.body.dataset.activeView = normalizedView;

  if (settings.updateHash) {
    pushViewHash(normalizedView);
  }

  if (settings.scroll) {
    scrollToView(normalizedView);
  }
}

function normalizeViewName(viewName) {
  const cleaned = String(viewName || "")
    .replace(/^#/, "")
    .trim()
    .toLowerCase();
  return VIEW_ALIASES[cleaned] || "";
}

function viewFromHash() {
  return normalizeViewName(window.location.hash);
}

function pushViewHash(viewName) {
  const nextHash = `#${VIEW_HASHES[viewName] || viewName}`;
  if (window.location.hash === nextHash) {
    return;
  }
  window.history.pushState({ view: viewName }, "", nextHash);
}

function scrollToView(viewName) {
  const target = viewName === "home"
    ? document.querySelector(".hero")
    : document.querySelector(".workspace-tabs");
  if (!target) {
    return;
  }

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  window.requestAnimationFrame(() => {
    target.scrollIntoView({
      behavior: prefersReducedMotion ? "auto" : "smooth",
      block: "start"
    });
  });
}

function bootstrapBackendData() {
  loadHealth();
  loadDrugs();
  loadCellLines();

  deferOptionalLoad(() => {
    loadDemoCases();
    loadModelPerformance();
  });
}

function deferOptionalLoad(callback) {
  if ("requestIdleCallback" in window) {
    window.requestIdleCallback(callback, { timeout: 1200 });
    return;
  }
  window.setTimeout(callback, 250);
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json")
    ? await response.json().catch(() => ({}))
    : { error: await response.text().catch(() => "") };

  if (!response.ok || data.status === "error" || data.success === false) {
    throw new Error(cleanError(data.error || data.message || `Request failed with HTTP ${response.status}.`));
  }

  return data;
}

function cleanError(message) {
  if (message && typeof message === "object") {
    message = message.message || message.detail || JSON.stringify(message);
  }
  const text = String(message || "Backend request failed.");
  if (/traceback|stack trace/i.test(text)) {
    return "Backend returned an internal error. Check the Flask console for details.";
  }
  return text.length > 260 ? `${text.slice(0, 257)}...` : text;
}

async function loadHealth() {
  try {
    const health = await apiJson(API.health);
    renderBackendHealth(health);
  } catch (error) {
    renderBackendHealthError(error.message);
  }
}

function renderBackendHealth(health) {
  const availableDrugs = health.available_drugs ?? "--";
  const availableCellLines = health.available_cell_lines ?? "--";
  const availableModels = health.model_count ?? "--";
  const featureColumns = health.feature_column_count ?? "--";
  const isReady = health.status === "success";
  const errors = Array.isArray(health.errors) ? health.errors.filter(Boolean) : [];

  setText("metric-drugs", availableDrugs);
  setText("metric-cell-lines", availableCellLines);
  setText("metric-features", featureColumns);
  setText("metric-models", availableModels);
  setText("health-drugs", availableDrugs);
  setText("health-features", featureColumns);
  setText("backend-status-title", isReady ? "" : "Backend error");
  setText("backend-status", isReady ? "" : errors[0] || health.message || "Backend health check failed.");
  updateBackendHealthTone(isReady ? "success" : "danger", isReady ? "Ready" : "Error");
}

function renderBackendHealthError(message) {
  setText("backend-status-title", "Backend error");
  setText("backend-status", message || "Could not reach /api/health.");
  setText("metric-models", "--");
  setText("health-drugs", "--");
  setText("health-features", "--");
  updateBackendHealthTone("danger", "Error");
}

async function loadModelPerformance() {
  try {
    const data = await apiJson(API.modelPerformance);
    state.modelPerformance = data;
    renderModelPerformance(data);
  } catch (error) {
    renderModelPerformanceError(error.message);
  }
}

function renderModelPerformance(data) {
  const assets = data.assets || {};
  const modelSummary = data.model_summary || {};
  const performance = data.performance || {};
  const deployedAverage = performance.deployed_final_average || {};

  setText("performance-cell-lines", assets.total_cell_lines ?? "--");
  setText("performance-drugs", assets.total_drugs ?? "--");
  setText("performance-features", `${assets.feature_vector ?? "--"} features`);
  setText("performance-models", assets.final_model_count ?? modelSummary.total_models ?? "--");
  setText("model-performance-explanation", data.explanation || "Model performance summary loaded from project result files.");
  setText("performance-average-summary", formatPerformanceSummary(deployedAverage));

  const status = document.getElementById("model-performance-status");
  if (status) {
    status.textContent = "Loaded";
    setToneClass(status, "success");
  }

  renderModelTypeCounts(modelSummary.count_per_model_type || {});
  renderPerformanceRows(performance.by_model_type || []);
}

function renderModelPerformanceError(message) {
  setText("model-performance-explanation", "Model performance summary could not be loaded.");
  setText("performance-average-summary", cleanError(message));
  const status = document.getElementById("model-performance-status");
  if (status) {
    status.textContent = "Unavailable";
    setToneClass(status, "danger");
  }

  const modelCounts = document.getElementById("model-type-counts");
  if (modelCounts) {
    modelCounts.innerHTML = `<span class="model-count-empty">${escapeHtml(cleanError(message))}</span>`;
  }

  const tableBody = document.getElementById("performance-table-body");
  if (tableBody) {
    tableBody.innerHTML = `<tr><td colspan="5">${escapeHtml(cleanError(message))}</td></tr>`;
  }
}

function renderModelTypeCounts(counts) {
  const container = document.getElementById("model-type-counts");
  if (!container) {
    return;
  }

  const preferredOrder = ["CatBoost", "LightGBM", "RandomForest", "XGBoost"];
  const entries = Object.entries(counts).sort((a, b) => {
    const aIndex = preferredOrder.indexOf(a[0]);
    const bIndex = preferredOrder.indexOf(b[0]);
    return (aIndex === -1 ? 99 : aIndex) - (bIndex === -1 ? 99 : bIndex);
  });

  if (!entries.length) {
    container.innerHTML = `<span class="model-count-empty">No model count data returned.</span>`;
    return;
  }

  container.innerHTML = entries.map(([model, count]) => `
    <div class="model-count-row">
      <span>${escapeHtml(model)}</span>
      <strong>${escapeHtml(String(count))}</strong>
    </div>
  `).join("");
}

function renderPerformanceRows(rows) {
  const tableBody = document.getElementById("performance-table-body");
  if (!tableBody) {
    return;
  }

  if (!rows.length) {
    tableBody.innerHTML = `<tr><td colspan="5">Average performance metrics are unavailable.</td></tr>`;
    return;
  }

  tableBody.innerHTML = rows.map((row) => `
    <tr>
      <td>${escapeHtml(row.model || "")}</td>
      <td class="mono">${escapeHtml(formatMetric(row.mean_r2_score, 3))}</td>
      <td class="mono">${escapeHtml(formatMetric(row.mean_pearson_rp, 3))}</td>
      <td class="mono">${escapeHtml(formatMetric(row.mean_rmse, 2))}</td>
      <td class="mono">${escapeHtml(formatMetric(row.mean_mae, 2))}</td>
    </tr>
  `).join("");
}

function formatPerformanceSummary(summary) {
  if (!summary || !Number.isFinite(Number(summary.mean_r2_score))) {
    return "Average performance metrics are loaded from saved project result files when available.";
  }

  return [
    `Final deployed models average R2 ${formatMetric(summary.mean_r2_score, 3)}`,
    `Pearson Rp ${formatMetric(summary.mean_pearson_rp, 3)}`,
    `RMSE ${formatMetric(summary.mean_rmse, 2)}`,
    `MAE ${formatMetric(summary.mean_mae, 2)}`
  ].join(" | ");
}

function formatMetric(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "n/a";
}

function updateBackendHealthTone(level, badgeText) {
  const card = document.getElementById("backend-health-card");
  const badge = document.getElementById("backend-status-badge");

  if (card) {
    card.classList.remove("is-ready", "is-error");
    card.classList.add(level === "success" ? "is-ready" : "is-error");
  }

  if (badge) {
    setToneClass(badge, level === "success" ? "success" : "danger");
    badge.textContent = badgeText;
  }
}

async function loadDrugs() {
  try {
    const data = await apiJson(`${API.drugs}?limit=all`);
    state.drugs = normalizeDrugList(data);
    state.drugIds = new Set(state.drugs.map((drug) => drug.id));
    state.drugsLoaded = true;
    state.drugLoadError = "";
    refreshOpenAutocompleteLists();
    if (state.drugs.length && document.getElementById("metric-drugs")?.textContent === "--") {
      setText("metric-drugs", state.drugs.length);
    }
  } catch (error) {
    state.drugsLoaded = false;
    state.drugLoadError = error.message;
    showAlert("predict-alert", `Drug list failed to load: ${error.message}`);
  } finally {
    refreshAllValidation();
  }
}

function normalizeDrugList(data) {
  const values = Array.isArray(data) ? data : data.drugs || data.items || [];
  return values
    .map((raw) => {
      if (typeof raw === "number" || typeof raw === "string") {
        const id = String(raw).trim();
        return { id, name: `NSC ${id}`, displayLabel: `NSC ${id}` };
      }

      const id = String(raw.id ?? raw.NSC ?? raw.nsc ?? raw.drug_id ?? raw.DRUG_ID ?? "").trim();
      const rawName = raw.name ?? raw.drug_name ?? raw.NAME ?? raw.id ?? "";
      const name = String(rawName || `NSC ${id}`).trim();
      const cleanName = /^\d+$/.test(name) ? `NSC ${name}` : name;
      const displayLabel = String(raw.label || `${cleanName} — NSC ${id}`).trim();
      return { id, name: cleanName, displayLabel };
    })
    .filter((drug) => drug.id);
}

async function loadCellLines(modelType = getSelectedModelType()) {
  const normalizedModelType = modelType === "quantum" ? "quantum" : "classical";
  const cached = state.cellLinesByModel[normalizedModelType] || [];
  if (cached.length) {
    applyCellLineList(cached, normalizedModelType);
    refreshAllValidation();
    return;
  }

  state.cellLinesLoaded = false;
  state.cellLineLoadError = "";
  fillCellLineSelect("cell-line", [], "Loading cell lines...");
  fillCellLineSelect("ecell-line", [], "Loading cell lines...");
  refreshAllValidation();

  try {
    const data = await apiJson(`${API.cellLines}?model_type=${encodeURIComponent(normalizedModelType)}`);
    const cellLines = normalizeCellLineList(data);
    state.cellLinesByModel[normalizedModelType] = cellLines;
    if (getSelectedModelType() === normalizedModelType) {
      applyCellLineList(cellLines, normalizedModelType);
    }
  } catch (error) {
    state.cellLinesLoaded = false;
    state.cellLineLoadError = error.message;
    fillCellLineSelect("cell-line", [], "Cell lines unavailable");
    fillCellLineSelect("ecell-line", [], "Cell lines unavailable");
    showAlert("predict-alert", `Cell-line list failed to load: ${error.message}`);
  } finally {
    refreshAllValidation();
  }
}

function applyCellLineList(cellLines, modelType) {
  state.cellLines = cellLines;
  state.cellLinesLoaded = true;
  state.cellLineLoadError = "";
  fillCellLineSelect("cell-line", state.cellLines);
  fillCellLineSelect("ecell-line", state.cellLines);
  if (modelType === "classical" && state.cellLines.length && document.getElementById("metric-cell-lines")?.textContent === "--") {
    setText("metric-cell-lines", state.cellLines.length);
  }
}

function normalizeCellLineList(data) {
  const values = Array.isArray(data) ? data : data.cell_lines || data.items || [];
  return values
    .map((value) => String(value ?? "").trim())
    .filter(Boolean);
}

function fillCellLineSelect(id, values, placeholder = "Select a cell line") {
  const select = document.getElementById(id);
  if (!select) {
    return;
  }

  const previousValue = select.value;
  select.innerHTML = [`<option value="">${escapeHtml(placeholder)}</option>`]
    .concat(values.map((value) => `<option value="${escapeAttribute(value)}">${escapeHtml(value)}</option>`))
    .join("");
  if (previousValue && values.includes(previousValue)) {
    select.value = previousValue;
  }
}

async function loadDemoCases() {
  try {
    const data = await apiJson(API.demoCases);
    const cases = normalizeDemoCases(data.demo_cases);
    state.demos = sortDemoCases(cases);
    renderDemoCases(state.demos);
    if (cases.length) {
      fillDemoCase(state.demos[0], { preserveExisting: true });
    }
  } catch (error) {
    renderDemoError(error.message);
  }
}

function normalizeDemoCases(rawCases) {
  if (Array.isArray(rawCases)) {
    return rawCases;
  }
  if (rawCases && typeof rawCases === "object") {
    return Object.values(rawCases);
  }
  return [];
}

function sortDemoCases(cases) {
  return [...cases].sort((a, b) => {
    const aIndex = DEMO_ORDER.indexOf(a.case_type);
    const bIndex = DEMO_ORDER.indexOf(b.case_type);
    return (aIndex === -1 ? 99 : aIndex) - (bIndex === -1 ? 99 : bIndex);
  });
}

function renderDemoCases(cases) {
  const container = document.getElementById("demo-cases");
  if (!container) {
    return;
  }

  if (!cases.length) {
    container.innerHTML = `<div class="demo-loading">No demo cases returned by the backend.</div>`;
    return;
  }

  container.innerHTML = cases.map((demo, index) => {
    const title = DEMO_TITLES[demo.case_type] || demo.case_type || `Demo ${index + 1}`;
    return `
      <button class="demo-case-card" type="button" data-demo-index="${index}">
        <strong>${escapeHtml(title)}</strong>
        <span>NSC ${escapeHtml(String(demo.NSC1))} + NSC ${escapeHtml(String(demo.NSC2))}</span>
        <span>${escapeHtml(String(demo.CELLNAME))}</span>
      </button>
    `;
  }).join("");

  container.querySelectorAll("[data-demo-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const demo = state.demos[Number(button.dataset.demoIndex)];
      if (!demo) {
        return;
      }
      fillDemoCase(demo);
      container.querySelectorAll(".demo-case-card").forEach((card) => card.classList.remove("is-selected"));
      button.classList.add("is-selected");
    });
  });
}

function renderDemoError(message) {
  const container = document.getElementById("demo-cases");
  if (container) {
    container.innerHTML = `<div class="demo-loading">Demo cases unavailable: ${escapeHtml(message)}</div>`;
  }
}

function fillDemoCase(demo, options = {}) {
  const preserveExisting = Boolean(options.preserveExisting);
  const nsc1 = resolveKnownDrugId(demo.NSC1) || String(demo.NSC1 ?? "").trim();
  const nsc2 = resolveKnownDrugId(demo.NSC2) || String(demo.NSC2 ?? "").trim();
  const fields = [
    ["drug1-input", "drug1-id", nsc1],
    ["drug2-input", "drug2-id", nsc2],
    ["edrug1-input", "edrug1-id", nsc1],
    ["edrug2-input", "edrug2-id", nsc2],
    ["ddrug1-input", "ddrug1-id", nsc1],
    ["ddrug2-input", "ddrug2-id", nsc2]
  ];

  fields.forEach(([inputId, hiddenId, value]) => {
    const input = document.getElementById(inputId);
    const hidden = document.getElementById(hiddenId);
    if (input && (!preserveExisting || !input.value) && value !== undefined && value !== null) {
      input.value = `NSC ${value}`;
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
    if (hidden && (!preserveExisting || !hidden.value) && value !== undefined && value !== null) {
      hidden.value = String(value);
    }
  });

  setSelectValueWhenAvailable("cell-line", demo.CELLNAME, preserveExisting);
  setSelectValueWhenAvailable("ecell-line", demo.CELLNAME, preserveExisting);
  clearAlert("predict-alert");
  refreshAllValidation();
  window.setTimeout(refreshAllValidation, 300);
  if (!preserveExisting) {
    switchView("predict");
    document.getElementById("predict-btn")?.focus();
  }
}

function setSelectValueWhenAvailable(id, value, preserveExisting = false) {
  if (!value) {
    return;
  }

  const select = document.getElementById(id);
  if (!select) {
    return;
  }

  const apply = () => {
    if (preserveExisting && select.value) {
      return;
    }
    if (Array.from(select.options).some((option) => option.value === value)) {
      select.value = value;
      select.dispatchEvent(new Event("change", { bubbles: true }));
    }
  };

  apply();
  window.setTimeout(apply, 250);
}

function bindHistoryActions() {
  const list = document.getElementById("history-list");
  if (list) {
    list.addEventListener("click", (event) => {
      const reloadButton = event.target.closest("[data-history-index]");
      if (!reloadButton) {
        return;
      }
      const index = Number(reloadButton.dataset.historyIndex);
      reloadHistoryItem(index);
    });
  }

  document.getElementById("clear-history-btn")?.addEventListener("click", clearPredictionHistory);
}

function loadPredictionHistory() {
  let parsed = [];
  try {
    const raw = window.localStorage.getItem(PREDICTION_HISTORY_KEY);
    parsed = raw ? JSON.parse(raw) : [];
  } catch (error) {
    parsed = [];
  }

  state.predictionHistory = Array.isArray(parsed)
    ? parsed.map(normalizeHistoryRecord).filter(Boolean).slice(0, MAX_PREDICTION_HISTORY)
    : [];
  persistPredictionHistory();
}

function normalizeHistoryRecord(record) {
  if (!record || typeof record !== "object") {
    return null;
  }

  const nsc1 = String(record.NSC1 ?? record.nsc1 ?? "").trim();
  const nsc2 = String(record.NSC2 ?? record.nsc2 ?? "").trim();
  const cellLine = String(record.CELLNAME ?? record.cellLine ?? record.cell_line ?? "").trim();
  const score = Number(
    record.final_predicted_COMBOSCORE ??
    record.score ??
    record.predicted_comboscore
  );

  if (!nsc1 || !nsc2 || !cellLine || !Number.isFinite(score)) {
    return null;
  }

  const timestamp = Number.isFinite(Date.parse(record.timestamp))
    ? new Date(record.timestamp).toISOString()
    : new Date().toISOString();

  return {
    NSC1: nsc1,
    NSC2: nsc2,
    CELLNAME: cellLine,
    model_used: String(record.model_used ?? record.model ?? "").trim(),
    model_type: String(record.model_type ?? record.modelType ?? "classical").trim().toLowerCase() === "quantum" ? "quantum" : "classical",
    final_predicted_COMBOSCORE: score,
    label: semanticLabelForScore(score),
    timestamp
  };
}

function savePredictionHistory(prediction) {
  const record = normalizeHistoryRecord({
    NSC1: prediction.nsc1,
    NSC2: prediction.nsc2,
    CELLNAME: prediction.cellLine,
    model_used: prediction.model,
    model_type: prediction.modelType || "classical",
    final_predicted_COMBOSCORE: prediction.score,
    label: semanticLabelForScore(prediction.score),
    timestamp: new Date().toISOString()
  });

  if (!record) {
    return;
  }

  state.predictionHistory = [record]
    .concat(state.predictionHistory)
    .slice(0, MAX_PREDICTION_HISTORY);
  persistPredictionHistory();
  renderPredictionHistory();
}

function persistPredictionHistory() {
  try {
    window.localStorage.setItem(PREDICTION_HISTORY_KEY, JSON.stringify(state.predictionHistory));
  } catch (error) {
    // History is a convenience feature; prediction should still work if storage is blocked.
  }
}

function renderPredictionHistory() {
  const list = document.getElementById("history-list");
  const clearButton = document.getElementById("clear-history-btn");
  if (!list) {
    return;
  }

  const records = state.predictionHistory
    .map(normalizeHistoryRecord)
    .filter(Boolean)
    .slice(0, MAX_PREDICTION_HISTORY);
  state.predictionHistory = records;

  if (clearButton) {
    clearButton.disabled = records.length === 0;
  }

  if (!records.length) {
    list.innerHTML = `<div class="history-empty">Successful predictions will appear here.</div>`;
    return;
  }

  list.innerHTML = records.map((record, index) => {
    const label = semanticLabelForScore(record.final_predicted_COMBOSCORE);
    const tone = historyToneForLabel(label);
    const labelClass = label.toLowerCase().replace(/[^a-z0-9_-]+/g, "-");
    return `
      <article class="history-item">
        <div class="history-item-top">
          <strong>NSC ${escapeHtml(record.NSC1)} + NSC ${escapeHtml(record.NSC2)}</strong>
          <span class="history-label history-label--${escapeAttribute(labelClass)}">${escapeHtml(label)}</span>
        </div>
        <div class="history-meta">
          <span>${escapeHtml(record.CELLNAME)}</span>
          <span>Score ${escapeHtml(formatScore(record.final_predicted_COMBOSCORE, 3))}</span>
          <span>${escapeHtml(record.model_used || MODEL_LABELS[record.model_type] || "Auto model")}</span>
          <span>${escapeHtml(formatHistoryTimestamp(record.timestamp))}</span>
        </div>
        <button type="button" class="history-reload-button ${escapeAttribute(tone)}" data-history-index="${index}">Reload</button>
      </article>
    `;
  }).join("");
}

function reloadHistoryItem(index) {
  const record = normalizeHistoryRecord(state.predictionHistory[index]);
  if (!record) {
    renderPredictionHistory();
    return;
  }

  setDrugFieldValue("drug1-input", "drug1-id", record.NSC1);
  setDrugFieldValue("drug2-input", "drug2-id", record.NSC2);
  const modelSelect = document.getElementById("model-type");
  if (modelSelect) {
    modelSelect.value = record.model_type;
  }
  loadCellLines(record.model_type).then(() => {
    setSelectValueWhenAvailable("cell-line", record.CELLNAME, false);
    refreshAllValidation();
  });
  clearAlert("predict-alert");
  switchView("predict");
  refreshAllValidation();
  window.setTimeout(refreshAllValidation, 300);
  document.getElementById("predict-btn")?.focus();
}

function setDrugFieldValue(inputId, hiddenId, nsc) {
  const value = String(nsc ?? "").trim();
  const input = document.getElementById(inputId);
  const hidden = document.getElementById(hiddenId);
  if (input) {
    input.value = value ? `NSC ${value}` : "";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }
  if (hidden) {
    hidden.value = value;
  }
}

function clearPredictionHistory() {
  state.predictionHistory = [];
  persistPredictionHistory();
  renderPredictionHistory();
}

function historyToneForLabel(label) {
  const normalized = String(label || "").toLowerCase();
  if (normalized.includes("synerg")) {
    return "history-reload-button--synergy";
  }
  if (normalized.includes("antag")) {
    return "history-reload-button--antagonism";
  }
  return "history-reload-button--neutral";
}

function formatHistoryTimestamp(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "recently";
  }
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(date);
}

function bindAutocomplete() {
  document.querySelectorAll("[data-drug-search]").forEach((input) => {
    input.addEventListener("input", () => handleDrugSearch(input));
    input.addEventListener("focus", () => handleDrugSearch(input));
    input.addEventListener("blur", () => syncDrugInputWithKnownValue(input));
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".autocomplete-wrap")) {
      closeAllAutocomplete();
    }
  });
}

function handleDrugSearch(input) {
  const query = input.value.trim();
  const listId = input.dataset.listId;
  const hiddenId = input.dataset.hiddenId;
  const list = document.getElementById(listId);
  const hidden = document.getElementById(hiddenId);

  if (!list || !hidden) {
    return;
  }

  hidden.value = "";
  clearTimeout(autocompleteTimers[listId]);

  autocompleteTimers[listId] = window.setTimeout(() => {
    if (!state.drugs.length) {
      renderAutocompleteMessage(list, "Loading valid NSC IDs...");
      return;
    }
    const matches = query.length ? searchDrugs(query) : state.drugs;
    renderDrugAutocomplete(list, input, hidden, matches);
  }, 160);
}

function searchDrugs(query) {
  const normalized = query.toLowerCase().replace(/^nsc\s*/, "").trim();
  return state.drugs.filter((drug) => (
    drug.id.includes(normalized) ||
    drug.name.toLowerCase().includes(query.toLowerCase()) ||
    drug.displayLabel.toLowerCase().includes(query.toLowerCase())
  ));
}

function refreshOpenAutocompleteLists() {
  document.querySelectorAll("[data-drug-search]").forEach((input) => {
    const list = document.getElementById(input.dataset.listId);
    if (list?.classList.contains("is-open")) {
      handleDrugSearch(input);
    }
  });
}

function renderAutocompleteMessage(list, message) {
  list.innerHTML = `<div class="autocomplete-message">${escapeHtml(message)}</div>`;
  setAutocompleteOpen(list, true);
}

function renderDrugAutocomplete(list, input, hidden, matches) {
  list.innerHTML = "";

  if (!matches.length) {
    setAutocompleteOpen(list, false);
    return;
  }

  matches.slice(0, 12).forEach((drug) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "autocomplete-item";
    item.setAttribute("role", "option");
    item.innerHTML = `
      <span>${escapeHtml(drug.name)}</span>
      <span class="autocomplete-id">NSC ${escapeHtml(drug.id)}</span>
    `;
    item.addEventListener("click", () => {
      input.value = drug.displayLabel;
      hidden.value = drug.id;
      setAutocompleteOpen(list, false);
      input.dispatchEvent(new Event("change", { bubbles: true }));
    });
    list.appendChild(item);
  });

  setAutocompleteOpen(list, true);
}

function setAutocompleteOpen(list, isOpen) {
  list.classList.toggle("is-open", isOpen);
  const input = document.querySelector(`[data-list-id="${list.id}"]`);
  if (input) {
    input.setAttribute("aria-expanded", String(isOpen));
  }
}

function syncDrugInputWithKnownValue(input) {
  window.setTimeout(() => {
    const hidden = document.getElementById(input.dataset.hiddenId);
    if (!hidden || hidden.value) {
      return;
    }
    const resolvedId = resolveKnownDrugId(input.value);
    if (resolvedId) {
      hidden.value = resolvedId;
      refreshAllValidation();
    }
  }, 120);
}

function closeAllAutocomplete() {
  document.querySelectorAll(".autocomplete-list").forEach((list) => {
    setAutocompleteOpen(list, false);
  });
}

function bindLiveValidation() {
  document.querySelectorAll("[data-drug-search]").forEach((input) => {
    input.addEventListener("input", refreshAllValidation);
    input.addEventListener("change", refreshAllValidation);
    input.addEventListener("blur", () => window.setTimeout(refreshAllValidation, 140));
  });

  ["cell-line", "ecell-line"].forEach((id) => {
    const select = document.getElementById(id);
    if (select) {
      select.addEventListener("change", refreshAllValidation);
      select.addEventListener("input", refreshAllValidation);
    }
  });

  document.getElementById("model-type")?.addEventListener("change", () => {
    loadCellLines(getSelectedModelType());
  });
}

function refreshAllValidation() {
  const results = {};
  Object.entries(VALIDATION_GROUPS).forEach(([name, config]) => {
    const groupResult = validateGroup(config);
    results[name] = groupResult.valid;
    setActionAvailability(config.buttonId, groupResult.valid);
  });
  return results;
}

function validateGroup(config) {
  const drugResults = config.drugFields.map((field) => validateDrugField(field));
  const cellResult = config.cellField ? validateCellLineField(config.cellField) : { valid: true };
  return {
    valid: drugResults.every((result) => result.valid) && cellResult.valid,
    drugResults,
    cellResult
  };
}

function validateDrugField({ inputId, hiddenId, messageId }) {
  const input = document.getElementById(inputId);
  const hidden = document.getElementById(hiddenId);
  const rawValue = input?.value?.trim() || "";

  if (!state.drugsLoaded) {
    return setValidationState(input, messageId, false, "neutral", state.drugLoadError ? "Drug list unavailable" : "Loading valid drugs...");
  }

  if (!rawValue) {
    if (hidden) {
      hidden.value = "";
    }
    return setValidationState(input, messageId, false, "neutral", "Required field");
  }

  const resolvedId = resolveKnownDrugId(rawValue);
  if (!resolvedId) {
    if (hidden) {
      hidden.value = "";
    }
    return setValidationState(input, messageId, false, "invalid", "Unknown drug");
  }

  if (hidden) {
    hidden.value = resolvedId;
  }
  return setValidationState(input, messageId, true, "valid", "Available drug");
}

function validateCellLineField({ selectId, messageId }) {
  const select = document.getElementById(selectId);
  const value = select?.value?.trim() || "";

  if (!state.cellLinesLoaded) {
    return setValidationState(select, messageId, false, "neutral", state.cellLineLoadError ? "Cell-line list unavailable" : "Loading valid cell lines...");
  }

  if (!value) {
    return setValidationState(select, messageId, false, "neutral", "Required field");
  }

  if (!state.cellLines.includes(value)) {
    return setValidationState(select, messageId, false, "invalid", "Unknown cell line");
  }

  return setValidationState(select, messageId, true, "valid", "Available cell line");
}

function setValidationState(input, messageId, isValid, tone, message) {
  const fieldGroup = input?.closest(".field-group");
  if (fieldGroup) {
    fieldGroup.classList.remove("is-valid", "is-invalid", "is-neutral");
    fieldGroup.classList.add(isValid ? "is-valid" : tone === "invalid" ? "is-invalid" : "is-neutral");
  }

  const messageElement = document.getElementById(messageId);
  if (messageElement) {
    messageElement.textContent = message;
    messageElement.classList.remove("is-valid", "is-invalid", "is-neutral");
    messageElement.classList.add(`is-${tone}`);
  }

  return { valid: isValid, message, tone };
}

function setActionAvailability(buttonId, isEnabled) {
  const button = document.getElementById(buttonId);
  if (!button || button.classList.contains("is-loading")) {
    return;
  }
  button.disabled = !isEnabled;
}

function bindActions() {
  document.getElementById("predict-btn")?.addEventListener("click", runPredict);
  document.getElementById("explain-btn")?.addEventListener("click", runExplain);
  document.getElementById("drug-info-btn")?.addEventListener("click", loadDrugInfo);
  document.getElementById("batch-run-btn")?.addEventListener("click", runSelectedBatchFile);
  document.getElementById("sample-csv-btn")?.addEventListener("click", downloadSampleBatchCsv);
  document.getElementById("download-btn")?.addEventListener("click", downloadBatchResults);
  document.getElementById("prediction-report-btn")?.addEventListener("click", downloadPredictionReport);
  document.getElementById("open-prediction-assistant-btn")?.addEventListener("click", () => switchView("prediction-chat"));
}

function bindBatchUpload() {
  const zone = document.getElementById("drop-zone");
  const input = document.getElementById("csv-file");
  if (!zone || !input) {
    return;
  }

  bindBatchTableScroll();
  window.addEventListener("resize", requestBatchScrollRefresh, { passive: true });

  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    zone.classList.add("is-dragging");
  });

  zone.addEventListener("dragleave", () => {
    zone.classList.remove("is-dragging");
  });

  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("is-dragging");
    const file = event.dataTransfer.files[0];
    if (file) {
      stageBatchFile(file);
    }
  });

  input.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
      stageBatchFile(file);
    }
  });
}

function bindBatchTableScroll() {
  const container = document.getElementById("batch-table-scroll") || document.querySelector(".batch-table-scroll");
  if (!container) {
    return;
  }

  if (container.dataset.batchScrollBound !== "true") {
    container.dataset.batchScrollBound = "true";
    container.addEventListener("wheel", handleBatchTableWheel, { passive: false });
  }

  updateBatchScrollHint(container);
}

function requestBatchScrollRefresh() {
  bindBatchTableScroll();
  window.requestAnimationFrame(() => updateBatchScrollHint());
}

function updateBatchScrollHint(container = document.getElementById("batch-table-scroll")) {
  const hint = document.getElementById("batch-scroll-hint");
  if (!container || !hint) {
    return;
  }

  const canScroll = container.scrollWidth > container.clientWidth + 1;
  hint.hidden = !canScroll;
}

function handleBatchTableWheel(event) {
  const container = event.currentTarget;
  if (!container || container.scrollWidth <= container.clientWidth + 1) {
    return;
  }

  const horizontalIntent = Math.abs(event.deltaX) > Math.abs(event.deltaY);
  let delta = horizontalIntent ? event.deltaX : 0;
  if (!delta && event.shiftKey) {
    delta = event.deltaY || event.deltaX;
  }
  if (!delta && !event.ctrlKey && Math.abs(event.deltaY) > 0) {
    delta = event.deltaY;
  }
  if (!delta) {
    return;
  }

  const maxScroll = container.scrollWidth - container.clientWidth;
  const nextScroll = Math.max(0, Math.min(maxScroll, container.scrollLeft + delta));
  if (nextScroll === container.scrollLeft) {
    return;
  }

  container.scrollLeft = nextScroll;
  event.preventDefault();
}

function bindProjectChat() {
  const toggle = document.getElementById("project-chat-toggle");
  const close = document.getElementById("project-chat-close");
  const form = document.getElementById("project-chat-form");
  const suggestions = document.getElementById("project-chat-suggestions");
  const panel = document.getElementById("project-chat-panel");

  loadProjectAssistantSize();
  loadProjectAssistantPosition();
  bindProjectAssistantDrag(toggle);
  bindProjectAssistantResize(panel);
  window.addEventListener("resize", () => {
    applyProjectAssistantSize(loadStoredProjectAssistantSize(), { persistCorrected: true });
    applyProjectAssistantPosition(loadStoredProjectAssistantPosition(), { persistCorrected: true });
  });

  toggle?.addEventListener("click", (event) => {
    if (state.projectChatSuppressClick) {
      event.preventDefault();
      state.projectChatSuppressClick = false;
      return;
    }
    toggleProjectChat();
  });
  close?.addEventListener("click", () => toggleProjectChat(false));

  form?.addEventListener("submit", (event) => {
    event.preventDefault();
    sendProjectChatMessage();
  });

  suggestions?.addEventListener("click", (event) => {
    const topicButton = event.target.closest("[data-chat-topic]");
    if (topicButton) {
      state.projectChatTopic = topicButton.dataset.chatTopic;
      state.projectChatExpanded = false;
      renderProjectChatSuggestions();
      return;
    }

    const moreButton = event.target.closest("[data-chat-more]");
    if (moreButton) {
      state.projectChatExpanded = !state.projectChatExpanded;
      renderProjectChatSuggestions();
      return;
    }

    const button = event.target.closest("[data-chat-question]");
    if (!button) {
      return;
    }
    sendProjectChatMessage(button.dataset.chatQuestion);
  });

  renderProjectChatSuggestions();
}

function bindProjectAssistantDrag(handle) {
  const widget = document.querySelector(".project-chat-widget");
  if (!handle || !widget) {
    return;
  }

  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== undefined && event.button !== 0) {
      return;
    }

    const rect = widget.getBoundingClientRect();
    state.projectChatDrag = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startLeft: rect.left,
      startTop: rect.top,
      moved: false
    };
    try {
      handle.setPointerCapture?.(event.pointerId);
    } catch (error) {
      // Synthetic pointer events used by automated checks may not be capturable.
    }
  });

  handle.addEventListener("pointermove", (event) => {
    const drag = state.projectChatDrag;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }

    const deltaX = event.clientX - drag.startX;
    const deltaY = event.clientY - drag.startY;
    if (!drag.moved && Math.hypot(deltaX, deltaY) < 6) {
      return;
    }

    drag.moved = true;
    state.projectChatSuppressClick = true;
    event.preventDefault();
    applyProjectAssistantPosition({
      left: drag.startLeft + deltaX,
      top: drag.startTop + deltaY
    });
  });

  const finishDrag = (event) => {
    const drag = state.projectChatDrag;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }

    const rect = widget.getBoundingClientRect();
    if (drag.moved) {
      saveProjectAssistantPosition({ left: rect.left, top: rect.top });
    }
    state.projectChatDrag = null;
    try {
      handle.releasePointerCapture?.(event.pointerId);
    } catch (error) {
      // Ignore release failures for synthetic or already-ended pointer captures.
    }
  };

  handle.addEventListener("pointerup", finishDrag);
  handle.addEventListener("pointercancel", finishDrag);
}

function bindProjectAssistantResize(panel) {
  if (!panel) {
    return;
  }

  panel.querySelectorAll("[data-chat-resize]").forEach((handle) => {
    handle.addEventListener("pointerdown", (event) => {
      if (event.button !== undefined && event.button !== 0) {
        return;
      }

      const rect = panel.getBoundingClientRect();
      state.projectChatResize = {
        pointerId: event.pointerId,
        direction: handle.dataset.chatResize || "corner",
        startX: event.clientX,
        startY: event.clientY,
        startWidth: rect.width,
        startHeight: rect.height,
        startLeft: rect.left,
        startTop: rect.top
      };
      event.preventDefault();
      event.stopPropagation();
      try {
        handle.setPointerCapture?.(event.pointerId);
      } catch (error) {
        // Synthetic pointer events used by automated checks may not be capturable.
      }
    });

    handle.addEventListener("pointermove", (event) => {
      const resize = state.projectChatResize;
      if (!resize || resize.pointerId !== event.pointerId) {
        return;
      }

      const deltaX = event.clientX - resize.startX;
      const deltaY = event.clientY - resize.startY;
      const direction = resize.direction;
      const nextSize = {
        width: direction === "right" || direction === "corner"
          ? resize.startWidth + deltaX
          : resize.startWidth,
        height: direction === "bottom" || direction === "corner"
          ? resize.startHeight + deltaY
          : resize.startHeight
      };

      event.preventDefault();
      applyProjectAssistantSize(nextSize);
      applyProjectAssistantPosition({
        left: resize.startLeft,
        top: resize.startTop
      });
    });

    const finishResize = (event) => {
      const resize = state.projectChatResize;
      if (!resize || resize.pointerId !== event.pointerId) {
        return;
      }

      const rect = panel.getBoundingClientRect();
      saveProjectAssistantSize({ width: rect.width, height: rect.height });
      saveProjectAssistantPosition({ left: rect.left, top: rect.top });
      state.projectChatResize = null;
      try {
        handle.releasePointerCapture?.(event.pointerId);
      } catch (error) {
        // Ignore release failures for synthetic or already-ended pointer captures.
      }
    };

    handle.addEventListener("pointerup", finishResize);
    handle.addEventListener("pointercancel", finishResize);
  });
}

function loadProjectAssistantSize() {
  applyProjectAssistantSize(loadStoredProjectAssistantSize(), { persistCorrected: true });
}

function loadStoredProjectAssistantSize() {
  try {
    const raw = window.localStorage.getItem(PROJECT_ASSISTANT_SIZE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (error) {
    return null;
  }
}

function saveProjectAssistantSize(size) {
  try {
    window.localStorage.setItem(PROJECT_ASSISTANT_SIZE_KEY, JSON.stringify(size));
  } catch (error) {
    // Size persistence is optional.
  }
}

function projectAssistantSizeLimits() {
  const margin = PROJECT_ASSISTANT_VIEWPORT_MARGIN;
  const panel = document.getElementById("project-chat-panel");
  const widget = document.querySelector(".project-chat-widget");
  const panelRect = panel && !panel.hidden ? panel.getBoundingClientRect() : null;
  const widgetRect = widget?.getBoundingClientRect();
  const launcherGapHeight = panelRect && widgetRect
    ? Math.max(0, widgetRect.height - panelRect.height)
    : 0;
  const maxWidth = Math.max(280, window.innerWidth - margin * 2);
  const maxHeight = Math.max(280, window.innerHeight - margin * 2 - launcherGapHeight);
  return {
    margin,
    minWidth: Math.min(320, maxWidth),
    minHeight: Math.min(420, maxHeight),
    maxWidth,
    maxHeight
  };
}

function applyProjectAssistantSize(size, options = {}) {
  const panel = document.getElementById("project-chat-panel");
  if (!panel) {
    return;
  }

  const limits = projectAssistantSizeLimits();
  panel.style.maxWidth = `${limits.maxWidth}px`;
  panel.style.maxHeight = `${limits.maxHeight}px`;

  const nextWidth = Number(size?.width);
  const nextHeight = Number(size?.height);
  if (!Number.isFinite(nextWidth) || !Number.isFinite(nextHeight)) {
    return;
  }

  const width = Math.min(limits.maxWidth, Math.max(limits.minWidth, nextWidth));
  const height = Math.min(limits.maxHeight, Math.max(limits.minHeight, nextHeight));
  panel.style.width = `${width}px`;
  panel.style.height = `${height}px`;
  if (options.persistCorrected && (width !== nextWidth || height !== nextHeight)) {
    saveProjectAssistantSize({ width, height });
  }
}

function loadProjectAssistantPosition() {
  applyProjectAssistantPosition(null);
}

function loadStoredProjectAssistantPosition() {
  try {
    const raw = window.localStorage.getItem(PROJECT_ASSISTANT_POSITION_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (error) {
    return null;
  }
}

function saveProjectAssistantPosition(position) {
  try {
    window.localStorage.setItem(PROJECT_ASSISTANT_POSITION_KEY, JSON.stringify(position));
  } catch (error) {
    // Position persistence is optional.
  }
}

function applyProjectAssistantPosition(position, options = {}) {
  const widget = document.querySelector(".project-chat-widget");
  if (!widget) {
    return;
  }

  const rect = widget.getBoundingClientRect();
  const margin = PROJECT_ASSISTANT_VIEWPORT_MARGIN;
  const fallbackLeft = window.innerWidth - rect.width - margin;
  const fallbackTop = window.innerHeight - rect.height - margin;
  const nextLeft = Number(position?.left);
  const nextTop = Number(position?.top);
  const width = rect.width || 260;
  const height = rect.height || 56;
  const maxLeft = Math.max(margin, window.innerWidth - width - margin);
  const maxTop = Math.max(margin, window.innerHeight - height - margin);
  const left = Number.isFinite(nextLeft) ? nextLeft : fallbackLeft;
  const top = Number.isFinite(nextTop) ? nextTop : fallbackTop;
  const clampedLeft = Math.min(maxLeft, Math.max(margin, left));
  const clampedTop = Math.min(maxTop, Math.max(margin, top));

  widget.style.left = `${clampedLeft}px`;
  widget.style.top = `${clampedTop}px`;
  widget.style.right = "auto";
  widget.style.bottom = "auto";
  if (options.persistCorrected && (clampedLeft !== nextLeft || clampedTop !== nextTop)) {
    saveProjectAssistantPosition({ left: clampedLeft, top: clampedTop });
  }
}

function toggleProjectChat(forceOpen) {
  const panel = document.getElementById("project-chat-panel");
  const toggle = document.getElementById("project-chat-toggle");
  const input = document.getElementById("project-chat-input");
  if (!panel) {
    return;
  }

  const shouldOpen = typeof forceOpen === "boolean" ? forceOpen : panel.hidden;
  panel.hidden = !shouldOpen;
  toggle?.setAttribute("aria-expanded", String(shouldOpen));

  if (shouldOpen) {
    const widget = document.querySelector(".project-chat-widget");
    applyProjectAssistantSize(loadStoredProjectAssistantSize(), { persistCorrected: true });
    if (widget) {
      const rect = widget.getBoundingClientRect();
      applyProjectAssistantPosition({ left: rect.left, top: rect.top }, { persistCorrected: true });
    }
    window.setTimeout(() => input?.focus(), 80);
    scrollProjectChatToBottom();
  }
}

async function sendProjectChatMessage(questionOverride) {
  if (state.projectChatBusy) {
    return;
  }

  const input = document.getElementById("project-chat-input");
  const sendButton = document.getElementById("project-chat-send");
  const question = String(questionOverride ?? input?.value ?? "").trim();
  if (!question) {
    return;
  }

  toggleProjectChat(true);
  renderProjectChatMessage("user", question);
  if (input) {
    input.value = "";
  }

  state.projectChatBusy = true;
  if (sendButton) {
    sendButton.disabled = true;
    sendButton.dataset.defaultText = sendButton.dataset.defaultText || sendButton.textContent || "Send";
    sendButton.textContent = "Thinking...";
  }
  const loadingBubble = renderProjectChatMessage("assistant", "Checking project notes", { loading: true });

  try {
    const data = await apiJson(API.chat, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "project", question })
    });
    replaceProjectChatLoading(loadingBubble, data.answer || "I could not find a project answer for that question.");
    renderProjectChatSuggestions(data.suggested_questions);
  } catch (error) {
    replaceProjectChatLoading(loadingBubble, error.message, { error: true });
  } finally {
    state.projectChatBusy = false;
    if (sendButton) {
      sendButton.disabled = false;
      sendButton.textContent = sendButton.dataset.defaultText || "Send";
    }
    input?.focus();
  }
}

function renderProjectChatMessage(role, message, options = {}) {
  const messages = document.getElementById("project-chat-messages");
  if (!messages) {
    return;
  }

  const bubble = document.createElement("div");
  const normalizedRole = role === "user" ? "user" : "assistant";
  bubble.className = `project-chat-message project-chat-message--${normalizedRole}`;
  if (options.error) {
    bubble.classList.add("project-chat-message--error");
  }
  if (options.loading) {
    bubble.classList.add("project-chat-message--loading");
    bubble.dataset.chatLoading = "true";
  }
  bubble.innerHTML = chatMessageHtml(message, options);
  messages.appendChild(bubble);
  scrollProjectChatToBottom();
  return bubble;
}

function renderProjectChatSuggestions() {
  const container = document.getElementById("project-chat-suggestions");
  if (!container) {
    return;
  }

  const activeKey = PROJECT_CHAT_TOPICS[state.projectChatTopic] ? state.projectChatTopic : "basics";
  state.projectChatTopic = activeKey;
  const activeTopic = PROJECT_CHAT_TOPICS[activeKey];
  const visibleQuestions = state.projectChatExpanded
    ? activeTopic.questions
    : activeTopic.questions.slice(0, 3);
  const topicButtons = Object.entries(PROJECT_CHAT_TOPICS)
    .map(([key, topic]) => {
      const isActive = key === activeKey;
      return `
        <button
          type="button"
          class="project-chat-topic-btn${isActive ? " is-active" : ""}"
          data-chat-topic="${escapeAttribute(key)}"
          aria-pressed="${isActive ? "true" : "false"}"
        >${escapeHtml(topic.label)}</button>
      `;
    })
    .join("");
  const questionButtons = visibleQuestions
    .map((question) => `<button type="button" class="project-chat-question-btn" data-chat-question="${escapeAttribute(question)}">${escapeHtml(question)}</button>`)
    .join("");
  const moreButton = activeTopic.questions.length > 3
    ? `<button type="button" class="project-chat-more-btn" data-chat-more="true">${state.projectChatExpanded ? "Less" : `More (${activeTopic.questions.length - 3})`}</button>`
    : "";

  container.innerHTML = `
    <div class="project-chat-suggestion-label">Suggested topics</div>
    <div class="project-chat-topics" role="list" aria-label="Project Assistant topics">${topicButtons}</div>
    <div class="project-chat-topic-questions" aria-label="${escapeAttribute(activeTopic.label)} questions">${questionButtons}${moreButton}</div>
  `;
}

function removeProjectChatLoading() {
  document.querySelectorAll("[data-chat-loading='true']").forEach((node) => node.remove());
}

function replaceProjectChatLoading(bubble, message, options = {}) {
  const target = bubble?.isConnected ? bubble : document.querySelector("[data-chat-loading='true']");
  if (!target) {
    renderProjectChatMessage("assistant", message, options);
    return;
  }
  delete target.dataset.chatLoading;
  target.classList.remove("project-chat-message--loading");
  if (options.error) {
    target.classList.add("project-chat-message--error");
  }
  target.innerHTML = chatMessageHtml(message, options);
  scrollProjectChatToBottom();
}

function scrollProjectChatToBottom() {
  const messages = document.getElementById("project-chat-messages");
  if (messages) {
    messages.scrollTop = messages.scrollHeight;
  }
}

function bindPredictionChat() {
  const form = document.getElementById("prediction-chat-form");
  const suggestions = document.getElementById("prediction-chat-suggestions");

  form?.addEventListener("submit", (event) => {
    event.preventDefault();
    sendPredictionChatMessage();
  });

  suggestions?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-prediction-chat-question]");
    if (!button) {
      return;
    }
    sendPredictionChatMessage(button.dataset.predictionChatQuestion);
  });
}

function showPredictionAssistant(prediction) {
  const panel = document.getElementById("prediction-assistant");
  const locked = document.getElementById("prediction-chat-locked");
  if (!panel) {
    return;
  }

  panel.hidden = false;
  if (locked) {
    locked.hidden = true;
  }
  setText("prediction-chat-pair", `NSC ${prediction.nsc1} + NSC ${prediction.nsc2}`);
  setText("prediction-chat-cell", prediction.cellLine);
  setText("prediction-chat-score", `ComboScore ${formatScore(prediction.score, 3)} | ${semanticLabelForScore(prediction.score)}`);
  setText("prediction-chat-model", prediction.model || "Auto-selected model");
  setPredictionChatStatus("Ready", "neutral");
  updatePredictionAssistantButton(true);

  const messages = document.getElementById("prediction-chat-messages");
  if (messages) {
    messages.innerHTML = "";
  }

  renderPredictionChatMessage(
    "assistant",
    `I can explain the latest prediction for NSC ${prediction.nsc1} + NSC ${prediction.nsc2} in ${prediction.cellLine}. Ask about the score, model, direction averaging, safety, or next checks.`
  );
  renderPredictionChatSuggestions(predictionChatSuggestionsForPrediction(prediction));
}

function resetPredictionAssistant() {
  const panel = document.getElementById("prediction-assistant");
  const locked = document.getElementById("prediction-chat-locked");
  if (panel) {
    panel.hidden = true;
  }
  if (locked) {
    locked.hidden = false;
  }
  const messages = document.getElementById("prediction-chat-messages");
  if (messages) {
    messages.innerHTML = "";
  }
  const input = document.getElementById("prediction-chat-input");
  if (input) {
    input.value = "";
  }
  updatePredictionAssistantButton(false);
  setPredictionChatStatus("Locked", "neutral");
}

async function sendPredictionChatMessage(questionOverride) {
  if (state.predictionChatBusy || !state.lastPrediction) {
    return;
  }

  const input = document.getElementById("prediction-chat-input");
  const sendButton = document.getElementById("prediction-chat-send");
  const question = String(questionOverride ?? input?.value ?? "").trim();
  if (!question) {
    return;
  }

  renderPredictionChatMessage("user", question);
  if (input) {
    input.value = "";
  }

  state.predictionChatBusy = true;
  if (sendButton) {
    sendButton.disabled = true;
    sendButton.dataset.defaultText = sendButton.dataset.defaultText || sendButton.textContent || "Ask";
    sendButton.textContent = "Thinking...";
  }
  setPredictionChatStatus("Thinking", "neutral");
  const loadingBubble = renderPredictionChatMessage("assistant", "Reading the latest prediction context", { loading: true });

  try {
    const data = await apiJson(API.chat, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPredictionChatPayload(question))
    });
    replacePredictionChatLoading(loadingBubble, data.answer || "I could not explain that prediction question.");
    renderPredictionChatSuggestions(data.suggested_questions);
    setPredictionChatStatus(predictionChatSourceLabel(data), data.llm_used ? "good" : "neutral");
  } catch (error) {
    replacePredictionChatLoading(loadingBubble, error.message, { error: true });
    setPredictionChatStatus("Error", "danger");
  } finally {
    state.predictionChatBusy = false;
    if (sendButton) {
      sendButton.disabled = false;
      sendButton.textContent = sendButton.dataset.defaultText || "Ask";
    }
    input?.focus();
  }
}

function renderPredictionChatMessage(role, message, options = {}) {
  const messages = document.getElementById("prediction-chat-messages");
  if (!messages) {
    return;
  }

  const bubble = document.createElement("div");
  const normalizedRole = role === "user" ? "user" : "assistant";
  bubble.className = `prediction-chat-message prediction-chat-message--${normalizedRole}`;
  if (options.error) {
    bubble.classList.add("prediction-chat-message--error");
  }
  if (options.loading) {
    bubble.classList.add("prediction-chat-message--loading");
    bubble.dataset.predictionChatLoading = "true";
  }
  bubble.innerHTML = chatMessageHtml(message, options);
  messages.appendChild(bubble);
  scrollPredictionChatToBottom();
  return bubble;
}

function renderPredictionChatSuggestions(suggestions) {
  const container = document.getElementById("prediction-chat-suggestions");
  if (!container) {
    return;
  }

  const values = Array.isArray(suggestions) && suggestions.length
    ? suggestions
    : predictionChatSuggestionsForPrediction(state.lastPrediction);

  container.innerHTML = values
    .slice(0, 7)
    .map((question) => `<button type="button" data-prediction-chat-question="${escapeAttribute(question)}">${escapeHtml(question)}</button>`)
    .join("");
}

function buildPredictionChatPayload(question) {
  const prediction = state.lastPrediction;
  return {
    mode: "prediction",
    question,
    prediction: {
      input: {
        NSC1: prediction.nsc1,
        NSC2: prediction.nsc2,
        CELLNAME: prediction.cellLine,
        model_type: prediction.modelType || "classical"
      },
      prediction_NSC1_to_NSC2: prediction.forward,
      prediction_NSC2_to_NSC1: prediction.reverse,
      final_predicted_COMBOSCORE: prediction.score,
      label: semanticLabelForScore(prediction.score),
      model_used: prediction.model,
      model_type: prediction.modelType || "classical"
    },
    explanation: explanationMatchesPrediction(state.lastExplanation, prediction) ? state.lastExplanation : null
  };
}

function explanationMatchesPrediction(explanation, prediction) {
  if (!explanation || !prediction || !explanation.input) {
    return false;
  }
  return (
    String(explanation.input.NSC1) === String(prediction.nsc1) &&
    String(explanation.input.NSC2) === String(prediction.nsc2) &&
    String(explanation.input.CELLNAME) === String(prediction.cellLine) &&
    String(explanation.model_type || explanation.input.model_type || "classical") === String(prediction.modelType || "classical")
  );
}

function predictionChatSuggestionsForPrediction(prediction) {
  const label = String(prediction?.label || semanticLabelForScore(prediction?.score)).toLowerCase();
  const labelQuestion = label.includes("antag")
    ? "Why is this antagonistic?"
    : label.includes("neutral")
      ? "Why is this neutral?"
      : "Why is this synergistic?";
  return [
    "Explain this result",
    labelQuestion,
    "What does this score mean?",
    "What model was used?",
    "Why predict both directions?",
    "Is this clinical advice?",
    "What should I check next?",
    "Which features caused this?"
  ];
}

function setPredictionChatStatus(text, tone = "neutral") {
  const status = document.getElementById("prediction-chat-status");
  if (!status) {
    return;
  }
  status.textContent = text;
  setToneClass(status, tone);
}

function predictionChatSourceLabel(data) {
  if (data?.provider_label === "AI Enhanced" || data?.provider_label === "Built-in Guide") {
    return data.provider_label;
  }
  return data?.llm_used ? "AI Enhanced" : "Built-in Guide";
}

function updatePredictionAssistantButton(isEnabled) {
  const button = document.getElementById("open-prediction-assistant-btn");
  if (!button) {
    return;
  }
  button.disabled = !isEnabled;
}

function removePredictionChatLoading() {
  document.querySelectorAll("[data-prediction-chat-loading='true']").forEach((node) => node.remove());
}

function replacePredictionChatLoading(bubble, message, options = {}) {
  const target = bubble?.isConnected ? bubble : document.querySelector("[data-prediction-chat-loading='true']");
  if (!target) {
    renderPredictionChatMessage("assistant", message, options);
    return;
  }
  delete target.dataset.predictionChatLoading;
  target.classList.remove("prediction-chat-message--loading");
  if (options.error) {
    target.classList.add("prediction-chat-message--error");
  }
  target.innerHTML = chatMessageHtml(message, options);
  scrollPredictionChatToBottom();
}

function scrollPredictionChatToBottom() {
  const messages = document.getElementById("prediction-chat-messages");
  if (messages) {
    messages.scrollTop = messages.scrollHeight;
  }
}

async function runPredict() {
  clearAlert("predict-alert");

  const validation = refreshAllValidation();
  if (!validation.predict) {
    resetPredictionAssistant();
    stopPredictionLoader({ restorePanel: true });
    showAlert("predict-alert", "Select valid drugs and a valid cell line before running prediction.");
    return;
  }

  const payload = buildModelPayload("drug1-id", "drug1-input", "drug2-id", "drug2-input", "cell-line");
  if (!payload) {
    resetPredictionAssistant();
    stopPredictionLoader({ restorePanel: true });
    showAlert("predict-alert", getPredictionInputError("drug1-input", "drug2-input", "cell-line"));
    return;
  }

  resetPredictionAssistant();
  setButtonLoading("predict-btn", "predict-btn-text", true, "Predict Synergy", "Running model");
  startPredictionLoader();

  try {
    const data = await apiJson(API.predict, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const prediction = normalizePredictionResponse(data, payload);
    state.lastPrediction = prediction;
    stopPredictionLoader();
    renderPrediction(prediction);
    showPredictionAssistant(prediction);
    savePredictionHistory(prediction);
    syncSelectionsFromPredict();
  } catch (error) {
    stopPredictionLoader({ restorePanel: true });
    showAlert("predict-alert", error.message);
  } finally {
    setButtonLoading("predict-btn", "predict-btn-text", false, "Predict Synergy", "Running model");
    refreshAllValidation();
  }
}

function startPredictionLoader() {
  const empty = document.getElementById("results-empty");
  const content = document.getElementById("results-content");

  if (empty) {
    empty.hidden = true;
  }
  if (content) {
    content.hidden = true;
  }

  const pill = document.getElementById("score-pill");
  if (pill) {
    setToneClass(pill, "neutral");
    pill.textContent = "Running model";
  }

  startInlineLoader("prediction", PREDICTION_FLOW_MESSAGES);
}

function stopPredictionLoader(options = {}) {
  const restorePanel = Boolean(options.restorePanel);

  stopInlineLoader("prediction");
  if (restorePanel) {
    restorePredictionPanel();
  }
}

function startInlineLoader(name, messages) {
  const loader = document.getElementById(`${name}-loader`);
  if (!loader) {
    return;
  }

  clearInlineLoader(name);
  state.inlineLoaders[name] = {
    index: 0,
    messages: [...messages],
    timer: null
  };
  setInlineLoaderMessage(name);
  loader.hidden = false;

  state.inlineLoaders[name].timer = window.setInterval(() => {
    const loaderState = state.inlineLoaders[name];
    if (!loaderState) {
      return;
    }
    loaderState.index = (loaderState.index + 1) % loaderState.messages.length;
    setInlineLoaderMessage(name);
  }, 700);
}

function stopInlineLoader(name) {
  clearInlineLoader(name);
  const loader = document.getElementById(`${name}-loader`);
  if (loader) {
    loader.hidden = true;
  }
}

function clearInlineLoader(name) {
  const loaderState = state.inlineLoaders[name];
  if (loaderState?.timer) {
    window.clearInterval(loaderState.timer);
  }
  delete state.inlineLoaders[name];
}

function setInlineLoaderMessage(name) {
  const loaderState = state.inlineLoaders[name];
  const message = document.getElementById(`${name}-loader-message`);
  if (!loaderState || !message) {
    return;
  }

  message.classList.remove("is-active");
  // Restart the small text animation each time the flow advances.
  void message.offsetWidth;
  message.textContent = `${loaderState.messages[loaderState.index]}...`;
  message.classList.add("is-active");
}

function restorePredictionPanel() {
  const empty = document.getElementById("results-empty");
  const content = document.getElementById("results-content");

  if (state.lastPrediction) {
    if (empty) {
      empty.hidden = true;
    }
    if (content) {
      content.hidden = false;
    }
    const pill = document.getElementById("score-pill");
    if (pill) {
      setToneClass(pill, state.lastPrediction.level);
      pill.textContent = state.lastPrediction.label;
    }
    return;
  }

  if (empty) {
    empty.hidden = false;
  }
  if (content) {
    content.hidden = true;
  }

  const pill = document.getElementById("score-pill");
  if (pill) {
    setToneClass(pill, "neutral");
    pill.textContent = "Awaiting input";
  }
}

function buildModelPayload(drug1Hidden, drug1Input, drug2Hidden, drug2Input, cellSelect) {
  const nsc1 = resolveDrugId(drug1Hidden, drug1Input);
  const nsc2 = resolveDrugId(drug2Hidden, drug2Input);
  const cellLine = document.getElementById(cellSelect)?.value || "";
  const modelType = getSelectedModelType();

  if (!nsc1 || !nsc2 || !cellLine) {
    return null;
  }

  if (state.drugIds.size && (!state.drugIds.has(String(nsc1)) || !state.drugIds.has(String(nsc2)))) {
    return null;
  }

  if (state.cellLines.length && !state.cellLines.includes(cellLine)) {
    return null;
  }

  return {
    NSC1: Number(nsc1),
    NSC2: Number(nsc2),
    CELLNAME: cellLine,
    drug1_id: String(nsc1),
    drug2_id: String(nsc2),
    cell_line: cellLine,
    model_type: modelType
  };
}

function getSelectedModelType() {
  const value = document.getElementById("model-type")?.value || "classical";
  return value === "quantum" ? "quantum" : "classical";
}

function getPredictionInputError(drug1Input, drug2Input, cellSelect) {
  const drug1Text = document.getElementById(drug1Input)?.value?.trim() || "";
  const drug2Text = document.getElementById(drug2Input)?.value?.trim() || "";
  const cellLine = document.getElementById(cellSelect)?.value || "";

  if (!drug1Text || !drug2Text || !cellLine) {
    return "Select both valid NSC drugs and a cell line before running the prediction.";
  }

  const drug1Candidate = resolveKnownDrugId(drug1Text);
  const drug2Candidate = resolveKnownDrugId(drug2Text);
  if (state.drugIds.size && (!drug1Candidate || !state.drugIds.has(drug1Candidate))) {
    return `Drug 1 NSC ${drug1Candidate || drug1Text} is not in the loaded valid drug list.`;
  }
  if (state.drugIds.size && (!drug2Candidate || !state.drugIds.has(drug2Candidate))) {
    return `Drug 2 NSC ${drug2Candidate || drug2Text} is not in the loaded valid drug list.`;
  }
  if (state.cellLines.length && !state.cellLines.includes(cellLine)) {
    return `Cell line ${cellLine} is not in the loaded valid cell-line list.`;
  }

  return "Select both valid NSC drugs and a cell line before running the prediction.";
}

function getDrugInfoInputError(drug1Input, drug2Input) {
  const drug1Text = document.getElementById(drug1Input)?.value?.trim() || "";
  const drug2Text = document.getElementById(drug2Input)?.value?.trim() || "";

  if (!drug1Text || !drug2Text) {
    return "Select two valid NSC drugs before loading compound details.";
  }

  const drug1Candidate = resolveKnownDrugId(drug1Text);
  const drug2Candidate = resolveKnownDrugId(drug2Text);
  if (state.drugIds.size && (!drug1Candidate || !state.drugIds.has(drug1Candidate))) {
    return `Drug 1 NSC ${drug1Candidate || drug1Text} is not in the loaded valid drug list.`;
  }
  if (state.drugIds.size && (!drug2Candidate || !state.drugIds.has(drug2Candidate))) {
    return `Drug 2 NSC ${drug2Candidate || drug2Text} is not in the loaded valid drug list.`;
  }

  return "Select two valid NSC drugs before loading compound details.";
}

function extractDrugCandidate(value) {
  const text = String(value ?? "").trim();
  if (/^\d+$/.test(text)) {
    return text;
  }
  const match = text.match(/\b(\d{2,})\b/);
  return match ? match[1] : "";
}

function normalizePredictionResponse(data, payload) {
  const input = data.input || {};
  const score = Number(
    data.final_predicted_COMBOSCORE ??
    data.final_prediction ??
    data.predicted_comboscore ??
    data.synergy_score ??
    data.score ??
    0
  );
  const label = String(data.prediction_label ?? data.label ?? data.interpretation ?? labelForScore(score));
  const category = String(data.prediction_category ?? data.category ?? categoryForLabel(label, score));
  const color = data.color || colorForCategory(category, score);

  return {
    score,
    label,
    category,
    color,
    level: toneForCategory(category, score),
    nsc1: input.NSC1 ?? data.NSC1 ?? payload.NSC1,
    nsc2: input.NSC2 ?? data.NSC2 ?? payload.NSC2,
    cellLine: input.CELLNAME ?? data.CELLNAME ?? payload.CELLNAME,
    modelType: data.model_type || input.model_type || payload.model_type || "classical",
    model: data.model_used || data.model_name || "",
    modelPath: data.model_path || "",
    forward: data.prediction_NSC1_to_NSC2,
    reverse: data.prediction_NSC2_to_NSC1,
    explanation: data.explanation || "",
    suggestion: data.suggestion || "",
    gaugeMin: SCORE_DISPLAY_MIN,
    gaugeMax: SCORE_DISPLAY_MAX
  };
}

function renderPrediction(data) {
  document.getElementById("results-empty").hidden = true;
  document.getElementById("results-content").hidden = false;

  setText("score-label", data.label);
  document.getElementById("score-label").style.color = data.color;
  setText("r-drug1", `NSC ${data.nsc1}`);
  setText("r-drug2", `NSC ${data.nsc2}`);
  setText("r-cell", data.cellLine);
  setText("r-cancer", "Step 6 cell-line model");
  setText("r-level", data.label);
  setText("r-cell-tile", data.cellLine);
  setText("r-cancer-tile", data.model || "Auto-selected model");
  renderPredictionStory(data);
  updatePredictionReportButton(true);

  const pill = document.getElementById("score-pill");
  setToneClass(pill, data.level);
  pill.textContent = data.label;

  setGauge(data.score, data.color, data.gaugeMin, data.gaugeMax);
}

function renderPredictionStory(data) {
  const container = document.getElementById("score-summary");
  if (!container) {
    return;
  }

  container.innerHTML = buildPredictionStoryHtml(data);
}

function buildPredictionStoryHtml(data) {
  const storyLabel = storyLabelForPrediction(data.label, data.score);
  const forward = formatScore(data.forward, 3);
  const reverse = formatScore(data.reverse, 3);
  const finalScore = formatScore(data.score, 3);
  const scoreMeaning = scoreMeaningForScore(data.score);

  return `
    <p>
      SynergyLens evaluated the drug pair <strong>NSC ${escapeHtml(String(data.nsc1))}</strong> and
      <strong>NSC ${escapeHtml(String(data.nsc2))}</strong> for the
      <strong>${escapeHtml(data.cellLine)}</strong> cell line. The final readout below summarizes
      what the predicted ComboScore represents for this pair.
    </p>
    <div class="story-score-grid">
      <div><span>NSC ${escapeHtml(String(data.nsc1))} -&gt; NSC ${escapeHtml(String(data.nsc2))}</span><strong>${escapeHtml(forward)}</strong></div>
      <div><span>NSC ${escapeHtml(String(data.nsc2))} -&gt; NSC ${escapeHtml(String(data.nsc1))}</span><strong>${escapeHtml(reverse)}</strong></div>
      <div><span>Final averaged ComboScore</span><strong>${escapeHtml(finalScore)}</strong></div>
      <div><span>Final label</span><strong>${escapeHtml(storyLabel)}</strong></div>
    </div>
    <p>
      The same pair was scored in both drug orders because combination features are order-sensitive.
      Averaging those two directional scores gives the final ComboScore.
      This result is <strong>${escapeHtml(storyLabel)}</strong>: ${escapeHtml(scoreMeaning)}
      Positive ComboScore suggests synergy, near zero suggests neutral or additive behavior,
      and negative ComboScore suggests antagonism.
    </p>
    <p class="story-safety">${escapeHtml(SAFETY_NOTE_TEXT)}</p>
  `;
}

function updatePredictionReportButton(isEnabled) {
  const button = document.getElementById("prediction-report-btn");
  if (!button) {
    return;
  }

  button.disabled = !isEnabled;
}

function downloadPredictionReport() {
  if (!state.lastPrediction) {
    showAlert("predict-alert", "Run a successful prediction before downloading a report.");
    updatePredictionReportButton(false);
    return;
  }

  const reportHtml = buildPredictionReportHtml(state.lastPrediction);
  const blob = new Blob([reportHtml], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  triggerDownload(url, predictionReportFilename());
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function buildPredictionReportHtml(data) {
  const generatedAt = new Date();
  const storyLabel = storyLabelForPrediction(data.label, data.score);
  const scoreMeaning = scoreMeaningForScore(data.score);
  const finalScore = formatScore(data.score, 3);
  const forward = formatScore(data.forward, 3);
  const reverse = formatScore(data.reverse, 3);
  const modelPath = data.modelPath || "Not provided";
  const generatedAtText = generatedAt.toLocaleString();

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SynergyLens Prediction Report</title>
  <style>
    :root {
      color-scheme: light;
      --text: #18312c;
      --muted: #5f716d;
      --accent: #0f766e;
      --danger: #c74747;
      --border: #dbe6e2;
      --surface: #f7faf8;
    }
    body {
      margin: 0;
      padding: 32px;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--text);
      background: #ffffff;
      line-height: 1.55;
    }
    main {
      max-width: 880px;
      margin: 0 auto;
    }
    h1, h2 {
      margin: 0;
      line-height: 1.15;
    }
    h1 {
      font-size: 34px;
      color: var(--accent);
    }
    h2 {
      margin-top: 28px;
      font-size: 18px;
    }
    .meta {
      margin-top: 8px;
      color: var(--muted);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .card {
      padding: 14px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--surface);
    }
    .card span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .card strong {
      display: block;
      margin-top: 6px;
      font-size: 16px;
      overflow-wrap: anywhere;
    }
    .score {
      font-size: 28px;
      color: ${escapeHtml(colorForCategory(data.category, data.score))};
    }
    .safety {
      margin-top: 24px;
      padding: 16px;
      border: 1px solid var(--border);
      border-left: 5px solid var(--accent);
      border-radius: 12px;
      background: var(--surface);
    }
    @media print {
      body { padding: 18px; }
      .card { break-inside: avoid; }
    }
  </style>
</head>
<body>
  <main>
    <h1>SynergyLens Prediction Report</h1>
    <p class="meta">Generated: ${escapeHtml(generatedAtText)}</p>

    <h2>Input</h2>
    <div class="grid">
      <div class="card"><span>NSC1</span><strong>${escapeHtml(String(data.nsc1))}</strong></div>
      <div class="card"><span>NSC2</span><strong>${escapeHtml(String(data.nsc2))}</strong></div>
      <div class="card"><span>Cell line</span><strong>${escapeHtml(data.cellLine)}</strong></div>
      <div class="card"><span>Project</span><strong>SynergyLens</strong></div>
    </div>

    <h2>Model</h2>
    <div class="grid">
      <div class="card"><span>Model used</span><strong>${escapeHtml(data.model || "Auto-selected cell-line model")}</strong></div>
      <div class="card"><span>Model path</span><strong>${escapeHtml(modelPath)}</strong></div>
    </div>

    <h2>Predictions</h2>
    <div class="grid">
      <div class="card"><span>NSC1 -&gt; NSC2 prediction</span><strong>${escapeHtml(forward)}</strong></div>
      <div class="card"><span>NSC2 -&gt; NSC1 prediction</span><strong>${escapeHtml(reverse)}</strong></div>
      <div class="card"><span>Final averaged ComboScore</span><strong class="score">${escapeHtml(finalScore)}</strong></div>
      <div class="card"><span>Final label</span><strong>${escapeHtml(storyLabel)}</strong></div>
    </div>

    <h2>ComboScore Interpretation</h2>
    <p>Positive ComboScore suggests synergy. Near zero ComboScore suggests neutral or additive behavior. Negative ComboScore suggests antagonism.</p>

    <h2>Result Story</h2>
    <p>
      SynergyLens evaluated NSC ${escapeHtml(String(data.nsc1))} with NSC ${escapeHtml(String(data.nsc2))}
      in the ${escapeHtml(data.cellLine)} cell line using the automatically selected
      ${escapeHtml(data.model || "cell-line")} model.
    </p>
    <p>
      The model predicted ${escapeHtml(forward)} for NSC ${escapeHtml(String(data.nsc1))} -&gt; NSC ${escapeHtml(String(data.nsc2))}
      and ${escapeHtml(reverse)} for NSC ${escapeHtml(String(data.nsc2))} -&gt; NSC ${escapeHtml(String(data.nsc1))}.
      The final averaged ComboScore is ${escapeHtml(finalScore)}, labeled ${escapeHtml(storyLabel)}.
      ${escapeHtml(scoreMeaning)}
    </p>

    <div class="safety">
      <strong>Safety note</strong>
      <p>${escapeHtml(SAFETY_NOTE_TEXT)}</p>
    </div>
  </main>
</body>
</html>`;
}

function predictionReportFilename() {
  const timestamp = new Date()
    .toISOString()
    .replace(/[:.]/g, "-");
  return `synergylens_prediction_report_${timestamp}.html`;
}

function storyLabelForPrediction(label, score) {
  const normalized = String(label || "").toLowerCase();
  if (normalized.includes("synerg")) {
    return "synergistic";
  }
  if (normalized.includes("antag")) {
    return "antagonistic";
  }
  if (normalized.includes("neutral") || normalized.includes("weak")) {
    return "neutral";
  }
  return semanticLabelForScore(score);
}

function semanticLabelForScore(score) {
  const value = Number(score);
  if (value > NEUTRAL_THRESHOLD) {
    return "synergistic";
  }
  if (value < -NEUTRAL_THRESHOLD) {
    return "antagonistic";
  }
  return "neutral / weak";
}

function scoreMeaningForScore(score) {
  const value = Number(score);
  if (value > NEUTRAL_THRESHOLD) {
    return "the model predicts a ComboScore above +4, which means the drug pair may perform better together than expected in this cell line. This suggests a synergistic interaction.";
  }
  if (value >= -NEUTRAL_THRESHOLD) {
    return "the predicted ComboScore is between -4 and +4, which means the drug pair behaves roughly as expected from the individual drugs. This suggests neutral or weak behavior.";
  }
  return "the model predicts a ComboScore below -4, which means the drug pair may perform worse together than expected in this cell line. This suggests antagonism.";
}

function setGauge(score, color, min, max) {
  const targetScore = Number(score);
  const safeTargetScore = Number.isFinite(targetScore) ? targetScore : 0;
  const scoreValue = document.getElementById("score-value");
  if (!scoreValue) {
    return;
  }
  const displayedScore = Number(scoreValue.textContent);
  const startScore = Number.isFinite(displayedScore)
    ? displayedScore
    : state.gaugeDisplayScore;
  const duration = GAUGE_ANIMATION_DURATION;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (state.gaugeAnimationFrame) {
    window.cancelAnimationFrame(state.gaugeAnimationFrame);
    state.gaugeAnimationFrame = null;
  }

  if (reducedMotion) {
    updateGaugeFrame(safeTargetScore, min, max, getGaugeColor(safeTargetScore));
    state.gaugeDisplayScore = safeTargetScore;
    state.gaugeInitialized = true;
    return;
  }

  state.gaugeInitialized = true;
  animateGauge(startScore, safeTargetScore, duration, min, max);
}

function initializeGauge() {
  renderGaugeTicks();
  updateGaugeFrame(state.gaugeDisplayScore, SCORE_DISPLAY_MIN, SCORE_DISPLAY_MAX, getGaugeColor(state.gaugeDisplayScore));
}

function animateGauge(fromScore, toScore, duration, min = SCORE_DISPLAY_MIN, max = SCORE_DISPLAY_MAX) {
  const safeFrom = Number.isFinite(Number(fromScore)) ? Number(fromScore) : 0;
  const safeTo = Number.isFinite(Number(toScore)) ? Number(toScore) : 0;
  const fromColor = getGaugeColor(safeFrom);
  const toColor = getGaugeColor(safeTo);
  const startedAt = window.performance.now();

  const step = (now) => {
    const elapsed = now - startedAt;
    const progress = Math.min(elapsed / duration, 1);
    const easedProgress = easeOutCubic(progress);
    const value = safeFrom + ((safeTo - safeFrom) * easedProgress);
    const frameColor = mixColors(fromColor, toColor, easedProgress);

    updateGaugeFrame(value, min, max, frameColor);

    if (progress < 1) {
      state.gaugeAnimationFrame = window.requestAnimationFrame(step);
      return;
    }

    state.gaugeAnimationFrame = null;
    updateGaugeFrame(safeTo, min, max, toColor);
    state.gaugeDisplayScore = safeTo;
  };

  state.gaugeAnimationFrame = window.requestAnimationFrame(step);
}

function updateGaugeFrame(score, min, max, color) {
  const gaugeCard = document.querySelector(".gauge-card");
  const arc = document.getElementById("gauge-arc");
  const needle = document.getElementById("gauge-needle");
  const scoreValue = document.getElementById("score-value");
  const scoreLabel = document.getElementById("score-label");
  if (!arc || !needle || !scoreValue) {
    return;
  }

  const dashOffset = 100 - (scoreToGaugeRatio(score, min, max) * 100);
  const needleAngle = scoreToNeedleAngle(score, min, max);
  const gaugeColor = color || getGaugeColor(score);

  if (gaugeCard) {
    gaugeCard.style.setProperty("--gauge-active-color", gaugeColor);
  }
  arc.style.strokeDashoffset = String(dashOffset);
  arc.style.stroke = gaugeColor;
  needle.style.transform = `rotate(${needleAngle}deg)`;
  needle.style.color = gaugeColor;
  scoreValue.textContent = formatScore(score, 2);
  scoreValue.style.color = "#f7fbfa";
  if (scoreLabel) {
    scoreLabel.style.color = gaugeColor;
  }
}

function easeOutCubic(value) {
  return 1 - Math.pow(1 - value, 3);
}

function clamp(value, min, max) {
  const numericValue = Number(value);
  const safeValue = Number.isFinite(numericValue) ? numericValue : 0;
  return Math.max(min, Math.min(max, safeValue));
}

function clampGaugeScore(score, min = SCORE_DISPLAY_MIN, max = SCORE_DISPLAY_MAX) {
  const { safeMin, safeMax } = normalizeGaugeBounds(min, max);
  return clamp(score, safeMin, safeMax);
}

function scoreToGaugeRatio(score, min = SCORE_DISPLAY_MIN, max = SCORE_DISPLAY_MAX) {
  const { safeMin, safeMax } = normalizeGaugeBounds(min, max);
  const clampedScore = clampGaugeScore(score, safeMin, safeMax);
  return (clampedScore - safeMin) / (safeMax - safeMin);
}

function scoreToNeedleAngle(score, min = SCORE_DISPLAY_MIN, max = SCORE_DISPLAY_MAX) {
  return -90 + (scoreToGaugeRatio(score, min, max) * 180);
}

function scoreToGaugeAngle(score, min = SCORE_DISPLAY_MIN, max = SCORE_DISPLAY_MAX) {
  const ratio = scoreToGaugeRatio(score, min, max);
  return GAUGE_GEOMETRY.startAngle + (ratio * (GAUGE_GEOMETRY.endAngle - GAUGE_GEOMETRY.startAngle));
}

function polarToCartesian(cx, cy, radius, angleDegrees) {
  const angleRadians = (angleDegrees * Math.PI) / 180;
  return {
    x: cx + (radius * Math.cos(angleRadians)),
    y: cy + (radius * Math.sin(angleRadians))
  };
}

function describeGaugeArc(cx, cy, radius, startAngle, endAngle) {
  const start = polarToCartesian(cx, cy, radius, startAngle);
  const end = polarToCartesian(cx, cy, radius, endAngle);
  const largeArcFlag = Math.abs(endAngle - startAngle) <= 180 ? 0 : 1;
  return [
    "M", formatSvgNumber(start.x), formatSvgNumber(start.y),
    "A", radius, radius, 0, largeArcFlag, 1, formatSvgNumber(end.x), formatSvgNumber(end.y)
  ].join(" ");
}

function renderGaugeTicks() {
  if (state.gaugeTicksRendered) {
    return;
  }

  const tickContainer = document.getElementById("gauge-ticks");
  const labelContainer = document.getElementById("gauge-scale-labels");
  if (!tickContainer || !labelContainer) {
    return;
  }

  tickContainer.innerHTML = [
    ...GAUGE_MINOR_TICKS.map((value) => buildGaugeTick(value, "minor")),
    ...GAUGE_MEDIUM_TICKS.map((value) => buildGaugeTick(value, "medium")),
    ...GAUGE_MAJOR_TICKS.map((value) => buildGaugeTick(value, "major"))
  ].join("");

  labelContainer.innerHTML = GAUGE_MAJOR_TICKS
    .map((value) => buildGaugeScaleLabel(value))
    .join("");

  updateGaugeScaleLabels();
  state.gaugeTicksRendered = true;
}

function buildGaugeTick(value, size) {
  const angle = scoreToGaugeAngle(value);
  const lengthMap = {
    minor: 5,
    medium: 8,
    major: 13
  };
  const outerRadius = GAUGE_GEOMETRY.tickOuterRadius;
  const innerRadius = outerRadius - lengthMap[size];
  const inner = polarToCartesian(GAUGE_GEOMETRY.cx, GAUGE_GEOMETRY.cy, innerRadius, angle);
  const outer = polarToCartesian(GAUGE_GEOMETRY.cx, GAUGE_GEOMETRY.cy, outerRadius, angle);
  const className = `gauge-${size}-tick`;
  const color = getGaugeColor(value);

  return `<line class="${className}" x1="${formatSvgNumber(inner.x)}" y1="${formatSvgNumber(inner.y)}" x2="${formatSvgNumber(outer.x)}" y2="${formatSvgNumber(outer.y)}" style="--gauge-tick-color: ${color}"></line>`;
}

function buildGaugeScaleLabel(value) {
  const angle = scoreToGaugeAngle(value);
  const point = polarToCartesian(GAUGE_GEOMETRY.cx, GAUGE_GEOMETRY.cy, GAUGE_GEOMETRY.labelRadius, angle);
  const color = getGaugeColor(value);
  return `<text class="gauge-scale-label" x="${formatSvgNumber(point.x)}" y="${formatSvgNumber(point.y)}" style="--gauge-label-color: ${color}">${formatGaugeTickLabel(value)}</text>`;
}

function updateGaugeScaleLabels() {
  document.querySelectorAll(".gauge-scale-label").forEach((label) => {
    const value = Number(label.textContent);
    if (Number.isFinite(value)) {
      label.style.setProperty("--gauge-label-color", getGaugeColor(value));
    }
  });
}

function formatGaugeTickLabel(value) {
  return Number(value) > 0 ? `+${value}` : String(value);
}

function formatSvgNumber(value) {
  return Number(value).toFixed(2).replace(/\.?0+$/, "");
}

function getGaugeColor(score) {
  const value = Number(score);
  if (value <= -STRONG_EFFECT_THRESHOLD) {
    return "#f46f63";
  }
  if (value < -NEUTRAL_THRESHOLD) {
    return "#ff936a";
  }
  if (value <= NEUTRAL_THRESHOLD) {
    return "#f6c94a";
  }
  if (value < STRONG_EFFECT_THRESHOLD) {
    return "#20d7bd";
  }
  return "#5cead8";
}

function mixColors(fromColor, toColor, ratio) {
  const from = hexToRgb(fromColor);
  const to = hexToRgb(toColor);
  const clampedRatio = clamp(ratio, 0, 1);
  const mixed = {
    r: Math.round(from.r + ((to.r - from.r) * clampedRatio)),
    g: Math.round(from.g + ((to.g - from.g) * clampedRatio)),
    b: Math.round(from.b + ((to.b - from.b) * clampedRatio))
  };
  return `rgb(${mixed.r}, ${mixed.g}, ${mixed.b})`;
}

function hexToRgb(color) {
  const normalized = String(color || "").replace("#", "");
  const value = normalized.length === 3
    ? normalized.split("").map((character) => character + character).join("")
    : normalized;
  const parsed = Number.parseInt(value, 16);
  if (!Number.isFinite(parsed)) {
    return { r: 246, g: 201, b: 74 };
  }
  return {
    r: (parsed >> 16) & 255,
    g: (parsed >> 8) & 255,
    b: parsed & 255
  };
}

function normalizeGaugeBounds(min, max) {
  const numericMin = Number(min);
  const numericMax = Number(max);
  const safeMin = Number.isFinite(numericMin) ? numericMin : SCORE_DISPLAY_MIN;
  const safeMax = Number.isFinite(numericMax) && numericMax > safeMin
    ? numericMax
    : SCORE_DISPLAY_MAX;
  return {
    safeMin,
    safeMax: safeMax > safeMin ? safeMax : safeMin + 1
  };
}

function syncSelectionsFromPredict() {
  copyField("drug1-input", "edrug1-input");
  copyField("drug1-id", "edrug1-id");
  copyField("drug2-input", "edrug2-input");
  copyField("drug2-id", "edrug2-id");
  copyField("cell-line", "ecell-line");
  copyField("drug1-input", "ddrug1-input");
  copyField("drug1-id", "ddrug1-id");
  copyField("drug2-input", "ddrug2-input");
  copyField("drug2-id", "ddrug2-id");
}

function copyField(sourceId, targetId) {
  const source = document.getElementById(sourceId);
  const target = document.getElementById(targetId);
  if (source && target) {
    target.value = source.value;
  }
}

async function runExplain() {
  clearAlert("explain-alert");

  const validation = refreshAllValidation();
  if (!validation.explain) {
    showAlert("explain-alert", "Select valid drugs and a valid cell line before requesting the XAI explanation.");
    return;
  }

  const payload = buildModelPayload("edrug1-id", "edrug1-input", "edrug2-id", "edrug2-input", "ecell-line");
  if (!payload) {
    showAlert("explain-alert", getPredictionInputError("edrug1-input", "edrug2-input", "ecell-line"));
    return;
  }

  setButtonLoading("explain-btn", "explain-btn-text", true, "Generate Explanation", "Computing XAI");
  startExplainLoader();

  try {
    const data = await apiJson(API.explain, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const explanation = normalizeExplanationResponse(data);
    state.lastExplanation = explanation;
    stopExplainLoader();
    renderExplanation(explanation);
  } catch (error) {
    stopExplainLoader({ restorePanel: true });
    showAlert("explain-alert", error.message);
  } finally {
    setButtonLoading("explain-btn", "explain-btn-text", false, "Generate Explanation", "Computing XAI");
    refreshAllValidation();
  }
}

function startExplainLoader() {
  document.getElementById("shap-empty").hidden = true;
  document.getElementById("shap-content").hidden = true;
  startInlineLoader("explain", EXPLAIN_FLOW_MESSAGES);
}

function stopExplainLoader(options = {}) {
  stopInlineLoader("explain");
  if (options.restorePanel) {
    restoreExplainPanel();
  }
}

function restoreExplainPanel() {
  const empty = document.getElementById("shap-empty");
  const content = document.getElementById("shap-content");

  if (state.lastExplanation) {
    empty.hidden = true;
    content.hidden = false;
    return;
  }

  empty.hidden = false;
  content.hidden = true;
}

function normalizeExplanationResponse(data) {
  const positive = normalizeShapRecords(
    data.top_positive_contributors || data.top_synergy_drivers || [],
    1
  );
  const negative = normalizeShapRecords(
    data.top_negative_contributors || data.top_antagonism_drivers || [],
    -1
  );
  const features = negative.concat(positive)
    .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
    .slice(0, 14)
    .reverse();

  return {
    features,
    top_positive_contributors: positive,
    top_negative_contributors: negative,
    input: data.input || {},
    model_used: data.model_used || "",
    model_type: data.model_type || data.input?.model_type || "classical",
    prediction: Number(data.final_predicted_COMBOSCORE ?? data.prediction ?? 0),
    baseValue: data.base_value ?? data.expected_value ?? null,
    summary: data.plain_english_explanation || data.explanation_summary || data.suggestion || "",
    explanationAvailable: data.explanation_available !== false
  };
}

function normalizeShapRecords(records, fallbackSign) {
  return records.map((record) => {
    const rawImpact = record.impact ?? record.shap_value ?? record.shap ?? 0;
    let shap = Number(rawImpact);
    if (!Number.isFinite(shap)) {
      shap = 0;
    }
    if (shap === 0 && fallbackSign) {
      shap = fallbackSign * 0.0001;
    }
    return {
      feature: String(record.readable_feature || record.feature_name || record.feature || "Feature"),
      shap,
      value: record.feature_value ?? record.value ?? ""
    };
  });
}

function renderExplanation(data) {
  document.getElementById("shap-empty").hidden = true;
  document.getElementById("shap-content").hidden = false;

  const themeColors = getThemeChartColors();
  const labels = data.features.map((feature) => feature.feature);
  const values = data.features.map((feature) => Number(feature.shap));
  const colors = values.map((value) => value >= 0 ? "rgba(15, 118, 110, 0.82)" : "rgba(199, 71, 71, 0.82)");
  const borders = values.map((value) => value >= 0 ? "rgba(15, 118, 110, 1)" : "rgba(199, 71, 71, 1)");
  const prediction = Number(data.prediction).toFixed(3);
  const baseValue = data.baseValue === null || data.baseValue === undefined
    ? "n/a"
    : Number(data.baseValue).toFixed(3);

  setText("shap-prediction", prediction);
  setText("shap-base", baseValue);
  setText("shap-base-info", data.summary || (
    data.baseValue === null || data.baseValue === undefined
      ? `Final prediction: ${prediction}. Base value was not returned by the explainer.`
      : `Base value: ${baseValue}. Final prediction: ${prediction}.`
  ));

  const chartShell = document.querySelector(".chart-shell");
  if (!data.features.length && !data.explanationAvailable) {
    if (state.shapChart) {
      state.shapChart.destroy();
      state.shapChart = null;
    }
    if (chartShell) {
      chartShell.innerHTML = `<div class="shap-fallback-list"><p>${escapeHtml(data.summary || "Detailed feature explanation is not available for the selected model.")}</p></div>`;
    }
    return;
  }

  if (!window.Chart) {
    if (state.shapChart) {
      state.shapChart.destroy();
      state.shapChart = null;
    }
    renderShapFallback(chartShell, data.features);
    showAlert("explain-alert", "Chart.js did not load, so a lightweight XAI list is shown instead.");
    return;
  }

  if (chartShell && !document.getElementById("shap-chart")) {
    chartShell.innerHTML = `<canvas id="shap-chart"></canvas>`;
  }

  if (state.shapChart) {
    state.shapChart.destroy();
  }

  const context = document.getElementById("shap-chart").getContext("2d");
  state.shapChart = new Chart(context, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderColor: borders,
        borderWidth: 1,
        borderRadius: 8
      }]
    },
    options: {
      animation: { duration: 500 },
      maintainAspectRatio: false,
      indexAxis: "y",
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (item) => ` XAI impact: ${item.parsed.x.toFixed(4)}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: themeColors.grid },
          ticks: {
            color: themeColors.tick,
            font: { family: "IBM Plex Mono", size: 11 }
          }
        },
        y: {
          grid: { display: false },
          ticks: {
            color: themeColors.label,
            font: { family: "Space Grotesk", size: 12, weight: "600" }
          }
        }
      }
    }
  });

}

function renderShapFallback(container, features) {
  if (!container) {
    return;
  }

  const rows = features.slice(0, 12).map((feature) => {
    const value = Number(feature.shap);
    const safeValue = Number.isFinite(value) ? value : 0;
    const magnitude = Math.min(100, Math.abs(safeValue) * 100);
    const direction = safeValue >= 0 ? "synergy" : "antagonism";
    const directionText = safeValue >= 0 ? "toward synergy" : "toward antagonism";
    return `
      <div class="shap-fallback-row">
        <div>
          <strong>${escapeHtml(feature.feature)}</strong>
          <span>${escapeHtml(directionText)}</span>
        </div>
        <div class="shap-fallback-bar" data-direction="${direction}">
          <i style="width: ${magnitude}%"></i>
        </div>
        <code>${escapeHtml(formatScore(value, 4))}</code>
      </div>
    `;
  }).join("");

  container.innerHTML = `<div class="shap-fallback-list">${rows || "<p>No XAI contributors returned.</p>"}</div>`;
}

async function loadDrugInfo() {
  clearAlert("drug-alert");

  const validation = refreshAllValidation();
  if (!validation.drugs) {
    showAlert("drug-alert", "Select two valid NSC drugs before loading compound details.");
    return;
  }

  const drug1 = resolveDrugId("ddrug1-id", "ddrug1-input");
  const drug2 = resolveDrugId("ddrug2-id", "ddrug2-input");
  const drug1Input = document.getElementById("ddrug1-input")?.value?.trim() || drug1;
  const drug2Input = document.getElementById("ddrug2-input")?.value?.trim() || drug2;

  if (!drug1 || !drug2) {
    showAlert("drug-alert", getDrugInfoInputError("ddrug1-input", "ddrug2-input"));
    return;
  }

  setButtonLoading("drug-info-btn", "dinfo-btn-text", true, "Load Drug Details", "Loading compounds");
  startDrugInfoLoader();

  try {
    const response = await fetch(API.moleculePair, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        NSC1: Number(drug1),
        NSC2: Number(drug2),
        drug1_input: drug1Input,
        drug2_input: drug2Input
      })
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok && !data.NSC1 && !data.NSC2) {
      throw new Error(cleanError(data.error || "Molecule lookup failed."));
    }
    stopDrugInfoLoader();
    renderMoleculeCards(data);
    if (data.status === "error") {
      showAlert("drug-alert", "One or more molecules could not be rendered. See the cards for details.");
    }
  } catch (error) {
    stopDrugInfoLoader({ restorePanel: true });
    showAlert("drug-alert", error.message);
  } finally {
    setButtonLoading("drug-info-btn", "dinfo-btn-text", false, "Load Drug Details", "Loading compounds");
    refreshAllValidation();
  }
}

function startDrugInfoLoader() {
  const area = document.getElementById("drug-cards-area");
  if (area) {
    area.hidden = true;
  }
  startInlineLoader("drug", DRUG_INFO_FLOW_MESSAGES);
}

function stopDrugInfoLoader(options = {}) {
  stopInlineLoader("drug");
  const area = document.getElementById("drug-cards-area");
  if (area) {
    area.hidden = false;
  }
}

function renderMoleculeCards(data) {
  const molecules = [
    normalizeMolecule(data.NSC1 || data.molecule_1, "Drug 1"),
    normalizeMolecule(data.NSC2 || data.molecule_2, "Drug 2")
  ];
  const area = document.getElementById("drug-cards-area");
  area.innerHTML = molecules.map((molecule, index) => moleculeCardShell(molecule, index)).join("");
  molecules.forEach((molecule, index) => {
    const container = document.querySelector(`[data-molecule-slot="${index}"]`);
    if (container && molecule.svg) {
      insertSanitizedSvg(container, molecule.svg);
    }
  });
}

function normalizeMolecule(raw, title) {
  const requested = raw?.requested_nsc ?? raw?.canonical_nsc ?? raw?.nsc ?? raw?.NSC ?? "";
  const used = raw?.used_nsc ?? raw?.resolved_nsc ?? raw?.canonical_nsc ?? requested;
  const drugName = String(raw?.drug_name || raw?.display_name || raw?.name || "").trim();
  const svg = raw?.structure_svg || raw?.svg || "";
  const structureAvailable = Boolean(svg);
  const profileFound = Boolean(raw && (raw.status === "success" || raw.metadata_found || raw.found || raw.molecule_found));
  const structureMessage = raw?.structure_message || raw?.message || raw?.error || "Structure image unavailable.";
  const fallbackDetail = raw?.rdkit_available === false
    ? "Install RDKit to render the 2D structure, formula, and molecular weight."
    : "No 2D structure is available for this compound.";
  return {
    title,
    ok: profileFound,
    requested,
    used,
    inputValue: raw?.input_value || raw?.requested_value || requested,
    drugName,
    aliasUsed: Boolean(raw?.alias_used ?? raw?.used_alias),
    molecularFormula: raw?.molecular_formula || "",
    molecularWeight: raw?.molecular_weight || "",
    source: raw?.source || "",
    svg,
    structureAvailable,
    structureStatus: raw?.structure_status || (structureAvailable ? "available" : "unavailable"),
    structureMessage,
    fallbackDetail,
    rdkitAvailable: raw?.rdkit_available !== false,
    error: raw?.error || raw?.message || "Molecule structure was not found."
  };
}

function moleculeCardShell(molecule, index) {
  if (!molecule.ok) {
    return `
      <article class="drug-card">
        <div class="drug-card-header">
          <h3>${escapeHtml(molecule.title)}</h3>
          <span>${escapeHtml(molecule.error)}</span>
        </div>
      </article>
    `;
  }

  const structureStatusText = molecule.structureAvailable
    ? (molecule.aliasUsed ? `Alias used: NSC ${escapeHtml(String(molecule.used))}` : "Direct molecule match")
    : "Structure image unavailable";
  const structureClass = molecule.structureAvailable
    ? "drug-structure"
    : "drug-structure drug-structure--unavailable";
  const structureContent = molecule.structureAvailable ? "" : `
    <div class="drug-structure-placeholder">
      <strong>Structure image unavailable</strong>
      <span>${escapeHtml(molecule.structureMessage || "No structure image is available for this compound.")}</span>
    </div>
  `;
  const formulaText = molecule.molecularFormula || (molecule.rdkitAvailable ? "n/a" : "Unavailable without RDKit");
  const weightText = molecule.molecularWeight || (molecule.rdkitAvailable ? "n/a" : "Unavailable without RDKit");

  return `
    <article class="drug-card">
      <div class="drug-card-header">
        <h3>${escapeHtml(molecule.title)}: ${escapeHtml(molecule.drugName || `NSC ${molecule.requested}`)}</h3>
        <span>NSC ${escapeHtml(String(molecule.requested))}</span>
        <span>${structureStatusText}</span>
      </div>
      <div class="drug-card-body">
        <div class="${structureClass}" data-molecule-slot="${index}">${structureContent}</div>
        <div class="drug-meta">
          <div class="drug-meta-row"><span>Drug name</span><span>${escapeHtml(molecule.drugName || "n/a")}</span></div>
          <div class="drug-meta-row"><span>NSC ID</span><span>${escapeHtml(String(molecule.requested))}</span></div>
          <div class="drug-meta-row"><span>Input drug value</span><span>${escapeHtml(String(molecule.inputValue || "n/a"))}</span></div>
          <div class="drug-meta-row"><span>Structure match</span><span>${structureStatusText}<small class="drug-meta-badge">${escapeHtml(molecule.structureStatus || "unavailable")}</small></span></div>
          ${molecule.structureAvailable ? "" : `<div class="drug-meta-row drug-meta-row--notice"><span>Fallback note</span><span>${escapeHtml(molecule.fallbackDetail)}</span></div>`}
          <div class="drug-meta-row"><span>Molecular Formula</span><span>${escapeHtml(formulaText)}</span></div>
          <div class="drug-meta-row"><span>Molecular Weight</span><span>${escapeHtml(String(weightText))}</span></div>
          <div class="drug-meta-row"><span>Structure Source</span><span>${escapeHtml(molecule.source || "n/a")}</span></div>
        </div>
      </div>
    </article>
  `;
}

function insertSanitizedSvg(container, svgText) {
  const parser = new DOMParser();
  const doc = parser.parseFromString(svgText, "image/svg+xml");
  const svg = doc.documentElement;
  if (!svg || svg.nodeName.toLowerCase() !== "svg") {
    container.textContent = "SVG could not be rendered.";
    return;
  }

  svg.querySelectorAll("script, foreignObject").forEach((node) => node.remove());
  svg.querySelectorAll("*").forEach((node) => {
    Array.from(node.attributes).forEach((attribute) => {
      if (/^on/i.test(attribute.name)) {
        node.removeAttribute(attribute.name);
      }
    });
  });
  container.replaceChildren(document.importNode(svg, true));
}

function stageBatchFile(file) {
  clearAlert("batch-alert");

  if (file.size > 10 * 1024 * 1024) {
    state.selectedBatchFile = null;
    showAlert("batch-alert", "Maximum upload size is 10 MB.");
    setText("drop-zone-title", "Drop your file here");
    const runButton = document.getElementById("batch-run-btn");
    if (runButton) {
      runButton.disabled = true;
    }
    return;
  }

  state.selectedBatchFile = file;
  setText("drop-zone-title", `Selected ${file.name}`);
  setButtonLoading("batch-run-btn", "batch-run-btn-text", false, "Run Batch Prediction", "Running Batch");
}

function runSelectedBatchFile() {
  if (!state.selectedBatchFile) {
    showAlert("batch-alert", "Select a CSV file before running batch prediction.");
    return;
  }

  processBatchFile(state.selectedBatchFile);
}

async function processBatchFile(file) {
  clearAlert("batch-alert");

  if (file.size > 10 * 1024 * 1024) {
    showAlert("batch-alert", "Maximum upload size is 10 MB.");
    return;
  }

  document.getElementById("drop-zone-title").textContent = `Processing ${file.name}`;
  setButtonLoading("batch-run-btn", "batch-run-btn-text", true, "Run Batch Prediction", "Running Batch");
  startBatchLoader();

  try {
    const uploadFile = await normalizeBatchFile(file);
    const formData = new FormData();
    formData.append("file", uploadFile, uploadFile.name || file.name);

    const response = await fetch(API.batch, { method: "POST", body: formData });
    const contentType = response.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.status === "error") {
        throw new Error(cleanError(data.error || "Batch prediction failed."));
      }
      state.batchBlob = null;
      state.batchDownloadUrl = makeDownloadUrl(data.output_file);
      stopBatchLoader();
      renderBatchJson(data);
    } else {
      if (!response.ok) {
        throw new Error("Batch prediction failed.");
      }
      state.batchBlob = await response.blob();
      state.batchDownloadUrl = "";
      const csvText = await state.batchBlob.text();
      stopBatchLoader();
      renderBatchCsv(csvText);
    }

    document.getElementById("batch-empty").hidden = true;
    document.getElementById("batch-result").classList.add("is-visible");
    state.lastBatchRendered = true;
    requestBatchScrollRefresh();
    document.getElementById("download-btn").disabled = false;
    document.getElementById("drop-zone-title").textContent = `Processed ${file.name}`;
    setButtonLoading("batch-run-btn", "batch-run-btn-text", false, "Run Again", "Running Batch");
  } catch (error) {
    stopBatchLoader({ restorePanel: true });
    showAlert("batch-alert", error.message);
    document.getElementById("drop-zone-title").textContent = `Selected ${file.name}`;
    setButtonLoading("batch-run-btn", "batch-run-btn-text", false, "Run Batch Prediction", "Running Batch");
  }
}

function startBatchLoader() {
  const empty = document.getElementById("batch-empty");
  const result = document.getElementById("batch-result");
  const download = document.getElementById("download-btn");

  if (empty) {
    empty.hidden = true;
  }
  if (result) {
    result.classList.remove("is-visible");
  }
  if (download) {
    download.disabled = true;
  }

  startInlineLoader("batch", BATCH_FLOW_MESSAGES);
}

function stopBatchLoader(options = {}) {
  stopInlineLoader("batch");
  if (options.restorePanel) {
    restoreBatchPanel();
  }
}

function restoreBatchPanel() {
  const empty = document.getElementById("batch-empty");
  const result = document.getElementById("batch-result");
  const download = document.getElementById("download-btn");

  if (state.lastBatchRendered) {
    if (empty) {
      empty.hidden = true;
    }
    if (result) {
      result.classList.add("is-visible");
    }
    if (download) {
      download.disabled = !state.batchBlob && !state.batchDownloadUrl;
    }
    return;
  }

  if (empty) {
    empty.hidden = false;
  }
  if (result) {
    result.classList.remove("is-visible");
  }
  if (download) {
    download.disabled = true;
  }
}

async function normalizeBatchFile(file) {
  const text = await file.text();
  const rows = parseCSV(text);
  if (!rows.length) {
    throw new Error("Uploaded CSV is empty.");
  }

  const headers = rows[0].map((header) => header.trim());
  const canonical = headers.map((header) => {
    const lower = header.toLowerCase();
    if (lower === "drug1_id" || lower === "drug1" || lower === "nsc1") return "NSC1";
    if (lower === "drug2_id" || lower === "drug2" || lower === "nsc2") return "NSC2";
    if (lower === "cell_line" || lower === "cellname") return "CELLNAME";
    return header;
  });

  const required = ["NSC1", "NSC2", "CELLNAME"];
  const indexes = required.map((column) => canonical.indexOf(column));
  if (indexes.some((index) => index === -1)) {
    throw new Error("CSV must include NSC1, NSC2, and CELLNAME.");
  }

  const outputRows = [required].concat(rows.slice(1).map((row) => indexes.map((index) => row[index] ?? "")));
  const csv = outputRows.map((row) => row.map(csvEscape).join(",")).join("\n");
  return new File([csv], file.name.replace(/\.csv$/i, "") + "_normalized.csv", { type: "text/csv" });
}

function renderBatchJson(data) {
  const rows = data.preview || [];
  const scores = rows
    .map((row) => Number(row.final_predicted_COMBOSCORE))
    .filter((value) => Number.isFinite(value));
  const averageScore = scores.length
    ? (scores.reduce((sum, value) => sum + value, 0) / scores.length).toFixed(2)
    : "n/a";

  document.getElementById("batch-stats").innerHTML = `
    <article class="stat-card"><strong>${escapeHtml(String(data.total_rows ?? rows.length))}</strong><span>Total Rows</span></article>
    <article class="stat-card"><strong>${escapeHtml(String(data.successful_rows ?? 0))}</strong><span>Successful</span></article>
    <article class="stat-card"><strong>${escapeHtml(String(data.failed_rows ?? 0))}</strong><span>Failed</span></article>
    <article class="stat-card"><strong>${escapeHtml(String(averageScore))}</strong><span>Preview Avg</span></article>
  `;

  renderBatchTable(rows);
}

function renderBatchCsv(csvText) {
  const rows = parseCSV(csvText);
  if (!rows.length) {
    showAlert("batch-alert", "The returned CSV preview was empty.");
    return;
  }

  const headers = rows[0];
  const dataRows = rows.slice(1);
  const scoreIndex = headers.indexOf("final_predicted_COMBOSCORE");
  const scores = dataRows.map((row) => Number.parseFloat(row[scoreIndex])).filter((value) => Number.isFinite(value));
  const averageScore = scores.length ? (scores.reduce((sum, value) => sum + value, 0) / scores.length).toFixed(2) : "n/a";

  document.getElementById("batch-stats").innerHTML = `
    <article class="stat-card"><strong>${dataRows.length}</strong><span>Total Rows</span></article>
    <article class="stat-card"><strong>${scores.length}</strong><span>Scored Rows</span></article>
    <article class="stat-card"><strong>${averageScore}</strong><span>Average Score</span></article>
    <article class="stat-card"><strong>CSV</strong><span>Download Ready</span></article>
  `;

  const objects = dataRows.slice(0, 50).map((row) => Object.fromEntries(headers.map((header, index) => [header, row[index] ?? ""])));
  renderBatchTable(objects);
}

function renderBatchTable(rows) {
  const tableHead = document.getElementById("batch-thead");
  const tableBody = document.getElementById("batch-tbody");

  if (!rows.length) {
    tableHead.innerHTML = "";
    tableBody.innerHTML = `<tr><td>No preview rows returned.</td></tr>`;
    requestBatchScrollRefresh();
    return;
  }

  const headers = Object.keys(rows[0]);
  tableHead.innerHTML = headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("");
  tableBody.innerHTML = rows.slice(0, 50).map((row) => `
    <tr>
      ${headers.map((header) => {
        const value = row[header] ?? "";
        if (header === "prediction_label" || header === "label") {
          return `<td>${renderBadge(batchLabelForRow(row, value))}</td>`;
        }
        if (header === "status") {
          return `<td>${renderBadge(value)}</td>`;
        }
        if (header === "final_predicted_COMBOSCORE") {
          return `<td class="mono">${escapeHtml(formatMaybeNumber(value))}</td>`;
        }
        return `<td>${escapeHtml(String(value))}</td>`;
      }).join("")}
    </tr>
  `).join("");
  requestBatchScrollRefresh();
}

function batchLabelForRow(row, fallbackLabel) {
  const score = Number(row.final_predicted_COMBOSCORE);
  return Number.isFinite(score) ? semanticLabelForScore(score) : fallbackLabel;
}

function downloadBatchResults() {
  if (state.batchBlob) {
    const url = URL.createObjectURL(state.batchBlob);
    triggerDownload(url, "synergy_predictions.csv");
    URL.revokeObjectURL(url);
    return;
  }

  if (state.batchDownloadUrl) {
    triggerDownload(state.batchDownloadUrl, "synergy_predictions.csv");
  }
}

function downloadSampleBatchCsv() {
  const blob = new Blob([SAMPLE_BATCH_CSV_CONTENT], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  triggerDownload(url, SAMPLE_BATCH_CSV_FILENAME);
  URL.revokeObjectURL(url);
}

function triggerDownload(url, filename) {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function makeDownloadUrl(outputFile) {
  const filename = String(outputFile || "").split(/[\\/]/).pop();
  return filename ? `${API.download}${encodeURIComponent(filename)}` : "";
}

function renderBadge(label) {
  const normalized = String(label || "").toLowerCase();
  let className = "badge badge-neutral";

  if (normalized.includes("synergy") || normalized.includes("synerg") || normalized === "success") {
    className = "badge badge-synergy";
  } else if (normalized.includes("weak") || normalized.includes("neutral")) {
    className = "badge badge-mild";
  } else if (normalized.includes("antag") || normalized === "error" || normalized.includes("fail")) {
    className = "badge badge-antag";
  }

  return `<span class="${className}">${escapeHtml(String(label || ""))}</span>`;
}

function showAlert(id, message) {
  const alert = document.getElementById(id);
  if (!alert) {
    return;
  }
  alert.textContent = message;
  alert.hidden = false;
}

function clearAlert(id) {
  const alert = document.getElementById(id);
  if (!alert) {
    return;
  }
  alert.hidden = true;
  alert.textContent = "";
}

function resolveDrugId(hiddenId, inputId) {
  const hiddenValue = document.getElementById(hiddenId)?.value?.trim() || "";
  if (hiddenValue && (!state.drugIds.size || state.drugIds.has(hiddenValue))) {
    return hiddenValue;
  }

  const typedValue = document.getElementById(inputId)?.value?.trim() || "";
  return resolveKnownDrugId(typedValue);
}

function resolveKnownDrugId(value) {
  const typedValue = String(value ?? "").trim();
  if (/^\d+(?:\.0+)?$/.test(typedValue)) {
    const numericId = String(Number.parseInt(typedValue, 10));
    return state.drugIds.size && !state.drugIds.has(numericId) ? "" : numericId;
  }

  const match = typedValue.match(/\b(\d{2,})\b/);
  if (match) {
    const matchedId = match[1];
    return state.drugIds.size && !state.drugIds.has(matchedId) ? "" : matchedId;
  }

  const normalized = typedValue.toLowerCase();
  const exactNameMatch = state.drugs.find((drug) => (
    drug.name.toLowerCase() === normalized ||
    drug.displayLabel.toLowerCase() === normalized
  ));
  return exactNameMatch?.id || "";
}

function setButtonLoading(buttonId, textId, isLoading, idleText, loadingText = idleText) {
  const button = document.getElementById(buttonId);
  const text = document.getElementById(textId);
  if (!button || !text) {
    return;
  }
  button.disabled = isLoading;
  button.classList.toggle("is-loading", isLoading);
  text.textContent = isLoading ? loadingText : idleText;
}

function setToneClass(element, level) {
  if (!element) {
    return;
  }
  element.classList.remove("tone-success", "tone-good", "tone-mild", "tone-neutral", "tone-danger");
  element.classList.add(`tone-${level || "neutral"}`);
}

function categoryForLabel(label, score) {
  const normalized = String(label || "").toLowerCase();
  if (normalized.includes("strong") && normalized.includes("synergy")) return "strong_synergy";
  if (normalized.includes("strong") && normalized.includes("antag")) return "strong_antagonism";
  const value = Number(score);
  if (Number.isFinite(value)) {
    if (value >= STRONG_EFFECT_THRESHOLD) return "strong_synergy";
    if (value > NEUTRAL_THRESHOLD) return "synergy";
    if (value <= -STRONG_EFFECT_THRESHOLD) return "strong_antagonism";
    if (value < -NEUTRAL_THRESHOLD) return "antagonism";
  }
  if (normalized.includes("synerg")) return "synergy";
  if (normalized.includes("antag")) return "antagonism";
  return "neutral";
}

function labelForScore(score) {
  if (score >= STRONG_EFFECT_THRESHOLD) return "Strong synergy";
  if (score > NEUTRAL_THRESHOLD) return "Synergistic";
  if (score >= -NEUTRAL_THRESHOLD) return "Neutral / weak";
  if (score > -STRONG_EFFECT_THRESHOLD) return "Antagonistic";
  return "Strong antagonism";
}

function colorForCategory(category, score) {
  const normalized = String(category || categoryForLabel("", score)).toLowerCase();
  if (normalized.includes("synergy")) return "#0f766e";
  if (normalized.includes("antagonism")) return "#c74747";
  return "#b7791f";
}

function toneForCategory(category, score) {
  const normalized = String(category || categoryForLabel("", score)).toLowerCase();
  if (normalized.includes("strong_synergy")) return "success";
  if (normalized.includes("synergy")) return "good";
  if (normalized.includes("antagonism")) return "danger";
  return "neutral";
}

function parseCSV(text) {
  const rows = [];
  let row = [];
  let value = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const nextChar = text[i + 1];

    if (char === "\"") {
      if (inQuotes && nextChar === "\"") {
        value += "\"";
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      row.push(value);
      value = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && nextChar === "\n") {
        i += 1;
      }
      row.push(value);
      if (row.some((cell) => cell !== "")) {
        rows.push(row);
      }
      row = [];
      value = "";
    } else {
      value += char;
    }
  }

  if (value !== "" || row.length) {
    row.push(value);
    rows.push(row);
  }

  return rows;
}

function csvEscape(value) {
  const text = String(value ?? "");
  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, "\"\"")}"`;
  }
  return text;
}

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = value;
  }
}

function formatScore(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : String(value ?? "");
}

function formatMaybeNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : String(value ?? "");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatChatText(value) {
  const escaped = escapeHtml(value);
  return escaped
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*\*/g, "")
    .replace(/\r\n|\r|\n/g, "<br>");
}

function chatMessageHtml(message, options = {}) {
  if (options.loading) {
    return `
      <p>
        <span>${formatChatText(message)}</span>
        <span class="chat-typing-dots" aria-hidden="true"><i></i><i></i><i></i></span>
      </p>
    `;
  }
  return `<p>${formatChatText(message)}</p>`;
}

function escapeAttribute(value) {
  return escapeHtml(value).replace(/`/g, "&#96;");
}

function getThemeChartColors() {
  const styles = window.getComputedStyle(document.body);
  return {
    grid: styles.getPropertyValue("--chart-grid").trim() || "rgba(24, 49, 44, 0.07)",
    tick: styles.getPropertyValue("--chart-tick").trim() || "#60746e",
    label: styles.getPropertyValue("--chart-label").trim() || "#18312c"
  };
}

function refreshChartTheme() {
  if (!state.shapChart) {
    return;
  }

  const themeColors = getThemeChartColors();
  state.shapChart.options.scales.x.grid.color = themeColors.grid;
  state.shapChart.options.scales.x.ticks.color = themeColors.tick;
  state.shapChart.options.scales.y.ticks.color = themeColors.label;
  state.shapChart.update();
}
