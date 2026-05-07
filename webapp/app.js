const toolConfig = {
  grammar: {
    label: "Gramática",
    title: "Explica una construcción latina",
    placeholder: "Ej.: Explica la diferencia entre ablativo absoluto y cum histórico.",
    chips: [
      "Explica el uso del dativo en: Rex civibus leges dat.",
      "Diferencia entre gerundio y gerundivo.",
      "Explica el ablativo absoluto con un ejemplo.",
    ],
    prefix: "Explica con rigor académico y ejemplos: ",
  },
  declensions: {
    label: "Declinaciones",
    title: "Paradigmas nominales",
    placeholder: "Ej.: servus (m, 2da)",
    chips: ["rosa (f, 1ra)", "servus (m, 2da)", "rex, regis (m, 3ra)", "corpus, corporis (n, 3ra)"],
    prefix: "Declina en todos los casos: ",
  },
  verbs: {
    label: "Verbos",
    title: "Conjugaciones y análisis verbal",
    placeholder: "Ej.: fero en presente de indicativo activo",
    chips: ["amo en presente de indicativo activo", "sum en imperfecto de indicativo", "fero en presente de indicativo activo"],
    prefix: "Conjuga el verbo ",
  },
  dictionary: {
    label: "Diccionario",
    title: "Lema, forma y sentido",
    placeholder: "Ej.: regibus",
    chips: ["regibus", "puellae", "amamus", "bello"],
    prefix: "Analiza morfológicamente y da lema, caso/persona, número y glosa breve: ",
  },
  corrector: {
    label: "Corrector",
    title: "Corrige y explica errores",
    placeholder: "Ej.: Agricola bonum est.",
    chips: ["Puella bonus est.", "Agricola bonum est.", "Dominus servi videt."],
    prefix: "Corrige el siguiente texto y explica cada error gramatical: ",
  },
  practice: {
    label: "Práctica",
    title: "Ejercicios graduados",
    placeholder: "Ej.: crea 5 ejercicios sobre acusativo singular.",
    chips: ["Crea 5 ejercicios sobre nominativo y acusativo.", "Hazme una mini prueba de verbos irregulares.", "Genera un diálogo latino A1."],
    prefix: "Crea una actividad didáctica breve: ",
  },
};

const state = {
  tool: "grammar",
  endpoint: localStorage.getItem("latinTutorEndpoint") || getDefaultEndpoint(),
};

const els = {
  toolList: document.querySelector("#toolList"),
  modeLabel: document.querySelector("#modeLabel"),
  modeTitle: document.querySelector("#modeTitle"),
  statusPill: document.querySelector("#statusPill"),
  userInput: document.querySelector("#userInput"),
  inputLabel: document.querySelector("#inputLabel"),
  quickPrompts: document.querySelector("#quickPrompts"),
  sendButton: document.querySelector("#sendButton"),
  chat: document.querySelector("#chat"),
  settingsButton: document.querySelector("#settingsButton"),
  clearButton: document.querySelector("#clearButton"),
  settingsDialog: document.querySelector("#settingsDialog"),
  endpointInput: document.querySelector("#endpointInput"),
  saveSettingsButton: document.querySelector("#saveSettingsButton"),
};

function init() {
  renderTool();
  updateStatus();
  bindEvents();
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./service-worker.js").catch(() => {});
  }
  window.lucide?.createIcons();
}

function bindEvents() {
  els.toolList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-tool]");
    if (!button) return;
    state.tool = button.dataset.tool;
    renderTool();
  });

  els.quickPrompts.addEventListener("click", (event) => {
    const chip = event.target.closest("[data-prompt]");
    if (!chip) return;
    els.userInput.value = chip.dataset.prompt;
    els.userInput.focus();
  });

  els.sendButton.addEventListener("click", sendPrompt);
  els.userInput.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      sendPrompt();
    }
  });

  els.settingsButton.addEventListener("click", () => {
    els.endpointInput.value = state.endpoint;
    els.settingsDialog.showModal();
  });

  els.saveSettingsButton.addEventListener("click", () => {
    state.endpoint = els.endpointInput.value.trim();
    localStorage.setItem("latinTutorEndpoint", state.endpoint);
    updateStatus();
  });

  els.clearButton.addEventListener("click", () => {
    els.chat.innerHTML = "";
  });
}

function renderTool() {
  const config = toolConfig[state.tool];
  document.querySelectorAll(".tool").forEach((button) => {
    button.classList.toggle("active", button.dataset.tool === state.tool);
  });
  els.modeLabel.textContent = config.label;
  els.modeTitle.textContent = config.title;
  els.userInput.placeholder = config.placeholder;
  els.inputLabel.textContent = `Consulta para ${config.label.toLowerCase()}`;
  els.quickPrompts.innerHTML = config.chips
    .map((chip) => `<button class="chip" data-prompt="${escapeAttr(chip)}">${escapeHtml(chip)}</button>`)
    .join("");
  window.lucide?.createIcons();
}

function updateStatus() {
  const ready = Boolean(state.endpoint);
  els.statusPill.textContent = ready ? "Endpoint listo" : "Sin endpoint";
  els.statusPill.classList.toggle("ready", ready);
}

async function sendPrompt() {
  const raw = els.userInput.value.trim();
  if (!raw) return;

  const config = toolConfig[state.tool];
  const instruction = raw.startsWith("[INST]") ? raw : `${config.prefix}${raw}`;

  addMessage("user", instruction);
  els.userInput.value = "";
  setLoading(true);

  try {
    const answer = state.endpoint ? await callEndpoint(instruction) : demoAnswer(instruction);
    addMessage("assistant", answer);
  } catch (error) {
    addMessage("assistant", `No pude consultar el endpoint.\n\n${error.message}`);
  } finally {
    setLoading(false);
  }
}

async function callEndpoint(instruction) {
  const response = await fetch(state.endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: instruction }),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const data = await response.json();
  return data.answer || data.text || "Respuesta vacía.";
}

function demoAnswer(instruction) {
  return [
    "Modo demo local. Configura un endpoint para usar el LoRA real.",
    "",
    "Prompt preparado:",
    instruction,
    "",
    "Sugerencia: usa el backend FastAPI en `server/latin_tutor_api.py` o un endpoint en Colab.",
  ].join("\n");
}

function addMessage(role, body) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  article.innerHTML = `
    <p class="message-role">${role === "user" ? "Consulta" : "Tutor Latinista"}</p>
    <div class="message-body">${escapeHtml(body)}</div>
  `;
  els.chat.prepend(article);
}

function setLoading(isLoading) {
  els.sendButton.disabled = isLoading;
  els.sendButton.querySelector("span").textContent = isLoading ? "Pensando" : "Enviar";
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  return escapeHtml(value);
}

function getDefaultEndpoint() {
  const host = window.location.hostname;
  const isLocal = ["", "localhost", "127.0.0.1"].includes(host);
  return isLocal ? "" : "/api/generate";
}

init();
