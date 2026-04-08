const PHASES = [
  { id: "ingestion", title: "Data ingestion", detail: "Parse PDF pages, structure, sections, equations, and references." },
  { id: "preprocessing", title: "Preprocessing", detail: "Clean text, preserve math notation, and build dual representations." },
  { id: "features", title: "Feature engineering", detail: "Extract keywords, entities, and mathematical structures." },
  { id: "models", title: "Transformer models", detail: "Generate summaries, classify sections, and compute similarity." },
  { id: "analysis", title: "Advanced NLP analysis", detail: "Discover topics, build semantic search, and classify equations." },
  { id: "groq", title: "LLM insights", detail: "Produce summary, ELI5, contributions, applications, and limitations." },
  { id: "output", title: "Report generation", detail: "Assemble JSON, markdown report, and visual outputs." }
];

const phaseList = document.getElementById("phaseList");
const fileInput = document.getElementById("fileInput");
const filePicker = document.getElementById("filePicker");
const analyzeButton = document.getElementById("analyzeButton");
const fileRow = document.getElementById("fileRow");
const fileName = document.getElementById("fileName");
const fileDetails = document.getElementById("fileDetails");
const loadDemoButton = document.getElementById("loadDemoButton");
const pipelineStatus = document.getElementById("pipelineStatus");
const pipelineProgressBar = document.getElementById("pipelineProgressBar");
const pipelineLabel = document.getElementById("pipelineLabel");
const pipelineHint = document.getElementById("pipelineHint");
const heroStatus = document.getElementById("heroStatus");
const groqStatus = document.getElementById("groqStatus");
const runtimeStatus = document.getElementById("runtimeStatus");
const groqKeyInput = document.getElementById("groqKeyInput");
const saveGroqButton = document.getElementById("saveGroqButton");
const clearGroqButton = document.getElementById("clearGroqButton");
const groqNote = document.getElementById("groqNote");

let selectedFile = null;
let pollingHandle = null;
let groqSettings = { configured: false };

function renderPhases(job = null) {
  const items = (job?.phases || PHASES.map((phase) => ({ ...phase, status: "idle" })))
    .map((phase, index) => {
      const cls = phase.status === "running" ? "active" : phase.status === "completed" ? "done" : phase.status === "failed" ? "failed" : "";
      const statusText = phase.status === "running"
        ? "Processing now"
        : phase.status === "completed"
          ? "Completed"
          : phase.status === "failed"
            ? "Failed"
            : `Step ${index + 1}`;
      return `
        <article class="phase-card ${cls}">
          <strong>${index + 1}. ${phase.title}</strong>
          <p>${phase.detail}</p>
          <span>${statusText}</span>
        </article>
      `;
    })
    .join("");
  phaseList.innerHTML = items;
}

function cleanupText(text) {
  if (!text) return "";
  return String(text)
    .replace(/â€”/g, "—")
    .replace(/â†’/g, "→")
    .replace(/â‰¥/g, "≥")
    .replace(/â‰¤/g, "≤")
    .replace(/âˆ’/g, "−")
    .replace(/âˆ¼/g, "∼")
    .replace(/Ï€/g, "π")
    .replace(/Î³/g, "γ")
    .replace(/\s+/g, " ")
    .trim();
}

function firstSentences(text, count = 2) {
  const cleaned = cleanupText(text);
  const parts = cleaned.split(/(?<=[.!?])\s+/).filter(Boolean);
  return parts.slice(0, count).join(" ");
}

function extractNumberedList(text) {
  const cleaned = String(text || "");
  const matches = [...cleaned.matchAll(/\d+\.\s+\*\*?([^:*]+?)\*\*?:?\s*([\s\S]*?)(?=\n\d+\.\s+\*\*?|\n?$)/g)];
  if (matches.length) {
    return matches.map((match) => cleanupText(`${match[1]} — ${match[2]}`));
  }
  return cleaned
    .split(/\n+/)
    .map((line) => cleanupText(line.replace(/^\d+\.\s*/, "")))
    .filter((line) => line.length > 30)
    .slice(0, 4);
}

function normalizeReport(report) {
  const keywords = (report.insights?.keywords || []).slice(0, 16);
  const topics = (report.topics?.details || []).slice(0, 6);
  const equationTypes = report.mathematics?.equation_types || {};
  const entities = report.insights?.entities || {};
  const mathStructures = report.mathematics?.math_structures || {};

  return {
    title: cleanupText(report.metadata?.title || "Untitled analysis"),
    subtitle: `${(report.metadata?.authors || []).slice(0, 4).join(", ")} • ${report.metadata?.source_pages || 0} pages • Pipeline v${report.metadata?.pipeline_version || "1.0.0"}`,
    abstract: cleanupText(report.abstract),
    summary: firstSentences(report.llm_analysis?.summary || report.summary?.abstractive || report.abstract, 5),
    eli5: firstSentences(report.llm_analysis?.eli5, 5),
    contributions: extractNumberedList(report.llm_analysis?.contributions),
    applications: firstSentences(report.llm_analysis?.applications, 4),
    limitations: firstSentences(report.llm_analysis?.limitations, 4),
    keywords,
    topics,
    equations: equationTypes,
    entities: {
      persons: (entities.persons || []).slice(0, 6).map(cleanupText),
      methods: (entities.methods || []).slice(0, 6).map(cleanupText),
      concepts: [...(entities.concepts || []), ...(entities.math_entities || [])].slice(0, 6).map(cleanupText)
    },
    mathStructures,
    sections: (report.structure?.sections || []).slice(0, 8),
    visuals: [
      { title: "Keyword frequency", detail: "Top-ranked extracted concepts", src: "/results/keyword_chart.png" },
      { title: "Equation distribution", detail: "Classified equation categories", src: "/results/equation_distribution.png" },
      { title: "Topic distribution", detail: "Discovered semantic clusters", src: "/results/topic_chart.png" }
    ],
    stats: [
      { label: "Pages", value: report.metadata?.source_pages || 0 },
      { label: "Sections", value: report.structure?.num_sections || 0 },
      { label: "Equations", value: report.mathematics?.total_equations || 0 },
      { label: "Topics", value: report.topics?.num_topics || 0 }
    ],
    citations: [
      `Unique references: ${report.citations?.unique_references || 0}`,
      `Citation density: ${report.citations?.citation_density || 0} per 100 words`,
      `Top cited entries captured: ${(report.citations?.top_cited || []).length}`
    ],
    similarity: [
      `Most similar section pairs captured: ${(report.similarity?.most_similar_sections || []).length}`,
      `Least similar section pairs captured: ${(report.similarity?.least_similar_sections || []).length}`,
      "Similarity helps surface repeated arguments, supporting sections, and redundant passages."
    ]
  };
}

function renderMetrics(target, items, className) {
  target.innerHTML = items
    .map((item) => `<div class="${className}"><strong>${item.value}</strong><span>${item.label}</span></div>`)
    .join("");
}

function renderAnalysis(report) {
  const data = normalizeReport(report);
  document.getElementById("paperTitle").textContent = data.title;
  document.getElementById("paperSubtitle").textContent = data.subtitle;
  document.getElementById("summaryText").textContent = data.summary;
  document.getElementById("eli5Text").textContent = data.eli5;
  document.getElementById("applicationsText").textContent = data.applications;
  document.getElementById("limitationsText").textContent = data.limitations;
  document.getElementById("abstractText").textContent = data.abstract;
  renderMetrics(document.getElementById("heroMetrics"), data.stats, "metric-chip");
  renderMetrics(document.getElementById("workspaceStats"), data.stats, "stat-chip");
  document.getElementById("contributionsList").innerHTML = data.contributions.map((item) => `<li>${item}</li>`).join("");
  document.getElementById("keywordCloud").innerHTML = data.keywords
    .map((keyword, index) => `<span class="keyword-pill" style="font-size:${0.86 + (data.keywords.length - index) * 0.012}rem">${cleanupText(keyword)}</span>`)
    .join("");
  document.getElementById("entityColumns").innerHTML = [
    { title: "People", values: data.entities.persons },
    { title: "Methods", values: data.entities.methods },
    { title: "Concepts", values: data.entities.concepts }
  ].map((group) => `
      <div class="entity-column">
        <strong>${group.title}</strong>
        <ul class="entity-list">
          ${group.values.map((value) => `<li class="entity-pill">${value}</li>`).join("") || "<li class='entity-pill'>No data captured</li>"}
        </ul>
      </div>
    `).join("");

  const maxTopicCount = Math.max(...data.topics.map((topic) => topic.count || 0), 1);
  document.getElementById("topicGrid").innerHTML = data.topics.map((topic) => `
      <article class="topic-card">
        <strong>Topic ${topic.id}</strong>
        <p>${cleanupText((topic.keywords || []).slice(0, 5).join(", "))}</p>
        <small>${topic.count} semantic units</small>
        <div class="topic-meter"><span style="width:${Math.max(12, ((topic.count || 0) / maxTopicCount) * 100)}%"></span></div>
      </article>
    `).join("");

  const palette = ["#215cf0", "#6cb8ff", "#ff8b3d", "#3bb273", "#6f5ef9", "#ff5f7d"];
  const equationEntries = Object.entries(data.equations);
  const totalEquations = equationEntries.reduce((sum, [, value]) => sum + (value.count || 0), 0);
  let cursor = 0;
  const stops = equationEntries.map(([name, value], index) => {
    const share = totalEquations ? ((value.count || 0) / totalEquations) * 100 : 0;
    const start = cursor;
    cursor += share;
    return `${palette[index % palette.length]} ${start}% ${cursor}%`;
  });
  document.getElementById("equationDonut").style.background = `conic-gradient(${stops.join(", ") || "#215cf0 0 100%"})`;
  document.getElementById("equationTotal").textContent = totalEquations;
  document.getElementById("equationBreakdown").innerHTML = equationEntries.map(([name, value], index) => `
      <div class="equation-item">
        <div class="legend">
          <span class="swatch" style="background:${palette[index % palette.length]}"></span>
          <strong>${cleanupText(name.replace(/_/g, " "))}</strong>
        </div>
        <span>${value.count} • ${value.percentage}%</span>
      </div>
    `).join("");
  document.getElementById("mathStructureBox").innerHTML = `
    <strong>Detected mathematical structure</strong>
    <p>${cleanupText(data.mathStructures.logical_flow || "No mathematical flow summary available.")}</p>
  `;
  document.getElementById("visualGrid").innerHTML = data.visuals.map((visual) => `
      <figure class="visual-card">
        <img src="${visual.src}" alt="${visual.title}">
        <figcaption><strong>${visual.title}</strong><span>${visual.detail}</span></figcaption>
      </figure>
    `).join("");
  document.getElementById("sectionList").innerHTML = data.sections.map((section) => `
      <div class="section-item">
        <strong>${cleanupText(section.title)}</strong>
        <span>${section.word_count} words • math density ${section.math_density}</span>
      </div>
    `).join("");
  document.getElementById("similarityList").innerHTML = data.similarity.map((item) => `<div class="insight-item">${cleanupText(item)}</div>`).join("");
  document.getElementById("citationList").innerHTML = data.citations.map((item) => `<div class="insight-item">${cleanupText(item)}</div>`).join("");
}

async function requestLatestReport() {
  const response = await fetch("/api/report/latest");
  if (!response.ok) throw new Error("Unable to load latest report");
  return response.json();
}

async function requestSettings() {
  const response = await fetch("/api/settings");
  if (!response.ok) throw new Error("Unable to load settings");
  return response.json();
}

function renderSettings(payload) {
  const groq = payload?.groq || {};
  const runtime = payload?.runtime || {};
  groqSettings = groq;

  const runtimeBits = [];
  if (runtime.device_name) {
    runtimeBits.push(runtime.device_name);
  }
  if (runtime.device_type) {
    runtimeBits.push(`backend: ${runtime.device_type}`);
  }
  if (typeof runtime.memory_gb === "number") {
    runtimeBits.push(`${runtime.memory_gb.toFixed(1)} GB`);
  }
  if (runtime.low_memory_mode) {
    runtimeBits.push("low-memory mode");
  }

  runtimeStatus.textContent = runtimeBits.length
    ? `Runtime detected for this backend: ${runtimeBits.join(" • ")}.`
    : "Runtime detection is not available yet.";

  if (groq.configured) {
    groqStatus.textContent = "Groq ready";
    const source = groq.source === "backend" ? "saved in the backend" : "available from the environment";
    groqNote.textContent = groq.masked_key
      ? `Groq key ${groq.masked_key} is ${source}. New analysis jobs will use it automatically.`
      : `Groq is configured and ${source}.`;
    groqKeyInput.placeholder = "Replace the saved Groq API key";
  } else {
    groqStatus.textContent = "Groq missing";
    groqNote.textContent = "Add a Groq API key to enable Phase 6 LLM insights. The rest of the pipeline can still run without it.";
    groqKeyInput.placeholder = "Paste Groq API key to enable Phase 6 insights";
  }
}

async function saveGroqKey(apiKey) {
  const response = await fetch("/api/settings/groq", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ groq_api_key: apiKey })
  });

  if (!response.ok) {
    throw new Error("Unable to save Groq key");
  }

  return response.json();
}

function setPipeline(job) {
  renderPhases(job);
  pipelineStatus.classList.remove("hidden");
  const phases = job.phases || [];
  const completed = phases.filter((phase) => phase.status === "completed").length;
  const running = phases.find((phase) => phase.status === "running");
  const failed = phases.find((phase) => phase.status === "failed");
  const percent = Math.round((completed / phases.length) * 100 + (running ? 8 : 0));
  pipelineProgressBar.style.width = `${Math.min(percent, 100)}%`;
  pipelineLabel.textContent = failed ? `${failed.title} failed` : running ? running.title : job.status === "completed" ? "Analysis complete" : "Preparing pipeline";
  pipelineHint.textContent = job.message || "The system is moving through the research analysis pipeline.";
  heroStatus.textContent = failed ? "Failed" : job.status === "completed" ? "Complete" : "Processing";
}

async function pollJob(jobId) {
  if (pollingHandle) clearInterval(pollingHandle);
  const tick = async () => {
    const response = await fetch(`/api/jobs/${jobId}`);
    if (!response.ok) return;
    const job = await response.json();
    setPipeline(job);
    if (job.report) renderAnalysis(job.report);
    if (job.status === "completed" || job.status === "failed") {
      clearInterval(pollingHandle);
      pollingHandle = null;
    }
  };
  await tick();
  pollingHandle = setInterval(tick, 1500);
}

async function startAnalysis() {
  if (!selectedFile) return;
  const formData = new FormData();
  formData.append("paper", selectedFile);
  heroStatus.textContent = "Queued";
  pipelineStatus.classList.remove("hidden");
  pipelineLabel.textContent = "Submitting paper";
  pipelineHint.textContent = groqSettings.configured
    ? "Your PDF is being handed to the backend pipeline."
    : "Your PDF is being handed to the backend pipeline. Groq is not configured yet, so Phase 6 insights may be skipped.";
  const response = await fetch("/api/analyze", { method: "POST", body: formData });
  if (!response.ok) {
    let payload = null;
    try {
      payload = await response.json();
    } catch (error) {
      payload = null;
    }
    if (response.status === 409 && payload?.active_job_id) {
      heroStatus.textContent = "Busy";
      pipelineLabel.textContent = "Analysis already running";
      pipelineHint.textContent = payload.message || "Another paper is already being analyzed.";
      pollJob(payload.active_job_id);
      return;
    }
    heroStatus.textContent = "Error";
    pipelineLabel.textContent = "Upload failed";
    pipelineHint.textContent = payload?.message || "The analysis request could not be started.";
    return;
  }
  const payload = await response.json();
  pollJob(payload.job_id);
}

function setupSectionNav() {
  document.querySelectorAll(".section-nav button").forEach((button) => {
    button.addEventListener("click", () => {
      document.getElementById(button.dataset.target)?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
}

filePicker.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  selectedFile = fileInput.files?.[0] || null;
  if (!selectedFile) return;
  fileRow.classList.remove("hidden");
  fileName.textContent = selectedFile.name;
  fileDetails.textContent = `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB • ready for analysis`;
  heroStatus.textContent = "Ready";
});

analyzeButton.addEventListener("click", startAnalysis);
saveGroqButton.addEventListener("click", async () => {
  const apiKey = groqKeyInput.value.trim();
  if (!apiKey) {
    groqNote.textContent = "Paste a Groq API key before saving it.";
    groqKeyInput.focus();
    return;
  }

  saveGroqButton.disabled = true;
  groqNote.textContent = "Saving Groq API key to the backend...";

  try {
    const settings = await saveGroqKey(apiKey);
    groqKeyInput.value = "";
    renderSettings(settings);
  } catch (error) {
    groqNote.textContent = "The Groq API key could not be saved right now.";
  } finally {
    saveGroqButton.disabled = false;
  }
});

clearGroqButton.addEventListener("click", async () => {
  clearGroqButton.disabled = true;
  groqNote.textContent = "Clearing the saved Groq API key...";

  try {
    const settings = await saveGroqKey("");
    groqKeyInput.value = "";
    renderSettings(settings);
  } catch (error) {
    groqNote.textContent = "The saved Groq API key could not be cleared right now.";
  } finally {
    clearGroqButton.disabled = false;
  }
});

loadDemoButton.addEventListener("click", async () => {
  try {
    const report = await requestLatestReport();
    renderAnalysis(report);
    heroStatus.textContent = "Loaded";
    pipelineStatus.classList.add("hidden");
  } catch (error) {
    heroStatus.textContent = "Unavailable";
  }
});

setupSectionNav();
renderPhases();
requestSettings().then(renderSettings).catch(() => {
  groqStatus.textContent = "Unavailable";
  runtimeStatus.textContent = "Backend settings could not be loaded.";
});
requestLatestReport().then(renderAnalysis).catch(() => {});
