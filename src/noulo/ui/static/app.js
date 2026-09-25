import { api, ApiError, getApiKey, setApiKey } from "./api.js";

// ---------------------------------------------------------------- constants

const DEFAULT_RUBRIC = ["insignificant", "low", "medium", "high", "critical"];
const LIMITS = { options: 20, rubricMin: 2, rubric: 11 };
const HEALTH_INTERVAL_MS = 5000;

const EXAMPLES = {
  noul: () => ({
    input: "I checked my account and you have taken the subscription payment twice.",
    proposition: "The customer reports being charged more than once.",
  }),
  choice: () => ({
    input: "The customer says their subscription payment was taken twice.",
    question: "Which department should handle this?",
    options: [
      { id: "A", text: "Billing" },
      { id: "B", text: "Technical Support" },
      { id: "C", text: "Sales" },
    ],
  }),
  score: () => ({
    input: "The production system is unavailable for every customer.",
    question: "How severe is this incident?",
    rubric: [...DEFAULT_RUBRIC],
  }),
};

const HINTS = {
  noul: "How likely is the proposition true, given the input? Returns a value from 0 to 1.",
  choice: "Which one of your options fits the input? Returns exactly one option id.",
  score: "Where does the input sit on your rubric? Returns a value from 0 (lowest level) to 1 (highest).",
};

const ICONS = {
  up: '<path d="M4 10l4-4 4 4"/>',
  down: '<path d="M4 6l4 4 4-4"/>',
  remove: '<path d="M4.5 4.5l7 7M11.5 4.5l-7 7"/>',
};

// ---------------------------------------------------------------- state

const state = {
  primitive: "noul",
  drafts: { noul: EXAMPLES.noul(), choice: EXAMPLES.choice(), score: EXAMPLES.score() },
  running: false,
  result: null,
  models: null,
  learningEnabled: null, // null = unknown / unavailable
  recordCount: 0,
};

// ---------------------------------------------------------------- DOM helpers

const $ = (selector) => document.querySelector(selector);

function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "value") node.value = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  return fill(node, ...children);
}

/** Replaces a node's children, skipping null/false and flattening nested arrays. */
function fill(node, ...children) {
  node.replaceChildren(...children.flat(Infinity).filter((c) => c != null && c !== false));
  return node;
}

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = ICONS[name];
  return svg;
}

function setBusy(button, busy) {
  button.disabled = busy;
  button.classList.toggle("is-busy", busy);
  button.setAttribute("aria-busy", String(busy));
}

function toast(message, tone = "ok") {
  const node = h("div", { class: "toast", "data-tone": tone }, message);
  $("#toasts").append(node);
  setTimeout(() => {
    node.classList.add("is-leaving");
    setTimeout(() => node.remove(), 300);
  }, 3500);
}

function showMessage(target, message, code) {
  target.replaceChildren(message || "", code ? h("span", { class: "code" }, code) : "");
}

// ---------------------------------------------------------------- formatting

const clamp01 = (n) => Math.min(1, Math.max(0, Number(n) || 0));
const fixed = (n, digits = 2) => Number(n).toFixed(digits);
const percent = (n) => `${Math.round(clamp01(n) * 100)}%`;
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const noulZone = (v) => (v < 0.35 ? "no" : v > 0.65 ? "yes" : "uncertain");
const ZONE_LABEL = { no: "No", uncertain: "Uncertain", yes: "Yes" };
const nearestLevel = (v, n) => Math.round(clamp01(v) * (n - 1));

function timeAgo(iso) {
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return "";
  const seconds = Math.round((then - Date.now()) / 1000);
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  const steps = [[60, "second"], [60, "minute"], [24, "hour"], [Infinity, "day"]];
  let amount = seconds;
  for (const [size, unit] of steps) {
    if (Math.abs(amount) < size) return rtf.format(amount, unit);
    amount = Math.round(amount / size);
  }
  return "";
}

// ---------------------------------------------------------------- status

async function pollHealthOnce() {
  const previous = $("#status").dataset.state;
  const { state: next, text } = await api.health();
  // Only touch the live region when something changed, so it is not re-announced.
  if (previous !== next || $("#status-text").textContent !== text) {
    $("#status").dataset.state = next;
    $("#status-text").textContent = text;
  }
  if (next === "ready" && (previous === "down" || previous === "warn")) refreshAll();
}

async function pollHealth() {
  await pollHealthOnce();
  setTimeout(pollHealth, HEALTH_INTERVAL_MS);
}

// ---------------------------------------------------------------- evaluate form

const fields = {
  input: $("#f-input"),
  proposition: $("#f-proposition"),
  question: $("#f-question"),
};

const draft = () => state.drafts[state.primitive];

function loadDraft() {
  const d = draft();
  fields.input.value = d.input;
  fields.proposition.value = d.proposition ?? "";
  fields.question.value = d.question ?? "";
  for (const node of document.querySelectorAll("[data-for]")) {
    node.hidden = !node.dataset.for.split(" ").includes(state.primitive);
  }
  $("#primitive-hint").textContent = HINTS[state.primitive];
  renderOptions();
  renderRubric();
  showMessage($("#run-message"), "");
}

function bindFields() {
  for (const [name, node] of Object.entries(fields)) {
    node.addEventListener("input", () => {
      if (name in draft()) draft()[name] = node.value;
    });
  }
  for (const radio of document.querySelectorAll('input[name="primitive"]')) {
    radio.addEventListener("change", () => {
      state.primitive = radio.value;
      loadDraft();
    });
  }
  $("#reset-example").addEventListener("click", () => {
    state.drafts[state.primitive] = EXAMPLES[state.primitive]();
    loadDraft();
    fields.input.focus();
  });
}

function focusRow(list, index, selector) {
  const row = list.children[Math.max(0, index)];
  const target = row && row.querySelector(selector);
  if (target && !target.disabled) target.focus();
}

function nextOptionId(options) {
  const used = new Set(options.map((o) => o.id.trim()));
  for (let i = 0; i < 26; i++) {
    const id = String.fromCharCode(65 + i);
    if (!used.has(id)) return id;
  }
  return `O${options.length + 1}`;
}

function renderOptions() {
  const list = $("#options-list");
  const options = state.drafts.choice.options;
  list.replaceChildren(...options.map((opt, i) => h("li", { class: "row" },
    h("input", {
      class: "row-id", type: "text", value: opt.id, maxlength: "24", spellcheck: "false",
      "aria-label": `Option ${i + 1} id`, oninput: (e) => { opt.id = e.target.value; },
    }),
    h("input", {
      class: "row-text", type: "text", value: opt.text, spellcheck: "false",
      "aria-label": `Option ${i + 1} text`, oninput: (e) => { opt.text = e.target.value; },
    }),
    h("button", {
      class: "icon-btn sm", type: "button", "aria-label": `Remove option ${i + 1}`,
      disabled: options.length <= 1,
      onclick: () => {
        options.splice(i, 1);
        renderOptions();
        focusRow(list, Math.min(i, options.length - 1), ".row-text");
      },
    }, icon("remove")),
  )));
  $("#option-add").disabled = options.length >= LIMITS.options;
}

function renderRubric() {
  const list = $("#rubric-list");
  const rubric = state.drafts.score.rubric;
  const move = (from, to, selector) => {
    [rubric[from], rubric[to]] = [rubric[to], rubric[from]];
    renderRubric();
    focusRow(list, to, selector);
  };
  list.replaceChildren(...rubric.map((level, i) => h("li", { class: "row" },
    h("span", { class: "row-index", "aria-hidden": "true" }, String(i + 1)),
    h("input", {
      class: "row-text", type: "text", value: level, spellcheck: "false",
      "aria-label": `Level ${i + 1} of ${rubric.length}`,
      oninput: (e) => { rubric[i] = e.target.value; },
    }),
    h("button", {
      class: "icon-btn sm move-up", type: "button", "aria-label": `Move level ${i + 1} up`,
      disabled: i === 0, onclick: () => move(i, i - 1, ".move-up"),
    }, icon("up")),
    h("button", {
      class: "icon-btn sm move-down", type: "button", "aria-label": `Move level ${i + 1} down`,
      disabled: i === rubric.length - 1, onclick: () => move(i, i + 1, ".move-down"),
    }, icon("down")),
    h("button", {
      class: "icon-btn sm", type: "button", "aria-label": `Remove level ${i + 1}`,
      disabled: rubric.length <= LIMITS.rubricMin,
      onclick: () => {
        rubric.splice(i, 1);
        renderRubric();
        focusRow(list, Math.min(i, rubric.length - 1), ".row-text");
      },
    }, icon("remove")),
  )));
  $("#rubric-add").disabled = rubric.length >= LIMITS.rubric;
}

function bindListEditors() {
  $("#option-add").addEventListener("click", () => {
    const options = state.drafts.choice.options;
    options.push({ id: nextOptionId(options), text: "" });
    renderOptions();
    focusRow($("#options-list"), options.length - 1, ".row-text");
  });
  $("#rubric-add").addEventListener("click", () => {
    state.drafts.score.rubric.push("");
    renderRubric();
    focusRow($("#rubric-list"), state.drafts.score.rubric.length - 1, ".row-text");
  });
}

/** Validates the current draft; returns { body } or { error, focus }. */
function buildRequest() {
  const d = draft();
  const input = d.input.trim();
  if (!input) return { error: "Enter an input to evaluate.", focus: fields.input };

  if (state.primitive === "noul") {
    const proposition = d.proposition.trim();
    if (!proposition) return { error: "Enter a proposition to test.", focus: fields.proposition };
    return { body: { input, proposition } };
  }

  const question = d.question.trim();
  if (!question) return { error: "Enter a question.", focus: fields.question };

  if (state.primitive === "choice") {
    const choices = d.options.map((o) => ({ id: o.id.trim(), text: o.text.trim() }));
    const blank = choices.findIndex((c) => !c.id || !c.text);
    if (blank >= 0) {
      return { error: `Option ${blank + 1} needs both an id and text.`, focus: $("#options-list").children[blank]?.querySelector("input") };
    }
    const ids = choices.map((c) => c.id);
    const dup = ids.find((id, i) => ids.indexOf(id) !== i);
    if (dup) return { error: `Option id "${dup}" is used twice. Ids must be unique.` };
    return { body: { input, question, choices } };
  }

  const rubric = d.rubric.map((level) => level.trim());
  const blank = rubric.findIndex((level) => !level);
  if (blank >= 0) return { error: `Level ${blank + 1} is empty.`, focus: $("#rubric-list").children[blank]?.querySelector("input") };
  const dup = rubric.find((level, i) => rubric.indexOf(level) !== i);
  if (dup) return { error: `Level "${dup}" appears twice. Levels must be unique.` };
  return { body: { input, question, rubric } };
}

// ---------------------------------------------------------------- run

function showRunError(err) {
  const target = $("#run-message");
  target.dataset.tone = "error";
  const code = err instanceof ApiError ? err.code : "CLIENT_ERROR";
  showMessage(target, err.message || "Something went wrong.", code);
  if (err.status === 401) {
    target.append(" ", h("button", { class: "btn-link", type: "button", onclick: openSettings }, "Set API key"));
  }
}

async function run() {
  if (state.running) return;
  const built = buildRequest();
  const message = $("#run-message");
  if (built.error) {
    message.dataset.tone = "error";
    showMessage(message, built.error);
    if (built.focus) built.focus.focus();
    return;
  }
  const primitive = state.primitive;
  state.running = true;
  setBusy($("#run"), true);
  $("#result").setAttribute("aria-busy", "true");
  showMessage(message, "");
  try {
    const onBusy = () => {
      message.dataset.tone = "busy";
      showMessage(message, "Engine busy, retrying…");
    };
    const res = await api.evaluate(primitive, built.body, { onBusy });
    showMessage(message, "");
    state.result = {
      primitive,
      request: built.body,
      value: res.data.value,
      diagnostics: res.data.diagnostics || null,
      recordId: res.headers.get("X-Record-Id"),
      elapsed: res.elapsed,
    };
    renderResult(state.result);
    if (state.result.recordId) refreshLearning();
  } catch (err) {
    showRunError(err);
  } finally {
    state.running = false;
    setBusy($("#run"), false);
    $("#result").removeAttribute("aria-busy");
  }
}

// ---------------------------------------------------------------- result

function renderResult(r) {
  $("#result-empty").hidden = true;
  $("#result-body").hidden = false;
  const model = r.diagnostics && r.diagnostics.model;
  $("#result-meta").textContent = [`${Math.round(r.elapsed)} ms`, model].filter(Boolean).join(" · ");
  const summary = r.primitive === "choice" ? renderChoiceVerdict(r) : renderScalarVerdict(r);
  renderFeedback(r);
  renderDiagnostics(r);
  $("#announcer").textContent = `${summary}. ${Math.round(r.elapsed)} milliseconds.`;
}

function renderScalarVerdict(r) {
  const v = clamp01(r.value);
  const word = $("#verdict-word");
  $("#verdict-tag").hidden = true;
  $("#verdict-value").textContent = fixed(v);
  let summary;
  if (r.primitive === "noul") {
    const zone = noulZone(v);
    word.textContent = ZONE_LABEL[zone];
    word.dataset.zone = zone;
    $("#verdict-caption").textContent = "Probability the proposition is true";
    buildNoulScale();
    summary = `Noul: ${ZONE_LABEL[zone]}, ${fixed(v)}`;
  } else {
    const levels = r.request.rubric;
    const index = nearestLevel(v, levels.length);
    word.textContent = levels[index];
    word.dataset.zone = "";
    $("#verdict-caption").textContent = `Nearest level: ${index + 1} of ${levels.length}`;
    buildScoreScale(levels, index);
    summary = `Score: ${fixed(v)}, nearest level ${levels[index]}`;
  }
  $("#scale").hidden = false;
  // Next frame so the needle glides from its previous position.
  requestAnimationFrame(() => $("#needle").style.setProperty("--pos", String(v)));
  return summary;
}

function renderChoiceVerdict(r) {
  const option = r.request.choices.find((c) => c.id === r.value);
  const tag = $("#verdict-tag");
  tag.hidden = false;
  tag.textContent = r.value;
  const word = $("#verdict-word");
  word.textContent = option ? option.text : "Unknown option";
  word.dataset.zone = "";
  const probs = r.diagnostics && r.diagnostics.probabilities;
  const p = probs ? probs[r.value] : undefined;
  $("#verdict-value").textContent = typeof p === "number" ? `p ${fixed(p)}` : "";
  $("#verdict-caption").textContent = `Chosen from ${plural(r.request.choices.length, "option")}`;
  $("#scale").hidden = true;
  return `Choice: ${r.value}, ${option ? option.text : ""}`;
}

function scaleLabel(text, pos, extra = "") {
  return h("span", { class: `scale-label ${extra}`.trim(), style: `--at:${pos}` }, text);
}

function buildNoulScale() {
  $("#scale-zones").replaceChildren(
    h("span", { class: "zone", "data-zone": "no", style: "--from:0;--to:0.35" }),
    h("span", { class: "zone", "data-zone": "uncertain", style: "--from:0.35;--to:0.65" }),
    h("span", { class: "zone", "data-zone": "yes", style: "--from:0.65;--to:1" }),
  );
  $("#scale-labels").replaceChildren(
    scaleLabel("0", 0), scaleLabel(".35", 0.35), scaleLabel(".65", 0.65), scaleLabel("1", 1),
  );
}

function buildScoreScale(levels, active) {
  const n = levels.length;
  const compact = n > 6;
  $("#scale-zones").replaceChildren(...levels.map((_, i) =>
    h("span", { class: `level-mark${i === active ? " is-active" : ""}`, style: `--at:${i / (n - 1)}` })));
  $("#scale-labels").replaceChildren(...levels.map((level, i) =>
    scaleLabel(compact ? String(i + 1) : level, i / (n - 1), i === active ? "is-active" : "")));
  $("#scale-labels").style.setProperty("--slot", `${100 / n}%`);
}

// ---------------------------------------------------------------- feedback

function renderFeedback(r) {
  const box = $("#feedback");
  if (!r.recordId) {
    const note = state.learningEnabled === false
      ? "Turn on learning to send feedback on results."
      : "This result was not stored, so it cannot take feedback.";
    box.replaceChildren(h("p", { class: "hint" }, note));
    return;
  }
  const controls = { noul: noulFeedback, choice: choiceFeedback, score: scoreFeedback }[r.primitive](r);
  box.replaceChildren(
    h("p", { class: "feedback-label", id: "feedback-label" }, "Was this right?"),
    h("div", { class: "feedback-controls", role: "group", "aria-labelledby": "feedback-label" }, controls),
    h("p", { class: "feedback-status", id: "feedback-status", "aria-live": "polite" }),
  );
}

async function sendFeedback(r, expected, shown) {
  const buttons = $("#feedback").querySelectorAll("button, select, input");
  buttons.forEach((b) => { b.disabled = true; });
  const status = $("#feedback-status");
  try {
    await api.feedback(r.recordId, expected);
    status.dataset.tone = "ok";
    showMessage(status, `Verified as ${shown}.`);
    toast(`Feedback sent. Verified as ${shown}.`);
    refreshLearning();
  } catch (err) {
    status.dataset.tone = "error";
    showMessage(status, err.message, err.code);
  } finally {
    buttons.forEach((b) => { b.disabled = false; });
  }
}

function noulFeedback(r) {
  const truth = r.value >= 0.5 ? 1 : 0;
  const describe = (v) => (v ? "true (1)" : "false (0)");
  const exact = h("input", {
    id: "feedback-exact", class: "exact-input", type: "number", min: "0", max: "1", step: "0.01",
    inputmode: "decimal", placeholder: "0–1",
  });
  const sendExact = () => {
    const n = Number.parseFloat(exact.value);
    if (!(n >= 0 && n <= 1)) {
      $("#feedback-status").dataset.tone = "error";
      showMessage($("#feedback-status"), "Enter a value between 0 and 1.");
      exact.focus();
      return;
    }
    sendFeedback(r, n, fixed(n));
  };
  exact.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.metaKey && !e.ctrlKey) { e.preventDefault(); sendExact(); }
  });
  return [
    h("button", { class: "btn", type: "button", onclick: () => sendFeedback(r, truth, describe(truth)) }, "Correct"),
    h("button", { class: "btn", type: "button", onclick: () => sendFeedback(r, 1 - truth, describe(1 - truth)) }, "Wrong"),
    h("span", { class: "feedback-or" }, "or"),
    h("label", { class: "sr-only", for: "feedback-exact" }, "Exact value"),
    exact,
    h("button", { class: "btn", type: "button", onclick: sendExact }, "Send value"),
  ];
}

function choiceFeedback(r) {
  const label = (c) => `${c.id} · ${c.text}`;
  const chosen = r.request.choices.find((c) => c.id === r.value);
  const others = r.request.choices.filter((c) => c.id !== r.value);
  const correct = h("button", {
    class: "btn", type: "button", onclick: () => sendFeedback(r, r.value, chosen ? label(chosen) : r.value),
  }, "Correct");
  if (!others.length) return [correct];
  const select = h("select", { id: "feedback-choice" },
    others.map((c) => h("option", { value: c.id }, label(c))));
  const send = () => {
    const pick = others.find((c) => c.id === select.value);
    sendFeedback(r, select.value, label(pick));
  };
  return [
    correct,
    h("span", { class: "feedback-or" }, "or it was"),
    h("label", { class: "sr-only", for: "feedback-choice" }, "Correct option"),
    select,
    h("button", { class: "btn", type: "button", onclick: send }, "Send"),
  ];
}

function scoreFeedback(r) {
  const levels = r.request.rubric;
  const n = levels.length;
  const current = nearestLevel(r.value, n);
  const select = h("select", { id: "feedback-level" },
    levels.map((level, i) => h("option", { value: String(i), selected: i === current }, `${i + 1} · ${level}`)));
  const send = () => {
    const index = Number(select.value);
    sendFeedback(r, index / (n - 1), `${levels[index]} (${fixed(index / (n - 1))})`);
  };
  return [
    h("label", { class: "feedback-or", for: "feedback-level" }, "Correct level"),
    select,
    h("button", { class: "btn", type: "button", onclick: send }, "Send"),
  ];
}

// ---------------------------------------------------------------- diagnostics

function kvPairs(pairs) {
  return pairs.filter(([, v]) => v != null && v !== "").flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, String(v))]);
}

function learningSentence(learning) {
  const matches = Array.isArray(learning.matches) ? learning.matches.length : Number(learning.matches) || 0;
  if (learning.applied) {
    return `Learning moved this result by ${percent(learning.influence)} from ${plural(matches, "similar case")}.`;
  }
  if (matches > 0) return `Learning found ${plural(matches, "similar case")} but did not change this result.`;
  return "Learning did not change this result.";
}

function probabilityBars(probs, highlight) {
  const rows = Object.entries(probs).map(([key, p]) => h("li", { class: `bar-row${key === highlight ? " is-active" : ""}` },
    h("span", { class: "bar-key", title: key }, key),
    h("span", { class: "bar-track" }, h("span", { class: "bar-fill", style: `--w:${clamp01(p)}` })),
    h("span", { class: "bar-value" }, fixed(p)),
  ));
  return [h("h4", { class: "subhead" }, "Probabilities"), h("ul", { class: "bars" }, rows)];
}

function renderDiagnostics(r) {
  const body = $("#diagnostics-body");
  const d = r.diagnostics;
  if (!d) {
    body.replaceChildren(h("p", { class: "hint" }, "The server returned no diagnostics for this result."));
    return;
  }
  const raw = typeof d.raw === "number" ? fixed(d.raw, 4) : null;
  const highlight = r.primitive === "choice" ? r.value
    : r.primitive === "score" ? r.request.rubric[nearestLevel(r.value, r.request.rubric.length)] : null;
  const hasProbs = d.probabilities && typeof d.probabilities === "object" && Object.keys(d.probabilities).length;
  fill(body,
    h("dl", { class: "kv" }, kvPairs([["Model", d.model], ["Backend", d.backend], ["Raw", raw], ["Record", r.recordId]])),
    hasProbs ? probabilityBars(d.probabilities, highlight) : null,
    d.learning ? h("p", { class: "learning-note" }, learningSentence(d.learning)) : null,
  );
}

// ---------------------------------------------------------------- models

function modelOptionLabel(m, active) {
  return [m.id, m.sizeMB && `${m.sizeMB} MB ${m.quantization}`, m.ramMB && `RAM ${m.ramMB} MB`,
    m.local ? null : "remote", m.id === active && "active", !m.installed && "not installed"]
    .filter(Boolean).join(" · ");
}

const pct = (x) => (x == null ? null : `${Math.round(x * 100)}%`);

function modelFacts(m) {
  const rows = [
    ["Tier", { basic: "basic", large: "larger (0.5-1 GB)", llm: "small LLM (4-bit, next-token scoring)",
      experimental: "experimental (measured below the default; not recommended)" }[m.tier] || null],
    ["Download", m.sizeMB ? `${m.sizeMB} MB` : null],
    ["Quantisation", m.quantization],
    ["RAM (measured peak)", m.ramMB ? `${m.ramMB} MB` : m.backend === "openai" ? "remote" : "not measured"],
    ["Noul accuracy", pct(m.noulAccuracy)],
    ["Choice accuracy", pct(m.choiceAccuracy)],
    ["Score error (MAE)", m.scoreMae == null ? null : m.scoreMae.toFixed(3)],
    ["Median latency", m.p50Ms == null ? null : `${Math.round(m.p50Ms)} ms`],
  ].filter(([, value]) => value);
  return h("dl", { class: "kv" }, ...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));
}

function renderModels() {
  const { active, models } = state.models;
  const select = $("#model-select");
  const keep = models.some((m) => m.id === select.value && m.installed) ? select.value : active;
  select.replaceChildren(...models.map((m) =>
    h("option", { value: m.id, disabled: !m.installed, selected: m.id === keep }, modelOptionLabel(m, active))));
  select.disabled = false;
  $("#model-hint").hidden = models.every((m) => m.installed);
  renderModelDetail();
}

function renderModelDetail() {
  const { active, models } = state.models;
  const m = models.find((x) => x.id === $("#model-select").value);
  $("#model-switch").disabled = !m || m.id === active || !m.installed;
  if (!m) { $("#model-detail").replaceChildren(); return; }
  fill($("#model-detail"),
    h("div", { class: "badges" },
      h("span", { class: "badge", "data-kind": m.local ? "local" : "remote" }, m.local ? "local" : "remote"),
      m.id === active ? h("span", { class: "badge", "data-kind": "active" }, "active") : null,
      h("span", { class: "detail-mono" }, [m.backend, m.quantization].filter(Boolean).join(" · "))),
    m.description ? h("p", { class: "hint" }, m.description) : null,
    modelFacts(m),
  );
}

async function loadModels() {
  try {
    state.models = await api.models();
    renderModels();
    showMessage($("#model-message"), "");
  } catch (err) {
    showMessage($("#model-message"), err.message, err.code);
  }
}

async function loadInfo() {
  try {
    const info = await api.info();
    const learning = typeof info.learning === "boolean" ? (info.learning ? "on" : "off") : info.learning;
    $("#info").replaceChildren(...kvPairs([
      ["Name", info.name], ["Version", info.version], ["Model", info.model],
      ["Quantization", info.quantization], ["Backend", info.backend],
      ["Runs", info.local ? "local" : "remote"], ["Device", info.device || "cpu"], ["Learning", learning],
    ]));
  } catch (err) {
    $("#info").replaceChildren(h("dt", {}, "Error"), h("dd", {}, err.message));
  }
}

async function switchModel() {
  const id = $("#model-select").value;
  const button = $("#model-switch");
  setBusy(button, true);
  $("#model-select").disabled = true;
  showMessage($("#model-message"), `Switching to ${id}…`);
  try {
    const res = await api.switchModel(id);
    toast(`Switched to ${(res.active && res.active.id) || id}.`);
    showMessage($("#model-message"), "");
  } catch (err) {
    showMessage($("#model-message"), err.message, err.code);
  } finally {
    button.classList.remove("is-busy");
    button.removeAttribute("aria-busy");
    $("#model-select").disabled = false;
    await Promise.all([loadModels(), loadInfo()]);
    pollHealthOnce();
  }
}

// ---------------------------------------------------------------- learning & memory

function applyLearning(data) {
  const toggle = $("#learning-toggle");
  state.learningEnabled = Boolean(data.enabled);
  toggle.checked = state.learningEnabled;
  toggle.disabled = false;
  const s = data.stats;
  $("#learning-stats").textContent = s
    ? `${plural(s.records ?? 0, "stored case")}, ${s.verified ?? 0} verified · embedder ${s.embedder || "unknown"}`
    : state.learningEnabled ? "No memory statistics available." : "Learning is off. Results are not stored.";
}

async function loadLearning() {
  try {
    applyLearning(await api.learning());
  } catch (err) {
    state.learningEnabled = null;
    $("#learning-toggle").disabled = true;
    $("#learning-stats").textContent = `Learning unavailable: ${err.message}`;
  }
}

async function toggleLearning() {
  const toggle = $("#learning-toggle");
  const wanted = toggle.checked;
  toggle.disabled = true;
  try {
    applyLearning(await api.setLearning(wanted));
    toast(wanted ? "Learning turned on." : "Learning turned off.");
    loadInfo();
    loadRecords();
  } catch (err) {
    toggle.checked = !wanted;
    toggle.disabled = false;
    toast(`${err.message} (${err.code})`, "error");
  }
}

function outcomeText(value, choice) {
  if (typeof value === "number") return fixed(value);
  return choice || null;
}

function taskLabel(task) {
  const parts = String(task || "").split("|");
  return parts.length > 1 ? parts[1] : parts[0];
}

function recordItem(rec) {
  const observed = outcomeText(rec.observedValue, rec.observedChoice);
  const verified = outcomeText(rec.verifiedValue, rec.verifiedChoice);
  const hits = Number(rec.hits) || 0;
  return h("li", { class: "record" },
    h("div", { class: "record-head" },
      h("span", { class: "tag" }, rec.primitive),
      h("span", { class: "record-task", title: rec.task }, taskLabel(rec.task))),
    h("p", { class: "record-input" }, rec.input),
    h("p", { class: "record-meta" },
      observed != null ? `observed ${observed}` : "no observation",
      verified != null ? h("span", { class: "verified" }, ` · verified ${verified}`) : " · unverified",
      ` · ${plural(hits, "hit")}`,
      ` · ${timeAgo(rec.updatedAt || rec.createdAt)}`),
  );
}

async function loadRecords() {
  const list = $("#memory-list");
  try {
    const data = await api.records(20);
    const records = (data && data.records) || [];
    state.recordCount = records.length;
    $("#memory-clear").disabled = records.length === 0;
    list.replaceChildren(...(records.length
      ? records.map(recordItem)
      : [h("li", { class: "empty" }, "No stored cases yet. Run an evaluation with learning on.")]));
    showMessage($("#memory-message"), "");
  } catch (err) {
    list.replaceChildren();
    $("#memory-clear").disabled = true;
    showMessage($("#memory-message"), err.message, err.code);
  }
}

let refreshTimer;
function refreshLearning() {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => { loadLearning(); loadRecords(); }, 250);
}

let disarmTimer;
function armClear(armed) {
  const button = $("#memory-clear");
  clearTimeout(disarmTimer);
  button.dataset.armed = armed ? "true" : "";
  button.textContent = armed ? "Confirm clear" : "Clear memory";
  $("#memory-cancel").hidden = !armed;
  showMessage($("#memory-message"), armed ? "Press Confirm clear to delete every stored case." : "");
  if (armed) disarmTimer = setTimeout(() => armClear(false), 6000);
}

async function clearMemory() {
  const button = $("#memory-clear");
  if (button.dataset.armed !== "true") { armClear(true); return; }
  armClear(false);
  button.disabled = true;
  try {
    const res = await api.clearRecords();
    toast(`Memory cleared. ${plural(Number(res && res.deleted) || 0, "record")} deleted.`);
  } catch (err) {
    showMessage($("#memory-message"), err.message, err.code);
  } finally {
    refreshLearning();
    button.focus();
  }
}

// Teach from a file: JSON Lines (one example per line) or a JSON array / {"examples": [...]}.
// Each example is an /evaluate request plus `expected` (see docs/learning.md).
function parseExamples(text, name) {
  const trimmed = text.trim();
  if (name.endsWith(".json") || trimmed.startsWith("[")) {
    const data = JSON.parse(trimmed);
    const items = Array.isArray(data) ? data : data && data.examples;
    if (!Array.isArray(items)) throw new Error('Expected a JSON array or {"examples": [...]}.');
    return items.map((item, index) => ({ line: index + 1, item }));
  }
  const examples = [];
  text.split(/\r?\n/).forEach((raw, index) => {
    const line = raw.trim();
    if (!line || line.startsWith("#")) return;
    try {
      examples.push({ line: index + 1, item: JSON.parse(line) });
    } catch {
      throw new Error(`Invalid JSON on line ${index + 1}.`);
    }
  });
  return examples;
}

function batchesOf(examples, maxBytes = 56000) {
  const batches = [];
  let batch = [];
  let size = 0;
  for (const example of examples) {
    const bytes = JSON.stringify(example.item).length + 2;
    if (batch.length && (size + bytes > maxBytes || batch.length >= 1000)) {
      batches.push(batch);
      batch = [];
      size = 0;
    }
    batch.push(example);
    size += bytes;
  }
  if (batch.length) batches.push(batch);
  return batches;
}

async function importExamples(file) {
  const result = $("#memory-import-result");
  const button = $("#memory-import");
  let examples;
  try {
    examples = parseExamples(await file.text(), file.name.toLowerCase());
  } catch (err) {
    showMessage(result, `${file.name}: ${err.message}`, "INVALID_FILE");
    return;
  }
  if (!examples.length) {
    showMessage(result, `${file.name} has no examples.`);
    return;
  }
  button.disabled = true;
  showMessage(result, `Teaching ${plural(examples.length, "example")}…`);
  let taught = 0;
  const skipped = [];
  try {
    for (const batch of batchesOf(examples)) {
      const res = await api.importExamples(batch.map((e) => e.item));
      taught += Number(res && res.imported) || 0;
      for (const failure of (res && res.failed) || []) {
        skipped.push(`line ${batch[failure.index].line}: ${failure.message}`);
      }
    }
    toast(`Taught ${plural(taught, "example")} from ${file.name}.`, skipped.length ? "warn" : "ok");
    showMessage(result, skipped.length
      ? `${plural(skipped.length, "example")} not taught. ${skipped.slice(0, 3).join(" · ")}${skipped.length > 3 ? " · …" : ""}`
      : `Taught ${plural(taught, "example")} from ${file.name}.`);
  } catch (err) {
    showMessage(result, err.message, err.code);
  } finally {
    button.disabled = false;
    refreshLearning();
  }
}

// ---------------------------------------------------------------- settings

function syncKeyIndicator() {
  $("#settings-open").dataset.key = getApiKey() ? "set" : "";
}

function openSettings() {
  $("#api-key").value = getApiKey();
  $("#api-key").type = "password";
  $("#api-key-show").checked = false;
  $("#settings").showModal();
  $("#api-key").focus();
}

function saveKey(key) {
  const persisted = setApiKey(key);
  syncKeyIndicator();
  $("#settings").close();
  if (!key) toast("API key removed.");
  else toast(persisted ? "API key saved." : "API key saved for this session only (browser storage is blocked).");
  refreshAll();
}

function bindSettings() {
  $("#settings-open").addEventListener("click", openSettings);
  $("#settings-cancel").addEventListener("click", () => $("#settings").close());
  $("#api-key-remove").addEventListener("click", () => saveKey(""));
  $("#api-key-show").addEventListener("change", (e) => {
    $("#api-key").type = e.target.checked ? "text" : "password";
  });
  $("#settings-form").addEventListener("submit", (e) => {
    e.preventDefault();
    saveKey($("#api-key").value.trim());
  });
  syncKeyIndicator();
}

// ---------------------------------------------------------------- init

function refreshAll() {
  loadModels();
  loadInfo();
  loadLearning();
  loadRecords();
}

function bindShortcuts() {
  const isMac = /Mac|iPhone|iPad/.test(navigator.userAgent);
  $("#run-shortcut").textContent = isMac ? "⌘ Enter" : "Ctrl Enter";
  $("#run").setAttribute("aria-keyshortcuts", isMac ? "Meta+Enter" : "Control+Enter");
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !$("#settings").open) {
      e.preventDefault();
      run();
    }
  });
}

function init() {
  bindFields();
  bindListEditors();
  bindSettings();
  bindShortcuts();
  loadDraft();
  $("#eval-form").addEventListener("submit", (e) => { e.preventDefault(); run(); });
  $("#model-select").addEventListener("change", renderModelDetail);
  $("#model-switch").addEventListener("click", switchModel);
  $("#learning-toggle").addEventListener("change", toggleLearning);
  $("#memory-refresh").addEventListener("click", refreshLearning);
  $("#memory-clear").addEventListener("click", clearMemory);
  $("#memory-import").addEventListener("click", () => $("#memory-file").click());
  $("#memory-file").addEventListener("change", (e) => {
    const [file] = e.target.files;
    if (file) importExamples(file);
    e.target.value = "";
  });
  $("#memory-cancel").addEventListener("click", () => { armClear(false); $("#memory-clear").focus(); });
  pollHealth();
  refreshAll();
}

init();
