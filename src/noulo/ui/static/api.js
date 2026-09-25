// Thin client for the noulo REST API. Same-origin, relative URLs only.

const KEY_STORAGE = "noulo.apiKey";
let sessionKey = ""; // fallback when localStorage is unavailable

export class ApiError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

export function getApiKey() {
  try {
    return localStorage.getItem(KEY_STORAGE) || "";
  } catch {
    return sessionKey;
  }
}

/** Stores the key; returns false when it could only be kept for this session. */
export function setApiKey(key) {
  sessionKey = key;
  try {
    if (key) localStorage.setItem(KEY_STORAGE, key);
    else localStorage.removeItem(KEY_STORAGE);
    return true;
  } catch {
    return false;
  }
}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function readJson(res) {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

async function send(method, path, body, timeoutMs) {
  const headers = { Accept: "application/json" };
  const key = getApiKey();
  if (key) headers.Authorization = `Bearer ${key}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const init = { method, headers, cache: "no-store" };
  if (body !== undefined) init.body = JSON.stringify(body);
  if (timeoutMs && typeof AbortSignal.timeout === "function") init.signal = AbortSignal.timeout(timeoutMs);
  let res;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(0, "NETWORK", "Cannot reach the noulo server. Check that it is running.");
  }
  return { res, data: await readJson(res) };
}

function toError(res, data) {
  const err = data && data.error;
  if (err && err.message) return new ApiError(res.status, err.code || `HTTP_${res.status}`, err.message);
  const detail = data && typeof data.detail === "string" ? data.detail : res.statusText;
  return new ApiError(res.status, `HTTP_${res.status}`, detail || "The request failed.");
}

/**
 * Sends a request and resolves with { data, headers, elapsed } for 2xx.
 * A 503 ENGINE_BUSY is retried once after 1s; onBusy is called before the retry.
 */
export async function request(method, path, body, { onBusy } = {}) {
  for (let attempt = 0; ; attempt++) {
    const started = performance.now();
    const { res, data } = await send(method, path, body);
    const elapsed = performance.now() - started;
    if (res.ok) return { data, headers: res.headers, elapsed };
    const error = toError(res, data);
    if (error.code === "ENGINE_BUSY" && attempt === 0) {
      if (onBusy) onBusy();
      await delay(1000);
      continue;
    }
    throw error;
  }
}

const call = async (method, path, body) => (await request(method, path, body)).data;

/** Health never throws: resolves to { state: "ready" | "warn" | "down", text }. */
async function health() {
  try {
    const { res, data } = await send("GET", "/health", undefined, 4000);
    if (res.ok && data && data.status === "ok" && data.modelLoaded) return { state: "ready", text: "ready" };
    const text = (data && (data.status || (data.error && data.error.message))) || `HTTP ${res.status}`;
    return { state: "warn", text: String(text) };
  } catch {
    return { state: "down", text: "offline" };
  }
}

export const api = {
  health,
  info: () => call("GET", "/api/v1/info"),
  models: () => call("GET", "/api/v1/models"),
  switchModel: (id) => call("PUT", "/api/v1/models/active", { id }),
  evaluate: (primitive, body, options) =>
    request("POST", `/api/v1/${primitive}?diagnostics=true`, body, options),
  learning: () => call("GET", "/api/v1/learning"),
  setLearning: (enabled) => call("PUT", "/api/v1/learning", { enabled }),
  feedback: (recordId, expected) => call("POST", "/api/v1/feedback", { recordId, expected }),
  records: (limit = 20) => call("GET", `/api/v1/learning/records?limit=${limit}`),
  clearRecords: () => call("DELETE", "/api/v1/learning/records"),
  importExamples: (items) => call("POST", "/api/v1/learning/import", { items }),
};
