(function (global) {
  const SESSION_ID_KEY = "playWebSessionId";
  const API_SECRETS_KEY = "playWebApiSecrets";
  const SESSION_HEADER = "X-Play-Session-Id";

  function generateUuid() {
    // crypto.randomUUID() is only available in secure contexts (HTTPS/localhost).
    // Fall back to a manual UUID v4 when serving over plain HTTP (e.g. direct VM IP).
    try {
      if (global.crypto && typeof global.crypto.randomUUID === "function") {
        return global.crypto.randomUUID();
      }
    } catch (e) {}
    const getRandom = (n) => {
      if (global.crypto && typeof global.crypto.getRandomValues === "function") {
        return global.crypto.getRandomValues(new Uint8Array(n));
      }
      const arr = new Uint8Array(n);
      for (let i = 0; i < n; i++) arr[i] = Math.floor(Math.random() * 256);
      return arr;
    };
    const b = getRandom(16);
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    const hex = Array.from(b, (x) => x.toString(16).padStart(2, "0"));
    return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10, 16).join("")}`;
  }

  function getOrCreateSessionId() {
    try {
      const existing = sessionStorage.getItem(SESSION_ID_KEY);
      if (existing && existing.length >= 8) return existing;
      const created = generateUuid();
      sessionStorage.setItem(SESSION_ID_KEY, created);
      return created;
    } catch (e) {
      return generateUuid();
    }
  }

  function readApiSecrets() {
    try {
      const raw = sessionStorage.getItem(API_SECRETS_KEY);
      if (!raw) return { hf_token: "", openrouter_api_key: "" };
      const parsed = JSON.parse(raw);
      return {
        hf_token: String(parsed.hf_token || ""),
        openrouter_api_key: String(parsed.openrouter_api_key || ""),
      };
    } catch (e) {
      return { hf_token: "", openrouter_api_key: "" };
    }
  }

  function writeApiSecrets(secrets) {
    const hf = String(secrets?.hf_token || "").trim();
    const openrouter = String(secrets?.openrouter_api_key || "").trim();
    const payload = {};
    if (hf) payload.hf_token = hf;
    if (openrouter) payload.openrouter_api_key = openrouter;
    try {
      if (Object.keys(payload).length > 0) {
        sessionStorage.setItem(API_SECRETS_KEY, JSON.stringify(payload));
      } else {
        sessionStorage.removeItem(API_SECRETS_KEY);
      }
    } catch (e) {}
    return payload;
  }

  function clearApiSecrets() {
    try {
      sessionStorage.removeItem(API_SECRETS_KEY);
    } catch (e) {}
  }

  function sessionHeaders(extraHeaders) {
    return {
      ...(extraHeaders || {}),
      [SESSION_HEADER]: getOrCreateSessionId(),
    };
  }

  function rememberSessionIdFromResponse(resp) {
    const serverId = resp?.headers?.get?.(SESSION_HEADER);
    if (!serverId) return;
    try {
      sessionStorage.setItem(SESSION_ID_KEY, serverId);
    } catch (e) {}
  }

  async function apiFetch(apiBase, path, options) {
    const opts = options || {};
    const headers = sessionHeaders(opts.headers || {});
    const resp = await fetch(`${String(apiBase || "").replace(/\/$/, "")}${path}`, {
      ...opts,
      headers,
    });
    rememberSessionIdFromResponse(resp);
    return resp;
  }

  function wsUrlWithSession(wsBasePath) {
    const base = String(wsBasePath || "").replace(/\?.*$/, "");
    const sessionId = encodeURIComponent(getOrCreateSessionId());
    return `${base}?session_id=${sessionId}`;
  }

  function isPlayDeployed() {
    const path = String(global.location?.pathname || "");
    return path === "/play" || path.startsWith("/play/");
  }

  function resolveApiBase() {
    if (isPlayDeployed()) {
      return `${global.location.origin}/play/api`;
    }
    const host = global.location?.hostname || "127.0.0.1";
    const protocol = global.location?.protocol === "https:" ? "https:" : "http:";
    const port = String(global.PLAY_WEB_API_PORT || "8001");
    return `${protocol}//${host}:${port}/api`;
  }

  function resolveWsUrl() {
    if (isPlayDeployed()) {
      const wsProto = global.location?.protocol === "https:" ? "wss:" : "ws:";
      return `${wsProto}//${global.location.host}/play/ws`;
    }
    const host = global.location?.hostname || "127.0.0.1";
    const wsProto = global.location?.protocol === "https:" ? "wss:" : "ws:";
    const port = String(global.PLAY_WEB_API_PORT || "8001");
    return `${wsProto}//${host}:${port}/ws`;
  }

  global.PlayWebSession = {
    SESSION_HEADER,
    getOrCreateSessionId,
    readApiSecrets,
    writeApiSecrets,
    clearApiSecrets,
    sessionHeaders,
    apiFetch,
    wsUrlWithSession,
    isPlayDeployed,
    resolveApiBase,
    resolveWsUrl,
  };
})(typeof globalThis !== "undefined" ? globalThis : window);
