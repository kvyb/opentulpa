const $ = (id) => document.getElementById(id);
let current = null;
let logSource = null;
let logCount = 0;
let logStreamId = "";
let lastLogSequence = 0;

$("connect-command").textContent = `opentulpa connect ${location.origin}`;

async function request(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (response.status === 204) return null;
  let payload = {};
  try { payload = await response.json(); } catch (_) { payload = {}; }
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

function setRuntime(status) {
  const dot = $("status-dot");
  dot.className = `dot ${status === "ready" ? "ready" : status === "failed" ? "failed" : ""}`;
  $("runtime-label").textContent = `RUNTIME ${String(status || "stopped").toUpperCase()}`;
}

function setSandbox(sandbox) {
  const panel = $("sandbox-panel");
  if (!panel) return;
  const ok = Boolean(sandbox?.ok);
  const checks = sandbox?.checks || {};
  const failed = Object.entries(checks)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
  panel.classList.remove("hidden", "failed", "ready");
  panel.classList.add(ok ? "ready" : "failed");
  $("sandbox-state").textContent = ok
    ? `SANDBOX READY / ${String(sandbox?.tier || "UNKNOWN").toUpperCase()}`
    : "SANDBOX FAILED";
  $("sandbox-detail").textContent = ok
    ? "Shell and SSH diagnostics can use sandbox execution."
    : `${sandbox?.error || "Sandbox worker is unavailable."}${failed.length ? ` Failed checks: ${failed.join(", ")}.` : ""}`;
}

function showAuth(status) {
  $("auth-panel").classList.remove("hidden");
  const needsClaim = !status.claimed;
  $("claim-form").classList.toggle("hidden", !needsClaim);
  $("login-form").classList.toggle("hidden", needsClaim);
  $("auth-title").innerHTML = needsClaim ? '<span>$</span> claim deployment' : '<span>$</span> unlock host';
  $("auth-copy").textContent = needsClaim
    ? "Local setup needs no token. Remote setup uses the one-time pairing code printed by the server."
    : "Enter the owner token to manage configuration and logs.";
}

function render(status) {
  current = status;
  setRuntime(status.runtime.status);
  setSandbox(status.sandbox);
  if (!status.authenticated) {
    showAuth(status);
    $("config-panel").classList.add("hidden");
    $("log-panel").classList.add("hidden");
    return;
  }
  $("auth-panel").classList.add("hidden");
  $("config-panel").classList.remove("hidden");
  $("log-panel").classList.remove("hidden");
  const config = status.config;
  if (config) {
    $("revision-label").textContent = `ACTIVE REV ${config.revision}`;
    $("base-url").value = config.base_url;
    $("model").value = config.model;
    $("api-key").placeholder = "configured; leave blank to keep";
    $("telegram-token").placeholder = config.telegram_configured
      ? "configured; leave blank to keep"
      : "123456:bot-token";
    $("telegram-user-id").value = config.telegram_user_id || "";
    $("restart-button").classList.remove("hidden");
  }
  startLogs();
}

async function refresh() {
  try { render(await request("/_host/api/status")); }
  catch (error) { $("form-status").textContent = error.message; }
}

function addLog(entry) {
  const incomingStreamId = String(entry.stream_id || "");
  const incomingSequence = Number(entry.sequence || 0);
  if (logStreamId && incomingStreamId && incomingStreamId !== logStreamId) {
    $("logs").replaceChildren();
    logCount = 0;
    lastLogSequence = 0;
  }
  if (incomingStreamId) logStreamId = incomingStreamId;
  if (incomingSequence <= lastLogSequence) return;
  const row = document.createElement("div");
  row.className = "log-line";
  const sequence = document.createElement("span");
  sequence.className = "log-seq";
  sequence.textContent = String(entry.sequence).padStart(4, "0");
  const stream = document.createElement("span");
  stream.className = "log-stream";
  stream.textContent = entry.stream;
  const text = document.createElement("span");
  text.className = "log-text";
  text.textContent = entry.text;
  row.append(sequence, stream, text);
  $("logs").appendChild(row);
  while ($("logs").children.length > 500) $("logs").firstChild.remove();
  lastLogSequence = incomingSequence;
  logCount += 1;
  $("log-count").textContent = `${logCount} LINES`;
  $("logs").scrollTop = $("logs").scrollHeight;
}

async function startLogs() {
  if (logSource) return;
  try {
    const payload = await request("/_host/api/logs");
    payload.logs.forEach(addLog);
    const last = payload.logs.at(-1)?.sequence || 0;
    const streamId = encodeURIComponent(payload.stream_id || "");
    logSource = new EventSource(
      `/_host/api/logs/stream?after=${last}&stream_id=${streamId}`
    );
    logSource.onmessage = (event) => addLog(JSON.parse(event.data));
  } catch (_) { /* Session state will be refreshed by the next owner action. */ }
}

$("claim-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const result = await request("/_host/api/claim", {
      method: "POST",
      body: JSON.stringify({ setup_token: $("setup-token").value || null }),
    });
    $("issued-token-value").textContent = result.owner_token;
    $("issued-connect-command").textContent =
      `opentulpa connect ${location.origin} --token '${result.owner_token}'`;
    $("issued-token").classList.remove("hidden");
    await refresh();
  } catch (error) {
    $("auth-copy").textContent = error.message;
  }
});

$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await request("/_host/api/session", {
      method: "POST",
      body: JSON.stringify({ token: $("owner-token").value }),
    });
    $("owner-token").value = "";
    await refresh();
  } catch (error) {
    $("auth-copy").textContent = error.message;
  }
});

$("config-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("apply-button");
  const status = $("form-status");
  button.disabled = true;
  button.textContent = "ACTIVATING...";
  status.className = "form-status";
  status.textContent = "Validating providers, starting Deep Agents, then activating interfaces.";
  const apiKey = $("api-key").value.trim();
  const telegramToken = $("telegram-token").value.trim();
  const telegramId = $("telegram-user-id").value.trim();
  try {
    await request("/_host/api/config", {
      method: "PUT",
      body: JSON.stringify({
        expected_revision: current.config?.revision || null,
        api_key: apiKey || null,
        base_url: $("base-url").value.trim(),
        model: $("model").value.trim(),
        telegram_bot_token: telegramToken || null,
        telegram_user_id: telegramId ? Number(telegramId) : null,
      }),
    });
    $("api-key").value = "";
    $("telegram-token").value = "";
    status.className = "form-status ok";
    status.textContent = "Runtime active. Connect with the local OpenTulpa terminal.";
    await refresh();
  } catch (error) {
    status.className = "form-status error";
    status.textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = current?.configured ? "APPLY REVISION" : "VALIDATE + START";
  }
});

$("restart-button").addEventListener("click", async () => {
  const status = $("form-status");
  try {
    status.textContent = "Restarting mutable runtime...";
    await request("/_host/api/runtime/restart", { method: "POST", body: "{}" });
    status.className = "form-status ok";
    status.textContent = "Runtime restarted.";
    await refresh();
  } catch (error) {
    status.className = "form-status error";
    status.textContent = error.message;
  }
});

refresh();
