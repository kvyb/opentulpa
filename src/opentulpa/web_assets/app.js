const byId = id => document.getElementById(id);
const messages = byId('messages');
const input = byId('input');
const send = byId('send');
const files = byId('files');
const fileQueue = byId('file-queue');
const runIndicator = byId('run-indicator');
const connectionStatus = byId('connection-status');
const connectionLabel = byId('connection-label');
const thread = localStorage.otThread || (localStorage.otThread = `web-${crypto.randomUUID()}`);
let token = '';
let sessionAvailable = true;
let activeRun = '';
let notificationEpoch = 0;
let notificationCursor = 0;
const seenNotifications = new Set();
const pendingApprovalNotifications = new Map();
const pendingRunText = 'Planning next moves';

function dismissEmptyState() {
  byId('empty-state')?.remove();
  messages.classList.add('has-messages');
}

function setConnectionState(state, label) {
  connectionStatus.dataset.state = state;
  connectionLabel.textContent = label;
}

function setRunState(state) {
  runIndicator.dataset.state = state;
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function setComposerBusy(busy) {
  send.disabled = busy;
  input.placeholder = busy ? 'OpenTulpa is working...' : 'Message OpenTulpa...';
}

function resizeInput() {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

function updateFileQueue() {
  const names = Array.from(files.files, file => file.name);
  fileQueue.hidden = names.length === 0;
  fileQueue.textContent = names.length ? `Attachments · ${names.join(' · ')}` : '';
}

function add(role, text, extra = '') {
  dismissEmptyState();
  const element = document.createElement('article');
  element.className = `message ${role} ${extra}`;
  const label = document.createElement('div');
  label.className = 'label';
  label.textContent = role === 'owner' ? 'Owner' : 'OpenTulpa';
  const body = document.createElement('div');
  body.className = 'content';
  body.textContent = text;
  element.append(label, body);
  messages.append(element);
  element.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return body;
}

function addPendingRun() {
  const body = add('agent', pendingRunText);
  body.dataset.pending = 'true';
  body.setAttribute('aria-busy', 'true');
  return body;
}

function clearPendingRun(body) {
  if (body.dataset.pending === 'true') body.textContent = '';
  delete body.dataset.pending;
  body.removeAttribute('aria-busy');
}

function activityLog(body) {
  const article = body.closest('.message');
  let log = article.querySelector('.activity-log');
  if (!log) {
    log = document.createElement('div');
    log.className = 'activity-log';
    article.insertBefore(log, body);
  }
  return log;
}

function activityLine(body, data, state) {
  const log = activityLog(body);
  const callId = String(data.call_id || '');
  let line = callId
    ? Array.from(log.children).find(item => item.dataset.callId === callId)
    : null;
  if (!line) {
    line = document.createElement('div');
    line.className = 'activity-line';
    if (callId) line.dataset.callId = callId;
    log.append(line);
  }
  const name = String(data.name || 'tool').replaceAll('_', ' ');
  line.dataset.state = state;
  line.textContent = state === 'artifact' ? `artifact · ${data.name || 'ready'}` : name;
}

function parse(block) {
  const data = block.split('\n').find(line => line.startsWith('data: '));
  if (!data) return null;
  try { return JSON.parse(data.slice(6)); } catch { return null; }
}

async function consume(response, onEvent) {
  if (!response.ok) {
    const error = new Error((await response.text()) || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split('\n\n');
    buffer = blocks.pop() || '';
    for (const block of blocks) {
      const event = parse(block);
      if (event) onEvent(event);
    }
    if (done) break;
  }
}

function headers() {
  const values = { 'content-type': 'application/json' };
  if (token) values.authorization = `Bearer ${token}`;
  return values;
}

function requestOwnerLogin() {
  sessionAvailable = false;
  setConnectionState('offline', 'token required');
  if (!byId('connection').open) byId('connection').showModal();
}

async function ackNotification(id) {
  const response = await fetch(`/v2/notifications/${id}/ack`, {
    method: 'POST',
    headers: headers(),
  });
  if (!response.ok) throw new Error(`Notification acknowledgement failed: HTTP ${response.status}`);
}

async function resolvedNotificationApproval(id, approvalId) {
  const pending = pendingApprovalNotifications.get(id);
  if (!pending) return;
  pending.delete(approvalId);
  if (!pending.size) {
    await ackNotification(id);
    pendingApprovalNotifications.delete(id);
  }
}

function approval(event, runId = event.run_id || activeRun, notificationId = null) {
  const data = event.data;
  const wrap = document.createElement('article');
  wrap.className = 'approval';
  const label = document.createElement('div');
  label.className = 'label';
  label.textContent = 'Approval required';
  const title = document.createElement('strong');
  title.textContent = data.tool_name || 'Action';
  const description = document.createElement('p');
  description.textContent = data.description || '';
  wrap.append(label, title, description);
  for (const decision of data.allowed_decisions || []) {
    const button = document.createElement('button');
    button.textContent = decision;
    button.className = decision === 'approve' ? '' : 'secondary';
    button.onclick = async () => {
      for (const item of wrap.querySelectorAll('button')) item.disabled = true;
      let edited_arguments = null;
      if (decision === 'edit') {
        const raw = prompt('Edited arguments as JSON', JSON.stringify(data.arguments || {}, null, 2));
        if (raw === null) {
          for (const item of wrap.querySelectorAll('button')) item.disabled = false;
          return;
        }
        try { edited_arguments = JSON.parse(raw); } catch {
          for (const item of wrap.querySelectorAll('button')) item.disabled = false;
          alert('Enter one valid JSON object.');
          return;
        }
      }
      const reply = addPendingRun();
      setRunState('working');
      try {
        const response = await fetch(`/v2/agent/runs/${runId}/resume`, {
          method: 'POST',
          headers: headers(),
          body: JSON.stringify({ approval_id: data.approval_id, decision, edited_arguments }),
        });
        await consume(response, item => render(item, reply));
        if (notificationId !== null) {
          await resolvedNotificationApproval(notificationId, data.approval_id);
        }
        wrap.remove();
      } catch (error) {
        clearPendingRun(reply);
        reply.textContent = `Approval failed. ${error.message}`;
        setRunState('error');
        for (const item of wrap.querySelectorAll('button')) item.disabled = false;
      }
    };
    wrap.append(button);
  }
  messages.append(wrap);
}

function renderNotification(item) {
  if (seenNotifications.has(item.id)) return;
  seenNotifications.add(item.id);
  add('agent', item.text || item.kind);
  const approvals = item.approvals || [];
  if (approvals.length) {
    pendingApprovalNotifications.set(item.id, new Set(approvals.map(value => value.approval_id)));
    for (const data of approvals) approval({ run_id: item.run_id, data }, item.run_id, item.id);
  }
}

const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

async function pollNotifications() {
  const epoch = ++notificationEpoch;
  notificationCursor = 0;
  seenNotifications.clear();
  let waitSeconds = 0;
  while (epoch === notificationEpoch) {
    try {
      const response = await fetch(
        `/v2/notifications?after_id=${notificationCursor}&limit=100&wait_seconds=${waitSeconds}`,
        { headers: headers() },
      );
      if (response.status === 401 || response.status === 503) {
        requestOwnerLogin();
        return;
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      sessionAvailable = true;
      setConnectionState('online', 'connected');
      waitSeconds = 20;
      const payload = await response.json();
      for (const item of payload.notifications || []) {
        renderNotification(item);
        if (!(item.approvals || []).length) await ackNotification(item.id);
        notificationCursor = Math.max(notificationCursor, Number(item.id) || 0);
      }
    } catch {
      if (epoch !== notificationEpoch) return;
      await sleep(1500);
    }
  }
}

async function uploadFiles() {
  const ids = [];
  for (const file of files.files) {
    const form = new FormData();
    form.append('upload', file);
    const uploadHeaders = {
      'Idempotency-Key': `web-file:${thread}:${crypto.randomUUID()}`,
    };
    if (token) uploadHeaders.authorization = `Bearer ${token}`;
    const response = await fetch('/v2/files', {
      method: 'POST',
      headers: uploadHeaders,
      body: form,
    });
    if (!response.ok) throw new Error(`Could not upload ${file.name}`);
    ids.push((await response.json()).file.id);
  }
  files.value = '';
  updateFileQueue();
  return ids;
}

function render(event, body) {
  activeRun = event.run_id || activeRun;
  const data = event.data || {};
  if (event.type === 'run.started' && !body.textContent) {
    setRunState('working');
    body.textContent = pendingRunText;
    body.dataset.pending = 'true';
    body.setAttribute('aria-busy', 'true');
  }
  if (event.type === 'message.delta' && data.text) {
    clearPendingRun(body);
    body.textContent += data.text;
  }
  if (event.type === 'tool.started' && data.name) {
    clearPendingRun(body);
    activityLine(body, data, 'pending');
  }
  if (event.type === 'tool.completed') {
    clearPendingRun(body);
    activityLine(body, data, data.ok === false ? 'error' : 'complete');
  }
  if (event.type === 'artifact.ready') {
    activityLine(body, data, 'artifact');
  }
  if (event.type === 'run.completed') {
    clearPendingRun(body);
    if (!body.textContent) body.textContent = data.text || 'Run completed without a response.';
    setRunState('idle');
  }
  if (event.type === 'run.failed') {
    clearPendingRun(body);
    body.textContent = data.message || 'Run failed.';
    setRunState('error');
  }
  if (event.type === 'approval.required') {
    clearPendingRun(body);
    if (!body.textContent) body.textContent = 'Waiting for your approval.';
    setRunState('approval');
    approval(event, event.run_id);
  }
  scrollToBottom();
}

byId('settings').onclick = () => {
  byId('token').value = token;
  byId('connection').showModal();
};
byId('new-thread').onclick = () => {
  localStorage.otThread = `web-${crypto.randomUUID()}`;
  location.reload();
};
byId('regenerate').onclick = () => {
  if (send.disabled) return;
  input.value = '/regenerate';
  resizeInput();
  byId('composer').requestSubmit();
};
byId('save-token').onclick = () => {
  token = byId('token').value.trim();
  byId('token').value = '';
  sessionAvailable = true;
  setConnectionState('online', 'connected');
  notificationEpoch += 1;
  pollNotifications();
};
byId('composer').onsubmit = async event => {
  event.preventDefault();
  if (send.disabled) return;
  if (!sessionAvailable && !token) {
    requestOwnerLogin();
    return;
  }
  const hasFiles = files.files.length > 0;
  const text = input.value.trim()
    || (hasFiles ? 'Please inspect and respond to the attached files.' : '');
  if (!text) return;
  input.value = '';
  resizeInput();
  add('owner', text);
  const body = addPendingRun();
  setComposerBusy(true);
  setRunState('working');
  try {
    const file_ids = await uploadFiles();
    const response = await fetch('/v2/agent/runs', {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify({ thread_id: thread, text, file_ids }),
    });
    await consume(response, item => render(item, body));
  } catch (error) {
    if (error.status === 401 || error.status === 503) requestOwnerLogin();
    clearPendingRun(body);
    body.textContent = `Connection failed. ${error.message}`;
    setRunState('error');
  } finally {
    setComposerBusy(false);
    input.focus();
  }
};

files.addEventListener('change', updateFileQueue);
input.addEventListener('input', resizeInput);
input.addEventListener('keydown', event => {
  if (event.key !== 'Enter' || event.shiftKey || event.isComposing) return;
  event.preventDefault();
  if (!send.disabled) byId('composer').requestSubmit();
});
input.focus();
resizeInput();
pollNotifications();
