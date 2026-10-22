const API = '';
const DEFAULT_DIRECTIVE = 'You are a military intelligence specialist in doctrine.';
let token = localStorage.getItem('cmd_token');
let activeModel = null;
let activeModelType = null;
let messages = [];
/** Per-channel chat history so Qwen / Shepherd / etc. stay isolated */
let chatThreads = {};
/** Per-channel in-flight generations — switching channel must not kill the other */
let generatingByModel = {};
let abortByModel = {};
let activeModelCaps = [];
let attachments = [];
let systemPrompts = JSON.parse(localStorage.getItem('cmd_directives') || '{}');
/** Per-channel vision view so yolo-military / yolo26 stay isolated (same idea as chatThreads) */
let visionByModel = {};
/** Last uploaded image File — re-infer when switching vision channels so labels differ */
let lastVisionImageFile = null;
/** Current mission tag (mission-v1, mission-v2, etc.) */
let currentMissionTag = null;

function ensureThread(modelName) {
  if (!modelName) return [];
  if (!chatThreads[modelName]) chatThreads[modelName] = [];
  return chatThreads[modelName];
}

function ensureVisionState(modelName) {
  if (!modelName) return { mode: 'live', imageUrl: null };
  if (!visionByModel[modelName]) {
    visionByModel[modelName] = { mode: 'live', imageUrl: null };
  }
  return visionByModel[modelName];
}

function revokeVisionImage(state) {
  if (state && state.imageUrl) {
    try { URL.revokeObjectURL(state.imageUrl); } catch {}
    state.imageUrl = null;
  }
}

function isGenerating(modelName) {
  return !!(modelName && generatingByModel[modelName]);
}

function setGenerating(modelName, on) {
  if (!modelName) return;
  if (on) generatingByModel[modelName] = true;
  else delete generatingByModel[modelName];
  refreshModelGeneratingBadges();
  syncSendStopUi();
}

function syncSendStopUi() {
  if (isGenerating(activeModel)) showStopButton();
  else showSendButton();
}

function refreshModelGeneratingBadges() {
  document.querySelectorAll('.model-item').forEach(el => {
    el.classList.toggle('generating', isGenerating(el.dataset.name));
  });
}

// Auth
async function login() {
  const pwd = document.getElementById('login-password').value;
  const status = document.getElementById('login-status');
  const btn = document.getElementById('login-btn');

  if (!pwd) return;
  btn.disabled = true;
  status.className = 'login-status';
  status.textContent = 'AUTHENTICATING...';

  try {
    const res = await fetch(`${API}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password: pwd }),
    });

    if (!res.ok) {
      status.className = 'login-status error';
      status.textContent = 'ACCESS DENIED';
      btn.disabled = false;
      return;
    }

    const data = await res.json();
    token = data.token;
    localStorage.setItem('cmd_token', token);

    status.className = 'login-status success';
    status.textContent = 'ACCESS GRANTED';

    setTimeout(() => showApp(), 600);
  } catch {
    status.className = 'login-status error';
    status.textContent = 'CONNECTION FAILED';
    btn.disabled = false;
  }
}

function logout() {
  token = null;
  localStorage.removeItem('cmd_token');
  activeModel = null;
  activeModelCaps = [];
  messages = [];
  chatThreads = {};
  Object.values(visionByModel).forEach(revokeVisionImage);
  visionByModel = {};
  lastVisionImageFile = null;
  Object.values(abortByModel).forEach(ac => { try { ac.abort(); } catch {} });
  abortByModel = {};
  generatingByModel = {};
  attachments = [];
  document.querySelector('.app-screen').classList.remove('active');
  document.querySelector('.login-screen').classList.remove('hidden');
  document.getElementById('login-password').value = '';
  document.getElementById('login-status').textContent = '';
  document.getElementById('login-btn').disabled = false;
}

function authHeaders() {
  return { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };
}

// App
async function showApp() {
  document.querySelector('.login-screen').classList.add('hidden');
  document.querySelector('.app-screen').classList.add('active');
  await loadModels();
  await refreshMissionStatus();
}

// Missions
function missionLabel(tag) {
  if (!tag) return '';
  return tag.startsWith('mission-') ? tag.slice('mission-'.length).toUpperCase() : tag.toUpperCase();
}

async function refreshMissionStatus() {
  const btn = document.getElementById('btn-mission');
  if (!btn || !token) return;
  try {
    const res = await fetch(`${API}/api/missions`, { headers: authHeaders() });
    if (res.status === 401) { logout(); return; }
    if (!res.ok) return;
    const data = await res.json();
    currentMissionTag = data.current || null;
    if (currentMissionTag) {
      btn.textContent = `MISSION: ${missionLabel(currentMissionTag)}`;
    } else {
      btn.textContent = 'SWITCH MISSION';
    }
  } catch {
    // Leave button as-is on failure
  }
}

async function openMissionModal() {
  const overlay = document.getElementById('mission-overlay');
  const select = document.getElementById('mission-select');
  const statusEl = document.getElementById('mission-status');
  if (!overlay || !select || !statusEl) return;

  statusEl.textContent = '';
  statusEl.className = 'mission-status';
  select.disabled = true;

  try {
    const res = await fetch(`${API}/api/missions`, { headers: authHeaders() });
    if (res.status === 401) { logout(); return; }
    if (!res.ok) {
      statusEl.textContent = 'Failed to load missions.';
      statusEl.classList.add('error');
      return;
    }
    const data = await res.json();
    const available = Array.isArray(data.available) ? data.available : [];
    currentMissionTag = data.current || null;

    select.innerHTML = '';
    if (available.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No missions available';
      select.appendChild(opt);
      select.disabled = true;
    } else {
      available.forEach(tag => {
        const opt = document.createElement('option');
        opt.value = tag;
        opt.textContent = missionLabel(tag);
        select.appendChild(opt);
      });
      select.disabled = false;
      if (currentMissionTag && available.includes(currentMissionTag)) {
        select.value = currentMissionTag;
      }
    }

    overlay.classList.add('active');
  } catch {
    statusEl.textContent = 'Failed to load missions.';
    statusEl.classList.add('error');
  }
}

function closeMissionModal() {
  const overlay = document.getElementById('mission-overlay');
  if (overlay) overlay.classList.remove('active');
}

async function applyMissionSwitch() {
  const select = document.getElementById('mission-select');
  const statusEl = document.getElementById('mission-status');
  const applyBtn = document.getElementById('mission-apply');
  const cancelBtn = document.getElementById('mission-cancel');
  if (!select || !statusEl || !applyBtn || !cancelBtn) return;

  const tag = select.value;
  if (!tag) return;

  statusEl.textContent = `Switching to mission ${missionLabel(tag)}...`;
  statusEl.className = 'mission-status';
  statusEl.classList.add('info');
  select.disabled = true;
  applyBtn.disabled = true;
  cancelBtn.disabled = true;

  try {
    const res = await fetch(`${API}/api/missions/switch`, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ tag }),
    });
    if (res.status === 401) { logout(); return; }
    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      statusEl.textContent = (detail || 'Mission switch failed.').slice(0, 200);
      statusEl.className = 'mission-status error';
      select.disabled = false;
      applyBtn.disabled = false;
      cancelBtn.disabled = false;
      return;
    }
    const data = await res.json();
    currentMissionTag = data.current || tag;
    statusEl.textContent = `Mission switched to ${missionLabel(currentMissionTag)}. Pods restarting...`;
    statusEl.className = 'mission-status success';
    await refreshMissionStatus();
    // Give the user a moment to read the status, then close
    setTimeout(closeMissionModal, 1200);
  } catch {
    statusEl.textContent = 'Mission switch failed.';
    statusEl.className = 'mission-status error';
    select.disabled = false;
    applyBtn.disabled = false;
    cancelBtn.disabled = false;
  }
}

async function loadModels() {
  const list = document.getElementById('model-list');
  try {
    const res = await fetch(`${API}/api/models`, { headers: authHeaders() });
    if (res.status === 401) { logout(); return; }
    const data = await res.json();

    list.innerHTML = '';
    const count = document.getElementById('model-count');
    const online = data.models.filter(m => m.ready).length;
    count.textContent = `${online}/${data.models.length} ONLINE`;

    data.models.forEach(m => {
      const el = document.createElement('div');
      el.className = [
        'model-item',
        m.ready ? '' : 'offline',
        activeModel === m.name ? 'active' : '',
        isGenerating(m.name) ? 'generating' : '',
      ].filter(Boolean).join(' ');
      el.dataset.name = m.name;
      el.innerHTML = `
        <div class="model-dot ${m.ready ? 'online' : 'offline'}"></div>
        <div class="model-info">
          <div class="model-name">${esc(m.display || m.name)}</div>
          <div class="model-id">${esc(m.description || m.name)}</div>
        </div>`;
      if (m.ready) {
        el.addEventListener('click', () => selectModel(m));
      }
      list.appendChild(el);
    });
  } catch {
    list.innerHTML = '<div style="padding:16px;color:var(--red);font-size:11px">FAILED TO LOAD UNITS</div>';
  }
}

function toggleSidebar() {
  document.querySelector('.sidebar').classList.toggle('open');
  document.getElementById('sidebar-backdrop').classList.toggle('open');
}

function closeSidebar() {
  document.querySelector('.sidebar').classList.remove('open');
  document.getElementById('sidebar-backdrop').classList.remove('open');
}

function isVisionModel(modelType) {
  return modelType === 'object-detection';
}

function selectModel(m) {
  // Do not abort other channels — generations continue in the background
  activeModel = m.name;
  activeModelType = m.model_type || '';
  activeModelCaps = m.capabilities || [];
  attachments = [];
  renderAttachments();
  document.querySelectorAll('.model-item').forEach(el => {
    el.classList.toggle('active', el.dataset.name === m.name);
  });
  document.getElementById('chat-target').textContent = `CHANNEL: ${(m.display || m.name).toUpperCase()}`;
  closeSidebar();

  if (isVisionModel(activeModelType)) {
    showVisionFeed();
    // One upload → switch channel → fresh inference (don’t reuse the other model’s JPEG)
    const st = ensureVisionState(activeModel);
    if (lastVisionImageFile && st.mode !== 'video') {
      handleVisionImageUpload(lastVisionImageFile);
    }
  } else {
    hideVisionFeed();
    messages = ensureThread(activeModel);
    document.getElementById('chat-input').focus();
    document.getElementById('btn-directive').style.display = '';
    updateAttachButton();
    syncSendStopUi();
    renderMessages();
  }
}

function showVisionFeed() {
  document.getElementById('chat-messages').style.display = 'none';
  document.getElementById('chat-input-area').style.display = 'none';
  document.getElementById('btn-directive').style.display = 'none';
  const feed = document.getElementById('vision-feed');
  feed.style.display = 'flex';
  applyVisionState(activeModel);
}

function hideVisionFeed() {
  const feed = document.getElementById('vision-feed');
  feed.style.display = 'none';
  const stream = document.getElementById('vision-stream');
  stream.onload = null;
  stream.onerror = null;
  stream.src = '';
  setVisionBrief('');
  document.getElementById('chat-messages').style.display = '';
  document.getElementById('chat-input-area').style.display = '';
}

function visionStreamUrl(kind, modelName) {
  // Cache-bust: browsers reuse the same MJPEG URL and never attach the new video
  const q = new URLSearchParams({
    token: token || '',
    model: modelName || '',
    _: String(Date.now()),
  });
  if (kind === 'video') return `${API}/api/vision/video/stream?${q}`;
  return `${API}/api/vision/stream?${q}`;
}

function applyVisionState(modelName) {
  const state = ensureVisionState(modelName);
  const stream = document.getElementById('vision-stream');
  const overlay = document.getElementById('vision-overlay');
  const indicator = document.getElementById('vision-indicator');
  const status = document.getElementById('vision-status');
  const label = document.getElementById('vision-label');
  const connectId = (state._connectId = (state._connectId || 0) + 1);

  stream.onload = null;
  stream.onerror = null;

  if (state.mode === 'image' && state.imageUrl) {
    label.textContent = 'UPLOADED IMAGE — INFERENCE RESULT';
    overlay.style.display = 'none';
    indicator.className = 'vision-indicator live';
    if (!state.briefPending) setVisionBrief(state.brief || '');
    stream.onload = () => {
      overlay.style.display = 'none';
      indicator.className = 'vision-indicator live';
    };
    stream.onerror = () => {
      overlay.style.display = 'flex';
      status.textContent = 'RESULT UNAVAILABLE';
      indicator.className = 'vision-indicator offline';
    };
    stream.src = state.imageUrl;
    return;
  }

  if (state.mode === 'video') {
    label.textContent = 'VIDEO INFERENCE FEED';
    overlay.style.display = 'flex';
    status.textContent = 'CONNECTING TO FEED...';
    indicator.className = 'vision-indicator connecting';
    if (!state.briefPending) setVisionBrief(state.brief || '');
    // multipart MJPEG often never fires img.onload — clear overlay after attach
    stream.onerror = () => {
      if (state._connectId !== connectId || state.mode !== 'video') return;
      overlay.style.display = 'flex';
      status.textContent = 'FEED UNAVAILABLE';
      indicator.className = 'vision-indicator offline';
    };
    stream.src = visionStreamUrl('video', modelName);
    setTimeout(() => {
      if (state._connectId !== connectId || state.mode !== 'video') return;
      if (indicator.classList.contains('offline')) return;
      if (state.briefPending) return;
      overlay.style.display = 'none';
      indicator.className = 'vision-indicator live';
    }, 600);
    return;
  }

  // live (default)
  state.mode = 'live';
  label.textContent = 'LIVE INFERENCE FEED';
  overlay.style.display = 'flex';
  status.textContent = 'CONNECTING TO FEED...';
  indicator.className = 'vision-indicator connecting';
  if (!state.briefPending) setVisionBrief(state.brief || '');
  stream.onload = () => {
    if (state._connectId !== connectId) return;
    overlay.style.display = 'none';
    indicator.className = 'vision-indicator live';
  };
  stream.onerror = () => {
    if (state._connectId !== connectId || state.mode !== 'live') return;
    overlay.style.display = 'flex';
    status.textContent = 'FEED UNAVAILABLE';
    indicator.className = 'vision-indicator offline';
  };
  stream.src = visionStreamUrl('live', modelName);
}

function triggerVisionUpload() {
  document.getElementById('vision-file-input').click();
}

/** Drop MJPEG connection + server-side player so a second upload can start cleanly */
async function teardownVisionPlayback() {
  const stream = document.getElementById('vision-stream');
  if (stream) {
    stream.onload = null;
    stream.onerror = null;
    // Do not assign src='' (loads the HTML page as an image and fires onerror)
    stream.removeAttribute('src');
  }
  try {
    await fetch(`${API}/api/vision/video`, {
      method: 'DELETE',
      headers: authHeaders(),
    });
  } catch {}
  // Let the browser drop the old socket before we open a new one
  await new Promise(r => setTimeout(r, 150));
}

async function handleVisionFileSelected(e) {
  const file = e.target.files[0];
  if (!file) return;
  e.target.value = '';

  if (file.type.startsWith('image/')) {
    await handleVisionImageUpload(file);
  } else {
    await handleVisionVideoUpload(file);
  }
}

function bumpBriefGen(state) {
  state.briefGen = (state.briefGen || 0) + 1;
  state.briefPending = true;
  state.brief = '';
  return state.briefGen;
}

function clearVisionBriefState(state) {
  if (!state) return;
  state.briefGen = (state.briefGen || 0) + 1;
  state.briefPending = false;
  state.brief = '';
}

function formatVisionBriefHtml(text) {
  const esc = (s) => String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  const bold = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  const raw = String(text || '').trim();
  if (!raw) return '';

  const lines = raw.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const labelRe = /^(Present|Threat|Risk|Action|What is present|Likely threat|Immediate risk|Recommended action)\s*[:\-—]\s*(.*)$/i;
  const labeled = [];
  for (const line of lines) {
    const cleaned = line.replace(/^[-*•]\s+/, '').replace(/^\d+\.\s+/, '');
    const m = cleaned.match(labelRe);
    if (!m) break;
    const key = m[1].toLowerCase();
    let label = 'Present';
    if (key.includes('threat')) label = 'Threat';
    else if (key.includes('risk')) label = 'Risk';
    else if (key.includes('action')) label = 'Action';
    else if (key.includes('present')) label = 'Present';
    labeled.push({ label, value: m[2] || '' });
  }
  if (labeled.length >= 3) {
    return `<div class="brief-rows">${labeled.map(r =>
      `<div class="brief-row"><div class="brief-label">${esc(r.label)}</div>` +
      `<div class="brief-value">${bold(r.value)}</div></div>`
    ).join('')}</div>`;
  }

  const bullets = lines.filter(l => /^[-*•]\s+/.test(l) || /^\d+\.\s+/.test(l));
  if (bullets.length >= 2) {
    return `<ul class="brief-list">${bullets.map(l =>
      `<li>${bold(l.replace(/^[-*•]\s+/, '').replace(/^\d+\.\s+/, ''))}</li>`
    ).join('')}</ul>`;
  }

  return `<p class="brief-fallback">${bold(raw)}</p>`;
}

function setVisionBrief(text) {
  const el = document.getElementById('vision-brief');
  if (!el) return;
  if (!text) {
    el.style.display = 'none';
    el.textContent = '';
    return;
  }
  el.innerHTML = formatVisionBriefHtml(text);
  el.style.display = 'block';
}

async function requestVisionBrief(detections) {
  if (!detections || !detections.length) {
    return { brief: 'No objects detected.', detections: detections || [] };
  }
  const briefRes = await fetch(`${API}/api/vision/brief`, {
    method: 'POST',
    headers: { ...authHeaders() },
    body: JSON.stringify({ detections, model_endpoint: 'shepherd-alpha' }),
  });
  if (!briefRes.ok) {
    const detail = await briefRes.text().catch(() => '');
    throw new Error(detail || `Brief HTTP ${briefRes.status}`);
  }
  const briefJson = await briefRes.json();
  return { brief: briefJson.brief, detections };
}

async function detectJsonFallback(file, channel) {
  const fd = new FormData();
  fd.append('file', file);
  const detRes = await fetch(
    `${API}/api/vision/detect?model=${encodeURIComponent(channel)}`,
    { method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: fd },
  );
  if (!detRes.ok) return [];
  const detJson = await detRes.json();
  return detJson.detections || [];
}

async function applyBriefResult(channel, gen, briefText, opts = {}) {
  const state = ensureVisionState(channel);
  if (state.briefGen !== gen) return false; // stale upload
  state.briefPending = false;
  state.brief = briefText || '';
  if (activeModel === channel) {
    if (state.brief) setVisionBrief(state.brief);
    else setVisionBrief(opts.emptyMsg || 'Brief unavailable (empty response).');
    if (opts.hideOverlay) {
      document.getElementById('vision-overlay').style.display = 'none';
      document.getElementById('vision-indicator').className = 'vision-indicator live';
    }
  }
  return true;
}

async function runShepherdBrief(channel, gen, detections, opts = {}) {
  const viewActive = () => activeModel === channel;
  const state = ensureVisionState(channel);
  if (viewActive()) {
    document.getElementById('vision-status').textContent = 'SHEPHERD BRIEFING...';
    document.getElementById('vision-overlay').style.display = 'flex';
    document.getElementById('vision-indicator').className = 'vision-indicator connecting';
  }
  try {
    const briefResult = await requestVisionBrief(detections);
    await applyBriefResult(channel, gen, (briefResult && briefResult.brief) || '', {
      hideOverlay: opts.hideOverlay !== false,
    });
  } catch (briefErr) {
    console.warn('Vision brief skipped:', briefErr);
    if (state.briefGen !== gen) return;
    state.briefPending = false;
    state.brief = '';
    if (viewActive()) {
      setVisionBrief('Brief failed: ' + (briefErr.message || 'unknown error').slice(0, 180));
      if (opts.hideOverlay !== false) {
        document.getElementById('vision-overlay').style.display = 'none';
        document.getElementById('vision-indicator').className = 'vision-indicator live';
      }
    }
  }
}

async function pollFeedDetections(channel, gen, attempts = 8) {
  for (let i = 0; i < attempts; i++) {
    const state = ensureVisionState(channel);
    if (state.briefGen !== gen) return null;
    try {
      const res = await fetch(
        `${API}/api/vision/detections?model=${encodeURIComponent(channel)}`,
        { headers: authHeaders() },
      );
      if (res.ok) {
        const data = await res.json();
        if ((data.detections || []).length) return data.detections;
      }
    } catch {}
    await new Promise(r => setTimeout(r, 700));
  }
  return [];
}

async function handleVisionImageUpload(file) {
  const channel = activeModel;
  if (!channel) return;
  lastVisionImageFile = file;

  const overlay = document.getElementById('vision-overlay');
  const status = document.getElementById('vision-status');
  const indicator = document.getElementById('vision-indicator');
  const viewActive = () => activeModel === channel;
  const state = ensureVisionState(channel);
  const gen = bumpBriefGen(state);

  if (viewActive()) {
    overlay.style.display = 'flex';
    status.textContent = 'PROCESSING UPLOAD...';
    indicator.className = 'vision-indicator connecting';
    setVisionBrief('');
  }

  const fd = new FormData();
  fd.append('file', file);

  let detectUrl = `${API}/api/vision/detect/annotate?model=${encodeURIComponent(channel)}`;

  try {
    const res = await fetch(detectUrl, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: fd,
    });
    if (res.status === 401) { logout(); return; }
    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new Error(detail || `HTTP ${res.status}`);
    }
    if (state.briefGen !== gen) return;

    let detections = [];
    const detectionsHdr = res.headers.get('X-Detections');
    if (detectionsHdr) {
      try { detections = JSON.parse(detectionsHdr); } catch (e) {
        console.warn('X-Detections parse failed', e);
      }
    }

    const blob = await res.blob();
    if (state.briefGen !== gen) return;
    const url = URL.createObjectURL(blob);
    revokeVisionImage(state);
    state.mode = 'image';
    state.imageUrl = url;
    if (viewActive()) {
      applyVisionState(channel);
      status.textContent = 'SHEPHERD BRIEFING...';
      overlay.style.display = 'flex';
    }

    // Header may be stripped by a proxy — fall back to JSON detect for images
    if (!detections.length) {
      try {
        detections = await detectJsonFallback(file, channel);
      } catch (e) {
        console.warn('detect fallback failed', e);
      }
    }
    if (state.briefGen !== gen) return;
    await runShepherdBrief(channel, gen, detections);
  } catch (err) {
    console.error('Image upload error:', err);
    if (state.briefGen !== gen) return;
    state.briefPending = false;
    if (viewActive()) {
      overlay.style.display = 'flex';
      status.textContent = 'UPLOAD FAILED: ' + (err.message || 'Unknown error');
      indicator.className = 'vision-indicator offline';
    }
  }
}

async function handleVisionVideoUpload(file) {
  const channel = activeModel;
  if (!channel) return;

  const overlay = document.getElementById('vision-overlay');
  const status = document.getElementById('vision-status');
  const indicator = document.getElementById('vision-indicator');
  const viewActive = () => activeModel === channel;
  const state = ensureVisionState(channel);
  const gen = bumpBriefGen(state);

  if (viewActive()) {
    overlay.style.display = 'flex';
    status.textContent = 'PROCESSING UPLOAD...';
    indicator.className = 'vision-indicator connecting';
    setVisionBrief('');
  }

  // Must close the previous MJPEG stream before POST — otherwise the server
  // stays busy and the next upload appears stuck / ignored.
  await teardownVisionPlayback();
  Object.keys(visionByModel).forEach(name => {
    if (visionByModel[name].mode === 'video') visionByModel[name].mode = 'live';
  });

  const fd = new FormData();
  fd.append('file', file);

  let videoUrl = `${API}/api/vision/video?model=${encodeURIComponent(channel)}`;

  const controller = new AbortController();
  const uploadTimeout = setTimeout(() => controller.abort(), 120000);

  try {
    const res = await fetch(videoUrl, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: fd,
      signal: controller.signal,
    });
    clearTimeout(uploadTimeout);
    if (res.status === 401) { logout(); return; }
    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new Error(detail || `HTTP ${res.status}`);
    }
    if (state.briefGen !== gen) return;

    revokeVisionImage(state);
    state.mode = 'video';
    // Brief pause so DELETE/teardown sockets are gone, then force a new MJPEG URL
    if (viewActive()) {
      await new Promise(r => setTimeout(r, 200));
      applyVisionState(channel);
      status.textContent = 'SHEPHERD BRIEFING...';
      overlay.style.display = 'flex';
      indicator.className = 'vision-indicator connecting';
    }

    const detections = await pollFeedDetections(channel, gen);
    if (state.briefGen !== gen) return;
    await runShepherdBrief(channel, gen, detections || []);
  } catch (err) {
    clearTimeout(uploadTimeout);
    console.error('Video upload error:', err);
    if (state.briefGen !== gen) return;
    state.briefPending = false;
    if (viewActive()) {
      overlay.style.display = 'flex';
      const msg = err.name === 'AbortError' ? 'UPLOAD TIMED OUT' : 'VIDEO UPLOAD FAILED: ' + (err.message || 'Unknown error');
      status.textContent = msg;
      indicator.className = 'vision-indicator offline';
    }
  }
}

async function backToLiveFeed() {
  const channel = activeModel;
  if (!channel) return;
  const state = ensureVisionState(channel);
  clearVisionBriefState(state);
  setVisionBrief('');
  await teardownVisionPlayback();
  Object.keys(visionByModel).forEach(name => {
    if (visionByModel[name].mode === 'video') visionByModel[name].mode = 'live';
  });
  revokeVisionImage(state);
  state.mode = 'live';
  lastVisionImageFile = null;
  applyVisionState(channel);
}

function getDirective(modelName) {
  return systemPrompts[modelName] !== undefined ? systemPrompts[modelName] : DEFAULT_DIRECTIVE;
}

function saveDirective(modelName, prompt) {
  systemPrompts[modelName] = prompt;
  localStorage.setItem('cmd_directives', JSON.stringify(systemPrompts));
}

function showDirectiveEditor() {
  if (!activeModel) return;
  document.getElementById('directive-input').value = getDirective(activeModel);
  document.getElementById('directive-overlay').classList.add('active');
  document.getElementById('directive-input').focus();
}

function hideDirectiveEditor() {
  document.getElementById('directive-overlay').classList.remove('active');
}

// Attachments
function updateAttachButton() {
  const btn = document.getElementById('btn-attach');
  const canAttach = activeModelCaps.includes('vision') || activeModelCaps.includes('audio-transcription');
  btn.style.display = canAttach ? '' : 'none';
  const input = document.getElementById('file-input');
  const accept = [];
  if (activeModelCaps.includes('vision')) accept.push('image/*');
  if (activeModelCaps.includes('audio-transcription')) accept.push('audio/*');
  input.accept = accept.join(',');
}

function triggerFileInput() {
  document.getElementById('file-input').click();
}

function handleFileSelected(e) {
  Array.from(e.target.files).forEach(file => {
    const reader = new FileReader();
    reader.onload = () => {
      attachments.push({
        type: file.type.startsWith('image/') ? 'image' : 'audio',
        file,
        dataUrl: reader.result,
        name: file.name,
      });
      renderAttachments();
    };
    reader.readAsDataURL(file);
  });
  e.target.value = '';
}

function removeAttachment(index) {
  attachments.splice(index, 1);
  renderAttachments();
}

function renderAttachments() {
  const strip = document.getElementById('attachment-strip');
  if (attachments.length === 0) {
    strip.innerHTML = '';
    strip.style.display = 'none';
    return;
  }
  strip.style.display = 'flex';
  strip.innerHTML = attachments.map((a, i) => {
    if (a.type === 'image') {
      return `<div class="attachment-preview">
        <img src="${a.dataUrl}" alt="${esc(a.name)}">
        <button class="attachment-remove" onclick="removeAttachment(${i})">&times;</button>
      </div>`;
    }
    return `<div class="attachment-preview audio">
      <span class="attachment-audio-label">AUDIO</span>
      <span class="attachment-audio-name">${esc(a.name)}</span>
      <button class="attachment-remove" onclick="removeAttachment(${i})">&times;</button>
    </div>`;
  }).join('');
}

// Chat
let streamingEl = null;
let streamStats = null;

function formatStatsHtml(stats, streaming) {
  const now = streaming ? performance.now() : stats.endTime;
  const elapsed = ((now - stats.startTime) / 1000).toFixed(1);
  const parts = [];

  if (stats.firstTokenTime) {
    parts.push(`TTFT ${((stats.firstTokenTime - stats.startTime) / 1000).toFixed(2)}s`);
  }
  if (stats.firstContentTime && stats.firstTokenTime &&
      stats.firstContentTime - stats.firstTokenTime > 50) {
    parts.push(`TTFO ${((stats.firstContentTime - stats.startTime) / 1000).toFixed(2)}s`);
  }

  const tokens = stats.completionTokens || stats.tokenCount;
  if (tokens > 0) {
    parts.push(`${tokens} TOKENS`);
    const genTime = stats.firstTokenTime ? (now - stats.firstTokenTime) / 1000 : 0;
    if (genTime > 0.1) {
      parts.push(`${(tokens / genTime).toFixed(1)} T/s`);
    }
  }
  if (stats.promptTokens != null) {
    parts.push(`${stats.promptTokens} PROMPT`);
  }
  parts.push(`${elapsed}s`);
  if (streaming) parts.push('<span class="stats-live">LIVE</span>');

  return parts.join(' <span class="stats-sep">|</span> ');
}

function buildMessageEl(msg) {
  const el = document.createElement('div');
  el.className = `message ${msg.role}${msg.blocked ? ' blocked' : ''}`;
  const label = msg.role === 'user' ? 'OPERATOR' : msg.blocked ? 'GUARDRAIL' : 'AI UNIT';
  let body = '';
  if (msg.thinking) {
    body += `<details class="message-thinking"><summary>REASONING LOG</summary>${esc(msg.thinking)}</details>`;
  }
  if (msg.attachments && msg.attachments.length > 0) {
    body += '<div class="message-attachments">';
    msg.attachments.forEach(a => {
      if (a.type === 'image') {
        body += `<img class="message-image" src="${a.dataUrl}" alt="${esc(a.name)}">`;
      } else {
        body += `<div class="message-audio-file">AUDIO: ${esc(a.name)}</div>`;
      }
    });
    body += '</div>';
  }
  if (msg.transcriptions && msg.transcriptions.length > 0) {
    msg.transcriptions.forEach(t => {
      body += `<div class="message-transcription"><span class="transcription-label">TRANSCRIPT</span>${esc(t)}</div>`;
    });
  }
  body += formatContent(msg.content || '');
  if (msg.stats) {
    body += `<div class="message-stats">${formatStatsHtml(msg.stats, false)}</div>`;
  }
  el.innerHTML = `<div class="message-label">${label}</div><div class="message-body">${body}</div>`;
  return el;
}

function renderMessages() {
  const container = document.getElementById('chat-messages');
  streamingEl = null;

  if (!activeModel) {
    container.innerHTML = '<div class="no-model-selected">SELECT A UNIT TO BEGIN</div>';
    return;
  }

  const channelBusy = isGenerating(activeModel);

  if (messages.length === 0 && !channelBusy) {
    container.innerHTML = `
      <div class="chat-empty">
        <div class="chat-empty-icon">&gt;_</div>
        <div class="chat-empty-text">AWAITING TRANSMISSION</div>
      </div>`;
    return;
  }

  container.innerHTML = '';
  messages.forEach(msg => container.appendChild(buildMessageEl(msg)));

  if (channelBusy) {
    const el = document.createElement('div');
    el.className = 'message assistant';
    el.innerHTML = `
      <div class="message-label">AI UNIT</div>
      <div class="message-body">
        <span class="typing-indicator">
          <span>PROCESSING</span>
          <span class="dot"></span><span class="dot"></span><span class="dot"></span>
        </span>
      </div>`;
    container.appendChild(el);
  }

  container.scrollTop = container.scrollHeight;
}

function updateStreamingMessage(msg) {
  const container = document.getElementById('chat-messages');

  if (!streamingEl) {
    const loading = container.querySelector('.typing-indicator');
    if (loading) loading.closest('.message').remove();
    streamingEl = document.createElement('div');
    streamingEl.className = 'message assistant';
    streamingEl.innerHTML = '<div class="message-label">AI UNIT</div><div class="message-body"></div>';
    container.appendChild(streamingEl);
  }

  if (msg.blocked) {
    streamingEl.className = 'message assistant blocked';
    streamingEl.querySelector('.message-label').textContent = 'GUARDRAIL';
  }

  const body = streamingEl.querySelector('.message-body');
  let thinkingEl = body.querySelector('.message-thinking');

  if (msg.thinking) {
    if (!thinkingEl) {
      thinkingEl = document.createElement('details');
      thinkingEl.className = 'message-thinking';
      thinkingEl.open = true;
      thinkingEl.innerHTML = '<summary>REASONING LOG</summary><div class="thinking-content"></div>';
      body.prepend(thinkingEl);
    }
    thinkingEl.querySelector('.thinking-content').textContent = msg.thinking;
  }

  let contentEl = body.querySelector('.content-area');
  if (!contentEl) {
    contentEl = document.createElement('span');
    contentEl.className = 'content-area';
    body.appendChild(contentEl);
  }
  contentEl.innerHTML = formatContent(msg.content || '');

  if (streamStats) {
    let statsEl = body.querySelector('.message-stats');
    if (!statsEl) {
      statsEl = document.createElement('div');
      statsEl.className = 'message-stats';
      body.appendChild(statsEl);
    }
    statsEl.innerHTML = formatStatsHtml(streamStats, true);
  }

  container.scrollTop = container.scrollHeight;
}

function parseGuardrailError(err) {
  const inner = err.error || err;
  const pf = inner.provider_specific_fields || err.provider_specific_fields;
  if (!pf || !pf.guardrail_name) return null;
  const name = pf.guardrail_name;
  const desc = pf.description || pf.error || inner.message || 'Content blocked';
  const severity = pf.severity ? pf.severity.toUpperCase() : null;
  const matched = pf.matched_phrase || pf.keyword || null;
  let text = `BLOCKED BY ${name.toUpperCase()}`;
  if (severity) text += ` [${severity}]`;
  text += `\n${desc}`;
  if (matched) text += `\nMatched: "${matched}"`;
  return text;
}

function formatContent(text) {
  if (!text) return '';
  let html = esc(text);
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/^#### (.+)$/gm, '<div class="md-heading md-h4">$1</div>');
  html = html.replace(/^### (.+)$/gm, '<div class="md-heading md-h3">$1</div>');
  html = html.replace(/^## (.+)$/gm, '<div class="md-heading md-h2">$1</div>');
  html = html.replace(/^# (.+)$/gm, '<div class="md-heading md-h1">$1</div>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  html = html.replace(/^- (.+)$/gm, '<div class="md-list-item">$1</div>');
  html = html.replace(/\n/g, '<br>');
  return html;
}

function esc(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function showStopButton() {
  document.getElementById('btn-send').style.display = 'none';
  document.getElementById('btn-stop').style.display = '';
}

function showSendButton() {
  document.getElementById('btn-stop').style.display = 'none';
  document.getElementById('btn-send').style.display = '';
}

function stopGenerating() {
  const ac = abortByModel[activeModel];
  if (ac) ac.abort();
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if ((!text && attachments.length === 0) || !activeModel || isGenerating(activeModel)) return;

  const threadModel = activeModel;
  const thread = ensureThread(threadModel);
  messages = thread;

  input.value = '';
  input.style.height = 'auto';

  const currentAttachments = [...attachments];
  attachments = [];
  renderAttachments();

  const userMsg = { role: 'user', content: text, attachments: currentAttachments };
  thread.push(userMsg);
  setGenerating(threadModel, true);
  renderMessages();

  const audioFiles = currentAttachments.filter(a => a.type === 'audio');
  const transcriptions = [];
  for (const a of audioFiles) {
    try {
      const fd = new FormData();
      fd.append('file', a.file);
      fd.append('model_endpoint', activeModel);
      const res = await fetch(`${API}/api/audio/transcribe`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: fd,
      });
      if (res.status === 401) { logout(); return; }
      if (res.ok) {
        const data = await res.json();
        transcriptions.push(data.text || '');
      } else {
        transcriptions.push('[TRANSCRIPTION FAILED]');
      }
    } catch {
      transcriptions.push('[TRANSCRIPTION FAILED]');
    }
  }
  if (transcriptions.length > 0) {
    userMsg.transcriptions = transcriptions;
    renderMessages();
  }

  const imageFiles = currentAttachments.filter(a => a.type === 'image');
  const allText = [text, ...transcriptions.filter(t => !t.startsWith('['))].filter(Boolean).join('\n\n');

  if (imageFiles.length > 0) {
    const parts = [];
    if (allText) parts.push({ type: 'text', text: allText });
    imageFiles.forEach(a => parts.push({ type: 'image_url', image_url: { url: a.dataUrl } }));
    userMsg.apiContent = parts;
  } else if (allText !== text) {
    userMsg.apiContent = allText;
  }

  const directive = getDirective(threadModel);
  const apiMessages = [];
  if (directive) {
    apiMessages.push({ role: 'system', content: directive });
  }
  thread
    .filter(m => {
      if (m.blocked) return false;
      const c = m.apiContent !== undefined ? m.apiContent : m.content;
      if (Array.isArray(c)) return c.length > 0;
      return c && !String(c).includes('[TRANSMISSION ERROR:');
    })
    .forEach(m => apiMessages.push({ role: m.role, content: m.apiContent !== undefined ? m.apiContent : m.content }));

  const stats = {
    startTime: performance.now(),
    firstTokenTime: null,
    firstContentTime: null,
    endTime: null,
    tokenCount: 0,
    promptTokens: null,
    completionTokens: null,
  };
  streamStats = stats;
  let assistantMsg = null;
  const viewActive = () => activeModel === threadModel;

  const abortController = new AbortController();
  abortByModel[threadModel] = abortController;

  try {
    const res = await fetch(`${API}/api/chat/completions`, {
      method: 'POST',
      headers: authHeaders(),
      signal: abortController.signal,
      body: JSON.stringify({
        model_endpoint: threadModel,
        model: 'default',
        messages: apiMessages,
        stream: true,
        stream_options: { include_usage: true },
      }),
    });

    if (res.status === 401) { logout(); return; }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    assistantMsg = { role: 'assistant', content: '', thinking: '', stats: null };
    thread.push(assistantMsg);

    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') break;

        try {
          const chunk = JSON.parse(data);

          if (chunk.error) {
            let errObj = chunk.error;
            if (typeof errObj.message === 'string') {
              try { errObj = JSON.parse(errObj.message); } catch {}
            }
            const guardrail = parseGuardrailError(errObj);
            if (guardrail) {
              assistantMsg.blocked = true;
              assistantMsg.content = guardrail;
              userMsg.blocked = true;
              if (viewActive()) updateStreamingMessage(assistantMsg);
              continue;
            }
            throw new Error(chunk.error.message || 'upstream error');
          }

          const delta = chunk.choices?.[0]?.delta;
          if (delta) {
            if (delta.reasoning_content) {
              assistantMsg.thinking += delta.reasoning_content;
              stats.tokenCount++;
              if (!stats.firstTokenTime) stats.firstTokenTime = performance.now();
            }
            if (delta.content) {
              assistantMsg.content += delta.content;
              stats.tokenCount++;
              if (!stats.firstTokenTime) stats.firstTokenTime = performance.now();
              if (!stats.firstContentTime) stats.firstContentTime = performance.now();
            }
            if (viewActive()) updateStreamingMessage(assistantMsg);
          }

          if (chunk.usage) {
            stats.promptTokens = chunk.usage.prompt_tokens;
            stats.completionTokens = chunk.usage.completion_tokens;
          }
        } catch (e) {
          if (e instanceof SyntaxError) continue;
          throw e;
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      if (assistantMsg && !assistantMsg.content && !assistantMsg.thinking) {
        const idx = thread.lastIndexOf(assistantMsg);
        if (idx >= 0) thread.splice(idx, 1);
      }
    } else {
      let guardrailMsg = null;
      try {
        const parsed = JSON.parse(err.message);
        guardrailMsg = parseGuardrailError(parsed);
      } catch {}
      if (guardrailMsg) {
        userMsg.blocked = true;
        if (assistantMsg) {
          assistantMsg.blocked = true;
          assistantMsg.content = guardrailMsg;
        } else {
          thread.push({ role: 'assistant', content: guardrailMsg, blocked: true });
        }
      } else if (assistantMsg) {
        assistantMsg.content += (assistantMsg.content ? '\n\n' : '') + `[TRANSMISSION ERROR: ${err.message}]`;
      } else {
        thread.push({ role: 'assistant', content: `[TRANSMISSION ERROR: ${err.message}]` });
      }
    }
  }

  stats.endTime = performance.now();
  if (assistantMsg) assistantMsg.stats = stats;
  if (streamStats === stats) streamStats = null;
  delete abortByModel[threadModel];
  setGenerating(threadModel, false);
  if (viewActive()) {
    messages = thread;
    renderMessages();
  }
}

async function clearChat() {
  if (isVisionModel(activeModelType)) {
    await backToLiveFeed();
    return;
  }
  if (activeModel) {
    chatThreads[activeModel] = [];
    messages = chatThreads[activeModel];
  } else {
    messages = [];
  }
  attachments = [];
  renderAttachments();
  renderMessages();
}

// Input handling
function handleInput(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

// Init
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('login-btn').addEventListener('click', login);
  document.getElementById('login-password').addEventListener('keydown', e => {
    if (e.key === 'Enter') login();
  });
  document.getElementById('btn-attach').addEventListener('click', triggerFileInput);
  document.getElementById('file-input').addEventListener('change', handleFileSelected);
  document.getElementById('btn-directive').addEventListener('click', showDirectiveEditor);
  document.getElementById('directive-close').addEventListener('click', hideDirectiveEditor);
  document.getElementById('directive-overlay').addEventListener('click', e => {
    if (e.target === e.currentTarget) hideDirectiveEditor();
  });
  document.getElementById('directive-save').addEventListener('click', () => {
    saveDirective(activeModel, document.getElementById('directive-input').value);
    hideDirectiveEditor();
  });
  document.getElementById('directive-reset').addEventListener('click', () => {
    document.getElementById('directive-input').value = DEFAULT_DIRECTIVE;
  });
  document.getElementById('btn-send').addEventListener('click', sendMessage);
  document.getElementById('btn-stop').addEventListener('click', stopGenerating);
  document.getElementById('chat-input').addEventListener('keydown', handleInput);
  document.getElementById('chat-input').addEventListener('input', function () { autoResize(this); });
  document.getElementById('btn-refresh').addEventListener('click', loadModels);
  document.getElementById('btn-clear').addEventListener('click', clearChat);
  document.getElementById('btn-logout').addEventListener('click', logout);
  document.getElementById('btn-sidebar-toggle').addEventListener('click', toggleSidebar);
  document.getElementById('sidebar-backdrop').addEventListener('click', closeSidebar);
  document.getElementById('btn-vision-upload').addEventListener('click', triggerVisionUpload);
  document.getElementById('vision-file-input').addEventListener('change', handleVisionFileSelected);
  document.getElementById('btn-fedora-watch').addEventListener('click', () => {
    toggleFedoraWatch().catch((e) => setVisionBrief(`Fedora error: ${e.message || e}`));
  });
  document.getElementById('btn-fedora-cam').addEventListener('click', () => {
    setFedoraFeed('camera').catch((e) => setVisionBrief(`Camera error: ${e.message || e}`));
  });
  document.getElementById('btn-fedora-upload-feed').addEventListener('click', () => {
    setFedoraFeed('upload').then(() => {
      document.getElementById('vision-file-input').click();
    }).catch((e) => setVisionBrief(`Upload feed error: ${e.message || e}`));
  });
  document.getElementById('btn-mission').addEventListener('click', openMissionModal);
  document.getElementById('mission-close').addEventListener('click', closeMissionModal);
  document.getElementById('mission-cancel').addEventListener('click', closeMissionModal);
  document.getElementById('mission-apply').addEventListener('click', applyMissionSwitch);
  document.getElementById('mission-overlay').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeMissionModal();
  });

  if (token) {
    fetch(`${API}/api/models`, { headers: authHeaders() })
      .then(r => { if (r.ok) showApp(); else logout(); })
      .catch(() => logout());
  }

  setInterval(() => { if (token) loadModels(); }, 30000);
});
