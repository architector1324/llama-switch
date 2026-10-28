document.addEventListener('DOMContentLoaded', () => {
    fetchConfig();
    startPolling();
    
    const copyBtn = document.getElementById('copy-logs-btn');
    const clearBtn = document.getElementById('clear-logs-btn');
    const scrollBtn = document.getElementById('scroll-logs-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', copyLogs);
        setupMobileButton(copyBtn, copyLogs);
    }
    if (clearBtn) {
        clearBtn.addEventListener('click', clearLogs);
        setupMobileButton(clearBtn, clearLogs);
    }
    if (scrollBtn) {
        scrollBtn.addEventListener('click', scrollToBottom);
        setupMobileButton(scrollBtn, scrollToBottom);
    }
    
    const ctxInput = document.getElementById('ctx-input');
    ctxInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            applyContextChange();
        }
    });
    // Once typed into, status polls stop overwriting the field.
    ctxInput.addEventListener('input', () => { ctxInputPristine = false; renderParallelHint(); });

    const parallelInput = document.getElementById('parallel-input');
    if (parallelInput) {
        parallelInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                applyParallelChange();
            }
        });
        parallelInput.addEventListener('input', () => {
            parallelInputPristine = false;
            renderParallelHint();
        });
    }

    const framesInput = document.getElementById('frames-input');
    framesInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            applyFramesChange();
        }
    });
    framesInput.addEventListener('input', () => { framesInputPristine = false; });

    // --- Theme Logic ---
    const themeBtn = document.getElementById('theme-toggle');
    const html = document.documentElement;

    const savedTheme = localStorage.getItem('theme') || 'dark';
    html.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);

    themeBtn.addEventListener('click', () => {
        const current = html.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        html.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
        updateThemeIcon(next);
    });

    function updateThemeIcon(theme) {
        // Show the icon of the theme the button switches to.
        themeBtn.textContent = theme === 'dark' ? '☀' : '☾'; 
    }
});

let currentConfig = {};
let lastStatus = null;
let currentLoadingModel = null;
let ctxInputPristine = true; // Field still mirrors the server, untouched
let framesInputPristine = true;
let parallelInputPristine = true;
// Device pickers are launch settings like the context window: once touched, the
// status poll must stop overwriting them, or a pick made while nothing is
// loaded is undone a few seconds later.
let devicesInputPristine = true;
// With several models up, the dashboard still shows one. Default: the instance
// of the open section; a click on a running model in the list picks that one.
let focusedModel = null;
let instances = [];

function instanceFor(section) {
    const here = instances.filter(i => i.kind === section);
    const kinds = (currentConfig && currentConfig.kinds) || {};
    // A row picked in this section wins, even when it is not loaded: then the
    // dashboard describes what Load would start, not somebody else's instance.
    if (focusedModel && kinds[focusedModel] === section) {
        return here.find(i => i.model === focusedModel) || null;
    }
    return here[0] || null;
}

// Not a config section: a view over every instance that is up, whatever its kind.
const RUNNING = 'running';

function runningInstance() {
    if (activeSection === RUNNING) {
        return instances.find(i => i.model === focusedModel) || instances[0] || null;
    }
    return instanceFor(activeSection);
}

// Everywhere else the tab is the kind. On Running the kinds are mixed, so the
// dashboard takes the kind of the model it is showing -- or none at all.
function dashboardKind() {
    if (activeSection !== RUNNING) return activeSection;
    const inst = runningInstance();
    return inst ? inst.kind : null;
}

// Mixed kinds need a kind label per row; the order is the tab order, so a
// restart does not shuffle the list.
function sortedInstances() {
    const order = Object.keys(SECTION_LABELS);
    return [...instances].sort((a, b) =>
        (order.indexOf(a.kind) - order.indexOf(b.kind)) || a.model.localeCompare(b.model));
}

let lastRunningSignature = '';

function renderRunningList(list) {
    lastRunningSignature = runningSignature();
    const rows = sortedInstances();
    if (!rows.length) {
        list.innerHTML = '<div class="loading">Nothing is running</div>';
        return;
    }
    rows.forEach(inst => {
        const item = document.createElement('div');
        item.className = 'model-item';
        item.dataset.key = inst.model;
        item.onclick = () => focusModel(inst.model);
        const quant = inst.quantization ? `<span class="quant-label">${inst.quantization}</span>` : '';
        item.innerHTML = `
            <div class="model-info">
                <div class="model-name">${inst.model}</div>
                <div class="running-meta">${quant}<span class="kind-label">${SECTION_LABELS[inst.kind] || inst.kind}</span></div>
            </div>
            <div class="actions">
                <button class="btn btn-primary btn-sm model-btn" onclick="event.stopPropagation(); handleModelClick('${inst.model}')">Unload</button>
            </div>
        `;
        list.appendChild(item);
    });
}

// Green only when something is really up: alive and past loading. A model
// still starting, or one whose process died, does not count.
function anythingLoaded() {
    return instances.some(i => i.running && i.ready);
}

function paintRunningTab() {
    const tab = document.querySelector('.section-tab.running-tab');
    if (tab) tab.classList.toggle('idle', !anythingLoaded());
}

function runningSignature() {
    return sortedInstances().map(i => `${i.model}:${i.quantization}:${i.kind}`).join('|');
}
let preserveInputPristine = true;

// Two identical cards are told apart only by index and bus address, so both go
// in the label. "All" stays the default: the backend spreads the model itself.
// The log belongs to one launched process; say which, so lines left by a model
// that has since exited are not read as the current one's.
// Returns the owner so the caller can decide whether this section sees the log.
async function fetchLogsOwner() {
    const el = document.getElementById('logs-owner');
    const inst = runningInstance();
    let owner = { model: null, kind: null };
    if (inst) {
        try {
            const res = await fetch(`/api/logs/owner?model=${encodeURIComponent(inst.model)}`);
            owner = await res.json();
        } catch (e) {
            // Leave it empty; the caller treats that as "nothing to show".
        }
    }
    const mine = !!(inst && owner.kind === dashboardKind());
    if (el) el.textContent = (mine && owner.model) ? `· ${owner.model}` : '';
    return { ...owner, mine, model: inst ? inst.model : null };
}

// Image pipelines are split across modules; sd-server takes a card per module.
const SD_DEVICE_ROLES = [['diffusion', 'sd-dev-diffusion'],
                         ['te', 'sd-dev-te'],
                         ['vae', 'sd-dev-vae']];

async function fetchDevices() {
    const sel = document.getElementById('devices-input');
    try {
        const res = await fetch('/api/devices');
        const data = await res.json();
        const devices = data.devices || [];
        const options = devices.map(d => {
            const gib = (d.total_mib / 1024).toFixed(0);
            const where = d.pci ? ` \u00b7 ${d.pci}` : '';
            return `<option value="${d.id}">${d.id} \u00b7 ${gib} GiB${where}</option>`;
        }).join('');

        if (sel) {
            const chosen = sel.value;
            sel.innerHTML = '<option value="">All GPUs</option>' + options;
            sel.value = chosen;
        }
        // "Profile" leaves the role where the profile put it, rather than
        // forcing every module onto one card.
        for (const [, id] of SD_DEVICE_ROLES) {
            const roleSel = document.getElementById(id);
            if (!roleSel) continue;
            const chosen = roleSel.value;
            roleSel.innerHTML = '<option value="">Profile</option>' + options;
            roleSel.value = chosen;
        }
    } catch (e) {
        console.error('Failed to load devices', e);
    }
}

// What goes to the server: a per-role assignment on the image tab, a plain
// device name everywhere else.
// What the profile itself says, so "Profile" can name the card it stands for.
function profileDeviceRoles(modelKey, quant) {
    const models = currentConfig ? currentConfig.models || {} : {};
    const entry = models[modelKey];
    if (!entry) return {};
    const cmd = entry.cmd || entry[quant] || entry[Object.keys(entry)[0]] || '';
    const m = /\$\{DEVICES:([^}]*)\}/.exec(cmd);
    if (!m) return {};
    return Object.fromEntries([...m[1].matchAll(/([A-Za-z_]+)=([^,\s]+)/g)]
        .map(x => [x[1], x[2]]));
}

function renderSdProfileLabels() {
    const sdInst = instanceFor('sd');
    const roles = sdInst ? profileDeviceRoles(sdInst.model, sdInst.quantization) : {};
    for (const [role, id] of SD_DEVICE_ROLES) {
        const el = document.getElementById(id);
        if (!el || !el.options.length) continue;
        const where = roles[role];
        el.options[0].textContent = where ? `Profile (${where})` : 'Profile';
    }
}

function selectedDevices() {
    if (dashboardKind() === 'sd') {
        const parts = [];
        for (const [role, id] of SD_DEVICE_ROLES) {
            const el = document.getElementById(id);
            if (el && el.value) parts.push(`${role}=${el.value}`);
        }
        return parts.join(',');
    }
    const el = document.getElementById('devices-input');
    return el ? el.value : '';
}

async function fetchConfig() {
    try {
        const res = await fetch('/api/config');
        const data = await res.json();
        currentConfig = data;
        // The server's default context window is the source of truth.
        if (data.default_ctx) {
            document.getElementById('ctx-input').value = data.default_ctx;
        }
        if (data.default_parallel) {
            const parallelInput = document.getElementById('parallel-input');
            if (parallelInput) parallelInput.value = data.default_parallel;
        }
        if (data.default_frames) {
            document.getElementById('frames-input').value = data.default_frames;
        }
        if (data.default_res) {
            const resInput = document.getElementById('res-input');
            if (resInput) resInput.value = data.default_res;
        }
        const up = (lastStatus && lastStatus.instances) || [];
        if (activeSection !== RUNNING && up.length && !up.some(i => i.kind === activeSection)) {
            activeSection = up[0].kind;
            localStorage.setItem('activeSection', activeSection);
        }
        renderSectionTabs();
        renderModelList(data.models || {});
        applyDashboardKind(dashboardKind());
        await fetchDevices();
    } catch (e) {
        console.error("Failed to load config", e);
        document.getElementById('model-list').innerHTML = '<div class="model-item">Error loading config</div>';
    }
}

const SECTION_LABELS = { llm: 'Language', sd: 'Image', tts: 'Speech', stt: 'Transcription', music: 'Music', decision: 'Decision' };

// The dashboard follows the tab, not the running model: otherwise it stays on
// Language until an image model is loaded for the first time.
let activeSection = localStorage.getItem('activeSection') || 'llm';

function renderSectionTabs() {
    const bar = document.getElementById('section-tabs');
    if (!bar || !currentConfig || !currentConfig.sections) return;
    const names = Object.keys(currentConfig.sections).filter(
        n => Object.keys(currentConfig.sections[n] || {}).length > 0
    );
    if (names.length && activeSection !== RUNNING && !names.includes(activeSection)) {
        activeSection = names[0];
    }
    bar.innerHTML = names.map(n =>
        `<button class="section-tab${n === activeSection ? ' active' : ''}" onclick="selectSection('${n}')">${SECTION_LABELS[n] || n}</button>`
    ).join('') +
        `<button class="section-tab running-tab${activeSection === RUNNING ? ' active' : ''}${anythingLoaded() ? '' : ' idle'}" onclick="selectSection('${RUNNING}')">Running</button>`;
}

function selectSection(name) {
    activeSection = name;
    resetControlPristine();
    localStorage.setItem('activeSection', name);
    renderSectionTabs();
    renderModelList(currentConfig ? currentConfig.models || {} : {});
    applyDashboardKind(dashboardKind());
    // Draw from the cached status first so the panel changes immediately, then
    // ask the backend: everything here is per-section, and waiting out the 5s
    // poll left the previous tab's answers on screen.
    if (lastStatus) updateStatusDisplay(lastStatus);
    pollStatus();
    refreshLogs();
}

function renderModelList(models) {
    const list = document.getElementById('model-list');
    list.innerHTML = '';

    if (activeSection === RUNNING) {
        renderRunningList(list);
        return;
    }

    if (Object.keys(models).length === 0) {
        list.innerHTML = '<div class="model-item">No models found in config</div>';
        return;
    }

    // Older servers send no "sections", so fall back to one unlabelled group.
    const sections = currentConfig.sections;
    if (!sections) {
        renderModelGroup(list, models, Object.keys(models).sort(), null);
        return;
    }

    const group = sections[activeSection] || {};
    const keys = Object.keys(group).sort();
    if (keys.length === 0) {
        list.innerHTML = '<div class="loading">No models in this section</div>';
        return;
    }
    // The tab already names the section, so no group header here.
    renderModelGroup(list, group, keys, null);
}

function renderModelGroup(list, models, modelKeys, sectionName) {
    if (sectionName) {
        const header = document.createElement('div');
        header.className = 'model-group-header';
        header.textContent = SECTION_LABELS[sectionName] || sectionName;
        list.appendChild(header);
    }

    modelKeys.forEach(key => {
        const modelInfo = models[key];
        const item = document.createElement('div');
        item.className = 'model-item';
        item.dataset.key = key;
        
        let quantSelector = '';
        if (!modelInfo.cmd) {
            // Config order matters: the first listed quant is the default.
            const quants = Object.keys(modelInfo);
            if (quants.length > 1) {
                quantSelector = `
                    <select class="quant-select" id="quant-select-${key}" onclick="event.stopPropagation()" onchange="handleQuantChange('${key}')">
                        ${quants.map((q, i) => `<option value="${q}" ${i === 0 ? 'selected' : ''}>${q}</option>`).join('')}
                    </select>
                `;
            } else if (quants.length === 1) {
                quantSelector = `<span class="quant-label">${quants[0]}</span>`;
            }
        }

        item.onclick = () => focusModel(key);
        item.innerHTML = `
            <div class="model-info">
                <div class="model-name">${key}</div>
                ${quantSelector}
            </div>
            <div class="actions">
                <button class="btn btn-primary btn-sm model-btn" onclick="event.stopPropagation(); handleModelClick('${key}')">Load</button>
            </div>
        `;
        list.appendChild(item);
    });
}

// Fields describe the model in focus, so a different focus must refill them
// even though the user typed in them for the previous one.
function resetControlPristine() {
    ctxInputPristine = true;
    parallelInputPristine = true;
    framesInputPristine = true;
    preserveInputPristine = true;
    devicesInputPristine = true;
}

// Any row can be picked, running or not: picking one that is not up yet is how
// its settings get dialled in before Load.
function focusModel(key) {
    focusedModel = key;
    resetControlPristine();
    if (lastStatus) updateStatusDisplay(lastStatus);
    updateActiveModel(lastStatus || {});
    refreshLogs();
}

function handleModelClick(key) {
    const quantSelect = document.getElementById(`quant-select-${key}`);
    const quantization = quantSelect ? quantSelect.value : null;
    const inst = instances.find(i => i.model === key);

    if (inst && inst.ready && (!quantization || inst.quantization === quantization)) {
        unloadModel(key);
        return;
    }

    // The dashboard follows whatever was last asked for in this section.
    focusedModel = key;
    loadModel(key, quantization);
}

function handleQuantChange(key) {
    if (instances.some(i => i.model === key)) {
        const quantSelect = document.getElementById(`quant-select-${key}`);
        const quantization = quantSelect ? quantSelect.value : null;
        loadModel(key, quantization);
    }
}

async function loadModel(key, quantization = null) {
    const ctxInput = document.getElementById('ctx-input');
    const ctx = parseInt(ctxInput.value) || 4096;
    const devices = selectedDevices();
    // Slots share that one context window, so the server needs both together.
    const parallelInput = document.getElementById('parallel-input');
    const parallel = parseInt(parallelInput ? parallelInput.value : '1') || 1;
    // Frames is tts-only: an llm-sized ctx makes no sense as a frame cap.
    const framesInput = document.getElementById('frames-input');
    const frames = parseInt(framesInput.value) || 2048;
    // sd-only: the picker lists sizes, profiles opt in with ${WIDTH}/${HEIGHT}.
    const resInput = document.getElementById('res-input');
    // Not "res": the fetch below binds that name in this block and would shadow it.
    const outputRes = resInput ? resInput.value : null;
    const hiresInput = document.getElementById('hires-input');
    const hires = hiresInput ? hiresInput.checked : false;
    // A launch flag, not a runtime one: it only takes effect on this load.
    const preserveInput = document.getElementById('preserve-think-input');
    const preserveThink = preserveInput ? preserveInput.checked : false;
    
    const kind = currentConfig && currentConfig.kinds ? currentConfig.kinds[key] : null;
    if (kind && kind !== activeSection && activeSection !== RUNNING) selectSection(kind);

    currentLoadingModel = key;
    updateButtonsState(lastStatus);

    try {
        const res = await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_key: key, quantization: quantization, ctx: ctx, parallel: parallel, devices: devices, frames: frames, res: outputRes, hires: hires, preserve_think: preserveThink })
        });
        
        if (!res.ok) {
            const err = await res.json();
            alert('Error: ' + err.detail);
            currentLoadingModel = null;
            updateButtonsState(lastStatus);
        }
    } catch (e) {
        alert('Network error: ' + e.message);
        currentLoadingModel = null;
        updateButtonsState(lastStatus);
    }
}

async function unloadModel(key) {
    // Without a model this stops everything, which is what Stop All wants.
    const url = key ? `/api/stop?model=${encodeURIComponent(key)}` : '/api/stop';
    await fetch(url, { method: 'POST' });
    currentLoadingModel = null;
    pollStatus();
}

// Every control reloads the model to take effect, so each one must first ask
// whether the running model is the one this tab is about. Otherwise changing
// Output Size on the Image tab restarted a language model.
function reloadCurrentModel() {
    const inst = runningInstance();
    if (!inst) {
        return; // Nothing of ours is loaded: the setting waits for the next start
    }
    loadModel(inst.model, inst.quantization);
}

async function applyContextChange() {
    reloadCurrentModel();
}

function renderPreserveThinkState(on) {
    const label = document.getElementById('preserve-think-state');
    if (!label) return;
    label.textContent = on ? 'On' : 'Off';
    label.classList.toggle('is-on', !!on);
}

async function applyPreserveThinkChange() {
    const input = document.getElementById('preserve-think-input');
    // Once toggled, status polls stop overwriting the box.
    preserveInputPristine = false;
    renderPreserveThinkState(input && input.checked);

    // The flag is baked into the command line, so the model has to come back up.
    reloadCurrentModel();
}

// Host-side buffers are named after the backend that pinned them (Vulkan_Host,
// CUDA_Host) or after the CPU itself; everything else is a card.
function isHostDevice(name) {
    return /^CPU/i.test(name) || /_Host$/i.test(name);
}

// One box per place the weights and the cache actually landed, cards kept apart
// and host buffers rolled into a single RAM figure.
function renderMemDevices(devices) {
    const row = document.getElementById('mem-devices-row');
    if (!row) return;
    if (!devices || !Object.keys(devices).length) {
        row.innerHTML = '';
        return;
    }

    const groups = new Map();
    for (const [name, vals] of Object.entries(devices)) {
        const key = isHostDevice(name) ? 'RAM' : name;
        const acc = groups.get(key) || { model: 0, context: 0, compute: 0 };
        acc.model += vals.model || 0;
        acc.context += vals.context || 0;
        acc.compute += vals.compute || 0;
        groups.set(key, acc);
    }

    // Cards first, in their own order; RAM last, where it reads as the remainder.
    const names = [...groups.keys()].sort((a, b) => {
        if (a === 'RAM') return 1;
        if (b === 'RAM') return -1;
        return a.localeCompare(b);
    });

    row.innerHTML = names.map(name => {
        const g = groups.get(name);
        const parts = [];
        if (g.model) parts.push(`model ${formatMiB(g.model)}`);
        if (g.context) parts.push(`context ${formatMiB(g.context)}`);
        if (g.compute) parts.push(`compute ${formatMiB(g.compute)}`);
        const title = parts.join(' \u00b7 ');
        return `<div class="stat-box"><span class="label" title="${title}">${name}</span>` +
               `<span class="value">${formatMiB(g.model + g.context + g.compute)}</span></div>`;
    }).join('');
}

// The backend reports MiB; GiB reads better once a model is past a gigabyte.
function formatMiB(mib) {
    if (!mib) return '-';
    return mib >= 1024 ? `${(mib / 1024).toFixed(2)} GiB` : `${Math.round(mib)} MiB`;
}

// The context window is per slot, so the KV the engine is asked for is that times
// the slot count. Spell the total out: it is what the memory bill follows.
function renderParallelHint() {
    const hint = document.getElementById('parallel-hint');
    if (!hint) return;
    const parallelInput = document.getElementById('parallel-input');
    const ctxInput = document.getElementById('ctx-input');
    const n = parseInt(parallelInput ? parallelInput.value : '1') || 1;
    const ctx = parseInt(ctxInput ? ctxInput.value : '0') || 0;
    hint.textContent = (n > 1 && ctx) ? `${n} \u00d7 ${ctx} = ${n * ctx} ctx total` : '';
}

function renderMultiState(on) {
    const label = document.getElementById('multi-state');
    if (!label) return;
    label.textContent = on ? 'On' : 'Off';
    label.classList.toggle('is-on', !!on);
}

async function applyMultiChange() {
    const input = document.getElementById('multi-input');
    const on = !!(input && input.checked);
    renderMultiState(on);
    try {
        await fetch('/api/multi', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ multi: on }),
        });
    } catch (e) {
        console.error('Failed to set multi-model', e);
    }
    pollStatus();
}

async function applyDevicesChange() {
    // Once picked, the choice is the user's until the page is reloaded.
    devicesInputPristine = false;
    reloadCurrentModel();
}

async function applyParallelChange() {
    reloadCurrentModel();
}

async function applyFramesChange() {
    reloadCurrentModel();
}

async function applyHiresChange() {
    const st = document.getElementById('hires-state');
    const input = document.getElementById('hires-input');
    if (st && input) st.innerText = input.checked ? 'On' : 'Off';
    reloadCurrentModel();
}

async function applyResChange() {
    reloadCurrentModel();
}

// A reset is every field changing at once, so like any field change it
// restarts the model it applies to. The server owns the defaults.
async function postReset(query) {
    try {
        const res = await fetch('/api/reset' + query, { method: 'POST' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            alert('Error: ' + (err.detail || res.status));
        }
    } catch (e) {
        alert('Network error: ' + e.message);
    }
    // The fields must take the defaults even where the user had typed.
    resetControlPristine();
    pollStatus();
}

async function resetFocused() {
    const inst = runningInstance();
    const kinds = (currentConfig && currentConfig.kinds) || {};
    const picked = focusedModel && kinds[focusedModel] === activeSection ? focusedModel : null;
    const key = inst ? inst.model : picked;
    if (!key && activeSection === RUNNING) return; // nothing up, nothing to reset
    await postReset(key ? `?model=${encodeURIComponent(key)}` : `?kind=${encodeURIComponent(activeSection)}`);
}

async function resetAll() {
    const n = instances.length;
    if (n && !confirm(n > 1
            ? `Reset every setting and restart all ${n} running models?`
            : 'Reset every setting and restart the running model?')) return;
    await postReset('');
}

async function stopServer() {
    const n = instances.length;
    if (!confirm(n > 1 ? `Stop all ${n} running models?` : 'Stop the running model?')) return;
    await unloadModel();
}

async function pollStatus() {
    try {
        const res = await fetch('/api/status');
        const status = await res.json();
        updateStatusDisplay(status);
        updateActiveModel(status);

        const loaded = (status.instances || []).find(i => i.model === currentLoadingModel);
        if (currentLoadingModel && loaded && loaded.ready) {
            currentLoadingModel = null;
        }
        // With eviction on, the start we were waiting for can be undone by the
        // next one; the instance simply is not there any more.
        if (currentLoadingModel && !loaded && (status.instances || []).length) {
            const anyStarting = (status.instances || []).some(i => !i.ready);
            if (!anyStarting) currentLoadingModel = null;
        }

        updateButtonsState(status);

    } catch (e) {
        console.log("Status check failed", e);
    }
}

async function refreshLogs() {
    const container = document.getElementById('logs-container');
    if (!container) return;
    try {
        // A log belongs to one instance, so it is fetched by name.
        const owner = await fetchLogsOwner();
        const res = await fetch(owner.model
            ? `/api/logs?model=${encodeURIComponent(owner.model)}`
            : '/api/logs');
        const lines = await res.json();

        const isAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 50;

        if (owner.mine && lines.length > 0) {
            container.innerHTML = lines.map(l => `<div class="log-line">${escapeHtml(l)}</div>`).join('');
        } else {
            container.innerHTML = '<div class="log-line">No logs yet...</div>';
        }

        if (isAtBottom) {
            container.scrollTop = container.scrollHeight;
        }
    } catch (e) {
        // Ignore
    }
}

function startPolling() {
    // Run once right away so the context field shows the remembered value.
    pollStatus();
    setInterval(pollStatus, 5000);

    refreshLogs();
    setInterval(refreshLogs, 5000);
}

function updateStatusDisplay(status) {
    lastStatus = status;
    instances = status.instances || [];
    // The instance this tab is about. Null means nothing of this kind is up,
    // whatever else may be running.
    const inst = runningInstance();

    const multiInput = document.getElementById('multi-input');
    if (multiInput && document.activeElement !== multiInput) {
        multiInput.checked = !!status.multi;
        renderMultiState(multiInput.checked);
    }

    // What the controls describe: the instance in focus, or -- with nothing of
    // this kind running -- the sticky choice for the next start.
    const shownCtx = inst && inst.ctx ? inst.ctx : status.selected_ctx;
    if (ctxInputPristine && shownCtx) {
        const ctxInput = document.getElementById('ctx-input');
        if (ctxInput && document.activeElement !== ctxInput) {
            ctxInput.value = shownCtx;
        }
    }

    const shownParallel = inst && inst.parallel ? inst.parallel : status.selected_parallel;
    if (parallelInputPristine && shownParallel) {
        const parallelInput = document.getElementById('parallel-input');
        if (parallelInput && document.activeElement !== parallelInput) {
            parallelInput.value = shownParallel;
        }
    }
    renderParallelHint();

    const perKind = status.selected_devices || {};
    const shownKind = dashboardKind();
    const chosen = inst ? (inst.devices || '') : (perKind[shownKind] || '');
    const devicesInput = document.getElementById('devices-input');
    if (devicesInputPristine && devicesInput && document.activeElement !== devicesInput
        && shownKind !== 'sd') {
        // Only when the option exists: the device list loads asynchronously.
        if ([...devicesInput.options].some(o => o.value === chosen)) {
            devicesInput.value = chosen;
        }
    }
    if (shownKind === 'sd' && devicesInputPristine) {
        const assigned = Object.fromEntries(
            chosen.split(',').filter(Boolean).map(p => {
                const [role, dev] = p.split('=');
                return [role, dev || ''];
            })
        );
        for (const [role, id] of SD_DEVICE_ROLES) {
            const el = document.getElementById(id);
            if (!el || document.activeElement === el) continue;
            const want = assigned[role] || '';
            if ([...el.options].some(o => o.value === want)) el.value = want;
        }
    }
    renderSdProfileLabels();

    // Mirror the server's remembered choice, except while a load is in flight:
    // status still carries the old value until the new process is up.
    if (preserveInputPristine && !currentLoadingModel) {
        const preserveInput = document.getElementById('preserve-think-input');
        const shownPreserve = inst ? !!inst.preserve_think : !!status.selected_preserve_think;
        if (preserveInput && document.activeElement !== preserveInput) {
            preserveInput.checked = shownPreserve;
            renderPreserveThinkState(preserveInput.checked);
        }
    }

    const hiresInput = document.getElementById('hires-input');
    const shownHires = inst ? !!inst.hires : !!status.selected_hires;
    if (hiresInput && document.activeElement !== hiresInput) {
        hiresInput.checked = shownHires;
        const hiresState = document.getElementById('hires-state');
        if (hiresState) hiresState.innerText = shownHires ? 'On' : 'Off';
    }

    const shownRes = (inst && inst.res) ? inst.res : status.selected_res;
    if (shownRes) {
        const resInput = document.getElementById('res-input');
        if (resInput && document.activeElement !== resInput) {
            resInput.value = shownRes;
        }
    }

    const shownFrames = (inst && inst.frames) ? inst.frames : status.selected_frames;
    if (framesInputPristine && shownFrames) {
        const framesInput = document.getElementById('frames-input');
        if (framesInput && document.activeElement !== framesInput) {
            framesInput.value = shownFrames;
        }
    }

    const indicator = document.getElementById('global-status');
    const statusText = document.getElementById('status-text');
    const modelText = document.getElementById('current-model');
    const webuiBtn = document.getElementById('webui-btn');
    
    applyDashboardKind(dashboardKind());

    paintRunningTab();

    // Models start and stop under the Running tab; redraw its list when the set
    // changes, not on every poll, so hover and scroll survive.
    if (activeSection === RUNNING && runningSignature() !== lastRunningSignature) {
        renderModelList(currentConfig ? currentConfig.models || {} : {});
    }

    // Looking at a tab whose backend is not the running one: the numbers on screen
    // would belong to someone else, so blank them rather than leave them stale.
    const showsRunningKind = !!inst;
    if (!showsRunningKind) {
        ['stat-gen-speed', 'stat-prompt-speed', 'stat-total-tokens', 'stat-ctx-usage',
         'stat-sd-speed', 'stat-sd-last', 'stat-sd-progress', 'stat-sd-images',
         'stat-tts-rtf', 'stat-tts-last', 'stat-tts-frames', 'stat-tts-clips',
         'stat-mem-model', 'stat-mem-context',
         'stat-sd-mem-params', 'stat-sd-mem-compute'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.innerText = '-';
        });
        // Not an innerText field: the row is built from scratch each time.
        renderMemDevices(null);
    }

    if (!inst) {
        // Something may well be running elsewhere; this section has nothing.
        const pickedHere = focusedModel
            && activeSection !== RUNNING
            && currentConfig.kinds
            && currentConfig.kinds[focusedModel] === activeSection;
        indicator.classList.remove('on');
        // "Stopped" only when there is truly nothing: no instance anywhere and no
        // row picked here. Otherwise this view is simply not loaded.
        statusText.innerText = (instances.length || pickedHere) ? 'Not loaded' : 'Stopped';
        statusText.style.color = 'var(--text-secondary)';
        modelText.innerText = pickedHere ? focusedModel : '-';
        webuiBtn.style.display = 'none';
    }

    if (inst) {
        {
            indicator.classList.add('on');
            statusText.innerText = `Running (Port: ${inst.port || '?'})`;
            statusText.style.color = 'var(--success)';
            modelText.innerText = inst.model || 'Unknown';
        }

        // Memory is reported by every llama.cpp-based backend, so it is filled
        // before the per-kind branches rather than inside the llm one. Guarded by
        // the same rule as the rest: another tab must not show these numbers.
        if (inst.stats) {
            const memModelEl = document.getElementById('stat-mem-model');
            if (memModelEl) memModelEl.innerText = formatMiB(inst.stats.mem_model);

            const memCtxEl = document.getElementById('stat-mem-context');
            if (memCtxEl) memCtxEl.innerText = formatMiB(inst.stats.mem_context);

            renderMemDevices(inst.stats.mem_devices);
        }

        if (inst.stats && inst.kind === 'sd') {
            const s = inst.stats;

            const speedEl = document.getElementById('stat-sd-speed');
            if (speedEl) speedEl.innerText = s.sd_speed ? `${s.sd_speed.toFixed(2)} s/it` : '-';

            const lastEl = document.getElementById('stat-sd-last');
            if (lastEl) lastEl.innerText = s.sd_last_time ? `${s.sd_last_time.toFixed(2)} s` : '-';

            const progEl = document.getElementById('stat-sd-progress');
            if (progEl) {
                progEl.innerText = s.sd_steps
                    ? `${s.sd_step} / ${s.sd_steps}${s.sd_size ? ` · ${s.sd_size}` : ''}`
                    : '-';
            }

            const imgEl = document.getElementById('stat-sd-images');
            if (imgEl) imgEl.innerText = s.sd_images || 0;

            const sdParamsEl = document.getElementById('stat-sd-mem-params');
            if (sdParamsEl) sdParamsEl.innerText = formatMiB(s.mem_model);

            // Only known once a graph has been built, so it stays '-' until the
            // first image.
            const sdComputeEl = document.getElementById('stat-sd-mem-compute');
            if (sdComputeEl) sdComputeEl.innerText = formatMiB(s.mem_compute);
        } else if (inst.stats && inst.kind === 'tts') {
            const s = inst.stats;

            // RTF below 1 is faster than realtime; the multiple reads better.
            const rtfEl = document.getElementById('stat-tts-rtf');
            if (rtfEl) rtfEl.innerText = s.tts_rtf ? `${(1 / s.tts_rtf).toFixed(1)}x realtime` : '-';

            const lastEl = document.getElementById('stat-tts-last');
            if (lastEl) {
                lastEl.innerText = s.tts_audio
                    ? `${s.tts_audio.toFixed(1)} s / ${s.tts_last_time.toFixed(1)} s`
                    : '-';
            }

            const frEl = document.getElementById('stat-tts-frames');
            if (frEl) frEl.innerText = s.tts_frames || '-';

            const clEl = document.getElementById('stat-tts-clips');
            if (clEl) clEl.innerText = s.tts_clips || 0;
        } else if (inst.stats && inst.kind === 'decision') {
            const s = inst.stats;

            // The server's own clock, so it excludes the trip to the client.
            const msEl = document.getElementById('stat-dec-ms');
            if (msEl) msEl.innerText = s.decision_ms ? `${Math.round(s.decision_ms)} ms` : '-';

            const tkEl = document.getElementById('stat-dec-tokens');
            if (tkEl) tkEl.innerText = s.decision_tokens ? `${s.decision_tokens} tokens` : '-';

            const cnEl = document.getElementById('stat-dec-count');
            if (cnEl) cnEl.innerText = s.decision_count || 0;
        } else if (inst.stats && inst.kind !== 'music' && inst.kind !== 'stt') {
            const used = inst.stats.ctx_used || 0;
            const limit = inst.ctx || 0;

            const ctxUsageEl = document.getElementById('stat-ctx-usage');
            if (ctxUsageEl) ctxUsageEl.innerText = `${used} / ${limit}`;

            const genSpeedEl = document.getElementById('stat-gen-speed');
            if (genSpeedEl) genSpeedEl.innerText = inst.stats.gen_speed ? `${inst.stats.gen_speed.toFixed(2)} t/s` : '-';

            const promptSpeedEl = document.getElementById('stat-prompt-speed');
            if (promptSpeedEl) promptSpeedEl.innerText = inst.stats.prompt_speed ? `${inst.stats.prompt_speed.toFixed(2)} t/s` : '-';

            const totalTokensEl = document.getElementById('stat-total-tokens');
            if(totalTokensEl) totalTokensEl.innerText = inst.stats.total_tokens || 0;
        }

        // Decision models serve /v1/systemone and nothing else: clef has no web page
        // at all, so the button would open a blank engine root.
        if (inst.ready && inst.kind !== 'decision') {
            webuiBtn.style.display = 'inline-block';
            // Straight to the engine's own address: every engine page calls its own
            // paths absolutely, so none of them survive being served under a prefix.
            const displayHost = (status.host === '0.0.0.0') ? window.location.hostname : status.host;
            webuiBtn.href = `http://${displayHost}:${inst.port}`;
        } else {
            webuiBtn.style.display = 'none';
        }

        
    } else {
        const ctxUsageEl = document.getElementById('stat-ctx-usage');
        if (ctxUsageEl) ctxUsageEl.innerText = '-';
        
        const genSpeedEl = document.getElementById('stat-gen-speed');
        if (genSpeedEl) genSpeedEl.innerText = '-';
        
        const promptSpeedEl = document.getElementById('stat-prompt-speed');
        if (promptSpeedEl) promptSpeedEl.innerText = '-';
        
        const totalTokensEl = document.getElementById('stat-total-tokens');
        if(totalTokensEl) totalTokensEl.innerText = '-';

        ['stat-sd-speed', 'stat-sd-last', 'stat-sd-progress', 'stat-sd-images',
         'stat-tts-rtf', 'stat-tts-last', 'stat-tts-frames', 'stat-tts-clips'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.innerText = '-';
        });
    }
}

// One stat block per backend, and only llm has a context window to set.
function applyDashboardKind(kind) {
    if (!kind) {
        ['stats-llm', 'stats-mem', 'mem-devices-row', 'stats-sd', 'stats-tts',
         'stats-decision', 'ctx-control', 'frames-control', 'res-control',
         'hires-control', 'parallel-control', 'devices-control',
         'sd-devices-control', 'preserve-think-control'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });
        return;
    }
    const isSd = kind === 'sd';
    const isTts = kind === 'tts';
    // Music has no counters of its own yet, but it is not a language model either:
    // showing the llm block would report someone else's tokens.
    const isMusic = kind === 'music';
    // Recognition has no counters of its own either, and no frames or output size.
    const isStt = kind === 'stt';
    // Decision models answer in one forward pass and generate no tokens, so the
    // llm counters would read zero; they do take -c, so the context control stays.
    const isDecision = kind === 'decision';
    const notLlm = isSd || isTts || isStt || isMusic || isDecision;
    const llmStats = document.getElementById('stats-llm');
    const memStats = document.getElementById('stats-mem');
    const memDevices = document.getElementById('mem-devices-row');
    const sdStats = document.getElementById('stats-sd');
    const ttsStats = document.getElementById('stats-tts');
    const decStats = document.getElementById('stats-decision');
    const ctxControl = document.getElementById('ctx-control');
    const framesControl = document.getElementById('frames-control');
    const resControl = document.getElementById('res-control');
    const hiresControl = document.getElementById('hires-control');
    const parallelControl = document.getElementById('parallel-control');
    const devicesControl = document.getElementById('devices-control');
    const sdDevicesControl = document.getElementById('sd-devices-control');
    const preserveControl = document.getElementById('preserve-think-control');

    if (llmStats) llmStats.style.display = notLlm ? 'none' : '';
    if (memStats) memStats.style.display = (notLlm && !isDecision) ? 'none' : '';
    if (memDevices) memDevices.style.display = (isTts || isStt || isMusic) ? 'none' : '';
    if (sdStats) sdStats.style.display = isSd ? '' : 'none';
    if (ttsStats) ttsStats.style.display = isTts ? '' : 'none';
    if (decStats) decStats.style.display = isDecision ? '' : 'none';
    if (ctxControl) ctxControl.style.display = (notLlm && !isDecision) ? 'none' : '';
    if (parallelControl) parallelControl.style.display = (notLlm && !isDecision) ? 'none' : '';
    // Every backend takes a device: -dev for llama.cpp, --backend role=dev for
    // sd, --backend vulkan --device N for audiocpp. Image models get one row per
    // module instead of a single picker.
    if (devicesControl) devicesControl.style.display = isSd ? 'none' : '';
    if (sdDevicesControl) sdDevicesControl.style.display = isSd ? '' : 'none';
    if (framesControl) framesControl.style.display = isTts ? '' : 'none';
    if (resControl) resControl.style.display = isSd ? '' : 'none';
    if (hiresControl) hiresControl.style.display = isSd ? '' : 'none';
    if (preserveControl) preserveControl.style.display = notLlm ? 'none' : '';
}

function updateActiveModel(status) {
    const byKey = new Map((status.instances || []).map(i => [i.model, i]));
    const shown = runningInstance();
    document.querySelectorAll('.model-item').forEach(el => {
        el.classList.remove('active', 'focused', 'selected');
        const inst = byKey.get(el.dataset.key);
        if (!inst) {
            // Picked but not loaded: the dashboard is dialling it in.
            if (el.dataset.key === focusedModel) el.classList.add('selected');
            return;
        }
        el.classList.add('active');
        // The one the dashboard is reporting on, when several are up.
        if (shown && shown.model === inst.model) el.classList.add('focused');

        const quantSelect = el.querySelector('.quant-select');
        if (quantSelect && inst.quantization && quantSelect.value !== inst.quantization) {
            quantSelect.value = inst.quantization;
        }
    });
}

function updateButtonsState(status) {
    const byKey = new Map(((status && status.instances) || []).map(i => [i.model, i]));

    document.querySelectorAll('.model-item').forEach(el => {
        const key = el.dataset.key;
        const btn = el.querySelector('.model-btn');
        const inst = byKey.get(key);

        btn.classList.remove('btn-primary', 'btn-danger', 'btn-warning', 'btn-success');
        btn.disabled = false;

        if (inst && inst.ready) {
            const quantSelect = el.querySelector('.quant-select');
            const selectedQuant = quantSelect ? quantSelect.value : null;

            if (!selectedQuant || selectedQuant === inst.quantization) {
                btn.innerText = "Unload";
                btn.classList.add('btn-danger');
            } else {
                btn.innerText = "Switch";
                btn.classList.add('btn-primary');
            }

            if (key === currentLoadingModel) {
                currentLoadingModel = null;
            }

        } else if (key === currentLoadingModel) {
            btn.innerText = "Starting...";
            btn.classList.add('btn-warning'); 
            btn.disabled = true;
        } else if (inst) {
            btn.innerText = "Starting...";
            btn.classList.add('btn-warning');
            btn.disabled = true;
        } else {
            btn.innerText = "Load";
            btn.classList.add('btn-primary');
        }
    });
}

function escapeHtml(text) {
    if (!text) return text;
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

async function copyLogs() {
    const btn = document.getElementById('copy-logs-btn');
    let ok = false;
    try {
        const owner = await fetchLogsOwner();
        if (!owner.mine) {
            throw new Error('log belongs to another section');
        }
        const res = await fetch(`/api/logs?model=${encodeURIComponent(owner.model)}`);
        const text = (await res.json()).join('\n');
        // navigator.clipboard exists only on a secure origin, and the dashboard is
        // plain http on the LAN, which is exactly where the phone opens it.
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
        } else {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.setAttribute('readonly', '');
            ta.style.position = 'fixed';
            ta.style.top = '-1000px';
            document.body.appendChild(ta);
            ta.select();
            ta.setSelectionRange(0, ta.value.length);
            document.execCommand('copy');
            document.body.removeChild(ta);
        }
        ok = true;
    } catch (e) {
        console.error('Failed to copy logs:', e);
    }
    if (btn) {
        const before = btn.innerText;
        btn.innerText = ok ? '✓' : '✗';
        setTimeout(() => { btn.innerText = before; }, 1200);
    }
}

async function clearLogs() {
    // Wiping another section's log from here would destroy the only record of a
    // crash the user cannot even see on this tab.
    const owner = await fetchLogsOwner();
    if (!owner.mine) return;
    try {
        await fetch(`/api/logs/clear?model=${encodeURIComponent(owner.model)}`,
                    { method: 'POST' });
        const container = document.getElementById('logs-container');
        container.innerHTML = '<div class="log-line">Waiting for logs...</div>';
    } catch (e) {
        console.error('Failed to clear logs:', e);
    }
}

// Touch support for mobile, plus ghost-click suppression.
function setupMobileButton(button, handler) {
    button.addEventListener('touchend', function(e) {
        e.preventDefault();
        e.stopPropagation();
        handler();
    }, { passive: false });
    
    button.addEventListener('touchstart', function(e) {
        e.stopPropagation();
    }, { passive: true });
}

function scrollToBottom() {
    const container = document.getElementById('logs-container');
    container.scrollTop = container.scrollHeight;
}
