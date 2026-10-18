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
let preserveInputPristine = true;

// Two identical cards are told apart only by index and bus address, so both go
// in the label. "All" stays the default: the backend spreads the model itself.
// The log belongs to one launched process; say which, so lines left by a model
// that has since exited are not read as the current one's.
async function fetchLogsOwner() {
    const el = document.getElementById('logs-owner');
    if (!el) return;
    try {
        const res = await fetch('/api/logs/owner');
        const data = await res.json();
        el.textContent = data.model ? `· ${data.model}` : '';
    } catch (e) {
        el.textContent = '';
    }
}

async function fetchDevices() {
    const sel = document.getElementById('devices-input');
    if (!sel) return;
    try {
        const res = await fetch('/api/devices');
        const data = await res.json();
        const devices = data.devices || [];
        const chosen = sel.value;
        sel.innerHTML = '<option value="">All GPUs</option>' + devices.map(d => {
            const gib = (d.total_mib / 1024).toFixed(0);
            const where = d.pci ? ` \u00b7 ${d.pci}` : '';
            return `<option value="${d.id}">${d.id} \u00b7 ${gib} GiB${where}</option>`;
        }).join('');
        sel.value = chosen;
    } catch (e) {
        console.error('Failed to load devices', e);
    }
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
        if (lastStatus && lastStatus.running && lastStatus.kind) {
            activeSection = lastStatus.kind;
            localStorage.setItem('activeSection', activeSection);
        }
        renderSectionTabs();
        renderModelList(data.models || {});
        applyDashboardKind(activeSection);
        await fetchDevices();
    } catch (e) {
        console.error("Failed to load config", e);
        document.getElementById('model-list').innerHTML = '<div class="model-item">Error loading config</div>';
    }
}

const SECTION_LABELS = { llm: 'Language', sd: 'Image', tts: 'Speech', music: 'Music', decision: 'Decision' };

// The dashboard follows the tab, not the running model: otherwise it stays on
// Language until an image model is loaded for the first time.
let activeSection = localStorage.getItem('activeSection') || 'llm';

function renderSectionTabs() {
    const bar = document.getElementById('section-tabs');
    if (!bar || !currentConfig || !currentConfig.sections) return;
    const names = Object.keys(currentConfig.sections).filter(
        n => Object.keys(currentConfig.sections[n] || {}).length > 0
    );
    if (names.length && !names.includes(activeSection)) activeSection = names[0];
    bar.innerHTML = names.map(n =>
        `<button class="section-tab${n === activeSection ? ' active' : ''}" onclick="selectSection('${n}')">${SECTION_LABELS[n] || n}</button>`
    ).join('');
}

function selectSection(name) {
    activeSection = name;
    localStorage.setItem('activeSection', name);
    renderSectionTabs();
    renderModelList(currentConfig ? currentConfig.models || {} : {});
    applyDashboardKind(activeSection);
    if (lastStatus) updateStatusDisplay(lastStatus);
}

function renderModelList(models) {
    const list = document.getElementById('model-list');
    list.innerHTML = '';

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
                    <select class="quant-select" id="quant-select-${key}" onchange="handleQuantChange('${key}')">
                        ${quants.map((q, i) => `<option value="${q}" ${i === 0 ? 'selected' : ''}>${q}</option>`).join('')}
                    </select>
                `;
            } else if (quants.length === 1) {
                quantSelector = `<span class="quant-label">${quants[0]}</span>`;
            }
        }

        item.innerHTML = `
            <div class="model-info">
                <div class="model-name">${key}</div>
                ${quantSelector}
            </div>
            <div class="actions">
                <button class="btn btn-primary btn-sm model-btn" onclick="handleModelClick('${key}')">Load</button>
            </div>
        `;
        list.appendChild(item);
    });
}

function handleModelClick(key) {
    const quantSelect = document.getElementById(`quant-select-${key}`);
    const quantization = quantSelect ? quantSelect.value : null;

    if (lastStatus && lastStatus.running && lastStatus.model === key && lastStatus.ready) {
        if (!quantization || lastStatus.quantization === quantization) {
            unloadModel();
            return;
        }
    }
    
    loadModel(key, quantization);
}

function handleQuantChange(key) {
    if (lastStatus && lastStatus.running && lastStatus.model === key) {
        const quantSelect = document.getElementById(`quant-select-${key}`);
        const quantization = quantSelect ? quantSelect.value : null;
        loadModel(key, quantization);
    }
}

async function loadModel(key, quantization = null) {
    const ctxInput = document.getElementById('ctx-input');
    const ctx = parseInt(ctxInput.value) || 4096;
    const devicesInput = document.getElementById('devices-input');
    const devices = devicesInput ? devicesInput.value : '';
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
    if (kind && kind !== activeSection) selectSection(kind);

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

async function unloadModel() {
    await fetch('/api/stop', { method: 'POST' });
    currentLoadingModel = null;
}

async function applyContextChange() {
    if (!lastStatus || !lastStatus.running || !lastStatus.model) {
        return; // Nothing to do if stopped
    }
    
    const currentModel = lastStatus.model;
    const currentQuant = lastStatus.quantization;
    loadModel(currentModel, currentQuant);
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
    if (!lastStatus || !lastStatus.running || !lastStatus.model) {
        return; // Nothing loaded: the checkbox just waits for the next start
    }
    loadModel(lastStatus.model, lastStatus.quantization);
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

async function applyDevicesChange() {
    if (!lastStatus || !lastStatus.running || !lastStatus.model) {
        return; // Nothing loaded: the pick just waits for the next start
    }
    loadModel(lastStatus.model, lastStatus.quantization);
}

async function applyParallelChange() {
    if (!lastStatus || !lastStatus.running || !lastStatus.model) {
        return; // Nothing loaded: the number just waits for the next start
    }
    loadModel(lastStatus.model, lastStatus.quantization);
}

async function applyFramesChange() {
    if (!lastStatus || !lastStatus.running || !lastStatus.model) {
        return; // Nothing to do if stopped
    }
    loadModel(lastStatus.model, lastStatus.quantization);
}

async function applyHiresChange() {
    const st = document.getElementById('hires-state');
    const input = document.getElementById('hires-input');
    if (st && input) st.innerText = input.checked ? 'On' : 'Off';
    if (!lastStatus || !lastStatus.running || !lastStatus.model) {
        return; // Nothing loaded: the toggle just waits for the next start
    }
    loadModel(lastStatus.model, lastStatus.quantization);
}

async function applyResChange() {
    if (!lastStatus || !lastStatus.running || !lastStatus.model) {
        return; // Nothing loaded: the pick just waits for the next start
    }
    loadModel(lastStatus.model, lastStatus.quantization);
}

async function stopServer() {
    if(!confirm("Stop current Llama server?")) return;
    await fetch('/api/stop', { method: 'POST' });
    currentLoadingModel = null;
}

async function pollStatus() {
    try {
        const res = await fetch('/api/status');
        const status = await res.json();
        updateStatusDisplay(status);
        updateActiveModel(status);

        if (currentLoadingModel && status.running && status.model === currentLoadingModel && status.ready) {
            currentLoadingModel = null;
        }

        if (currentLoadingModel && (!status.running || (status.model && status.model !== currentLoadingModel))) {
            currentLoadingModel = null;
        }

        updateButtonsState(status);

    } catch (e) {
        console.log("Status check failed", e);
    }
}

function startPolling() {
    // Run once right away so the context field shows the remembered value.
    pollStatus();
    setInterval(pollStatus, 5000);

    setInterval(async () => {
        const container = document.getElementById('logs-container');
        try {
            const res = await fetch('/api/logs');
            const lines = await res.json();
            fetchLogsOwner();
            
            const isAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 50;
            
            if (lines.length > 0) {
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
    }, 5000);
}

function updateStatusDisplay(status) {
    lastStatus = status;

    // Mirror the server's remembered ctx so a reload keeps the chosen value.
    if (ctxInputPristine && status.selected_ctx) {
        const ctxInput = document.getElementById('ctx-input');
        if (ctxInput && document.activeElement !== ctxInput) {
            ctxInput.value = status.selected_ctx;
        }
    }

    if (parallelInputPristine && status.selected_parallel) {
        const parallelInput = document.getElementById('parallel-input');
        if (parallelInput && document.activeElement !== parallelInput) {
            parallelInput.value = status.selected_parallel;
        }
    }
    renderParallelHint();

    const devicesInput = document.getElementById('devices-input');
    if (devicesInput && document.activeElement !== devicesInput
        && status.selected_devices !== undefined) {
        // Only when the option exists: the device list loads asynchronously.
        if ([...devicesInput.options].some(o => o.value === status.selected_devices)) {
            devicesInput.value = status.selected_devices;
        }
    }

    // Mirror the server's remembered choice, except while a load is in flight:
    // status still carries the old value until the new process is up.
    if (preserveInputPristine && !currentLoadingModel) {
        const preserveInput = document.getElementById('preserve-think-input');
        if (preserveInput && document.activeElement !== preserveInput) {
            preserveInput.checked = !!status.selected_preserve_think;
            renderPreserveThinkState(preserveInput.checked);
        }
    }

    const hiresInput = document.getElementById('hires-input');
    if (hiresInput && document.activeElement !== hiresInput) {
        hiresInput.checked = !!status.selected_hires;
        const hiresState = document.getElementById('hires-state');
        if (hiresState) hiresState.innerText = status.selected_hires ? 'On' : 'Off';
    }

    if (status.selected_res) {
        const resInput = document.getElementById('res-input');
        if (resInput && document.activeElement !== resInput) {
            resInput.value = status.selected_res;
        }
    }

    if (framesInputPristine && status.selected_frames) {
        const framesInput = document.getElementById('frames-input');
        if (framesInput && document.activeElement !== framesInput) {
            framesInput.value = status.selected_frames;
        }
    }

    const indicator = document.getElementById('global-status');
    const statusText = document.getElementById('status-text');
    const modelText = document.getElementById('current-model');
    const webuiBtn = document.getElementById('webui-btn');
    
    applyDashboardKind(activeSection);

    // Looking at a tab whose backend is not the running one: the numbers on screen
    // would belong to someone else, so blank them rather than leave them stale.
    const showsRunningKind = status.running && status.kind === activeSection;
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

    if (status.running && !showsRunningKind) {
        // Something runs, but not in this section. Saying its name here claimed a
        // speech model was loaded when it was a language one.
        indicator.classList.remove('on');
        statusText.innerText = 'Not loaded';
        statusText.style.color = 'var(--text-secondary)';
        modelText.innerText = '-';
        webuiBtn.style.display = 'none';
    }

    if (status.running) {
        if (showsRunningKind) {
            indicator.classList.add('on');
            statusText.innerText = `Running (Port: ${status.port || '?'})`;
            statusText.style.color = 'var(--success)';
            modelText.innerText = status.model || 'Unknown';
        }

        // Memory is reported by every llama.cpp-based backend, so it is filled
        // before the per-kind branches rather than inside the llm one. Guarded by
        // the same rule as the rest: another tab must not show these numbers.
        if (status.stats && showsRunningKind) {
            const memModelEl = document.getElementById('stat-mem-model');
            if (memModelEl) memModelEl.innerText = formatMiB(status.stats.mem_model);

            const memCtxEl = document.getElementById('stat-mem-context');
            if (memCtxEl) memCtxEl.innerText = formatMiB(status.stats.mem_context);

            renderMemDevices(status.stats.mem_devices);
        }

        if (status.stats && status.kind === 'sd') {
            const s = status.stats;

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
        } else if (status.stats && status.kind === 'tts') {
            const s = status.stats;

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
        } else if (status.stats && status.kind === 'decision') {
            const s = status.stats;

            // The server's own clock, so it excludes the trip to the client.
            const msEl = document.getElementById('stat-dec-ms');
            if (msEl) msEl.innerText = s.decision_ms ? `${Math.round(s.decision_ms)} ms` : '-';

            const tkEl = document.getElementById('stat-dec-tokens');
            if (tkEl) tkEl.innerText = s.decision_tokens ? `${s.decision_tokens} tokens` : '-';

            const cnEl = document.getElementById('stat-dec-count');
            if (cnEl) cnEl.innerText = s.decision_count || 0;
        } else if (status.stats && status.kind !== 'music') {
            const used = status.stats.ctx_used || 0;
            const limit = status.ctx || 0;

            const ctxUsageEl = document.getElementById('stat-ctx-usage');
            if (ctxUsageEl) ctxUsageEl.innerText = `${used} / ${limit}`;

            const genSpeedEl = document.getElementById('stat-gen-speed');
            if (genSpeedEl) genSpeedEl.innerText = status.stats.gen_speed ? `${status.stats.gen_speed.toFixed(2)} t/s` : '-';

            const promptSpeedEl = document.getElementById('stat-prompt-speed');
            if (promptSpeedEl) promptSpeedEl.innerText = status.stats.prompt_speed ? `${status.stats.prompt_speed.toFixed(2)} t/s` : '-';

            const totalTokensEl = document.getElementById('stat-total-tokens');
            if(totalTokensEl) totalTokensEl.innerText = status.stats.total_tokens || 0;
        }

        // Decision models serve /v1/systemone and nothing else: clef has no web page
        // at all, so the button would open a blank engine root.
        if (status.ready && status.kind !== 'decision' && showsRunningKind) {
            webuiBtn.style.display = 'inline-block';
            // Straight to the engine's own address: every engine page calls its own
            // paths absolutely, so none of them survive being served under a prefix.
            const displayHost = (status.host === '0.0.0.0') ? window.location.hostname : status.host;
            webuiBtn.href = `http://${displayHost}:${status.port}`;
        } else {
            webuiBtn.style.display = 'none';
        }

        
    } else {
        indicator.classList.remove('on');
        statusText.innerText = 'Stopped';
        statusText.style.color = 'var(--text-secondary)';
        if (!status.model) modelText.innerText = '-';
        webuiBtn.style.display = 'none';
        
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
    const isSd = kind === 'sd';
    const isTts = kind === 'tts';
    // Music has no counters of its own yet, but it is not a language model either:
    // showing the llm block would report someone else's tokens.
    const isMusic = kind === 'music';
    // Decision models answer in one forward pass and generate no tokens, so the
    // llm counters would read zero; they do take -c, so the context control stays.
    const isDecision = kind === 'decision';
    const notLlm = isSd || isTts || isMusic || isDecision;
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
    const preserveControl = document.getElementById('preserve-think-control');

    if (llmStats) llmStats.style.display = notLlm ? 'none' : '';
    if (memStats) memStats.style.display = (notLlm && !isDecision) ? 'none' : '';
    if (memDevices) memDevices.style.display = (isTts || isMusic) ? 'none' : '';
    if (sdStats) sdStats.style.display = isSd ? '' : 'none';
    if (ttsStats) ttsStats.style.display = isTts ? '' : 'none';
    if (decStats) decStats.style.display = isDecision ? '' : 'none';
    if (ctxControl) ctxControl.style.display = (notLlm && !isDecision) ? 'none' : '';
    if (parallelControl) parallelControl.style.display = (notLlm && !isDecision) ? 'none' : '';
    // Every backend takes a device: -dev for llama.cpp, --backend role=dev for
    // sd, --backend vulkan --device N for audiocpp.
    if (devicesControl) devicesControl.style.display = '';
    if (framesControl) framesControl.style.display = isTts ? '' : 'none';
    if (resControl) resControl.style.display = isSd ? '' : 'none';
    if (hiresControl) hiresControl.style.display = isSd ? '' : 'none';
    if (preserveControl) preserveControl.style.display = notLlm ? 'none' : '';
}

function updateActiveModel(status) {
    document.querySelectorAll('.model-item').forEach(el => {
        el.classList.remove('active');
        if (status.running && status.model === el.dataset.key) {
            el.classList.add('active');
            
            const quantSelect = el.querySelector('.quant-select');
            if (quantSelect && status.quantization && quantSelect.value !== status.quantization) {
                quantSelect.value = status.quantization;
            }
        }
    });
}

function updateButtonsState(status) {
    const isRunning = status ? status.running : false;
    const runningModel = status ? status.model : null;
    const isReady = status ? status.ready : false;

    document.querySelectorAll('.model-item').forEach(el => {
        const key = el.dataset.key;
        const btn = el.querySelector('.model-btn');
        
        btn.classList.remove('btn-primary', 'btn-danger', 'btn-warning', 'btn-success');
        btn.disabled = false;

        if (isRunning && key === runningModel && isReady) {
            const quantSelect = el.querySelector('.quant-select');
            const selectedQuant = quantSelect ? quantSelect.value : null;
            
            if (!selectedQuant || selectedQuant === status.quantization) {
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
        } else if (isRunning && key === runningModel) {
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
        const res = await fetch('/api/logs');
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

function clearLogs() {
    fetch('/api/logs/clear', { method: 'POST' })
        .then(() => {
            const container = document.getElementById('logs-container');
            container.innerHTML = '<div class="log-line">Waiting for logs...</div>';
        })
        .catch(e => console.error('Failed to clear logs:', e));
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
