document.addEventListener('DOMContentLoaded', () => {
    fetchConfig();
    startPolling();
    
    const clearBtn = document.getElementById('clear-logs-btn');
    const scrollBtn = document.getElementById('scroll-logs-btn');
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
    ctxInput.addEventListener('input', () => { ctxInputPristine = false; });

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

async function fetchConfig() {
    try {
        const res = await fetch('/api/config');
        const data = await res.json();
        currentConfig = data;
        // The server's default context window is the source of truth.
        if (data.default_ctx) {
            document.getElementById('ctx-input').value = data.default_ctx;
        }
        if (data.default_frames) {
            document.getElementById('frames-input').value = data.default_frames;
        }
        renderModelList(data.models || {});
    } catch (e) {
        console.error("Failed to load config", e);
        document.getElementById('model-list').innerHTML = '<div class="model-item">Error loading config</div>';
    }
}

const SECTION_LABELS = { llm: 'Language', sd: 'Image', tts: 'Speech', music: 'Music' };

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

    Object.keys(sections).forEach(name => {
        const keys = Object.keys(sections[name] || {}).sort();
        if (keys.length === 0) return;
        renderModelGroup(list, sections[name], keys, name);
    });
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
    // Frames is tts-only: an llm-sized ctx makes no sense as a frame cap.
    const framesInput = document.getElementById('frames-input');
    const frames = parseInt(framesInput.value) || 2048;
    
    currentLoadingModel = key;
    updateButtonsState(lastStatus);

    try {
        const res = await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_key: key, quantization: quantization, ctx: ctx, frames: frames })
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

async function applyFramesChange() {
    if (!lastStatus || !lastStatus.running || !lastStatus.model) {
        return; // Nothing to do if stopped
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
    
    applyDashboardKind(status.kind);

    if (status.running) {
        indicator.classList.add('on');
        statusText.innerText = `Running (Port: ${status.port || '?'})`;
        statusText.style.color = 'var(--success)';
        modelText.innerText = status.model || 'Unknown';

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

        if (status.ready) {
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
    const notLlm = isSd || isTts || isMusic;
    const llmStats = document.getElementById('stats-llm');
    const sdStats = document.getElementById('stats-sd');
    const ttsStats = document.getElementById('stats-tts');
    const ctxControl = document.getElementById('ctx-control');
    const framesControl = document.getElementById('frames-control');

    if (llmStats) llmStats.style.display = notLlm ? 'none' : '';
    if (sdStats) sdStats.style.display = isSd ? '' : 'none';
    if (ttsStats) ttsStats.style.display = isTts ? '' : 'none';
    if (ctxControl) ctxControl.style.display = notLlm ? 'none' : '';
    if (framesControl) framesControl.style.display = isTts ? '' : 'none';
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
