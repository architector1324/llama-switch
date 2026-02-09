document.addEventListener('DOMContentLoaded', () => {
    fetchConfig();
    startPolling();
    
    // Setup mobile touch support for log buttons
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
    
    // Context Enter Listener
    const ctxInput = document.getElementById('ctx-input');
    ctxInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            applyContextChange();
        }
    });

    // --- Theme Logic ---
    const themeBtn = document.getElementById('theme-toggle');
    const html = document.documentElement;

    // Load saved
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
        // If dark, show Sun to switch to light. If light, show Moon to switch to dark.
        themeBtn.textContent = theme === 'dark' ? '☀' : '☾'; 
    }
});

let currentConfig = {};
let lastStatus = null;
let currentLoadingModel = null; // Track which model is being loaded

async function fetchConfig() {
    try {
        const res = await fetch('/api/config');
        const data = await res.json();
        currentConfig = data;
        renderModelList(data.models || {});
    } catch (e) {
        console.error("Failed to load config", e);
        document.getElementById('model-list').innerHTML = '<div class="model-item">Error loading config</div>';
    }
}

function renderModelList(models) {
    const list = document.getElementById('model-list');
    list.innerHTML = '';
    
    // Convert to array and sort
    const modelKeys = Object.keys(models).sort();
    
    if (modelKeys.length === 0) {
        list.innerHTML = '<div class="model-item">No models found in config</div>';
        return;
    }

    modelKeys.forEach(key => {
        const item = document.createElement('div');
        item.className = 'model-item';
        item.dataset.key = key;
        
        item.innerHTML = `
            <div class="model-name">${key}</div>
            <div class="actions">
                <button class="btn btn-primary btn-sm model-btn" onclick="handleModelClick('${key}')">Load</button>
            </div>
        `;
        list.appendChild(item);
    });
}

function handleModelClick(key) {
    // Check state to decide action
    // We allow unload only if we are fully ready. If starting, maybe allow stop too?
    // Let's stick to "Unload" button availability.
    if (lastStatus && lastStatus.running && lastStatus.model === key && lastStatus.ready) {
        unloadModel(); 
    } else {
        loadModel(key);
    }
}

async function loadModel(key) {
    const ctxInput = document.getElementById('ctx-input');
    const ctx = parseInt(ctxInput.value) || 4096;
    
    currentLoadingModel = key;
    updateButtonsState(lastStatus); // Reflect loading state immediately

    try {
        const res = await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_key: key, ctx: ctx })
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
    
    // Silent Reload current model with new context
    const currentModel = lastStatus.model;
    loadModel(currentModel);
}

async function stopServer() {
    if(!confirm("Stop current Llama server?")) return;
    await fetch('/api/stop', { method: 'POST' });
    currentLoadingModel = null;
}

function startPolling() {
    // Status Poll
    setInterval(async () => {
        try {
            const res = await fetch('/api/status');
            const status = await res.json();
            updateStatusDisplay(status);
            updateActiveModel(status);
            
            // Clear loading state only when READY
            if (currentLoadingModel && status.running && status.model === currentLoadingModel && status.ready) {
                currentLoadingModel = null;
            }
            
            // Also safety: if we were loading but server stopped or switched model
            if (currentLoadingModel && (!status.running || (status.model && status.model !== currentLoadingModel))) {
                currentLoadingModel = null;
            }
            
            updateButtonsState(status);
            
        } catch (e) {
            console.log("Status check failed", e);
        }
    }, 5000);

    // Logs Poll
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
    const indicator = document.getElementById('global-status');
    const statusText = document.getElementById('status-text');
    const modelText = document.getElementById('current-model');
    // const ctxText = document.getElementById('current-ctx'); // Removed from DOM
    const webuiBtn = document.getElementById('webui-btn');
    
    if (status.running) {
        indicator.classList.add('on');
        statusText.innerText = `Running (Port: ${status.port || '?'})`;
        statusText.style.color = 'var(--success)';
        modelText.innerText = status.model || 'Unknown';
        // ctxText.innerText = status.ctx || '-';
        
        // Update Stats
        if (status.stats) {
            const used = status.stats.ctx_used || 0;
            const limit = status.ctx || 0; // Use status.ctx as the limit source
            
            const ctxUsageEl = document.getElementById('stat-ctx-usage');
            if (ctxUsageEl) ctxUsageEl.innerText = `${used} / ${limit}`;
            
            const genSpeedEl = document.getElementById('stat-gen-speed');
            if (genSpeedEl) genSpeedEl.innerText = status.stats.gen_speed ? `${status.stats.gen_speed.toFixed(2)} t/s` : '-';
            
            const promptSpeedEl = document.getElementById('stat-prompt-speed');
            if (promptSpeedEl) promptSpeedEl.innerText = status.stats.prompt_speed ? `${status.stats.prompt_speed.toFixed(2)} t/s` : '-';
            
            const totalTokensEl = document.getElementById('stat-total-tokens');
            if(totalTokensEl) totalTokensEl.innerText = status.stats.total_tokens || 0;
        }

        // WebUI Button
        if (status.ready) {
            webuiBtn.style.display = 'inline-block';
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
        // if (!status.ctx) ctxText.innerText = '-';
        webuiBtn.style.display = 'none';
        
        // Reset stats display
        const ctxUsageEl = document.getElementById('stat-ctx-usage');
        if (ctxUsageEl) ctxUsageEl.innerText = '-';
        
        const genSpeedEl = document.getElementById('stat-gen-speed');
        if (genSpeedEl) genSpeedEl.innerText = '-';
        
        const promptSpeedEl = document.getElementById('stat-prompt-speed');
        if (promptSpeedEl) promptSpeedEl.innerText = '-';
        
        const totalTokensEl = document.getElementById('stat-total-tokens');
        if(totalTokensEl) totalTokensEl.innerText = '-';
    }
}

function updateActiveModel(status) {
    document.querySelectorAll('.model-item').forEach(el => {
        el.classList.remove('active');
        if (status.running && status.model === el.dataset.key) {
            el.classList.add('active');
        }
    });
}

function updateButtonsState(status) {
    // defaults
    const isRunning = status ? status.running : false;
    const runningModel = status ? status.model : null;
    const isReady = status ? status.ready : false;

    document.querySelectorAll('.model-item').forEach(el => {
        const key = el.dataset.key;
        const btn = el.querySelector('.model-btn');
        
        // Reset styles first
        btn.classList.remove('btn-primary', 'btn-danger', 'btn-warning', 'btn-success');
        btn.disabled = false;

        // If backend says it's running and ready, show Unload (Source of Truth)
        if (isRunning && key === runningModel && isReady) {
            btn.innerText = "Unload"; 
            btn.classList.add('btn-danger');
            
            // Safety: Clear loading tracker if it was stuck
            if (key === currentLoadingModel) {
                currentLoadingModel = null;
            }
            
        } else if (key === currentLoadingModel) {
            // Loading state (User initiated)
            btn.innerText = "Starting...";
            btn.classList.add('btn-warning'); 
            btn.disabled = true;
        } else if (isRunning && key === runningModel) {
            // Running but not ready (Starting...)
            btn.innerText = "Starting...";
            btn.classList.add('btn-warning');
            btn.disabled = true;
        } else {
            // Idle state
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

// Add touch event support for mobile
function setupMobileButton(button, handler) {
    // Handle both touch and mouse events
    button.addEventListener('touchend', function(e) {
        e.preventDefault();
        e.stopPropagation();
        handler();
    }, { passive: false });
    
    // Prevent ghost clicks
    button.addEventListener('touchstart', function(e) {
        e.stopPropagation();
    }, { passive: true });
}

function scrollToBottom() {
    const container = document.getElementById('logs-container');
    container.scrollTop = container.scrollHeight;
}
