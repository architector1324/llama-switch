import os
import sys
import yaml
import json
import time
import asyncio
import socket
import signal
import subprocess
import threading
import argparse
import httpx
from collections import deque
from typing import Optional, List, Dict, Callable
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    yield
    # Shutdown logic
    print("\n[Service] Shutting down... cleaning up processes")
    # We reference global 'state' and '_stop_process_unsafe' which are defined below.
    # This works because lifespan is called at runtime after module load.
    if "state" in globals() and state.process:  # type: ignore
        with state.lock:
            _stop_process_unsafe()


app = FastAPI(lifespan=lifespan)


# --- Utilities ---
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# --- Config Manager ---
class ConfigManager:
    def __init__(
        self,
        config_file: str,
        watch: bool = False,
        on_change: Optional[Callable] = None,
    ):
        self.config_file = config_file
        self.watch = watch
        self.models = {}
        self.last_mtime = 0
        self.lock = threading.Lock()
        self.on_change = on_change

        # Initial load
        self.reload()

        if self.watch:
            t = threading.Thread(target=self._watch_loop, daemon=True)
            t.start()
            print(f"[Config] Watching {config_file} for changes...")

    def reload(self):
        with self.lock:
            if not os.path.exists(self.config_file):
                print(f"[Config] Error: {self.config_file} not found")
                self.models = {}
                return

            try:
                mtime = os.stat(self.config_file).st_mtime
                with open(self.config_file, "r") as f:
                    data = yaml.safe_load(f)
                    self.models = data.get("models", {})
                    self.last_mtime = mtime
                print(
                    f"[Config] Loaded {len(self.models)} models from {self.config_file}"
                )
            except Exception as e:
                print(f"[Config] Failed to load config: {e}")

    def _watch_loop(self):
        while True:
            time.sleep(2)
            try:
                if not os.path.exists(self.config_file):
                    continue

                current_mtime = os.stat(self.config_file).st_mtime
                if current_mtime > self.last_mtime:
                    print("[Config] Change detected, reloading...")
                    self.reload()
                    if self.on_change:
                        self.on_change()
            except Exception as e:
                print(f"[Config] Watch error: {e}")

    def get_models(self):
        with self.lock:
            return self.models


import re

# ... (existing imports)


# --- Global State ---
class ServiceState:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.current_model: Optional[str] = None
        self.current_ctx: int = 0
        self.current_port: int = 0
        self.default_ctx: int = 4096
        self.host: str = "0.0.0.0"
        self.ready: bool = False
        self.logs = deque(maxlen=2000)
        self.lock = threading.Lock()
        self.config_mgr: Optional[ConfigManager] = None
        self.stats = {
            "ctx_used": 0,  # From 'stop processing: n_tokens = X'
            "ctx_limit": 0,
            "total_tokens": 0,  # Accumulated generation
            "prompt_speed": 0.0,
            "gen_speed": 0.0,
        }


state = ServiceState()


# --- Unsafe Process Control ---
def _stop_process_unsafe():
    if state.process:
        print("[Service] Stopping current process...")
        if state.process.poll() is None:
            try:
                os.killpg(os.getpgid(state.process.pid), signal.SIGTERM)
                try:
                    state.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(state.process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        state.process = None
        state.current_model = None
        state.current_port = 0
        state.ready = False
        # Reset stats
        state.stats = {
            "ctx_used": 0,
            "ctx_limit": 0,
            "total_tokens": 0,
            "prompt_speed": 0.0,
            "gen_speed": 0.0,
        }
        print("[Service] Process stopped.")


def on_config_change():
    """Callback to stop the server when config changes."""
    print("[Service] Config change detected. Stopping any running model...")
    with state.lock:
        _stop_process_unsafe()


# --- Log Reader ---
def log_reader(proc, log_queue):
    # Regex patterns
    # prompt eval time =       4.67 ms /    11 tokens (    0.42 ms per token,  2355.46 tokens per second)
    re_prompt = re.compile(
        r"prompt eval time\s*=\s*[\d\.]+\s*ms\s*/\s*\d+\s*tokens\s*\(\s*[\d\.]+\s*ms per token,\s*([\d\.]+)\s*tokens per second\)"
    )

    # eval time =     492.12 ms /     9 tokens (   54.68 ms per token,    18.29 tokens per second)
    re_eval = re.compile(
        r"\s+eval time\s*=\s*[\d\.]+\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*[\d\.]+\s*ms per token,\s*([\d\.]+)\s*tokens per second\)"
    )

    # slot      release: id  3 | task 10 | stop processing: n_tokens = 73, truncated = 0
    re_release = re.compile(r"stop processing: n_tokens = (\d+)")

    try:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                # Process ended
                break

            if line:
                # line is already string due to text=True
                decoded = line.rstrip()
                log_queue.append(decoded)

                # Check for ready state (Robust check)

                if (
                    "main: model loaded" in decoded
                    or '"msg":"model loaded"' in decoded
                    or "server is listening on" in decoded
                    or "main: server is listening" in decoded
                ):
                    with state.lock:
                        if not state.ready:
                            state.ready = True

                # Parsing Stats

                try:
                    # Prompt Speed
                    pm = re_prompt.search(decoded)
                    if pm:
                        val = float(pm.group(1))
                        with state.lock:
                            state.stats["prompt_speed"] = val

                    # Gen Speed & Token Accumulation
                    em = re_eval.search(decoded)
                    if em:
                        tokens_count = int(em.group(1))
                        speed_val = float(em.group(2))
                        with state.lock:
                            state.stats["gen_speed"] = speed_val
                            state.stats["total_tokens"] += tokens_count

                    # Context Usage (Total session tokens for that slot)
                    rm = re_release.search(decoded)
                    if rm:
                        used = int(rm.group(1))
                        with state.lock:
                            state.stats["ctx_used"] = used
                            if state.current_ctx > 0:
                                state.stats["ctx_limit"] = state.current_ctx
                except Exception as e:
                    print(f"[Service] Log parsing error: {e}")

    except Exception as e:
        print(f"[Service] Log Reader Thread Crashed: {e}")


# --- API Models ---
class StartRequest(BaseModel):
    model_key: str
    ctx: Optional[int] = None


# --- Internal Start Logic ---
def _start_model_server(model_key: str, ctx: Optional[int] = None) -> Dict:
    if not state.config_mgr:
        raise RuntimeError("Config not initialized")

    models = state.config_mgr.get_models()
    if model_key not in models:
        raise ValueError("Model not found in config")

    model_conf = models[model_key]
    cmd_template = model_conf.get("cmd", "")

    # Determined context
    ctx = ctx if ctx is not None else state.default_ctx

    # Find a free port
    port = find_free_port()

    cmd_str = cmd_template.replace("${PORT}", str(port))
    cmd_str = cmd_str.replace("${CTX}", str(ctx))
    cmd_str = cmd_str.replace("${HOST}", state.host)
    # Fallback
    cmd_str = cmd_str.replace("$PORT", str(port))
    cmd_str = cmd_str.replace("$CTX", str(ctx))
    cmd_str = cmd_str.replace("$HOST", state.host)

    print(f"Starting model {model_key} on {state.host}:{port} with command: {cmd_str}")

    with state.lock:
        _stop_process_unsafe()

        try:
            state.process = subprocess.Popen(
                cmd_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                text=True,  # Treat as text (decodes automatically)
                bufsize=1,  # Line buffered
            )
            state.current_model = model_key
            state.current_ctx = ctx
            state.current_port = port

            t = threading.Thread(
                target=log_reader, args=(state.process, state.logs), daemon=True
            )
            t.start()

        except Exception as e:
            state.process = None
            raise RuntimeError(str(e))

    return {"port": port, "command": cmd_str}


# --- Routes ---


@app.get("/api/config")
def get_config():
    if not state.config_mgr:
        models = {}
    else:
        models = state.config_mgr.get_models()

    return {"models": models, "default_ctx": state.default_ctx}


@app.get("/v1/models")
def get_v1_models():
    """OpenAI-compatible models list"""
    if not state.config_mgr:
        return {"object": "list", "data": [], "models": []}

    models_data = state.config_mgr.get_models()
    model_list_openai = []
    model_list_custom = []

    for key, model_info in models_data.items():
        # Check capabilities
        cmd_str = model_info.get("cmd", "")
        capabilities = ["completion", "chat"]
        if "mmproj" in cmd_str:
            capabilities.append("multimodal")

        # OpenAI Data Format
        model_list_openai.append(
            {
                "id": key,
                "object": "model",
                "created": 1677619200,
                "owned_by": "llamacpp",
                "meta": {
                    "vocab_type": 1,
                    "n_vocab": 32000,  # Dummy
                    "n_ctx_train": 4096,  # Dummy
                    "n_embd": 4096,  # Dummy
                    "n_params": 7000000000,  # Dummy
                    "size": 4000000000,  # Dummy
                },
            }
        )

        # Custom "models" Format
        model_list_custom.append(
            {
                "name": key,
                "model": key,
                "type": "model",
                "modified_at": "",
                "size": "",
                "digest": "",
                "tags": [],
                "capabilities": capabilities,
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "",
                    "families": [],
                    "parameter_size": "",
                    "quantization_level": "",
                },
            }
        )

    return {"object": "list", "data": model_list_openai, "models": model_list_custom}


@app.get("/api/status")
def get_status():
    with state.lock:
        is_running = state.process is not None and state.process.poll() is None
        return {
            "running": is_running,
            "ready": state.ready,
            "model": state.current_model,
            "ctx": state.current_ctx,
            "port": state.current_port if is_running else None,
            "host": state.host,
            "pid": state.process.pid if state.process and is_running else None,
            "stats": state.stats,  # Add stats to response
        }


@app.get("/api/logs")
def get_logs():
    return list(state.logs)


@app.post("/api/stop")
def stop_server():
    with state.lock:
        _stop_process_unsafe()
    return {"status": "stopped"}


@app.post("/api/start")
def start_server(req: StartRequest):
    try:
        updated_data = _start_model_server(req.model_key, req.ctx)
        return {"status": "started", **updated_data}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
@app.post("/v1/completions")
async def proxy_to_llama(request: Request):
    """
    Transparent proxy that auto-loads the requested model.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    requested_model = body.get("model")
    if not requested_model:
        raise HTTPException(status_code=400, detail="Model field required")

    # Check if we need to load the model
    with state.lock:
        current_model = state.current_model
        is_running = state.process is not None and state.process.poll() is None

    # Reload/Load if model different or not running
    if requested_model != current_model or not is_running:
        print(f"[Proxy] Auto-loading model: {requested_model}")
        try:
            _start_model_server(requested_model)
        except ValueError:
            raise HTTPException(
                status_code=404, detail=f"Model {requested_model} not found"
            )
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Wait for ready
    retries = 0
    while retries < 60:  # 60 seconds timeout
        if state.ready:
            break
        await asyncio.sleep(1)
        retries += 1

    if not state.ready:
        raise HTTPException(
            status_code=504, detail="Model failed to load within timeout"
        )

    # Forward the request
    target_url = f"http://{state.host}:{state.current_port}{request.url.path}"

    # filter headers
    filtered_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("content-length", "host")
    }

    try:
        client = httpx.AsyncClient()
        req = client.build_request(
            method=request.method,
            url=target_url,
            headers=filtered_headers,
            json=body,
            timeout=None,
        )
        r = await client.send(req, stream=True)

        async def cleanup():
            await r.aclose()
            await client.aclose()

        return StreamingResponse(
            r.aiter_bytes(),
            status_code=r.status_code,
            media_type=r.headers.get("content-type"),
            background=BackgroundTask(cleanup),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")


# --- Static files ---
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_index():
    return FileResponse("templates/index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama Switch Server")
    parser.add_argument(
        "-H",
        "--host",
        type=str,
        default="localhost",
        help="Host interface (default: localhost)",
    )
    parser.add_argument(
        "-p", "--port", type=int, default=11435, help="UI Port (default: 11435)"
    )
    parser.add_argument(
        "-c",
        "--ctx",
        type=int,
        default=4096,
        help="Default Context Window (default: 4096)",
    )
    parser.add_argument(
        "-f", "--config", type=str, default="config.yaml", help="Config file path"
    )
    parser.add_argument(
        "-w", "--watch", action="store_true", help="Watch config file for changes"
    )

    args = parser.parse_args()

    # Store default ctx
    state.default_ctx = args.ctx
    state.host = args.host

    # Initialize Config Manager with callback
    state.config_mgr = ConfigManager(
        args.config, watch=args.watch, on_change=on_config_change
    )

    print(f"Starting UI on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
