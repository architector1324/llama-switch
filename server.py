import argparse
import asyncio
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Callable, Dict, List, Optional, Tuple

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    print("\n[Service] Shutting down... cleaning up processes")
    if "state" in globals() and state.process:  # type: ignore
        with state.lock:
            _stop_process_unsafe()


app = FastAPI(lifespan=lifespan)

# Browser clients on other origins need CORS, same as llama.cpp grants.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Utilities ---
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# --- Config Manager ---
SECTIONS = ("llm", "sd", "tts", "music")


class ConfigManager:
    def __init__(
        self,
        config_file: str,
        watch: bool = False,
        on_change: Optional[Callable] = None,
    ):
        self.config_file = config_file
        self.watch = watch
        self.sections = {name: {} for name in SECTIONS}
        self.models = {}
        self.model_kind = {}
        self.last_mtime = 0
        self.lock = threading.Lock()
        self.on_change = on_change
        self.reload()

        if self.watch:
            t = threading.Thread(target=self._watch_loop, daemon=True)
            t.start()
            print(f"[Config] Watching {config_file} for changes...")

    def reload(self):
        with self.lock:
            if not os.path.exists(self.config_file):
                print(f"[Config] Error: {self.config_file} not found")
                self.sections = {name: {} for name in SECTIONS}
                self.models = {}
                self.model_kind = {}
                return

            try:
                mtime = os.stat(self.config_file).st_mtime
                with open(self.config_file, "r") as f:
                    data = yaml.safe_load(f) or {}

                sections = {name: (data.get(name) or {}) for name in SECTIONS}

                # Configs written before sections put everything under "models".
                legacy = data.get("models")
                if legacy and not sections["llm"]:
                    sections["llm"] = legacy
                    print("[Config] legacy 'models:' section loaded as 'llm'")

                models = {}
                model_kind = {}
                for name in SECTIONS:
                    for key, conf in sections[name].items():
                        if key in models:
                            print(
                                f"[Config] Duplicate model '{key}' in section "
                                f"'{name}', keeping the one from "
                                f"'{model_kind[key]}'"
                            )
                            continue
                        models[key] = conf
                        model_kind[key] = name

                self.sections = sections
                self.models = models
                self.model_kind = model_kind
                self.last_mtime = mtime

                counts = ", ".join(f"{n}: {len(sections[n])}" for n in SECTIONS)
                print(f"[Config] Loaded {len(models)} models ({counts}) from {self.config_file}")
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

    def get_models(self, kind: Optional[str] = None):
        """All models merged across sections, or just one section's models."""
        with self.lock:
            if kind is None:
                return self.models
            return self.sections.get(kind, {})

    def get_kind(self, model_key: str) -> Optional[str]:
        """Which section a model came from, or None if it is unknown."""
        with self.lock:
            return self.model_kind.get(model_key)


# --- Global State ---
class ServiceState:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.current_model: Optional[str] = None
        self.current_quant: Optional[str] = None
        self.current_ctx: int = 0
        self.current_frames: int = 0
        self.current_port: int = 0
        self.current_kind: Optional[str] = None
        self.default_ctx: int = 4096
        # Survives sd loads and stops, so a round trip keeps the user's choice.
        self.selected_ctx: int = 0
        # Frame cap for tts, apart from ctx: 2048 frames at 12.5/s is about 164 s.
        self.default_frames: int = 2048
        self.selected_frames: int = 0
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
            # sd_speed is seconds per step whichever unit sd.cpp printed.
            "sd_step": 0,
            "sd_steps": 0,
            "sd_speed": 0.0,
            "sd_last_time": 0.0,
            "sd_images": 0,
            "sd_size": "",
            "tts_frames": 0,
            "tts_rtf": 0.0,
            "tts_audio": 0.0,
            "tts_last_time": 0.0,
            "tts_clips": 0,
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
        state.current_quant = None
        state.current_port = 0
        state.current_kind = None
        state.ready = False
        state.stats = {
            "ctx_used": 0,
            "ctx_limit": 0,
            "total_tokens": 0,
            "prompt_speed": 0.0,
            "gen_speed": 0.0,
            "sd_step": 0,
            "sd_steps": 0,
            "sd_speed": 0.0,
            "sd_last_time": 0.0,
            "sd_images": 0,
            "sd_size": "",
        }
        print("[Service] Process stopped.")


def on_config_change():
    """Stop the running model when the config changes."""
    print("[Service] Config change detected. Stopping any running model...")
    with state.lock:
        _stop_process_unsafe()


# --- Log Reader ---
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _iter_output_chunks(proc):
    """Split on CR too: sd-server redraws progress in place, without LF."""
    buf = ""
    while True:
        chunk = proc.stdout.read(1)
        if not chunk:
            if proc.poll() is not None:
                break
            continue
        if chunk in ("\r", "\n"):
            if buf:
                yield _ANSI_RE.sub("", buf).rstrip()
                buf = ""
            continue
        buf += chunk
    if buf:
        yield _ANSI_RE.sub("", buf).rstrip()


def log_reader(proc, log_queue):
    # prompt eval time =       4.67 ms /    11 tokens (    0.42 ms per token,  2355.46 tokens per second)
    re_prompt = re.compile(
        r"prompt eval time\s*=\s*[\d\.]+\s*ms\s*/\s*\d+\s*tokens\s*\(\s*[\d\.]+\s*ms per token,\s*([\d\.]+)\s*tokens per second\)"
    )

    # Same line as above minus the "prompt " prefix; ^ fails on the log's prefix.
    re_eval = re.compile(
        r"(?<!prompt )eval time\s*=\s*[\d\.]+\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*[\d\.]+\s*ms per token,\s*([\d\.]+)\s*tokens per second\)"
    )

    # slot      release: id  3 | task 10 | stop processing: n_tokens = 73, truncated = 0
    re_release = re.compile(r"stop processing: n_tokens = (\d+)")

    # sd-server sampling progress: "|====>     | 3/8 - 6.59s/it" (or "12.30it/s")
    re_sd_step = re.compile(r"\|\s*(\d+)/(\d+)\s*-\s*([\d\.]+)(s/it|it/s)")

    # [INFO ] stable-diffusion.cpp:5675 - sampling completed, taking 23.84s
    re_sd_done = re.compile(r"sampling completed, taking\s*([\d\.]+)s")

    # [INFO ] stable-diffusion.cpp:5592 - generate_image 1024x1024
    re_sd_size = re.compile(r"generate_image\s+(\d+)x(\d+)")

    # qwentts.cpp progress: "[Pipeline] Generated 1920 frames (slot 0)"
    re_tts_step = re.compile(r"\[Pipeline\] Generated (\d+) frames")

    # "[Perf] Total 36762.3 ms (2048 frames, 16.67 ms/frame AR, audio 163.84 s, RTF 0.224)"
    re_tts_done = re.compile(
        r"\[Perf\] Total\s+([\d\.]+)\s*ms\s*\((\d+)\s*frames.*?audio\s+([\d\.]+)\s*s,\s*RTF\s+([\d\.]+)\)"
    )

    try:
        for decoded in _iter_output_chunks(proc):
            if decoded:
                log_queue.append(decoded)

                # Substrings that survive llama.cpp log renames ("main:" -> "llama_server:").
                if (
                    "model loaded" in decoded
                    or '"msg":"model loaded"' in decoded
                    or "listening on" in decoded
                    or "server is listening" in decoded
                ):
                    with state.lock:
                        if not state.ready:
                            state.ready = True

                try:
                    pm = re_prompt.search(decoded)
                    if pm:
                        val = float(pm.group(1))
                        with state.lock:
                            state.stats["prompt_speed"] = val

                    em = re_eval.search(decoded)
                    if em:
                        tokens_count = int(em.group(1))
                        speed_val = float(em.group(2))
                        with state.lock:
                            state.stats["gen_speed"] = speed_val
                            state.stats["total_tokens"] += tokens_count

                    rm = re_release.search(decoded)
                    if rm:
                        used = int(rm.group(1))
                        with state.lock:
                            state.stats["ctx_used"] = used
                            if state.current_ctx > 0:
                                state.stats["ctx_limit"] = state.current_ctx

                    # --- sd-server ---

                    sm = re_sd_step.search(decoded)
                    if sm:
                        step, steps = int(sm.group(1)), int(sm.group(2))
                        value, unit = float(sm.group(3)), sm.group(4)
                        # sd.cpp flips the unit below 1s per step; normalise to s/it.
                        sec_per_step = value if unit == "s/it" else (
                            1.0 / value if value > 0 else 0.0
                        )
                        with state.lock:
                            state.stats["sd_step"] = step
                            state.stats["sd_steps"] = steps
                            state.stats["sd_speed"] = round(sec_per_step, 2)

                    dm = re_sd_done.search(decoded)
                    if dm:
                        with state.lock:
                            state.stats["sd_last_time"] = float(dm.group(1))
                            state.stats["sd_images"] += 1
                            # Park the bar at 100% rather than the last partial step.
                            if state.stats["sd_steps"]:
                                state.stats["sd_step"] = state.stats["sd_steps"]

                    zm = re_sd_size.search(decoded)
                    if zm:
                        with state.lock:
                            state.stats["sd_size"] = f"{zm.group(1)}x{zm.group(2)}"
                            state.stats["sd_step"] = 0

                    # --- tts-server ---

                    tm = re_tts_step.search(decoded)
                    if tm:
                        with state.lock:
                            state.stats["tts_frames"] = int(tm.group(1))

                    td = re_tts_done.search(decoded)
                    if td:
                        with state.lock:
                            state.stats["tts_last_time"] = float(td.group(1)) / 1000.0
                            state.stats["tts_frames"] = int(td.group(2))
                            state.stats["tts_audio"] = float(td.group(3))
                            state.stats["tts_rtf"] = float(td.group(4))
                            state.stats["tts_clips"] += 1
                except Exception as e:
                    print(f"[Service] Log parsing error: {e}")

    except Exception as e:
        print(f"[Service] Log Reader Thread Crashed: {e}")


# --- API Models ---
class StartRequest(BaseModel):
    model_key: str
    quantization: Optional[str] = None
    ctx: Optional[int] = None
    frames: Optional[int] = None


# --- Internal Start Logic ---
def _default_quant(model_conf: Dict) -> Optional[str]:
    """First quant listed in the config, or None for the old single-cmd format."""
    if "cmd" in model_conf:
        return None
    return next(iter(model_conf), None)


def _resolve_model(requested_model: str, kind: str) -> Tuple[str, Optional[str]]:
    """Model id to (model_key, quant); a bare id means the first-listed quant."""
    if not state.config_mgr:
        raise HTTPException(status_code=500, detail="Config not initialized")

    models_data = state.config_mgr.get_models(kind)

    if requested_model in models_data:
        return requested_model, _default_quant(models_data[requested_model])

    for model_key, model_info in models_data.items():
        if "cmd" in model_info:
            continue  # Old format doesn't support suffixes

        default_quant = _default_quant(model_info)
        for quant_key in model_info.keys():
            if quant_key == default_quant:
                continue  # Already covered by the bare model id above
            if requested_model == f"{model_key}-{quant_key}":
                return model_key, quant_key

    raise HTTPException(status_code=404, detail=f"Model {requested_model} not found")


def _autoload_model(model_key: str, quant: Optional[str]) -> None:
    """Load the model unless it is already the running one."""
    with state.lock:
        current_model = state.current_model
        current_quant = state.current_quant
        is_running = state.process is not None and state.process.poll() is None

    if model_key == current_model and quant == current_quant and is_running:
        return

    print(f"[Proxy] Auto-loading model: {model_key} (quant: {quant})")
    try:
        _start_model_server(model_key, quant)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


def _start_model_server(
    model_key: str,
    quantization: Optional[str] = None,
    ctx: Optional[int] = None,
    frames: Optional[int] = None,
) -> Dict:
    if not state.config_mgr:
        raise RuntimeError("Config not initialized")

    models = state.config_mgr.get_models()
    if model_key not in models:
        raise ValueError("Model not found in config")

    model_conf = models[model_key]
    kind = state.config_mgr.get_kind(model_key) or "llm"

    if "cmd" in model_conf:
        # Old single-cmd format, no quants
        cmd_template = model_conf.get("cmd", "")
        actual_quant = None
    else:
        if not quantization:
            quantization = _default_quant(model_conf)

        if quantization not in model_conf:
            raise ValueError(
                f"Quantization {quantization} not found for model {model_key}"
            )

        cmd_template = model_conf[quantization]
        actual_quant = quantization

    # An explicit request wins, then the last llm choice, then the startup default.
    if ctx is None:
        ctx = state.selected_ctx or state.default_ctx

    # Deliberately not ${CTX}: an llm context window is far too big for a frame cap.
    if frames is None:
        frames = state.selected_frames or state.default_frames

    port = find_free_port()

    cmd_str = cmd_template.replace("${PORT}", str(port))
    cmd_str = cmd_str.replace("${CTX}", str(ctx))
    cmd_str = cmd_str.replace("${FRAMES}", str(frames))
    cmd_str = cmd_str.replace("${HOST}", state.host)
    # Bare forms, for configs written without the braces.
    cmd_str = cmd_str.replace("$PORT", str(port))
    cmd_str = cmd_str.replace("$CTX", str(ctx))
    cmd_str = cmd_str.replace("$FRAMES", str(frames))
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
                errors="replace",  # Replace invalid characters instead of crashing
            )
            state.current_model = model_key
            state.current_quant = actual_quant
            # sd-server has no context window; reporting one skews the UI gauge.
            state.current_ctx = ctx if kind == "llm" else 0
            if kind == "llm":
                state.selected_ctx = ctx
            state.current_frames = frames if kind == "tts" else 0
            if kind == "tts":
                state.selected_frames = frames
            state.current_port = port
            state.current_kind = kind

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
        return {
            "models": {},
            "sections": {name: {} for name in SECTIONS},
            "kinds": {},
            "default_ctx": state.default_ctx,
            "default_frames": state.default_frames,
        }

    models = state.config_mgr.get_models()
    return {
        # Flat map kept for older clients; "sections" is the grouped view.
        "models": models,
        "sections": {
            name: state.config_mgr.get_models(name) for name in SECTIONS
        },
        "kinds": {key: state.config_mgr.get_kind(key) for key in models},
        "default_ctx": state.default_ctx,
        "default_frames": state.default_frames,
    }


@app.get("/v1/models")
def get_v1_models():
    """OpenAI-compatible models list"""
    if not state.config_mgr:
        return {"object": "list", "data": [], "models": []}

    # Text models only, or chat clients would offer sd-server as a chat model.
    models_data = state.config_mgr.get_models("llm")
    model_list_openai = []
    model_list_custom = []

    for model_key, model_info in models_data.items():
        if "cmd" in model_info:
            quants = {None: model_info["cmd"]}
        else:
            quants = model_info

        # The first-listed quant is the one published under the bare model id.
        default_quant = next(iter(quants), None)

        for quant_key, cmd_str in quants.items():
            if quant_key == default_quant:
                openai_id = model_key
            else:
                openai_id = f"{model_key}-{quant_key}"

            capabilities = ["completion", "chat"]
            if "mmproj" in cmd_str:
                capabilities.append("multimodal")

            model_list_openai.append(
                {
                    "id": openai_id,
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

            model_list_custom.append(
                {
                    "name": openai_id,
                    "model": openai_id,
                    "type": "model",
                    "modified_at": "",
                    "size": "",
                    "digest": "",
                    "tags": [],
                    "capabilities": capabilities,
                    "details": {
                        "parent_model": model_key,
                        "format": "gguf",
                        "family": "",
                        "families": [],
                        "parameter_size": "",
                        "quantization_level": quant_key or "",
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
            "quantization": state.current_quant,
            "kind": state.current_kind,
            "ctx": state.current_ctx,
            "selected_ctx": state.selected_ctx or state.default_ctx,
            "frames": state.current_frames,
            "selected_frames": state.selected_frames or state.default_frames,
            "port": state.current_port if is_running else None,
            "host": state.host,
            "pid": state.process.pid if state.process and is_running else None,
            "stats": state.stats,  # Add stats to response
        }


@app.get("/api/logs")
def get_logs():
    return list(state.logs)


@app.post("/api/logs/clear")
def clear_logs():
    with state.lock:
        state.logs.clear()
    return {"status": "cleared"}


@app.post("/api/stop")
def stop_server():
    with state.lock:
        _stop_process_unsafe()
    return {"status": "stopped"}


@app.post("/api/start")
def start_server(req: StartRequest):
    try:
        updated_data = _start_model_server(
            req.model_key, req.quantization, req.ctx, req.frames
        )
        return {"status": "started", **updated_data}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
@app.post("/v1/completions")
@app.post("/v1/responses")
async def proxy_to_llama(request: Request):
    """Transparent proxy that auto-loads the requested model."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    requested_model = body.get("model")
    if not requested_model:
        raise HTTPException(status_code=400, detail="Model field required")

    # Text endpoints resolve against text models only, matching /v1/models.
    _autoload_model(*_resolve_model(requested_model, "llm"))

    retries = 0
    while retries < 60:  # seconds
        if state.ready:
            break
        await asyncio.sleep(1)
        retries += 1

    if not state.ready:
        raise HTTPException(
            status_code=504, detail="Model failed to load within timeout"
        )

    target_url = f"http://{state.host}:{state.current_port}{request.url.path}"

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

        content_type = r.headers.get("content-type") or ""
        body = r.aiter_bytes()
        if r.status_code == 200:
            body = _clean_chat(body, content_type)

        return StreamingResponse(
            body,
            status_code=r.status_code,
            media_type=content_type or None,
            background=BackgroundTask(cleanup),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")


# --- Image proxy: /v1/images/* OpenAI, /sdapi/v1/* A1111, /sdcpp/v1/* native ---
IMAGE_PREFIXES = ("/v1/images/", "/sdapi/v1/", "/sdcpp/v1/")

_BOUNDARY_RE = re.compile(r'boundary=(?:"([^"]+)"|([^\s;]+))', re.IGNORECASE)


def _model_from_multipart(raw_body: bytes, content_type: str) -> Optional[str]:
    """Model id from a multipart `model` field, without copying the file part."""
    match = _BOUNDARY_RE.search(content_type)
    if not match:
        return None

    delimiter = b"--" + (match.group(1) or match.group(2)).encode("utf-8", "replace")
    pos = 0
    while True:
        start = raw_body.find(delimiter, pos)
        if start < 0:
            return None

        head_start = start + len(delimiter)
        head_end = raw_body.find(b"\r\n\r\n", head_start)
        if head_end < 0:
            return None
        pos = head_end + 4

        head = raw_body[head_start:head_end].lower()
        if b'name="model"' not in head or b"filename=" in head:
            continue

        end = raw_body.find(delimiter, pos)
        value = raw_body[pos:end] if end >= 0 else raw_body[pos:]
        if len(value) > 256:
            return None  # not a model name, whatever it is

        try:
            model = value.decode("utf-8").strip()
        except UnicodeDecodeError:
            return None
        return model or None


def _model_from_body(raw_body: bytes, content_type: str = "") -> Optional[str]:
    """Model id from a request body, or None when it names no model."""
    if not raw_body:
        return None

    if content_type.lower().lstrip().startswith("multipart/form-data"):
        return _model_from_multipart(raw_body, content_type)

    try:
        body = json.loads(raw_body)
    except ValueError:
        return None

    if not isinstance(body, dict):
        return None

    model = body.get("model")
    return model if isinstance(model, str) and model else None


async def _proxy_to_current(
    request: Request,
    path: str,
    kind: str = "sd",
    rewrite: Optional[Callable] = None,
):
    raw_body = await request.body()

    requested_model = _model_from_body(
        raw_body, request.headers.get("content-type", "")
    )
    if requested_model:
        _autoload_model(*_resolve_model(requested_model, kind))

    with state.lock:
        is_running = state.process is not None and state.process.poll() is None
        current_kind = state.current_kind
        port = state.current_port

    if not is_running:
        raise HTTPException(status_code=503, detail="No model is running")
    if current_kind != kind:
        raise HTTPException(
            status_code=409,
            detail=f"Current model is '{current_kind}', load a {kind} model first",
        )

    retries = 0
    while retries < 60 and not state.ready:
        await asyncio.sleep(1)
        retries += 1
    if not state.ready:
        raise HTTPException(status_code=504, detail="Model failed to load in time")

    target_url = f"http://{state.host}:{port}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

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
            content=raw_body or None,
            timeout=None,
        )
        r = await client.send(req, stream=True)

        async def cleanup():
            await r.aclose()
            await client.aclose()

        content_type = r.headers.get("content-type") or ""
        body = r.aiter_bytes()
        if rewrite is not None and r.status_code == 200:
            body = rewrite(body, content_type)

        return StreamingResponse(
            body,
            status_code=r.status_code,
            media_type=content_type or None,
            background=BackgroundTask(cleanup),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")


@app.api_route("/v1/images/{path:path}", methods=["GET", "POST"])
async def proxy_openai_images(request: Request, path: str):
    return await _proxy_to_current(request, f"v1/images/{path}")


@app.api_route("/sdapi/v1/{path:path}", methods=["GET", "POST"])
async def proxy_sdapi(request: Request, path: str):
    return await _proxy_to_current(request, f"sdapi/v1/{path}")


@app.api_route("/sdcpp/v1/{path:path}", methods=["GET", "POST"])
async def proxy_sdcpp(request: Request, path: str):
    return await _proxy_to_current(request, f"sdcpp/v1/{path}")


# --- Audio proxy: speech and voices are tts, transcriptions are llama-server ---
AUDIO_LLM_ENDPOINTS = ("transcriptions", "translations")

# Qwen3-ASR answers with a "language <Name><asr_text>" header: llama.cpp#26749.
_ASR_TAG = "<asr_text>"
_ASR_LANG_PREFIX = "language "
_ASR_HEADER_LIMIT = 64


def _asr_header_forming(head: str) -> bool:
    # True while the text could still grow into the header rather than a transcript.
    head = head.lstrip()
    if len(head) >= _ASR_HEADER_LIMIT or "\n" in head:
        return False
    return _ASR_LANG_PREFIX.startswith(head) or head.startswith(_ASR_LANG_PREFIX)


def _strip_asr_header(text: str) -> str:
    head, tag, rest = text.partition(_ASR_TAG)
    return rest if tag and _asr_header_forming(head) else text


class _AsrHeaderFilter:
    # The header arrives one token at a time, so hold deltas back until it resolves.
    def __init__(self):
        self.held = ""
        self.settled = False

    def feed(self, delta: str) -> str:
        if self.settled:
            return delta
        self.held += delta
        if _ASR_TAG in self.held:
            out = _strip_asr_header(self.held)
        elif _asr_header_forming(self.held):
            return ""
        else:
            out = self.held
        self.settled, self.held = True, ""
        return out

    def flush(self) -> str:
        # A stream can end while the header is still undecided; nothing may be lost.
        out, self.held, self.settled = self.held, "", True
        return out


def _sse_line(event: Dict) -> bytes:
    return b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n"


async def _read_body(chunks) -> Tuple[bytes, Optional[Dict]]:
    body = b"".join([chunk async for chunk in chunks])
    try:
        payload = json.loads(body)
    except ValueError:
        return body, None
    return body, payload if isinstance(payload, dict) else None


async def _rewrite_sse(chunks, edit: Callable[[Dict], bool], finish: Callable[[], bytes]):
    # edit() rewrites one event; finish() releases whatever edit() still holds.
    buf = b""
    released = False

    def release() -> bytes:
        nonlocal released
        if released:
            return b""
        released = True
        return finish()

    def line_out(line: bytes) -> bytes:
        if not line.startswith(b"data: "):
            return line + b"\n"
        try:
            event = json.loads(line[6:])
        except ValueError:
            return release() + line + b"\n"
        if not isinstance(event, dict) or not edit(event):
            return line + b"\n"
        return _sse_line(event)

    async for chunk in chunks:
        buf += chunk
        while b"\n" in buf:
            line, _, buf = buf.partition(b"\n")
            yield line_out(line.rstrip(b"\r"))
    if buf:
        yield line_out(buf.rstrip(b"\r"))
    yield release()


async def _clean_transcription(chunks, content_type: str):
    if "text/event-stream" not in content_type:
        body, payload = await _read_body(chunks)
        if payload and isinstance(payload.get("text"), str):
            cleaned = _strip_asr_header(payload["text"])
            if cleaned != payload["text"]:
                payload["text"] = cleaned
                body = json.dumps(payload, ensure_ascii=False).encode()
        yield body
        return

    header = _AsrHeaderFilter()
    last: Optional[Dict] = None

    def edit(event: Dict) -> bool:
        nonlocal last
        # The closing event repeats the whole transcript, header included.
        if isinstance(event.get("text"), str):
            cleaned = _strip_asr_header(event["text"])
            if cleaned != event["text"]:
                event["text"] = cleaned
                return True
            return False
        if isinstance(event.get("delta"), str):
            last = event
            cleaned = header.feed(event["delta"])
            if cleaned != event["delta"]:
                event["delta"] = cleaned
                return True
        return False

    def finish() -> bytes:
        held = header.flush()
        if not held or last is None:
            return b""
        last["delta"] = held
        return _sse_line(last) + b"\n"

    async for out in _rewrite_sse(chunks, edit, finish):
        yield out


async def _clean_chat(chunks, content_type: str):
    if "text/event-stream" not in content_type:
        body, payload = await _read_body(chunks)
        changed = False
        for choice in (payload or {}).get("choices") or []:
            message = choice.get("message") or {}
            if isinstance(message.get("content"), str):
                cleaned = _strip_asr_header(message["content"])
                if cleaned != message["content"]:
                    message["content"] = cleaned
                    changed = True
        if changed:
            body = json.dumps(payload, ensure_ascii=False).encode()
        yield body
        return

    header = _AsrHeaderFilter()
    last: Optional[Dict] = None

    def edit(event: Dict) -> bool:
        nonlocal last
        changed = False
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                last = event
                cleaned = header.feed(delta["content"])
                if cleaned != delta["content"]:
                    delta["content"] = cleaned
                    changed = True
        return changed

    def finish() -> bytes:
        held = header.flush()
        if not held or last is None:
            return b""
        for choice in last.get("choices") or []:
            if isinstance((choice.get("delta") or {}).get("content"), str):
                choice["delta"]["content"] = held
        return _sse_line(last) + b"\n"

    async for out in _rewrite_sse(chunks, edit, finish):
        yield out


@app.api_route("/v1/audio/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy_audio(request: Request, path: str):
    head = path.split("/", 1)[0]
    is_asr = head in AUDIO_LLM_ENDPOINTS
    return await _proxy_to_current(
        request,
        f"v1/audio/{path}",
        kind="llm" if is_asr else "tts",
        rewrite=_clean_transcription if is_asr else None,
    )


# audio.cpp runs generation through its own task endpoint, not the OpenAI audio one.
@app.api_route("/v1/tasks/{path:path}", methods=["GET", "POST"])
async def proxy_tasks(request: Request, path: str):
    return await _proxy_to_current(request, f"v1/tasks/{path}", kind="music")


# Engine UIs are not republished here. They call their own paths absolutely, and
# those collide with this server's; "Open WebUI" points straight at the engine port.


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

    state.default_ctx = args.ctx
    state.host = args.host

    state.config_mgr = ConfigManager(
        args.config, watch=args.watch, on_change=on_config_change
    )

    print(f"Starting UI on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
