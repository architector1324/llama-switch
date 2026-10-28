# Llama Switch

![](./ref-dark.png)

A lightweight, modern WebUI manager for running local inference servers side by side.

Llama Switch puts one dashboard and one OpenAI-compatible endpoint in front of several engines: [llama.cpp](https://github.com/ggerganov/llama.cpp) for text, speech recognition, audio/vision input and decision models, [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) for images, [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) or [audio.cpp](https://github.com/0xShug0/audio.cpp) for speech synthesis, and audio.cpp for speech recognition and music generation. Several models can run at once, each on its own port and its own cards; with **Multi-model** off it works the old way, one model at a time.

## Features

- **Six kinds of model in one config**: `llm`, `sd`, `tts`, `stt`, `music` and `decision` sections, shown in the UI as **Language**, **Image**, **Speech**, **Transcription**, **Music** and **Decision**.
- **Several models at once**: with **Multi-model** on, loading a model leaves the others running, so a language model, an image model and a voice can be up together. Off, every start stops everything else.
- **Running tab**: every model that is up, whatever its kind. The tab is green while at least one model is loaded and red when nothing is.
- **A dashboard per model**: click any row, running or not, and the dashboard describes that model. Every field change restarts it with the new value; whatever is set on a model that is not loaded yet is what **Load** starts it with.
- **Per-model settings**:
  - **GPU**: which card the model runs on, or all of them. Image models get one row per module (diffusion, text encoder, VAE); **Profile** keeps what the config says for that module.
  - **Context Window** (per slot) and **Parallel Slots** for language and decision models.
  - **Preserve Thinking** for language models, **Output Size** and **Hires Fix** for image models, **Max Frames** for speech.
- **Reset and Reset All**: a model's settings, or every section's, back to the defaults. Like any field change, a reset restarts what it applies to.
- **Memory, as the engine reports it**: model and context memory for language and decision models, weights and compute buffer for image models, each split per card and RAM. audio.cpp prints no memory figures, so speech, transcription and music on it show none.
- **Real-time counters**, per kind:
  - **Language:** generation speed (t/s), prompt speed (t/s), context usage, total tokens.
  - **Image:** step time, last image time, step progress, images generated.
  - **Speech:** speed as a multiple of realtime, last clip's length against the time it took, frames generated, clips.
  - **Decision:** time of the last decision, its prompt size, requests served.
  - **Transcription** and **Music:** none yet.
- **Auto-switching from the API**: a request that names a `model` loads it first, whether the name arrives in a JSON body (chat, images, speech) or in a multipart upload (transcriptions, image edits).
- **Upstream Logs**: each model keeps its own log, and the panel names whose log it is showing.
- **Theme Support**: built-in Dark and Light modes.
- **WebUI Link**: opens the running engine's own interface on the engine's own address. Engine pages request their paths absolutely, so none of them survive being served under a prefix here.
- **Config hot-reload**: with `--watch`, edits to the config are picked up without a restart.

## Prerequisites

- **Go 1.22+**, only to build. The result is one static binary with the dashboard embedded in it; it needs nothing else at runtime.
- **llama-server**: the binary from `llama.cpp`, in `PATH` or given by absolute path in the config. The GPU picker lists the cards that `llama-server --list-devices` reports, so this one is needed even if you only run other engines.
- Optional, only for the sections you use: `sd-server` from `stable-diffusion.cpp`, `tts-server` from `qwentts.cpp`, `audiocpp-server` from `audio.cpp` (upstream builds it as `audiocpp_server`).
- Optional, for the `stt` section: `ffmpeg`. audio.cpp transcribes WAV only, so any other upload (mp3 from Open WebUI, webm from a browser recording) is converted to 16 kHz mono WAV first.
- Optional: `vulkaninfo`. With it, each card in the GPU picker is labelled with its PCI address, which is the only thing that tells two identical cards apart.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/architector1324/llama-switch.git
   cd llama-switch
   ```

2. **Build the binary:**
   ```bash
   cd src && CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o ~/.local/bin/llama-switch .
   ```
   The sources, dashboard included, live in `src/`; the binary goes to `~/.local/bin`, where the bundled systemd unit expects it, and runs from any directory.

## Configuration

Create a config file. The bundled systemd unit reads `~/.config/llama-switch/config.yaml`; started by hand, the binary reads `config.yaml` in the current directory unless `-f` names another. Models are grouped into up to six sections; you can define as many models as you like in each, and leave out any section you do not use.

Each profile is one command line. The server fills in these placeholders before running it:

| Placeholder | Kinds | Becomes |
| --- | --- | --- |
| `${HOST}`, `${PORT}` | all | the interface and a free port for this model |
| `${CTX}` | `llm`, `decision` | the context window **per slot** times the slot count: what `-c` expects |
| `${PARALLEL}` | `llm`, `decision` | the slot count, for `--parallel` |
| `${DEVICES}` | all | the chosen card in the engine's own syntax: `-dev Vulkan1` for llama.cpp, `--backend vulkan --device 1` for audio.cpp |
| `${DEVICES:<flags>}` | all | as above, but with **All GPUs** / **Profile** chosen it becomes `<flags>`. For `sd` the roles named in `<flags>` are the ones the GPU rows reassign |
| `${PRESERVE_THINK}` | `llm` | `--reasoning-preserve` or `--no-reasoning-preserve` |
| `${FRAMES}` | `tts` | the audio frame cap |
| `${WIDTH}`, `${HEIGHT}` | `sd` | the chosen Output Size |
| `${HIRES}`, `${HIRES:<flags>}` | `sd` | with Hires Fix on, `--hires` or `<flags>`; off, nothing |

The bare forms `$HOST`, `$PORT`, `$CTX`, `$PARALLEL`, `$FRAMES`, `$WIDTH`, `$HEIGHT` and `$PRESERVE_THINK` work too.

**Example `config.yaml`:**

```yaml
llm:
  # Model with multiple quantizations
  qwen-3:
    q4_k_m: llama-server --parallel ${PARALLEL} ${DEVICES} --host ${HOST} --port ${PORT} --model /path/to/qwen3-instruct-q4_k_m.gguf -c ${CTX} -ngl 99
    q8_k_xl: llama-server --parallel ${PARALLEL} ${DEVICES} --host ${HOST} --port ${PORT} --model /path/to/qwen3-instruct-UD-Q8_K_XL.gguf -c ${CTX} -ngl 99

  # Multimodal model, vision and/or audio input (with mmproj)
  gemma-3:
    q4_k_m: llama-server --parallel ${PARALLEL} ${DEVICES} --host ${HOST} --port ${PORT} --model /path/to/gemma-3-it-q4_k_m.gguf --mmproj /path/to/gemma-3-it-mmproj.gguf -c ${CTX} -ngl 99

  # Speech recognition on llama-server lives here; on audio.cpp it goes under stt
  qwen3-asr:
    q8: llama-server --parallel ${PARALLEL} ${DEVICES} --host ${HOST} --port ${PORT} --model /path/to/Qwen3-ASR-1.7B-Q8_0.gguf --mmproj /path/to/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf -c ${CTX} -ngl 99

sd:
  # Profile layout: diffusion on one card, text encoder and VAE on the other
  z-image:
    q8_0: sd-server -l ${HOST} --listen-port ${PORT} --diffusion-model /path/to/z_image_turbo-Q8_0.gguf --llm /path/to/Qwen3-4B-Q8_0.gguf --vae /path/to/ae.safetensors ${DEVICES:--backend diffusion=Vulkan1,te=Vulkan0,vae=Vulkan0} -W ${WIDTH} -H ${HEIGHT} --steps 8 --cfg-scale 1.0 ${HIRES:--hires --hires-denoising-strength 0.5} -v

tts:
  qwen3-tts:
    q8: tts-server --host ${HOST} --port ${PORT} --model /path/to/qwen-talker-1.7b-base-Q8_0.gguf --codec /path/to/qwen-tokenizer-12hz-Q8_0.gguf --max-new ${FRAMES}

stt:
  # Speech recognition on audio.cpp; uploads that are not WAV go through ffmpeg first
  nemotron-asr:
    f16: audiocpp-server --config /path/to/nemotron-asr.json --host ${HOST} --port ${PORT} ${DEVICES} --busy-timeout-ms 0

music:
  minimax-music3:
    q4: audiocpp-server --config /path/to/music.json --host ${HOST} --port ${PORT} ${DEVICES} --busy-timeout-ms 0

decision:
  # Answers typed questions in one forward pass; served at /v1/systemone
  clef:
    q8_0: llama-server --parallel ${PARALLEL} ${DEVICES} --host ${HOST} --port ${PORT} --model /path/to/Clef-Flash-Q8_0.gguf -c ${CTX} -ngl 99
```

- `-ngl 99`: offloads layers to GPU (adjust based on your hardware).
- `-c ${CTX}`, `--parallel ${PARALLEL}`: the context window and slots, set per model in the UI. Two slots of 8192 ask the engine for 16384.
- `${DEVICES}`: drop it from a profile and that model ignores the GPU picker.

**Default quantization:** the **first** quant listed for a model is its default. Through the OpenAI-compatible API (`/v1/models`, `/v1/chat/completions`) that default is exposed under the bare model id (e.g. `qwen-3`), while the rest get a `-<quant>` suffix (e.g. `qwen-3-q8_k_xl`). Requesting the bare id loads the first-listed quant, so order each model's variants with the preferred one first.

**Model ids are global.** If the same key appears in two sections, the first one wins, in the order `llm`, `sd`, `tts`, `stt`, `music`, `decision`, and the duplicate is logged and skipped.

**Older configs still work.** A pre-sections config that puts everything under a single `models:` key is read as the `llm` section; an explicit `llm:` section always wins over it.

**Memory figures need trace logging.** llama.cpp prints its buffer sizes only at log level 4, so language and decision models are started with `LLAMA_ARG_LOG_VERBOSITY=4`. That level is noisier for the whole run, not just the load: each request adds prompt-cache lines. An idle model would also log `all slots are idle` once a second; the panel drops that one line.

## API

A request that names a `model` goes to the instance serving that model, loading it first if it is not up. A request that names none (the chat endpoints always need one) goes to the one running model of that kind: with none running it gets `503`, with several it gets `409` asking which. Naming a model of another kind, say an image model on the chat endpoint, gets `409` with the endpoint that serves it.

Requests to the engines go straight to them and ignore `http_proxy`, so a system-wide proxy does not get in the way.

| Endpoint | Section |
| --- | --- |
| `/v1/models` | every section; each entry carries `type` and, in the ollama-style list, `kind` and `capabilities` |
| `/v1/chat/completions`, `/v1/completions`, `/v1/responses` | `llm` |
| `/v1/audio/transcriptions`, `/v1/audio/translations` | `stt` when the request names an `stt` model, or names none while one is running; otherwise `llm`, where these are `llama-server`'s own ASR endpoints |
| `/v1/images/*` (OpenAI), `/sdapi/v1/*` (AUTOMATIC1111), `/sdcpp/v1/*` (native) | `sd` |
| `/v1/audio/speech`, `/v1/audio/voices` | `tts` |
| `/v1/tasks/*` | `music`, `audio.cpp`'s own generation endpoint |
| `/v1/systemone` | `decision` |

Qwen3-ASR starts its transcripts with a `language <Name><asr_text>` header. It is stripped from transcription and chat answers alike, streamed or not, so clients get the text alone.

Management:

| Endpoint | Does |
| --- | --- |
| `GET /api/status[?model=]` | every running instance, plus the remembered settings |
| `POST /api/start` | start a model with the given settings |
| `POST /api/stop[?model=]` | stop one model, or all of them |
| `POST /api/reset[?model=\|?kind=]` | settings back to defaults, restarting what they apply to; without a query, everything |
| `POST /api/multi` | `{"multi": true}` to keep several models, `false` to evict on every start |
| `GET /api/logs[?model=]`, `GET /api/logs/owner[?model=]`, `POST /api/logs/clear[?model=]` | a model's log |
| `GET /api/devices` | the cards the engine reports, with PCI addresses |
| `GET /api/config` | the config, grouped by section |

## Usage

1. **Start the server:**
   ```bash
   llama-switch --config config.yaml --watch
   ```

2. **Open the Dashboard:**
   Navigate to `http://localhost:11435` in your browser.

3. **Control Models:**
   - Pick a section tab, click a model, set its GPU, context and the rest, then click **Load**.
   - Turn on **Multi-model** to keep several models up; click any running row, here or on the **Running** tab, to switch the dashboard to it.
   - Change any field to restart that model with it; **Reset** puts the model back on defaults, **Reset All** does it for everything.
   - Click **Open WebUI** to access the running engine's native interface.
   - Click **Unload** on a model, or **Stop All**.

### Command line

| Flag | Default | Meaning |
| --- | --- | --- |
| `-H`, `--host` | `localhost` | interface for the UI **and** for the engines it launches |
| `-p`, `--port` | `11435` | UI port (engines get a free port each) |
| `-c`, `--ctx` | `4096` | default context window, per slot |
| `-np`, `--parallel` | `1` | default number of slots |
| `-f`, `--config` | `config.yaml` | config file path |
| `-w`, `--watch` | off | reload the config when it changes on disk |

### As a systemd service

The repository carries a user unit, [llama-switch.service](llama-switch.service), that runs `~/.local/bin/llama-switch` on `~/.config/llama-switch/config.yaml` with `--watch`. Add whatever flags you need to its `ExecStart` (`--host 0.0.0.0` to reach the dashboard from other machines), then:

```bash
cp llama-switch.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-switch
```

The server's own messages, config reloads and the command line of every start among them, go to the journal: `journalctl --user -u llama-switch -f`. What the engines print is in each model's **Upstream Logs** on the dashboard.

### The microphone

Everything is served over plain `http`. Browsers only hand over the microphone in a secure context, and `localhost` is the only plain-http address that counts as one, so recording controls are unavailable from a phone or a laptop reaching this machine by hostname or IP.

> Note that `--watch` reloads the config by **stopping every running model**. Do not edit the config while a model is mid-generation.
