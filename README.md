# Llama Switch

![](./ref-dark.png)

A lightweight, modern WebUI manager for switching between [llama.cpp](https://github.com/ggerganov/llama.cpp) models seamlessly.

Llama Switch provides a clean dashboard to list your local LLM models, start/stop them on demand, and monitor real-time generation statistics (tokens/sec, prompt processing speed, context usage).

## Features

- **Model Management**: Define multiple models in a simple YAML config.
- **One-Click Switching**: Automatically stops the current model and starts the new one.
- **Real-time Dashboard**:
  - **Generation Speed:** Tokens per second (t/s).
  - **Prompt Speed:** Prompt processing t/s.
  - **Context Tracker:** Visual usage of the context window.
  - **Total Tokens:** Token count for the current session.
- **Configurable Context**: Adjust the context window size on the fly before loading a model.
- **Upstream Logs**: View raw `llama-server` logs directly in the UI.
- **Theme Support**: Built-in Dark and Light modes.
- **Proxy/WebUI Link**: Provides a direct link to the running `llama-server` WebUI.

## Prerequisites

- **Python 3.8+**
- **llama-server**: The binary from `llama.cpp` must be installed and accessible in your system/shell path, or specified by absolute path in the config.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/llama-switch.git
   cd llama-switch
   ```

2. **Install Python dependencies:**
   ```bash
   pip install fastapi uvicorn pyyaml httpx
   ```

## Configuration

Create a `config.yaml` file in the root directory. You can define as many models as you like.

Use `${PORT}` and `${CTX}` placeholders in your command string; the server will inject the values automatically (default port is `11435`, or the one assigned by the app).

**Example `config.yaml`:**

```yaml
models:
  # Simple Text Model
  mistral-7b:
    cmd: llama-server --port ${PORT} --model /path/to/models/mistral-7b-v0.3.Q4_K_M.gguf -c ${CTX} -ngl 99

  # Vision Model (with mmproj)
  llava-v1.6:
    cmd: llama-server --port ${PORT} --model /path/to/models/llava-v1.6-7b.Q4_K_M.gguf --mmproj /path/to/models/mmproj-model-f16.gguf -c ${CTX} -ngl 99
```

- `-ngl 99`: Offloads layers to GPU (adjust based on your hardware).
- `-c ${CTX}`: Sets the context window (controllable via UI).

## Usage

1. **Start the server:**
   ```bash
   python server.py --config config.yaml
   ```

2. **Open the Dashboard:**
   Navigate to `http://localhost:11435` in your browser.

3. **Control Models:**
   - Select a model from the left sidebar and click **Load**.
   - Change the **Context Window** in the dashboard if needed (default 4096).
   - Click **Open WebUI** to access the native `llama.cpp` interface.
   - Click **Stop** or load another model to terminate the current session.
