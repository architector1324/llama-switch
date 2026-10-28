package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// The engines the tests start are this test binary, run again with
// fakeEngineEnv set: no llama.cpp, GPU or interpreter needed, and the real
// process control, pipes and log parsing all take part.
const (
	fakeEngineEnv = "LLAMA_SWITCH_FAKE_ENGINE"
	fakeParentEnv = "LLAMA_SWITCH_FAKE_PARENT"
)

func TestMain(m *testing.M) {
	if kind := os.Getenv(fakeEngineEnv); kind != "" {
		runFakeEngine(kind, os.Args[1:])
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// The start of a profile that runs a fake engine of this kind.
func fakeEngine(kind string) string {
	exe, err := os.Executable()
	if err != nil {
		panic(err)
	}
	return fmt.Sprintf("%s=%s %s=%d '%s'", fakeEngineEnv, kind, fakeParentEnv, os.Getpid(), exe)
}

func argValue(args []string, name string) string {
	for i, a := range args {
		if a == name && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

func runFakeEngine(kind string, args []string) {
	// Engines run in a session of their own, so a test run that dies would leave
	// them behind; they leave with it instead.
	if parent, err := strconv.Atoi(os.Getenv(fakeParentEnv)); err == nil {
		go func() {
			for range time.Tick(200 * time.Millisecond) {
				if syscall.Kill(parent, 0) != nil {
					os.Exit(0)
				}
			}
		}()
	}

	say := func(format string, a ...any) { fmt.Printf(format+"\n", a...) }
	if kind == "crash" {
		say("fatal: model file not found")
		os.Exit(1)
	}

	say("\x1b[32mfake %s: %s\x1b[0m", kind, strings.Join(args, " "))
	if kind == "llm" || kind == "decision" {
		say("load_tensors:      Vulkan0 model buffer size =   610.56 MiB")
		say("load_tensors:   CPU_Mapped model buffer size =   100.00 MiB")
		say("llama_kv_cache:    Vulkan0 KV buffer size =    51.00 MiB")
		say("srv load_model: [mtmd] estimated worst-case memory usage of mmproj is 1817.01 MiB")
		say("clip_ctx: CLIP using Vulkan1 backend")
		say("clip_ctx: CLIP using Vulkan1 backend")
		say("0.00.100.000 I srv  update_slots: all slots are idle")
	}
	if ms, _ := strconv.Atoi(argValue(args, "--delay-ms")); ms > 0 {
		time.Sleep(time.Duration(ms) * time.Millisecond)
	}

	l, err := net.Listen("tcp", net.JoinHostPort(argValue(args, "--host"), argValue(args, "--port")))
	if err != nil {
		say("bind failed: %v", err)
		os.Exit(1)
	}
	say("main: server is listening on http://%s", l.Addr())
	http.Serve(l, fakeHandler(kind, args))
}

// What the fake engine saw of a request, sent back so a test can check what
// the proxy forwarded.
func fakeEcho(kind string, args []string, r *http.Request, body []byte) map[string]any {
	echo := map[string]any{
		"kind":              kind,
		"pid":               os.Getpid(),
		"args":              strings.Join(args, " "),
		"method":            r.Method,
		"path":              r.URL.EscapedPath(),
		"query":             r.URL.RawQuery,
		"content_type":      r.Header.Get("Content-Type"),
		"transfer_encoding": strings.Join(r.TransferEncoding, ","),
		"len":               len(body),
	}
	if i := bytes.Index(body, []byte("filename=")); i >= 0 {
		part := body[i:]
		if j := bytes.Index(part, []byte("\r\n\r\n")); j >= 0 && len(part) >= j+8 {
			echo["file_head"] = string(part[:j])
			echo["file_magic"] = string(part[j+4 : j+8])
		}
	}
	return echo
}

func fakeHandler(kind string, args []string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		echo := fakeEcho(kind, args, r, body)
		send := func(v any) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(v)
		}

		switch r.URL.Path {
		case "/slots":
			send([]map[string]any{{
				"id": 0, "n_ctx": 8192, "n_prompt_tokens": 42, "is_processing": false,
				"next_token": []map[string]any{{"n_decoded": 0}},
			}})

		case "/v1/chat/completions", "/v1/completions":
			fmt.Println("prompt eval time =       4.67 ms /    11 tokens (    0.42 ms per token,  2355.46 tokens per second)")
			fmt.Println("       eval time =      50.00 ms /    20 tokens (    2.50 ms per token,   400.00 tokens per second)")
			fmt.Println("slot      release: id  0 | task 0 | stop processing: n_tokens = 73, truncated = 0")
			var req map[string]any
			json.Unmarshal(body, &req)
			if req["stream"] != true {
				send(map[string]any{
					"id":   "x",
					"echo": echo,
					"choices": []map[string]any{{"index": 0, "message": map[string]any{
						"role": "assistant", "content": "language English<asr_text>Hi & <bye>",
					}}},
				})
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			for _, tok := range []string{"language", " English", "<asr", "_text>", "Hello", " world"} {
				quoted, _ := json.Marshal(tok)
				fmt.Fprintf(w, `data: {"id":"x","created":12345678901234567,"choices":[{"index":0,"delta":{"content":%s}}]}`+"\n\n", quoted)
				w.(http.Flusher).Flush()
			}
			fmt.Fprint(w, "data: [DONE]\n\n")

		case "/v1/images/generations":
			echo["data"] = []map[string]any{{"b64_json": base64.StdEncoding.EncodeToString([]byte("fake image"))}}
			echo["output_format"] = "png"
			send(echo)

		case "/v1/audio/transcriptions":
			send(map[string]any{"text": "language English<asr_text>hello", "echo": echo})

		default:
			send(echo)
		}
	})
}
