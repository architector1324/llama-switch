package main

import (
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Feed lines through the reader the way a backend's pipe would.
func readLog(t *testing.T, ctx int, output string) *Instance {
	t.Helper()
	inst := &Instance{Ctx: ctx, stats: newStats()}
	logReader(io.NopCloser(strings.NewReader(output)), inst)
	return inst
}

func approx(a, b float64) bool { return math.Abs(a-b) < 0.001 }

func TestEachOutputLineSplitsOnCR(t *testing.T) {
	var got []string
	eachOutputLine(strings.NewReader("a\rb\r\nc\n\nd \x1b[32mgreen\x1b[0m  \ncaf\xe9"), func(l string) {
		got = append(got, l)
	})
	want := []string{"a", "b", "c", "d green", "caf�"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestMatchEvalSkipsPromptEval(t *testing.T) {
	prompt := "prompt eval time =       4.67 ms /    11 tokens (    0.42 ms per token,  2355.46 tokens per second)"
	if m := matchEval(prompt); m != nil {
		t.Errorf("prompt line read as generation: %q", m)
	}
	gen := "       eval time =      50.00 ms /    20 tokens (    2.50 ms per token,   400.00 tokens per second)"
	if m := matchEval(gen); m == nil || m[1] != "20" || m[2] != "400.00" {
		t.Errorf("generation line: %q", m)
	}
}

func TestLogReaderLlama(t *testing.T) {
	inst := readLog(t, 2048, strings.Join([]string{
		"\x1b[32mbuild: 1234\x1b[0m",
		"load_tensors:      Vulkan0 model buffer size =   610.56 MiB",
		"load_tensors:   CPU_Mapped model buffer size =   100.00 MiB",
		"llama_kv_cache:    Vulkan0 KV buffer size =    51.00 MiB",
		"llama_kv_cache:    Vulkan1 KV buffer size =    49.00 MiB",
		"srv load_model: [mtmd] estimated worst-case memory usage of mmproj is 1817.01 MiB",
		"clip_ctx: CLIP using Vulkan1 backend",
		"clip_ctx: CLIP using Vulkan1 backend",
		"0.46.251.374 I srv  update_slots: all slots are idle",
		"main: server is listening on http://127.0.0.1:8080",
		"prompt eval time =       4.67 ms /    11 tokens (    0.42 ms per token,  2355.46 tokens per second)",
		"       eval time =      50.00 ms /    20 tokens (    2.50 ms per token,   400.00 tokens per second)",
		"slot      release: id  3 | task 10 | stop processing: n_tokens = 73, truncated = 0",
		"       eval time =      30.00 ms /    15 tokens (    2.00 ms per token,   500.00 tokens per second)",
	}, "\n"))

	if !inst.ready {
		t.Error("not ready after 'server is listening'")
	}
	if len(inst.logs) != 13 || inst.logs[0] != "build: 1234" {
		t.Errorf("logs %q: the idle line goes, colours are stripped", inst.logs)
	}

	s := inst.stats
	if !approx(s.MemModel, 610.56+100+1817.01) || !approx(s.MemContext, 100) {
		t.Errorf("memory %v / %v", s.MemModel, s.MemContext)
	}
	if !approx(s.MemDevices["Vulkan1"].Model, 1817.01) || !approx(s.MemDevices["Vulkan1"].Context, 49) {
		t.Errorf("Vulkan1 %+v: the tower is charged once", s.MemDevices["Vulkan1"])
	}
	if !approx(s.MemDevices["Vulkan0"].Model, 610.56) || !approx(s.MemDevices["CPU_Mapped"].Model, 100) {
		t.Errorf("devices %+v", s.MemDevices)
	}
	if !approx(s.PromptSpeed, 2355.46) || !approx(s.GenSpeed, 500) || s.TotalTokens != 35 {
		t.Errorf("speeds %v / %v, tokens %d", s.PromptSpeed, s.GenSpeed, s.TotalTokens)
	}
	if s.CtxUsed != 73 || s.CtxLimit != 2048 {
		t.Errorf("ctx %d / %d", s.CtxUsed, s.CtxLimit)
	}
}

func TestLogReaderReadyLines(t *testing.T) {
	for _, line := range []string{
		"main: model loaded",
		`{"msg":"model loaded"}`,
		"llama_server: server is listening on http://0.0.0.0:1",
		"[INFO] listening on 127.0.0.1:1",
	} {
		if inst := readLog(t, 0, line+"\n"); !inst.ready {
			t.Errorf("%q does not mark the engine ready", line)
		}
	}
	if inst := readLog(t, 0, "loading model\n"); inst.ready {
		t.Error("ready too early")
	}
}

func TestLogReaderSd(t *testing.T) {
	inst := readLog(t, 0, strings.Join([]string{
		"backend_fit.cpp - DiT params 6272 MiB, compute reserve 2048 MiB -> compute Vulkan1, params Vulkan1",
		"backend_fit.cpp - Conditioner params 4000 MiB, compute reserve 100 MiB -> compute Vulkan0, params Vulkan0",
		"backend_fit.cpp - VAE params 319 MiB, compute reserve 100 MiB -> compute Vulkan0, params Vulkan0",
		"diffusion_engine.cpp - total params memory size = 10509.14MB (VRAM 10509.14MB, RAM 0.00MB): " +
			"text_encoders 4076.43MB(VRAM), diffusion_model 6272.71MB(VRAM), vae 160.00MB(VRAM)",
		"[INFO ] stable-diffusion.cpp:5592 - generate_image 512x768",
		"  |====>     | 3/8 - 6.59s/it\r  |========> | 8/8 - 1.50it/s\r",
		"[INFO ] stable-diffusion.cpp:5675 - sampling completed, taking 23.84s",
		"ggml_runner.cpp - z_image compute buffer size: 690.12 MB(VRAM) on Vulkan1 (peak across 1 segment)",
		"ggml_runner.cpp - vae compute buffer size: 300.00 MB(VRAM) on Vulkan0 (peak across 1 segment)",
		"ggml_runner.cpp - qwen3 compute buffer size: 100.00 MB(VRAM) on Vulkan0 (peak across 1 segment)",
		// A component re-reports its peak as it grows: replaced, not added.
		"ggml_runner.cpp - vae compute buffer size: 350.00 MB(VRAM) on Vulkan0 (peak across 1 segment)",
	}, "\n"))

	s := inst.stats
	if !approx(s.MemModel, 10509.14) || !approx(s.MemCompute, 690.12+350+100) {
		t.Errorf("params %v, compute %v", s.MemModel, s.MemCompute)
	}
	v0, v1 := s.MemDevices["Vulkan0"], s.MemDevices["Vulkan1"]
	if !approx(v0.Model, 4076.43+160) || !approx(v0.Compute, 450) {
		t.Errorf("Vulkan0 %+v", v0)
	}
	if !approx(v1.Model, 6272.71) || !approx(v1.Compute, 690.12) {
		t.Errorf("Vulkan1 %+v", v1)
	}
	// 1.50 it/s is 0.67 s/it; the bar parks at 100% once sampling is done.
	if s.SdSpeed != 0.67 || s.SdStep != 8 || s.SdSteps != 8 || s.SdImages != 1 ||
		s.SdLastTime != 23.84 || s.SdSize != "512x768" {
		t.Errorf("sd counters %+v", s)
	}
}

func TestLogReaderTts(t *testing.T) {
	inst := readLog(t, 0, "[Pipeline] Generated 1920 frames (slot 0)\n"+
		"[Perf] Total 36762.3 ms (2048 frames, 16.67 ms/frame AR, audio 163.84 s, RTF 0.224)\n")
	s := inst.stats
	if s.TtsFrames != 2048 || !approx(s.TtsLastTime, 36.7623) || s.TtsAudio != 163.84 ||
		s.TtsRtf != 0.224 || s.TtsClips != 1 {
		t.Errorf("tts counters %+v", s)
	}
}

func TestLogReaderDecision(t *testing.T) {
	inst := readLog(t, 4096, "1.17.598.882 I slot launch_slot_: id  0 | task 0 | processing task\n"+
		"1.18.108.321 I slot      release: id  0 | task 0 | stop processing: n_tokens = 449\n"+
		// A release with no launch before it is not a decision.
		"1.19.000.000 I slot      release: id  0 | task 1 | stop processing: n_tokens = 10\n")
	s := inst.stats
	// The engine's own clock: 78.108321 s - 77.598882 s.
	if !approx(s.DecisionMs, 509.439) || s.DecisionTokens != 449 || s.DecisionCount != 1 {
		t.Errorf("decision %v ms, %d tokens, %d served", s.DecisionMs, s.DecisionTokens, s.DecisionCount)
	}
}

func TestSlotsPollerRates(t *testing.T) {
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/slots" {
			http.NotFound(w, r)
			return
		}
		n := calls.Add(1)
		// Two slots: the gauge reads the fullest one, the rate sums them.
		fmt.Fprintf(w, `[{"n_ctx":4096,"n_prompt_tokens":100,"is_processing":true,"n_decoded":%d,"n_prompt_tokens_processed":100},
			{"n_ctx":4096,"n_prompt_tokens":300,"is_processing":false,"next_token":[{"n_decoded":%d}]}]`, 25*n, 25*n)
	}))
	defer srv.Close()
	_, portStr, _ := net.SplitHostPort(strings.TrimPrefix(srv.URL, "http://"))
	port, _ := strconv.Atoi(portStr)

	inst := &Instance{Port: port, done: make(chan struct{}), stats: newStats()}
	finished := make(chan struct{})
	go func() {
		slotsPoller(inst)
		close(finished)
	}()

	deadline := time.Now().Add(6 * time.Second)
	for {
		inst.mu.Lock()
		s := inst.stats
		inst.mu.Unlock()
		if s.GenSpeed > 0 {
			// 50 tokens a poll, a poll a second.
			if s.GenSpeed < 35 || s.GenSpeed > 70 {
				t.Errorf("gen speed %v, want about 50", s.GenSpeed)
			}
			if s.PromptSpeed != 0 {
				t.Errorf("prompt speed %v from a counter that never moved", s.PromptSpeed)
			}
			if s.CtxUsed != 300 || s.CtxLimit != 4096 {
				t.Errorf("ctx %d / %d", s.CtxUsed, s.CtxLimit)
			}
			// One slot at work, and past its prompt: it has written tokens.
			if s.SlotsBusy != 1 || s.SlotsPrefill != 0 {
				t.Errorf("%d slots busy, %d on the prompt", s.SlotsBusy, s.SlotsPrefill)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("no rate after several polls")
		}
		time.Sleep(50 * time.Millisecond)
	}

	close(inst.done)
	select {
	case <-finished:
	case <-time.After(3 * time.Second):
		t.Error("the poller outlived its process")
	}
}
