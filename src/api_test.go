package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// --- Harness ---

var testDevices = []Device{
	{ID: "Vulkan0", Name: "Fake GPU", TotalMiB: 16304, FreeMiB: 16000},
	{ID: "Vulkan1", Name: "Fake GPU", TotalMiB: 16304, FreeMiB: 16000},
}

const testConfigTemplate = `
llm:
  zeta:
    q4: {llm} --port ${PORT} --host ${HOST} ${DEVICES} -c ${CTX} --parallel ${PARALLEL} ${PRESERVE_THINK} --mmproj x
    q8: {llm} --port $PORT --host $HOST -c $CTX
  legacy:
    cmd: {llm} --port ${PORT} --host ${HOST}
  slow:
    q1: {llm} --port ${PORT} --host ${HOST} --delay-ms 300
  broken:
    q1: {crash}
sd:
  img:
    q8_0: {sd} --port ${PORT} --host ${HOST} ${DEVICES:--backend diffusion=Vulkan1,te=Vulkan0,vae=Vulkan0} -W ${WIDTH} -H ${HEIGHT} ${HIRES:--hires --strength 0.5}
  img2:
    q8: {sd} --port ${PORT} --host ${HOST}
tts:
  voice:
    q8: {tts} --port ${PORT} --host ${HOST} ${DEVICES} --max-new ${FRAMES}
stt:
  ears:
    f16: {stt} --port ${PORT} --host ${HOST} ${DEVICES}
music:
  song:
    q8: {music} --port ${PORT} --host ${HOST}
    sfx: {music} --port ${PORT} --host ${HOST} --sfx
decision:
  clef:
    q8_0: {decision} --port ${PORT} --host ${HOST} ${DEVICES} -c ${CTX}
  zeta:
    dup: echo duplicate
`

func testConfig() string {
	r := strings.NewReplacer(
		"{llm}", fakeEngine("llm"), "{sd}", fakeEngine("sd"), "{tts}", fakeEngine("tts"),
		"{stt}", fakeEngine("stt"), "{music}", fakeEngine("music"),
		"{decision}", fakeEngine("decision"), "{crash}", fakeEngine("crash"),
	)
	return r.Replace(testConfigTemplate)
}

// Everything back to how a fresh server starts, minus the device probe: the
// fake cards stand in for llama-server --list-devices.
func resetState() {
	opMu.Lock()
	stopAll()
	opMu.Unlock()

	state.mu.Lock()
	state.multi = true
	state.defaultCtx, state.selectedCtx = 4096, 0
	state.defaultParallel, state.selectedParallel = 1, 0
	state.defaultFrames, state.selectedFrames = 2048, 0
	state.defaultRes, state.selectedRes = "1024x1024", ""
	state.selectedHires, state.selectedPreserveThink = false, false
	state.selectedDevices = map[string]string{}
	state.host = "127.0.0.1"
	state.mu.Unlock()

	devicesMu.Lock()
	devicesCache = testDevices
	devicesMu.Unlock()
}

func useConfig(t *testing.T, text string, watch bool) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(text), 0o600); err != nil {
		t.Fatal(err)
	}
	state.cfg = newConfigManager(path, watch, onConfigChange)
	return path
}

// Never through the proxy the environment may name.
var testClient = &http.Client{Transport: &http.Transport{}, Timeout: 30 * time.Second}

type harness struct {
	t   *testing.T
	url string
}

func newHarnessWith(t *testing.T, watch bool) (*harness, string) {
	t.Helper()
	resetState()
	path := useConfig(t, testConfig(), watch)
	srv := httptest.NewServer(routes())
	t.Cleanup(func() {
		srv.Close()
		resetState()
	})
	return &harness{t: t, url: srv.URL}, path
}

func newHarness(t *testing.T) *harness {
	h, _ := newHarnessWith(t, false)
	return h
}

// body goes as JSON unless it is already []byte or an io.Reader; header is
// name, value pairs.
func (h *harness) request(method, path string, body any, header ...string) *http.Response {
	h.t.Helper()
	var rd io.Reader
	switch b := body.(type) {
	case nil:
	case []byte:
		rd = bytes.NewReader(b)
	case io.Reader:
		rd = b
	default:
		raw, err := json.Marshal(b)
		if err != nil {
			h.t.Fatal(err)
		}
		rd = bytes.NewReader(raw)
		header = append([]string{"Content-Type", "application/json"}, header...)
	}
	req, err := http.NewRequest(method, h.url+path, rd)
	if err != nil {
		h.t.Fatal(err)
	}
	for i := 0; i+1 < len(header); i += 2 {
		req.Header.Set(header[i], header[i+1])
	}
	resp, err := testClient.Do(req)
	if err != nil {
		h.t.Fatalf("%s %s: %v", method, path, err)
	}
	return resp
}

func (h *harness) raw(method, path string, body any, header ...string) (int, []byte, http.Header) {
	h.t.Helper()
	resp := h.request(method, path, body, header...)
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, raw, resp.Header
}

func (h *harness) call(method, path string, body any, header ...string) (int, map[string]any) {
	h.t.Helper()
	code, raw, _ := h.raw(method, path, body, header...)
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		h.t.Fatalf("%s %s: answer is not a JSON object: %q", method, path, raw)
	}
	return code, out
}

func (h *harness) ok(method, path string, body any, header ...string) map[string]any {
	h.t.Helper()
	code, out := h.call(method, path, body, header...)
	if code != http.StatusOK {
		h.t.Fatalf("%s %s: status %d, %v", method, path, code, out)
	}
	return out
}

func (h *harness) status() map[string]any {
	h.t.Helper()
	return h.ok("GET", "/api/status", nil)
}

func instancesOf(status map[string]any) []map[string]any {
	var out []map[string]any
	for _, v := range status["instances"].([]any) {
		out = append(out, v.(map[string]any))
	}
	return out
}

func (h *harness) running() []string {
	h.t.Helper()
	var names []string
	for _, i := range instancesOf(h.status()) {
		names = append(names, i["model"].(string))
	}
	sort.Strings(names)
	return names
}

// The model's entry in /api/status, or nil when it is not there.
func (h *harness) instance(model string) map[string]any {
	h.t.Helper()
	for _, i := range instancesOf(h.status()) {
		if i["model"] == model {
			return i
		}
	}
	return nil
}

func (h *harness) waitFor(what string, cond func() bool) {
	h.t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			h.t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func (h *harness) waitReady(model string) map[string]any {
	h.t.Helper()
	var inst map[string]any
	h.waitFor(model+" to be ready", func() bool {
		inst = h.instance(model)
		return inst != nil && inst["ready"] == true
	})
	return inst
}

func (h *harness) start(req map[string]any) map[string]any {
	h.t.Helper()
	res := h.ok("POST", "/api/start", req)
	h.waitReady(req["model_key"].(string))
	return res
}

func detail(m map[string]any) string {
	s, _ := m["detail"].(string)
	return s
}

func statsOf(inst map[string]any) map[string]any {
	return inst["stats"].(map[string]any)
}

func near(a any, b float64) bool {
	f, ok := a.(float64)
	return ok && math.Abs(f-b) < 0.01
}

// Top-level keys of a JSON object, in the order they were sent.
func objectKeys(t *testing.T, raw json.RawMessage) []string {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(raw))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		t.Fatalf("not an object: %s", raw)
	}
	var keys []string
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			t.Fatal(err)
		}
		keys = append(keys, tok.(string))
		var skip json.RawMessage
		if err := dec.Decode(&skip); err != nil {
			t.Fatal(err)
		}
	}
	return keys
}

func multipartBody(t *testing.T, fields map[string]string, filename string, file []byte) ([]byte, string) {
	t.Helper()
	var b bytes.Buffer
	mw := multipart.NewWriter(&b)
	names := make([]string, 0, len(fields))
	for k := range fields {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		mw.WriteField(k, fields[k])
	}
	if filename != "" {
		fw, _ := mw.CreateFormFile("file", filename)
		fw.Write(file)
	}
	mw.Close()
	return b.Bytes(), mw.FormDataContentType()
}

// The whole process group is gone, not just the shell that led it.
func groupGone(pid int) bool {
	return syscall.Kill(-pid, 0) != nil
}

// --- Config and listings ---

func TestConfigEndpointKeepsConfigOrder(t *testing.T) {
	h := newHarness(t)
	_, raw, _ := h.raw("GET", "/api/config", nil)

	var top map[string]json.RawMessage
	if err := json.Unmarshal(raw, &top); err != nil {
		t.Fatal(err)
	}
	if got := objectKeys(t, top["sections"]); !reflect.DeepEqual(got, sections) {
		t.Errorf("section order %v, want %v", got, sections)
	}

	var grouped map[string]json.RawMessage
	json.Unmarshal(top["sections"], &grouped)
	if got, want := objectKeys(t, grouped["llm"]), []string{"zeta", "legacy", "slow", "broken"}; !reflect.DeepEqual(got, want) {
		t.Errorf("llm model order %v, want %v", got, want)
	}
	var llm map[string]json.RawMessage
	json.Unmarshal(grouped["llm"], &llm)
	if got := objectKeys(t, llm["zeta"]); !reflect.DeepEqual(got, []string{"q4", "q8"}) {
		t.Errorf("zeta quant order %v: the first one is the default", got)
	}

	var kinds map[string]string
	json.Unmarshal(top["kinds"], &kinds)
	if kinds["zeta"] != "llm" || kinds["clef"] != "decision" || kinds["ears"] != "stt" {
		t.Errorf("kinds %v", kinds)
	}
	// The duplicate under decision is skipped in the merged view.
	var models map[string]map[string]string
	json.Unmarshal(top["models"], &models)
	if _, dup := models["zeta"]["dup"]; dup {
		t.Error("decision's duplicate zeta replaced the llm one")
	}
}

func TestV1ModelsListsEverySection(t *testing.T) {
	h := newHarness(t)
	out := h.ok("GET", "/v1/models", nil)

	byID := map[string]map[string]any{}
	for _, m := range out["data"].([]any) {
		entry := m.(map[string]any)
		byID[entry["id"].(string)] = entry
	}
	for id, kind := range map[string]string{
		"zeta": "llm", "zeta-q8": "llm", "legacy": "llm", "img": "sd", "voice": "tts",
		"ears": "stt", "song": "music", "song-sfx": "music", "clef": "decision",
	} {
		if byID[id] == nil || byID[id]["type"] != kind {
			t.Errorf("%s: %v, want type %s", id, byID[id], kind)
		}
	}
	for _, missing := range []string{"zeta-q4", "legacy-cmd", "song-q8"} {
		if byID[missing] != nil {
			t.Errorf("%s listed: the default quant goes under the bare id only", missing)
		}
	}
	if meta := byID["zeta"]["meta"].(map[string]any); meta["n_ctx"] != 4096.0 {
		t.Errorf("zeta n_ctx %v, want the default 4096", meta["n_ctx"])
	}
	if meta := byID["img"]["meta"].(map[string]any); meta["n_ctx"] != 0.0 {
		t.Errorf("img n_ctx %v: an image model has no context window", meta["n_ctx"])
	}

	caps := map[string][]any{}
	for _, m := range out["models"].([]any) {
		entry := m.(map[string]any)
		caps[entry["name"].(string)] = entry["capabilities"].([]any)
	}
	if !reflect.DeepEqual(caps["zeta"], []any{"completion", "chat", "multimodal"}) {
		t.Errorf("zeta capabilities %v: its profile loads an mmproj", caps["zeta"])
	}
	if !reflect.DeepEqual(caps["zeta-q8"], []any{"completion", "chat"}) {
		t.Errorf("zeta-q8 capabilities %v", caps["zeta-q8"])
	}
	if !reflect.DeepEqual(caps["ears"], []any{"audio", "transcription"}) {
		t.Errorf("ears capabilities %v", caps["ears"])
	}
}

// --- Starting and stopping ---

func TestStartReportsItsSettings(t *testing.T) {
	h := newHarness(t)
	res := h.ok("POST", "/api/start", map[string]any{
		"model_key": "zeta", "quantization": "q4", "ctx": 2048, "parallel": 2,
		"devices": " Vulkan1 ", "preserve_think": true,
	})
	cmd := res["command"].(string)
	port := int(res["port"].(float64))
	for _, want := range []string{
		fmt.Sprintf("--port %d", port), "--host 127.0.0.1", "-dev Vulkan1",
		"-c 4096", "--parallel 2", "--reasoning-preserve",
	} {
		if !strings.Contains(cmd, want) {
			t.Errorf("command %q lacks %q", cmd, want)
		}
	}

	inst := h.waitReady("zeta")
	for field, want := range map[string]any{
		"label": "zeta:q4", "kind": "llm", "ctx": 2048.0, "parallel": 2.0, "devices": "Vulkan1",
		"preserve_think": true, "port": float64(port), "running": true,
	} {
		if inst[field] != want {
			t.Errorf("%s = %v, want %v", field, inst[field], want)
		}
	}

	stats := statsOf(inst)
	if !near(stats["mem_model"], 610.56+100+1817.01) || !near(stats["mem_context"], 51) {
		t.Errorf("memory %v / %v", stats["mem_model"], stats["mem_context"])
	}
	devs := stats["mem_devices"].(map[string]any)
	if !near(devs["Vulkan1"].(map[string]any)["model"], 1817.01) {
		t.Errorf("Vulkan1 %v: the vision tower is charged once, where clip put it", devs["Vulkan1"])
	}
	if !near(devs["CPU_Mapped"].(map[string]any)["model"], 100) {
		t.Errorf("CPU_Mapped %v", devs["CPU_Mapped"])
	}

	status := h.status()
	if status["selected_ctx"] != 2048.0 || status["selected_parallel"] != 2.0 ||
		status["selected_preserve_think"] != true {
		t.Errorf("sticky choices not remembered: %v", status)
	}
	if devs := status["selected_devices"].(map[string]any); devs["llm"] != "Vulkan1" {
		t.Errorf("selected_devices %v", devs)
	}
	// The one instance is also described by the flat fields.
	if status["model"] != "zeta" || status["quantization"] != "q4" {
		t.Errorf("flat fields %v / %v", status["model"], status["quantization"])
	}

	// The window the engine really got, polled from /slots.
	h.waitFor("the slots poller", func() bool {
		return statsOf(h.instance("zeta"))["ctx_limit"] == 8192.0
	})

	pid := int(inst["pid"].(float64))
	if out := h.ok("POST", "/api/stop?model=zeta", nil); out["stopped"] != 1.0 {
		t.Errorf("stop: %v", out)
	}
	h.waitFor("the engine to exit", func() bool { return groupGone(pid) })
	if out := h.ok("POST", "/api/stop?model=zeta", nil); out["stopped"] != 0.0 {
		t.Errorf("second stop: %v", out)
	}
	if got := h.running(); len(got) != 0 {
		t.Errorf("still running: %v", got)
	}
}

func TestStartRejectsBadSettings(t *testing.T) {
	h := newHarness(t)
	for _, tc := range []struct {
		req    any
		code   int
		detail string
	}{
		{map[string]any{"model_key": "img", "res": "abc"}, 404, "Resolution must look like 1024x1024, got 'abc'"},
		{map[string]any{"model_key": "img", "res": "1024x1024; rm -rf /"}, 404, "Resolution must look like 1024x1024, got '1024x1024; rm -rf /'"},
		{map[string]any{"model_key": "zeta", "parallel": 0}, 404, "Parallel slots must be 1..64, got 0"},
		{map[string]any{"model_key": "zeta", "parallel": 65}, 404, "Parallel slots must be 1..64, got 65"},
		{map[string]any{"model_key": "zeta", "devices": "Vulkan9, te=Vulkan0"}, 404, "Unknown device(s): Vulkan9; available: Vulkan0, Vulkan1"},
		{map[string]any{"model_key": "zeta", "devices": "Vulkan0;reboot"}, 404, "Unknown device(s): Vulkan0;reboot; available: Vulkan0, Vulkan1"},
		{map[string]any{"model_key": "nope"}, 404, "Model not found in config"},
		{map[string]any{"model_key": "zeta", "quantization": "q1"}, 404, "Quantization q1 not found for model zeta"},
		{map[string]any{"ctx": 4096}, 422, "Field required: model_key"},
	} {
		code, out := h.call("POST", "/api/start", tc.req)
		if code != tc.code || detail(out) != tc.detail {
			t.Errorf("%v: %d %q, want %d %q", tc.req, code, detail(out), tc.code, tc.detail)
		}
	}
	if code, _ := h.call("POST", "/api/start", []byte("{nope"), "Content-Type", "application/json"); code != 422 {
		t.Errorf("invalid JSON: %d, want 422", code)
	}
	if got := h.running(); len(got) != 0 {
		t.Errorf("a rejected start left %v running", got)
	}
}

func TestEvictionFollowsMultiModel(t *testing.T) {
	h := newHarness(t)
	// On from the start: a load leaves the others up.
	if multi := h.status()["multi"]; multi != true {
		t.Errorf("multi at start: %v", multi)
	}
	h.start(map[string]any{"model_key": "zeta"})
	h.start(map[string]any{"model_key": "voice"})
	h.start(map[string]any{"model_key": "img"})
	if got := h.running(); !reflect.DeepEqual(got, []string{"img", "voice", "zeta"}) {
		t.Errorf("multi on: %v", got)
	}
	// Loading a model again replaces it rather than doubling it.
	h.start(map[string]any{"model_key": "zeta", "quantization": "q8"})
	if got := h.running(); !reflect.DeepEqual(got, []string{"img", "voice", "zeta"}) {
		t.Errorf("reload: %v", got)
	}
	if q := h.instance("zeta")["quantization"]; q != "q8" {
		t.Errorf("zeta quant %v after reload", q)
	}

	if code, out := h.call("POST", "/api/multi", map[string]any{}); code != 422 {
		t.Errorf("multi without a value: %d %v", code, out)
	}

	if out := h.ok("POST", "/api/multi", map[string]any{"multi": false}); out["multi"] != false {
		t.Fatalf("multi: %v", out)
	}
	h.start(map[string]any{"model_key": "voice"})
	if got := h.running(); !reflect.DeepEqual(got, []string{"voice"}) {
		t.Errorf("multi off: %v, want only the last start", got)
	}

	if out := h.ok("POST", "/api/stop", nil); out["stopped"] != 1.0 {
		t.Errorf("stop all: %v", out)
	}
}

func TestEngineThatExitsWhileStarting(t *testing.T) {
	h := newHarness(t)
	code, out := h.call("POST", "/v1/chat/completions", map[string]any{"model": "broken"})
	if code != 500 || detail(out) != "broken:q1 exited while starting; see its log" {
		t.Errorf("%d %q", code, detail(out))
	}
	// The log outlives the process: that is where the reason is.
	_, raw, _ := h.raw("GET", "/api/logs?model=broken", nil)
	if !strings.Contains(string(raw), "fatal: model file not found") {
		t.Errorf("log %s", raw)
	}
}

func TestConcurrentRequestsStartAModelOnce(t *testing.T) {
	h := newHarness(t)
	var wg sync.WaitGroup
	pids := make([]any, 2)
	errs := make([]error, 2)
	for n := range pids {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			body := strings.NewReader(`{"model":"slow"}`)
			resp, err := testClient.Post(h.url+"/v1/chat/completions", "application/json", body)
			if err != nil {
				errs[n] = err
				return
			}
			defer resp.Body.Close()
			var out map[string]any
			json.NewDecoder(resp.Body).Decode(&out)
			if resp.StatusCode != 200 {
				errs[n] = fmt.Errorf("status %d: %v", resp.StatusCode, out)
				return
			}
			pids[n] = out["echo"].(map[string]any)["pid"]
		}(n)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if pids[0] != pids[1] {
		t.Errorf("answered by engines %v and %v: the second start killed the first", pids[0], pids[1])
	}
}

// --- Reset ---

func TestResetRestoresDefaultsAndRestarts(t *testing.T) {
	h := newHarness(t)
	h.ok("POST", "/api/multi", map[string]any{"multi": true})

	res := h.start(map[string]any{
		"model_key": "img", "res": " 512X768 ", "hires": true, "devices": "diffusion=Vulkan0",
	})
	for _, want := range []string{
		"--backend diffusion=Vulkan0,te=Vulkan0,vae=Vulkan0", "-W 512 -H 768", "--hires --strength 0.5",
	} {
		if !strings.Contains(res["command"].(string), want) {
			t.Errorf("command %q lacks %q", res["command"], want)
		}
	}
	h.start(map[string]any{"model_key": "voice", "frames": 1024})

	out := h.ok("POST", "/api/reset?model=img", nil)
	if !reflect.DeepEqual(out["restarted"], []any{"img"}) {
		t.Errorf("reset img restarted %v", out["restarted"])
	}
	img := h.waitReady("img")
	if img["res"] != "1024x1024" || img["hires"] != false || img["devices"] != "" {
		t.Errorf("img after reset: res %v hires %v devices %v", img["res"], img["hires"], img["devices"])
	}
	if !strings.Contains(h.status()["selected_res"].(string), "1024x1024") {
		t.Error("selected_res not reset")
	}
	if v := h.instance("voice"); v == nil || v["frames"] != 1024.0 {
		t.Errorf("resetting img touched voice: %v", v)
	}

	// A section with nothing picked: the choices go, nothing restarts.
	out = h.ok("POST", "/api/reset?kind=tts", nil)
	if len(out["restarted"].([]any)) != 0 || h.status()["selected_frames"] != 2048.0 {
		t.Errorf("reset kind: %v, selected_frames %v", out, h.status()["selected_frames"])
	}

	for query, want := range map[string]string{
		"?kind=bogus": "Unknown section bogus",
		"?model=nope": "Model nope not found",
	} {
		if code, out := h.call("POST", "/api/reset"+query, nil); code != 404 || detail(out) != want {
			t.Errorf("reset%s: %d %q", query, code, detail(out))
		}
	}

	// Reset All restarts everything, and must not let the first restart evict
	// the rest even with Multi-model off.
	h.ok("POST", "/api/multi", map[string]any{"multi": false})
	out = h.ok("POST", "/api/reset", nil)
	got := out["restarted"].([]any)
	if len(got) != 2 {
		t.Errorf("reset all restarted %v", got)
	}
	h.waitReady("img")
	if v := h.waitReady("voice"); v["frames"] != 2048.0 {
		t.Errorf("voice frames %v after reset all", v["frames"])
	}
}

// --- Proxying ---

func TestChatAutoloadsAndStripsTheASRHeader(t *testing.T) {
	h := newHarness(t)

	out := h.ok("POST", "/v1/chat/completions", map[string]any{"model": "zeta", "messages": []any{}})
	msg := out["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != "Hi & <bye>" {
		t.Errorf("content %q", msg["content"])
	}
	if q := h.instance("zeta")["quantization"]; q != "q4" {
		t.Errorf("bare id loaded %v, want the first-listed quant", q)
	}
	if echo := out["echo"].(map[string]any); echo["path"] != "/v1/chat/completions" {
		t.Errorf("forwarded to %v", echo["path"])
	}

	code, raw, header := h.raw("POST", "/v1/chat/completions", map[string]any{"model": "zeta", "stream": true})
	if code != 200 || header.Get("Content-Type") != "text/event-stream; charset=utf-8" {
		t.Fatalf("stream: %d %q", code, header.Get("Content-Type"))
	}
	var text strings.Builder
	for _, line := range strings.Split(string(raw), "\n") {
		if !strings.HasPrefix(line, "data: {") {
			continue
		}
		var ev map[string]any
		if err := json.Unmarshal([]byte(line[6:]), &ev); err != nil {
			t.Fatalf("%q: %v", line, err)
		}
		text.WriteString(ev["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)["content"].(string))
	}
	if text.String() != "Hello world" {
		t.Errorf("streamed text %q", text.String())
	}
	if !strings.Contains(string(raw), `"created":12345678901234567`) {
		t.Error("an integer past float precision did not survive the rewrite")
	}
	if !strings.HasSuffix(strings.TrimSpace(string(raw)), "data: [DONE]") {
		t.Errorf("stream does not end with [DONE]: %q", raw)
	}

	// Token counters come from the engine's own timing lines.
	h.waitFor("the timing lines", func() bool {
		return statsOf(h.instance("zeta"))["total_tokens"] == 40.0
	})

	h.ok("POST", "/v1/completions", map[string]any{"model": "zeta-q8", "prompt": "x"})
	if q := h.instance("zeta")["quantization"]; q != "q8" {
		t.Errorf("suffixed id loaded %v", q)
	}
	h.ok("POST", "/api/multi", map[string]any{"multi": false})
	h.ok("POST", "/v1/responses", map[string]any{"model": "legacy"})
	if got := h.running(); !reflect.DeepEqual(got, []string{"legacy"}) {
		t.Errorf("running %v: with Multi-model off a request evicts the rest", got)
	}
}

func TestChatErrors(t *testing.T) {
	h := newHarness(t)
	for _, tc := range []struct {
		body   any
		code   int
		detail string
	}{
		{[]byte("{nope"), 400, "Invalid JSON"},
		{[]byte(""), 400, "Invalid JSON"},
		{map[string]any{"messages": []any{}}, 400, "Model field required"},
		{[]any{"zeta"}, 400, "Model field required"},
		{map[string]any{"model": 5}, 400, "Model field required"},
		{map[string]any{"model": "img"}, 409, "Model img is a 'sd' model, not 'llm'; it is served at /v1/images/generations"},
		{map[string]any{"model": "song"}, 409, "Model song is a 'music' model, not 'llm'; it is served at /v1/tasks/"},
		{map[string]any{"model": "nope"}, 404, "Model nope not found"},
		{map[string]any{"model": "zeta-q4"}, 404, "Model zeta-q4 not found"},
	} {
		code, out := h.call("POST", "/v1/chat/completions", tc.body)
		if code != tc.code || detail(out) != tc.detail {
			t.Errorf("%v: %d %q, want %d %q", tc.body, code, detail(out), tc.code, tc.detail)
		}
	}
	if got := h.running(); len(got) != 0 {
		t.Errorf("a failed request left %v running", got)
	}
}

func TestRequestsWithoutAModelGoToTheOneRunning(t *testing.T) {
	h := newHarness(t)

	if code, out := h.call("POST", "/v1/images/generations", map[string]any{"prompt": "a"}); code != 503 ||
		detail(out) != "No sd model is running" {
		t.Errorf("nothing running: %d %q", code, detail(out))
	}

	h.ok("POST", "/api/multi", map[string]any{"multi": true})
	h.start(map[string]any{"model_key": "img"})
	h.start(map[string]any{"model_key": "img2"})
	if code, out := h.call("POST", "/v1/images/generations", map[string]any{"prompt": "a"}); code != 409 ||
		detail(out) != "Several sd models are running (img, img2); name one in the request" {
		t.Errorf("two running: %d %q", code, detail(out))
	}

	img2 := h.instance("img2")["pid"]
	echo := h.ok("POST", "/v1/images/generations?x=1&y=%20z", map[string]any{"model": "img2", "prompt": "a"})
	if echo["pid"] != img2 || echo["path"] != "/v1/images/generations" || echo["query"] != "x=1&y=%20z" {
		t.Errorf("named JSON request: %v", echo)
	}

	body, ctype := multipartBody(t, map[string]string{"model": "img", "prompt": "p"}, "a.png", []byte("GIF89a-data"))
	echo = h.ok("POST", "/v1/images/edits", body, "Content-Type", ctype)
	if echo["pid"] != h.instance("img")["pid"] || echo["file_magic"] != "GIF8" || echo["len"] != float64(len(body)) {
		t.Errorf("named multipart request: %v", echo)
	}

	h.ok("POST", "/api/stop?model=img2", nil)
	for _, path := range []string{"/sdapi/v1/sd-models", "/sdcpp/v1/x/y"} {
		if echo := h.ok("GET", path, nil); echo["path"] != path || echo["method"] != "GET" || echo["kind"] != "sd" {
			t.Errorf("GET %s: %v", path, echo)
		}
	}

	// Each endpoint resolves within its own section.
	echo = h.ok("POST", "/v1/tasks/run", map[string]any{"model": "song-sfx"})
	if echo["kind"] != "music" || !strings.Contains(echo["args"].(string), "--sfx") {
		t.Errorf("tasks: %v", echo)
	}
	echo = h.ok("POST", "/v1/systemone", map[string]any{"model": "clef"})
	if echo["kind"] != "decision" || echo["path"] != "/v1/systemone" {
		t.Errorf("systemone: %v", echo)
	}
	echo = h.ok("POST", "/v1/audio/speech", map[string]any{"model": "voice", "input": "hi"})
	if echo["kind"] != "tts" {
		t.Errorf("speech: %v", echo)
	}
	echo = h.ok("DELETE", "/v1/audio/voices/abc", nil)
	if echo["method"] != "DELETE" || echo["path"] != "/v1/audio/voices/abc" {
		t.Errorf("voices delete: %v", echo)
	}
}

func TestChunkedUploadIsResentWithALength(t *testing.T) {
	h := newHarness(t)
	h.start(map[string]any{"model_key": "voice"})
	payload := `{"input":"hi"}`
	// A reader of unknown length makes the client send it chunked.
	body := io.MultiReader(strings.NewReader(payload))
	echo := h.ok("POST", "/v1/audio/speech", body, "Content-Type", "application/json")
	if echo["transfer_encoding"] != "" || echo["len"] != float64(len(payload)) {
		t.Errorf("engine saw %v", echo)
	}
}

func TestTranscriptionsPickSttOrLlm(t *testing.T) {
	h := newHarness(t)
	h.ok("POST", "/api/multi", map[string]any{"multi": true})
	wav := []byte("RIFF\x00\x00\x00\x00WAVEdata")
	send := func(fields map[string]string) (int, map[string]any) {
		body, ctype := multipartBody(t, fields, "x.wav", wav)
		return h.call("POST", "/v1/audio/transcriptions", body, "Content-Type", ctype)
	}
	kindOf := func(out map[string]any) any { return out["echo"].(map[string]any)["kind"] }

	code, out := send(map[string]string{"model": "ears"})
	if code != 200 || kindOf(out) != "stt" || out["text"] != "hello" {
		t.Errorf("stt model named: %d %v", code, out)
	}
	if out["echo"].(map[string]any)["file_magic"] != "RIFF" {
		t.Errorf("a WAV upload was rewritten: %v", out["echo"])
	}

	if _, out = send(map[string]string{"model": "zeta"}); kindOf(out) != "llm" || out["text"] != "hello" {
		t.Errorf("llm model named: %v", out)
	}
	if _, out = send(nil); kindOf(out) != "stt" {
		t.Errorf("none named with stt up: %v", out)
	}
	h.ok("POST", "/api/stop?model=ears", nil)
	if _, out = send(nil); kindOf(out) != "llm" {
		t.Errorf("none named without stt: %v", out)
	}
	if code, out := send(map[string]string{"model": "img"}); code != 409 {
		t.Errorf("image model named: %d %v", code, out)
	}

	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("no ffmpeg: conversion not checked")
	}
	mp3 := filepath.Join(t.TempDir(), "in.mp3")
	if err := exec.Command("ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
		"sine=frequency=440:duration=0.5", mp3).Run(); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(mp3)
	body, ctype := multipartBody(t, map[string]string{"model": "ears"}, "x.mp3", data)
	out = h.ok("POST", "/v1/audio/transcriptions", body, "Content-Type", ctype)
	echo := out["echo"].(map[string]any)
	if echo["file_magic"] != "RIFF" || !strings.Contains(echo["file_head"].(string), `filename="audio.wav"`) ||
		!strings.Contains(echo["file_head"].(string), "Content-Type: audio/wav") {
		t.Errorf("mp3 not converted: %v", echo)
	}
}

// --- Logs, CORS, dashboard ---

func TestLogsBelongToTheirInstance(t *testing.T) {
	h := newHarness(t)
	if out := h.ok("GET", "/api/logs/owner", nil); out["model"] != nil || out["kind"] != nil {
		t.Errorf("owner with nothing up: %v", out)
	}

	h.start(map[string]any{"model_key": "zeta"})
	var lines []string
	_, raw, _ := h.raw("GET", "/api/logs?model=zeta", nil)
	json.Unmarshal(raw, &lines)
	if len(lines) == 0 || !strings.HasPrefix(lines[0], "fake llm: ") {
		t.Fatalf("log %q", lines)
	}
	for _, l := range lines {
		if strings.Contains(l, "\x1b") || strings.Contains(l, idleLine) {
			t.Errorf("line kept as is: %q", l)
		}
	}
	// With one instance up, the bare endpoints mean that one.
	_, bare, _ := h.raw("GET", "/api/logs", nil)
	if !bytes.Equal(bare, raw) {
		t.Error("bare /api/logs differs from ?model=zeta")
	}
	if out := h.ok("GET", "/api/logs/owner", nil); out["model"] != "zeta:q4" || out["kind"] != "llm" {
		t.Errorf("owner %v", out)
	}

	h.ok("POST", "/api/logs/clear?model=zeta", nil)
	if _, raw, _ = h.raw("GET", "/api/logs?model=zeta", nil); string(raw) != "[]" {
		t.Errorf("after clear: %s", raw)
	}
	if _, raw, _ = h.raw("GET", "/api/logs?model=nope", nil); string(raw) != "[]" {
		t.Errorf("unknown model: %s", raw)
	}
}

func TestCORS(t *testing.T) {
	h := newHarness(t)
	code, _, header := h.raw("OPTIONS", "/v1/chat/completions", nil,
		"Origin", "http://elsewhere", "Access-Control-Request-Method", "POST",
		"Access-Control-Request-Headers", "authorization, content-type")
	if code != 200 || header.Get("Access-Control-Allow-Origin") != "*" ||
		header.Get("Access-Control-Allow-Headers") != "authorization, content-type" ||
		!strings.Contains(header.Get("Access-Control-Allow-Methods"), "POST") {
		t.Errorf("preflight: %d %v", code, header)
	}
	if _, _, header = h.raw("GET", "/api/status", nil, "Origin", "http://elsewhere"); header.Get("Access-Control-Allow-Origin") != "*" {
		t.Errorf("simple request: %v", header)
	}
	if _, _, header = h.raw("GET", "/api/status", nil); header.Get("Access-Control-Allow-Origin") != "" {
		t.Errorf("no Origin, still CORS headers: %v", header)
	}
}

func TestDashboardIsEmbedded(t *testing.T) {
	h := newHarness(t)
	// Not where the sources are: the files must come from the binary.
	wd, _ := os.Getwd()
	os.Chdir(t.TempDir())
	defer os.Chdir(wd)

	code, raw, header := h.raw("GET", "/", nil)
	if code != 200 || header.Get("Content-Type") != "text/html; charset=utf-8" ||
		!strings.Contains(string(raw), "<title>Llama Switch</title>") {
		t.Errorf("index: %d %q", code, header.Get("Content-Type"))
	}
	for path, ctype := range map[string]string{
		"/static/js/app.js?v=43": "text/javascript; charset=utf-8",
		"/static/css/style.css":  "text/css; charset=utf-8",
	} {
		if code, raw, header := h.raw("GET", path, nil); code != 200 || header.Get("Content-Type") != ctype || len(raw) == 0 {
			t.Errorf("%s: %d %q", path, code, header.Get("Content-Type"))
		}
	}
	for _, path := range []string{"/static/", "/static/js/", "/static/nope.js"} {
		if code, _, _ := h.raw("GET", path, nil); code != 404 {
			t.Errorf("%s: %d, want 404", path, code)
		}
	}
}

func TestConfigChangeStopsModels(t *testing.T) {
	h, path := newHarnessWith(t, true)
	h.start(map[string]any{"model_key": "zeta"})
	pid := int(h.instance("zeta")["pid"].(float64))

	f, _ := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	f.WriteString("# edited\n")
	f.Close()
	later := time.Now().Add(5 * time.Second)
	os.Chtimes(path, later, later)

	h.waitFor("the watcher to stop zeta", func() bool { return len(h.running()) == 0 })
	h.waitFor("the engine to exit", func() bool { return groupGone(pid) })
}
