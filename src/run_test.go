package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// --- Flags and settings ---

func TestCheckRunFlags(t *testing.T) {
	for _, tc := range []struct {
		flags []string
		kind  string
		want  string
	}{
		{[]string{"gpu", "multi", "verbose"}, "sd", ""},
		{[]string{"ctx", "parallel", "preserve-thinking", "hide-thinking"}, "llm", ""},
		{[]string{"ctx", "questions"}, "decision", ""},
		{[]string{"size", "hires", "o"}, "sd", ""},
		{[]string{"frames", "voice", "no-play"}, "tts", ""},
		{[]string{"ctx"}, "sd", "--ctx is for llm and decision models, and m is a sd model"},
		{[]string{"o"}, "llm", "-o is for sd, tts and music models, and m is a llm model"},
		{[]string{"frames"}, "music", "--frames is for tts models, and m is a music model"},
		{[]string{"preserve-thinking"}, "decision", "--preserve-thinking is for llm models, and m is a decision model"},
	} {
		set := map[string]bool{}
		for _, f := range tc.flags {
			set[f] = true
		}
		err := checkRunFlags(set, "m", tc.kind)
		if got := ""; err != nil {
			got = err.Error()
			if got != tc.want {
				t.Errorf("%v on %s: %q, want %q", tc.flags, tc.kind, got, tc.want)
			}
		} else if tc.want != "" {
			t.Errorf("%v on %s: accepted, want %q", tc.flags, tc.kind, tc.want)
		}
	}
}

func settings(flags map[string]any) runSettings {
	s := runSettings{set: map[string]bool{}}
	for name, v := range flags {
		s.set[name] = true
		switch name {
		case "gpu":
			s.gpu = v.(string)
		case "size":
			s.size = v.(string)
		case "ctx":
			s.ctx = v.(int)
		case "parallel":
			s.parallel = v.(int)
		case "frames":
			s.frames = v.(int)
		case "preserve-thinking":
			s.preserve = v.(bool)
		case "hires":
			s.hires = v.(bool)
		case "multi":
			s.multi = v.(bool)
		}
	}
	return s
}

func TestApplySettingsRestartsLikeTheDashboard(t *testing.T) {
	h := newHarness(t)
	ctx := context.Background()
	zeta := runOptions{Server: h.url, Model: "zeta", ModelKey: "zeta"}

	if err := applySettings(ctx, zeta, "q4", settings(map[string]any{"ctx": 2048, "parallel": 2})); err != nil {
		t.Fatal(err)
	}
	inst := h.waitReady("zeta")
	if inst["ctx"] != 2048.0 || inst["parallel"] != 2.0 || inst["quantization"] != "q4" {
		t.Fatalf("started with %v", inst)
	}
	pid := inst["pid"]

	// Already running exactly so: left alone.
	applySettings(ctx, zeta, "q4", settings(map[string]any{"ctx": 2048}))
	if got := h.instance("zeta")["pid"]; got != pid {
		t.Errorf("restarted for nothing: pid %v -> %v", pid, got)
	}

	// One setting changes, the rest are carried over.
	applySettings(ctx, zeta, "q4", settings(map[string]any{"gpu": "Vulkan1", "preserve-thinking": true}))
	inst = h.waitReady("zeta")
	if inst["pid"] == pid || inst["devices"] != "Vulkan1" || inst["ctx"] != 2048.0 ||
		inst["parallel"] != 2.0 || inst["preserve_think"] != true {
		t.Errorf("after --gpu: %v", inst)
	}
	applySettings(ctx, zeta, "q4", settings(map[string]any{"gpu": "all"}))
	if inst = h.waitReady("zeta"); inst["devices"] != "" || inst["ctx"] != 2048.0 {
		t.Errorf("after --gpu all: %v", inst)
	}

	// Nothing given: nothing touched, not even a model that is down.
	if err := applySettings(ctx, runOptions{Server: h.url, Model: "voice", ModelKey: "voice"}, "q8", settings(nil)); err != nil {
		t.Fatal(err)
	}
	if h.instance("voice") != nil {
		t.Error("started without a setting to apply")
	}

	img := runOptions{Server: h.url, Model: "img", ModelKey: "img"}
	if err := applySettings(ctx, img, "q8_0", settings(map[string]any{"multi": true, "size": "512X768", "hires": true,
		"gpu": "vae=Vulkan1"})); err != nil {
		t.Fatal(err)
	}
	if inst = h.waitReady("img"); inst["res"] != "512x768" || inst["hires"] != true || inst["devices"] != "vae=Vulkan1" {
		t.Errorf("img: %v", inst)
	}
	if h.status()["multi"] != true || h.instance("zeta") == nil {
		t.Error("--multi did not keep zeta up")
	}

	err := applySettings(ctx, img, "q8_0", settings(map[string]any{"size": "big"}))
	if err == nil || err.Error() != "Resolution must look like 1024x1024, got 'big'" {
		t.Errorf("bad size: %v", err)
	}
}

func TestFindModel(t *testing.T) {
	h := newHarness(t)
	m, err := findModel(h.url, "zeta-q8")
	if err != nil || m.Kind != "llm" || m.Details.ParentModel != "zeta" || m.Details.QuantizationLevel != "q8" {
		t.Errorf("%+v %v", m, err)
	}
	if _, err := findModel(h.url, "nope"); err == nil || !strings.Contains(err.Error(), "Model nope not found") {
		t.Errorf("unknown: %v", err)
	}
}

func TestProgressSaysWhatIsAwaited(t *testing.T) {
	steps := `{"running":true,"ready":true,"stats":{"sd_step":3,"sd_steps":8}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("model") == "img" {
			io.WriteString(w, steps)
		} else {
			io.WriteString(w, `{"running":false,"ready":false,"stats":{}}`)
		}
	}))
	defer srv.Close()
	if got := progress(runOptions{Server: srv.URL, Model: "img", ModelKey: "img"})(); got != "step 3/8" {
		t.Errorf("sd: %q", got)
	}
	if got := progress(runOptions{Server: srv.URL, Model: "zeta-q8", ModelKey: "zeta"})(); got != "loading zeta-q8" {
		t.Errorf("down: %q", got)
	}
	steps = `{"running":true,"ready":true,"stats":{"sd_step":8,"sd_steps":8}}`
	if got := progress(runOptions{Server: srv.URL, Model: "img", ModelKey: "img"})(); got != "" {
		t.Errorf("finished image still shown: %q", got)
	}
}

// --- Images and audio ---

func tinyPNG(t *testing.T) []byte {
	t.Helper()
	var b bytes.Buffer
	png.Encode(&b, image.NewRGBA(image.Rect(0, 0, 2, 3)))
	return b.Bytes()
}

// A second of 8 kHz mono 16-bit silence.
func tinyWAV() []byte {
	data := make([]byte, 16000)
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+len(data)))
	b.WriteString("WAVEfmt ")
	// PCM, one channel, 8000 Hz, 16000 bytes a second, 2-byte frames, 16 bits.
	for _, v := range []any{uint32(16), uint16(1), uint16(1), uint32(8000), uint32(16000), uint16(2), uint16(16)} {
		binary.Write(&b, binary.LittleEndian, v)
	}
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(len(data)))
	b.Write(data)
	return b.Bytes()
}

// A server with one endpoint that records what it was sent.
func endpoint(t *testing.T, path string, reply func(w http.ResponseWriter, r *http.Request)) (string, *[]byte) {
	t.Helper()
	var got []byte
	mux := http.NewServeMux()
	mux.HandleFunc(path, func(w http.ResponseWriter, r *http.Request) {
		got, _ = io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewReader(got))
		reply(w, r)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv.URL, &got
}

func TestDeliver(t *testing.T) {
	dir := t.TempDir()
	wd, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(wd)

	// Piped: the bytes themselves.
	var out, errOut bytes.Buffer
	if path, err := deliver(runOptions{Model: "z-image"}, []byte("img"), "png", &out, &errOut); err != nil || path != "" || out.String() != "img" {
		t.Errorf("piped: %q %q %v", path, out.String(), err)
	}

	// A terminal: a new file here, named after the model, never over another.
	out.Reset()
	first, _ := deliver(runOptions{Model: "flux2-klein:4b", Styled: true}, []byte("a"), "png", &out, &errOut)
	second, _ := deliver(runOptions{Model: "flux2-klein:4b", Styled: true}, []byte("b"), "png", &out, &errOut)
	if !strings.HasPrefix(first, "flux2-klein-4b-") || !strings.HasSuffix(first, ".png") || first == second {
		t.Errorf("names %q %q", first, second)
	}
	if data, _ := os.ReadFile(second); string(data) != "b" || !strings.Contains(out.String(), "Saved "+first+"\n") {
		t.Errorf("second %q, output %q", data, out.String())
	}

	// -o, with stdout piped: the note stays out of the pipe.
	out.Reset()
	errOut.Reset()
	target := filepath.Join(dir, "cat.png")
	if _, err := deliver(runOptions{Model: "m", Output: target}, []byte("cat"), "png", &out, &errOut); err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(target); string(data) != "cat" || out.Len() != 0 || errOut.String() != "Saved "+target+"\n" {
		t.Errorf("-o: %q, stdout %q, stderr %q", data, out.String(), errOut.String())
	}
}

func TestGenerateImage(t *testing.T) {
	pic := tinyPNG(t)
	url, sent := endpoint(t, "/v1/images/generations", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"output_format": "png",
			"data":          []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(pic)}},
		})
	})
	var out, errOut bytes.Buffer
	if err := generateImage(context.Background(), runOptions{Server: url, Model: "z-image", Verbose: true},
		"a red cat", &out, &errOut); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out.Bytes(), pic) || !strings.Contains(errOut.String(), "image size:           2x3\n") {
		t.Errorf("out %d bytes, stats %q", out.Len(), errOut.String())
	}
	var req map[string]any
	json.Unmarshal(*sent, &req)
	if req["model"] != "z-image" || req["prompt"] != "a red cat" || req["n"] != 1.0 {
		t.Errorf("request %v", req)
	}

	url, _ = endpoint(t, "/v1/images/generations", func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"data":[]}`)
	})
	if err := generateImage(context.Background(), runOptions{Server: url, Model: "m"}, "x", io.Discard, io.Discard); err == nil {
		t.Error("an answer without an image passed")
	}
}

func TestGenerateImageThroughTheServer(t *testing.T) {
	h := newHarness(t)
	var out bytes.Buffer
	if err := generateImage(context.Background(), runOptions{Server: h.url, Model: "img", ModelKey: "img"},
		"a cat", &out, io.Discard); err != nil {
		t.Fatal(err)
	}
	if out.String() != "fake image" || h.instance("img") == nil {
		t.Errorf("got %q", out.String())
	}
}

func TestSpeakAndCompose(t *testing.T) {
	wav := tinyWAV()
	url, sent := endpoint(t, "/v1/audio/speech", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "audio/wav")
		w.Write(wav)
	})
	var out, errOut bytes.Buffer
	opts := runOptions{Server: url, Model: "qwen3-tts", Voice: "Kore", Verbose: true}
	if err := speak(context.Background(), opts, "Привет", &out, &errOut); err != nil {
		t.Fatal(err)
	}
	var req map[string]any
	json.Unmarshal(*sent, &req)
	if !bytes.Equal(out.Bytes(), wav) || req["input"] != "Привет" || req["voice"] != "Kore" {
		t.Errorf("speech: %d bytes, request %v", out.Len(), req)
	}
	if !strings.Contains(errOut.String(), "audio duration:       1s\n") {
		t.Errorf("stats %q", errOut.String())
	}

	url, sent = endpoint(t, "/v1/tasks/run", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"audio": base64.StdEncoding.EncodeToString(wav), "sample_rate": 8000})
	})
	out.Reset()
	opts = runOptions{Server: url, Model: "ace-step", Duration: 3, Lyrics: "la la"}
	if err := compose(context.Background(), opts, "lofi beat", &out, io.Discard); err != nil {
		t.Fatal(err)
	}
	var task struct {
		Model   string         `json:"model"`
		Request map[string]any `json:"request"`
	}
	json.Unmarshal(*sent, &task)
	if !bytes.Equal(out.Bytes(), wav) || task.Model != "ace-step" || task.Request["text"] != "lofi beat" ||
		task.Request["duration_seconds"] != 3.0 || task.Request["lyrics"] != "la la" {
		t.Errorf("music: %d bytes, request %+v", out.Len(), task)
	}
}

func TestFinishAudioPlaysInATerminal(t *testing.T) {
	dir := t.TempDir()
	played := filepath.Join(dir, "played.wav")
	saved := filepath.Join(dir, "clip.wav")
	defer func(p [][]string) { audioPlayers = p }(audioPlayers)
	audioPlayers = [][]string{{"sh", "-c", `cp "$0" ` + played}}

	opts := runOptions{Model: "m", Output: saved, Styled: true, Play: true}
	if err := finishAudio(context.Background(), opts, tinyWAV(), "wav", time.Now(), io.Discard, io.Discard); err != nil {
		t.Fatal(err)
	}
	if a, b := mustRead(t, saved), mustRead(t, played); !bytes.Equal(a, b) || len(a) == 0 {
		t.Error("the saved clip is not what was played")
	}

	os.Remove(played)
	opts.Play = false
	finishAudio(context.Background(), opts, tinyWAV(), "wav", time.Now(), io.Discard, io.Discard)
	if _, err := os.Stat(played); err == nil {
		t.Error("played with --no-play")
	}

	audioPlayers = [][]string{{"no-such-player-anywhere"}}
	opts.Play = true
	var errOut bytes.Buffer
	if err := finishAudio(context.Background(), opts, tinyWAV(), "wav", time.Now(), io.Discard, &errOut); err != nil ||
		!strings.Contains(errOut.String(), "Not played: no audio player found") {
		t.Errorf("no player: %v %q", err, errOut.String())
	}
}

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestWavDuration(t *testing.T) {
	if d, ok := wavDuration(tinyWAV()); !ok || d != time.Second {
		t.Errorf("%v %v", d, ok)
	}
	// Streamed: the size left open, the bytes that came count.
	streamed := tinyWAV()
	binary.LittleEndian.PutUint32(streamed[40:], 0xFFFFFFFF)
	if d, ok := wavDuration(streamed[:len(streamed)-8000]); !ok || d != 500*time.Millisecond {
		t.Errorf("streamed %v %v", d, ok)
	}
	for _, bad := range [][]byte{nil, []byte("RIFF"), []byte("ID3\x03 not a wav at all")} {
		if _, ok := wavDuration(bad); ok {
			t.Errorf("%q read as a WAV", bad)
		}
	}
}

// --- Transcription ---

func TestUnquotePath(t *testing.T) {
	home, _ := os.UserHomeDir()
	for in, want := range map[string]string{
		"/tmp/a.wav":                "/tmp/a.wav",
		"  '/tmp/my clip.wav'  ":    "/tmp/my clip.wav",
		`"/tmp/my clip.wav"`:        "/tmp/my clip.wav",
		`/tmp/my\ clip.wav`:         "/tmp/my clip.wav",
		"file:///tmp/my%20clip.wav": "/tmp/my clip.wav",
		"~/rec.wav":                 filepath.Join(home, "rec.wav"),
	} {
		if got := unquotePath(in); got != want {
			t.Errorf("unquotePath(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestTranscribe(t *testing.T) {
	var model, filename string
	var size int
	url, _ := endpoint(t, "/v1/audio/transcriptions", func(w http.ResponseWriter, r *http.Request) {
		r.ParseMultipartForm(1 << 20)
		model = r.FormValue("model")
		if f, h, err := r.FormFile("file"); err == nil {
			data, _ := io.ReadAll(f)
			filename, size = h.Filename, len(data)
		}
		io.WriteString(w, `{"text":"  Привет, мир. ","timing":{"wall_ms":368,"audio_duration_ms":2480,"rtf":0.15}}`)
	})

	clip := filepath.Join(t.TempDir(), "my clip.wav")
	os.WriteFile(clip, tinyWAV(), 0o600)
	var out, errOut bytes.Buffer
	opts := runOptions{Server: url, Model: "qwen3-asr", Verbose: true}
	if err := transcribeFile(context.Background(), opts, "'"+clip+"'", &out, &errOut); err != nil {
		t.Fatal(err)
	}
	if out.String() != "Привет, мир.\n" || model != "qwen3-asr" || filename != "my clip.wav" || size != len(tinyWAV()) {
		t.Errorf("printed %q; sent model %q, file %q of %d bytes", out.String(), model, filename, size)
	}
	if !strings.Contains(errOut.String(), "audio duration:       2.48s\n") ||
		!strings.Contains(errOut.String(), "speed:                6.67x realtime\n") {
		t.Errorf("stats %q", errOut.String())
	}

	if err := transcribeFile(context.Background(), opts, "/no/such/file.wav", io.Discard, io.Discard); err == nil {
		t.Error("a missing file passed")
	}
}

func TestReplTakesPathsForTranscription(t *testing.T) {
	url, _ := endpoint(t, "/v1/audio/transcriptions", func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"text":"hello"}`)
	})
	clip := filepath.Join(t.TempDir(), "a.wav")
	os.WriteFile(clip, tinyWAV(), 0o600)

	opts := runOptions{Server: url, Model: "qwen3-asr", Kind: "stt"}
	var out, errOut bytes.Buffer
	task := newRunTask(opts, &out, &errOut)
	// A path starts with a slash and still is not a command.
	if err := repl(strings.NewReader(clip+"\n/bye\n"), &out, &errOut, nil, task); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "hello\n") || strings.Contains(out.String(), "Unknown command") || errOut.Len() != 0 {
		t.Errorf("out %q, err %q", out.String(), errOut.String())
	}
	if strings.Contains(replHelp(task), "/clear") {
		t.Error("/clear offered with no conversation to clear")
	}
}

// --- Decision ---

const clefAnswer = `{"model":"/m/Clef.gguf","answers":{
  "severity":{"type":"score","score":2.68,"legend":{"0":"No impact","1":"Minor","2":"Major","3":"Critical"},"probabilities":{"0":0.01,"1":0.02,"2":0.25,"3":0.72},"confidence":0.68},
  "team":{"type":"choice","choice":"billing","probabilities":{"billing":0.9355,"sales":0.018,"technical":0.046},"confidence":0.903},
  "urgent":{"type":"noul","noul":0.9532}},
  "usage":{"input_tokens":336,"output_tokens":0}}`

func TestDecide(t *testing.T) {
	url, sent := endpoint(t, "/v1/systemone", func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, clefAnswer)
	})
	file := filepath.Join(t.TempDir(), "q.json")
	os.WriteFile(file, []byte(`{"questions":{
		"urgent":{"type":"noul","instructions":"Is this urgent?"},
		"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"","technical":"","sales":""}},
		"severity":{"type":"score","instructions":"How severe?","criteria":["No impact","Minor","Major","Critical"]}}}`), 0o600)
	questions, err := readQuestions(file)
	if err != nil {
		t.Fatal(err)
	}

	var out, errOut bytes.Buffer
	opts := runOptions{Server: url, Model: "clef:9b", Styled: true, Verbose: true}
	if err := decide(context.Background(), opts, questions, "charged twice", &out, &errOut); err != nil {
		t.Fatal(err)
	}
	// In the order the file asks them, not the order they came back in.
	want := "urgent    yes       95.3%\n" +
		"team      billing   93.5%, confidence 90.3%\n" +
		"severity  Critical  2.68 of 0-3, confidence 68.0%\n"
	if out.String() != want {
		t.Errorf("table:\n%s\nwant\n%s", out.String(), want)
	}
	if !strings.Contains(errOut.String(), "prompt eval count:    336 token(s)\n") {
		t.Errorf("stats %q", errOut.String())
	}
	var req struct {
		Model, State string
		Questions    map[string]any
	}
	json.Unmarshal(*sent, &req)
	if req.Model != "clef:9b" || req.State != "charged twice" || len(req.Questions) != 3 {
		t.Errorf("request %+v", req)
	}

	// The one yes/no question a bare prompt asks.
	url, sent = endpoint(t, "/v1/systemone", func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"answers":{"answer":{"type":"noul","noul":0.2}}}`)
	})
	out.Reset()
	opts.Server, opts.Verbose = url, false
	decide(context.Background(), opts, yesNoQuestion("Is it spam?"), "hello", &out, io.Discard)
	if out.String() != "no  80.0%\n" || !strings.Contains(string(*sent), `"instructions":"Is it spam?"`) {
		t.Errorf("yes/no: %q, sent %s", out.String(), *sent)
	}

	// For a pipe: the JSON itself.
	out.Reset()
	opts.Styled = false
	decide(context.Background(), opts, yesNoQuestion("Is it spam?"), "hello", &out, io.Discard)
	var back map[string]any
	if err := json.Unmarshal(out.Bytes(), &back); err != nil || back["answers"] == nil {
		t.Errorf("piped %q", out.String())
	}
}

func TestReadQuestions(t *testing.T) {
	dir := t.TempDir()
	write := func(text string) string {
		p := filepath.Join(dir, "q.json")
		os.WriteFile(p, []byte(text), 0o600)
		return p
	}
	q, err := readQuestions(write(`{"a":{"type":"noul"},"b":{"type":"noul"}}`))
	if err != nil || strings.Join(jsonKeys(q), ",") != "a,b" {
		t.Errorf("bare: %s %v", q, err)
	}
	for _, bad := range []string{`{}`, `[1]`, `{nope`} {
		if _, err := readQuestions(write(bad)); err == nil {
			t.Errorf("%s accepted", bad)
		}
	}
	if _, err := readQuestions(filepath.Join(dir, "missing.json")); err == nil {
		t.Error("missing file accepted")
	}
}

func TestSystemPromptLeadsEveryRequest(t *testing.T) {
	s, url := scripted(t, sse(delta("content", "ok")))
	opts := runOptions{Server: url, Model: "m", System: "Answer in one word."}
	task := chatTask(opts, io.Discard, io.Discard)
	if err := repl(strings.NewReader("hi\n/clear\nagain\n/bye\n"), io.Discard, io.Discard, nil, task); err != nil {
		t.Fatal(err)
	}
	sys := chatMessage{"system", "Answer in one word."}
	want := [][]chatMessage{
		{sys, {"user", "hi"}},
		{sys, {"user", "again"}}, // /clear forgets the talk, not the system prompt
	}
	if got := s.sent(); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("sent %v, want %v", got, want)
	}
	if err := checkRunFlags(map[string]bool{"system": true}, "img", "sd"); err == nil {
		t.Error("--system accepted for an image model")
	}
}

// --- Editing images ---

func TestExtractImages(t *testing.T) {
	dir := t.TempDir()
	cat := filepath.Join(dir, "cat.png")
	spaced := filepath.Join(dir, "my dog.JPG")
	notes := filepath.Join(dir, "notes.txt")
	for _, p := range []string{cat, spaced, notes} {
		os.WriteFile(p, []byte("x"), 0o600)
	}
	escaped := strings.ReplaceAll(spaced, " ", `\ `)
	url := "file://" + strings.ReplaceAll(spaced, " ", "%20")

	for _, tc := range []struct {
		in, prompt string
		paths      []string
	}{
		{"a red cat", "a red cat", nil},
		{"make it black " + cat, "make it black", []string{cat}},
		{cat + "  make it   black", "make it black", []string{cat}},
		{"'" + spaced + "' add a hat", "add a hat", []string{spaced}},
		{`"` + spaced + `" add a hat`, "add a hat", []string{spaced}},
		{escaped + " add a hat", "add a hat", []string{spaced}},
		{url + " add a hat", "add a hat", []string{spaced}},
		{"merge " + cat + " and " + spaced[:len(dir)] + "/my\\ dog.JPG", "merge and", []string{cat, spaced}},
		// Only images, and only files: the rest is the prompt.
		{"rewrite " + notes, "rewrite " + notes, nil},
		{"a logo.png style poster", "a logo.png style poster", nil},
	} {
		prompt, paths, err := extractImages(tc.in)
		if err != nil || prompt != tc.prompt || fmt.Sprint(paths) != fmt.Sprint(tc.paths) {
			t.Errorf("extractImages(%q) = %q %v %v, want %q %v", tc.in, prompt, paths, err, tc.prompt, tc.paths)
		}
	}

	if _, _, err := extractImages("fix " + filepath.Join(dir, "gone.png")); err == nil ||
		!strings.HasSuffix(err.Error(), "gone.png: no such image") {
		t.Errorf("missing image: %v", err)
	}
}

type editCall struct {
	model, prompt string
	images        []string // file names, in order
	mask          string
}

func editEndpoint(t *testing.T, pic []byte) (string, *[]editCall, *int) {
	t.Helper()
	var calls []editCall
	generations := 0
	mux := http.NewServeMux()
	answer := func(w http.ResponseWriter) {
		json.NewEncoder(w).Encode(map[string]any{
			"output_format": "png",
			"data":          []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(pic)}},
		})
	}
	mux.HandleFunc("/v1/images/edits", func(w http.ResponseWriter, r *http.Request) {
		r.ParseMultipartForm(1 << 20)
		c := editCall{model: r.FormValue("model"), prompt: r.FormValue("prompt")}
		for _, fh := range r.MultipartForm.File["image[]"] {
			c.images = append(c.images, fh.Filename)
		}
		if fh := r.MultipartForm.File["mask"]; len(fh) > 0 {
			c.mask = fh[0].Filename
		}
		calls = append(calls, c)
		answer(w)
	})
	mux.HandleFunc("/v1/images/generations", func(w http.ResponseWriter, r *http.Request) {
		generations++
		answer(w)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv.URL, &calls, &generations
}

func TestGenerateImageEditsWhatItIsGiven(t *testing.T) {
	pic := tinyPNG(t)
	url, calls, generations := editEndpoint(t, pic)
	dir := t.TempDir()
	cat, hat, mask := filepath.Join(dir, "cat.png"), filepath.Join(dir, "hat.png"), filepath.Join(dir, "mask.png")
	for _, p := range []string{cat, hat, mask} {
		os.WriteFile(p, tinyPNG(t), 0o600)
	}
	opts := runOptions{Server: url, Model: "mage-flow-edit", Styled: true, Output: filepath.Join(dir, "out.png")}

	// A path in the message.
	var out bytes.Buffer
	if err := generateImage(context.Background(), opts, "make the cat black "+cat, &out, io.Discard); err != nil {
		t.Fatal(err)
	}
	want := editCall{model: "mage-flow-edit", prompt: "make the cat black", images: []string{"cat.png"}}
	if len(*calls) != 1 || fmt.Sprint((*calls)[0]) != fmt.Sprint(want) {
		t.Errorf("sent %+v, want %+v", *calls, want)
	}
	if !strings.HasPrefix(out.String(), "Added image "+cat+"\nSaved ") || !bytes.Equal(mustRead(t, opts.Output), pic) {
		t.Errorf("printed %q", out.String())
	}

	// --image and --mask, with a path in the message besides.
	opts.Images, opts.Mask = []string{hat}, mask
	if err := generateImage(context.Background(), opts, cat+" put the hat on the cat", io.Discard, io.Discard); err != nil {
		t.Fatal(err)
	}
	want = editCall{model: "mage-flow-edit", prompt: "put the hat on the cat", images: []string{"hat.png", "cat.png"}, mask: "mask.png"}
	if fmt.Sprint((*calls)[1]) != fmt.Sprint(want) {
		t.Errorf("sent %+v, want %+v", (*calls)[1], want)
	}

	// No image: a new one is drawn.
	opts.Images, opts.Mask = nil, ""
	generateImage(context.Background(), opts, "a red cat", io.Discard, io.Discard)
	if *generations != 1 || len(*calls) != 2 {
		t.Errorf("generations %d, edits %d", *generations, len(*calls))
	}

	for _, tc := range []struct {
		opts   runOptions
		prompt string
		want   string
	}{
		{runOptions{Server: url, Model: "m", Mask: mask}, "a cat", "--mask needs an image to apply to"},
		{runOptions{Server: url, Model: "m"}, cat, "say what to do with the image"},
		{runOptions{Server: url, Model: "m", Images: []string{filepath.Join(dir, "gone.png")}}, "x", "no such file"},
	} {
		if err := generateImage(context.Background(), tc.opts, tc.prompt, io.Discard, io.Discard); err == nil ||
			!strings.Contains(err.Error(), tc.want) {
			t.Errorf("%q: %v, want %q", tc.prompt, err, tc.want)
		}
	}
}

func TestEditImageThroughTheServer(t *testing.T) {
	h := newHarness(t)
	cat := filepath.Join(t.TempDir(), "cat.png")
	os.WriteFile(cat, tinyPNG(t), 0o600)
	var out bytes.Buffer
	opts := runOptions{Server: h.url, Model: "img2", ModelKey: "img2"}
	if err := generateImage(context.Background(), opts, "black cat "+cat, &out, io.Discard); err != nil {
		t.Fatal(err)
	}
	// The model is named inside the upload, and the server loads it from there.
	if out.String() != "fake image" || h.instance("img2") == nil {
		t.Errorf("got %q, running %v", out.String(), h.running())
	}
}

func TestReplAttachesImages(t *testing.T) {
	pic := tinyPNG(t)
	url, calls, generations := editEndpoint(t, pic)
	dir := t.TempDir()
	cat := filepath.Join(dir, "cat.png")
	os.WriteFile(cat, pic, 0o600)
	wd, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(wd)

	opts := runOptions{Server: url, Model: "m", Kind: "sd", Styled: true}
	var out, errOut bytes.Buffer
	in := "a red cat\n'" + cat + "' make it black\n/bye\n"
	if err := repl(strings.NewReader(in), &out, &errOut, nil, newRunTask(opts, &out, &errOut)); err != nil {
		t.Fatal(err)
	}
	if *generations != 1 || len(*calls) != 1 || (*calls)[0].prompt != "make it black" {
		t.Errorf("generations %d, edits %+v", *generations, *calls)
	}
	if strings.Count(out.String(), "Saved ") != 2 || !strings.Contains(out.String(), "Added image "+cat) || errOut.Len() != 0 {
		t.Errorf("out %q, err %q", out.String(), errOut.String())
	}
}
