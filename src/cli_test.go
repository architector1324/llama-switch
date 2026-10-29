package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestServerURL(t *testing.T) {
	for env, want := range map[string]string{
		"":                    "http://localhost:11435",
		"0.0.0.0:9000":        "http://0.0.0.0:9000",
		"https://box.lan/":    "https://box.lan",
		" 192.168.0.2:11435 ": "http://192.168.0.2:11435",
	} {
		t.Setenv("LLMSW_HOST", env)
		if got := serverURL(); got != want {
			t.Errorf("LLMSW_HOST=%q: %q, want %q", env, got, want)
		}
	}
}

func TestRunAnswersOnce(t *testing.T) {
	h := newHarness(t)
	var out, errOut bytes.Buffer
	answer, err := chat(context.Background(), runOptions{Server: h.url, Model: "zeta"},
		[]chatMessage{{"user", "hi"}}, &out, &errOut)
	if err != nil {
		t.Fatal(err)
	}
	// The ASR header the fake model streams is gone by the time it gets here.
	if answer != "Hello world" || out.String() != "Hello world\n" {
		t.Errorf("answer %q, printed %q", answer, out.String())
	}
	if errOut.Len() != 0 {
		t.Errorf("stderr %q without a terminal", errOut.String())
	}
}

func TestRunReportsServerErrors(t *testing.T) {
	h := newHarness(t)
	for model, want := range map[string]string{
		"nope":   "Model nope not found",
		"img":    "Model img is a 'sd' model, not 'llm'; it is served at /v1/images/generations",
		"broken": "broken:q1 exited while starting; see its log",
	} {
		_, err := chat(context.Background(), runOptions{Server: h.url, Model: model},
			[]chatMessage{{"user", "hi"}}, io.Discard, io.Discard)
		if err == nil || err.Error() != want {
			t.Errorf("%s: %v, want %q", model, err, want)
		}
	}

	_, err := chat(context.Background(), runOptions{Server: "http://127.0.0.1:1", Model: "zeta"},
		[]chatMessage{{"user", "hi"}}, io.Discard, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "is `llmsw serve` running?") {
		t.Errorf("no server: %v", err)
	}
}

// A server that answers each chat request with the next of these.
type scriptedChat struct {
	mu       sync.Mutex
	replies  []func(w http.ResponseWriter, r *http.Request)
	requests [][]chatMessage
}

func (s *scriptedChat) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Messages []chatMessage `json:"messages"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	s.mu.Lock()
	n := len(s.requests)
	s.requests = append(s.requests, body.Messages)
	reply := s.replies[len(s.replies)-1]
	if n < len(s.replies) {
		reply = s.replies[n]
	}
	s.mu.Unlock()
	reply(w, r)
}

func (s *scriptedChat) sent() [][]chatMessage {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([][]chatMessage{}, s.requests...)
}

func sse(events ...string) func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range events {
			fmt.Fprintf(w, "data: %s\n\n", ev)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}
}

func delta(field, text string) string {
	q, _ := json.Marshal(text)
	return fmt.Sprintf(`{"choices":[{"index":0,"delta":{%q:%s}}]}`, field, q)
}

func scripted(t *testing.T, replies ...func(w http.ResponseWriter, r *http.Request)) (*scriptedChat, string) {
	t.Helper()
	s := &scriptedChat{replies: replies}
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)
	return s, srv.URL
}

func TestRunWaitsForASlowLoad(t *testing.T) {
	timeout := func(w http.ResponseWriter, r *http.Request) {
		writeError(w, newErr(http.StatusGatewayTimeout, "big:q8 failed to load within timeout"))
	}
	s, url := scripted(t, timeout, timeout, sse(delta("content", "ok")))
	var out bytes.Buffer
	answer, err := chat(context.Background(), runOptions{Server: url, Model: "big"},
		[]chatMessage{{"user", "hi"}}, &out, io.Discard)
	if err != nil || answer != "ok" || len(s.sent()) != 3 {
		t.Errorf("answer %q, err %v, after %d requests", answer, err, len(s.sent()))
	}

	// Any other failure is reported, not retried.
	s, url = scripted(t, func(w http.ResponseWriter, r *http.Request) {
		writeError(w, newErr(http.StatusGatewayTimeout, "upstream timed out"))
	})
	if _, err := chat(context.Background(), runOptions{Server: url, Model: "big"},
		[]chatMessage{{"user", "hi"}}, io.Discard, io.Discard); err == nil || len(s.sent()) != 1 {
		t.Errorf("err %v after %d requests", err, len(s.sent()))
	}
}

func TestRunShowsOrHidesThinking(t *testing.T) {
	reply := sse(delta("reasoning_content", "Let me"), delta("reasoning_content", " think."),
		delta("content", "\n\nAnswer"), delta("content", "."))
	_, url := scripted(t, reply)

	var out bytes.Buffer
	answer, err := chat(context.Background(), runOptions{Server: url, Model: "m", ShowThinking: true},
		[]chatMessage{{"user", "hi"}}, &out, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if want := "Thinking...\nLet me think.\n...done thinking.\n\nAnswer.\n"; out.String() != want {
		t.Errorf("shown:\n%q\nwant\n%q", out.String(), want)
	}
	if answer != "Answer." {
		t.Errorf("answer kept for the history: %q", answer)
	}

	out.Reset()
	chat(context.Background(), runOptions{Server: url, Model: "m", ShowThinking: true, Styled: true},
		[]chatMessage{{"user", "hi"}}, &out, io.Discard)
	if !strings.Contains(out.String(), "\x1b[90mLet me\x1b[0m") || strings.Contains(out.String(), "\x1b[90mAnswer") {
		t.Errorf("styled: %q", out.String())
	}

	out.Reset()
	chat(context.Background(), runOptions{Server: url, Model: "m"}, []chatMessage{{"user", "hi"}}, &out, io.Discard)
	if out.String() != "Answer.\n" {
		t.Errorf("hidden: %q", out.String())
	}
}

func TestRunReportsAnErrorInTheStream(t *testing.T) {
	_, url := scripted(t, sse(delta("content", "par"), `{"error":{"code":500,"message":"context overflow"}}`))
	var out bytes.Buffer
	answer, err := chat(context.Background(), runOptions{Server: url, Model: "m"},
		[]chatMessage{{"user", "hi"}}, &out, io.Discard)
	if err == nil || err.Error() != "context overflow" || answer != "par" || out.String() != "par\n" {
		t.Errorf("err %v, answer %q, printed %q", err, answer, out.String())
	}
}

func TestRunVerbosePrintsTimings(t *testing.T) {
	// The closing event as llama-server sends it: no choices, usage and timings.
	final := `{"choices":[],"usage":{"completion_tokens":4,"prompt_tokens":18},` +
		`"timings":{"cache_n":6,"prompt_n":18,"prompt_ms":86.662,"prompt_per_second":207.70349,` +
		`"predicted_n":4,"predicted_ms":42.556,"predicted_per_second":70.49534,"draft_n":4,"draft_n_accepted":3}}`
	// A third of a second before the engine starts, as if it were loading.
	reply := sse(delta("content", "Hi"), final)
	s, url := scripted(t, func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(300 * time.Millisecond)
		reply(w, r)
	})

	var out, errOut bytes.Buffer
	if _, err := chat(context.Background(), runOptions{Server: url, Model: "m", Verbose: true},
		[]chatMessage{{"user", "hi"}}, &out, &errOut); err != nil {
		t.Fatal(err)
	}
	// Total less what the engine says it spent on the prompt and the answer.
	var total, load time.Duration
	for _, l := range strings.Split(errOut.String(), "\n") {
		if v, ok := strings.CutPrefix(l, "total duration:"); ok {
			total, _ = time.ParseDuration(strings.TrimSpace(v))
		}
		if v, ok := strings.CutPrefix(l, "load duration:"); ok {
			load, _ = time.ParseDuration(strings.TrimSpace(v))
		}
	}
	if total < 300*time.Millisecond || (total-load-129218*time.Microsecond).Abs() > time.Millisecond {
		t.Errorf("total %v, load %v: load must be total minus 129.2ms of engine time", total, load)
	}
	// The answer stays clean for a pipe: the figures go to stderr.
	if out.String() != "Hi\n" {
		t.Errorf("stdout %q", out.String())
	}
	for _, want := range []string{
		"total duration:       ", "load duration:        ",
		"prompt eval count:    18 token(s)\n", "prompt cache:         6 token(s)\n",
		"prompt eval duration: 86.7ms\n", "prompt eval rate:     207.70 tokens/s\n",
		"eval count:           4 token(s)\n", "eval duration:        42.6ms\n",
		"eval rate:            70.50 tokens/s\n", "draft accepted:       3/4 (75%)\n",
	} {
		if !strings.Contains(errOut.String(), want) {
			t.Errorf("stats lack %q:\n%s", want, errOut.String())
		}
	}
	if sent := s.sent(); len(sent) != 1 {
		t.Fatalf("sent %v", sent)
	}

	// Not asked for, not printed; another engine's counts without timings.
	errOut.Reset()
	chat(context.Background(), runOptions{Server: url, Model: "m"}, []chatMessage{{"user", "hi"}}, io.Discard, &errOut)
	if errOut.Len() != 0 {
		t.Errorf("stats without --verbose: %q", errOut.String())
	}
	_, url = scripted(t, sse(delta("content", "Hi"), `{"choices":[],"usage":{"completion_tokens":2,"prompt_tokens":5}}`))
	errOut.Reset()
	chat(context.Background(), runOptions{Server: url, Model: "m", Verbose: true},
		[]chatMessage{{"user", "hi"}}, io.Discard, &errOut)
	if !strings.Contains(errOut.String(), "prompt eval count:    5 token(s)\neval count:           2 token(s)\n") ||
		strings.Contains(errOut.String(), "rate") {
		t.Errorf("usage only: %q", errOut.String())
	}
}

func TestRunAsksForUsage(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		sse(delta("content", "ok"))(w, r)
	}))
	defer srv.Close()
	chat(context.Background(), runOptions{Server: srv.URL, Model: "m"}, []chatMessage{{"user", "hi"}}, io.Discard, io.Discard)
	if opts, _ := body["stream_options"].(map[string]any); opts["include_usage"] != true || body["stream"] != true {
		t.Errorf("request %v", body)
	}
}

func TestReplKeepsTheConversation(t *testing.T) {
	s, url := scripted(t, sse(delta("content", "ok")))
	input := strings.Join([]string{
		"hello", "", "again", "/clear", `"""`, "line one", `line two"""`,
		"/nope", "/?", `"""one line"""`, "/bye", "never sent",
	}, "\n") + "\n"

	var out, errOut bytes.Buffer
	if err := repl(strings.NewReader(input), &out, &errOut, nil, chatTask(runOptions{Server: url, Model: "m"}, &out, &errOut)); err != nil {
		t.Fatal(err)
	}

	sent := s.sent()
	want := [][]chatMessage{
		{{"user", "hello"}},
		{{"user", "hello"}, {"assistant", "ok"}, {"user", "again"}},
		{{"user", "line one\nline two"}},
		{{"user", "line one\nline two"}, {"assistant", "ok"}, {"user", "one line"}},
	}
	if fmt.Sprint(sent) != fmt.Sprint(want) {
		t.Errorf("sent\n%v\nwant\n%v", sent, want)
	}
	for _, s := range []string{"\n> ", "... ", "Cleared session context",
		"Unknown command '/nope'. Type /? for help", "/clear    forget"} {
		if !strings.Contains(out.String(), s) {
			t.Errorf("output lacks %q:\n%s", s, out.String())
		}
	}
}

func TestReplEndsOnEOFAndSurvivesErrors(t *testing.T) {
	_, url := scripted(t, func(w http.ResponseWriter, r *http.Request) {
		writeError(w, newErr(http.StatusNotFound, "Model m not found"))
	})
	var out, errOut bytes.Buffer
	if err := repl(strings.NewReader("hi\nagain\n"), &out, &errOut, nil, chatTask(runOptions{Server: url, Model: "m"}, &out, &errOut)); err != nil {
		t.Fatal(err)
	}
	if strings.Count(errOut.String(), "Error: Model m not found") != 2 {
		t.Errorf("stderr %q: each failure is reported and the session goes on", errOut.String())
	}
}

// Collects output and says when a piece of text has shown up in it.
type watchWriter struct {
	mu   sync.Mutex
	buf  bytes.Buffer
	want string
	seen chan struct{}
	once sync.Once
}

func (w *watchWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.buf.Write(p)
	if strings.Contains(w.buf.String(), w.want) {
		w.once.Do(func() { close(w.seen) })
	}
	return len(p), nil
}

func (w *watchWriter) String() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.String()
}

func TestReplCtrlCStopsTheAnswer(t *testing.T) {
	s, url := scripted(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\n", delta("content", "Once upon"))
		w.(http.Flusher).Flush()
		<-r.Context().Done() // writes forever unless stopped
	}, sse(delta("content", "ok")))

	// Pressed once the answer is on screen, the way a person would.
	out := &watchWriter{want: "Once upon", seen: make(chan struct{})}
	interrupts := make(chan os.Signal, 1)
	go func() {
		<-out.seen
		interrupts <- os.Interrupt
	}()

	done := make(chan error, 1)
	go func() {
		done <- repl(strings.NewReader("tell a story\nhi\n/bye\n"), out, io.Discard, interrupts,
			chatTask(runOptions{Server: url, Model: "m"}, out, io.Discard))
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Ctrl+C did not stop the answer")
	}

	sent := s.sent()
	// The stopped answer is dropped from the conversation.
	if len(sent) != 2 || fmt.Sprint(sent[1]) != fmt.Sprint([]chatMessage{{"user", "hi"}}) {
		t.Errorf("sent %v", sent)
	}
	if !strings.Contains(out.String(), "Once upon") {
		t.Errorf("output %q", out.String())
	}
}

func TestListAndPs(t *testing.T) {
	h := newHarness(t)
	h.start(map[string]any{"model_key": "zeta", "quantization": "q8", "parallel": 2})
	h.start(map[string]any{"model_key": "img"})

	rows := func(text string) map[string][]string {
		out := map[string][]string{}
		for _, line := range strings.Split(strings.TrimSpace(text), "\n")[1:] {
			f := strings.Fields(line)
			out[f[0]] = f[1:]
		}
		return out
	}

	var buf bytes.Buffer
	if err := printModels(h.url, "", &buf); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(buf.String(), "NAME") {
		t.Errorf("no header: %q", buf.String())
	}
	list := rows(buf.String())
	for name, want := range map[string]string{
		"zeta": "llm q4", "zeta-q8": "llm q8 idle", "eta": "llm q1",
		"img": "sd q8_0 idle", "song-sfx": "music sfx", "ears": "stt f16",
	} {
		if got := strings.Join(list[name], " "); got != want {
			t.Errorf("list %s: %q, want %q", name, got, want)
		}
	}

	buf.Reset()
	printModels(h.url, "sd", &buf)
	if got := rows(buf.String()); len(got) != 2 || got["img"] == nil || got["img2"] == nil {
		t.Errorf("list sd: %v", got)
	}

	buf.Reset()
	if err := printRunning(h.url, &buf); err != nil {
		t.Fatal(err)
	}
	ps := rows(buf.String())
	// Named the way run takes them; memory as the dashboard rounds it.
	if got := strings.Join(ps["zeta-q8"], " "); !strings.HasPrefix(got, "llm idle 2.5 GiB default 2 x 4096 ") {
		t.Errorf("ps zeta-q8: %q", got)
	}
	if got := strings.Join(ps["img"], " "); !strings.HasPrefix(got, "sd idle - default - ") {
		t.Errorf("ps img: %q", got)
	}
	if len(ps) != 2 {
		t.Errorf("ps rows %v", ps)
	}

	if err := printRunning("http://127.0.0.1:1", io.Discard); err == nil ||
		!strings.Contains(err.Error(), "is `llmsw serve` running?") {
		t.Errorf("no server: %v", err)
	}
}

func TestStop(t *testing.T) {
	h := newHarness(t)
	h.start(map[string]any{"model_key": "zeta", "quantization": "q8"})
	h.start(map[string]any{"model_key": "img"})
	h.start(map[string]any{"model_key": "voice"})

	var out bytes.Buffer
	for names, want := range map[string]string{
		"nope":         "Model nope not found; `llmsw list` shows the ones there are",
		"zeta":         "zeta is not running; `llmsw ps` shows what is", // the q4 is not the one up
		"img ears":     "ears is not running; `llmsw ps` shows what is",
		"zeta-q8 nope": "Model nope not found; `llmsw list` shows the ones there are",
	} {
		if err := stopModels(h.url, strings.Fields(names), false, &out); err == nil || err.Error() != want {
			t.Errorf("stop %s: %v, want %q", names, err, want)
		}
	}
	// A bad name anywhere in the list stops nothing.
	if got := h.running(); len(got) != 3 || out.Len() != 0 {
		t.Fatalf("after failed stops: %v, printed %q", got, out.String())
	}

	if err := stopModels(h.url, []string{"zeta-q8", "img"}, false, &out); err != nil {
		t.Fatal(err)
	}
	if got := h.running(); fmt.Sprint(got) != "[voice]" || out.String() != "Stopped zeta-q8\nStopped img\n" {
		t.Errorf("running %v, printed %q", got, out.String())
	}

	out.Reset()
	h.start(map[string]any{"model_key": "img"})
	if err := stopModels(h.url, nil, true, &out); err != nil {
		t.Fatal(err)
	}
	// Named as ps names them, in the order they started.
	if got := h.running(); len(got) != 0 || out.String() != "Stopped voice\nStopped img\n" {
		t.Errorf("--all: running %v, printed %q", got, out.String())
	}

	out.Reset()
	if err := stopModels(h.url, nil, true, &out); err != nil || out.String() != "Nothing was running\n" {
		t.Errorf("--all with nothing up: %v, printed %q", err, out.String())
	}
}

func TestPsShowsDetails(t *testing.T) {
	h := newHarness(t)
	h.start(map[string]any{"model_key": "zeta", "quantization": "q8", "parallel": 2})
	h.start(map[string]any{"model_key": "img"})
	h.start(map[string]any{"model_key": "voice"})
	h.ok("POST", "/v1/chat/completions", map[string]any{"model": "zeta-q8", "messages": []any{}})
	h.waitFor("the answer's speeds", func() bool {
		s, _ := h.instance("zeta")["stats"].(map[string]any)
		return s["gen_speed"] != 0.0
	})

	var out bytes.Buffer
	if err := printDetails(h.url, []string{"zeta-q8", "img", "voice"}, &out); err != nil {
		t.Fatal(err)
	}
	got := regexp.MustCompile(`port \d+, pid \d+`).ReplaceAllString(out.String(), "port N, pid N")
	// The dashboard's panel for each kind: memory rolled up per card, the host's as RAM.
	want := `zeta-q8
  state:             idle, port N, pid N
  type:              llm, q8
  gpu:               default
  context:           2 x 4096
  preserve thinking: off
  ctx usage:         73 / 4096
  prompt speed:      2355.46 tokens/s
  gen speed:         400.00 tokens/s
  total tokens:      20
  model memory:      2.5 GiB
  context memory:    51 MiB
  Vulkan0:           662 MiB (model 611 MiB, context 51 MiB)
  Vulkan1:           1.8 GiB
  RAM:               100 MiB

img
  state:             idle, port N, pid N
  type:              sd, q8_0
  gpu:               default
  size:              1024x1024
  hires:             off
  step time:         -
  last image:        -
  progress:          -
  images:            0
  params memory:     -
  compute buffer:    -

voice
  state:             idle, port N, pid N
  type:              tts, q8
  gpu:               default
  max frames:        2048
  speed:             -
  last clip:         -
  frames:            -
  clips:             0
`
	if got != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}

	// As with stop, every name is checked first.
	for names, want := range map[string]string{
		"img nope": "Model nope not found; `llmsw list` shows the ones there are",
		"img zeta": "zeta is not running; `llmsw ps` shows what is",
	} {
		out.Reset()
		if err := printDetails(h.url, strings.Fields(names), &out); err == nil || err.Error() != want || out.Len() != 0 {
			t.Errorf("ps %s: %v, printed %q", names, err, out.String())
		}
	}
}

func TestInstanceState(t *testing.T) {
	up := runningInstance{Kind: "llm", Running: true, Ready: true, Parallel: 1}
	with := func(kind string, f func(i *runningInstance)) runningInstance {
		i := up
		i.Kind = kind
		f(&i)
		return i
	}
	for _, tc := range []struct {
		i            runningInstance
		state, shown string
	}{
		{runningInstance{Kind: "llm"}, "exited", "exited"},
		{runningInstance{Kind: "llm", Running: true}, "loading", "loading"},
		{up, "idle", "idle"},
		{with("llm", func(i *runningInstance) { i.Active = 1 }), "generating", "generating"},
		// Sent to the engine's port directly: only its slots know.
		{with("llm", func(i *runningInstance) { i.Stats.SlotsBusy = 1 }), "generating", "generating"},
		{with("llm", func(i *runningInstance) { i.Active, i.Stats.SlotsBusy, i.Stats.SlotsPrefill = 1, 1, 1 }),
			"processing prompt", "processing prompt"},
		{with("llm", func(i *runningInstance) { i.Parallel, i.Active, i.Stats.SlotsBusy, i.Stats.SlotsPrefill = 4, 2, 2, 1 }),
			"generating", "generating, 2 of 4 slots"},
		{with("sd", func(i *runningInstance) { i.Active = 2 }), "generating", "generating, 2 requests"},
		{with("stt", func(i *runningInstance) { i.Active = 1 }), "transcribing", "transcribing"},
		{with("decision", func(i *runningInstance) { i.Active = 1 }), "deciding", "deciding"},
	} {
		if got, shown := tc.i.state(), tc.i.stateDetail(); got != tc.state || shown != tc.shown {
			t.Errorf("%+v: %q, %q; want %q, %q", tc.i, got, shown, tc.state, tc.shown)
		}
	}
}

func TestPsTellsWorkFromRest(t *testing.T) {
	h := newHarness(t)
	h.start(map[string]any{"model_key": "zeta", "quantization": "q4"})
	zeta := func() runningInstance {
		instances, err := namedInstances(h.url, []string{"zeta"})
		if err != nil {
			t.Fatal(err)
		}
		return instances[0]
	}
	if got := zeta().state(); got != "idle" {
		t.Fatalf("before: %q", got)
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		body := strings.NewReader(`{"model": "zeta", "messages": [], "fake_delay_ms": 1500}`)
		if resp, err := testClient.Post(h.url+"/v1/chat/completions", "application/json", body); err == nil {
			resp.Body.Close()
		}
	}()
	// At once from the proxy; then, from the engine's slots, that the prompt is going in.
	h.waitFor("the request at work", func() bool { return zeta().Active == 1 })
	h.waitFor("the slot on the prompt", func() bool { return zeta().state() == "processing prompt" })
	<-done
	if i := zeta(); i.state() != "idle" || i.Active != 0 || i.Stats.SlotsBusy != 0 {
		t.Errorf("after: %q, %d active, %d slots busy", i.state(), i.Active, i.Stats.SlotsBusy)
	}
}
