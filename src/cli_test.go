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
	if err := repl(runOptions{Server: url, Model: "m"}, strings.NewReader(input), &out, &errOut, nil); err != nil {
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
	if err := repl(runOptions{Server: url, Model: "m"}, strings.NewReader("hi\nagain\n"), &out, &errOut, nil); err != nil {
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
		done <- repl(runOptions{Server: url, Model: "m"}, strings.NewReader("tell a story\nhi\n/bye\n"),
			out, io.Discard, interrupts)
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
	h.ok("POST", "/api/multi", map[string]any{"multi": true})
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
		"zeta": "llm q4", "zeta-q8": "llm q8 running", "legacy": "llm -",
		"img": "sd q8_0 running", "song-sfx": "music sfx", "ears": "stt f16",
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
	if got := strings.Join(ps["zeta-q8"], " "); !strings.HasPrefix(got, "llm running 2.5 GiB default 2 x 4096 ") {
		t.Errorf("ps zeta-q8: %q", got)
	}
	if got := strings.Join(ps["img"], " "); !strings.HasPrefix(got, "sd running - default - ") {
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
