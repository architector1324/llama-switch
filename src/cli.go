package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"text/tabwriter"
	"time"
)

// --- Client commands: run, list, ps ---
//
// They talk to a running server over its own HTTP API, the way ollama's do, so
// they work the same against this machine or another one.

const usageText = `llmsw runs local inference engines side by side behind one dashboard and one
OpenAI-compatible endpoint.

Usage:
  llmsw serve [flags]          start the server and its dashboard
  llmsw run MODEL [PROMPT]     chat with a model, loading it first if needed
  llmsw list [TYPE]            models in the config (also: ls)
  llmsw ps                     models that are up (also: status)

The client commands talk to the server at $LLMSW_HOST (default localhost:11435).
Run "llmsw COMMAND -h" for a command's flags.
`

// Where the client commands find the server: host:port, or a full URL.
func serverURL() string {
	host := strings.TrimSpace(os.Getenv("LLMSW_HOST"))
	if host == "" {
		host = "localhost:11435"
	}
	if !strings.Contains(host, "://") {
		host = "http://" + host
	}
	return strings.TrimRight(host, "/")
}

// No timeout: an answer streams for as long as the model writes.
var cliClient = &http.Client{}

func fail(err error) {
	fmt.Fprintf(os.Stderr, "Error: %v\n", err)
	os.Exit(1)
}

func isTerminal(f *os.File) bool {
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}

func unreachable(server string, err error) error {
	return fmt.Errorf("could not reach the server at %s; is `llmsw serve` running? (%v)", server, err)
}

// The reason an answer gives for failing: this server says {"detail": ...},
// llama-server {"error": {"message": ...}}.
func errorMessage(resp *http.Response) string {
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var body struct {
		Detail any `json:"detail"`
		Error  any `json:"error"`
	}
	if json.Unmarshal(raw, &body) == nil {
		if s, ok := body.Detail.(string); ok && s != "" {
			return s
		}
		switch e := body.Error.(type) {
		case string:
			return e
		case map[string]any:
			if s, ok := e["message"].(string); ok {
				return s
			}
		}
	}
	if s := strings.TrimSpace(string(raw)); s != "" {
		return s
	}
	return resp.Status
}

func getJSON(server, path string, v any) error {
	resp, err := cliClient.Get(server + path)
	if err != nil {
		return unreachable(server, err)
	}
	if resp.StatusCode != http.StatusOK {
		return errors.New(errorMessage(resp))
	}
	defer resp.Body.Close()
	return json.NewDecoder(resp.Body).Decode(v)
}

// --- list and ps ---

type listedModel struct {
	Name    string `json:"name"`
	Kind    string `json:"kind"`
	Details struct {
		ParentModel       string `json:"parent_model"`
		QuantizationLevel string `json:"quantization_level"`
	} `json:"details"`
}

type runningInstance struct {
	Model        string `json:"model"`
	Quantization string `json:"quantization"`
	Kind         string `json:"kind"`
	Label        string `json:"label"`
	Devices      string `json:"devices"`
	Port         int    `json:"port"`
	Running      bool   `json:"running"`
	Ready        bool   `json:"ready"`
	Ctx          int    `json:"ctx"`
	Parallel     int    `json:"parallel"`
	Stats        struct {
		MemModel   float64 `json:"mem_model"`
		MemContext float64 `json:"mem_context"`
		MemCompute float64 `json:"mem_compute"`
	} `json:"stats"`
}

func instanceKey(model, quant string) string { return model + "\x00" + quant }

func (i runningInstance) state() string {
	switch {
	case !i.Running:
		return "exited"
	case !i.Ready:
		return "loading"
	}
	return "running"
}

func fetchModels(server string) ([]listedModel, error) {
	var out struct {
		Models []listedModel `json:"models"`
	}
	err := getJSON(server, "/v1/models", &out)
	return out.Models, err
}

func fetchInstances(server string) ([]runningInstance, error) {
	var out struct {
		Instances []runningInstance `json:"instances"`
	}
	err := getJSON(server, "/api/status", &out)
	return out.Instances, err
}

// The same MiB/GiB the dashboard shows.
func formatMemory(mib float64) string {
	if mib <= 0 {
		return "-"
	}
	if mib >= 1024 {
		return fmt.Sprintf("%.1f GiB", mib/1024)
	}
	return fmt.Sprintf("%.0f MiB", mib)
}

func printModels(server, kind string, w io.Writer) error {
	models, err := fetchModels(server)
	if err != nil {
		return err
	}
	instances, err := fetchInstances(server)
	if err != nil {
		return err
	}
	up := map[string]string{}
	for _, i := range instances {
		up[instanceKey(i.Model, i.Quantization)] = i.state()
	}

	tw := tabwriter.NewWriter(w, 0, 0, 3, ' ', 0)
	fmt.Fprintln(tw, "NAME\tTYPE\tQUANT\tSTATE")
	for _, m := range models {
		if kind != "" && m.Kind != kind {
			continue
		}
		quant := m.Details.QuantizationLevel
		if quant == "" {
			quant = "-"
		}
		state := up[instanceKey(m.Details.ParentModel, m.Details.QuantizationLevel)]
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", m.Name, m.Kind, quant, state)
	}
	return tw.Flush()
}

func printRunning(server string, w io.Writer) error {
	instances, err := fetchInstances(server)
	if err != nil {
		return err
	}
	// Named the way run and the API take them, not by the internal label.
	models, err := fetchModels(server)
	if err != nil {
		return err
	}
	names := map[string]string{}
	for _, m := range models {
		names[instanceKey(m.Details.ParentModel, m.Details.QuantizationLevel)] = m.Name
	}

	tw := tabwriter.NewWriter(w, 0, 0, 3, ' ', 0)
	fmt.Fprintln(tw, "NAME\tTYPE\tSTATE\tMEMORY\tGPU\tCONTEXT\tPORT")
	for _, i := range instances {
		name := names[instanceKey(i.Model, i.Quantization)]
		if name == "" {
			name = i.Label
		}
		gpu := i.Devices
		if gpu == "" {
			gpu = "default"
		}
		context := "-"
		if i.Ctx > 0 {
			context = strconv.Itoa(i.Ctx)
			if i.Parallel > 1 {
				context = fmt.Sprintf("%d x %d", i.Parallel, i.Ctx)
			}
		}
		port := "-"
		if i.Port > 0 {
			port = strconv.Itoa(i.Port)
		}
		mem := i.Stats.MemModel + i.Stats.MemContext + i.Stats.MemCompute
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			name, i.Kind, i.state(), formatMemory(mem), gpu, context, port)
	}
	return tw.Flush()
}

func listCommand(args []string) {
	fs := flag.NewFlagSet("list", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Usage: llmsw list [TYPE]\n\nModels in the config, with the type, quant and state of each.\nTYPE is one of %s.\n",
			strings.Join(sections, ", "))
	}
	fs.Parse(args)
	kind := fs.Arg(0)
	if fs.NArg() > 1 || (kind != "" && !isSection(kind)) {
		fs.Usage()
		os.Exit(2)
	}
	if err := printModels(serverURL(), kind, os.Stdout); err != nil {
		fail(err)
	}
}

func psCommand(args []string) {
	fs := flag.NewFlagSet("ps", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "Usage: llmsw ps\n\nModels that are up: their state, memory, cards, context window and port.")
	}
	fs.Parse(args)
	if fs.NArg() > 0 {
		fs.Usage()
		os.Exit(2)
	}
	if err := printRunning(serverURL(), os.Stdout); err != nil {
		fail(err)
	}
}

// --- run ---

type runOptions struct {
	Server       string
	Model        string
	ShowThinking bool
	Verbose      bool // timings after each answer
	Styled       bool // stdout is a terminal
	Spinner      bool // stderr is a terminal
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func runCommand(args []string) {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	hide := fs.Bool("hide-thinking", false, "leave the model's reasoning out of the answer")
	verbose := fs.Bool("verbose", false, "print token counts and timings after each answer")
	fs.Usage = func() {
		fmt.Fprint(fs.Output(), `Usage: llmsw run [flags] MODEL [PROMPT]

Chat with a language model; the server loads it first if it is not up. With
no prompt and a terminal, an interactive session; with a prompt, or text piped
in, one answer.

Flags:
`)
		fs.PrintDefaults()
	}
	fs.Parse(args)
	if fs.NArg() == 0 {
		fs.Usage()
		os.Exit(2)
	}
	model := fs.Arg(0)
	// Flags may follow the model too.
	fs.Parse(fs.Args()[1:])
	prompt := strings.Join(fs.Args(), " ")

	opts := runOptions{
		Server:  serverURL(),
		Model:   model,
		Verbose: *verbose,
		Styled:  isTerminal(os.Stdout),
		Spinner: isTerminal(os.Stderr),
	}
	// Piped somewhere else, only the answer is wanted.
	opts.ShowThinking = opts.Styled && !*hide

	interactive := isTerminal(os.Stdin)
	if !interactive {
		piped, err := io.ReadAll(os.Stdin)
		if err != nil {
			fail(err)
		}
		// The text first, what to do with it after, as with ollama.
		if text := strings.TrimSpace(string(piped)); text != "" {
			prompt = strings.TrimSpace(text + "\n\n" + prompt)
		}
	}

	if prompt != "" {
		if _, err := chat(context.Background(), opts, []chatMessage{{"user", prompt}}, os.Stdout, os.Stderr); err != nil {
			fail(err)
		}
		return
	}
	if !interactive {
		fail(errors.New("nothing to send: give a prompt or pipe one in"))
	}

	interrupts := make(chan os.Signal, 1)
	signal.Notify(interrupts, os.Interrupt)
	if err := repl(opts, os.Stdin, os.Stdout, os.Stderr, interrupts); err != nil {
		fail(err)
	}
}

const replHelp = `Available commands:
  /clear    forget the conversation so far
  /bye      exit (Ctrl+D works too)
  /?        this help

Use """ to begin and end a message that spans lines.
Ctrl+C stops the answer being written.
`

const replPrompt = "> "

// An interactive session: the conversation so far goes with every message.
func repl(opts runOptions, in io.Reader, out, errOut io.Writer, interrupts <-chan os.Signal) error {
	br := bufio.NewReader(in)
	var history []chatMessage

	// Ctrl+C stops the answer in progress; at the prompt it says how to leave.
	var mu sync.Mutex
	var cancelAnswer context.CancelFunc
	if interrupts != nil {
		go func() {
			for range interrupts {
				mu.Lock()
				cancel := cancelAnswer
				mu.Unlock()
				if cancel != nil {
					cancel()
				} else {
					fmt.Fprint(out, "\nUse Ctrl + d or /bye to exit.\n"+replPrompt)
				}
			}
		}()
	}

	for {
		fmt.Fprint(out, replPrompt)
		text, err := readMessage(br, out)
		if err != nil {
			fmt.Fprintln(out)
			return nil // Ctrl+D
		}

		switch cmd := strings.TrimSpace(text); {
		case cmd == "":
			continue
		case cmd == "/bye" || cmd == "/exit":
			return nil
		case cmd == "/clear":
			history = nil
			fmt.Fprintln(out, "Cleared session context")
			continue
		case cmd == "/?" || cmd == "/help":
			fmt.Fprint(out, replHelp)
			continue
		case strings.HasPrefix(cmd, "/") && !strings.Contains(cmd, "\n"):
			fmt.Fprintf(out, "Unknown command '%s'. Type /? for help\n", strings.Fields(cmd)[0])
			continue
		}

		history = append(history, chatMessage{"user", text})
		ctx, cancel := context.WithCancel(context.Background())
		mu.Lock()
		cancelAnswer = cancel
		mu.Unlock()

		answer, err := chat(ctx, opts, history, out, errOut)

		mu.Lock()
		cancelAnswer = nil
		mu.Unlock()
		stopped := ctx.Err() != nil
		cancel()

		if err != nil {
			// An answer that did not finish is not part of the conversation.
			history = history[:len(history)-1]
			if stopped {
				fmt.Fprintln(out)
			} else {
				fmt.Fprintf(errOut, "Error: %v\n", err)
			}
			continue
		}
		history = append(history, chatMessage{"assistant", answer})
		fmt.Fprintln(out)
	}
}

// One message: a line, or everything between a pair of """.
func readMessage(br *bufio.Reader, out io.Writer) (string, error) {
	line, err := br.ReadString('\n')
	if err != nil && line == "" {
		return "", err
	}
	line = strings.TrimRight(line, "\r\n")
	if !strings.HasPrefix(line, `"""`) {
		return line, nil
	}
	first := strings.TrimPrefix(line, `"""`)
	if strings.HasSuffix(first, `"""`) {
		return strings.TrimSuffix(first, `"""`), nil
	}

	var lines []string
	if first != "" {
		lines = append(lines, first)
	}
	for {
		fmt.Fprint(out, "... ")
		next, err := br.ReadString('\n')
		next = strings.TrimRight(next, "\r\n")
		if strings.HasSuffix(next, `"""`) {
			lines = append(lines, strings.TrimSuffix(next, `"""`))
			break
		}
		lines = append(lines, next)
		if err != nil {
			break
		}
	}
	return strings.Join(lines, "\n"), nil
}

// Send the conversation and print the answer as it streams. The text it
// returns is the answer alone, reasoning left out, as the next turn needs it.
func chat(ctx context.Context, opts runOptions, messages []chatMessage, out, errOut io.Writer) (string, error) {
	body, _ := json.Marshal(map[string]any{
		"model": opts.Model, "messages": messages, "stream": true,
		// The last event then carries the token counts, and from llama-server the timings too.
		"stream_options": map[string]any{"include_usage": true},
	})
	started := time.Now()
	var usage *chatUsage
	var timings *chatTimings

	spin := startSpinner(opts.Spinner, errOut)
	defer spin.stop()

	resp, err := postChat(ctx, opts.Server, body)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	p := &answerPrinter{out: out, showThinking: opts.ShowThinking, styled: opts.Styled}
	defer p.finish()

	br := bufio.NewReader(resp.Body)
	for {
		line, readErr := br.ReadString('\n')
		line = strings.TrimRight(line, "\r\n")
		if data, ok := strings.CutPrefix(line, "data: "); ok {
			if data == "[DONE]" {
				break
			}
			var ev struct {
				Choices []struct {
					Delta struct {
						Content          string `json:"content"`
						ReasoningContent string `json:"reasoning_content"`
						Reasoning        string `json:"reasoning"`
					} `json:"delta"`
				} `json:"choices"`
				Error *struct {
					Message string `json:"message"`
				} `json:"error"`
				Usage   *chatUsage   `json:"usage"`
				Timings *chatTimings `json:"timings"`
			}
			if json.Unmarshal([]byte(data), &ev) == nil {
				if ev.Error != nil {
					spin.stop()
					return p.answer.String(), errors.New(ev.Error.Message)
				}
				if ev.Usage != nil {
					usage = ev.Usage
				}
				if ev.Timings != nil {
					timings = ev.Timings
				}
				for _, c := range ev.Choices {
					if r := c.Delta.ReasoningContent + c.Delta.Reasoning; r != "" {
						spin.stop()
						p.think(r)
					}
					if c.Delta.Content != "" {
						spin.stop()
						p.say(c.Delta.Content)
					}
				}
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return p.answer.String(), readErr
		}
	}

	p.finish()
	spin.stop()
	if opts.Verbose {
		printStats(errOut, time.Since(started), usage, timings)
	}
	return p.answer.String(), nil
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// What llama-server reports about one answer, in its last event.
type chatTimings struct {
	CacheN             int     `json:"cache_n"`
	PromptN            int     `json:"prompt_n"`
	PromptMs           float64 `json:"prompt_ms"`
	PromptPerSecond    float64 `json:"prompt_per_second"`
	PredictedN         int     `json:"predicted_n"`
	PredictedMs        float64 `json:"predicted_ms"`
	PredictedPerSecond float64 `json:"predicted_per_second"`
	DraftN             int     `json:"draft_n"`
	DraftNAccepted     int     `json:"draft_n_accepted"`
}

func milliseconds(ms float64) time.Duration {
	return time.Duration(ms * float64(time.Millisecond))
}

// The summary ollama's --verbose prints, from the engine's own figures where it
// gives them. Load duration is whatever the request spent before the engine
// took the prompt: the load itself when the model was not up, next to nothing
// when it was.
func printStats(w io.Writer, total time.Duration, usage *chatUsage, t *chatTimings) {
	line := func(label, format string, a ...any) {
		fmt.Fprintf(w, "%-22s"+format+"\n", append([]any{label + ":"}, a...)...)
	}
	round := func(d time.Duration) time.Duration { return d.Round(100 * time.Microsecond) }

	line("total duration", "%v", round(total))
	switch {
	case t != nil:
		if load := total - milliseconds(t.PromptMs+t.PredictedMs); load > 0 {
			line("load duration", "%v", round(load))
		}
		line("prompt eval count", "%d token(s)", t.PromptN)
		if t.CacheN > 0 {
			line("prompt cache", "%d token(s)", t.CacheN)
		}
		line("prompt eval duration", "%v", round(milliseconds(t.PromptMs)))
		line("prompt eval rate", "%.2f tokens/s", t.PromptPerSecond)
		line("eval count", "%d token(s)", t.PredictedN)
		line("eval duration", "%v", round(milliseconds(t.PredictedMs)))
		line("eval rate", "%.2f tokens/s", t.PredictedPerSecond)
		if t.DraftN > 0 {
			line("draft accepted", "%d/%d (%.0f%%)", t.DraftNAccepted, t.DraftN,
				100*float64(t.DraftNAccepted)/float64(t.DraftN))
		}
	case usage != nil:
		line("prompt eval count", "%d token(s)", usage.PromptTokens)
		line("eval count", "%d token(s)", usage.CompletionTokens)
	}
}

// The server gives up on a request after a minute of loading, not on the
// model: it keeps loading, and asking again waits for the same load. So a
// big model is simply waited for.
func postChat(ctx context.Context, server string, body []byte) (*http.Response, error) {
	for {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, server+"/v1/chat/completions", bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := cliClient.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			return nil, unreachable(server, err)
		}
		if resp.StatusCode == http.StatusOK {
			return resp, nil
		}
		msg := errorMessage(resp)
		if resp.StatusCode == http.StatusGatewayTimeout && strings.HasSuffix(msg, "failed to load within timeout") {
			continue
		}
		return nil, errors.New(msg)
	}
}

// Writes the streamed answer: reasoning dimmed between two markers, the way
// ollama shows it, then the answer itself.
type answerPrinter struct {
	out          io.Writer
	showThinking bool
	styled       bool
	thinking     bool
	reasoned     bool // reasoning came, shown or not, and the answer has not started
	wrote        bool
	lastNewline  bool
	answer       strings.Builder
}

func (p *answerPrinter) write(s string, dim bool) {
	if s == "" {
		return
	}
	if dim && p.styled {
		fmt.Fprint(p.out, "\x1b[90m"+s+"\x1b[0m")
	} else {
		fmt.Fprint(p.out, s)
	}
	p.wrote = true
	p.lastNewline = strings.HasSuffix(s, "\n")
}

func (p *answerPrinter) think(s string) {
	if p.answer.Len() == 0 {
		p.reasoned = true
	}
	if !p.showThinking {
		return
	}
	if !p.thinking {
		p.thinking = true
		p.write("Thinking...\n", true)
	}
	p.write(s, true)
}

func (p *answerPrinter) endThinking() {
	if !p.thinking {
		return
	}
	p.thinking = false
	if !p.lastNewline {
		p.write("\n", true)
	}
	p.write("...done thinking.\n\n", true)
}

func (p *answerPrinter) say(s string) {
	p.endThinking()
	if p.reasoned {
		// The answer often opens with the newlines that closed the reasoning.
		if s = strings.TrimLeft(s, "\n"); s == "" {
			return
		}
		p.reasoned = false
	}
	p.write(s, false)
	p.answer.WriteString(s)
}

func (p *answerPrinter) finish() {
	p.endThinking()
	if p.wrote && !p.lastNewline {
		p.write("\n", false)
	}
}

// Spins on stderr until the first token, so a model loading is not mistaken
// for a hang.
type spinner struct {
	once    sync.Once
	done    chan struct{}
	stopped chan struct{}
}

func startSpinner(enabled bool, w io.Writer) *spinner {
	s := &spinner{done: make(chan struct{}), stopped: make(chan struct{})}
	if !enabled {
		close(s.stopped)
		return s
	}
	go func() {
		defer close(s.stopped)
		frames := []rune("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
		tick := time.NewTicker(100 * time.Millisecond)
		defer tick.Stop()
		for i := 0; ; i++ {
			select {
			case <-s.done:
				fmt.Fprint(w, "\r\x1b[K")
				return
			case <-tick.C:
				fmt.Fprintf(w, "\r%c ", frames[i%len(frames)])
			}
		}
	}()
	return s
}

func (s *spinner) stop() {
	s.once.Do(func() { close(s.done) })
	<-s.stopped
}
