package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// --- run ---
//
// One command for every kind: what it does with a message depends on what the
// model is. A language model answers it, an image model draws it, a speech
// model says it, a music model plays it, a transcription model takes it as the
// path of a recording, and a decision model as the state its questions are about.

type runOptions struct {
	Server       string
	Model        string // the id the API takes
	ModelKey     string // the config key behind it, which /api/status goes by
	Kind         string
	System       string // llm: the system prompt
	ShowThinking bool
	Verbose      bool // figures after each result
	Styled       bool // stdout is a terminal
	Spinner      bool // stderr is a terminal

	// llm, sd: text piped in, ahead of the prompt as it is. It is read, not
	// searched for files to attach: only the prompt names those.
	Piped string

	Output    string   // where an image or a clip goes
	Images    []string // sd: images to edit rather than draw anew
	Mask      string   // sd: which part of them to edit
	Play      bool     // play a clip once it is saved
	Voice     string
	Duration  float64
	Lyrics    string
	Questions json.RawMessage // a decision model's questions, in the file's order
}

const runUsage = `Usage: llmsw run [flags] MODEL [PROMPT]

Run a model; the server loads it first if it is not up. With a prompt, or
input piped in, one result; with neither and a terminal, a session.

What a model does with its input depends on its type:
  llm       answers it; a session keeps the conversation. Image, audio and
            video files named in it go along, for a model that takes them
  sd        draws it into a PNG; image files named in it, or with --image,
            it edits instead
  tts       says it, into a WAV it then plays
  music     composes from it, into a WAV it then plays
  stt       transcribes the recordings named, or the audio piped in
  decision  takes the prompt as a yes/no question about the piped state,
            or the state for the questions in --questions

A file is named by its path: as it is, quoted, with its spaces escaped, or
dragged in. Images and audio made go to -o, to stdout when that is not a
terminal, or else to a new file in the current directory.

The dashboard's settings are flags too: --gpu, --ctx, --parallel,
--preserve-thinking, --size, --hires, --frames, --multi. Given one, the model
starts with it, as a change on the dashboard would restart it; whatever is not
given stays as it was.

Flags:
`

// The kinds each flag means something for; a flag missing here suits all.
var runFlagKinds = map[string][]string{
	"hide-thinking":     {"llm"},
	"system":            {"llm"},
	"ctx":               {"llm", "decision"},
	"parallel":          {"llm", "decision"},
	"preserve-thinking": {"llm"},
	"size":              {"sd"},
	"image":             {"sd"},
	"mask":              {"sd"},
	"hires":             {"sd"},
	"frames":            {"tts"},
	"o":                 {"sd", "tts", "music"},
	"no-play":           {"tts", "music"},
	"voice":             {"tts"},
	"duration":          {"music"},
	"lyrics":            {"music"},
	"questions":         {"decision"},
}

// Where run's flags land.
type runArgs struct {
	hide, verbose, noPlay                          bool
	system, output, voice, lyrics, questions, mask string
	duration                                       float64
	images                                         fileList
	settings                                       runSettings
}

func runFlags(a *runArgs) *flag.FlagSet {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	fs.BoolVar(&a.hide, "hide-thinking", false, "llm: leave the reasoning out of the answer")
	fs.StringVar(&a.system, "system", "", "llm: the system prompt")
	fs.BoolVar(&a.verbose, "verbose", false, "print timings and counts after each result")
	fs.StringVar(&a.output, "o", "", "sd, tts, music: write the result to this `file`")
	fs.BoolVar(&a.noPlay, "no-play", false, "tts, music: save the audio without playing it")
	fs.StringVar(&a.voice, "voice", "", "tts: the voice to speak with")
	fs.Float64Var(&a.duration, "duration", 0, "music: length in seconds (default: the model's)")
	fs.StringVar(&a.lyrics, "lyrics", "", "music: lyrics to sing")
	fs.StringVar(&a.questions, "questions", "", "decision: a JSON `file` of the questions to ask")
	fs.Var(&a.images, "image", "sd: an image `file` to edit; give it again for several")
	fs.StringVar(&a.mask, "mask", "", "sd: a mask `file` for the part of the image to edit")
	s := &a.settings
	fs.StringVar(&s.gpu, "gpu", "", `card(s) to run on, as the dashboard names them, or "all"; sd also takes role=card pairs: diffusion, te, vae`)
	fs.IntVar(&s.ctx, "ctx", 0, "llm, decision: context window per slot")
	fs.IntVar(&s.parallel, "parallel", 0, "llm, decision: parallel slots")
	fs.BoolVar(&s.preserve, "preserve-thinking", false, "llm: feed earlier reasoning back into the prompt")
	fs.StringVar(&s.size, "size", "", "sd: output size, WIDTHxHEIGHT")
	fs.BoolVar(&s.hires, "hires", false, "sd: hires fix")
	fs.IntVar(&s.frames, "frames", 0, "tts: max frames")
	fs.BoolVar(&s.multi, "multi", false, "turn Multi-model on, or off with --multi=false, before loading")
	fs.Usage = func() {
		fmt.Fprint(fs.Output(), runUsage)
		printFlags(fs, nil)
	}
	return fs
}

func runCommand(args []string) {
	var a runArgs
	fs := runFlags(&a)
	fs.Parse(args)
	if fs.NArg() == 0 {
		fs.Usage()
		os.Exit(2)
	}
	model := fs.Arg(0)
	// Flags may follow the model too.
	fs.Parse(fs.Args()[1:])
	rest := fs.Args()
	prompt := strings.Join(rest, " ")
	s := a.settings
	s.set = map[string]bool{}
	fs.Visit(func(f *flag.Flag) { s.set[f.Name] = true })

	server := serverURL()
	entry, err := findModel(server, model)
	if err != nil {
		fail(err)
	}
	if err := checkRunFlags(s.set, model, entry.Kind); err != nil {
		fs.Usage()
		fail(err)
	}
	opts := runOptions{
		Server:   server,
		Model:    model,
		ModelKey: entry.Details.ParentModel,
		Kind:     entry.Kind,
		System:   a.system,
		Verbose:  a.verbose,
		Styled:   isTerminal(os.Stdout),
		Spinner:  isTerminal(os.Stderr),
		Output:   a.output,
		Images:   a.images,
		Mask:     a.mask,
		Play:     !a.noPlay,
		Voice:    a.voice,
		Duration: a.duration,
		Lyrics:   a.lyrics,
	}
	// Piped somewhere else, only the answer is wanted.
	opts.ShowThinking = opts.Styled && !a.hide
	if a.questions != "" {
		if opts.Questions, err = readQuestions(a.questions); err != nil {
			fail(err)
		}
	}

	interactive := isTerminal(os.Stdin)
	var piped []byte
	if !interactive {
		if piped, err = io.ReadAll(os.Stdin); err != nil {
			fail(err)
		}
	}
	text := joinInput(string(piped), prompt)
	if opts.Kind == "llm" || opts.Kind == "sd" {
		opts.Piped, text = strings.TrimSpace(string(piped)), prompt
	}

	ctx := context.Background()
	if err := applySettings(ctx, opts, entry.Details.QuantizationLevel, s); err != nil {
		fail(err)
	}
	task := newRunTask(opts, os.Stdout, os.Stderr)
	switch {
	case opts.Kind == "stt" && len(piped) > 0:
		err = transcribe(ctx, opts, "audio", piped, os.Stdout, os.Stderr)
	case opts.Kind == "stt" && len(rest) > 0:
		for _, path := range rest {
			if len(rest) > 1 {
				fmt.Printf("%s:\n", path)
			}
			if err = transcribeFile(ctx, opts, path, os.Stdout, os.Stderr); err != nil {
				break
			}
		}
	case opts.Kind == "decision" && opts.Questions == nil && prompt != "":
		err = decide(ctx, opts, yesNoQuestion(prompt), strings.TrimSpace(string(piped)), os.Stdout, os.Stderr)
	case text != "" || opts.Piped != "":
		err = task.turn(ctx, text)
	case !interactive:
		err = errors.New("nothing to send: give a prompt or pipe one in")
	case opts.Output != "":
		err = errors.New("-o takes one result: give a prompt to use it")
	case opts.Kind == "decision" && opts.Questions == nil:
		err = errors.New("a session with a decision model needs --questions FILE; " +
			`or ask one yes/no question: llmsw run ` + model + ` "Is this urgent?" < state.txt`)
	default:
		interrupts := make(chan os.Signal, 1)
		signal.Notify(interrupts, os.Interrupt)
		err = repl(os.Stdin, os.Stdout, os.Stderr, interrupts, task)
	}
	if err != nil {
		fail(err)
	}
}

// A flag that may be given more than once, each time a file.
type fileList []string

func (l *fileList) String() string     { return strings.Join(*l, ",") }
func (l *fileList) Set(v string) error { *l = append(*l, v); return nil }

// A flag for another kind of model would be quietly ignored; say so instead.
func checkRunFlags(set map[string]bool, model, kind string) error {
	var names []string
	for name := range set {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		kinds, limited := runFlagKinds[name]
		if limited && !slices.Contains(kinds, kind) {
			dash := "--"
			if len(name) == 1 {
				dash = "-"
			}
			return fmt.Errorf("%s%s is for %s models, and %s is a %s model",
				dash, name, joinWords(kinds), model, kind)
		}
	}
	return nil
}

// "a", "a and b", "a, b and c".
func joinWords(words []string) string {
	if len(words) < 3 {
		return strings.Join(words, " and ")
	}
	return strings.Join(words[:len(words)-1], ", ") + " and " + words[len(words)-1]
}

// The dashboard's settings, as run's flags give them.
type runSettings struct {
	set                    map[string]bool // the flags given
	gpu, size              string
	ctx, parallel, frames  int
	preserve, hires, multi bool
}

// Do what the dashboard does when a field changes: start the model with the
// new value. The fields not given keep the running instance's values, and a
// model already running exactly so is left alone.
func applySettings(ctx context.Context, opts runOptions, quant string, s runSettings) error {
	if s.set["multi"] {
		body, _ := json.Marshal(map[string]bool{"multi": s.multi})
		resp, err := post(ctx, opts.Server, "/api/multi", "application/json", body)
		if err != nil {
			return err
		}
		resp.Body.Close()
	}

	want := map[string]any{}
	if s.set["gpu"] {
		devices := strings.TrimSpace(s.gpu)
		if devices == "all" {
			devices = ""
		}
		want["devices"] = devices
	}
	if s.set["ctx"] {
		want["ctx"] = s.ctx
	}
	if s.set["parallel"] {
		want["parallel"] = s.parallel
	}
	if s.set["preserve-thinking"] {
		want["preserve_think"] = s.preserve
	}
	if s.set["size"] {
		want["res"] = strings.ToLower(strings.TrimSpace(s.size))
	}
	if s.set["hires"] {
		want["hires"] = s.hires
	}
	if s.set["frames"] {
		want["frames"] = s.frames
	}
	if len(want) == 0 {
		return nil
	}

	var up struct {
		Running       bool   `json:"running"`
		Quantization  string `json:"quantization"`
		Devices       string `json:"devices"`
		Ctx           int    `json:"ctx"`
		Parallel      int    `json:"parallel"`
		PreserveThink bool   `json:"preserve_think"`
		Res           string `json:"res"`
		Hires         bool   `json:"hires"`
		Frames        int    `json:"frames"`
	}
	if err := getJSON(opts.Server, "/api/status?model="+url.QueryEscape(opts.ModelKey), &up); err != nil {
		return err
	}
	req := map[string]any{"model_key": opts.ModelKey, "quantization": quant}
	if up.Running && up.Quantization == quant {
		current := map[string]any{
			"devices": up.Devices, "ctx": up.Ctx, "parallel": up.Parallel,
			"preserve_think": up.PreserveThink, "res": up.Res, "hires": up.Hires, "frames": up.Frames,
		}
		same := true
		for k, v := range want {
			if current[k] != v {
				same = false
			}
		}
		if same {
			return nil
		}
		// Zero means the kind has no such setting; the server fills in its own.
		for k, v := range current {
			if v != 0 && v != "" {
				req[k] = v
			}
		}
		req["devices"] = up.Devices
	}
	for k, v := range want {
		req[k] = v
	}

	body, _ := json.Marshal(req)
	resp, err := post(ctx, opts.Server, "/api/start", "application/json", body)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

// The model's entry in the server's list, which says what kind it is.
func findModel(server, name string) (listedModel, error) {
	models, err := fetchModels(server)
	if err != nil {
		return listedModel{}, err
	}
	for _, m := range models {
		if m.Name == name {
			return m, nil
		}
	}
	return listedModel{}, fmt.Errorf("Model %s not found; `llmsw list` shows the ones there are", name)
}

// What run does with one message, for the kind of model it runs.
type runTask struct {
	turn  func(ctx context.Context, text string) error
	clear func() // forgets the conversation; nil where there is none
	hint  string // what a message is, for /?
}

func newRunTask(opts runOptions, out, errOut io.Writer) runTask {
	switch opts.Kind {
	case "sd":
		return runTask{
			turn: func(ctx context.Context, text string) error { return generateImage(ctx, opts, text, out, errOut) },
			hint: "Each message is a prompt for a new image. Name an image file in it, or\n" +
				"drag one in, and the prompt edits that image instead.",
		}
	case "tts":
		return runTask{
			turn: func(ctx context.Context, text string) error { return speak(ctx, opts, text, out, errOut) },
			hint: "Each message is spoken.",
		}
	case "music":
		return runTask{
			turn: func(ctx context.Context, text string) error { return compose(ctx, opts, text, out, errOut) },
			hint: "Each message is a prompt for a new piece.",
		}
	case "stt":
		return runTask{
			turn: func(ctx context.Context, text string) error { return transcribeFile(ctx, opts, text, out, errOut) },
			hint: "Each message is the path of a recording to transcribe.",
		}
	case "decision":
		return runTask{
			turn: func(ctx context.Context, text string) error {
				return decide(ctx, opts, opts.Questions, text, out, errOut)
			},
			hint: "Each message is the state the questions are asked about.",
		}
	}
	return chatTask(opts, out, errOut)
}

type chatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"` // the text, or parts: files and the text
}

// The conversation so far goes with every message, after the system prompt,
// which /clear keeps.
func chatTask(opts runOptions, out, errOut io.Writer) runTask {
	var history []chatMessage
	return runTask{
		turn: func(ctx context.Context, text string) error {
			content, err := userContent(ctx, opts, text, out, errOut)
			if err != nil {
				return err
			}
			history = append(history, chatMessage{"user", content})
			messages := history
			if opts.System != "" {
				messages = append([]chatMessage{{"system", opts.System}}, history...)
			}
			answer, err := chat(ctx, opts, messages, out, errOut)
			if err != nil {
				// An answer that did not finish is not part of the conversation.
				history = history[:len(history)-1]
				return err
			}
			history = append(history, chatMessage{"assistant", answer})
			return nil
		},
		clear: func() { history = nil },
		hint: "Each message goes with the conversation so far. Name an image, audio or\n" +
			"video file in it, or drag one in, and the file goes along.",
	}
}

// A message as the chat API takes it. The files it names go as parts of their
// own, ahead of the text, where Gemma and Qwen want them; whether the model
// takes them is for the engine to say.
func userContent(ctx context.Context, opts runOptions, typed string, out, errOut io.Writer) (any, error) {
	prompt, paths, _ := extractFiles(typed, false, "image", "audio", "video")
	text := joinInput(opts.Piped, prompt)
	if len(paths) == 0 {
		return text, nil
	}
	var parts []map[string]any
	for _, p := range paths {
		part, err := mediaPart(ctx, p)
		if err != nil {
			return nil, err
		}
		fmt.Fprintf(noteWriter(opts, out, errOut), "Added %s %s\n", mediaKind(p), p)
		parts = append(parts, part)
	}
	if text != "" {
		parts = append(parts, map[string]any{"type": "text", "text": text})
	}
	return parts, nil
}

// A file as llama-server takes it: an image or a video as a data URL, audio
// as base64 with its format. WAV, MP3 and FLAC it decodes itself; anything
// else goes through ffmpeg into WAV first.
func mediaPart(ctx context.Context, path string) (map[string]any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	ext := strings.ToLower(filepath.Ext(path))
	dataURL := map[string]string{"url": "data:" + mediaTypes[ext] + ";base64," + base64.StdEncoding.EncodeToString(data)}
	switch mediaKind(path) {
	case "image":
		return map[string]any{"type": "image_url", "image_url": dataURL}, nil
	case "video":
		return map[string]any{"type": "video_url", "video_url": dataURL}, nil
	}
	format := strings.TrimPrefix(ext, ".")
	if format != "wav" && format != "mp3" && format != "flac" {
		if data, err = convertToWav(ctx, data); err != nil {
			return nil, fmt.Errorf("%s: %v", path, err)
		}
		format = "wav"
	}
	return map[string]any{"type": "input_audio", "input_audio": map[string]string{
		"data": base64.StdEncoding.EncodeToString(data), "format": format,
	}}, nil
}

const replPrompt = "> "

func replHelp(task runTask) string {
	clear := ""
	if task.clear != nil {
		clear = "  /clear    forget the conversation so far\n"
	}
	return "Available commands:\n" + clear + `  /bye      exit (Ctrl+D works too)
  /?        this help

` + task.hint + `
Use """ to begin and end a message that spans lines.
Ctrl+C stops the result being made.
`
}

// An interactive session, one message at a time.
func repl(in io.Reader, out, errOut io.Writer, interrupts <-chan os.Signal, task runTask) error {
	br := bufio.NewReader(in)

	// Ctrl+C stops the result in progress; at the prompt it says how to leave.
	var mu sync.Mutex
	var cancelTurn context.CancelFunc
	if interrupts != nil {
		go func() {
			for range interrupts {
				mu.Lock()
				cancel := cancelTurn
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
		case cmd == "/clear" && task.clear != nil:
			task.clear()
			fmt.Fprintln(out, "Cleared session context")
			continue
		case cmd == "/?" || cmd == "/help":
			fmt.Fprint(out, replHelp(task))
			continue
		// A command is one word; a path typed or dragged in has more slashes, and is a message.
		case strings.HasPrefix(cmd, "/") && !strings.Contains(cmd, "\n") && !strings.Contains(strings.Fields(cmd)[0][1:], "/"):
			fmt.Fprintf(out, "Unknown command '%s'. Type /? for help\n", strings.Fields(cmd)[0])
			continue
		}

		ctx, cancel := context.WithCancel(context.Background())
		mu.Lock()
		cancelTurn = cancel
		mu.Unlock()

		err = task.turn(ctx, text)

		mu.Lock()
		cancelTurn = nil
		mu.Unlock()
		stopped := ctx.Err() != nil
		cancel()

		if err != nil {
			if stopped {
				fmt.Fprintln(out)
			} else {
				fmt.Fprintf(errOut, "Error: %v\n", err)
			}
			continue
		}
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

// --- Talking to the server ---

// The server gives up on a request after a minute of loading, not on the
// model: it keeps loading, and asking again waits for the same load. So a
// big model is simply waited for.
func post(ctx context.Context, server, path, contentType string, body []byte) (*http.Response, error) {
	for {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, server+path, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", contentType)
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

// A JSON request whose JSON answer lands in v, with the spinner going meanwhile.
func runJSON(ctx context.Context, opts runOptions, errOut io.Writer, path string, req, v any) error {
	body, _ := json.Marshal(req)
	spin := startSpinner(opts.Spinner, errOut, progress(opts))
	defer spin.stop()
	resp, err := post(ctx, opts.Server, path, "application/json", body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(v); err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return fmt.Errorf("unreadable answer: %v", err)
	}
	return nil
}

// What the spinner says while the server works: that the model is loading,
// or how far an image has got.
func progress(opts runOptions) func() string {
	if opts.ModelKey == "" {
		return nil
	}
	path := "/api/status?model=" + url.QueryEscape(opts.ModelKey)
	return func() string {
		var st struct {
			Running bool `json:"running"`
			Ready   bool `json:"ready"`
			Stats   struct {
				SdStep  int `json:"sd_step"`
				SdSteps int `json:"sd_steps"`
			} `json:"stats"`
		}
		if getJSON(opts.Server, path, &st) != nil {
			return ""
		}
		switch {
		case !st.Running || !st.Ready:
			return "loading " + opts.Model
		case st.Stats.SdStep > 0 && st.Stats.SdStep < st.Stats.SdSteps:
			return fmt.Sprintf("step %d/%d", st.Stats.SdStep, st.Stats.SdSteps)
		}
		return ""
	}
}

// --- Chat ---

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

	spin := startSpinner(opts.Spinner, errOut, progress(opts))
	defer spin.stop()

	resp, err := post(ctx, opts.Server, "/v1/chat/completions", "application/json", body)
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

// One line of the --verbose summary, aligned the way ollama aligns it.
func statLine(w io.Writer, label, format string, a ...any) {
	fmt.Fprintf(w, "%-22s"+format+"\n", append([]any{label + ":"}, a...)...)
}

func roundDuration(d time.Duration) time.Duration { return d.Round(100 * time.Microsecond) }

// The summary ollama's --verbose prints, from the engine's own figures where it
// gives them. Load duration is whatever the request spent before the engine
// took the prompt: the load itself when the model was not up, next to nothing
// when it was.
func printStats(w io.Writer, total time.Duration, usage *chatUsage, t *chatTimings) {
	statLine(w, "total duration", "%v", roundDuration(total))
	switch {
	case t != nil:
		if load := total - milliseconds(t.PromptMs+t.PredictedMs); load > 0 {
			statLine(w, "load duration", "%v", roundDuration(load))
		}
		statLine(w, "prompt eval count", "%d token(s)", t.PromptN)
		if t.CacheN > 0 {
			statLine(w, "prompt cache", "%d token(s)", t.CacheN)
		}
		statLine(w, "prompt eval duration", "%v", roundDuration(milliseconds(t.PromptMs)))
		statLine(w, "prompt eval rate", "%.2f tokens/s", t.PromptPerSecond)
		statLine(w, "eval count", "%d token(s)", t.PredictedN)
		statLine(w, "eval duration", "%v", roundDuration(milliseconds(t.PredictedMs)))
		statLine(w, "eval rate", "%.2f tokens/s", t.PredictedPerSecond)
		if t.DraftN > 0 {
			statLine(w, "draft accepted", "%d/%d (%.0f%%)", t.DraftNAccepted, t.DraftN,
				100*float64(t.DraftNAccepted)/float64(t.DraftN))
		}
	case usage != nil:
		statLine(w, "prompt eval count", "%d token(s)", usage.PromptTokens)
		statLine(w, "eval count", "%d token(s)", usage.CompletionTokens)
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

// Spins on stderr until the first result, so a model loading is not mistaken
// for a hang. label, polled on the side, says what is being waited for.
type spinner struct {
	once    sync.Once
	done    chan struct{}
	stopped chan struct{}
}

func startSpinner(enabled bool, w io.Writer, label func() string) *spinner {
	s := &spinner{done: make(chan struct{}), stopped: make(chan struct{})}
	if !enabled {
		close(s.stopped)
		return s
	}

	var text atomic.Pointer[string]
	if label != nil {
		go func() {
			for {
				l := label()
				text.Store(&l)
				select {
				case <-s.done:
					return
				case <-time.After(500 * time.Millisecond):
				}
			}
		}()
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
				l := ""
				if p := text.Load(); p != nil {
					l = *p
				}
				fmt.Fprintf(w, "\r%c %s\x1b[K", frames[i%len(frames)], l)
			}
		}
	}()
	return s
}

func (s *spinner) stop() {
	s.once.Do(func() { close(s.done) })
	<-s.stopped
}
