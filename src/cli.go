package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
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
  llmsw ps [MODEL...]          models that are up, or the details of those named
                               (also: status)
  llmsw stop MODEL...          unload models; --all unloads every one
  llmsw completion fish        print the completion script for fish

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

// For the short calls: listings, status, the spinner's polls.
var apiClient = &http.Client{Timeout: 10 * time.Second}

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
	resp, err := apiClient.Get(server + path)
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
	Model         string `json:"model"`
	Quantization  string `json:"quantization"`
	Kind          string `json:"kind"`
	Label         string `json:"label"`
	Devices       string `json:"devices"`
	Port          int    `json:"port"`
	PID           int    `json:"pid"`
	Running       bool   `json:"running"`
	Ready         bool   `json:"ready"`
	Active        int    `json:"active"`
	Ctx           int    `json:"ctx"`
	Parallel      int    `json:"parallel"`
	PreserveThink bool   `json:"preserve_think"`
	Res           string `json:"res"`
	Hires         bool   `json:"hires"`
	Frames        int    `json:"frames"`
	Stats         Stats  `json:"stats"`
}

func instanceKey(model, quant string) string { return model + "\x00" + quant }

// What a model at work is doing, by kind; the rest generate.
var workingStates = map[string]string{"stt": "transcribing", "decision": "deciding"}

// Requests through the switch tell work from rest at once; the engine's own
// slots also see what came to its port directly, and when it is still
// reading the prompt.
func (i runningInstance) state() string {
	switch {
	case !i.Running:
		return "exited"
	case !i.Ready:
		return "loading"
	case i.Stats.SlotsPrefill > 0 && i.Stats.SlotsPrefill == i.Stats.SlotsBusy:
		return "processing prompt"
	case i.Active > 0 || i.Stats.SlotsBusy > 0:
		if s, ok := workingStates[i.Kind]; ok {
			return s
		}
		return "generating"
	}
	return "idle"
}

// The state, and for a model at work how much of it there is.
func (i runningInstance) stateDetail() string {
	s := i.state()
	switch {
	case s == "idle" || !i.Running || !i.Ready:
	case i.Parallel > 1 && i.Stats.SlotsBusy > 0:
		s += fmt.Sprintf(", %d of %d slots", i.Stats.SlotsBusy, i.Parallel)
	case i.Active > 1:
		s += fmt.Sprintf(", %d requests", i.Active)
	}
	return s
}

func (i runningInstance) gpu() string {
	if i.Devices == "" {
		return "default"
	}
	return i.Devices
}

func (i runningInstance) context() string {
	switch {
	case i.Ctx <= 0:
		return "-"
	case i.Parallel > 1:
		return fmt.Sprintf("%d x %d", i.Parallel, i.Ctx)
	}
	return strconv.Itoa(i.Ctx)
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
		port := "-"
		if i.Port > 0 {
			port = strconv.Itoa(i.Port)
		}
		mem := i.Stats.MemModel + i.Stats.MemContext + i.Stats.MemCompute
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			name, i.Kind, i.state(), formatMemory(mem), i.gpu(), i.context(), port)
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
		fmt.Fprintln(fs.Output(), "Usage: llmsw ps [MODEL...]\n\nModels that are up: their state, memory, cards, context window and port.\n"+
			"Given names, as ps names them, what the dashboard shows of each.")
	}
	fs.Parse(args)
	var err error
	if fs.NArg() > 0 {
		err = printDetails(serverURL(), fs.Args(), os.Stdout)
	} else {
		err = printRunning(serverURL(), os.Stdout)
	}
	if err != nil {
		fail(err)
	}
}

// What the dashboard shows of each model named: its settings, the counters
// for its kind and where its memory went.
func printDetails(server string, names []string, w io.Writer) error {
	instances, err := namedInstances(server, names)
	if err != nil {
		return err
	}
	for n, i := range instances {
		if n > 0 {
			fmt.Fprintln(w)
		}
		printInstance(w, names[n], i)
	}
	return nil
}

func printInstance(w io.Writer, name string, i runningInstance) {
	line := func(label, value string) { fmt.Fprintf(w, "  %-19s%s\n", label+":", value) }
	// "-" for a figure not there yet, as the dashboard has it.
	figure := func(ok bool, format string, a ...any) string {
		if !ok {
			return "-"
		}
		return fmt.Sprintf(format, a...)
	}
	onOff := func(b bool) string {
		if b {
			return "on"
		}
		return "off"
	}
	s := i.Stats

	fmt.Fprintln(w, name)
	state := i.stateDetail()
	if i.Running && i.Port > 0 {
		state += fmt.Sprintf(", port %d, pid %d", i.Port, i.PID)
	}
	line("state", state)
	kind := i.Kind
	if i.Quantization != "" {
		kind += ", " + i.Quantization
	}
	line("type", kind)
	line("gpu", i.gpu())

	switch i.Kind {
	case "llm":
		line("context", i.context())
		line("preserve thinking", onOff(i.PreserveThink))
		line("ctx usage", fmt.Sprintf("%d / %d", s.CtxUsed, i.Ctx))
		line("prompt speed", figure(s.PromptSpeed > 0, "%.2f tokens/s", s.PromptSpeed))
		line("gen speed", figure(s.GenSpeed > 0, "%.2f tokens/s", s.GenSpeed))
		line("total tokens", strconv.Itoa(s.TotalTokens))
	case "decision":
		line("context", i.context())
		line("decision", figure(s.DecisionMs > 0, "%.0f ms", s.DecisionMs))
		line("prompt", figure(s.DecisionTokens > 0, "%d tokens", s.DecisionTokens))
		line("requests", strconv.Itoa(s.DecisionCount))
	case "sd":
		line("size", figure(i.Res != "", "%s", i.Res))
		line("hires", onOff(i.Hires))
		line("step time", figure(s.SdSpeed > 0, "%.2f s/it", s.SdSpeed))
		line("last image", figure(s.SdLastTime > 0, "%.2f s", s.SdLastTime))
		progress := figure(s.SdSteps > 0, "%d / %d", s.SdStep, s.SdSteps)
		if s.SdSteps > 0 && s.SdSize != "" {
			progress += " · " + s.SdSize
		}
		line("progress", progress)
		line("images", strconv.Itoa(s.SdImages))
		line("params memory", formatMemory(s.MemModel))
		line("compute buffer", formatMemory(s.MemCompute))
	case "tts":
		line("max frames", figure(i.Frames > 0, "%d", i.Frames))
		line("speed", figure(s.TtsRtf > 0, "%.1fx realtime", 1/s.TtsRtf))
		line("last clip", figure(s.TtsAudio > 0, "%.1f s / %.1f s", s.TtsAudio, s.TtsLastTime))
		line("frames", figure(s.TtsFrames > 0, "%d", s.TtsFrames))
		line("clips", strconv.Itoa(s.TtsClips))
	}

	if i.Kind == "llm" || i.Kind == "decision" {
		line("model memory", formatMemory(s.MemModel))
		line("context memory", formatMemory(s.MemContext))
	}
	if i.Kind == "llm" || i.Kind == "decision" || i.Kind == "sd" {
		for _, d := range deviceMemory(s.MemDevices) {
			line(d[0], d[1])
		}
	}
}

// Memory by where it landed, rolled up as the dashboard does it: each card on
// its own, the host's buffers together as RAM, after them.
func deviceMemory(devices map[string]MemSlot) [][2]string {
	groups := map[string]MemSlot{}
	for name, m := range devices {
		if up := strings.ToUpper(name); strings.HasPrefix(up, "CPU") || strings.HasSuffix(up, "_HOST") {
			name = "RAM"
		}
		g := groups[name]
		g.Model, g.Context, g.Compute = g.Model+m.Model, g.Context+m.Context, g.Compute+m.Compute
		groups[name] = g
	}
	names := make([]string, 0, len(groups))
	for name := range groups {
		names = append(names, name)
	}
	sort.Slice(names, func(a, b int) bool {
		if (names[a] == "RAM") != (names[b] == "RAM") {
			return names[b] == "RAM"
		}
		return names[a] < names[b]
	})

	var out [][2]string
	for _, name := range names {
		g := groups[name]
		var parts []string
		for _, p := range []struct {
			what string
			mib  float64
		}{{"model", g.Model}, {"context", g.Context}, {"compute", g.Compute}} {
			if p.mib > 0 {
				parts = append(parts, p.what+" "+formatMemory(p.mib))
			}
		}
		text := formatMemory(g.Model + g.Context + g.Compute)
		if len(parts) > 1 {
			text += " (" + strings.Join(parts, ", ") + ")"
		}
		out = append(out, [2]string{name, text})
	}
	return out
}

func stopFlags(all *bool) *flag.FlagSet {
	fs := flag.NewFlagSet("stop", flag.ExitOnError)
	fs.BoolVar(all, "all", false, "unload every model that is up")
	fs.Usage = func() {
		fmt.Fprint(fs.Output(), "Usage: llmsw stop MODEL...\n       llmsw stop --all\n\n"+
			"Unload models, named as ps names them.\n\nFlags:\n")
		printFlags(fs, nil)
	}
	return fs
}

func stopCommand(args []string) {
	all := new(bool)
	fs := stopFlags(all)
	fs.Parse(args)
	// Neither would be a stop-all nobody asked for; both, a contradiction.
	if *all == (fs.NArg() > 0) {
		fs.Usage()
		os.Exit(2)
	}
	if err := stopModels(serverURL(), fs.Args(), *all, os.Stdout); err != nil {
		fail(err)
	}
}

// Every name is checked before anything stops, so a typo in the list does not
// leave half of it unloaded.
func stopModels(server string, names []string, all bool, out io.Writer) error {
	stop := func(query string) (int, error) {
		resp, err := post(context.Background(), server, "/api/stop"+query, "application/json", nil)
		if err != nil {
			return 0, err
		}
		defer resp.Body.Close()
		var res struct {
			Stopped int `json:"stopped"`
		}
		err = json.NewDecoder(resp.Body).Decode(&res)
		return res.Stopped, err
	}

	if all {
		n, err := stop("")
		if err != nil {
			return err
		}
		fmt.Fprintf(out, "Stopped %d model(s)\n", n)
		return nil
	}

	instances, err := namedInstances(server, names)
	if err != nil {
		return err
	}
	for n, i := range instances {
		if _, err := stop("?model=" + url.QueryEscape(i.Model)); err != nil {
			return err
		}
		fmt.Fprintf(out, "Stopped %s\n", names[n])
	}
	return nil
}

// The instances behind the models named, as ps names them. Every name is
// checked before any is used, so a typo in the list does nothing half-way.
func namedInstances(server string, names []string) ([]runningInstance, error) {
	models, err := fetchModels(server)
	if err != nil {
		return nil, err
	}
	instances, err := fetchInstances(server)
	if err != nil {
		return nil, err
	}
	up := map[string]runningInstance{}
	for _, i := range instances {
		up[instanceKey(i.Model, i.Quantization)] = i
	}
	byName := map[string]listedModel{}
	for _, m := range models {
		byName[m.Name] = m
	}
	var out []runningInstance
	for _, name := range names {
		m, ok := byName[name]
		if !ok {
			return nil, fmt.Errorf("Model %s not found; `llmsw list` shows the ones there are", name)
		}
		i, ok := up[instanceKey(m.Details.ParentModel, m.Details.QuantizationLevel)]
		if !ok {
			return nil, fmt.Errorf("%s is not running; `llmsw ps` shows what is", name)
		}
		out = append(out, i)
	}
	return out, nil
}
