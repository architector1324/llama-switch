package main

import (
	"fmt"
	"net"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// --- Global State ---
type MemSlot struct {
	Model   float64 `json:"model"`
	Context float64 `json:"context"`
	Compute float64 `json:"compute"`
}

// Counters for one backend process. Per instance, never shared.
type Stats struct {
	CtxUsed     int     `json:"ctx_used"` // From 'stop processing: n_tokens = X'
	CtxLimit    int     `json:"ctx_limit"`
	TotalTokens int     `json:"total_tokens"` // Accumulated generation
	PromptSpeed float64 `json:"prompt_speed"`
	GenSpeed    float64 `json:"gen_speed"`
	// Parsed out of the backend's own load-time lines; zero until it loads.
	MemModel   float64            `json:"mem_model"`
	MemContext float64            `json:"mem_context"`
	MemCompute float64            `json:"mem_compute"`
	MemDevices map[string]MemSlot `json:"mem_devices"`
	// SdSpeed is seconds per step whichever unit sd.cpp printed.
	SdStep      int     `json:"sd_step"`
	SdSteps     int     `json:"sd_steps"`
	SdSpeed     float64 `json:"sd_speed"`
	SdLastTime  float64 `json:"sd_last_time"`
	SdImages    int     `json:"sd_images"`
	SdSize      string  `json:"sd_size"`
	TtsFrames   int     `json:"tts_frames"`
	TtsRtf      float64 `json:"tts_rtf"`
	TtsAudio    float64 `json:"tts_audio"`
	TtsLastTime float64 `json:"tts_last_time"`
	TtsClips    int     `json:"tts_clips"`
	// Only a decision model ever reports these.
	DecisionMs     float64 `json:"decision_ms,omitempty"`
	DecisionTokens int     `json:"decision_tokens,omitempty"`
	DecisionCount  int     `json:"decision_count,omitempty"`
	// llm: the slots at work at the last poll, and how many of them are still
	// reading the prompt.
	SlotsBusy    int `json:"slots_busy"`
	SlotsPrefill int `json:"slots_prefill"`
}

func newStats() Stats {
	return Stats{MemDevices: map[string]MemSlot{}}
}

const maxLogLines = 2000

// One launched backend: its process, its port, its counters, its log.
//
// Keeping all of it per instance is what lets an image model and a language
// model be up at the same time. Everything above mu is set before the process
// starts and never changes after.
type Instance struct {
	ModelKey      string
	Quant         string // "" for the old single-cmd format
	Kind          string
	Port          int
	Ctx           int
	Parallel      int
	Devices       string
	Frames        int
	Res           string
	Hires         bool
	PreserveThink bool

	cmd  *exec.Cmd
	done chan struct{} // closed once the process has exited

	mu     sync.Mutex
	ready  bool
	active int // requests through the proxy still at work on it
	stats  Stats
	logs   []string
}

func (i *Instance) label() string {
	if i.Quant != "" {
		return i.ModelKey + ":" + i.Quant
	}
	return i.ModelKey
}

func (i *Instance) alive() bool {
	if i.done == nil {
		return false
	}
	select {
	case <-i.done:
		return false
	default:
		return true
	}
}

func (i *Instance) isReady() bool {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.ready
}

// Marks a request at work on the engine until end is called. Once none is,
// the busy slots of the last poll are stale: the request that had them is
// done, and the next poll finds any other.
func (i *Instance) begin() (end func()) {
	i.mu.Lock()
	i.active++
	i.mu.Unlock()
	return func() {
		i.mu.Lock()
		if i.active--; i.active == 0 {
			i.stats.SlotsBusy, i.stats.SlotsPrefill = 0, 0
		}
		i.mu.Unlock()
	}
}

func (i *Instance) update(fn func(s *Stats)) {
	i.mu.Lock()
	fn(&i.stats)
	i.mu.Unlock()
}

func (i *Instance) appendLog(line string) {
	i.mu.Lock()
	i.logs = append(i.logs, line)
	if len(i.logs) > maxLogLines {
		i.logs = i.logs[len(i.logs)-maxLogLines:]
	}
	i.mu.Unlock()
}

func (i *Instance) logLines() []string {
	i.mu.Lock()
	defer i.mu.Unlock()
	return append([]string{}, i.logs...)
}

func (i *Instance) clearLogs() {
	i.mu.Lock()
	i.logs = nil
	i.mu.Unlock()
}

func nullable(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func (i *Instance) toDict() map[string]any {
	alive := i.alive()
	var port, pid any
	if alive {
		port, pid = i.Port, i.cmd.Process.Pid
	}
	i.mu.Lock()
	ready, active, stats := i.ready, i.active, i.stats
	i.mu.Unlock()
	return map[string]any{
		"model":          i.ModelKey,
		"quantization":   nullable(i.Quant),
		"kind":           i.Kind,
		"label":          i.label(),
		"port":           port,
		"pid":            pid,
		"running":        alive,
		"ready":          ready,
		"active":         active,
		"ctx":            i.Ctx,
		"parallel":       i.Parallel,
		"devices":        i.Devices,
		"frames":         i.Frames,
		"res":            i.Res,
		"hires":          i.Hires,
		"preserve_think": i.PreserveThink,
		"stats":          stats,
	}
}

type ServiceState struct {
	mu sync.Mutex
	// Every running backend, keyed by model, in start order. One entry per model,
	// so loading the same model again replaces it rather than doubling it.
	instances map[string]*Instance
	order     []string
	// On from the start. Off keeps the old behaviour: starting anything stops
	// everything else.
	multi      bool
	defaultCtx int
	// Survives sd loads and stops, so a round trip keeps the user's choice.
	selectedCtx int
	// Whether the next llm load pulls previous reasoning back into the prompt.
	// Sticky like selectedCtx: a stop or an sd load must not reset the choice.
	selectedPreserveThink bool
	// Server slots: how many clients may talk to one model at once. Sticky like
	// selectedCtx. The context window is per slot, so N slots ask the engine for
	// N times as much KV -- more slots cost memory, not context.
	defaultParallel  int
	selectedParallel int
	// Which offload devices a model may use, comma-separated as the backend
	// names them ("Vulkan0"); empty means every device, the old behaviour.
	// Sticky per kind, like selectedRes is to sd: a card picked for a
	// language model says nothing about where an image model should go.
	selectedDevices map[string]string
	// Frame cap for tts, apart from ctx: 2048 frames at 12.5/s is about 164 s.
	defaultFrames  int
	selectedFrames int
	// Output size for sd, as "WxH". Sticky like selectedFrames: a stop or an
	// llm load must not reset it.
	defaultRes  string
	selectedRes string
	// Second sd pass. Sticky like selectedRes.
	selectedHires bool
	host          string
	cfg           *ConfigManager
}

var state = &ServiceState{
	instances:       map[string]*Instance{},
	multi:           true,
	defaultCtx:      4096,
	defaultParallel: 1,
	selectedDevices: map[string]string{},
	defaultFrames:   2048,
	defaultRes:      "1024x1024",
	host:            "0.0.0.0",
}

// Starts, stops and resets run one at a time. A request that finds its model
// not loaded checks and starts under this, so two of them cannot both start it
// and have the second kill the first. Kept apart from state.mu so that status
// polls are not held up while a backend takes its time to exit.
var opMu sync.Mutex

// Caller holds state.mu.
func (s *ServiceState) list() []*Instance {
	out := make([]*Instance, 0, len(s.order))
	for _, k := range s.order {
		out = append(out, s.instances[k])
	}
	return out
}

// Caller holds state.mu.
func (s *ServiceState) ofKind(kind string) []*Instance {
	var out []*Instance
	for _, i := range s.list() {
		if i.Kind == kind {
			out = append(out, i)
		}
	}
	return out
}

func (s *ServiceState) instanceFor(modelKey string) *Instance {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.instances[modelKey]
}

// Caller holds state.mu.
func (s *ServiceState) add(inst *Instance) {
	s.instances[inst.ModelKey] = inst
	s.order = append(s.order, inst.ModelKey)
}

// Caller holds state.mu.
func (s *ServiceState) remove(modelKey string) *Instance {
	inst, ok := s.instances[modelKey]
	if !ok {
		return nil
	}
	delete(s.instances, modelKey)
	for i, k := range s.order {
		if k == modelKey {
			s.order = append(s.order[:i], s.order[i+1:]...)
			break
		}
	}
	return inst
}

// --- Process Control ---

// Kill the whole process group: profiles are shell lines that fork.
func terminate(inst *Instance) {
	if !inst.alive() {
		return
	}
	pgid := inst.cmd.Process.Pid // setsid made it the group leader
	if err := syscall.Kill(-pgid, syscall.SIGTERM); err != nil {
		return
	}
	select {
	case <-inst.done:
	case <-time.After(5 * time.Second):
		syscall.Kill(-pgid, syscall.SIGKILL)
	}
}

// Stop one instance. Caller holds opMu.
func stopInstance(modelKey string) bool {
	state.mu.Lock()
	inst := state.remove(modelKey)
	state.mu.Unlock()
	if inst == nil {
		return false
	}
	fmt.Printf("[Service] Stopping %s...\n", inst.label())
	terminate(inst)
	inst.mu.Lock()
	inst.ready = false
	inst.mu.Unlock()
	fmt.Printf("[Service] %s stopped.\n", inst.label())
	return true
}

// Stop every instance. Caller holds opMu.
func stopAll() int {
	state.mu.Lock()
	keys := append([]string{}, state.order...)
	state.mu.Unlock()
	stopped := 0
	for _, k := range keys {
		if stopInstance(k) {
			stopped++
		}
	}
	return stopped
}

// Stop running models when the config changes.
//
// A profile that was edited no longer describes the process that is up, and
// the command line cannot be changed under a running backend.
func onConfigChange() {
	fmt.Println("[Service] Config change detected. Stopping any running model...")
	opMu.Lock()
	defer opMu.Unlock()
	stopAll()
}

func findFreePort() (int, error) {
	l, err := net.Listen("tcp4", ":0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

// --- Start Logic ---

// A setting the caller got wrong: unknown model, bad size, unknown card. The API
// answers it with 404, as it always has; anything else is a 500.
type badValue string

func (e badValue) Error() string { return string(e) }

type StartRequest struct {
	ModelKey      *string `json:"model_key"`
	Quantization  *string `json:"quantization"`
	Ctx           *int    `json:"ctx"`
	Frames        *int    `json:"frames"`
	Res           *string `json:"res"`
	Hires         *bool   `json:"hires"`
	PreserveThink *bool   `json:"preserve_think"`
	Parallel      *int    `json:"parallel"`
	Devices       *string `json:"devices"`
}

type StartResult struct {
	Port    int    `json:"port"`
	Command string `json:"command"`
}

// Goes straight into a shell command line, so nothing but digits and one "x".
var reRes = regexp.MustCompile(`^(\d{2,5})x(\d{2,5})$`)

// ${HIRES:<flags>} keeps each model's own hires settings next to the model, because
// they differ: a denoise strength for the image models, a whole upscaler block for ltx.
// On, the flags go in verbatim; off, the whole placeholder disappears.
var reHires = regexp.MustCompile(`\$\{HIRES:([^}]*)\}`)

// re.sub with a function: fn gets the submatches, "" for a group that did not take part.
func replaceSubmatchFunc(re *regexp.Regexp, s string, fn func(groups []string) string) string {
	var b strings.Builder
	last := 0
	for _, idx := range re.FindAllStringSubmatchIndex(s, -1) {
		groups := make([]string, len(idx)/2)
		for g := range groups {
			if idx[2*g] >= 0 {
				groups[g] = s[idx[2*g]:idx[2*g+1]]
			}
		}
		b.WriteString(s[last:idx[0]])
		b.WriteString(fn(groups))
		last = idx[1]
	}
	b.WriteString(s[last:])
	return b.String()
}

// The values a profile's placeholders take for one start, already validated.
type profileValues struct {
	Kind, Host, Devices  string
	Port, Ctx, Parallel  int
	Frames               int
	Width, Height        string
	Hires, PreserveThink bool
}

// The command line a profile stands for, placeholders filled in.
func expandProfile(tmpl string, v profileValues) string {
	port := strconv.Itoa(v.Port)
	// -c is the whole KV pool and llama.cpp hands each slot an equal share of it, so
	// the window the user asked for is per slot and the pool has to be N times it.
	ctxTotal := strconv.Itoa(v.Ctx * v.Parallel)
	frames := strconv.Itoa(v.Frames)
	parallel := strconv.Itoa(v.Parallel)
	preserveFlag := "--no-reasoning-preserve"
	if v.PreserveThink {
		preserveFlag = "--reasoning-preserve"
	}
	hiresFlag := ""
	if v.Hires {
		hiresFlag = "--hires"
	}

	cmd := strings.ReplaceAll(tmpl, "${PORT}", port)
	cmd = strings.ReplaceAll(cmd, "${CTX}", ctxTotal)
	cmd = strings.ReplaceAll(cmd, "${FRAMES}", frames)
	cmd = replaceSubmatchFunc(reHires, cmd, func(g []string) string {
		if v.Hires {
			return g[1]
		}
		return ""
	})
	cmd = strings.ReplaceAll(cmd, "${HIRES}", hiresFlag)
	cmd = strings.ReplaceAll(cmd, "${WIDTH}", v.Width)
	cmd = strings.ReplaceAll(cmd, "${HEIGHT}", v.Height)
	cmd = strings.ReplaceAll(cmd, "${HOST}", v.Host)
	cmd = strings.ReplaceAll(cmd, "${PRESERVE_THINK}", preserveFlag)
	cmd = strings.ReplaceAll(cmd, "${PARALLEL}", parallel)
	// The draft model follows --device unless told otherwise, so llm needs no
	// second flag for it.
	cmd = replaceSubmatchFunc(reDevices, cmd, func(g []string) string {
		return deviceFlags(v.Devices, v.Kind, g[1])
	})
	// Bare forms, for configs written without the braces.
	cmd = strings.ReplaceAll(cmd, "$PORT", port)
	cmd = strings.ReplaceAll(cmd, "$CTX", ctxTotal)
	cmd = strings.ReplaceAll(cmd, "$FRAMES", frames)
	cmd = strings.ReplaceAll(cmd, "$WIDTH", v.Width)
	cmd = strings.ReplaceAll(cmd, "$HEIGHT", v.Height)
	cmd = strings.ReplaceAll(cmd, "$HOST", v.Host)
	cmd = strings.ReplaceAll(cmd, "$PRESERVE_THINK", preserveFlag)
	cmd = strings.ReplaceAll(cmd, "$PARALLEL", parallel)
	return cmd
}

func orStr(p *string, def string) string {
	if p == nil {
		return def
	}
	return *p
}

// Start a model, replacing whatever the eviction rule says it replaces. Caller
// holds opMu.
func startModelServer(req StartRequest, replaceOnly bool) (StartResult, error) {
	if state.cfg == nil {
		return StartResult{}, fmt.Errorf("Config not initialized")
	}

	modelKey := orStr(req.ModelKey, "")
	cfg := state.cfg.get()
	modelConf, ok := cfg.Models.Get(modelKey)
	if !ok {
		return StartResult{}, badValue("Model not found in config")
	}
	kind := cfg.Kinds[modelKey]
	if kind == "" {
		kind = "llm"
	}

	var cmdTemplate, actualQuant string
	if isLegacy(modelConf) {
		// Old single-cmd format, no quants
		cmdTemplate = modelConf.Values["cmd"]
	} else {
		quantization := orStr(req.Quantization, "")
		if quantization == "" {
			quantization = defaultQuant(modelConf)
		}
		tmpl, ok := modelConf.Get(quantization)
		if !ok {
			return StartResult{}, badValue(fmt.Sprintf(
				"Quantization %s not found for model %s", quantization, modelKey))
		}
		cmdTemplate = tmpl
		actualQuant = quantization
	}

	state.mu.Lock()
	selCtx, defCtx := state.selectedCtx, state.defaultCtx
	selFrames, defFrames := state.selectedFrames, state.defaultFrames
	selRes, defRes := state.selectedRes, state.defaultRes
	selHires := state.selectedHires
	selPreserve := state.selectedPreserveThink
	selParallel, defParallel := state.selectedParallel, state.defaultParallel
	selDevices := state.selectedDevices[kind]
	host := state.host
	state.mu.Unlock()

	firstNonZero := func(a, b int) int {
		if a != 0 {
			return a
		}
		return b
	}

	// An explicit request wins, then the last llm choice, then the startup default.
	ctx := firstNonZero(selCtx, defCtx)
	if req.Ctx != nil {
		ctx = *req.Ctx
	}

	// Deliberately not ${CTX}: an llm context window is far too big for a frame cap.
	frames := firstNonZero(selFrames, defFrames)
	if req.Frames != nil {
		frames = *req.Frames
	}

	// Same precedence as frames. Only profiles that spell ${WIDTH}/${HEIGHT} use it.
	res := selRes
	if res == "" {
		res = defRes
	}
	res = orStr(req.Res, res)
	mRes := reRes.FindStringSubmatch(strings.ToLower(strings.TrimSpace(res)))
	if mRes == nil {
		return StartResult{}, badValue(fmt.Sprintf(
			"Resolution must look like 1024x1024, got '%s'", res))
	}
	width, height := mRes[1], mRes[2]
	res = width + "x" + height

	hires := selHires
	if req.Hires != nil {
		hires = *req.Hires
	}

	// Same precedence as ctx: an explicit request, then the last llm choice.
	preserveThink := selPreserve
	if req.PreserveThink != nil {
		preserveThink = *req.PreserveThink
	}

	// Same precedence as ctx. Goes straight onto a shell command line.
	parallel := firstNonZero(selParallel, defParallel)
	if req.Parallel != nil {
		parallel = *req.Parallel
	}
	if parallel < 1 || parallel > 64 {
		return StartResult{}, badValue(fmt.Sprintf("Parallel slots must be 1..64, got %d", parallel))
	}

	// Same precedence as ctx. Whitelisted against what the backend reports, both
	// because a typo would fail obscurely at launch and because this string is
	// pasted into a shell command line.
	devices := strings.TrimSpace(orStr(req.Devices, selDevices))
	if devices != "" {
		known := map[string]bool{}
		for _, d := range listDevices(false) {
			known[d.ID] = true
		}
		var picked, unknown []string
		for _, p := range strings.Split(devices, ",") {
			if p = strings.TrimSpace(p); p != "" {
				picked = append(picked, p)
			}
		}
		// Entries are either a device or "role=device" (sd assigns per module).
		for _, p := range picked {
			_, dev, _ := strings.Cut(p, "=")
			if !known[strings.TrimSpace(dev)] && !known[p] {
				unknown = append(unknown, p)
			}
		}
		if len(unknown) > 0 {
			names := make([]string, 0, len(known))
			for k := range known {
				names = append(names, k)
			}
			sort.Strings(names)
			available := strings.Join(names, ", ")
			if available == "" {
				available = "none"
			}
			return StartResult{}, badValue(fmt.Sprintf(
				"Unknown device(s): %s; available: %s", strings.Join(unknown, ", "), available))
		}
		devices = strings.Join(picked, ",")
	}

	port, err := findFreePort()
	if err != nil {
		return StartResult{}, err
	}

	cmdStr := expandProfile(cmdTemplate, profileValues{
		Kind:          kind,
		Host:          host,
		Port:          port,
		Ctx:           ctx,
		Parallel:      parallel,
		Frames:        frames,
		Width:         width,
		Height:        height,
		Hires:         hires,
		PreserveThink: preserveThink,
		Devices:       devices,
	})

	fmt.Printf("Starting model %s on %s:%d with command: %s\n", modelKey, host, port, cmdStr)

	// The memory panel reads llama.cpp's own buffer-size lines, and those are
	// LLAMA_LOG_INFO, which maps to trace level -- invisible at the default 3. Set
	// here rather than in the profiles: the panel depends on it, so it must not be
	// something a profile can quietly drop. Other backends ignore the variable.
	childEnv := os.Environ()
	if kind == "llm" || kind == "decision" {
		childEnv = append(childEnv, "LLAMA_ARG_LOG_VERBOSITY=4")
	}

	// Off: starting anything clears the board, which is how it always worked.
	// On: only the same model is replaced, so the rest keep running.
	state.mu.Lock()
	multi := state.multi
	state.mu.Unlock()
	if multi || replaceOnly {
		stopInstance(modelKey)
	} else {
		stopAll()
	}

	inst := &Instance{
		ModelKey: modelKey,
		Quant:    actualQuant,
		Kind:     kind,
		Port:     port,
		Devices:  devices,
		stats:    newStats(),
	}
	// sd-server has no context window; reporting one skews the UI gauge.
	// Decision models do take -c, so they report it too; only the sticky
	// selectedCtx stays an llm notion.
	if kind == "llm" || kind == "decision" {
		inst.Ctx = ctx
		// Only llama-server takes --parallel; sd and tts have no slots.
		inst.Parallel = parallel
	}
	if kind == "tts" {
		inst.Frames = frames
	}
	if kind == "sd" {
		inst.Res = res
		inst.Hires = hires
	}
	if kind == "llm" {
		inst.PreserveThink = preserveThink
	}

	state.mu.Lock()
	state.selectedDevices[kind] = devices
	if kind == "llm" {
		state.selectedCtx = ctx
		state.selectedPreserveThink = preserveThink
		state.selectedParallel = parallel
	}
	if kind == "tts" {
		state.selectedFrames = frames
	}
	if kind == "sd" {
		state.selectedRes = res
		state.selectedHires = hires
	}
	state.mu.Unlock()

	// One pipe for both streams, as the backends interleave them.
	pr, pw, err := os.Pipe()
	if err != nil {
		return StartResult{}, err
	}
	cmd := exec.Command("/bin/sh", "-c", cmdStr)
	cmd.Stdout = pw
	cmd.Stderr = pw
	cmd.Env = childEnv
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		pr.Close()
		pw.Close()
		return StartResult{}, err
	}
	pw.Close()

	inst.cmd = cmd
	inst.done = make(chan struct{})
	go func() {
		cmd.Wait()
		close(inst.done)
	}()

	state.mu.Lock()
	state.add(inst)
	state.mu.Unlock()

	go logReader(pr, inst)
	if kind == "llm" {
		go slotsPoller(inst)
	}

	return StartResult{Port: port, Command: cmdStr}, nil
}
