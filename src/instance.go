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
	// llm: slots at work at the last poll, and how many are still reading the prompt.
	SlotsBusy    int `json:"slots_busy"`
	SlotsPrefill int `json:"slots_prefill"`
}

func newStats() Stats {
	return Stats{MemDevices: map[string]MemSlot{}}
}

const maxLogLines = 2000

// One launched backend: process, port, counters, log; fields above mu never change after start.
type Instance struct {
	ModelKey      string
	Quant         string
	Kind          string
	Port          int
	Ctx           int
	Parallel      int
	Devices       string
	Frames        int
	Res           string
	Hires         bool
	PreserveThink bool
	Command       string

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

// Marks a request at work until end is called; the last one out clears the stale slot counts.
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
		"command":        i.Command,
		"stats":          stats,
	}
}

type ServiceState struct {
	mu sync.Mutex
	// Running backends by model, one per model, in start order.
	instances map[string]*Instance
	order     []string
	// On by default; off, starting anything stops everything else.
	multi      bool
	defaultCtx int
	// Survives sd loads and stops, so a round trip keeps the user's choice.
	selectedCtx int
	// Whether the next llm load keeps previous reasoning; sticky like selectedCtx.
	selectedPreserveThink bool
	// Slots per model; sticky like selectedCtx. The context is per slot, so more slots cost memory.
	defaultParallel  int
	selectedParallel int
	// Cards a model may use, per kind, as the backend names them ("Vulkan0"); empty means all.
	selectedDevices map[string]string
	// Frame cap for tts, apart from ctx: 2048 frames at 12.5/s is about 164 s.
	defaultFrames  int
	selectedFrames int
	// sd output size as "WxH"; sticky like selectedFrames.
	defaultRes  string
	selectedRes string
	// Second sd pass. Sticky like selectedRes.
	selectedHires bool
	// Fixed ports by model; a model missing here gets a free port each start.
	selectedPorts map[string]int
	host          string
	cfg           *ConfigManager
}

var state = &ServiceState{
	instances:       map[string]*Instance{},
	multi:           true,
	defaultCtx:      4096,
	defaultParallel: 1,
	selectedDevices: map[string]string{},
	selectedPorts:   map[string]int{},
	defaultFrames:   2048,
	defaultRes:      "1024x1024",
	host:            "0.0.0.0",
}

// Starts, stops and resets run one at a time; apart from state.mu so status polls never wait.
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
	if err := killTree(inst.cmd.Process, false); err != nil {
		return
	}
	select {
	case <-inst.done:
	case <-time.After(5 * time.Second):
		killTree(inst.cmd.Process, true)
	}
}

// Stop one instance; the one stopped, nil when none was up. Caller holds opMu.
func stopInstance(modelKey string) *Instance {
	state.mu.Lock()
	inst := state.remove(modelKey)
	state.mu.Unlock()
	if inst == nil {
		return nil
	}
	fmt.Printf("[Service] Stopping %s...\n", inst.label())
	terminate(inst)
	inst.mu.Lock()
	inst.ready = false
	inst.mu.Unlock()
	fmt.Printf("[Service] %s stopped.\n", inst.label())
	return inst
}

// Stop every instance; the ones stopped. Caller holds opMu.
func stopAll() []*Instance {
	state.mu.Lock()
	keys := append([]string{}, state.order...)
	state.mu.Unlock()
	var stopped []*Instance
	for _, k := range keys {
		if inst := stopInstance(k); inst != nil {
			stopped = append(stopped, inst)
		}
	}
	return stopped
}

// An edited profile no longer matches its running process, so a config change stops every model.
func onConfigChange() {
	fmt.Println("[Service] Config change detected. Stopping any running model...")
	opMu.Lock()
	defer opMu.Unlock()
	stopAll()
}

// A fixed port must be free, or held by a model this start stops: itself, or any with othersStay false.
func checkPort(modelKey string, port int, othersStay bool) error {
	state.mu.Lock()
	var holder *Instance
	for _, i := range state.list() {
		if i.Port == port && i.alive() {
			holder = i
		}
	}
	state.mu.Unlock()
	switch {
	case holder == nil:
		l, err := net.Listen("tcp4", ":"+strconv.Itoa(port))
		if err != nil {
			return badValue(fmt.Sprintf("Port %d is in use", port))
		}
		l.Close()
	case holder.ModelKey != modelKey && othersStay:
		return badValue(fmt.Sprintf("Port %d is taken by %s", port, holder.label()))
	}
	return nil
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

// A setting the caller got wrong, answered with 404; anything else is a 500.
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
	Port          *int    `json:"port"`
}

type StartResult struct {
	Port    int    `json:"port"`
	Command string `json:"command"`
}

// Goes straight into a shell command line, so nothing but digits and one "x".
var reRes = regexp.MustCompile(`^(\d{2,5})x(\d{2,5})$`)

// ${HIRES:<flags>}: the flags verbatim with Hires Fix on, nothing with it off.
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
	// -c is the whole KV pool, shared equally by the slots, so it is the per-slot window times N.
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
	// The draft model follows --device, so llm needs no second flag.
	cmd = replaceSubmatchFunc(reDevices, cmd, func(g []string) string {
		return deviceFlags(v.Devices, v.Kind, g[1])
	})
	return cmd
}

func orStr(p *string, def string) string {
	if p == nil {
		return def
	}
	return *p
}

// Start a model, replacing what the eviction rule says. Caller holds opMu.
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

	actualQuant := orStr(req.Quantization, "")
	if actualQuant == "" {
		actualQuant = defaultQuant(modelConf)
	}
	cmdTemplate, ok := modelConf.Get(actualQuant)
	if !ok {
		return StartResult{}, badValue(fmt.Sprintf(
			"Quantization %s not found for model %s", actualQuant, modelKey))
	}

	state.mu.Lock()
	selCtx, defCtx := state.selectedCtx, state.defaultCtx
	selFrames, defFrames := state.selectedFrames, state.defaultFrames
	selRes, defRes := state.selectedRes, state.defaultRes
	selHires := state.selectedHires
	selPreserve := state.selectedPreserveThink
	selParallel, defParallel := state.selectedParallel, state.defaultParallel
	selDevices := state.selectedDevices[kind]
	selPort := state.selectedPorts[modelKey]
	multi := state.multi
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

	// Same precedence as ctx; checked against the backend's cards, since it goes into a shell command.
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

	// -1 asks for a free port each start; a fixed one stays with the model.
	fixedPort := selPort
	if req.Port != nil {
		if p := *req.Port; p != -1 && (p < 1 || p > 65535) {
			return StartResult{}, badValue(fmt.Sprintf("Port must be -1 or 1..65535, got %d", p))
		}
		fixedPort = max(*req.Port, 0)
	}
	port := fixedPort
	if port > 0 {
		if err := checkPort(modelKey, port, multi || replaceOnly); err != nil {
			return StartResult{}, err
		}
	} else {
		var err error
		if port, err = findFreePort(); err != nil {
			return StartResult{}, err
		}
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

	// The memory panel needs llama.cpp's buffer sizes, which only trace level (4) prints.
	childEnv := os.Environ()
	if kind == "llm" || kind == "decision" {
		childEnv = append(childEnv, "LLAMA_ARG_LOG_VERBOSITY=4")
	}

	// Multi-model off: a start clears the board; on: only the same model is replaced.
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
		Command:  cmdStr,
		stats:    newStats(),
	}
	// Only llm and decision models have a context window.
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
	if fixedPort > 0 {
		state.selectedPorts[modelKey] = fixedPort
	} else {
		delete(state.selectedPorts, modelKey)
	}
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
	cmd := shellCommand(cmdStr)
	cmd.Stdout = pw
	cmd.Stderr = pw
	cmd.Env = childEnv
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
