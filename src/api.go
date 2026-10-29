package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// An error the API answers with its own status, as {"detail": ...}.
type httpErr struct {
	status int
	detail string
}

func (e *httpErr) Error() string { return e.detail }

func newErr(status int, format string, a ...any) *httpErr {
	return &httpErr{status, fmt.Sprintf(format, a...)}
}

// JSON as the dashboard reads it: commands keep their "&&" rather than &.
func marshalJSON(v any) ([]byte, error) {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(b.Bytes(), []byte("\n")), nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	body, err := marshalJSON(v)
	if err != nil {
		status, body = http.StatusInternalServerError, []byte(`{"detail":"Internal Server Error"}`)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(body)
}

func writeError(w http.ResponseWriter, err error) {
	var he *httpErr
	if errors.As(err, &he) {
		writeJSON(w, he.status, map[string]string{"detail": he.detail})
		return
	}
	writeJSON(w, http.StatusInternalServerError, map[string]string{"detail": err.Error()})
}

func decodeRequest(r *http.Request, v any) error {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		return newErr(http.StatusUnprocessableEntity, "Invalid request body: %v", err)
	}
	return nil
}

func handleConfig(w http.ResponseWriter, r *http.Request) {
	cfg := state.cfg.get()

	grouped := OrderedMap[*Section]{}
	for _, name := range sections {
		grouped.Set(name, cfg.Sections[name])
	}
	kinds := OrderedMap[string]{}
	for _, key := range cfg.Models.Keys {
		kinds.Set(key, cfg.Kinds[key])
	}

	state.mu.Lock()
	defer state.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{
		// Flat map kept for older clients; "sections" is the grouped view.
		"models":           cfg.Models,
		"sections":         grouped,
		"kinds":            kinds,
		"default_ctx":      state.defaultCtx,
		"default_parallel": state.defaultParallel,
		"default_frames":   state.defaultFrames,
		"default_res":      state.defaultRes,
	})
}

type modelMeta struct {
	VocabType int   `json:"vocab_type"`
	NVocab    int   `json:"n_vocab"` // Dummy
	NCtx      int   `json:"n_ctx"`
	NCtxTrain int   `json:"n_ctx_train"`
	NEmbd     int   `json:"n_embd"`   // Dummy
	NParams   int64 `json:"n_params"` // Dummy
	Size      int64 `json:"size"`     // Dummy
}

type openaiModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
	// Not in the OpenAI schema; LM Studio adds it too, and clients ignore it.
	Type string    `json:"type"`
	Meta modelMeta `json:"meta"`
}

type customModelDetails struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

type customModel struct {
	Name         string             `json:"name"`
	Model        string             `json:"model"`
	Type         string             `json:"type"`
	Kind         string             `json:"kind"`
	ModifiedAt   string             `json:"modified_at"`
	Size         string             `json:"size"`
	Digest       string             `json:"digest"`
	Tags         []string           `json:"tags"`
	Capabilities []string           `json:"capabilities"`
	Details      customModelDetails `json:"details"`
}

// OpenAI-compatible list of every section's models, tagged by type.
func handleV1Models(w http.ResponseWriter, r *http.Request) {
	cfg := state.cfg.get()

	// The window the next llm load will actually get: same expression the load path uses.
	state.mu.Lock()
	effectiveCtx := state.selectedCtx
	if effectiveCtx == 0 {
		effectiveCtx = state.defaultCtx
	}
	state.mu.Unlock()

	listOpenAI := []openaiModel{}
	listCustom := []customModel{}

	for _, modelKey := range cfg.Models.Keys {
		info := cfg.Models.Values[modelKey]
		kind := cfg.Kinds[modelKey]
		if kind == "" {
			kind = "llm"
		}
		quants := info.Keys
		if isLegacy(info) {
			quants = []string{""}
		}

		// A context window is an llm notion: sd and tts have none.
		nCtx := 0
		if kind == "llm" || kind == "decision" {
			nCtx = effectiveCtx
		}

		for i, quant := range quants {
			cmd := info.Values[quant]
			if isLegacy(info) {
				cmd = info.Values["cmd"]
			}
			// The first-listed quant is the one published under the bare model id.
			id := modelKey
			if i > 0 {
				id = modelKey + "-" + quant
			}

			capabilities, ok := kindCapabilities[kind]
			if !ok {
				capabilities = []string{"completion"}
			}
			capabilities = append([]string{}, capabilities...)
			if strings.Contains(cmd, "mmproj") {
				capabilities = append(capabilities, "multimodal")
			}

			listOpenAI = append(listOpenAI, openaiModel{
				ID:      id,
				Object:  "model",
				Created: 1677619200,
				OwnedBy: "llamacpp",
				Type:    kind,
				Meta: modelMeta{
					VocabType: 1, NVocab: 32000, NCtx: nCtx, NCtxTrain: nCtx,
					NEmbd: 4096, NParams: 7000000000, Size: 4000000000,
				},
			})
			listCustom = append(listCustom, customModel{
				Name:         id,
				Model:        id,
				Type:         "model",
				Kind:         kind,
				Tags:         []string{},
				Capabilities: capabilities,
				Details: customModelDetails{
					ParentModel:       modelKey,
					Format:            "gguf",
					Family:            kind,
					Families:          []string{},
					QuantizationLevel: quant,
				},
			})
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list", "data": listOpenAI, "models": listCustom,
	})
}

// Every running instance plus the sticky settings; ?model= picks the one the flat fields describe.
func handleStatus(w http.ResponseWriter, r *http.Request) {
	model := r.URL.Query().Get("model")

	state.mu.Lock()
	defer state.mu.Unlock()

	instances := []map[string]any{}
	for _, inst := range state.list() {
		instances = append(instances, inst.toDict())
	}
	var picked map[string]any
	if model != "" {
		if inst := state.instances[model]; inst != nil {
			picked = inst.toDict()
		}
	} else if len(instances) == 1 {
		picked = instances[0]
	}

	resp := map[string]any{
		"running":      false,
		"ready":        false,
		"model":        nil,
		"quantization": nil,
		"kind":         nil,
		"ctx":          0,
		"parallel":     0,
		"devices":      "",
		"frames":       0,
		"res":          "",
		"hires":        false,
		"port":         nil,
		"pid":          nil,
		"stats":        newStats(),
	}
	if picked != nil {
		resp = map[string]any{}
		for k, v := range picked {
			resp[k] = v
		}
	}

	orDefault := func(v, def int) int {
		if v != 0 {
			return v
		}
		return def
	}
	selectedRes := state.selectedRes
	if selectedRes == "" {
		selectedRes = state.defaultRes
	}
	selectedDevices := map[string]string{}
	for k, v := range state.selectedDevices {
		selectedDevices[k] = v
	}

	resp["instances"] = instances
	resp["multi"] = state.multi
	resp["host"] = state.host
	resp["selected_ctx"] = orDefault(state.selectedCtx, state.defaultCtx)
	resp["selected_parallel"] = orDefault(state.selectedParallel, state.defaultParallel)
	// A map, not one value: the picker shows the choice for the open tab.
	resp["selected_devices"] = selectedDevices
	resp["selected_preserve_think"] = state.selectedPreserveThink
	resp["selected_frames"] = orDefault(state.selectedFrames, state.defaultFrames)
	resp["selected_res"] = selectedRes
	resp["selected_hires"] = state.selectedHires
	writeJSON(w, http.StatusOK, resp)
}

// Hold several models at once, or evict everything on each start.
func handleMulti(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Multi *bool `json:"multi"`
	}
	if err := decodeRequest(r, &req); err != nil {
		writeError(w, err)
		return
	}
	if req.Multi == nil {
		writeError(w, newErr(http.StatusUnprocessableEntity, "Field required: multi"))
		return
	}
	state.mu.Lock()
	// Turning it off leaves whatever is up; the next start clears the board.
	state.multi = *req.Multi
	multi := state.multi
	state.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]bool{"multi": multi})
}

// Offload devices the backend reports, for the dashboard's picker.
func handleDevices(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"devices": listDevices(true)})
}

// The instance whose log is being asked for.
func logInstance(model string) *Instance {
	state.mu.Lock()
	defer state.mu.Unlock()
	if model != "" {
		return state.instances[model]
	}
	if len(state.order) == 1 {
		return state.instances[state.order[0]]
	}
	return nil
}

func handleLogs(w http.ResponseWriter, r *http.Request) {
	// Kept after the process exits: a crash is read from the log it left behind.
	lines := []string{}
	if inst := logInstance(r.URL.Query().Get("model")); inst != nil {
		lines = inst.logLines()
	}
	writeJSON(w, http.StatusOK, lines)
}

// Which model produced the log being served, and of what kind.
func handleLogsOwner(w http.ResponseWriter, r *http.Request) {
	inst := logInstance(r.URL.Query().Get("model"))
	if inst == nil {
		writeJSON(w, http.StatusOK, map[string]any{"model": nil, "kind": nil})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"model": inst.label(), "kind": inst.Kind})
}

func handleLogsClear(w http.ResponseWriter, r *http.Request) {
	if inst := logInstance(r.URL.Query().Get("model")); inst != nil {
		inst.clearLogs()
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "cleared"})
}

// Stop one model, or everything when no model is named.
func handleStop(w http.ResponseWriter, r *http.Request) {
	model := r.URL.Query().Get("model")
	opMu.Lock()
	var stopped []*Instance
	if model != "" {
		if inst := stopInstance(model); inst != nil {
			stopped = append(stopped, inst)
		}
	} else {
		stopped = stopAll()
	}
	opMu.Unlock()
	// Which ones, so that stopping them all can say what it stopped.
	models := []map[string]any{}
	for _, inst := range stopped {
		models = append(models, map[string]any{
			"model": inst.ModelKey, "quantization": nullable(inst.Quant), "kind": inst.Kind, "label": inst.label(),
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "stopped", "stopped": len(stopped), "models": models})
}

// One section's remembered choices back to the defaults. Caller holds state.mu.
func resetSection(kind string) {
	delete(state.selectedDevices, kind)
	switch kind {
	case "llm":
		state.selectedCtx = 0
		state.selectedParallel = 0
		state.selectedPreserveThink = false
	case "sd":
		state.selectedRes = ""
		state.selectedHires = false
	case "tts":
		state.selectedFrames = 0
	}
}

// Settings back to defaults, restarting what they apply to: ?model=, ?kind= or everything.
func handleReset(w http.ResponseWriter, r *http.Request) {
	model := r.URL.Query().Get("model")
	kind := r.URL.Query().Get("kind")

	opMu.Lock()
	defer opMu.Unlock()

	type target struct{ key, quant string }
	var targets []target

	state.mu.Lock()
	switch {
	case model != "":
		modelKind := state.cfg.kind(model)
		if modelKind == "" {
			state.mu.Unlock()
			writeError(w, newErr(http.StatusNotFound, "Model %s not found", model))
			return
		}
		resetSection(modelKind)
		if inst := state.instances[model]; inst != nil && inst.alive() {
			targets = append(targets, target{inst.ModelKey, inst.Quant})
		}
	case kind != "":
		if !isSection(kind) {
			state.mu.Unlock()
			writeError(w, newErr(http.StatusNotFound, "Unknown section %s", kind))
			return
		}
		resetSection(kind)
	default:
		for _, name := range sections {
			resetSection(name)
		}
		for _, inst := range state.list() {
			if inst.alive() {
				targets = append(targets, target{inst.ModelKey, inst.Quant})
			}
		}
	}
	defCtx, defFrames, defRes, defParallel :=
		state.defaultCtx, state.defaultFrames, state.defaultRes, state.defaultParallel
	state.mu.Unlock()

	restarted := []string{}
	for _, t := range targets {
		key, quant := t.key, t.quant
		hires, preserve, devices := false, false, ""
		_, err := startModelServer(StartRequest{
			ModelKey:      &key,
			Quantization:  &quant,
			Ctx:           &defCtx,
			Frames:        &defFrames,
			Res:           &defRes,
			Hires:         &hires,
			PreserveThink: &preserve,
			Parallel:      &defParallel,
			Devices:       &devices,
		},
			// Reset All restarts every model, so the first restart must not evict the rest.
			model == "")
		if err != nil {
			writeError(w, newErr(http.StatusInternalServerError, "%s: %v", key, err))
			return
		}
		restarted = append(restarted, key)
	}

	writeJSON(w, http.StatusOK, map[string]any{"status": "reset", "restarted": restarted})
}

func handleStart(w http.ResponseWriter, r *http.Request) {
	var req StartRequest
	if err := decodeRequest(r, &req); err != nil {
		writeError(w, err)
		return
	}
	if req.ModelKey == nil {
		writeError(w, newErr(http.StatusUnprocessableEntity, "Field required: model_key"))
		return
	}

	opMu.Lock()
	res, err := startModelServer(req, false)
	opMu.Unlock()
	if err != nil {
		var bad badValue
		if errors.As(err, &bad) {
			writeError(w, newErr(http.StatusNotFound, "%s", bad))
		} else {
			writeError(w, newErr(http.StatusInternalServerError, "%v", err))
		}
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "started", "port": res.Port, "command": res.Command,
	})
}
