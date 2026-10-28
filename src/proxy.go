package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

// --- Model routing ---

// Model id to (model_key, quant) within one section.
func lookupModel(requested, kind string) (string, string, bool) {
	sec := state.cfg.models(kind)
	if conf, ok := sec.Get(requested); ok {
		return requested, defaultQuant(conf), true
	}
	for _, key := range sec.Keys {
		info := sec.Values[key]
		if isLegacy(info) {
			continue // Old format doesn't support suffixes
		}
		def := defaultQuant(info)
		for _, quant := range info.Keys {
			if quant == def {
				continue // Already covered by the bare model id above
			}
			if requested == key+"-"+quant {
				return key, quant, true
			}
		}
	}
	return "", "", false
}

// Model id to (model_key, quant); a bare id means the first-listed quant.
func resolveModel(requested, kind string) (string, string, error) {
	if key, quant, ok := lookupModel(requested, kind); ok {
		return key, quant, nil
	}
	// The list is served whole, so a chat client can offer an image model and the
	// user can pick it. Calling that "not found" when it plainly is in the list
	// would be a lie; name the endpoint that does serve it.
	for _, other := range sections {
		if other == kind {
			continue
		}
		if _, _, ok := lookupModel(requested, other); ok {
			endpoint := kindEndpoint[other]
			if endpoint == "" {
				endpoint = "?"
			}
			return "", "", newErr(http.StatusConflict,
				"Model %s is a '%s' model, not '%s'; it is served at %s",
				requested, other, kind, endpoint)
		}
	}
	return "", "", newErr(http.StatusNotFound, "Model %s not found", requested)
}

func serving(modelKey, quant string) *Instance {
	if inst := state.instanceFor(modelKey); inst != nil && inst.alive() && inst.Quant == quant {
		return inst
	}
	return nil
}

// The instance serving this model, starting it if needed.
func autoloadModel(modelKey, quant string) (*Instance, error) {
	if inst := serving(modelKey, quant); inst != nil {
		return inst, nil
	}

	opMu.Lock()
	defer opMu.Unlock()
	// Checked again under the lock: a request that came in alongside this one
	// may have started it meanwhile, and starting it twice kills the first.
	if inst := serving(modelKey, quant); inst != nil {
		return inst, nil
	}

	shown := quant
	if shown == "" {
		shown = "None"
	}
	fmt.Printf("[Proxy] Auto-loading model: %s (quant: %s)\n", modelKey, shown)
	if _, err := startModelServer(StartRequest{ModelKey: &modelKey, Quantization: &quant}, false); err != nil {
		var bad badValue
		if errors.As(err, &bad) {
			return nil, newErr(http.StatusNotFound, "%s", bad)
		}
		return nil, newErr(http.StatusInternalServerError, "%v", err)
	}

	inst := state.instanceFor(modelKey)
	if inst == nil {
		return nil, newErr(http.StatusInternalServerError, "Model failed to start")
	}
	return inst, nil
}

// Wait for the backend to announce itself, or fail with a clear timeout.
func awaitReady(ctx context.Context, inst *Instance) error {
	for i := 0; i < 60; i++ {
		if inst.isReady() {
			return nil
		}
		if !inst.alive() {
			return newErr(http.StatusInternalServerError,
				"%s exited while starting; see its log", inst.label())
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Second):
		}
	}
	return newErr(http.StatusGatewayTimeout, "%s failed to load within timeout", inst.label())
}

// --- Forwarding ---

var upstream = &http.Client{
	Transport: directTransport(),
	// Answers go back to the client as they are, redirects included.
	CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
}

// Not forwarded. The body is resent whole with its own length, so a chunked
// upload's Transfer-Encoding must not travel with it: both headers together make
// llama-server answer 400 (Open WebUI streams its STT uploads chunked). The
// transport negotiates compression itself and hands back the decoded body.
var dropHeaders = map[string]bool{
	"Content-Length":    true,
	"Host":              true,
	"Transfer-Encoding": true,
	"Accept-Encoding":   true,
	"Connection":        true,
	"Keep-Alive":        true,
	"Proxy-Connection":  true,
	"Te":                true,
	"Trailer":           true,
	"Upgrade":           true,
}

// A rewriter edits a successful answer on its way back; write sends a piece to
// the client at once.
type rewriter func(body io.Reader, contentType string, write func([]byte))

func forward(w http.ResponseWriter, r *http.Request, target string, body []byte, rewrite rewriter) {
	var rd io.Reader = http.NoBody
	if len(body) > 0 {
		rd = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(r.Context(), r.Method, target, rd)
	if err != nil {
		writeError(w, newErr(http.StatusInternalServerError, "Proxy error: %v", err))
		return
	}
	for k, vs := range r.Header {
		if !dropHeaders[http.CanonicalHeaderKey(k)] {
			req.Header[k] = vs
		}
	}

	resp, err := upstream.Do(req)
	if err != nil {
		writeError(w, newErr(http.StatusInternalServerError, "Proxy error: %v", err))
		return
	}
	defer resp.Body.Close()

	contentType := resp.Header.Get("Content-Type")
	if contentType != "" {
		shown := contentType
		if strings.HasPrefix(shown, "text/") && !strings.Contains(shown, "charset=") {
			shown += "; charset=utf-8"
		}
		w.Header().Set("Content-Type", shown)
	} else {
		w.Header()["Content-Type"] = nil // no sniffing either
	}
	w.WriteHeader(resp.StatusCode)

	rc := http.NewResponseController(w)
	write := func(b []byte) {
		if len(b) == 0 {
			return
		}
		w.Write(b)
		rc.Flush()
	}

	if rewrite != nil && resp.StatusCode == http.StatusOK {
		rewrite(resp.Body, contentType, write)
		return
	}
	buf := make([]byte, 32<<10)
	for {
		n, err := resp.Body.Read(buf)
		write(buf[:n])
		if err != nil {
			return
		}
	}
}

// Transparent proxy that auto-loads the requested model.
func proxyToLlama(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, newErr(http.StatusBadRequest, "Invalid JSON"))
		return
	}
	parsed, err := decodeJSON(body)
	if err != nil {
		writeError(w, newErr(http.StatusBadRequest, "Invalid JSON"))
		return
	}
	payload, _ := parsed.(map[string]any)
	requested, _ := payload["model"].(string)
	if requested == "" {
		writeError(w, newErr(http.StatusBadRequest, "Model field required"))
		return
	}

	// Text endpoints resolve against text models only, matching /v1/models.
	key, quant, err := resolveModel(requested, "llm")
	if err != nil {
		writeError(w, err)
		return
	}
	inst, err := autoloadModel(key, quant)
	if err != nil {
		writeError(w, err)
		return
	}
	if err := awaitReady(r.Context(), inst); err != nil {
		writeError(w, err)
		return
	}

	if r.Header.Get("Content-Type") == "" {
		r.Header.Set("Content-Type", "application/json")
	}
	// Its own port, so another model being up changes nothing for this request.
	target := fmt.Sprintf("http://%s:%d%s", state.host, inst.Port, r.URL.EscapedPath())
	forward(w, r, target, body, cleanChat)
}

// --- Image, audio, decision and task proxies ---

var reBoundary = regexp.MustCompile(`(?i)boundary=(?:"([^"]+)"|([^\s;]+))`)

func multipartDelimiter(contentType string) []byte {
	m := reBoundary.FindStringSubmatch(contentType)
	if m == nil {
		return nil
	}
	b := m[1]
	if b == "" {
		b = m[2]
	}
	return []byte("--" + b)
}

// bytes.Index from an offset, -1 when absent.
func indexFrom(b, sep []byte, from int) int {
	if from > len(b) {
		return -1
	}
	i := bytes.Index(b[from:], sep)
	if i < 0 {
		return -1
	}
	return from + i
}

// Model id from a multipart `model` field, without copying the file part.
func modelFromMultipart(raw []byte, contentType string) string {
	delim := multipartDelimiter(contentType)
	if delim == nil {
		return ""
	}
	pos := 0
	for {
		start := indexFrom(raw, delim, pos)
		if start < 0 {
			return ""
		}
		headStart := start + len(delim)
		headEnd := indexFrom(raw, []byte("\r\n\r\n"), headStart)
		if headEnd < 0 {
			return ""
		}
		pos = headEnd + 4

		head := bytes.ToLower(raw[headStart:headEnd])
		if !bytes.Contains(head, []byte(`name="model"`)) || bytes.Contains(head, []byte("filename=")) {
			continue
		}

		value := raw[pos:]
		if end := indexFrom(raw, delim, pos); end >= 0 {
			value = raw[pos:end]
		}
		if len(value) > 256 || !utf8.Valid(value) {
			return "" // not a model name, whatever it is
		}
		return strings.TrimSpace(string(value))
	}
}

// Model id from a request body, or "" when it names no model.
func modelFromBody(raw []byte, contentType string) string {
	if len(raw) == 0 {
		return ""
	}
	ct := strings.TrimLeftFunc(strings.ToLower(contentType), unicode.IsSpace)
	if strings.HasPrefix(ct, "multipart/form-data") {
		return modelFromMultipart(raw, contentType)
	}
	parsed, err := decodeJSON(raw)
	if err != nil {
		return ""
	}
	payload, _ := parsed.(map[string]any)
	model, _ := payload["model"].(string)
	return model
}

// Forward to the instance the body names, or to the one running instance of
// this kind. Every proxied path is passed on as it came in.
func proxyToCurrent(w http.ResponseWriter, r *http.Request, kind string, rewrite rewriter, raw []byte) {
	if raw == nil {
		var err error
		if raw, err = io.ReadAll(r.Body); err != nil {
			writeError(w, newErr(http.StatusBadRequest, "Could not read the request body"))
			return
		}
	}

	var inst *Instance
	if requested := modelFromBody(raw, r.Header.Get("Content-Type")); requested != "" {
		key, quant, err := resolveModel(requested, kind)
		if err != nil {
			writeError(w, err)
			return
		}
		if inst, err = autoloadModel(key, quant); err != nil {
			writeError(w, err)
			return
		}
	} else {
		// No model named: serve it from the one instance of this kind, if there
		// is exactly one. With several up, the request has to say which.
		state.mu.Lock()
		var candidates []*Instance
		for _, i := range state.ofKind(kind) {
			if i.alive() {
				candidates = append(candidates, i)
			}
		}
		state.mu.Unlock()
		if len(candidates) == 0 {
			writeError(w, newErr(http.StatusServiceUnavailable, "No %s model is running", kind))
			return
		}
		if len(candidates) > 1 {
			names := make([]string, len(candidates))
			for i, c := range candidates {
				names[i] = c.ModelKey
			}
			sort.Strings(names)
			writeError(w, newErr(http.StatusConflict,
				"Several %s models are running (%s); name one in the request",
				kind, strings.Join(names, ", ")))
			return
		}
		inst = candidates[0]
	}

	if err := awaitReady(r.Context(), inst); err != nil {
		writeError(w, err)
		return
	}
	target := fmt.Sprintf("http://%s:%d%s", state.host, inst.Port, r.URL.EscapedPath())
	if r.URL.RawQuery != "" {
		target += "?" + r.URL.RawQuery
	}
	forward(w, r, target, raw, rewrite)
}

// TypeSafe-compatible System One API of the running decision model.
//
// Decision models answer typed questions in one forward pass and generate no
// text, so they get their own section rather than sharing the llm one. clef in
// particular serves this endpoint only.
func proxySystemOne(w http.ResponseWriter, r *http.Request) {
	proxyToCurrent(w, r, "decision", nil, nil)
}

// /v1/images/* OpenAI, /sdapi/v1/* A1111, /sdcpp/v1/* native.
func proxyImages(w http.ResponseWriter, r *http.Request) {
	proxyToCurrent(w, r, "sd", nil, nil)
}

// audio.cpp runs generation through its own task endpoint, not the OpenAI audio one.
func proxyTasks(w http.ResponseWriter, r *http.Request) {
	proxyToCurrent(w, r, "music", nil, nil)
}

// Speech and voices are tts; transcriptions are stt or llama-server.
func proxyAudio(w http.ResponseWriter, r *http.Request) {
	head, _, _ := strings.Cut(r.PathValue("path"), "/")
	isASR := head == "transcriptions" || head == "translations"
	contentType := r.Header.Get("Content-Type")
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, newErr(http.StatusBadRequest, "Could not read the request body"))
		return
	}

	kind := "tts"
	if isASR {
		kind = asrKind(raw, contentType)
	}
	if kind == "stt" {
		if raw, err = wavMultipart(r.Context(), raw, contentType); err != nil {
			writeError(w, err)
			return
		}
	}
	var rewrite rewriter
	if isASR {
		rewrite = cleanTranscription
	}
	proxyToCurrent(w, r, kind, rewrite, raw)
}

// Transcriptions go to an stt model when one is named or running, else to an llm.
//
// The llm path stays because a multimodal llama-server (gemma4) transcribes too.
func asrKind(raw []byte, contentType string) string {
	if requested := modelFromBody(raw, contentType); requested != "" {
		if _, _, ok := lookupModel(requested, "stt"); ok {
			return "stt"
		}
		return "llm"
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	for _, i := range state.ofKind("stt") {
		if i.alive() {
			return "stt"
		}
	}
	return "llm"
}

var (
	reFilename    = regexp.MustCompile(`filename="[^"]*"`)
	rePartCtyLine = regexp.MustCompile(`(?im)^content-type:.*$`)
)

// The upload's file part as WAV: audio.cpp transcribes WAV only.
//
// Open WebUI sends mp3 and browsers record webm, so anything that is not RIFF
// goes through ffmpeg. Temp files both ways: mp4/m4a keep their index at the end,
// and ffmpeg cannot seek a pipe to find it.
func wavMultipart(ctx context.Context, raw []byte, contentType string) ([]byte, error) {
	delim := multipartDelimiter(contentType)
	if delim == nil {
		return raw, nil
	}

	pos, headStart := 0, 0
	var head []byte
	for {
		start := indexFrom(raw, delim, pos)
		if start < 0 {
			return raw, nil
		}
		headStart = start + len(delim)
		headEnd := indexFrom(raw, []byte("\r\n\r\n"), headStart)
		if headEnd < 0 {
			return raw, nil
		}
		pos = headEnd + 4
		head = raw[headStart:headEnd]
		if bytes.Contains(bytes.ToLower(head), []byte("filename=")) {
			break
		}
	}

	dataEnd := indexFrom(raw, append([]byte("\r\n"), delim...), pos)
	if dataEnd < 0 {
		return raw, nil
	}
	data := raw[pos:dataEnd]
	if bytes.HasPrefix(data, []byte("RIFF")) {
		return raw, nil
	}

	tmp, err := os.MkdirTemp("", "llama-switch-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmp)
	src, dst := filepath.Join(tmp, "in"), filepath.Join(tmp, "out.wav")
	if err := os.WriteFile(src, data, 0o600); err != nil {
		return nil, err
	}

	// A file, not pipe:1: piped WAV has no sizes in its header, and audio.cpp
	// rejects that ("failed to read WAV data chunk").
	var stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, "ffmpeg", "-loglevel", "error", "-i", src,
		"-ac", "1", "-ar", "16000", dst)
	cmd.Stderr = &stderr
	runErr := cmd.Run()
	var exitErr *exec.ExitError
	if runErr != nil && !errors.As(runErr, &exitErr) {
		return nil, newErr(http.StatusInternalServerError, "Could not run ffmpeg: %v", runErr)
	}
	wav, _ := os.ReadFile(dst)
	if runErr != nil || len(wav) == 0 {
		return nil, newErr(http.StatusBadRequest, "Could not convert the upload to WAV: %s",
			strings.TrimSpace(strings.ToValidUTF8(stderr.String(), "�")))
	}

	head = reFilename.ReplaceAll(head, []byte(`filename="audio.wav"`))
	head = rePartCtyLine.ReplaceAll(head, []byte("Content-Type: audio/wav"))
	out := make([]byte, 0, headStart+len(head)+4+len(wav)+len(raw)-dataEnd)
	out = append(out, raw[:headStart]...)
	out = append(out, head...)
	out = append(out, "\r\n\r\n"...)
	out = append(out, wav...)
	out = append(out, raw[dataEnd:]...)
	return out, nil
}

// --- Qwen3-ASR answers with a "language <Name><asr_text>" header: llama.cpp#26749 ---

const (
	asrTag         = "<asr_text>"
	asrLangPrefix  = "language "
	asrHeaderLimit = 64
)

// True while the text could still grow into the header rather than a transcript.
func asrHeaderForming(head string) bool {
	head = strings.TrimLeftFunc(head, unicode.IsSpace)
	if utf8.RuneCountInString(head) >= asrHeaderLimit || strings.Contains(head, "\n") {
		return false
	}
	return strings.HasPrefix(asrLangPrefix, head) || strings.HasPrefix(head, asrLangPrefix)
}

func stripAsrHeader(text string) string {
	head, rest, found := strings.Cut(text, asrTag)
	if found && asrHeaderForming(head) {
		return rest
	}
	return text
}

// The header arrives one token at a time, so hold deltas back until it resolves.
type asrHeaderFilter struct {
	held    string
	settled bool
}

func (f *asrHeaderFilter) feed(delta string) string {
	if f.settled {
		return delta
	}
	f.held += delta
	var out string
	switch {
	case strings.Contains(f.held, asrTag):
		out = stripAsrHeader(f.held)
	case asrHeaderForming(f.held):
		return ""
	default:
		out = f.held
	}
	f.settled, f.held = true, ""
	return out
}

// A stream can end while the header is still undecided; nothing may be lost.
func (f *asrHeaderFilter) flush() string {
	out := f.held
	f.held, f.settled = "", true
	return out
}

// One JSON value and nothing after it, numbers kept as written so that a
// re-encoded event says exactly what the engine said.
func decodeJSON(b []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, errors.New("trailing data after JSON value")
	}
	return v, nil
}

func sseLine(event map[string]any) []byte {
	b, _ := marshalJSON(event)
	return append(append([]byte("data: "), b...), '\n')
}

func readBody(body io.Reader) ([]byte, map[string]any) {
	raw, _ := io.ReadAll(body)
	parsed, err := decodeJSON(raw)
	if err != nil {
		return raw, nil
	}
	payload, _ := parsed.(map[string]any)
	return raw, payload
}

// edit rewrites one event in place and says whether it did; finish releases
// whatever edit still holds, before the first line that is not an event
// ("data: [DONE]") or at the end of the stream.
func rewriteSSE(body io.Reader, write func([]byte), edit func(map[string]any) bool, finish func() []byte) {
	released := false
	release := func() []byte {
		if released {
			return nil
		}
		released = true
		return finish()
	}

	lineOut := func(line []byte) []byte {
		line = bytes.TrimRight(line, "\r")
		if !bytes.HasPrefix(line, []byte("data: ")) {
			return append(line, '\n')
		}
		parsed, err := decodeJSON(line[6:])
		if err != nil {
			return append(release(), append(line, '\n')...)
		}
		event, ok := parsed.(map[string]any)
		if !ok || !edit(event) {
			return append(line, '\n')
		}
		return sseLine(event)
	}

	br := bufio.NewReader(body)
	for {
		line, err := br.ReadBytes('\n')
		if len(line) > 0 {
			write(lineOut(bytes.TrimSuffix(line, []byte("\n"))))
		}
		if err != nil {
			break
		}
	}
	write(release())
}

func cleanTranscription(body io.Reader, contentType string, write func([]byte)) {
	if !strings.Contains(contentType, "text/event-stream") {
		raw, payload := readBody(body)
		if text, ok := payload["text"].(string); ok {
			if cleaned := stripAsrHeader(text); cleaned != text {
				payload["text"] = cleaned
				raw, _ = marshalJSON(payload)
			}
		}
		write(raw)
		return
	}

	header := &asrHeaderFilter{}
	var last map[string]any

	edit := func(event map[string]any) bool {
		// The closing event repeats the whole transcript, header included.
		if text, ok := event["text"].(string); ok {
			cleaned := stripAsrHeader(text)
			if cleaned == text {
				return false
			}
			event["text"] = cleaned
			return true
		}
		if delta, ok := event["delta"].(string); ok {
			last = event
			if cleaned := header.feed(delta); cleaned != delta {
				event["delta"] = cleaned
				return true
			}
		}
		return false
	}

	finish := func() []byte {
		held := header.flush()
		if held == "" || last == nil {
			return nil
		}
		last["delta"] = held
		return append(sseLine(last), '\n')
	}

	rewriteSSE(body, write, edit, finish)
}

func choicesOf(payload map[string]any) []map[string]any {
	list, _ := payload["choices"].([]any)
	var out []map[string]any
	for _, c := range list {
		if choice, ok := c.(map[string]any); ok {
			out = append(out, choice)
		}
	}
	return out
}

func cleanChat(body io.Reader, contentType string, write func([]byte)) {
	if !strings.Contains(contentType, "text/event-stream") {
		raw, payload := readBody(body)
		changed := false
		for _, choice := range choicesOf(payload) {
			message, _ := choice["message"].(map[string]any)
			if content, ok := message["content"].(string); ok {
				if cleaned := stripAsrHeader(content); cleaned != content {
					message["content"] = cleaned
					changed = true
				}
			}
		}
		if changed {
			raw, _ = marshalJSON(payload)
		}
		write(raw)
		return
	}

	header := &asrHeaderFilter{}
	var last map[string]any

	edit := func(event map[string]any) bool {
		changed := false
		for _, choice := range choicesOf(event) {
			delta, _ := choice["delta"].(map[string]any)
			if content, ok := delta["content"].(string); ok {
				last = event
				if cleaned := header.feed(content); cleaned != content {
					delta["content"] = cleaned
					changed = true
				}
			}
		}
		return changed
	}

	finish := func() []byte {
		held := header.flush()
		if held == "" || last == nil {
			return nil
		}
		for _, choice := range choicesOf(last) {
			if delta, ok := choice["delta"].(map[string]any); ok {
				if _, ok := delta["content"].(string); ok {
					delta["content"] = held
				}
			}
		}
		return append(sseLine(last), '\n')
	}

	rewriteSSE(body, write, edit, finish)
}
