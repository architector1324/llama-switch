package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"mime/multipart"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"
)

// --- run for the kinds that do not chat ---

// Where a generated file goes: the -o path; stdout when that is not a
// terminal, so `llmsw run z-image "a cat" > cat.png` works; else a new file in
// the current directory. Returns the path written, "" for stdout.
func deliver(opts runOptions, data []byte, ext string, out, errOut io.Writer) (string, error) {
	path := opts.Output
	if path == "" {
		if !opts.Styled {
			_, err := out.Write(data)
			return "", err
		}
		path = freshName(opts.Model, ext)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return "", err
	}
	fmt.Fprintf(noteWriter(opts, out, errOut), "Saved %s\n", path)
	return path, nil
}

// Where a note about the result goes: into the terminal, or out of the way of
// a pipe that carries the result itself.
func noteWriter(opts runOptions, out, errOut io.Writer) io.Writer {
	if opts.Styled {
		return out
	}
	return errOut
}

// The files a message can carry, by extension.
var mediaTypes = map[string]string{
	".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
	".bmp": "image/bmp", ".gif": "image/gif",
	".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg",
	".opus": "audio/opus", ".m4a": "audio/mp4", ".aac": "audio/aac",
	".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska",
	".mov": "video/quicktime", ".avi": "video/x-msvideo", ".mpeg": "video/mpeg", ".mpg": "video/mpeg",
}

// "image", "audio" or "video"; "" for any other file.
func mediaKind(path string) string {
	kind, _, _ := strings.Cut(mediaTypes[strings.ToLower(filepath.Ext(path))], "/")
	return kind
}

// Piped text, then the prompt that says what to do with it, as with ollama.
func joinInput(piped, prompt string) string {
	return strings.TrimSpace(strings.TrimSpace(piped) + "\n\n" + prompt)
}

// Files of the kinds given that a message names come out of its text and go
// with the request, the way ollama attaches them: a path, quoted, with its
// spaces escaped, or dragged in as a file:// URL. A word that only looks like
// a file name ("logo.png style") stays text. strict makes a path to nothing an
// error rather than a prompt nobody meant; otherwise it stays text, as in a
// question about that path.
func extractFiles(text string, strict bool, kinds ...string) (string, []string, error) {
	type span struct{ start, end int }
	var cut []span
	var paths []string

	for i := 0; i < len(text); {
		if text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r' {
			i++
			continue
		}
		start := i
		if q := text[i]; q == '\'' || q == '"' {
			if j := strings.IndexByte(text[i+1:], q); j >= 0 {
				i += j + 2
			} else {
				i = len(text)
			}
		} else {
			for i < len(text) && !strings.ContainsRune(" \t\n\r", rune(text[i])) {
				if text[i] == '\\' {
					i++
				}
				i++
			}
			i = min(i, len(text))
		}

		word := text[start:i]
		path := unquotePath(word)
		kind := mediaKind(path)
		if !slices.Contains(kinds, kind) {
			continue
		}
		if st, err := os.Stat(path); err == nil && st.Mode().IsRegular() {
			cut = append(cut, span{start, i})
			paths = append(paths, path)
		} else if strict && (strings.ContainsAny(word, `/\`) || strings.HasPrefix(word, "~")) {
			return "", nil, fmt.Errorf("%s: no such %s", path, kind)
		}
	}

	// A path goes with the blanks after it; the rest keeps its lines, code and all.
	var rest strings.Builder
	last := 0
	for _, c := range cut {
		rest.WriteString(text[last:c.start])
		for last = c.end; last < len(text) && (text[last] == ' ' || text[last] == '\t'); last++ {
		}
	}
	rest.WriteString(text[last:])
	return strings.TrimSpace(rest.String()), paths, nil
}

// model-20261028-221530.png, numbered on if that is taken.
func freshName(model, ext string) string {
	base := strings.NewReplacer(":", "-", "/", "-").Replace(model) + "-" + time.Now().Format("20060102-150405")
	name := base + "." + ext
	for i := 2; ; i++ {
		if _, err := os.Stat(name); os.IsNotExist(err) {
			return name
		}
		name = fmt.Sprintf("%s-%d.%s", base, i, ext)
	}
}

// What both image routes answer with.
type imageAnswer struct {
	Data []struct {
		B64JSON string `json:"b64_json"`
	} `json:"data"`
	OutputFormat string `json:"output_format"`
}

// A prompt draws a new image; image files named in it, or given with --image,
// are edited by it instead.
func generateImage(ctx context.Context, opts runOptions, prompt string, out, errOut io.Writer) error {
	started := time.Now()
	prompt, found, err := extractFiles(prompt, true, "image")
	if err != nil {
		return err
	}
	prompt = joinInput(opts.Piped, prompt)
	for _, p := range found {
		fmt.Fprintf(noteWriter(opts, out, errOut), "Added image %s\n", p)
	}
	images := append(append([]string{}, opts.Images...), found...)

	var res imageAnswer
	switch {
	case len(images) == 0 && opts.Mask != "":
		return errors.New("--mask needs an image to apply to")
	case len(images) == 0:
		req := map[string]any{"model": opts.Model, "prompt": prompt, "n": 1, "output_format": "png"}
		err = runJSON(ctx, opts, errOut, "/v1/images/generations", req, &res)
	case prompt == "":
		return errors.New("say what to do with the image")
	default:
		err = editImage(ctx, opts, prompt, images, errOut, &res)
	}
	if err != nil {
		return err
	}
	return saveImage(opts, res, started, out, errOut)
}

// The edit route takes the images as a multipart upload, the way OpenAI's does.
func editImage(ctx context.Context, opts runOptions, prompt string, images []string, errOut io.Writer, res *imageAnswer) error {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	mw.WriteField("model", opts.Model)
	mw.WriteField("prompt", prompt)
	mw.WriteField("n", "1")
	mw.WriteField("output_format", "png")
	attach := func(field, path string) error {
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		fw, _ := mw.CreateFormFile(field, filepath.Base(path))
		_, err = fw.Write(data)
		return err
	}
	for _, p := range images {
		if err := attach("image[]", p); err != nil {
			return err
		}
	}
	if opts.Mask != "" {
		if err := attach("mask", opts.Mask); err != nil {
			return err
		}
	}
	mw.Close()

	spin := startSpinner(opts.Spinner, errOut, progress(opts))
	defer spin.stop()
	resp, err := post(ctx, opts.Server, "/v1/images/edits", mw.FormDataContentType(), body.Bytes())
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(res); err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return fmt.Errorf("unreadable answer: %v", err)
	}
	return nil
}

func saveImage(opts runOptions, res imageAnswer, started time.Time, out, errOut io.Writer) error {
	if len(res.Data) == 0 {
		return errors.New("the answer carries no image")
	}
	img, err := base64.StdEncoding.DecodeString(res.Data[0].B64JSON)
	if err != nil {
		return fmt.Errorf("unreadable image: %v", err)
	}
	ext := res.OutputFormat
	switch ext {
	case "":
		ext = "png"
	case "jpeg":
		ext = "jpg"
	}
	if _, err := deliver(opts, img, ext, out, errOut); err != nil {
		return err
	}
	if opts.Verbose {
		statLine(errOut, "total duration", "%v", roundDuration(time.Since(started)))
		if cfg, _, err := image.DecodeConfig(bytes.NewReader(img)); err == nil {
			statLine(errOut, "image size", "%dx%d", cfg.Width, cfg.Height)
		}
	}
	return nil
}

func speak(ctx context.Context, opts runOptions, text string, out, errOut io.Writer) error {
	started := time.Now()
	req := map[string]any{"model": opts.Model, "input": text}
	if opts.Voice != "" {
		req["voice"] = opts.Voice
	}
	body, _ := json.Marshal(req)

	spin := startSpinner(opts.Spinner, errOut, progress(opts))
	resp, err := post(ctx, opts.Server, "/v1/audio/speech", "application/json", body)
	if err != nil {
		spin.stop()
		return err
	}
	audio, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	spin.stop()
	if err != nil {
		return err
	}
	ext := "wav"
	if strings.Contains(resp.Header.Get("Content-Type"), "mpeg") {
		ext = "mp3"
	}
	return finishAudio(ctx, opts, audio, ext, started, out, errOut)
}

func compose(ctx context.Context, opts runOptions, text string, out, errOut io.Writer) error {
	started := time.Now()
	request := map[string]any{"text": text}
	if opts.Duration > 0 {
		request["duration_seconds"] = opts.Duration
	}
	if opts.Lyrics != "" {
		request["lyrics"] = opts.Lyrics
	}
	var res struct {
		Audio string `json:"audio"`
	}
	// audio.cpp's own task route: music is not an OpenAI speech request.
	if err := runJSON(ctx, opts, errOut, "/v1/tasks/run", map[string]any{"model": opts.Model, "request": request}, &res); err != nil {
		return err
	}
	if res.Audio == "" {
		return errors.New("the answer carries no audio")
	}
	audio, err := base64.StdEncoding.DecodeString(res.Audio)
	if err != nil {
		return fmt.Errorf("unreadable audio: %v", err)
	}
	return finishAudio(ctx, opts, audio, "wav", started, out, errOut)
}

// Save the clip and, in a terminal, play it.
func finishAudio(ctx context.Context, opts runOptions, audio []byte, ext string, started time.Time, out, errOut io.Writer) error {
	path, err := deliver(opts, audio, ext, out, errOut)
	if err != nil {
		return err
	}
	if opts.Verbose {
		total := time.Since(started)
		statLine(errOut, "total duration", "%v", roundDuration(total))
		if d, ok := wavDuration(audio); ok {
			statLine(errOut, "audio duration", "%v", roundDuration(d))
			statLine(errOut, "speed", "%.2fx realtime", d.Seconds()/total.Seconds())
		}
	}
	if path == "" || !opts.Play || !opts.Styled {
		return nil
	}
	if err := play(ctx, path); err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		fmt.Fprintf(errOut, "Not played: %v\n", err)
	}
	return nil
}

// The first of these that is installed plays the clip.
var audioPlayers = [][]string{
	{"pw-play"},
	{"paplay"},
	{"aplay", "-q"},
	{"ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"},
	{"afplay"},
}

func play(ctx context.Context, path string) error {
	for _, p := range audioPlayers {
		if _, err := exec.LookPath(p[0]); err == nil {
			return exec.CommandContext(ctx, p[0], append(p[1:], path)...).Run()
		}
	}
	return errors.New("no audio player found: pw-play, paplay, aplay or ffplay")
}

// How long a PCM WAV plays, from its header.
func wavDuration(b []byte) (time.Duration, bool) {
	if len(b) < 12 || string(b[:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return 0, false
	}
	var rate, channels, bits, dataSize int
	for pos := 12; pos+8 <= len(b); {
		id, size := string(b[pos:pos+4]), int(binary.LittleEndian.Uint32(b[pos+4:pos+8]))
		body := pos + 8
		switch id {
		case "fmt ":
			if body+16 > len(b) {
				return 0, false
			}
			channels = int(binary.LittleEndian.Uint16(b[body+2:]))
			rate = int(binary.LittleEndian.Uint32(b[body+4:]))
			bits = int(binary.LittleEndian.Uint16(b[body+14:]))
		case "data":
			// A streamed WAV leaves the size open; the bytes that came are the data.
			dataSize = min(size, len(b)-body)
		}
		if id == "data" {
			break
		}
		pos = body + size + size%2
	}
	bytesPerSecond := rate * channels * bits / 8
	if bytesPerSecond == 0 || dataSize == 0 {
		return 0, false
	}
	return time.Duration(float64(dataSize) / float64(bytesPerSecond) * float64(time.Second)), true
}

// --- stt ---

// A path as a terminal hands it over: quoted, escaped, or a file:// URL when
// dragged in.
func unquotePath(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 && (s[0] == '\'' || s[0] == '"') && s[len(s)-1] == s[0] {
		s = s[1 : len(s)-1]
	} else {
		s = strings.ReplaceAll(s, `\ `, " ")
	}
	if rest, ok := strings.CutPrefix(s, "file://"); ok {
		if p, err := url.PathUnescape(rest); err == nil {
			s = p
		}
	}
	if rest, ok := strings.CutPrefix(s, "~/"); ok {
		if home, err := os.UserHomeDir(); err == nil {
			s = filepath.Join(home, rest)
		}
	}
	return s
}

func transcribeFile(ctx context.Context, opts runOptions, path string, out, errOut io.Writer) error {
	path = unquotePath(path)
	audio, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return transcribe(ctx, opts, path, audio, out, errOut)
}

// Upload one recording and print what was said. The server turns anything
// that is not WAV into WAV first.
func transcribe(ctx context.Context, opts runOptions, name string, audio []byte, out, errOut io.Writer) error {
	started := time.Now()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	mw.WriteField("model", opts.Model)
	fw, _ := mw.CreateFormFile("file", filepath.Base(name))
	fw.Write(audio)
	mw.Close()

	spin := startSpinner(opts.Spinner, errOut, progress(opts))
	resp, err := post(ctx, opts.Server, "/v1/audio/transcriptions", mw.FormDataContentType(), body.Bytes())
	if err != nil {
		spin.stop()
		return err
	}
	var res struct {
		Text   string `json:"text"`
		Timing *struct {
			AudioDurationMs float64 `json:"audio_duration_ms"`
			Rtf             float64 `json:"rtf"`
		} `json:"timing"`
	}
	err = json.NewDecoder(resp.Body).Decode(&res)
	resp.Body.Close()
	spin.stop()
	if err != nil {
		return fmt.Errorf("unreadable answer: %v", err)
	}

	fmt.Fprintln(out, strings.TrimSpace(res.Text))
	if opts.Verbose {
		statLine(errOut, "total duration", "%v", roundDuration(time.Since(started)))
		if t := res.Timing; t != nil && t.AudioDurationMs > 0 {
			statLine(errOut, "audio duration", "%v", roundDuration(milliseconds(t.AudioDurationMs)))
			if t.Rtf > 0 {
				statLine(errOut, "speed", "%.2fx realtime", 1/t.Rtf)
			}
		}
	}
	return nil
}

// --- decision ---

// The questions file holds {"questions": {...}}, the shape the API takes, or
// just the questions themselves.
func readQuestions(path string) (json.RawMessage, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("%s: %v", path, err)
	}
	if q, ok := top["questions"]; ok && bytes.HasPrefix(bytes.TrimSpace(q), []byte("{")) {
		return q, nil
	}
	if len(top) == 0 {
		return nil, fmt.Errorf("%s: no questions in it", path)
	}
	return json.RawMessage(raw), nil
}

// The question a bare prompt asks.
const yesNoName = "answer"

func yesNoQuestion(question string) json.RawMessage {
	raw, _ := json.Marshal(map[string]any{yesNoName: map[string]string{"type": "noul", "instructions": question}})
	return raw
}

type decisionAnswer struct {
	Type          string             `json:"type"`
	Noul          *float64           `json:"noul"`
	Choice        string             `json:"choice"`
	Score         *float64           `json:"score"`
	Legend        map[string]string  `json:"legend"`
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    *float64           `json:"confidence"`
}

func percent(p float64) string { return fmt.Sprintf("%.1f%%", 100*p) }

// The answer in words, and how sure the model is of it.
func (a decisionAnswer) describe() (string, string) {
	confidence := ""
	if a.Confidence != nil {
		confidence = ", confidence " + percent(*a.Confidence)
	}
	switch {
	case a.Type == "noul" && a.Noul != nil:
		if *a.Noul >= 0.5 {
			return "yes", percent(*a.Noul)
		}
		return "no", percent(1 - *a.Noul)
	case a.Type == "choice":
		return a.Choice, percent(a.Probabilities[a.Choice]) + confidence
	case a.Type == "score" && a.Score != nil:
		level := strconv.Itoa(int(math.Round(*a.Score)))
		name := a.Legend[level]
		if name == "" {
			name = level
		}
		return name, fmt.Sprintf("%.2f of 0-%d%s", *a.Score, len(a.Legend)-1, confidence)
	}
	return "?", a.Type
}

func decide(ctx context.Context, opts runOptions, questions json.RawMessage, state string, out, errOut io.Writer) error {
	started := time.Now()
	var raw json.RawMessage
	req := map[string]any{"model": opts.Model, "state": state, "questions": questions}
	if err := runJSON(ctx, opts, errOut, "/v1/systemone", req, &raw); err != nil {
		return err
	}
	var res struct {
		Answers map[string]decisionAnswer `json:"answers"`
		Usage   struct {
			InputTokens int `json:"input_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &res); err != nil {
		return fmt.Errorf("unreadable answer: %v", err)
	}

	if !opts.Styled {
		// For a script: the answers whole, probabilities and all.
		var pretty bytes.Buffer
		json.Indent(&pretty, raw, "", "  ")
		fmt.Fprintln(out, pretty.String())
	} else if names := jsonKeys(questions); len(names) == 1 && names[0] == yesNoName {
		answer, detail := res.Answers[yesNoName].describe()
		fmt.Fprintf(out, "%s  %s\n", answer, detail)
	} else {
		// In the order the questions were asked; any the file did not name, after.
		seen := map[string]bool{}
		for _, n := range names {
			seen[n] = true
		}
		var extra []string
		for n := range res.Answers {
			if !seen[n] {
				extra = append(extra, n)
			}
		}
		sort.Strings(extra)
		tw := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)
		for _, n := range append(names, extra...) {
			a, ok := res.Answers[n]
			if !ok {
				continue
			}
			answer, detail := a.describe()
			fmt.Fprintf(tw, "%s\t%s\t%s\n", n, answer, detail)
		}
		tw.Flush()
	}

	if opts.Verbose {
		statLine(errOut, "total duration", "%v", roundDuration(time.Since(started)))
		if res.Usage.InputTokens > 0 {
			statLine(errOut, "prompt eval count", "%d token(s)", res.Usage.InputTokens)
		}
	}
	return nil
}

// Top-level keys of a JSON object, in their order.
func jsonKeys(raw json.RawMessage) []string {
	dec := json.NewDecoder(bytes.NewReader(raw))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return nil
	}
	var keys []string
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return keys
		}
		key, _ := tok.(string)
		keys = append(keys, key)
		var skip json.RawMessage
		if dec.Decode(&skip) != nil {
			return keys
		}
	}
	return keys
}
