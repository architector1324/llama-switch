package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime/multipart"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// --- Qwen3-ASR header ---

func TestStripAsrHeader(t *testing.T) {
	for in, want := range map[string]string{
		"language English<asr_text>hello":                     "hello",
		"  language Russian<asr_text>Привет":                  "Привет",
		"<asr_text>hello":                                     "hello",
		"hello":                                               "hello",
		"hello<asr_text>world":                                "hello<asr_text>world",
		"language a\nb<asr_text>x":                            "language a\nb<asr_text>x",
		"language " + strings.Repeat("я", 60) + "<asr_text>x": "language " + strings.Repeat("я", 60) + "<asr_text>x",
	} {
		if got := stripAsrHeader(in); got != want {
			t.Errorf("stripAsrHeader(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestAsrHeaderFilterHoldsTheHeader(t *testing.T) {
	f := &asrHeaderFilter{}
	var out []string
	for _, d := range []string{"lang", "uage", " English", "<asr", "_text>", "Hi", " there"} {
		out = append(out, f.feed(d))
	}
	if got := strings.Join(out, "|"); got != "|||||Hi| there" {
		t.Errorf("deltas %q", got)
	}

	f = &asrHeaderFilter{}
	if got := f.feed("Hello"); got != "Hello" {
		t.Errorf("plain text held: %q", got)
	}

	// The stream ends while the header could still be one: nothing is lost.
	f = &asrHeaderFilter{}
	if f.feed("lang") != "" || f.flush() != "lang" || f.feed("x") != "x" {
		t.Error("undecided text not released")
	}
}

func rewrite(fn rewriter, contentType, body string) string {
	var out bytes.Buffer
	fn(strings.NewReader(body), contentType, func(b []byte) { out.Write(b) })
	return out.String()
}

func sseEvents(t *testing.T, stream, field string) []string {
	t.Helper()
	var got []string
	for _, line := range strings.Split(stream, "\n") {
		if !strings.HasPrefix(line, "data: {") {
			continue
		}
		var ev map[string]any
		if err := json.Unmarshal([]byte(line[6:]), &ev); err != nil {
			t.Fatalf("%q: %v", line, err)
		}
		if field == "content" {
			got = append(got, ev["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)["content"].(string))
		} else if v, ok := ev[field].(string); ok {
			got = append(got, v)
		}
	}
	return got
}

func chatEvent(content string) string {
	quoted, _ := json.Marshal(content)
	return `data: {"id":"x","created":12345678901234567,"choices":[{"index":0,"delta":{"content":` + string(quoted) + `}}]}`
}

func TestCleanChatStream(t *testing.T) {
	in := strings.Join([]string{
		": keep-alive", "",
		chatEvent("language"), "", chatEvent(" English<asr"), "", chatEvent("_text>Hel"), "",
		chatEvent("lo"), "", "data: [DONE]", "", "",
	}, "\n")
	out := rewrite(cleanChat, "text/event-stream", in)
	if got := strings.Join(sseEvents(t, out, "content"), ""); got != "Hello" {
		t.Errorf("text %q", got)
	}
	if !strings.HasPrefix(out, ": keep-alive\n\n") || !strings.HasSuffix(out, "data: [DONE]\n\n") {
		t.Errorf("framing lost: %q", out)
	}
	if !strings.Contains(out, `"created":12345678901234567`) {
		t.Error("a big integer lost precision in a rewritten event")
	}

	// Lines it had no reason to touch go out as they came.
	plain := chatEvent("Hello") + "\n\n" + chatEvent(" world") + "\n\ndata: [DONE]\n\n"
	if out := rewrite(cleanChat, "text/event-stream", plain); out != plain {
		t.Errorf("plain stream changed:\n%q\n%q", plain, out)
	}

	// Still undecided at [DONE]: what was held goes out before it.
	out = rewrite(cleanChat, "text/event-stream", chatEvent("lang")+"\n\ndata: [DONE]\n\n")
	if i, j := strings.Index(out, `"content":"lang"`), strings.Index(out, "[DONE]"); i < 0 || i > j {
		t.Errorf("held text not released before [DONE]: %q", out)
	}
}

func TestCleanChatJSON(t *testing.T) {
	in := `{"id":"x","created":12345678901234567,"choices":[{"index":0,"message":{"role":"assistant","content":"language English<asr_text>Hi & <bye>"}}]}`
	var got map[string]any
	dec := json.NewDecoder(strings.NewReader(rewrite(cleanChat, "application/json", in)))
	dec.UseNumber()
	if err := dec.Decode(&got); err != nil {
		t.Fatal(err)
	}
	msg := got["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != "Hi & <bye>" || got["created"] != json.Number("12345678901234567") {
		t.Errorf("rewritten %v", got)
	}

	plain := `{"choices": [{"message": {"content": "Hello"}}], "x": 1.50}`
	if out := rewrite(cleanChat, "application/json", plain); out != plain {
		t.Errorf("untouched answer re-encoded: %q", out)
	}
	if out := rewrite(cleanChat, "text/plain", "not json"); out != "not json" {
		t.Errorf("non-JSON answer changed: %q", out)
	}
}

func TestCleanTranscription(t *testing.T) {
	out := rewrite(cleanTranscription, "application/json", `{"text":"language English<asr_text>hello"}`)
	if out != `{"text":"hello"}` {
		t.Errorf("JSON %q", out)
	}

	delta := func(s string) string {
		q, _ := json.Marshal(s)
		return `data: {"type":"transcript.text.delta","delta":` + string(q) + "}\n\n"
	}
	stream := delta("language") + delta(" Russian<asr_text>") + delta("Privet") +
		`data: {"type":"transcript.text.done","text":"language Russian<asr_text>Privet"}` + "\n\n"
	out = rewrite(cleanTranscription, "text/event-stream", stream)
	if got := strings.Join(sseEvents(t, out, "delta"), ""); got != "Privet" {
		t.Errorf("deltas %q", got)
	}
	if got := sseEvents(t, out, "text"); len(got) != 1 || got[0] != "Privet" {
		t.Errorf("closing event %q", got)
	}

	// No [DONE] at all: the held text goes out when the stream ends.
	out = rewrite(cleanTranscription, "text/event-stream", delta("lang"))
	if got := strings.Join(sseEvents(t, out, "delta"), ""); got != "lang" {
		t.Errorf("held at the end: %q", got)
	}
}

// --- Request bodies ---

func TestModelFromMultipart(t *testing.T) {
	body, ctype := multipartBody(t, map[string]string{"model": " ears \r\n", "prompt": "p"}, "x.wav", []byte("RIFF"))
	if got := modelFromBody(body, ctype); got != "ears" {
		t.Errorf("model %q", got)
	}

	// After the file, and with a quoted boundary.
	var b bytes.Buffer
	mw := multipart.NewWriter(&b)
	mw.SetBoundary("b-1")
	fw, _ := mw.CreateFormFile("file", "x.wav")
	fw.Write([]byte(`name="model"` + "\r\n\r\nnot this"))
	mw.WriteField("model", "img")
	mw.Close()
	if got := modelFromMultipart(b.Bytes(), `multipart/form-data; boundary="b-1"`); got != "img" {
		t.Errorf("model after the file part: %q", got)
	}

	// A file that happens to be called model is not the field.
	b.Reset()
	mw = multipart.NewWriter(&b)
	fw, _ = mw.CreateFormFile("model", "model.bin")
	fw.Write([]byte("weights"))
	mw.Close()
	if got := modelFromMultipart(b.Bytes(), mw.FormDataContentType()); got != "" {
		t.Errorf("file part read as the model: %q", got)
	}

	body, ctype = multipartBody(t, map[string]string{"model": strings.Repeat("x", 300)}, "", nil)
	if got := modelFromMultipart(body, ctype); got != "" {
		t.Errorf("a 300-byte value read as a model name")
	}
	if got := modelFromMultipart(body, "multipart/form-data"); got != "" {
		t.Errorf("no boundary, still a model: %q", got)
	}
}

func TestModelFromBody(t *testing.T) {
	for body, want := range map[string]string{
		`{"model":"zeta","x":1}`: "zeta",
		`{"model":5}`:            "",
		`{"model":""}`:           "",
		`["zeta"]`:               "",
		`{"model":"zeta"} junk`:  "",
		`{nope`:                  "",
		``:                       "",
	} {
		if got := modelFromBody([]byte(body), "application/json"); got != want {
			t.Errorf("modelFromBody(%q) = %q, want %q", body, got, want)
		}
	}
}

func TestDecodeJSON(t *testing.T) {
	v, err := decodeJSON([]byte(" {\"n\": 12345678901234567} \n"))
	if err != nil || v.(map[string]any)["n"] != json.Number("12345678901234567") {
		t.Errorf("%v %v", v, err)
	}
	if _, err := decodeJSON([]byte(`{} {}`)); err == nil {
		t.Error("trailing data accepted")
	}
}

func TestWavMultipartLeavesWavAlone(t *testing.T) {
	ctx := context.Background()
	body, ctype := multipartBody(t, map[string]string{"model": "ears"}, "x.wav", []byte("RIFF....WAVE"))
	for _, tc := range []struct {
		body  []byte
		ctype string
	}{
		{body, ctype},
		{body, "multipart/form-data"}, // no boundary
		{[]byte(`{"model":"ears"}`), "application/json"},
	} {
		got, err := wavMultipart(ctx, tc.body, tc.ctype)
		if err != nil || !bytes.Equal(got, tc.body) {
			t.Errorf("%q changed: %v", tc.ctype, err)
		}
	}
	noFile, ctype := multipartBody(t, map[string]string{"model": "ears"}, "", nil)
	if got, err := wavMultipart(ctx, noFile, ctype); err != nil || !bytes.Equal(got, noFile) {
		t.Errorf("no file part: %v", err)
	}
}

func TestWavMultipartConverts(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("no ffmpeg")
	}
	ctx := context.Background()
	mp3 := filepath.Join(t.TempDir(), "in.mp3")
	if err := exec.Command("ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
		"sine=frequency=440:duration=0.5", mp3).Run(); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(mp3)
	body, ctype := multipartBody(t, map[string]string{"model": "ears", "zlanguage": "ru"}, "x.mp3", data)

	got, err := wavMultipart(ctx, body, ctype)
	if err != nil {
		t.Fatal(err)
	}
	// Still a well-formed form, with the other fields where they were.
	_, params, _ := strings.Cut(ctype, "boundary=")
	mr := multipart.NewReader(bytes.NewReader(got), params)
	fields := map[string]string{}
	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("converted form does not parse: %v", err)
		}
		content, _ := io.ReadAll(part)
		if part.FileName() != "" {
			if part.FileName() != "audio.wav" || part.Header.Get("Content-Type") != "audio/wav" ||
				!bytes.HasPrefix(content, []byte("RIFF")) {
				t.Errorf("file part %q %q %q", part.FileName(), part.Header.Get("Content-Type"), content[:4])
			}
			continue
		}
		fields[part.FormName()] = string(content)
	}
	if fields["model"] != "ears" || fields["zlanguage"] != "ru" {
		t.Errorf("fields %v", fields)
	}

	garbage, ctype := multipartBody(t, nil, "x.bin", []byte("not audio at all"))
	_, err = wavMultipart(ctx, garbage, ctype)
	if he, ok := err.(*httpErr); !ok || he.status != 400 ||
		!strings.HasPrefix(he.detail, "Could not convert the upload to WAV: ") {
		t.Errorf("garbage: %v", err)
	}
}
