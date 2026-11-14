package main

import (
	"bytes"
	"context"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"testing"
)

func Test3DProxyAndClientCancellation(t *testing.T) {
	h := newHarness(t)
	useConfig(t, testConfig()+"\n3d:\n  pixal3d:\n    f16: "+fakeEngine("3d")+" --port ${PORT} --host ${HOST} ${DEVICES:--backend Vulkan --gpu 0}\n", false)
	code, _, _ := h.raw("GET", "/v1/3d/pixal3d/health", nil)
	if code != http.StatusServiceUnavailable {
		t.Fatalf("health before start: %d", code)
	}
	h.start(map[string]any{"model_key": "pixal3d", "devices": "Vulkan1"})
	code, page, _ := h.raw("GET", "/studio/?model=pixal3d", nil)
	if code != http.StatusOK || !bytes.Contains(page, []byte("Trellis Studio")) {
		t.Fatalf("studio page: %d %s", code, page)
	}
	code, _, _ = h.raw("GET", "/studio/assets/", nil)
	if code != http.StatusNotFound {
		t.Fatalf("studio directory listing: %d", code)
	}
	code, body, _ := h.raw("GET", "/v1/3d/pixal3d/health", nil)
	if code != http.StatusOK || string(body) != "ok" {
		t.Fatalf("health %d %q", code, body)
	}

	var b bytes.Buffer
	mw := multipart.NewWriter(&b)
	part, _ := mw.CreateFormFile("image", "in.png")
	part.Write([]byte("fake input image"))
	mw.Close()
	code, body, headers := h.raw("POST", "/v1/3d/pixal3d/generate", b.Bytes(), "Content-Type", mw.FormDataContentType())
	if code != http.StatusOK || string(body) != "glTF fake model" || headers.Get("Content-Type") != "model/gltf-binary" {
		t.Fatalf("result: %d %q %v", code, body, headers)
	}

	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, "POST", h.url+"/v1/3d/pixal3d/generate", bytes.NewReader(b.Bytes()))
	req.Header.Set("Content-Type", mw.FormDataContentType())
	req.Header.Set("X-Test-Slow", "1")
	done := make(chan struct{})
	go func() {
		defer close(done)
		if resp, err := testClient.Do(req); err == nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}
	}()
	h.waitFor("3D generation", func() bool { return h.instance("pixal3d")["active"].(float64) == 1 })
	cancel()
	<-done
	if got := h.instance("pixal3d")["active"]; got != float64(1) {
		t.Fatalf("lost active request on disconnect: %v", got)
	}
	h.waitFor("3D generation to finish", func() bool { return h.instance("pixal3d")["active"].(float64) == 0 })
}

func TestRun3D(t *testing.T) {
	h := newHarness(t)
	useConfig(t, testConfig()+"\n3d:\n  pixal3d:\n    f16: "+fakeEngine("3d")+" --port ${PORT} --host ${HOST}\n", false)
	image := filepath.Join(t.TempDir(), "input.png")
	if err := os.WriteFile(image, []byte("fake input image"), 0600); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	opts := runOptions{Server: h.url, Model: "pixal3d", Kind: "3d"}
	if err := generate3D(context.Background(), opts, image, &out, io.Discard); err != nil {
		t.Fatal(err)
	}
	if out.String() != "glTF fake model" {
		t.Fatalf("output %q", out.String())
	}
	if err := generate3D(context.Background(), opts, "no image", io.Discard, io.Discard); err == nil {
		t.Fatal("missing image accepted")
	}
}
