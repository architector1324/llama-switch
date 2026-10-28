package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
)

func TestComplete(t *testing.T) {
	h := newHarness(t)
	has := func(got []string, want ...string) bool {
		for _, w := range want {
			if !slices.Contains(got, w) {
				return false
			}
		}
		return true
	}
	names := func(got []string) []string {
		var out []string
		for _, c := range got {
			name, _, _ := strings.Cut(c, "\t")
			out = append(out, name)
		}
		return out
	}

	for _, tc := range []struct {
		words  string
		cur    string
		check  func([]string) bool
		reason string
	}{
		{"", "", func(c []string) bool { return strings.HasPrefix(c[0], "serve\t") && has(names(c), "run", "stop") }, "commands"},
		{"run", "", func(c []string) bool { return c[0] == "zeta\tllm" && has(c, "img\tsd", "clef\tdecision") }, "models, in the config's order"},
		{"run", "-", func(c []string) bool { return has(names(c), "--system", "--size", "--gpu") }, "every flag before the model"},
		{"run img", "-", func(c []string) bool {
			return has(names(c), "--size", "--image", "-o", "--gpu", "--verbose") && !has(names(c), "--system")
		}, "an image model's flags"},
		{"run --ctx 8192", "", func(c []string) bool { return has(names(c), "zeta") }, "a flag's value is not the model"},
		{"run --multi zeta", "-", func(c []string) bool { return has(names(c), "--system") && !has(names(c), "--size") }, "a bool flag takes no value"},
		{"run img --size", "", func(c []string) bool { return has(c, "1024x1024\t1:1", "2752x1536\t16:9") }, "the dashboard's sizes"},
		{"run img -o", "", func(c []string) bool { return fmt.Sprint(c) == "["+filesMarker+"]" }, "a file"},
		{"run zeta", "wh", func(c []string) bool { return fmt.Sprint(c) == "["+filesMarker+"]" }, "files a message names"},
		{"run zeta hello", "-", func(c []string) bool { return fmt.Sprint(c) == "["+filesMarker+"]" }, "no flags in the prompt"},
		{"run voice", "", func(c []string) bool { return len(c) == 0 }, "speech is text only"},
		{"stop", "-", func(c []string) bool { return fmt.Sprint(names(c)) == "[--all]" }, "stop's flag"},
		{"stop", "", func(c []string) bool { return len(c) == 0 }, "nothing up"},
		{"ls", "", func(c []string) bool { return fmt.Sprint(c) == fmt.Sprint(sections) }, "the types"},
		{"serve", "-", func(c []string) bool { return has(names(c), "--host", "--config") && !has(names(c), "-H") }, "long names only"},
		{"serve --config", "", func(c []string) bool { return fmt.Sprint(c) == "["+filesMarker+"]" }, "the config file"},
		{"completion", "", func(c []string) bool { return fmt.Sprint(c) == "[fish]" }, "the shells"},
	} {
		got := complete(h.url, strings.Fields(tc.words), tc.cur)
		if !tc.check(got) {
			t.Errorf("%q %q: %s, got %q", tc.words, tc.cur, tc.reason, got)
		}
	}

	// The cards: /api/devices asks the machine, not the harness.
	cards := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"devices": testDevices})
	}))
	defer cards.Close()
	if got := complete(cards.URL, []string{"run", "zeta", "--gpu"}, "Vulkan0,"); fmt.Sprint(got) != "[Vulkan0,Vulkan1\tFake GPU]" {
		t.Errorf("--gpu after a comma: %q", got)
	}
	if got := complete(cards.URL, []string{"run"}, "--gpu=V"); !has(got, "--gpu=Vulkan0\tFake GPU", "--gpu=Vulkan1\tFake GPU") {
		t.Errorf("--gpu=: %q", got)
	}

	h.start(map[string]any{"model_key": "zeta", "quantization": "q4"})
	h.waitReady("zeta")
	if got := complete(h.url, []string{"stop"}, ""); fmt.Sprint(got) != "[zeta\tllm]" {
		t.Errorf("stop: %q", got)
	}
	if got := complete(h.url, []string{"stop", "zeta"}, ""); len(got) != 0 {
		t.Errorf("stop zeta: %q", got)
	}
}
