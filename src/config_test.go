package main

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func writeConfig(t *testing.T, text string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(text), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func mustLoad(t *testing.T, text string) *Config {
	t.Helper()
	cfg, err := loadConfig(writeConfig(t, text))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestLoadConfigKeepsOrder(t *testing.T) {
	cfg := mustLoad(t, `
llm:
  b:
    q8: cmd b q8
    q4: cmd b q4
  a:
    q1: cmd a
sd:
`)
	llm := cfg.Sections["llm"]
	if !reflect.DeepEqual(llm.Keys, []string{"b", "a"}) {
		t.Errorf("model order %v", llm.Keys)
	}
	b := llm.Values["b"]
	if !reflect.DeepEqual(b.Keys, []string{"q8", "q4"}) || defaultQuant(b) != "q8" {
		t.Errorf("quants %v, default %q: the first listed is the default", b.Keys, defaultQuant(b))
	}
	for _, name := range sections {
		if cfg.Sections[name] == nil {
			t.Errorf("section %s missing", name)
		}
	}
	if !reflect.DeepEqual(cfg.Models.Keys, []string{"b", "a"}) || cfg.Kinds["a"] != "llm" {
		t.Errorf("merged %v, kinds %v", cfg.Models.Keys, cfg.Kinds)
	}
}

func TestLoadConfigKeepsTheFirstOfADuplicate(t *testing.T) {
	cfg := mustLoad(t, `
decision:
  x:
    q: from decision
llm:
  x:
    q: from llm
`)
	// Section order decides, not file order.
	if cfg.Kinds["x"] != "llm" || cfg.Models.Values["x"].Values["q"] != "from llm" {
		t.Errorf("x is %s: %v", cfg.Kinds["x"], cfg.Models.Values["x"])
	}
	if _, ok := cfg.Sections["decision"].Get("x"); !ok {
		t.Error("the grouped view lost the duplicate")
	}
}

func TestLoadConfigAcceptsEmptyAndExtraParts(t *testing.T) {
	for _, text := range []string{"", "llm:\n", "llm: {}\n", "settings: 5\nllm:\n  m:\n    q:\n"} {
		if _, err := loadConfig(writeConfig(t, text)); err != nil {
			t.Errorf("%q: %v", text, err)
		}
	}
	// Anchors work as they do in any YAML.
	cfg := mustLoad(t, "base: &shared\n  q4: shared cmd\nllm:\n  m: *shared\n")
	if cfg.Models.Values["m"].Values["q4"] != "shared cmd" {
		t.Errorf("alias: %v", cfg.Models.Values["m"])
	}
}

func TestLoadConfigRejectsBadShapes(t *testing.T) {
	for _, text := range []string{
		"llm: [1, 2]\n",
		"llm:\n  m: just a string\n",
		"llm:\n  m:\n    q: [not, a, command]\n",
		"- a\n- b\n",
		"llm: {m: {q: c}\n",
	} {
		if _, err := loadConfig(writeConfig(t, text)); err == nil {
			t.Errorf("%q loaded", text)
		}
	}
}

func TestConfigManagerKeepsTheLastGoodConfig(t *testing.T) {
	path := writeConfig(t, "llm:\n  m:\n    q: c\n")
	c := newConfigManager(path, false, nil)
	if c.kind("m") != "llm" {
		t.Fatalf("not loaded: %v", c.get().Kinds)
	}

	os.WriteFile(path, []byte("llm: [broken\n"), 0o600)
	c.reload()
	if c.kind("m") != "llm" {
		t.Error("a broken edit threw away the working config")
	}

	os.Remove(path)
	c.reload()
	if len(c.models("").Keys) != 0 {
		t.Error("a missing file still serves models")
	}
	if c.models("nope") == nil {
		t.Error("an unknown section must read as empty")
	}
}

func TestOrderedMapJSON(t *testing.T) {
	var m OrderedMap[string]
	m.Set("z", "x && y <b>")
	m.Set("a", "1")
	m.Set("z", "again && <b>")
	if !reflect.DeepEqual(m.Keys, []string{"z", "a"}) {
		t.Errorf("keys %v: a key set again keeps its place", m.Keys)
	}
	got, err := marshalJSON(map[string]any{"m": m})
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"m":{"z":"again && <b>","a":"1"}}`; string(got) != want {
		t.Errorf("got %s, want %s", got, want)
	}
}
