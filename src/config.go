package main

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

// --- Config Manager ---
var sections = []string{"llm", "sd", "tts", "stt", "music", "decision"}

// What each kind can do, published in /v1/models like ollama's "capabilities".
var kindCapabilities = map[string][]string{
	"llm":      {"completion", "chat"},
	"sd":       {"image"},
	"tts":      {"audio", "speech"},
	"stt":      {"audio", "transcription"},
	"music":    {"audio", "music"},
	"decision": {"completion"},
}

var kindEndpoint = map[string]string{
	"llm": "/v1/chat/completions",
	"sd":  "/v1/images/generations",
	"tts": "/v1/audio/speech",
	"stt": "/v1/audio/transcriptions",
	// audio.cpp generates music through its own task endpoint; /v1/audio/* is tts.
	"music":    "/v1/tasks/",
	"decision": "/v1/systemone",
}

func isSection(name string) bool {
	for _, s := range sections {
		if s == name {
			return true
		}
	}
	return false
}

// A YAML mapping that keeps key order: the first quant listed is the default.
type OrderedMap[V any] struct {
	Keys   []string
	Values map[string]V
}

func (m *OrderedMap[V]) Set(key string, v V) {
	if m.Values == nil {
		m.Values = map[string]V{}
	}
	if _, ok := m.Values[key]; !ok {
		m.Keys = append(m.Keys, key)
	}
	m.Values[key] = v
}

func (m *OrderedMap[V]) Get(key string) (V, bool) {
	v, ok := m.Values[key]
	return v, ok
}

func (m *OrderedMap[V]) UnmarshalYAML(n *yaml.Node) error {
	if n.Kind != yaml.MappingNode {
		return fmt.Errorf("line %d: expected a mapping", n.Line)
	}
	for i := 0; i+1 < len(n.Content); i += 2 {
		var key string
		if err := n.Content[i].Decode(&key); err != nil {
			return err
		}
		var v V
		if err := n.Content[i+1].Decode(&v); err != nil {
			return err
		}
		m.Set(key, v)
	}
	return nil
}

func (m OrderedMap[V]) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, k := range m.Keys {
		if i > 0 {
			b.WriteByte(',')
		}
		kb, _ := marshalJSON(k)
		b.Write(kb)
		b.WriteByte(':')
		vb, err := marshalJSON(m.Values[k])
		if err != nil {
			return nil, err
		}
		b.Write(vb)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// One model's command lines by quant.
type Profiles = OrderedMap[string]

type Section = OrderedMap[Profiles]

// First quant listed in the config.
func defaultQuant(p Profiles) string {
	if len(p.Keys) == 0 {
		return ""
	}
	return p.Keys[0]
}

// One loaded snapshot, never modified, so it is shared without copying.
type Config struct {
	Sections map[string]*Section
	Models   *Section // every section merged
	Kinds    map[string]string
}

func emptyConfig() *Config {
	c := &Config{Sections: map[string]*Section{}, Models: &Section{}, Kinds: map[string]string{}}
	for _, name := range sections {
		c.Sections[name] = &Section{}
	}
	return c
}

func loadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw map[string]yaml.Node
	if err := yaml.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	decode := func(name string) (*Section, error) {
		sec := &Section{}
		if node, ok := raw[name]; ok {
			if err := node.Decode(sec); err != nil {
				return nil, fmt.Errorf("section '%s': %w", name, err)
			}
		}
		return sec, nil
	}

	cfg := emptyConfig()
	for _, name := range sections {
		sec, err := decode(name)
		if err != nil {
			return nil, err
		}
		cfg.Sections[name] = sec
	}

	for _, name := range sections {
		sec := cfg.Sections[name]
		for _, key := range sec.Keys {
			if prev, dup := cfg.Kinds[key]; dup {
				fmt.Printf("[Config] Duplicate model '%s' in section '%s', keeping the one from '%s'\n",
					key, name, prev)
				continue
			}
			cfg.Models.Set(key, sec.Values[key])
			cfg.Kinds[key] = name
		}
	}
	return cfg, nil
}

type ConfigManager struct {
	path      string
	onChange  func()
	mu        sync.Mutex
	cfg       *Config
	lastMtime time.Time
}

func newConfigManager(path string, watch bool, onChange func()) *ConfigManager {
	c := &ConfigManager{path: path, onChange: onChange, cfg: emptyConfig()}
	c.reload()
	if watch {
		go c.watchLoop()
		fmt.Printf("[Config] Watching %s for changes...\n", path)
	}
	return c
}

func (c *ConfigManager) reload() {
	c.mu.Lock()
	defer c.mu.Unlock()

	st, err := os.Stat(c.path)
	if err != nil {
		fmt.Printf("[Config] Error: %s not found\n", c.path)
		c.cfg = emptyConfig()
		return
	}

	cfg, err := loadConfig(c.path)
	if err != nil {
		fmt.Printf("[Config] Failed to load config: %v\n", err)
		return
	}
	c.cfg = cfg
	c.lastMtime = st.ModTime()

	counts := make([]string, 0, len(sections))
	for _, n := range sections {
		counts = append(counts, fmt.Sprintf("%s: %d", n, len(cfg.Sections[n].Keys)))
	}
	fmt.Printf("[Config] Loaded %d models (%s) from %s\n",
		len(cfg.Models.Keys), strings.Join(counts, ", "), c.path)
}

func (c *ConfigManager) watchLoop() {
	for {
		time.Sleep(2 * time.Second)
		st, err := os.Stat(c.path)
		if err != nil {
			continue
		}
		c.mu.Lock()
		changed := st.ModTime().After(c.lastMtime)
		c.mu.Unlock()
		if changed {
			fmt.Println("[Config] Change detected, reloading...")
			c.reload()
			if c.onChange != nil {
				c.onChange()
			}
		}
	}
}

// The current snapshot.
func (c *ConfigManager) get() *Config {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cfg
}

// All models merged across sections, or just one section's models.
func (c *ConfigManager) models(kind string) *Section {
	cfg := c.get()
	if kind == "" {
		return cfg.Models
	}
	if sec, ok := cfg.Sections[kind]; ok {
		return sec
	}
	return &Section{}
}

// Which section a model came from, or "" if it is unknown.
func (c *ConfigManager) kind(modelKey string) string {
	return c.get().Kinds[modelKey]
}
