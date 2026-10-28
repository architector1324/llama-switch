package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"
)

// --- Client commands: run, list, ps ---
//
// They talk to a running server over its own HTTP API, the way ollama's do, so
// they work the same against this machine or another one.

const usageText = `llmsw runs local inference engines side by side behind one dashboard and one
OpenAI-compatible endpoint.

Usage:
  llmsw serve [flags]          start the server and its dashboard
  llmsw run MODEL [PROMPT]     chat with a model, loading it first if needed
  llmsw list [TYPE]            models in the config (also: ls)
  llmsw ps                     models that are up (also: status)

The client commands talk to the server at $LLMSW_HOST (default localhost:11435).
Run "llmsw COMMAND -h" for a command's flags.
`

// Where the client commands find the server: host:port, or a full URL.
func serverURL() string {
	host := strings.TrimSpace(os.Getenv("LLMSW_HOST"))
	if host == "" {
		host = "localhost:11435"
	}
	if !strings.Contains(host, "://") {
		host = "http://" + host
	}
	return strings.TrimRight(host, "/")
}

// No timeout: an answer streams for as long as the model writes.
var cliClient = &http.Client{}

// For the short calls: listings, status, the spinner's polls.
var apiClient = &http.Client{Timeout: 10 * time.Second}

func fail(err error) {
	fmt.Fprintf(os.Stderr, "Error: %v\n", err)
	os.Exit(1)
}

func isTerminal(f *os.File) bool {
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}

func unreachable(server string, err error) error {
	return fmt.Errorf("could not reach the server at %s; is `llmsw serve` running? (%v)", server, err)
}

// The reason an answer gives for failing: this server says {"detail": ...},
// llama-server {"error": {"message": ...}}.
func errorMessage(resp *http.Response) string {
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var body struct {
		Detail any `json:"detail"`
		Error  any `json:"error"`
	}
	if json.Unmarshal(raw, &body) == nil {
		if s, ok := body.Detail.(string); ok && s != "" {
			return s
		}
		switch e := body.Error.(type) {
		case string:
			return e
		case map[string]any:
			if s, ok := e["message"].(string); ok {
				return s
			}
		}
	}
	if s := strings.TrimSpace(string(raw)); s != "" {
		return s
	}
	return resp.Status
}

func getJSON(server, path string, v any) error {
	resp, err := apiClient.Get(server + path)
	if err != nil {
		return unreachable(server, err)
	}
	if resp.StatusCode != http.StatusOK {
		return errors.New(errorMessage(resp))
	}
	defer resp.Body.Close()
	return json.NewDecoder(resp.Body).Decode(v)
}

// --- list and ps ---

type listedModel struct {
	Name    string `json:"name"`
	Kind    string `json:"kind"`
	Details struct {
		ParentModel       string `json:"parent_model"`
		QuantizationLevel string `json:"quantization_level"`
	} `json:"details"`
}

type runningInstance struct {
	Model        string `json:"model"`
	Quantization string `json:"quantization"`
	Kind         string `json:"kind"`
	Label        string `json:"label"`
	Devices      string `json:"devices"`
	Port         int    `json:"port"`
	Running      bool   `json:"running"`
	Ready        bool   `json:"ready"`
	Ctx          int    `json:"ctx"`
	Parallel     int    `json:"parallel"`
	Stats        struct {
		MemModel   float64 `json:"mem_model"`
		MemContext float64 `json:"mem_context"`
		MemCompute float64 `json:"mem_compute"`
	} `json:"stats"`
}

func instanceKey(model, quant string) string { return model + "\x00" + quant }

func (i runningInstance) state() string {
	switch {
	case !i.Running:
		return "exited"
	case !i.Ready:
		return "loading"
	}
	return "running"
}

func fetchModels(server string) ([]listedModel, error) {
	var out struct {
		Models []listedModel `json:"models"`
	}
	err := getJSON(server, "/v1/models", &out)
	return out.Models, err
}

func fetchInstances(server string) ([]runningInstance, error) {
	var out struct {
		Instances []runningInstance `json:"instances"`
	}
	err := getJSON(server, "/api/status", &out)
	return out.Instances, err
}

// The same MiB/GiB the dashboard shows.
func formatMemory(mib float64) string {
	if mib <= 0 {
		return "-"
	}
	if mib >= 1024 {
		return fmt.Sprintf("%.1f GiB", mib/1024)
	}
	return fmt.Sprintf("%.0f MiB", mib)
}

func printModels(server, kind string, w io.Writer) error {
	models, err := fetchModels(server)
	if err != nil {
		return err
	}
	instances, err := fetchInstances(server)
	if err != nil {
		return err
	}
	up := map[string]string{}
	for _, i := range instances {
		up[instanceKey(i.Model, i.Quantization)] = i.state()
	}

	tw := tabwriter.NewWriter(w, 0, 0, 3, ' ', 0)
	fmt.Fprintln(tw, "NAME\tTYPE\tQUANT\tSTATE")
	for _, m := range models {
		if kind != "" && m.Kind != kind {
			continue
		}
		quant := m.Details.QuantizationLevel
		if quant == "" {
			quant = "-"
		}
		state := up[instanceKey(m.Details.ParentModel, m.Details.QuantizationLevel)]
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", m.Name, m.Kind, quant, state)
	}
	return tw.Flush()
}

func printRunning(server string, w io.Writer) error {
	instances, err := fetchInstances(server)
	if err != nil {
		return err
	}
	// Named the way run and the API take them, not by the internal label.
	models, err := fetchModels(server)
	if err != nil {
		return err
	}
	names := map[string]string{}
	for _, m := range models {
		names[instanceKey(m.Details.ParentModel, m.Details.QuantizationLevel)] = m.Name
	}

	tw := tabwriter.NewWriter(w, 0, 0, 3, ' ', 0)
	fmt.Fprintln(tw, "NAME\tTYPE\tSTATE\tMEMORY\tGPU\tCONTEXT\tPORT")
	for _, i := range instances {
		name := names[instanceKey(i.Model, i.Quantization)]
		if name == "" {
			name = i.Label
		}
		gpu := i.Devices
		if gpu == "" {
			gpu = "default"
		}
		context := "-"
		if i.Ctx > 0 {
			context = strconv.Itoa(i.Ctx)
			if i.Parallel > 1 {
				context = fmt.Sprintf("%d x %d", i.Parallel, i.Ctx)
			}
		}
		port := "-"
		if i.Port > 0 {
			port = strconv.Itoa(i.Port)
		}
		mem := i.Stats.MemModel + i.Stats.MemContext + i.Stats.MemCompute
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			name, i.Kind, i.state(), formatMemory(mem), gpu, context, port)
	}
	return tw.Flush()
}

func listCommand(args []string) {
	fs := flag.NewFlagSet("list", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Usage: llmsw list [TYPE]\n\nModels in the config, with the type, quant and state of each.\nTYPE is one of %s.\n",
			strings.Join(sections, ", "))
	}
	fs.Parse(args)
	kind := fs.Arg(0)
	if fs.NArg() > 1 || (kind != "" && !isSection(kind)) {
		fs.Usage()
		os.Exit(2)
	}
	if err := printModels(serverURL(), kind, os.Stdout); err != nil {
		fail(err)
	}
}

func psCommand(args []string) {
	fs := flag.NewFlagSet("ps", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "Usage: llmsw ps\n\nModels that are up: their state, memory, cards, context window and port.")
	}
	fs.Parse(args)
	if fs.NArg() > 0 {
		fs.Usage()
		os.Exit(2)
	}
	if err := printRunning(serverURL(), os.Stdout); err != nil {
		fail(err)
	}
}
