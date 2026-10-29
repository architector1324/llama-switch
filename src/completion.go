package main

import (
	"flag"
	"fmt"
	"os"
	"regexp"
	"slices"
	"strings"
)

// --- Shell completion: the script asks the binary, so it follows the config ---

const fishCompletion = `# llmsw completions for fish. Install with
#   llmsw completion fish > ~/.config/fish/completions/llmsw.fish
function __llmsw_complete
    llmsw __complete (commandline -xpc)[2..] "$(commandline -ct)" 2>/dev/null
end
complete -c llmsw -f -k -a '(__llmsw_complete | string match -v __files)'
complete -c llmsw -n '__llmsw_complete | string match -q __files' -F
`

// A candidate that tells the script to offer file names.
const filesMarker = "__files"

var commandHelp = [][2]string{
	{"serve", "start the server and its dashboard"},
	{"run", "run a model, loading it first if needed"},
	{"list", "models in the config"},
	{"ps", "models that are up"},
	{"stop", "unload models"},
	{"completion", "print a shell's completion script"},
	{"help", "the commands there are"},
}

func completionCommand(args []string) {
	fs := flag.NewFlagSet("completion", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprint(fs.Output(), "Usage: llmsw completion fish\n\n"+
			"Print the completion script for fish. It asks llmsw what fits as you type,\n"+
			"so it is installed once and keeps up with the config:\n\n"+
			"  llmsw completion fish > ~/.config/fish/completions/llmsw.fish\n")
	}
	fs.Parse(args)
	if fs.NArg() != 1 || fs.Arg(0) != "fish" {
		fs.Usage()
		os.Exit(2)
	}
	fmt.Print(fishCompletion)
}

func completeCommand(args []string) {
	if len(args) == 0 {
		return
	}
	for _, c := range complete(serverURL(), args[:len(args)-1], args[len(args)-1]) {
		fmt.Println(c)
	}
}

// Candidates for cur after words, one per line with a tab before its description, as fish takes them.
func complete(server string, words []string, cur string) []string {
	if len(words) == 0 {
		var out []string
		for _, c := range commandHelp {
			out = append(out, c[0]+"\t"+c[1])
		}
		return out
	}
	switch cmd, words := words[0], words[1:]; cmd {
	case "serve":
		fs := serveFlags(&serveArgs{})
		if _, f := positionals(fs, words, 1); f != nil {
			return flagValues(server, f, cur)
		}
		return flagNames(fs, serveShortFlags, nil)
	case "run":
		return completeRun(server, words, cur)
	case "list", "ls":
		if len(words) == 0 {
			return sections
		}
	case "ps", "status":
		return runningNames(server, words)
	case "stop":
		fs := stopFlags(new(bool))
		args, _ := positionals(fs, words, 1)
		if strings.HasPrefix(cur, "-") && len(args) == 0 {
			return flagNames(fs, nil, nil)
		}
		return runningNames(server, args)
	case "completion":
		if len(words) == 0 {
			return []string{"fish"}
		}
	}
	return nil
}

// Once the model is named, only the flags for its kind.
func completeRun(server string, words []string, cur string) []string {
	fs := runFlags(&runArgs{})
	args, pending := positionals(fs, words, -1)
	kind := ""
	if len(args) > 0 {
		if m, err := findModel(server, args[0]); err == nil {
			kind = m.Kind
		}
	}
	fits := func(name string) bool {
		kinds, limited := runFlagKinds[name]
		return kind == "" || !limited || slices.Contains(kinds, kind)
	}

	switch {
	case pending != nil:
		return flagValues(server, pending, cur)
	case strings.HasPrefix(cur, "-") && !slices.Contains(words, "--"):
		// --gpu=Vulkan1 is one word.
		if name, value, ok := strings.Cut(strings.TrimLeft(cur, "-"), "="); ok {
			f := fs.Lookup(name)
			if f == nil {
				return nil
			}
			prefix := cur[:len(cur)-len(value)]
			var out []string
			for _, v := range flagValues(server, f, value) {
				if v != filesMarker {
					v = prefix + v
				}
				out = append(out, v)
			}
			return out
		}
		return flagNames(fs, nil, fits)
	case len(args) == 0:
		return modelNames(server)
	case kind == "llm" || kind == "sd" || kind == "stt":
		// The files a message names go along.
		return []string{filesMarker}
	}
	return nil
}

// Non-flag words and the flag awaiting a value; flags end at "--" or after flagsUntil words (-1: never).
func positionals(fs *flag.FlagSet, words []string, flagsUntil int) ([]string, *flag.Flag) {
	var args []string
	for i := 0; i < len(words); i++ {
		w := words[i]
		if w == "--" {
			return append(args, words[i+1:]...), nil
		}
		if flagsUntil >= 0 && len(args) >= flagsUntil || !strings.HasPrefix(w, "-") || w == "-" {
			args = append(args, w)
			continue
		}
		name, _, inline := strings.Cut(strings.TrimLeft(w, "-"), "=")
		f := fs.Lookup(name)
		if f == nil || inline || isBoolFlag(f) {
			continue
		}
		if i+1 == len(words) {
			return args, f
		}
		i++
	}
	return args, nil
}

func isBoolFlag(f *flag.Flag) bool {
	b, ok := f.Value.(interface{ IsBoolFlag() bool })
	return ok && b.IsBoolFlag()
}

// Long flag names with two dashes; short aliases left out.
func flagNames(fs *flag.FlagSet, short map[string]string, fits func(string) bool) []string {
	aliases := map[string]bool{}
	for _, s := range short {
		aliases[s] = true
	}
	var out []string
	fs.VisitAll(func(f *flag.Flag) {
		if aliases[f.Name] || (fits != nil && !fits(f.Name)) {
			return
		}
		dash := "--"
		if len(f.Name) == 1 {
			dash = "-"
		}
		_, usage := flag.UnquoteUsage(f)
		out = append(out, dash+f.Name+"\t"+usage)
	})
	return out
}

// What a flag's value can be, where there is a list of it.
func flagValues(server string, f *flag.Flag, cur string) []string {
	switch f.Name {
	case "o", "image", "mask", "questions", "config", "f":
		return []string{filesMarker}
	case "gpu":
		return gpuValues(server, cur)
	case "size":
		return dashboardSizes()
	}
	if isBoolFlag(f) {
		return []string{"true", "false"}
	}
	return nil
}

// The server's cards for --gpu; after a comma, the ones not named yet.
func gpuValues(server, cur string) []string {
	var res struct {
		Devices []struct {
			ID   string `json:"id"`
			Name string `json:"name"`
		} `json:"devices"`
	}
	if getJSON(server, "/api/devices", &res) != nil {
		return nil
	}
	named := ""
	if i := strings.LastIndexByte(cur, ','); i >= 0 {
		named = cur[:i+1]
	}
	var out []string
	if named == "" {
		out = append(out, "all\tevery card")
	}
	for _, d := range res.Devices {
		if !slices.Contains(strings.Split(named, ","), d.ID) {
			out = append(out, named+d.ID+"\t"+d.Name)
		}
	}
	return out
}

// The dashboard's Output Size choices, read from the page so the two agree.
var reSizeOption = regexp.MustCompile(`<option value="(\d+x\d+)">[^<\[]*\[([^\]]+)\]`)

func dashboardSizes() []string {
	page, _ := assets.ReadFile("templates/index.html")
	var out []string
	for _, m := range reSizeOption.FindAllSubmatch(page, -1) {
		out = append(out, string(m[1])+"\t"+string(m[2]))
	}
	return out
}

func modelNames(server string) []string {
	models, err := fetchModels(server)
	if err != nil {
		return nil
	}
	var out []string
	for _, m := range models {
		out = append(out, m.Name+"\t"+m.Kind)
	}
	return out
}

// The models that are up, as stop takes them, but for those named already.
func runningNames(server string, named []string) []string {
	instances, err := fetchInstances(server)
	if err != nil {
		return nil
	}
	names, err := instanceNames(server)
	if err != nil {
		return nil
	}
	var out []string
	for _, i := range instances {
		name := names[instanceKey(i.Model, i.Quantization)]
		if name != "" && !slices.Contains(named, name) {
			out = append(out, name+"\t"+i.Kind)
		}
	}
	return out
}
