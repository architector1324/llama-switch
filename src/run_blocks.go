package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
)

// --- The code blocks of an answer, saved to files ---
//
// What a language model writes between fences is often a file in its own
// right: JSON, an SVG, a script. In a terminal each one is saved the way an
// image model's picture is, named for the model and the time, with the
// extension its language has.

type codeBlock struct {
	lang string // the first word after the opening fence, lowercased
	text string
}

// The fenced blocks of a Markdown answer, closed ones only: a block the answer
// broke off in is not a whole file.
func codeBlocks(answer string) []codeBlock {
	var blocks []codeBlock
	var fence, lang string
	var indent int
	var lines []string
	open := false
	for _, line := range strings.Split(answer, "\n") {
		line = strings.TrimSuffix(line, "\r")
		trimmed := strings.TrimLeft(line, " ")
		f := fenceOf(trimmed)
		if !open {
			if f != "" {
				open, fence, indent, lines = true, f, len(line)-len(trimmed), nil
				lang = strings.ToLower(strings.Trim(firstWord(trimmed[len(f):]), "{}."))
			}
			continue
		}
		// Closed by the same character, at least as many, and nothing after.
		if f != "" && f[0] == fence[0] && len(f) >= len(fence) && strings.TrimSpace(trimmed[len(f):]) == "" {
			if text := strings.Join(lines, "\n"); strings.TrimSpace(text) != "" {
				blocks = append(blocks, codeBlock{lang, text + "\n"})
			}
			open = false
			continue
		}
		// A fence indented in a list item: its lines lose as much.
		for n := 0; n < indent && strings.HasPrefix(line, " "); n++ {
			line = line[1:]
		}
		lines = append(lines, line)
	}
	return blocks
}

// The fence s starts with: three or more backticks or tildes; "" for none. A
// backtick in what follows makes it inline code instead, as in ```x```.
func fenceOf(s string) string {
	for _, c := range []byte{'`', '~'} {
		n := 0
		for n < len(s) && s[n] == c {
			n++
		}
		if n >= 3 {
			if c == '`' && strings.Contains(s[n:], "`") {
				return ""
			}
			return s[:n]
		}
	}
	return ""
}

func firstWord(s string) string {
	if f := strings.Fields(s); len(f) > 0 {
		return f[0]
	}
	return ""
}

// Languages whose name is not their extension.
var langExts = map[string]string{
	"python": "py", "python3": "py", "javascript": "js", "typescript": "ts",
	"bash": "sh", "shell": "sh", "zsh": "sh", "powershell": "ps1",
	"golang": "go", "rust": "rs", "ruby": "rb", "kotlin": "kt", "perl": "pl",
	"haskell": "hs", "csharp": "cs", "c#": "cs", "c++": "cpp", "objective-c": "m",
	"markdown": "md", "yml": "yaml", "latex": "tex", "makefile": "mk",
	"text": "txt", "plaintext": "txt", "plain": "txt", "console": "txt", "output": "txt",
}

var reExt = regexp.MustCompile(`^[a-z0-9]{1,12}$`)

// The extension a block's file gets: from its language, or, where it names
// none or only a generic one, from what it holds.
func blockExt(b codeBlock) string {
	body := strings.TrimSpace(b.text)
	generic := b.lang == "" || b.lang == "xml" || b.lang == "html" || b.lang == "text" || b.lang == "plaintext"
	switch {
	case generic && (strings.HasPrefix(body, "<svg") || strings.HasPrefix(body, "<?xml") && strings.Contains(body, "<svg")):
		return "svg"
	case b.lang == "" && json.Valid([]byte(body)):
		return "json"
	case b.lang == "" && (strings.HasPrefix(strings.ToLower(body), "<!doctype html") || strings.HasPrefix(body, "<html")):
		return "html"
	}
	if ext, ok := langExts[b.lang]; ok {
		return ext
	}
	if reExt.MatchString(b.lang) {
		return b.lang
	}
	return "txt"
}

// Each block to a new file in the current directory: model-time.ext, numbered
// on when several land in the same second with the same extension.
func saveBlocks(opts runOptions, answer string, out io.Writer) error {
	for _, b := range codeBlocks(answer) {
		path := freshName(opts.Model, blockExt(b))
		if err := os.WriteFile(path, []byte(b.text), 0o644); err != nil {
			return err
		}
		fmt.Fprintf(out, "Saved %s\n", path)
	}
	return nil
}
