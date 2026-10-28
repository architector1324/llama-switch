package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode"
)

// --- Log Reader ---
var reANSI = regexp.MustCompile(`\x1b\[[0-9;]*[A-Za-z]`)

func cleanLine(b []byte) string {
	s := strings.ToValidUTF8(string(b), "�")
	s = reANSI.ReplaceAllString(s, "")
	return strings.TrimRightFunc(s, unicode.IsSpace)
}

// Split on CR too: sd-server redraws progress in place, without LF.
func eachOutputLine(r io.Reader, fn func(string)) {
	br := bufio.NewReaderSize(r, 64<<10)
	var buf []byte
	emit := func() {
		if len(buf) > 0 {
			fn(cleanLine(buf))
			buf = buf[:0]
		}
	}
	for {
		c, err := br.ReadByte()
		if err != nil {
			break
		}
		if c == '\r' || c == '\n' {
			emit()
			continue
		}
		buf = append(buf, c)
	}
	emit()
}

// 0.46.251.374 I srv  update_slots: all slots are idle
const idleLine = "update_slots: all slots are idle"

var (
	// prompt eval time =       4.67 ms /    11 tokens (    0.42 ms per token,  2355.46 tokens per second)
	rePrompt = regexp.MustCompile(`prompt eval time\s*=\s*[\d\.]+\s*ms\s*/\s*\d+\s*tokens\s*\(\s*[\d\.]+\s*ms per token,\s*([\d\.]+)\s*tokens per second\)`)

	// Same line as above minus the "prompt " prefix. RE2 has no lookbehind, so
	// matchEval skips the matches that sit right after "prompt ".
	reEval = regexp.MustCompile(`eval time\s*=\s*[\d\.]+\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*[\d\.]+\s*ms per token,\s*([\d\.]+)\s*tokens per second\)`)

	// slot      release: id  3 | task 10 | stop processing: n_tokens = 73, truncated = 0
	reRelease = regexp.MustCompile(`stop processing: n_tokens = (\d+)`)

	// sd-server sampling progress: "|====>     | 3/8 - 6.59s/it" (or "12.30it/s")
	reSdStep = regexp.MustCompile(`\|\s*(\d+)/(\d+)\s*-\s*([\d\.]+)(s/it|it/s)`)

	// [INFO ] stable-diffusion.cpp:5675 - sampling completed, taking 23.84s
	reSdDone = regexp.MustCompile(`sampling completed, taking\s*([\d\.]+)s`)

	// [INFO ] stable-diffusion.cpp:5592 - generate_image 1024x1024
	reSdSize = regexp.MustCompile(`generate_image\s+(\d+)x(\d+)`)

	// qwentts.cpp progress: "[Pipeline] Generated 1920 frames (slot 0)"
	reTtsStep = regexp.MustCompile(`\[Pipeline\] Generated (\d+) frames`)

	// "[Perf] Total 36762.3 ms (2048 frames, 16.67 ms/frame AR, audio 163.84 s, RTF 0.224)"
	reTtsDone = regexp.MustCompile(`\[Perf\] Total\s+([\d\.]+)\s*ms\s*\((\d+)\s*frames.*?audio\s+([\d\.]+)\s*s,\s*RTF\s+([\d\.]+)\)`)

	// Decision models print no print_timing at all, only the slot lifecycle. The
	// request's cost is the gap between these two lines; llama.cpp's own log clock
	// is used rather than ours, so pipe latency does not creep into the number.
	//   1.17.598.882 I slot launch_slot_: id  0 | task 0 | processing task
	//   1.18.108.321 I slot      release: id  0 | task 0 | stop processing: n_tokens = 449
	reDecStart = regexp.MustCompile(`^(\d+)\.(\d+)\.(\d+)\.(\d+).*launch_slot_.*processing task`)
	reDecDone  = regexp.MustCompile(`^(\d+)\.(\d+)\.(\d+)\.(\d+).*release:.*stop processing: n_tokens = (\d+)`)

	// Memory, from llama.cpp's own lines. Backend-agnostic: the device name is
	// whatever the backend calls itself, Vulkan0 or CUDA0 or CPU_Mapped.
	//   load_tensors:      Vulkan0 model buffer size =   610.56 MiB
	reMemModel = regexp.MustCompile(`(\S+) model buffer size\s*=\s*([\d\.]+)\s*MiB`)

	// Only buffers that were really allocated. The "llama_kv_cache: size =" line
	// looks tempting because it needs no device, but a draft model prints it for a
	// cache it does not own -- it shares the target's ("sharing with layer N").
	//   llama_kv_cache:    Vulkan0 KV buffer size =    51.00 MiB
	reMemKV = regexp.MustCompile(`(\S+) KV buffer size\s*=\s*([\d\.]+)\s*MiB`)

	// The vision tower is loaded by clip, which names its device on one line and
	// never prints a buffer size; mtmd prints the size once, earlier, with no device.
	//   srv load_model: [mtmd] estimated worst-case memory usage of mmproj is 1817.01 MiB
	//   clip_ctx: CLIP using Vulkan0 backend        (twice: vision and audio, one copy)
	reMemMmproj = regexp.MustCompile(`memory usage of mmproj is\s*([\d\.]+)\s*MiB`)
	reClipDev   = regexp.MustCompile(`CLIP using (\S+) backend`)

	// sd.cpp keeps its own books. The fit plan says which card each component
	// landed on, and a separate line gives what each one really weighed -- the plan
	// is only an estimate (it budgeted 319 MiB for a VAE that came out 160).
	//   backend_fit.cpp - DiT params 6272 MiB, compute reserve 2048 MiB -> compute Vulkan1, params Vulkan1
	reSdFit = regexp.MustCompile(`(\S+)\s+params\s+[\d\.]+\s*MiB,\s*compute reserve.*?params (\S+)`)

	//   diffusion_engine.cpp - total params memory size = 10509.14MB (VRAM 10509.14MB, RAM 0.00MB):
	//       text_encoders 4076.43MB(VRAM), diffusion_model 6272.71MB(VRAM), vae 160.00MB(VRAM), ...
	reSdParams = regexp.MustCompile(`total params memory size\s*=\s*([\d\.]+)MB`)
	reSdPart   = regexp.MustCompile(`(\w+) ([\d\.]+)MB\((?:VRAM|RAM)\)`)

	// Each component reports its own peak and re-reports it as it grows, so the
	// figure is kept per component and summed per device -- replacing by device
	// alone let the vae overwrite the text encoder.
	//   ggml_runner.cpp - z_image compute buffer size: 690.12 MB(VRAM) on Vulkan1 (peak across 1 segment)
	reSdCompute = regexp.MustCompile(`(\S+) compute buffer size:\s*([\d\.]+)\s*MB\((?:VRAM|RAM)\) on (\S+)`)
)

// The fit plan and the weight report name the same parts differently.
var sdPartDeviceKey = map[string]string{
	"text_encoders":   "Conditioner",
	"diffusion_model": "DiT",
	"vae":             "VAE",
	"controlnet":      "ControlNet",
	"extensions":      "PhotoMaker",
}

func matchEval(s string) []string {
	for _, idx := range reEval.FindAllStringSubmatchIndex(s, -1) {
		if strings.HasSuffix(s[:idx[0]], "prompt ") {
			continue
		}
		return []string{s[idx[0]:idx[1]], s[idx[2]:idx[3]], s[idx[4]:idx[5]]}
	}
	return nil
}

func atof(s string) float64 {
	v, _ := strconv.ParseFloat(s, 64)
	return v
}

func atoi(s string) int {
	v, _ := strconv.Atoi(s)
	return v
}

// M.SS.mmm.uuu since server start.
func logSeconds(m []string) float64 {
	return float64(atoi(m[1]))*60 + float64(atoi(m[2])) +
		float64(atoi(m[3]))/1000 + float64(atoi(m[4]))/1_000_000
}

type computeKey struct{ part, dev string }

func logReader(r io.ReadCloser, inst *Instance) {
	defer r.Close()

	var decStarted *float64

	memModel, memContext, memMmproj := 0.0, 0.0, 0.0
	memDevices := map[string]*MemSlot{}
	var memOrder []string
	sdComponentDevice := map[string]string{}
	sdCompute := map[computeKey]float64{}

	slot := func(dev string) *MemSlot {
		s, ok := memDevices[dev]
		if !ok {
			s = &MemSlot{}
			memDevices[dev] = s
			memOrder = append(memOrder, dev)
		}
		return s
	}

	eachOutputLine(r, func(line string) {
		// At trace level, which the memory panel needs, llama.cpp reports an
		// idle server once a second. Nothing reads it, and it buried the
		// request lines in the panel, so it is dropped before it is stored.
		if line == "" || strings.Contains(line, idleLine) {
			return
		}
		inst.appendLog(line)

		// Substrings that survive llama.cpp log renames ("main:" -> "llama_server:").
		if strings.Contains(line, "model loaded") ||
			strings.Contains(line, "listening on") ||
			strings.Contains(line, "server is listening") {
			inst.mu.Lock()
			inst.ready = true
			inst.mu.Unlock()
		}

		if ds := reDecStart.FindStringSubmatch(line); ds != nil {
			t := logSeconds(ds)
			decStarted = &t
		}
		if dd := reDecDone.FindStringSubmatch(line); dd != nil && decStarted != nil {
			ms := (logSeconds(dd) - *decStarted) * 1000
			inst.update(func(s *Stats) {
				s.DecisionMs = ms
				s.DecisionTokens = atoi(dd[5])
				s.DecisionCount++
			})
			decStarted = nil
		}

		// --- memory ---

		memChanged := false

		if mm := reMemModel.FindStringSubmatch(line); mm != nil {
			memModel += atof(mm[2])
			slot(mm[1]).Model += atof(mm[2])
			memChanged = true
		}

		if km := reMemKV.FindStringSubmatch(line); km != nil {
			memContext += atof(km[2])
			slot(km[1]).Context += atof(km[2])
			memChanged = true
		}

		if pm := reMemMmproj.FindStringSubmatch(line); pm != nil {
			memMmproj = atof(pm[1])
			memModel += memMmproj
			memChanged = true
		}

		// Charged once, when clip says where it put the tower.
		if cd := reClipDev.FindStringSubmatch(line); cd != nil {
			if memMmproj != 0 {
				slot(cd[1]).Model += memMmproj
				memMmproj = 0
			}
			memChanged = true
		}

		// --- sd-server keeps its own books ---

		if fit := reSdFit.FindStringSubmatch(line); fit != nil {
			sdComponentDevice[fit[1]] = fit[2]
		}

		if sp := reSdParams.FindStringSubmatch(line); sp != nil {
			memModel = atof(sp[1])
			for _, part := range reSdPart.FindAllStringSubmatch(line, -1) {
				key := part[1]
				if alias, ok := sdPartDeviceKey[key]; ok {
					key = alias
				}
				if dev, ok := sdComponentDevice[key]; ok && dev != "" {
					slot(dev).Model += atof(part[2])
				}
			}
			memChanged = true
		}

		if sc := reSdCompute.FindStringSubmatch(line); sc != nil {
			sdCompute[computeKey{sc[1], sc[3]}] = atof(sc[2])
			sums := map[string]float64{}
			for k, v := range sdCompute {
				sums[k.dev] += v
			}
			for dev, sum := range sums {
				slot(dev).Compute = sum
			}
			memChanged = true
		}

		if memChanged {
			devices := make(map[string]MemSlot, len(memDevices))
			compute := 0.0
			for _, dev := range memOrder {
				devices[dev] = *memDevices[dev]
				compute += memDevices[dev].Compute
			}
			inst.update(func(s *Stats) {
				s.MemModel = memModel
				s.MemContext = memContext
				s.MemCompute = compute
				s.MemDevices = devices
			})
		}

		// --- llama-server ---

		if pm := rePrompt.FindStringSubmatch(line); pm != nil {
			inst.update(func(s *Stats) { s.PromptSpeed = atof(pm[1]) })
		}

		if em := matchEval(line); em != nil {
			inst.update(func(s *Stats) {
				s.GenSpeed = atof(em[2])
				s.TotalTokens += atoi(em[1])
			})
		}

		if rm := reRelease.FindStringSubmatch(line); rm != nil {
			inst.update(func(s *Stats) {
				s.CtxUsed = atoi(rm[1])
				if inst.Ctx > 0 {
					s.CtxLimit = inst.Ctx
				}
			})
		}

		// --- sd-server ---

		if sm := reSdStep.FindStringSubmatch(line); sm != nil {
			value := atof(sm[3])
			// sd.cpp flips the unit below 1s per step; normalise to s/it.
			secPerStep := value
			if sm[4] != "s/it" {
				secPerStep = 0
				if value > 0 {
					secPerStep = 1 / value
				}
			}
			inst.update(func(s *Stats) {
				s.SdStep = atoi(sm[1])
				s.SdSteps = atoi(sm[2])
				s.SdSpeed = math.Round(secPerStep*100) / 100
			})
		}

		if dm := reSdDone.FindStringSubmatch(line); dm != nil {
			inst.update(func(s *Stats) {
				s.SdLastTime = atof(dm[1])
				s.SdImages++
				// Park the bar at 100% rather than the last partial step.
				if s.SdSteps != 0 {
					s.SdStep = s.SdSteps
				}
			})
		}

		if zm := reSdSize.FindStringSubmatch(line); zm != nil {
			inst.update(func(s *Stats) {
				s.SdSize = zm[1] + "x" + zm[2]
				s.SdStep = 0
			})
		}

		// --- tts-server ---

		if tm := reTtsStep.FindStringSubmatch(line); tm != nil {
			inst.update(func(s *Stats) { s.TtsFrames = atoi(tm[1]) })
		}

		if td := reTtsDone.FindStringSubmatch(line); td != nil {
			inst.update(func(s *Stats) {
				s.TtsLastTime = atof(td[1]) / 1000
				s.TtsFrames = atoi(td[2])
				s.TtsAudio = atof(td[3])
				s.TtsRtf = atof(td[4])
				s.TtsClips++
			})
		}
	})
}

// Never through a proxy: the backend is our own child on loopback. With
// http_proxy set and no_proxy unset -- which is what the service inherits --
// every hop went to privoxy and came back 502.
func directTransport() *http.Transport {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.Proxy = nil
	return t
}

func num(m map[string]any, key string) float64 {
	v, _ := m[key].(float64)
	return v
}

// Live pp/tg while a request is in flight.
//
// llama-server prints its timings only when a request finishes, so the dashboard
// sat on the previous request's numbers until the next one ended. /slots is on by
// default and counts tokens as they come, so the rate is a difference over time.
// The log parser still overwrites both with the exact figures at the end.
func slotsPoller(inst *Instance) {
	url := fmt.Sprintf("http://127.0.0.1:%d/slots", inst.Port)
	client := &http.Client{Timeout: time.Second, Transport: directTransport()}
	defer client.CloseIdleConnections()

	var prevT time.Time
	var prevDec, prevPP float64
	havePrev, prevBusy := false, false

	tick := time.NewTicker(time.Second)
	defer tick.Stop()
	for {
		select {
		case <-inst.done:
			return
		case <-tick.C:
		}

		slots, err := fetchSlots(client, url)
		if err != nil {
			// --no-slots, a backend that has no such endpoint, or a restart.
			havePrev, prevBusy = false, false
			inst.update(func(s *Stats) { s.SlotsBusy, s.SlotsPrefill = 0, 0 })
			continue
		}

		// Context fill, live. The log only says it once the request is released, and
		// n_ctx is what the engine actually ended up with rather than what we asked for.
		// The fullest slot, not the sum: every slot has its own window of this size,
		// so the gauge reads against one of them. Identical while there is one slot.
		ctxUsed, ctxLimit := 0.0, 0.0
		busySlots, prefill := 0, 0
		dec, pp := 0.0, 0.0
		for _, sl := range slots {
			ctxUsed = math.Max(ctxUsed, num(sl, "n_prompt_tokens"))
			ctxLimit = math.Max(ctxLimit, num(sl, "n_ctx"))
			decoded := num(sl, "n_decoded")
			if nt, ok := sl["next_token"].([]any); ok && len(nt) > 0 {
				if first, ok := nt[0].(map[string]any); ok {
					decoded += num(first, "n_decoded")
				}
			}
			if b, _ := sl["is_processing"].(bool); b {
				busySlots++
				// Nothing written yet: the prompt is still going in.
				if decoded == 0 {
					prefill++
				}
			}
			dec += decoded
			pp += num(sl, "n_prompt_tokens_processed")
		}
		busy := busySlots > 0
		inst.update(func(s *Stats) {
			s.SlotsBusy, s.SlotsPrefill = busySlots, prefill
			if ctxUsed > 0 {
				s.CtxUsed = int(ctxUsed)
			}
			if ctxLimit > 0 {
				s.CtxLimit = int(ctxLimit)
			}
		})

		now := time.Now()
		// Both ends of the window must fall inside the request. Otherwise the first
		// window also covers the idle time before it and reports far too little: a
		// prefill that takes a tenth of a second looked like a whole second of it.
		if busy && prevBusy && havePrev {
			// A counter that dropped means a new request started: rebase, do not
			// report a negative rate.
			if dt := now.Sub(prevT).Seconds(); dt > 0 {
				inst.update(func(s *Stats) {
					if dec > prevDec {
						s.GenSpeed = (dec - prevDec) / dt
					}
					if pp > prevPP {
						s.PromptSpeed = (pp - prevPP) / dt
					}
				})
			}
		}
		prevT, prevDec, prevPP, prevBusy, havePrev = now, dec, pp, busy, true
	}
}

func fetchSlots(client *http.Client, url string) ([]map[string]any, error) {
	resp, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var slots []map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&slots); err != nil {
		return nil, err
	}
	if slots == nil {
		return nil, fmt.Errorf("no slots")
	}
	return slots, nil
}
