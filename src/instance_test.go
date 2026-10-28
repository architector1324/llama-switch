package main

import (
	"net/http"
	"testing"
)

func TestExpandProfile(t *testing.T) {
	base := profileValues{
		Kind: "llm", Host: "0.0.0.0", Port: 4242, Ctx: 4096, Parallel: 2, Frames: 2048,
		Width: "832", Height: "1216",
	}
	with := func(edit func(v *profileValues)) profileValues {
		v := base
		edit(&v)
		return v
	}
	for _, tc := range []struct {
		tmpl string
		v    profileValues
		want string
	}{
		// The pool is the per-slot window times the slots.
		{"s --host ${HOST} --port ${PORT} -c ${CTX} --parallel ${PARALLEL} ${PRESERVE_THINK}", base,
			"s --host 0.0.0.0 --port 4242 -c 8192 --parallel 2 --no-reasoning-preserve"},
		{"s ${PRESERVE_THINK}", with(func(v *profileValues) { v.PreserveThink = true }), "s --reasoning-preserve"},
		{"s $HOST $PORT $CTX $PARALLEL $FRAMES $WIDTH $HEIGHT $PRESERVE_THINK", base,
			"s 0.0.0.0 4242 8192 2 2048 832 1216 --no-reasoning-preserve"},
		{"s -W ${WIDTH} -H ${HEIGHT} --max-new ${FRAMES}", base, "s -W 832 -H 1216 --max-new 2048"},

		// Hires: on puts the profile's own flags in, off removes the placeholder.
		{"s ${HIRES:--hires --strength 0.5} ${HIRES}", with(func(v *profileValues) { v.Hires = true }),
			"s --hires --strength 0.5 --hires"},
		{"s ${HIRES:--hires --strength 0.5} ${HIRES}.", base, "s  ."},

		{"s ${DEVICES}.", base, "s ."},
		{"s ${DEVICES:--fit on}", base, "s --fit on"},
		{"s ${DEVICES}", with(func(v *profileValues) { v.Devices = "Vulkan1" }), "s -dev Vulkan1"},
		{"s ${DEVICES:--backend diffusion=Vulkan1,vae=Vulkan0}",
			with(func(v *profileValues) { v.Kind, v.Devices = "sd", "vae=Vulkan1" }),
			"s --backend diffusion=Vulkan1,vae=Vulkan1"},

		// Shell syntax around a placeholder is left for the shell.
		{`sed "s/@CTX@/$((2*${FRAMES}+1024))/"`, base, `sed "s/@CTX@/$((2*2048+1024))/"`},
	} {
		if got := expandProfile(tc.tmpl, tc.v); got != tc.want {
			t.Errorf("expandProfile(%q)\n got %q\nwant %q", tc.tmpl, got, tc.want)
		}
	}
}

func TestLookupAndResolveModel(t *testing.T) {
	resetState()
	useConfig(t, testConfig(), false)

	for _, tc := range []struct {
		id, kind, key, quant string
		ok                   bool
	}{
		{"zeta", "llm", "zeta", "q4", true},
		{"zeta-q8", "llm", "zeta", "q8", true},
		{"zeta-q4", "llm", "", "", false}, // the default has the bare id only
		{"legacy", "llm", "legacy", "", true},
		{"legacy-cmd", "llm", "", "", false},
		{"song-sfx", "music", "song", "sfx", true},
		{"song-sfx", "llm", "", "", false},
		{"ears", "stt", "ears", "f16", true},
	} {
		key, quant, ok := lookupModel(tc.id, tc.kind)
		if key != tc.key || quant != tc.quant || ok != tc.ok {
			t.Errorf("lookupModel(%s, %s) = %q %q %v", tc.id, tc.kind, key, quant, ok)
		}
	}

	if _, _, err := resolveModel("img", "llm"); !isHTTPErr(err, http.StatusConflict,
		"Model img is a 'sd' model, not 'llm'; it is served at /v1/images/generations") {
		t.Errorf("wrong section: %v", err)
	}
	if _, _, err := resolveModel("nope", "sd"); !isHTTPErr(err, http.StatusNotFound, "Model nope not found") {
		t.Errorf("unknown: %v", err)
	}
}

func isHTTPErr(err error, status int, detail string) bool {
	he, ok := err.(*httpErr)
	return ok && he.status == status && he.detail == detail
}
