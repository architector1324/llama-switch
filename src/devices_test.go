package main

import "testing"

func TestParseDeviceList(t *testing.T) {
	devices := parseDeviceList(`ggml_vulkan: Found 2 Vulkan devices:
Available devices:
  Vulkan0: AMD Radeon RX 9060 XT (RADV GFX1200) (16304 MiB, 2048 MiB free)
  Vulkan1: AMD Radeon RX 9060 XT (RADV GFX1200) (16304 MiB, 5497 MiB free)
`)
	if len(devices) != 2 {
		t.Fatalf("%d devices: %+v", len(devices), devices)
	}
	d := devices[1]
	if d.ID != "Vulkan1" || d.Name != "AMD Radeon RX 9060 XT (RADV GFX1200)" || d.TotalMiB != 16304 || d.FreeMiB != 5497 {
		t.Errorf("parsed %+v", d)
	}
	// JSON [] rather than null for the picker.
	if empty := parseDeviceList("no devices\n"); empty == nil || len(empty) != 0 {
		t.Errorf("empty listing %#v", empty)
	}
}

func TestPCIAddressesLabelDevices(t *testing.T) {
	pci := parsePCIAddresses(`GPU0:
	deviceUUID         = 00000000-0600-0000-0000-000000000000
GPU1:
	deviceUUID         = 00000000-0300-0000-0000-000000000000
`)
	if len(pci) != 2 || pci[0] != "0000:06:00.0" || pci[1] != "0000:03:00.0" {
		t.Fatalf("addresses %v", pci)
	}

	devices := []Device{{ID: "Vulkan0"}, {ID: "Vulkan1"}}
	labelDevices(devices, pci)
	if devices[0].PCI == nil || *devices[0].PCI != "0000:06:00.0" {
		t.Errorf("Vulkan0 labelled %v", devices[0].PCI)
	}

	// The tools disagree about the cards: no label beats a wrong one.
	devices = []Device{{ID: "Vulkan0"}}
	labelDevices(devices, pci)
	if devices[0].PCI != nil {
		t.Errorf("mismatched count still labelled %s", *devices[0].PCI)
	}
}

func TestDeviceFlags(t *testing.T) {
	const sdDefault = "--backend diffusion=Vulkan1,te=Vulkan0,vae=Vulkan0"
	for _, tc := range []struct {
		devices, kind, def, want string
	}{
		// Nothing picked keeps what the profile says.
		{"", "llm", "", ""},
		{"", "sd", sdDefault, sdDefault},
		{"", "tts", "--backend vulkan --device 0", "--backend vulkan --device 0"},

		{"Vulkan1", "llm", "", "-dev Vulkan1"},
		{"Vulkan0,Vulkan1", "llm", "", "-dev Vulkan0,Vulkan1"},
		{"Vulkan1", "decision", "", "-dev Vulkan1"},

		// A bare card moves every role the profile names.
		{"Vulkan1", "sd", sdDefault, "--backend diffusion=Vulkan1,te=Vulkan1,vae=Vulkan1"},
		// Named roles override only themselves, and keep the profile's order.
		{"te=Vulkan1", "sd", sdDefault, "--backend diffusion=Vulkan1,te=Vulkan1,vae=Vulkan0"},
		{"vae=Vulkan1, diffusion=Vulkan0", "sd", sdDefault, "--backend diffusion=Vulkan0,te=Vulkan0,vae=Vulkan1"},
		{"clip=Vulkan1", "sd", sdDefault, "--backend diffusion=Vulkan1,te=Vulkan0,vae=Vulkan0,clip=Vulkan1"},
		{"Vulkan0", "sd", "", "--backend Vulkan0"},

		// audio.cpp takes a backend and an index within it.
		{"Vulkan1", "tts", "", "--backend vulkan --device 1"},
		{"CUDA0", "stt", "", "--backend cuda --device 0"},
		{"Vulkan12", "music", "", "--backend vulkan --device 12"},
		{"weird", "music", "--fallback", "--fallback"},
	} {
		if got := deviceFlags(tc.devices, tc.kind, tc.def); got != tc.want {
			t.Errorf("deviceFlags(%q, %s, %q) = %q, want %q", tc.devices, tc.kind, tc.def, got, tc.want)
		}
	}
}
