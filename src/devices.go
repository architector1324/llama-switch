package main

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Vulkan0: AMD Radeon RX 9060 XT (RADV GFX1200) (16304 MiB, 15586 MiB free)
var reDevice = regexp.MustCompile(`^\s*(\S+):\s*(.+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)\s*$`)

// vulkaninfo encodes the PCI address in deviceUUID: 00000000-0600-... is bus 0x06.
var reVkUUID = regexp.MustCompile(`deviceUUID\s*=\s*[0-9a-f]{8}-([0-9a-f]{2})([0-9a-f]{2})-`)

type Device struct {
	ID       string  `json:"id"`
	Name     string  `json:"name"`
	TotalMiB int     `json:"total_mib"`
	FreeMiB  int     `json:"free_mib"`
	PCI      *string `json:"pci"`
}

var (
	devicesMu    sync.Mutex
	devicesCache []Device // nil until the first successful listing
)

// Stdout of a command, even when it exits non-zero; an error only when it could
// not be run at all or ran out of time.
func runOutput(timeout time.Duration, name string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).Output()
	if ctx.Err() != nil {
		return "", fmt.Errorf("%s timed out after %s", name, timeout)
	}
	var exitErr *exec.ExitError
	if err != nil && !errors.As(err, &exitErr) {
		return "", err
	}
	return string(out), nil
}

func splitLines(s string) []string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	return strings.FieldsFunc(s, func(r rune) bool { return r == '\n' || r == '\r' })
}

// PCI address per Vulkan device, in the order Vulkan enumerates them.
//
// Two identical cards are indistinguishable by name, so the bus address is the
// only thing that says which is which. Best-effort: no vulkaninfo, no addresses.
func devicePCIAddresses() []string {
	out, err := runOutput(15*time.Second, "vulkaninfo", "--summary")
	if err != nil {
		return nil
	}
	return parsePCIAddresses(out)
}

func parsePCIAddresses(out string) []string {
	var addrs []string
	for _, m := range reVkUUID.FindAllStringSubmatch(out, -1) {
		addrs = append(addrs, fmt.Sprintf("0000:%s:%s.0", m[1], m[2]))
	}
	return addrs
}

// Offload devices as the backend itself enumerates them.
//
// Asking llama-server keeps the list in step with whatever backend the build
// carries -- Vulkan here, CUDA elsewhere -- instead of hardcoding one vendor.
func listDevices(refresh bool) []Device {
	devicesMu.Lock()
	defer devicesMu.Unlock()
	if devicesCache != nil && !refresh {
		return devicesCache
	}

	out, err := runOutput(60*time.Second, "llama-server", "--list-devices")
	if err != nil {
		fmt.Printf("[Service] Could not list devices: %v\n", err)
		return []Device{}
	}
	devices := parseDeviceList(out)
	labelDevices(devices, devicePCIAddresses())
	devicesCache = devices
	return devices
}

func parseDeviceList(out string) []Device {
	devices := []Device{}
	for _, line := range splitLines(out) {
		m := reDevice.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		total, _ := strconv.Atoi(m[3])
		free, _ := strconv.Atoi(m[4])
		devices = append(devices, Device{ID: m[1], Name: m[2], TotalMiB: total, FreeMiB: free})
	}
	return devices
}

// Addresses go on only when there is one per device: a count that differs means
// the two tools disagree about the cards, and a wrong label is worse than none.
func labelDevices(devices []Device, pci []string) {
	if len(pci) != len(devices) {
		return
	}
	for i := range devices {
		addr := pci[i]
		devices[i].PCI = &addr
	}
}

// ${DEVICES} or ${DEVICES:<flags the profile uses by default>}. Every backend
// spells device selection differently, so the flags are built per kind rather
// than written into the profiles.
var reDevices = regexp.MustCompile(`\$\{DEVICES(?::([^}]*))?\}`)

// "Vulkan1" -> ("vulkan", 1), for backends that take a name and an index.
var reDeviceSplit = regexp.MustCompile(`^([A-Za-z_]+)(\d+)$`)

var reRoleAssign = regexp.MustCompile(`([A-Za-z_]+)=([^,\s]+)`)

// Device-selection flags for one backend family.
//
// Empty selection keeps whatever the profile already said, so "all" is exactly
// today's behaviour rather than a guess at it.
func deviceFlags(devices, kind, defaultText string) string {
	if devices == "" {
		return defaultText
	}

	if kind == "llm" || kind == "decision" {
		return "-dev " + devices
	}

	if kind == "sd" {
		// sd-server assigns per module, and which modules exist differs per
		// pipeline, so the roles come from the profile's own default. A selection
		// may name roles ("diffusion=Vulkan1,vae=Vulkan0") and then overrides only
		// those, leaving the rest as the profile had them; a bare device name
		// moves every role to that card.
		var roles []string
		assigned := map[string]string{}
		assign := func(role, dev string) {
			if _, ok := assigned[role]; !ok {
				roles = append(roles, role)
			}
			assigned[role] = dev
		}
		for _, m := range reRoleAssign.FindAllStringSubmatch(defaultText, -1) {
			assign(m[1], m[2])
		}
		if strings.Contains(devices, "=") {
			for _, part := range strings.Split(devices, ",") {
				role, dev, _ := strings.Cut(part, "=")
				role, dev = strings.TrimSpace(role), strings.TrimSpace(dev)
				if role != "" && dev != "" {
					assign(role, dev)
				}
			}
		} else {
			for _, role := range roles {
				assigned[role] = devices
			}
		}
		if len(roles) == 0 {
			return "--backend " + devices
		}
		parts := make([]string, len(roles))
		for i, r := range roles {
			parts[i] = r + "=" + assigned[r]
		}
		return "--backend " + strings.Join(parts, ",")
	}

	// audiocpp-server takes a backend name plus an index within it.
	if m := reDeviceSplit.FindStringSubmatch(devices); m != nil {
		return fmt.Sprintf("--backend %s --device %s", strings.ToLower(m[1]), m[2])
	}
	return defaultText
}
