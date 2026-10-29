//go:build !windows

package main

import (
	"os"
	"os/exec"
	"syscall"
)

// A profile line in a session of its own, so its whole group can be signalled.
func shellCommand(line string) *exec.Cmd {
	cmd := exec.Command("/bin/sh", "-c", line)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	return cmd
}

// SIGTERM, or SIGKILL when forced, to the group setsid made p the leader of.
func killTree(p *os.Process, force bool) error {
	sig := syscall.SIGTERM
	if force {
		sig = syscall.SIGKILL
	}
	return syscall.Kill(-p.Pid, sig)
}
