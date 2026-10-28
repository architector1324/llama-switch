package main

import (
	"context"
	"embed"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// The dashboard ships inside the binary, so it runs from any directory.
//
//go:embed static templates
var assets embed.FS

// Browser clients on other origins need CORS, same as llama.cpp grants.
func cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Origin") == "" {
			next.ServeHTTP(w, r)
			return
		}
		h := w.Header()
		h.Set("Access-Control-Allow-Origin", "*")
		if r.Method == http.MethodOptions && r.Header.Get("Access-Control-Request-Method") != "" {
			h.Set("Access-Control-Allow-Methods", "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT")
			if asked := r.Header.Get("Access-Control-Request-Headers"); asked != "" {
				h.Set("Access-Control-Allow-Headers", asked)
			}
			h.Set("Access-Control-Max-Age", "600")
			w.WriteHeader(http.StatusOK)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func routes() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /api/config", handleConfig)
	mux.HandleFunc("GET /v1/models", handleV1Models)
	mux.HandleFunc("GET /api/status", handleStatus)
	mux.HandleFunc("POST /api/multi", handleMulti)
	mux.HandleFunc("GET /api/devices", handleDevices)
	mux.HandleFunc("GET /api/logs", handleLogs)
	mux.HandleFunc("GET /api/logs/owner", handleLogsOwner)
	mux.HandleFunc("POST /api/logs/clear", handleLogsClear)
	mux.HandleFunc("POST /api/stop", handleStop)
	mux.HandleFunc("POST /api/reset", handleReset)
	mux.HandleFunc("POST /api/start", handleStart)

	mux.HandleFunc("POST /v1/chat/completions", proxyToLlama)
	mux.HandleFunc("POST /v1/completions", proxyToLlama)
	mux.HandleFunc("POST /v1/responses", proxyToLlama)
	mux.HandleFunc("POST /v1/systemone", proxySystemOne)
	for _, method := range []string{"GET", "POST"} {
		mux.HandleFunc(method+" /v1/images/{path...}", proxyImages)
		mux.HandleFunc(method+" /sdapi/v1/{path...}", proxyImages)
		mux.HandleFunc(method+" /sdcpp/v1/{path...}", proxyImages)
		mux.HandleFunc(method+" /v1/tasks/{path...}", proxyTasks)
	}
	for _, method := range []string{"GET", "POST", "DELETE"} {
		mux.HandleFunc(method+" /v1/audio/{path...}", proxyAudio)
	}

	// Engine UIs are not republished here. They call their own paths absolutely, and
	// those collide with this server's; "Open WebUI" points straight at the engine port.

	static, _ := fs.Sub(assets, "static")
	files := http.StripPrefix("/static", http.FileServerFS(static))
	mux.HandleFunc("GET /static/", func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/") {
			http.NotFound(w, r) // files only, no directory listings
			return
		}
		files.ServeHTTP(w, r)
	})

	index, _ := assets.ReadFile("templates/index.html")
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(index)
	})

	return cors(mux)
}

func main() {
	var (
		host, configPath    string
		port, ctx, parallel int
		watch               bool
	)
	flag.StringVar(&host, "H", "localhost", "interface for the UI and for the engines it launches")
	flag.StringVar(&host, "host", "localhost", "same as -H")
	flag.IntVar(&port, "p", 11435, "UI port (engines get a free port each)")
	flag.IntVar(&port, "port", 11435, "same as -p")
	flag.IntVar(&ctx, "c", 4096, "default context window, per slot")
	flag.IntVar(&ctx, "ctx", 4096, "same as -c")
	flag.IntVar(&parallel, "np", 1, "default server slots per model")
	flag.IntVar(&parallel, "parallel", 1, "same as -np")
	flag.StringVar(&configPath, "f", "config.yaml", "config file path")
	flag.StringVar(&configPath, "config", "config.yaml", "same as -f")
	flag.BoolVar(&watch, "w", false, "reload the config when it changes on disk")
	flag.BoolVar(&watch, "watch", false, "same as -w")
	flag.Parse()

	state.defaultCtx = ctx
	state.defaultParallel = parallel
	state.host = host
	state.cfg = newConfigManager(configPath, watch, onConfigChange)

	srv := &http.Server{
		Addr:    net.JoinHostPort(host, strconv.Itoa(port)),
		Handler: routes(),
	}

	sigCtx, stopSignals := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stopSignals()

	fmt.Printf("Starting UI on http://%s:%d\n", host, port)
	served := make(chan error, 1)
	go func() { served <- srv.ListenAndServe() }()

	select {
	case err := <-served:
		if !errors.Is(err, http.ErrServerClosed) {
			fmt.Printf("[Service] %v\n", err)
			os.Exit(1)
		}
	case <-sigCtx.Done():
	}
	// A second Ctrl+C kills outright.
	stopSignals()

	fmt.Println("\n[Service] Shutting down... cleaning up processes")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	srv.Shutdown(shutdownCtx)

	opMu.Lock()
	stopAll()
	opMu.Unlock()
}
