---
name: playwright-local-webgl-capture
description: Reliable headless Playwright capture for locally running WebGL and Three.js apps, especially Next.js dev servers inside WSL. Use when Codex needs to open a local app, prefer localhost over 127.0.0.1 for dev-resource and HMR correctness, force software WebGL, wait for heavy scenes to settle, capture a screenshot of the rendered canvas, and return browser logs.
---

# Playwright Local WebGL Capture

Use this skill to verify a local WebGL scene in headless Playwright without re-deriving the working launch configuration.

## Quick Start

Run the durable capture script from the repo root:

```bash
node skills/playwright-local-webgl-capture/scripts/capture_local_webgl.mjs
```

Default behavior:

- Open `http://localhost:3000`
- Use Playwright's default Chromium resolution from `~/.cache/ms-playwright`
- Force Playwright temp/runtime state into repo-local `tmp/playwright-runtime`
- Check whether the target server is reachable before launching the browser
- Start `bun dev` automatically if the target URL is down
- Force software WebGL with ANGLE + SwiftShader
- Wait for a visible `canvas`
- Wait for a default ready control of `[aria-label="Play"], [aria-label="Pause"]`
- Wait an additional 45 seconds for heavy scenes to settle
- Capture the rendered `canvas` into `tmp/playwright-localhost-3000.png`
- Write logs to `tmp/playwright-localhost-3000.log`

## Workflow

1. Confirm the local app is already running.
2. Prefer `http://localhost:3000` over `http://127.0.0.1:3000` for Next.js dev servers.
3. Launch Playwright with `chromium.launch({ headless: true })`.
4. Let the script manage `TMPDIR`, `TMP`, `TEMP`, and XDG directories inside the repo so the run stays sandbox-friendly.
5. Do not set `PLAYWRIGHT_BROWSERS_PATH=0` unless the task explicitly requires a repo-local browser bundle.
6. Use the script instead of an inline one-off snippet.
7. After the script finishes, inspect the JSON summary, then read the log file if the user wants logs or if the run failed.
8. If the user wants the image attached or you need to inspect it visually, call `view_image` on the screenshot path.

## Command Patterns

Deterministic server check without launching the browser:

```bash
node skills/playwright-local-webgl-capture/scripts/capture_local_webgl.mjs --check-only
```

Default capture:

```bash
node skills/playwright-local-webgl-capture/scripts/capture_local_webgl.mjs
```

Custom URL and longer settle window:

```bash
node skills/playwright-local-webgl-capture/scripts/capture_local_webgl.mjs \
  --url http://localhost:3001 \
  --settle-ms 90000
```

Custom readiness selector and titles:

```bash
node skills/playwright-local-webgl-capture/scripts/capture_local_webgl.mjs \
  --ready-selector '[data-testid="scene-ready"]' \
  --ready-titles 'Ready,Loaded'
```

Skip readiness control and rely on canvas + settle timing:

```bash
node skills/playwright-local-webgl-capture/scripts/capture_local_webgl.mjs --skip-ready
```

Custom artifact paths:

```bash
node skills/playwright-local-webgl-capture/scripts/capture_local_webgl.mjs \
  --screenshot tmp/my-scene.png \
  --log tmp/my-scene.log
```

Custom server command and server log:

```bash
node skills/playwright-local-webgl-capture/scripts/capture_local_webgl.mjs \
  --url http://localhost:3001 \
  --dev-command 'bun dev -- --port 3001' \
  --server-log tmp/my-bun-dev.log
```

## Script Behavior

The script does the durable parts that were easy to get wrong manually:

- Use `localhost` by default
- Use Playwright's default cached Chromium path instead of overriding `executablePath`
- Put Playwright temp artifacts, browser profiles, and XDG state under repo-local `tmp/playwright-runtime`
- Probe the target URL first and auto-run `bun dev` when the server is down
- Leave a server running if the script launched it
- Apply the working headless WebGL flags:
  - `--use-gl=angle`
  - `--use-angle=swiftshader`
  - `--enable-webgl`
  - `--ignore-gpu-blocklist`
- Wait for a real, visible `canvas`
- Prefer `locator("canvas").screenshot()` over `page.screenshot()` for heavy scenes
- Record console logs, page errors, request failures, HTTP 4xx/5xx responses, and tracked resource fetches
- Sanitize local filesystem paths in emitted logs

## Options

Supported script options:

- `--url <url>`: Target URL. Default: `http://localhost:3000`
- `--screenshot <path>`: Screenshot output path. Default: derived under `tmp/`
- `--log <path>`: Log output path. Default: derived under `tmp/`
- `--server-log <path>`: Server log path for auto-started `bun dev`. Default: `tmp/playwright-runtime/bun-dev.log`
- `--viewport <width>x<height>`: Viewport size. Default: `1600x900`
- `--timeout-ms <ms>`: Selector and navigation timeout. Default: `90000`
- `--server-timeout-ms <ms>`: Time to wait for auto-started server readiness. Default: `120000`
- `--settle-ms <ms>`: Extra post-ready wait. Default: `45000`
- `--screenshot-timeout-ms <ms>`: Canvas screenshot timeout. Default: `120000`
- `--canvas-selector <selector>`: Canvas locator. Default: `canvas`
- `--ready-selector <selector>`: Optional readiness control selector. Default: `[aria-label="Play"], [aria-label="Pause"]`
- `--ready-titles <csv>`: Optional accepted `title` values for the readiness control. Default: `Play,Pause`
- `--skip-ready`: Skip readiness-control waiting
- `--check-only`: Only check server reachability and print JSON, without launching Playwright
- `--dev-command <command>`: Override the auto-start command. Default: `bun dev`, or `bun dev -- --port <port>` when the target URL uses a non-3000 port
- `--track-url-substring <text>`: Log matching resource requests. Default: `earth-day.jpg`

## Failure Handling

If the run fails:

- Read the JSON summary printed by the script
- Read the log file for console and network details
- Read the server log if the script auto-started `bun dev`
- Use the failure screenshot artifact if one was captured
- Check whether the page mounted a `canvas`
- Check whether HMR connected on `localhost`
- Increase `--settle-ms` for very heavy scenes before changing the browser setup
- If startup fails while another `next dev` is already running for the same repo, treat that as a workspace lock issue rather than a Playwright issue

## Resources

### scripts/

Use [`scripts/capture_local_webgl.mjs`](scripts/capture_local_webgl.mjs) as the reusable capture entrypoint.
