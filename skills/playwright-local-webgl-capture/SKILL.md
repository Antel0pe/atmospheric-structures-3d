---
name: playwright-local-webgl-capture
description: Reliable headless Playwright capture for this repo's locally running WebGL app. Use when Codex needs one command that starts the local app if needed, opens the automation capture route, waits for the paused ready state, captures a screenshot quickly, and returns timings plus browser logs.
---

# Playwright Local WebGL Capture

Use this skill to verify the local WebGL scene in headless Playwright without re-deriving the working launch configuration. The repo now exposes an automation capture path that pauses rendering once the scene is ready, which avoids the slow live-WebGL screenshot path.

For interaction and state restoration, prefer the runtime automation API on `window.__ATMOS_AUTOMATION__`. It exposes stable methods for movement, time stepping, saved views, and direct view restore, so you do not need to synthesize keyboard input or guess selectors.

## Quick Start

Run the durable capture script from the repo root:

```bash
bun run capture:webgl
```

Default behavior:

- Open `http://localhost:3000/?automation=1`
- Use Playwright's default Chromium resolution
- Force Playwright temp/runtime state into repo-local `tmp/playwright-runtime`
- Check whether the target server is reachable before launching the browser
- Start `bun dev --hostname localhost --port 3000` automatically if the target URL is down
- Use ANGLE + SwiftShader by default for stable headless WebGL logs
- Wait for a visible `canvas`
- Wait for `window.__ATMOS_AUTOMATION__.paused`
- Capture a full browser screenshot into `tmp/playwright-localhost-3000.png`
- Write logs to `tmp/playwright-localhost-3000.log`
- Print timing breakdowns for launch, ready, pause, render, and screenshot

## Workflow

1. Prefer `bun run capture:webgl` over an inline Playwright snippet.
2. Prefer `http://localhost:3000/?automation=1` over `127.0.0.1` for this repo.
3. Let the script manage `TMPDIR`, `TMP`, `TEMP`, and XDG directories inside the repo so the run stays sandbox-friendly.
4. Let the script auto-start the local Next.js dev server when the app is down.
5. Use the automation route by default. It pauses rendering once the scene is ready so screenshots become fast again.
6. After Playwright opens the page, wait for the automation object with `await page.waitForFunction(() => Boolean(window.__ATMOS_AUTOMATION__))`.
7. Then call `await page.evaluate(() => window.__ATMOS_AUTOMATION__.waitForReady())` before interacting.
8. Read the control manifest with `await page.evaluate(() => window.__ATMOS_AUTOMATION__.describe())` if you need the exact buttons/selectors.
9. Prefer automation API methods over DOM clicks for view changes and saved-view operations.
10. After the script finishes, inspect the JSON summary first, then the log file if the user wants browser logs or the run failed.
11. If the user wants the image attached or you need to inspect it visually, call `view_image` on the screenshot path.

## Default Interaction Pattern

Use this exact sequence when you need to drive the viewer reliably:

1. Open `http://localhost:3000/?automation=1`.
2. Wait for `window.__ATMOS_AUTOMATION__` to exist.
3. Wait for `window.__ATMOS_AUTOMATION__.waitForReady()`.
4. Call `window.__ATMOS_AUTOMATION__.describe()` once to inspect the available controls and selectors.
5. Use `runNavigationCommand(...)`, `stepTime(...)`, `setTimestamp(...)`, `saveView(...)`, `applySavedView(...)`, or `applyViewState(...)`.
6. Use DOM selectors only when you specifically need to verify visible buttons/fields.

Minimal Playwright example:

```js
await page.goto("http://localhost:3000/?automation=1", { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => Boolean(window.__ATMOS_AUTOMATION__));
await page.evaluate(() => window.__ATMOS_AUTOMATION__.waitForReady());

const manifest = await page.evaluate(() => window.__ATMOS_AUTOMATION__.describe());

await page.evaluate(() =>
  window.__ATMOS_AUTOMATION__.runNavigationCommand("move-left")
);

await page.evaluate(() =>
  window.__ATMOS_AUTOMATION__.stepTime("forward")
);

const savedView = await page.evaluate(() =>
  window.__ATMOS_AUTOMATION__.saveView({
    title: "Skill Example",
    description: "Saved from the Playwright skill",
  })
);

await page.evaluate((id) =>
  window.__ATMOS_AUTOMATION__.applySavedView({ id }),
savedView.id);
```

## Runtime Automation API

Available on every viewer page as `window.__ATMOS_AUTOMATION__`.

Properties:

- `enabled`: `true` only when the page was opened with `?automation=1`
- `paused`: whether automation capture mode is currently paused on a ready frame
- `ready`: whether the globe texture and registered layers are ready for the current timestamp

Read methods:

- `describe()`: returns the control manifest, selectors, recommended URL, and current snapshot
- `getSnapshot()`: returns `{ ready, paused, timestamp, zoom01, earthView, savedViews }`
- `waitForReady(timeoutMs?)`: resolves once the current timestamp is fully ready

Camera and time methods:

- `runNavigationCommand(command)`: runs the same action as the dev sidebar movement/look buttons
- `stepTime("backward" | "forward", timeoutMs?)`: clicks the existing time-step buttons and waits for readiness
- `setTimestamp(timestamp, timeoutMs?)`: sets a timestamp directly and waits for readiness
- `applyViewState({ timestamp?, earthView }, timeoutMs?)`: applies an exact serialized view and re-applies after timestamp readiness when needed

Saved-view methods:

- `listSavedViews()`
- `saveView({ title, description? })`
- `applySavedView({ id? , title? }, timeoutMs?)`
- `deleteSavedView({ id? , title? })`

UI helper methods:

- `ensureLayersSidebarOpen()`: opens the left layers/dev sidebar if it is closed
- `freeze()`, `resume()`, `renderOnce()`, `capturePngDataUrl()`

## Available Buttons And Selectors

Navigation buttons in the dev viewer pane:

- `viewer-move-forward`: `Move forward`
- `viewer-move-backward`: `Move backward`
- `viewer-move-left`: `Move left`
- `viewer-move-right`: `Move right`
- `viewer-move-up`: `Move up`
- `viewer-move-down`: `Move down`
- `viewer-look-left`: `Look left`
- `viewer-look-right`: `Look right`
- `viewer-look-up`: `Look up`
- `viewer-look-down`: `Look down`

Time controls:

- `time-step-backward`: `Step backward time`
- `time-step-forward`: `Step forward time`

Saved-view controls:

- Left sidebar toggle: aria-label `Open layers` or `Close layers`
- `dev-viewer-pane`: dev viewer section
- `viewer-current-zoom`
- `viewer-current-timestamp`
- `saved-view-title`: `Saved view title`
- `saved-view-description`: `Saved view description`
- `saved-view-save`: `Save current view`
- `saved-view-refresh`: `Refresh saved views`
- `saved-views-list`: `Saved views list`
- Apply button pattern: aria-label `Apply saved view <title>`, test id `saved-view-apply-<id>`
- Delete button pattern: aria-label `Delete saved view <title>`, test id `saved-view-delete-<id>`

## Guidance

- Prefer `runNavigationCommand(...)` over keyboard input when you want deterministic movement in automation.
- Prefer `applySavedView(...)` or `applyViewState(...)` over manual multi-step DOM replay when you need an exact view.
- Prefer `stepTime(...)` over clicking the time buttons manually unless the test specifically needs DOM assertions.
- Use `describe()` first when you are unsure which control names are available.
- If you do need visible UI assertions, call `ensureLayersSidebarOpen()` first so the dev pane is guaranteed to be accessible.

## Command Patterns

Deterministic server check without launching the browser:

```bash
bun run capture:webgl -- --check-only
```

Default capture:

```bash
bun run capture:webgl
```

Scene-only export from the renderer without browser chrome:

```bash
bun run capture:webgl -- --scene-only
```

Custom URL:

```bash
bun run capture:webgl -- --url http://localhost:3001
```

Fallback to the older non-automation capture path:

```bash
bun run capture:webgl -- --no-automation
```

Custom artifact paths:

```bash
bun run capture:webgl -- \
  --screenshot tmp/my-scene.png \
  --log tmp/my-scene.log
```

Custom server command and server log:

```bash
bun run capture:webgl -- \
  --url http://localhost:3001 \
  --dev-command 'bun dev --hostname localhost --port 3001' \
  --server-log tmp/my-bun-dev.log
```

## Script Behavior

The script does the durable parts that were easy to get wrong manually:

- Use `localhost` by default
- Add `?automation=1` by default so the repo enables fast automation capture mode
- Use Playwright's default cached Chromium path instead of overriding `executablePath`
- Put Playwright temp artifacts, browser profiles, and XDG state under repo-local `tmp/playwright-runtime`
- Probe the target URL first and auto-run `bun dev` when the server is down
- Leave a server running if the script launched it
- Launch headless Chromium with WebGL enabled and an optional `--swiftshader` fallback
- Launch headless Chromium with WebGL enabled and SwiftShader by default for deterministic CI-style runs
- Wait for a real, visible `canvas`
- Wait for the repo's automation pause hook, which only flips true after the globe texture and registered layers are ready
- Support renderer-only export with `--scene-only`
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
- `--settle-ms <ms>`: Extra post-ready wait for non-automation fallback mode. Default: `45000`
- `--screenshot-timeout-ms <ms>`: Canvas screenshot timeout. Default: `120000`
- `--canvas-selector <selector>`: Canvas locator. Default: `canvas`
- `--ready-selector <selector>`: Optional readiness control selector for non-automation fallback mode. Default: `[aria-label="Play"], [aria-label="Pause"]`
- `--ready-titles <csv>`: Optional accepted `title` values for the readiness control. Default: `Play,Pause`
- `--skip-ready`: Skip readiness-control waiting in non-automation fallback mode
- `--check-only`: Only check server reachability and print JSON, without launching Playwright
- `--dev-command <command>`: Override the auto-start command. Default: `bun dev --hostname localhost --port <port>`
- `--track-url-substring <text>`: Log matching resource requests. Default: `earth-day.jpg`
- `--no-automation`: Disable the repo's automation capture route and use the slower fallback path
- `--scene-only`: Use `window.__ATMOS_AUTOMATION__.capturePngDataUrl()` instead of a full browser screenshot
- `--swiftshader`: Force ANGLE + SwiftShader explicitly
- `--native-gpu`: Opt out of the default SwiftShader path and use the native GPU path

## Failure Handling

If the run fails:

- Read the JSON summary printed by the script
- Read the log file for console and network details
- Read the server log if the script auto-started `bun dev`
- Use the failure screenshot artifact if one was captured
- Check whether the page mounted a `canvas`
- Check whether `window.__ATMOS_AUTOMATION__` was reached and whether `paused` ever became `true`
- Try `--native-gpu` only when you specifically need to debug the host GPU path
- Use `--no-automation` only when the repo automation hook is unavailable
- If startup fails while another `next dev` is already running for the same repo, treat that as a workspace lock issue rather than a Playwright issue

## Resources

### scripts/

Use [`scripts/capture_local_webgl.mjs`](scripts/capture_local_webgl.mjs) as the reusable capture entrypoint.
