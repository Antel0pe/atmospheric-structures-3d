import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { chromium } from "playwright";

const cwd = process.cwd();
const home = process.env.HOME ?? "";

function sanitize(value) {
  if (!value) return value;
  let next = String(value);
  if (cwd) next = next.split(cwd).join("<repo>");
  if (home) next = next.split(home).join("~");
  return next;
}

function parseArgs(argv) {
  const options = {
    url: "http://localhost:3000",
    screenshot: "",
    log: "",
    serverLog: "",
    viewport: "1600x900",
    timeoutMs: 90_000,
    serverTimeoutMs: 120_000,
    settleMs: 45_000,
    screenshotTimeoutMs: 120_000,
    canvasSelector: "canvas",
    readySelector: '[aria-label="Play"], [aria-label="Pause"]',
    readyTitles: ["Play", "Pause"],
    skipReady: false,
    checkOnly: false,
    devCommand: "",
    trackUrlSubstring: "earth-day.jpg",
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case "--url":
        options.url = argv[++i];
        break;
      case "--screenshot":
        options.screenshot = argv[++i];
        break;
      case "--log":
        options.log = argv[++i];
        break;
      case "--server-log":
        options.serverLog = argv[++i];
        break;
      case "--viewport":
        options.viewport = argv[++i];
        break;
      case "--timeout-ms":
        options.timeoutMs = Number(argv[++i]);
        break;
      case "--server-timeout-ms":
        options.serverTimeoutMs = Number(argv[++i]);
        break;
      case "--settle-ms":
        options.settleMs = Number(argv[++i]);
        break;
      case "--screenshot-timeout-ms":
        options.screenshotTimeoutMs = Number(argv[++i]);
        break;
      case "--canvas-selector":
        options.canvasSelector = argv[++i];
        break;
      case "--ready-selector":
        options.readySelector = argv[++i];
        break;
      case "--ready-titles":
        options.readyTitles = argv[++i]
          .split(",")
          .map((value) => value.trim())
          .filter(Boolean);
        break;
      case "--skip-ready":
        options.skipReady = true;
        break;
      case "--check-only":
        options.checkOnly = true;
        break;
      case "--dev-command":
        options.devCommand = argv[++i];
        break;
      case "--track-url-substring":
        options.trackUrlSubstring = argv[++i];
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function parseViewport(viewport) {
  const match = /^(\d+)x(\d+)$/i.exec(viewport);
  if (!match) {
    throw new Error(`Invalid viewport '${viewport}'. Expected <width>x<height>.`);
  }
  return { width: Number(match[1]), height: Number(match[2]) };
}

function deriveArtifactPath(kind, urlString) {
  const url = new URL(urlString);
  const host = url.hostname.replace(/[^a-z0-9.-]/gi, "-");
  const port = url.port || (url.protocol === "https:" ? "443" : "80");
  const ext = kind === "screenshot" ? "png" : "log";
  return path.join("tmp", `playwright-${host}-${port}.${ext}`);
}

async function ensureParentDir(filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
}

async function isServerUp(urlString, timeoutMs = 2_500) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(urlString, {
      method: "GET",
      signal: controller.signal,
      redirect: "manual",
    });
    return response.status > 0;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

async function waitForServer(urlString, timeoutMs, intervalMs = 1_000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (await isServerUp(urlString)) return true;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  return false;
}

function defaultDevCommandForUrl(urlString) {
  const url = new URL(urlString);
  const port = url.port || (url.protocol === "https:" ? "443" : "80");
  if (port === "3000") return "bun dev";
  return `bun dev -- --port ${port}`;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const viewport = parseViewport(options.viewport);
  const screenshotPath = options.screenshot || deriveArtifactPath("screenshot", options.url);
  const logPath = options.log || deriveArtifactPath("log", options.url);
  const runtimeRoot = path.join(cwd, "tmp", "playwright-runtime");
  const browserTempRoot = path.join(runtimeRoot, "tmp");
  const xdgConfigHome = path.join(runtimeRoot, "xdg-config");
  const xdgCacheHome = path.join(runtimeRoot, "xdg-cache");
  const xdgDataHome = path.join(runtimeRoot, "xdg-data");
  const serverLogPath = options.serverLog || path.join(runtimeRoot, "bun-dev.log");
  const logs = [];
  const push = (kind, message) => {
    const line = `[${new Date().toISOString()}] ${kind}: ${sanitize(message)}`;
    logs.push(line);
    return line;
  };

  await ensureParentDir(screenshotPath);
  await ensureParentDir(logPath);
  await fs.mkdir(browserTempRoot, { recursive: true });
  await fs.mkdir(xdgConfigHome, { recursive: true });
  await fs.mkdir(xdgCacheHome, { recursive: true });
  await fs.mkdir(xdgDataHome, { recursive: true });
  await ensureParentDir(serverLogPath);

  const launchEnv = {
    ...process.env,
    TMPDIR: browserTempRoot,
    TMP: browserTempRoot,
    TEMP: browserTempRoot,
    XDG_CONFIG_HOME: xdgConfigHome,
    XDG_CACHE_HOME: xdgCacheHome,
    XDG_DATA_HOME: xdgDataHome,
  };

  process.env.TMPDIR = browserTempRoot;
  process.env.TMP = browserTempRoot;
  process.env.TEMP = browserTempRoot;
  process.env.XDG_CONFIG_HOME = xdgConfigHome;
  process.env.XDG_CACHE_HOME = xdgCacheHome;
  process.env.XDG_DATA_HOME = xdgDataHome;

  let browser;
  let context;
  let page;
  let devServer;

  await fs.writeFile(logPath, "", "utf8");

  const appendLog = async (kind, message) => {
    const line = push(kind, message);
    await fs.appendFile(logPath, `${line}\n`, "utf8");
  };

  try {
    await appendLog("browser-path", chromium.executablePath());
    await appendLog("runtime-root", runtimeRoot);

    const serverInitiallyUp = await isServerUp(options.url);
    await appendLog("server-check", serverInitiallyUp ? `${options.url} reachable` : `${options.url} unreachable`);

    if (options.checkOnly) {
      console.log(
        JSON.stringify(
          {
            ok: true,
            mode: "check-only",
            url: options.url,
            reachable: serverInitiallyUp,
            browserPath: sanitize(chromium.executablePath()),
            runtimeRoot: sanitize(runtimeRoot),
            logPath: sanitize(logPath),
          },
          null,
          2
        )
      );
      return;
    }

    if (!serverInitiallyUp) {
      const devCommand = options.devCommand || defaultDevCommandForUrl(options.url);
      await appendLog("server", `starting '${devCommand}' because ${options.url} is not reachable`);
      const serverLogHandle = await fs.open(serverLogPath, "a");
      devServer = spawn("bash", ["-lc", devCommand], {
        cwd,
        detached: true,
        env: launchEnv,
        stdio: ["ignore", serverLogHandle.fd, serverLogHandle.fd],
      });
      devServer.unref();
      await serverLogHandle.close();
      const ready = await waitForServer(options.url, options.serverTimeoutMs);
      if (!ready) {
        throw new Error(`Server command did not make ${options.url} reachable within ${options.serverTimeoutMs}ms. See ${sanitize(serverLogPath)}`);
      }
      await appendLog("server", `server command is serving ${options.url}`);
    } else {
      await appendLog("server", `${options.url} already reachable`);
    }

    browser = await chromium.launch({
      headless: true,
      env: launchEnv,
      args: [
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--enable-webgl",
        "--ignore-gpu-blocklist",
      ],
    });

    context = await browser.newContext({
      viewport,
      deviceScaleFactor: 1,
    });
    page = await context.newPage();

    page.on("console", (msg) => {
      void fs.appendFile(
        logPath,
        `${push(`console.${msg.type()}`, msg.text())}\n`,
        "utf8"
      );
    });
    page.on("pageerror", (err) => {
      void fs.appendFile(logPath, `${push("pageerror", err.stack || err.message)}\n`, "utf8");
    });
    page.on("requestfailed", (req) => {
      const failure = req.failure();
      void fs.appendFile(
        logPath,
        `${push(
          "requestfailed",
          `${req.method()} ${req.url()} ${failure ? failure.errorText : "unknown"}`
        )}\n`,
        "utf8"
      );
    });
    page.on("response", (res) => {
      if (res.status() >= 400) {
        void fs.appendFile(logPath, `${push("response", `${res.status()} ${res.url()}`)}\n`, "utf8");
      }
    });

    if (options.trackUrlSubstring) {
      page.on("request", (req) => {
        if (req.url().includes(options.trackUrlSubstring)) {
          void fs.appendFile(logPath, `${push("tracked-request", req.url())}\n`, "utf8");
        }
      });
      page.on("requestfinished", (req) => {
        if (req.url().includes(options.trackUrlSubstring)) {
          void fs.appendFile(logPath, `${push("tracked-finished", req.url())}\n`, "utf8");
        }
      });
    }

    await page.goto(options.url, {
      waitUntil: "domcontentloaded",
      timeout: options.timeoutMs,
    });
    await appendLog("nav", `loaded ${page.url()}`);
    await appendLog("title", await page.title());

    await page.waitForSelector(options.canvasSelector, {
      state: "visible",
      timeout: options.timeoutMs,
    });
    await page.waitForFunction(
      (selector) => {
        const canvas = document.querySelector(selector);
        if (!(canvas instanceof HTMLCanvasElement)) return false;
        const rect = canvas.getBoundingClientRect();
        return rect.width > 100 && rect.height > 100;
      },
      options.canvasSelector,
      { timeout: options.timeoutMs }
    );
    await appendLog("canvas", "visible and sized");

    if (!options.skipReady && options.readySelector) {
      await page.waitForSelector(options.readySelector, {
        state: "visible",
        timeout: options.timeoutMs,
      });
      await page.waitForFunction(
        ({ selector, titles }) => {
          const node = document.querySelector(selector);
          if (!(node instanceof HTMLElement)) return false;
          const title = node.getAttribute("title") || "";
          return titles.length === 0 ? true : titles.includes(title);
        },
        { selector: options.readySelector, titles: options.readyTitles },
        { timeout: options.timeoutMs }
      );
      await appendLog("ready", "readiness control indicates scene ready");
    }

    await appendLog("settle", `${options.settleMs}ms`);
    await page.waitForTimeout(options.settleMs);

    const domInfo = await page.evaluate(({ canvasSelector, readySelector }) => {
      const canvas = document.querySelector(canvasSelector);
      const rect = canvas?.getBoundingClientRect();
      const readyNode = readySelector ? document.querySelector(readySelector) : null;
      return {
        href: location.href,
        title: document.title,
        canvas: rect ? { width: rect.width, height: rect.height } : null,
        readyTitle: readyNode?.getAttribute("title") || null,
        textSample: document.body?.innerText?.slice(0, 500) || "",
      };
    }, {
      canvasSelector: options.canvasSelector,
      readySelector: options.skipReady ? "" : options.readySelector,
    });
    await appendLog("dom", JSON.stringify(domInfo));

    await page.locator(options.canvasSelector).first().screenshot({
      path: screenshotPath,
      timeout: options.screenshotTimeoutMs,
    });
    await appendLog("screenshot", screenshotPath);

    console.log(
      JSON.stringify(
        {
          ok: true,
          url: options.url,
          browserPath: sanitize(chromium.executablePath()),
          screenshotPath: sanitize(screenshotPath),
          logPath: sanitize(logPath),
          logs,
        },
        null,
        2
      )
    );
  } catch (error) {
    await appendLog("fatal", error.stack || error.message);
    if (page) {
      try {
        const canvasLocator = page.locator(options.canvasSelector).first();
        if ((await canvasLocator.count()) > 0) {
          await canvasLocator.screenshot({
            path: screenshotPath,
            timeout: options.screenshotTimeoutMs,
          });
        } else {
          await page.locator("body").screenshot({
            path: screenshotPath,
            timeout: options.screenshotTimeoutMs,
          });
        }
        await appendLog("failure-screenshot", screenshotPath);
      } catch (screenshotError) {
        await appendLog("failure-screenshot-error", screenshotError.stack || screenshotError.message);
      }
    }
    console.log(
      JSON.stringify(
        {
          ok: false,
          url: options.url,
          browserPath: sanitize(chromium.executablePath()),
          screenshotPath: sanitize(screenshotPath),
          logPath: sanitize(logPath),
          logs,
          error: sanitize(error.stack || error.message),
        },
        null,
        2
      )
    );
    process.exitCode = 1;
  } finally {
    await context?.close().catch(() => {});
    await browser?.close().catch(() => {});
  }
}

await main();
