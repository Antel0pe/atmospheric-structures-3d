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

function slugify(value) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64) || "debug-case";
}

function parseTarget(value, index) {
  const [xText, yText] = value.split(",", 2).map((part) => part.trim());
  const x = Number(xText);
  const y = Number(yText);
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    throw new Error(`Invalid --target '${value}'. Expected '<x>,<y>'.`);
  }
  return {
    id: `target-${index + 1}`,
    label: `Target ${index + 1}`,
    x,
    y,
  };
}

function parseArgs(argv) {
  const options = {
    url: "http://localhost:3000",
    analyzer: "moisture-structure",
    caseFile: "",
    savedViewPath: "",
    savedViewTitle: "",
    targets: [],
    title: "",
    notes: "",
    outputDir: "",
    serverTimeoutMs: 120_000,
    timeoutMs: 120_000,
    python: true,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case "--url":
        options.url = argv[++i];
        break;
      case "--analyzer":
        options.analyzer = argv[++i];
        break;
      case "--case-file":
        options.caseFile = argv[++i];
        break;
      case "--saved-view-path":
        options.savedViewPath = argv[++i];
        break;
      case "--saved-view-title":
        options.savedViewTitle = argv[++i];
        break;
      case "--target":
        options.targets.push(argv[++i]);
        break;
      case "--title":
        options.title = argv[++i];
        break;
      case "--notes":
        options.notes = argv[++i];
        break;
      case "--output-dir":
        options.outputDir = argv[++i];
        break;
      case "--server-timeout-ms":
        options.serverTimeoutMs = Number(argv[++i]);
        break;
      case "--timeout-ms":
        options.timeoutMs = Number(argv[++i]);
        break;
      case "--no-python":
        options.python = false;
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!options.caseFile && !options.savedViewPath && !options.savedViewTitle) {
    throw new Error("Provide --case-file, --saved-view-path, or --saved-view-title.");
  }

  return options;
}

function withAutomationQuery(urlString) {
  const url = new URL(urlString);
  url.searchParams.set("automation", "1");
  return url.toString();
}

async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
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
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (await isServerUp(urlString)) return true;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  return false;
}

function defaultDevCommandForUrl(urlString) {
  const url = new URL(urlString);
  const port = url.port || (url.protocol === "https:" ? "443" : "80");
  if (port === "3000") return "bun dev --hostname localhost --port 3000";
  return `bun dev --hostname localhost --port ${port}`;
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function buildComparisonCases(debugCase) {
  if (debugCase.analyzer !== "moisture-structure") return [];

  const simplified = cloneJson(debugCase);
  simplified.title = `${debugCase.title} · simplified-render`;
  simplified.layerState.moistureStructureLayer = {
    ...simplified.layerState.moistureStructureLayer,
    visualPreset: "solidShell",
    structurePreset: "currentDepth",
    legibilityExperiment: "none",
    surfaceCueMode: "none",
    surfaceBrightness: 1,
    surfaceShadowStrength: 1,
    cameraCutawayEnabled: false,
    pickMode: false,
    focusMode: "none",
    selectedComponentId: null,
    verticalWallFadeEnabled: false,
    footprintOverlayEnabled: false,
  };

  const simpleVoxelShell = cloneJson(simplified);
  simpleVoxelShell.title = `${debugCase.title} · simple-voxel-shell`;
  simpleVoxelShell.layerState.moistureStructureLayer = {
    ...simpleVoxelShell.layerState.moistureStructureLayer,
    segmentationMode: "simple-voxel-shell",
  };

  return [
    { label: "baseline", debugCase },
    { label: "simplified-render", debugCase: simplified },
    { label: "simple-voxel-shell", debugCase: simpleVoxelShell },
  ];
}

async function writePngDataUrl(filePath, dataUrl) {
  const base64 = dataUrl.split(",")[1] || "";
  await fs.writeFile(filePath, Buffer.from(base64, "base64"));
}

async function captureScenario(page, label, debugCase, outputDir) {
  const appliedState = await page.evaluate(
    async (nextCase) => window.__ATMOS_AUTOMATION__.applyViewDebugCase(nextCase, 120000),
    debugCase
  );
  const hits = [];
  for (const target of debugCase.targets) {
    hits.push(
      await page.evaluate(
        async (request) => window.__ATMOS_AUTOMATION__.hitTestDebugTarget(request),
        {
          analyzer: debugCase.analyzer,
          target,
        }
      )
    );
  }

  const screenshotPath = path.join(outputDir, `${label}.png`);
  const dataUrl = await page.evaluate(
    () => window.__ATMOS_AUTOMATION__.capturePngDataUrl()
  );
  if (!dataUrl) {
    throw new Error(`Capture failed for scenario '${label}'.`);
  }
  await writePngDataUrl(screenshotPath, dataUrl);

  return {
    label,
    debugCase,
    appliedState,
    hits,
    screenshot: path.relative(cwd, screenshotPath).replace(/\\/g, "/"),
  };
}

async function runCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      env: {
        ...process.env,
        ...options.env,
      },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(
          new Error(
            `${command} ${args.join(" ")} failed with code ${code}\n${stderr || stdout}`
          )
        );
      }
    });
  });
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const targetUrl = withAutomationQuery(options.url);
  const runtimeRoot = path.join(cwd, "tmp", "playwright-runtime");
  const browserTempRoot = path.join(runtimeRoot, "tmp");
  const xdgConfigHome = path.join(runtimeRoot, "xdg-config");
  const xdgCacheHome = path.join(runtimeRoot, "xdg-cache");
  const xdgDataHome = path.join(runtimeRoot, "xdg-data");
  const serverLogPath = path.join(runtimeRoot, "view-debug-bun-dev.log");

  await ensureDir(browserTempRoot);
  await ensureDir(xdgConfigHome);
  await ensureDir(xdgCacheHome);
  await ensureDir(xdgDataHome);
  await ensureDir(path.join(runtimeRoot, "mpl-config"));

  process.env.TMPDIR = browserTempRoot;
  process.env.TMP = browserTempRoot;
  process.env.TEMP = browserTempRoot;
  process.env.XDG_CONFIG_HOME = xdgConfigHome;
  process.env.XDG_CACHE_HOME = xdgCacheHome;
  process.env.XDG_DATA_HOME = xdgDataHome;
  process.env.PLAYWRIGHT_BROWSERS_PATH = path.join(home, ".cache", "ms-playwright");

  const launchEnv = {
    ...process.env,
    TMPDIR: browserTempRoot,
    TMP: browserTempRoot,
    TEMP: browserTempRoot,
    XDG_CONFIG_HOME: xdgConfigHome,
    XDG_CACHE_HOME: xdgCacheHome,
    XDG_DATA_HOME: xdgDataHome,
    PLAYWRIGHT_BROWSERS_PATH: path.join(home, ".cache", "ms-playwright"),
  };

  let browser;
  let devServer = null;
  let startedServer = false;

  try {
    const serverReady = await isServerUp(targetUrl);
    if (!serverReady) {
      const devCommand = defaultDevCommandForUrl(options.url);
      await ensureDir(path.dirname(serverLogPath));
      const serverLogHandle = await fs.open(serverLogPath, "a");
      devServer = spawn("bash", ["-lc", devCommand], {
        cwd,
        detached: true,
        env: launchEnv,
        stdio: ["ignore", serverLogHandle.fd, serverLogHandle.fd],
      });
      devServer.unref();
      await serverLogHandle.close();
      startedServer = true;
      const ready = await waitForServer(targetUrl, options.serverTimeoutMs);
      if (!ready) {
        throw new Error(
          `Server did not become reachable at ${sanitize(targetUrl)} within ${options.serverTimeoutMs}ms.`
        );
      }
    }

    browser = await chromium.launch({
      headless: true,
      env: launchEnv,
      args: ["--enable-webgl", "--ignore-gpu-blocklist", "--use-gl=angle", "--use-angle=swiftshader"],
    });
    const context = await browser.newContext({
      viewport: { width: 1600, height: 900 },
      deviceScaleFactor: 1,
    });
    const page = await context.newPage();
    const consoleLogs = [];
    page.on("console", (msg) => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    page.on("pageerror", (error) => {
      consoleLogs.push({ type: "pageerror", text: error.stack || error.message });
    });

    await page.goto(targetUrl, { waitUntil: "domcontentloaded", timeout: options.timeoutMs });
    await page.waitForFunction(() => Boolean(window.__ATMOS_AUTOMATION__?.enabled), undefined, {
      timeout: options.timeoutMs,
    });
    await page.evaluate(() => window.__ATMOS_AUTOMATION__.waitForReady(120000));

    let debugCase;
    if (options.caseFile) {
      debugCase = JSON.parse(await fs.readFile(options.caseFile, "utf8"));
    } else {
      const targets = options.targets.map((value, index) => parseTarget(value, index));
      if (targets.length === 0) {
        throw new Error("Provide at least one --target when building a new debug case.");
      }

      let savedView;
      let source;
      if (options.savedViewPath) {
        const savedViewPath = path.resolve(cwd, options.savedViewPath);
        savedView = JSON.parse(await fs.readFile(savedViewPath, "utf8"));
        source = {
          kind: "saved-view",
          id: savedView.id,
          title: savedView.title,
          path: path.relative(cwd, savedViewPath).replace(/\\/g, "/"),
        };
        await page.evaluate(
          async (input) => window.__ATMOS_AUTOMATION__.applyViewState(input, 120000),
          {
            timestamp: savedView.timestamp,
            earthView: savedView.earthView,
          }
        );
      } else {
        savedView = await page.evaluate(async (title) => {
          const views = await window.__ATMOS_AUTOMATION__.listSavedViews();
          const exact = views.find((view) => view.title === title);
          if (exact) return exact;
          const folded = title.toLowerCase();
          return views.find((view) => view.title.toLowerCase() === folded) ?? null;
        }, options.savedViewTitle);
        if (!savedView) {
          throw new Error(`Saved view '${options.savedViewTitle}' was not found.`);
        }
        source = {
          kind: "saved-view",
          id: savedView.id,
          title: savedView.title,
        };
        await page.evaluate(
          async (title) => window.__ATMOS_AUTOMATION__.applySavedView({ title }, 120000),
          savedView.title
        );
      }

      debugCase = await page.evaluate(
        (input) => window.__ATMOS_AUTOMATION__.buildViewDebugCase(input),
        {
          analyzer: options.analyzer,
          title: options.title || `${savedView.title} debug`,
          source,
          targets,
          notes: options.notes || undefined,
        }
      );
    }

    const caseSlug = slugify(debugCase.title || "view-debug");
    const outputDir =
      options.outputDir ||
      path.join("tmp", "view-debug", `${caseSlug}-${debugCase.timestamp.replace(/[:]/g, "-")}`);
    const absoluteOutputDir = path.resolve(cwd, outputDir);
    await ensureDir(absoluteOutputDir);

    const debugCasePath = path.join(absoluteOutputDir, "debug-case.json");
    await fs.writeFile(debugCasePath, `${JSON.stringify(debugCase, null, 2)}\n`, "utf8");

    const scenarios = [];
    for (const scenario of buildComparisonCases(debugCase)) {
      scenarios.push(await captureScenario(page, scenario.label, scenario.debugCase, absoluteOutputDir));
    }

    const captureContext = {
      version: 1,
      analyzer: debugCase.analyzer,
      url: targetUrl,
      outputDir: path.relative(cwd, absoluteOutputDir).replace(/\\/g, "/"),
      consoleLogs,
      scenarios,
      startedServer,
    };
    const captureContextPath = path.join(absoluteOutputDir, "capture-context.json");
    await fs.writeFile(
      captureContextPath,
      `${JSON.stringify(captureContext, null, 2)}\n`,
      "utf8"
    );

    let analyzerResult = null;
    if (options.python && debugCase.analyzer === "moisture-structure") {
      const { stdout } = await runCommand("conda", [
        "run",
        "-n",
        "atmospheric-structures-3d",
        "python",
        "-m",
        "scripts.view_debug_moisture",
        "--debug-case",
        path.relative(cwd, debugCasePath).replace(/\\/g, "/"),
        "--capture-context",
        path.relative(cwd, captureContextPath).replace(/\\/g, "/"),
        "--output-dir",
        path.relative(cwd, absoluteOutputDir).replace(/\\/g, "/"),
      ], {
        env: {
          TMPDIR: browserTempRoot,
          TMP: browserTempRoot,
          TEMP: browserTempRoot,
          MPLCONFIGDIR: path.join(runtimeRoot, "mpl-config"),
          XDG_CONFIG_HOME: xdgConfigHome,
          XDG_CACHE_HOME: xdgCacheHome,
          XDG_DATA_HOME: xdgDataHome,
          PYTHONPATH: cwd,
        },
      });
      analyzerResult = stdout.trim();
    }

    console.log(
      JSON.stringify(
        {
          ok: true,
          analyzer: debugCase.analyzer,
          outputDir: path.relative(cwd, absoluteOutputDir).replace(/\\/g, "/"),
          debugCase: path.relative(cwd, debugCasePath).replace(/\\/g, "/"),
          captureContext: path.relative(cwd, captureContextPath).replace(/\\/g, "/"),
          analyzerResult: analyzerResult || null,
        },
        null,
        2
      )
    );

    await browser.close();
  } finally {
    if (browser) {
      await browser.close().catch(() => {});
    }
    if (devServer) {
      devServer.unref();
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
