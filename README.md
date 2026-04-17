This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

Run `source scripts/project-conda-env.sh --env-name atmospheric-structures-3d`
to create the `atmospheric-structures-3d` conda environment and activate it.

First, run the development server:

```bash
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

## Fast Playwright Capture

Use the repo-local capture workflow when you want a deterministic screenshot of the fully loaded WebGL app:

```bash
bun run capture:webgl
```

Default behavior:

- Targets `http://localhost:3000/?automation=1`
- Starts `bun dev --hostname localhost --port 3000` automatically if the app is down
- Waits for the repo's automation renderer to pause after the globe texture and layers are ready
- Captures a full browser screenshot to `tmp/playwright-localhost-3000.png`
- Writes logs to `tmp/playwright-localhost-3000.log`
- Prints JSON timings so you can see launch, ready, and screenshot cost separately

For a scene-only export without the browser chrome:

```bash
bun run capture:webgl -- --scene-only
```

## App Config

App configuration is selected by `NEXT_PUBLIC_APP_MODE`.

- `dev` uses `config/app-config.dev.json`
- `deployed` uses `config/app-config.deployed.json`

If `NEXT_PUBLIC_APP_MODE` is unset, the app defaults to `deployed`.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## To do
- make diagnostic 3d viewer so can export generated data 3d structure in browser and view it raw in 3d without integrating in layer. maybe help test if the structure looks correct and good?
- make a skill or something to integrate 3d structure into product and figure out issues and iterate on the rendering side once iterating on data side is done