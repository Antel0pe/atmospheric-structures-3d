# Worktree Setup

Use `worktree-scripts/setup-worktree.sh` after creating a new git worktree:

```bash
git worktree add ../atmospheric-structures-3d-feature -b codex/feature
cd ../atmospheric-structures-3d-feature
worktree-scripts/setup-worktree.sh
```

The setup links local-only state into a shared root owned by the main checkout:

- `data/`: raw ERA5 and related source data, read by diagnostics and builders.
- `public/moisture-structures/`, `public/potential-temperature-structures/`, `public/precipitable-water-proxy/`, `public/precipitation-radar/`, `public/relative-humidity-shell/`, and `public/temperature-slices/`: generated viewer assets.
- `notes/`: collaboration memory shared across worktrees.
- `saved-views/`: local viewer state used by capture and debug workflows.
- `skills/structure-of-data/logs/` and `skills/structure-probe/logs/`: diagnostic run logs.

The script keeps these per worktree:

- `node_modules/`: installed with `bun install --frozen-lockfile`.
- `.next/`: Next.js build/dev cache.
- `tmp/`: Playwright runtime state, WebGL screenshots, debug bundles, and other scratch outputs.

`public/air-mass-structures/` currently contains tracked baseline assets, so setup leaves it local instead of replacing it with a symlink. New ignored processed folders can be shared automatically; tracked public assets remain branch-local and should be reviewed like code.

Run this from the populated original checkout once if the shared root does not exist yet. Existing ignored directories are moved into the shared root and replaced by symlinks. To inspect without changing files:

```bash
worktree-scripts/setup-worktree.sh --check-only
worktree-scripts/setup-worktree.sh --dry-run
```

To use a different shared root:

```bash
ATMOS_WORKTREE_SHARED_ROOT=../atmos-shared worktree-scripts/setup-worktree.sh
```

For WSL2, keep the repo and shared root on the Linux filesystem when possible. Symlinks and Next.js file watching are usually slower and less predictable under `/mnt/c`.

## Notes

`notes/` is intentionally shared so conclusions and preferences survive across worktrees. To reduce clashing:

- Re-read a note immediately before editing it.
- Prefer short append-only updates.
- For scripted or bulk note edits, hold the shared lock:

```bash
worktree-scripts/with-shared-notes-lock.sh -- <command>
```

## Validation

Setup ends with:

```bash
bun run capture:webgl -- --check-only
```

That verifies the locked JavaScript dependencies are usable and the local WebGL capture skill can import Playwright. It does not require the dev server to already be running.

For the full capture path:

```bash
bun run capture:webgl
```

For the deeper view-debug path, the Python analyzer expects the conda environment named `atmospheric-structures-3d`.

## Cleanup

Before removing a worktree, the symlinks can be detached without deleting shared data:

```bash
worktree-scripts/cleanup-worktree.sh
```

To also remove local runtime caches:

```bash
worktree-scripts/cleanup-worktree.sh --prune-caches
```

