import { mkdir, readdir, readFile, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";

import type { SavedViewInput, SavedViewRecord } from "../../lib/viewerTypes";
import { isSavedViewRecord } from "../../lib/viewerTypes";

const SAVED_VIEWS_DIRNAME = "saved-views";

function sanitizeSlug(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
}

export function getSavedViewsRootPath() {
  return path.join(process.cwd(), SAVED_VIEWS_DIRNAME);
}

async function readSavedViewEntries() {
  const root = getSavedViewsRootPath();

  try {
    const entries = await readdir(root, { withFileTypes: true });
    const views = await Promise.all(
      entries
        .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
        .map(async (entry) => {
          try {
            const filePath = path.join(root, entry.name);
            const raw = await readFile(filePath, "utf-8");
            const parsed: unknown = JSON.parse(raw);
            return isSavedViewRecord(parsed)
              ? {
                  filePath,
                  savedView: parsed,
                }
              : null;
          } catch {
            return null;
          }
        })
    );

    return views.filter(
      (
        view
      ): view is {
        filePath: string;
        savedView: SavedViewRecord;
      } => view !== null
    );
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return [];
    }
    throw error;
  }
}

export async function readSavedViews(): Promise<SavedViewRecord[]> {
  const entries = await readSavedViewEntries();
  return entries
    .map((entry) => entry.savedView)
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export async function writeSavedView(input: SavedViewInput): Promise<SavedViewRecord> {
  const root = getSavedViewsRootPath();
  await mkdir(root, { recursive: true });

  const createdAt = new Date().toISOString();
  const id = randomUUID();
  const slug = sanitizeSlug(input.title) || "saved-view";
  const filename = `${createdAt.replace(/[:.]/g, "-")}-${slug}-${id.slice(0, 8)}.json`;

  const savedView: SavedViewRecord = {
    schemaVersion: 1,
    id,
    title: input.title.trim(),
    description: input.description.trim(),
    createdAt,
    timestamp: input.timestamp,
    earthView: input.earthView,
  };

  await writeFile(
    path.join(root, filename),
    `${JSON.stringify(savedView, null, 2)}\n`,
    "utf-8"
  );

  return savedView;
}

export async function deleteSavedViewById(id: string): Promise<boolean> {
  const entries = await readSavedViewEntries();
  const matchingEntry = entries.find((entry) => entry.savedView.id === id);

  if (!matchingEntry) {
    return false;
  }

  await unlink(matchingEntry.filePath);
  return true;
}
