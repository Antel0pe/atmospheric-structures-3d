import { NextResponse } from "next/server";

import { isDevMode } from "../../../lib/appConfig";
import { isSavedViewInput } from "../../../lib/viewerTypes";
import {
  deleteSavedViewById,
  readSavedViews,
  writeSavedView,
} from "../../_lib/savedViews";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function devOnlyResponse() {
  return NextResponse.json({ error: "Saved views are only available in dev mode." }, { status: 404 });
}

export async function GET() {
  if (!isDevMode()) return devOnlyResponse();

  const savedViews = await readSavedViews();
  return NextResponse.json(savedViews, {
    headers: {
      "Cache-Control": "no-store",
    },
  });
}

export async function POST(request: Request) {
  if (!isDevMode()) return devOnlyResponse();

  const payload: unknown = await request.json();
  if (!isSavedViewInput(payload)) {
    return NextResponse.json(
      { error: "Invalid saved view payload." },
      { status: 400 }
    );
  }

  const title = payload.title.trim();
  if (!title) {
    return NextResponse.json(
      { error: "Saved view title is required." },
      { status: 400 }
    );
  }

  const savedView = await writeSavedView({
    ...payload,
    title,
  });
  return NextResponse.json(savedView, {
    headers: {
      "Cache-Control": "no-store",
    },
  });
}

export async function DELETE(request: Request) {
  if (!isDevMode()) return devOnlyResponse();

  const { searchParams } = new URL(request.url);
  const id = searchParams.get("id")?.trim() ?? "";

  if (!id) {
    return NextResponse.json(
      { error: "Saved view id is required." },
      { status: 400 }
    );
  }

  const deleted = await deleteSavedViewById(id);
  if (!deleted) {
    return NextResponse.json(
      { error: "Saved view not found." },
      { status: 404 }
    );
  }

  return NextResponse.json(
    { ok: true, id },
    {
      headers: {
        "Cache-Control": "no-store",
      },
    }
  );
}
