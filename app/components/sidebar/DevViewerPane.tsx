"use client";

import { useEffect, useMemo, useState } from "react";
import { isDevMode } from "../../lib/appConfig";
import { VIEWER_NAVIGATION_CONTROLS } from "../../lib/viewerAutomation";
import type { SavedViewRecord } from "../../lib/viewerTypes";
import { useViewerStore } from "../../state/viewerStore";

function sectionStyle() {
  return {
    margin: 8,
    padding: 12,
    borderRadius: 12,
    background: "rgba(255,255,255,0.06)",
    border: "1px solid rgba(255,255,255,0.08)",
    color: "#e9eef7",
    font: "500 12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto",
  } as const;
}

function commandButtonStyle() {
  return {
    borderRadius: 10,
    border: "1px solid rgba(140, 190, 255, 0.28)",
    background: "rgba(82, 146, 255, 0.14)",
    color: "#e9eef7",
    padding: "10px 12px",
    fontWeight: 700,
    cursor: "pointer",
    minHeight: 42,
  } as const;
}

function inputStyle() {
  return {
    width: "100%",
    borderRadius: 10,
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(6, 10, 18, 0.88)",
    color: "#e9eef7",
    padding: "10px 12px",
    outline: "none",
  } as const;
}

function SavedViewCard({
  savedView,
  onApply,
  onDelete,
  deleting,
}: {
  savedView: SavedViewRecord;
  onApply: (savedView: SavedViewRecord) => void;
  onDelete: (savedView: SavedViewRecord) => void;
  deleting: boolean;
}) {
  return (
    <div
      style={{
        display: "grid",
        gap: 8,
        padding: 10,
        borderRadius: 10,
        border: "1px solid rgba(255,255,255,0.08)",
        background: "rgba(255,255,255,0.04)",
      }}
    >
      <div style={{ display: "grid", gap: 4 }}>
        <div style={{ fontWeight: 700 }}>{savedView.title}</div>
        {savedView.description ? (
          <div style={{ opacity: 0.72, lineHeight: 1.4 }}>{savedView.description}</div>
        ) : null}
        <div style={{ opacity: 0.62 }}>
          {savedView.timestamp} UTC
        </div>
      </div>

      <div style={{ display: "grid", gap: 8, gridTemplateColumns: "repeat(2, minmax(0, 1fr))" }}>
        <button
          type="button"
          onClick={() => onApply(savedView)}
          aria-label={`Apply saved view ${savedView.title}`}
          data-testid={`saved-view-apply-${savedView.id}`}
          style={commandButtonStyle()}
        >
          Apply
        </button>

        <button
          type="button"
          onClick={() => onDelete(savedView)}
          disabled={deleting}
          aria-label={`Delete saved view ${savedView.title}`}
          data-testid={`saved-view-delete-${savedView.id}`}
          style={{
            ...commandButtonStyle(),
            border: "1px solid rgba(255, 122, 149, 0.34)",
            background: "rgba(196, 53, 86, 0.16)",
            opacity: deleting ? 0.55 : 1,
            cursor: deleting ? "not-allowed" : "pointer",
          }}
        >
          {deleting ? "Deleting..." : "Delete"}
        </button>
      </div>
    </div>
  );
}

export default function DevViewerPane() {
  if (!isDevMode()) return null;

  return <DevViewerPaneInner />;
}

function DevViewerPaneInner() {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const requestNavigationCommand = useViewerStore(
    (state) => state.requestNavigationCommand
  );
  const requestApplySavedView = useViewerStore(
    (state) => state.requestApplySavedView
  );
  const loadSavedViews = useViewerStore((state) => state.loadSavedViews);
  const saveSavedView = useViewerStore((state) => state.saveSavedView);
  const deleteSavedView = useViewerStore((state) => state.deleteSavedView);
  const savedViews = useViewerStore((state) => state.savedViews);
  const savedViewsLoading = useViewerStore((state) => state.savedViewsLoading);
  const savedViewsSaving = useViewerStore((state) => state.savedViewsSaving);
  const savedViewDeletingId = useViewerStore((state) => state.savedViewDeletingId);
  const savedViewsError = useViewerStore((state) => state.savedViewsError);
  const zoom01 = useViewerStore((state) => state.zoom01);
  const timestamp = useViewerStore((state) => state.timestamp);

  useEffect(() => {
    void loadSavedViews();
  }, [loadSavedViews]);

  const canSave = useMemo(() => title.trim().length > 0, [title]);

  return (
    <section
      aria-label="Dev viewer controls"
      data-testid="dev-viewer-pane"
      style={sectionStyle()}
    >
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          gap: 8,
          marginBottom: 12,
        }}
      >
        <div style={{ fontWeight: 800, letterSpacing: ".02em", textTransform: "uppercase" }}>
          Dev Viewer
        </div>
        <div data-testid="viewer-current-zoom" style={{ opacity: 0.65 }}>
          Zoom {zoom01.toFixed(3)}
        </div>
      </div>

      <div style={{ display: "grid", gap: 8, marginBottom: 12 }}>
        <div style={{ opacity: 0.72 }}>Timestamp</div>
        <div data-testid="viewer-current-timestamp" style={{ fontWeight: 700 }}>
          {timestamp}
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gap: 8,
          gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
          marginBottom: 14,
        }}
      >
        {VIEWER_NAVIGATION_CONTROLS.map((entry) => (
          <button
            key={entry.command}
            type="button"
            onClick={() => requestNavigationCommand(entry.command)}
            aria-label={entry.ariaLabel}
            data-testid={entry.testId}
            style={commandButtonStyle()}
          >
            {entry.label}
          </button>
        ))}
      </div>

      <div
        style={{
          display: "grid",
          gap: 10,
          paddingTop: 12,
          borderTop: "1px solid rgba(255,255,255,0.08)",
        }}
      >
        <label style={{ display: "grid", gap: 6 }}>
          <span style={{ fontWeight: 700 }}>Saved View Title</span>
          <input
            value={title}
            onChange={(event) => setTitle(event.currentTarget.value)}
            aria-label="Saved view title"
            data-testid="saved-view-title"
            placeholder="Warm sector overview"
            style={inputStyle()}
          />
        </label>

        <label style={{ display: "grid", gap: 6 }}>
          <span style={{ fontWeight: 700 }}>Saved View Description</span>
          <textarea
            value={description}
            onChange={(event) => setDescription(event.currentTarget.value)}
            aria-label="Saved view description"
            data-testid="saved-view-description"
            placeholder="Optional note about what this view shows."
            rows={3}
            style={{ ...inputStyle(), resize: "vertical" as const }}
          />
        </label>

        <div style={{ display: "grid", gap: 8, gridTemplateColumns: "repeat(2, minmax(0, 1fr))" }}>
          <button
            type="button"
            onClick={async () => {
              const savedView = await saveSavedView({
                title,
                description,
              });
              if (!savedView) return;
              setTitle("");
              setDescription("");
            }}
            disabled={!canSave || savedViewsSaving}
            aria-label="Save current view"
            data-testid="saved-view-save"
            style={{
              ...commandButtonStyle(),
              opacity: !canSave || savedViewsSaving ? 0.55 : 1,
              cursor: !canSave || savedViewsSaving ? "not-allowed" : "pointer",
            }}
          >
            {savedViewsSaving ? "Saving..." : "Save Current View"}
          </button>

          <button
            type="button"
            onClick={() => void loadSavedViews()}
            aria-label="Refresh saved views"
            data-testid="saved-view-refresh"
            style={commandButtonStyle()}
          >
            Refresh List
          </button>
        </div>

        {savedViewsError ? (
          <div
            aria-label="Saved view error"
            data-testid="saved-view-error"
            style={{ color: "#ffb7c0", lineHeight: 1.5 }}
          >
            {savedViewsError}
          </div>
        ) : null}

        <div style={{ display: "grid", gap: 8 }}>
          <div style={{ fontWeight: 700 }}>Saved Views</div>
          <div
            aria-label="Saved views list"
            data-testid="saved-views-list"
            style={{
              display: "grid",
              gap: 8,
              maxHeight: 320,
              overflowY: "auto",
            }}
          >
            {savedViewsLoading ? (
              <div style={{ opacity: 0.72 }}>Loading saved views…</div>
            ) : savedViews.length === 0 ? (
              <div style={{ opacity: 0.72 }}>No saved views yet.</div>
            ) : (
              savedViews.map((savedView) => (
                <SavedViewCard
                  key={savedView.id}
                  savedView={savedView}
                  onApply={requestApplySavedView}
                  onDelete={(entry) => void deleteSavedView(entry.id)}
                  deleting={savedViewDeletingId === savedView.id}
                />
              ))
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
