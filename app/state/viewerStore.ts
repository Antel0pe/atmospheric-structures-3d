import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import { getAppConfig } from "../lib/appConfig";
import type {
  EarthViewState,
  SavedViewApplyRequest,
  SavedViewInput,
  SavedViewRecord,
  ViewerNavigationCommand,
  ViewerNavigationRequest,
} from "../lib/viewerTypes";

type SavedViewsState = {
  savedViews: SavedViewRecord[];
  savedViewsLoading: boolean;
  savedViewsSaving: boolean;
  savedViewDeletingId: string | null;
  savedViewsError: string | null;
  timestamp: string;
  earthView: EarthViewState | null;
  zoom01: number;
  applySavedViewRequest: SavedViewApplyRequest | null;
  navigationRequest: ViewerNavigationRequest | null;
  setTimestamp: (timestamp: string) => void;
  publishEarthView: (earthView: EarthViewState) => void;
  setZoom01: (zoom01: number) => void;
  requestNavigationCommand: (command: ViewerNavigationCommand) => void;
  requestApplySavedView: (savedView: SavedViewRecord) => void;
  promoteApplySavedViewReady: (requestId: number) => void;
  clearApplySavedViewRequest: (requestId: number) => void;
  setSavedViews: (savedViews: SavedViewRecord[]) => void;
  setSavedViewsLoading: (loading: boolean) => void;
  setSavedViewsSaving: (saving: boolean) => void;
  setSavedViewDeletingId: (id: string | null) => void;
  setSavedViewsError: (message: string | null) => void;
  loadSavedViews: () => Promise<void>;
  saveSavedView: (input: Pick<SavedViewInput, "title" | "description">) => Promise<SavedViewRecord | null>;
  deleteSavedView: (id: string) => Promise<boolean>;
};

let nextNavigationRequestId = 1;
let nextApplyRequestId = 1;

async function parseJsonResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as T;
}

export const useViewerStore = create<SavedViewsState>()(
  subscribeWithSelector((set, get) => ({
    savedViews: [],
    savedViewsLoading: false,
    savedViewsSaving: false,
    savedViewDeletingId: null,
    savedViewsError: null,
    timestamp: getAppConfig().sliderDateRange.startDate,
    earthView: null,
    zoom01: 0,
    applySavedViewRequest: null,
    navigationRequest: null,
    setTimestamp: (timestamp) => set({ timestamp }),
    publishEarthView: (earthView) =>
      set({
        earthView,
        zoom01: earthView.zoom01,
      }),
    setZoom01: (zoom01) =>
      set((state) => ({
        zoom01,
        earthView: state.earthView
          ? {
              ...state.earthView,
              zoom01,
            }
          : null,
      })),
    requestNavigationCommand: (command) =>
      set({
        navigationRequest: {
          requestId: nextNavigationRequestId++,
          command,
        },
      }),
    requestApplySavedView: (savedView) =>
      set({
        applySavedViewRequest: {
          requestId: nextApplyRequestId++,
          phase: "initial",
          savedView,
        },
      }),
    promoteApplySavedViewReady: (requestId) =>
      set((state) => {
        if (
          !state.applySavedViewRequest ||
          state.applySavedViewRequest.requestId !== requestId ||
          state.applySavedViewRequest.phase === "ready"
        ) {
          return state;
        }

        return {
          applySavedViewRequest: {
            ...state.applySavedViewRequest,
            phase: "ready",
          },
        };
      }),
    clearApplySavedViewRequest: (requestId) =>
      set((state) => {
        if (
          !state.applySavedViewRequest ||
          state.applySavedViewRequest.requestId !== requestId
        ) {
          return state;
        }

        return {
          applySavedViewRequest: null,
        };
      }),
    setSavedViews: (savedViews) => set({ savedViews }),
    setSavedViewsLoading: (savedViewsLoading) => set({ savedViewsLoading }),
    setSavedViewsSaving: (savedViewsSaving) => set({ savedViewsSaving }),
    setSavedViewDeletingId: (savedViewDeletingId) => set({ savedViewDeletingId }),
    setSavedViewsError: (savedViewsError) => set({ savedViewsError }),
    async loadSavedViews() {
      set({ savedViewsLoading: true, savedViewsError: null });
      try {
        const savedViews = await parseJsonResponse<SavedViewRecord[]>(
          await fetch("/api/dev/saved-views", {
            cache: "no-store",
          })
        );
        set({
          savedViews,
          savedViewsLoading: false,
        });
      } catch (error) {
        set({
          savedViewsLoading: false,
          savedViewsError:
            error instanceof Error ? error.message : "Failed to load saved views",
        });
      }
    },
    async saveSavedView(input) {
      const earthView = get().earthView;
      if (!earthView) {
        set({ savedViewsError: "The earth view is not ready to save yet." });
        return null;
      }

      set({ savedViewsSaving: true, savedViewsError: null });
      try {
        const savedView = await parseJsonResponse<SavedViewRecord>(
          await fetch("/api/dev/saved-views", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              title: input.title,
              description: input.description,
              timestamp: get().timestamp,
              earthView,
            } satisfies SavedViewInput),
          })
        );

        set((state) => ({
          savedViewsSaving: false,
          savedViews: [savedView, ...state.savedViews.filter((entry) => entry.id !== savedView.id)],
        }));

        return savedView;
      } catch (error) {
        set({
          savedViewsSaving: false,
          savedViewsError:
            error instanceof Error ? error.message : "Failed to save the current view",
        });
        return null;
      }
    },
    async deleteSavedView(id) {
      set({ savedViewDeletingId: id, savedViewsError: null });
      try {
        await parseJsonResponse<{ ok: true; id: string }>(
          await fetch(`/api/dev/saved-views?id=${encodeURIComponent(id)}`, {
            method: "DELETE",
          })
        );

        set((state) => ({
          savedViewDeletingId: null,
          savedViews: state.savedViews.filter((entry) => entry.id !== id),
        }));
        return true;
      } catch (error) {
        set({
          savedViewDeletingId: null,
          savedViewsError:
            error instanceof Error ? error.message : "Failed to delete the saved view",
        });
        return false;
      }
    },
  }))
);
