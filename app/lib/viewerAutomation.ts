import type {
  EarthViewState,
  SavedViewRecord,
  ViewerNavigationCommand,
} from "./viewerTypes";
import type {
  MoistureLegibilityExperiment,
  MoistureStructureLayerState,
  MoistureVisualPreset,
} from "../state/controlsStore";
import type { MoistureSegmentationMode } from "../components/utils/ApiResponses";

export const VIEWER_NAVIGATION_CONTROLS = [
  {
    label: "Forward",
    command: "move-forward",
    ariaLabel: "Move forward",
    testId: "viewer-move-forward",
  },
  {
    label: "Backward",
    command: "move-backward",
    ariaLabel: "Move backward",
    testId: "viewer-move-backward",
  },
  {
    label: "Left",
    command: "move-left",
    ariaLabel: "Move left",
    testId: "viewer-move-left",
  },
  {
    label: "Right",
    command: "move-right",
    ariaLabel: "Move right",
    testId: "viewer-move-right",
  },
  {
    label: "Up",
    command: "move-up",
    ariaLabel: "Move up",
    testId: "viewer-move-up",
  },
  {
    label: "Down",
    command: "move-down",
    ariaLabel: "Move down",
    testId: "viewer-move-down",
  },
  {
    label: "Look Left",
    command: "look-left",
    ariaLabel: "Look left",
    testId: "viewer-look-left",
  },
  {
    label: "Look Right",
    command: "look-right",
    ariaLabel: "Look right",
    testId: "viewer-look-right",
  },
  {
    label: "Look Up",
    command: "look-up",
    ariaLabel: "Look up",
    testId: "viewer-look-up",
  },
  {
    label: "Look Down",
    command: "look-down",
    ariaLabel: "Look down",
    testId: "viewer-look-down",
  },
] as const satisfies ReadonlyArray<{
  label: string;
  command: ViewerNavigationCommand;
  ariaLabel: string;
  testId: string;
}>;

export const VIEWER_TIME_CONTROLS = [
  {
    action: "step-backward-time",
    label: "Step Backward Time",
    ariaLabel: "Step backward time",
    testId: "time-step-backward",
  },
  {
    action: "step-forward-time",
    label: "Step Forward Time",
    ariaLabel: "Step forward time",
    testId: "time-step-forward",
  },
] as const;

export const VIEWER_AUTOMATION_SELECTORS = {
  layersSidebarToggle: {
    openAriaLabel: "Open layers",
    closeAriaLabel: "Close layers",
  },
  devViewerPane: {
    testId: "dev-viewer-pane",
    ariaLabel: "Dev viewer controls",
  },
  currentZoom: {
    testId: "viewer-current-zoom",
  },
  currentTimestamp: {
    testId: "viewer-current-timestamp",
  },
  savedViewTitle: {
    testId: "saved-view-title",
    ariaLabel: "Saved view title",
  },
  savedViewDescription: {
    testId: "saved-view-description",
    ariaLabel: "Saved view description",
  },
  savedViewSave: {
    testId: "saved-view-save",
    ariaLabel: "Save current view",
  },
  savedViewRefresh: {
    testId: "saved-view-refresh",
    ariaLabel: "Refresh saved views",
  },
  savedViewsList: {
    testId: "saved-views-list",
    ariaLabel: "Saved views list",
  },
  savedViewApply: {
    ariaLabelPattern: "Apply saved view <title>",
    testIdPattern: "saved-view-apply-<id>",
  },
  savedViewDelete: {
    ariaLabelPattern: "Delete saved view <title>",
    testIdPattern: "saved-view-delete-<id>",
  },
  canvas: {
    selector: "canvas",
  },
} as const;

export type ViewerAutomationSnapshot = {
  ready: boolean;
  paused: boolean;
  timestamp: string;
  zoom01: number;
  moistureLegibilityExperiment: MoistureLegibilityExperiment;
  moistureSegmentationMode: MoistureSegmentationMode;
  earthView: EarthViewState | null;
  savedViews: SavedViewRecord[];
};

export type ViewerAutomationSavedViewTarget = {
  id?: string;
  title?: string;
};

export type ViewerAutomationViewInput = {
  timestamp?: string;
  earthView: EarthViewState;
};

export type ViewerAutomationDescribeResult = {
  version: 1;
  recommendedUrl: "/?automation=1";
  selectors: typeof VIEWER_AUTOMATION_SELECTORS;
  navigationControls: typeof VIEWER_NAVIGATION_CONTROLS;
  timeControls: typeof VIEWER_TIME_CONTROLS;
  snapshot: ViewerAutomationSnapshot;
};

export type ViewerAutomationApi = {
  enabled: boolean;
  readonly paused: boolean;
  readonly ready: boolean;
  freeze: () => void;
  resume: () => void;
  renderOnce: () => void;
  capturePngDataUrl: () => string | null;
  describe: () => ViewerAutomationDescribeResult;
  getSnapshot: () => ViewerAutomationSnapshot;
  waitForReady: (timeoutMs?: number) => Promise<ViewerAutomationSnapshot>;
  ensureLayersSidebarOpen: () => Promise<boolean>;
  runNavigationCommand: (
    command: ViewerNavigationCommand
  ) => Promise<ViewerAutomationSnapshot>;
  stepTime: (
    direction: "forward" | "backward",
    timeoutMs?: number
  ) => Promise<ViewerAutomationSnapshot>;
  setTimestamp: (
    timestamp: string,
    timeoutMs?: number
  ) => Promise<ViewerAutomationSnapshot>;
  applyViewState: (
    input: ViewerAutomationViewInput,
    timeoutMs?: number
  ) => Promise<ViewerAutomationSnapshot>;
  setMoistureLegibilityExperiment: (
    experiment: MoistureLegibilityExperiment,
    timeoutMs?: number
  ) => Promise<ViewerAutomationSnapshot>;
  setMoistureVisualPreset: (
    preset: MoistureVisualPreset,
    timeoutMs?: number
  ) => Promise<ViewerAutomationSnapshot>;
  setMoistureLayerPatch: (
    patch: Partial<MoistureStructureLayerState>,
    timeoutMs?: number
  ) => Promise<ViewerAutomationSnapshot>;
  resetMoistureLegibilityExperiment: (
    timeoutMs?: number
  ) => Promise<ViewerAutomationSnapshot>;
  setMoistureSegmentationMode: (
    segmentationMode: MoistureSegmentationMode,
    timeoutMs?: number
  ) => Promise<ViewerAutomationSnapshot>;
  listSavedViews: () => Promise<SavedViewRecord[]>;
  saveView: (input: {
    title: string;
    description?: string;
  }) => Promise<SavedViewRecord>;
  applySavedView: (
    target: ViewerAutomationSavedViewTarget,
    timeoutMs?: number
  ) => Promise<ViewerAutomationSnapshot>;
  deleteSavedView: (
    target: ViewerAutomationSavedViewTarget
  ) => Promise<{ ok: true; id: string }>;
};
