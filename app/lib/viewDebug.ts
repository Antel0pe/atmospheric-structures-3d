import type {
  ExampleContoursLayerState,
  ExampleParticleLayerState,
  ExampleShaderMeshLayerState,
  MoistureStructureLayerState,
  PrecipitableWaterLayerState,
  PrecipitationRadarLayerState,
  RelativeHumidityLayerState,
} from "../state/controlsStore";
import { isEarthViewState, type EarthViewState } from "./viewerTypes";

export type ViewDebugAnalyzerName = "moisture-structure" | (string & {});

export type NormalizedScreenTarget = {
  id?: string;
  label?: string;
  x: number;
  y: number;
};

export type ViewDebugSavedViewSource = {
  kind: "saved-view";
  id?: string;
  title?: string;
  path?: string;
};

export type ViewDebugExplicitSource = {
  kind: "explicit-view";
  description?: string;
};

export type ViewDebugSource = ViewDebugSavedViewSource | ViewDebugExplicitSource;

export type ViewDebugLayerStateSnapshot = {
  moistureStructureLayer: MoistureStructureLayerState;
  precipitationRadarLayer: PrecipitationRadarLayerState;
  precipitableWaterLayer: PrecipitableWaterLayerState;
  relativeHumidityLayer: RelativeHumidityLayerState;
  exampleShaderMeshLayer: ExampleShaderMeshLayerState;
  exampleContoursLayer: ExampleContoursLayerState;
  exampleParticleLayer: ExampleParticleLayerState;
};

export type ViewDebugCase = {
  version: 1;
  analyzer: ViewDebugAnalyzerName;
  title: string;
  createdAt: string;
  source: ViewDebugSource;
  timestamp: string;
  earthView: EarthViewState;
  layerState: ViewDebugLayerStateSnapshot;
  targets: NormalizedScreenTarget[];
  notes?: string;
};

export type ViewDebugCaseInput = {
  analyzer: ViewDebugAnalyzerName;
  title: string;
  source: ViewDebugSource;
  targets: NormalizedScreenTarget[];
  notes?: string;
};

export type ViewDebugAnalyzerAdapter = {
  analyzer: ViewDebugAnalyzerName;
  getState?: () => unknown;
  hitTest?: (target: NormalizedScreenTarget) => unknown;
  selectTarget?: (targetId: unknown) => unknown;
};

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function isNormalizedScreenTarget(value: unknown): value is NormalizedScreenTarget {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    isFiniteNumber(candidate.x) &&
    isFiniteNumber(candidate.y) &&
    candidate.x >= 0 &&
    candidate.x <= 1 &&
    candidate.y >= 0 &&
    candidate.y <= 1 &&
    (candidate.id === undefined || typeof candidate.id === "string") &&
    (candidate.label === undefined || typeof candidate.label === "string")
  );
}

function isViewDebugSource(value: unknown): value is ViewDebugSource {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  if (candidate.kind === "saved-view") {
    return (
      (candidate.id === undefined || typeof candidate.id === "string") &&
      (candidate.title === undefined || typeof candidate.title === "string") &&
      (candidate.path === undefined || typeof candidate.path === "string")
    );
  }
  if (candidate.kind === "explicit-view") {
    return candidate.description === undefined || typeof candidate.description === "string";
  }
  return false;
}

function isLayerStateSnapshot(value: unknown): value is ViewDebugLayerStateSnapshot {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    !!candidate.moistureStructureLayer &&
    !!candidate.precipitationRadarLayer &&
    !!candidate.precipitableWaterLayer &&
    !!candidate.relativeHumidityLayer &&
    !!candidate.exampleShaderMeshLayer &&
    !!candidate.exampleContoursLayer &&
    !!candidate.exampleParticleLayer
  );
}

export function isViewDebugCase(value: unknown): value is ViewDebugCase {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    candidate.version === 1 &&
    typeof candidate.analyzer === "string" &&
    typeof candidate.title === "string" &&
    typeof candidate.createdAt === "string" &&
    isViewDebugSource(candidate.source) &&
    typeof candidate.timestamp === "string" &&
    isEarthViewState(candidate.earthView) &&
    isLayerStateSnapshot(candidate.layerState) &&
    Array.isArray(candidate.targets) &&
    candidate.targets.every((target) => isNormalizedScreenTarget(target)) &&
    (candidate.notes === undefined || typeof candidate.notes === "string")
  );
}
