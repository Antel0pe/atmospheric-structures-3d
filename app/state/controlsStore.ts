import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import type {
  MoistureComponentFootprint,
  MoistureSegmentationMode,
  MoistureStructureComponentMetadata,
} from "../components/utils/ApiResponses";

export const EXAMPLE_SHADER_MESH_PRESSURE_OPTIONS = [
  { value: "none", label: "Hidden" },
  { value: 250, label: "250 hPa" },
  { value: 500, label: "500 hPa" },
  { value: 925, label: "925 hPa" },
] as const;

export type ExampleShaderMeshPressure =
  (typeof EXAMPLE_SHADER_MESH_PRESSURE_OPTIONS)[number]["value"];

export const EXAMPLE_CONTOURS_PRESSURE_OPTIONS = [
  { value: "none", label: "Hidden" },
  { value: "msl", label: "MSL" },
  { value: "250", label: "250 hPa" },
  { value: "500", label: "500 hPa" },
  { value: "925", label: "925 hPa" },
] as const;

export type ExampleContoursPressure =
  (typeof EXAMPLE_CONTOURS_PRESSURE_OPTIONS)[number]["value"];

export const EXAMPLE_PARTICLE_PRESSURE_OPTIONS = [
  { value: "none", label: "Hidden" },
  { value: 250, label: "250 hPa" },
  { value: 500, label: "500 hPa" },
  { value: 925, label: "925 hPa" },
] as const;

export type ExampleParticlePressure =
  (typeof EXAMPLE_PARTICLE_PRESSURE_OPTIONS)[number]["value"];

export const MOISTURE_VISUAL_PRESET_OPTIONS = [
  { value: "none", label: "None" },
  { value: "baseline", label: "Baseline" },
  { value: "solidShell", label: "Solid Shell" },
  { value: "lightingPass", label: "Lighting Pass" },
  { value: "interiorCutaway", label: "Interior Cutaway" },
  { value: "distanceCue", label: "Distance Cue" },
  { value: "fullPop", label: "Full Pop" },
] as const;

export type MoistureVisualPreset =
  (typeof MOISTURE_VISUAL_PRESET_OPTIONS)[number]["value"];

export const MOISTURE_STRUCTURE_PRESET_OPTIONS = [
  { value: "none", label: "None" },
  { value: "currentDepth", label: "Current Depth" },
  { value: "componentRead", label: "Component Read" },
  { value: "selectedFocus", label: "Selected Focus" },
  { value: "macroForm", label: "Macro Form" },
  { value: "thresholdCompare", label: "Threshold Compare" },
] as const;

export type MoistureStructurePreset =
  (typeof MOISTURE_STRUCTURE_PRESET_OPTIONS)[number]["value"];

export const MOISTURE_FOCUS_MODE_OPTIONS = [
  { value: "none", label: "No Focus" },
  { value: "dimOthers", label: "Dim Others" },
  { value: "showSelectedOnly", label: "Show Selected Only" },
] as const;

export type MoistureFocusMode =
  (typeof MOISTURE_FOCUS_MODE_OPTIONS)[number]["value"];

export const MOISTURE_COLOR_MODE_OPTIONS = [
  { value: "pressureBands", label: "Pressure Bands" },
  { value: "componentSolid", label: "Component Solid" },
  { value: "componentHybrid", label: "Component Hybrid" },
  { value: "selectedMonochrome", label: "Selected Monochrome" },
] as const;

export type MoistureColorMode =
  (typeof MOISTURE_COLOR_MODE_OPTIONS)[number]["value"];

export const MOISTURE_COMPONENT_SORT_OPTIONS = [
  { value: "size", label: "Largest First" },
  { value: "pressureSpan", label: "Pressure Span" },
  { value: "peakHumidity", label: "Peak Humidity" },
] as const;

export type MoistureComponentSort =
  (typeof MOISTURE_COMPONENT_SORT_OPTIONS)[number]["value"];

export const MOISTURE_SEGMENTATION_MODE_OPTIONS: ReadonlyArray<{
  value: MoistureSegmentationMode;
  label: string;
}> = [
  { value: "p95-close", label: "p95 + Closing" },
  { value: "p95-open", label: "p95 No Closing" },
  { value: "p97-close", label: "p97 + Closing" },
] as const;

type ExampleShaderMeshLayerState = {
  pressureLevel: ExampleShaderMeshPressure;
  uValueMin: number;
  uValueMax: number;
  uGamma: number;
  uAlpha: number;
  uZeroEps: number;
  uAsinhK: number;
};

type ExampleContoursLayerState = {
  pressureLevel: ExampleContoursPressure;
  contrast: number;
  opacity: number;
};

type ExampleParticleLayerState = {
  pressureLevel: ExampleParticlePressure;
};

export type MoistureStructureLayerState = {
  visible: boolean;
  opacity: number;
  verticalExaggeration: number;
  cameraCutawayEnabled: boolean;
  cameraCutawayRadius: number;
  visualPreset: MoistureVisualPreset;
  structurePreset: MoistureStructurePreset;
  solidShellEnabled: boolean;
  lightingEnabled: boolean;
  interiorBackfaceEnabled: boolean;
  rimEnabled: boolean;
  distanceFadeEnabled: boolean;
  frontOpacity: number;
  backfaceOpacity: number;
  ambientIntensity: number;
  keyLightIntensity: number;
  headLightIntensity: number;
  rimStrength: number;
  distanceFadeStrength: number;
  focusMode: MoistureFocusMode;
  selectedComponentId: number | null;
  pickMode: boolean;
  nonSelectedOpacity: number;
  colorMode: MoistureColorMode;
  componentSort: MoistureComponentSort;
  verticalWallFadeEnabled: boolean;
  verticalWallFadeStrength: number;
  segmentationMode: MoistureSegmentationMode;
  footprintOverlayEnabled: boolean;
};

export type MoistureVisualPresetState = Pick<
  MoistureStructureLayerState,
  | "solidShellEnabled"
  | "lightingEnabled"
  | "interiorBackfaceEnabled"
  | "rimEnabled"
  | "distanceFadeEnabled"
  | "frontOpacity"
  | "backfaceOpacity"
  | "ambientIntensity"
  | "keyLightIntensity"
  | "headLightIntensity"
  | "rimStrength"
  | "distanceFadeStrength"
>;

export type MoistureStructurePresetState = Pick<
  MoistureStructureLayerState,
  | "focusMode"
  | "pickMode"
  | "nonSelectedOpacity"
  | "colorMode"
  | "verticalWallFadeEnabled"
  | "verticalWallFadeStrength"
  | "segmentationMode"
  | "footprintOverlayEnabled"
>;

export type MoistureSidebarFrameState = {
  timestamp: string;
  segmentationMode: MoistureSegmentationMode;
  thresholdedVoxelCount: number;
  components: MoistureStructureComponentMetadata[];
  footprints: MoistureComponentFootprint[];
} | null;

const MOISTURE_VISUAL_PRESET_STATE: Record<
  MoistureVisualPreset,
  MoistureVisualPresetState
> = {
  none: {
    solidShellEnabled: false,
    lightingEnabled: false,
    interiorBackfaceEnabled: false,
    rimEnabled: false,
    distanceFadeEnabled: false,
    frontOpacity: 1,
    backfaceOpacity: 1,
    ambientIntensity: 2,
    keyLightIntensity: 0,
    headLightIntensity: 0,
    rimStrength: 0,
    distanceFadeStrength: 0,
  },
  baseline: {
    solidShellEnabled: false,
    lightingEnabled: false,
    interiorBackfaceEnabled: false,
    rimEnabled: false,
    distanceFadeEnabled: false,
    frontOpacity: 1,
    backfaceOpacity: 1,
    ambientIntensity: 2,
    keyLightIntensity: 0,
    headLightIntensity: 0,
    rimStrength: 0.45,
    distanceFadeStrength: 0.3,
  },
  solidShell: {
    solidShellEnabled: true,
    lightingEnabled: false,
    interiorBackfaceEnabled: false,
    rimEnabled: false,
    distanceFadeEnabled: false,
    frontOpacity: 1.2,
    backfaceOpacity: 0.28,
    ambientIntensity: 2,
    keyLightIntensity: 0,
    headLightIntensity: 0,
    rimStrength: 0.6,
    distanceFadeStrength: 0.35,
  },
  lightingPass: {
    solidShellEnabled: false,
    lightingEnabled: true,
    interiorBackfaceEnabled: false,
    rimEnabled: false,
    distanceFadeEnabled: false,
    frontOpacity: 1,
    backfaceOpacity: 1,
    ambientIntensity: 0.82,
    keyLightIntensity: 1.25,
    headLightIntensity: 0.7,
    rimStrength: 0.7,
    distanceFadeStrength: 0.35,
  },
  interiorCutaway: {
    solidShellEnabled: true,
    lightingEnabled: false,
    interiorBackfaceEnabled: true,
    rimEnabled: false,
    distanceFadeEnabled: false,
    frontOpacity: 1.2,
    backfaceOpacity: 0.32,
    ambientIntensity: 2,
    keyLightIntensity: 0,
    headLightIntensity: 0,
    rimStrength: 0.7,
    distanceFadeStrength: 0.35,
  },
  distanceCue: {
    solidShellEnabled: true,
    lightingEnabled: false,
    interiorBackfaceEnabled: false,
    rimEnabled: false,
    distanceFadeEnabled: true,
    frontOpacity: 1.2,
    backfaceOpacity: 0.28,
    ambientIntensity: 2,
    keyLightIntensity: 0,
    headLightIntensity: 0,
    rimStrength: 0.75,
    distanceFadeStrength: 0.72,
  },
  fullPop: {
    solidShellEnabled: true,
    lightingEnabled: true,
    interiorBackfaceEnabled: true,
    rimEnabled: true,
    distanceFadeEnabled: true,
    frontOpacity: 1.24,
    backfaceOpacity: 0.32,
    ambientIntensity: 0.78,
    keyLightIntensity: 1.35,
    headLightIntensity: 0.9,
    rimStrength: 0.92,
    distanceFadeStrength: 0.68,
  },
};

const MOISTURE_STRUCTURE_PRESET_STATE: Record<
  MoistureStructurePreset,
  MoistureStructurePresetState
> = {
  none: {
    focusMode: "none",
    pickMode: false,
    nonSelectedOpacity: 0.06,
    colorMode: "pressureBands",
    verticalWallFadeEnabled: false,
    verticalWallFadeStrength: 0,
    segmentationMode: "p95-close",
    footprintOverlayEnabled: false,
  },
  currentDepth: {
    focusMode: "none",
    pickMode: false,
    nonSelectedOpacity: 0.06,
    colorMode: "pressureBands",
    verticalWallFadeEnabled: false,
    verticalWallFadeStrength: 0.55,
    segmentationMode: "p95-close",
    footprintOverlayEnabled: false,
  },
  componentRead: {
    focusMode: "none",
    pickMode: false,
    nonSelectedOpacity: 0.06,
    colorMode: "componentHybrid",
    verticalWallFadeEnabled: true,
    verticalWallFadeStrength: 0.55,
    segmentationMode: "p95-close",
    footprintOverlayEnabled: false,
  },
  selectedFocus: {
    focusMode: "dimOthers",
    pickMode: true,
    nonSelectedOpacity: 0.06,
    colorMode: "selectedMonochrome",
    verticalWallFadeEnabled: true,
    verticalWallFadeStrength: 0.55,
    segmentationMode: "p95-close",
    footprintOverlayEnabled: true,
  },
  macroForm: {
    focusMode: "none",
    pickMode: false,
    nonSelectedOpacity: 0.06,
    colorMode: "componentSolid",
    verticalWallFadeEnabled: true,
    verticalWallFadeStrength: 0.72,
    segmentationMode: "p95-close",
    footprintOverlayEnabled: true,
  },
  thresholdCompare: {
    focusMode: "none",
    pickMode: false,
    nonSelectedOpacity: 0.06,
    colorMode: "componentHybrid",
    verticalWallFadeEnabled: true,
    verticalWallFadeStrength: 0.55,
    segmentationMode: "p95-close",
    footprintOverlayEnabled: false,
  },
};

export function getMoistureVisualPresetState(
  preset: MoistureVisualPreset
): MoistureVisualPresetState {
  return MOISTURE_VISUAL_PRESET_STATE[preset];
}

export function getMoistureStructurePresetState(
  preset: MoistureStructurePreset
): MoistureStructurePresetState {
  return MOISTURE_STRUCTURE_PRESET_STATE[preset];
}

export const EXAMPLE_LAYER_PRESETS = {
  shaderMeshVisible: {
    pressureLevel: 250 as ExampleShaderMeshPressure,
    uValueMin: -7.0e-4,
    uValueMax: 7.0e-4,
    uGamma: 0.5,
    uAlpha: 0.95,
    uZeroEps: 0.08,
    uAsinhK: 3,
  },
  contoursVisible: {
    pressureLevel: "msl" as ExampleContoursPressure,
    contrast: 2.0,
    opacity: 0.95,
  },
  particleVisible: {
    pressureLevel: 925 as ExampleParticlePressure,
  },
} as const;

type ControlsState = {
  moistureStructureLayer: MoistureStructureLayerState;
  moistureStructureFrame: MoistureSidebarFrameState;
  exampleShaderMeshLayer: ExampleShaderMeshLayerState;
  exampleContoursLayer: ExampleContoursLayerState;
  exampleParticleLayer: ExampleParticleLayerState;
  setMoistureStructureLayer: (
    patch: Partial<MoistureStructureLayerState>
  ) => void;
  setMoistureVisualPreset: (preset: MoistureVisualPreset) => void;
  resetMoistureVisualPreset: () => void;
  setMoistureStructurePreset: (preset: MoistureStructurePreset) => void;
  resetMoistureStructurePreset: () => void;
  setMoistureStructureFrame: (frame: MoistureSidebarFrameState) => void;
  setExampleShaderMeshLayer: (
    patch: Partial<ExampleShaderMeshLayerState>
  ) => void;
  setExampleContoursLayer: (patch: Partial<ExampleContoursLayerState>) => void;
  setExampleParticleLayer: (patch: Partial<ExampleParticleLayerState>) => void;
};

export const useControls = create<ControlsState>()(
  subscribeWithSelector((set) => ({
    moistureStructureLayer: {
      visible: true,
      opacity: 0.78,
      verticalExaggeration: 4,
      cameraCutawayEnabled: true,
      cameraCutawayRadius: 40,
      visualPreset: "fullPop",
      structurePreset: "componentRead",
      ...getMoistureVisualPresetState("fullPop"),
      ...getMoistureStructurePresetState("componentRead"),
      selectedComponentId: null,
      componentSort: "size",
    },
    moistureStructureFrame: null,
    // Defaults keep the examples hidden from the UI. Use EXAMPLE_LAYER_PRESETS
    // in code when you want to turn one on for local experimentation.
    exampleShaderMeshLayer: {
      pressureLevel: "none",
      uValueMin: -7.0e-4,
      uValueMax: 7.0e-4,
      uGamma: 0.5,
      uAlpha: 0.95,
      uZeroEps: 0.08,
      uAsinhK: 3,
    },
    exampleContoursLayer: {
      pressureLevel: "none",
      contrast: 2.0,
      opacity: 0.95,
    },
    exampleParticleLayer: {
      pressureLevel: "none",
    },
    setMoistureStructureLayer: (patch) =>
      set((state) => ({
        moistureStructureLayer: {
          ...state.moistureStructureLayer,
          ...patch,
        },
      })),
    setMoistureVisualPreset: (preset) =>
      set((state) => ({
        moistureStructureLayer: {
          ...state.moistureStructureLayer,
          visualPreset: preset,
          ...getMoistureVisualPresetState(preset),
        },
      })),
    resetMoistureVisualPreset: () =>
      set((state) => ({
        moistureStructureLayer: {
          ...state.moistureStructureLayer,
          ...getMoistureVisualPresetState(state.moistureStructureLayer.visualPreset),
        },
      })),
    setMoistureStructurePreset: (preset) =>
      set((state) => ({
        moistureStructureLayer: {
          ...state.moistureStructureLayer,
          structurePreset: preset,
          ...getMoistureStructurePresetState(preset),
        },
      })),
    resetMoistureStructurePreset: () =>
      set((state) => ({
        moistureStructureLayer: {
          ...state.moistureStructureLayer,
          ...getMoistureStructurePresetState(
            state.moistureStructureLayer.structurePreset
          ),
        },
      })),
    setMoistureStructureFrame: (frame) =>
      set(() => ({
        moistureStructureFrame: frame,
      })),
    setExampleShaderMeshLayer: (patch) =>
      set((state) => ({
        exampleShaderMeshLayer: {
          ...state.exampleShaderMeshLayer,
          ...patch,
        },
      })),
    setExampleContoursLayer: (patch) =>
      set((state) => ({
        exampleContoursLayer: {
          ...state.exampleContoursLayer,
          ...patch,
        },
      })),
    setExampleParticleLayer: (patch) =>
      set((state) => ({
        exampleParticleLayer: {
          ...state.exampleParticleLayer,
          ...patch,
        },
      })),
  }))
);
