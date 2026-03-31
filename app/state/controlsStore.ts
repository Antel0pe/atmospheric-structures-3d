import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";

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

type MoistureStructureLayerState = {
  visible: boolean;
  opacity: number;
};

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
  exampleShaderMeshLayer: ExampleShaderMeshLayerState;
  exampleContoursLayer: ExampleContoursLayerState;
  exampleParticleLayer: ExampleParticleLayerState;
  setMoistureStructureLayer: (
    patch: Partial<MoistureStructureLayerState>
  ) => void;
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
    },
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
