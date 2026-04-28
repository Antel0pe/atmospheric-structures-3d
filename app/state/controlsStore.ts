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

export const POTENTIAL_TEMPERATURE_VARIANT_OPTIONS = [
  { value: "bridge-gap-1", label: "Bridge 1 Missing Level" },
  { value: "bridge-gap-2", label: "Bridge Up To 2 Levels" },
  { value: "fill-between-anchors", label: "Fill Between Anchors" },
  {
    value: "raw-temperature-midpoint-cold-side",
    label: "Raw Temperature Midpoint Polar Side",
  },
  {
    value: "top10-components-sign-growth",
    label: "Top 10% Components + Sign Growth",
  },
] as const;

export type PotentialTemperatureVariant =
  (typeof POTENTIAL_TEMPERATURE_VARIANT_OPTIONS)[number]["value"];

export const AIR_MASS_CLASSIFICATION_VARIANT_OPTIONS = [
  {
    value: "temperature-rh-latmean",
    label: "Temperature + RH Anomaly",
  },
  {
    value: "theta-rh-latmean",
    label: "Theta + RH Anomaly",
  },
  {
    value: "theta-q-latmean",
    label: "Theta + Specific Humidity",
  },
  {
    value: "surface-attached-theta-rh-latmean",
    label: "Surface-Attached Theta + RH",
  },
  {
    value: "theta-anomaly-percentile-tails",
    label: "Theta Percentile Tail Buckets",
  },
  {
    value: "theta-anomaly-stddev-tails",
    label: "Theta Std Dev Tail Buckets",
  },
  {
    value: "theta-anomaly-stddev-side-6neighbor-min100k",
    label: "Theta Std Dev Side Tails >=100k",
  },
  {
    value: "theta-anomaly-stddev-side-6neighbor-min100k-open-top",
    label: "Theta Std Dev Side Tails >=100k Open Top",
  },
] as const;

export type AirMassClassificationVariant =
  (typeof AIR_MASS_CLASSIFICATION_VARIANT_OPTIONS)[number]["value"];

export const POTENTIAL_TEMPERATURE_COLOR_MODE_OPTIONS = [
  { value: "pressureBands", label: "Current Default" },
  {
    value: "precipitableWaterProxy",
    label: "Precipitable Water Proxy",
  },
  { value: "thermalContrast", label: "Hot/Cold Depth Ramp" },
] as const;

export type PotentialTemperatureColorMode =
  (typeof POTENTIAL_TEMPERATURE_COLOR_MODE_OPTIONS)[number]["value"];

export type ExampleShaderMeshLayerState = {
  pressureLevel: ExampleShaderMeshPressure;
  uValueMin: number;
  uValueMax: number;
  uGamma: number;
  uAlpha: number;
  uZeroEps: number;
  uAsinhK: number;
};

export type ExampleContoursLayerState = {
  pressureLevel: ExampleContoursPressure;
  contrast: number;
  opacity: number;
};

export type ExampleParticleLayerState = {
  pressureLevel: ExampleParticlePressure;
};

export type PrecipitationRadarLayerState = {
  visible: boolean;
  opacity: number;
};

export type PrecipitableWaterLayerState = {
  visible: boolean;
  opacity: number;
};

export type PotentialTemperatureLayerState = {
  visible: boolean;
  opacity: number;
  colorMode: PotentialTemperatureColorMode;
  variant: PotentialTemperatureVariant;
  showCellGrid: boolean;
};

export type AirMassClassificationLayerState = {
  visible: boolean;
  opacity: number;
  variant: AirMassClassificationVariant;
  showCellGrid: boolean;
  altitudeRange01: {
    min: number;
    max: number;
  };
  cameraCutawayEnabled: boolean;
  cameraCutawayRadius: number;
  hiddenClassKeys: string[];
};

type ControlsState = {
  verticalExaggeration: number;
  exampleShaderMeshLayer: ExampleShaderMeshLayerState;
  exampleContoursLayer: ExampleContoursLayerState;
  exampleParticleLayer: ExampleParticleLayerState;
  precipitationRadarLayer: PrecipitationRadarLayerState;
  precipitableWaterLayer: PrecipitableWaterLayerState;
  potentialTemperatureLayer: PotentialTemperatureLayerState;
  airMassLayer: AirMassClassificationLayerState;
  setVerticalExaggeration: (verticalExaggeration: number) => void;
  setExampleShaderMeshLayer: (
    patch: Partial<ExampleShaderMeshLayerState>
  ) => void;
  setExampleContoursLayer: (patch: Partial<ExampleContoursLayerState>) => void;
  setExampleParticleLayer: (patch: Partial<ExampleParticleLayerState>) => void;
  setPrecipitationRadarLayer: (
    patch: Partial<PrecipitationRadarLayerState>
  ) => void;
  setPrecipitableWaterLayer: (
    patch: Partial<PrecipitableWaterLayerState>
  ) => void;
  setPotentialTemperatureLayer: (
    patch: Partial<PotentialTemperatureLayerState>
  ) => void;
  setAirMassLayer: (patch: Partial<AirMassClassificationLayerState>) => void;
};

export const useControls = create<ControlsState>()(
  subscribeWithSelector((set) => ({
    verticalExaggeration: 2.35,
    exampleShaderMeshLayer: {
      pressureLevel: "none",
      uValueMin: -0.0006,
      uValueMax: 0.001,
      uGamma: 0.7,
      uAlpha: 0.72,
      uZeroEps: 0.00002,
      uAsinhK: 9000,
    },
    exampleContoursLayer: {
      pressureLevel: "none",
      contrast: 1,
      opacity: 0.75,
    },
    exampleParticleLayer: {
      pressureLevel: "none",
    },
    precipitationRadarLayer: {
      visible: false,
      opacity: 0.92,
    },
    precipitableWaterLayer: {
      visible: false,
      opacity: 1,
    },
    potentialTemperatureLayer: {
      visible: false,
      opacity: 1,
      colorMode: "pressureBands",
      variant: "bridge-gap-1",
      showCellGrid: false,
    },
    airMassLayer: {
      visible: true,
      opacity: 1,
      variant: "theta-anomaly-stddev-side-6neighbor-min100k",
      showCellGrid: false,
      altitudeRange01: { min: 0, max: 1 },
      cameraCutawayEnabled: false,
      cameraCutawayRadius: 40,
      hiddenClassKeys: [],
    },
    setVerticalExaggeration: (verticalExaggeration) =>
      set(() => ({
        verticalExaggeration,
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
    setPrecipitationRadarLayer: (patch) =>
      set((state) => ({
        precipitationRadarLayer: {
          ...state.precipitationRadarLayer,
          ...patch,
        },
      })),
    setPrecipitableWaterLayer: (patch) =>
      set((state) => ({
        precipitableWaterLayer: {
          ...state.precipitableWaterLayer,
          ...patch,
          opacity: 1,
        },
      })),
    setPotentialTemperatureLayer: (patch) =>
      set((state) => ({
        potentialTemperatureLayer: {
          ...state.potentialTemperatureLayer,
          ...patch,
          opacity: 1,
        },
      })),
    setAirMassLayer: (patch) =>
      set((state) => ({
        airMassLayer: {
          ...state.airMassLayer,
          ...patch,
          opacity: 1,
        },
      })),
  }))
);
