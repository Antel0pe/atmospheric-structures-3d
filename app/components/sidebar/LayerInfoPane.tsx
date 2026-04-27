"use client";

import { useEffect, useMemo, useState } from "react";
import {
  MOISTURE_COMPONENT_SORT_OPTIONS,
  MOISTURE_STRUCTURE_PRESET_OPTIONS,
  MOISTURE_VISUAL_PRESET_OPTIONS,
  getMoistureStructurePresetState,
  getMoistureVisualPresetState,
  moistureLegibilityExperimentLabel,
  moistureSegmentationModeLabel,
  moistureSurfaceCueModeLabel,
  resolveMoistureStructureLayerState,
  useControls,
  type AirMassClassificationVariant,
  type MoistureStructureLayerState,
} from "@/app/state/controlsStore";
import { useViewerStore } from "@/app/state/viewerStore";
import type {
  MoistureStructureComponentMetadata,
} from "../utils/ApiResponses";
import { snapTimestampToAvailable } from "../utils/ApiResponses";
import {
  fetchAirMassStructureManifest,
  type AirMassStructureClassKey,
} from "../utils/airMassStructureAssets";

type ActiveExampleId =
  | "moistureStructureLayer"
  | "potentialTemperatureLayer"
  | "airMassLayer"
  | "exampleShaderMeshLayer"
  | "exampleContoursLayer"
  | "exampleParticleLayer";

type LegendItem = {
  label: string;
  detail: string;
  swatch: string;
};

type LayerInfoEntry = {
  id: ActiveExampleId;
  title: string;
  summary: string;
  detail: string;
  legend: LegendItem[];
};

type AirMassComponentEntry = {
  key: AirMassStructureClassKey;
  label: string;
  color: string;
  voxelCount: number;
  componentCount: number;
};

const LAYER_INFO: Record<ActiveExampleId, Omit<LayerInfoEntry, "id">> = {
  moistureStructureLayer: {
    title: "Moisture Structures Layer",
    summary:
      "A 3D humidity-body renderer that loads precomputed moisture meshes, exaggerates their height in the client, applies a camera-following cutaway, and now adds structure-focused presets for component reading, picking, wall suppression, and macro footprints.",
    detail:
      "Humid regions are detected with pressure-relative thresholds, meshed offline, and rendered here as globe-space moisture bodies. The review harness now lets you compare pressure bands versus component-aware colors, focus individual components, suppress side-wall clutter, and project selected footprints back onto the globe.",
    legend: [
      {
        label: "Pressure Bands",
        detail: "Discrete per-level coloring for seeing vertical structure and stacked altitude slices.",
        swatch:
          "linear-gradient(135deg, rgba(185, 92, 255, 0.96), rgba(149, 69, 235, 0.84))",
      },
      {
        label: "Component Colors",
        detail: "Stable per-component hues for reading continuity and large-scale shape.",
        swatch:
          "linear-gradient(135deg, rgba(94, 134, 255, 0.96), rgba(66, 99, 220, 0.84))",
      },
      {
        label: "Footprint Overlay",
        detail: "Projected outlines of selected or dominant components for macro context.",
        swatch:
          "linear-gradient(135deg, rgba(45, 198, 214, 0.96), rgba(255, 138, 99, 0.84))",
      },
    ],
  },
  potentialTemperatureLayer: {
    title: "Potential Temperature Layer",
    summary:
      "A thermal compare layer that now mixes two structure families: dry-potential-temperature anomaly shells relative to the matched climatology, and a raw-temperature midpoint cold-side shell that isolates the polar-side body directly from smoothed temperature.",
    detail:
      "Most variants still derive dry potential temperature from pressure-level temperature, subtract the matched climatological mean field, and render separate warm and cold voxel shells. The raw-temperature midpoint variant instead smooths raw temperature level by level, finds T_mid = 0.5 * (T_min + T_max), and keeps only the cold side of that boundary through the 1000-250 hPa window, with only pole-edge cleanup if the sampled cap would otherwise break topological coherence.",
    legend: [
      {
        label: "Default Pressure Bands",
        detail: "Separate warm and cold palettes keyed by pressure band for reading anomaly sign and depth together.",
        swatch:
          "linear-gradient(135deg, rgba(255, 90, 54, 0.96), rgba(255, 211, 77, 0.9))",
      },
      {
        label: "Proxy Ramp",
        detail: "The precipitable-water-proxy ramp applied to potential temperature so layer shapes can be compared with matching depth colors.",
        swatch:
          "linear-gradient(135deg, rgba(255, 138, 99, 0.96), rgba(185, 92, 255, 0.9))",
      },
      {
        label: "Hot/Cold Depth Ramp",
        detail: "Thermal red-for-warm and blue-for-cold shading that darkens toward the lower atmosphere.",
        swatch:
          "linear-gradient(135deg, rgba(110, 14, 24, 0.96), rgba(33, 86, 191, 0.9))",
      },
    ],
  },
  airMassLayer: {
    title: "Air Mass Classification Layer",
    summary:
      "A 3D proxy-classification layer that combines warm/cold and moist/dry anomaly axes into four quadrant shells, so the globe shows coherent tropical, polar, continental, and maritime-like bodies without claiming full source-region analysis.",
    detail:
      "Each recipe clips the main troposphere, computes either raw-temperature or dry-potential-temperature anomalies together with a moisture anomaly axis, standardizes both per pressure level, keeps only the strongest combined cells, and then separates the surviving volume into warm-dry, warm-moist, cold-dry, and cold-moist shells. These are proxy classes derived from local thermodynamic structure, not true Bergeron air-mass trajectories.",
    legend: [
      {
        label: "Warm Dry",
        detail: "Continental-tropical and superior-style proxy bodies.",
        swatch:
          "linear-gradient(135deg, rgba(207, 95, 33, 0.96), rgba(255, 223, 123, 0.9))",
      },
      {
        label: "Warm Moist",
        detail: "Maritime-tropical and monsoon-style proxy bodies.",
        swatch:
          "linear-gradient(135deg, rgba(181, 72, 62, 0.96), rgba(255, 176, 135, 0.9))",
      },
      {
        label: "Cold Dry / Moist",
        detail: "Polar-continental and maritime-polar proxy shells, split between deep blue and teal families.",
        swatch:
          "linear-gradient(135deg, rgba(39, 87, 191, 0.96), rgba(73, 196, 190, 0.9))",
      },
    ],
  },
  exampleShaderMeshLayer: {
    title: "Example Shader Mesh Layer",
    summary:
      "A reference globe-mesh shader that crossfades hourly raster textures and maps signed values into a warm/cool palette.",
    detail:
      "Use this as the boilerplate for mesh-based overlays that decode image data onto a sphere and animate transitions between timestamps.",
    legend: [
      {
        label: "Warm values",
        detail: "Positive-side values from the decoded raster range.",
        swatch:
          "linear-gradient(135deg, rgba(255, 219, 64, 0.96), rgba(255, 168, 0, 0.88))",
      },
      {
        label: "Cool values",
        detail: "Negative-side values from the decoded raster range.",
        swatch:
          "linear-gradient(135deg, rgba(41, 199, 102, 0.92), rgba(18, 111, 66, 0.84))",
      },
    ],
  },
  exampleContoursLayer: {
    title: "Example Contours Layer",
    summary:
      "A reference contour renderer that loads JSON linework, colors each level, and crossfades between timestamps.",
    detail:
      "Use this as the boilerplate for line-based overlays that need per-level styling and smooth replacement of contour groups.",
    legend: [
      {
        label: "Lower levels",
        detail: "Lower contour values in the current level range.",
        swatch:
          "linear-gradient(90deg, rgba(0, 255, 38, 0.95), rgba(68, 255, 126, 0.9))",
      },
      {
        label: "Higher levels",
        detail: "Higher contour values in the current level range.",
        swatch:
          "linear-gradient(90deg, rgba(255, 78, 129, 0.95), rgba(255, 0, 89, 0.9))",
      },
    ],
  },
  exampleParticleLayer: {
    title: "Example Particle Layer",
    summary:
      "A reference GPU particle system that advects positions from a wind texture and projects the result back onto the globe.",
    detail:
      "Use this as the boilerplate for particle or trail layers that simulate motion on the GPU and then render the result in globe space.",
    legend: [
      {
        label: "Particle trails",
        detail: "Advected particles rendered as cyan globe-space points and trails.",
        swatch:
          "linear-gradient(135deg, rgba(36, 223, 223, 0.96), rgba(124, 239, 255, 0.9))",
      },
    ],
  },
};

function airMassInfoForVariant(
  variant: AirMassClassificationVariant
): Omit<LayerInfoEntry, "id"> {
  if (variant.startsWith("theta-anomaly-")) {
    return {
      title: "Theta Anomaly Bucket Layer",
      summary:
        "A thermal tail-bucket shell that shows dry-potential-temperature anomaly structures, not moisture-based air-mass quadrants.",
      detail:
        "This variant keeps only cells outside +/-1 standard deviation from the per-level anomaly distribution, groups 3D components by cold or warm side, and renders the surviving bucket shells. Blue means cold anomaly, red means warm anomaly; lightness tracks anomaly bucket strength, while saturation increases upward so higher-altitude structure reads stronger.",
      legend: [
        {
          label: "Cold Buckets",
          detail: "Buckets 0-2: colder dry-theta anomaly tails, from strongest/darkest to weaker/lighter blue.",
          swatch:
            "linear-gradient(135deg, hsl(212 100% 24%), hsl(212 46% 64%))",
        },
        {
          label: "Warm Buckets",
          detail: "Buckets 7-9: warmer dry-theta anomaly tails, from weaker/lighter red-orange to strongest/darkest red.",
          swatch:
            "linear-gradient(135deg, hsl(9 46% 62%), hsl(0 100% 27%))",
        },
        {
          label: "Altitude Saturation",
          detail: "Same bucket hue/lightness, with stronger saturation aloft and softer saturation lower in the structure.",
          swatch:
            "linear-gradient(135deg, hsl(212 46% 42%), hsl(212 100% 42%), hsl(0 100% 43%))",
        },
      ],
    };
  }
  return LAYER_INFO.airMassLayer;
}

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

function actionButtonStyle() {
  return {
    borderRadius: 999,
    border: "1px solid rgba(255,255,255,0.14)",
    background: "rgba(255,255,255,0.08)",
    color: "#e9eef7",
    padding: "5px 9px",
    cursor: "pointer",
    font: "700 11px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto",
  } as const;
}

type MoistureVisualOverrideKey = keyof ReturnType<typeof getMoistureVisualPresetState>;
type MoistureStructureOverrideKey = keyof ReturnType<
  typeof getMoistureStructurePresetState
>;

const COMPONENT_PALETTE = [
  "#ff8a63",
  "#ffd166",
  "#8ac926",
  "#2dc6d6",
  "#5e86ff",
  "#b95cff",
  "#f72585",
  "#06d6a0",
  "#4895ef",
  "#f4a261",
  "#90be6d",
  "#c77dff",
] as const;

const AIR_MASS_MIN_ALTITUDE_RANGE_SPAN = 0.01;

type AirMassPressureWindow = {
  min: number;
  max: number;
};

const MOISTURE_VISUAL_OVERRIDE_LABELS: Record<MoistureVisualOverrideKey, string> = {
  solidShellEnabled: "Shell",
  lightingEnabled: "Light",
  interiorBackfaceEnabled: "Interior",
  rimEnabled: "Rim",
  distanceFadeEnabled: "Distance",
  frontOpacity: "Front",
  backfaceOpacity: "Back",
  ambientIntensity: "Ambient",
  keyLightIntensity: "Key",
  headLightIntensity: "Head",
  rimStrength: "Rim",
  distanceFadeStrength: "Fade",
};

const MOISTURE_STRUCTURE_OVERRIDE_LABELS: Record<
  MoistureStructureOverrideKey,
  string
> = {
  focusMode: "Focus",
  pickMode: "Pick",
  nonSelectedOpacity: "Dim",
  colorMode: "Color",
  verticalWallFadeEnabled: "Walls",
  verticalWallFadeStrength: "Walls",
  segmentationMode: "Seg",
  footprintOverlayEnabled: "Footprint",
};

function moisturePresetLabel(value: MoistureStructureLayerState["visualPreset"]) {
  return (
    MOISTURE_VISUAL_PRESET_OPTIONS.find((option) => option.value === value)?.label ??
    value
  );
}

function moistureStructurePresetLabel(
  value: MoistureStructureLayerState["structurePreset"]
) {
  return (
    MOISTURE_STRUCTURE_PRESET_OPTIONS.find((option) => option.value === value)
      ?.label ?? value
  );
}

function moistureOverrideSummary(state: MoistureStructureLayerState) {
  const presetState = getMoistureVisualPresetState(state.visualPreset);
  const overrideKeys = (Object.keys(presetState) as MoistureVisualOverrideKey[]).filter(
    (key) => state[key] !== presetState[key]
  );
  if (overrideKeys.length === 0) return "Preset";
  const labels = overrideKeys
    .slice(0, 3)
    .map((key) => MOISTURE_VISUAL_OVERRIDE_LABELS[key]);
  const suffix = overrideKeys.length > 3 ? ` +${overrideKeys.length - 3}` : "";
  return `${labels.join(" / ")}${suffix}`;
}

function moistureStructureOverrideSummary(state: MoistureStructureLayerState) {
  const presetState = getMoistureStructurePresetState(state.structurePreset);
  const overrideKeys = (
    Object.keys(presetState) as MoistureStructureOverrideKey[]
  ).filter((key) => state[key] !== presetState[key]);
  if (overrideKeys.length === 0) return "Preset";
  const labels = overrideKeys
    .slice(0, 3)
    .map((key) => MOISTURE_STRUCTURE_OVERRIDE_LABELS[key]);
  const suffix = overrideKeys.length > 3 ? ` +${overrideKeys.length - 3}` : "";
  return `${labels.join(" / ")}${suffix}`;
}

function componentColorForId(id: number) {
  return COMPONENT_PALETTE[id % COMPONENT_PALETTE.length];
}

function moistureComponentLabel(component: MoistureStructureComponentMetadata) {
  if (typeof component.bucket_index === "number") {
    return `Bucket ${component.bucket_index + 1}`;
  }
  return `Component ${component.id}`;
}

function moistureComponentColor(component: MoistureStructureComponentMetadata) {
  return componentColorForId(component.bucket_index ?? component.id);
}

function airMassFallbackColor(classKey: AirMassStructureClassKey, index: number) {
  const colors: Record<string, string> = {
    warm_dry: "#d36a2f",
    warm_moist: "#c2534d",
    cold_dry: "#2252a8",
    cold_moist: "#188f99",
    bucket_0: "#08306b",
    bucket_1: "#2171b5",
    bucket_2: "#6baed6",
    bucket_7: "#fb6a4a",
    bucket_8: "#cb181d",
    bucket_9: "#67000d",
  };
  return colors[classKey] ?? COMPONENT_PALETTE[index % COMPONENT_PALETTE.length];
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(value >= 0.1 ? 0 : 1)}%`;
}

function pressureToStandardAtmosphereHeightM(pressureHpa: number) {
  const safePressure = Math.max(pressureHpa, 1);
  return 44330.0 * (1.0 - (safePressure / 1013.25) ** 0.1903);
}

function standardAtmosphereHeightMToPressure(heightM: number) {
  const normalized = Math.min(Math.max(1.0 - heightM / 44330.0, 1e-6), 1);
  return 1013.25 * normalized ** (1 / 0.1903);
}

function pressureForAltitudeMix(
  pressureWindow: AirMassPressureWindow,
  altitudeMix: number
) {
  const lowerPressure = Math.max(pressureWindow.min, pressureWindow.max);
  const upperPressure = Math.min(pressureWindow.min, pressureWindow.max);
  const lowerHeight = pressureToStandardAtmosphereHeightM(lowerPressure);
  const upperHeight = pressureToStandardAtmosphereHeightM(upperPressure);
  return standardAtmosphereHeightMToPressure(
    lowerHeight + Math.min(Math.max(altitudeMix, 0), 1) * (upperHeight - lowerHeight)
  );
}

function normalizeAirMassAltitudeRange(
  range: {
    min: number;
    max: number;
  }
) {
  const min = Math.min(Math.max(range.min, 0), 1);
  const max = Math.min(Math.max(range.max, 0), 1);
  if (max - min >= AIR_MASS_MIN_ALTITUDE_RANGE_SPAN) {
    return { min, max };
  }
  if (min <= 1 - AIR_MASS_MIN_ALTITUDE_RANGE_SPAN) {
    return { min, max: min + AIR_MASS_MIN_ALTITUDE_RANGE_SPAN };
  }
  return { min: 1 - AIR_MASS_MIN_ALTITUDE_RANGE_SPAN, max: 1 };
}

function formatAirMassAltitudeRangeLabel(
  pressureWindow: AirMassPressureWindow,
  range: {
    min: number;
    max: number;
  }
) {
  const normalizedRange = normalizeAirMassAltitudeRange(range);
  const lowerPressure = pressureForAltitudeMix(pressureWindow, normalizedRange.min);
  const upperPressure = pressureForAltitudeMix(pressureWindow, normalizedRange.max);
  return `${lowerPressure.toFixed(0)}-${upperPressure.toFixed(0)} hPa`;
}

function AirMassAltitudeRangeControl({
  pressureWindow,
  range,
  onChange,
}: {
  pressureWindow: AirMassPressureWindow;
  range: { min: number; max: number };
  onChange: (range: { min: number; max: number }) => void;
}) {
  const normalizedRange = normalizeAirMassAltitudeRange(range);

  const minPercent = normalizedRange.min * 100;
  const maxPercent = normalizedRange.max * 100;

  const updateMin = (value: number) => {
    onChange(
      normalizeAirMassAltitudeRange({
        min: Math.min(
          value / 100,
          normalizedRange.max - AIR_MASS_MIN_ALTITUDE_RANGE_SPAN
        ),
        max: normalizedRange.max,
      })
    );
  };

  const updateMax = (value: number) => {
    onChange(
      normalizeAirMassAltitudeRange({
        min: normalizedRange.min,
        max: Math.max(
          value / 100,
          normalizedRange.min + AIR_MASS_MIN_ALTITUDE_RANGE_SPAN
        ),
      })
    );
  };
  return (
    <div style={{ display: "grid", gap: 8 }}>
      <style>{`
        .air-mass-altitude-range {
          position: relative;
          height: 30px;
        }
        .air-mass-altitude-range input[type="range"] {
          position: absolute;
          inset: 0;
          width: 100%;
          height: 30px;
          margin: 0;
          pointer-events: none;
          appearance: none;
          background: transparent;
        }
        .air-mass-altitude-range input[type="range"]::-webkit-slider-thumb {
          pointer-events: auto;
          appearance: none;
          width: 16px;
          height: 16px;
          border-radius: 999px;
          border: 2px solid rgba(255, 255, 255, 0.92);
          background: #8fe7c7;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.45);
          cursor: pointer;
        }
        .air-mass-altitude-range input[type="range"]::-moz-range-thumb {
          pointer-events: auto;
          width: 16px;
          height: 16px;
          border-radius: 999px;
          border: 2px solid rgba(255, 255, 255, 0.92);
          background: #8fe7c7;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.45);
          cursor: pointer;
        }
        .air-mass-altitude-range input[type="range"]::-webkit-slider-runnable-track {
          appearance: none;
          height: 30px;
          background: transparent;
        }
        .air-mass-altitude-range input[type="range"]::-moz-range-track {
          height: 30px;
          background: transparent;
        }
      `}</style>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: 8,
          fontWeight: 600,
        }}
      >
        <span>Altitude Range</span>
        <span style={{ opacity: 0.68 }}>
          {formatAirMassAltitudeRangeLabel(pressureWindow, normalizedRange)}
        </span>
      </div>
      <div className="air-mass-altitude-range">
        <div
          aria-hidden
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            top: 13,
            height: 4,
            borderRadius: 999,
            background: "rgba(255,255,255,0.16)",
          }}
        />
        <div
          aria-hidden
          style={{
            position: "absolute",
            left: `${minPercent}%`,
            right: `${100 - maxPercent}%`,
            top: 13,
            height: 4,
            borderRadius: 999,
            background:
              "linear-gradient(90deg, rgba(143,231,199,0.92), rgba(122,173,255,0.92))",
          }}
        />
        <input
          type="range"
          min={0}
          max={100}
          step={1}
          value={minPercent}
          aria-label="Air mass lower altitude bound"
          onChange={(event) => updateMin(Number(event.currentTarget.value))}
        />
        <input
          type="range"
          min={0}
          max={100}
          step={1}
          value={maxPercent}
          aria-label="Air mass upper altitude bound"
          onChange={(event) => updateMax(Number(event.currentTarget.value))}
        />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", opacity: 0.62 }}>
        <span>Surface</span>
        <span>Top</span>
      </div>
    </div>
  );
}

function CompactRangeControl({
  label,
  value,
  valueLabel,
  min,
  max,
  step,
  disabled,
  onChange,
}: {
  label: string;
  value: number;
  valueLabel: string;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <label style={{ display: "grid", gap: 8, opacity: disabled ? 0.52 : 1 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: 8,
          fontWeight: 600,
        }}
      >
        <span>{label}</span>
        <span style={{ opacity: 0.68 }}>{valueLabel}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
        style={{ width: "100%", accentColor: "#8fe7c7" }}
      />
    </label>
  );
}

function componentPressureSpan(component: MoistureStructureComponentMetadata) {
  return component.pressure_max_hpa - component.pressure_min_hpa;
}

function sortComponents(
  components: MoistureStructureComponentMetadata[],
  sort: MoistureStructureLayerState["componentSort"]
) {
  return components.slice().sort((left, right) => {
    if (sort === "pressureSpan") {
      return componentPressureSpan(right) - componentPressureSpan(left);
    }
    if (sort === "peakHumidity") {
      return right.max_specific_humidity - left.max_specific_humidity;
    }
    return right.voxel_count - left.voxel_count;
  });
}

function splitRingForMap(ring: Array<[number, number]>) {
  if (ring.length === 0) return [];

  const segments: Array<Array<[number, number]>> = [];
  let current: Array<[number, number]> = [];

  for (let index = 0; index < ring.length; index += 1) {
    const [longitude, latitude] = ring[index];
    const normalizedLongitude = ((longitude % 360) + 360) % 360;
    const point: [number, number] = [normalizedLongitude, latitude];

    if (current.length > 0) {
      const previousLongitude = current[current.length - 1][0];
      if (Math.abs(normalizedLongitude - previousLongitude) > 180) {
        segments.push(current);
        current = [];
      }
    }

    current.push(point);
  }

  if (current.length > 1) {
    segments.push(current);
  }

  return segments;
}

function ringPath(segment: Array<[number, number]>) {
  if (segment.length < 2) return "";
  return segment
    .map(([longitude, latitude], index) => {
      const x = longitude;
      const y = 90 - latitude;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

export default function LayerInfoPane() {
  const timestamp = useViewerStore((state) => state.timestamp);
  const moistureStructureLayerState = useControls(
    (state) => state.moistureStructureLayer
  );
  const moistureStructureLayer = useMemo(
    () => resolveMoistureStructureLayerState(moistureStructureLayerState),
    [moistureStructureLayerState]
  );
  const moistureStructureFrame = useControls((state) => state.moistureStructureFrame);
  const setMoistureStructureLayer = useControls(
    (state) => state.setMoistureStructureLayer
  );
  const potentialTemperatureLayer = useControls(
    (state) => state.potentialTemperatureLayer
  );
  const airMassLayer = useControls((state) => state.airMassLayer);
  const setAirMassLayer = useControls((state) => state.setAirMassLayer);
  const exampleShaderMeshLayer = useControls(
    (state) => state.exampleShaderMeshLayer
  );
  const exampleContoursLayer = useControls(
    (state) => state.exampleContoursLayer
  );
  const exampleParticleLayer = useControls(
    (state) => state.exampleParticleLayer
  );

  const sortedMoistureComponents = useMemo(() => {
    if (!moistureStructureFrame) return [];
    return sortComponents(
      moistureStructureFrame.components,
      moistureStructureLayer.componentSort
    );
  }, [moistureStructureFrame, moistureStructureLayer.componentSort]);

  const selectedMoistureComponent = useMemo(() => {
    if (!moistureStructureFrame || moistureStructureLayer.selectedComponentId === null) {
      return null;
    }
    return (
      moistureStructureFrame.components.find(
        (component) => component.id === moistureStructureLayer.selectedComponentId
      ) ?? null
    );
  }, [moistureStructureFrame, moistureStructureLayer.selectedComponentId]);

  const selectedMoistureFootprint = useMemo(() => {
    if (!moistureStructureFrame || moistureStructureLayer.selectedComponentId === null) {
      return null;
    }
    return (
      moistureStructureFrame.footprints.find(
        (footprint) => footprint.id === moistureStructureLayer.selectedComponentId
      ) ?? null
    );
  }, [moistureStructureFrame, moistureStructureLayer.selectedComponentId]);

  const [airMassComponentState, setAirMassComponentState] = useState<{
    variant: AirMassClassificationVariant;
    entries: AirMassComponentEntry[];
    pressureWindow: AirMassPressureWindow;
  } | null>(null);

  useEffect(() => {
    if (!airMassLayer.visible) {
      return;
    }

    let cancelled = false;
    void fetchAirMassStructureManifest({
      variant: airMassLayer.variant,
      notifyOnError: false,
    })
      .then((manifest) => {
        if (cancelled) return;
        const availableTimestamps = manifest.timestamps.map((entry) => entry.timestamp);
        const resolvedTimestamp = snapTimestampToAvailable(
          timestamp,
          availableTimestamps
        );
        const timestampEntry =
          manifest.timestamps.find((entry) => entry.timestamp === resolvedTimestamp) ??
          manifest.timestamps[0];
        const entries = manifest.classification.classes.map((classEntry, index) => {
          const counts = timestampEntry?.class_counts[classEntry.key];
          return {
            key: classEntry.key,
            label: classEntry.label,
            color:
              classEntry.color ??
              airMassFallbackColor(classEntry.key, index),
            voxelCount: counts?.voxel_count ?? 0,
            componentCount: counts?.component_count ?? 0,
          };
        });
        setAirMassComponentState({
          variant: airMassLayer.variant,
          entries,
          pressureWindow: manifest.pressure_window_hpa,
        });
      })
      .catch((error) => {
        if (cancelled) return;
        console.error("Failed to load air-mass component list", error);
        setAirMassComponentState({
          variant: airMassLayer.variant,
          entries: [],
          pressureWindow: { min: 250, max: 1000 },
        });
      });

    return () => {
      cancelled = true;
    };
  }, [airMassLayer.variant, airMassLayer.visible, timestamp]);

  const airMassComponents =
    airMassLayer.visible && airMassComponentState?.variant === airMassLayer.variant
      ? airMassComponentState.entries
      : [];
  const airMassPressureWindow =
    airMassLayer.visible && airMassComponentState?.variant === airMassLayer.variant
      ? airMassComponentState.pressureWindow
      : { min: 250, max: 1000 };

  const hiddenAirMassClassKeys = useMemo(
    () => new Set(airMassLayer.hiddenClassKeys),
    [airMassLayer.hiddenClassKeys]
  );

  const visibleAirMassComponentCount = airMassComponents.filter(
    (component) => !hiddenAirMassClassKeys.has(component.key)
  ).length;

  const setAirMassClassVisible = (
    classKey: AirMassStructureClassKey,
    visible: boolean
  ) => {
    const hiddenClassKeys = new Set(airMassLayer.hiddenClassKeys);
    if (visible) {
      hiddenClassKeys.delete(classKey);
    } else {
      hiddenClassKeys.add(classKey);
    }
    setAirMassLayer({
      hiddenClassKeys: Array.from(hiddenClassKeys).sort(),
    });
  };

  const showAllAirMassComponents = () => {
    const currentKeys = new Set(airMassComponents.map((component) => component.key));
    setAirMassLayer({
      hiddenClassKeys: airMassLayer.hiddenClassKeys.filter(
        (classKey) => !currentKeys.has(classKey)
      ),
    });
  };

  const hideAllAirMassComponents = () => {
    setAirMassLayer({
      hiddenClassKeys: Array.from(
        new Set([
          ...airMassLayer.hiddenClassKeys,
          ...airMassComponents.map((component) => component.key),
        ])
      ).sort(),
    });
  };

  const activeEntries = (() => {
    const entries: Array<LayerInfoEntry & { tag: string }> = [];

    if (moistureStructureLayer.visible) {
      entries.push({
        id: "moistureStructureLayer",
        tag: `${moisturePresetLabel(
          moistureStructureLayer.visualPreset
        )} / ${moistureStructurePresetLabel(
          moistureStructureLayer.structurePreset
        )} | ${moistureLegibilityExperimentLabel(
          moistureStructureLayer.legibilityExperiment
        )} | Cue ${moistureSurfaceCueModeLabel(
          moistureStructureLayer.surfaceCueMode
        )} | ${moistureOverrideSummary(
          moistureStructureLayerState
        )} + ${moistureStructureOverrideSummary(
          moistureStructureLayerState
        )} | ${moistureSegmentationModeLabel(
          moistureStructureLayer.segmentationMode
        )} | Clip ${
          moistureStructureLayer.cameraCutawayEnabled
            ? moistureStructureLayer.cameraCutawayRadius.toFixed(0)
            : "Off"
        }`,
        ...LAYER_INFO.moistureStructureLayer,
      });
    }

    if (potentialTemperatureLayer.visible) {
      const tag =
        potentialTemperatureLayer.variant === "raw-temperature-midpoint-cold-side"
          ? "Raw temperature | Midpoint cold-side shell"
          : potentialTemperatureLayer.variant === "top10-components-sign-growth"
            ? "Climatology dry-theta | Top-10%-component sign growth"
            : "Climatology dry-theta | Bridge / fill shells";
      entries.push({
        id: "potentialTemperatureLayer",
        tag,
        ...LAYER_INFO.potentialTemperatureLayer,
      });
    }

    if (airMassLayer.visible) {
      const tag =
        airMassLayer.variant === "surface-attached-theta-rh-latmean"
          ? "Surface-attached theta / RH proxy"
          : airMassLayer.variant === "theta-rh-latmean"
          ? "Theta / RH anomaly proxy"
          : airMassLayer.variant === "theta-q-latmean"
            ? "Theta / q anomaly proxy"
            : "Temperature / RH anomaly proxy";
      entries.push({
        id: "airMassLayer",
        tag: `${tag} | Range ${formatAirMassAltitudeRangeLabel(
          airMassPressureWindow,
          airMassLayer.altitudeRange01
        )} | Clip ${
          airMassLayer.cameraCutawayEnabled
            ? airMassLayer.cameraCutawayRadius.toFixed(0)
            : "Off"
        }`,
        ...airMassInfoForVariant(airMassLayer.variant),
      });
    }

    if (exampleShaderMeshLayer.pressureLevel !== "none") {
      entries.push({
        id: "exampleShaderMeshLayer",
        tag: `${exampleShaderMeshLayer.pressureLevel} hPa`,
        ...LAYER_INFO.exampleShaderMeshLayer,
      });
    }

    if (exampleContoursLayer.pressureLevel !== "none") {
      const tag =
        exampleContoursLayer.pressureLevel === "msl"
          ? "MSL"
          : `${exampleContoursLayer.pressureLevel} hPa`;
      entries.push({
        id: "exampleContoursLayer",
        tag,
        ...LAYER_INFO.exampleContoursLayer,
      });
    }

    if (exampleParticleLayer.pressureLevel !== "none") {
      entries.push({
        id: "exampleParticleLayer",
        tag: `${exampleParticleLayer.pressureLevel} hPa`,
        ...LAYER_INFO.exampleParticleLayer,
      });
    }

    return entries;
  })();

  return (
    <aside
      style={{
        position: "relative",
        top: 0,
        right: 0,
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        backdropFilter: "blur(6px)",
        background: "transparent",
        borderLeft: "1px solid rgba(255,255,255,0.08)",
        zIndex: 1000,
        overflowY: "auto",
      }}
    >
      {activeEntries.length === 0 ? (
        <section style={sectionStyle()}>
          <div
            style={{
              fontSize: 12,
              fontWeight: 700,
              letterSpacing: ".02em",
              textTransform: "uppercase",
              opacity: 0.9,
              marginBottom: 10,
            }}
          >
            Example Layers
          </div>
          <div style={{ opacity: 0.78, lineHeight: 1.45 }}>
            The moisture structures layer is available from the sidebar, and the
            example shader mesh, contour, and particle layers remain scaffolded in
            code for local experimentation.
          </div>
          <div style={{ opacity: 0.62, lineHeight: 1.45, marginTop: 10 }}>
            Enable one of the presets in `app/state/controlsStore.ts` when you want
            to turn a specific example on for local experimentation.
          </div>
        </section>
      ) : (
        activeEntries.map((entry) => (
          <section key={entry.id} style={sectionStyle()}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 8,
                marginBottom: 10,
              }}
            >
              <div
                style={{
                  fontSize: 12,
                  fontWeight: 700,
                  letterSpacing: ".02em",
                  textTransform: "uppercase",
                  opacity: 0.9,
                }}
              >
                {entry.title}
              </div>
              <div
                style={{
                  padding: "4px 8px",
                  borderRadius: 999,
                  background: "rgba(255,255,255,0.08)",
                  border: "1px solid rgba(255,255,255,0.12)",
                  fontSize: 11,
                  opacity: 0.75,
                }}
              >
                {entry.tag}
              </div>
            </div>
            <div style={{ opacity: 0.86, lineHeight: 1.45 }}>{entry.summary}</div>
            <div style={{ opacity: 0.62, lineHeight: 1.45, marginTop: 8 }}>
              {entry.detail}
            </div>
            <div style={{ display: "grid", gap: 8, marginTop: 12 }}>
              {entry.legend.map((item) => (
                <div
                  key={`${entry.id}-${item.label}`}
                  style={{ display: "grid", gridTemplateColumns: "32px 1fr", gap: 10 }}
                >
                  <div
                    aria-hidden
                    style={{
                      width: 32,
                      height: 32,
                      borderRadius: 10,
                      background: item.swatch,
                      border: "1px solid rgba(255,255,255,0.12)",
                    }}
                  />
                  <div>
                    <div style={{ fontWeight: 700, marginBottom: 2 }}>{item.label}</div>
                    <div style={{ opacity: 0.64, lineHeight: 1.35 }}>{item.detail}</div>
                  </div>
                </div>
              ))}
            </div>
            {entry.id === "moistureStructureLayer" && moistureStructureFrame ? (
              <div
                style={{
                  display: "grid",
                  gap: 10,
                  marginTop: 14,
                  paddingTop: 12,
                  borderTop: "1px solid rgba(255,255,255,0.08)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 8,
                  }}
                >
                  <div
                    style={{
                      fontSize: 11,
                      fontWeight: 700,
                      letterSpacing: "0.08em",
                      textTransform: "uppercase",
                      opacity: 0.7,
                    }}
                  >
                    Components
                  </div>
                  <div style={{ opacity: 0.62 }}>
                    {
                      MOISTURE_COMPONENT_SORT_OPTIONS.find(
                        (option) =>
                          option.value === moistureStructureLayer.componentSort
                      )?.label
                    }
                  </div>
                </div>
                <div style={{ opacity: 0.72, lineHeight: 1.45 }}>
                  {moistureStructureFrame.timestamp} ·{" "}
                  {moistureStructureFrame.components.length} components ·{" "}
                  {moistureSegmentationModeLabel(
                    moistureStructureFrame.segmentationMode
                  )}
                </div>

                {selectedMoistureComponent ? (
                  <div
                    style={{
                      display: "grid",
                      gap: 8,
                      padding: 10,
                      borderRadius: 12,
                      background: "rgba(255,255,255,0.04)",
                      border: "1px solid rgba(255,255,255,0.08)",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        gap: 8,
                      }}
                    >
                      <div style={{ fontWeight: 700 }}>
                        {moistureComponentLabel(selectedMoistureComponent)}
                      </div>
                      <button
                        type="button"
                        onClick={() =>
                          setMoistureStructureLayer({ selectedComponentId: null })
                        }
                        style={{
                          borderRadius: 999,
                          border: "1px solid rgba(255,255,255,0.12)",
                          background: "rgba(255,255,255,0.06)",
                          color: "#e9eef7",
                          padding: "4px 8px",
                          cursor: "pointer",
                        }}
                      >
                        Clear
                      </button>
                    </div>
                    <div style={{ opacity: 0.72, lineHeight: 1.4 }}>
                      {formatPercent(
                        selectedMoistureComponent.voxel_count /
                          Math.max(moistureStructureFrame.thresholdedVoxelCount, 1)
                      )}{" "}
                      of thresholded voxels ·{" "}
                      {selectedMoistureComponent.pressure_min_hpa.toFixed(0)}-
                      {selectedMoistureComponent.pressure_max_hpa.toFixed(0)} hPa ·
                      peak {selectedMoistureComponent.max_specific_humidity.toExponential(2)}
                    </div>
                    <div style={{ opacity: 0.64, lineHeight: 1.4 }}>
                      Lat {selectedMoistureComponent.latitude_min_deg.toFixed(1)} to{" "}
                      {selectedMoistureComponent.latitude_max_deg.toFixed(1)} · Lon{" "}
                      {selectedMoistureComponent.longitude_min_deg.toFixed(1)} to{" "}
                      {selectedMoistureComponent.longitude_max_deg.toFixed(1)} ·{" "}
                      {selectedMoistureComponent.wraps_longitude_seam
                        ? "Wraps seam"
                        : "No seam wrap"}
                    </div>
                    {selectedMoistureFootprint ? (
                      <svg
                        viewBox="0 0 360 180"
                        style={{
                          width: "100%",
                          height: 110,
                          borderRadius: 10,
                          background: "rgba(7, 12, 20, 0.8)",
                          border: "1px solid rgba(255,255,255,0.08)",
                        }}
                      >
                        <rect
                          x="0"
                          y="0"
                          width="360"
                          height="180"
                          fill="rgba(12, 18, 28, 0.9)"
                        />
                        {selectedMoistureFootprint.rings.flatMap((ring, ringIndex) =>
                          splitRingForMap(ring).map((segment, segmentIndex) => (
                            <path
                              key={`${ringIndex}-${segmentIndex}`}
                              d={ringPath(segment)}
                              fill="none"
                              stroke={moistureComponentColor(selectedMoistureComponent)}
                              strokeWidth="2.5"
                              strokeLinejoin="round"
                              strokeLinecap="round"
                            />
                          ))
                        )}
                      </svg>
                    ) : null}
                  </div>
                ) : null}

                <div style={{ display: "grid", gap: 8 }}>
                  {sortedMoistureComponents.map((component, index) => {
                    const selected =
                      component.id === moistureStructureLayer.selectedComponentId;
                    return (
                      <button
                        key={component.id}
                        type="button"
                        onClick={() =>
                          setMoistureStructureLayer({
                            selectedComponentId:
                              selected && moistureStructureLayer.focusMode === "none"
                                ? null
                                : component.id,
                          })
                        }
                        style={{
                          display: "grid",
                          gap: 6,
                          textAlign: "left",
                          padding: 10,
                          borderRadius: 12,
                          border: selected
                            ? "1px solid rgba(152, 200, 255, 0.45)"
                            : "1px solid rgba(255,255,255,0.08)",
                          background: selected
                            ? "rgba(91, 152, 255, 0.14)"
                            : "rgba(255,255,255,0.04)",
                          color: "#e9eef7",
                          cursor: "pointer",
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            gap: 8,
                          }}
                        >
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <span
                              aria-hidden
                              style={{
                                width: 10,
                                height: 10,
                                borderRadius: 999,
                                background: moistureComponentColor(component),
                                boxShadow: "0 0 0 1px rgba(255,255,255,0.18)",
                              }}
                            />
                            <span style={{ fontWeight: 700 }}>
                              #{index + 1} · {moistureComponentLabel(component)}
                            </span>
                          </div>
                          <span style={{ opacity: 0.7 }}>
                            {formatPercent(
                              component.voxel_count /
                                Math.max(
                                  moistureStructureFrame.thresholdedVoxelCount,
                                  1
                                )
                            )}
                          </span>
                        </div>
                        <div style={{ opacity: 0.75, lineHeight: 1.35 }}>
                          {component.voxel_count.toLocaleString()} voxels ·{" "}
                          {component.pressure_min_hpa.toFixed(0)}-
                          {component.pressure_max_hpa.toFixed(0)} hPa
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            ) : null}
            {entry.id === "airMassLayer" && airMassComponents.length > 0 ? (
              <div
                style={{
                  display: "grid",
                  gap: 10,
                  marginTop: 14,
                  paddingTop: 12,
                  borderTop: "1px solid rgba(255,255,255,0.08)",
                }}
              >
                <AirMassAltitudeRangeControl
                  pressureWindow={airMassPressureWindow}
                  range={airMassLayer.altitudeRange01}
                  onChange={(altitudeRange01) =>
                    setAirMassLayer({ altitudeRange01 })
                  }
                />

                <div
                  style={{
                    display: "grid",
                    gap: 10,
                    paddingTop: 10,
                    borderTop: "1px solid rgba(255,255,255,0.08)",
                  }}
                >
                  <label
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 10,
                    }}
                  >
                    <span style={{ fontWeight: 700 }}>Camera Cutaway</span>
                    <input
                      type="checkbox"
                      checked={airMassLayer.cameraCutawayEnabled}
                      onChange={(event) =>
                        setAirMassLayer({
                          cameraCutawayEnabled: event.currentTarget.checked,
                        })
                      }
                      style={{ accentColor: "#8fe7c7" }}
                    />
                  </label>
                  <CompactRangeControl
                    label="Cutaway Radius"
                    value={airMassLayer.cameraCutawayRadius}
                    valueLabel={airMassLayer.cameraCutawayRadius.toFixed(0)}
                    min={8}
                    max={120}
                    step={1}
                    disabled={!airMassLayer.cameraCutawayEnabled}
                    onChange={(cameraCutawayRadius) =>
                      setAirMassLayer({ cameraCutawayRadius })
                    }
                  />
                </div>

                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 8,
                  }}
                >
                  <div
                    style={{
                      fontSize: 11,
                      fontWeight: 700,
                      letterSpacing: "0.08em",
                      textTransform: "uppercase",
                      opacity: 0.7,
                    }}
                  >
                    Components
                  </div>
                  <div style={{ opacity: 0.62 }}>
                    {visibleAirMassComponentCount}/{airMassComponents.length} visible
                  </div>
                </div>

                <div style={{ display: "flex", gap: 8 }}>
                  <button
                    type="button"
                    onClick={showAllAirMassComponents}
                    style={actionButtonStyle()}
                  >
                    Show All
                  </button>
                  <button
                    type="button"
                    onClick={hideAllAirMassComponents}
                    style={{
                      ...actionButtonStyle(),
                      background: "rgba(255,255,255,0.04)",
                    }}
                  >
                    Hide All
                  </button>
                </div>

                <div style={{ display: "grid", gap: 8 }}>
                  {airMassComponents.map((component, index) => {
                    const checked = !hiddenAirMassClassKeys.has(component.key);
                    return (
                      <label
                        key={component.key}
                        style={{
                          display: "grid",
                          gridTemplateColumns: "minmax(0, 1fr) auto",
                          gap: 10,
                          alignItems: "center",
                          padding: 10,
                          borderRadius: 12,
                          border: checked
                            ? "1px solid rgba(152, 200, 255, 0.28)"
                            : "1px solid rgba(255,255,255,0.08)",
                          background: checked
                            ? "rgba(255,255,255,0.05)"
                            : "rgba(255,255,255,0.025)",
                          cursor: "pointer",
                        }}
                      >
                        <span style={{ display: "grid", gap: 6, minWidth: 0 }}>
                          <span
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 8,
                              minWidth: 0,
                            }}
                          >
                            <span
                              aria-hidden
                              style={{
                                width: 10,
                                height: 10,
                                borderRadius: 999,
                                background: component.color,
                                boxShadow: "0 0 0 1px rgba(255,255,255,0.18)",
                                flex: "0 0 auto",
                              }}
                            />
                            <span
                              style={{
                                fontWeight: 700,
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                whiteSpace: "nowrap",
                              }}
                            >
                              #{index + 1} · {component.label}
                            </span>
                          </span>
                          <span style={{ opacity: 0.72, lineHeight: 1.35 }}>
                            {component.componentCount.toLocaleString()} connected
                            components · {component.voxelCount.toLocaleString()} voxels
                          </span>
                        </span>
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={(event) =>
                            setAirMassClassVisible(
                              component.key,
                              event.currentTarget.checked
                            )
                          }
                          style={{ accentColor: component.color }}
                        />
                      </label>
                    );
                  })}
                </div>
              </div>
            ) : null}
          </section>
        ))
      )}
    </aside>
  );
}
