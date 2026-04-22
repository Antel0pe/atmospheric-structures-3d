"use client";

import { useMemo } from "react";
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
  type MoistureStructureLayerState,
} from "@/app/state/controlsStore";
import type {
  MoistureStructureComponentMetadata,
} from "../utils/ApiResponses";

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

function formatPercent(value: number) {
  return `${(value * 100).toFixed(value >= 0.1 ? 0 : 1)}%`;
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
        airMassLayer.variant === "theta-rh-latmean"
          ? "Theta / RH anomaly proxy"
          : airMassLayer.variant === "theta-q-latmean"
            ? "Theta / q anomaly proxy"
            : "Temperature / RH anomaly proxy";
      entries.push({
        id: "airMassLayer",
        tag,
        ...LAYER_INFO.airMassLayer,
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
          </section>
        ))
      )}
    </aside>
  );
}
