"use client";

import { useMemo } from "react";
import { useControls } from "@/app/state/controlsStore";

type ActiveExampleId =
  | "moistureStructureLayer"
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
      "A 3D humidity-body renderer that loads precomputed moisture meshes, colors them by vertical character, and crossfades between timestamps.",
    detail:
      "Humid regions are detected with pressure-relative thresholds, lightly smoothed, turned into closed 3D surfaces offline, and then rendered here as semi-transparent globe-space volumes.",
    legend: [
      {
        label: "Higher-altitude structures",
        detail: "More magenta-toned components with lower pressure centroids.",
        swatch:
          "linear-gradient(135deg, rgba(214, 77, 246, 0.96), rgba(170, 68, 255, 0.84))",
      },
      {
        label: "Lower-altitude structures",
        detail: "Warmer coral-toned components with higher pressure centroids.",
        swatch:
          "linear-gradient(135deg, rgba(255, 123, 99, 0.96), rgba(255, 157, 117, 0.88))",
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

export default function LayerInfoPane() {
  const moistureStructureLayer = useControls(
    (state) => state.moistureStructureLayer
  );
  const exampleShaderMeshLayer = useControls(
    (state) => state.exampleShaderMeshLayer
  );
  const exampleContoursLayer = useControls(
    (state) => state.exampleContoursLayer
  );
  const exampleParticleLayer = useControls(
    (state) => state.exampleParticleLayer
  );

  const activeEntries = useMemo(() => {
    const entries: Array<LayerInfoEntry & { tag: string }> = [];

    if (moistureStructureLayer.visible) {
      entries.push({
        id: "moistureStructureLayer",
        tag: `Opacity ${Math.round(moistureStructureLayer.opacity * 100)}%`,
        ...LAYER_INFO.moistureStructureLayer,
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
  }, [
    moistureStructureLayer.opacity,
    moistureStructureLayer.visible,
    exampleContoursLayer.pressureLevel,
    exampleParticleLayer.pressureLevel,
    exampleShaderMeshLayer.pressureLevel,
  ]);

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
          </section>
        ))
      )}
    </aside>
  );
}
