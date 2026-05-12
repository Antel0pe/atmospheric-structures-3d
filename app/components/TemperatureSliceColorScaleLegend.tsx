"use client";

import { useEffect, useMemo, useState } from "react";
import { type TemperatureSliceVariant, useControls } from "../state/controlsStore";
import {
  fetchTemperatureSliceFrame,
  type TemperatureSliceFrame,
} from "./utils/temperatureSliceAssets";

type TemperatureRange = {
  min: number;
  max: number;
};

type LegendProps = {
  timestamp: string;
  sidebarOpen: boolean;
  sidebarWidth: string;
  layerInfoOpen: boolean;
  layerInfoWidth: string;
};

type LoadedFrame = {
  frame: TemperatureSliceFrame;
  timestamp: string;
  pressureHpa: number;
  variant: TemperatureSliceVariant;
};

const CONTINUOUS_STOPS = [
  { position: 0, color: "rgb(14, 42, 199)" },
  { position: 0.25, color: "rgb(23, 185, 240)" },
  { position: 0.5, color: "rgb(246, 247, 240)" },
  { position: 0.75, color: "rgb(255, 156, 31)" },
  { position: 1, color: "rgb(224, 9, 5)" },
];

const DISCRETE_COLORS = [
  "rgb(11, 33, 153)",
  "rgb(14, 68, 209)",
  "rgb(105, 181, 245)",
  "rgb(246, 247, 240)",
  "rgb(251, 159, 144)",
  "rgb(227, 54, 40)",
  "rgb(158, 5, 6)",
];

const INFERNO_STOPS = [
  { position: 0, color: "rgb(0, 0, 4)" },
  { position: 1 / 9, color: "rgb(27, 12, 65)" },
  { position: 2 / 9, color: "rgb(74, 12, 107)" },
  { position: 3 / 9, color: "rgb(120, 28, 109)" },
  { position: 4 / 9, color: "rgb(165, 44, 96)" },
  { position: 5 / 9, color: "rgb(207, 68, 70)" },
  { position: 6 / 9, color: "rgb(237, 105, 37)" },
  { position: 7 / 9, color: "rgb(251, 155, 6)" },
  { position: 8 / 9, color: "rgb(247, 209, 61)" },
  { position: 1, color: "rgb(252, 255, 164)" },
];

function parseCssPx(value: string) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function getViewportSize() {
  if (typeof window === "undefined") {
    return { width: 1600, height: 900 };
  }
  return { width: window.innerWidth, height: window.innerHeight };
}

function pressurePairRange(frame: TemperatureSliceFrame): TemperatureRange {
  const { lower, upper, mix } = frame.pressurePair;
  return {
    min:
      lower.temperature_min_k +
      (upper.temperature_min_k - lower.temperature_min_k) * mix,
    max:
      lower.temperature_max_k +
      (upper.temperature_max_k - lower.temperature_max_k) * mix,
  };
}

function isAbsoluteTemperatureFrame(frame: TemperatureSliceFrame) {
  return (
    frame.manifest.field_kind === undefined ||
    frame.manifest.field_kind === "raw-temperature" ||
    frame.manifest.field_kind === "raw-temperature-front-overlay" ||
    frame.manifest.field_kind === "raw-temperature-vertical-coherence" ||
    frame.manifest.field_kind === "raw-temperature-anomaly-strength" ||
    frame.manifest.field_kind === "raw-temperature-anomaly-agreement"
  );
}

function isThermalDisplacementFrame(frame: TemperatureSliceFrame) {
  return (
    frame.manifest.field_kind === "thermal-displacement-latitude" ||
    frame.manifest.field_kind === "thermal-displacement-latitude-smoothed" ||
    frame.manifest.field_kind === "thermal-displacement-zonal-mean-latitude" ||
    frame.manifest.field_kind ===
      "thermal-displacement-zonal-trimmed-mean-latitude"
  );
}

function isThermalConflictFrame(frame: TemperatureSliceFrame) {
  return frame.manifest.field_kind === "thermal-conflict-neighborhood";
}

function displayValue(value: number, frame: TemperatureSliceFrame) {
  if (isThermalDisplacementFrame(frame)) {
    return 90 - value * 90;
  }
  return isAbsoluteTemperatureFrame(frame) ? value - 273.15 : value;
}

function formatLegendValue(value: number, frame: TemperatureSliceFrame) {
  const display = displayValue(value, frame);
  if (isThermalDisplacementFrame(frame)) {
    return `${display.toFixed(0)}°`;
  }
  const abs = Math.abs(display);
  if (abs >= 100) return display.toFixed(0);
  if (abs >= 10) return display.toFixed(1);
  return display.toFixed(2);
}

function valueAt(range: TemperatureRange, position: number) {
  return range.min + (range.max - range.min) * position;
}

function shiftedStopPosition(position: number, midpoint01: number) {
  const midpoint = Math.min(Math.max(midpoint01, 0), 1);
  if (position <= 0.5) {
    return Math.min(Math.max(position * 2 * midpoint, 0), 1);
  }
  return Math.min(
    Math.max(midpoint + (position - 0.5) * 2 * (1 - midpoint), 0),
    1
  );
}

function continuousGradientCss(midpoint01: number) {
  return `linear-gradient(90deg, ${CONTINUOUS_STOPS.map(
    (stop) =>
      `${stop.color} ${shiftedStopPosition(stop.position, midpoint01) * 100}%`
  ).join(", ")})`;
}

function infernoGradientCss(midpoint01: number) {
  return `linear-gradient(90deg, ${INFERNO_STOPS.map(
    (stop) =>
      `${stop.color} ${shiftedStopPosition(stop.position, midpoint01) * 100}%`
  ).join(", ")})`;
}

function thermalConflictIdentityGradientCss() {
  return "linear-gradient(90deg, rgb(12, 48, 224) 0%, rgb(73, 71, 74) 50%, rgb(227, 27, 11) 100%)";
}

function thermalConflictSignalGradientCss() {
  return "linear-gradient(90deg, rgba(73, 71, 74, 0.35) 0%, rgb(255, 219, 20) 62%, rgb(255, 250, 217) 100%)";
}

function discreteGradientCss(midpoint01: number) {
  return `linear-gradient(90deg, ${DISCRETE_COLORS.map((color, index) => {
    const start =
      shiftedStopPosition(index / DISCRETE_COLORS.length, midpoint01) * 100;
    const end =
      shiftedStopPosition((index + 1) / DISCRETE_COLORS.length, midpoint01) *
      100;
    return `${color} ${start}% ${end}%`;
  }).join(", ")})`;
}

export default function TemperatureSliceColorScaleLegend({
  timestamp,
  sidebarOpen,
  sidebarWidth,
  layerInfoOpen,
  layerInfoWidth,
}: LegendProps) {
  const layer = useControls((state) => state.temperatureSliceLayer);
  const [loadedFrame, setLoadedFrame] = useState<LoadedFrame | null>(null);
  const [viewportSize, setViewportSize] = useState(getViewportSize);

  useEffect(() => {
    const onResize = () => setViewportSize(getViewportSize());
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    let cancelled = false;

    if (!layer.visible) {
      return () => {
        cancelled = true;
      };
    }

    void fetchTemperatureSliceFrame(timestamp, layer.pressureHpa, {
      variant: layer.variant,
      notifyOnError: false,
    })
      .then((nextFrame) => {
        if (!cancelled) {
          setLoadedFrame({
            frame: nextFrame,
            timestamp,
            pressureHpa: layer.pressureHpa,
            variant: layer.variant,
          });
        }
      })
      .catch((error) => {
        if (!cancelled) {
          console.error("Failed to load temperature slice legend", error);
          setLoadedFrame(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [layer.pressureHpa, layer.variant, layer.visible, timestamp]);

  const legend = useMemo(() => {
    const frame = loadedFrame?.frame;
    if (!frame || !layer.visible) return null;
    if (
      loadedFrame.timestamp !== timestamp ||
      loadedFrame.pressureHpa !== layer.pressureHpa ||
      loadedFrame.variant !== layer.variant
    ) {
      return null;
    }

    if (isThermalConflictFrame(frame)) {
      return {
        thermalConflict: true,
        discrete: false,
        inferno: false,
        ticks: [],
      };
    }

    const range =
      layer.colorScaleMode === "global"
        ? frame.manifest.temperature_range_k
        : pressurePairRange(frame);
    const discrete = layer.colorScaleMode === "perLevelDiscrete";
    const inferno = layer.colorScaleMode === "perLevelInferno";
    const ticks = discrete
      ? Array.from({ length: DISCRETE_COLORS.length + 1 }, (_, index) => ({
          key: `boundary-${index}`,
          position: shiftedStopPosition(
            index / DISCRETE_COLORS.length,
            layer.colorMidpoint01
          ),
          label: formatLegendValue(
            valueAt(
              range,
              shiftedStopPosition(
                index / DISCRETE_COLORS.length,
                layer.colorMidpoint01
              )
            ),
            frame
          ),
        }))
      : CONTINUOUS_STOPS.map((stop) => ({
          key: `stop-${stop.position}`,
          position: shiftedStopPosition(stop.position, layer.colorMidpoint01),
          label: formatLegendValue(
            valueAt(range, shiftedStopPosition(stop.position, layer.colorMidpoint01)),
            frame
          ),
        }));

    return {
      discrete,
      inferno,
      ticks,
    };
  }, [
    layer.colorScaleMode,
    layer.colorMidpoint01,
    layer.pressureHpa,
    layer.variant,
    layer.visible,
    loadedFrame,
    timestamp,
  ]);

  if (!legend) return null;

  const sidebarPx = sidebarOpen ? parseCssPx(sidebarWidth) : 0;
  const layerInfoPx = layerInfoOpen ? parseCssPx(layerInfoWidth) : 0;
  const availableWidth = viewportSize.width - sidebarPx - layerInfoPx - 54;
  if (availableWidth < 300 || viewportSize.height < 320) {
    return null;
  }
  const legendWidth = Math.min(372, availableWidth);

  return (
    <section
      aria-label="Temperature slice color scale"
      style={{
        position: "absolute",
        left: sidebarOpen ? `calc(${sidebarWidth} + 18px)` : 18,
        bottom: 156,
        zIndex: 76,
        width: legendWidth,
        pointerEvents: "none",
        padding: "10px 11px 9px",
        borderRadius: 8,
        border: "1px solid rgba(148, 163, 184, 0.22)",
        background: "rgba(4, 9, 16, 0.82)",
        color: "var(--atm-text)",
        boxShadow: "0 16px 38px rgba(0, 0, 0, 0.36)",
        backdropFilter: "blur(12px)",
        transition: "left 220ms ease, max-width 220ms ease",
      }}
    >
      <div
        style={{
          marginBottom: 7,
          font: "700 10px var(--font-sans)",
          letterSpacing: "0.02em",
          textTransform: "uppercase",
        }}
      >
        Scale
      </div>

      {legend.thermalConflict ? (
        <div style={{ display: "grid", gap: 9 }}>
          <div style={{ display: "grid", gap: 4 }}>
            <div
              style={{
                height: 16,
                borderRadius: 4,
                border: "1px solid rgba(255, 255, 255, 0.24)",
                background: thermalConflictIdentityGradientCss(),
                boxShadow: "inset 0 0 0 1px rgba(0, 0, 0, 0.18)",
              }}
            />
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                color: "#f3f7fd",
                font: "750 9px var(--font-mono)",
                textShadow: "0 1px 3px rgba(0, 0, 0, 0.95)",
              }}
            >
              <span>Cold-side</span>
              <span>Muted</span>
              <span>Warm-side</span>
            </div>
          </div>
          <div style={{ display: "grid", gap: 4 }}>
            <div
              style={{
                height: 10,
                borderRadius: 4,
                border: "1px solid rgba(255, 255, 255, 0.18)",
                background: thermalConflictSignalGradientCss(),
                boxShadow: "inset 0 0 0 1px rgba(0, 0, 0, 0.18)",
              }}
            />
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                color: "#f3f7fd",
                font: "750 9px var(--font-mono)",
                textShadow: "0 1px 3px rgba(0, 0, 0, 0.95)",
              }}
            >
              <span>Low conflict</span>
              <span>High conflict</span>
            </div>
          </div>
        </div>
      ) : (
      <div style={{ position: "relative", height: 38 }}>
        <div
          style={{
            position: "absolute",
            inset: "0 0 auto 0",
            height: 16,
            borderRadius: 4,
            border: "1px solid rgba(255, 255, 255, 0.24)",
            background: legend.discrete
              ? discreteGradientCss(layer.colorMidpoint01)
              : legend.inferno
                ? infernoGradientCss(layer.colorMidpoint01)
              : continuousGradientCss(layer.colorMidpoint01),
            boxShadow: "inset 0 0 0 1px rgba(0, 0, 0, 0.18)",
          }}
        />
        {legend.ticks.map((tick) => (
          <div
            key={tick.key}
            style={{
              position: "absolute",
              left: `${tick.position * 100}%`,
              top: 0,
              transform:
                tick.position === 0
                  ? "translateX(0)"
                  : tick.position === 1
                    ? "translateX(-100%)"
                    : "translateX(-50%)",
              display: "grid",
              justifyItems:
                tick.position === 0
                  ? "start"
                  : tick.position === 1
                    ? "end"
                    : "center",
              gap: 3,
              color: "#f3f7fd",
              font: "750 9px var(--font-mono)",
              whiteSpace: "nowrap",
              textShadow: "0 1px 3px rgba(0, 0, 0, 0.95)",
            }}
          >
            <span
              aria-hidden
              style={{
                width: 1,
                height: 19,
                background: "rgba(255, 255, 255, 0.86)",
                boxShadow: "0 0 0 1px rgba(0, 0, 0, 0.32)",
              }}
            />
            <span>{tick.label}</span>
          </div>
        ))}
      </div>
      )}
    </section>
  );
}
