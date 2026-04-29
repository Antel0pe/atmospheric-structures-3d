"use client";

import { useEffect, useMemo, useState } from "react";
import {
  type AirMassClassificationVariant,
  useControls,
} from "@/app/state/controlsStore";
import {
  fetchAirMassStructureManifest,
  type AirMassStructureClassKey,
} from "../utils/airMassStructureAssets";

type ActiveLayerId =
  | "precipitableWaterLayer"
  | "potentialTemperatureLayer"
  | "airMassLayer"
  | "precipitationRadarLayer";

type LayerInfoEntry = {
  id: ActiveLayerId;
  title: string;
  tag: string;
  summary: string;
};

type AirMassComponentEntry = {
  key: AirMassStructureClassKey;
  label: string;
  color: string;
  voxelCount: number;
  componentCount: number;
};

type AirMassPressureWindow = {
  min: number;
  max: number;
};

const AIR_MASS_MIN_ALTITUDE_RANGE_SPAN = 0.01;

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

function airMassFallbackColor(classKey: AirMassStructureClassKey, index: number) {
  if (classKey.includes("warm")) return "#ff9b63";
  if (classKey.includes("cold")) return "#66b7ff";
  const fallback = ["#7fe7c7", "#ffb86b", "#8fb7ff", "#e58bff"];
  return fallback[index % fallback.length];
}

function normalizeAirMassAltitudeRange(range: { min: number; max: number }) {
  const min = Math.max(0, Math.min(1, range.min));
  const max = Math.max(0, Math.min(1, range.max));
  if (max >= min) return { min, max };
  return { min: max, max: min };
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
    lowerHeight +
      Math.min(Math.max(altitudeMix, 0), 1) * (upperHeight - lowerHeight)
  );
}

function formatAirMassAltitudeRangeLabel(
  pressureWindow: AirMassPressureWindow,
  range: { min: number; max: number }
) {
  const normalized = normalizeAirMassAltitudeRange(range);
  return `${pressureForAltitudeMix(pressureWindow, normalized.min).toFixed(
    0
  )}-${pressureForAltitudeMix(
    pressureWindow,
    normalized.max
  ).toFixed(0)} hPa`;
}

function variantLabel(variant: AirMassClassificationVariant) {
  if (variant.startsWith("theta-anomaly-")) return "Theta anomaly buckets";
  if (variant === "theta-q-latmean") return "Theta + q anomaly";
  if (variant === "theta-rh-latmean") return "Theta + RH anomaly";
  if (variant === "surface-attached-theta-rh-latmean") {
    return "Surface-attached theta + RH";
  }
  return "Temperature + RH anomaly";
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
    <label
      style={{
        display: "grid",
        gap: 7,
        opacity: disabled ? 0.48 : 1,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: 10,
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
  const minPercent = Math.round(normalizedRange.min * 100);
  const maxPercent = Math.round(normalizedRange.max * 100);
  const updateMin = (percent: number) => {
    const min = Math.min(
      percent / 100,
      Math.max(
        0,
        normalizedRange.max - AIR_MASS_MIN_ALTITUDE_RANGE_SPAN
      )
    );
    onChange({ min, max: normalizedRange.max });
  };
  const updateMax = (percent: number) => {
    const max = Math.max(
      percent / 100,
      Math.min(
        1,
        normalizedRange.min + AIR_MASS_MIN_ALTITUDE_RANGE_SPAN
      )
    );
    onChange({ min: normalizedRange.min, max });
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
          gap: 10,
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
            background: "linear-gradient(90deg, #5e86ff, #8fe7c7)",
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

export default function LayerInfoPane() {
  const precipitableWaterLayer = useControls(
    (state) => state.precipitableWaterLayer
  );
  const potentialTemperatureLayer = useControls(
    (state) => state.potentialTemperatureLayer
  );
  const airMassLayer = useControls((state) => state.airMassLayer);
  const precipitationLayer = useControls((state) => state.precipitationRadarLayer);
  const setAirMassLayer = useControls((state) => state.setAirMassLayer);
  const [airMassComponentState, setAirMassComponentState] = useState<{
    variant: AirMassClassificationVariant;
    entries: AirMassComponentEntry[];
    pressureWindow: AirMassPressureWindow;
  } | null>(null);

  useEffect(() => {
    let cancelled = false;
    void fetchAirMassStructureManifest({
      variant: airMassLayer.variant,
      notifyOnError: false,
    })
      .then((manifest) => {
        if (cancelled) return;
        const timestampSummary = manifest.timestamps[0];
        const entries = manifest.classification.classes.map((entry, index) => {
          const counts = timestampSummary?.class_counts?.[entry.key];
          return {
            key: entry.key,
            label: entry.label,
            color: entry.color ?? airMassFallbackColor(entry.key, index),
            voxelCount: counts?.voxel_count ?? 0,
            componentCount: counts?.component_count ?? 0,
          };
        });
        setAirMassComponentState({
          variant: manifest.variant,
          entries,
          pressureWindow: manifest.pressure_window_hpa,
        });
      })
      .catch(() => {
        if (!cancelled) setAirMassComponentState(null);
      });

    return () => {
      cancelled = true;
    };
  }, [airMassLayer.variant]);

  const activeLayers = useMemo(() => {
    const layers: LayerInfoEntry[] = [];

    if (precipitableWaterLayer.visible) {
      layers.push({
        id: "precipitableWaterLayer",
        title: "Precipitable Water Proxy",
        tag: "500-1000 hPa",
        summary:
          "A low-level proxy shell built from high specific humidity, high RH, and multi-level depth gates.",
      });
    }

    if (potentialTemperatureLayer.visible) {
      layers.push({
        id: "potentialTemperatureLayer",
        title: "Potential Temperature",
        tag: `${potentialTemperatureLayer.variant} / ${potentialTemperatureLayer.colorMode}`,
        summary:
          "Thermal structure shells derived from potential-temperature or raw-temperature recipes.",
      });
    }

    if (airMassLayer.visible) {
      const pressureWindow = airMassComponentState?.pressureWindow ?? {
        min: 250,
        max: 1000,
      };
      layers.push({
        id: "airMassLayer",
        title: airMassLayer.variant.startsWith("theta-anomaly-")
          ? "Theta Anomaly Bucket Layer"
          : "Air Mass Classification",
        tag: `${variantLabel(airMassLayer.variant)} / ${formatAirMassAltitudeRangeLabel(
          pressureWindow,
          airMassLayer.altitudeRange01
        )}`,
        summary:
          "Proxy-classified thermodynamic shells. The theta-anomaly variants show dry-theta tail buckets rather than source-region air masses.",
      });
    }

    if (precipitationLayer.visible) {
      layers.push({
        id: "precipitationRadarLayer",
        title: "Precipitation Radar",
        tag: "Surface overlay",
        summary: "Static radar texture overlay for precipitation context.",
      });
    }

    return layers;
  }, [
    airMassComponentState?.pressureWindow,
    airMassLayer.altitudeRange01,
    airMassLayer.variant,
    airMassLayer.visible,
    precipitableWaterLayer.visible,
    precipitationLayer.visible,
    potentialTemperatureLayer.colorMode,
    potentialTemperatureLayer.variant,
    potentialTemperatureLayer.visible,
  ]);

  const hiddenAirMassClassKeys = useMemo(
    () => new Set(airMassLayer.hiddenClassKeys),
    [airMassLayer.hiddenClassKeys]
  );
  const airMassComponents =
    airMassComponentState?.variant === airMassLayer.variant
      ? airMassComponentState.entries
      : [];
  const airMassPressureWindow =
    airMassComponentState?.variant === airMassLayer.variant
      ? airMassComponentState.pressureWindow
      : { min: 250, max: 1000 };

  const setAirMassClassVisible = (
    classKey: AirMassStructureClassKey,
    visible: boolean
  ) => {
    const nextHidden = new Set(airMassLayer.hiddenClassKeys);
    if (visible) {
      nextHidden.delete(classKey);
    } else {
      nextHidden.add(classKey);
    }
    setAirMassLayer({ hiddenClassKeys: Array.from(nextHidden) });
  };

  return (
    <section style={sectionStyle()}>
      <div
        style={{
          fontWeight: 800,
          letterSpacing: ".02em",
          textTransform: "uppercase",
          marginBottom: 12,
        }}
      >
        Layer Info
      </div>

      {activeLayers.length === 0 ? (
        <div style={{ lineHeight: 1.45, opacity: 0.72 }}>
          Enable a layer from the left sidebar to inspect its current recipe.
        </div>
      ) : (
        <div style={{ display: "grid", gap: 12 }}>
          {activeLayers.map((layer) => (
            <article
              key={layer.id}
              style={{
                display: "grid",
                gap: 8,
                paddingBottom: 12,
                borderBottom: "1px solid rgba(255,255,255,0.1)",
              }}
            >
              <div style={{ display: "grid", gap: 3 }}>
                <div style={{ fontWeight: 800 }}>{layer.title}</div>
                <div style={{ opacity: 0.65, fontSize: 11 }}>{layer.tag}</div>
              </div>
              <div style={{ opacity: 0.8, lineHeight: 1.45 }}>{layer.summary}</div>
            </article>
          ))}
        </div>
      )}

      {airMassLayer.visible && airMassComponents.length > 0 ? (
        <div style={{ display: "grid", gap: 10, marginTop: 14 }}>
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

          <div style={{ display: "flex", gap: 8 }}>
            <button
              type="button"
              style={actionButtonStyle()}
              onClick={() => setAirMassLayer({ hiddenClassKeys: [] })}
            >
              Show all
            </button>
            <button
              type="button"
              style={actionButtonStyle()}
              onClick={() =>
                setAirMassLayer({
                  hiddenClassKeys: airMassComponents.map((entry) => entry.key),
                })
              }
            >
              Hide all
            </button>
          </div>

          {airMassComponents.map((component) => {
            const checked = !hiddenAirMassClassKeys.has(component.key);
            return (
              <label
                key={component.key}
                style={{
                  display: "grid",
                  gridTemplateColumns: "auto 1fr auto",
                  gap: 8,
                  alignItems: "center",
                }}
              >
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
                <span>
                  <span style={{ fontWeight: 700 }}>{component.label}</span>
                  <span style={{ opacity: 0.65 }}>
                    {" "}
                    {component.componentCount} comps /{" "}
                    {component.voxelCount.toLocaleString()} cells
                  </span>
                </span>
                <span
                  aria-hidden
                  style={{
                    width: 16,
                    height: 16,
                    borderRadius: 4,
                    background: component.color,
                    boxShadow: "0 0 0 1px rgba(255,255,255,0.18) inset",
                  }}
                />
              </label>
            );
          })}
        </div>
      ) : null}
    </section>
  );
}
