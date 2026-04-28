"use client";

import { useSyncExternalStore } from "react";
import {
  AIR_MASS_CLASSIFICATION_VARIANT_OPTIONS,
  POTENTIAL_TEMPERATURE_COLOR_MODE_OPTIONS,
  POTENTIAL_TEMPERATURE_VARIANT_OPTIONS,
  type AirMassClassificationVariant,
  type PotentialTemperatureColorMode,
  type PotentialTemperatureVariant,
  useControls,
} from "../../state/controlsStore";

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

function rowStyle() {
  return {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  } as const;
}

function selectStyle() {
  return {
    width: "100%",
    borderRadius: 10,
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(6, 10, 18, 0.88)",
    color: "#e9eef7",
    padding: "10px 12px",
    outline: "none",
  } as const;
}

function CheckboxRow({
  label,
  checked,
  accentColor,
  onChange,
}: {
  label: string;
  checked: boolean;
  accentColor: string;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label style={rowStyle()}>
      <span style={{ fontWeight: 700 }}>{label}</span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.currentTarget.checked)}
        style={{ accentColor }}
      />
    </label>
  );
}

function subscribeToHydrationStore() {
  return () => {};
}

function getClientHydrationSnapshot() {
  return true;
}

function getServerHydrationSnapshot() {
  return false;
}

export default function LayerVisibilityPane() {
  const mounted = useSyncExternalStore(
    subscribeToHydrationStore,
    getClientHydrationSnapshot,
    getServerHydrationSnapshot
  );
  const verticalExaggeration = useControls((state) => state.verticalExaggeration);
  const precipitationLayer = useControls((state) => state.precipitationRadarLayer);
  const precipitableWaterLayer = useControls(
    (state) => state.precipitableWaterLayer
  );
  const potentialTemperatureLayer = useControls(
    (state) => state.potentialTemperatureLayer
  );
  const airMassLayer = useControls((state) => state.airMassLayer);
  const setVerticalExaggeration = useControls(
    (state) => state.setVerticalExaggeration
  );
  const setPrecipitationRadarLayer = useControls(
    (state) => state.setPrecipitationRadarLayer
  );
  const setPrecipitableWaterLayer = useControls(
    (state) => state.setPrecipitableWaterLayer
  );
  const setPotentialTemperatureLayer = useControls(
    (state) => state.setPotentialTemperatureLayer
  );
  const setAirMassLayer = useControls((state) => state.setAirMassLayer);

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
        Quick Layers
      </div>

      <div style={{ display: "grid", gap: 12 }}>
        <label style={{ display: "grid", gap: 8 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              gap: 8,
              fontWeight: 600,
            }}
          >
            <span>Vertical Exaggeration</span>
            <span>{verticalExaggeration.toFixed(1)}x</span>
          </div>
          <input
            type="range"
            min={1}
            max={8}
            step={0.5}
            value={verticalExaggeration}
            onChange={(event) =>
              setVerticalExaggeration(Number(event.currentTarget.value))
            }
            style={{ width: "100%", accentColor: "#9ebcff" }}
          />
        </label>

        <CheckboxRow
          label="Precipitable Water Proxy"
          checked={precipitableWaterLayer.visible}
          accentColor="#f3de6f"
          onChange={(checked) => setPrecipitableWaterLayer({ visible: checked })}
        />

        <CheckboxRow
          label="Potential Temperature"
          checked={potentialTemperatureLayer.visible}
          accentColor="#ff9b7d"
          onChange={(checked) => setPotentialTemperatureLayer({ visible: checked })}
        />

        <div
          style={{
            display: potentialTemperatureLayer.visible ? "grid" : "none",
            gap: 12,
          }}
        >
          <label style={{ display: "grid", gap: 8 }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                gap: 8,
                fontWeight: 600,
              }}
            >
              <span>Structure Recipe</span>
              <span style={{ opacity: 0.68 }}>
                {potentialTemperatureLayer.variant ===
                "raw-temperature-midpoint-cold-side"
                  ? "Raw temperature midpoint"
                  : "Climatology anomaly"}
              </span>
            </div>
            <select
              value={potentialTemperatureLayer.variant}
              onChange={(event) =>
                setPotentialTemperatureLayer({
                  variant: event.currentTarget.value as PotentialTemperatureVariant,
                })
              }
              style={selectStyle()}
            >
              {POTENTIAL_TEMPERATURE_VARIANT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <label style={{ display: "grid", gap: 8 }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                gap: 8,
                fontWeight: 600,
              }}
            >
              <span>Color Mode</span>
              <span style={{ opacity: 0.68 }}>Warm/cold shell</span>
            </div>
            <select
              value={potentialTemperatureLayer.colorMode}
              onChange={(event) =>
                setPotentialTemperatureLayer({
                  colorMode:
                    event.currentTarget.value as PotentialTemperatureColorMode,
                })
              }
              style={selectStyle()}
            >
              {POTENTIAL_TEMPERATURE_COLOR_MODE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <CheckboxRow
            label="Show Cell Grid"
            checked={potentialTemperatureLayer.showCellGrid}
            accentColor="#cfe0ff"
            onChange={(checked) =>
              setPotentialTemperatureLayer({ showCellGrid: checked })
            }
          />
        </div>

        <CheckboxRow
          label="Air Mass Classification"
          checked={airMassLayer.visible}
          accentColor="#8fe7c7"
          onChange={(checked) => setAirMassLayer({ visible: checked })}
        />

        <div
          style={{
            display: airMassLayer.visible ? "grid" : "none",
            gap: 12,
          }}
        >
          <label style={{ display: "grid", gap: 8 }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                gap: 8,
                fontWeight: 600,
              }}
            >
              <span>Proxy Recipe</span>
              <span style={{ opacity: 0.68 }}>Warm/cold + moist/dry</span>
            </div>
            <select
              value={airMassLayer.variant}
              onChange={(event) =>
                setAirMassLayer({
                  variant:
                    event.currentTarget.value as AirMassClassificationVariant,
                  hiddenClassKeys: [],
                })
              }
              style={selectStyle()}
            >
              {AIR_MASS_CLASSIFICATION_VARIANT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          {mounted ? (
            <CheckboxRow
              label="Show Cell Grid"
              checked={airMassLayer.showCellGrid}
              accentColor="#cfe0ff"
              onChange={(checked) => setAirMassLayer({ showCellGrid: checked })}
            />
          ) : null}
        </div>

        <CheckboxRow
          label="Precipitation Radar"
          checked={precipitationLayer.visible}
          accentColor="#8dff75"
          onChange={(checked) => setPrecipitationRadarLayer({ visible: checked })}
        />
      </div>
    </section>
  );
}
