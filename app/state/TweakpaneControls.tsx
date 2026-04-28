"use client";

import { useState } from "react";
import {
  AIR_MASS_CLASSIFICATION_VARIANT_OPTIONS,
  POTENTIAL_TEMPERATURE_COLOR_MODE_OPTIONS,
  POTENTIAL_TEMPERATURE_VARIANT_OPTIONS,
  useControls,
  type AirMassClassificationVariant,
  type PotentialTemperatureColorMode,
  type PotentialTemperatureVariant,
} from "./controlsStore";

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

function SliderField({
  label,
  value,
  valueLabel,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  valueLabel?: string;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label style={{ display: "grid", gap: 8 }}>
      <div style={rowStyle()}>
        <span>{label}</span>
        <span>{valueLabel ?? value.toFixed(step < 1 ? 2 : 0)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
        style={{ width: "100%", accentColor: "#86b7ff" }}
      />
    </label>
  );
}

function CheckboxField({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label style={rowStyle()}>
      <span>{label}</span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.currentTarget.checked)}
        style={{ accentColor: "#86b7ff" }}
      />
    </label>
  );
}

export default function TweakpaneControls() {
  const [open, setOpen] = useState(false);
  const verticalExaggeration = useControls((state) => state.verticalExaggeration);
  const precipitationRadarLayer = useControls(
    (state) => state.precipitationRadarLayer
  );
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
      <button
        onClick={() => setOpen((value) => !value)}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
          background: "transparent",
          border: "none",
          color: "inherit",
          cursor: "pointer",
          padding: 0,
          marginBottom: open ? 10 : 0,
          fontWeight: 700,
          letterSpacing: ".02em",
          textTransform: "uppercase",
          opacity: 0.9,
        }}
      >
        <span>Tweakpane Controls</span>
        <span style={{ opacity: 0.7 }}>{open ? "-" : "+"}</span>
      </button>

      {open ? (
        <div style={{ display: "grid", gap: 12 }}>
          <SliderField
            label="Vertical exaggeration"
            value={verticalExaggeration}
            min={1}
            max={8}
            step={0.05}
            onChange={setVerticalExaggeration}
          />

          <CheckboxField
            label="Precipitation radar"
            checked={precipitationRadarLayer.visible}
            onChange={(visible) => setPrecipitationRadarLayer({ visible })}
          />
          <SliderField
            label="Radar opacity"
            value={precipitationRadarLayer.opacity}
            min={0}
            max={1}
            step={0.01}
            onChange={(opacity) => setPrecipitationRadarLayer({ opacity })}
          />

          <CheckboxField
            label="Precipitable water proxy"
            checked={precipitableWaterLayer.visible}
            onChange={(visible) => setPrecipitableWaterLayer({ visible })}
          />

          <CheckboxField
            label="Potential temperature"
            checked={potentialTemperatureLayer.visible}
            onChange={(visible) => setPotentialTemperatureLayer({ visible })}
          />
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
          <select
            value={potentialTemperatureLayer.colorMode}
            onChange={(event) =>
              setPotentialTemperatureLayer({
                colorMode: event.currentTarget.value as PotentialTemperatureColorMode,
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

          <CheckboxField
            label="Air mass"
            checked={airMassLayer.visible}
            onChange={(visible) => setAirMassLayer({ visible })}
          />
          <select
            value={airMassLayer.variant}
            onChange={(event) =>
              setAirMassLayer({
                variant: event.currentTarget.value as AirMassClassificationVariant,
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
          <CheckboxField
            label="Air mass cell grid"
            checked={airMassLayer.showCellGrid}
            onChange={(showCellGrid) => setAirMassLayer({ showCellGrid })}
          />
        </div>
      ) : null}
    </section>
  );
}
