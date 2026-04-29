"use client";

import { useState } from "react";
import {
  useControls,
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

export default function TweakpaneControls() {
  const [open, setOpen] = useState(false);
  const precipitationRadarLayer = useControls(
    (state) => state.precipitationRadarLayer
  );
  const setPrecipitationRadarLayer = useControls(
    (state) => state.setPrecipitationRadarLayer
  );

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
            label="Radar opacity"
            value={precipitationRadarLayer.opacity}
            min={0}
            max={1}
            step={0.01}
            onChange={(opacity) => setPrecipitationRadarLayer({ opacity })}
          />
        </div>
      ) : null}
    </section>
  );
}
