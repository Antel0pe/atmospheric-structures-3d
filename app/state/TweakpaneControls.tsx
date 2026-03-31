"use client";

import { useState } from "react";
import { useControls } from "./controlsStore";

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

export default function TweakpaneControls() {
  const [open, setOpen] = useState(true);
  const moistureLayer = useControls((state) => state.moistureStructureLayer);
  const setMoistureLayer = useControls((state) => state.setMoistureStructureLayer);

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
        <span>Moisture Structures</span>
        <span style={{ opacity: 0.7 }}>{open ? "–" : "+"}</span>
      </button>

      {open ? (
        <div style={{ display: "grid", gap: 12 }}>
          <label
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 12,
            }}
          >
            <span style={{ fontWeight: 600 }}>Visible</span>
            <input
              type="checkbox"
              checked={moistureLayer.visible}
              onChange={(event) =>
                setMoistureLayer({ visible: event.currentTarget.checked })
              }
              style={{ accentColor: "#ff8a9a" }}
            />
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
              <span>Opacity</span>
              <span>{Math.round(moistureLayer.opacity * 100)}%</span>
            </div>
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={moistureLayer.opacity}
              onChange={(event) =>
                setMoistureLayer({ opacity: Number(event.currentTarget.value) })
              }
              style={{ width: "100%", accentColor: "#ff8a9a" }}
            />
          </label>

          <div style={{ display: "grid", gap: 4 }}>
            <div
              style={{
                fontSize: 11,
                fontWeight: 700,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                opacity: 0.66,
              }}
            >
              Threshold Mode
            </div>
            <div style={{ lineHeight: 1.45, opacity: 0.85 }}>
              Pressure-relative p95 by pressure level, with one light closing pass
              and small components removed.
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
