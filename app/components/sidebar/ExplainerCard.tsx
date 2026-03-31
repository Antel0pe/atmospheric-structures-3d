"use client";
import { useState } from "react";

export default function ExplainerCard() {
  const [open, setOpen] = useState(true);

  return (
    <section
      aria-label="explainer card"
      style={{
        margin: 8,
        padding: 12,
        borderRadius: 12,
        background: "rgba(255,255,255,0.06)",
        border: "1px solid rgba(255,255,255,0.08)",
        color: "#e9eef7",
        font: "500 12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto",
      }}
    >
      <button
        onClick={() => setOpen(!open)}
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
        <span>Pacific Northwest Floods (2021) — Atmospheric River Anomaly</span>
        <span style={{ opacity: 0.7 }}>{open ? "–" : "+"}</span>
      </button>

      {open && (
        <div style={{ display: "grid", gap: 10 }}>
          {/* What you're looking at */}
          <div style={rowStyle()}>
            <RiverIcon />
            <div>
              What this map shows
              <div style={muted()}>
                An atmospheric river anomaly: where moisture transport was unusually strong compared to typical conditions.
                Think of it like extra moisture being funneled along a corridor.
              </div>
            </div>
          </div>

          {/* Pineapple express */}
          <div style={rowStyle()}>
            <PineappleIcon />
            <div>
              “Pineapple Express”
              <div style={muted()}>
                A nickname for a setup where tropical moisture is carried from near Hawaii toward the Pacific coast, feeding a long,
                narrow plume of very wet air.
              </div>
            </div>
          </div>

        </div>
      )}
    </section>
  );
}

function rowStyle() {
  return {
    display: "grid",
    gridTemplateColumns: "auto 1fr",
    gap: 10,
    alignItems: "center",
  } as const;
}

function muted() {
  return { opacity: 0.7, fontWeight: 400, lineHeight: 1.35 } as const;
}

function iconStroke() {
  return {
    stroke: "rgba(255,255,255,0.28)",
    strokeWidth: 1.2,
  } as const;
}

function iconFill() {
  return {
    fill: "rgba(255,255,255,0.08)",
    stroke: "rgba(255,255,255,0.22)",
    strokeWidth: 1.2,
  } as const;
}

function RiverIcon() {
  return (
    <svg width="42" height="42" viewBox="0 0 42 42" aria-hidden>
      <circle cx="21" cy="21" r="18" fill="rgba(255,255,255,0.03)" />
      <path
        d="M10 28 C14 22, 18 30, 22 24 C26 18, 28 26, 32 20"
        fill="none"
        stroke="rgba(255,255,255,0.35)"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M10 22 C14 16, 18 24, 22 18 C26 12, 28 20, 32 14"
        fill="none"
        stroke="rgba(255,255,255,0.22)"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <circle cx="32" cy="20" r="1.6" fill="rgba(255,255,255,0.35)" />
    </svg>
  );
}

function PineappleIcon() {
  return (
    <svg width="42" height="42" viewBox="0 0 42 42" aria-hidden>
      <circle cx="21" cy="21" r="18" fill="rgba(255,255,255,0.03)" />
      {/* leaves */}
      <path d="M21 9 C18 10, 17 13, 21 14 C25 13, 24 10, 21 9 Z" {...iconFill()} />
      <path d="M16 11 C14 12, 14 15, 18 15 C20 14, 18 12, 16 11 Z" {...iconFill()} />
      <path d="M26 11 C28 12, 28 15, 24 15 C22 14, 24 12, 26 11 Z" {...iconFill()} />
      {/* body */}
      <ellipse cx="21" cy="24" rx="8.5" ry="10" {...iconFill()} />
      {/* simple cross-hatch */}
      <path d="M16 22 L26 30" {...iconStroke()} />
      <path d="M26 22 L16 30" {...iconStroke()} />
      <path d="M15 26 L27 26" {...iconStroke()} />
    </svg>
  );
}
