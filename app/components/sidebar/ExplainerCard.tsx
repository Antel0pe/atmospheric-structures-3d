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
        <span>Pacific Northwest Floods (2021) - Atmospheric River Setup</span>
        <span style={{ opacity: 0.7 }}>{open ? "-" : "+"}</span>
      </button>

      {open ? (
        <div style={{ display: "grid", gap: 10 }}>
          <div style={rowStyle()}>
            <RiverIcon />
            <div>
              What this map shows
              <div style={muted()}>
                A fast atmospheric-structure viewer for comparing layered
                fields around the November 2021 Pacific Northwest flood setup.
              </div>
            </div>
          </div>

          <div style={rowStyle()}>
            <MapIcon />
            <div>
              Globe and flat map
              <div style={muted()}>
                The same layer controls can drive either the 3D globe or the
                quick flat-map projection at /2d.
              </div>
            </div>
          </div>
        </div>
      ) : null}
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

function MapIcon() {
  return (
    <svg width="42" height="42" viewBox="0 0 42 42" aria-hidden>
      <circle cx="21" cy="21" r="18" fill="rgba(255,255,255,0.03)" />
      <path
        d="M10 14 L18 11 L25 14 L32 11 L32 28 L25 31 L18 28 L10 31 Z"
        fill="rgba(255,255,255,0.08)"
        stroke="rgba(255,255,255,0.26)"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
      <path
        d="M18 11 L18 28 M25 14 L25 31"
        fill="none"
        stroke="rgba(255,255,255,0.22)"
        strokeWidth="1.2"
      />
    </svg>
  );
}
