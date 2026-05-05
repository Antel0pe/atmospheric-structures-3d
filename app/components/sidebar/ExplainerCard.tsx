"use client";

import { useState } from "react";

export default function ExplainerCard() {
  const [open, setOpen] = useState(true);

  return (
    <section className="atm-panel-section" aria-label="About">
      <button
        type="button"
        className="atm-section-toggle"
        onClick={() => setOpen(!open)}
      >
        <span>About</span>
        <span aria-hidden>{open ? "⌄" : "›"}</span>
      </button>

      {open ? (
        <div className="atm-about-copy">
          <p>Explore dynamic patterns in the atmosphere across space and time.</p>
          <a href="https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5" target="_blank" rel="noreferrer">
            Learn more ↗
          </a>
        </div>
      ) : null}
    </section>
  );
}
