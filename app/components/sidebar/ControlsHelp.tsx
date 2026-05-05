"use client";

import { useState } from "react";

export default function ControlsHelp() {
  const [open, setOpen] = useState(true);

  return (
    <section className="atm-panel-section" aria-label="Shortcuts">
      <button
        type="button"
        className="atm-section-toggle"
        onClick={() => setOpen(!open)}
      >
        <span>Shortcuts</span>
        <span aria-hidden>{open ? "⌄" : "›"}</span>
      </button>

      {open ? (
        <div className="atm-shortcut-list">
          <ShortcutRow keys={["W", "A", "S", "D"]} label="Move / pan" />
          <ShortcutRow keys={["Shift", "Space"]} label="Altitude" />
          <ShortcutRow keys={["←", "→"]} label="Step time" />
          <ShortcutRow keys={["Mouse", "Q / Esc"]} label="Look around" />
        </div>
      ) : null}
    </section>
  );
}

function ShortcutRow({ keys, label }: { keys: string[]; label: string }) {
  return (
    <div className="atm-shortcut-row">
      <div className="atm-shortcut-keys">
        {keys.map((key) => (
          <span key={key}>{key}</span>
        ))}
      </div>
      <span>{label}</span>
    </div>
  );
}
