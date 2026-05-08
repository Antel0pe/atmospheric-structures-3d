"use client";

import { useMemo, type KeyboardEvent, type PointerEvent } from "react";
import {
  temperatureSliceColorScaleLabel,
  type TemperatureSliceVariant,
  temperatureSliceVariantLabel,
  useControls,
} from "@/app/state/controlsStore";

type ActiveLayerInfo = {
  title: string;
  badge?: string;
  tag: string;
  description: string;
};

const PRESSURE_LEVELS = [250, 500, 700, 850, 1000];

function panelSectionStyle() {
  return {
    padding: "15px 18px",
    borderBottom: "1px solid rgba(148, 163, 184, 0.13)",
    color: "var(--atm-text)",
    font: "500 11px var(--font-sans)",
  } as const;
}

function titleStyle() {
  return {
    marginBottom: 12,
    color: "var(--atm-text)",
    fontSize: 11,
    fontWeight: 850,
    letterSpacing: "0.04em",
    textTransform: "uppercase" as const,
  };
}

function metaRow(label: string, value: string) {
  return (
    <div className="atm-meta-row" key={label}>
      <span>{label}</span>
      <span>{value}</span>
    </div>
  );
}

function temperatureSliceDescription(variant: TemperatureSliceVariant) {
  if (variant === "thermal-displacement-latitude-smoothed") {
    return "A full-map pressure slice colored by closest climatology latitude after smoothing the raw temperature map at a 20-cell scale.";
  }
  if (variant === "thermal-displacement-latitude") {
    return "A full-map pressure slice colored by the climatology latitude whose temperature is closest to each raw cell at the same longitude.";
  }
  if (variant === "raw-temperature-anomaly-agreement") {
    return "A full-map raw-temperature pressure slice. Climatology departure adjusts saturation by sign agreement: same-side anomalies get vivid, opposite-side anomalies get muted.";
  }
  if (variant === "raw-temperature-anomaly-strength") {
    return "A full-map raw-temperature pressure slice. Climatology-departure magnitude makes unusual cells more vivid without changing their raw-temperature hue.";
  }
  return "A full-map pressure slice of the selected temperature field. Cold values render blue and warm values render red.";
}

function temperatureSliceUnits(variant: TemperatureSliceVariant) {
  if (
    variant === "thermal-displacement-latitude" ||
    variant === "thermal-displacement-latitude-smoothed"
  ) {
    return "Matched latitude";
  }
  return "°C";
}

function TemperaturePressureControl({
  pressureHpa,
  onChange,
}: {
  pressureHpa: number;
  onChange: (pressureHpa: number) => void;
}) {
  const clampedPressure = Math.min(Math.max(Math.round(pressureHpa), 250), 1000);
  const pressurePercent = ((clampedPressure - 250) / (1000 - 250)) * 100;
  const updateFromClientY = (element: HTMLElement, clientY: number) => {
    const rect = element.getBoundingClientRect();
    const mix = Math.min(Math.max((clientY - rect.top) / rect.height, 0), 1);
    onChange(Math.round(250 + mix * (1000 - 250)));
  };
  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    updateFromClientY(event.currentTarget, event.clientY);
  };
  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
    updateFromClientY(event.currentTarget, event.clientY);
  };
  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 25 : 10;
    if (event.key === "ArrowUp") {
      onChange(Math.max(250, clampedPressure - step));
      event.preventDefault();
    } else if (event.key === "ArrowDown") {
      onChange(Math.min(1000, clampedPressure + step));
      event.preventDefault();
    } else if (event.key === "Home") {
      onChange(250);
      event.preventDefault();
    } else if (event.key === "End") {
      onChange(1000);
      event.preventDefault();
    }
  };

  return (
    <div className="atm-vertical-level">
      <div className="atm-level-column">
        <div>Pressure</div>
        {PRESSURE_LEVELS.map((level) => (
          <span
            key={level}
            data-active={Math.abs(clampedPressure - level) < 75}
          >
            {level} hPa
          </span>
        ))}
      </div>

      <div
        className="atm-pressure-track"
        role="slider"
        tabIndex={0}
        aria-label="Temperature slice pressure"
        aria-valuemin={250}
        aria-valuemax={1000}
        aria-valuenow={clampedPressure}
        aria-valuetext={`${clampedPressure} hPa`}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onKeyDown={handleKeyDown}
      >
        <div className="atm-pressure-track-line" aria-hidden />
        <div
          className="atm-pressure-thumb"
          aria-hidden
          style={{ top: `${pressurePercent}%` }}
        />
      </div>
    </div>
  );
}

export default function LayerInfoPane() {
  const precipitableWaterLayer = useControls(
    (state) => state.precipitableWaterLayer
  );
  const temperatureSliceLayer = useControls(
    (state) => state.temperatureSliceLayer
  );
  const potentialTemperatureLayer = useControls(
    (state) => state.potentialTemperatureLayer
  );
  const airMassLayer = useControls((state) => state.airMassLayer);
  const precipitationLayer = useControls((state) => state.precipitationRadarLayer);
  const setTemperatureSliceLayer = useControls(
    (state) => state.setTemperatureSliceLayer
  );

  const primaryLayer = useMemo<ActiveLayerInfo | null>(() => {
    if (temperatureSliceLayer.visible) {
      return {
        title: "Temperature Slice",
        badge: "Active",
        tag: `${temperatureSliceLayer.pressureHpa.toFixed(
          0
        )} hPa • ${temperatureSliceVariantLabel(
          temperatureSliceLayer.variant
        )} • ${temperatureSliceColorScaleLabel(
          temperatureSliceLayer.colorScaleMode
        )}`,
        description: temperatureSliceDescription(temperatureSliceLayer.variant),
      };
    }
    if (potentialTemperatureLayer.visible) {
      return {
        title: "Potential Temperature",
        badge: "Active",
        tag: `${potentialTemperatureLayer.variant} • ${potentialTemperatureLayer.colorMode}`,
        description:
          "Thermal structure shells derived from potential-temperature or raw-temperature recipes.",
      };
    }
    if (airMassLayer.visible) {
      return {
        title: "Air Mass Classification",
        badge: "Active",
        tag: airMassLayer.variant,
        description:
          "Proxy-classified thermodynamic shells for comparing warm, cold, moist, and dry bodies.",
      };
    }
    if (precipitableWaterLayer.visible) {
      return {
        title: "Precipitable Water Proxy",
        badge: "Active",
        tag: "500-1000 hPa",
        description:
          "A low-level proxy shell built from humidity, RH, and multi-level depth gates.",
      };
    }
    if (precipitationLayer.visible) {
      return {
        title: "Precipitation Radar",
        badge: "Active",
        tag: "Surface overlay",
        description: "Static radar texture overlay for precipitation context.",
      };
    }
    return null;
  }, [
    airMassLayer.variant,
    airMassLayer.visible,
    precipitableWaterLayer.visible,
    precipitationLayer.visible,
    potentialTemperatureLayer.colorMode,
    potentialTemperatureLayer.variant,
    potentialTemperatureLayer.visible,
    temperatureSliceLayer.colorScaleMode,
    temperatureSliceLayer.pressureHpa,
    temperatureSliceLayer.variant,
    temperatureSliceLayer.visible,
  ]);

  return (
    <aside className="atm-info-pane">
      <div className="atm-info-scroll">
        <section style={panelSectionStyle()}>
          <div style={titleStyle()}>Layer Info</div>

          {primaryLayer ? (
            <div className="atm-info-heading">
              <div>
                <strong>{primaryLayer.title}</strong>
                {primaryLayer.badge ? <span>{primaryLayer.badge}</span> : null}
              </div>
              <p>{primaryLayer.tag}</p>
            </div>
          ) : (
            <p className="atm-muted-copy">
              Enable a layer from the left sidebar to inspect its current recipe.
            </p>
          )}
        </section>

        {primaryLayer ? (
          <section style={panelSectionStyle()}>
            <div style={titleStyle()}>Description</div>
            <p className="atm-muted-copy">{primaryLayer.description}</p>
          </section>
        ) : null}

        {temperatureSliceLayer.visible ? (
          <>
            <section style={panelSectionStyle()}>
              <div style={titleStyle()}>Metadata</div>
              <div className="atm-meta-stack">
                {metaRow("Parameter", "Temperature")}
                {metaRow("Pressure Level", `${temperatureSliceLayer.pressureHpa.toFixed(0)} hPa`)}
                {metaRow("Data Variant", temperatureSliceVariantLabel(temperatureSliceLayer.variant))}
                {metaRow("Units", temperatureSliceUnits(temperatureSliceLayer.variant))}
                {metaRow("Source", "Reanalysis")}
              </div>
            </section>

            <section style={panelSectionStyle()}>
              <div style={titleStyle()}>Vertical Level</div>
              <TemperaturePressureControl
                pressureHpa={temperatureSliceLayer.pressureHpa}
                onChange={(pressureHpa) =>
                  setTemperatureSliceLayer({ pressureHpa })
                }
              />
            </section>

            <section style={panelSectionStyle()}>
              <a className="atm-view-3d-link" href="/3d">
                <span>◎</span>
                <span>View in 3D</span>
                <span aria-hidden>↗</span>
              </a>
            </section>
          </>
        ) : null}
      </div>
    </aside>
  );
}
