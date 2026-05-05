"use client";

import { useSyncExternalStore, type ReactNode } from "react";
import {
  AIR_MASS_CLASSIFICATION_VARIANT_OPTIONS,
  POTENTIAL_TEMPERATURE_COLOR_MODE_OPTIONS,
  POTENTIAL_TEMPERATURE_VARIANT_OPTIONS,
  TEMPERATURE_SLICE_COLOR_SCALE_OPTIONS,
  TEMPERATURE_SLICE_VARIANT_OPTIONS,
  isSmoothAirMassClassificationVariant,
  temperatureSliceVariantLabel,
  type AirMassClassificationVariant,
  type PotentialTemperatureColorMode,
  type PotentialTemperatureVariant,
  type TemperatureSliceColorScaleMode,
  type TemperatureSliceVariant,
  useControls,
} from "../../state/controlsStore";

function subscribeToHydrationStore() {
  return () => {};
}

function getClientHydrationSnapshot() {
  return true;
}

function getServerHydrationSnapshot() {
  return false;
}

function panelSectionStyle() {
  return {
    padding: "14px 18px",
    borderTop: "1px solid rgba(148, 163, 184, 0.14)",
    color: "var(--atm-text)",
    font: "500 11px var(--font-sans)",
  } as const;
}

function sectionHeadingStyle() {
  return {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
    marginBottom: 12,
    color: "var(--atm-text)",
    fontSize: 11,
    fontWeight: 800,
    letterSpacing: "0.04em",
    textTransform: "uppercase" as const,
  };
}

function selectStyle() {
  return {
    width: "100%",
    minHeight: 30,
    borderRadius: 7,
    border: "1px solid rgba(148, 163, 184, 0.23)",
    background: "rgba(4, 9, 16, 0.88)",
    color: "var(--atm-text)",
    padding: "5px 8px",
    outline: "none",
    fontFamily: "var(--font-sans)",
    fontSize: 10,
    fontWeight: 600,
    lineHeight: 1.2,
  } as const;
}

function ToggleSwitch({
  checked,
  accentColor,
  onChange,
}: {
  checked: boolean;
  accentColor: string;
  onChange: (checked: boolean) => void;
}) {
  return (
    <span className="atm-switch" data-checked={checked}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.currentTarget.checked)}
        style={{ accentColor }}
      />
      <span aria-hidden style={{ background: checked ? accentColor : undefined }} />
    </span>
  );
}

function LayerRow({
  label,
  checked,
  onChange,
  children,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  children?: ReactNode;
}) {
  return (
    <div className="atm-layer-row" data-active={checked}>
      <label className="atm-layer-row-header">
        <span className="atm-layer-chevron" aria-hidden>⌄</span>
        <span
          className="atm-layer-dot"
          aria-hidden
          style={{ background: checked ? "var(--atm-green)" : "var(--atm-dot-off)" }}
        />
        <span>{label}</span>
        <ToggleSwitch checked={checked} accentColor="var(--atm-green)" onChange={onChange} />
      </label>
      {checked && children ? <div className="atm-layer-controls">{children}</div> : null}
    </div>
  );
}

function FieldLabel({
  label,
  value,
  children,
}: {
  label: string;
  value?: string;
  children: ReactNode;
}) {
  return (
    <label className="atm-field">
      <div>
        <span>{label}</span>
        {value ? <span>{value}</span> : null}
      </div>
      {children}
    </label>
  );
}

function SliderField({
  label,
  value,
  valueLabel,
  min,
  max,
  step,
  accent,
  onChange,
}: {
  label: string;
  value: number;
  valueLabel: string;
  min: number;
  max: number;
  step: number;
  accent: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="atm-field">
      <div>
        <span>{label}</span>
        <span>{valueLabel}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
        style={{ width: "100%", accentColor: accent }}
      />
    </label>
  );
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
  const temperatureSliceLayer = useControls(
    (state) => state.temperatureSliceLayer
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
  const setTemperatureSliceLayer = useControls(
    (state) => state.setTemperatureSliceLayer
  );
  const setPotentialTemperatureLayer = useControls(
    (state) => state.setPotentialTemperatureLayer
  );
  const setAirMassLayer = useControls((state) => state.setAirMassLayer);

  return (
    <>
      <section style={panelSectionStyle()}>
        <div style={sectionHeadingStyle()}>
          <span>Layers</span>
        </div>

        <div className="atm-layer-stack">
          <LayerRow
            label="Temperature Slice"
            checked={temperatureSliceLayer.visible}
            onChange={(checked) => setTemperatureSliceLayer({ visible: checked })}
          >
            <FieldLabel
              label="Data Variant"
              value={temperatureSliceVariantLabel(temperatureSliceLayer.variant)}
            >
              <select
                value={temperatureSliceLayer.variant}
                onChange={(event) =>
                  setTemperatureSliceLayer({
                    variant: event.currentTarget.value as TemperatureSliceVariant,
                  })
                }
                style={selectStyle()}
              >
                {TEMPERATURE_SLICE_VARIANT_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </FieldLabel>

            <FieldLabel label="Color Scale Mode" value="Air Pressure Levels">
              <select
                value={temperatureSliceLayer.colorScaleMode}
                onChange={(event) =>
                  setTemperatureSliceLayer({
                    colorScaleMode:
                      event.currentTarget.value as TemperatureSliceColorScaleMode,
                  })
                }
                style={selectStyle()}
              >
                {TEMPERATURE_SLICE_COLOR_SCALE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </FieldLabel>
          </LayerRow>

          <LayerRow
            label="Potential Temperature"
            checked={potentialTemperatureLayer.visible}
            onChange={(checked) => setPotentialTemperatureLayer({ visible: checked })}
          >
            <FieldLabel label="Structure Recipe">
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
            </FieldLabel>

            <FieldLabel label="Color Mode">
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
            </FieldLabel>

            <label className="atm-layer-row-header atm-sub-toggle">
              <span>Show Cell Grid</span>
              <ToggleSwitch
                checked={potentialTemperatureLayer.showCellGrid}
                accentColor="#cfe0ff"
                onChange={(checked) =>
                  setPotentialTemperatureLayer({ showCellGrid: checked })
                }
              />
            </label>
          </LayerRow>

          <LayerRow
            label="Precipitable Water"
            checked={precipitableWaterLayer.visible}
            onChange={(checked) => setPrecipitableWaterLayer({ visible: checked })}
          />

          <LayerRow
            label="Air Mass Classification"
            checked={airMassLayer.visible}
            onChange={(checked) => setAirMassLayer({ visible: checked })}
          >
            <FieldLabel label="Proxy Recipe">
              <select
                value={airMassLayer.variant}
                onChange={(event) => {
                  const variant = event.currentTarget
                    .value as AirMassClassificationVariant;
                  setAirMassLayer({
                    variant,
                    showCellGrid:
                      airMassLayer.showCellGrid ||
                      isSmoothAirMassClassificationVariant(variant),
                    hiddenClassKeys: [],
                  });
                }}
                style={selectStyle()}
              >
                {AIR_MASS_CLASSIFICATION_VARIANT_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </FieldLabel>

            {mounted ? (
              <label className="atm-layer-row-header atm-sub-toggle">
                <span>Show Cell Grid</span>
                <ToggleSwitch
                  checked={airMassLayer.showCellGrid}
                  accentColor="#cfe0ff"
                  onChange={(checked) => setAirMassLayer({ showCellGrid: checked })}
                />
              </label>
            ) : null}
          </LayerRow>

          <LayerRow
            label="Precipitation Radar"
            checked={precipitationLayer.visible}
            onChange={(checked) => setPrecipitationRadarLayer({ visible: checked })}
          />
        </div>
      </section>

      <section style={panelSectionStyle()}>
        <div style={sectionHeadingStyle()}>
          <span>General Controls</span>
        </div>
        <div className="atm-control-stack">
          <SliderField
            label="Opacity"
            value={temperatureSliceLayer.opacity}
            valueLabel={`${Math.round(temperatureSliceLayer.opacity * 100)}%`}
            min={0}
            max={1}
            step={0.01}
            accent="#4f8cff"
            onChange={(opacity) => setTemperatureSliceLayer({ opacity })}
          />
          <SliderField
            label="Vertical Exaggeration"
            value={verticalExaggeration}
            valueLabel={`${verticalExaggeration.toFixed(1)}x`}
            min={1}
            max={8}
            step={0.5}
            accent="#8fb6ff"
            onChange={setVerticalExaggeration}
          />
        </div>
      </section>
    </>
  );
}
