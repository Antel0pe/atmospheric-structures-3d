"use client";

import { useState } from "react";
import {
  MOISTURE_COLOR_MODE_OPTIONS,
  MOISTURE_COMPONENT_SORT_OPTIONS,
  MOISTURE_FOCUS_MODE_OPTIONS,
  MOISTURE_LEGIBILITY_EXPERIMENT_OPTIONS,
  MOISTURE_SEGMENTATION_MODE_OPTIONS,
  MOISTURE_STRUCTURE_PRESET_OPTIONS,
  MOISTURE_VISUAL_PRESET_OPTIONS,
  type MoistureColorMode,
  type MoistureComponentSort,
  type MoistureFocusMode,
  type MoistureLegibilityExperiment,
  type MoistureStructurePreset,
  type MoistureVisualPreset,
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

function infoTitleStyle() {
  return {
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    opacity: 0.66,
  } as const;
}

function checkboxRowStyle() {
  return {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  } as const;
}

function sliderLabelStyle() {
  return {
    display: "flex",
    justifyContent: "space-between",
    gap: 8,
    fontWeight: 600,
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

function actionButtonStyle() {
  return {
    width: "100%",
    borderRadius: 10,
    border: "1px solid rgba(110, 170, 255, 0.28)",
    background: "rgba(82, 146, 255, 0.16)",
    color: "#e9eef7",
    padding: "10px 12px",
    fontWeight: 700,
    cursor: "pointer",
  } as const;
}

type SliderFieldProps = {
  label: string;
  valueLabel: string;
  min: number;
  max: number;
  step: number;
  value: number;
  accentColor: string;
  disabled?: boolean;
  onChange: (value: number) => void;
};

function SliderField({
  label,
  valueLabel,
  min,
  max,
  step,
  value,
  accentColor,
  disabled,
  onChange,
}: SliderFieldProps) {
  return (
    <label
      style={{
        display: "grid",
        gap: 8,
        opacity: disabled ? 0.5 : 1,
      }}
    >
      <div style={sliderLabelStyle()}>
        <span>{label}</span>
        <span>{valueLabel}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
        style={{ width: "100%", accentColor }}
      />
    </label>
  );
}

type CheckboxFieldProps = {
  label: string;
  checked: boolean;
  accentColor: string;
  onChange: (checked: boolean) => void;
};

function CheckboxField({
  label,
  checked,
  accentColor,
  onChange,
}: CheckboxFieldProps) {
  return (
    <label style={checkboxRowStyle()}>
      <span style={{ fontWeight: 600 }}>{label}</span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.currentTarget.checked)}
        style={{ accentColor }}
      />
    </label>
  );
}

export default function TweakpaneControls() {
  const [open, setOpen] = useState(true);
  const moistureLayer = useControls((state) => state.moistureStructureLayer);
  const setMoistureLayer = useControls((state) => state.setMoistureStructureLayer);
  const setMoistureVisualPreset = useControls(
    (state) => state.setMoistureVisualPreset
  );
  const resetMoistureVisualPreset = useControls(
    (state) => state.resetMoistureVisualPreset
  );
  const setMoistureStructurePreset = useControls(
    (state) => state.setMoistureStructurePreset
  );
  const resetMoistureStructurePreset = useControls(
    (state) => state.resetMoistureStructurePreset
  );
  const setMoistureLegibilityExperiment = useControls(
    (state) => state.setMoistureLegibilityExperiment
  );
  const resetMoistureLegibilityExperiment = useControls(
    (state) => state.resetMoistureLegibilityExperiment
  );

  const backfaceControlsEnabled =
    !moistureLayer.solidShellEnabled || moistureLayer.interiorBackfaceEnabled;

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
          <CheckboxField
            label="Visible"
            checked={moistureLayer.visible}
            accentColor="#ff8a9a"
            onChange={(checked) => setMoistureLayer({ visible: checked })}
          />

          <SliderField
            label="Opacity"
            valueLabel={`${Math.round(moistureLayer.opacity * 100)}%`}
            min={0.1}
            max={1}
            step={0.05}
            value={moistureLayer.opacity}
            accentColor="#ff8a9a"
            onChange={(value) => setMoistureLayer({ opacity: value })}
          />

          <SliderField
            label="Vertical Exaggeration"
            valueLabel={`${moistureLayer.verticalExaggeration.toFixed(1)}x`}
            min={1}
            max={8}
            step={0.5}
            value={moistureLayer.verticalExaggeration}
            accentColor="#8ad4ff"
            onChange={(value) =>
              setMoistureLayer({ verticalExaggeration: value })
            }
          />

          <CheckboxField
            label="Camera Cutaway"
            checked={moistureLayer.cameraCutawayEnabled}
            accentColor="#8ad4ff"
            onChange={(checked) =>
              setMoistureLayer({ cameraCutawayEnabled: checked })
            }
          />

          <SliderField
            label="Cutaway Radius"
            valueLabel={moistureLayer.cameraCutawayRadius.toFixed(0)}
            min={4}
            max={40}
            step={1}
            value={moistureLayer.cameraCutawayRadius}
            accentColor="#8ad4ff"
            disabled={!moistureLayer.cameraCutawayEnabled}
            onChange={(value) =>
              setMoistureLayer({ cameraCutawayRadius: value })
            }
          />

          <div
            style={{
              display: "grid",
              gap: 10,
              marginTop: 2,
              paddingTop: 12,
              borderTop: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <div style={infoTitleStyle()}>Legibility Experiments</div>

            <label style={{ display: "grid", gap: 8 }}>
              <div style={sliderLabelStyle()}>
                <span>Experiment</span>
                <span style={{ opacity: 0.68 }}>Comparison mode</span>
              </div>
              <select
                value={moistureLayer.legibilityExperiment}
                onChange={(event) =>
                  setMoistureLegibilityExperiment(
                    event.currentTarget.value as MoistureLegibilityExperiment
                  )
                }
                style={selectStyle()}
              >
                {MOISTURE_LEGIBILITY_EXPERIMENT_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <button
              type="button"
              onClick={() => resetMoistureLegibilityExperiment()}
              style={actionButtonStyle()}
            >
              Reset To Default
            </button>

            <div style={{ lineHeight: 1.45, opacity: 0.78 }}>
              `Bridge Pruned` is the default comparison state. Segmentation now
              controls the data variant directly, so `None` and `Bridge Pruned`
              use the same selected dataset, `Bridge Pruned + Shell First`
              matches `Shell First`, and the matte variant only softens the
              lighting response.
            </div>
          </div>

          <div
            style={{
              display: "grid",
              gap: 10,
              marginTop: 2,
              paddingTop: 12,
              borderTop: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <div style={infoTitleStyle()}>Visual Experiments</div>

            <label style={{ display: "grid", gap: 8 }}>
              <div style={sliderLabelStyle()}>
                <span>Preset</span>
                <span style={{ opacity: 0.68 }}>Review mode</span>
              </div>
              <select
                value={moistureLayer.visualPreset}
                onChange={(event) =>
                  setMoistureVisualPreset(
                    event.currentTarget.value as MoistureVisualPreset
                  )
                }
                style={selectStyle()}
              >
                {MOISTURE_VISUAL_PRESET_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <button
              type="button"
              onClick={() => resetMoistureVisualPreset()}
              style={actionButtonStyle()}
            >
              Reset To Preset
            </button>

            <CheckboxField
              label="Solid Shell"
              checked={moistureLayer.solidShellEnabled}
              accentColor="#ffb266"
              onChange={(checked) =>
                setMoistureLayer({ solidShellEnabled: checked })
              }
            />

            <CheckboxField
              label="Lighting Pass"
              checked={moistureLayer.lightingEnabled}
              accentColor="#ffe68a"
              onChange={(checked) =>
                setMoistureLayer({ lightingEnabled: checked })
              }
            />

            <CheckboxField
              label="Interior Backfaces"
              checked={moistureLayer.interiorBackfaceEnabled}
              accentColor="#9fc1ff"
              onChange={(checked) =>
                setMoistureLayer({ interiorBackfaceEnabled: checked })
              }
            />

            <CheckboxField
              label="Rim Emphasis"
              checked={moistureLayer.rimEnabled}
              accentColor="#d89bff"
              onChange={(checked) => setMoistureLayer({ rimEnabled: checked })}
            />

            <CheckboxField
              label="Distance Cue"
              checked={moistureLayer.distanceFadeEnabled}
              accentColor="#8ffff2"
              onChange={(checked) =>
                setMoistureLayer({ distanceFadeEnabled: checked })
              }
            />

            <SliderField
              label="Front Opacity"
              valueLabel={`${moistureLayer.frontOpacity.toFixed(2)}x`}
              min={0.4}
              max={1.5}
              step={0.05}
              value={moistureLayer.frontOpacity}
              accentColor="#ffb266"
              onChange={(value) => setMoistureLayer({ frontOpacity: value })}
            />

            <SliderField
              label="Backface Opacity"
              valueLabel={`${moistureLayer.backfaceOpacity.toFixed(2)}x`}
              min={0}
              max={1}
              step={0.02}
              value={moistureLayer.backfaceOpacity}
              accentColor="#9fc1ff"
              disabled={!backfaceControlsEnabled}
              onChange={(value) => setMoistureLayer({ backfaceOpacity: value })}
            />

            <SliderField
              label="Ambient Light"
              valueLabel={moistureLayer.ambientIntensity.toFixed(2)}
              min={0}
              max={2.5}
              step={0.05}
              value={moistureLayer.ambientIntensity}
              accentColor="#ffe68a"
              disabled={!moistureLayer.lightingEnabled}
              onChange={(value) => setMoistureLayer({ ambientIntensity: value })}
            />

            <SliderField
              label="Key Light"
              valueLabel={moistureLayer.keyLightIntensity.toFixed(2)}
              min={0}
              max={2.5}
              step={0.05}
              value={moistureLayer.keyLightIntensity}
              accentColor="#ffe68a"
              disabled={!moistureLayer.lightingEnabled}
              onChange={(value) =>
                setMoistureLayer({ keyLightIntensity: value })
              }
            />

            <SliderField
              label="Head Light"
              valueLabel={moistureLayer.headLightIntensity.toFixed(2)}
              min={0}
              max={2.5}
              step={0.05}
              value={moistureLayer.headLightIntensity}
              accentColor="#ffe68a"
              disabled={!moistureLayer.lightingEnabled}
              onChange={(value) =>
                setMoistureLayer({ headLightIntensity: value })
              }
            />

            <SliderField
              label="Rim Strength"
              valueLabel={moistureLayer.rimStrength.toFixed(2)}
              min={0}
              max={1.5}
              step={0.05}
              value={moistureLayer.rimStrength}
              accentColor="#d89bff"
              disabled={!moistureLayer.rimEnabled}
              onChange={(value) => setMoistureLayer({ rimStrength: value })}
            />

            <SliderField
              label="Distance Fade"
              valueLabel={moistureLayer.distanceFadeStrength.toFixed(2)}
              min={0}
              max={1.25}
              step={0.05}
              value={moistureLayer.distanceFadeStrength}
              accentColor="#8ffff2"
              disabled={!moistureLayer.distanceFadeEnabled}
              onChange={(value) =>
                setMoistureLayer({ distanceFadeStrength: value })
              }
            />
          </div>

          <div
            style={{
              display: "grid",
              gap: 10,
              marginTop: 2,
              paddingTop: 12,
              borderTop: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <div style={infoTitleStyle()}>Structure Experiments</div>

            <label style={{ display: "grid", gap: 8 }}>
              <div style={sliderLabelStyle()}>
                <span>Structure Preset</span>
                <span style={{ opacity: 0.68 }}>Legibility mode</span>
              </div>
              <select
                value={moistureLayer.structurePreset}
                onChange={(event) =>
                  setMoistureStructurePreset(
                    event.currentTarget.value as MoistureStructurePreset
                  )
                }
                style={selectStyle()}
              >
                {MOISTURE_STRUCTURE_PRESET_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <button
              type="button"
              onClick={() => resetMoistureStructurePreset()}
              style={actionButtonStyle()}
            >
              Reset To Structure Preset
            </button>

            <CheckboxField
              label="Pick Mode"
              checked={moistureLayer.pickMode}
              accentColor="#7de0ff"
              onChange={(checked) => setMoistureLayer({ pickMode: checked })}
            />

            <CheckboxField
              label="Vertical Wall Fade"
              checked={moistureLayer.verticalWallFadeEnabled}
              accentColor="#92ffb3"
              onChange={(checked) =>
                setMoistureLayer({ verticalWallFadeEnabled: checked })
              }
            />

            <CheckboxField
              label="Footprint Overlay"
              checked={moistureLayer.footprintOverlayEnabled}
              accentColor="#ffc56f"
              onChange={(checked) =>
                setMoistureLayer({ footprintOverlayEnabled: checked })
              }
            />

            <label style={{ display: "grid", gap: 8 }}>
              <div style={sliderLabelStyle()}>
                <span>Focus Mode</span>
                <span style={{ opacity: 0.68 }}>Selection context</span>
              </div>
              <select
                value={moistureLayer.focusMode}
                onChange={(event) =>
                  setMoistureLayer({
                    focusMode: event.currentTarget.value as MoistureFocusMode,
                  })
                }
                style={selectStyle()}
              >
                {MOISTURE_FOCUS_MODE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label style={{ display: "grid", gap: 8 }}>
              <div style={sliderLabelStyle()}>
                <span>Color Mode</span>
                <span style={{ opacity: 0.68 }}>Structure encoding</span>
              </div>
              <select
                value={moistureLayer.colorMode}
                onChange={(event) =>
                  setMoistureLayer({
                    colorMode: event.currentTarget.value as MoistureColorMode,
                  })
                }
                style={selectStyle()}
              >
                {MOISTURE_COLOR_MODE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label style={{ display: "grid", gap: 8 }}>
              <div style={sliderLabelStyle()}>
                <span>Component Sort</span>
                <span style={{ opacity: 0.68 }}>Sidebar ordering</span>
              </div>
              <select
                value={moistureLayer.componentSort}
                onChange={(event) =>
                  setMoistureLayer({
                    componentSort:
                      event.currentTarget.value as MoistureComponentSort,
                  })
                }
                style={selectStyle()}
              >
                {MOISTURE_COMPONENT_SORT_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label style={{ display: "grid", gap: 8 }}>
              <div style={sliderLabelStyle()}>
                <span>Segmentation</span>
                <span style={{ opacity: 0.68 }}>Data variant</span>
              </div>
              <select
                value={moistureLayer.segmentationMode}
                onChange={(event) =>
                  setMoistureLayer({
                    segmentationMode: event.currentTarget.value as typeof moistureLayer.segmentationMode,
                  })
                }
                style={selectStyle()}
              >
                {MOISTURE_SEGMENTATION_MODE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <SliderField
              label="Wall Fade Strength"
              valueLabel={moistureLayer.verticalWallFadeStrength.toFixed(2)}
              min={0}
              max={1}
              step={0.05}
              value={moistureLayer.verticalWallFadeStrength}
              accentColor="#92ffb3"
              disabled={!moistureLayer.verticalWallFadeEnabled}
              onChange={(value) =>
                setMoistureLayer({ verticalWallFadeStrength: value })
              }
            />

            <SliderField
              label="Non-selected Opacity"
              valueLabel={moistureLayer.nonSelectedOpacity.toFixed(2)}
              min={0}
              max={0.4}
              step={0.02}
              value={moistureLayer.nonSelectedOpacity}
              accentColor="#7de0ff"
              disabled={moistureLayer.focusMode === "none"}
              onChange={(value) =>
                setMoistureLayer({ nonSelectedOpacity: value })
              }
            />
          </div>

          <div style={{ display: "grid", gap: 4 }}>
            <div style={infoTitleStyle()}>Threshold Mode</div>
            <div style={{ lineHeight: 1.45, opacity: 0.85 }}>
              Switch between `p95 + closing`, `p95 no closing`, and `p97 +
              closing` to compare how segmentation changes the dominant
              structures.
            </div>
          </div>

          <div style={{ display: "grid", gap: 4 }}>
            <div style={infoTitleStyle()}>Display Mode</div>
            <div style={{ lineHeight: 1.45, opacity: 0.85 }}>
              Combine the depth harness with component-aware colors, picking,
              focus modes, wall suppression, and macro footprints to test what
              makes structure easiest to read.
            </div>
          </div>

          <div style={{ display: "grid", gap: 4 }}>
            <div style={infoTitleStyle()}>Legibility Mode</div>
            <div style={{ lineHeight: 1.45, opacity: 0.85 }}>
              Use the named legibility experiments for apples-to-apples
              screenshot comparisons. Segmentation now changes the moisture
              recipe; legibility only changes how that recipe is rendered.
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
