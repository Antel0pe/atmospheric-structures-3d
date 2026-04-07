"use client";

import {
  RELATIVE_HUMIDITY_COLOR_MODE_OPTIONS,
  type RelativeHumidityColorMode,
  useControls,
} from "../../state/controlsStore";

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

function CheckboxRow({
  label,
  checked,
  accentColor,
  onChange,
}: {
  label: string;
  checked: boolean;
  accentColor: string;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label style={rowStyle()}>
      <span style={{ fontWeight: 700 }}>{label}</span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.currentTarget.checked)}
        style={{ accentColor }}
      />
    </label>
  );
}

export default function LayerVisibilityPane() {
  const moistureVisible = useControls(
    (state) => state.moistureStructureLayer.visible
  );
  const precipitationLayer = useControls((state) => state.precipitationRadarLayer);
  const relativeHumidityLayer = useControls(
    (state) => state.relativeHumidityLayer
  );
  const setMoistureStructureLayer = useControls(
    (state) => state.setMoistureStructureLayer
  );
  const setPrecipitationRadarLayer = useControls(
    (state) => state.setPrecipitationRadarLayer
  );
  const setRelativeHumidityLayer = useControls(
    (state) => state.setRelativeHumidityLayer
  );

  return (
    <section style={sectionStyle()}>
      <div
        style={{
          fontWeight: 800,
          letterSpacing: ".02em",
          textTransform: "uppercase",
          marginBottom: 12,
        }}
      >
        Quick Layers
      </div>

      <div style={{ display: "grid", gap: 12 }}>
        <CheckboxRow
          label="Moisture Structures"
          checked={moistureVisible}
          accentColor="#ff8a9a"
          onChange={(checked) => setMoistureStructureLayer({ visible: checked })}
        />

        <CheckboxRow
          label="Relative Humidity Shell"
          checked={relativeHumidityLayer.visible}
          accentColor="#7fe7ff"
          onChange={(checked) => setRelativeHumidityLayer({ visible: checked })}
        />

        {relativeHumidityLayer.visible ? (
          <label style={{ display: "grid", gap: 8 }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                gap: 8,
                fontWeight: 600,
              }}
            >
              <span>RH Color Mode</span>
              <span style={{ opacity: 0.68 }}>Voxel shell</span>
            </div>
            <select
              value={relativeHumidityLayer.colorMode}
              onChange={(event) =>
                setRelativeHumidityLayer({
                  colorMode: event.currentTarget.value as RelativeHumidityColorMode,
                })
              }
              style={selectStyle()}
            >
              {RELATIVE_HUMIDITY_COLOR_MODE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        ) : null}

        <CheckboxRow
          label="Precipitation Radar"
          checked={precipitationLayer.visible}
          accentColor="#8dff75"
          onChange={(checked) => setPrecipitationRadarLayer({ visible: checked })}
        />

        <div style={{ lineHeight: 1.45, opacity: 0.75 }}>
          Relative humidity currently uses the generated shell asset nearest to the
          active timestamp. The precipitation layer uses static radar textures.
        </div>
      </div>
    </section>
  );
}
