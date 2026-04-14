// @ts-expect-error Bun provides this module at test runtime, but the repo
// does not include Bun's type package for `tsc --noEmit`.
import { describe, expect, test } from "bun:test";

import {
  DEFAULT_VISIBLE_MOISTURE_BUCKET_INDICES,
  DEFAULT_MOISTURE_SEGMENTATION_MODE,
  getMoistureStructurePresetState,
  getMoistureVisualPresetState,
  resolveMoistureStructureLayerState,
  useControls,
  type MoistureStructureLayerState,
} from "./controlsStore";

function buildBaseState(): MoistureStructureLayerState {
  return {
    visible: true,
    opacity: 0.95,
    verticalExaggeration: 4,
    cameraCutawayEnabled: false,
    cameraCutawayRadius: 40,
    visualPreset: "solidShell",
    structurePreset: "componentRead",
    ...getMoistureVisualPresetState("solidShell"),
    ...getMoistureStructurePresetState("componentRead"),
    selectedComponentId: null,
    componentSort: "size",
    visibleBucketIndices: DEFAULT_VISIBLE_MOISTURE_BUCKET_INDICES,
    legibilityExperiment: "none",
    surfaceCueMode: "none",
    surfaceBrightness: 1,
    surfaceShadowStrength: 1,
  };
}

describe("resolveMoistureStructureLayerState", () => {
  test("the controls store defaults to the neutral lit solid-shell baseline", () => {
    expect(useControls.getState().moistureStructureLayer.legibilityExperiment).toBe(
      "bridgePruned"
    );
    expect(useControls.getState().moistureStructureLayer.visible).toBe(false);
    expect(useControls.getState().relativeHumidityLayer.visible).toBe(false);
    expect(useControls.getState().relativeHumidityLayer.opacity).toBe(1);
    expect(useControls.getState().relativeHumidityLayer.colorMode).toBe(
      "pressureBands"
    );
    expect(useControls.getState().relativeHumidityLayer.variant).toBe(
      "baseline"
    );
    expect(useControls.getState().precipitationRadarLayer.visible).toBe(false);
    expect(useControls.getState().potentialTemperatureLayer.visible).toBe(true);
    expect(useControls.getState().potentialTemperatureLayer.opacity).toBe(1);
    expect(useControls.getState().moistureStructureLayer.verticalExaggeration).toBe(
      2.35
    );
    expect(useControls.getState().moistureStructureLayer.structurePreset).toBe(
      "currentDepth"
    );
    expect(useControls.getState().moistureStructureLayer.surfaceCueMode).toBe(
      "none"
    );
    expect(useControls.getState().moistureStructureLayer.surfaceBrightness).toBe(
      1
    );
    expect(
      useControls.getState().moistureStructureLayer.surfaceShadowStrength
    ).toBe(1);
    expect(useControls.getState().moistureStructureLayer.cameraCutawayEnabled).toBe(
      false
    );
    expect(useControls.getState().moistureStructureLayer.segmentationMode).toBe(
      DEFAULT_MOISTURE_SEGMENTATION_MODE
    );
  });

  test("none preserves the baseline state and disables experiment-only uniforms", () => {
    const base = buildBaseState();
    const resolved = resolveMoistureStructureLayerState(base);

    expect(resolved).toMatchObject(base);
  });

  test("shellFirst prioritizes the near shell without changing the data variant", () => {
    const resolved = resolveMoistureStructureLayerState({
      ...buildBaseState(),
      segmentationMode: "p95-smooth-open1",
      legibilityExperiment: "shellFirst",
    });

    expect(resolved.segmentationMode).toBe("p95-smooth-open1");
    expect(resolved.solidShellEnabled).toBe(true);
    expect(resolved.distanceFadeEnabled).toBe(false);
    expect(resolved.interiorBackfaceEnabled).toBe(false);
    expect(resolved.opacity).toBeGreaterThanOrEqual(0.95);
    expect(resolved.frontOpacity).toBeGreaterThanOrEqual(1.5);
    expect(resolved.backfaceOpacity).toBe(0);
  });

  test("bridgePruned preserves the selected data variant", () => {
    const resolved = resolveMoistureStructureLayerState({
      ...buildBaseState(),
      segmentationMode: "p95-local-anomaly",
      legibilityExperiment: "bridgePruned",
    });

    expect(resolved.segmentationMode).toBe("p95-local-anomaly");
  });

  test("bridgePrunedShellFirst combines shell-first styling without changing the selected data variant", () => {
    const resolved = resolveMoistureStructureLayerState({
      ...buildBaseState(),
      segmentationMode: "p95-smooth-open1",
      legibilityExperiment: "bridgePrunedShellFirst",
    });

    expect(resolved.segmentationMode).toBe("p95-smooth-open1");
    expect(resolved.solidShellEnabled).toBe(true);
    expect(resolved.distanceFadeEnabled).toBe(false);
    expect(resolved.interiorBackfaceEnabled).toBe(false);
    expect(resolved.opacity).toBeGreaterThanOrEqual(0.95);
    expect(resolved.frontOpacity).toBeGreaterThanOrEqual(1.5);
    expect(resolved.backfaceOpacity).toBe(0);
  });

  test("bridgePrunedShellFirstMatte keeps the shell-first shape, preserves the data variant, and softens the direct lights", () => {
    const resolved = resolveMoistureStructureLayerState({
      ...buildBaseState(),
      segmentationMode: "p95-local-anomaly",
      legibilityExperiment: "bridgePrunedShellFirstMatte",
    });

    expect(resolved.segmentationMode).toBe("p95-local-anomaly");
    expect(resolved.solidShellEnabled).toBe(true);
    expect(resolved.distanceFadeEnabled).toBe(false);
    expect(resolved.interiorBackfaceEnabled).toBe(false);
    expect(resolved.opacity).toBeGreaterThanOrEqual(0.95);
    expect(resolved.frontOpacity).toBeGreaterThanOrEqual(1.5);
    expect(resolved.backfaceOpacity).toBe(0);
    expect(resolved.keyLightIntensity).toBeLessThanOrEqual(0.72);
    expect(resolved.headLightIntensity).toBeLessThanOrEqual(0.42);
    expect(resolved.ambientIntensity).toBeGreaterThanOrEqual(0.92);
  });
});
