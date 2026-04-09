// @ts-expect-error Bun provides this module at test runtime, but the repo
// does not include Bun's type package for `tsc --noEmit`.
import { describe, expect, test } from "bun:test";

import { useControls } from "../state/controlsStore";
import {
  isNormalizedScreenTarget,
  isViewDebugCase,
  type ViewDebugCase,
} from "./viewDebug";

function buildEarthView() {
  return {
    cameraPosition: { x: 1, y: 2, z: 3 },
    cameraQuaternion: { x: 0, y: 0, z: 0, w: 1 },
    cameraUp: { x: 0, y: 1, z: 0 },
    controlsTarget: { x: 0, y: 0, z: 0 },
    yaw: 0.1,
    pitch: -0.2,
    zoom01: 0.5,
  };
}

function buildCase(): ViewDebugCase {
  const controls = useControls.getState();
  return {
    version: 1,
    analyzer: "moisture-structure",
    title: "Random walls debug",
    createdAt: "2026-04-07T00:00:00.000Z",
    source: {
      kind: "saved-view",
      id: "view-1",
      title: "random walls",
      path: "saved-views/random-walls.json",
    },
    timestamp: "2021-11-08T12:00",
    earthView: buildEarthView(),
    layerState: {
      moistureStructureLayer: {
        ...controls.moistureStructureLayer,
        visibleBucketIndices: [...controls.moistureStructureLayer.visibleBucketIndices],
      },
      precipitationRadarLayer: { ...controls.precipitationRadarLayer },
      relativeHumidityLayer: { ...controls.relativeHumidityLayer },
      exampleShaderMeshLayer: { ...controls.exampleShaderMeshLayer },
      exampleContoursLayer: { ...controls.exampleContoursLayer },
      exampleParticleLayer: { ...controls.exampleParticleLayer },
    },
    targets: [{ id: "target-1", label: "Target 1", x: 0.5, y: 0.5 }],
    notes: "Local debug case for a wall-like target.",
  };
}

describe("view-debug validators", () => {
  test("accepts normalized screen targets inside the viewport bounds", () => {
    expect(isNormalizedScreenTarget({ x: 0, y: 0 })).toBe(true);
    expect(isNormalizedScreenTarget({ x: 0.5, y: 1, id: "t1" })).toBe(true);
  });

  test("rejects normalized screen targets outside the viewport bounds", () => {
    expect(isNormalizedScreenTarget({ x: -0.01, y: 0.5 })).toBe(false);
    expect(isNormalizedScreenTarget({ x: 0.5, y: 1.01 })).toBe(false);
  });

  test("accepts a complete view debug case snapshot", () => {
    expect(isViewDebugCase(buildCase())).toBe(true);
  });

  test("rejects a view debug case with an invalid target", () => {
    const value = buildCase() as unknown as { targets: Array<{ x: number; y: number }> };
    value.targets = [{ x: 1.2, y: 0.5 }];
    expect(isViewDebugCase(value)).toBe(false);
  });
});
