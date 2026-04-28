// @ts-expect-error Bun provides this module at test runtime, but the repo
// does not include Bun's type package for `tsc --noEmit`.
import { describe, expect, test } from "bun:test";

import { useControls } from "./controlsStore";

describe("controls store", () => {
  test("defaults to the current active layer surface", () => {
    const state = useControls.getState();

    expect(state.verticalExaggeration).toBe(2.35);
    expect(state.precipitationRadarLayer).toEqual({
      visible: false,
      opacity: 0.92,
    });
    expect(state.precipitableWaterLayer).toEqual({
      visible: false,
      opacity: 1,
    });
    expect(state.potentialTemperatureLayer).toEqual({
      visible: false,
      opacity: 1,
      colorMode: "pressureBands",
      variant: "bridge-gap-1",
      showCellGrid: false,
    });
    expect(state.airMassLayer).toEqual({
      visible: true,
      opacity: 1,
      variant: "theta-anomaly-stddev-side-6neighbor-min100k",
      showCellGrid: false,
      altitudeRange01: { min: 0, max: 1 },
      cameraCutawayEnabled: false,
      cameraCutawayRadius: 40,
      hiddenClassKeys: [],
    });
  });

  test("layer setters merge patches into the existing layer state", () => {
    const controls = useControls.getState();

    controls.setVerticalExaggeration(3.5);
    controls.setPrecipitationRadarLayer({ visible: true, opacity: 0.4 });
    controls.setPotentialTemperatureLayer({
      visible: true,
      variant: "fill-between-anchors",
    });
    controls.setAirMassLayer({
      showCellGrid: true,
      altitudeRange01: { min: 0.2, max: 0.75 },
      cameraCutawayEnabled: true,
      cameraCutawayRadius: 72,
      hiddenClassKeys: ["bucket_0"],
    });

    const nextState = useControls.getState();
    expect(nextState.verticalExaggeration).toBe(3.5);
    expect(nextState.precipitationRadarLayer).toEqual({
      visible: true,
      opacity: 0.4,
    });
    expect(nextState.potentialTemperatureLayer).toMatchObject({
      visible: true,
      opacity: 1,
      colorMode: "pressureBands",
      variant: "fill-between-anchors",
      showCellGrid: false,
    });
    expect(nextState.airMassLayer).toMatchObject({
      visible: true,
      opacity: 1,
      showCellGrid: true,
      altitudeRange01: { min: 0.2, max: 0.75 },
      cameraCutawayEnabled: true,
      cameraCutawayRadius: 72,
      hiddenClassKeys: ["bucket_0"],
    });
  });
});
