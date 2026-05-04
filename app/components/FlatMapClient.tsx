"use client";

import { useEffect } from "react";
import HomeClient from "./HomeClient";
import { useControls } from "../state/controlsStore";

export default function FlatMapClient() {
  useEffect(() => {
    const controls = useControls.getState();
    controls.setVerticalExaggeration(8);
    controls.setAirMassLayer({
      visible: false,
      opacity: 1,
      showCellGrid: false,
      altitudeRange01: { min: 0, max: 1 },
      cameraCutawayEnabled: false,
      hiddenClassKeys: [],
    });
    controls.setTemperatureSliceLayer({
      visible: true,
      pressureHpa: 500,
      opacity: 1,
      variant: "raw-temperature",
      colorScaleMode: "global",
    });
  }, []);

  return <HomeClient projectionMode="flat2d" />;
}
