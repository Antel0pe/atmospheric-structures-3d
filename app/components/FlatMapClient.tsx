"use client";

import { useEffect } from "react";
import HomeClient from "./HomeClient";
import { useControls } from "../state/controlsStore";

export default function FlatMapClient() {
  useEffect(() => {
    const controls = useControls.getState();
    // QUICK AND DIRTY NEED TO REDO ROUTE-LEVEL PRESETS BETTER: force the first /2d pass
    // toward the requested theta std-dev side-tail air-mass view without adding route state.
    controls.setAirMassLayer({
      visible: true,
      variant: "theta-anomaly-stddev-side-6neighbor-min100k",
      opacity: 1,
      showCellGrid: false,
      altitudeRange01: { min: 0, max: 1 },
      cameraCutawayEnabled: false,
      hiddenClassKeys: [],
    });
  }, []);

  return <HomeClient projectionMode="flat2d" />;
}
