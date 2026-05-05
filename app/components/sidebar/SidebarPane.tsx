"use client";

import dynamic from "next/dynamic";
import ControlsHelp from "./ControlsHelp";

const DevViewerPane = dynamic(() => import("./DevViewerPane"), {
  ssr: false,
  loading: () => null,
});

const LayerVisibilityPane = dynamic(() => import("./LayerVisibilityPane"), {
  ssr: false,
  loading: () => null,
});

export default function SidebarPane() {
  return (
    <aside
      className="atm-sidebar"
      style={{
        position: "relative",
        top: 0,
        right: 0,
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        zIndex: 1000,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          overflowX: "hidden",
          WebkitOverflowScrolling: "touch",
        }}
      >
        <ControlsHelp />
        <LayerVisibilityPane />
        <DevViewerPane />
      </div>
    </aside>
  );
}
