"use client";

import dynamic from "next/dynamic";
import ControlsHelp from "./ControlsHelp";
import ExplainerCard from "./ExplainerCard";

const DevViewerPane = dynamic(() => import("./DevViewerPane"), {
  ssr: false,
  loading: () => null,
});

const LayerVisibilityPane = dynamic(() => import("./LayerVisibilityPane"), {
  ssr: false,
  loading: () => null,
});

const TweakpaneControls = dynamic(() => import("@/app/state/TweakpaneControls"), {
  ssr: false,
  loading: () => null,
});

export default function SidebarPane() {
  return (
    <aside
      style={{
        position: "relative",
        top: 0,
        right: 0,
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        backdropFilter: "blur(6px)",
        background: "transparent",
        borderRight: "1px solid rgba(255,255,255,0.08)",
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
        <ExplainerCard />
        <ControlsHelp />
        <DevViewerPane />
        <LayerVisibilityPane />
        <TweakpaneControls />
      </div>
    </aside>
  );
}
