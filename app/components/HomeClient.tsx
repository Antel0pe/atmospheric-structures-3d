"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import SidebarPane from "./sidebar/SidebarPane";
import DataNoticeOverlay from "./DataNoticeOverlay";
import { useViewerStore } from "../state/viewerStore";

const EarthBase = dynamic(() => import("./layers/EarthBase"), {
  ssr: false,
  loading: () => <div style={{ width: "100%", height: "100%" }} />,
});

const MoistureStructureLayer = dynamic(
  () => import("./layers/MoistureStructureLayer"),
  {
    ssr: false,
    loading: () => <div style={{ width: "100%", height: "100%" }} />,
  }
);

const RelativeHumidityVoxelLayer = dynamic(
  () => import("./layers/RelativeHumidityVoxelLayer"),
  {
    ssr: false,
    loading: () => <div style={{ width: "100%", height: "100%" }} />,
  }
);

const PrecipitableWaterProxyLayer = dynamic(
  () => import("./layers/PrecipitableWaterProxyLayer"),
  {
    ssr: false,
    loading: () => <div style={{ width: "100%", height: "100%" }} />,
  }
);

const PrecipitationRadarLayer = dynamic(
  () => import("./layers/PrecipitationRadarLayer"),
  {
    ssr: false,
    loading: () => <div style={{ width: "100%", height: "100%" }} />,
  }
);

const ExampleShaderMeshLayer = dynamic(
  () => import("./layers/ExampleShaderMeshLayer"),
  {
    ssr: false,
    loading: () => <div style={{ width: "100%", height: "100%" }} />,
  }
);

const ExampleContoursLayer = dynamic(
  () => import("./layers/ExampleContoursLayer"),
  {
    ssr: false,
    loading: () => <div style={{ width: "100%", height: "100%" }} />,
  }
);

const ExampleParticleLayer = dynamic(
  () => import("./layers/ExampleParticleLayer"),
  {
    ssr: false,
    loading: () => <div style={{ width: "100%", height: "100%" }} />,
  }
);

const TimeSlider = dynamic(() => import("./TimeSlider"), {
  ssr: false,
  loading: () => <div style={{ height: "100%" }} />,
});

const LayerInfoPane = dynamic(() => import("./sidebar/LayerInfoPane"), {
  ssr: false,
  loading: () => <div style={{ width: "100%", height: "100%" }} />,
});

export default function HomeClient() {
  const [allReady, setAllReady] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [layerInfoOpen, setLayerInfoOpen] = useState(true);
  const datehour = useViewerStore((state) => state.timestamp);
  const setTimestamp = useViewerStore((state) => state.setTimestamp);
  const applySavedViewRequest = useViewerStore(
    (state) => state.applySavedViewRequest
  );
  const sidebarWidth = "max(15vw, 320px)";
  const layerInfoWidth = sidebarWidth;

  useEffect(() => {
    if (!applySavedViewRequest) return;
    if (applySavedViewRequest.phase !== "initial") return;
    if (applySavedViewRequest.savedView.timestamp === datehour) return;

    setTimestamp(applySavedViewRequest.savedView.timestamp);
  }, [applySavedViewRequest, datehour, setTimestamp]);

  useEffect(() => {
    if (!allReady || !applySavedViewRequest) return;
    if (applySavedViewRequest.phase !== "initial") return;
    if (applySavedViewRequest.savedView.timestamp !== datehour) return;

    useViewerStore
      .getState()
      .promoteApplySavedViewReady(applySavedViewRequest.requestId);
  }, [allReady, applySavedViewRequest, datehour]);

  return (
    <div
      style={{
        position: "relative",
        width: "100vw",
        height: "100vh",
        overflow: "hidden",
        background: "#04070d",
      }}
    >
      <DataNoticeOverlay />

      <div
        style={{
          position: "absolute",
          inset: 0,
        }}
      >
        <EarthBase
          timestamp={datehour}
          onAllReadyChange={(ready, timestamp) => {
            if (timestamp === datehour) setAllReady(ready);
          }}
        >
          <PrecipitationRadarLayer />
          <MoistureStructureLayer />
          <RelativeHumidityVoxelLayer />
          <PrecipitableWaterProxyLayer />
          <ExampleShaderMeshLayer />
          <ExampleContoursLayer />
          <ExampleParticleLayer heightTex={null} />
        </EarthBase>
      </div>

      <div
        style={{
          position: "absolute",
          left: 24,
          right: 24,
          bottom: 24,
          zIndex: 35,
          display: "flex",
          justifyContent: "center",
          pointerEvents: "none",
        }}
      >
        <div
          style={{
            width: "min(960px, 100%)",
            pointerEvents: "auto",
          }}
        >
          <TimeSlider
            key={applySavedViewRequest ? `saved-view-${applySavedViewRequest.requestId}` : "time-slider"}
            value={datehour}
            onChange={setTimestamp}
            allReady={allReady}
          />
        </div>
      </div>

      <button
        onClick={() => setSidebarOpen((value) => !value)}
        aria-label={sidebarOpen ? "Close layers" : "Open layers"}
        style={{
          position: "absolute",
          top: 14,
          left: sidebarOpen ? `calc(${sidebarWidth} - 48px)` : 14,
          zIndex: 50,
          width: sidebarOpen ? 34 : 100,
          height: sidebarOpen ? 34 : 50,
          borderRadius: 12,
          background: "rgba(70, 140, 255, 0.24)",
          border: "1px solid rgba(140, 190, 255, 0.32)",
          color: "white",
          cursor: "pointer",
          backdropFilter: "blur(10px)",
          display: "grid",
          placeItems: "center",
          userSelect: "none",
          boxShadow: "0 6px 18px rgba(0,0,0,0.35)",
          transition:
            "left 220ms cubic-bezier(0.2, 0.8, 0.2, 1), width 220ms cubic-bezier(0.2, 0.8, 0.2, 1), height 220ms cubic-bezier(0.2, 0.8, 0.2, 1)",
        }}
      >
        <span
          style={{
            fontSize: sidebarOpen ? 22 : 14,
            fontWeight: 600,
            lineHeight: 1,
            opacity: 0.95,
          }}
        >
          {sidebarOpen ? "×" : "Layers"}
        </span>
      </button>

      <aside
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          bottom: 0,
          width: sidebarWidth,
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
          backdropFilter: "blur(6px)",
          background: "transparent",
          borderRight: "1px solid rgba(255,255,255,0.12)",
          boxShadow: "20px 0 40px rgba(0,0,0,0.25)",
          transform: sidebarOpen ? "translateX(0)" : "translateX(-100%)",
          opacity: sidebarOpen ? 1 : 0,
          transition: "transform 220ms cubic-bezier(0.2, 0.8, 0.2, 1), opacity 180ms ease",
          zIndex: 45,
          pointerEvents: sidebarOpen ? "auto" : "none",
        }}
      >
        <SidebarPane />
      </aside>

      <button
        onClick={() => setLayerInfoOpen((value) => !value)}
        aria-label={layerInfoOpen ? "Close layer info" : "Open layer info"}
        style={{
          position: "absolute",
          top: 14,
          right: 14,
          zIndex: 50,
          width: layerInfoOpen ? 34 : 114,
          height: layerInfoOpen ? 34 : 50,
          borderRadius: 12,
          background: "rgba(70, 140, 255, 0.24)",
          border: "1px solid rgba(140, 190, 255, 0.32)",
          color: "white",
          cursor: "pointer",
          backdropFilter: "blur(10px)",
          display: "grid",
          placeItems: "center",
          userSelect: "none",
          boxShadow: "0 6px 18px rgba(0,0,0,0.35)",
          transition:
            "width 220ms cubic-bezier(0.2, 0.8, 0.2, 1), height 220ms cubic-bezier(0.2, 0.8, 0.2, 1)",
        }}
      >
        <span
          style={{
            fontSize: layerInfoOpen ? 22 : 14,
            fontWeight: 600,
            lineHeight: 1,
            opacity: 0.95,
          }}
        >
          {layerInfoOpen ? "×" : "Layer Info"}
        </span>
      </button>

      <aside
        style={{
          position: "absolute",
          top: 0,
          right: 0,
          bottom: 0,
          width: layerInfoWidth,
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
          backdropFilter: "blur(6px)",
          background: "transparent",
          borderLeft: "1px solid rgba(255,255,255,0.12)",
          boxShadow: "-20px 0 40px rgba(0,0,0,0.25)",
          transform: layerInfoOpen ? "translateX(0)" : "translateX(100%)",
          opacity: layerInfoOpen ? 1 : 0,
          transition: "transform 220ms cubic-bezier(0.2, 0.8, 0.2, 1), opacity 180ms ease",
          zIndex: 45,
          pointerEvents: layerInfoOpen ? "auto" : "none",
        }}
      >
        <LayerInfoPane />
      </aside>
    </div>
  );
}
