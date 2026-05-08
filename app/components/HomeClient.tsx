"use client";

import dynamic from "next/dynamic";
import { useEffect, useState, type ReactNode } from "react";
import SidebarPane from "./sidebar/SidebarPane";
import DataNoticeOverlay from "./DataNoticeOverlay";
import TemperatureSliceColorScaleLegend from "./TemperatureSliceColorScaleLegend";
import { useViewerStore } from "../state/viewerStore";
import type { EarthProjectionMode } from "./layers/EarthBase";

const EarthBase = dynamic(() => import("./layers/EarthBase"), {
  ssr: false,
  loading: () => <div style={{ width: "100%", height: "100%" }} />,
});

const PrecipitableWaterProxyLayer = dynamic(
  () => import("./layers/PrecipitableWaterProxyLayer"),
  {
    ssr: false,
    loading: () => <div style={{ width: "100%", height: "100%" }} />,
  }
);

const PotentialTemperatureStructuresLayer = dynamic(
  () => import("./layers/PotentialTemperatureStructuresLayer"),
  {
    ssr: false,
    loading: () => <div style={{ width: "100%", height: "100%" }} />,
  }
);

const TemperatureSliceLayer = dynamic(
  () => import("./layers/TemperatureSliceLayer"),
  {
    ssr: false,
    loading: () => <div style={{ width: "100%", height: "100%" }} />,
  }
);

const AirMassClassificationLayer = dynamic(
  () => import("./layers/AirMassClassificationLayer"),
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

const TimeSlider = dynamic(() => import("./TimeSlider"), {
  ssr: false,
  loading: () => <div style={{ height: "100%" }} />,
});

const LayerInfoPane = dynamic(() => import("./sidebar/LayerInfoPane"), {
  ssr: false,
  loading: () => <div style={{ width: "100%", height: "100%" }} />,
});

type HomeClientProps = {
  projectionMode?: EarthProjectionMode;
};

function formatWorkbenchTimestamp(value: string) {
  const parsed = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/.exec(value);
  if (!parsed) return `${value} UTC`;
  return `${parsed[1]}-${parsed[2]}-${parsed[3]}  ${parsed[4]}:${parsed[5]} UTC`;
}

function stepTimestamp(value: string, hours: number) {
  const parsed = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/.exec(value);
  if (!parsed) return value;
  const next = new Date(
    Date.UTC(
      Number(parsed[1]),
      Number(parsed[2]) - 1,
      Number(parsed[3]),
      Number(parsed[4]) + hours,
      Number(parsed[5])
    )
  );
  const year = next.getUTCFullYear();
  const month = String(next.getUTCMonth() + 1).padStart(2, "0");
  const day = String(next.getUTCDate()).padStart(2, "0");
  const hour = String(next.getUTCHours()).padStart(2, "0");
  const minute = String(next.getUTCMinutes()).padStart(2, "0");
  return `${year}-${month}-${day}T${hour}:${minute}`;
}

function WorkbenchIcon({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <button type="button" className="atm-icon-button" aria-label={label} title={label}>
      {children}
    </button>
  );
}

function TopWorkbenchBar({
  timestamp,
  onTimestampChange,
}: {
  timestamp: string;
  onTimestampChange: (timestamp: string) => void;
}) {
  return (
    <header className="atm-topbar">
      <div className="atm-brand">
        <div className="atm-brand-mark" aria-hidden>
          <svg viewBox="0 0 32 32">
            <path d="M16 3.5a12.5 12.5 0 1 1-8.84 3.66" />
            <path d="M16 7.5a8.5 8.5 0 1 0 6.01 2.49" />
            <path d="M16 2v8M16 22v8M2 16h8M22 16h8" />
            <path d="M8.2 8.2l5.6 5.6M18.2 18.2l5.6 5.6M23.8 8.2l-5.6 5.6M13.8 18.2l-5.6 5.6" />
          </svg>
        </div>
        <span>Atmospheric Workbench</span>
      </div>

      <div className="atm-time-jump" aria-label="Current timestamp">
        <button
          type="button"
          className="atm-step-button"
          aria-label="Previous time"
          onClick={() => onTimestampChange(stepTimestamp(timestamp, -3))}
        >
          ‹
        </button>
        <span className="atm-calendar" aria-hidden>
          <svg viewBox="0 0 24 24">
            <path d="M7 3v4M17 3v4M4.5 9h15M6 5h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z" />
          </svg>
        </span>
        <span>{formatWorkbenchTimestamp(timestamp)}</span>
        <button
          type="button"
          className="atm-step-button"
          aria-label="Next time"
          onClick={() => onTimestampChange(stepTimestamp(timestamp, 3))}
        >
          ›
        </button>
      </div>

      <nav className="atm-toolbar" aria-label="Workbench tools">
        <WorkbenchIcon label="Globe">
          <svg viewBox="0 0 24 24"><path d="M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18ZM3.8 12h16.4M12 3c2.2 2.4 3.3 5.4 3.3 9S14.2 18.6 12 21c-2.2-2.4-3.3-5.4-3.3-9S9.8 5.4 12 3Z" /></svg>
        </WorkbenchIcon>
        <WorkbenchIcon label="Analyze">
          <svg viewBox="0 0 24 24"><path d="M4 19h16M6 16l4-5 3 3 5-8M18 6h2v2" /></svg>
        </WorkbenchIcon>
        <WorkbenchIcon label="Layers">
          <svg viewBox="0 0 24 24"><path d="m12 4 8 4-8 4-8-4 8-4Zm-8 8 8 4 8-4M4 16l8 4 8-4" /></svg>
        </WorkbenchIcon>
        <WorkbenchIcon label="Capture">
          <svg viewBox="0 0 24 24"><path d="M8 7h1.5l1-2h3l1 2H16a3 3 0 0 1 3 3v6a3 3 0 0 1-3 3H8a3 3 0 0 1-3-3v-6a3 3 0 0 1 3-3Zm4 9a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z" /></svg>
        </WorkbenchIcon>
        <WorkbenchIcon label="Share">
          <svg viewBox="0 0 24 24"><path d="M12 4v11M8 8l4-4 4 4M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" /></svg>
        </WorkbenchIcon>
      </nav>

      <div className="atm-user-menu">
        <span>MW</span>
        <span aria-hidden>⌄</span>
      </div>
    </header>
  );
}

export default function HomeClient({
  projectionMode = "globe",
}: HomeClientProps = {}) {
  const [allReady, setAllReady] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [layerInfoOpen, setLayerInfoOpen] = useState(true);
  const datehour = useViewerStore((state) => state.timestamp);
  const setTimestamp = useViewerStore((state) => state.setTimestamp);
  const applySavedViewRequest = useViewerStore(
    (state) => state.applySavedViewRequest
  );
  const sidebarWidth = "288px";
  const layerInfoWidth = "304px";

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
      <TopWorkbenchBar timestamp={datehour} onTimestampChange={setTimestamp} />

      <div
        style={{
          position: "absolute",
          inset: 0,
          zIndex: 0,
        }}
      >
        <EarthBase
          timestamp={datehour}
          projectionMode={projectionMode}
          onAllReadyChange={(ready, timestamp) => {
            if (timestamp === datehour) setAllReady(ready);
          }}
        >
          <PrecipitationRadarLayer />
          <PrecipitableWaterProxyLayer />
          <TemperatureSliceLayer />
          <PotentialTemperatureStructuresLayer />
          <AirMassClassificationLayer />
        </EarthBase>
      </div>

      <TemperatureSliceColorScaleLegend
        timestamp={datehour}
        sidebarOpen={sidebarOpen}
        sidebarWidth={sidebarWidth}
        layerInfoOpen={layerInfoOpen}
        layerInfoWidth={layerInfoWidth}
      />

      <div
        style={{
          position: "absolute",
          left: sidebarOpen ? `calc(${sidebarWidth} + 28px)` : 28,
          right: layerInfoOpen ? `calc(${layerInfoWidth} + 28px)` : 28,
          bottom: 14,
          zIndex: 80,
          display: "flex",
          justifyContent: "center",
          pointerEvents: "none",
          transition: "left 220ms ease, right 220ms ease",
        }}
      >
        <div
          style={{
            width: "min(1080px, 100%)",
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
        className="atm-floating-tab"
        style={{
          position: "absolute",
          top: 72,
          left: sidebarOpen ? `calc(${sidebarWidth} - 46px)` : 14,
          zIndex: 130,
          width: sidebarOpen ? 30 : 92,
          height: 32,
          transition:
            "left 220ms cubic-bezier(0.2, 0.8, 0.2, 1), width 220ms cubic-bezier(0.2, 0.8, 0.2, 1)",
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
        className="atm-side-shell atm-side-shell-left"
        style={{
          position: "absolute",
          top: 60,
          left: 0,
          bottom: 0,
          width: sidebarWidth,
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
          transform: sidebarOpen ? "translateX(0)" : "translateX(-100%)",
          opacity: sidebarOpen ? 1 : 0,
          transition: "transform 220ms cubic-bezier(0.2, 0.8, 0.2, 1), opacity 180ms ease",
          zIndex: 110,
          pointerEvents: sidebarOpen ? "auto" : "none",
        }}
      >
        <SidebarPane />
      </aside>

      <button
        onClick={() => setLayerInfoOpen((value) => !value)}
        aria-label={layerInfoOpen ? "Close layer info" : "Open layer info"}
        className="atm-floating-tab"
        style={{
          position: "absolute",
          top: 72,
          right: 14,
          zIndex: 130,
          width: layerInfoOpen ? 30 : 110,
          height: 32,
          transition:
            "right 220ms cubic-bezier(0.2, 0.8, 0.2, 1), width 220ms cubic-bezier(0.2, 0.8, 0.2, 1)",
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
        className="atm-side-shell atm-side-shell-right"
        style={{
          position: "absolute",
          top: 60,
          right: 0,
          bottom: 0,
          width: layerInfoWidth,
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
          transform: layerInfoOpen ? "translateX(0)" : "translateX(100%)",
          opacity: layerInfoOpen ? 1 : 0,
          transition: "transform 220ms cubic-bezier(0.2, 0.8, 0.2, 1), opacity 180ms ease",
          zIndex: 110,
          pointerEvents: layerInfoOpen ? "auto" : "none",
        }}
      >
        <LayerInfoPane />
      </aside>
    </div>
  );
}
