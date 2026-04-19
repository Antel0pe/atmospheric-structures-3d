import { useCallback, useEffect, useRef } from "react";
import * as THREE from "three";
import { useEarthLayer } from "./EarthBase";
import {
  useControls,
  type PotentialTemperatureColorMode,
} from "../../state/controlsStore";
import {
  fetchPotentialTemperatureStructureFrame,
  type PotentialTemperatureStructureFrame,
} from "../utils/potentialTemperatureStructureAssets";

const LAYER_CLEARANCE = 10.8;
const LAYER_RENDER_ORDER = 68;
const WARM_PRESSURE_COLOR_BANDS = [
  { minHpa: 850, color: new THREE.Color("#ff5a36") },
  { minHpa: 700, color: new THREE.Color("#ff7f2a") },
  { minHpa: 500, color: new THREE.Color("#ffab2e") },
  { minHpa: 350, color: new THREE.Color("#ffd34d") },
  { minHpa: 250, color: new THREE.Color("#fff07a") },
  { minHpa: 150, color: new THREE.Color("#fff3a6") },
  { minHpa: 70, color: new THREE.Color("#fff7c7") },
  { minHpa: 0, color: new THREE.Color("#fffbe1") },
] as const;
const COLD_PRESSURE_COLOR_BANDS = [
  { minHpa: 850, color: new THREE.Color("#1b5fd6") },
  { minHpa: 700, color: new THREE.Color("#355ee8") },
  { minHpa: 500, color: new THREE.Color("#654fe0") },
  { minHpa: 350, color: new THREE.Color("#8b48cf") },
  { minHpa: 250, color: new THREE.Color("#4aa0d8") },
  { minHpa: 150, color: new THREE.Color("#35b8b0") },
  { minHpa: 70, color: new THREE.Color("#4fcb8f") },
  { minHpa: 0, color: new THREE.Color("#7ad56e") },
] as const;
const PRECIPITABLE_WATER_PROXY_COLOR_STOPS = [
  new THREE.Color("#ff8a63"),
  new THREE.Color("#2dc6d6"),
  new THREE.Color("#5e86ff"),
  new THREE.Color("#b95cff"),
] as const;
const WARM_THERMAL_COLOR_BANDS = [
  { minHpa: 850, color: new THREE.Color("#7a1018") },
  { minHpa: 700, color: new THREE.Color("#972023") },
  { minHpa: 500, color: new THREE.Color("#bc3528") },
  { minHpa: 350, color: new THREE.Color("#dd5332") },
  { minHpa: 250, color: new THREE.Color("#f67b4b") },
  { minHpa: 150, color: new THREE.Color("#ffab78") },
  { minHpa: 70, color: new THREE.Color("#ffd0b0") },
  { minHpa: 0, color: new THREE.Color("#ffe9da") },
] as const;
const COLD_THERMAL_COLOR_BANDS = [
  { minHpa: 850, color: new THREE.Color("#0f245f") },
  { minHpa: 700, color: new THREE.Color("#17398a") },
  { minHpa: 500, color: new THREE.Color("#2254b5") },
  { minHpa: 350, color: new THREE.Color("#3378da") },
  { minHpa: 250, color: new THREE.Color("#57a2f0") },
  { minHpa: 150, color: new THREE.Color("#84c2ff") },
  { minHpa: 70, color: new THREE.Color("#b6ddff") },
  { minHpa: 0, color: new THREE.Color("#dff0ff") },
] as const;

type ShellKind = "warm" | "cold";

function pressureToStandardAtmosphereHeightM(pressureHpa: number) {
  const safePressure = Math.max(pressureHpa, 1);
  return 44330.0 * (1.0 - (safePressure / 1013.25) ** 0.1903);
}

function standardAtmosphereHeightMToPressure(heightM: number) {
  const normalized = THREE.MathUtils.clamp(1.0 - heightM / 44330.0, 1e-6, 1);
  return 1013.25 * normalized ** (1 / 0.1903);
}

function pressureToRadius(
  frame: PotentialTemperatureStructureFrame,
  pressureHpa: number
) {
  const {
    base_radius: baseRadius,
    vertical_span: verticalSpan,
    reference_pressure_hpa: { min: minPressure, max: maxPressure },
  } = frame.manifest.globe;

  const minHeight = pressureToStandardAtmosphereHeightM(maxPressure);
  const maxHeight = pressureToStandardAtmosphereHeightM(minPressure);
  const scale = verticalSpan / Math.max(maxHeight - minHeight, 1e-9);

  return (
    baseRadius +
    (pressureToStandardAtmosphereHeightM(pressureHpa) - minHeight) * scale
  );
}

function radiusToPressureHpa(frame: PotentialTemperatureStructureFrame, radius: number) {
  const {
    base_radius: baseRadius,
    vertical_span: verticalSpan,
    reference_pressure_hpa: { min: minPressure, max: maxPressure },
  } = frame.manifest.globe;

  const minHeight = pressureToStandardAtmosphereHeightM(maxPressure);
  const maxHeight = pressureToStandardAtmosphereHeightM(minPressure);
  const scale = (maxHeight - minHeight) / Math.max(verticalSpan, 1e-9);
  const heightM = minHeight + Math.max(radius - baseRadius, 0) * scale;
  return standardAtmosphereHeightMToPressure(heightM);
}

function colorAtStopPosition(t: number, stops: readonly THREE.Color[]) {
  const scaled = THREE.MathUtils.clamp(t, 0, 1) * (stops.length - 1);
  const startIndex = Math.floor(scaled);
  const endIndex = Math.min(startIndex + 1, stops.length - 1);
  const mix = scaled - startIndex;
  return stops[startIndex].clone().lerp(stops[endIndex], mix);
}

function colorForPressureBand(
  pressureHpa: number,
  colorMode: PotentialTemperatureColorMode,
  kind: ShellKind
) {
  const bands =
    colorMode === "thermalContrast"
      ? kind === "warm"
        ? WARM_THERMAL_COLOR_BANDS
        : COLD_THERMAL_COLOR_BANDS
      : kind === "warm"
        ? WARM_PRESSURE_COLOR_BANDS
        : COLD_PRESSURE_COLOR_BANDS;
  for (const band of bands) {
    if (pressureHpa >= band.minHpa) {
      return band.color;
    }
  }
  return bands[bands.length - 1].color;
}

function buildBandScale(
  frame: PotentialTemperatureStructureFrame,
  colorMode: PotentialTemperatureColorMode,
  kind: ShellKind
) {
  const thresholds = frame.metadata.selection.thresholds_by_pressure_level;
  if (thresholds.length === 0) {
    return { boundaryRadii: [] as number[], levelColors: [] as THREE.Color[] };
  }
  const radii = thresholds.map((entry) => pressureToRadius(frame, entry.pressure_hpa));
  const boundaryRadii = radii
    .slice(0, -1)
    .map((radius, index) => (radius + radii[index + 1]) / 2);
  const levelColors =
    colorMode === "precipitableWaterProxy"
      ? radii.map((_, index) =>
          colorAtStopPosition(
            index / Math.max(radii.length - 1, 1),
            PRECIPITABLE_WATER_PROXY_COLOR_STOPS
          )
        )
      : thresholds.map((entry) =>
          colorForPressureBand(entry.pressure_hpa, colorMode, kind)
        );

  return { boundaryRadii, levelColors };
}

function levelIndexForRadius(radius: number, boundaryRadii: number[]) {
  let low = 0;
  let high = boundaryRadii.length;

  while (low < high) {
    const mid = (low + high) >> 1;
    if (radius > boundaryRadii[mid]) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
}

function buildGeometry(
  frame: PotentialTemperatureStructureFrame,
  verticalExaggeration: number,
  kind: ShellKind,
  colorMode: PotentialTemperatureColorMode
) {
  const bandScale = buildBandScale(frame, colorMode, kind);
  const positions =
    kind === "warm" ? frame.warmPositions.slice() : frame.coldPositions.slice();
  const indices =
    kind === "warm" ? frame.warmIndices.slice() : frame.coldIndices.slice();
  const colors = new Float32Array((positions.length / 3) * 3);
  const baseRadius = frame.manifest.globe.base_radius;

  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    const radius = Math.hypot(x, y, z);
    if (radius <= 1e-6) continue;

    const radialOffset = Math.max(radius - baseRadius, 0);
    const exaggeratedRadius =
      baseRadius + LAYER_CLEARANCE + radialOffset * verticalExaggeration;
    const scale = exaggeratedRadius / radius;

    positions[i] *= scale;
    positions[i + 1] *= scale;
    positions[i + 2] *= scale;

    const levelIndex = levelIndexForRadius(radius, bandScale.boundaryRadii);
    const pressureHpa = radiusToPressureHpa(frame, radius);
    const color =
      bandScale.levelColors[levelIndex] ??
      colorForPressureBand(pressureHpa, colorMode, kind);
    colors[i] = color.r;
    colors[i + 1] = color.g;
    colors[i + 2] = color.b;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

export default function PotentialTemperatureStructuresLayer() {
  const layerState = useControls((state) => state.potentialTemperatureLayer);
  const verticalExaggeration = useControls(
    (state) => state.moistureStructureLayer.verticalExaggeration
  );
  const { engineReady, sceneRef, globeRef, signalReady, timestamp } =
    useEarthLayer("potential-temperature-structures");

  const rootRef = useRef<THREE.Group | null>(null);
  const warmMeshRef = useRef<THREE.Mesh | null>(null);
  const coldMeshRef = useRef<THREE.Mesh | null>(null);
  const materialRef = useRef<THREE.MeshLambertMaterial | null>(null);
  const frameRef = useRef<PotentialTemperatureStructureFrame | null>(null);
  const reqIdRef = useRef(0);

  const rebuildMesh = useCallback(() => {
    const root = rootRef.current;
    const material = materialRef.current;
    const frame = frameRef.current;
    if (!root || !material || !frame) return;
    root.userData.variant = frame.manifest.variant ?? layerState.variant;

    warmMeshRef.current?.removeFromParent();
    warmMeshRef.current?.geometry.dispose();
    coldMeshRef.current?.removeFromParent();
    coldMeshRef.current?.geometry.dispose();

    const warmGeometry = buildGeometry(
      frame,
      verticalExaggeration,
      "warm",
      layerState.colorMode
    );
    const warmMesh = new THREE.Mesh(warmGeometry, material);
    warmMesh.name = "potential-temperature-warm-shell";
    warmMesh.renderOrder = LAYER_RENDER_ORDER;
    warmMesh.frustumCulled = false;
    root.add(warmMesh);
    warmMeshRef.current = warmMesh;

    const coldGeometry = buildGeometry(
      frame,
      verticalExaggeration,
      "cold",
      layerState.colorMode
    );
    const coldMesh = new THREE.Mesh(coldGeometry, material);
    coldMesh.name = "potential-temperature-cold-shell";
    coldMesh.renderOrder = LAYER_RENDER_ORDER;
    coldMesh.frustumCulled = false;
    root.add(coldMesh);
    coldMeshRef.current = coldMesh;
  }, [layerState.colorMode, verticalExaggeration]);

  useEffect(() => {
    if (!engineReady) return;
    if (!sceneRef.current || !globeRef.current) return;

    const root = new THREE.Group();
    root.name = "potential-temperature-structures-root";
    root.visible = layerState.visible;
    root.renderOrder = LAYER_RENDER_ORDER;
    root.frustumCulled = false;
    sceneRef.current.add(root);
    rootRef.current = root;

    const material = new THREE.MeshLambertMaterial({
      vertexColors: true,
      transparent: false,
      opacity: 1,
      depthWrite: true,
      depthTest: true,
      side: THREE.FrontSide,
      flatShading: true,
    });
    materialRef.current = material;

    return () => {
      warmMeshRef.current?.geometry.dispose();
      coldMeshRef.current?.geometry.dispose();
      material.dispose();
      materialRef.current = null;
      warmMeshRef.current = null;
      coldMeshRef.current = null;
      rootRef.current = null;
      root.removeFromParent();
    };
  }, [engineReady, globeRef, sceneRef]);

  useEffect(() => {
    if (!engineReady) return;
    const root = rootRef.current;
    const material = materialRef.current;
    if (!root || !material) return;

    root.visible = layerState.visible;
    material.transparent = layerState.opacity < 0.999;
    material.opacity = layerState.opacity;
    material.depthWrite = layerState.opacity >= 0.999;
    material.side = THREE.FrontSide;
    if (frameRef.current) {
      rebuildMesh();
    }
  }, [engineReady, layerState.opacity, layerState.visible, rebuildMesh]);

  useEffect(() => {
    if (!engineReady) return;
    const root = rootRef.current;
    if (!root) return;

    let cancelled = false;
    const requestId = ++reqIdRef.current;
    const isCancelled = () => cancelled || requestId !== reqIdRef.current;

    if (!layerState.visible) {
      root.visible = false;
      signalReady(timestamp);
      return () => {
        cancelled = true;
      };
    }

    root.visible = true;

    void fetchPotentialTemperatureStructureFrame(timestamp, {
      variant: layerState.variant,
    })
      .then((frame) => {
        if (isCancelled()) return;
        frameRef.current = frame;
        rebuildMesh();
        signalReady(timestamp);
      })
      .catch((error) => {
        if (isCancelled()) return;
        console.error("Failed to load potential temperature layer", error);
        signalReady(timestamp);
      });

    return () => {
      cancelled = true;
    };
  }, [
    engineReady,
    layerState.variant,
    layerState.visible,
    rebuildMesh,
    signalReady,
    timestamp,
  ]);

  return null;
}
