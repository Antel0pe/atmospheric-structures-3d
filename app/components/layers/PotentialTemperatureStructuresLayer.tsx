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
const CELL_GRID_RADIAL_OFFSET = 0.08;
const CELL_GRID_OPACITY = 0.82;
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

function buildCellGridGeometry(geometry: THREE.BufferGeometry) {
  const positionAttribute = geometry.getAttribute("position");
  const colorAttribute = geometry.getAttribute("color");
  const indexAttribute = geometry.getIndex();
  if (
    !(positionAttribute instanceof THREE.BufferAttribute) ||
    !(colorAttribute instanceof THREE.BufferAttribute) ||
    !(indexAttribute instanceof THREE.BufferAttribute)
  ) {
    return new THREE.BufferGeometry();
  }

  const indices = indexAttribute.array;
  const linePositions: number[] = [];
  const lineColors: number[] = [];
  const seenEdges = new Set<string>();
  const edgeColor = new THREE.Color();
  const colorA = new THREE.Color();
  const colorB = new THREE.Color();
  const edgeHighlight = new THREE.Color(1, 1, 1);
  const liftedA = new THREE.Vector3();
  const liftedB = new THREE.Vector3();

  const appendEdge = (startIndex: number, endIndex: number) => {
    if (startIndex === endIndex) return;
    const key =
      startIndex < endIndex
        ? `${startIndex}:${endIndex}`
        : `${endIndex}:${startIndex}`;
    if (seenEdges.has(key)) return;
    seenEdges.add(key);

    liftedA.fromBufferAttribute(positionAttribute, startIndex);
    liftedB.fromBufferAttribute(positionAttribute, endIndex);
    const radiusA = Math.max(liftedA.length(), 1e-6);
    const radiusB = Math.max(liftedB.length(), 1e-6);
    liftedA.multiplyScalar((radiusA + CELL_GRID_RADIAL_OFFSET) / radiusA);
    liftedB.multiplyScalar((radiusB + CELL_GRID_RADIAL_OFFSET) / radiusB);

    linePositions.push(
      liftedA.x,
      liftedA.y,
      liftedA.z,
      liftedB.x,
      liftedB.y,
      liftedB.z
    );

    colorA.fromBufferAttribute(colorAttribute, startIndex);
    colorB.fromBufferAttribute(colorAttribute, endIndex);
    edgeColor.copy(colorA).lerp(colorB, 0.5).lerp(edgeHighlight, 0.38);
    lineColors.push(
      edgeColor.r,
      edgeColor.g,
      edgeColor.b,
      edgeColor.r,
      edgeColor.g,
      edgeColor.b
    );
  };

  for (let index = 0; index + 5 < indices.length; index += 6) {
    const corner0 = Number(indices[index]);
    const tri1b = Number(indices[index + 1]);
    const tri1c = Number(indices[index + 2]);
    const corner0Repeat = Number(indices[index + 3]);
    const tri2b = Number(indices[index + 4]);
    const tri2c = Number(indices[index + 5]);
    if (corner0 !== corner0Repeat) continue;

    let corner1 = -1;
    let corner2 = -1;
    let corner3 = -1;

    if (tri1c === tri2b) {
      corner1 = tri1b;
      corner2 = tri1c;
      corner3 = tri2c;
    } else if (tri1b === tri2c) {
      corner1 = tri1c;
      corner2 = tri1b;
      corner3 = tri2b;
    } else {
      continue;
    }

    appendEdge(corner0, corner1);
    appendEdge(corner1, corner2);
    appendEdge(corner2, corner3);
    appendEdge(corner3, corner0);
  }

  const lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(linePositions), 3)
  );
  lineGeometry.setAttribute(
    "color",
    new THREE.BufferAttribute(new Float32Array(lineColors), 3)
  );
  lineGeometry.computeBoundingSphere();
  return lineGeometry;
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
  const warmGridRef = useRef<THREE.LineSegments | null>(null);
  const coldGridRef = useRef<THREE.LineSegments | null>(null);
  const materialRef = useRef<THREE.MeshLambertMaterial | null>(null);
  const gridMaterialRef = useRef<THREE.LineBasicMaterial | null>(null);
  const frameRef = useRef<PotentialTemperatureStructureFrame | null>(null);
  const reqIdRef = useRef(0);

  const rebuildMesh = useCallback(() => {
    const root = rootRef.current;
    const material = materialRef.current;
    const gridMaterial = gridMaterialRef.current;
    const frame = frameRef.current;
    if (!root || !material || !gridMaterial || !frame) return;
    root.userData.variant = frame.manifest.variant ?? layerState.variant;

    warmMeshRef.current?.removeFromParent();
    warmMeshRef.current?.geometry.dispose();
    coldMeshRef.current?.removeFromParent();
    coldMeshRef.current?.geometry.dispose();
    warmGridRef.current?.removeFromParent();
    warmGridRef.current?.geometry.dispose();
    coldGridRef.current?.removeFromParent();
    coldGridRef.current?.geometry.dispose();

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

    if (layerState.showCellGrid && warmGeometry.index && warmGeometry.index.count > 0) {
      const warmGrid = new THREE.LineSegments(
        buildCellGridGeometry(warmGeometry),
        gridMaterial
      );
      warmGrid.name = "potential-temperature-warm-cell-grid";
      warmGrid.renderOrder = LAYER_RENDER_ORDER + 1;
      warmGrid.frustumCulled = false;
      root.add(warmGrid);
      warmGridRef.current = warmGrid;
    } else {
      warmGridRef.current = null;
    }

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

    if (layerState.showCellGrid && coldGeometry.index && coldGeometry.index.count > 0) {
      const coldGrid = new THREE.LineSegments(
        buildCellGridGeometry(coldGeometry),
        gridMaterial
      );
      coldGrid.name = "potential-temperature-cold-cell-grid";
      coldGrid.renderOrder = LAYER_RENDER_ORDER + 1;
      coldGrid.frustumCulled = false;
      root.add(coldGrid);
      coldGridRef.current = coldGrid;
    } else {
      coldGridRef.current = null;
    }
  }, [layerState.colorMode, layerState.showCellGrid, layerState.variant, verticalExaggeration]);

  useEffect(() => {
    if (!engineReady) return;
    if (!sceneRef.current || !globeRef.current) return;

    const root = new THREE.Group();
    root.name = "potential-temperature-structures-root";
    root.visible = false;
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

    const gridMaterial = new THREE.LineBasicMaterial({
      color: new THREE.Color("#eef4ff"),
      transparent: true,
      opacity: CELL_GRID_OPACITY,
      depthTest: true,
      depthWrite: false,
    });
    gridMaterialRef.current = gridMaterial;

    return () => {
      warmMeshRef.current?.geometry.dispose();
      coldMeshRef.current?.geometry.dispose();
      warmGridRef.current?.geometry.dispose();
      coldGridRef.current?.geometry.dispose();
      material.dispose();
      gridMaterial.dispose();
      materialRef.current = null;
      gridMaterialRef.current = null;
      warmMeshRef.current = null;
      coldMeshRef.current = null;
      warmGridRef.current = null;
      coldGridRef.current = null;
      rootRef.current = null;
      root.removeFromParent();
    };
  }, [engineReady, globeRef, sceneRef]);

  useEffect(() => {
    if (!engineReady) return;
    const root = rootRef.current;
    const material = materialRef.current;
    const gridMaterial = gridMaterialRef.current;
    if (!root || !material || !gridMaterial) return;

    root.visible = layerState.visible;
    material.transparent = layerState.opacity < 0.999;
    material.opacity = layerState.opacity;
    material.depthWrite = layerState.opacity >= 0.999;
    material.side = THREE.FrontSide;
    gridMaterial.opacity = Math.max(layerState.opacity * CELL_GRID_OPACITY, 0.12);
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
