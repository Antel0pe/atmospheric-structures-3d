import { useCallback, useEffect, useRef } from "react";
import * as THREE from "three";
import { useEarthLayer } from "./EarthBase";
import { useControls } from "../../state/controlsStore";
import {
  AIR_MASS_CLASS_ORDER,
  type AirMassStructureClassKey,
  type AirMassStructureFrame,
  fetchAirMassStructureFrame,
} from "../utils/airMassStructureAssets";

const LAYER_CLEARANCE = 11.1;
const LAYER_RENDER_ORDER = 69;
const CELL_GRID_RADIAL_OFFSET = 0.08;
const CELL_GRID_OPACITY = 0.82;

const CLASS_COLOR_STOPS: Partial<Record<
  AirMassStructureClassKey,
  readonly THREE.Color[]
>> = {
  warm_dry: [
    new THREE.Color("#7c2e18"),
    new THREE.Color("#d36a2f"),
    new THREE.Color("#ffb44d"),
    new THREE.Color("#ffe49c"),
  ],
  warm_moist: [
    new THREE.Color("#7f2531"),
    new THREE.Color("#c2534d"),
    new THREE.Color("#ff8f6b"),
    new THREE.Color("#ffd3b3"),
  ],
  cold_dry: [
    new THREE.Color("#11295f"),
    new THREE.Color("#2252a8"),
    new THREE.Color("#5a9ef0"),
    new THREE.Color("#d5ebff"),
  ],
  cold_moist: [
    new THREE.Color("#0f555e"),
    new THREE.Color("#188f99"),
    new THREE.Color("#4fd2c3"),
    new THREE.Color("#dcfff6"),
  ],
};

const BUCKET_CLASS_COLORS: Record<string, THREE.Color> = {
  bucket_0: new THREE.Color("#08306b"),
  bucket_1: new THREE.Color("#2171b5"),
  bucket_2: new THREE.Color("#6baed6"),
  bucket_3: new THREE.Color("#c6dbef"),
  bucket_4: new THREE.Color("#f7fbff"),
  bucket_5: new THREE.Color("#fff5f0"),
  bucket_6: new THREE.Color("#fcbba1"),
  bucket_7: new THREE.Color("#fb6a4a"),
  bucket_8: new THREE.Color("#cb181d"),
  bucket_9: new THREE.Color("#67000d"),
};

function pressureToStandardAtmosphereHeightM(pressureHpa: number) {
  const safePressure = Math.max(pressureHpa, 1);
  return 44330.0 * (1.0 - (safePressure / 1013.25) ** 0.1903);
}

function standardAtmosphereHeightMToPressure(heightM: number) {
  const normalized = THREE.MathUtils.clamp(1.0 - heightM / 44330.0, 1e-6, 1);
  return 1013.25 * normalized ** (1 / 0.1903);
}

function pressureToRadius(frame: AirMassStructureFrame, pressureHpa: number) {
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

function radiusToPressureHpa(frame: AirMassStructureFrame, radius: number) {
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

function buildBandScale(frame: AirMassStructureFrame, classKey: AirMassStructureClassKey) {
  const stops = CLASS_COLOR_STOPS[classKey];
  if (!stops) {
    return { boundaryRadii: [] as number[], levelColors: [] as THREE.Color[] };
  }
  const pressures = frame.metadata.pressure_levels_hpa;
  if (pressures.length === 0) {
    return { boundaryRadii: [] as number[], levelColors: [] as THREE.Color[] };
  }
  const radii = pressures.map((pressure) => pressureToRadius(frame, pressure));
  const boundaryRadii = radii
    .slice(0, -1)
    .map((radius, index) => (radius + radii[index + 1]) / 2);
  const levelColors = radii.map((_, index) =>
    colorAtStopPosition(
      index / Math.max(radii.length - 1, 1),
      stops
    )
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
  frame: AirMassStructureFrame,
  verticalExaggeration: number,
  classKey: AirMassStructureClassKey
) {
  const bandScale = buildBandScale(frame, classKey);
  const bucketColor = BUCKET_CLASS_COLORS[classKey] ?? null;
  const fallbackStops = CLASS_COLOR_STOPS[classKey] ?? [
    new THREE.Color("#7f2531"),
    new THREE.Color("#c2534d"),
    new THREE.Color("#ff8f6b"),
    new THREE.Color("#ffd3b3"),
  ];
  const classBuffer = frame.classBuffers[classKey];
  const positions = classBuffer.positions.slice();
  const indices = classBuffer.indices.slice();
  const colors = new Float32Array(positions.length);
  const baseRadius = frame.manifest.globe.base_radius;

  for (let index = 0; index < positions.length; index += 3) {
    const x = positions[index];
    const y = positions[index + 1];
    const z = positions[index + 2];
    const radius = Math.hypot(x, y, z);
    if (radius <= 1e-6) continue;

    const radialOffset = Math.max(radius - baseRadius, 0);
    const exaggeratedRadius =
      baseRadius + LAYER_CLEARANCE + radialOffset * verticalExaggeration;
    const scale = exaggeratedRadius / radius;

    positions[index] *= scale;
    positions[index + 1] *= scale;
    positions[index + 2] *= scale;

    const levelIndex = levelIndexForRadius(radius, bandScale.boundaryRadii);
    const pressureHpa = radiusToPressureHpa(frame, radius);
    const color =
      bucketColor ??
      bandScale.levelColors[levelIndex] ??
      colorAtStopPosition(
        1 - THREE.MathUtils.clamp((pressureHpa - 250) / 750, 0, 1),
        fallbackStops
      );
    colors[index] = color.r;
    colors[index + 1] = color.g;
    colors[index + 2] = color.b;
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

export default function AirMassClassificationLayer() {
  const layerState = useControls((state) => state.airMassLayer);
  const verticalExaggeration = useControls(
    (state) => state.moistureStructureLayer.verticalExaggeration
  );
  const { engineReady, sceneRef, globeRef, signalReady, timestamp } =
    useEarthLayer("air-mass-classification");

  const rootRef = useRef<THREE.Group | null>(null);
  const materialRef = useRef<THREE.MeshLambertMaterial | null>(null);
  const gridMaterialRef = useRef<THREE.LineBasicMaterial | null>(null);
  const meshRefs = useRef<Partial<Record<AirMassStructureClassKey, THREE.Mesh | null>>>(
    {}
  );
  const gridRefs = useRef<
    Partial<Record<AirMassStructureClassKey, THREE.LineSegments | null>>
  >({});
  const frameRef = useRef<AirMassStructureFrame | null>(null);
  const reqIdRef = useRef(0);

  const clearMeshes = useCallback(() => {
    const classKeys = new Set<AirMassStructureClassKey>([
      ...AIR_MASS_CLASS_ORDER,
      ...Object.keys(meshRefs.current),
      ...Object.keys(gridRefs.current),
      ...(frameRef.current?.classKeys ?? []),
    ]);
    for (const classKey of classKeys) {
      const mesh = meshRefs.current[classKey];
      if (mesh) {
        mesh.removeFromParent();
        mesh.geometry.dispose();
        meshRefs.current[classKey] = null;
      }
      const grid = gridRefs.current[classKey];
      if (grid) {
        grid.removeFromParent();
        grid.geometry.dispose();
        gridRefs.current[classKey] = null;
      }
    }
  }, []);

  const rebuildMesh = useCallback(() => {
    const root = rootRef.current;
    const material = materialRef.current;
    const gridMaterial = gridMaterialRef.current;
    const frame = frameRef.current;
    if (!root || !material || !gridMaterial || !frame) return;

    root.userData.variant = frame.manifest.variant;
    clearMeshes();

    for (const classKey of frame.classKeys) {
      const classBuffer = frame.classBuffers[classKey];
      if (classBuffer.positions.length === 0 || classBuffer.indices.length === 0) {
        meshRefs.current[classKey] = null;
        continue;
      }

      const geometry = buildGeometry(frame, verticalExaggeration, classKey);
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `air-mass-${classKey}-shell`;
      mesh.renderOrder = LAYER_RENDER_ORDER;
      mesh.frustumCulled = false;
      root.add(mesh);
      meshRefs.current[classKey] = mesh;

      if (layerState.showCellGrid && geometry.index && geometry.index.count > 0) {
        const grid = new THREE.LineSegments(
          buildCellGridGeometry(geometry),
          gridMaterial
        );
        grid.name = `air-mass-${classKey}-cell-grid`;
        grid.renderOrder = LAYER_RENDER_ORDER + 1;
        grid.frustumCulled = false;
        root.add(grid);
        gridRefs.current[classKey] = grid;
      } else {
        gridRefs.current[classKey] = null;
      }
    }
  }, [clearMeshes, layerState.showCellGrid, verticalExaggeration]);

  useEffect(() => {
    if (!engineReady) return;
    if (!sceneRef.current || !globeRef.current) return;

    const root = new THREE.Group();
    root.name = "air-mass-classification-root";
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
      vertexColors: true,
      transparent: true,
      opacity: CELL_GRID_OPACITY,
      depthTest: true,
      depthWrite: false,
    });
    gridMaterialRef.current = gridMaterial;

    return () => {
      clearMeshes();
      material.dispose();
      gridMaterial.dispose();
      materialRef.current = null;
      gridMaterialRef.current = null;
      rootRef.current = null;
      root.removeFromParent();
    };
  }, [clearMeshes, engineReady, globeRef, sceneRef]);

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
    gridMaterial.opacity = Math.max(layerState.opacity * CELL_GRID_OPACITY, 0.12);
  }, [engineReady, layerState.opacity, layerState.visible]);

  useEffect(() => {
    if (!engineReady || !frameRef.current || !layerState.visible) return;
    rebuildMesh();
  }, [
    engineReady,
    layerState.showCellGrid,
    layerState.visible,
    rebuildMesh,
    verticalExaggeration,
  ]);

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

    void fetchAirMassStructureFrame(timestamp, {
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
        console.error("Failed to load air-mass classification layer", error);
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
