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

const CLASS_COLOR_STOPS: Record<
  AirMassStructureClassKey,
  readonly THREE.Color[]
> = {
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
      CLASS_COLOR_STOPS[classKey]
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
      bandScale.levelColors[levelIndex] ??
      colorAtStopPosition(
        1 - THREE.MathUtils.clamp((pressureHpa - 250) / 750, 0, 1),
        CLASS_COLOR_STOPS[classKey]
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

export default function AirMassClassificationLayer() {
  const layerState = useControls((state) => state.airMassLayer);
  const verticalExaggeration = useControls(
    (state) => state.moistureStructureLayer.verticalExaggeration
  );
  const { engineReady, sceneRef, globeRef, signalReady, timestamp } =
    useEarthLayer("air-mass-classification");

  const rootRef = useRef<THREE.Group | null>(null);
  const materialRef = useRef<THREE.MeshLambertMaterial | null>(null);
  const meshRefs = useRef<Partial<Record<AirMassStructureClassKey, THREE.Mesh | null>>>(
    {}
  );
  const frameRef = useRef<AirMassStructureFrame | null>(null);
  const reqIdRef = useRef(0);

  const clearMeshes = useCallback(() => {
    for (const classKey of AIR_MASS_CLASS_ORDER) {
      const mesh = meshRefs.current[classKey];
      if (!mesh) continue;
      mesh.removeFromParent();
      mesh.geometry.dispose();
      meshRefs.current[classKey] = null;
    }
  }, []);

  const rebuildMesh = useCallback(() => {
    const root = rootRef.current;
    const material = materialRef.current;
    const frame = frameRef.current;
    if (!root || !material || !frame) return;

    root.userData.variant = frame.manifest.variant;
    clearMeshes();

    for (const classKey of AIR_MASS_CLASS_ORDER) {
      const classBuffer = frame.classBuffers[classKey];
      if (classBuffer.positions.length === 0 || classBuffer.indices.length === 0) {
        meshRefs.current[classKey] = null;
        continue;
      }

      const mesh = new THREE.Mesh(
        buildGeometry(frame, verticalExaggeration, classKey),
        material
      );
      mesh.name = `air-mass-${classKey}-shell`;
      mesh.renderOrder = LAYER_RENDER_ORDER;
      mesh.frustumCulled = false;
      root.add(mesh);
      meshRefs.current[classKey] = mesh;
    }
  }, [clearMeshes, verticalExaggeration]);

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

    return () => {
      clearMeshes();
      material.dispose();
      materialRef.current = null;
      rootRef.current = null;
      root.removeFromParent();
    };
  }, [clearMeshes, engineReady, globeRef, sceneRef]);

  useEffect(() => {
    if (!engineReady) return;
    const root = rootRef.current;
    const material = materialRef.current;
    if (!root || !material) return;

    root.visible = layerState.visible;
    material.transparent = layerState.opacity < 0.999;
    material.opacity = layerState.opacity;
    material.depthWrite = layerState.opacity >= 0.999;
  }, [engineReady, layerState.opacity, layerState.visible]);

  useEffect(() => {
    if (!engineReady || !frameRef.current || !layerState.visible) return;
    rebuildMesh();
  }, [engineReady, layerState.visible, rebuildMesh, verticalExaggeration]);

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
