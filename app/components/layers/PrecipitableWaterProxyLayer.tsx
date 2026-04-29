import { useCallback, useEffect, useRef } from "react";
import * as THREE from "three";
import { type EarthProjectionMode, useEarthLayer } from "./EarthBase";
import { useControls } from "../../state/controlsStore";
import {
  fetchPrecipitableWaterProxyFrame,
  type PrecipitableWaterProxyFrame,
} from "../utils/precipitableWaterProxyAssets";
import {
  flatMapFaceStaysWithinSeam,
  globeVec3ToLatLon,
  latLonHeightToFlatMapVec3,
} from "../utils/EarthUtils";

const LAYER_CLEARANCE = 10.8;
const LAYER_RENDER_ORDER = 67;
const PROXY_COLOR_STOPS = [
  new THREE.Color("#ff8a63"),
  new THREE.Color("#2dc6d6"),
  new THREE.Color("#5e86ff"),
  new THREE.Color("#b95cff"),
] as const;

function pressureToStandardAtmosphereHeightM(pressureHpa: number) {
  const safePressure = Math.max(pressureHpa, 1);
  return 44330.0 * (1.0 - (safePressure / 1013.25) ** 0.1903);
}

function pressureToRadius(
  frame: PrecipitableWaterProxyFrame,
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

function colorAtStopPosition(t: number, stops: readonly THREE.Color[]) {
  const scaled = THREE.MathUtils.clamp(t, 0, 1) * (stops.length - 1);
  const startIndex = Math.floor(scaled);
  const endIndex = Math.min(startIndex + 1, stops.length - 1);
  const mix = scaled - startIndex;
  return stops[startIndex].clone().lerp(stops[endIndex], mix);
}

function buildBandScale(frame: PrecipitableWaterProxyFrame) {
  const activeThresholds = frame.manifest.thresholds.filter(
    (entry) => entry.active_pressure_window
  );
  const radii = activeThresholds.map((entry) =>
    pressureToRadius(frame, entry.pressure_hpa)
  );
  const boundaryRadii = radii
    .slice(0, -1)
    .map((radius, index) => (radius + radii[index + 1]) / 2);
  const levelColors = radii.map((_, index) =>
    colorAtStopPosition(
      index / Math.max(radii.length - 1, 1),
      PROXY_COLOR_STOPS
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
  frame: PrecipitableWaterProxyFrame,
  verticalExaggeration: number,
  projectionMode: EarthProjectionMode
) {
  const bandScale = buildBandScale(frame);
  const positions = frame.positions.slice();
  const indices = frame.indices.slice();
  const colors = new Float32Array((positions.length / 3) * 3);
  const baseRadius = frame.manifest.globe.base_radius;
  const globePosition = new THREE.Vector3();

  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    const radius = Math.hypot(x, y, z);
    if (radius <= 1e-6) continue;

    const radialOffset = Math.max(radius - baseRadius, 0);
    if (projectionMode === "flat2d") {
      // QUICK AND DIRTY NEED TO REDO FLAT PROJECTION WITH SHARED PROJECTOR/2D
      // ASSETS: unwrap existing globe vertices into lon/lat map coordinates.
      globePosition.set(x, y, z);
      const { lat, lon } = globeVec3ToLatLon(globePosition);
      const flatPosition = latLonHeightToFlatMapVec3(
        lat,
        lon,
        LAYER_CLEARANCE + radialOffset * verticalExaggeration
      );
      positions[i] = flatPosition.x;
      positions[i + 1] = flatPosition.y;
      positions[i + 2] = flatPosition.z;
    } else {
      const exaggeratedRadius =
        baseRadius + LAYER_CLEARANCE + radialOffset * verticalExaggeration;
      const scale = exaggeratedRadius / radius;

      positions[i] *= scale;
      positions[i + 1] *= scale;
      positions[i + 2] *= scale;
    }

    const levelIndex = levelIndexForRadius(radius, bandScale.boundaryRadii);
    const color =
      bandScale.levelColors[levelIndex] ??
      bandScale.levelColors[bandScale.levelColors.length - 1] ??
      PROXY_COLOR_STOPS[0];
    colors[i] = color.r;
    colors[i + 1] = color.g;
    colors[i + 2] = color.b;
  }

  if (frame.manifest.version < 2) {
    for (let i = 0; i < indices.length; i += 3) {
      const second = indices[i + 1];
      indices[i + 1] = indices[i + 2];
      indices[i + 2] = second;
    }
  }

  let geometryIndices = indices;
  if (projectionMode === "flat2d") {
    const filteredIndices: number[] = [];
    for (let i = 0; i + 2 < indices.length; i += 3) {
      const faceIndices = [
        Number(indices[i]),
        Number(indices[i + 1]),
        Number(indices[i + 2]),
      ];
      if (flatMapFaceStaysWithinSeam(positions, faceIndices)) {
        filteredIndices.push(...faceIndices);
      }
    }
    geometryIndices = new Uint32Array(filteredIndices);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(new THREE.BufferAttribute(geometryIndices, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

export default function PrecipitableWaterProxyLayer() {
  const layerState = useControls((state) => state.precipitableWaterLayer);
  const verticalExaggeration = useControls((state) => state.verticalExaggeration);
  const { engineReady, sceneRef, globeRef, projectionMode, signalReady, timestamp } =
    useEarthLayer("precipitable-water-proxy");

  const rootRef = useRef<THREE.Group | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const materialRef = useRef<THREE.MeshLambertMaterial | null>(null);
  const frameRef = useRef<PrecipitableWaterProxyFrame | null>(null);
  const reqIdRef = useRef(0);

  const rebuildMesh = useCallback(() => {
    const root = rootRef.current;
    const material = materialRef.current;
    const frame = frameRef.current;
    if (!root || !material || !frame) return;

    meshRef.current?.removeFromParent();
    meshRef.current?.geometry.dispose();

    const geometry = buildGeometry(frame, verticalExaggeration, projectionMode);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = "precipitable-water-proxy";
    mesh.renderOrder = LAYER_RENDER_ORDER;
    mesh.frustumCulled = false;
    root.add(mesh);
    meshRef.current = mesh;
  }, [projectionMode, verticalExaggeration]);

  useEffect(() => {
    if (!engineReady) return;
    if (!sceneRef.current) return;
    if (projectionMode === "globe" && !globeRef.current) return;

    const root = new THREE.Group();
    root.name = "precipitable-water-proxy-root";
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
      side: projectionMode === "flat2d" ? THREE.DoubleSide : THREE.FrontSide,
      flatShading: true,
    });
    materialRef.current = material;

    return () => {
      meshRef.current?.geometry.dispose();
      material.dispose();
      materialRef.current = null;
      meshRef.current = null;
      rootRef.current = null;
      root.removeFromParent();
    };
  }, [engineReady, globeRef, layerState.visible, projectionMode, sceneRef]);

  useEffect(() => {
    if (!engineReady) return;
    const root = rootRef.current;
    const material = materialRef.current;
    if (!root || !material) return;

    root.visible = layerState.visible;
    material.opacity = 1;
    material.depthWrite = true;
    material.side = projectionMode === "flat2d" ? THREE.DoubleSide : THREE.FrontSide;
    if (frameRef.current) {
      rebuildMesh();
    }
  }, [
    engineReady,
    projectionMode,
    rebuildMesh,
    layerState.opacity,
    layerState.visible,
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

    void fetchPrecipitableWaterProxyFrame(timestamp)
      .then((frame) => {
        if (isCancelled()) return;
        frameRef.current = frame;
        rebuildMesh();
        signalReady(timestamp);
      })
      .catch(() => {
        if (isCancelled()) return;
        signalReady(timestamp);
      });

    return () => {
      cancelled = true;
    };
  }, [engineReady, layerState.visible, rebuildMesh, signalReady, timestamp]);

  return null;
}
