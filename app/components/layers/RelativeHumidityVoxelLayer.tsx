import { useCallback, useEffect, useRef } from "react";
import * as THREE from "three";
import { useEarthLayer } from "./EarthBase";
import { useControls, type RelativeHumidityColorMode } from "../../state/controlsStore";
import {
  fetchRelativeHumidityShellFrame,
  type RelativeHumidityShellFrame,
} from "../utils/relativeHumidityShellAssets";

const LAYER_CLEARANCE = 10.4;
const LAYER_RENDER_ORDER = 66;
const PRESSURE_COLOR_BANDS = [
  { minHpa: 850, color: new THREE.Color("#ff7a59") },
  { minHpa: 700, color: new THREE.Color("#ffb347") },
  { minHpa: 500, color: new THREE.Color("#d7e95b") },
  { minHpa: 350, color: new THREE.Color("#4fd6a7") },
  { minHpa: 250, color: new THREE.Color("#39b8ff") },
  { minHpa: 150, color: new THREE.Color("#4e7dff") },
  { minHpa: 70, color: new THREE.Color("#7a68ff") },
  { minHpa: 0, color: new THREE.Color("#b06bff") },
] as const;
const SOLID_COLOR = new THREE.Color("#8feeff");

function pressureToStandardAtmosphereHeightM(pressureHpa: number) {
  const safePressure = Math.max(pressureHpa, 1);
  return 44330.0 * (1.0 - (safePressure / 1013.25) ** 0.1903);
}

function pressureToRadius(frame: RelativeHumidityShellFrame, pressureHpa: number) {
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

function colorForPressureBand(pressureHpa: number) {
  for (const band of PRESSURE_COLOR_BANDS) {
    if (pressureHpa >= band.minHpa) {
      return band.color.clone();
    }
  }

  return PRESSURE_COLOR_BANDS[PRESSURE_COLOR_BANDS.length - 1].color.clone();
}

function buildBandScale(frame: RelativeHumidityShellFrame) {
  const radii = frame.manifest.thresholds.map((entry) =>
    pressureToRadius(frame, entry.pressure_hpa)
  );
  const boundaryRadii = radii
    .slice(0, -1)
    .map((radius, index) => (radius + radii[index + 1]) / 2);
  const levelColors = frame.manifest.thresholds.map((entry) =>
    colorForPressureBand(entry.pressure_hpa)
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
  frame: RelativeHumidityShellFrame,
  verticalExaggeration: number,
  colorMode: RelativeHumidityColorMode
) {
  const bandScale = buildBandScale(frame);
  const positions = frame.positions.slice();
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
    const color =
      colorMode === "solidAqua"
        ? SOLID_COLOR
        : bandScale.levelColors[levelIndex] ?? SOLID_COLOR;
    colors[i] = color.r;
    colors[i + 1] = color.g;
    colors[i + 2] = color.b;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(new THREE.BufferAttribute(frame.indices, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

export default function RelativeHumidityVoxelLayer() {
  const rhLayer = useControls((state) => state.relativeHumidityLayer);
  const verticalExaggeration = useControls(
    (state) => state.moistureStructureLayer.verticalExaggeration
  );
  const { engineReady, sceneRef, globeRef, signalReady, timestamp } =
    useEarthLayer("relative-humidity-shell");

  const rootRef = useRef<THREE.Group | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const materialRef = useRef<THREE.MeshLambertMaterial | null>(null);
  const frameRef = useRef<RelativeHumidityShellFrame | null>(null);
  const reqIdRef = useRef(0);

  const rebuildMesh = useCallback(() => {
    const root = rootRef.current;
    const material = materialRef.current;
    const frame = frameRef.current;
    if (!root || !material || !frame) return;

    meshRef.current?.removeFromParent();
    meshRef.current?.geometry.dispose();

    const geometry = buildGeometry(
      frame,
      verticalExaggeration,
      rhLayer.colorMode
    );
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = "relative-humidity-shell";
    mesh.renderOrder = LAYER_RENDER_ORDER;
    mesh.frustumCulled = false;
    root.add(mesh);
    meshRef.current = mesh;
  }, [rhLayer.colorMode, verticalExaggeration]);

  useEffect(() => {
    if (!engineReady) return;
    if (!sceneRef.current || !globeRef.current) return;

    const root = new THREE.Group();
    root.name = "relative-humidity-shell-root";
    root.visible = rhLayer.visible;
    root.renderOrder = LAYER_RENDER_ORDER;
    root.frustumCulled = false;
    sceneRef.current.add(root);
    rootRef.current = root;

    const material = new THREE.MeshLambertMaterial({
      vertexColors: true,
      transparent: true,
      opacity: rhLayer.opacity,
      depthWrite: rhLayer.opacity >= 0.99,
      depthTest: true,
      side: THREE.DoubleSide,
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
  }, [engineReady, globeRef, sceneRef]);

  useEffect(() => {
    if (!engineReady) return;
    const root = rootRef.current;
    const material = materialRef.current;
    if (!root || !material) return;

    root.visible = rhLayer.visible;
    material.opacity = rhLayer.opacity;
    material.depthWrite = rhLayer.opacity >= 0.99;
    if (frameRef.current) {
      rebuildMesh();
    }
  }, [engineReady, rebuildMesh, rhLayer.opacity, rhLayer.visible]);

  useEffect(() => {
    if (!engineReady) return;
    const root = rootRef.current;
    if (!root) return;

    let cancelled = false;
    const requestId = ++reqIdRef.current;
    const isCancelled = () => cancelled || requestId !== reqIdRef.current;

    if (!rhLayer.visible) {
      root.visible = false;
      signalReady(timestamp);
      return () => {
        cancelled = true;
      };
    }

    root.visible = true;

    void fetchRelativeHumidityShellFrame(timestamp)
      .then((frame) => {
        if (isCancelled()) return;
        frameRef.current = frame;
        rebuildMesh();
        signalReady(timestamp);
      })
      .catch((error) => {
        if (isCancelled()) return;
        console.error("Failed to load relative humidity shell layer", error);
        signalReady(timestamp);
      });

    return () => {
      cancelled = true;
    };
  }, [engineReady, rebuildMesh, rhLayer.visible, signalReady, timestamp]);

  return null;
}
