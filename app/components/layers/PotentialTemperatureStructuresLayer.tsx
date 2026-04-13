import { useCallback, useEffect, useRef } from "react";
import * as THREE from "three";
import { useEarthLayer } from "./EarthBase";
import { useControls } from "../../state/controlsStore";
import {
  fetchPotentialTemperatureStructureFrame,
  type PotentialTemperatureStructureFrame,
} from "../utils/potentialTemperatureStructureAssets";

const LAYER_CLEARANCE = 10.8;
const LAYER_RENDER_ORDER = 68;
const COLDNESS_COLOR_STOPS = [
  new THREE.Color("#dff4ff"),
  new THREE.Color("#7dc4ff"),
  new THREE.Color("#3f7ff7"),
  new THREE.Color("#172d88"),
] as const;

function colorAtStopPosition(t: number, stops: readonly THREE.Color[]) {
  const scaled = THREE.MathUtils.clamp(t, 0, 1) * (stops.length - 1);
  const startIndex = Math.floor(scaled);
  const endIndex = Math.min(startIndex + 1, stops.length - 1);
  const mix = scaled - startIndex;
  return stops[startIndex].clone().lerp(stops[endIndex], mix);
}

function buildGeometry(
  frame: PotentialTemperatureStructureFrame,
  verticalExaggeration: number
) {
  const positions = frame.positions.slice();
  const indices = frame.indices.slice();
  const coldnessSigma = frame.coldnessSigma;
  const colors = new Float32Array((positions.length / 3) * 3);
  const baseRadius = frame.manifest.globe.base_radius;
  const threshold = frame.manifest.selection.z_threshold_sigma;
  const maxColdness = Math.max(frame.metadata.coldness_sigma_max, threshold + 1e-6);
  const coldnessRange = Math.max(maxColdness - threshold, 1e-6);

  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    const radius = Math.hypot(x, y, z);
    if (radius > 1e-6) {
      const radialOffset = Math.max(radius - baseRadius, 0);
      const exaggeratedRadius =
        baseRadius + LAYER_CLEARANCE + radialOffset * verticalExaggeration;
      const scale = exaggeratedRadius / radius;
      positions[i] *= scale;
      positions[i + 1] *= scale;
      positions[i + 2] *= scale;
    }

    const coldnessValue = coldnessSigma[i / 3] ?? threshold;
    const normalized = (coldnessValue - threshold) / coldnessRange;
    const color = colorAtStopPosition(normalized, COLDNESS_COLOR_STOPS);
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
  const meshRef = useRef<THREE.Mesh | null>(null);
  const materialRef = useRef<THREE.MeshLambertMaterial | null>(null);
  const frameRef = useRef<PotentialTemperatureStructureFrame | null>(null);
  const reqIdRef = useRef(0);

  const rebuildMesh = useCallback(() => {
    const root = rootRef.current;
    const material = materialRef.current;
    const frame = frameRef.current;
    if (!root || !material || !frame) return;

    meshRef.current?.removeFromParent();
    meshRef.current?.geometry.dispose();

    const geometry = buildGeometry(frame, verticalExaggeration);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = "potential-temperature-cold-anomaly-shell";
    mesh.renderOrder = LAYER_RENDER_ORDER;
    mesh.frustumCulled = false;
    root.add(mesh);
    meshRef.current = mesh;
  }, [verticalExaggeration]);

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
      transparent: true,
      opacity: 1,
      depthWrite: true,
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
  }, [engineReady, globeRef, layerState.visible, sceneRef]);

  useEffect(() => {
    if (!engineReady) return;
    const root = rootRef.current;
    const material = materialRef.current;
    if (!root || !material) return;

    root.visible = layerState.visible;
    material.opacity = layerState.opacity;
    material.depthWrite = layerState.opacity >= 0.999;
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

    void fetchPotentialTemperatureStructureFrame(timestamp)
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
  }, [engineReady, layerState.visible, rebuildMesh, signalReady, timestamp]);

  return null;
}
