import { useEffect, useRef } from "react";
import * as THREE from "three";
import { useEarthLayer } from "./EarthBase";
import { useControls } from "../../state/controlsStore";
import {
  fetchMoistureStructureFrame,
  type MoistureStructureComponentMetadata,
  type MoistureStructureFrame,
} from "../utils/ApiResponses";

type MoistureLayerStyle = ReturnType<
  typeof useControls.getState
>["moistureStructureLayer"];

type MoistureSlice = {
  group: THREE.Group;
  materials: THREE.MeshPhongMaterial[];
  opacityWeights: number[];
};

const MAGENTA = new THREE.Color("#d64df6");
const CORAL = new THREE.Color("#ff7b63");

function animateT(
  ms: number,
  isCancelled: () => boolean,
  onUpdate: (t: number) => void,
  onDone?: () => void
) {
  const start = performance.now();
  function step(now: number) {
    if (isCancelled()) return;
    const t = Math.min(1, (now - start) / Math.max(ms, 1));
    onUpdate(t);
    if (t < 1) requestAnimationFrame(step);
    else onDone?.();
  }
  requestAnimationFrame(step);
}

function disposeSlice(slice: MoistureSlice | null) {
  if (!slice) return;
  slice.group.traverse((object) => {
    if (object instanceof THREE.Mesh) {
      object.geometry.dispose();
    }
  });
  for (const material of slice.materials) {
    material.dispose();
  }
  slice.group.removeFromParent();
}

function componentColor(component: MoistureStructureComponentMetadata) {
  const midpoint = (component.pressure_min_hpa + component.pressure_max_hpa) / 2;
  const t = THREE.MathUtils.clamp((midpoint - 150) / (1000 - 150), 0, 1);
  return MAGENTA.clone().lerp(CORAL, t);
}

function buildComponentOpacityWeights(frame: MoistureStructureFrame) {
  const maxVoxelCount = Math.max(
    ...frame.metadata.components.map((component) => component.voxel_count),
    1
  );
  const maxSpecificHumidity = Math.max(
    ...frame.metadata.components.map((component) => component.max_specific_humidity),
    1e-6
  );

  return frame.metadata.components.map((component) => {
    const voxelWeight = Math.sqrt(component.voxel_count / maxVoxelCount);
    const humidityWeight = Math.min(
      1,
      component.max_specific_humidity / maxSpecificHumidity
    );
    return THREE.MathUtils.lerp(0.42, 1.0, voxelWeight * 0.6 + humidityWeight * 0.4);
  });
}

function buildComponentGeometry(
  positions: Float32Array,
  indices: Uint32Array,
  component: MoistureStructureComponentMetadata
) {
  const positionStart = component.vertex_offset * 3;
  const positionEnd = positionStart + component.vertex_count * 3;
  const indexStart = component.index_offset;

  const localPositions = positions.slice(positionStart, positionEnd);
  const localIndices = new Uint32Array(component.index_count);

  for (let i = 0; i < component.index_count; i += 1) {
    localIndices[i] = indices[indexStart + i] - component.vertex_offset;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(localPositions, 3));
  geometry.setIndex(new THREE.BufferAttribute(localIndices, 1));
  geometry.computeVertexNormals();
  return geometry;
}

function buildSlice(frame: MoistureStructureFrame, opacity: number): MoistureSlice {
  const group = new THREE.Group();
  group.name = "moisture-structures-slice";
  group.renderOrder = 64;
  group.frustumCulled = false;

  const materials: THREE.MeshPhongMaterial[] = [];
  const allOpacityWeights = buildComponentOpacityWeights(frame);
  const opacityWeights: number[] = [];

  frame.metadata.components.forEach((component, index) => {
    if (component.vertex_count <= 0 || component.index_count < 3) return;

    const geometry = buildComponentGeometry(frame.positions, frame.indices, component);
    const color = componentColor(component);
    const opacityWeight = allOpacityWeights[index] ?? 1;
    const material = new THREE.MeshPhongMaterial({
      color,
      emissive: color.clone().multiplyScalar(0.16),
      specular: new THREE.Color("#f7d2ff"),
      shininess: 44,
      transparent: true,
      opacity: opacity * opacityWeight,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = `moisture-component-${component.id}`;
    mesh.renderOrder = 64;
    mesh.frustumCulled = false;
    group.add(mesh);
    materials.push(material);
    opacityWeights.push(opacityWeight);
  });

  return { group, materials, opacityWeights };
}

function setSliceOpacity(slice: MoistureSlice | null, opacity: number) {
  if (!slice) return;
  for (let i = 0; i < slice.materials.length; i += 1) {
    slice.materials[i].opacity = opacity * (slice.opacityWeights[i] ?? 1);
  }
}

export default function MoistureStructureLayer() {
  const moistureLayer = useControls((state) => state.moistureStructureLayer);
  const visible = moistureLayer.visible;

  const { engineReady, sceneRef, timestamp, signalReady } =
    useEarthLayer("moisture-structures");

  const rootRef = useRef<THREE.Group | null>(null);
  const currentRef = useRef<MoistureSlice | null>(null);
  const transitionRef = useRef<MoistureSlice | null>(null);
  const styleRef = useRef<MoistureLayerStyle>(moistureLayer);
  const fadeMixRef = useRef<number | null>(null);
  const reqIdRef = useRef(0);

  const applyVisibleOpacity = (targetOpacity: number) => {
    const mix = fadeMixRef.current;
    const current = currentRef.current;
    const transition = transitionRef.current;

    if (mix === null || !transition) {
      setSliceOpacity(current, targetOpacity);
      return;
    }

    setSliceOpacity(current, targetOpacity * (1 - mix));
    setSliceOpacity(transition, targetOpacity * mix);
  };

  useEffect(() => {
    if (!engineReady || !sceneRef.current) return;

    const root = new THREE.Group();
    const state = useControls.getState().moistureStructureLayer;
    root.name = "moisture-structures-root";
    root.visible = state.visible;
    root.renderOrder = 64;
    root.frustumCulled = false;
    styleRef.current = state;

    sceneRef.current.add(root);
    rootRef.current = root;

    const unsubscribe = useControls.subscribe(
      (state) => state.moistureStructureLayer,
      (next) => {
        styleRef.current = next;
        root.visible = next.visible;
        applyVisibleOpacity(next.opacity);
      }
    );

    return () => {
      unsubscribe();
      disposeSlice(transitionRef.current);
      transitionRef.current = null;
      fadeMixRef.current = null;
      disposeSlice(currentRef.current);
      currentRef.current = null;
      rootRef.current = null;
      root.removeFromParent();
    };
  }, [engineReady, sceneRef]);

  useEffect(() => {
    if (!engineReady) return;
    const root = rootRef.current;
    if (!root) return;

    let cancelled = false;
    const requestId = ++reqIdRef.current;
    const isCancelled = () => cancelled || requestId !== reqIdRef.current;

    if (!visible) {
      disposeSlice(transitionRef.current);
      transitionRef.current = null;
      fadeMixRef.current = null;
      disposeSlice(currentRef.current);
      currentRef.current = null;
      root.visible = false;
      signalReady(timestamp);
      return () => {
        cancelled = true;
      };
    }

    root.visible = true;

    void (async () => {
      try {
        const style = styleRef.current;
        const frame = await fetchMoistureStructureFrame(timestamp);
        if (isCancelled()) return;

        const next = buildSlice(frame, 0.0);
        root.add(next.group);
        transitionRef.current = next;

        const prev = currentRef.current;
        fadeMixRef.current = 0;
        applyVisibleOpacity(style.opacity);

        animateT(
          260,
          isCancelled,
          (t) => {
            fadeMixRef.current = t;
            applyVisibleOpacity(styleRef.current.opacity);
          },
          () => {
            if (isCancelled()) {
              if (transitionRef.current === next) transitionRef.current = null;
              fadeMixRef.current = null;
              disposeSlice(next);
              return;
            }

            disposeSlice(prev);
            currentRef.current = next;
            if (transitionRef.current === next) transitionRef.current = null;
            fadeMixRef.current = null;
            applyVisibleOpacity(styleRef.current.opacity);
          }
        );

        signalReady(timestamp);
      } catch (error) {
        if (isCancelled()) return;
        console.error("Failed to load moisture structures", error);
        signalReady(timestamp);
      }
    })();

    return () => {
      cancelled = true;
      if (transitionRef.current) {
        disposeSlice(transitionRef.current);
        transitionRef.current = null;
      }
      fadeMixRef.current = null;
    };
  }, [engineReady, signalReady, timestamp, visible]);

  return null;
}
