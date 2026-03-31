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

type LevelBandScale = {
  boundaryRadii: number[];
  levelColors: THREE.Color[];
};

const LEVEL_COLOR_STOPS = [
  new THREE.Color("#ff8a63"),
  new THREE.Color("#2dc6d6"),
  new THREE.Color("#5e86ff"),
  new THREE.Color("#b95cff"),
] as const;
const LEVEL_SPECULAR = new THREE.Color("#f6f8ff");
const LEVEL_EMISSIVE = new THREE.Color("#151b28");

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

function pressureToStandardAtmosphereHeightM(pressureHpa: number) {
  const safePressure = Math.max(pressureHpa, 1);
  return 44330.0 * (1.0 - (safePressure / 1013.25) ** 0.1903);
}

function pressureToRadius(frame: MoistureStructureFrame, pressureHpa: number) {
  const {
    base_radius: baseRadius,
    vertical_span: verticalSpan,
    reference_pressure_hpa: { min: minPressure, max: maxPressure },
  } = frame.manifest.globe;

  const minHeight = pressureToStandardAtmosphereHeightM(maxPressure);
  const maxHeight = pressureToStandardAtmosphereHeightM(minPressure);
  const scale = verticalSpan / Math.max(maxHeight - minHeight, 1e-9);

  return (
    baseRadius + (pressureToStandardAtmosphereHeightM(pressureHpa) - minHeight) * scale
  );
}

function colorAtStopPosition(t: number) {
  const scaled = THREE.MathUtils.clamp(t, 0, 1) * (LEVEL_COLOR_STOPS.length - 1);
  const startIndex = Math.floor(scaled);
  const endIndex = Math.min(startIndex + 1, LEVEL_COLOR_STOPS.length - 1);
  const mix = scaled - startIndex;
  return LEVEL_COLOR_STOPS[startIndex].clone().lerp(LEVEL_COLOR_STOPS[endIndex], mix);
}

function buildLevelBandScale(frame: MoistureStructureFrame): LevelBandScale {
  const radii = frame.manifest.thresholds.map((entry) =>
    pressureToRadius(frame, entry.pressure_hpa)
  );
  const boundaryRadii = radii
    .slice(0, -1)
    .map((radius, index) => (radius + radii[index + 1]) / 2);
  const levelColors = radii.map((_, index) =>
    colorAtStopPosition(index / Math.max(radii.length - 1, 1))
  );

  return { boundaryRadii, levelColors };
}

function levelIndexForRadius(radius: number, bandScale: LevelBandScale) {
  let low = 0;
  let high = bandScale.boundaryRadii.length;

  while (low < high) {
    const mid = (low + high) >> 1;
    if (radius > bandScale.boundaryRadii[mid]) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
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
  component: MoistureStructureComponentMetadata,
  baseRadius: number,
  verticalExaggeration: number,
  bandScale: LevelBandScale
) {
  const positionStart = component.vertex_offset * 3;
  const positionEnd = positionStart + component.vertex_count * 3;
  const indexStart = component.index_offset;

  const localPositions = positions.slice(positionStart, positionEnd);
  const localIndices = new Uint32Array(component.index_count);
  const localColors = new Float32Array(component.vertex_count * 3);

  if (verticalExaggeration !== 1) {
    for (let i = 0; i < localPositions.length; i += 3) {
      const x = localPositions[i];
      const y = localPositions[i + 1];
      const z = localPositions[i + 2];
      const radius = Math.hypot(x, y, z);
      if (radius <= 1e-6) continue;

      const levelColor = bandScale.levelColors[levelIndexForRadius(radius, bandScale)];
      localColors[i] = levelColor.r;
      localColors[i + 1] = levelColor.g;
      localColors[i + 2] = levelColor.b;

      const radialOffset = Math.max(radius - baseRadius, 0);
      const exaggeratedRadius = baseRadius + radialOffset * verticalExaggeration;
      const scale = exaggeratedRadius / radius;

      localPositions[i] *= scale;
      localPositions[i + 1] *= scale;
      localPositions[i + 2] *= scale;
    }
  } else {
    for (let i = 0; i < localPositions.length; i += 3) {
      const x = localPositions[i];
      const y = localPositions[i + 1];
      const z = localPositions[i + 2];
      const radius = Math.hypot(x, y, z);
      if (radius <= 1e-6) continue;

      const levelColor = bandScale.levelColors[levelIndexForRadius(radius, bandScale)];
      localColors[i] = levelColor.r;
      localColors[i + 1] = levelColor.g;
      localColors[i + 2] = levelColor.b;
    }
  }

  for (let i = 0; i < component.index_count; i += 1) {
    localIndices[i] = indices[indexStart + i] - component.vertex_offset;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(localPositions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(localColors, 3));
  geometry.setIndex(new THREE.BufferAttribute(localIndices, 1));
  geometry.computeVertexNormals();
  return geometry;
}

function buildSlice(
  frame: MoistureStructureFrame,
  style: MoistureLayerStyle,
  opacity: number
): MoistureSlice {
  const group = new THREE.Group();
  group.name = "moisture-structures-slice";
  group.renderOrder = 64;
  group.frustumCulled = false;

  const materials: THREE.MeshPhongMaterial[] = [];
  const allOpacityWeights = buildComponentOpacityWeights(frame);
  const opacityWeights: number[] = [];
  const baseRadius = frame.manifest.globe.base_radius;
  const bandScale = buildLevelBandScale(frame);

  frame.metadata.components.forEach((component, index) => {
    if (component.vertex_count <= 0 || component.index_count < 3) return;

    const geometry = buildComponentGeometry(
      frame.positions,
      frame.indices,
      component,
      baseRadius,
      style.verticalExaggeration,
      bandScale
    );
    const opacityWeight = allOpacityWeights[index] ?? 1;
    const material = new THREE.MeshPhongMaterial({
      color: new THREE.Color("#ffffff"),
      emissive: LEVEL_EMISSIVE.clone(),
      specular: LEVEL_SPECULAR.clone(),
      shininess: 44,
      transparent: true,
      opacity: opacity * opacityWeight,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      vertexColors: true,
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
  const currentFrameRef = useRef<MoistureStructureFrame | null>(null);
  const transitionFrameRef = useRef<MoistureStructureFrame | null>(null);
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

  const rebuildVisibleSlices = (root: THREE.Group, style: MoistureLayerStyle) => {
    const currentFrame = currentFrameRef.current;
    const transitionFrame = transitionFrameRef.current;
    const mix = fadeMixRef.current;

    const nextCurrent = currentFrame
      ? buildSlice(
          currentFrame,
          style,
          mix === null || !transitionFrame ? style.opacity : style.opacity * (1 - mix)
        )
      : null;
    const nextTransition =
      transitionFrame && mix !== null
        ? buildSlice(transitionFrame, style, style.opacity * mix)
        : null;

    disposeSlice(transitionRef.current);
    transitionRef.current = null;
    disposeSlice(currentRef.current);
    currentRef.current = null;

    if (nextCurrent) {
      root.add(nextCurrent.group);
      currentRef.current = nextCurrent;
    }

    if (nextTransition) {
      root.add(nextTransition.group);
      transitionRef.current = nextTransition;
    }
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
        const prev = styleRef.current;
        styleRef.current = next;
        root.visible = next.visible;
        if (next.verticalExaggeration !== prev.verticalExaggeration) {
          rebuildVisibleSlices(root, next);
        }
        applyVisibleOpacity(next.opacity);
      }
    );

    return () => {
      unsubscribe();
      disposeSlice(transitionRef.current);
      transitionRef.current = null;
      transitionFrameRef.current = null;
      fadeMixRef.current = null;
      disposeSlice(currentRef.current);
      currentRef.current = null;
      currentFrameRef.current = null;
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
      transitionFrameRef.current = null;
      fadeMixRef.current = null;
      disposeSlice(currentRef.current);
      currentRef.current = null;
      currentFrameRef.current = null;
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

        const next = buildSlice(frame, style, 0.0);
        root.add(next.group);
        transitionRef.current = next;
        transitionFrameRef.current = frame;

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
            currentFrameRef.current = frame;
            if (transitionRef.current === next) transitionRef.current = null;
            transitionFrameRef.current = null;
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
      transitionFrameRef.current = null;
      fadeMixRef.current = null;
    };
  }, [engineReady, signalReady, timestamp, visible]);

  return null;
}
