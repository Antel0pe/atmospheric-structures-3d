import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { useEarthLayer } from "./EarthBase";
import { useControls } from "../../state/controlsStore";
import {
  fetchExampleContours,
  type ExampleContoursFile,
} from "../utils/ApiResponses";
import { latLonToVec3 } from "../utils/EarthUtils";

type ExampleContoursPressure =
  ReturnType<typeof useControls.getState>["exampleContoursLayer"]["pressureLevel"];
type PressureNonNone = Exclude<ExampleContoursPressure, "none">;
type ExampleContoursStyle =
  ReturnType<typeof useControls.getState>["exampleContoursLayer"];
type ContoursSlice = {
  group: THREE.Group;
  materials: Map<string, THREE.LineBasicMaterial>;
  minValue: number;
  maxValue: number;
};

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

function disposeGroupLines(group: THREE.Group) {
  group.traverse((object) => {
    if (object instanceof THREE.Line) {
      object.geometry.dispose();
    }
  });
}

function disposeMaterialCache(cache: Map<string, THREE.LineBasicMaterial>) {
  for (const material of cache.values()) material.dispose();
  cache.clear();
}

function disposeSlice(slice: ContoursSlice | null) {
  if (!slice) return;
  disposeGroupLines(slice.group);
  slice.group.removeFromParent();
  disposeMaterialCache(slice.materials);
}

function computeMinMaxValue(
  file: ExampleContoursFile,
  pressure: PressureNonNone
): { min: number; max: number } {
  const fallback: Record<PressureNonNone, { min: number; max: number }> = {
    msl: { min: 920, max: 1060 },
    "250": { min: 9600, max: 11200 },
    "500": { min: 4600, max: 6000 },
    "925": { min: 500, max: 1100 },
  };

  const keys = Object.keys(file.levels);
  let min = Infinity;
  let max = -Infinity;
  for (const key of keys) {
    const value = Number(key);
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }

  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return fallback[pressure];
  }
  return { min, max };
}

function valueToColor(
  levelValue: number,
  minValue: number,
  maxValue: number,
  contrast: number
) {
  let t = (levelValue - minValue) / (maxValue - minValue);
  t = THREE.MathUtils.clamp(t, 0, 1);
  const width = 0.5 / Math.max(contrast, 1e-6);
  t = THREE.MathUtils.smoothstep(t, 0.5 - width, 0.5 + width);

  const red = new THREE.Color(1.0, 0.0, 0.35);
  const green = new THREE.Color(0.0, 1.0, 0.15);
  return green.clone().lerp(red, t);
}

function buildContoursGroup(opts: {
  file: ExampleContoursFile;
  pressure: PressureNonNone;
  radius: number;
  opacity: number;
  contrast: number;
  renderOrder: number;
}): ContoursSlice {
  const { file, pressure, radius, opacity, contrast, renderOrder } = opts;

  const group = new THREE.Group();
  group.name = "example-contours-slice";
  group.renderOrder = renderOrder;
  group.frustumCulled = false;

  const lift = radius * 0.002;
  const { min, max } = computeMinMaxValue(file, pressure);
  const materials = new Map<string, THREE.LineBasicMaterial>();
  const levelKeys = Object.keys(file.levels).sort(
    (a, b) => parseFloat(a) - parseFloat(b)
  );

  const getMaterialForLevel = (levelKey: string) => {
    const cached = materials.get(levelKey);
    if (cached) return cached;

    const levelValue = parseFloat(levelKey);
    const color = valueToColor(levelValue, min, max, contrast);
    const material = new THREE.LineBasicMaterial({
      transparent: true,
      opacity,
      depthTest: true,
      depthWrite: false,
      color,
    });

    materials.set(levelKey, material);
    return material;
  };

  for (const levelKey of levelKeys) {
    const lines = file.levels[levelKey];
    if (!lines || lines.length === 0) continue;

    const material = getMaterialForLevel(levelKey);
    for (const line of lines) {
      if (!line || line.length < 2) continue;

      const positions = new Float32Array(line.length * 3);
      for (let i = 0; i < line.length; i++) {
        const [lonDeg, latDeg] = line[i];
        const point = latLonToVec3(latDeg, lonDeg, radius + lift);
        const j = i * 3;
        positions[j + 0] = point.x;
        positions[j + 1] = point.y;
        positions[j + 2] = point.z;
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(positions, 3)
      );

      const threeLine = new THREE.Line(geometry, material);
      threeLine.frustumCulled = false;
      group.add(threeLine);
    }
  }

  return { group, materials, minValue: min, maxValue: max };
}

function setMaterialsOpacity(
  materials: Map<string, THREE.LineBasicMaterial>,
  opacity: number
) {
  for (const material of materials.values()) material.opacity = opacity;
}

function setSliceOpacity(slice: ContoursSlice | null, opacity: number) {
  if (!slice) return;
  setMaterialsOpacity(slice.materials, opacity);
}

function setSliceContrast(slice: ContoursSlice | null, contrast: number) {
  if (!slice) return;
  for (const [levelKey, material] of slice.materials.entries()) {
    const levelValue = Number(levelKey);
    if (!Number.isFinite(levelValue)) continue;
    material.color.copy(
      valueToColor(levelValue, slice.minValue, slice.maxValue, contrast)
    );
  }
}

export default function ExampleContoursLayer() {
  const exampleContours = useControls((state) => state.exampleContoursLayer);
  const pressureLevel = exampleContours.pressureLevel;

  const layerKey = useMemo(
    () => `example-contours-${pressureLevel}`,
    [pressureLevel]
  );
  const { engineReady, sceneRef, globeRef, timestamp, signalReady } =
    useEarthLayer(layerKey);

  const rootRef = useRef<THREE.Group | null>(null);
  const currentRef = useRef<ContoursSlice | null>(null);
  const transitionRef = useRef<ContoursSlice | null>(null);
  const styleRef = useRef<ExampleContoursStyle>(exampleContours);
  const fadeMixRef = useRef<number | null>(null);
  const reqIdRef = useRef(0);

  const applyVisibleOpacity = (targetOpacity: number) => {
    const mix = fadeMixRef.current;
    const current = currentRef.current;
    const transition = transitionRef.current;

    if (transition && mix !== null) {
      if (current) setSliceOpacity(current, targetOpacity * (1 - mix));
      setSliceOpacity(transition, targetOpacity * mix);
      return;
    }

    setSliceOpacity(current, targetOpacity);
  };

  useEffect(() => {
    if (!engineReady) return;
    if (!sceneRef.current || !globeRef.current) return;

    const scene = sceneRef.current;
    const root = new THREE.Group();
    root.name = "example-contours-root";
    root.renderOrder = 60;
    root.frustumCulled = false;

    const state = useControls.getState().exampleContoursLayer;
    root.visible = state.pressureLevel !== "none";
    styleRef.current = state;

    scene.add(root);
    rootRef.current = root;

    const unsubscribe = useControls.subscribe(
      (next) => next.exampleContoursLayer,
      (next) => {
        styleRef.current = next;
        root.visible = next.pressureLevel !== "none";
        setSliceContrast(currentRef.current, next.contrast);
        setSliceContrast(transitionRef.current, next.contrast);
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
  }, [engineReady, globeRef, sceneRef]);

  useEffect(() => {
    if (!engineReady) return;
    const root = rootRef.current;
    if (!root) return;

    let cancelled = false;
    const requestId = ++reqIdRef.current;
    const isCancelled = () => cancelled || requestId !== reqIdRef.current;

    if (pressureLevel === "none") {
      disposeSlice(transitionRef.current);
      transitionRef.current = null;
      fadeMixRef.current = null;
      disposeSlice(currentRef.current);
      currentRef.current = null;
      signalReady(timestamp);
      return () => {
        cancelled = true;
      };
    }

    void (async () => {
      try {
        const style = styleRef.current;
        const file = await fetchExampleContours(timestamp, pressureLevel);
        if (isCancelled()) return;

        const next = buildContoursGroup({
          file,
          pressure: pressureLevel,
          radius: 100,
          opacity: 0.0,
          contrast: style.contrast,
          renderOrder: 60,
        });

        root.add(next.group);
        transitionRef.current = next;

        const prev = currentRef.current;
        const fadeMs = 220;

        if (prev) setSliceContrast(prev, styleRef.current.contrast);
        fadeMixRef.current = 0;
        applyVisibleOpacity(styleRef.current.opacity);

        animateT(
          fadeMs,
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
        console.error("Failed to load example contours", error);
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
  }, [engineReady, pressureLevel, signalReady, timestamp]);

  return null;
}
