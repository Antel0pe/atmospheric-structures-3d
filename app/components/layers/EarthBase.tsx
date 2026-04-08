"use client";

import {
  Children,
  createContext,
  ReactNode,
  RefObject,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import ThreeGlobe from "three-globe";
import type { ViewerAutomationApi } from "../../lib/viewerAutomation";
import {
  VIEWER_AUTOMATION_SELECTORS,
  VIEWER_NAVIGATION_CONTROLS,
  VIEWER_TIME_CONTROLS,
} from "../../lib/viewerAutomation";
import type { EarthViewState } from "../../lib/viewerTypes";
import {
  resolveMoistureStructureLayerState,
  useControls,
  type ExampleContoursLayerState,
  type ExampleParticleLayerState,
  type ExampleShaderMeshLayerState,
  type MoistureLegibilityExperiment,
  type MoistureStructureLayerState,
  type PrecipitationRadarLayerState,
  type RelativeHumidityLayerState,
} from "../../state/controlsStore";
import type { MoistureSegmentationMode } from "../utils/ApiResponses";
import { useViewerStore } from "../../state/viewerStore";
import { lookAtLatLon } from "../utils/EarthUtils";
import type {
  ViewDebugAnalyzerAdapter,
  ViewDebugCase,
  ViewDebugCaseInput,
  ViewDebugLayerStateSnapshot,
} from "../../lib/viewDebug";
import {
  applyDiscreteNavigationCommand,
  applyMouseLookDelta,
  canonicalizeCameraForNavigation,
  deriveYawPitchFromCamera,
  moveCameraAlongGlobe,
  syncCameraFromYawPitch,
  stabilizeNavigationPose,
  ZOOM_RADIUS_MAX,
  ZOOM_RADIUS_MIN,
} from "./earthNavigation";

const LOCAL_GLOBE_TEXTURE_URL = "/earth-day.jpg";

declare global {
  interface Window {
    __ATMOS_AUTOMATION__?: ViewerAutomationApi;
  }
}

export type EarthEngine = {
  engineReady: boolean;
  hostRef: RefObject<HTMLDivElement | null>;
  rendererRef: RefObject<THREE.WebGLRenderer | null>;
  sceneRef: RefObject<THREE.Scene | null>;
  cameraRef: RefObject<THREE.PerspectiveCamera | null>;
  controlsRef: RefObject<OrbitControls | null>;
  globeRef: RefObject<ThreeGlobe | null>;
  ambientLightRef: RefObject<THREE.AmbientLight | null>;
  moistureKeyLightRef: RefObject<THREE.DirectionalLight | null>;
  moistureHeadLightRef: RefObject<THREE.DirectionalLight | null>;
  timestamp: string;
  registerLayer: (key: string) => void;
  unregisterLayer: (key: string) => void;
  signalLayerReady: (ts: string, key: string) => void;
  allLayersReady: boolean;
  registerFramePass: (key: string, pass: FramePass) => void;
  unregisterFramePass: (key: string) => void;
  registerDebugAdapter: (adapter: ViewDebugAnalyzerAdapter) => void;
  unregisterDebugAdapter: (analyzer: string) => void;
  zoom01: number;
  setZoom01: (z: number) => void;
};

const EarthContext = createContext<EarthEngine | null>(null);

function useEarth() {
  const ctx = useContext(EarthContext);
  if (!ctx) throw new Error("useEarth must be used inside <EarthBase />");
  return ctx;
}

export function useEarthLayer(key: string) {
  const earth = useEarth();
  const { registerLayer, unregisterLayer, signalLayerReady } = earth;

  useEffect(() => {
    registerLayer(key);
    return () => unregisterLayer(key);
  }, [key, registerLayer, unregisterLayer]);

  const signalReady = useCallback(
    (ts: string) => signalLayerReady(ts, key),
    [key, signalLayerReady]
  );

  return { ...earth, signalReady };
}

export type FrameTick = {
  dt: number;
  t: number;
  timestamp: string;
};

export type FramePass = (tick: FrameTick) => void;

type Props = {
  timestamp: string;
  onAllReadyChange?: (ready: boolean, timestamp: string) => void;
  children?: ReactNode;
};

function vector3ToState(vector: THREE.Vector3) {
  return { x: vector.x, y: vector.y, z: vector.z };
}

function quaternionToState(quaternion: THREE.Quaternion) {
  return {
    x: quaternion.x,
    y: quaternion.y,
    z: quaternion.z,
    w: quaternion.w,
  };
}

function isEditableKeyboardTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;

  const tagName = target.tagName;
  if (tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT") {
    return true;
  }

  return target.isContentEditable;
}

export default function EarthBase({
  timestamp,
  onAllReadyChange,
  children,
}: Props) {
  const expectedLayerCount = Children.count(children);
  const hostRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const globeRef = useRef<ThreeGlobe | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const ambientLightRef = useRef<THREE.AmbientLight | null>(null);
  const moistureKeyLightRef = useRef<THREE.DirectionalLight | null>(null);
  const moistureHeadLightRef = useRef<THREE.DirectionalLight | null>(null);
  const sunRef = useRef<THREE.DirectionalLight | null>(null);
  const roRef = useRef<ResizeObserver | null>(null);
  const automationEnabledRef = useRef(false);
  const automationPausedRef = useRef(false);
  const globeReadyRef = useRef(false);
  const allLayersReadyRef = useRef(false);
  const latestTimestampRef = useRef(timestamp);
  const readyLayersRef = useRef(new Map<string, string>());
  const registeredLayersRef = useRef(new Map<string, number>());
  const framePassesRef = useRef(new Map<string, FramePass>());
  const debugAdaptersRef = useRef(new Map<string, ViewDebugAnalyzerAdapter>());
  const zoomCommitTimerRef = useRef<number | null>(null);
  const pendingZoomRef = useRef(0);
  const navigationPoseRef = useRef({ yaw: 0, pitch: 0 });
  const [globeReady, setGlobeReady] = useState(false);
  const [engineReady, setEngineReady] = useState(false);
  const [allLayersReady, setAllLayersReady] = useState(false);
  const [zoom01, _setZoom01] = useState(0);

  latestTimestampRef.current = timestamp;

  useEffect(() => {
    globeReadyRef.current = globeReady;
  }, [globeReady]);

  useEffect(() => {
    allLayersReadyRef.current = allLayersReady;
  }, [allLayersReady]);

  const radiusToZoom01 = useCallback((radius: number) => {
    const zoom = (radius - ZOOM_RADIUS_MIN) / (ZOOM_RADIUS_MAX - ZOOM_RADIUS_MIN);
    return THREE.MathUtils.clamp(zoom, 0, 1);
  }, []);

  const zoom01ToRadius = useCallback((zoom: number) => {
    return (
      ZOOM_RADIUS_MIN +
      THREE.MathUtils.clamp(zoom, 0, 1) * (ZOOM_RADIUS_MAX - ZOOM_RADIUS_MIN)
    );
  }, []);

  const render = useCallback(() => {
    const renderer = rendererRef.current;
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    if (!renderer || !scene || !camera) return;
    renderer.render(scene, camera);
  }, []);

  const publishEarthView = useCallback(() => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return null;

    const nextZoom01 = radiusToZoom01(camera.position.length());
    const earthView: EarthViewState = {
      cameraPosition: vector3ToState(camera.position),
      cameraQuaternion: quaternionToState(camera.quaternion),
      cameraUp: vector3ToState(camera.up),
      controlsTarget: vector3ToState(controls.target),
      yaw: navigationPoseRef.current.yaw,
      pitch: navigationPoseRef.current.pitch,
      zoom01: nextZoom01,
    };

    useViewerStore.getState().publishEarthView(earthView);
    return earthView;
  }, [radiusToZoom01]);

  const syncZoomState = useCallback(
    (nextZoom01: number, immediate = false) => {
      pendingZoomRef.current = nextZoom01;
      useViewerStore.getState().setZoom01(nextZoom01);

      if (immediate) {
        if (zoomCommitTimerRef.current != null) {
          window.clearTimeout(zoomCommitTimerRef.current);
          zoomCommitTimerRef.current = null;
        }
        _setZoom01(nextZoom01);
        return;
      }

      if (zoomCommitTimerRef.current != null) {
        window.clearTimeout(zoomCommitTimerRef.current);
      }

      zoomCommitTimerRef.current = window.setTimeout(() => {
        _setZoom01(pendingZoomRef.current);
        zoomCommitTimerRef.current = null;
      }, 500);
    },
    []
  );

  const syncViewerState = useCallback(
    (immediateZoom = false) => {
      const camera = cameraRef.current;
      if (!camera) return null;

      const nextZoom01 = radiusToZoom01(camera.position.length());
      syncZoomState(nextZoom01, immediateZoom);
      return publishEarthView();
    },
    [publishEarthView, radiusToZoom01, syncZoomState]
  );

  const applyEarthViewState = useCallback(
    (earthView: EarthViewState) => {
      const camera = cameraRef.current;
      const controls = controlsRef.current;
      if (!camera || !controls) return;

      camera.position.set(
        earthView.cameraPosition.x,
        earthView.cameraPosition.y,
        earthView.cameraPosition.z
      );
      camera.quaternion.set(
        earthView.cameraQuaternion.x,
        earthView.cameraQuaternion.y,
        earthView.cameraQuaternion.z,
        earthView.cameraQuaternion.w
      );
      camera.up.set(
        earthView.cameraUp.x,
        earthView.cameraUp.y,
        earthView.cameraUp.z
      );
      controls.target.set(
        earthView.controlsTarget.x,
        earthView.controlsTarget.y,
        earthView.controlsTarget.z
      );

      navigationPoseRef.current = stabilizeNavigationPose({
        yaw: earthView.yaw,
        pitch: earthView.pitch,
      });
      syncCameraFromYawPitch(
        camera,
        controls,
        navigationPoseRef.current.yaw,
        navigationPoseRef.current.pitch
      );
      syncZoomState(earthView.zoom01, true);
      publishEarthView();
      render();
    },
    [publishEarthView, render, syncZoomState]
  );

  const recomputeAllReady = useCallback(() => {
    if (!engineReady) {
      setAllLayersReady(false);
      return;
    }

    const registered = registeredLayersRef.current;
    const ready = readyLayersRef.current;
    const currentTimestamp = latestTimestampRef.current;
    let registeredCount = 0;

    for (const count of registered.values()) {
      if (count > 0) registeredCount += count;
    }

    if (registeredCount < expectedLayerCount) {
      setAllLayersReady(false);
      return;
    }

    let nextReady = true;
    for (const [key, count] of registered.entries()) {
      if (count <= 0) continue;
      if (ready.get(key) !== currentTimestamp) {
        nextReady = false;
        break;
      }
    }

    setAllLayersReady(nextReady);
  }, [engineReady, expectedLayerCount]);

  useEffect(() => {
    recomputeAllReady();
  }, [engineReady, recomputeAllReady, timestamp]);

  const registerLayer = useCallback(
    (key: string) => {
      const current = registeredLayersRef.current.get(key) ?? 0;
      registeredLayersRef.current.set(key, current + 1);
      recomputeAllReady();
    },
    [recomputeAllReady]
  );

  const unregisterLayer = useCallback(
    (key: string) => {
      const current = registeredLayersRef.current.get(key) ?? 0;
      if (current <= 1) {
        registeredLayersRef.current.delete(key);
        readyLayersRef.current.delete(key);
      } else {
        registeredLayersRef.current.set(key, current - 1);
      }
      recomputeAllReady();
    },
    [recomputeAllReady]
  );

  const signalLayerReady = useCallback(
    (ts: string, key: string) => {
      if (ts !== latestTimestampRef.current) return;
      readyLayersRef.current.set(key, ts);
      recomputeAllReady();
    },
    [recomputeAllReady]
  );

  useEffect(() => {
    onAllReadyChange?.(allLayersReady, timestamp);
  }, [allLayersReady, onAllReadyChange, timestamp]);

  const registerFramePass = useCallback((key: string, pass: FramePass) => {
    framePassesRef.current.set(key, pass);
  }, []);

  const unregisterFramePass = useCallback((key: string) => {
    framePassesRef.current.delete(key);
  }, []);

  const registerDebugAdapter = useCallback((adapter: ViewDebugAnalyzerAdapter) => {
    debugAdaptersRef.current.set(adapter.analyzer, adapter);
  }, []);

  const unregisterDebugAdapter = useCallback((analyzer: string) => {
    debugAdaptersRef.current.delete(analyzer);
  }, []);

  const setZoom01 = useCallback(
    (zoom: number) => {
      const camera = cameraRef.current;
      const controls = controlsRef.current;
      if (!camera || !controls) return;

      const radius = zoom01ToRadius(zoom);
      camera.position.normalize().multiplyScalar(radius);
      controls.update();
      navigationPoseRef.current = stabilizeNavigationPose(
        deriveYawPitchFromCamera(camera.position, camera.quaternion)
      );
      syncZoomState(radiusToZoom01(radius), true);
      publishEarthView();
      render();
    },
    [publishEarthView, radiusToZoom01, render, syncZoomState, zoom01ToRadius]
  );

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const automationEnabled =
      new URLSearchParams(window.location.search).get("automation") === "1";
    automationEnabledRef.current = automationEnabled;
    automationPausedRef.current = false;
    globeReadyRef.current = false;
    setGlobeReady(false);

    const getSize = () => {
      const rect = host.getBoundingClientRect();
      return { w: Math.max(1, rect.width), h: Math.max(1, rect.height) };
    };

    const { w, h } = getSize();

    const renderer = new THREE.WebGLRenderer({
      antialias: automationEnabled ? false : window.devicePixelRatio < 2,
      powerPreference: automationEnabled ? "high-performance" : "default",
      preserveDrawingBuffer: automationEnabled,
    });
    renderer.autoClear = false;
    renderer.setPixelRatio(automationEnabled ? 1 : Math.min(window.devicePixelRatio ?? 1, 2));
    renderer.setSize(w, h);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0c10);

    const globe = new ThreeGlobe()
      .onGlobeReady(() => {
        globeReadyRef.current = true;
        setGlobeReady(true);
      })
      .globeImageUrl(LOCAL_GLOBE_TEXTURE_URL);
    scene.add(globe);

    const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1e9);
    camera.up.set(0, 1, 0);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enableRotate = false;
    controls.minPolarAngle = 0.0001;
    controls.maxPolarAngle = Math.PI - 0.0001;
    controls.minAzimuthAngle = -Infinity;
    controls.maxAzimuthAngle = Infinity;

    camera.position.set(0, -300, 150);
    controls.target.set(0, 0, 0);
    controls.update();
    renderer.render(scene, camera);

    lookAtLatLon(30, -135, camera, controls, globe, 100);
    navigationPoseRef.current = canonicalizeCameraForNavigation(camera, controls);
    syncZoomState(radiusToZoom01(camera.position.length()), true);

    const ambient = new THREE.AmbientLight(0xffffff, 2);
    scene.add(ambient);

    const moistureLightTarget = new THREE.Object3D();
    moistureLightTarget.position.set(0, 0, 0);
    scene.add(moistureLightTarget);

    const moistureKeyLight = new THREE.DirectionalLight(0xf5f9ff, 0);
    moistureKeyLight.position.set(220, -160, 260);
    moistureKeyLight.target = moistureLightTarget;
    scene.add(moistureKeyLight);

    const moistureHeadLight = new THREE.DirectionalLight(0xcfe4ff, 0);
    moistureHeadLight.position.copy(camera.position);
    moistureHeadLight.target = moistureLightTarget;
    scene.add(moistureHeadLight);

    const sun = null;
    const canvas = renderer.domElement;
    const pressed = new Set<string>();
    let moving = false;
    let lastT = performance.now();

    const publishAndRender = (immediateZoom = false) => {
      syncViewerState(immediateZoom);
      render();
    };

    const onControlsChange = () => {
      navigationPoseRef.current = stabilizeNavigationPose(
        deriveYawPitchFromCamera(camera.position, camera.quaternion)
      );
      syncViewerState();
    };

    controls.addEventListener("change", onControlsChange);

    function onPointerLockChange() {
      const locked = document.pointerLockElement === canvas;
      canvas.style.cursor = locked ? "none" : "grab";

      if (locked) {
        canvas.addEventListener("mousemove", onMouseMove);
      } else {
        canvas.removeEventListener("mousemove", onMouseMove);
      }
    }

    function onPointerLockError() {
      console.warn("[PointerLock] request failed (browser/permission)");
    }

    function onCanvasClick(event: MouseEvent) {
      if (event.button === 0) {
        canvas.requestPointerLock();
      }
    }

    function onReleaseKey(event: KeyboardEvent) {
      if (
        event.key.toLowerCase() === "q" &&
        document.pointerLockElement === canvas
      ) {
        document.exitPointerLock();
      }
    }

    function onMouseMove(event: MouseEvent) {
      navigationPoseRef.current = applyMouseLookDelta(
        camera,
        controls,
        navigationPoseRef.current,
        event.movementX || 0,
        event.movementY || 0
      );
      publishAndRender();
    }

    function clearMovementState() {
      pressed.clear();
      moving = false;
    }

    function shouldIgnoreKeyboardEvent(event: KeyboardEvent) {
      return (
        isEditableKeyboardTarget(event.target) ||
        isEditableKeyboardTarget(document.activeElement)
      );
    }

    function stepMovement(dt: number) {
      const result = moveCameraAlongGlobe({
        camera,
        controls,
        pose: navigationPoseRef.current,
        dt,
        input: {
          forward: (pressed.has("w") ? 1 : 0) + (pressed.has("s") ? -1 : 0),
          right: (pressed.has("d") ? 1 : 0) + (pressed.has("a") ? -1 : 0),
          vertical: (pressed.has(" ") ? 1 : 0) + (pressed.has("shift") ? -1 : 0),
        },
      });

      if (!result.didMove) return false;

      navigationPoseRef.current = {
        yaw: result.yaw,
        pitch: result.pitch,
      };
      publishAndRender();
      return true;
    }

    function startMoveLoop() {
      if (moving) return;
      moving = true;
      lastT = performance.now();

      const step = () => {
        if (!moving) return;
        if (pressed.size === 0) {
          moving = false;
          return;
        }

        const now = performance.now();
        const dt = Math.min(0.05, (now - lastT) / 1000);
        lastT = now;
        stepMovement(dt);
        requestAnimationFrame(step);
      };

      requestAnimationFrame(step);
    }

    function onKeyDown(event: KeyboardEvent) {
      if (shouldIgnoreKeyboardEvent(event)) {
        clearMovementState();
        return;
      }

      const key = event.key.toLowerCase();
      if (key === " ") event.preventDefault();
      pressed.add(key);
      startMoveLoop();
    }

    function onKeyUp(event: KeyboardEvent) {
      if (shouldIgnoreKeyboardEvent(event)) {
        clearMovementState();
        return;
      }

      pressed.delete(event.key.toLowerCase());
    }

    function onVisibilityChange() {
      if (document.hidden) clearMovementState();
    }

    function onFocusIn(event: FocusEvent) {
      if (isEditableKeyboardTarget(event.target)) {
        clearMovementState();
      }
    }

    const resizeObserver = new ResizeObserver(() => {
      const next = getSize();
      renderer.setSize(next.w, next.h);
      camera.aspect = next.w / next.h;
      camera.updateProjectionMatrix();
    });
    resizeObserver.observe(host);

    function getReadyState() {
      return globeReadyRef.current && allLayersReadyRef.current;
    }

    function getViewerSnapshot() {
      const state = useViewerStore.getState();
      const moistureLayer = useControls.getState().moistureStructureLayer;
      const resolvedMoistureLayer = resolveMoistureStructureLayerState(
        moistureLayer
      );
      return {
        ready: getReadyState(),
        paused: automationPausedRef.current,
        timestamp: state.timestamp,
        zoom01: state.zoom01,
        moistureLegibilityExperiment: moistureLayer.legibilityExperiment,
        moistureSegmentationMode: resolvedMoistureLayer.segmentationMode,
        earthView: state.earthView,
        savedViews: state.savedViews,
      };
    }

    function cloneMoistureStructureLayerState(
      state: MoistureStructureLayerState
    ): MoistureStructureLayerState {
      return {
        ...state,
        visibleBucketIndices: [...state.visibleBucketIndices],
      };
    }

    function cloneRelativeHumidityLayerState(
      state: RelativeHumidityLayerState
    ): RelativeHumidityLayerState {
      return { ...state };
    }

    function clonePrecipitationRadarLayerState(
      state: PrecipitationRadarLayerState
    ): PrecipitationRadarLayerState {
      return { ...state };
    }

    function cloneExampleShaderMeshLayerState(
      state: ExampleShaderMeshLayerState
    ): ExampleShaderMeshLayerState {
      return { ...state };
    }

    function cloneExampleContoursLayerState(
      state: ExampleContoursLayerState
    ): ExampleContoursLayerState {
      return { ...state };
    }

    function cloneExampleParticleLayerState(
      state: ExampleParticleLayerState
    ): ExampleParticleLayerState {
      return { ...state };
    }

    function getLayerStateSnapshot(): ViewDebugLayerStateSnapshot {
      const controlsState = useControls.getState();
      return {
        moistureStructureLayer: cloneMoistureStructureLayerState(
          controlsState.moistureStructureLayer
        ),
        precipitationRadarLayer: clonePrecipitationRadarLayerState(
          controlsState.precipitationRadarLayer
        ),
        relativeHumidityLayer: cloneRelativeHumidityLayerState(
          controlsState.relativeHumidityLayer
        ),
        exampleShaderMeshLayer: cloneExampleShaderMeshLayerState(
          controlsState.exampleShaderMeshLayer
        ),
        exampleContoursLayer: cloneExampleContoursLayerState(
          controlsState.exampleContoursLayer
        ),
        exampleParticleLayer: cloneExampleParticleLayerState(
          controlsState.exampleParticleLayer
        ),
      };
    }

    function applyLayerStateSnapshot(snapshot: ViewDebugLayerStateSnapshot) {
      const controlsState = useControls.getState();
      controlsState.setMoistureStructureLayer(
        cloneMoistureStructureLayerState(snapshot.moistureStructureLayer)
      );
      controlsState.setPrecipitationRadarLayer(
        clonePrecipitationRadarLayerState(snapshot.precipitationRadarLayer)
      );
      controlsState.setRelativeHumidityLayer(
        cloneRelativeHumidityLayerState(snapshot.relativeHumidityLayer)
      );
      controlsState.setExampleShaderMeshLayer(
        cloneExampleShaderMeshLayerState(snapshot.exampleShaderMeshLayer)
      );
      controlsState.setExampleContoursLayer(
        cloneExampleContoursLayerState(snapshot.exampleContoursLayer)
      );
      controlsState.setExampleParticleLayer(
        cloneExampleParticleLayerState(snapshot.exampleParticleLayer)
      );
    }

    function getViewDebugState() {
      const viewerSnapshot = getViewerSnapshot();
      const analyzers = Object.fromEntries(
        Array.from(debugAdaptersRef.current.entries()).map(([analyzer, adapter]) => [
          analyzer,
          adapter.getState?.() ?? null,
        ])
      );

      return {
        version: 1 as const,
        timestamp: viewerSnapshot.timestamp,
        ready: viewerSnapshot.ready,
        paused: viewerSnapshot.paused,
        zoom01: viewerSnapshot.zoom01,
        earthView: viewerSnapshot.earthView,
        layerState: getLayerStateSnapshot(),
        analyzers,
        savedViews: viewerSnapshot.savedViews,
      };
    }

    function buildViewDebugCase(input: ViewDebugCaseInput): ViewDebugCase {
      const earthView = useViewerStore.getState().earthView;
      if (!earthView) {
        throw new Error("The earth view is not ready yet.");
      }

      return {
        version: 1,
        analyzer: input.analyzer,
        title: input.title.trim(),
        createdAt: new Date().toISOString(),
        source: input.source,
        timestamp: useViewerStore.getState().timestamp,
        earthView,
        layerState: getLayerStateSnapshot(),
        targets: input.targets.map((target) => ({ ...target })),
        notes: input.notes?.trim() || undefined,
      };
    }

    function isInteractableElement(element: HTMLElement | null) {
      if (!element) return false;
      if (!element.isConnected) return false;

      const style = window.getComputedStyle(element);
      return (
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        style.opacity !== "0" &&
        !element.hasAttribute("disabled")
      );
    }

    function findElementByTestId(testId: string) {
      return document.querySelector<HTMLElement>(`[data-testid="${testId}"]`);
    }

    function clickElementByTestId(testId: string) {
      const element = findElementByTestId(testId);
      if (!element || !isInteractableElement(element)) {
        throw new Error(`Could not find an interactable element for data-testid="${testId}".`);
      }
      element.click();
    }

    async function waitForCondition(
      predicate: () => boolean,
      timeoutMs: number,
      description: string
    ) {
      const startedAt = performance.now();

      while (performance.now() - startedAt < timeoutMs) {
        if (predicate()) return;
        await new Promise<void>((resolve) => {
          window.setTimeout(resolve, 50);
        });
      }

      throw new Error(`Timed out waiting for ${description}.`);
    }

    async function waitForReady(timeoutMs = 90_000) {
      await waitForCondition(getReadyState, timeoutMs, "the viewer to become ready");
      return getViewerSnapshot();
    }

    async function waitForMoistureState(timeoutMs = 90_000) {
      const moistureLayer = useControls.getState().moistureStructureLayer;
      const resolved = resolveMoistureStructureLayerState(moistureLayer);
      const expectedTimestamp = useViewerStore.getState().timestamp;

      await waitForCondition(
        () => {
          const frame = useControls.getState().moistureStructureFrame;
          return (
            !resolved.visible ||
            (frame !== null &&
              frame.timestamp === expectedTimestamp &&
              frame.segmentationMode === resolved.segmentationMode)
          );
        },
        timeoutMs,
        `moisture state for ${resolved.segmentationMode} to settle`
      );

      await new Promise<void>((resolve) => {
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            render();
            resolve();
          })
        );
      });

      return getViewerSnapshot();
    }

    async function loadSavedViews() {
      await useViewerStore.getState().loadSavedViews();
      return useViewerStore.getState().savedViews;
    }

    async function resolveSavedView(target: { id?: string; title?: string }) {
      const savedViews = await loadSavedViews();

      if (target.id) {
        const byId = savedViews.find((savedView) => savedView.id === target.id);
        if (byId) return byId;
      }

      if (target.title) {
        const exactTitleMatch = savedViews.find(
          (savedView) => savedView.title === target.title
        );
        if (exactTitleMatch) return exactTitleMatch;

        const foldedTitle = target.title.toLowerCase();
        const caseInsensitiveMatches = savedViews.filter(
          (savedView) => savedView.title.toLowerCase() === foldedTitle
        );
        if (caseInsensitiveMatches.length === 1) {
          return caseInsensitiveMatches[0];
        }
      }

      throw new Error("Saved view not found for the provided id/title.");
    }

    const unsubscribeNavigation = useViewerStore.subscribe(
      (state) => state.navigationRequest,
      (request, previousRequest) => {
        if (!request) return;
        if (request.requestId === previousRequest?.requestId) return;

        const result = applyDiscreteNavigationCommand(
          camera,
          controls,
          navigationPoseRef.current,
          request.command
        );
        navigationPoseRef.current = {
          yaw: result.yaw,
          pitch: result.pitch,
        };
        publishAndRender();
      }
    );

    const unsubscribeApply = useViewerStore.subscribe(
      (state) => state.applySavedViewRequest,
      (request, previousRequest) => {
        if (!request) return;
        const isSameRequest =
          request.requestId === previousRequest?.requestId &&
          request.phase === previousRequest?.phase;
        if (isSameRequest) return;

        applyEarthViewState(request.savedView.earthView);

        if (request.phase === "ready") {
          useViewerStore.getState().clearApplySavedViewRequest(request.requestId);
        }
      }
    );

    rendererRef.current = renderer;
    sceneRef.current = scene;
    cameraRef.current = camera;
    controlsRef.current = controls;
    globeRef.current = globe;
    ambientLightRef.current = ambient;
    moistureKeyLightRef.current = moistureKeyLight;
    moistureHeadLightRef.current = moistureHeadLight;
    sunRef.current = sun;
    roRef.current = resizeObserver;

    window.__ATMOS_AUTOMATION__ = {
      enabled: automationEnabled,
      get paused() {
        return automationPausedRef.current;
      },
      get ready() {
        return getReadyState();
      },
      freeze() {
        automationPausedRef.current = true;
      },
      resume() {
        automationPausedRef.current = false;
      },
      renderOnce() {
        controls.update();
        render();
      },
      capturePngDataUrl() {
        controls.update();
        render();
        const gl = renderer.getContext();
        if (typeof gl.finish === "function") {
          gl.finish();
        }
        return renderer.domElement.toDataURL("image/png");
      },
      describe() {
        return {
          version: 1,
          recommendedUrl: "/?automation=1",
          selectors: VIEWER_AUTOMATION_SELECTORS,
          navigationControls: VIEWER_NAVIGATION_CONTROLS,
          timeControls: VIEWER_TIME_CONTROLS,
          snapshot: getViewerSnapshot(),
        };
      },
      getSnapshot() {
        return getViewerSnapshot();
      },
      getViewDebugState,
      buildViewDebugCase,
      waitForReady,
      async ensureLayersSidebarOpen() {
        const openLabel = VIEWER_AUTOMATION_SELECTORS.layersSidebarToggle.openAriaLabel;
        const closeLabel = VIEWER_AUTOMATION_SELECTORS.layersSidebarToggle.closeAriaLabel;
        const toggle = document.querySelector<HTMLElement>(
          `[aria-label="${openLabel}"], [aria-label="${closeLabel}"]`
        );
        if (!toggle || !isInteractableElement(toggle)) {
          throw new Error("Could not find the layers sidebar toggle.");
        }

        if (toggle.getAttribute("aria-label") === closeLabel) {
          return true;
        }

        toggle.click();
        await waitForCondition(
          () =>
            document
              .querySelector<HTMLElement>(
                `[aria-label="${openLabel}"], [aria-label="${closeLabel}"]`
              )
              ?.getAttribute("aria-label") === closeLabel,
          5_000,
          "the layers sidebar to open"
        );
        return true;
      },
      async runNavigationCommand(command) {
        const result = applyDiscreteNavigationCommand(
          camera,
          controls,
          navigationPoseRef.current,
          command
        );
        navigationPoseRef.current = {
          yaw: result.yaw,
          pitch: result.pitch,
        };
        publishAndRender();
        await new Promise<void>((resolve) => {
          requestAnimationFrame(() => resolve());
        });
        return getViewerSnapshot();
      },
      async stepTime(direction, timeoutMs = 90_000) {
        const previousTimestamp = useViewerStore.getState().timestamp;
        clickElementByTestId(
          direction === "backward"
            ? VIEWER_TIME_CONTROLS[0].testId
            : VIEWER_TIME_CONTROLS[1].testId
        );

        await new Promise<void>((resolve) => {
          window.setTimeout(resolve, 0);
        });

        if (useViewerStore.getState().timestamp === previousTimestamp) {
          return getViewerSnapshot();
        }

        return waitForReady(timeoutMs);
      },
      async setTimestamp(nextTimestamp, timeoutMs = 90_000) {
        const timestampValue = nextTimestamp.trim();
        if (!timestampValue) {
          throw new Error("Timestamp is required.");
        }

        if (useViewerStore.getState().timestamp === timestampValue) {
          return getViewerSnapshot();
        }

        useViewerStore.getState().setTimestamp(timestampValue);
        return waitForReady(timeoutMs);
      },
      async applyViewState(input, timeoutMs = 90_000) {
        const nextTimestamp = input.timestamp?.trim();
        const currentTimestamp = useViewerStore.getState().timestamp;

        if (nextTimestamp && nextTimestamp !== currentTimestamp) {
          useViewerStore.getState().setTimestamp(nextTimestamp);
          applyEarthViewState(input.earthView);
          await waitForReady(timeoutMs);
          applyEarthViewState(input.earthView);
          return getViewerSnapshot();
        }

        applyEarthViewState(input.earthView);
        return getViewerSnapshot();
      },
      async applyViewDebugCase(debugCase, timeoutMs = 90_000) {
        applyLayerStateSnapshot(debugCase.layerState);
        const nextTimestamp = debugCase.timestamp.trim();
        const currentTimestamp = useViewerStore.getState().timestamp;

        if (nextTimestamp && nextTimestamp !== currentTimestamp) {
          useViewerStore.getState().setTimestamp(nextTimestamp);
          applyEarthViewState(debugCase.earthView);
          await waitForReady(timeoutMs);
        }

        applyEarthViewState(debugCase.earthView);
        await waitForReady(timeoutMs);
        await waitForMoistureState(timeoutMs);
        applyEarthViewState(debugCase.earthView);
        render();
        return getViewDebugState();
      },
      async hitTestDebugTarget(request) {
        const adapter = debugAdaptersRef.current.get(request.analyzer);
        if (!adapter?.hitTest) {
          throw new Error(`No debug analyzer is registered for "${request.analyzer}".`);
        }

        await waitForReady();
        return adapter.hitTest(request.target);
      },
      async selectDebugTarget(request) {
        const adapter = debugAdaptersRef.current.get(request.analyzer);
        if (!adapter?.selectTarget) {
          throw new Error(
            `The debug analyzer "${request.analyzer}" does not support selection.`
          );
        }

        const result = adapter.selectTarget(request.targetId);
        render();
        return result ?? null;
      },
      async setMoistureLegibilityExperiment(
        experiment: MoistureLegibilityExperiment,
        timeoutMs = 90_000
      ) {
        if (useControls.getState().moistureStructureLayer.legibilityExperiment === experiment) {
          return waitForMoistureState(timeoutMs);
        }

        useControls.getState().setMoistureLegibilityExperiment(experiment);
        return waitForMoistureState(timeoutMs);
      },
      async resetMoistureLegibilityExperiment(timeoutMs = 90_000) {
        if (
          useControls.getState().moistureStructureLayer.legibilityExperiment ===
          "bridgePruned"
        ) {
          return waitForMoistureState(timeoutMs);
        }

        useControls.getState().resetMoistureLegibilityExperiment();
        return waitForMoistureState(timeoutMs);
      },
      async setMoistureSegmentationMode(
        segmentationMode: MoistureSegmentationMode,
        timeoutMs = 90_000
      ) {
        if (useControls.getState().moistureStructureLayer.segmentationMode === segmentationMode) {
          return waitForMoistureState(timeoutMs);
        }

        useControls.getState().setMoistureStructureLayer({ segmentationMode });
        return waitForMoistureState(timeoutMs);
      },
      async listSavedViews() {
        return loadSavedViews();
      },
      async saveView(input) {
        const savedView = await useViewerStore.getState().saveSavedView({
          title: input.title,
          description: input.description ?? "",
        });

        if (!savedView) {
          throw new Error(
            useViewerStore.getState().savedViewsError ??
              "Failed to save the current view."
          );
        }

        return savedView;
      },
      async applySavedView(target, timeoutMs = 90_000) {
        const savedView = await resolveSavedView(target);
        useViewerStore.getState().requestApplySavedView(savedView);

        const requestId = useViewerStore.getState().applySavedViewRequest?.requestId;
        if (!requestId) {
          throw new Error("Failed to create a saved view apply request.");
        }

        await waitForCondition(
          () => {
            const request = useViewerStore.getState().applySavedViewRequest;
            return !request || request.requestId !== requestId;
          },
          timeoutMs,
          `saved view "${savedView.title}" to apply`
        );

        return getViewerSnapshot();
      },
      async deleteSavedView(target) {
        const savedView = await resolveSavedView(target);
        const deleted = await useViewerStore.getState().deleteSavedView(savedView.id);

        if (!deleted) {
          throw new Error(
            useViewerStore.getState().savedViewsError ??
              `Failed to delete saved view "${savedView.title}".`
          );
        }

        return {
          ok: true,
          id: savedView.id,
        };
      },
    };

    canvas.addEventListener("click", onCanvasClick);
    document.addEventListener("pointerlockchange", onPointerLockChange);
    document.addEventListener("pointerlockerror", onPointerLockError);
    window.addEventListener("keydown", onReleaseKey);
    window.addEventListener("keydown", onKeyDown, { passive: false });
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", clearMovementState);
    document.addEventListener("focusin", onFocusIn);
    document.addEventListener("visibilitychange", onVisibilityChange);

    setEngineReady(true);
    publishEarthView();

    return () => {
      setEngineReady(false);
      unsubscribeNavigation();
      unsubscribeApply();
      resizeObserver.disconnect();
      controls.removeEventListener("change", onControlsChange);
      controls.dispose();

      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", clearMovementState);
      window.removeEventListener("keydown", onReleaseKey);
      document.removeEventListener("focusin", onFocusIn);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      document.removeEventListener("pointerlockchange", onPointerLockChange);
      document.removeEventListener("pointerlockerror", onPointerLockError);
      canvas.removeEventListener("click", onCanvasClick);
      canvas.removeEventListener("mousemove", onMouseMove);

      if (document.pointerLockElement === canvas) {
        document.exitPointerLock();
      }

      if (zoomCommitTimerRef.current != null) {
        window.clearTimeout(zoomCommitTimerRef.current);
        zoomCommitTimerRef.current = null;
      }

      renderer.dispose();
      if (renderer.domElement.parentElement === host) {
        host.removeChild(renderer.domElement);
      }

      debugAdaptersRef.current.clear();
      ambientLightRef.current = null;
      moistureKeyLightRef.current = null;
      moistureHeadLightRef.current = null;
      delete window.__ATMOS_AUTOMATION__;
    };
  }, [
    applyEarthViewState,
    publishEarthView,
    radiusToZoom01,
    render,
    syncViewerState,
    syncZoomState,
  ]);

  useEffect(() => {
    if (!automationEnabledRef.current) return;
    const shouldPause = allLayersReady && globeReady;
    automationPausedRef.current = shouldPause;
    if (shouldPause) {
      render();
    }
  }, [allLayersReady, globeReady, render]);

  useEffect(() => {
    const renderer = rendererRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!renderer || !camera || !controls) return;
    const sceneRenderer = renderer;
    const sceneControls = controls;

    let running = true;

    function runFramePassSafely(pass: FramePass, tick: FrameTick) {
      const prevRT = sceneRenderer.getRenderTarget();
      const prevViewport = new THREE.Vector4();
      const prevScissor = new THREE.Vector4();
      const prevScissorTest = sceneRenderer.getScissorTest();

      sceneRenderer.getViewport(prevViewport);
      sceneRenderer.getScissor(prevScissor);

      const prevAutoClear = sceneRenderer.autoClear;
      const prevClearAlpha = sceneRenderer.getClearAlpha();
      const prevClearColor = new THREE.Color();
      sceneRenderer.getClearColor(prevClearColor);

      try {
        pass(tick);
      } catch (error) {
        console.error("[FramePass] failed:", error);
      } finally {
        sceneRenderer.setRenderTarget(prevRT);
        sceneRenderer.setViewport(
          prevViewport.x,
          prevViewport.y,
          prevViewport.z,
          prevViewport.w
        );
        sceneRenderer.setScissor(
          prevScissor.x,
          prevScissor.y,
          prevScissor.z,
          prevScissor.w
        );
        sceneRenderer.setScissorTest(prevScissorTest);
        sceneRenderer.autoClear = prevAutoClear;
        sceneRenderer.setClearColor(prevClearColor, prevClearAlpha);
      }
    }

    const loop = () => {
      if (!running) return;

      if (automationPausedRef.current) {
        requestAnimationFrame(loop);
        return;
      }

      const now = performance.now();
      const dt = 15;

      const prevViewport = new THREE.Vector4();
      const prevScissor = new THREE.Vector4();
      const prevScissorTest = sceneRenderer.getScissorTest();
      sceneRenderer.getViewport(prevViewport);
      sceneRenderer.getScissor(prevScissor);

      const tick: FrameTick = { dt, t: now, timestamp };
      for (const pass of framePassesRef.current.values()) {
        runFramePassSafely(pass, tick);
      }

      sceneRenderer.setViewport(
        prevViewport.x,
        prevViewport.y,
        prevViewport.z,
        prevViewport.w
      );
      sceneRenderer.setScissor(
        prevScissor.x,
        prevScissor.y,
        prevScissor.z,
        prevScissor.w
      );
      sceneRenderer.setScissorTest(prevScissorTest);

      sceneControls.update();
      render();
      requestAnimationFrame(loop);
    };

    requestAnimationFrame(loop);
    return () => {
      running = false;
    };
  }, [engineReady, render, timestamp]);

  const earthValue = useMemo(
    () => ({
      engineReady,
      hostRef,
      rendererRef,
      sceneRef,
      cameraRef,
      controlsRef,
      globeRef,
      ambientLightRef,
      moistureKeyLightRef,
      moistureHeadLightRef,
      timestamp,
      registerLayer,
      unregisterLayer,
      signalLayerReady,
      allLayersReady,
      registerFramePass,
      unregisterFramePass,
      registerDebugAdapter,
      unregisterDebugAdapter,
      zoom01,
      setZoom01,
    }),
    [
      allLayersReady,
      engineReady,
      registerFramePass,
      registerDebugAdapter,
      registerLayer,
      setZoom01,
      signalLayerReady,
      timestamp,
      unregisterDebugAdapter,
      unregisterFramePass,
      unregisterLayer,
      zoom01,
    ]
  );

  return (
    <EarthContext.Provider value={earthValue}>
      <div ref={hostRef} style={{ position: "absolute", inset: 0 }}>
        {engineReady ? children : null}
      </div>
    </EarthContext.Provider>
  );
}
