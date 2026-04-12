import { useCallback, useEffect, useRef } from "react";
import * as THREE from "three";
import { useEarthLayer } from "./EarthBase";
import { useControls } from "../../state/controlsStore";
import {
  fetchPotentialTemperatureStructureFrame,
  type PotentialTemperatureSideMesh,
  type PotentialTemperatureStructureFrame,
} from "../utils/potentialTemperatureStructureAssets";

const COOL_LAYER_CLEARANCE = 10.8;
const HOT_LAYER_CLEARANCE = 11.2;
const COOL_RENDER_ORDER = 68;
const HOT_RENDER_ORDER = 69;
const HOT_COLOR = new THREE.Color("#ff9b6a");
const COOL_COLOR = new THREE.Color("#6a96ff");

type ThermalCutawayUniforms = {
  cutawayCenter: { value: THREE.Vector3 };
  cutawayRadius: { value: number };
  enabled: { value: number };
};

function attachThermalCutawayShader(material: THREE.MeshLambertMaterial) {
  const uniforms: ThermalCutawayUniforms = {
    cutawayCenter: { value: new THREE.Vector3() },
    cutawayRadius: { value: 0 },
    enabled: { value: 1 },
  };

  material.userData.thermalCutawayUniforms = uniforms;
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uThermalCutawayCenter = uniforms.cutawayCenter;
    shader.uniforms.uThermalCutawayRadius = uniforms.cutawayRadius;
    shader.uniforms.uThermalCutawayEnabled = uniforms.enabled;

    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        `#include <common>
varying vec3 vThermalWorldPosition;`
      )
      .replace(
        "#include <worldpos_vertex>",
        `#include <worldpos_vertex>
vec4 thermalWorldPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
thermalWorldPosition = batchingMatrix * thermalWorldPosition;
#endif
#ifdef USE_INSTANCING
thermalWorldPosition = instanceMatrix * thermalWorldPosition;
#endif
thermalWorldPosition = modelMatrix * thermalWorldPosition;
vThermalWorldPosition = thermalWorldPosition.xyz;`
      );

    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
uniform vec3 uThermalCutawayCenter;
uniform float uThermalCutawayRadius;
uniform float uThermalCutawayEnabled;
varying vec3 vThermalWorldPosition;`
      )
      .replace(
        "#include <clipping_planes_fragment>",
        `#include <clipping_planes_fragment>
if ( uThermalCutawayEnabled > 0.5 ) {
  vec3 cutawayDelta = vThermalWorldPosition - uThermalCutawayCenter;
  if ( dot( cutawayDelta, cutawayDelta ) < uThermalCutawayRadius * uThermalCutawayRadius ) {
    discard;
  }
}`
      );
  };
  material.customProgramCacheKey = () => "potential-temperature-cutaway-v1";
  material.needsUpdate = true;
  return uniforms;
}

function buildGeometry(
  frame: PotentialTemperatureStructureFrame,
  side: PotentialTemperatureSideMesh,
  verticalExaggeration: number,
  clearance: number
) {
  const positions = side.positions.slice();
  const indices = side.indices.slice();
  const baseRadius = frame.manifest.globe.base_radius;

  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    const radius = Math.hypot(x, y, z);
    if (radius <= 1e-6) continue;

    const radialOffset = Math.max(radius - baseRadius, 0);
    const exaggeratedRadius = baseRadius + clearance + radialOffset * verticalExaggeration;
    const scale = exaggeratedRadius / radius;
    positions[i] *= scale;
    positions[i + 1] *= scale;
    positions[i + 2] *= scale;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
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
  const {
    cameraRef,
    engineReady,
    globeRef,
    registerFramePass,
    sceneRef,
    signalReady,
    timestamp,
    unregisterFramePass,
  } = useEarthLayer("potential-temperature-structures");

  const rootRef = useRef<THREE.Group | null>(null);
  const hotMeshRef = useRef<THREE.Mesh | null>(null);
  const coolMeshRef = useRef<THREE.Mesh | null>(null);
  const hotMaterialRef = useRef<THREE.MeshLambertMaterial | null>(null);
  const coolMaterialRef = useRef<THREE.MeshLambertMaterial | null>(null);
  const frameRef = useRef<PotentialTemperatureStructureFrame | null>(null);
  const reqIdRef = useRef(0);
  const cameraPositionRef = useRef(new THREE.Vector3());

  const rebuildMeshes = useCallback(() => {
    const root = rootRef.current;
    const hotMaterial = hotMaterialRef.current;
    const coolMaterial = coolMaterialRef.current;
    const frame = frameRef.current;
    if (!root || !hotMaterial || !coolMaterial || !frame) return;

    hotMeshRef.current?.removeFromParent();
    hotMeshRef.current?.geometry.dispose();
    coolMeshRef.current?.removeFromParent();
    coolMeshRef.current?.geometry.dispose();

    if (frame.coolSide.indices.length > 0) {
      const coolGeometry = buildGeometry(
        frame,
        frame.coolSide,
        verticalExaggeration,
        COOL_LAYER_CLEARANCE
      );
      const coolMesh = new THREE.Mesh(coolGeometry, coolMaterial);
      coolMesh.name = "potential-temperature-cool-side";
      coolMesh.renderOrder = COOL_RENDER_ORDER;
      coolMesh.frustumCulled = false;
      root.add(coolMesh);
      coolMeshRef.current = coolMesh;
    } else {
      coolMeshRef.current = null;
    }

    if (frame.hotSide.indices.length > 0) {
      const hotGeometry = buildGeometry(
        frame,
        frame.hotSide,
        verticalExaggeration,
        HOT_LAYER_CLEARANCE
      );
      const hotMesh = new THREE.Mesh(hotGeometry, hotMaterial);
      hotMesh.name = "potential-temperature-hot-side";
      hotMesh.renderOrder = HOT_RENDER_ORDER;
      hotMesh.frustumCulled = false;
      root.add(hotMesh);
      hotMeshRef.current = hotMesh;
    } else {
      hotMeshRef.current = null;
    }
  }, [verticalExaggeration]);

  useEffect(() => {
    if (!engineReady) return;
    if (!sceneRef.current || !globeRef.current) return;

    const root = new THREE.Group();
    root.name = "potential-temperature-structures-root";
    root.visible = layerState.visible;
    root.renderOrder = HOT_RENDER_ORDER;
    root.frustumCulled = false;
    sceneRef.current.add(root);
    rootRef.current = root;

    const baseMaterial = {
      transparent: false,
      opacity: 1,
      depthWrite: true,
      depthTest: true,
      side: THREE.DoubleSide,
      flatShading: true,
    } as const;
    coolMaterialRef.current = new THREE.MeshLambertMaterial({
      ...baseMaterial,
      color: COOL_COLOR,
    });
    hotMaterialRef.current = new THREE.MeshLambertMaterial({
      ...baseMaterial,
      color: HOT_COLOR,
    });
    attachThermalCutawayShader(coolMaterialRef.current);
    attachThermalCutawayShader(hotMaterialRef.current);

    return () => {
      hotMeshRef.current?.geometry.dispose();
      coolMeshRef.current?.geometry.dispose();
      hotMaterialRef.current?.dispose();
      coolMaterialRef.current?.dispose();
      hotMaterialRef.current = null;
      coolMaterialRef.current = null;
      hotMeshRef.current = null;
      coolMeshRef.current = null;
      rootRef.current = null;
      root.removeFromParent();
    };
  }, [engineReady, globeRef, layerState.visible, sceneRef]);

  useEffect(() => {
    if (!engineReady) return;
    const root = rootRef.current;
    const hotMaterial = hotMaterialRef.current;
    const coolMaterial = coolMaterialRef.current;
    if (!root || !hotMaterial || !coolMaterial) return;

    root.visible = layerState.visible;
    hotMaterial.opacity = 1;
    hotMaterial.depthWrite = true;
    hotMaterial.side = THREE.DoubleSide;
    coolMaterial.opacity = 1;
    coolMaterial.depthWrite = true;
    coolMaterial.side = THREE.DoubleSide;
    if (frameRef.current) {
      rebuildMeshes();
    }
  }, [engineReady, layerState.visible, rebuildMeshes]);

  useEffect(() => {
    if (!engineReady) return;

    const applyCutawayState = () => {
      const camera = cameraRef.current;
      if (!camera) return;

      camera.getWorldPosition(cameraPositionRef.current);
      const cutawayRadius = Math.max(cameraPositionRef.current.length() - 90, 0);

      for (const material of [hotMaterialRef.current, coolMaterialRef.current]) {
        const uniforms = material?.userData
          ?.thermalCutawayUniforms as ThermalCutawayUniforms | undefined;
        if (!uniforms) continue;
        uniforms.cutawayCenter.value.copy(cameraPositionRef.current);
        uniforms.cutawayRadius.value = cutawayRadius;
        uniforms.enabled.value = 1;
      }
    };

    const framePassKey = "potential-temperature-cutaway";
    applyCutawayState();
    registerFramePass(framePassKey, applyCutawayState);
    return () => {
      unregisterFramePass(framePassKey);
    };
  }, [cameraRef, engineReady, registerFramePass, unregisterFramePass]);

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
        rebuildMeshes();
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
  }, [engineReady, layerState.visible, rebuildMeshes, signalReady, timestamp]);

  return null;
}
