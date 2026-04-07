import { useEffect, useRef } from "react";
import * as THREE from "three";
import { useEarthLayer } from "./EarthBase";
import { useControls } from "../../state/controlsStore";
import {
  fetchPrecipitationRadarFrame,
  precipitationRadarImageUrl,
} from "../utils/precipitationRadarAssets";
import {
  animateUniform,
  configureDataTexture,
  crossfadeTextureUniforms,
  disposeCrossfadeTextures,
  loadDataTextureFromApi,
} from "./shaderUtils";

const LAYER_LIFT = 0.32;
const DEFAULT_FADE_MS = 220;

type PrecipitationLayerParams =
  ReturnType<typeof useControls.getState>["precipitationRadarLayer"];

function applyLayerParams(
  material: THREE.ShaderMaterial,
  params: PrecipitationLayerParams,
) {
  material.uniforms.uLayerOpacity.value = params.opacity;
}

export default function PrecipitationRadarLayer() {
  const { engineReady, sceneRef, globeRef, signalReady, timestamp } =
    useEarthLayer("precipitation-radar");

  const layerState = useControls((state) => state.precipitationRadarLayer);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const pendingRef = useRef<PrecipitationLayerParams | null>(null);
  const hasContentRef = useRef(false);
  const reqIdRef = useRef(0);

  useEffect(() => {
    if (!engineReady) return;
    if (!sceneRef.current || !globeRef.current) return;

    const scene = sceneRef.current;
    const state = useControls.getState();
    pendingRef.current = state.precipitationRadarLayer;

    const geometry = new THREE.SphereGeometry(100 + LAYER_LIFT, 128, 128);
    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: true,
      uniforms: {
        uTexA: { value: null as THREE.Texture | null },
        uTexB: { value: null as THREE.Texture | null },
        uMix: { value: 0.0 },
        uLonOffset: { value: 0.25 },
        uMaxMm: { value: 20.0 },
        uThresholdsMm: {
          value: [
            0.2,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
            20.0,
          ],
        },
        uLayerOpacity: { value: state.precipitationRadarLayer.opacity },
      },
      vertexShader: `
        varying vec2 vUv;

        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D uTexA;
        uniform sampler2D uTexB;
        uniform float uMix;
        uniform float uLonOffset;
        uniform float uMaxMm;
        uniform float uThresholdsMm[7];
        uniform float uLayerOpacity;

        varying vec2 vUv;

        float decodeMm(float encodedValue) {
          float normalized = encodedValue * encodedValue;
          return normalized * uMaxMm;
        }

        vec4 bandColor(float mm) {
          if (mm < uThresholdsMm[0]) {
            return vec4(0.0);
          }
          if (mm < uThresholdsMm[1]) {
            return vec4(0.52, 0.94, 0.72, 0.34);
          }
          if (mm < uThresholdsMm[2]) {
            return vec4(0.10, 0.82, 0.36, 0.42);
          }
          if (mm < uThresholdsMm[3]) {
            return vec4(0.14, 0.66, 0.96, 0.54);
          }
          if (mm < uThresholdsMm[4]) {
            return vec4(0.98, 0.88, 0.22, 0.68);
          }
          if (mm < uThresholdsMm[5]) {
            return vec4(0.98, 0.52, 0.18, 0.80);
          }
          if (mm < uThresholdsMm[6]) {
            return vec4(0.90, 0.18, 0.16, 0.88);
          }
          return vec4(0.84, 0.22, 0.88, 0.94);
        }

        void main() {
          vec2 uv = vUv;
          uv.x = fract(uv.x + uLonOffset);

          float encodedA = texture2D(uTexA, uv).r;
          float encodedB = texture2D(uTexB, uv).r;
          float encoded = mix(encodedA, encodedB, clamp(uMix, 0.0, 1.0));
          float mm = decodeMm(encoded);
          vec4 color = bandColor(mm);
          gl_FragColor = vec4(color.rgb, color.a * clamp(uLayerOpacity, 0.0, 1.0));
        }
      `,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = "precipitation-radar-layer";
    mesh.renderOrder = 58;
    mesh.frustumCulled = false;
    mesh.visible = state.precipitationRadarLayer.visible && hasContentRef.current;

    scene.add(mesh);
    meshRef.current = mesh;

    return () => {
      meshRef.current = null;
      mesh.removeFromParent();
      geometry.dispose();
      disposeCrossfadeTextures(material);
      material.dispose();
    };
  }, [engineReady, globeRef, sceneRef]);

  useEffect(() => {
    if (!engineReady) return;
    const mesh = meshRef.current;
    if (!mesh) return;

    pendingRef.current = useControls.getState().precipitationRadarLayer;
    const unsubscribe = useControls.subscribe(
      (state) => state.precipitationRadarLayer,
      (params) => {
        pendingRef.current = params;
        mesh.visible = params.visible && hasContentRef.current;
        applyLayerParams(mesh.material as THREE.ShaderMaterial, params);
      }
    );

    return () => unsubscribe();
  }, [engineReady]);

  useEffect(() => {
    if (!engineReady) return;
    const mesh = meshRef.current;
    if (!mesh) return;

    const material = mesh.material as THREE.ShaderMaterial;
    const visible = layerState.visible;
    let cancelled = false;
    const requestId = ++reqIdRef.current;
    const isCancelled = () => cancelled || requestId !== reqIdRef.current;

    if (!visible) {
      mesh.visible = false;
      signalReady(timestamp);
      return () => {
        cancelled = true;
      };
    }

    void fetchPrecipitationRadarFrame(timestamp)
      .then((frame) =>
        loadDataTextureFromApi({
          url: precipitationRadarImageUrl(frame.entry),
          fallbackMessage: "Failed to load precipitation radar texture.",
          layerLabel: "Precipitation radar",
        }).then((texture) => ({ frame, texture }))
      )
      .then(({ frame, texture }) => {
        if (isCancelled()) {
          texture.dispose();
          return;
        }

        configureDataTexture(texture);
        material.uniforms.uMaxMm.value = frame.manifest.encoding.max_mm;
        material.uniforms.uThresholdsMm.value = frame.manifest.thresholds_mm;
        applyLayerParams(
          material,
          pendingRef.current ?? useControls.getState().precipitationRadarLayer
        );

        const hadVisibleContent = hasContentRef.current;
        hasContentRef.current = true;
        mesh.visible = true;

        if (!hadVisibleContent) {
          const targetOpacity =
            pendingRef.current?.opacity ??
            useControls.getState().precipitationRadarLayer.opacity;
          disposeCrossfadeTextures(material);
          material.uniforms.uTexA.value = texture;
          material.uniforms.uTexB.value = texture;
          material.uniforms.uMix.value = 0.0;
          material.needsUpdate = true;
          animateUniform(
            material,
            "uLayerOpacity",
            0.0,
            targetOpacity,
            DEFAULT_FADE_MS,
            isCancelled
          );
          signalReady(timestamp);
          return;
        }

        crossfadeTextureUniforms({
          material,
          nextTexture: texture,
          isCancelled,
        });
        signalReady(timestamp);
      })
      .catch((error) => {
        if (isCancelled()) return;
        console.error("Failed to load precipitation radar layer", error);
        if (!hasContentRef.current) {
          mesh.visible = false;
        }
        signalReady(timestamp);
      });

    return () => {
      cancelled = true;
    };
  }, [engineReady, layerState.visible, signalReady, timestamp]);

  return null;
}
