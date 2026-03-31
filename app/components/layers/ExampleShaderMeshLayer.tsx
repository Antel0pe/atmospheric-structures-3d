import { useEffect, useRef } from "react";
import * as THREE from "three";
import { useEarthLayer } from "./EarthBase";
import { exampleShaderMeshLayerApiUrl } from "../utils/ApiResponses";
import {
  ExampleShaderMeshPressure,
  useControls,
} from "../../state/controlsStore";
import {
  animateUniform,
  configureDataTexture,
  crossfadeTextureUniforms,
  disposeCrossfadeTextures,
  loadDataTextureFromApi,
} from "./shaderUtils";

const SUPPORTED_LEVELS = [250, 500, 925] as const;
type SupportedLevel = (typeof SUPPORTED_LEVELS)[number];
type DataRange = { min: number; max: number };

type ExampleShaderMeshParams =
  ReturnType<typeof useControls.getState>["exampleShaderMeshLayer"];

function defaultRangeForLevel(level: SupportedLevel): DataRange {
  if (level === 250) {
    return { min: -0.0005787129048258066, max: 0.0010109632275998592 };
  }
  if (level === 500) {
    return { min: -0.0005457177758216858, max: 0.0009189110714942217 };
  }
  return { min: -0.0011868530418723822, max: 0.0008237080182880163 };
}

function resolveLevel(
  pressure: ExampleShaderMeshPressure
): SupportedLevel | null {
  if (pressure === "none") return null;
  return SUPPORTED_LEVELS.includes(pressure) ? pressure : 250;
}

function applyExampleShaderMeshDisplayParams(
  material: THREE.ShaderMaterial,
  params: ExampleShaderMeshParams
) {
  material.uniforms.uDisplayMin.value = params.uValueMin;
  material.uniforms.uDisplayMax.value = params.uValueMax;
  material.uniforms.uGamma.value = params.uGamma;
  material.uniforms.uAlpha.value = params.uAlpha;
  material.uniforms.uZeroEps.value = params.uZeroEps;
  material.uniforms.uAsinhK.value = params.uAsinhK;
}

function applyExampleShaderMeshDecodeRange(
  material: THREE.ShaderMaterial,
  slot: "A" | "B",
  range: DataRange
) {
  if (slot === "A") {
    material.uniforms.uDataMinA.value = range.min;
    material.uniforms.uDataMaxA.value = range.max;
    return;
  }

  material.uniforms.uDataMinB.value = range.min;
  material.uniforms.uDataMaxB.value = range.max;
}

export default function ExampleShaderMeshLayer() {
  const { engineReady, sceneRef, globeRef, timestamp, signalReady } =
    useEarthLayer("example-shader-mesh");

  const pressureLevel = useControls(
    (state) => state.exampleShaderMeshLayer.pressureLevel
  );

  const meshRef = useRef<THREE.Mesh | null>(null);
  const reqIdRef = useRef(0);
  const pendingRef = useRef<ExampleShaderMeshParams | null>(null);
  const hasContentRef = useRef(false);

  useEffect(() => {
    if (!engineReady) return;
    if (!sceneRef.current || !globeRef.current) return;

    const scene = sceneRef.current;
    const state = useControls.getState();
    pendingRef.current = state.exampleShaderMeshLayer;

    const radius = 100;
    const lift = radius * 0.0024;
    const geometry = new THREE.SphereGeometry(radius + lift, 128, 128);

    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: true,
      uniforms: {
        uTexA: { value: null as THREE.Texture | null },
        uTexB: { value: null as THREE.Texture | null },
        uMix: { value: 0.0 },
        uLonOffset: { value: 0.25 },
        uDataMinA: { value: 0 },
        uDataMaxA: { value: 1 },
        uDataMinB: { value: 0 },
        uDataMaxB: { value: 1 },
        uDisplayMin: { value: state.exampleShaderMeshLayer.uValueMin },
        uDisplayMax: { value: state.exampleShaderMeshLayer.uValueMax },
        uGamma: { value: state.exampleShaderMeshLayer.uGamma },
        uAlpha: { value: state.exampleShaderMeshLayer.uAlpha },
        uZeroEps: { value: state.exampleShaderMeshLayer.uZeroEps },
        uAsinhK: { value: state.exampleShaderMeshLayer.uAsinhK },
        uLayerOpacity: { value: 1.0 },
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
uniform float uDataMinA;
uniform float uDataMaxA;
uniform float uDataMinB;
uniform float uDataMaxB;

uniform float uDisplayMin;
uniform float uDisplayMax;

uniform float uGamma;
uniform float uAlpha;

uniform float uZeroEps;
uniform float uAsinhK;

uniform float uLayerOpacity;

varying vec2 vUv;

vec3 WARM = vec3(1.00, 0.85, 0.10);
vec3 COOL = vec3(0.12, 0.78, 0.28);
vec3 NEU  = vec3(0.86, 0.90, 1.00);

float magMap(float m) {
  if (uAsinhK > 1e-6) {
    float k = uAsinhK;
    m = asinh(k * m) / asinh(k);
  }
  return pow(m, max(uGamma, 1e-6));
}

float saturateFast(float m) {
  m = clamp(m, 0.0, 1.0);
  float p = 3.5;
  return 1.0 - pow(1.0 - m, p);
}

void main() {
  vec2 uv = vUv;
  uv.x = fract(uv.x + uLonOffset);

  float xA = texture2D(uTexA, uv).r;
  float xB = texture2D(uTexB, uv).r;
  float valueA = mix(uDataMinA, uDataMaxA, xA);
  float valueB = mix(uDataMinB, uDataMaxB, xB);
  float value = mix(valueA, valueB, clamp(uMix, 0.0, 1.0));

  float v = clamp(value, uDisplayMin, uDisplayMax);
  float scale = max(abs(uDisplayMin), abs(uDisplayMax));
  scale = max(scale, 1e-12);

  float z = clamp(v / scale, -1.0, 1.0);
  float m0 = abs(z);
  float m = magMap(m0);
  float s = saturateFast(m);

  float eps = max(uZeroEps, 1e-6);
  float near0 = smoothstep(eps, eps * 2.0, m0);

  float a = s * near0 * clamp(uAlpha, 0.0, 1.0);
  a *= clamp(uLayerOpacity, 0.0, 1.0);

  vec3 signCol = (z >= 0.0) ? WARM : COOL;
  vec3 col = mix(NEU, signCol, s);

  gl_FragColor = vec4(col, a);
}
      `,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = "example-shader-mesh-layer";
    mesh.renderOrder = 57;
    mesh.frustumCulled = false;
    mesh.visible =
      state.exampleShaderMeshLayer.pressureLevel !== "none" &&
      hasContentRef.current;

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

    pendingRef.current = useControls.getState().exampleShaderMeshLayer;

    const unsubscribe = useControls.subscribe(
      (state) => state.exampleShaderMeshLayer,
      (params) => {
        pendingRef.current = params;
        mesh.visible = params.pressureLevel !== "none" && hasContentRef.current;
        if (hasContentRef.current) {
          applyExampleShaderMeshDisplayParams(
            mesh.material as THREE.ShaderMaterial,
            params
          );
        }
      }
    );

    return () => unsubscribe();
  }, [engineReady]);

  useEffect(() => {
    if (!engineReady) return;
    const mesh = meshRef.current;
    if (!mesh) return;

    const material = mesh.material as THREE.ShaderMaterial;

    let cancelled = false;
    const requestId = ++reqIdRef.current;
    const isCancelled = () => cancelled || requestId !== reqIdRef.current;

    const level = resolveLevel(pressureLevel);
    if (level === null) {
      hasContentRef.current = false;
      mesh.visible = false;
      disposeCrossfadeTextures(material);
      material.uniforms.uTexA.value = null;
      material.uniforms.uTexB.value = null;
      material.uniforms.uMix.value = 0.0;
      material.uniforms.uLayerOpacity.value = 0.0;
      material.needsUpdate = true;
      signalReady(timestamp);
      return () => {
        cancelled = true;
      };
    }

    mesh.visible = hasContentRef.current;

    const url = exampleShaderMeshLayerApiUrl(timestamp, level);
    void loadDataTextureFromApi({
      url,
      fallbackMessage: "Failed to load example shader mesh data.",
      layerLabel: `Example shader mesh (${level} hPa)`,
    })
      .then((texture) => {
        if (isCancelled()) {
          texture.dispose();
          return;
        }

        configureDataTexture(texture);

        const latest =
          pendingRef.current ?? useControls.getState().exampleShaderMeshLayer;
        const nextRange = defaultRangeForLevel(level);
        applyExampleShaderMeshDisplayParams(material, latest);

        const hadVisibleContent = hasContentRef.current;
        hasContentRef.current = true;
        mesh.visible = true;

        if (!hadVisibleContent) {
          disposeCrossfadeTextures(material);
          applyExampleShaderMeshDecodeRange(material, "A", nextRange);
          applyExampleShaderMeshDecodeRange(material, "B", nextRange);
          material.uniforms.uTexA.value = texture;
          material.uniforms.uTexB.value = texture;
          material.uniforms.uMix.value = 0.0;
          material.uniforms.uLayerOpacity.value = 0.0;
          material.needsUpdate = true;

          animateUniform(material, "uLayerOpacity", 0.0, 1.0, 220, isCancelled);
          signalReady(timestamp);
          return;
        }

        applyExampleShaderMeshDecodeRange(material, "B", nextRange);
        material.uniforms.uLayerOpacity.value = 1.0;
        crossfadeTextureUniforms({
          material,
          nextTexture: texture,
          isCancelled,
          onPromote: () => {
            applyExampleShaderMeshDecodeRange(material, "A", nextRange);
            applyExampleShaderMeshDecodeRange(material, "B", nextRange);
          },
        });
        signalReady(timestamp);
      })
      .catch((error) => {
        if (isCancelled()) return;
        console.error("Failed to load example shader mesh png", error);
        if (!hasContentRef.current) {
          mesh.visible = false;
        }
        signalReady(timestamp);
      });

    return () => {
      cancelled = true;
    };
  }, [engineReady, pressureLevel, signalReady, timestamp]);

  return null;
}
