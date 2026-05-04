import { useEffect, useRef } from "react";
import * as THREE from "three";
import {
  useControls,
  type TemperatureSliceVariant,
} from "../../state/controlsStore";
import {
  FLAT_EARTH_MAP_HEIGHT,
  FLAT_EARTH_MAP_WIDTH,
} from "../utils/EarthUtils";
import {
  fetchTemperatureSliceFrame,
  fetchTemperatureSliceManifest,
  temperatureSliceBorderTextureUrl,
  temperatureSliceImageUrl,
  type TemperatureSliceLevelEntry,
  type TemperatureSliceManifest,
  type TemperatureSliceManifestTimestamp,
} from "../utils/temperatureSliceAssets";
import {
  configureDataTexture,
  loadDataTextureFromApi,
} from "./shaderUtils";
import { useEarthLayer } from "./EarthBase";

const LAYER_LIFT = 0.5;
const VERTICAL_SPAN = 4.5;

type TemperatureSliceLayerParams =
  ReturnType<typeof useControls.getState>["temperatureSliceLayer"];
type TemperatureRangeK = { min: number; max: number };

function pressureToStandardAtmosphereHeightM(pressureHpa: number) {
  const safePressure = Math.max(pressureHpa, 1);
  return 44330.0 * (1.0 - (safePressure / 1013.25) ** 0.1903);
}

function pressureToFlatHeight(pressureHpa: number, verticalExaggeration: number) {
  const lowerHeight = pressureToStandardAtmosphereHeightM(1000);
  const upperHeight = pressureToStandardAtmosphereHeightM(250);
  const mix = THREE.MathUtils.clamp(
    (pressureToStandardAtmosphereHeightM(pressureHpa) - lowerHeight) /
      Math.max(upperHeight - lowerHeight, 1e-6),
    0,
    1
  );
  return LAYER_LIFT + mix * VERTICAL_SPAN * verticalExaggeration;
}

function applyLayerParams(
  mesh: THREE.Mesh,
  material: THREE.ShaderMaterial,
  params: TemperatureSliceLayerParams,
  verticalExaggeration: number
) {
  material.uniforms.uLayerOpacity.value = params.opacity;
  material.uniforms.uColorScaleMode.value =
    params.colorScaleMode === "perLevel" ? 1.0 : 0.0;
  mesh.position.y = pressureToFlatHeight(params.pressureHpa, verticalExaggeration);
}

function createTransparentTexture() {
  const texture = new THREE.DataTexture(
    new Uint8Array([0, 0, 0, 0]),
    1,
    1,
    THREE.RGBAFormat
  );
  texture.needsUpdate = true;
  return texture;
}

function pressurePairFor(
  entry: TemperatureSliceManifestTimestamp,
  pressureHpa: number
) {
  const sortedLevels = [...entry.levels].sort(
    (a, b) => a.pressure_hpa - b.pressure_hpa
  );
  const clampedPressure = THREE.MathUtils.clamp(
    pressureHpa,
    sortedLevels[0].pressure_hpa,
    sortedLevels[sortedLevels.length - 1].pressure_hpa
  );

  let lower = sortedLevels[0];
  let upper = sortedLevels[sortedLevels.length - 1];
  for (let index = 0; index < sortedLevels.length - 1; index += 1) {
    const candidateLower = sortedLevels[index];
    const candidateUpper = sortedLevels[index + 1];
    if (
      candidateLower.pressure_hpa <= clampedPressure &&
      clampedPressure <= candidateUpper.pressure_hpa
    ) {
      lower = candidateLower;
      upper = candidateUpper;
      break;
    }
  }

  const span = Math.max(upper.pressure_hpa - lower.pressure_hpa, 1e-6);
  return {
    lower,
    upper,
    mix: THREE.MathUtils.clamp((clampedPressure - lower.pressure_hpa) / span, 0, 1),
  };
}

function textureCacheKey(
  variant: TemperatureSliceVariant,
  level: TemperatureSliceLevelEntry
) {
  return `${variant}:${level.pressure_hpa}:${level.image}`;
}

function levelRangeToEncodedRange(
  level: TemperatureSliceLevelEntry,
  globalRange: TemperatureRangeK | null
) {
  if (!globalRange) {
    return new THREE.Vector2(0, 1);
  }
  const globalSpan = Math.max(globalRange.max - globalRange.min, 1e-6);
  return new THREE.Vector2(
    THREE.MathUtils.clamp(
      (level.temperature_min_k - globalRange.min) / globalSpan,
      0,
      1
    ),
    THREE.MathUtils.clamp(
      (level.temperature_max_k - globalRange.min) / globalSpan,
      0,
      1
    )
  );
}

function isSaturationEncodedManifest(manifest: TemperatureSliceManifest) {
  return (
    manifest.rendering.encoding ===
    "raw-temperature-uint16-rg-saturation-strength-b"
  );
}

export default function TemperatureSliceLayer() {
  const {
    engineReady,
    sceneRef,
    projectionMode,
    signalReady,
    timestamp,
  } = useEarthLayer("temperature-slice");

  const layerState = useControls((state) => state.temperatureSliceLayer);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const pendingRef = useRef<TemperatureSliceLayerParams | null>(null);
  const hasContentRef = useRef(false);
  const frameEntryRef = useRef<TemperatureSliceManifestTimestamp | null>(null);
  const frameVariantRef = useRef<TemperatureSliceVariant | null>(null);
  const globalTemperatureRangeRef = useRef<TemperatureRangeK | null>(null);
  const textureCacheRef = useRef(new Map<string, THREE.Texture>());
  const texturePromiseRef = useRef(new Map<string, Promise<THREE.Texture>>());
  const borderTextureUrlRef = useRef<string | null>(null);
  const reqIdRef = useRef(0);

  useEffect(() => {
    if (!engineReady) return;
    if (projectionMode !== "flat2d") {
      signalReady(timestamp);
      return;
    }
    if (!sceneRef.current) return;

    const scene = sceneRef.current;
    const state = useControls.getState();
    const textureCache = textureCacheRef.current;
    const texturePromises = texturePromiseRef.current;
    pendingRef.current = state.temperatureSliceLayer;

    const geometry = new THREE.PlaneGeometry(
      FLAT_EARTH_MAP_WIDTH,
      FLAT_EARTH_MAP_HEIGHT,
      1,
      1
    );
    geometry.rotateX(-Math.PI / 2);

    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.DoubleSide,
      uniforms: {
        uTexA: { value: null as THREE.Texture | null },
        uTexB: { value: null as THREE.Texture | null },
        uBorderTex: { value: createTransparentTexture() },
        uBorderTexel: { value: new THREE.Vector2(1 / 1440, 1 / 713) },
        uTexALevelRange: { value: new THREE.Vector2(0, 1) },
        uTexBLevelRange: { value: new THREE.Vector2(0, 1) },
        uPressureMix: { value: 0.0 },
        uLayerOpacity: { value: state.temperatureSliceLayer.opacity },
        uColorScaleMode: {
          value:
            state.temperatureSliceLayer.colorScaleMode === "perLevel" ? 1.0 : 0.0,
        },
        uSaturationMode: { value: 0.0 },
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
        uniform sampler2D uBorderTex;
        uniform vec2 uBorderTexel;
        uniform vec2 uTexALevelRange;
        uniform vec2 uTexBLevelRange;
        uniform float uPressureMix;
        uniform float uLayerOpacity;
        uniform float uColorScaleMode;
        uniform float uSaturationMode;

        varying vec2 vUv;

        float decodePackedTemperature(vec4 sampleColor) {
          float highByte = floor(sampleColor.r * 255.0 + 0.5);
          float lowByte = floor(sampleColor.g * 255.0 + 0.5);
          return (highByte * 256.0 + lowByte) / 65535.0;
        }

        vec3 temperatureColor(float t) {
          t = clamp(t, 0.0, 1.0);
          vec3 deepBlue = vec3(0.055, 0.165, 0.780);
          vec3 cyan = vec3(0.090, 0.725, 0.940);
          vec3 white = vec3(0.965, 0.970, 0.940);
          vec3 orange = vec3(1.000, 0.610, 0.120);
          vec3 red = vec3(0.880, 0.035, 0.020);

          if (t < 0.25) {
            return mix(deepBlue, cyan, smoothstep(0.0, 0.25, t));
          }
          if (t < 0.50) {
            return mix(cyan, white, smoothstep(0.25, 0.50, t));
          }
          if (t < 0.75) {
            return mix(white, orange, smoothstep(0.50, 0.75, t));
          }
          return mix(orange, red, smoothstep(0.75, 1.0, t));
        }

        float borderAlpha(vec2 uv) {
          vec2 clampedUv = clamp(uv, vec2(0.0), vec2(1.0));
          return texture2D(uBorderTex, clampedUv).a;
        }

        void main() {
          vec4 sampleA = texture2D(uTexA, vUv);
          vec4 sampleB = texture2D(uTexB, vUv);
          float valueA = decodePackedTemperature(sampleA);
          float valueB = decodePackedTemperature(sampleB);
          float levelValueA = clamp(
            (valueA - uTexALevelRange.x) /
              max(uTexALevelRange.y - uTexALevelRange.x, 0.000001),
            0.0,
            1.0
          );
          float levelValueB = clamp(
            (valueB - uTexBLevelRange.x) /
              max(uTexBLevelRange.y - uTexBLevelRange.x, 0.000001),
            0.0,
            1.0
          );
          float globalValue = mix(valueA, valueB, clamp(uPressureMix, 0.0, 1.0));
          float levelValue = mix(levelValueA, levelValueB, clamp(uPressureMix, 0.0, 1.0));
          float value = mix(globalValue, levelValue, step(0.5, uColorScaleMode));
          vec3 color = temperatureColor(value);
          float encodedStrength = mix(sampleA.b, sampleB.b, clamp(uPressureMix, 0.0, 1.0));
          float saturation = mix(1.0, 0.18 + 0.92 * encodedStrength, step(0.5, uSaturationMode));
          float luminance = dot(color, vec3(0.299, 0.587, 0.114));
          color = mix(vec3(luminance), color, clamp(saturation, 0.0, 1.0));
          float border = borderAlpha(vUv);
          border = max(border, borderAlpha(vUv + vec2(uBorderTexel.x, 0.0)));
          border = max(border, borderAlpha(vUv - vec2(uBorderTexel.x, 0.0)));
          border = max(border, borderAlpha(vUv + vec2(0.0, uBorderTexel.y)));
          border = max(border, borderAlpha(vUv - vec2(0.0, uBorderTexel.y)));
          border = max(border, borderAlpha(vUv + uBorderTexel));
          border = max(border, borderAlpha(vUv - uBorderTexel));
          float borderStrength = smoothstep(0.02, 0.36, border);
          color = mix(color, vec3(0.015, 0.018, 0.022), borderStrength * 0.92);
          gl_FragColor = vec4(color, clamp(uLayerOpacity, 0.0, 1.0));
        }
      `,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = "temperature-slice-layer";
    mesh.renderOrder = 62;
    mesh.frustumCulled = false;
    mesh.visible = state.temperatureSliceLayer.visible && hasContentRef.current;
    applyLayerParams(mesh, material, state.temperatureSliceLayer, state.verticalExaggeration);

    scene.add(mesh);
    meshRef.current = mesh;

    return () => {
      meshRef.current = null;
      frameEntryRef.current = null;
      frameVariantRef.current = null;
      globalTemperatureRangeRef.current = null;
      borderTextureUrlRef.current = null;
      mesh.removeFromParent();
      geometry.dispose();
      const uniforms = material.uniforms as {
        uBorderTex?: { value: THREE.Texture | null };
      };
      uniforms.uBorderTex?.value?.dispose();
      for (const texture of textureCache.values()) {
        texture.dispose();
      }
      textureCache.clear();
      texturePromises.clear();
      material.dispose();
    };
  }, [engineReady, projectionMode, sceneRef, signalReady, timestamp]);

  const applyCachedPressureTextures = (
    material: THREE.ShaderMaterial,
    variant: TemperatureSliceVariant,
    pressureHpa: number
  ) => {
    const entry = frameEntryRef.current;
    if (!entry) return false;
    if (frameVariantRef.current !== variant) return false;

    const pair = pressurePairFor(entry, pressureHpa);
    const lowerTexture = textureCacheRef.current.get(
      textureCacheKey(variant, pair.lower)
    );
    const upperTexture = textureCacheRef.current.get(
      textureCacheKey(variant, pair.upper)
    );
    if (!lowerTexture || !upperTexture) return false;

    material.uniforms.uTexA.value = lowerTexture;
    material.uniforms.uTexB.value = upperTexture;
    material.uniforms.uTexALevelRange.value.copy(
      levelRangeToEncodedRange(pair.lower, globalTemperatureRangeRef.current)
    );
    material.uniforms.uTexBLevelRange.value.copy(
      levelRangeToEncodedRange(pair.upper, globalTemperatureRangeRef.current)
    );
    material.uniforms.uPressureMix.value = pair.mix;
    return true;
  };

  const loadTemperatureTexture = (
    variant: TemperatureSliceVariant,
    level: TemperatureSliceLevelEntry
  ) => {
    const key = textureCacheKey(variant, level);
    const cached = textureCacheRef.current.get(key);
    if (cached) return Promise.resolve(cached);

    const existing = texturePromiseRef.current.get(key);
    if (existing) return existing;

    const promise = loadDataTextureFromApi({
      url: temperatureSliceImageUrl(variant, level),
      fallbackMessage: "Failed to load temperature slice texture.",
      layerLabel: "Temperature slice",
    }).then((texture) => {
      configureDataTexture(texture, {
        wrapS: THREE.ClampToEdgeWrapping,
        wrapT: THREE.ClampToEdgeWrapping,
      });
      textureCacheRef.current.set(key, texture);
      texturePromiseRef.current.delete(key);
      return texture;
    });

    texturePromiseRef.current.set(key, promise);
    return promise;
  };

  useEffect(() => {
    if (!engineReady) return;
    const mesh = meshRef.current;
    if (!mesh) return;

    pendingRef.current = useControls.getState().temperatureSliceLayer;
    const unsubscribe = useControls.subscribe(
      (state) => ({
        layer: state.temperatureSliceLayer,
        verticalExaggeration: state.verticalExaggeration,
      }),
      ({ layer, verticalExaggeration: nextVerticalExaggeration }) => {
        pendingRef.current = layer;
        mesh.visible = layer.visible && hasContentRef.current;
        applyLayerParams(
          mesh,
          mesh.material as THREE.ShaderMaterial,
          layer,
          nextVerticalExaggeration
        );
        if (layer.visible && hasContentRef.current) {
          applyCachedPressureTextures(
            mesh.material as THREE.ShaderMaterial,
            layer.variant,
            layer.pressureHpa
          );
        }
      }
    );

    return () => unsubscribe();
  }, [engineReady]);

  useEffect(() => {
    if (!engineReady) return;
    if (projectionMode !== "flat2d") {
      signalReady(timestamp);
      return;
    }

    const mesh = meshRef.current;
    if (!mesh) return;
    const material = mesh.material as THREE.ShaderMaterial;

    let cancelled = false;
    const requestId = ++reqIdRef.current;
    const isCancelled = () => cancelled || requestId !== reqIdRef.current;

    if (!layerState.visible) {
      mesh.visible = false;
      signalReady(timestamp);
      return () => {
        cancelled = true;
      };
    }

    if (
      hasContentRef.current &&
      applyCachedPressureTextures(
        material,
        layerState.variant,
        layerState.pressureHpa
      )
    ) {
      signalReady(timestamp);
      return () => {
        cancelled = true;
      };
    }

    void fetchTemperatureSliceFrame(timestamp, layerState.pressureHpa, {
      variant: layerState.variant,
    })
      .then((frame) => {
        frameEntryRef.current = frame.entry;
        frameVariantRef.current = layerState.variant;
        globalTemperatureRangeRef.current = frame.manifest.temperature_range_k;
        material.uniforms.uSaturationMode.value = isSaturationEncodedManifest(
          frame.manifest
        )
          ? 1.0
          : 0.0;

        return Promise.all([
          loadTemperatureTexture(layerState.variant, frame.pressurePair.lower),
          loadTemperatureTexture(layerState.variant, frame.pressurePair.upper),
        ]).then(() => ({ frame }));
      })
      .then(async (payload) => {
        if (isCancelled()) {
          return;
        }

        const latest =
          pendingRef.current ?? useControls.getState().temperatureSliceLayer;
        applyLayerParams(mesh, material, latest, useControls.getState().verticalExaggeration);
        applyCachedPressureTextures(material, latest.variant, latest.pressureHpa);

        hasContentRef.current = true;
        mesh.visible = latest.visible;
        material.uniforms.uLayerOpacity.value = latest.opacity;

        const borderUrl = temperatureSliceBorderTextureUrl(
          latest.variant,
          payload.frame.manifest
        );
        if (borderUrl && borderTextureUrlRef.current !== borderUrl) {
          const borderTexture = await loadDataTextureFromApi({
            url: borderUrl,
            fallbackMessage: "Failed to load temperature slice border texture.",
            layerLabel: "Temperature slice borders",
            notifyOnError: false,
          });
          if (isCancelled()) {
            borderTexture.dispose();
            return;
          }
          configureDataTexture(borderTexture, {
            wrapS: THREE.ClampToEdgeWrapping,
            wrapT: THREE.ClampToEdgeWrapping,
          });
          const borderImage = borderTexture.image as
            | { width?: number; height?: number }
            | undefined;
          if (borderImage?.width && borderImage.height) {
            material.uniforms.uBorderTexel.value.set(
              1 / borderImage.width,
              1 / borderImage.height
            );
          }
          const previous = material.uniforms.uBorderTex.value as THREE.Texture | null;
          material.uniforms.uBorderTex.value = borderTexture;
          material.needsUpdate = true;
          borderTextureUrlRef.current = borderUrl;
          previous?.dispose();
        }

        void fetchTemperatureSliceManifest({
          notifyOnError: false,
          variant: latest.variant,
        }).then((manifest) => {
          const entry = manifest.timestamps.find(
            (candidate) => candidate.timestamp === payload.frame.entry.timestamp
          );
          if (!entry || isCancelled()) return;
          void Promise.allSettled(
            entry.levels.map((level) => loadTemperatureTexture(latest.variant, level))
          );
        });

        signalReady(timestamp);
      })
      .catch((error) => {
        if (isCancelled()) return;
        console.error("Failed to load temperature slice layer", error);
        if (!hasContentRef.current) {
          mesh.visible = false;
        }
        signalReady(timestamp);
      });

    return () => {
      cancelled = true;
    };
  }, [
    engineReady,
    layerState.pressureHpa,
    layerState.variant,
    layerState.visible,
    projectionMode,
    signalReady,
    timestamp,
  ]);

  return null;
}
