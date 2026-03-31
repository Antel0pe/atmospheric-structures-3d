import * as THREE from "three";
import {
  fetchBlobOrThrow,
  notifyDataFetchError,
} from "../utils/dataFetchErrors";

export const DEFAULT_TEXTURE_FADE_MS = 220;

type TextureCrossfadeUniformNames = {
  texA?: string;
  texB?: string;
  mix?: string;
};

function getUniformRef<T>(
  material: THREE.ShaderMaterial,
  uniformName: string
): { value: T } {
  const uniform = (material.uniforms as Record<string, { value: T } | undefined>)[uniformName];
  if (!uniform) {
    throw new Error(`Missing shader uniform "${uniformName}"`);
  }
  return uniform;
}

export function configureDataTexture(
  texture: THREE.Texture,
  opts?: {
    flipY?: boolean;
    wrapS?: THREE.Wrapping;
    wrapT?: THREE.Wrapping;
  }
) {
  texture.colorSpace = THREE.NoColorSpace;
  texture.flipY = opts?.flipY ?? true;
  texture.wrapS = opts?.wrapS ?? THREE.RepeatWrapping;
  texture.wrapT = opts?.wrapT ?? THREE.RepeatWrapping;
}

export function animateUniform(
  material: THREE.ShaderMaterial,
  uniformName: string,
  from: number,
  to: number,
  ms: number,
  isCancelled: () => boolean
) {
  const uniform = getUniformRef<number>(material, uniformName);
  const start = performance.now();
  uniform.value = from;

  const step = (now: number) => {
    if (isCancelled()) return;
    const t = Math.min(1, (now - start) / Math.max(ms, 1));
    uniform.value = from + (to - from) * t;
    if (t < 1) requestAnimationFrame(step);
  };

  requestAnimationFrame(step);
}

export function crossfadeTextureUniforms(args: {
  material: THREE.ShaderMaterial;
  nextTexture: THREE.Texture;
  isCancelled: () => boolean;
  fadeMs?: number;
  onPromote?: () => void;
  uniforms?: TextureCrossfadeUniformNames;
}): number | null {
  const {
    material,
    nextTexture,
    isCancelled,
    fadeMs = DEFAULT_TEXTURE_FADE_MS,
    onPromote,
    uniforms,
  } = args;

  const texAName = uniforms?.texA ?? "uTexA";
  const texBName = uniforms?.texB ?? "uTexB";
  const mixName = uniforms?.mix ?? "uMix";

  const texAUniform = getUniformRef<THREE.Texture | null>(material, texAName);
  const texBUniform = getUniformRef<THREE.Texture | null>(material, texBName);
  const mixUniform = getUniformRef<number>(material, mixName);

  const texA = texAUniform.value;

  // First texture: keep both samplers valid.
  if (!texA) {
    texAUniform.value = nextTexture;
    texBUniform.value = nextTexture;
    mixUniform.value = 0.0;
    material.needsUpdate = true;
    return null;
  }

  const prevB = texBUniform.value;
  if (prevB && prevB !== texA && prevB !== nextTexture) {
    prevB.dispose();
  }

  texBUniform.value = nextTexture;
  mixUniform.value = 0.0;
  material.needsUpdate = true;

  animateUniform(material, mixName, 0.0, 1.0, fadeMs, isCancelled);

  return window.setTimeout(() => {
    if (isCancelled()) return;

    const aOld = texAUniform.value;
    const bNew = texBUniform.value;

    if (bNew) {
      texAUniform.value = bNew;
      texBUniform.value = bNew;
      mixUniform.value = 0.0;
      onPromote?.();
      material.needsUpdate = true;
    }

    if (aOld && aOld !== bNew) {
      aOld.dispose();
    }
  }, fadeMs + 20);
}

export function disposeCrossfadeTextures(
  material: THREE.ShaderMaterial,
  uniforms?: TextureCrossfadeUniformNames
) {
  const texAName = uniforms?.texA ?? "uTexA";
  const texBName = uniforms?.texB ?? "uTexB";

  const texA = getUniformRef<THREE.Texture | null>(material, texAName).value;
  const texB = getUniformRef<THREE.Texture | null>(material, texBName).value;

  if (texA) texA.dispose();
  if (texB && texB !== texA) texB.dispose();
}

export async function loadDataTextureFromApi(args: {
  url: string;
  fallbackMessage: string;
  layerLabel?: string;
  notifyOnError?: boolean;
  requestInit?: RequestInit;
}) {
  const { url, fallbackMessage, layerLabel, notifyOnError = true, requestInit } =
    args;
  const blob = await fetchBlobOrThrow(url, fallbackMessage, {
    layerLabel,
    notifyOnError,
    ...requestInit,
  });

  const objectUrl = URL.createObjectURL(blob);

  try {
    return await new Promise<THREE.Texture>((resolve, reject) => {
      new THREE.TextureLoader().load(objectUrl, resolve, undefined, reject);
    });
  } catch (error) {
    if (notifyOnError) {
      notifyDataFetchError(error, fallbackMessage, { layerLabel });
    }
    throw error;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}
