// WindUvLayer.tsx
"use client";
import * as THREE from "three";
import { useEffect, useMemo, useRef } from "react";
import { type FrameTick, useEarthLayer } from "./EarthBase";
import { exampleParticleLayerApiUrl } from "../utils/ApiResponses"; // you said to use this
import { useControls } from "@/app/state/controlsStore";
import { loadDataTextureFromApi } from "./shaderUtils";

export const min_max_gph_ranges_glsl = `
uniform float uPressure;
void getGphRange(float pressure, out float minRange, out float maxRange) {
    if (pressure == 250.0) {
        minRange = 9600.0;
        maxRange = 11200.0;
    } else if (pressure == 500.0) {
        minRange = 4600.0;
        maxRange = 6000.0;
    } else if (pressure == 850.0) {
        minRange = 1200.0;
        maxRange = 1600.0;
    } else {
        // Default/fallback values
        minRange = 0.0;
        maxRange = 0.0;
    }
}
`;

export const GET_POSITION_Z_SHARED_GLSL3 = `
  ${min_max_gph_ranges_glsl}
  float decodeElevation(vec3 rgb) {
    float R = floor(rgb.r * 255.0 + 0.5);
    float G = floor(rgb.g * 255.0 + 0.5);
    float B = floor(rgb.b * 255.0 + 0.5);
    return (R * 65536.0 + G * 256.0 + B) * 0.1 - 10000.0;
  }
  float get_position_z_glsl3(sampler2D tex, vec2 uv, float exaggeration) {
    float minGPHRange, maxGPHRange;
    getGphRange(uPressure, minGPHRange, maxGPHRange);

    float elev = decodeElevation(texture(tex, uv).rgb);
    float t = clamp((elev - minGPHRange) / (maxGPHRange - minGPHRange), 0.0, 1.0);
    return exaggeration * t;
  }
`;

export const getWindMotionRangesPerPressureLevel = `
// tiny tolerance to avoid float == pitfalls
const float WIND_PRESSURE_LEVEL_EPS = 0.5;

void getUVRange(float pressure, out float minU, out float maxU, out float minV, out float maxV) {
  // UV_RANGES_MPS:
  //  850: (-60, 60)
  //  500: (-80, 80)
  //  250: (-120,120)
  if (abs(pressure - 250.0) < WIND_PRESSURE_LEVEL_EPS) {
    minU = -120.0; maxU = 120.0;
    minV = -120.0; maxV = 120.0;
  } else if (abs(pressure - 500.0) < WIND_PRESSURE_LEVEL_EPS) {
    minU =  -80.0; maxU =  80.0;
    minV =  -80.0; maxV =  80.0;
  } else if (abs(pressure - 850.0) < WIND_PRESSURE_LEVEL_EPS) {
    minU =  -60.0; maxU =  60.0;
    minV =  -60.0; maxV =  60.0;
} else if (abs(pressure - 925.0) < WIND_PRESSURE_LEVEL_EPS) {
    minU = -40.0; maxU = 40.0;
    minV = -40.0; maxV = 40.0;
  } else {
    // fallback: conservative
    minU = -80.0; maxU = 80.0;
    minV = -80.0; maxV = 80.0;
  }
}

void getZRange(out float minW, out float maxW) {
  // Z_RANGE_MPS = (-5, 5)
  minW = -5.0;
  maxW =  5.0;
}
`

const mapUVtoLatLng = `
  float globeRadius = 100.0;
  
  // deg→rad
  float d2r(float d) { return d * 0.017453292519943295; }

  // lat/lon (degrees) -> XYZ in three-globe orientation
    vec3 latLonToXYZ(float latDeg, float lonDeg, float radius) {
      float phi   = d2r(90.0 - latDeg);       // polar
      float theta = d2r(lonDeg + 270.0);      // azimuth
      float x = radius * sin(phi) * cos(theta);
      float z =  -radius * sin(phi) * sin(theta);
      float y =  radius * cos(phi);
      return vec3(x, y, z);
    }
  // get lat/lon from uv
  vec2 getLatLon(vec2 uv){
    return vec2(mix(  90.0, -90.0, uv.y), mix(-180.0, 180.0, uv.x));
  }
`

// GLSL3 helper: map gl_VertexID to subsampled UVs using a fixed integer step
const GET_UV_SUBSAMPLED_GLSL3 = `
  vec2 get_uv_from_vertex_id_subsampled(int gridW, int gridH, int step) {
    int outW = (gridW + step - 1) / step;
    int ii = gl_VertexID % outW;
    int jj = gl_VertexID / outW;
    int srcI = min(gridW - 1, ii * step);
    int srcJ = min(gridH - 1, jj * step);
    return vec2(float(srcI) / float(gridW - 1),
                float(srcJ) / float(gridH - 1));
  }
`;

// UV wind points shader (GLSL3): derive per-vertex UV/XY from gl_VertexID
const UV_POINTS_VERT = `
  ${GET_POSITION_Z_SHARED_GLSL3}
  ${GET_UV_SUBSAMPLED_GLSL3}
  ${mapUVtoLatLng}
  uniform sampler2D uTerrainTexture;
  uniform sampler2D uCurrentPosition;
  uniform vec2 uSimSize;
  uniform float uExaggeration;
  uniform float uAspect;
  uniform float uPointSize;
  uniform int uGridW;
  uniform int uGridH;
  uniform int uStep;
  uniform float zOffset;
  void main(){
    vec2 uvIdx = get_uv_from_vertex_id_subsampled(uGridW, uGridH, uStep);
    vec2 uv = texture(uCurrentPosition, uvIdx).rg;
    vec2 latlon = getLatLon(uv);
    vec3 basePos = latLonToXYZ(latlon.x, latlon.y, globeRadius);
    // 3) sample field height (0..uExaggeration mapped by your get_position_z_glsl3)
    //    and lift along the outward normal
    float hNorm = get_position_z_glsl3(uTerrainTexture, uv, 1.0); // returns t in [0,1] (because we pass 1.0)
    float hWorld = uExaggeration * 50.0 * hNorm;

    vec3 normal = normalize(basePos);
    vec3 worldPos = basePos + normal * hWorld;

    // 4) position
    gl_Position = projectionMatrix * modelViewMatrix * vec4(worldPos + normal * zOffset, 1.0);

    float totalLife = texture(uCurrentPosition, uvIdx).b;
    float lifeExpended = texture(uCurrentPosition, uvIdx).a;
    float p = clamp(lifeExpended / totalLife, 0.0, 1.0);
    // 0→1 from birth to 0.25
    float fadeIn  = smoothstep(0.0, 0.25, p);
    // 1→0 from 0.75 to death
    float fadeOut = 1.0 - smoothstep(0.75, 1.0, p);

    // full curve: up → hold → down
    float fade = fadeIn * fadeOut;
    gl_PointSize = uPointSize * max(fade, 0.001); // shrink away
  }`;
const UV_POINTS_FRAG = `
  precision highp float;
  out vec4 fragColor;
  void main(){
    vec2 d = gl_PointCoord - 0.5;
    if(dot(d,d) > 0.25) discard;
    fragColor = vec4(0.2, 0.9, 0.9, 1.0);
  }
`;

const LAT_LNG_TO_UV_CONVERSION = `
// --- constants & helpers (put above main) ---
const float PI = 3.14159265358979323846264;
const float EARTH_R = 6371000.0;                // meters
const float M_PER_DEG_LAT = (2.0 * PI * EARTH_R) / 360.0; // ≈ 111320 m/deg

// Plate carrée mapping helpers
float latFromV(float vTex) {
  // vTex: 0 (top) → 1 (bottom) maps to +90° → −90°
  return 90.0 - 180.0 * vTex;                   // degrees
}

// Convert (u,v) in m/s at latitude (deg) over dt seconds → ΔUV on plate carrée
vec2 deltaUV_from_ms(vec2 uv_mps, float lat_deg, float dt) {
  float phi = radians(lat_deg);
  float cosphi = cos(phi);
  // meters per degree of longitude shrinks by cos(lat); avoid blow-ups near poles
  float m_per_deg_lon = max(M_PER_DEG_LAT * max(cosphi, 1e-6), 1e-6);

  // degrees moved this step
  float dlat_deg = (uv_mps.y * dt) / M_PER_DEG_LAT;
  float dlon_deg = (uv_mps.x * dt) / m_per_deg_lon;

  // degrees → normalized texture UV (note: V increases downward ⇒ minus sign on dlat)
  // WHEN MOVING TO GLOBE RATHER THAN RECTANGLE, REMOVE COSPHI
  float du = (dlon_deg / 360.0);
  float dv = -dlat_deg / 180.0;
  return vec2(du, dv);
}

// Wrap only longitude (U); clamp latitude (V) to avoid pole wrap
vec2 wrapClampUV(vec2 uv) {
  uv.x = fract(uv.x);
  uv.y = clamp(uv.y, 0.0, 1.0);
  return uv;
}
`

const SIM_VERT = `
out vec2 vUv;
void main() {
  vUv = uv;                    
  gl_Position = vec4(position.xy, 0.0, 1.0);  
}
`;

const SIM_FRAG = `
    ${LAT_LNG_TO_UV_CONVERSION}
    ${getWindMotionRangesPerPressureLevel}
    precision highp float;
    in vec2 vUv;
    out vec4 fragColor;

    uniform sampler2D uPrev;
    uniform float uDt;
    uniform vec2  uSize;
    uniform sampler2D uWindTexture;
    uniform float uPressure;

    uniform float uWindGain;
    uniform float uLifetimeTarget;
    uniform float uMinDistancePerTimeStep;

    float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453123); }
    vec2 sampleWindUV(vec2 uv) {
      uv = fract(uv);

      // Read packed wind from texture
      vec2 rg = texture(uWindTexture, uv).rg;

      // Pressure-aware ranges
      float uMin, uMax, vMin, vMax;
      getUVRange(uPressure, uMin, uMax, vMin, vMax);

      // Decode to physical units (m/s)
      float u_ms = mix(uMin, uMax, rg.r);
      float v_ms = mix(vMin, vMax, rg.g);

      return vec2(u_ms, v_ms);
    }

    void main() {
      vec2 st = (gl_FragCoord.xy - 0.5) / uSize;

      vec4 prev = texture(uPrev, st);
      vec2 position = prev.rg;
      float totalLifeThreshold = prev.b;

      // --- RK2 with physical advection ---
      // Step 1: sample wind at current pos (assumed m/s), convert to ΔUV over (0.5*dt)
      vec2 wind1_ms = sampleWindUV(position) * uWindGain;                // m/s
      float lat1_deg = latFromV(position.y);
      vec2 duv1 = deltaUV_from_ms(wind1_ms, lat1_deg, 0.5 * uDt);

      // Midpoint position
      vec2 midPos = wrapClampUV(position + duv1);

      // Step 2: sample at midpoint and advance full dt with midpoint slope
      vec2 wind2_ms = sampleWindUV(midPos) * uWindGain;                  // m/s
      // wind2_ms = vec2(-1,-1);
      float lat2_deg = latFromV(midPos.y);
      vec2 duv2 = deltaUV_from_ms(wind2_ms, lat2_deg, uDt);

      vec2 newPos = wrapClampUV(position + duv2);
      float lifeExpended = prev.a;
      float movedUV  = length(newPos - position);
      float distanceParticleMoved = max(movedUV, uMinDistancePerTimeStep);
      lifeExpended += distanceParticleMoved / uLifetimeTarget;

      bool particleIsDead = (totalLifeThreshold <= lifeExpended);

      if (particleIsDead) {
        newPos =  st;
        lifeExpended = 0.0;
        totalLifeThreshold = hash(newPos + st) + 1.0;
      }

      fragColor = vec4(newPos, totalLifeThreshold, lifeExpended);
  }
  `
export const TRAIL_STAMP_MIN_VERT = /* glsl */`
${GET_UV_SUBSAMPLED_GLSL3}         // you already have this chunk
uniform sampler2D uCurrentPosition; // RG = (u,v)
uniform int   uGridW, uGridH, uStep;
uniform float uPointSize;

void main() {
  vec2 uvIdx = get_uv_from_vertex_id_subsampled(uGridW, uGridH, uStep);
  vec2 uv    = texture(uCurrentPosition, uvIdx).rg;

  // map (u,v in 0..1) → clip-space (-1..+1), flip V so v=0 is top row
  vec2 ndc = vec2(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
  gl_Position = vec4(ndc, 0.0, 1.0);
  gl_PointSize = uPointSize;
}
`;

// Stamp points (draw second, additively)
const TRAIL_STAMP_FRAG = /* glsl */ `
  precision highp float;
  out vec4 fragColor;
  void main() {
    vec2 d = gl_PointCoord - 0.5;
    if (dot(d,d) > 0.16) discard;
    fragColor = vec4(1.0);
  }
`;

export const TRAIL_GLOBE_VERT = /* glsl */`
out vec3 vWorld;
void main(){
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorld = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
  gl_PointSize = 1.0;
}
`;

export const TRAIL_GLOBE_FRAG = /* glsl */`
precision highp float;
in vec3 vWorld; out vec4 fragColor;
uniform sampler2D uTrailTex;
uniform float uOpacity;
uniform float uLonOffset;  // seam shift; +270° → 0.75
uniform bool  uFlipV;
uniform vec3 trailColor;

// world → equirect UV (match your latLonToXYZ that used z = -sin(theta))
vec2 worldToUV(vec3 p){
  vec3 n = normalize(p);
  float lat = asin(clamp(n.y, -1.0, 1.0));    // [-pi/2, pi/2]
  float lon = atan(-n.z, n.x);                 // NOTE the minus on z
  float u = fract(lon / (2.0*3.14159265) + 0.5 + uLonOffset);
  float v = 0.5 - lat / 3.14159265;
  if (uFlipV) v = 1.0 - v;
  return vec2(u, v);
}

void main(){
  vec2 uv = worldToUV(vWorld);
  vec3 t  = texture(uTrailTex, uv).rgb;
  float I = clamp(max(max(t.r, t.g), t.b), 0.0, 1.0);
  fragColor = vec4(trailColor, I * uOpacity);
}
`;

export const TRAIL_DECAY_VERT = `
// COPY_MIN_VERT (GLSL3)
out vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

export const TRAIL_DECAY_FRAG = `
// COPY_MIN_FRAG (GLSL3)
precision highp float;
in vec2 vUv;
out vec4 fragColor;
uniform sampler2D uSrc;
void main() {
  vec4 original = texture(uSrc, vUv);
  // fragColor = vec4(0.0, original.g * 0.5, 0.0, 1.0);
  fragColor = original * 0.99;
}
`;

function configureWindTexture(texture: THREE.Texture) {
  texture.flipY = false;
  texture.colorSpace = THREE.NoColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
}

function getTextureWH(texture: THREE.Texture): { w: number; h: number } {
  const img = texture.image as { width?: number; height?: number } | undefined;
  const w = typeof img?.width === "number" ? img.width : 0;
  const h = typeof img?.height === "number" ? img.height : 0;
  return { w, h };
}

function getZoomStep(zoomLevel: number): number {
  return Math.round(THREE.MathUtils.clamp(zoomLevel * 15, 3, 20));
}

function clearRT(
  renderer: THREE.WebGLRenderer,
  rt: THREE.WebGLRenderTarget,
  alpha: number // 0 or 0.0
) {
  const prevRT = renderer.getRenderTarget();
  const prevClr = renderer.getClearColor(new THREE.Color()).clone();
  const prevA = renderer.getClearAlpha();

  renderer.setRenderTarget(rt);
  renderer.setClearColor(0x000000, alpha);
  renderer.clear(true, false, false);

  renderer.setRenderTarget(prevRT);
  renderer.setClearColor(prevClr, prevA);
}

function clearRTFloat(renderer: THREE.WebGLRenderer, rt: THREE.WebGLRenderTarget) {
  clearRT(renderer, rt, 0);
}
function clearRTByte(renderer: THREE.WebGLRenderer, rt: THREE.WebGLRenderTarget) {
  clearRT(renderer, rt, 0.0);
}

function disposeWindLayer(
  args: {
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    unregisterFramePass: (key: string) => void;
    passKey: string;
    uvPointsRef: React.MutableRefObject<THREE.Points | null>;
    windTexRef: React.MutableRefObject<THREE.Texture | null>;
    apiRef: React.MutableRefObject<WindLayerAPI | null>;
  }
) {
  const { scene, unregisterFramePass, passKey, uvPointsRef, windTexRef, apiRef } = args;
  const L = apiRef.current;
  if (!L) return;

  unregisterFramePass(passKey);

  // points
  if (uvPointsRef.current) scene.remove(uvPointsRef.current);
  uvPointsRef.current?.geometry?.dispose();
  const pointsMaterial = uvPointsRef.current?.material;
  if (Array.isArray(pointsMaterial)) {
    for (const material of pointsMaterial) material.dispose();
  } else {
    pointsMaterial?.dispose();
  }
  uvPointsRef.current = null;

  // overlay
  L.trailOverlayMesh.removeFromParent();
  L.trailOverlayMesh.geometry.dispose();
  L.trailOverlayMat.dispose();

  // RTs
  L.readRT.dispose();
  L.writeRT.dispose();
  L.trailReadRT.dispose();
  L.trailWriteRT.dispose();

  // materials
  L.simMat.dispose();
  L.trailStampMat.dispose();
  L.decayMat.dispose();

  // best-effort dispose any geometry on helper scenes
  for (const obj of L.simScene.children) {
    if (obj instanceof THREE.Mesh) obj.geometry.dispose();
  }
  for (const obj of L.trailScene.children) {
    if (obj instanceof THREE.Mesh) obj.geometry.dispose();
  }

  apiRef.current = null;

  // wind texture
  windTexRef.current?.dispose();
  windTexRef.current = null;
}

function buildWindLayer(args: {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;

  pressureLevel: number;
  heightTex: THREE.Texture | null;
  exaggeration: number;

  texW: number;
  texH: number;
  windTexture: THREE.Texture;

  registerFramePass: (key: string, fn: (tick: FrameTick) => void) => void;
  passKey: string;

  // refs (so we can keep using your existing pattern)
  apiRef: React.MutableRefObject<WindLayerAPI | null>;
  uvPointsRef: React.MutableRefObject<THREE.Points | null>;

  zoomLevel: number;
}) {
  const {
    renderer,
    scene,
    pressureLevel,
    heightTex,
    exaggeration,
    texW,
    texH,
    windTexture,
    registerFramePass,
    passKey,
    apiRef,
    uvPointsRef,
    zoomLevel,
  } = args;

  // const UV_POINTS_STEP = 10;
  // const UV_POINTS_STEP = zoomLevel * 15;
  const UV_POINTS_STEP = getZoomStep(zoomLevel);
  const outW = Math.ceil(texW / UV_POINTS_STEP);
  const outH = Math.ceil(texH / UV_POINTS_STEP);

  const makeFloatRT = (w: number, h: number) =>
    new THREE.WebGLRenderTarget(w, h, {
      type: THREE.FloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      wrapS: THREE.ClampToEdgeWrapping,
      wrapT: THREE.ClampToEdgeWrapping,
      depthBuffer: false,
      stencilBuffer: false,
    });

  const rtRead = makeFloatRT(outW, outH);
  const rtWrite = makeFloatRT(outW, outH);
  rtRead.texture.generateMipmaps = false;
  rtWrite.texture.generateMipmaps = false;
  clearRTFloat(renderer, rtRead);
  clearRTFloat(renderer, rtWrite);

  // dummy geo for gl_VertexID
  const geo = new THREE.BufferGeometry();
  geo.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(outW * outH * 3), 3)
  );

  const aspect = texW / texH;

  const ptsMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: UV_POINTS_VERT,
    fragmentShader: UV_POINTS_FRAG,
    transparent: true,
    blending: THREE.NormalBlending,
    depthWrite: false,
    depthTest: true,
    uniforms: {
      uTerrainTexture: { value: heightTex },
      uExaggeration: { value: exaggeration },
      uAspect: { value: aspect },
      uPointSize: { value: 1.5 * (window.devicePixelRatio || 1) * 3.0 },
      uGridW: { value: texW },
      uGridH: { value: texH },
      uStep: { value: UV_POINTS_STEP },
      uCurrentPosition: { value: rtRead.texture },
      uSimSize: { value: new THREE.Vector2(outW, outH) },
      uPressure: { value: pressureLevel },
      zOffset: { value: 1 },
    },
  });

  const pts = new THREE.Points(geo, ptsMat);
  pts.frustumCulled = false;
  pts.name = `wind-uv-points-${pressureLevel}`;
  scene.add(pts);

  // sim
  const simScene = new THREE.Scene();
  const simCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const simGeom = new THREE.PlaneGeometry(2, 2);
  const simMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: SIM_VERT,
    fragmentShader: SIM_FRAG,
    uniforms: {
      uPrev: { value: rtRead.texture },
      uDt: { value: 0.0 },
      uSize: { value: new THREE.Vector2(outW, outH) },
      uWindTexture: { value: windTexture },
      uPressure: { value: pressureLevel },
      uWindGain: { value: 20 },
      uLifetimeTarget: { value: 8.0 },
      uMinDistancePerTimeStep: { value: 0.05 },
    },
  });
  simScene.add(new THREE.Mesh(simGeom, simMat));

  // trails RTs
  const makeTrailRT = (w: number, h: number) =>
    new THREE.WebGLRenderTarget(w, h, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      stencilBuffer: false,
    });

  const TRAIL_SCALE = 1;
  // const TRAIL_SCALE = THREE.MathUtils.clamp(
  //   0.5 + zoomLevel,   // 0 -> 0.5, 0.5 -> 1.0, 1.0 -> 1.5
  //   0.5,
  //   1.5
  // );
  const trailW = Math.max(1, Math.round(texW * TRAIL_SCALE));
  const trailH = Math.max(1, Math.round(texH * TRAIL_SCALE));

  const trailReadRT = makeTrailRT(trailW, trailH);
  const trailWriteRT = makeTrailRT(trailW, trailH);
  clearRTByte(renderer, trailReadRT);
  clearRTByte(renderer, trailWriteRT);

  // combined trail scene
  const trailCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const trailScene = new THREE.Scene();

  const decayGeom = new THREE.PlaneGeometry(2, 2);
  const decayMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: TRAIL_DECAY_VERT,
    fragmentShader: TRAIL_DECAY_FRAG,
    depthTest: false,
    depthWrite: false,
    transparent: false,
    blending: THREE.NoBlending,
    uniforms: { uSrc: { value: trailReadRT.texture } },
  });
  const decayMesh = new THREE.Mesh(decayGeom, decayMat);
  decayMesh.frustumCulled = false;
  decayMesh.renderOrder = 0;
  trailScene.add(decayMesh);

  const trailStampMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: TRAIL_STAMP_MIN_VERT,
    fragmentShader: TRAIL_STAMP_FRAG,
    depthTest: false,
    depthWrite: false,
    transparent: true,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uCurrentPosition: { value: rtRead.texture },
      uGridW: { value: texW },
      uGridH: { value: texH },
      uStep: { value: UV_POINTS_STEP },
      uPointSize: { value: 1.0 },
    },
  });
  const trailStampPoints = new THREE.Points(geo, trailStampMat);
  trailStampPoints.frustumCulled = false;
  trailStampPoints.renderOrder = 1;
  trailScene.add(trailStampPoints);

  // overlay globe
  const globeRadius = 100;
  const trailOverlayGeom = new THREE.SphereGeometry(globeRadius + 1, 256, 128);
  const trailOverlayMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: TRAIL_GLOBE_VERT,
    fragmentShader: TRAIL_GLOBE_FRAG,
    transparent: true,
    depthTest: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uTrailTex: { value: trailReadRT.texture },
      uOpacity: { value: 0.9 },
      uLonOffset: { value: 0.25 },
      uFlipV: { value: true },
      trailColor: { value: new THREE.Color(0x99ffff) },
    },
  });
  trailOverlayMat.toneMapped = false;

  const trailOverlayMesh = new THREE.Mesh(trailOverlayGeom, trailOverlayMat);
  trailOverlayMesh.frustumCulled = false;
  trailOverlayMesh.renderOrder = 10;
  trailOverlayMesh.name = `wind-uv-trails-${pressureLevel}`;
  scene.add(trailOverlayMesh);

  uvPointsRef.current = pts;

  apiRef.current = {
    simScene,
    simCam,
    simMat,
    readRT: rtRead,
    writeRT: rtWrite,
    ptsMat,
    outW,
    outH,

    trailReadRT,
    trailWriteRT,

    trailOverlayMesh,
    trailOverlayMat,

    trailScene,
    trailCam,
    trailStampMat,
    decayMat,
    trailStampPoints,

    texW,
    texH,

    zoomStep: UV_POINTS_STEP,
  };

  // frame pass ONCE
  registerFramePass(passKey, (tick) => {
    const L = apiRef.current;
    if (!L) return;

    const simTimeStep = tick.dt;

    // 1) sim
    L.simMat.uniforms.uPrev.value = L.readRT.texture;
    L.simMat.uniforms.uDt.value = simTimeStep;

    renderer.setRenderTarget(L.writeRT);
    renderer.setViewport(0, 0, L.outW, L.outH);
    renderer.setScissorTest(false);
    renderer.clear();
    renderer.render(L.simScene, L.simCam);
    renderer.setRenderTarget(null);

    // swap sim RTs
    {
      const tmp = L.readRT;
      L.readRT = L.writeRT;
      L.writeRT = tmp;
    }

    L.ptsMat.uniforms.uCurrentPosition.value = L.readRT.texture;

    // 2) trails
    L.decayMat.uniforms.uSrc.value = L.trailReadRT.texture;
    L.trailStampMat.uniforms.uCurrentPosition.value = L.readRT.texture;

    renderer.setRenderTarget(L.trailWriteRT);
    renderer.setViewport(0, 0, L.trailWriteRT.width, L.trailWriteRT.height);
    renderer.setScissorTest(false);
    renderer.clear();

    // keep the “extra bind” that prevents disco
    renderer.setRenderTarget(L.trailWriteRT);
    renderer.render(L.trailScene, L.trailCam);

    // swap trail RTs
    {
      const t = L.trailReadRT;
      L.trailReadRT = L.trailWriteRT;
      L.trailWriteRT = t;
    }

    L.trailOverlayMat.uniforms.uTrailTex.value = L.trailReadRT.texture;

    renderer.setRenderTarget(null);
  });
}

export type WindLayerAPI = {
  // sim ping-pong
  simScene: THREE.Scene;
  simCam: THREE.OrthographicCamera;
  simMat: THREE.ShaderMaterial;
  readRT: THREE.WebGLRenderTarget;
  writeRT: THREE.WebGLRenderTarget;
  ptsMat: THREE.ShaderMaterial;
  outW: number;
  outH: number;

  // trails ping-pong
  trailReadRT: THREE.WebGLRenderTarget;
  trailWriteRT: THREE.WebGLRenderTarget;

  trailOverlayMesh: THREE.Mesh;
  trailOverlayMat: THREE.ShaderMaterial;

  // trails (decay + stamp share one scene/cam)
  trailScene: THREE.Scene;
  trailCam: THREE.OrthographicCamera;
  trailStampMat: THREE.ShaderMaterial;
  decayMat: THREE.ShaderMaterial;

  // stamping points object (so we can dispose its material/geo cleanly if needed)
  trailStampPoints: THREE.Points;

  texW: number;
  texH: number;

  zoomStep: number;
};

type Props = {
  heightTex: THREE.Texture | null;
  exaggeration?: number;
  setWindTex?: (tex: THREE.Texture) => void;
};

export default function ExampleParticleLayer({
  heightTex,
  exaggeration,
  setWindTex,
}: Props) {
  const exampleParticlePressureLevel = useControls((s) => s.exampleParticleLayer.pressureLevel);
  // Use EarthBase plumbing
  const layerKey = useMemo(() => `example-particle-${exampleParticlePressureLevel}`, [exampleParticlePressureLevel]);
  const {
    engineReady,
    rendererRef,
    sceneRef,
    timestamp,
    signalReady,
    registerFramePass,
    unregisterFramePass,
    zoom01,
  } = useEarthLayer(layerKey);

  const apiRef = useRef<WindLayerAPI | null>(null);
  const reqIdRef = useRef(0);

  // keep these so we can dispose easily
  const windTexRef = useRef<THREE.Texture | null>(null);
  const uvPointsRef = useRef<THREE.Points | null>(null);

  // A unique pass key (so multiple WindUvLayer instances don’t collide)
  const passKey = useMemo(() => `${layerKey}:framepass`, [layerKey]);

  useEffect(() => {
    if (!engineReady) return;

    const renderer = rendererRef.current;
    const scene = sceneRef.current;
    if (!renderer || !scene) return;
    if (exampleParticlePressureLevel === "none") {
      signalReady(timestamp);
      return;
    }

    let disposed = false;
    const myReqId = ++reqIdRef.current;
    const isCancelled = () => disposed || myReqId !== reqIdRef.current;
    const url = exampleParticleLayerApiUrl(timestamp, exampleParticlePressureLevel);

    void loadDataTextureFromApi({
      url,
      fallbackMessage: "Failed to load wind data.",
      layerLabel: `Example particle layer (${exampleParticlePressureLevel} hPa)`,
    })
      .then((texture) => {
        if (isCancelled()) {
          texture.dispose();
          return;
        }

        configureWindTexture(texture);

        const { w: texW, h: texH } = getTextureWH(texture);
        if (texW === 0 || texH === 0) {
          texture.dispose();
          return;
        }

        const existing = apiRef.current;
        // 1) If first time: build
        if (!existing) {
          if (windTexRef.current) windTexRef.current.dispose();
          windTexRef.current = texture;
          setWindTex?.(texture);

          buildWindLayer({
            renderer,
            scene,
            pressureLevel: exampleParticlePressureLevel,
            heightTex,
            exaggeration: exaggeration ?? 0.5,
            texW,
            texH,
            windTexture: texture,
            registerFramePass,
            passKey,
            apiRef,
            uvPointsRef,
            zoomLevel: zoom01,
          });

          signalReady(timestamp);
          return;
        }

        // 2) If dims changed: rebuild
        if (existing.texW !== texW || existing.texH !== texH) {
          disposeWindLayer({
            renderer,
            scene,
            unregisterFramePass,
            passKey,
            uvPointsRef,
            windTexRef,
            apiRef,
          });

          // after disposal, build again
          windTexRef.current = texture;
          setWindTex?.(texture);

          buildWindLayer({
            renderer,
            scene,
            pressureLevel: exampleParticlePressureLevel,
            heightTex,
            exaggeration: exaggeration ?? 0.5,
            texW,
            texH,
            windTexture: texture,
            registerFramePass,
            passKey,
            apiRef,
            uvPointsRef,
            zoomLevel: zoom01,
          });

          signalReady(timestamp);
          return;
        }

        const desiredZoomStep = getZoomStep(zoom01);
        if (existing.zoomStep !== desiredZoomStep) {
          // IMPORTANT: keep the new texture or reuse the same one;
          // we already have `texture` loaded and configured.
          disposeWindLayer({
            renderer,
            scene,
            unregisterFramePass,
            passKey,
            uvPointsRef,
            windTexRef,
            apiRef,
          });

          // after disposal, build again using the SAME already-loaded texture
          windTexRef.current = texture;
          setWindTex?.(texture);

          buildWindLayer({
            renderer,
            scene,
            pressureLevel: exampleParticlePressureLevel,
            heightTex,
            exaggeration: exaggeration ?? 0.5,
            texW,
            texH,
            windTexture: texture,
            registerFramePass,
            passKey,
            apiRef,
            uvPointsRef,
            zoomLevel: zoom01,
          });

          signalReady(timestamp);
          return;
        }
        // 3) Normal case: swap wind texture + reset RTs
        if (windTexRef.current) windTexRef.current.dispose();
        windTexRef.current = texture;
        setWindTex?.(texture);

        existing.simMat.uniforms.uWindTexture.value = texture;
        // prevent particles/trails from clearing out on timestamp change
        // clearRTFloat(renderer, existing.readRT);
        // clearRTFloat(renderer, existing.writeRT);
        // clearRTByte(renderer, existing.trailReadRT);
        // clearRTByte(renderer, existing.trailWriteRT);

        existing.ptsMat.uniforms.uCurrentPosition.value = existing.readRT.texture;
        existing.trailStampMat.uniforms.uCurrentPosition.value = existing.readRT.texture;

        existing.decayMat.uniforms.uSrc.value = existing.trailReadRT.texture;
        existing.trailOverlayMat.uniforms.uTrailTex.value = existing.trailReadRT.texture;

        signalReady(timestamp);
      })
      .catch((err) => {
        if (isCancelled()) return;
        console.error("Failed to load example particle png", err);
        signalReady(timestamp);
      });

    return () => {
      disposed = true;
    };
  }, [
    engineReady,
    rendererRef,
    sceneRef,
    timestamp,
    exampleParticlePressureLevel,
    heightTex,
    exaggeration,
    registerFramePass,
    unregisterFramePass,
    passKey,
    signalReady,
    setWindTex,
    zoom01,
  ]);

  useEffect(() => {
    if (!engineReady) return;
    const renderer = rendererRef.current;
    const scene = sceneRef.current;
    if (!renderer || !scene) return;

    return () => {
      disposeWindLayer({
        renderer,
        scene,
        unregisterFramePass,
        passKey,
        uvPointsRef,
        windTexRef,
        apiRef,
      });
    };
  }, [engineReady, rendererRef, sceneRef, unregisterFramePass, passKey]);

  useEffect(() => {
    const L = apiRef.current;
    if (!L) return;
    L.ptsMat.uniforms.uTerrainTexture.value = heightTex;
    L.ptsMat.uniforms.uExaggeration.value = exaggeration ?? 0.5;
  }, [heightTex, exaggeration]);

  return null;
}
