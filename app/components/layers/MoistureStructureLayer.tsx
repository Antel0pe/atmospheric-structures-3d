import { useCallback, useEffect, useRef } from "react";
import * as THREE from "three";
import { useEarthLayer } from "./EarthBase";
import {
  resolveMoistureStructureLayerState,
  useControls,
  type ResolvedMoistureStructureLayerState,
} from "../../state/controlsStore";
import {
  fetchMoistureStructureFrame,
  type MoistureSegmentationMode,
  type MoistureStructureComponentMetadata,
  type MoistureStructureFrame,
} from "../utils/ApiResponses";
import type {
  NormalizedScreenTarget,
  ViewDebugAnalyzerAdapter,
} from "../../lib/viewDebug";

type MoistureLayerStyle = ResolvedMoistureStructureLayerState;

type MoistureShaderUniforms = {
  cutawayCenter: { value: THREE.Vector3 };
  cameraDirection: { value: THREE.Vector3 };
  cutawayRadius: { value: number };
  cutawayEnabled: { value: number };
  solidShellEnabled: { value: number };
  depthCueStrength: { value: number };
  surfaceCueMode: { value: number };
  surfaceBrightness: { value: number };
  surfaceShadowStrength: { value: number };
  rimEnabled: { value: number };
  rimStrength: { value: number };
  distanceFadeEnabled: { value: number };
  distanceFadeStrength: { value: number };
  colorMode: { value: number };
  pressureBandCount: { value: number };
  componentColor: { value: THREE.Color };
  neutralColor: { value: THREE.Color };
  isSelected: { value: number };
  verticalWallFadeEnabled: { value: number };
  verticalWallFadeStrength: { value: number };
  wallFadePassScale: { value: number };
  backPass: { value: number };
};

type MoistureComponentVisual = {
  frontMesh: THREE.Object3D;
  backMesh: THREE.Object3D;
  outlineMesh: THREE.Object3D;
  frontMaterial: THREE.Material;
  backMaterial: THREE.Material;
  outlineMaterial: THREE.Material;
  opacityWeight: number;
  componentId: number;
  bucketIndex: number | null;
  componentColor: THREE.Color;
  renderKind: "default" | "simpleVoxelShell";
};

type MoistureSlice = {
  group: THREE.Group;
  components: MoistureComponentVisual[];
  materials: THREE.Material[];
  geometries: THREE.BufferGeometry[];
  baseOpacity: number;
  pressureBandCount: number;
};

type LevelBandScale = {
  boundaryRadii: number[];
  levelColors: THREE.Color[];
};

type MoistureFootprintVisual = {
  group: THREE.Group;
  geometries: THREE.BufferGeometry[];
  materials: THREE.LineBasicMaterial[];
};

const LEVEL_COLOR_STOPS = [
  new THREE.Color("#ff8a63"),
  new THREE.Color("#2dc6d6"),
  new THREE.Color("#5e86ff"),
  new THREE.Color("#b95cff"),
] as const;
const FRONT_SPECULAR = new THREE.Color("#f6f8ff");
const BACK_SPECULAR = new THREE.Color("#b7c7e8");
const MOISTURE_EMISSIVE = new THREE.Color("#151b28");
const INTERIOR_TINT = new THREE.Color("#42577f");
const FRONT_TINT = new THREE.Color("#ffffff");
const OUTLINE_TINT = new THREE.Color("#6f7d93");
const DEFAULT_AMBIENT_INTENSITY = 2;
const OUTLINE_RENDER_ORDER = 63;
const MOISTURE_GLOBE_CLEARANCE = 10;
const FRONT_RENDER_ORDER = 64;
const BACK_RENDER_ORDER = 65;
const FOOTPRINT_RENDER_ORDER = 68;
const KEY_LIGHT_POSITION = new THREE.Vector3(220, -160, 260);
const NEUTRAL_COMPONENT_COLOR = new THREE.Color("#5b6474");
const COMPONENT_PALETTE = [
  "#ff8a63",
  "#ffd166",
  "#8ac926",
  "#2dc6d6",
  "#5e86ff",
  "#b95cff",
  "#f72585",
  "#06d6a0",
  "#4895ef",
  "#f4a261",
  "#90be6d",
  "#c77dff",
].map((value) => new THREE.Color(value));
const COLOR_MODE_VALUES = {
  pressureBands: 0,
  componentSolid: 1,
  componentHybrid: 2,
  componentHeightSteps: 3,
  selectedMonochrome: 4,
} as const;
const SURFACE_CUE_MODE_VALUES = {
  none: 0,
  globalLift: 1,
  topLight: 2,
  edgeEnhance: 3,
  aerialPerspective: 4,
  heightCue: 5,
  roofRelief: 6,
  viewAdaptive: 7,
} as const;

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
  for (const geometry of slice.geometries) {
    geometry.dispose();
  }
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

function latLonToXYZ(latDeg: number, lonDeg: number, radius: number) {
  const lat = THREE.MathUtils.degToRad(latDeg);
  const lon = THREE.MathUtils.degToRad(-(lonDeg + 270.0));
  return new THREE.Vector3(
    radius * Math.cos(lat) * Math.cos(lon),
    radius * Math.sin(lat),
    radius * Math.cos(lat) * Math.sin(lon)
  );
}

function componentColorForId(id: number) {
  return COMPONENT_PALETTE[id % COMPONENT_PALETTE.length].clone();
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
  if (frame.metadata.components.length === 0) {
    return [];
  }
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

function attachMoistureShader(material: THREE.MeshPhongMaterial) {
  const uniforms: MoistureShaderUniforms = {
    cutawayCenter: { value: new THREE.Vector3() },
    cameraDirection: { value: new THREE.Vector3(0, 0, -1) },
    cutawayRadius: { value: 0 },
    cutawayEnabled: { value: 0 },
    solidShellEnabled: { value: 0 },
    depthCueStrength: { value: 0 },
    surfaceCueMode: { value: SURFACE_CUE_MODE_VALUES.none },
    surfaceBrightness: { value: 1 },
    surfaceShadowStrength: { value: 1 },
    rimEnabled: { value: 0 },
    rimStrength: { value: 0 },
    distanceFadeEnabled: { value: 0 },
    distanceFadeStrength: { value: 0 },
    colorMode: { value: COLOR_MODE_VALUES.pressureBands },
    pressureBandCount: { value: 1 },
    componentColor: { value: FRONT_TINT.clone() },
    neutralColor: { value: NEUTRAL_COMPONENT_COLOR.clone() },
    isSelected: { value: 0 },
    verticalWallFadeEnabled: { value: 0 },
    verticalWallFadeStrength: { value: 0 },
    wallFadePassScale: { value: 1 },
    backPass: { value: 0 },
  };

  material.userData.moistureShaderUniforms = uniforms;
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uCameraCutawayCenter = uniforms.cutawayCenter;
    shader.uniforms.uMoistureCameraDirection = uniforms.cameraDirection;
    shader.uniforms.uCameraCutawayRadius = uniforms.cutawayRadius;
    shader.uniforms.uCameraCutawayEnabled = uniforms.cutawayEnabled;
    shader.uniforms.uMoistureSolidShellEnabled = uniforms.solidShellEnabled;
    shader.uniforms.uMoistureDepthCueStrength = uniforms.depthCueStrength;
    shader.uniforms.uMoistureSurfaceCueMode = uniforms.surfaceCueMode;
    shader.uniforms.uMoistureSurfaceBrightness = uniforms.surfaceBrightness;
    shader.uniforms.uMoistureSurfaceShadowStrength =
      uniforms.surfaceShadowStrength;
    shader.uniforms.uMoistureRimEnabled = uniforms.rimEnabled;
    shader.uniforms.uMoistureRimStrength = uniforms.rimStrength;
    shader.uniforms.uMoistureDistanceFadeEnabled = uniforms.distanceFadeEnabled;
    shader.uniforms.uMoistureDistanceFadeStrength = uniforms.distanceFadeStrength;
    shader.uniforms.uMoistureColorMode = uniforms.colorMode;
    shader.uniforms.uMoisturePressureBandCount = uniforms.pressureBandCount;
    shader.uniforms.uMoistureComponentColor = uniforms.componentColor;
    shader.uniforms.uMoistureNeutralColor = uniforms.neutralColor;
    shader.uniforms.uMoistureIsSelected = uniforms.isSelected;
    shader.uniforms.uMoistureVerticalWallFadeEnabled =
      uniforms.verticalWallFadeEnabled;
    shader.uniforms.uMoistureVerticalWallFadeStrength =
      uniforms.verticalWallFadeStrength;
    shader.uniforms.uMoistureWallFadePassScale = uniforms.wallFadePassScale;
    shader.uniforms.uMoistureBackPass = uniforms.backPass;

    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        `#include <common>
attribute float pressureMix;
varying vec3 vWorldPosition;
varying vec3 vMoistureWorldNormal;
varying float vPressureMix;`
      )
      .replace(
        "#include <beginnormal_vertex>",
        `#include <beginnormal_vertex>
vMoistureWorldNormal = normalize( mat3( modelMatrix ) * objectNormal );
vPressureMix = pressureMix;`
      )
      .replace(
        "#include <worldpos_vertex>",
        `#include <worldpos_vertex>
vec4 moistureWorldPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
moistureWorldPosition = batchingMatrix * moistureWorldPosition;
#endif
#ifdef USE_INSTANCING
moistureWorldPosition = instanceMatrix * moistureWorldPosition;
#endif
moistureWorldPosition = modelMatrix * moistureWorldPosition;
vWorldPosition = moistureWorldPosition.xyz;`
      );

    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
uniform vec3 uCameraCutawayCenter;
uniform vec3 uMoistureCameraDirection;
uniform float uCameraCutawayRadius;
uniform float uCameraCutawayEnabled;
uniform float uMoistureSolidShellEnabled;
uniform float uMoistureDepthCueStrength;
uniform float uMoistureSurfaceCueMode;
uniform float uMoistureSurfaceBrightness;
uniform float uMoistureSurfaceShadowStrength;
uniform float uMoistureRimEnabled;
uniform float uMoistureRimStrength;
uniform float uMoistureDistanceFadeEnabled;
uniform float uMoistureDistanceFadeStrength;
uniform float uMoistureColorMode;
uniform float uMoisturePressureBandCount;
uniform vec3 uMoistureComponentColor;
uniform vec3 uMoistureNeutralColor;
uniform float uMoistureIsSelected;
uniform float uMoistureVerticalWallFadeEnabled;
uniform float uMoistureVerticalWallFadeStrength;
uniform float uMoistureWallFadePassScale;
uniform float uMoistureBackPass;
varying vec3 vWorldPosition;
varying vec3 vMoistureWorldNormal;
varying float vPressureMix;`
      )
      .replace(
        "#include <clipping_planes_fragment>",
        `#include <clipping_planes_fragment>
if ( uCameraCutawayEnabled > 0.5 ) {
  vec3 cutawayDelta = vWorldPosition - uCameraCutawayCenter;
  if ( dot( cutawayDelta, cutawayDelta ) < uCameraCutawayRadius * uCameraCutawayRadius ) {
    discard;
  }
}`
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
if ( uMoistureColorMode > 0.5 ) {
  vec3 moistureColor = uMoistureComponentColor;
  float bandCount = max( uMoisturePressureBandCount, 1.0 );
  float bandIndex = floor( vPressureMix * max( bandCount - 1.0, 1.0 ) + 0.5 );
  float altitudeMix = bandCount > 1.0 ? bandIndex / max( bandCount - 1.0, 1.0 ) : 0.0;
  if ( uMoistureColorMode > 1.5 && uMoistureColorMode < 2.5 ) {
    moistureColor *= ( 0.88 + 0.24 * vPressureMix );
  } else if ( uMoistureColorMode > 2.5 && uMoistureColorMode < 3.5 ) {
    vec3 steppedAltitudeColor = moistureColor;
    float saturationDrop = mix( 0.58, 0.0, altitudeMix );
    vec3 moistureLuma = vec3( dot( moistureColor, vec3( 0.2126, 0.7152, 0.0722 ) ) );
    steppedAltitudeColor = mix( steppedAltitudeColor, moistureLuma, saturationDrop );
    float parity = mod( bandIndex, 2.0 );
    if ( parity > 0.5 ) {
      steppedAltitudeColor *= 0.94;
    } else {
      steppedAltitudeColor *= 1.02;
    }
    moistureColor = steppedAltitudeColor;
  } else if ( uMoistureColorMode > 3.5 ) {
    moistureColor = uMoistureIsSelected > 0.5 ? uMoistureComponentColor : uMoistureNeutralColor;
  }
  diffuseColor.rgb = moistureColor;
}`
      )
      .replace(
        "#include <envmap_fragment>",
        `#include <envmap_fragment>
vec3 moistureViewDir = normalize( vViewPosition );
vec3 moistureViewNormal = normalize( normal );
vec3 radialUp = normalize( vWorldPosition );
float wallness = 1.0 - abs( dot( normalize( vMoistureWorldNormal ), radialUp ) );
vec3 cameraToFragment = vWorldPosition - uCameraCutawayCenter;
float forwardDistance = dot( cameraToFragment, normalize( uMoistureCameraDirection ) );
float farDepth = smoothstep( 16.0, 104.0, forwardDistance );
float topFacing = 1.0 - smoothstep( 0.24, 0.88, wallness );
float wallFacing = smoothstep( 0.28, 0.92, wallness );
float silhouette = pow( 1.0 - abs( dot( moistureViewNormal, moistureViewDir ) ), 2.2 );
float shadowStrength = uMoistureSurfaceShadowStrength;
float cameraTopDown = smoothstep(
  0.76,
  0.97,
  dot( normalize( uCameraCutawayCenter ), normalize( -uMoistureCameraDirection ) )
);
float roofFirstStrength = 0.0;
if ( uMoistureSurfaceCueMode > 5.5 ) {
  roofFirstStrength =
    cameraTopDown * uMoistureSolidShellEnabled * ( 1.0 - uMoistureBackPass );
}
float topDownWallSuppression = roofFirstStrength * smoothstep( 0.22, 0.9, wallness );
vec3 lowRoofTint = vec3( 0.97, 0.91, 0.78 );
vec3 highRoofTint = vec3( 0.75, 0.86, 0.99 );
vec3 roofHeightTint = mix( lowRoofTint, highRoofTint, smoothstep( 0.0, 1.0, vPressureMix ) );
float moistureWallFade = 0.0;
if ( uMoistureVerticalWallFadeEnabled > 0.5 ) {
  moistureWallFade = smoothstep( 0.35, 0.9, wallness ) * uMoistureVerticalWallFadeStrength * uMoistureWallFadePassScale;
  float wallAlphaFade = uMoistureSolidShellEnabled > 0.5 && uMoistureBackPass < 0.5 ? mix( 0.12, 0.68, roofFirstStrength ) : 0.75;
  diffuseColor.a *= 1.0 - wallAlphaFade * moistureWallFade;
}
if ( roofFirstStrength > 0.001 ) {
  diffuseColor.a *= 1.0 - topDownWallSuppression * 0.45;
}
if ( uMoistureSolidShellEnabled > 0.5 && uMoistureBackPass < 0.5 ) {
  float roofBias = 1.0 - smoothstep( 0.18, 0.95, wallness );
  outgoingLight *= mix( 1.0 - 0.18 * shadowStrength, 1.04, roofBias );
}
if ( roofFirstStrength > 0.001 ) {
  vec3 roofLit = outgoingLight * mix( 1.01, 1.15, topFacing );
  roofLit = mix( roofLit, roofHeightTint, topFacing * roofFirstStrength * 0.42 );
  vec3 wallLit = mix(
    outgoingLight,
    vec3( dot( outgoingLight, vec3( 0.2126, 0.7152, 0.0722 ) ) ) * vec3( 0.96, 0.94, 0.91 ),
    0.58
  );
  wallLit *= 1.0 - topDownWallSuppression * 0.3;
  outgoingLight = mix( wallLit, roofLit, topFacing );
}
if ( uMoistureDepthCueStrength > 0.001 && uMoistureBackPass < 0.5 ) {
  float depthShade = mix( 1.08, 1.0 - 0.32 * shadowStrength, farDepth );
  outgoingLight *= mix( 1.0, depthShade, uMoistureDepthCueStrength );
  vec3 depthLuma = vec3( dot( outgoingLight, vec3( 0.2126, 0.7152, 0.0722 ) ) );
  outgoingLight = mix(
    outgoingLight,
    mix( outgoingLight, depthLuma, 0.55 ),
    farDepth * 0.28 * uMoistureDepthCueStrength * shadowStrength
  );
}
if ( uMoistureRimEnabled > 0.5 ) {
  float rim = pow( 1.0 - max( dot( moistureViewNormal, moistureViewDir ), 0.0 ), 2.2 );
  outgoingLight += diffuseColor.rgb * rim * uMoistureRimStrength * ( 1.0 - moistureWallFade );
}
if ( uMoistureDistanceFadeEnabled > 0.5 ) {
  float moistureDistance = length( vViewPosition );
  float fade = smoothstep( 32.0, 180.0, moistureDistance ) * uMoistureDistanceFadeStrength;
  float luminance = dot( outgoingLight, vec3( 0.2126, 0.7152, 0.0722 ) );
  outgoingLight = mix( outgoingLight, vec3( luminance ), fade * 0.45 );
  outgoingLight *= 1.0 - fade * 0.18;
  diffuseColor.a *= 1.0 - fade * 0.55;
}
  if ( uMoistureSurfaceCueMode > 0.5 && uMoistureBackPass < 0.5 ) {
  vec3 surfaceCueBase = outgoingLight;
  vec3 surfaceCueLight = surfaceCueBase;
  vec3 surfaceCueLuma = vec3( dot( surfaceCueBase, vec3( 0.2126, 0.7152, 0.0722 ) ) );
  if ( uMoistureSurfaceCueMode > 0.5 && uMoistureSurfaceCueMode < 1.5 ) {
    surfaceCueLight = surfaceCueBase * 1.12 + diffuseColor.rgb * 0.035;
  } else if ( uMoistureSurfaceCueMode > 1.5 && uMoistureSurfaceCueMode < 2.5 ) {
    surfaceCueLight *= mix( 1.0 - 0.06 * shadowStrength, 1.12, topFacing );
    surfaceCueLight *= 1.0 - wallFacing * 0.12 * shadowStrength;
  } else if ( uMoistureSurfaceCueMode > 2.5 && uMoistureSurfaceCueMode < 3.5 ) {
    surfaceCueLight *= 1.0 - silhouette * 0.24 * shadowStrength;
    surfaceCueLight += diffuseColor.rgb * topFacing * 0.035;
  } else if ( uMoistureSurfaceCueMode > 3.5 && uMoistureSurfaceCueMode < 4.5 ) {
    vec3 hazeTarget = mix( surfaceCueBase, surfaceCueLuma, 0.3 ) + vec3( 0.03, 0.04, 0.06 );
    surfaceCueLight = mix( surfaceCueBase, hazeTarget, farDepth * 0.26 );
  } else if ( uMoistureSurfaceCueMode > 4.5 && uMoistureSurfaceCueMode < 5.5 ) {
    float heightLift = mix( 0.97, 1.14, vPressureMix );
    surfaceCueLight *= mix( 1.0, heightLift, topFacing * 0.68 );
    surfaceCueLight += diffuseColor.rgb * topFacing * vPressureMix * 0.04;
  } else if ( uMoistureSurfaceCueMode > 5.5 && uMoistureSurfaceCueMode < 6.5 ) {
    vec3 lowRoofColor = vec3( 0.92, 0.77, 0.45 );
    vec3 highRoofColor = vec3( 0.50, 0.72, 0.94 );
    vec3 roofColor = mix( lowRoofColor, highRoofColor, smoothstep( 0.0, 1.0, vPressureMix ) );
    vec3 wallColor = mix(
      surfaceCueBase,
      surfaceCueLuma * 0.86 + vec3( 0.05, 0.045, 0.04 ),
      0.6
    );
    vec3 roofLight = surfaceCueBase * mix( 1.0, 1.14, topFacing );
    roofLight = mix( roofLight, roofColor, topFacing * 0.58 );
    roofLight += roofColor * topFacing * 0.05;
    wallColor *= 1.0 - wallFacing * 0.14 * shadowStrength;
    surfaceCueLight = mix( wallColor, roofLight, topFacing );
    surfaceCueLight = mix( surfaceCueLight, roofColor, cameraTopDown * topFacing * 0.18 );
  } else if ( uMoistureSurfaceCueMode > 6.5 ) {
    vec3 topDownLight = surfaceCueBase;
    vec3 obliqueLight = surfaceCueBase;
    topDownLight *= mix( 1.0 - 0.05 * shadowStrength, 1.12, topFacing );
    topDownLight *= 1.0 - wallFacing * 0.1 * shadowStrength;
    topDownLight *= mix( 0.99, 1.1, topFacing * vPressureMix );
    obliqueLight *= 1.0 - silhouette * 0.18 * shadowStrength;
    vec3 obliqueLuma = vec3( dot( obliqueLight, vec3( 0.2126, 0.7152, 0.0722 ) ) );
    vec3 obliqueHaze = mix( obliqueLight, obliqueLuma, 0.22 ) + vec3( 0.02, 0.03, 0.05 );
    obliqueLight = mix( obliqueLight, obliqueHaze, farDepth * 0.2 );
    surfaceCueLight = mix( obliqueLight, topDownLight, cameraTopDown );
  }
  outgoingLight = surfaceCueLight;
}
outgoingLight *= 1.0 - moistureWallFade * 0.22 * shadowStrength;
outgoingLight = mix(
  outgoingLight,
  outgoingLight * vec3( 1.04, 1.03, 1.0 ),
  topFacing * roofFirstStrength * 0.22
);
outgoingLight *= uMoistureSurfaceBrightness;`
      );
  };
  material.customProgramCacheKey = () => "moisture-structure-legibility-v3";
}

function setMoistureShaderUniforms(
  material: THREE.MeshPhongMaterial,
  cameraPosition: THREE.Vector3,
  cameraDirection: THREE.Vector3,
  style: MoistureLayerStyle,
  options: {
    componentColor: THREE.Color;
    isSelected: boolean;
    pressureBandCount: number;
    wallFadePassScale: number;
    backPass: boolean;
  }
) {
  const uniforms = material.userData.moistureShaderUniforms as
    | MoistureShaderUniforms
    | undefined;
  if (!uniforms) return;

  uniforms.cutawayCenter.value.copy(cameraPosition);
  uniforms.cameraDirection.value.copy(cameraDirection);
  uniforms.cutawayRadius.value = style.cameraCutawayRadius;
  uniforms.cutawayEnabled.value = style.cameraCutawayEnabled ? 1 : 0;
  uniforms.solidShellEnabled.value = style.solidShellEnabled ? 1 : 0;
  uniforms.depthCueStrength.value = style.solidShellEnabled
    ? style.lightingEnabled
      ? 0.18
      : 0.32
    : 0;
  uniforms.surfaceCueMode.value = SURFACE_CUE_MODE_VALUES[style.surfaceCueMode];
  uniforms.surfaceBrightness.value = style.surfaceBrightness;
  uniforms.surfaceShadowStrength.value = style.surfaceShadowStrength;
  uniforms.rimEnabled.value = style.rimEnabled ? 1 : 0;
  uniforms.rimStrength.value = style.rimStrength;
  uniforms.distanceFadeEnabled.value = style.distanceFadeEnabled ? 1 : 0;
  uniforms.distanceFadeStrength.value = style.distanceFadeStrength;
  uniforms.colorMode.value = COLOR_MODE_VALUES[style.colorMode];
  uniforms.pressureBandCount.value = options.pressureBandCount;
  uniforms.componentColor.value.copy(options.componentColor);
  uniforms.neutralColor.value.copy(NEUTRAL_COMPONENT_COLOR);
  uniforms.isSelected.value = options.isSelected ? 1 : 0;
  uniforms.verticalWallFadeEnabled.value = style.verticalWallFadeEnabled ? 1 : 0;
  uniforms.verticalWallFadeStrength.value = style.verticalWallFadeStrength;
  uniforms.wallFadePassScale.value = options.wallFadePassScale;
  uniforms.backPass.value = options.backPass ? 1 : 0;
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
  const pressureMix = new Float32Array(component.vertex_count);

  if (verticalExaggeration !== 1) {
    for (let i = 0; i < localPositions.length; i += 3) {
      const x = localPositions[i];
      const y = localPositions[i + 1];
      const z = localPositions[i + 2];
      const radius = Math.hypot(x, y, z);
      if (radius <= 1e-6) continue;

      const levelIndex = levelIndexForRadius(radius, bandScale);
      const levelColor = bandScale.levelColors[levelIndex];
      pressureMix[i / 3] = levelIndex / Math.max(bandScale.levelColors.length - 1, 1);
      localColors[i] = levelColor.r;
      localColors[i + 1] = levelColor.g;
      localColors[i + 2] = levelColor.b;

      const radialOffset = Math.max(radius - baseRadius, 0);
      const exaggeratedRadius =
        baseRadius + MOISTURE_GLOBE_CLEARANCE + radialOffset * verticalExaggeration;
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

      const levelIndex = levelIndexForRadius(radius, bandScale);
      const levelColor = bandScale.levelColors[levelIndex];
      pressureMix[i / 3] = levelIndex / Math.max(bandScale.levelColors.length - 1, 1);
      localColors[i] = levelColor.r;
      localColors[i + 1] = levelColor.g;
      localColors[i + 2] = levelColor.b;

      const elevatedRadius = radius + MOISTURE_GLOBE_CLEARANCE;
      const scale = elevatedRadius / radius;
      localPositions[i] *= scale;
      localPositions[i + 1] *= scale;
      localPositions[i + 2] *= scale;
    }
  }

  for (let i = 0; i < component.index_count; i += 1) {
    localIndices[i] = indices[indexStart + i] - component.vertex_offset;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(localPositions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(localColors, 3));
  geometry.setAttribute("pressureMix", new THREE.BufferAttribute(pressureMix, 1));
  geometry.setIndex(new THREE.BufferAttribute(localIndices, 1));
  geometry.computeVertexNormals();
  return geometry;
}

function createMoistureMaterial(
  side: THREE.Side,
  flatShading: boolean
): THREE.MeshPhongMaterial {
  const material = new THREE.MeshPhongMaterial({
    color: FRONT_TINT.clone(),
    emissive: MOISTURE_EMISSIVE.clone(),
    specular: FRONT_SPECULAR.clone(),
    shininess: 44,
    flatShading,
    transparent: true,
    opacity: 1,
    depthWrite: false,
    depthTest: true,
    side,
    vertexColors: true,
  });
  attachMoistureShader(material);
  return material;
}

function createMoistureOutlineMaterial() {
  return new THREE.MeshBasicMaterial({
    color: OUTLINE_TINT.clone(),
    transparent: true,
    opacity: 0.4,
    depthWrite: false,
    depthTest: true,
    side: THREE.BackSide,
  });
}

function buildSimpleVoxelShellGeometry(frame: MoistureStructureFrame) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(frame.positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(frame.indices, 1));
  geometry.computeBoundingSphere();
  return geometry;
}

function createSimpleVoxelShellMaterial() {
  return new THREE.MeshBasicMaterial({
    color: FRONT_TINT.clone(),
    transparent: false,
    opacity: 1,
    depthWrite: true,
    depthTest: true,
    side: THREE.DoubleSide,
  });
}

function buildSlice(
  frame: MoistureStructureFrame,
  style: MoistureLayerStyle,
  baseOpacity: number
): MoistureSlice {
  const group = new THREE.Group();
  group.name = "moisture-structures-slice";
  group.renderOrder = FRONT_RENDER_ORDER;
  group.frustumCulled = false;

  const components: MoistureComponentVisual[] = [];
  const materials: THREE.Material[] = [];
  const geometries: THREE.BufferGeometry[] = [];
  const opacityWeights = buildComponentOpacityWeights(frame);
  const baseRadius = frame.manifest.globe.base_radius;
  const bandScale = buildLevelBandScale(frame);
  const simpleVoxelShell = style.segmentationMode === "simple-voxel-shell";
  const flatShading = frame.manifest.geometry_mode === "voxel-faces";

  if (simpleVoxelShell) {
    const component = frame.metadata.components[0];
    if (component && frame.indices.length >= 3 && frame.positions.length >= 9) {
      const geometry = buildSimpleVoxelShellGeometry(frame);
      const frontMaterial = createSimpleVoxelShellMaterial();
      const backMaterial = new THREE.MeshBasicMaterial({
        transparent: true,
        opacity: 0,
        depthWrite: false,
        depthTest: false,
      });
      const outlineMaterial = new THREE.MeshBasicMaterial({
        transparent: true,
        opacity: 0,
        depthWrite: false,
        depthTest: false,
      });

      const frontMesh = new THREE.Mesh(geometry, frontMaterial);
      frontMesh.name = `moisture-component-${component.id}-simple-voxel-shell`;
      frontMesh.renderOrder = FRONT_RENDER_ORDER;
      frontMesh.frustumCulled = false;
      frontMesh.userData.moistureComponentId = component.id;

      const backMesh = new THREE.Group();
      backMesh.visible = false;

      const outlineMesh = new THREE.Group();
      outlineMesh.visible = false;

      group.add(frontMesh);

      components.push({
        outlineMesh,
        frontMesh,
        backMesh,
        outlineMaterial,
        frontMaterial,
        backMaterial,
        opacityWeight: 1,
        componentId: component.id,
        bucketIndex: component.bucket_index ?? null,
        componentColor: componentColorForId(component.bucket_index ?? component.id),
        renderKind: "simpleVoxelShell",
      });
      materials.push(outlineMaterial, frontMaterial, backMaterial);
      geometries.push(geometry);
    }

    return {
      group,
      components,
      materials,
      geometries,
      baseOpacity,
      pressureBandCount: bandScale.levelColors.length,
    };
  }

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
    const frontMaterial = createMoistureMaterial(THREE.FrontSide, flatShading);
    const backMaterial = createMoistureMaterial(THREE.BackSide, flatShading);
    const outlineMaterial = createMoistureOutlineMaterial();
    const componentColor = componentColorForId(
      component.bucket_index ?? component.id
    );

    const outlineMesh = new THREE.Mesh(geometry, outlineMaterial);
    outlineMesh.name = `moisture-component-${component.id}-outline`;
    outlineMesh.renderOrder = OUTLINE_RENDER_ORDER;
    outlineMesh.frustumCulled = false;
    outlineMesh.scale.setScalar(1.003);
    outlineMesh.userData.moistureComponentId = component.id;

    const frontMesh = new THREE.Mesh(geometry, frontMaterial);
    frontMesh.name = `moisture-component-${component.id}-front`;
    frontMesh.renderOrder = FRONT_RENDER_ORDER;
    frontMesh.frustumCulled = false;
    frontMesh.userData.moistureComponentId = component.id;

    const backMesh = new THREE.Mesh(geometry, backMaterial);
    backMesh.name = `moisture-component-${component.id}-back`;
    backMesh.renderOrder = BACK_RENDER_ORDER;
    backMesh.frustumCulled = false;
    backMesh.userData.moistureComponentId = component.id;

    group.add(outlineMesh);
    group.add(frontMesh);
    group.add(backMesh);

    components.push({
      outlineMesh,
      frontMesh,
      backMesh,
      outlineMaterial,
      frontMaterial,
      backMaterial,
      opacityWeight: opacityWeights[index] ?? 1,
      componentId: component.id,
      bucketIndex: component.bucket_index ?? null,
      componentColor,
      renderKind: "default",
    });
    materials.push(outlineMaterial, frontMaterial, backMaterial);
    geometries.push(geometry);
  });

  return {
    group,
    components,
    materials,
    geometries,
    baseOpacity,
    pressureBandCount: bandScale.levelColors.length,
  };
}

function normalizeOpacityWeight(weight: number) {
  return THREE.MathUtils.clamp((weight - 0.42) / 0.58, 0, 1);
}

function boostedFrontOpacityWeight(weight: number, solidShellEnabled: boolean) {
  if (!solidShellEnabled) return weight;
  return THREE.MathUtils.lerp(0.88, 1.0, normalizeOpacityWeight(weight));
}

function updateTransparentMode(material: THREE.MeshPhongMaterial, transparent: boolean) {
  if (material.transparent !== transparent) {
    material.transparent = transparent;
    material.needsUpdate = true;
  }
}

function applySliceRenderStyle(
  slice: MoistureSlice | null,
  style: MoistureLayerStyle,
  isTransitioning: boolean
) {
  if (!slice) return;

  const simpleVoxelShell = style.segmentationMode === "simple-voxel-shell";

  const frontDepthWrite =
    simpleVoxelShell || (style.solidShellEnabled && !isTransitioning);
  const frontEmissiveIntensity = style.lightingEnabled
    ? 0.025
    : style.solidShellEnabled
      ? 0.18
      : 0.56;
  const frontShininess = style.lightingEnabled ? 58 : 40;
  const showBackfaces =
    simpleVoxelShell
      ? false
      : !style.solidShellEnabled || style.interiorBackfaceEnabled;
  const backDarkened = style.solidShellEnabled && style.interiorBackfaceEnabled;
  const matteShellFirst =
    style.legibilityExperiment === "bridgePrunedShellFirstMatte";
  const matteSolidShell =
    style.solidShellEnabled ||
    style.legibilityExperiment === "shellFirst" ||
    style.legibilityExperiment === "bridgePrunedShellFirst" ||
    matteShellFirst;
  const visibleBuckets =
    style.segmentationMode === "buckets" ||
    style.segmentationMode === "buckets-global"
      ? new Set(style.visibleBucketIndices)
      : null;

  for (const component of slice.components) {
    const bucketVisible =
      visibleBuckets === null ||
      component.bucketIndex === null ||
      visibleBuckets.has(component.bucketIndex);
    if (!bucketVisible) {
      component.outlineMesh.visible = false;
      component.frontMesh.visible = false;
      component.backMesh.visible = false;
      continue;
    }

    const isSelected =
      style.selectedComponentId !== null &&
      component.componentId === style.selectedComponentId;
    const nonSelectedMultiplier =
      style.selectedComponentId === null || isSelected
        ? 1
        : style.focusMode === "dimOthers"
          ? style.nonSelectedOpacity
          : style.focusMode === "showSelectedOnly"
            ? 0
            : 1;
    let frontWeight = boostedFrontOpacityWeight(
      component.opacityWeight,
      style.solidShellEnabled
    );
    if (
      style.legibilityExperiment === "shellFirst" ||
      style.legibilityExperiment === "bridgePrunedShellFirst" ||
      style.legibilityExperiment === "bridgePrunedShellFirstMatte"
    ) {
      frontWeight = Math.max(frontWeight, 0.96);
    }
    const frontAlpha = THREE.MathUtils.clamp(
      slice.baseOpacity *
        frontWeight *
        style.frontOpacity *
        nonSelectedMultiplier,
      0,
      1
    );
    const backAlpha = THREE.MathUtils.clamp(
      slice.baseOpacity *
        (style.solidShellEnabled ? 1 : component.opacityWeight) *
        style.backfaceOpacity *
        nonSelectedMultiplier,
      0,
      1
    );
    const opaqueFrontPreferred =
      (simpleVoxelShell || frontDepthWrite) &&
      !style.distanceFadeEnabled &&
      nonSelectedMultiplier >= 0.98 &&
      (style.solidShellEnabled || frontAlpha >= 0.94);
    const outlineVisible =
      !simpleVoxelShell &&
      style.solidShellEnabled &&
      !style.interiorBackfaceEnabled &&
      frontAlpha > 0.001;

    if (component.renderKind === "simpleVoxelShell") {
      const frontMaterial = component.frontMaterial as THREE.MeshBasicMaterial;
      component.outlineMesh.visible = false;
      component.backMesh.visible = false;
      component.frontMesh.visible = frontAlpha > 0.001;
      frontMaterial.color.copy(FRONT_TINT);
      frontMaterial.opacity = frontAlpha;
      frontMaterial.transparent = frontAlpha < 0.999;
      frontMaterial.depthWrite = true;
      frontMaterial.depthTest = true;
      frontMaterial.side = THREE.DoubleSide;
      continue;
    }

    component.outlineMesh.visible = outlineVisible;
    component.frontMesh.visible = frontAlpha > 0.001;
    component.backMesh.visible = showBackfaces && backAlpha > 0.001;

    const outlineMaterial = component.outlineMaterial as THREE.MeshBasicMaterial;
    const frontMaterial = component.frontMaterial as THREE.MeshPhongMaterial;
    const backMaterial = component.backMaterial as THREE.MeshPhongMaterial;

    outlineMaterial.opacity = outlineVisible
      ? THREE.MathUtils.clamp(0.04 + frontAlpha * 0.1, 0, 0.14)
      : 0;
    outlineMaterial.color.copy(OUTLINE_TINT);

    updateTransparentMode(
      frontMaterial,
      !opaqueFrontPreferred && (frontAlpha < 0.999 || style.distanceFadeEnabled)
    );
    frontMaterial.opacity = opaqueFrontPreferred ? 1 : frontAlpha;
    frontMaterial.depthWrite = frontDepthWrite;
    frontMaterial.depthTest = true;
    frontMaterial.color.copy(FRONT_TINT);
    frontMaterial.emissive.copy(MOISTURE_EMISSIVE);
    frontMaterial.emissiveIntensity = frontEmissiveIntensity;
    frontMaterial.specular.copy(FRONT_SPECULAR);
    frontMaterial.shininess = matteSolidShell
      ? style.lightingEnabled
        ? 8
        : 8
      : frontShininess;
    frontMaterial.specular.multiplyScalar(
      matteSolidShell ? 0.03 : style.lightingEnabled ? 0.08 : 0.12
    );

    if (simpleVoxelShell) {
      updateTransparentMode(frontMaterial, frontAlpha < 0.999);
      frontMaterial.opacity = frontAlpha;
      frontMaterial.depthWrite = true;
      frontMaterial.depthTest = true;
      frontMaterial.shininess = 8;
      frontMaterial.specular.setScalar(0.05);
    }

    updateTransparentMode(backMaterial, true);
    backMaterial.opacity = backAlpha;
    backMaterial.depthWrite = false;
    backMaterial.depthTest = true;
    backMaterial.color.copy(backDarkened ? INTERIOR_TINT : FRONT_TINT);
    backMaterial.emissive.copy(MOISTURE_EMISSIVE);
    backMaterial.emissiveIntensity = backDarkened
      ? 0.04
      : frontEmissiveIntensity;
    backMaterial.specular.copy(backDarkened ? BACK_SPECULAR : FRONT_SPECULAR);
    backMaterial.shininess = matteSolidShell
      ? backDarkened
        ? 8
        : 10
      : backDarkened
        ? 18
        : frontShininess;
    backMaterial.specular.multiplyScalar(
      matteSolidShell ? (backDarkened ? 0.08 : 0.05) : 0.12
    );
  }
}

function applySliceCameraState(
  slice: MoistureSlice | null,
  cameraPosition: THREE.Vector3,
  cameraDirection: THREE.Vector3,
  style: MoistureLayerStyle
) {
  if (!slice) return;
  const pressureBandCount = Math.max(slice.pressureBandCount, 1);
  for (const component of slice.components) {
    if (component.renderKind === "simpleVoxelShell") {
      continue;
    }
    const isSelected =
      style.selectedComponentId !== null &&
      component.componentId === style.selectedComponentId;
    setMoistureShaderUniforms(
      component.frontMaterial as THREE.MeshPhongMaterial,
      cameraPosition,
      cameraDirection,
      style,
      {
        componentColor: component.componentColor,
        isSelected,
        pressureBandCount,
        wallFadePassScale: 1,
        backPass: false,
      }
    );
    setMoistureShaderUniforms(
      component.backMaterial as THREE.MeshPhongMaterial,
      cameraPosition,
      cameraDirection,
      style,
      {
        componentColor: component.componentColor,
        isSelected,
        pressureBandCount,
        wallFadePassScale: style.interiorBackfaceEnabled ? 0.5 : 0,
        backPass: true,
      }
    );
  }
}

function restoreMoistureLighting(
  ambientLight: THREE.AmbientLight | null,
  keyLight: THREE.DirectionalLight | null,
  headLight: THREE.DirectionalLight | null
) {
  if (ambientLight) ambientLight.intensity = DEFAULT_AMBIENT_INTENSITY;
  if (keyLight) keyLight.intensity = 0;
  if (headLight) headLight.intensity = 0;
}

function applyMoistureLighting(
  style: MoistureLayerStyle,
  cameraPosition: THREE.Vector3,
  cameraDirection: THREE.Vector3,
  ambientLight: THREE.AmbientLight | null,
  keyLight: THREE.DirectionalLight | null,
  headLight: THREE.DirectionalLight | null
) {
  if (!style.visible || !style.lightingEnabled) {
    restoreMoistureLighting(ambientLight, keyLight, headLight);
    return;
  }

  if (ambientLight) ambientLight.intensity = style.ambientIntensity;

  if (keyLight) {
    keyLight.intensity = style.keyLightIntensity;
    keyLight.position.copy(KEY_LIGHT_POSITION);
    keyLight.target.position.set(0, 0, 0);
    keyLight.target.updateMatrixWorld();
  }

  if (headLight) {
    headLight.intensity = style.headLightIntensity;
    headLight.position.copy(cameraPosition);
    headLight.target.position.copy(cameraPosition).addScaledVector(cameraDirection, 120);
    headLight.target.updateMatrixWorld();
  }
}

function disposeFootprintVisual(visual: MoistureFootprintVisual | null) {
  if (!visual) return;
  for (const geometry of visual.geometries) {
    geometry.dispose();
  }
  for (const material of visual.materials) {
    material.dispose();
  }
  visual.group.removeFromParent();
}

function buildFootprintVisual(
  frame: MoistureStructureFrame,
  style: MoistureLayerStyle
): MoistureFootprintVisual | null {
  if (!style.footprintOverlayEnabled || frame.footprints.length === 0) {
    return null;
  }

  const visibleBuckets =
    style.segmentationMode === "buckets" ||
    style.segmentationMode === "buckets-global"
      ? new Set(style.visibleBucketIndices)
      : null;
  const visibleComponents = frame.metadata.components.filter(
    (component) =>
      visibleBuckets === null ||
      component.bucket_index === undefined ||
      visibleBuckets.has(component.bucket_index)
  );

  const selectedIds =
    style.selectedComponentId !== null
      ? [style.selectedComponentId]
      : visibleComponents
          .slice()
          .sort((left, right) => right.voxel_count - left.voxel_count)
          .slice(0, 3)
          .map((component) => component.id);

  const group = new THREE.Group();
  group.name = "moisture-footprints";
  group.renderOrder = FOOTPRINT_RENDER_ORDER;
  group.frustumCulled = false;

  const geometries: THREE.BufferGeometry[] = [];
  const materials: THREE.LineBasicMaterial[] = [];
  const radius = frame.manifest.globe.base_radius + 0.25;
  const footprintLookup = new Map(frame.footprints.map((footprint) => [footprint.id, footprint]));

  for (const componentId of selectedIds) {
    const footprint = footprintLookup.get(componentId);
    if (!footprint) continue;
    const componentMetadata = frame.metadata.components.find(
      (component) => component.id === componentId
    );

    const componentColor = componentColorForId(
      componentMetadata?.bucket_index ?? componentId
    );
    const opacity =
      style.selectedComponentId !== null && componentId === style.selectedComponentId
        ? 0.96
        : 0.78;

    for (const ring of footprint.rings) {
      if (ring.length < 3) continue;

      const linePositions = new Float32Array(ring.length * 3);
      ring.forEach(([longitude, latitude], index) => {
        const vertex = latLonToXYZ(latitude, longitude, radius);
        linePositions[index * 3] = vertex.x;
        linePositions[index * 3 + 1] = vertex.y;
        linePositions[index * 3 + 2] = vertex.z;
      });

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
      const material = new THREE.LineBasicMaterial({
        color: componentColor.clone(),
        transparent: true,
        opacity,
        depthWrite: false,
        depthTest: true,
      });
      const line = new THREE.LineLoop(geometry, material);
      line.renderOrder = FOOTPRINT_RENDER_ORDER;
      line.frustumCulled = false;
      group.add(line);
      geometries.push(geometry);
      materials.push(material);
    }
  }

  if (group.children.length === 0) {
    disposeFootprintVisual({ group, geometries, materials });
    return null;
  }

  return { group, geometries, materials };
}

function publishMoistureFrameState(
  frame: MoistureStructureFrame,
  segmentationMode: MoistureSegmentationMode
) {
  useControls.getState().setMoistureStructureFrame({
    timestamp: frame.metadata.timestamp,
    segmentationMode,
    thresholdedVoxelCount: frame.metadata.thresholded_voxel_count,
    components: frame.metadata.components,
    footprints: frame.footprints,
  });
}

function vector3ToDebugState(vector: THREE.Vector3) {
  return { x: vector.x, y: vector.y, z: vector.z };
}

export default function MoistureStructureLayer() {
  const rawMoistureLayer = useControls((state) => state.moistureStructureLayer);
  const moistureLayer = resolveMoistureStructureLayerState(rawMoistureLayer);
  const visible = moistureLayer.visible;
  const segmentationMode = moistureLayer.segmentationMode;

  const {
    ambientLightRef,
    cameraRef,
    engineReady,
    moistureHeadLightRef,
    moistureKeyLightRef,
    registerDebugAdapter,
    registerFramePass,
    rendererRef,
    sceneRef,
    signalReady,
    timestamp,
    unregisterDebugAdapter,
    unregisterFramePass,
  } = useEarthLayer("moisture-structures");

  const rootRef = useRef<THREE.Group | null>(null);
  const footprintRef = useRef<MoistureFootprintVisual | null>(null);
  const currentRef = useRef<MoistureSlice | null>(null);
  const transitionRef = useRef<MoistureSlice | null>(null);
  const currentFrameRef = useRef<MoistureStructureFrame | null>(null);
  const transitionFrameRef = useRef<MoistureStructureFrame | null>(null);
  const styleRef = useRef<MoistureLayerStyle>(moistureLayer);
  const fadeMixRef = useRef<number | null>(null);
  const reqIdRef = useRef(0);
  const cameraPositionRef = useRef(new THREE.Vector3());
  const cameraDirectionRef = useRef(new THREE.Vector3(0, 0, -1));

  const applyRenderStyle = useCallback((style: MoistureLayerStyle) => {
    const isTransitioning =
      fadeMixRef.current !== null && transitionRef.current !== null;
    applySliceRenderStyle(currentRef.current, style, isTransitioning);
    applySliceRenderStyle(transitionRef.current, style, isTransitioning);
  }, []);

  const applyCameraDrivenState = useCallback(
    (
      cameraPosition: THREE.Vector3,
      cameraDirection: THREE.Vector3,
      style: MoistureLayerStyle
    ) => {
      applySliceCameraState(currentRef.current, cameraPosition, cameraDirection, style);
      applySliceCameraState(
        transitionRef.current,
        cameraPosition,
        cameraDirection,
        style
      );
      applyMoistureLighting(
        style,
        cameraPosition,
        cameraDirection,
        ambientLightRef.current,
        moistureKeyLightRef.current,
        moistureHeadLightRef.current
      );
    },
    [ambientLightRef, moistureHeadLightRef, moistureKeyLightRef]
  );

  const applyVisibleOpacity = useCallback((targetOpacity: number) => {
    const mix = fadeMixRef.current;
    const current = currentRef.current;
    const transition = transitionRef.current;

    if (mix === null || !transition) {
      if (current) current.baseOpacity = targetOpacity;
      applyRenderStyle(styleRef.current);
      return;
    }

    if (current) current.baseOpacity = targetOpacity * (1 - mix);
    transition.baseOpacity = targetOpacity * mix;
    applyRenderStyle(styleRef.current);
  }, [applyRenderStyle]);

  const updateFootprints = useCallback(
    (root: THREE.Group, frame: MoistureStructureFrame | null, style: MoistureLayerStyle) => {
      disposeFootprintVisual(footprintRef.current);
      footprintRef.current = null;

      if (!frame) return;
      const nextFootprints = buildFootprintVisual(frame, style);
      if (!nextFootprints) return;
      root.add(nextFootprints.group);
      footprintRef.current = nextFootprints;
    },
    []
  );

  const rebuildVisibleSlices = useCallback((root: THREE.Group, style: MoistureLayerStyle) => {
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

    applyRenderStyle(style);
    applyCameraDrivenState(
      cameraPositionRef.current,
      cameraDirectionRef.current,
      style
    );
    updateFootprints(root, currentFrame ?? transitionFrame, style);
  }, [applyCameraDrivenState, applyRenderStyle, updateFootprints]);

  const getMoistureDebugState = useCallback(() => {
    const rawLayer = useControls.getState().moistureStructureLayer;
    const resolvedLayer = resolveMoistureStructureLayerState(rawLayer);
    const frame =
      currentFrameRef.current ?? transitionFrameRef.current ?? null;

    return {
      visible: resolvedLayer.visible,
      rawLayer: {
        ...rawLayer,
        visibleBucketIndices: [...rawLayer.visibleBucketIndices],
      },
      resolvedLayer: {
        ...resolvedLayer,
        visibleBucketIndices: [...resolvedLayer.visibleBucketIndices],
      },
      frame: frame
        ? {
            timestamp: frame.metadata.timestamp,
            segmentationMode: resolvedLayer.segmentationMode,
            dataset: frame.manifest.dataset,
            geometryMode: frame.manifest.geometry_mode ?? null,
            thresholdMode: frame.manifest.threshold_mode,
            thresholds: frame.manifest.thresholds,
            entry: frame.entry,
            componentCount: frame.metadata.component_count,
            thresholdedVoxelCount: frame.metadata.thresholded_voxel_count,
            components: frame.metadata.components,
            footprints: frame.footprints,
            positionsFile: frame.metadata.positions_file,
            indicesFile: frame.metadata.indices_file,
          }
        : null,
    };
  }, []);

  const hitTestMoistureTarget = useCallback(
    (target: NormalizedScreenTarget) => {
      const renderer = rendererRef.current;
      const camera = cameraRef.current;
      if (!renderer || !camera) {
        throw new Error("The renderer or camera is not ready.");
      }

      const currentComponents = currentRef.current?.components ?? [];
      const transitionComponents = transitionRef.current?.components ?? [];
      const meshes = [
        ...currentComponents.map((component) => component.frontMesh),
        ...transitionComponents.map((component) => component.frontMesh),
      ];
      if (meshes.length === 0) {
        return {
          analyzer: "moisture-structure",
          target,
          didHit: false,
          reason: "no-moisture-meshes",
        };
      }

      const pointer = new THREE.Vector2(target.x * 2 - 1, -(target.y * 2 - 1));
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(pointer, camera);

      const hit = raycaster.intersectObjects(meshes, false)[0];
      if (!hit) {
        return {
          analyzer: "moisture-structure",
          target,
          didHit: false,
          reason: "no-intersection",
        };
      }

      const componentId =
        typeof hit.object.userData.moistureComponentId === "number"
          ? hit.object.userData.moistureComponentId
          : null;
      const frame =
        currentFrameRef.current ?? transitionFrameRef.current ?? null;
      const resolvedLayer = resolveMoistureStructureLayerState(
        useControls.getState().moistureStructureLayer
      );
      const component =
        frame?.metadata.components.find((entry) => entry.id === componentId) ?? null;
      const footprint =
        frame?.footprints.find((entry) => entry.id === componentId) ?? null;

      return {
        analyzer: "moisture-structure",
        target,
        didHit: true,
        meshName: hit.object.name,
        distance: hit.distance,
        faceIndex: hit.faceIndex ?? null,
        worldPosition: vector3ToDebugState(hit.point),
        componentId,
        component,
        footprint,
        frame: frame
          ? {
              timestamp: frame.metadata.timestamp,
              segmentationMode: resolvedLayer.segmentationMode,
              dataset: frame.manifest.dataset,
              geometryMode: frame.manifest.geometry_mode ?? null,
            }
          : null,
      };
    },
    [cameraRef, rendererRef]
  );

  const selectMoistureDebugTarget = useCallback((targetId: unknown) => {
    const selectedComponentId =
      typeof targetId === "number" ? targetId : targetId === null ? null : null;
    useControls.getState().setMoistureStructureLayer({ selectedComponentId });
    return {
      analyzer: "moisture-structure",
      selectedComponentId,
    };
  }, []);

  useEffect(() => {
    if (!engineReady) return;

    const adapter: ViewDebugAnalyzerAdapter = {
      analyzer: "moisture-structure",
      getState: getMoistureDebugState,
      hitTest: hitTestMoistureTarget,
      selectTarget: selectMoistureDebugTarget,
    };

    registerDebugAdapter(adapter);
    return () => {
      unregisterDebugAdapter(adapter.analyzer);
    };
  }, [
    engineReady,
    getMoistureDebugState,
    hitTestMoistureTarget,
    registerDebugAdapter,
    selectMoistureDebugTarget,
    unregisterDebugAdapter,
  ]);

  useEffect(() => {
    if (!engineReady || !sceneRef.current) return;

    const root = new THREE.Group();
    const state = resolveMoistureStructureLayerState(
      useControls.getState().moistureStructureLayer
    );
    const ambientLight = ambientLightRef.current;
    const keyLight = moistureKeyLightRef.current;
    const headLight = moistureHeadLightRef.current;
    root.name = "moisture-structures-root";
    root.visible = state.visible;
    root.renderOrder = FRONT_RENDER_ORDER;
    root.frustumCulled = false;
    styleRef.current = state;

    sceneRef.current.add(root);
    rootRef.current = root;
    if (cameraRef.current) {
      cameraRef.current.getWorldPosition(cameraPositionRef.current);
      cameraRef.current.getWorldDirection(cameraDirectionRef.current);
      applyCameraDrivenState(
        cameraPositionRef.current,
        cameraDirectionRef.current,
        state
      );
    } else {
      applyMoistureLighting(
        state,
        cameraPositionRef.current,
        cameraDirectionRef.current,
        ambientLightRef.current,
        moistureKeyLightRef.current,
        moistureHeadLightRef.current
      );
    }

    const framePassKey = "moisture-structures-camera-review";
    registerFramePass(framePassKey, () => {
      const camera = cameraRef.current;
      if (!camera) return;
      camera.getWorldPosition(cameraPositionRef.current);
      camera.getWorldDirection(cameraDirectionRef.current);
      applyCameraDrivenState(
        cameraPositionRef.current,
        cameraDirectionRef.current,
        styleRef.current
      );
    });

    const unsubscribe = useControls.subscribe(
      (state) => state.moistureStructureLayer,
      (nextRaw) => {
        const next = resolveMoistureStructureLayerState(nextRaw);
        const prev = styleRef.current;
        styleRef.current = next;
        root.visible = next.visible;
        if (next.verticalExaggeration !== prev.verticalExaggeration) {
          rebuildVisibleSlices(root, next);
          return;
        }
        applyVisibleOpacity(next.opacity);
        applyCameraDrivenState(
          cameraPositionRef.current,
          cameraDirectionRef.current,
          next
        );
        if (
          next.footprintOverlayEnabled !== prev.footprintOverlayEnabled ||
          next.selectedComponentId !== prev.selectedComponentId ||
          next.structurePreset !== prev.structurePreset ||
          next.legibilityExperiment !== prev.legibilityExperiment
        ) {
          updateFootprints(root, currentFrameRef.current, next);
        }
      }
    );

    return () => {
      unsubscribe();
      unregisterFramePass(framePassKey);
      restoreMoistureLighting(ambientLight, keyLight, headLight);
      disposeSlice(transitionRef.current);
      transitionRef.current = null;
      transitionFrameRef.current = null;
      fadeMixRef.current = null;
      disposeFootprintVisual(footprintRef.current);
      footprintRef.current = null;
      disposeSlice(currentRef.current);
      currentRef.current = null;
      currentFrameRef.current = null;
      useControls.getState().setMoistureStructureFrame(null);
      rootRef.current = null;
      root.removeFromParent();
    };
  }, [
    ambientLightRef,
    cameraRef,
    engineReady,
    moistureHeadLightRef,
    moistureKeyLightRef,
    registerFramePass,
    sceneRef,
    unregisterFramePass,
    applyCameraDrivenState,
    applyVisibleOpacity,
    rebuildVisibleSlices,
    updateFootprints,
  ]);

  useEffect(() => {
    if (!engineReady || !rendererRef.current) return;

    const element = rendererRef.current.domElement;
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();

    function onCaptureClick(event: MouseEvent) {
      const style = styleRef.current;
      if (!style.pickMode || !event.altKey || event.button !== 0) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      const camera = cameraRef.current;
      if (!camera) return;

      const meshes =
        currentRef.current?.components.map((component) => component.frontMesh) ??
        transitionRef.current?.components.map((component) => component.frontMesh) ??
        [];
      if (meshes.length === 0) return;

      const rect = element.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);

      const hit = raycaster.intersectObjects(meshes, false)[0];
      const componentId = hit?.object.userData.moistureComponentId;
      useControls.getState().setMoistureStructureLayer({
        selectedComponentId: typeof componentId === "number" ? componentId : null,
      });
    }

    element.addEventListener("click", onCaptureClick, true);
    return () => {
      element.removeEventListener("click", onCaptureClick, true);
    };
  }, [cameraRef, engineReady, rendererRef]);

  useEffect(() => {
    if (!engineReady) return;
    const root = rootRef.current;
    if (!root) return;

    let cancelled = false;
    const requestId = ++reqIdRef.current;
    const isCancelled = () => cancelled || requestId !== reqIdRef.current;
    useControls.getState().setMoistureStructureLayer({ selectedComponentId: null });

    if (!visible) {
      disposeSlice(transitionRef.current);
      transitionRef.current = null;
      transitionFrameRef.current = null;
      fadeMixRef.current = null;
      disposeFootprintVisual(footprintRef.current);
      footprintRef.current = null;
      disposeSlice(currentRef.current);
      currentRef.current = null;
      currentFrameRef.current = null;
      useControls.getState().setMoistureStructureFrame(null);
      root.visible = false;
      restoreMoistureLighting(
        ambientLightRef.current,
        moistureKeyLightRef.current,
        moistureHeadLightRef.current
      );
      signalReady(timestamp);
      return () => {
        cancelled = true;
      };
    }

    root.visible = true;

    void (async () => {
      try {
        const style = styleRef.current;
        const prev = currentRef.current;
        // Crossfading from an empty scene only delays the first visible render.
        const skipInitialFade = !prev;
        const frame = await fetchMoistureStructureFrame(timestamp, {
          segmentationMode,
        });
        if (isCancelled()) return;

        const next = buildSlice(frame, style, skipInitialFade ? style.opacity : 0.0);
        root.add(next.group);
        if (skipInitialFade) {
          currentRef.current = next;
          currentFrameRef.current = frame;
          transitionRef.current = null;
          transitionFrameRef.current = null;
          fadeMixRef.current = null;
          applyVisibleOpacity(style.opacity);
          applyCameraDrivenState(
            cameraPositionRef.current,
            cameraDirectionRef.current,
            style
          );
          updateFootprints(root, frame, style);
          publishMoistureFrameState(frame, style.segmentationMode);
          signalReady(timestamp);
          return;
        }

        transitionRef.current = next;
        transitionFrameRef.current = frame;
        fadeMixRef.current = 0;
        applyVisibleOpacity(style.opacity);
        applyCameraDrivenState(
          cameraPositionRef.current,
          cameraDirectionRef.current,
          style
        );

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
            applyCameraDrivenState(
              cameraPositionRef.current,
              cameraDirectionRef.current,
              styleRef.current
            );
            updateFootprints(root, frame, styleRef.current);
            publishMoistureFrameState(frame, styleRef.current.segmentationMode);
            signalReady(timestamp);
          }
        );
      } catch (error) {
        if (isCancelled()) return;
        console.error("Failed to load moisture structures", error);
        useControls.getState().setMoistureStructureFrame(null);
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
  }, [
    ambientLightRef,
    engineReady,
    moistureHeadLightRef,
    moistureKeyLightRef,
    applyCameraDrivenState,
    applyVisibleOpacity,
    segmentationMode,
    signalReady,
    timestamp,
    updateFootprints,
    visible,
  ]);

  return null;
}
