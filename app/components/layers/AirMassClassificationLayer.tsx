import { useCallback, useEffect, useRef } from "react";
import * as THREE from "three";
import { type EarthProjectionMode, useEarthLayer } from "./EarthBase";
import {
  isSmoothAirMassClassificationVariant,
  useControls,
  type AirMassClassificationLayerState,
} from "../../state/controlsStore";
import {
  flatMapFaceStaysWithinSeam,
  globeVec3ToLatLon,
  latLonHeightToFlatMapVec3,
} from "../utils/EarthUtils";
import {
  AIR_MASS_CLASS_ORDER,
  type AirMassStructureClassKey,
  type AirMassStructureFrame,
  fetchAirMassStructureFrame,
} from "../utils/airMassStructureAssets";

const LAYER_CLEARANCE = 11.1;
const LAYER_RENDER_ORDER = 69;
const CELL_GRID_RADIAL_OFFSET = 0.08;
const CELL_GRID_OPACITY = 0.82;
const MIN_ALTITUDE_RANGE_SPAN = 0.01;
const SMOOTH_VERTEX_ITERATIONS = 2;
const SMOOTH_VERTEX_ALPHA = 0.34;
const SMOOTH_VERTEX_KEY_SCALE = 100000;

type AirMassAltitudeRange01 =
  AirMassClassificationLayerState["altitudeRange01"];

type AirMassShaderUniforms = {
  cutawayCenter: { value: THREE.Vector3 };
  cutawayRadius: { value: number };
  cutawayEnabled: { value: number };
  altitudeMin: { value: number };
  altitudeMax: { value: number };
  altitudeClipEnabled: { value: number };
};

type AirMassLevelRange = {
  start: number;
  count: number;
  altitudeMix: number;
};

const CLASS_COLOR_STOPS: Partial<Record<
  AirMassStructureClassKey,
  readonly THREE.Color[]
>> = {
  warm_dry: [
    new THREE.Color("#7c2e18"),
    new THREE.Color("#d36a2f"),
    new THREE.Color("#ffb44d"),
    new THREE.Color("#ffe49c"),
  ],
  warm_moist: [
    new THREE.Color("#7f2531"),
    new THREE.Color("#c2534d"),
    new THREE.Color("#ff8f6b"),
    new THREE.Color("#ffd3b3"),
  ],
  cold_dry: [
    new THREE.Color("#11295f"),
    new THREE.Color("#2252a8"),
    new THREE.Color("#5a9ef0"),
    new THREE.Color("#d5ebff"),
  ],
  cold_moist: [
    new THREE.Color("#0f555e"),
    new THREE.Color("#188f99"),
    new THREE.Color("#4fd2c3"),
    new THREE.Color("#dcfff6"),
  ],
};

const BUCKET_COLOR_CONFIG: Record<
  string,
  {
    hue: number;
    lightness: number;
  }
> = {
  bucket_0: { hue: 0.59, lightness: 0.24 },
  bucket_1: { hue: 0.59, lightness: 0.42 },
  bucket_2: { hue: 0.59, lightness: 0.64 },
  bucket_7: { hue: 0.025, lightness: 0.62 },
  bucket_8: { hue: 0.01, lightness: 0.43 },
  bucket_9: { hue: 0.0, lightness: 0.27 },
};

function pressureToStandardAtmosphereHeightM(pressureHpa: number) {
  const safePressure = Math.max(pressureHpa, 1);
  return 44330.0 * (1.0 - (safePressure / 1013.25) ** 0.1903);
}

function standardAtmosphereHeightMToPressure(heightM: number) {
  const normalized = THREE.MathUtils.clamp(1.0 - heightM / 44330.0, 1e-6, 1);
  return 1013.25 * normalized ** (1 / 0.1903);
}

function pressureToRadius(frame: AirMassStructureFrame, pressureHpa: number) {
  const {
    base_radius: baseRadius,
    vertical_span: verticalSpan,
    reference_pressure_hpa: { min: minPressure, max: maxPressure },
  } = frame.manifest.globe;

  const minHeight = pressureToStandardAtmosphereHeightM(maxPressure);
  const maxHeight = pressureToStandardAtmosphereHeightM(minPressure);
  const scale = verticalSpan / Math.max(maxHeight - minHeight, 1e-9);

  return (
    baseRadius +
    (pressureToStandardAtmosphereHeightM(pressureHpa) - minHeight) * scale
  );
}

function radiusToPressureHpa(frame: AirMassStructureFrame, radius: number) {
  const {
    base_radius: baseRadius,
    vertical_span: verticalSpan,
    reference_pressure_hpa: { min: minPressure, max: maxPressure },
  } = frame.manifest.globe;

  const minHeight = pressureToStandardAtmosphereHeightM(maxPressure);
  const maxHeight = pressureToStandardAtmosphereHeightM(minPressure);
  const scale = (maxHeight - minHeight) / Math.max(verticalSpan, 1e-9);
  const heightM = minHeight + Math.max(radius - baseRadius, 0) * scale;
  return standardAtmosphereHeightMToPressure(heightM);
}

function normalizeAltitudeRange01(
  range: AirMassAltitudeRange01 | undefined
): AirMassAltitudeRange01 {
  const min = THREE.MathUtils.clamp(range?.min ?? 0, 0, 1);
  const max = THREE.MathUtils.clamp(range?.max ?? 1, 0, 1);
  if (max - min >= MIN_ALTITUDE_RANGE_SPAN) {
    return { min, max };
  }
  if (min <= 1 - MIN_ALTITUDE_RANGE_SPAN) {
    return { min, max: min + MIN_ALTITUDE_RANGE_SPAN };
  }
  return { min: 1 - MIN_ALTITUDE_RANGE_SPAN, max: 1 };
}

function altitudeMixForPressure(frame: AirMassStructureFrame, pressureHpa: number) {
  const pressureWindow = frame.manifest.pressure_window_hpa;
  const lowerPressure = Math.max(pressureWindow.min, pressureWindow.max);
  const upperPressure = Math.min(pressureWindow.min, pressureWindow.max);
  return THREE.MathUtils.clamp(
    (lowerPressure - pressureHpa) / Math.max(lowerPressure - upperPressure, 1e-6),
    0,
    1
  );
}

function colorAtStopPosition(t: number, stops: readonly THREE.Color[]) {
  const scaled = THREE.MathUtils.clamp(t, 0, 1) * (stops.length - 1);
  const startIndex = Math.floor(scaled);
  const endIndex = Math.min(startIndex + 1, stops.length - 1);
  const mix = scaled - startIndex;
  return stops[startIndex].clone().lerp(stops[endIndex], mix);
}

function bucketColorForPressure(
  frame: AirMassStructureFrame,
  classKey: AirMassStructureClassKey,
  pressureHpa: number
) {
  const config = BUCKET_COLOR_CONFIG[classKey];
  if (!config) return null;
  const pressureWindow = frame.manifest.pressure_window_hpa;
  const lowerPressure = Math.max(pressureWindow.min, pressureWindow.max);
  const upperPressure = Math.min(pressureWindow.min, pressureWindow.max);
  const altitudeT = THREE.MathUtils.clamp(
    (lowerPressure - pressureHpa) / Math.max(lowerPressure - upperPressure, 1e-6),
    0,
    1
  );
  const saturation = THREE.MathUtils.lerp(0.46, 1.0, altitudeT);
  return new THREE.Color().setHSL(config.hue, saturation, config.lightness);
}

function buildBandScale(frame: AirMassStructureFrame, classKey: AirMassStructureClassKey) {
  const pressures = frame.metadata.pressure_levels_hpa;
  if (pressures.length === 0) {
    return { boundaryRadii: [] as number[], levelColors: [] as THREE.Color[] };
  }
  const radii = pressures.map((pressure) => pressureToRadius(frame, pressure));
  const boundaryRadii = radii
    .slice(0, -1)
    .map((radius, index) => (radius + radii[index + 1]) / 2);
  const stops = CLASS_COLOR_STOPS[classKey];
  const levelColors = stops
    ? radii.map((_, index) =>
        colorAtStopPosition(
          index / Math.max(radii.length - 1, 1),
          stops
        )
      )
    : [];
  return { boundaryRadii, levelColors };
}

function levelIndexForRadius(radius: number, boundaryRadii: number[]) {
  let low = 0;
  let high = boundaryRadii.length;

  while (low < high) {
    const mid = (low + high) >> 1;
    if (radius > boundaryRadii[mid]) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
}

function positionKey(positions: Float32Array, vertexIndex: number) {
  const offset = vertexIndex * 3;
  return `${Math.round(positions[offset] * SMOOTH_VERTEX_KEY_SCALE)}:${Math.round(
    positions[offset + 1] * SMOOTH_VERTEX_KEY_SCALE
  )}:${Math.round(positions[offset + 2] * SMOOTH_VERTEX_KEY_SCALE)}`;
}

function connectNeighbor(
  neighbors: Array<Set<number>>,
  groupA: number,
  groupB: number
) {
  if (groupA === groupB) return;
  neighbors[groupA].add(groupB);
  neighbors[groupB].add(groupA);
}

function smoothPositionsByAdjacency(
  positions: Float32Array,
  indices: Uint32Array
) {
  const vertexCount = positions.length / 3;
  if (vertexCount === 0 || indices.length < 3) return positions;

  const groupByVertex = new Uint32Array(vertexCount);
  const groupMap = new Map<string, number>();
  const groupSums: number[] = [];
  const groupCounts: number[] = [];

  for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex += 1) {
    const key = positionKey(positions, vertexIndex);
    let groupIndex = groupMap.get(key);
    if (groupIndex === undefined) {
      groupIndex = groupCounts.length;
      groupMap.set(key, groupIndex);
      groupSums.push(0, 0, 0);
      groupCounts.push(0);
    }

    groupByVertex[vertexIndex] = groupIndex;
    const sourceOffset = vertexIndex * 3;
    const groupOffset = groupIndex * 3;
    groupSums[groupOffset] += positions[sourceOffset];
    groupSums[groupOffset + 1] += positions[sourceOffset + 1];
    groupSums[groupOffset + 2] += positions[sourceOffset + 2];
    groupCounts[groupIndex] += 1;
  }

  const groupPositions = new Float32Array(groupCounts.length * 3);
  for (let groupIndex = 0; groupIndex < groupCounts.length; groupIndex += 1) {
    const groupOffset = groupIndex * 3;
    const count = Math.max(groupCounts[groupIndex], 1);
    groupPositions[groupOffset] = groupSums[groupOffset] / count;
    groupPositions[groupOffset + 1] = groupSums[groupOffset + 1] / count;
    groupPositions[groupOffset + 2] = groupSums[groupOffset + 2] / count;
  }

  const neighbors = Array.from(
    { length: groupCounts.length },
    () => new Set<number>()
  );
  for (let index = 0; index + 2 < indices.length; index += 3) {
    const groupA = groupByVertex[Number(indices[index])];
    const groupB = groupByVertex[Number(indices[index + 1])];
    const groupC = groupByVertex[Number(indices[index + 2])];
    connectNeighbor(neighbors, groupA, groupB);
    connectNeighbor(neighbors, groupB, groupC);
    connectNeighbor(neighbors, groupC, groupA);
  }

  let current = groupPositions;
  for (let iteration = 0; iteration < SMOOTH_VERTEX_ITERATIONS; iteration += 1) {
    const next = current.slice();
    for (let groupIndex = 0; groupIndex < neighbors.length; groupIndex += 1) {
      const neighborSet = neighbors[groupIndex];
      if (neighborSet.size < 2) continue;

      let averageX = 0;
      let averageY = 0;
      let averageZ = 0;
      for (const neighborIndex of neighborSet) {
        const neighborOffset = neighborIndex * 3;
        averageX += current[neighborOffset];
        averageY += current[neighborOffset + 1];
        averageZ += current[neighborOffset + 2];
      }

      const groupOffset = groupIndex * 3;
      const inverseNeighborCount = 1 / neighborSet.size;
      next[groupOffset] = THREE.MathUtils.lerp(
        current[groupOffset],
        averageX * inverseNeighborCount,
        SMOOTH_VERTEX_ALPHA
      );
      next[groupOffset + 1] = THREE.MathUtils.lerp(
        current[groupOffset + 1],
        averageY * inverseNeighborCount,
        SMOOTH_VERTEX_ALPHA
      );
      next[groupOffset + 2] = THREE.MathUtils.lerp(
        current[groupOffset + 2],
        averageZ * inverseNeighborCount,
        SMOOTH_VERTEX_ALPHA
      );
    }
    current = next;
  }

  for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex += 1) {
    const sourceOffset = groupByVertex[vertexIndex] * 3;
    const targetOffset = vertexIndex * 3;
    positions[targetOffset] = current[sourceOffset];
    positions[targetOffset + 1] = current[sourceOffset + 1];
    positions[targetOffset + 2] = current[sourceOffset + 2];
  }

  return positions;
}

function buildGeometry(
  frame: AirMassStructureFrame,
  verticalExaggeration: number,
  classKey: AirMassStructureClassKey,
  projectionMode: EarthProjectionMode,
  smoothAltitudeClip: boolean
) {
  const bandScale = buildBandScale(frame, classKey);
  const fallbackStops = CLASS_COLOR_STOPS[classKey] ?? [
    new THREE.Color("#7f2531"),
    new THREE.Color("#c2534d"),
    new THREE.Color("#ff8f6b"),
    new THREE.Color("#ffd3b3"),
  ];
  const classBuffer = frame.classBuffers[classKey];
  const indices = classBuffer.indices;
  const positions = smoothAltitudeClip
    ? smoothPositionsByAdjacency(classBuffer.positions.slice(), indices)
    : classBuffer.positions.slice();
  const colors = new Float32Array(positions.length);
  const altitudeMixes = new Float32Array(positions.length / 3);
  const vertexLevelIndices = new Uint16Array(positions.length / 3);
  const baseRadius = frame.manifest.globe.base_radius;
  const pressureLevels = frame.metadata.pressure_levels_hpa;
  const levelCount = Math.max(pressureLevels.length, 1);
  const globePosition = new THREE.Vector3();
  const altitudeMixByLevel =
    pressureLevels.length > 0
      ? pressureLevels.map((pressure) => altitudeMixForPressure(frame, pressure))
      : [0];
  const indicesByLevel = Array.from({ length: levelCount }, () => [] as number[]);

  for (let index = 0; index < positions.length; index += 3) {
    const x = positions[index];
    const y = positions[index + 1];
    const z = positions[index + 2];
    const radius = Math.hypot(x, y, z);
    if (radius <= 1e-6) continue;

    const radialOffset = Math.max(radius - baseRadius, 0);
    if (projectionMode === "flat2d") {
      // QUICK AND DIRTY NEED TO REDO FLAT PROJECTION WITH SHARED PROJECTOR/2D
      // ASSETS: unwrap existing globe classification vertices into lon/lat map space.
      globePosition.set(x, y, z);
      const { lat, lon } = globeVec3ToLatLon(globePosition);
      const flatPosition = latLonHeightToFlatMapVec3(
        lat,
        lon,
        LAYER_CLEARANCE + radialOffset * verticalExaggeration
      );
      positions[index] = flatPosition.x;
      positions[index + 1] = flatPosition.y;
      positions[index + 2] = flatPosition.z;
    } else {
      const exaggeratedRadius =
        baseRadius + LAYER_CLEARANCE + radialOffset * verticalExaggeration;
      const scale = exaggeratedRadius / radius;

      positions[index] *= scale;
      positions[index + 1] *= scale;
      positions[index + 2] *= scale;
    }

    const levelIndex = levelIndexForRadius(radius, bandScale.boundaryRadii);
    const pressureHpa = radiusToPressureHpa(frame, radius);
    altitudeMixes[index / 3] = altitudeMixForPressure(frame, pressureHpa);
    vertexLevelIndices[index / 3] = THREE.MathUtils.clamp(
      levelIndex,
      0,
      levelCount - 1
    );
    const color =
      bucketColorForPressure(frame, classKey, pressureHpa) ??
      bandScale.levelColors[levelIndex] ??
      colorAtStopPosition(
        1 - THREE.MathUtils.clamp((pressureHpa - 250) / 750, 0, 1),
        fallbackStops
      );
    colors[index] = color.r;
    colors[index + 1] = color.g;
    colors[index + 2] = color.b;
  }

  for (let index = 0; index < indices.length; index += 6) {
    const faceIndices =
      index + 5 < indices.length
        ? [
            Number(indices[index]),
            Number(indices[index + 1]),
            Number(indices[index + 2]),
            Number(indices[index + 3]),
            Number(indices[index + 4]),
            Number(indices[index + 5]),
          ]
        : Array.from(indices.slice(index, Math.min(index + 3, indices.length)), Number);
    if (
      projectionMode === "flat2d" &&
      !flatMapFaceStaysWithinSeam(positions, faceIndices)
    ) {
      continue;
    }
    const levelIndex = THREE.MathUtils.clamp(
      Math.round(
        faceIndices.reduce((sum, item) => sum + vertexLevelIndices[item], 0) /
          Math.max(faceIndices.length, 1)
      ),
      0,
      levelCount - 1
    );
    indicesByLevel[levelIndex].push(...faceIndices);
  }

  const totalIndexCount = indicesByLevel.reduce(
    (sum, item) => sum + item.length,
    0
  );
  const sortedIndices = new Uint32Array(totalIndexCount);
  const levelRanges: AirMassLevelRange[] = [];
  let sortedIndexOffset = 0;
  for (let levelIndex = 0; levelIndex < indicesByLevel.length; levelIndex += 1) {
    const levelIndicesForGeometry = indicesByLevel[levelIndex];
    const start = sortedIndexOffset;
    sortedIndices.set(levelIndicesForGeometry, sortedIndexOffset);
    sortedIndexOffset += levelIndicesForGeometry.length;
    levelRanges.push({
      start,
      count: levelIndicesForGeometry.length,
      altitudeMix:
        altitudeMixByLevel[levelIndex] ??
        levelIndex / Math.max(levelCount - 1, 1),
    });
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute(
    "airMassAltitudeMix",
    new THREE.BufferAttribute(altitudeMixes, 1)
  );
  geometry.setIndex(
    new THREE.BufferAttribute(sortedIndices, 1)
  );
  geometry.userData.airMassLevelRanges = smoothAltitudeClip
    ? [{ start: 0, count: totalIndexCount, altitudeMix: 0.5 }]
    : levelRanges;
  geometry.userData.airMassSmoothAltitudeClip = smoothAltitudeClip;
  geometry.computeBoundingSphere();
  return geometry;
}

function attachAirMassShader(
  material: THREE.MeshBasicMaterial | THREE.LineBasicMaterial
) {
  const uniforms: AirMassShaderUniforms = {
    cutawayCenter: { value: new THREE.Vector3() },
    cutawayRadius: { value: 0 },
    cutawayEnabled: { value: 0 },
    altitudeMin: { value: 0 },
    altitudeMax: { value: 1 },
    altitudeClipEnabled: { value: 0 },
  };

  material.userData.airMassShaderUniforms = uniforms;
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uAirMassCutawayCenter = uniforms.cutawayCenter;
    shader.uniforms.uAirMassCutawayRadius = uniforms.cutawayRadius;
    shader.uniforms.uAirMassCutawayEnabled = uniforms.cutawayEnabled;
    shader.uniforms.uAirMassAltitudeMin = uniforms.altitudeMin;
    shader.uniforms.uAirMassAltitudeMax = uniforms.altitudeMax;
    shader.uniforms.uAirMassAltitudeClipEnabled =
      uniforms.altitudeClipEnabled;

    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        `#include <common>
attribute float airMassAltitudeMix;
varying vec3 vAirMassWorldPosition;
varying float vAirMassAltitudeMix;`
      )
      .replace(
        "#include <begin_vertex>",
        `#include <begin_vertex>
vAirMassAltitudeMix = airMassAltitudeMix;`
      )
      .replace(
        "#include <worldpos_vertex>",
        `#include <worldpos_vertex>
vec4 airMassWorldPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
airMassWorldPosition = batchingMatrix * airMassWorldPosition;
#endif
#ifdef USE_INSTANCING
airMassWorldPosition = instanceMatrix * airMassWorldPosition;
#endif
airMassWorldPosition = modelMatrix * airMassWorldPosition;
vAirMassWorldPosition = airMassWorldPosition.xyz;`
      );

    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
uniform vec3 uAirMassCutawayCenter;
uniform float uAirMassCutawayRadius;
uniform float uAirMassCutawayEnabled;
uniform float uAirMassAltitudeMin;
uniform float uAirMassAltitudeMax;
uniform float uAirMassAltitudeClipEnabled;
varying vec3 vAirMassWorldPosition;
varying float vAirMassAltitudeMix;`
      )
      .replace(
        "#include <clipping_planes_fragment>",
        `#include <clipping_planes_fragment>
if ( uAirMassAltitudeClipEnabled > 0.5 ) {
  if ( vAirMassAltitudeMix < uAirMassAltitudeMin - 0.001 || vAirMassAltitudeMix > uAirMassAltitudeMax + 0.001 ) {
    discard;
  }
}
if ( uAirMassCutawayEnabled > 0.5 ) {
  vec3 cutawayDelta = vAirMassWorldPosition - uAirMassCutawayCenter;
  if ( dot( cutawayDelta, cutawayDelta ) < uAirMassCutawayRadius * uAirMassCutawayRadius ) {
    discard;
  }
}`
      );
  };
  material.customProgramCacheKey = () => "air-mass-classification-cutaway-v4";
}

function setAirMassShaderUniforms(
  material: THREE.MeshBasicMaterial | THREE.LineBasicMaterial,
  cameraPosition: THREE.Vector3,
  layerState: {
    altitudeRange01: AirMassAltitudeRange01;
    cameraCutawayEnabled: boolean;
    cameraCutawayRadius: number;
    variant: AirMassClassificationLayerState["variant"];
  }
) {
  const uniforms = material.userData.airMassShaderUniforms as
    | AirMassShaderUniforms
    | undefined;
  if (!uniforms) return;

  uniforms.cutawayCenter.value.copy(cameraPosition);
  uniforms.cutawayRadius.value = layerState.cameraCutawayRadius;
  uniforms.cutawayEnabled.value = layerState.cameraCutawayEnabled ? 1 : 0;
  const altitudeRange = normalizeAltitudeRange01(layerState.altitudeRange01);
  uniforms.altitudeMin.value = altitudeRange.min;
  uniforms.altitudeMax.value = altitudeRange.max;
  uniforms.altitudeClipEnabled.value = isSmoothAirMassClassificationVariant(
    layerState.variant
  )
    ? 1
    : 0;
}

function buildCellGridGeometry(
  geometry: THREE.BufferGeometry,
  projectionMode: EarthProjectionMode
) {
  const positionAttribute = geometry.getAttribute("position");
  const colorAttribute = geometry.getAttribute("color");
  const altitudeAttribute = geometry.getAttribute("airMassAltitudeMix");
  const indexAttribute = geometry.getIndex();
  if (
    !(positionAttribute instanceof THREE.BufferAttribute) ||
    !(colorAttribute instanceof THREE.BufferAttribute) ||
    !(altitudeAttribute instanceof THREE.BufferAttribute) ||
    !(indexAttribute instanceof THREE.BufferAttribute)
  ) {
    return new THREE.BufferGeometry();
  }

  const indices = indexAttribute.array;
  const shellRanges = (geometry.userData.airMassLevelRanges ??
    []) as AirMassLevelRange[];
  const linePositions: number[] = [];
  const lineColors: number[] = [];
  const lineAltitudeMixes: number[] = [];
  const lineRanges: AirMassLevelRange[] = [];
  const seenEdges = new Set<string>();
  const edgeColor = new THREE.Color();
  const colorA = new THREE.Color();
  const colorB = new THREE.Color();
  const edgeHighlight = new THREE.Color(1, 1, 1);
  const liftedA = new THREE.Vector3();
  const liftedB = new THREE.Vector3();

  const appendEdge = (startIndex: number, endIndex: number) => {
    if (startIndex === endIndex) return;
    const key =
      startIndex < endIndex
        ? `${startIndex}:${endIndex}`
        : `${endIndex}:${startIndex}`;
    if (seenEdges.has(key)) return;
    seenEdges.add(key);

    liftedA.fromBufferAttribute(positionAttribute, startIndex);
    liftedB.fromBufferAttribute(positionAttribute, endIndex);
    if (projectionMode === "flat2d") {
      liftedA.y += CELL_GRID_RADIAL_OFFSET;
      liftedB.y += CELL_GRID_RADIAL_OFFSET;
    } else {
      const radiusA = Math.max(liftedA.length(), 1e-6);
      const radiusB = Math.max(liftedB.length(), 1e-6);
      liftedA.multiplyScalar((radiusA + CELL_GRID_RADIAL_OFFSET) / radiusA);
      liftedB.multiplyScalar((radiusB + CELL_GRID_RADIAL_OFFSET) / radiusB);
    }

    linePositions.push(
      liftedA.x,
      liftedA.y,
      liftedA.z,
      liftedB.x,
      liftedB.y,
      liftedB.z
    );

    colorA.fromBufferAttribute(colorAttribute, startIndex);
    colorB.fromBufferAttribute(colorAttribute, endIndex);
    edgeColor.copy(colorA).lerp(colorB, 0.5).lerp(edgeHighlight, 0.38);
    lineColors.push(
      edgeColor.r,
      edgeColor.g,
      edgeColor.b,
      edgeColor.r,
      edgeColor.g,
      edgeColor.b
    );
    lineAltitudeMixes.push(
      altitudeAttribute.getX(startIndex),
      altitudeAttribute.getX(endIndex)
    );
  };

  const appendFaceRange = (
    start: number,
    count: number,
    altitudeMix: number
  ) => {
    const lineStart = linePositions.length / 3;
    const end = Math.min(start + count, indices.length);

    for (let index = start; index + 5 < end; index += 6) {
      const corner0 = Number(indices[index]);
      const tri1b = Number(indices[index + 1]);
      const tri1c = Number(indices[index + 2]);
      const corner0Repeat = Number(indices[index + 3]);
      const tri2b = Number(indices[index + 4]);
      const tri2c = Number(indices[index + 5]);
      if (corner0 !== corner0Repeat) continue;

      let corner1 = -1;
      let corner2 = -1;
      let corner3 = -1;

      if (tri1c === tri2b) {
        corner1 = tri1b;
        corner2 = tri1c;
        corner3 = tri2c;
      } else if (tri1b === tri2c) {
        corner1 = tri1c;
        corner2 = tri1b;
        corner3 = tri2b;
      } else {
        continue;
      }

      appendEdge(corner0, corner1);
      appendEdge(corner1, corner2);
      appendEdge(corner2, corner3);
      appendEdge(corner3, corner0);
    }

    lineRanges.push({
      start: lineStart,
      count: linePositions.length / 3 - lineStart,
      altitudeMix,
    });
  };

  if (shellRanges.length > 0) {
    for (const range of shellRanges) {
      appendFaceRange(range.start, range.count, range.altitudeMix);
    }
  } else {
    appendFaceRange(0, indices.length, 0);
  }

  const lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(linePositions), 3)
  );
  lineGeometry.setAttribute(
    "color",
    new THREE.BufferAttribute(new Float32Array(lineColors), 3)
  );
  lineGeometry.setAttribute(
    "airMassAltitudeMix",
    new THREE.BufferAttribute(new Float32Array(lineAltitudeMixes), 1)
  );
  lineGeometry.userData.airMassLevelRanges = lineRanges;
  lineGeometry.userData.airMassSmoothAltitudeClip =
    geometry.userData.airMassSmoothAltitudeClip === true;
  lineGeometry.computeBoundingSphere();
  return lineGeometry;
}

function shouldRenderDoubleSided(frame: AirMassStructureFrame) {
  return (
    frame.metadata.radial_top_faces_drawn === false ||
    frame.manifest.classification.radial_top_faces_drawn === false
  );
}

function drawRangeForAltitude(
  geometry: THREE.BufferGeometry,
  altitudeRange01: AirMassAltitudeRange01
) {
  if (geometry.userData.airMassSmoothAltitudeClip === true) {
    const indexCount = geometry.getIndex()?.count ?? 0;
    return { start: 0, count: indexCount };
  }

  const ranges = (geometry.userData.airMassLevelRanges ??
    []) as AirMassLevelRange[];
  if (ranges.length === 0) {
    const indexCount = geometry.getIndex()?.count ?? 0;
    return { start: 0, count: indexCount };
  }

  const altitudeRange = normalizeAltitudeRange01(altitudeRange01);
  let start = -1;
  let end = 0;

  for (const range of ranges) {
    if (
      range.count <= 0 ||
      range.altitudeMix < altitudeRange.min ||
      range.altitudeMix > altitudeRange.max
    ) {
      continue;
    }
    if (start < 0) {
      start = range.start;
    }
    end = range.start + range.count;
  }

  if (start < 0) {
    return { start: 0, count: 0 };
  }

  return {
    start,
    count: end - start,
  };
}

function setGeometryDrawRangeIfChanged(
  geometry: THREE.BufferGeometry,
  start: number,
  count: number
) {
  if (geometry.drawRange.start === start && geometry.drawRange.count === count) {
    return;
  }
  geometry.setDrawRange(start, count);
}

export default function AirMassClassificationLayer() {
  const layerVisible = useControls((state) => state.airMassLayer.visible);
  const layerVariant = useControls((state) => state.airMassLayer.variant);
  const verticalExaggeration = useControls((state) => state.verticalExaggeration);
  const {
    cameraRef,
    engineReady,
    sceneRef,
    globeRef,
    projectionMode,
    registerFramePass,
    signalReady,
    timestamp,
    unregisterFramePass,
  } = useEarthLayer("air-mass-classification");

  const rootRef = useRef<THREE.Group | null>(null);
  const materialRef = useRef<THREE.MeshBasicMaterial | null>(null);
  const gridMaterialRef = useRef<THREE.LineBasicMaterial | null>(null);
  const meshRefs = useRef<Partial<Record<AirMassStructureClassKey, THREE.Mesh | null>>>(
    {}
  );
  const gridRefs = useRef<
    Partial<Record<AirMassStructureClassKey, THREE.LineSegments | null>>
  >({});
  const frameRef = useRef<AirMassStructureFrame | null>(null);
  const layerStateRef = useRef(useControls.getState().airMassLayer);
  const cameraPositionRef = useRef(new THREE.Vector3());
  const reqIdRef = useRef(0);

  const clearMeshes = useCallback(() => {
    const classKeys = new Set<AirMassStructureClassKey>([
      ...AIR_MASS_CLASS_ORDER,
      ...Object.keys(meshRefs.current),
      ...Object.keys(gridRefs.current),
      ...(frameRef.current?.classKeys ?? []),
    ]);
    for (const classKey of classKeys) {
      const mesh = meshRefs.current[classKey];
      if (mesh) {
        mesh.removeFromParent();
        mesh.geometry.dispose();
        meshRefs.current[classKey] = null;
      }
      const grid = gridRefs.current[classKey];
      if (grid) {
        grid.removeFromParent();
        grid.geometry.dispose();
        gridRefs.current[classKey] = null;
      }
    }
  }, []);

  const applyAltitudeAndClassVisibility = useCallback(() => {
    const state = layerStateRef.current;
    const hiddenClassKeys = new Set(state.hiddenClassKeys);
    for (const classKey of new Set([
      ...Object.keys(meshRefs.current),
      ...Object.keys(gridRefs.current),
    ])) {
      const classVisible = !hiddenClassKeys.has(classKey);
      const mesh = meshRefs.current[classKey];
      if (mesh) {
        const drawRange = drawRangeForAltitude(
          mesh.geometry,
          state.altitudeRange01
        );
        setGeometryDrawRangeIfChanged(
          mesh.geometry,
          drawRange.start,
          drawRange.count
        );
        mesh.visible = classVisible && drawRange.count > 0;
      }
      const grid = gridRefs.current[classKey];
      if (grid) {
        const drawRange = drawRangeForAltitude(
          grid.geometry,
          state.altitudeRange01
        );
        setGeometryDrawRangeIfChanged(
          grid.geometry,
          drawRange.start,
          drawRange.count
        );
        grid.visible =
          classVisible &&
          state.showCellGrid &&
          drawRange.count > 0;
      }
    }
  }, []);

  const applyLayerRuntimeState = useCallback(
    (state: AirMassClassificationLayerState) => {
      const root = rootRef.current;
      const material = materialRef.current;
      const gridMaterial = gridMaterialRef.current;
      if (root) {
        root.visible = state.visible;
      }
      if (material) {
        material.transparent = state.opacity < 0.999;
        material.opacity = state.opacity;
        material.depthWrite = state.opacity >= 0.999;
      }
      if (gridMaterial) {
        gridMaterial.opacity = Math.max(state.opacity * CELL_GRID_OPACITY, 0.12);
      }

      const camera = cameraRef.current;
      if (camera && material && gridMaterial) {
        camera.getWorldPosition(cameraPositionRef.current);
        setAirMassShaderUniforms(
          material,
          cameraPositionRef.current,
          state
        );
        setAirMassShaderUniforms(
          gridMaterial,
          cameraPositionRef.current,
          state
        );
      }

      applyAltitudeAndClassVisibility();
    },
    [applyAltitudeAndClassVisibility, cameraRef]
  );

  useEffect(() => {
    const applyState = (state: AirMassClassificationLayerState) => {
      layerStateRef.current = state;
      applyLayerRuntimeState(state);
    };
    applyState(useControls.getState().airMassLayer);
    return useControls.subscribe((state, previousState) => {
      if (state.airMassLayer === previousState.airMassLayer) return;
      applyState(state.airMassLayer);
    });
  }, [applyLayerRuntimeState]);

  const rebuildMesh = useCallback(() => {
    const root = rootRef.current;
    const material = materialRef.current;
    const gridMaterial = gridMaterialRef.current;
    const frame = frameRef.current;
    if (!root || !material || !gridMaterial || !frame) return;
    root.userData.variant = frame.manifest.variant;
    material.side =
      projectionMode === "flat2d" || shouldRenderDoubleSided(frame)
        ? THREE.DoubleSide
        : THREE.FrontSide;
    material.needsUpdate = true;
    clearMeshes();
    const smoothAltitudeClip = isSmoothAirMassClassificationVariant(
      layerStateRef.current.variant
    );

    for (const classKey of frame.classKeys) {
      const classBuffer = frame.classBuffers[classKey];
      if (classBuffer.positions.length === 0 || classBuffer.indices.length === 0) {
        meshRefs.current[classKey] = null;
        gridRefs.current[classKey] = null;
        continue;
      }

      const geometry = buildGeometry(
        frame,
        verticalExaggeration,
        classKey,
        projectionMode,
        smoothAltitudeClip
      );
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `air-mass-${classKey}-shell`;
      mesh.renderOrder = LAYER_RENDER_ORDER;
      mesh.frustumCulled = false;
      root.add(mesh);
      meshRefs.current[classKey] = mesh;

      if (geometry.index && geometry.index.count > 0) {
        const gridGeometry = buildCellGridGeometry(geometry, projectionMode);
        const grid = new THREE.LineSegments(
          gridGeometry,
          gridMaterial
        );
        grid.name = `air-mass-${classKey}-cell-grid`;
        grid.renderOrder = LAYER_RENDER_ORDER + 1;
        grid.frustumCulled = false;
        root.add(grid);
        gridRefs.current[classKey] = grid;
      } else {
        gridRefs.current[classKey] = null;
      }
    }
    applyAltitudeAndClassVisibility();
  }, [
    applyAltitudeAndClassVisibility,
    clearMeshes,
    projectionMode,
    verticalExaggeration,
  ]);

  useEffect(() => {
    if (!engineReady) return;
    if (!sceneRef.current) return;
    if (projectionMode === "globe" && !globeRef.current) return;

    const root = new THREE.Group();
    root.name = "air-mass-classification-root";
    root.visible = false;
    root.renderOrder = LAYER_RENDER_ORDER;
    root.frustumCulled = false;
    sceneRef.current.add(root);
    rootRef.current = root;

    const material = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: false,
      opacity: 1,
      depthWrite: true,
      depthTest: true,
      side: THREE.FrontSide,
    });
    attachAirMassShader(material);
    materialRef.current = material;

    const gridMaterial = new THREE.LineBasicMaterial({
      color: 0x808080,
      vertexColors: false,
      transparent: true,
      opacity: CELL_GRID_OPACITY,
      depthTest: true,
      depthWrite: false,
    });
    attachAirMassShader(gridMaterial);
    gridMaterialRef.current = gridMaterial;
    applyLayerRuntimeState(layerStateRef.current);

    const framePassKey = "air-mass-classification-camera-cutaway";
    registerFramePass(framePassKey, () => {
      const camera = cameraRef.current;
      const currentMaterial = materialRef.current;
      const currentGridMaterial = gridMaterialRef.current;
      if (!camera || !currentMaterial || !currentGridMaterial) return;
      camera.getWorldPosition(cameraPositionRef.current);
      setAirMassShaderUniforms(
        currentMaterial,
        cameraPositionRef.current,
        layerStateRef.current
      );
      setAirMassShaderUniforms(
        currentGridMaterial,
        cameraPositionRef.current,
        layerStateRef.current
      );
    });

    return () => {
      unregisterFramePass(framePassKey);
      clearMeshes();
      material.dispose();
      gridMaterial.dispose();
      materialRef.current = null;
      gridMaterialRef.current = null;
      rootRef.current = null;
      root.removeFromParent();
    };
  }, [
    applyLayerRuntimeState,
    cameraRef,
    clearMeshes,
    engineReady,
    globeRef,
    projectionMode,
    registerFramePass,
    sceneRef,
    unregisterFramePass,
  ]);

  useEffect(() => {
    if (!engineReady || !frameRef.current || !layerVisible) return;
    rebuildMesh();
  }, [
    engineReady,
    layerVisible,
    rebuildMesh,
    verticalExaggeration,
  ]);

  useEffect(() => {
    if (!engineReady) return;
    const root = rootRef.current;
    if (!root) return;

    let cancelled = false;
    const requestId = ++reqIdRef.current;
    const isCancelled = () => cancelled || requestId !== reqIdRef.current;

    if (!layerVisible) {
      root.visible = false;
      signalReady(timestamp);
      return () => {
        cancelled = true;
      };
    }

    root.visible = true;

    void fetchAirMassStructureFrame(timestamp, {
      variant: layerVariant,
    })
      .then((frame) => {
        if (isCancelled()) return;
        frameRef.current = frame;
        rebuildMesh();
        signalReady(timestamp);
      })
      .catch((error) => {
        if (isCancelled()) return;
        console.error("Failed to load air-mass classification layer", error);
        signalReady(timestamp);
      });

    return () => {
      cancelled = true;
    };
  }, [
    engineReady,
    layerVariant,
    layerVisible,
    rebuildMesh,
    signalReady,
    timestamp,
  ]);

  return null;
}
