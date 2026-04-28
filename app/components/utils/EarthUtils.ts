import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export const FLAT_EARTH_MAP_WIDTH = 360;
export const FLAT_EARTH_MAP_HEIGHT = 180;

// 1) lat/lon -> world position on your globe (origin-centered)
export function latLonToVec3(latDeg: number, lonDeg: number, radius: number, lonOffsetDeg = 270, latOffsetDeg = 0) {
    const lat = THREE.MathUtils.degToRad(latDeg + latOffsetDeg);
    const lon = THREE.MathUtils.degToRad(-(lonDeg + lonOffsetDeg)); // inverting for threejs coord system
    const x = radius * Math.cos(lat) * Math.cos(lon);
    const y = radius * Math.sin(lat);
    const z = radius * Math.cos(lat) * Math.sin(lon);
    return new THREE.Vector3(x, y, z);
}

export function globeVec3ToLatLon(vector: THREE.Vector3) {
    const radius = Math.max(vector.length(), 1e-6);
    const lat = THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(vector.y / radius, -1, 1)));
    const lonRaw = -THREE.MathUtils.radToDeg(Math.atan2(vector.z, vector.x)) - 270;
    const lon = ((((lonRaw + 180) % 360) + 360) % 360) - 180;
    return { lat, lon, radius };
}

export function latLonHeightToFlatMapVec3(latDeg: number, lonDeg: number, height = 0) {
    return new THREE.Vector3(lonDeg, height, -latDeg);
}

// 2) compute globe radius from the ThreeGlobe mesh
export function getGlobeRadius(globe: THREE.Object3D) {
    const sphere = new THREE.Sphere();
    new THREE.Box3().setFromObject(globe).getBoundingSphere(sphere);
    return sphere.radius;
}

// 3) fly the camera to a given lat/lon
export function lookAtLatLon(
    lat: number,
    lon: number,
    camera: THREE.PerspectiveCamera,
    controls: OrbitControls,
    globe: THREE.Object3D,
    altitude = 0 // extra distance above surface, in world units
) {
    const R = getGlobeRadius(globe);
    const target = latLonToVec3(lat, lon, R);      // point on surface
    const normal = target.clone().normalize();

    // keep roughly the same viewing distance unless you specify altitude
    const keepDist = camera.position.distanceTo(controls.target);
    const dist = altitude > 0 ? altitude : keepDist;

    const newPos = normal.clone().multiplyScalar(R + dist);

    // snap (or tween if you prefer)
    controls.target.copy(target);
    camera.position.copy(newPos);
    camera.lookAt(controls.target);
    controls.update();
}
