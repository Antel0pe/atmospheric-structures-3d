import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { ViewerNavigationCommand } from "../../lib/viewerTypes";

const CENTER = new THREE.Vector3(0, 0, 0);

export const ZOOM_RADIUS_MIN = 115;
export const ZOOM_RADIUS_MAX = 400;
export const SURFACE_SPEED = 200;
export const MOUSE_SENS = 0.002;
export const PITCH_MAX = THREE.MathUtils.degToRad(89.99);
export const BUTTON_MOVE_DT = 0.2;
export const LOOK_STEP_RADIANS = 0.14;
export const NAVIGATION_PITCH_EPSILON = 1e-4;

type NavigationPose = {
  yaw: number;
  pitch: number;
};

export function clampNavigationPitch(pitch: number) {
  return THREE.MathUtils.clamp(
    pitch,
    -PITCH_MAX + NAVIGATION_PITCH_EPSILON,
    PITCH_MAX - NAVIGATION_PITCH_EPSILON
  );
}

export function stabilizeNavigationPose(pose: NavigationPose): NavigationPose {
  return {
    yaw: pose.yaw,
    pitch: clampNavigationPitch(pose.pitch),
  };
}

function buildLocalFrame(position: THREE.Vector3) {
  const up = position.clone().sub(CENTER).normalize();
  const ref =
    Math.abs(up.y) > 0.99
      ? new THREE.Vector3(1, 0, 0)
      : new THREE.Vector3(0, 1, 0);
  const east = new THREE.Vector3().crossVectors(ref, up).normalize();
  const north = new THREE.Vector3().crossVectors(up, east).normalize();

  return { up, east, north };
}

export function previewCameraOrientationFromYawPitch(
  cameraPosition: THREE.Vector3,
  yaw: number,
  pitch: number
) {
  const { up, east, north } = buildLocalFrame(cameraPosition);
  const pitchedNorth = north.clone().applyAxisAngle(east, pitch);
  const forward = pitchedNorth.clone().applyAxisAngle(up, yaw);
  const right = forward.clone().cross(up).normalize();
  const cameraUp = right.clone().cross(forward).normalize();
  const zAxis = forward.clone().negate();
  const rotationMatrix = new THREE.Matrix4().makeBasis(right, cameraUp, zAxis);
  const cameraQuaternion = new THREE.Quaternion().setFromRotationMatrix(rotationMatrix);

  return {
    forward,
    cameraUp,
    cameraQuaternion,
  };
}

export function deriveYawPitchFromCamera(
  cameraPosition: THREE.Vector3,
  cameraQuaternion: THREE.Quaternion
): NavigationPose {
  const { up, east, north } = buildLocalFrame(cameraPosition);
  const forward = new THREE.Vector3(0, 0, -1)
    .applyQuaternion(cameraQuaternion)
    .normalize();
  const verticalComponent = THREE.MathUtils.clamp(forward.dot(up), -1, 1);
  const pitch = Math.asin(verticalComponent);
  const horizontal = forward
    .clone()
    .addScaledVector(up, -verticalComponent);

  if (horizontal.lengthSq() < 1e-12) {
    return stabilizeNavigationPose({ yaw: 0, pitch });
  }

  horizontal.normalize();
  const yaw = Math.atan2(-horizontal.dot(east), horizontal.dot(north));
  return stabilizeNavigationPose({ yaw, pitch });
}

export function syncCameraFromYawPitch(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  yaw: number,
  pitch: number
) {
  const safePitch = clampNavigationPitch(pitch);
  const { forward, cameraUp, cameraQuaternion } =
    previewCameraOrientationFromYawPitch(camera.position, yaw, safePitch);
  camera.up.copy(cameraUp);
  camera.quaternion.copy(cameraQuaternion);
  controls.target.copy(camera.position).add(forward);
  controls.update();
}

export function canonicalizeCameraForNavigation(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls
) {
  const pose = deriveYawPitchFromCamera(camera.position, camera.quaternion);
  syncCameraFromYawPitch(camera, controls, pose.yaw, pose.pitch);
  return pose;
}

export function applyMouseLookDelta(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  pose: NavigationPose,
  deltaX: number,
  deltaY: number
) {
  const yaw = pose.yaw - deltaX * MOUSE_SENS;
  const pitch = THREE.MathUtils.clamp(
    pose.pitch - deltaY * MOUSE_SENS,
    -PITCH_MAX,
    PITCH_MAX
  );

  syncCameraFromYawPitch(camera, controls, yaw, pitch);
  return { yaw, pitch };
}

type MovementInput = {
  forward: number;
  right: number;
  vertical: number;
};

type MoveCameraParams = {
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  pose: NavigationPose;
  dt: number;
  input: MovementInput;
};

export function moveCameraAlongGlobe({
  camera,
  controls,
  pose,
  dt,
  input,
}: MoveCameraParams) {
  const localUp = camera.position.clone().sub(CENTER).normalize();
  const viewForward = new THREE.Vector3(0, 0, -1).applyQuaternion(
    camera.quaternion
  );
  const viewRight = new THREE.Vector3(1, 0, 0).applyQuaternion(
    camera.quaternion
  );

  const forwardTangent = viewForward
    .clone()
    .addScaledVector(localUp, -viewForward.dot(localUp));
  const rightTangent = viewRight
    .clone()
    .addScaledVector(localUp, -viewRight.dot(localUp));

  if (forwardTangent.lengthSq() < 1e-12) {
    forwardTangent.copy(rightTangent);
  }
  forwardTangent.normalize();

  if (rightTangent.lengthSq() < 1e-12) {
    rightTangent.crossVectors(localUp, forwardTangent).normalize();
  } else {
    rightTangent.normalize();
  }

  const direction = new THREE.Vector3();
  if (input.forward !== 0) {
    direction.addScaledVector(forwardTangent, input.forward);
  }
  if (input.right !== 0) {
    direction.addScaledVector(rightTangent, input.right);
  }

  let didMove = false;

  if (direction.lengthSq() > 1e-10) {
    direction.normalize();
    const radius = camera.position.distanceTo(CENTER);
    const angle = (SURFACE_SPEED / Math.max(1e-6, radius)) * dt;
    const axis = new THREE.Vector3().crossVectors(localUp, direction).normalize();
    const rotation = new THREE.Quaternion().setFromAxisAngle(axis, angle);
    camera.position.sub(CENTER).applyQuaternion(rotation).add(CENTER);
    didMove = true;
  }

  if (input.vertical !== 0) {
    const climb = SURFACE_SPEED * 0.5 * dt * input.vertical;
    const nextPosition = camera.position
      .clone()
      .add(localUp.clone().multiplyScalar(climb));
    const radius = THREE.MathUtils.clamp(
      nextPosition.length(),
      ZOOM_RADIUS_MIN,
      ZOOM_RADIUS_MAX
    );
    camera.position.copy(nextPosition.normalize().multiplyScalar(radius));
    didMove = true;
  }

  if (!didMove) {
    return {
      didMove: false,
      yaw: pose.yaw,
      pitch: pose.pitch,
      zoom01: camera.position.length(),
    };
  }

  syncCameraFromYawPitch(camera, controls, pose.yaw, pose.pitch);
  return {
    didMove: true,
    yaw: pose.yaw,
    pitch: pose.pitch,
    zoom01: camera.position.length(),
  };
}

export function applyDiscreteNavigationCommand(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  pose: NavigationPose,
  command: ViewerNavigationCommand
) {
  switch (command) {
    case "move-forward":
      return moveCameraAlongGlobe({
        camera,
        controls,
        pose,
        dt: BUTTON_MOVE_DT,
        input: { forward: 1, right: 0, vertical: 0 },
      });
    case "move-backward":
      return moveCameraAlongGlobe({
        camera,
        controls,
        pose,
        dt: BUTTON_MOVE_DT,
        input: { forward: -1, right: 0, vertical: 0 },
      });
    case "move-left":
      return moveCameraAlongGlobe({
        camera,
        controls,
        pose,
        dt: BUTTON_MOVE_DT,
        input: { forward: 0, right: -1, vertical: 0 },
      });
    case "move-right":
      return moveCameraAlongGlobe({
        camera,
        controls,
        pose,
        dt: BUTTON_MOVE_DT,
        input: { forward: 0, right: 1, vertical: 0 },
      });
    case "move-up":
      return moveCameraAlongGlobe({
        camera,
        controls,
        pose,
        dt: BUTTON_MOVE_DT,
        input: { forward: 0, right: 0, vertical: 1 },
      });
    case "move-down":
      return moveCameraAlongGlobe({
        camera,
        controls,
        pose,
        dt: BUTTON_MOVE_DT,
        input: { forward: 0, right: 0, vertical: -1 },
      });
    case "look-left": {
      const yaw = pose.yaw + LOOK_STEP_RADIANS;
      syncCameraFromYawPitch(camera, controls, yaw, pose.pitch);
      return { didMove: true, yaw, pitch: pose.pitch, zoom01: camera.position.length() };
    }
    case "look-right": {
      const yaw = pose.yaw - LOOK_STEP_RADIANS;
      syncCameraFromYawPitch(camera, controls, yaw, pose.pitch);
      return { didMove: true, yaw, pitch: pose.pitch, zoom01: camera.position.length() };
    }
    case "look-up": {
      const pitch = THREE.MathUtils.clamp(
        pose.pitch + LOOK_STEP_RADIANS,
        -PITCH_MAX,
        PITCH_MAX
      );
      syncCameraFromYawPitch(camera, controls, pose.yaw, pitch);
      return { didMove: true, yaw: pose.yaw, pitch, zoom01: camera.position.length() };
    }
    case "look-down": {
      const pitch = THREE.MathUtils.clamp(
        pose.pitch - LOOK_STEP_RADIANS,
        -PITCH_MAX,
        PITCH_MAX
      );
      syncCameraFromYawPitch(camera, controls, pose.yaw, pitch);
      return { didMove: true, yaw: pose.yaw, pitch, zoom01: camera.position.length() };
    }
  }
}
