export type Vector3State = {
  x: number;
  y: number;
  z: number;
};

export type QuaternionState = {
  x: number;
  y: number;
  z: number;
  w: number;
};

export type EarthViewState = {
  cameraPosition: Vector3State;
  cameraQuaternion: QuaternionState;
  cameraUp: Vector3State;
  controlsTarget: Vector3State;
  yaw: number;
  pitch: number;
  zoom01: number;
};

export type SavedViewRecord = {
  schemaVersion: 1;
  id: string;
  title: string;
  description: string;
  createdAt: string;
  timestamp: string;
  earthView: EarthViewState;
};

export type SavedViewInput = {
  title: string;
  description: string;
  timestamp: string;
  earthView: EarthViewState;
};

export type SavedViewApplyPhase = "initial" | "ready";

export type SavedViewApplyRequest = {
  requestId: number;
  phase: SavedViewApplyPhase;
  savedView: SavedViewRecord;
};

export type ViewerNavigationCommand =
  | "move-forward"
  | "move-backward"
  | "move-left"
  | "move-right"
  | "move-up"
  | "move-down"
  | "look-left"
  | "look-right"
  | "look-up"
  | "look-down";

export type ViewerNavigationRequest = {
  requestId: number;
  command: ViewerNavigationCommand;
};

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isVector3State(value: unknown): value is Vector3State {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    isFiniteNumber(candidate.x) &&
    isFiniteNumber(candidate.y) &&
    isFiniteNumber(candidate.z)
  );
}

function isQuaternionState(value: unknown): value is QuaternionState {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    isFiniteNumber(candidate.x) &&
    isFiniteNumber(candidate.y) &&
    isFiniteNumber(candidate.z) &&
    isFiniteNumber(candidate.w)
  );
}

export function isEarthViewState(value: unknown): value is EarthViewState {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    isVector3State(candidate.cameraPosition) &&
    isQuaternionState(candidate.cameraQuaternion) &&
    isVector3State(candidate.cameraUp) &&
    isVector3State(candidate.controlsTarget) &&
    isFiniteNumber(candidate.yaw) &&
    isFiniteNumber(candidate.pitch) &&
    isFiniteNumber(candidate.zoom01)
  );
}

export function isSavedViewInput(value: unknown): value is SavedViewInput {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.title === "string" &&
    typeof candidate.description === "string" &&
    typeof candidate.timestamp === "string" &&
    isEarthViewState(candidate.earthView)
  );
}

export function isSavedViewRecord(value: unknown): value is SavedViewRecord {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    candidate.schemaVersion === 1 &&
    typeof candidate.id === "string" &&
    typeof candidate.title === "string" &&
    typeof candidate.description === "string" &&
    typeof candidate.createdAt === "string" &&
    typeof candidate.timestamp === "string" &&
    isEarthViewState(candidate.earthView)
  );
}
