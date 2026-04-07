import {
  fetchBlobOrThrow,
  fetchJsonOrThrow,
} from "./dataFetchErrors";
import { snapTimestampToAvailable } from "./ApiResponses";

export type RelativeHumidityThresholdEntry = {
  pressure_hpa: number;
  threshold: number;
};

export type RelativeHumidityShellManifestTimestamp = {
  timestamp: string;
  metadata: string;
  positions: string;
  indices: string;
  component_count: number;
  vertex_count: number;
  index_count: number;
};

export type RelativeHumidityShellManifest = {
  version: number;
  dataset: string;
  variable: string;
  units: string;
  structure_kind: "relative-humidity-voxel-shell";
  threshold_percent: number;
  geometry_mode: "voxel-faces";
  globe: {
    base_radius: number;
    vertical_span: number;
    reference_pressure_hpa: {
      min: number;
      max: number;
    };
  };
  grid: {
    pressure_level_count: number;
    latitude_count: number;
    longitude_count: number;
    latitude_step_degrees: number | null;
    longitude_step_degrees: number | null;
  };
  thresholds: RelativeHumidityThresholdEntry[];
  timestamps: RelativeHumidityShellManifestTimestamp[];
};

export type RelativeHumidityShellComponentMetadata = {
  id: number;
  vertex_offset: number;
  vertex_count: number;
  index_offset: number;
  index_count: number;
  voxel_count: number;
  mean_relative_humidity: number;
  max_relative_humidity: number;
  pressure_min_hpa: number;
  pressure_max_hpa: number;
  latitude_min_deg: number;
  latitude_max_deg: number;
  longitude_min_deg: number;
  longitude_max_deg: number;
  wraps_longitude_seam: boolean;
};

export type RelativeHumidityShellMetadata = {
  version: number;
  timestamp: string;
  component_count: number;
  vertex_count: number;
  index_count: number;
  thresholded_voxel_count: number;
  components: RelativeHumidityShellComponentMetadata[];
  positions_file: string;
  indices_file: string;
};

export type RelativeHumidityShellFrame = {
  manifest: RelativeHumidityShellManifest;
  entry: RelativeHumidityShellManifestTimestamp;
  metadata: RelativeHumidityShellMetadata;
  positions: Float32Array;
  indices: Uint32Array;
};

const relativeHumidityManifestPromise: {
  current: Promise<RelativeHumidityShellManifest> | null;
} = {
  current: null,
};

function buildRelativeHumidityShellUrl(...segments: string[]) {
  return `/${["relative-humidity-shell", ...segments].join("/")}`;
}

export async function fetchRelativeHumidityShellManifest(opts?: {
  refresh?: boolean;
  notifyOnError?: boolean;
}) {
  const refresh = opts?.refresh ?? false;
  const notifyOnError = opts?.notifyOnError ?? true;

  if (!refresh && relativeHumidityManifestPromise.current) {
    return relativeHumidityManifestPromise.current;
  }

  relativeHumidityManifestPromise.current =
    fetchJsonOrThrow<RelativeHumidityShellManifest>(
      buildRelativeHumidityShellUrl("index.json"),
      "Failed to load relative humidity shell manifest.",
      {
        layerLabel: "Relative humidity shell",
        notifyOnError,
      }
    ).catch((error) => {
      relativeHumidityManifestPromise.current = null;
      throw error;
    });

  return relativeHumidityManifestPromise.current;
}

export async function fetchRelativeHumidityShellFrame(
  datehour: string,
  opts?: { notifyOnError?: boolean }
): Promise<RelativeHumidityShellFrame> {
  const notifyOnError = opts?.notifyOnError ?? true;
  const manifest = await fetchRelativeHumidityShellManifest({ notifyOnError });
  const availableValues = manifest.timestamps.map((item) => item.timestamp);
  const resolvedTimestamp = snapTimestampToAvailable(datehour, availableValues);
  const entry =
    manifest.timestamps.find((item) => item.timestamp === resolvedTimestamp) ??
    manifest.timestamps[0];

  if (!entry) {
    throw new Error("No relative humidity shell assets are available.");
  }

  const [metadata, positionsBlob, indicesBlob] = await Promise.all([
    fetchJsonOrThrow<RelativeHumidityShellMetadata>(
      buildRelativeHumidityShellUrl(entry.metadata),
      "Failed to load relative humidity shell metadata.",
      {
        layerLabel: "Relative humidity shell",
        notifyOnError,
      }
    ),
    fetchBlobOrThrow(
      buildRelativeHumidityShellUrl(entry.positions),
      "Failed to load relative humidity shell positions.",
      {
        layerLabel: "Relative humidity shell",
        notifyOnError,
      }
    ),
    fetchBlobOrThrow(
      buildRelativeHumidityShellUrl(entry.indices),
      "Failed to load relative humidity shell indices.",
      {
        layerLabel: "Relative humidity shell",
        notifyOnError,
      }
    ),
  ]);

  const [positionsBuffer, indicesBuffer] = await Promise.all([
    positionsBlob.arrayBuffer(),
    indicesBlob.arrayBuffer(),
  ]);

  return {
    manifest,
    entry,
    metadata,
    positions: new Float32Array(positionsBuffer),
    indices: new Uint32Array(indicesBuffer),
  };
}
