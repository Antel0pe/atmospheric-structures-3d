import {
  fetchBlobOrThrow,
  fetchJsonOrThrow,
} from "./dataFetchErrors";
import { snapTimestampToAvailable } from "./ApiResponses";
import type { RelativeHumidityVariant } from "../../state/controlsStore";

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
  variant?: string;
  threshold_percent: number;
  geometry_mode: "voxel-faces";
  postprocess?: {
    minimum_component_size: number;
    connectivity: string;
    wraps_longitude: boolean;
  };
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
  postprocess?: {
    minimum_component_size: number;
    connectivity: string;
    wraps_longitude: boolean;
    component_count_before_filter: number;
    component_count_after_filter: number;
    removed_component_count: number;
    removed_voxel_count: number;
  };
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

const relativeHumidityManifestPromiseCache = new Map<
  RelativeHumidityVariant,
  Promise<RelativeHumidityShellManifest>
>();

function relativeHumidityShellBaseSegments(variant: RelativeHumidityVariant) {
  if (variant === "baseline") {
    return ["relative-humidity-shell"];
  }

  return ["relative-humidity-shell", "variants", variant];
}

function buildRelativeHumidityShellUrl(
  variant: RelativeHumidityVariant,
  ...segments: string[]
) {
  return `/${[...relativeHumidityShellBaseSegments(variant), ...segments].join("/")}`;
}

export async function fetchRelativeHumidityShellManifest(opts?: {
  variant?: RelativeHumidityVariant;
  refresh?: boolean;
  notifyOnError?: boolean;
}) {
  const variant = opts?.variant ?? "baseline";
  const refresh = opts?.refresh ?? false;
  const notifyOnError = opts?.notifyOnError ?? true;

  if (!refresh) {
    const cachedPromise = relativeHumidityManifestPromiseCache.get(variant);
    if (cachedPromise) {
      return cachedPromise;
    }
  }

  const manifestPromise = fetchJsonOrThrow<RelativeHumidityShellManifest>(
    buildRelativeHumidityShellUrl(variant, "index.json"),
    "Failed to load relative humidity shell manifest.",
    {
      cache: "no-store",
      layerLabel: "Relative humidity shell",
      notifyOnError,
    }
  ).catch((error) => {
    relativeHumidityManifestPromiseCache.delete(variant);
    throw error;
  });

  relativeHumidityManifestPromiseCache.set(variant, manifestPromise);
  return manifestPromise;
}

export async function fetchRelativeHumidityShellFrame(
  datehour: string,
  opts?: { notifyOnError?: boolean; variant?: RelativeHumidityVariant }
): Promise<RelativeHumidityShellFrame> {
  const notifyOnError = opts?.notifyOnError ?? true;
  const variant = opts?.variant ?? "baseline";
  const manifest = await fetchRelativeHumidityShellManifest({
    notifyOnError,
    refresh: true,
    variant,
  });
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
      buildRelativeHumidityShellUrl(variant, entry.metadata),
      "Failed to load relative humidity shell metadata.",
      {
        layerLabel: "Relative humidity shell",
        notifyOnError,
      }
    ),
    fetchBlobOrThrow(
      buildRelativeHumidityShellUrl(variant, entry.positions),
      "Failed to load relative humidity shell positions.",
      {
        layerLabel: "Relative humidity shell",
        notifyOnError,
      }
    ),
    fetchBlobOrThrow(
      buildRelativeHumidityShellUrl(variant, entry.indices),
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
