import {
  fetchBlobOrThrow,
  fetchJsonOrThrow,
} from "./dataFetchErrors";
import { snapTimestampToAvailable } from "./ApiResponses";

export type PrecipitableWaterProxyThresholdEntry = {
  pressure_hpa: number;
  active_pressure_window: boolean;
  specific_humidity_threshold: number | null;
  relative_humidity_threshold: number | null;
};

export type PrecipitableWaterProxyManifestTimestamp = {
  timestamp: string;
  metadata: string;
  positions: string;
  indices: string;
  component_count: number;
  vertex_count: number;
  index_count: number;
};

export type PrecipitableWaterProxyManifest = {
  version: number;
  datasets: {
    specific_humidity: string;
    relative_humidity: string;
  };
  variables: {
    specific_humidity: string;
    relative_humidity: string;
  };
  units: {
    specific_humidity: string;
    relative_humidity: string;
  };
  structure_kind: "precipitable-water-proxy-voxel-shell";
  geometry_mode: "voxel-faces";
  gates: {
    specific_humidity: {
      kind: string;
      kept_top_percent: number;
      quantile: number;
      threshold_seed: string;
    };
    relative_humidity: {
      minimum_percent: number;
    };
    vertical_depth: {
      minimum_adjacent_levels: number;
    };
    pressure_window: {
      min_hpa: number;
      max_hpa: number;
    };
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
  thresholds: PrecipitableWaterProxyThresholdEntry[];
  timestamps: PrecipitableWaterProxyManifestTimestamp[];
};

export type PrecipitableWaterProxyComponentMetadata = {
  id: number;
  vertex_offset: number;
  vertex_count: number;
  index_offset: number;
  index_count: number;
  voxel_count: number;
  mean_specific_humidity: number;
  max_specific_humidity: number;
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

export type PrecipitableWaterProxyMetadata = {
  version: number;
  timestamp: string;
  component_count: number;
  vertex_count: number;
  index_count: number;
  thresholded_voxel_count: number;
  gate_counts: {
    finite_pressure_window_voxel_count: number;
    specific_humidity_gate_voxel_count: number;
    relative_humidity_gate_voxel_count: number;
    combined_gate_voxel_count: number;
    depth_gate_voxel_count: number;
  };
  components: PrecipitableWaterProxyComponentMetadata[];
  positions_file: string;
  indices_file: string;
};

export type PrecipitableWaterProxyFrame = {
  manifest: PrecipitableWaterProxyManifest;
  entry: PrecipitableWaterProxyManifestTimestamp;
  metadata: PrecipitableWaterProxyMetadata;
  positions: Float32Array;
  indices: Uint32Array;
};

let precipitableWaterProxyManifestPromise: Promise<PrecipitableWaterProxyManifest> | null =
  null;

function buildPrecipitableWaterProxyUrl(...segments: string[]) {
  return `/precipitable-water-proxy/${segments.join("/")}`;
}

export async function fetchPrecipitableWaterProxyManifest(opts?: {
  refresh?: boolean;
  notifyOnError?: boolean;
}) {
  const refresh = opts?.refresh ?? false;
  const notifyOnError = opts?.notifyOnError ?? true;

  if (!refresh && precipitableWaterProxyManifestPromise) {
    return precipitableWaterProxyManifestPromise;
  }

  precipitableWaterProxyManifestPromise = fetchJsonOrThrow<PrecipitableWaterProxyManifest>(
    buildPrecipitableWaterProxyUrl("index.json"),
    "Failed to load precipitable water proxy manifest.",
    {
      cache: "no-store",
      layerLabel: "Precipitable water proxy",
      notifyOnError,
    }
  ).catch((error) => {
    precipitableWaterProxyManifestPromise = null;
    throw error;
  });

  return precipitableWaterProxyManifestPromise;
}

export async function fetchPrecipitableWaterProxyFrame(
  datehour: string,
  opts?: { notifyOnError?: boolean }
): Promise<PrecipitableWaterProxyFrame> {
  const notifyOnError = opts?.notifyOnError ?? true;
  const manifest = await fetchPrecipitableWaterProxyManifest({
    notifyOnError,
    refresh: true,
  });
  const availableValues = manifest.timestamps.map((item) => item.timestamp);
  const resolvedTimestamp = snapTimestampToAvailable(datehour, availableValues);
  const entry =
    manifest.timestamps.find((item) => item.timestamp === resolvedTimestamp) ??
    manifest.timestamps[0];

  if (!entry) {
    throw new Error("No precipitable water proxy assets are available.");
  }

  const [metadata, positionsBlob, indicesBlob] = await Promise.all([
    fetchJsonOrThrow<PrecipitableWaterProxyMetadata>(
      buildPrecipitableWaterProxyUrl(entry.metadata),
      "Failed to load precipitable water proxy metadata.",
      {
        layerLabel: "Precipitable water proxy",
        notifyOnError,
      }
    ),
    fetchBlobOrThrow(
      buildPrecipitableWaterProxyUrl(entry.positions),
      "Failed to load precipitable water proxy positions.",
      {
        layerLabel: "Precipitable water proxy",
        notifyOnError,
      }
    ),
    fetchBlobOrThrow(
      buildPrecipitableWaterProxyUrl(entry.indices),
      "Failed to load precipitable water proxy indices.",
      {
        layerLabel: "Precipitable water proxy",
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
