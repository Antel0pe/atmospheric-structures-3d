import { fetchBlobOrThrow, fetchJsonOrThrow } from "./dataFetchErrors";
import { snapTimestampToAvailable } from "./ApiResponses";

export type PotentialTemperatureManifestTimestamp = {
  timestamp: string;
  metadata: string;
  warm_positions: string;
  warm_indices: string;
  cold_positions: string;
  cold_indices: string;
  voxel_count: number;
  component_count: number;
  positive_component_count: number;
  negative_component_count: number;
};

export type PotentialTemperatureStructureManifest = {
  version: number;
  dataset: string;
  variable: string;
  units: string;
  derived_variable: {
    name: "dry_potential_temperature";
    units: string;
    reference_pressure_hpa: number;
    kappa: number;
  };
  structure_kind: "potential-temperature-latitude-mean-anomaly-shell";
  geometry_mode: "voxel-faces";
  selection: {
    background: "per-level_latitude-band_mean";
    threshold_basis: "per-level_absolute-anomaly_percentile";
    absolute_anomaly_percentile: number;
    smoothing_sigma_cells: number;
    keep_signs: ["negative", "positive"];
    volume_connectivity: string;
    wraps_longitude: true;
  };
  sampling: {
    latitude_stride: number;
    longitude_stride: number;
    method: string;
  };
  pressure_window_hpa: {
    min: number;
    max: number;
    level_count: number;
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
  timestamps: PotentialTemperatureManifestTimestamp[];
};

export type PotentialTemperatureStructureMetadata = {
  version: number;
  timestamp: string;
  component_count: number;
  positive_component_count: number;
  negative_component_count: number;
  largest_component_voxel_count: number;
  largest_positive_component_voxel_count: number;
  largest_negative_component_voxel_count: number;
  voxel_count: number;
  positive_voxel_count: number;
  negative_voxel_count: number;
  selected_voxel_count_before_smoothing: number;
  vertex_count: number;
  index_count: number;
  warm_vertex_count: number;
  warm_index_count: number;
  cold_vertex_count: number;
  cold_index_count: number;
  theta_min: number;
  theta_max: number;
  theta_mean: number;
  anomaly_min: number;
  anomaly_max: number;
  anomaly_mean: number;
  anomaly_abs_max: number;
  pressure_min_hpa: number;
  pressure_max_hpa: number;
  latitude_min_deg: number;
  latitude_max_deg: number;
  longitude_min_deg: number;
  longitude_max_deg: number;
  absolute_anomaly_percentile: number;
  smoothing_sigma_cells: number;
  selection: {
    kept_signs: ["negative", "positive"];
    thresholds_by_pressure_level: Array<{
      pressure_hpa: number;
      absolute_anomaly_threshold: number;
    }>;
  };
  warm_positions_file: string;
  warm_indices_file: string;
  cold_positions_file: string;
  cold_indices_file: string;
};

export type PotentialTemperatureStructureFrame = {
  manifest: PotentialTemperatureStructureManifest;
  entry: PotentialTemperatureManifestTimestamp;
  metadata: PotentialTemperatureStructureMetadata;
  warmPositions: Float32Array;
  warmIndices: Uint32Array;
  coldPositions: Float32Array;
  coldIndices: Uint32Array;
};

let potentialTemperatureManifestPromise: Promise<PotentialTemperatureStructureManifest> | null =
  null;

function buildPotentialTemperatureStructureUrl(...segments: string[]) {
  return `/potential-temperature-structures/${segments.join("/")}`;
}

export async function fetchPotentialTemperatureStructureManifest(opts?: {
  refresh?: boolean;
  notifyOnError?: boolean;
}) {
  const refresh = opts?.refresh ?? false;
  const notifyOnError = opts?.notifyOnError ?? true;

  if (!refresh && potentialTemperatureManifestPromise) {
    return potentialTemperatureManifestPromise;
  }

  potentialTemperatureManifestPromise =
    fetchJsonOrThrow<PotentialTemperatureStructureManifest>(
      buildPotentialTemperatureStructureUrl("index.json"),
      "Failed to load potential temperature structure manifest.",
      {
        cache: "no-store",
        layerLabel: "Potential temperature structures",
        notifyOnError,
      }
    ).catch((error) => {
      potentialTemperatureManifestPromise = null;
      throw error;
    });

  return potentialTemperatureManifestPromise;
}

export async function fetchPotentialTemperatureStructureFrame(
  datehour: string,
  opts?: { notifyOnError?: boolean }
): Promise<PotentialTemperatureStructureFrame> {
  const notifyOnError = opts?.notifyOnError ?? true;
  const manifest = await fetchPotentialTemperatureStructureManifest({
    notifyOnError,
    refresh: true,
  });
  const availableValues = manifest.timestamps.map((item) => item.timestamp);
  const resolvedTimestamp = snapTimestampToAvailable(datehour, availableValues);
  const entry =
    manifest.timestamps.find((item) => item.timestamp === resolvedTimestamp) ??
    manifest.timestamps[0];

  if (!entry) {
    throw new Error("No potential temperature structure assets are available.");
  }

  const [metadata, warmPositionsBlob, warmIndicesBlob, coldPositionsBlob, coldIndicesBlob] =
    await Promise.all([
      fetchJsonOrThrow<PotentialTemperatureStructureMetadata>(
        buildPotentialTemperatureStructureUrl(entry.metadata),
        "Failed to load potential temperature structure metadata.",
        {
          layerLabel: "Potential temperature structures",
          notifyOnError,
        }
      ),
      fetchBlobOrThrow(
        buildPotentialTemperatureStructureUrl(entry.warm_positions),
        "Failed to load potential temperature warm structure positions.",
        {
          layerLabel: "Potential temperature structures",
          notifyOnError,
        }
      ),
      fetchBlobOrThrow(
        buildPotentialTemperatureStructureUrl(entry.warm_indices),
        "Failed to load potential temperature warm structure indices.",
        {
          layerLabel: "Potential temperature structures",
          notifyOnError,
        }
      ),
      fetchBlobOrThrow(
        buildPotentialTemperatureStructureUrl(entry.cold_positions),
        "Failed to load potential temperature cold structure positions.",
        {
          layerLabel: "Potential temperature structures",
          notifyOnError,
        }
      ),
      fetchBlobOrThrow(
        buildPotentialTemperatureStructureUrl(entry.cold_indices),
        "Failed to load potential temperature cold structure indices.",
        {
          layerLabel: "Potential temperature structures",
          notifyOnError,
        }
      ),
    ]);

  const [warmPositionsBuffer, warmIndicesBuffer, coldPositionsBuffer, coldIndicesBuffer] =
    await Promise.all([
      warmPositionsBlob.arrayBuffer(),
      warmIndicesBlob.arrayBuffer(),
      coldPositionsBlob.arrayBuffer(),
      coldIndicesBlob.arrayBuffer(),
    ]);

  return {
    manifest,
    entry,
    metadata,
    warmPositions: new Float32Array(warmPositionsBuffer),
    warmIndices: new Uint32Array(warmIndicesBuffer),
    coldPositions: new Float32Array(coldPositionsBuffer),
    coldIndices: new Uint32Array(coldIndicesBuffer),
  };
}
