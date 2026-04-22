import { fetchBlobOrThrow, fetchJsonOrThrow } from "./dataFetchErrors";
import { snapTimestampToAvailable } from "./ApiResponses";
import type { PotentialTemperatureVariant } from "../../state/controlsStore";

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
  climatology_dataset?: string;
  variable: string;
  units: string;
  variant?: string;
  derived_variable:
    | {
        name: "dry_potential_temperature";
        units: string;
        reference_pressure_hpa: number;
        kappa: number;
      }
    | {
        name: "temperature";
        units: string;
      };
  structure_kind:
    | "potential-temperature-latitude-mean-anomaly-shell"
    | "potential-temperature-climatology-anomaly-shell"
    | "raw-temperature-midpoint-cold-side-shell";
  geometry_mode: "voxel-faces";
  selection: {
    background:
      | "per-level_latitude-band_mean"
      | "matched_gridpoint_climatological_theta_mean"
      | "per-level_smoothed_raw_temperature_midpoint";
    threshold_basis:
      | "per-level_sign-tail_top-percent"
      | "per-level_absolute-anomaly_top-percent"
      | "per-level_absolute-anomaly_top-percent_then_top-component-share"
      | "per-level_absolute-anomaly_percentile"
      | "per-level_smoothed-temperature_midpoint";
    keep_top_percent?: number;
    component_keep_top_percent?: number;
    absolute_anomaly_percentile?: number;
    smoothing_sigma_cells: number;
    vertical_connection_mode?: string;
    vertical_connection_label?: string;
    core_component_connectivity?: string;
    keep_signs: Array<"negative" | "positive">;
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
  selected_voxel_count_before_connection_fill?: number;
  connection_fill_voxel_count?: number;
  selected_voxel_count_before_smoothing: number;
  vertex_count: number;
  index_count: number;
  warm_vertex_count: number;
  warm_index_count: number;
  cold_vertex_count: number;
  cold_index_count: number;
  field_name: string;
  field_units: string;
  field_min: number;
  field_max: number;
  field_mean: number;
  theta_min?: number;
  theta_max?: number;
  theta_mean?: number;
  anomaly_name: string;
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
  keep_top_percent?: number;
  component_keep_top_percent?: number;
  absolute_anomaly_percentile?: number;
  smoothing_sigma_cells: number;
  connection_mode?: string;
  selected_voxel_count_before_component_filter?: number;
  core_voxel_count?: number;
  selection: {
    kept_signs: Array<"negative" | "positive">;
    keep_top_percent?: number;
    component_keep_top_percent?: number;
    absolute_anomaly_percentile?: number;
    vertical_connection_mode?: string;
    vertical_connection_label?: string;
    core_component_connectivity?: string;
    threshold_rule?: string;
    kept_side?: string;
    polar_cap_cleanup_rule?: string;
    thresholds_by_pressure_level: Array<{
      pressure_hpa: number;
      absolute_anomaly_threshold?: number;
      hot_anomaly_threshold?: number;
      cold_anomaly_threshold?: number;
      temperature_min?: number;
      temperature_max?: number;
      midpoint_temperature?: number;
      north_polar_edge_row_cleanup_applied?: boolean;
      south_polar_edge_row_cleanup_applied?: boolean;
      selected_cell_count?: number;
      component_count?: number;
      kept_component_count?: number;
      kept_component_size_threshold?: number;
      largest_component_size?: number;
      largest_kept_component_size?: number;
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
const potentialTemperatureManifestPromiseCache = new Map<
  PotentialTemperatureVariant,
  Promise<PotentialTemperatureStructureManifest>
>();

function potentialTemperatureStructureBaseSegments(
  variant: PotentialTemperatureVariant
) {
  return ["potential-temperature-structures", "variants", variant];
}

function buildPotentialTemperatureStructureUrl(
  variant: PotentialTemperatureVariant,
  ...segments: string[]
) {
  return `/${[...potentialTemperatureStructureBaseSegments(variant), ...segments].join("/")}`;
}

export async function fetchPotentialTemperatureStructureManifest(opts?: {
  variant?: PotentialTemperatureVariant;
  refresh?: boolean;
  notifyOnError?: boolean;
}) {
  const variant = opts?.variant ?? "bridge-gap-1";
  const refresh = opts?.refresh ?? false;
  const notifyOnError = opts?.notifyOnError ?? true;

  if (!refresh) {
    const cachedPromise = potentialTemperatureManifestPromiseCache.get(variant);
    if (cachedPromise) {
      return cachedPromise;
    }
  }

  const manifestPromise = fetchJsonOrThrow<PotentialTemperatureStructureManifest>(
    buildPotentialTemperatureStructureUrl(variant, "index.json"),
    "Failed to load potential temperature structure manifest.",
    {
      cache: "no-store",
      layerLabel: "Potential temperature structures",
      notifyOnError,
    }
  ).catch((error) => {
    potentialTemperatureManifestPromiseCache.delete(variant);
    if (potentialTemperatureManifestPromise === manifestPromise) {
      potentialTemperatureManifestPromise = null;
    }
    throw error;
  });

  potentialTemperatureManifestPromise = manifestPromise;
  potentialTemperatureManifestPromiseCache.set(variant, manifestPromise);
  return manifestPromise;
}

export async function fetchPotentialTemperatureStructureFrame(
  datehour: string,
  opts?: {
    notifyOnError?: boolean;
    variant?: PotentialTemperatureVariant;
  }
): Promise<PotentialTemperatureStructureFrame> {
  const notifyOnError = opts?.notifyOnError ?? true;
  const variant = opts?.variant ?? "bridge-gap-1";
  const manifest = await fetchPotentialTemperatureStructureManifest({
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
    throw new Error("No potential temperature structure assets are available.");
  }

  const [metadata, warmPositionsBlob, warmIndicesBlob, coldPositionsBlob, coldIndicesBlob] =
    await Promise.all([
      fetchJsonOrThrow<PotentialTemperatureStructureMetadata>(
        buildPotentialTemperatureStructureUrl(variant, entry.metadata),
        "Failed to load potential temperature structure metadata.",
        {
          layerLabel: "Potential temperature structures",
          notifyOnError,
        }
      ),
      fetchBlobOrThrow(
        buildPotentialTemperatureStructureUrl(variant, entry.warm_positions),
        "Failed to load potential temperature warm structure positions.",
        {
          layerLabel: "Potential temperature structures",
          notifyOnError,
        }
      ),
      fetchBlobOrThrow(
        buildPotentialTemperatureStructureUrl(variant, entry.warm_indices),
        "Failed to load potential temperature warm structure indices.",
        {
          layerLabel: "Potential temperature structures",
          notifyOnError,
        }
      ),
      fetchBlobOrThrow(
        buildPotentialTemperatureStructureUrl(variant, entry.cold_positions),
        "Failed to load potential temperature cold structure positions.",
        {
          layerLabel: "Potential temperature structures",
          notifyOnError,
        }
      ),
      fetchBlobOrThrow(
        buildPotentialTemperatureStructureUrl(variant, entry.cold_indices),
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
