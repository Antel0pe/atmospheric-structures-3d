import { fetchBlobOrThrow, fetchJsonOrThrow } from "./dataFetchErrors";
import { snapTimestampToAvailable } from "./ApiResponses";

export type PotentialTemperatureManifestTimestamp = {
  timestamp: string;
  metadata: string;
  positions: string;
  indices: string;
  coldness_sigma: string;
  voxel_count: number;
  component_count: number;
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
  structure_kind: "potential-temperature-cold-zonal-anomaly-shell";
  geometry_mode: "voxel-faces";
  selection: {
    background: "per-level_zonal_mean";
    standardization: "per-level_stddev";
    keep_side: "cold_only";
    z_threshold_sigma: number;
    minimum_level_component_size: number;
    level_component_connectivity: string;
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
  largest_component_voxel_count: number;
  thresholded_voxel_count: number;
  vertex_count: number;
  index_count: number;
  theta_min: number;
  theta_max: number;
  theta_mean: number;
  coldness_sigma_min: number;
  coldness_sigma_max: number;
  coldness_sigma_mean: number;
  pressure_min_hpa: number;
  pressure_max_hpa: number;
  latitude_min_deg: number;
  latitude_max_deg: number;
  longitude_min_deg: number;
  longitude_max_deg: number;
  z_threshold_sigma: number;
  selection: {
    raw_voxel_count: number;
    removed_voxel_count: number;
    postprocess: {
      minimum_component_size: number;
      connectivity: string;
      wraps_longitude: boolean;
      component_count_before_filter: number;
      component_count_after_filter: number;
      removed_component_count: number;
      removed_voxel_count: number;
    };
  };
  positions_file: string;
  indices_file: string;
  coldness_sigma_file: string;
};

export type PotentialTemperatureStructureFrame = {
  manifest: PotentialTemperatureStructureManifest;
  entry: PotentialTemperatureManifestTimestamp;
  metadata: PotentialTemperatureStructureMetadata;
  positions: Float32Array;
  indices: Uint32Array;
  coldnessSigma: Float32Array;
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

  const [metadata, positionsBlob, indicesBlob, coldnessSigmaBlob] = await Promise.all([
    fetchJsonOrThrow<PotentialTemperatureStructureMetadata>(
      buildPotentialTemperatureStructureUrl(entry.metadata),
      "Failed to load potential temperature structure metadata.",
      {
        layerLabel: "Potential temperature structures",
        notifyOnError,
      }
    ),
    fetchBlobOrThrow(
      buildPotentialTemperatureStructureUrl(entry.positions),
      "Failed to load potential temperature structure positions.",
      {
        layerLabel: "Potential temperature structures",
        notifyOnError,
      }
    ),
    fetchBlobOrThrow(
      buildPotentialTemperatureStructureUrl(entry.indices),
      "Failed to load potential temperature structure indices.",
      {
        layerLabel: "Potential temperature structures",
        notifyOnError,
      }
    ),
    fetchBlobOrThrow(
      buildPotentialTemperatureStructureUrl(entry.coldness_sigma),
      "Failed to load potential temperature coldness values.",
      {
        layerLabel: "Potential temperature structures",
        notifyOnError,
      }
    ),
  ]);

  const [positionsBuffer, indicesBuffer, coldnessSigmaBuffer] = await Promise.all([
    positionsBlob.arrayBuffer(),
    indicesBlob.arrayBuffer(),
    coldnessSigmaBlob.arrayBuffer(),
  ]);

  return {
    manifest,
    entry,
    metadata,
    positions: new Float32Array(positionsBuffer),
    indices: new Uint32Array(indicesBuffer),
    coldnessSigma: new Float32Array(coldnessSigmaBuffer),
  };
}
