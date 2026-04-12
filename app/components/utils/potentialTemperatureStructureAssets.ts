import { fetchBlobOrThrow, fetchJsonOrThrow } from "./dataFetchErrors";
import { snapTimestampToAvailable } from "./ApiResponses";

export type PotentialTemperatureManifestTimestamp = {
  timestamp: string;
  metadata: string;
  threshold_value: number;
  hot_voxel_count: number;
  cool_voxel_count: number;
  hot_component_count: number;
  cool_component_count: number;
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
  structure_kind: "potential-temperature-threshold-shells";
  geometry_mode: "voxel-faces";
  threshold: {
    kind: "global_top_percent";
    top_percent: number;
    quantile: number;
  };
  selection: {
    connectivity: "26-connected";
    wraps_longitude: true;
    side_rule: "components_touching_opposite_threshold_side";
    hot_interface_faces_visible: boolean;
    cool_interface_faces_visible: boolean;
  };
  sampling: {
    latitude_stride: number;
    longitude_stride: number;
    method: string;
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

export type PotentialTemperatureThresholdSideMetadata = {
  voxel_count: number;
  component_count: number;
  touching_component_count: number;
  vertex_count: number;
  index_count: number;
  theta_min: number | null;
  theta_max: number | null;
  theta_mean: number | null;
  pressure_min_hpa: number | null;
  pressure_max_hpa: number | null;
  positions_file: string;
  indices_file: string;
};

export type PotentialTemperatureStructureMetadata = {
  version: number;
  timestamp: string;
  top_percent: number;
  threshold_value: number;
  finite_voxel_count: number;
  hot_side: PotentialTemperatureThresholdSideMetadata;
  cool_side: PotentialTemperatureThresholdSideMetadata;
};

export type PotentialTemperatureSideMesh = {
  positions: Float32Array;
  indices: Uint32Array;
};

export type PotentialTemperatureStructureFrame = {
  manifest: PotentialTemperatureStructureManifest;
  entry: PotentialTemperatureManifestTimestamp;
  metadata: PotentialTemperatureStructureMetadata;
  hotSide: PotentialTemperatureSideMesh;
  coolSide: PotentialTemperatureSideMesh;
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

  const metadata = await fetchJsonOrThrow<PotentialTemperatureStructureMetadata>(
    buildPotentialTemperatureStructureUrl(entry.metadata),
    "Failed to load potential temperature structure metadata.",
    {
      layerLabel: "Potential temperature structures",
      notifyOnError,
    }
  );

  const [hotPositionsBlob, hotIndicesBlob, coolPositionsBlob, coolIndicesBlob] =
    await Promise.all([
      fetchBlobOrThrow(
        buildPotentialTemperatureStructureUrl(metadata.hot_side.positions_file),
        "Failed to load hot-side potential temperature positions.",
        {
          layerLabel: "Potential temperature structures",
          notifyOnError,
        }
      ),
      fetchBlobOrThrow(
        buildPotentialTemperatureStructureUrl(metadata.hot_side.indices_file),
        "Failed to load hot-side potential temperature indices.",
        {
          layerLabel: "Potential temperature structures",
          notifyOnError,
        }
      ),
      fetchBlobOrThrow(
        buildPotentialTemperatureStructureUrl(metadata.cool_side.positions_file),
        "Failed to load cool-side potential temperature positions.",
        {
          layerLabel: "Potential temperature structures",
          notifyOnError,
        }
      ),
      fetchBlobOrThrow(
        buildPotentialTemperatureStructureUrl(metadata.cool_side.indices_file),
        "Failed to load cool-side potential temperature indices.",
        {
          layerLabel: "Potential temperature structures",
          notifyOnError,
        }
      ),
    ]);

  const [hotPositionsBuffer, hotIndicesBuffer, coolPositionsBuffer, coolIndicesBuffer] =
    await Promise.all([
      hotPositionsBlob.arrayBuffer(),
      hotIndicesBlob.arrayBuffer(),
      coolPositionsBlob.arrayBuffer(),
      coolIndicesBlob.arrayBuffer(),
    ]);

  return {
    manifest,
    entry,
    metadata,
    hotSide: {
      positions: new Float32Array(hotPositionsBuffer),
      indices: new Uint32Array(hotIndicesBuffer),
    },
    coolSide: {
      positions: new Float32Array(coolPositionsBuffer),
      indices: new Uint32Array(coolIndicesBuffer),
    },
  };
}
