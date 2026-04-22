import { fetchBlobOrThrow, fetchJsonOrThrow } from "./dataFetchErrors";
import { snapTimestampToAvailable } from "./ApiResponses";
import type { AirMassClassificationVariant } from "../../state/controlsStore";

export const AIR_MASS_CLASS_ORDER = [
  "warm_dry",
  "warm_moist",
  "cold_dry",
  "cold_moist",
] as const;

export type AirMassStructureClassKey = (typeof AIR_MASS_CLASS_ORDER)[number];

export type AirMassStructureManifestTimestamp = {
  timestamp: string;
  metadata: string;
  voxel_count: number;
  component_count: number;
  class_counts: Record<
    AirMassStructureClassKey,
    {
      voxel_count: number;
      component_count: number;
    }
  >;
};

export type AirMassStructureManifest = {
  version: number;
  dataset: string;
  variant: AirMassClassificationVariant;
  variant_label: string;
  structure_kind: "air-mass-proxy-shells";
  geometry_mode: "voxel-faces";
  variables: {
    temperature: string;
    relative_humidity: string;
    specific_humidity: string;
  };
  classification: {
    thermal_axis_field: string;
    moisture_axis_field: string;
    thermal_transform: string;
    moisture_transform: string;
    score_basis: string;
    keep_top_percent: number;
    axis_min_abs_zscore: number;
    bridge_gap_levels: number;
    min_component_voxels: number;
    min_component_pressure_span_levels: number;
    classes: Array<{
      key: AirMassStructureClassKey;
      label: string;
    }>;
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
  timestamps: AirMassStructureManifestTimestamp[];
};

export type AirMassStructureMetadata = {
  version: number;
  timestamp: string;
  voxel_count: number;
  component_count: number;
  class_summaries: Record<
    AirMassStructureClassKey,
    {
      label: string;
      voxel_count: number;
      component_count: number;
      largest_component_voxel_count: number;
      positions_file: string;
      indices_file: string;
      vertex_count: number;
      index_count: number;
    }
  >;
  pressure_levels_hpa: number[];
  score_thresholds_by_pressure_level: Array<{
    pressure_hpa: number;
    score_threshold: number | null;
    kept_cell_count: number;
  }>;
  thermal_axis: {
    field: string;
    transform: string;
    scale_by_pressure_level: number[];
  };
  moisture_axis: {
    field: string;
    transform: string;
    scale_by_pressure_level: number[];
  };
  selection: {
    keep_top_percent: number;
    axis_min_abs_zscore: number;
    score_basis: string;
    bridge_gap_levels: number;
    min_component_voxels: number;
    min_component_pressure_span_levels: number;
    classes: Array<{
      key: AirMassStructureClassKey;
      label: string;
    }>;
  };
  smoothing_sigma_cells: number;
  pressure_min_hpa: number;
  pressure_max_hpa: number;
  latitude_min_deg: number;
  latitude_max_deg: number;
  longitude_min_deg: number;
  longitude_max_deg: number;
};

export type AirMassStructureFrame = {
  manifest: AirMassStructureManifest;
  entry: AirMassStructureManifestTimestamp;
  metadata: AirMassStructureMetadata;
  classBuffers: Record<
    AirMassStructureClassKey,
    {
      positions: Float32Array;
      indices: Uint32Array;
    }
  >;
};

const manifestPromiseCache = new Map<
  AirMassClassificationVariant,
  Promise<AirMassStructureManifest>
>();

function airMassStructureBaseSegments(variant: AirMassClassificationVariant) {
  return ["air-mass-structures", "variants", variant];
}

function buildAirMassStructureUrl(
  variant: AirMassClassificationVariant,
  ...segments: string[]
) {
  return `/${[...airMassStructureBaseSegments(variant), ...segments].join("/")}`;
}

export async function fetchAirMassStructureManifest(opts?: {
  variant?: AirMassClassificationVariant;
  refresh?: boolean;
  notifyOnError?: boolean;
}) {
  const variant = opts?.variant ?? "temperature-rh-latmean";
  const refresh = opts?.refresh ?? false;
  const notifyOnError = opts?.notifyOnError ?? true;

  if (!refresh) {
    const cachedPromise = manifestPromiseCache.get(variant);
    if (cachedPromise) return cachedPromise;
  }

  const manifestPromise = fetchJsonOrThrow<AirMassStructureManifest>(
    buildAirMassStructureUrl(variant, "index.json"),
    "Failed to load air-mass structure manifest.",
    {
      cache: "no-store",
      layerLabel: "Air-mass structures",
      notifyOnError,
    }
  ).catch((error) => {
    manifestPromiseCache.delete(variant);
    throw error;
  });

  manifestPromiseCache.set(variant, manifestPromise);
  return manifestPromise;
}

export async function fetchAirMassStructureFrame(
  datehour: string,
  opts?: {
    notifyOnError?: boolean;
    variant?: AirMassClassificationVariant;
  }
): Promise<AirMassStructureFrame> {
  const notifyOnError = opts?.notifyOnError ?? true;
  const variant = opts?.variant ?? "temperature-rh-latmean";
  const manifest = await fetchAirMassStructureManifest({
    variant,
    refresh: true,
    notifyOnError,
  });
  const availableValues = manifest.timestamps.map((item) => item.timestamp);
  const resolvedTimestamp = snapTimestampToAvailable(datehour, availableValues);
  const entry =
    manifest.timestamps.find((item) => item.timestamp === resolvedTimestamp) ??
    manifest.timestamps[0];

  if (!entry) {
    throw new Error("No air-mass structure assets are available.");
  }

  const metadata = await fetchJsonOrThrow<AirMassStructureMetadata>(
    buildAirMassStructureUrl(variant, entry.metadata),
    "Failed to load air-mass structure metadata.",
    {
      layerLabel: "Air-mass structures",
      notifyOnError,
    }
  );

  const classBlobs = await Promise.all(
    AIR_MASS_CLASS_ORDER.flatMap((classKey) => [
      fetchBlobOrThrow(
        buildAirMassStructureUrl(
          variant,
          metadata.class_summaries[classKey].positions_file
        ),
        `Failed to load ${classKey} air-mass structure positions.`,
        {
          layerLabel: "Air-mass structures",
          notifyOnError,
        }
      ),
      fetchBlobOrThrow(
        buildAirMassStructureUrl(
          variant,
          metadata.class_summaries[classKey].indices_file
        ),
        `Failed to load ${classKey} air-mass structure indices.`,
        {
          layerLabel: "Air-mass structures",
          notifyOnError,
        }
      ),
    ])
  );

  const classBuffers = {} as AirMassStructureFrame["classBuffers"];
  await Promise.all(
    AIR_MASS_CLASS_ORDER.map(async (classKey, index) => {
      const positionsBlob = classBlobs[index * 2];
      const indicesBlob = classBlobs[index * 2 + 1];
      const [positionsBuffer, indicesBuffer] = await Promise.all([
        positionsBlob.arrayBuffer(),
        indicesBlob.arrayBuffer(),
      ]);
      classBuffers[classKey] = {
        positions: new Float32Array(positionsBuffer),
        indices: new Uint32Array(indicesBuffer),
      };
    })
  );

  return {
    manifest,
    entry,
    metadata,
    classBuffers,
  };
}
