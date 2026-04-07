import {
  type AvailableRange,
  DataFetchError,
  fetchBlobOrThrow,
  fetchJsonOrThrow,
  notifyDataFetchError,
} from "./dataFetchErrors";

const DATA_SOURCE_KIND = process.env.NEXT_PUBLIC_DATA_SOURCE_KIND;
const PUBLIC_DATA_BASE_PATH = process.env.NEXT_PUBLIC_DATA_BASE_PATH?.trim() || "";

function usesPublicDataAssets() {
  return DATA_SOURCE_KIND === "public";
}

function buildPublicDataUrl(...segments: string[]) {
  const prefix = PUBLIC_DATA_BASE_PATH.replace(/^\/+|\/+$/g, "");
  return `/${[prefix, ...segments].filter(Boolean).join("/")}`;
}

export type MoistureSegmentationMode =
  | "p95-close"
  | "simple-voxel-shell"
  | "p95-close-voxel-shell"
  | "p95-smooth-open1-voxel-shell"
  | "p95-close-smoothmesh"
  | "p95-smooth-open1"
  | "p95-local-anomaly"
  | "p95-open"
  | "p97-close"
  | "p95-close-open1"
  | "buckets"
  | "buckets-global";

function moistureStructuresBaseSegments(
  segmentationMode: MoistureSegmentationMode
) {
  if (segmentationMode === "p95-close") {
    return ["moisture-structures"];
  }

  return ["moisture-structures", "variants", segmentationMode];
}

function buildMoistureStructuresUrl(
  segmentationMode: MoistureSegmentationMode,
  ...segments: string[]
) {
  return buildPublicDataUrl(
    ...moistureStructuresBaseSegments(segmentationMode),
    ...segments
  );
}

function parseDatehour(datehour: string): Date {
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/.exec(datehour);
  if (!match) throw new Error("Invalid datehour");

  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const hour = Number(match[4]);
  const minute = Number(match[5]);

  if (
    month < 1 ||
    month > 12 ||
    day < 1 ||
    day > 31 ||
    hour < 0 ||
    hour > 23 ||
    minute < 0 ||
    minute > 59
  ) {
    throw new Error("Invalid datehour");
  }

  const dt = new Date(Date.UTC(year, month - 1, day, hour, minute, 0, 0));
  if (
    dt.getUTCFullYear() !== year ||
    dt.getUTCMonth() !== month - 1 ||
    dt.getUTCDate() !== day ||
    dt.getUTCHours() !== hour ||
    dt.getUTCMinutes() !== minute
  ) {
    throw new Error("Invalid datehour");
  }

  return dt;
}

function snapToHour(dt: Date): Date {
  return new Date(
    Date.UTC(
      dt.getUTCFullYear(),
      dt.getUTCMonth(),
      dt.getUTCDate(),
      dt.getUTCHours(),
      0,
      0,
      0
    )
  );
}

function toHourlyPngFilename(datehour: string) {
  const dtHourly = snapToHour(parseDatehour(datehour));
  const y = dtHourly.getUTCFullYear();
  const mo = String(dtHourly.getUTCMonth() + 1).padStart(2, "0");
  const d = String(dtHourly.getUTCDate()).padStart(2, "0");
  const h = String(dtHourly.getUTCHours()).padStart(2, "0");
  return `${y}-${mo}-${d}T${h}-00-00.png`;
}

function toHourlyJsonFilename(datehour: string) {
  const dtHourly = snapToHour(parseDatehour(datehour));
  const y = dtHourly.getUTCFullYear();
  const mo = String(dtHourly.getUTCMonth() + 1).padStart(2, "0");
  const d = String(dtHourly.getUTCDate()).padStart(2, "0");
  const h = String(dtHourly.getUTCHours()).padStart(2, "0");
  return `${y}-${mo}-${d}T${h}-00-00.json`;
}

export type ExampleContoursPressure = "msl" | "250" | "500" | "925";

export function exampleShaderMeshLayerApiUrl(
  datehour: string,
  pressureLevel: number
) {
  if (usesPublicDataAssets()) {
    return buildPublicDataUrl(
      "divergence-rg",
      String(pressureLevel),
      toHourlyPngFilename(datehour)
    );
  }

  return `/api/divergence/${encodeURIComponent(String(pressureLevel))}/${encodeURIComponent(datehour)}`;
}

export function exampleParticleLayerApiUrl(
  datehour: string,
  pressureLevel: number
) {
  if (usesPublicDataAssets()) {
    return buildPublicDataUrl(
      "wind-uv-rg",
      String(pressureLevel),
      toHourlyPngFilename(datehour)
    );
  }

  return `/api/wind_uv/${encodeURIComponent(String(pressureLevel))}/${encodeURIComponent(datehour)}`;
}

export function exampleContoursApiUrl(
  datehour: string,
  pressure: ExampleContoursPressure
) {
  if (usesPublicDataAssets()) {
    return buildPublicDataUrl(
      "gph_contours",
      String(pressure),
      toHourlyJsonFilename(datehour)
    );
  }

  return `/api/msl_contours/${encodeURIComponent(String(pressure))}/${encodeURIComponent(datehour)}`;
}

export type LonLat = [number, number];
export type ContourLine = LonLat[];
export type ContourLevels = Record<string, ContourLine[]>;

export type ExampleContoursFile = {
  timestamp: string;
  contour_step_hpa: number;
  levels: ContourLevels;
};

export type MoistureThresholdEntry = {
  pressure_hpa: number;
  threshold: number;
};

export type MoistureStructureManifestTimestamp = {
  timestamp: string;
  metadata: string;
  positions: string;
  indices: string;
  footprints?: string;
  component_count: number;
  vertex_count: number;
  index_count: number;
};

export type MoistureStructureManifest = {
  version: number;
  dataset: string;
  variable: string;
  units: string;
  geometry_mode?: string;
  segmentation_mode?: MoistureSegmentationMode;
  threshold_mode: {
    kind: string;
    quantile: number;
    minimum_component_size: number;
    smoothing: {
      binary_closing_radius_cells: number;
      binary_opening_radius_cells?: number;
      gaussian_sigma: number;
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
  thresholds: MoistureThresholdEntry[];
  timestamps: MoistureStructureManifestTimestamp[];
};

export type MoistureStructureComponentMetadata = {
  id: number;
  bucket_index?: number;
  vertex_offset: number;
  vertex_count: number;
  index_offset: number;
  index_count: number;
  voxel_count: number;
  mean_specific_humidity: number;
  max_specific_humidity: number;
  pressure_min_hpa: number;
  pressure_max_hpa: number;
  latitude_min_deg: number;
  latitude_max_deg: number;
  longitude_min_deg: number;
  longitude_max_deg: number;
  wraps_longitude_seam: boolean;
};

export type MoistureStructureMetadata = {
  version: number;
  timestamp: string;
  component_count: number;
  vertex_count: number;
  index_count: number;
  thresholded_voxel_count: number;
  components: MoistureStructureComponentMetadata[];
  positions_file: string;
  indices_file: string;
};

export type MoistureComponentFootprint = {
  id: number;
  rings: Array<Array<[number, number]>>;
  occupied_cell_count: number;
  latitude_min_deg: number;
  latitude_max_deg: number;
  longitude_min_deg: number;
  longitude_max_deg: number;
};

export type MoistureFootprintFile = {
  version: number;
  timestamp: string;
  components: MoistureComponentFootprint[];
};

export type MoistureStructureFrame = {
  manifest: MoistureStructureManifest;
  entry: MoistureStructureManifestTimestamp;
  metadata: MoistureStructureMetadata;
  positions: Float32Array;
  indices: Uint32Array;
  footprints: MoistureComponentFootprint[];
};

const moistureManifestPromises = new Map<
  MoistureSegmentationMode,
  Promise<MoistureStructureManifest>
>();

export async function fetchExampleContours(
  datehour: string,
  pressure: ExampleContoursPressure
): Promise<ExampleContoursFile> {
  const layerLabel =
    pressure === "msl"
      ? "Example contours (MSL)"
      : `Example contours (${pressure} hPa)`;

  return fetchJsonOrThrow<ExampleContoursFile>(
    exampleContoursApiUrl(datehour, pressure),
    "Failed to load example contour data.",
    { layerLabel }
  );
}

function availableRangeFromManifest(
  manifest: MoistureStructureManifest
): AvailableRange | undefined {
  const first = manifest.timestamps[0]?.timestamp;
  const last = manifest.timestamps[manifest.timestamps.length - 1]?.timestamp;
  if (!first || !last) return undefined;
  return { start: first, end: last };
}

function moistureTimestampNotFoundError(
  datehour: string,
  manifest: MoistureStructureManifest
) {
  return new DataFetchError({
    message: `No moisture structure data exists for ${datehour}.`,
    status: 404,
    availableRange: availableRangeFromManifest(manifest),
  });
}

export function snapTimestampToAvailable(
  datehour: string,
  availableValues: string[]
) {
  if (availableValues.length === 0) return datehour;
  if (availableValues.includes(datehour)) return datehour;

  let targetMs = Number.NaN;
  try {
    targetMs = parseDatehour(datehour).getTime();
  } catch {
    return availableValues[0];
  }

  let bestValue = availableValues[0];
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const candidate of availableValues) {
    try {
      const distance = Math.abs(parseDatehour(candidate).getTime() - targetMs);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestValue = candidate;
      }
    } catch {
      continue;
    }
  }

  return bestValue;
}

export async function fetchMoistureStructureManifest(opts?: {
  segmentationMode?: MoistureSegmentationMode;
  refresh?: boolean;
  notifyOnError?: boolean;
}) {
  const segmentationMode = opts?.segmentationMode ?? "p95-close";
  const refresh = opts?.refresh ?? false;
  const notifyOnError = opts?.notifyOnError ?? true;

  if (!refresh) {
    const existingPromise = moistureManifestPromises.get(segmentationMode);
    if (existingPromise) {
      return existingPromise;
    }
  }

  const promise = fetchJsonOrThrow<MoistureStructureManifest>(
    buildMoistureStructuresUrl(segmentationMode, "index.json"),
    "Failed to load moisture structure manifest.",
    {
      layerLabel: "Moisture structures",
      notifyOnError,
    }
  );

  moistureManifestPromises.set(
    segmentationMode,
    promise.catch((error) => {
      moistureManifestPromises.delete(segmentationMode);
      throw error;
    })
  );

  return moistureManifestPromises.get(segmentationMode)!;
}

export async function fetchMoistureStructureFrame(
  datehour: string,
  opts?: {
    segmentationMode?: MoistureSegmentationMode;
    notifyOnError?: boolean;
  }
): Promise<MoistureStructureFrame> {
  const segmentationMode = opts?.segmentationMode ?? "p95-close";
  const notifyOnError = opts?.notifyOnError ?? true;
  const manifest = await fetchMoistureStructureManifest({
    segmentationMode,
    notifyOnError,
  });
  let entry = manifest.timestamps.find((item) => item.timestamp === datehour);

  if (!entry && segmentationMode === "simple-voxel-shell") {
    const availableValues = manifest.timestamps.map((item) => item.timestamp);
    const fallbackTimestamp = snapTimestampToAvailable(datehour, availableValues);
    entry =
      manifest.timestamps.find((item) => item.timestamp === fallbackTimestamp) ??
      manifest.timestamps[0];
  }

  if (!entry) {
    const error = moistureTimestampNotFoundError(datehour, manifest);
    if (notifyOnError) {
      notifyDataFetchError(error, "Failed to load moisture structures.", {
        layerLabel: "Moisture structures",
      });
    }
    throw error;
  }

  const [metadata, positionsBlob, indicesBlob, footprintFile] = await Promise.all([
    fetchJsonOrThrow<MoistureStructureMetadata>(
      buildMoistureStructuresUrl(segmentationMode, entry.metadata),
      "Failed to load moisture structure metadata.",
      {
        layerLabel: "Moisture structures",
        notifyOnError,
      }
    ),
    fetchBlobOrThrow(
      buildMoistureStructuresUrl(segmentationMode, entry.positions),
      "Failed to load moisture structure positions.",
      {
        layerLabel: "Moisture structures",
        notifyOnError,
      }
    ),
    fetchBlobOrThrow(
      buildMoistureStructuresUrl(segmentationMode, entry.indices),
      "Failed to load moisture structure indices.",
      {
        layerLabel: "Moisture structures",
        notifyOnError,
      }
    ),
    entry.footprints
      ? fetchJsonOrThrow<MoistureFootprintFile>(
          buildMoistureStructuresUrl(segmentationMode, entry.footprints),
          "Failed to load moisture footprint data.",
          {
            layerLabel: "Moisture structures",
            notifyOnError,
          }
        )
      : Promise.resolve(null),
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
    footprints: footprintFile?.components ?? [],
  };
}
