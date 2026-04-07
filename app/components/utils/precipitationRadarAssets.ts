import {
  fetchJsonOrThrow,
} from "./dataFetchErrors";
import { snapTimestampToAvailable } from "./ApiResponses";

export type PrecipitationRadarManifestTimestamp = {
  timestamp: string;
  image: string;
};

export type PrecipitationRadarManifest = {
  version: number;
  dataset: string;
  variable: string;
  units: string;
  display_units: "mm";
  encoding: {
    curve: "sqrt";
    max_mm: number;
  };
  thresholds_mm: number[];
  grid: {
    latitude_count: number;
    longitude_count: number;
    latitude_step_degrees: number | null;
    longitude_step_degrees: number | null;
  };
  timestamps: PrecipitationRadarManifestTimestamp[];
};

export type PrecipitationRadarFrame = {
  manifest: PrecipitationRadarManifest;
  entry: PrecipitationRadarManifestTimestamp;
};

const precipitationManifestPromise: {
  current: Promise<PrecipitationRadarManifest> | null;
} = {
  current: null,
};

function buildPrecipitationRadarUrl(...segments: string[]) {
  return `/${["precipitation-radar", ...segments].join("/")}`;
}

export async function fetchPrecipitationRadarManifest(opts?: {
  refresh?: boolean;
  notifyOnError?: boolean;
}) {
  const refresh = opts?.refresh ?? false;
  const notifyOnError = opts?.notifyOnError ?? true;

  if (!refresh && precipitationManifestPromise.current) {
    return precipitationManifestPromise.current;
  }

  precipitationManifestPromise.current = fetchJsonOrThrow<PrecipitationRadarManifest>(
    buildPrecipitationRadarUrl("index.json"),
    "Failed to load precipitation radar manifest.",
    {
      layerLabel: "Precipitation radar",
      notifyOnError,
    }
  ).catch((error) => {
    precipitationManifestPromise.current = null;
    throw error;
  });

  return precipitationManifestPromise.current;
}

export async function fetchPrecipitationRadarFrame(
  datehour: string,
  opts?: { notifyOnError?: boolean }
): Promise<PrecipitationRadarFrame> {
  const notifyOnError = opts?.notifyOnError ?? true;
  const manifest = await fetchPrecipitationRadarManifest({ notifyOnError });
  const availableValues = manifest.timestamps.map((item) => item.timestamp);
  const resolvedTimestamp = snapTimestampToAvailable(datehour, availableValues);
  const entry =
    manifest.timestamps.find((item) => item.timestamp === resolvedTimestamp) ??
    manifest.timestamps[0];

  if (!entry) {
    throw new Error("No precipitation radar assets are available.");
  }

  return { manifest, entry };
}

export function precipitationRadarImageUrl(entry: PrecipitationRadarManifestTimestamp) {
  return buildPrecipitationRadarUrl(entry.image);
}
