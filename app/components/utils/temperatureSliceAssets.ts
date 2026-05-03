import { snapTimestampToAvailable } from "./ApiResponses";
import { fetchJsonOrThrow } from "./dataFetchErrors";

export type TemperatureSliceLevelEntry = {
  pressure_hpa: number;
  image: string;
  temperature_min_k: number;
  temperature_max_k: number;
};

export type TemperatureSliceManifestTimestamp = {
  timestamp: string;
  levels: TemperatureSliceLevelEntry[];
};

export type TemperatureSliceManifest = {
  version: number;
  dataset: string;
  variable: string;
  units: string;
  display_units: "K";
  rendering: {
    kind: "full-field-pressure-slice";
    filtering: "none";
    color_scale: "global-min-blue-to-global-max-red";
    encoding: "normalized-temperature-uint16-packed-rg";
  };
  temperature_range_k: {
    min: number;
    max: number;
  };
  pressure_window_hpa: {
    min: number;
    max: number;
  };
  pressure_levels_hpa: number[];
  grid: {
    latitude_count: number;
    longitude_count: number;
    latitude_step_degrees: number | null;
    longitude_step_degrees: number | null;
    latitude_min_degrees: number;
    latitude_max_degrees: number;
    longitude_min_degrees: number;
    longitude_max_degrees: number;
  };
  border_texture?: string;
  timestamps: TemperatureSliceManifestTimestamp[];
};

export type TemperatureSlicePressurePair = {
  lower: TemperatureSliceLevelEntry;
  upper: TemperatureSliceLevelEntry;
  mix: number;
};

export type TemperatureSliceFrame = {
  manifest: TemperatureSliceManifest;
  entry: TemperatureSliceManifestTimestamp;
  pressurePair: TemperatureSlicePressurePair;
};

const temperatureSliceManifestPromise: {
  current: Promise<TemperatureSliceManifest> | null;
} = {
  current: null,
};

function buildTemperatureSliceUrl(...segments: string[]) {
  return `/${["temperature-slices", ...segments].join("/")}`;
}

export async function fetchTemperatureSliceManifest(opts?: {
  refresh?: boolean;
  notifyOnError?: boolean;
}) {
  const refresh = opts?.refresh ?? false;
  const notifyOnError = opts?.notifyOnError ?? true;

  if (!refresh && temperatureSliceManifestPromise.current) {
    return temperatureSliceManifestPromise.current;
  }

  temperatureSliceManifestPromise.current =
    fetchJsonOrThrow<TemperatureSliceManifest>(
      buildTemperatureSliceUrl("index.json"),
      "Failed to load temperature slice manifest.",
      {
        layerLabel: "Temperature slice",
        notifyOnError,
      }
    ).catch((error) => {
      temperatureSliceManifestPromise.current = null;
      throw error;
    });

  return temperatureSliceManifestPromise.current;
}

export async function fetchTemperatureSliceFrame(
  datehour: string,
  pressureHpa: number,
  opts?: { notifyOnError?: boolean }
): Promise<TemperatureSliceFrame> {
  const notifyOnError = opts?.notifyOnError ?? true;
  const manifest = await fetchTemperatureSliceManifest({ notifyOnError });
  const availableValues = manifest.timestamps.map((item) => item.timestamp);
  const resolvedTimestamp = snapTimestampToAvailable(datehour, availableValues);
  const entry =
    manifest.timestamps.find((item) => item.timestamp === resolvedTimestamp) ??
    manifest.timestamps[0];

  if (!entry) {
    throw new Error("No temperature slice assets are available.");
  }

  const sortedLevels = [...entry.levels].sort(
    (a, b) => a.pressure_hpa - b.pressure_hpa
  );
  if (sortedLevels.length === 0) {
    throw new Error("No temperature slice pressure levels are available.");
  }

  const clampedPressure = Math.min(
    Math.max(pressureHpa, sortedLevels[0].pressure_hpa),
    sortedLevels[sortedLevels.length - 1].pressure_hpa
  );

  let lower = sortedLevels[0];
  let upper = sortedLevels[sortedLevels.length - 1];

  for (let index = 0; index < sortedLevels.length - 1; index += 1) {
    const candidateLower = sortedLevels[index];
    const candidateUpper = sortedLevels[index + 1];
    if (
      candidateLower.pressure_hpa <= clampedPressure &&
      clampedPressure <= candidateUpper.pressure_hpa
    ) {
      lower = candidateLower;
      upper = candidateUpper;
      break;
    }
  }

  const span = Math.max(upper.pressure_hpa - lower.pressure_hpa, 1e-6);
  const mix = Math.min(
    Math.max((clampedPressure - lower.pressure_hpa) / span, 0),
    1
  );

  return {
    manifest,
    entry,
    pressurePair: { lower, upper, mix },
  };
}

export function temperatureSliceImageUrl(entry: TemperatureSliceLevelEntry) {
  return buildTemperatureSliceUrl(entry.image);
}

export function temperatureSliceBorderTextureUrl(
  manifest: TemperatureSliceManifest
) {
  return manifest.border_texture
    ? buildTemperatureSliceUrl(manifest.border_texture)
    : null;
}
