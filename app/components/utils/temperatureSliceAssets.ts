import { snapTimestampToAvailable } from "./ApiResponses";
import { fetchJsonOrThrow } from "./dataFetchErrors";
import type { TemperatureSliceVariant } from "../../state/controlsStore";

export type TemperatureSliceLevelEntry = {
  pressure_hpa: number;
  image: string;
  temperature_min_k: number;
  temperature_max_k: number;
  value_min?: number;
  value_max?: number;
  saturation_strength_min?: number;
  saturation_strength_max?: number;
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
  display_units: "K" | "equator-to-pole latitude match";
  variant?: string;
  variant_label?: string;
  field_kind?:
    | "raw-temperature"
    | "equivalent-potential-temperature"
    | "temperature-climatology-anomaly"
    | "potential-temperature-climatology-anomaly"
    | "raw-temperature-vertical-coherence"
    | "raw-temperature-anomaly-strength"
    | "raw-temperature-anomaly-agreement"
    | "raw-temperature-front-overlay"
    | "thermal-displacement-latitude"
    | "thermal-displacement-latitude-smoothed"
    | "thermal-displacement-zonal-mean-latitude"
    | "thermal-displacement-zonal-trimmed-mean-latitude"
    | "thermal-conflict-neighborhood";
  climatology_dataset?: string | null;
  rendering: {
    kind: "full-field-pressure-slice";
    filtering: "none";
    color_scale:
      | "global-min-blue-to-global-max-red"
      | "per-level-min-blue-to-per-level-max-red"
      | "global-symmetric-zero-white-blue-red";
    encoding:
      | "normalized-temperature-uint16-packed-rg"
      | "raw-temperature-uint16-rg-saturation-strength-b"
      | "raw-temperature-uint16-rg-signed-saturation-b"
      | "raw-temperature-uint16-rg-front-mask-b"
      | "thermal-conflict-warmness-uint16-rg-conflict-b";
  };
  temperature_range_k: {
    min: number;
    max: number;
  };
  value_range?: {
    min: number;
    max: number;
  };
  saturation_strength_range?: {
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

const temperatureSliceManifestPromiseCache = new Map<
  TemperatureSliceVariant,
  Promise<TemperatureSliceManifest>
>();

const DEFAULT_TEMPERATURE_SLICE_VARIANT: TemperatureSliceVariant =
  "raw-temperature";

function temperatureSliceBaseSegments(variant: TemperatureSliceVariant) {
  return ["temperature-slices", "variants", variant];
}

function buildTemperatureSliceUrl(
  variant: TemperatureSliceVariant,
  ...segments: string[]
) {
  return `/${[...temperatureSliceBaseSegments(variant), ...segments].join("/")}`;
}

export async function fetchTemperatureSliceManifest(opts?: {
  variant?: TemperatureSliceVariant;
  refresh?: boolean;
  notifyOnError?: boolean;
}) {
  const variant = opts?.variant ?? DEFAULT_TEMPERATURE_SLICE_VARIANT;
  const refresh = opts?.refresh ?? false;
  const notifyOnError = opts?.notifyOnError ?? true;

  if (!refresh) {
    const cachedPromise = temperatureSliceManifestPromiseCache.get(variant);
    if (cachedPromise) {
      return cachedPromise;
    }
  }

  const manifestPromise = fetchJsonOrThrow<TemperatureSliceManifest>(
    buildTemperatureSliceUrl(variant, "index.json"),
    "Failed to load temperature slice manifest.",
    {
      layerLabel: "Temperature slice",
      notifyOnError,
    }
  ).catch((error) => {
    temperatureSliceManifestPromiseCache.delete(variant);
    throw error;
  });

  temperatureSliceManifestPromiseCache.set(variant, manifestPromise);
  return manifestPromise;
}

export async function fetchTemperatureSliceFrame(
  datehour: string,
  pressureHpa: number,
  opts?: {
    notifyOnError?: boolean;
    variant?: TemperatureSliceVariant;
  }
): Promise<TemperatureSliceFrame> {
  const notifyOnError = opts?.notifyOnError ?? true;
  const variant = opts?.variant ?? DEFAULT_TEMPERATURE_SLICE_VARIANT;
  const manifest = await fetchTemperatureSliceManifest({
    notifyOnError,
    variant,
  });
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

export function temperatureSliceImageUrl(
  variant: TemperatureSliceVariant,
  entry: TemperatureSliceLevelEntry
) {
  return buildTemperatureSliceUrl(variant, entry.image);
}

export function temperatureSliceBorderTextureUrl(
  variant: TemperatureSliceVariant,
  manifest: TemperatureSliceManifest
) {
  return manifest.border_texture
    ? buildTemperatureSliceUrl(variant, manifest.border_texture)
    : null;
}
