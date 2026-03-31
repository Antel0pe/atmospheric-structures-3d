import { fetchJsonOrThrow } from "./dataFetchErrors";

const DATA_SOURCE_KIND = process.env.NEXT_PUBLIC_DATA_SOURCE_KIND;
const PUBLIC_DATA_BASE_PATH = process.env.NEXT_PUBLIC_DATA_BASE_PATH?.trim() || "";

function usesPublicDataAssets() {
  return DATA_SOURCE_KIND === "public";
}

function buildPublicDataUrl(...segments: string[]) {
  const prefix = PUBLIC_DATA_BASE_PATH.replace(/^\/+|\/+$/g, "");
  return `/${[prefix, ...segments].filter(Boolean).join("/")}`;
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
