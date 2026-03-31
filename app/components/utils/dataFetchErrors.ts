"use client";

export type AvailableRange = {
  start: string;
  end: string;
};

type ApiErrorBody = {
  error?: string;
  code?: string;
  available_range?: AvailableRange;
};

export type DataFetchNotice = {
  title: string;
  message: string;
  dedupeKey: string;
};

type DataFetchNoticeOptions = {
  layerLabel?: string;
};

type DataFetchRequestOptions = RequestInit &
  DataFetchNoticeOptions & {
    notifyOnError?: boolean;
  };

const NOTICE_EVENT = "water-transport:data-fetch-notice";
const NOTICE_DEDUPE_MS = 1800;
const recentNoticeTimes = new Map<string, number>();

export class DataFetchError extends Error {
  status?: number;
  code?: string;
  availableRange?: AvailableRange;
  apiError?: string;
  notified = false;

  constructor(args: {
    message: string;
    status?: number;
    code?: string;
    availableRange?: AvailableRange;
    apiError?: string;
  }) {
    super(args.message);
    this.name = "DataFetchError";
    this.status = args.status;
    this.code = args.code;
    this.availableRange = args.availableRange;
    this.apiError = args.apiError;
  }
}

function isAvailableRange(value: unknown): value is AvailableRange {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<AvailableRange>;
  return typeof candidate.start === "string" && typeof candidate.end === "string";
}

function normalizeApiErrorBody(body: unknown): ApiErrorBody | null {
  if (!body || typeof body !== "object") return null;
  const candidate = body as ApiErrorBody;
  return {
    error: typeof candidate.error === "string" ? candidate.error : undefined,
    code: typeof candidate.code === "string" ? candidate.code : undefined,
    available_range: isAvailableRange(candidate.available_range)
      ? candidate.available_range
      : undefined,
  };
}

function formatPopupDate(value: string) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return value;

  const day = String(date.getUTCDate()).padStart(2, "0");
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const year = String(date.getUTCFullYear()).slice(-2);
  const hours = String(date.getUTCHours()).padStart(2, "0");
  const minutes = String(date.getUTCMinutes()).padStart(2, "0");

  return `${day}/${month}/${year} ${hours}:${minutes}`;
}

function formatRange(range: AvailableRange) {
  const formattedStart = formatPopupDate(range.start);
  const formattedEnd = formatPopupDate(range.end);
  const usedFallback =
    formattedStart === range.start || formattedEnd === range.end;

  if (usedFallback) {
    return `${range.start} to ${range.end}`;
  }

  return `${formattedStart} to ${formattedEnd} UTC`;
}

function buildNoticeTitle(options?: DataFetchNoticeOptions) {
  return options?.layerLabel?.trim() || "Data loading";
}

function buildUserMessage(error: unknown, fallbackMessage: string) {
  if (error instanceof DataFetchError) {
    if (error.status === 404 && error.availableRange) {
      return `No data exists for this date. Data only exists between ${formatRange(
        error.availableRange
      )}.`;
    }

    if (error.status === 404) {
      return "No data exists for this date.";
    }

    if (error.apiError) {
      return `Data error: ${error.apiError}.`;
    }

    if (error.message) {
      return error.message;
    }
  }

  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }

  return fallbackMessage;
}

function buildDedupeKey(
  error: unknown,
  fallbackMessage: string,
  options?: DataFetchNoticeOptions
) {
  const scope = options?.layerLabel?.trim() || "data";

  if (error instanceof DataFetchError) {
    if (error.status === 404 && error.availableRange) {
      return `${scope}:404:${error.availableRange.start}:${error.availableRange.end}`;
    }

    if (error.status && error.apiError) {
      return `${scope}:${error.status}:${error.apiError}`;
    }

    if (error.message) {
      return `${scope}:${error.message}`;
    }
  }

  if (error instanceof Error && error.message.trim()) {
    return `${scope}:${error.message}`;
  }

  return `${scope}:${fallbackMessage}`;
}

export function emitDataFetchNotice(notice: DataFetchNotice) {
  if (typeof window === "undefined") return;

  const now = Date.now();
  const lastSeen = recentNoticeTimes.get(notice.dedupeKey) ?? 0;
  if (now - lastSeen < NOTICE_DEDUPE_MS) return;

  recentNoticeTimes.set(notice.dedupeKey, now);
  window.dispatchEvent(new CustomEvent<DataFetchNotice>(NOTICE_EVENT, { detail: notice }));
}

export function subscribeToDataFetchNotices(
  listener: (notice: DataFetchNotice) => void
) {
  if (typeof window === "undefined") return () => {};

  const handler = (event: Event) => {
    const customEvent = event as CustomEvent<DataFetchNotice>;
    if (!customEvent.detail) return;
    listener(customEvent.detail);
  };

  window.addEventListener(NOTICE_EVENT, handler as EventListener);
  return () => {
    window.removeEventListener(NOTICE_EVENT, handler as EventListener);
  };
}

export function notifyDataFetchError(
  error: unknown,
  fallbackMessage = "Failed to load data.",
  options?: DataFetchNoticeOptions
) {
  if (error instanceof DataFetchError && error.notified) return;

  emitDataFetchNotice({
    title: buildNoticeTitle(options),
    message: buildUserMessage(error, fallbackMessage),
    dedupeKey: buildDedupeKey(error, fallbackMessage, options),
  });

  if (error instanceof DataFetchError) {
    error.notified = true;
  }
}

async function parseErrorBody(response: Response) {
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) return null;

  try {
    return normalizeApiErrorBody(await response.json());
  } catch {
    return null;
  }
}

export async function createDataFetchErrorFromResponse(
  response: Response,
  fallbackMessage: string
) {
  const body = await parseErrorBody(response);
  const apiError = body?.error;

  return new DataFetchError({
    message:
      apiError ||
      `${fallbackMessage} (${response.status} ${response.statusText})`,
    status: response.status,
    code: body?.code,
    availableRange: body?.available_range,
    apiError,
  });
}

export async function fetchJsonOrThrow<T>(
  url: string,
  fallbackMessage: string,
  opts?: DataFetchRequestOptions
) {
  const { notifyOnError = true, layerLabel, ...requestInit } = opts ?? {};

  try {
    const response = await fetch(url, requestInit);
    if (!response.ok) {
      const error = await createDataFetchErrorFromResponse(response, fallbackMessage);
      if (notifyOnError) {
        notifyDataFetchError(error, fallbackMessage, { layerLabel });
      }
      throw error;
    }
    return (await response.json()) as T;
  } catch (error) {
    if (notifyOnError) {
      notifyDataFetchError(error, fallbackMessage, { layerLabel });
    }
    throw error;
  }
}

export async function fetchBlobOrThrow(
  url: string,
  fallbackMessage: string,
  opts?: DataFetchRequestOptions
) {
  const { notifyOnError = true, layerLabel, ...requestInit } = opts ?? {};

  try {
    const response = await fetch(url, requestInit);
    if (!response.ok) {
      const error = await createDataFetchErrorFromResponse(response, fallbackMessage);
      if (notifyOnError) {
        notifyDataFetchError(error, fallbackMessage, { layerLabel });
      }
      throw error;
    }
    return await response.blob();
  } catch (error) {
    if (notifyOnError) {
      notifyDataFetchError(error, fallbackMessage, { layerLabel });
    }
    throw error;
  }
}
