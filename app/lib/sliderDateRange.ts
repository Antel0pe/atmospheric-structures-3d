const DEFAULT_DATA_START_DATE = "2021-11-01T00:00";
const DEFAULT_DATA_END_DATE = "2021-11-30T23:00";

export type SliderDateRange = {
  startDate: string;
  endDate: string;
};

const SLIDER_DATE_RANGE: SliderDateRange = {
  startDate: process.env.NEXT_PUBLIC_DATA_START_DATE?.trim() || DEFAULT_DATA_START_DATE,
  endDate: process.env.NEXT_PUBLIC_DATA_END_DATE?.trim() || DEFAULT_DATA_END_DATE,
};

export function getSliderDateRangeFromEnv(): SliderDateRange {
  return SLIDER_DATE_RANGE;
}
