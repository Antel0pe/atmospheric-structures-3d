import { NextResponse } from "next/server";

function keyToIsoUtc(key: string) {
  let match =
    /^(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})\.(?:png|json)$/.exec(key);
  if (match) {
    return `${match[1]}T${match[2]}:${match[3]}:${match[4]}Z`;
  }

  match = /^(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})\.json$/.exec(key);
  if (match) {
    return `${match[1]}T${match[2]}:${match[3]}:00Z`;
  }

  match = /^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})$/.exec(key);
  if (match) {
    return `${match[1]}T${match[2]}:${match[3]}:00Z`;
  }

  return key;
}

export function noDataForDateResponse(firstKey: string, lastKey: string) {
  return NextResponse.json(
    {
      error: "no data exists for this date",
      code: "NO_DATA_FOR_DATE",
      available_range: {
        start: keyToIsoUtc(firstKey),
        end: keyToIsoUtc(lastKey),
      },
    },
    { status: 404 }
  );
}
