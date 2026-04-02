"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getAppConfig } from "../lib/appConfig";

// -----------------------------
// Constants
// -----------------------------
const MS_PER_HOUR = 3_600_000;
const COMMIT_DELAY_MS = 100;

// -----------------------------
// Date helpers (UTC-only)
// -----------------------------
function parseDateTimeUTC(value: string): Date {
  // Expect "YYYY-MM-DDTHH:mm" interpreted as UTC
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/.exec(value);
  if (!m) throw new Error("Invalid datetime format. Expected YYYY-MM-DDTHH:mm");

  const y = Number(m[1]);
  const mo = Number(m[2]) - 1;
  const d = Number(m[3]);
  const h = Number(m[4]);
  const min = Number(m[5]);

  return new Date(Date.UTC(y, mo, d, h, min, 0));
}

function formatDateTimeUTC(dt: Date): string {
  const y = dt.getUTCFullYear();
  const mo = String(dt.getUTCMonth() + 1).padStart(2, "0");
  const d = String(dt.getUTCDate()).padStart(2, "0");
  const h = String(dt.getUTCHours()).padStart(2, "0");
  const min = String(dt.getUTCMinutes()).padStart(2, "0");
  return `${y}-${mo}-${d}T${h}:${min}`;
}

function formatPrettyUTC(dt: Date): string {
  return new Intl.DateTimeFormat("en-US", {
    timeZone: "UTC",
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(dt);
}

function formatCompactUTC(dt: Date): string {
  const mo = String(dt.getUTCMonth() + 1).padStart(2, "0");
  const d = String(dt.getUTCDate()).padStart(2, "0");
  const y = String(dt.getUTCFullYear()).slice(-2);
  const rawHour = dt.getUTCHours();
  const hour = rawHour % 12 || 12;
  const suffix = rawHour >= 12 ? "PM" : "AM";
  const min = dt.getUTCMinutes();
  const minutePart = min === 0 ? "" : `:${String(min).padStart(2, "0")}`;
  return `${mo}/${d}/${y} ${hour}${minutePart} ${suffix}`;
}

function prettyFromValueStrUTC(valueStr: string): string {
  return formatPrettyUTC(parseDateTimeUTC(valueStr));
}

function clampValueToRange(value: string, start: Date, end: Date) {
  let curMs: number;
  try {
    curMs = parseDateTimeUTC(value).getTime();
  } catch {
    curMs = start.getTime();
  }

  const clampedMs = Math.max(start.getTime(), Math.min(end.getTime(), curMs));
  return formatDateTimeUTC(new Date(clampedMs));
}

// -----------------------------
// Component
// -----------------------------
export interface TimeSliderProps {
  value: string; // "YYYY-MM-DDTHH:mm" (UTC)
  onChange: (next: string) => void;
  allReady: boolean;
}

export default function TimeSlider({
  value,
  onChange,
  allReady,
}: TimeSliderProps) {
  const { startDate, endDate } = getAppConfig().sliderDateRange;
  const start = useMemo(() => parseDateTimeUTC(startDate), [startDate]);
  const end = useMemo(() => parseDateTimeUTC(endDate), [endDate]);
  const clampedValue = useMemo(() => clampValueToRange(value, start, end), [value, start, end]);
  const [stepHours, setStepHours] = useState<number>(3);
  const stepHoursRef = useRef<number>(3);
  const [collapsed, setCollapsed] = useState(true);
  useEffect(() => { stepHoursRef.current = stepHours; }, [stepHours]);

  useEffect(() => {
    if (clampedValue !== value) {
      onChange(clampedValue);
    }
  }, [clampedValue, onChange, value]);

  // Total slider span in whole hours
  const totalHours = useMemo(() => {
    const spanMs = end.getTime() - start.getTime();
    return Math.max(0, Math.floor(spanMs / MS_PER_HOUR));
  }, [start, end]);

  // Clamp incoming prop value into [start, end], then convert to hour offset
  const currentHours = useMemo(() => {
    const clampedMs = parseDateTimeUTC(clampedValue).getTime();
    const hrs = Math.floor((clampedMs - start.getTime()) / MS_PER_HOUR);
    return Math.max(0, Math.min(totalHours, hrs));
  }, [clampedValue, start, totalHours]);

  // Draft (UI) state
  const [draftHours, setDraftHours] = useState<number>(currentHours);

  // Refs to avoid stale closures in event handlers / timers
  const totalHoursRef = useRef<number>(totalHours);
  const draftHoursRef = useRef<number>(draftHours);
  const commitTimerRef = useRef<number | null>(null);

  useEffect(() => {
    totalHoursRef.current = totalHours;
  }, [totalHours]);

  useEffect(() => {
    draftHoursRef.current = draftHours;
  }, [draftHours]);

  // Commit draft -> onChange (UTC string)
  const commitHours = useCallback(
    (hours: number) => {
      const dt = new Date(start.getTime() + hours * MS_PER_HOUR);
      onChange(formatDateTimeUTC(dt));
    },
    [onChange, start]
  );

  // Debounced commit while dragging / holding keys
  const scheduleCommit = useCallback(
    (hours: number) => {
      if (commitTimerRef.current !== null) window.clearTimeout(commitTimerRef.current);

      commitTimerRef.current = window.setTimeout(() => {
        commitTimerRef.current = null;
        commitHours(hours);
      }, COMMIT_DELAY_MS);
    },
    [commitHours]
  );

  // One helper for “update UI + schedule commit”
  const setDraftAndSchedule = useCallback(
    (hours: number) => {
      const clamped = Math.max(0, Math.min(totalHoursRef.current, hours));
      setDraftHours(clamped);
      scheduleCommit(clamped);
    },
    [scheduleCommit]
  );

  // Keyboard stepping
  const step = useCallback(
    (delta: -1 | 1) => {
      setDraftAndSchedule(draftHoursRef.current + delta * stepHoursRef.current);
    },
    [setDraftAndSchedule]
  );

  const [isPlaying, setIsPlaying] = useState(false);
  const playTimerRef = useRef<number | null>(null);
  const nudgeByStep = useCallback(
    (delta: -1 | 1) => {
      setIsPlaying(false);
      if (commitTimerRef.current !== null) {
        window.clearTimeout(commitTimerRef.current);
        commitTimerRef.current = null;
      }

      const next = Math.max(
        0,
        Math.min(totalHoursRef.current, draftHoursRef.current + delta * stepHoursRef.current)
      );

      setDraftHours(next);
      commitHours(next);
    },
    [commitHours]
  );

  // Keyboard listeners (ArrowLeft/ArrowRight)
  useEffect(() => {
    const isTypingTarget = (el: Element | null) => {
      if (!el) return false;
      const node = el as HTMLElement;
      const tag = node.tagName;
      return tag === "INPUT" || tag === "TEXTAREA" || node.isContentEditable;
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (isTypingTarget(document.activeElement)) return;

      if (e.key === "ArrowLeft") {
        setIsPlaying(false);
        step(-1);
        e.preventDefault();
      } else if (e.key === "ArrowRight") {
        setIsPlaying(false);
        step(1);
        e.preventDefault();
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        commitHours(draftHoursRef.current); // commit immediately on release
      }
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [step, commitHours]);

  const currentHoursRef = useRef(currentHours);
  useEffect(() => { currentHoursRef.current = currentHours; }, [currentHours]);

  useEffect(() => {
    if (!isPlaying) return;

    // Don't advance until the scene reports ready for the current timestamp
    if (!allReady) return;

    // After ready, wait 200ms, then advance by 1 hour
    playTimerRef.current = window.setTimeout(() => {
      const next = Math.min(totalHoursRef.current, currentHoursRef.current + stepHoursRef.current);

      // stop at end
      if (next === currentHoursRef.current) {
        setIsPlaying(false);
        return;
      }

      // advance UI immediately so slider moves
      setDraftHours(next);

      // commit the timestamp change (this will flip allReady false in HomeClient per your wiring)
      commitHours(next);
    }, 700);

    return () => {
      if (playTimerRef.current !== null) {
        window.clearTimeout(playTimerRef.current);
        playTimerRef.current = null;
      }
    };
  }, [allReady, isPlaying, commitHours]);


  // Cleanup any pending timer on unmount
  useEffect(() => {
    return () => {
      if (commitTimerRef.current !== null) window.clearTimeout(commitTimerRef.current);
    };
  }, []);

  // Display value (based on draftHours)
  const displayDate = useMemo(
    () => new Date(start.getTime() + draftHours * MS_PER_HOUR),
    [start, draftHours]
  );
  const displayValueStr = useMemo(() => formatDateTimeUTC(displayDate), [displayDate]);
  const compactDisplayValue = useMemo(() => formatCompactUTC(displayDate), [displayDate]);

  const clamp = (v: number) => Math.max(1, Math.min(24, v));
  const chromeButtonStyle: React.CSSProperties = {
    width: 34,
    height: 34,
    borderRadius: 999,
    border: "1px solid rgba(255,255,255,0.25)",
    background: "rgba(0,0,0,0.35)",
    color: "white",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    opacity: 1,
    userSelect: "none",
    lineHeight: 1,
    padding: 0,
    flex: "0 0 auto",
  };

  return (
    <div
      style={{
        position: "relative",
        display: "flex",
        flexDirection: "column",
        gap: collapsed ? 0 : 8,
        padding: collapsed ? "14px 18px 12px" : "14px 18px",
        width: collapsed ? "fit-content" : "100%",
        maxWidth: "100%",
        margin: "0 auto",
        justifyContent: "center",
        borderRadius: 22,
        border: "1px solid rgba(255,255,255,0.2)",
        background: "rgba(6,10,18,0.14)",
        backdropFilter: "blur(14px)",
        boxShadow: "0 18px 40px rgba(0,0,0,0.22)",
        color: "rgba(255,255,255,0.96)",
      }}
    >
      {collapsed ? (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 8,
            minWidth: 180,
            paddingTop: 4,
          }}
        >
          <button
            onClick={() => setCollapsed(false)}
            aria-label="Expand time slider"
            title="Expand time slider"
            style={{
              ...chromeButtonStyle,
              position: "absolute",
              top: 14,
              right: 10,
              width: 28,
              height: 28,
            }}
          >
            <span style={{ fontSize: 16, transform: "translateY(-1px)" }}>▴</span>
          </button>

          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <button
              onClick={() => nudgeByStep(-1)}
              aria-label={`Step backward ${stepHours} hours`}
              title={`Step backward ${stepHours} hours`}
              style={chromeButtonStyle}
            >
              <span style={{ fontSize: 16, transform: "translateX(-1px)" }}>←</span>
            </button>

            <button
              onClick={() => setIsPlaying((p) => !p)}
              title={
                isPlaying
                  ? allReady
                    ? "Pause"
                    : "Pause (waiting for render...)"
                  : allReady
                    ? "Play"
                    : "Play (will resume when ready)"
              }
              aria-label={isPlaying ? "Pause" : "Play"}
              style={chromeButtonStyle}
            >
              <span style={{ fontSize: 14, transform: isPlaying ? "none" : "translateX(1px)" }}>
                {isPlaying ? "❚❚" : "▶"}
              </span>
            </button>

            <button
              onClick={() => nudgeByStep(1)}
              aria-label={`Step forward ${stepHours} hours`}
              title={`Step forward ${stepHours} hours`}
              style={chromeButtonStyle}
            >
              <span style={{ fontSize: 16, transform: "translateX(1px)" }}>→</span>
            </button>
          </div>

          <div
            style={{
              textAlign: "center",
              fontSize: 14,
              fontWeight: 600,
              letterSpacing: 0.2,
              whiteSpace: "nowrap",
            }}
            title={`${formatPrettyUTC(displayDate)} UTC`}
          >
            {compactDisplayValue}
          </div>
        </div>
      ) : (
        <>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              fontSize: 12,
            }}
          >
            <div style={{ flex: 1, textAlign: "left" }}>
              {prettyFromValueStrUTC(startDate)} UTC
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <button
                onClick={() => nudgeByStep(-1)}
                aria-label={`Step backward ${stepHours} hours`}
                title={`Step backward ${stepHours} hours`}
                style={chromeButtonStyle}
              >
                <span style={{ fontSize: 16, transform: "translateX(-1px)" }}>←</span>
              </button>

              <button
                onClick={() => setIsPlaying((p) => !p)}
                title={
                  isPlaying
                    ? allReady
                      ? "Pause"
                      : "Pause (waiting for render...)"
                    : allReady
                      ? "Play"
                      : "Play (will resume when ready)"
                }
                aria-label={isPlaying ? "Pause" : "Play"}
                style={chromeButtonStyle}
              >
                <span style={{ fontSize: 14, transform: isPlaying ? "none" : "translateX(1px)" }}>
                  {isPlaying ? "❚❚" : "▶"}
                </span>
              </button>

              <button
                onClick={() => nudgeByStep(1)}
                aria-label={`Step forward ${stepHours} hours`}
                title={`Step forward ${stepHours} hours`}
                style={chromeButtonStyle}
              >
                <span style={{ fontSize: 16, transform: "translateX(1px)" }}>→</span>
              </button>
            </div>

            <div style={{ flex: 1, textAlign: "right" }}>
              {prettyFromValueStrUTC(endDate)} UTC
            </div>
          </div>

          <input
            type="range"
            min={0}
            max={totalHours}
            step={1}
            value={draftHours}
            onChange={(e) => {
              setIsPlaying(false);
              const h = Number(e.target.value);
              setDraftHours(h);
              scheduleCommit(h);
            }}
            style={{ width: "100%", accentColor: "#9fc8ff" }}
          />

          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 160, display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
              <div style={{ opacity: 0.85, whiteSpace: "nowrap" }}>Step (hours)</div>

              <input
                type="number"
                min={1}
                max={24}
                step={1}
                value={stepHours}
                onChange={(e) => setStepHours(clamp(Number(e.target.value)))}
                onWheel={(e) => {
                  e.preventDefault();
                  const dir = e.deltaY > 0 ? -1 : 1;
                  setStepHours((s) => clamp(s + dir));
                }}
                style={{
                  width: 64,
                  padding: "6px 8px",
                  borderRadius: 8,
                  border: "1px solid rgba(255,255,255,0.25)",
                  background: "rgba(0,0,0,0.35)",
                  color: "white",
                  outline: "none",
                  textAlign: "center",
                }}
              />

              <div style={{ opacity: 0.75, whiteSpace: "nowrap" }}>h</div>
            </div>

            <div style={{ flex: 1, textAlign: "center", fontSize: 12 }}>
              {formatPrettyUTC(parseDateTimeUTC(displayValueStr))} UTC
            </div>

            <div style={{ width: 160, display: "flex", justifyContent: "flex-end" }}>
              <button
                onClick={() => setCollapsed(true)}
                aria-label="Collapse time slider"
                title="Collapse time slider"
                style={chromeButtonStyle}
              >
                <span style={{ fontSize: 16, transform: "translateY(1px)" }}>▾</span>
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
