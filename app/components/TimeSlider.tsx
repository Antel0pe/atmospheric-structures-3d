"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getAppConfig } from "../lib/appConfig";

const MS_PER_HOUR = 3_600_000;
const COMMIT_DELAY_MS = 100;

function parseDateTimeUTC(value: string): Date {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/.exec(value);
  if (!m) throw new Error("Invalid datetime format. Expected YYYY-MM-DDTHH:mm");

  return new Date(
    Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]), Number(m[4]), Number(m[5]), 0)
  );
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

function formatDayLabelUTC(dt: Date): string {
  return new Intl.DateTimeFormat("en-US", {
    timeZone: "UTC",
    month: "short",
    day: "numeric",
  }).format(dt);
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

export interface TimeSliderProps {
  value: string;
  onChange: (next: string) => void;
  allReady: boolean;
}

export default function TimeSlider({ value, onChange, allReady }: TimeSliderProps) {
  const { startDate, endDate } = getAppConfig().sliderDateRange;
  const start = useMemo(() => parseDateTimeUTC(startDate), [startDate]);
  const end = useMemo(() => parseDateTimeUTC(endDate), [endDate]);
  const clampedValue = useMemo(
    () => clampValueToRange(value, start, end),
    [value, start, end]
  );
  const [stepHours, setStepHours] = useState<number>(3);
  const stepHoursRef = useRef<number>(3);

  useEffect(() => {
    stepHoursRef.current = stepHours;
  }, [stepHours]);

  useEffect(() => {
    if (clampedValue !== value) {
      onChange(clampedValue);
    }
  }, [clampedValue, onChange, value]);

  const totalHours = useMemo(() => {
    const spanMs = end.getTime() - start.getTime();
    return Math.max(0, Math.floor(spanMs / MS_PER_HOUR));
  }, [start, end]);

  const currentHours = useMemo(() => {
    const clampedMs = parseDateTimeUTC(clampedValue).getTime();
    const hrs = Math.floor((clampedMs - start.getTime()) / MS_PER_HOUR);
    return Math.max(0, Math.min(totalHours, hrs));
  }, [clampedValue, start, totalHours]);

  const [draftHours, setDraftHours] = useState<number>(currentHours);
  const [isPlaying, setIsPlaying] = useState(false);

  const totalHoursRef = useRef<number>(totalHours);
  const draftHoursRef = useRef<number>(draftHours);
  const commitTimerRef = useRef<number | null>(null);
  const playTimerRef = useRef<number | null>(null);
  const currentHoursRef = useRef(currentHours);

  useEffect(() => {
    totalHoursRef.current = totalHours;
  }, [totalHours]);

  useEffect(() => {
    draftHoursRef.current = draftHours;
  }, [draftHours]);

  useEffect(() => {
    currentHoursRef.current = currentHours;
  }, [currentHours]);

  const commitHours = useCallback(
    (hours: number) => {
      const dt = new Date(start.getTime() + hours * MS_PER_HOUR);
      onChange(formatDateTimeUTC(dt));
    },
    [onChange, start]
  );

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

  const setDraftAndSchedule = useCallback(
    (hours: number) => {
      const clamped = Math.max(0, Math.min(totalHoursRef.current, hours));
      setDraftHours(clamped);
      scheduleCommit(clamped);
    },
    [scheduleCommit]
  );

  const step = useCallback(
    (delta: -1 | 1) => {
      setDraftAndSchedule(draftHoursRef.current + delta * stepHoursRef.current);
    },
    [setDraftAndSchedule]
  );

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

  useEffect(() => {
    const isTypingTarget = (el: Element | null) => {
      if (!el) return false;
      const node = el as HTMLElement;
      const tag = node.tagName;
      return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || node.isContentEditable;
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
        commitHours(draftHoursRef.current);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [step, commitHours]);

  useEffect(() => {
    if (!isPlaying) return;
    if (!allReady) return;

    playTimerRef.current = window.setTimeout(() => {
      const next = Math.min(totalHoursRef.current, currentHoursRef.current + stepHoursRef.current);

      if (next === currentHoursRef.current) {
        setIsPlaying(false);
        return;
      }

      setDraftHours(next);
      commitHours(next);
    }, 700);

    return () => {
      if (playTimerRef.current !== null) {
        window.clearTimeout(playTimerRef.current);
        playTimerRef.current = null;
      }
    };
  }, [allReady, isPlaying, commitHours]);

  useEffect(() => {
    return () => {
      if (commitTimerRef.current !== null) window.clearTimeout(commitTimerRef.current);
      if (playTimerRef.current !== null) window.clearTimeout(playTimerRef.current);
    };
  }, []);

  const displayDate = useMemo(
    () => new Date(start.getTime() + draftHours * MS_PER_HOUR),
    [start, draftHours]
  );
  const currentPercent = totalHours > 0 ? (draftHours / totalHours) * 100 : 0;
  const tickMarks = useMemo(() => {
    if (totalHours <= 0) {
      return [{ label: formatDayLabelUTC(start), position: 0 }];
    }

    const maxTicks = 7;
    const spanDays = Math.max(1, Math.ceil(totalHours / 24));
    const intervalDays = Math.max(1, Math.ceil(spanDays / maxTicks));
    const ticks: { label: string; position: number }[] = [];
    const firstTick = new Date(
      Date.UTC(start.getUTCFullYear(), start.getUTCMonth(), start.getUTCDate())
    );

    if (firstTick.getTime() < start.getTime()) {
      firstTick.setUTCDate(firstTick.getUTCDate() + 1);
    }

    for (
      const tickDate = new Date(firstTick);
      tickDate.getTime() <= end.getTime();
      tickDate.setUTCDate(tickDate.getUTCDate() + intervalDays)
    ) {
      const hours = (tickDate.getTime() - start.getTime()) / MS_PER_HOUR;
      ticks.push({
        label: formatDayLabelUTC(tickDate),
        position: Math.max(0, Math.min(100, (hours / totalHours) * 100)),
      });
    }

    if (ticks.length === 0 || ticks[0].position > 4) {
      ticks.unshift({ label: formatDayLabelUTC(start), position: 0 });
    }

    const endLabel = formatDayLabelUTC(end);
    if (ticks[ticks.length - 1].position < 96 && ticks[ticks.length - 1].label !== endLabel) {
      ticks.push({ label: endLabel, position: 100 });
    }

    return ticks;
  }, [end, start, totalHours]);

  const timebarStyle: React.CSSProperties = {
    width: "100%",
    height: 128,
    display: "grid",
    gridTemplateRows: "50px 56px",
    gap: 5,
    boxSizing: "border-box",
    padding: "15px 20px 12px",
    border: "1px solid rgba(48, 62, 78, 0.7)",
    borderRadius: 4,
    background: "rgba(4, 10, 17, 0.94)",
    color: "rgba(226, 235, 248, 0.94)",
    boxShadow: "0 16px 34px rgba(0, 0, 0, 0.42), inset 0 1px 0 rgba(255,255,255,0.04)",
    backdropFilter: "blur(18px)",
    fontSize: 11,
    lineHeight: 1,
  };

  const headerStyle: React.CSSProperties = {
    position: "relative",
    display: "grid",
    gridTemplateColumns: "minmax(0, 1fr) auto minmax(0, 1fr)",
    alignItems: "start",
    columnGap: 16,
  };

  const buttonStyle: React.CSSProperties = {
    width: 32,
    height: 30,
    display: "inline-grid",
    placeItems: "center",
    border: "1px solid rgba(76, 93, 114, 0.48)",
    borderRadius: 5,
    background: "rgba(7, 15, 26, 0.82)",
    color: "rgba(204, 218, 237, 0.88)",
    cursor: "pointer",
    padding: 0,
    boxShadow: "0 5px 14px rgba(0, 0, 0, 0.34), inset 0 1px 0 rgba(255,255,255,0.04)",
  };

  const primaryButtonStyle: React.CSSProperties = {
    ...buttonStyle,
    width: 38,
    height: 34,
    color: "#3f8fff",
    borderColor: "rgba(67, 119, 198, 0.56)",
    background: "rgba(18, 36, 61, 0.92)",
  };

  const iconStyle: React.CSSProperties = {
    width: 15,
    height: 15,
    display: "block",
    fill: "currentColor",
  };

  return (
    <div className="atm-timebar" style={timebarStyle}>
      <div className="atm-timebar-header" style={headerStyle}>
        <div
          className="atm-timebar-current"
          title={`${formatPrettyUTC(displayDate)} UTC`}
          style={{
            minWidth: 0,
            overflow: "hidden",
            color: "rgba(220, 229, 243, 0.88)",
            fontSize: 13,
            fontWeight: 700,
            lineHeight: "34px",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {formatPrettyUTC(displayDate)} UTC
        </div>

        <div
          className="atm-timebar-controls"
          aria-label="Time controls"
          style={{ display: "inline-flex", alignItems: "center", gap: 12 }}
        >
          <button
            type="button"
            onClick={() => nudgeByStep(-1)}
            aria-label="Step backward time"
            data-testid="time-step-backward"
            title={`Step backward ${stepHours} hours`}
            className="atm-timebar-button"
            style={buttonStyle}
          >
            <svg viewBox="0 0 24 24" aria-hidden style={iconStyle}>
              <path d="M11 7 6 12l5 5V7Z" />
              <path d="M18 7l-5 5 5 5V7Z" />
            </svg>
          </button>

          <button
            type="button"
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
            className="atm-timebar-button atm-timebar-button-primary"
            data-playing={isPlaying}
            style={primaryButtonStyle}
          >
            {isPlaying ? (
              <svg
                viewBox="0 0 24 24"
                aria-hidden
                style={{ ...iconStyle, fill: "none", stroke: "currentColor", strokeWidth: 2.4 }}
              >
                <path d="M8 6v12M16 6v12" strokeLinecap="round" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" aria-hidden style={{ ...iconStyle, width: 17, height: 17 }}>
                <path d="m9 6 9 6-9 6V6Z" />
              </svg>
            )}
          </button>

          <button
            type="button"
            onClick={() => nudgeByStep(1)}
            aria-label="Step forward time"
            data-testid="time-step-forward"
            title={`Step forward ${stepHours} hours`}
            className="atm-timebar-button"
            style={buttonStyle}
          >
            <svg viewBox="0 0 24 24" aria-hidden style={iconStyle}>
              <path d="m13 7 5 5-5 5V7Z" />
              <path d="m6 7 5 5-5 5V7Z" />
            </svg>
          </button>
        </div>

        <label
          className="atm-timebar-step"
          style={{
            justifySelf: "end",
            display: "inline-flex",
            alignItems: "center",
            gap: 4,
            color: "rgba(202, 213, 228, 0.82)",
            fontSize: 11,
            fontWeight: 760,
            lineHeight: "34px",
            whiteSpace: "nowrap",
          }}
        >
          <span>Step:</span>
          <select
            aria-label="Time step"
            value={stepHours}
            onChange={(e) => setStepHours(Number(e.target.value))}
            style={{
              width: 66,
              height: 34,
              border: 0,
              appearance: "none",
              background: "transparent",
              color: "rgba(225, 234, 247, 0.92)",
              cursor: "pointer",
              font: "inherit",
              fontWeight: 800,
              outline: "none",
              padding: "0 16px 0 0",
            }}
          >
            {[1, 3, 6, 12, 24].map((hours) => (
              <option key={hours} value={hours}>
                {hours} {hours === 1 ? "hour" : "hours"}
              </option>
            ))}
          </select>
          <svg
            aria-hidden
            viewBox="0 0 16 16"
            style={{
              width: 12,
              height: 12,
              marginLeft: -13,
              color: "rgba(188, 201, 219, 0.88)",
              fill: "none",
              stroke: "currentColor",
              strokeWidth: 2,
              strokeLinecap: "round",
              strokeLinejoin: "round",
              pointerEvents: "none",
            }}
          >
            <path d="m4 6 4 4 4-4" />
          </svg>
        </label>
      </div>

      <div
        className="atm-timebar-rail"
        style={{ position: "relative", minHeight: 56, paddingTop: 11 }}
      >
        <div
          aria-hidden
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            top: 13,
            height: 2,
            borderRadius: 999,
            background: "rgba(72, 86, 104, 0.52)",
            boxShadow: "0 0 0 1px rgba(10, 17, 27, 0.72)",
          }}
        >
          <div
            style={{
              width: `${currentPercent}%`,
              height: "100%",
              borderRadius: 999,
              background: "rgba(62, 124, 216, 0.34)",
            }}
          />
        </div>

        <div
          aria-hidden
          style={{
            position: "absolute",
            left: `${currentPercent}%`,
            top: 6,
            width: 14,
            height: 14,
            transform: "translateX(-50%)",
            border: "2px solid rgba(126, 142, 168, 0.98)",
            borderRadius: 999,
            background: "#050b13",
            boxShadow: "0 0 0 2px rgba(5, 10, 18, 0.92), 0 2px 8px rgba(0, 0, 0, 0.52)",
          }}
        />

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
          className="atm-time-range"
          style={{
            position: "absolute",
            inset: "0 0 auto",
            zIndex: 2,
            width: "100%",
            height: 20,
            margin: 0,
            opacity: 0,
            cursor: "pointer",
          }}
        />

        <div
          className="atm-timebar-ticks"
          aria-hidden
          style={{
            position: "absolute",
            inset: "34px 0 auto",
            height: 13,
            color: "rgba(157, 172, 193, 0.78)",
            fontSize: 10,
            fontWeight: 760,
            pointerEvents: "none",
          }}
        >
          {tickMarks.map((tick) => (
            <span
              key={`${tick.label}-${tick.position}`}
              style={{
                position: "absolute",
                top: 0,
                left: `${tick.position}%`,
                transform:
                  tick.position <= 0
                    ? "translateX(0)"
                    : tick.position >= 100
                      ? "translateX(-100%)"
                      : "translateX(-50%)",
                whiteSpace: "nowrap",
              }}
            >
              {tick.label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
