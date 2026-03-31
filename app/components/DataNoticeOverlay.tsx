"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  subscribeToDataFetchNotices,
  type DataFetchNotice,
} from "./utils/dataFetchErrors";

type NoticeItem = DataFetchNotice & {
  id: number;
  visible: boolean;
};

const AUTO_DISMISS_MS = 4200;
const EXIT_MS = 260;

export default function DataNoticeOverlay() {
  const [notices, setNotices] = useState<NoticeItem[]>([]);
  const nextIdRef = useRef(1);
  const hideTimersRef = useRef<Map<number, number>>(new Map());
  const removeTimersRef = useRef<Map<number, number>>(new Map());

  const hideNotice = useCallback((id: number) => {
    setNotices((current) =>
      current.map((notice) =>
        notice.id === id ? { ...notice, visible: false } : notice
      )
    );

    const removeTimers = removeTimersRef.current;
    const existingRemoveTimer = removeTimers.get(id);
    if (existingRemoveTimer) window.clearTimeout(existingRemoveTimer);

    const removeTimer = window.setTimeout(() => {
      removeTimers.delete(id);
      setNotices((current) => current.filter((notice) => notice.id !== id));
    }, EXIT_MS);

    removeTimers.set(id, removeTimer);
  }, []);

  const dismissNotice = useCallback((id: number) => {
    const hideTimers = hideTimersRef.current;
    const hideTimer = hideTimers.get(id);
    if (hideTimer) {
      window.clearTimeout(hideTimer);
      hideTimers.delete(id);
    }

    hideNotice(id);
  }, [hideNotice]);

  useEffect(() => {
    const hideTimers = hideTimersRef.current;
    const removeTimers = removeTimersRef.current;

    const unsubscribe = subscribeToDataFetchNotices((notice) => {
      const id = nextIdRef.current++;
      const item: NoticeItem = { ...notice, id, visible: false };

      setNotices((current) => [...current, item]);

      requestAnimationFrame(() => {
        setNotices((current) =>
          current.map((entry) =>
            entry.id === id ? { ...entry, visible: true } : entry
          )
        );
      });

      const hideTimer = window.setTimeout(() => {
        hideTimers.delete(id);
        hideNotice(id);
      }, AUTO_DISMISS_MS);

      hideTimers.set(id, hideTimer);
    });

    return () => {
      unsubscribe();

      for (const timer of hideTimers.values()) {
        window.clearTimeout(timer);
      }
      hideTimers.clear();

      for (const timer of removeTimers.values()) {
        window.clearTimeout(timer);
      }
      removeTimers.clear();
    };
  }, [hideNotice]);

  if (notices.length === 0) return null;

  return (
    <div
      style={{
        position: "fixed",
        top: 14,
        left: 0,
        right: 0,
        zIndex: 90,
        display: "grid",
        justifyItems: "center",
        gap: 10,
        pointerEvents: "none",
        padding: "0 16px",
      }}
    >
      {notices.map((notice) => (
        <div
          key={notice.id}
          role="status"
          aria-live="polite"
          style={{
            position: "relative",
            width: "min(760px, 100%)",
            padding: "12px 48px 12px 16px",
            borderRadius: 14,
            border: "1px solid rgba(255, 178, 164, 0.28)",
            background:
              "linear-gradient(180deg, rgba(66, 16, 16, 0.92), rgba(28, 10, 12, 0.9))",
            color: "#fff1ec",
            boxShadow: "0 16px 34px rgba(0,0,0,0.3)",
            backdropFilter: "blur(12px)",
            font: "600 13px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto",
            lineHeight: 1.4,
            opacity: notice.visible ? 1 : 0,
            transform: notice.visible
              ? "translateY(0)"
              : "translateY(-18px)",
            transition:
              "transform 220ms cubic-bezier(0.2, 0.8, 0.2, 1), opacity 180ms ease",
            pointerEvents: "auto",
          }}
        >
          <button
            type="button"
            aria-label={`Dismiss ${notice.title} notice`}
            onClick={() => dismissNotice(notice.id)}
            style={{
              position: "absolute",
              top: 8,
              right: 8,
              width: 28,
              height: 28,
              borderRadius: 999,
              border: "1px solid rgba(255,255,255,0.16)",
              background: "rgba(255,255,255,0.08)",
              color: "#fff1ec",
              cursor: "pointer",
              display: "grid",
              placeItems: "center",
              fontSize: 15,
              lineHeight: 1,
            }}
          >
            ×
          </button>

          <div
            style={{
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              opacity: 0.74,
              marginBottom: 4,
            }}
          >
            {notice.title}
          </div>
          <div style={{ fontSize: 13, fontWeight: 600 }}>{notice.message}</div>
        </div>
      ))}
    </div>
  );
}
