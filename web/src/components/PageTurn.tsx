import { useEffect, useRef, useState, type ReactNode } from "react";
import { useLocation } from "react-router-dom";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";

const SHARED_KEY = "repointel:shared-run";

export function markSharedRun(id: string, el: HTMLElement) {
  const r = el.getBoundingClientRect();
  sessionStorage.setItem(
    SHARED_KEY,
    JSON.stringify({ id, top: r.top, left: r.left, width: r.width, height: r.height }),
  );
}

export function consumeSharedRun(): {
  id: string;
  top: number;
  left: number;
  width: number;
  height: number;
} | null {
  const raw = sessionStorage.getItem(SHARED_KEY);
  if (!raw) return null;
  sessionStorage.removeItem(SHARED_KEY);
  return JSON.parse(raw) as {
    id: string;
    top: number;
    left: number;
    width: number;
    height: number;
  };
}

function intoLiveRun(from: string, to: string) {
  return to.startsWith("/app/runs/") && (from === "/app" || from === "/app/");
}

/** Feature 10 — M8 Page Turn, except Live Run which uses a shared-element FLIP. */
export function PageTurn({ children }: { children: ReactNode }) {
  const location = useLocation();
  const reduced = usePrefersReducedMotion();
  const prevPath = useRef(location.pathname);
  const shownRef = useRef<ReactNode>(children);
  const [shown, setShown] = useState(children);
  const [leaving, setLeaving] = useState<ReactNode | null>(null);

  shownRef.current = shown;

  useEffect(() => {
    const from = prevPath.current;
    const to = location.pathname;
    if (from === to) {
      setShown(children);
      return;
    }
    prevPath.current = to;

    if (reduced || intoLiveRun(from, to)) {
      setLeaving(null);
      setShown(children);
      return;
    }

    setLeaving(shownRef.current);
    setShown(children);
    const t = window.setTimeout(() => setLeaving(null), 380);
    return () => window.clearTimeout(t);
  }, [location.pathname, children, reduced]);

  return (
    <div className="page-stage" data-motion="M8">
      {leaving ? (
        <div className="page-layer page-layer-out" aria-hidden>
          {leaving}
        </div>
      ) : null}
      <div className={leaving ? "page-layer page-layer-in" : undefined}>{shown}</div>
    </div>
  );
}
