import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { Checkpoint } from "../types/pipeline";
import { VerdictBadge } from "./VerdictBadge";
import { SignalPulse } from "./SignalPulse";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";

type Props = {
  checkpoints: Checkpoint[];
  selectedId: string | null;
  onScrub: (id: string) => void;
  live?: boolean;
};

/** Feature 6 — M1 Ink Rise on new rows; M2 Pin Snap on selection (FLIP bar). */
export function RunTimeline({ checkpoints, selectedId, onScrub, live = false }: Props) {
  const reduced = usePrefersReducedMotion();
  const listRef = useRef<HTMLUListElement>(null);
  const barRef = useRef<HTMLSpanElement>(null);
  const prevBar = useRef<DOMRect | null>(null);
  const seen = useRef<Set<string>>(new Set());
  const primed = useRef(false);
  const [entering, setEntering] = useState<Set<string>>(new Set());

  useLayoutEffect(() => {
    if (!primed.current) {
      checkpoints.forEach((cp) => seen.current.add(cp.id));
      primed.current = true;
      return;
    }
    const fresh = checkpoints.filter((cp) => !seen.current.has(cp.id)).map((cp) => cp.id);
    fresh.forEach((id) => seen.current.add(id));
    setEntering(new Set(fresh));
  }, [checkpoints]);

  useEffect(() => {
    const list = listRef.current;
    if (!list || checkpoints.length === 0) return;
    if (reduced) {
      list.scrollTop = list.scrollHeight;
      return;
    }
    const from = list.scrollTop;
    const to = Math.max(0, list.scrollHeight - list.clientHeight);
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / 400);
      const ease = 1 - Math.pow(1 - t, 3);
      list.scrollTop = from + (to - from) * ease;
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [checkpoints.length, reduced]);

  useLayoutEffect(() => {
    const el = barRef.current;
    if (!el) return;
    const next = el.getBoundingClientRect();
    const prev = prevBar.current;
    prevBar.current = next;
    if (!prev || reduced) return;
    const dy = prev.top - next.top;
    const sh = prev.height / (next.height || 1);
    el.animate([{ transform: `translateY(${dy}px) scaleY(${sh})` }, { transform: "none" }], {
      duration: 250,
      easing: "cubic-bezier(.34,1.56,.64,1)",
    });
  }, [selectedId, reduced]);

  const lastId = checkpoints[checkpoints.length - 1]?.id;

  return (
    <ul className="timeline" ref={listRef}>
      {checkpoints.map((cp) => {
        const selected = cp.id === selectedId;
        const isLive = live && cp.id === lastId;
        const rise = entering.has(cp.id);
        return (
          <li key={cp.id}>
            <button
              type="button"
              className={`timeline-row${selected ? " is-selected" : ""}${rise ? " is-entering" : ""}`}
              onClick={() => onScrub(cp.id)}
            >
              {selected ? <span className="timeline-bar" ref={barRef} /> : null}
              <span className="timeline-time">{cp.timestamp}</span>
              <span className="timeline-node">
                {cp.node}
                {isLive ? (
                  <span className="timeline-live">
                    <SignalPulse />
                  </span>
                ) : null}
              </span>
              {cp.verdict ? <VerdictBadge verdict={cp.verdict} /> : <span />}
            </button>
          </li>
        );
      })}
    </ul>
  );
}
