import { useEffect, useState } from "react";
import type { RubricItem } from "../types/pipeline";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";

const ORDER: RubricItem["id"][] = ["groundedness", "completeness", "structure", "citations"];

type Props = { items: RubricItem[]; live?: boolean };

/** Feature 7 — M2 Pin Snap per row, 180ms stagger, never simultaneous. */
export function RubricChecklist({ items, live = true }: Props) {
  const reduced = usePrefersReducedMotion();
  const [shown, setShown] = useState(0);

  useEffect(() => {
    const resolved = items.filter((i) => i.passed !== undefined).length;
    if (!live || reduced) {
      setShown(resolved);
      return;
    }
    setShown(0);
    const timers: number[] = [];
    for (let i = 0; i < resolved; i += 1) {
      timers.push(window.setTimeout(() => setShown(i + 1), 180 * i));
    }
    return () => timers.forEach((t) => window.clearTimeout(t));
  }, [items, live, reduced]);

  const byId = Object.fromEntries(items.map((i) => [i.id, i]));

  return (
    <div className="rubric" aria-label="Reviewer rubric">
      {ORDER.map((id, index) => {
        const item = byId[id];
        const resolved = item && item.passed !== undefined && index < shown;
        return (
          <div
            key={id}
            className={`rubric-row${resolved ? " is-resolved" : ""}${
              resolved && item.passed ? " is-pass" : ""
            }${resolved && item.passed === false ? " is-fail" : ""}`}
          >
            <div className="rubric-head">
              <span className="rubric-dot" data-motion="M2" />
              <span>{item?.label ?? id}</span>
            </div>
            {resolved && item.passed === false && item.reason ? (
              <div className="rubric-reason">{item.reason}</div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
