import { useCallback, useEffect, useRef, useState } from "react";
import type { PipelineEvent } from "../types/pipeline";
import { usePrefersReducedMotion } from "./usePrefersReducedMotion";

type Source =
  | { kind: "scripted"; events: PipelineEvent[]; loop?: boolean }
  | { kind: "live"; events: PipelineEvent[] };

/**
 * Drives PipelineDiagram from either a scripted array or a live event list.
 * Animation logic in the diagram is identical; only the data source differs.
 */
export function usePipelinePlayback(source: Source) {
  const reduced = usePrefersReducedMotion();
  const [index, setIndex] = useState(-1);
  const [frozen, setFrozen] = useState(false);
  const timer = useRef<number | null>(null);
  const events = source.events;
  const looping = source.kind === "scripted" && source.loop;
  const live = source.kind === "live";

  const clear = () => {
    if (timer.current != null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
  };

  const schedule = useCallback(
    (nextIndex: number) => {
      clear();
      if (frozen) return;
      const event = events[nextIndex];
      if (!event) {
        if (looping) {
          const pause = reduced ? 400 : 1600;
          timer.current = window.setTimeout(() => {
            setFrozen(false);
            setIndex(-1);
          }, pause);
        }
        return;
      }
      const delay = reduced ? 80 : (event.delayMs ?? 700);
      timer.current = window.setTimeout(() => {
        setIndex(nextIndex);
        if (event.route === "needs_human_review") setFrozen(true);
      }, delay);
    },
    [events, frozen, looping, reduced],
  );

  useEffect(() => {
    if (live) {
      setIndex(events.length - 1);
      const last = events[events.length - 1];
      setFrozen(last?.route === "needs_human_review");
      return;
    }
    schedule(index + 1);
    return clear;
  }, [index, schedule, live, events.length]);

  const resume = () => {
    setFrozen(false);
  };

  return { current: index >= 0 ? events[index] : undefined, index, frozen, resume, setFrozen };
}
