import { useEffect, useMemo, useRef, useState } from "react";
import type { PipelineEvent, PipelineNodeId, ReviewerVerdict } from "../types/pipeline";
import { VerdictBadge } from "./VerdictBadge";
import { Odometer } from "./Odometer";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";

const NODES: { id: PipelineNodeId; label: string; x: number; y: number; w: number; h: number }[] = [
  { id: "planner", label: "Planner", x: 24, y: 78, w: 108, h: 36 },
  { id: "retriever", label: "Retriever", x: 168, y: 78, w: 108, h: 36 },
  { id: "synthesis", label: "Synthesis", x: 352, y: 46, w: 112, h: 36 },
  { id: "reviewer", label: "Reviewer", x: 496, y: 46, w: 112, h: 36 },
  { id: "finalize", label: "Finalize", x: 680, y: 46, w: 108, h: 36 },
  { id: "needs_human_review", label: "Human review", x: 668, y: 132, w: 132, h: 36 },
];

const EDGES: { from: PipelineNodeId; to: PipelineNodeId; d: string }[] = [
  { from: "planner", to: "retriever", d: "M132 96 H168" },
  { from: "retriever", to: "synthesis", d: "M276 96 H352" },
  { from: "synthesis", to: "reviewer", d: "M464 64 H496" },
  { from: "reviewer", to: "synthesis", d: "M496 90 C 470 118, 430 118, 408 82" },
  { from: "reviewer", to: "finalize", d: "M608 64 H680" },
  { from: "reviewer", to: "needs_human_review", d: "M552 82 V150 H668" },
];

function edgeKey(from: PipelineNodeId, to: PipelineNodeId) {
  return `${from}->${to}`;
}

function traceMs(length: number) {
  return Math.min(900, Math.max(400, length * 4.2));
}

type Props = {
  /** Latest stream event — same shape for live WebSocket and scripted demo. */
  event?: PipelineEvent;
  /** M9 Hatch Freeze — halt in-flight Trace Draw / dash marching in place. */
  frozen?: boolean;
  simulated?: boolean;
};

export function PipelineDiagram({ event, frozen = false, simulated = false }: Props) {
  const reduced = usePrefersReducedMotion();
  const [active, setActive] = useState<PipelineNodeId | null>(null);
  const [traced, setTraced] = useState<Set<string>>(new Set());
  const [currentEdge, setCurrentEdge] = useState<string | null>(null);
  const [verdict, setVerdict] = useState<ReviewerVerdict | undefined>();
  const [iteration, setIteration] = useState(0);
  const [live, setLive] = useState("");
  const drawRefs = useRef<Record<string, SVGPathElement | null>>({});
  const drawTimer = useRef<number | null>(null);
  const eventKey = event?.id ?? "";

  const nodeMap = useMemo(() => Object.fromEntries(NODES.map((n) => [n.id, n])), []);

  useEffect(() => {
    const paths = drawRefs.current;
    const resetDraw = (key?: string) => {
      const targets = key ? [paths[key]] : Object.values(paths);
      targets.forEach((path) => {
        if (!path) return;
        path.style.transition = "none";
        path.style.strokeDasharray = "";
        path.style.strokeDashoffset = "";
      });
    };

    if (!event) {
      setActive(null);
      setTraced(new Set());
      setCurrentEdge(null);
      setVerdict(undefined);
      setIteration(0);
      setLive("");
      resetDraw();
      return;
    }

    const prev = event.previousNode;
    const next = event.node;
    const ek = prev ? edgeKey(prev, next) : null;
    setLive(event.liveDescription);
    // Node Wake (fill/stroke 300ms) — previous node falls back via CSS.
    setActive(next);
    if (event.verdict) setVerdict(event.verdict);
    if (typeof event.iteration === "number") setIteration(event.iteration);

    if (!ek) {
      setCurrentEdge(null);
      return;
    }

    setCurrentEdge(ek);
    const path = paths[ek];
    if (reduced || !path) {
      setTraced((s) => new Set(s).add(ek));
      setCurrentEdge(null);
      return;
    }

    const len = path.getTotalLength();
    const ms = traceMs(len);
    path.style.transition = "none";
    path.style.strokeDasharray = `${len}`;
    path.style.strokeDashoffset = `${len}`;
    path.getBoundingClientRect();
    path.style.transition = `stroke-dashoffset ${ms}ms linear`;
    path.style.strokeDashoffset = "0";

    drawTimer.current = window.setTimeout(() => {
      setTraced((s) => new Set(s).add(ek));
      setCurrentEdge(null);
      resetDraw(ek);
    }, ms);

    return () => {
      if (drawTimer.current != null) window.clearTimeout(drawTimer.current);
    };
  }, [eventKey, event, reduced]);

  useEffect(() => {
    if (!frozen) return;
    if (drawTimer.current != null) {
      window.clearTimeout(drawTimer.current);
      drawTimer.current = null;
    }
    Object.values(drawRefs.current).forEach((path) => {
      if (!path) return;
      const computed = getComputedStyle(path).strokeDashoffset;
      path.style.transition = "none";
      path.style.strokeDashoffset = computed;
    });
  }, [frozen]);

  return (
    <figure className="pipeline" data-motion="M4">
      <svg
        viewBox="0 0 820 200"
        role="img"
        aria-label="LangGraph pipeline: Planner, Retriever, reflection subgraph, Finalize"
      >
        <defs>
          <pattern
            id="hatch"
            width="8"
            height="8"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <line x1="0" y1="0" x2="0" y2="8" stroke="#2B2828" strokeWidth="1" />
          </pattern>
        </defs>

        <rect className="subgraph-box" x="332" y="18" width="292" height="108" rx="3" />
        <text className="subgraph-caption" x="348" y="34">
          reflection subgraph
        </text>
        <text className="subgraph-caption" x="348" y="118">
          iter
        </text>
        <foreignObject x="378" y="104" width="36" height="16">
          <div>
            <Odometer value={String(iteration)} />
          </div>
        </foreignObject>

        {EDGES.map((e) => {
          const key = edgeKey(e.from, e.to);
          const isCurrent = currentEdge === key && !frozen;
          const isTraced = traced.has(key);
          return (
            <g key={key}>
              <path
                className={`edge-path${isTraced ? " is-traced" : ""}${frozen ? " is-frozen" : ""}`}
                d={e.d}
              />
              <path
                ref={(el) => {
                  drawRefs.current[key] = el;
                }}
                className={`edge-draw${isCurrent || (frozen && currentEdge === key) ? " is-drawing" : ""}`}
                d={e.d}
              />
              {isCurrent ? <path className="edge-march" d={e.d} /> : null}
            </g>
          );
        })}

        {NODES.map((n) => (
          <g key={n.id}>
            <rect
              className={`pipeline-node${active === n.id ? " is-active" : ""}`}
              x={n.x}
              y={n.y}
              width={n.w}
              height={n.h}
              rx="3"
            />
            <text className="pipeline-label" x={n.x + n.w / 2} y={n.y + n.h / 2 + 1}>
              {n.label}
            </text>
          </g>
        ))}

        {verdict ? (
          <foreignObject x={nodeMap.reviewer.x + 8} y={nodeMap.reviewer.y - 28} width="160" height="28">
            <div>
              <VerdictBadge verdict={verdict} />
            </div>
          </foreignObject>
        ) : null}

        <rect
          className={`hatch-overlay${frozen ? " is-on" : ""}`}
          x="0"
          y="0"
          width="820"
          height="200"
          fill="url(#hatch)"
        />
      </svg>
      <div className="pipeline-live" aria-live="polite">
        {live}
      </div>
      {simulated ? (
        <figcaption className="sim-caption">simulated run — see it live in the app</figcaption>
      ) : null}
    </figure>
  );
}
