import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useParams } from "react-router-dom";
import { ArchetypeBadge } from "../components/ArchetypeBadge";
import { HumanReviewPanel } from "../components/HumanReviewPanel";
import { PipelineDiagram } from "../components/PipelineDiagram";
import { RubricChecklist } from "../components/RubricChecklist";
import { RunTimeline } from "../components/RunTimeline";
import { StreamingDraft } from "../components/StreamingDraft";
import { TelemetryTicker } from "../components/TelemetryTicker";
import { consumeSharedRun } from "../components/PageTurn";
import {
  DEMO_CHECKPOINTS,
  DEMO_DRAFT,
  DEMO_ISSUES,
  DEMO_RUBRIC,
  DEMO_RUNS,
  LIVE_PIPELINE,
} from "../data/demo";
import { usePipelinePlayback } from "../hooks/usePipelinePlayback";
import type { Checkpoint, PipelineEvent } from "../types/pipeline";

export function LiveRun() {
  const { runId } = useParams();
  const run = DEMO_RUNS.find((r) => r.id === runId) ?? DEMO_RUNS[0];
  const headerRef = useRef<HTMLDivElement>(null);
  const [extra, setExtra] = useState<PipelineEvent | undefined>();
  const [exiting, setExiting] = useState(false);
  const [complete, setComplete] = useState(false);
  const [tokens, setTokens] = useState(1280);
  const [cost, setCost] = useState(0.0142);
  const [elapsed, setElapsed] = useState(4000);
  const playback = usePipelinePlayback({ kind: "scripted", events: LIVE_PIPELINE });
  const event = extra ?? playback.current;

  const checkpoints: Checkpoint[] = useMemo(() => {
    const idx = extra ? DEMO_CHECKPOINTS.length : Math.max(1, playback.index + 1);
    return DEMO_CHECKPOINTS.slice(0, Math.min(idx, DEMO_CHECKPOINTS.length));
  }, [playback.index, extra]);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  useEffect(() => {
    setSelectedId(checkpoints[checkpoints.length - 1]?.id ?? null);
  }, [checkpoints]);

  const human = event?.route === "needs_human_review" || (playback.frozen && !extra);
  const reviewer = event?.node === "reviewer" && !human;
  const synthesizing = event?.node === "synthesis";

  useEffect(() => {
    if (complete) return;
    const id = window.setInterval(() => {
      setElapsed((e) => e + 1000);
      if (!playback.frozen) {
        setTokens((t) => t + 17);
        setCost((c) => c + 0.0003);
      }
    }, 1000);
    return () => window.clearInterval(id);
  }, [complete, playback.frozen]);

  useLayoutEffect(() => {
    const shared = consumeSharedRun();
    const el = headerRef.current;
    if (!shared || !el || shared.id !== run.id) return;
    const next = el.getBoundingClientRect();
    const dx = shared.left - next.left;
    const dy = shared.top - next.top;
    const sx = shared.width / next.width;
    const sy = shared.height / next.height;
    el.animate(
      [
        { transform: `translate(${dx}px, ${dy}px) scale(${sx}, ${sy})`, opacity: 0.92 },
        { transform: "none", opacity: 1 },
      ],
      { duration: 420, easing: "cubic-bezier(.19,1,.22,1)" },
    );
  }, [run.id]);

  const onApprove = () => {
    setExiting(true);
    window.setTimeout(() => {
      setExiting(false);
      playback.setFrozen(false);
      setExtra({
        id: "finalize",
        node: "finalize",
        previousNode: "reviewer",
        route: "finalize",
        verdict: "approve",
        liveDescription: "Human approved as-is. Run finalized.",
      });
      setComplete(true);
    }, 300);
  };

  const onRevise = () => {
    setExiting(true);
    window.setTimeout(() => {
      setExiting(false);
      playback.setFrozen(false);
      setExtra({
        id: "resynth",
        node: "synthesis",
        previousNode: "reviewer",
        iteration: 1,
        liveDescription: "Sent back for revision. Synthesis waking up.",
      });
    }, 300);
  };

  let pane: ReactNode;
  if (human && !extra) {
    pane = (
      <HumanReviewPanel
        issues={DEMO_ISSUES}
        onApprove={onApprove}
        onRevise={onRevise}
        exiting={exiting}
      />
    );
  } else if (reviewer) {
    pane = <RubricChecklist items={DEMO_RUBRIC} />;
  } else if (synthesizing) {
    pane = <StreamingDraft sections={DEMO_DRAFT} />;
  } else {
    pane = (
      <div className="pane-pending">
        <p>Waiting for Synthesis. The diagram is the live moment on this screen.</p>
      </div>
    );
  }

  return (
    <div className="live-run">
      <div className="run-layout">
        <aside>
          <div className="run-header shared-header" ref={headerRef}>
            <div className="run-meta">
              <ArchetypeBadge type={run.archetype} />
              <span>{run.repo}</span>
            </div>
            <h1>Live run</h1>
          </div>
          <RunTimeline
            checkpoints={checkpoints}
            selectedId={selectedId}
            onScrub={setSelectedId}
            live={!complete}
          />
        </aside>
        <div className="run-main">
          <PipelineDiagram event={event} frozen={playback.frozen && !extra} />
        </div>
        <div className="run-pane">{pane}</div>
      </div>
      <TelemetryTicker tokens={tokens} cost={cost} durationMs={elapsed} complete={complete} />
    </div>
  );
}
