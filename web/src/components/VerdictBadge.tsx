import { useEffect, useRef, useState } from "react";
import type { ReviewerVerdict } from "../types/pipeline";
import { SignalPulse } from "./SignalPulse";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";

const LABELS: Record<ReviewerVerdict, string> = {
  approve: "Approve",
  revise: "Revise",
  needs_human_review: "Needs review",
};

type Props = { verdict: ReviewerVerdict };

function Icon({ verdict }: { verdict: ReviewerVerdict }) {
  if (verdict === "approve") {
    return (
      <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
        <path className="icon-stroke" d="M2 6.2 L4.6 9 L10 3.2" />
      </svg>
    );
  }
  if (verdict === "revise") {
    return (
      <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
        <path className="icon-stroke" d="M3 10 V2 L9 4.2 L3 6.4" />
      </svg>
    );
  }
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
      <circle className="icon-stroke" cx="6" cy="4" r="2" />
      <path className="icon-stroke" d="M3.2 10 C3.2 8 8.8 8 8.8 10" />
    </svg>
  );
}

/** Feature 5 — M6 Verdict Morph on change; M3 pulse only while awaiting a human. */
export function VerdictBadge({ verdict }: Props) {
  const reduced = usePrefersReducedMotion();
  const first = useRef(true);
  const prev = useRef(verdict);
  const [morph, setMorph] = useState(false);

  useEffect(() => {
    if (first.current) {
      first.current = false;
      prev.current = verdict;
      return;
    }
    if (prev.current === verdict) return;
    prev.current = verdict;
    if (reduced) return;
    setMorph(true);
    const t = window.setTimeout(() => setMorph(false), 450);
    return () => window.clearTimeout(t);
  }, [verdict, reduced]);

  return (
    <span
      className={`verdict-badge${morph ? " is-morphing" : ""}`}
      data-verdict={verdict}
      data-motion="M6"
    >
      <Icon verdict={verdict} />
      {LABELS[verdict]}
      {verdict === "needs_human_review" ? (
        <SignalPulse title="awaiting human input" />
      ) : null}
    </span>
  );
}
