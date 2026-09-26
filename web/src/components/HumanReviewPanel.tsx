import { useState } from "react";
import type { ReviewIssue } from "../types/pipeline";
import { SignalPulse } from "./SignalPulse";

type Props = {
  issues: ReviewIssue[];
  onApprove: () => void;
  onRevise: () => void;
  exiting?: boolean;
};

/** Feature 9 — M1 Ink Rise (inverted) on entrance. */
export function HumanReviewPanel({ issues, onApprove, onRevise, exiting }: Props) {
  const [loading, setLoading] = useState<"approve" | "revise" | null>(null);
  const [notes, setNotes] = useState<Record<number, string>>({});

  const submit = (kind: "approve" | "revise") => {
    setLoading(kind);
    window.setTimeout(() => {
      if (kind === "approve") onApprove();
      else onRevise();
    }, 700);
  };

  return (
    <aside className={`human-panel${exiting ? " is-exiting" : ""}`} data-motion="M1">
      <h2>Paused for review</h2>
      <p style={{ color: "var(--ink-soft)", marginTop: 0 }}>
        The machine is waiting. Hatch Freeze is on the diagram until you respond.
      </p>
      {issues.map((issue, i) => (
        <div className="issue-card" key={`${issue.kind}-${i}`}>
          <span className="issue-kind" data-kind={issue.kind}>
            {issue.kind}
          </span>
          <div>{issue.description}</div>
          <textarea
            placeholder="Comment (optional)"
            value={notes[i] ?? ""}
            onChange={(e) => setNotes((n) => ({ ...n, [i]: e.target.value }))}
            aria-label={`Comment on ${issue.kind}`}
          />
        </div>
      ))}
      <div className="issue-actions">
        <button
          type="button"
          className="btn btn-ghost"
          disabled={loading !== null}
          aria-busy={loading === "approve"}
          onClick={() => submit("approve")}
        >
          {loading === "approve" ? <SignalPulse title="submitting" /> : "Approve as-is"}
        </button>
        <button
          type="button"
          className="btn btn-primary"
          disabled={loading !== null}
          aria-busy={loading === "revise"}
          onClick={() => submit("revise")}
        >
          {loading === "revise" ? <SignalPulse title="submitting" /> : "Send back for revision"}
        </button>
      </div>
    </aside>
  );
}
