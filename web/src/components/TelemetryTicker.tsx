import { useEffect, useRef, useState } from "react";
import { Odometer } from "./Odometer";

type Props = {
  tokens: number;
  cost: number;
  durationMs: number;
  complete?: boolean;
};

function pad2(n: number) {
  return n.toString().padStart(2, "0");
}

/** Feature 4 — M5 Odometer Roll on changed digits. */
export function TelemetryTicker({ tokens, cost, durationMs, complete = false }: Props) {
  const [pulse, setPulse] = useState<"tokens" | "cost" | "time" | null>(null);
  const prev = useRef({ tokens, cost, durationMs });

  useEffect(() => {
    const p = prev.current;
    if (p.tokens !== tokens) setPulse("tokens");
    else if (p.cost !== cost) setPulse("cost");
    else if (Math.floor(p.durationMs / 1000) !== Math.floor(durationMs / 1000)) setPulse("time");
    prev.current = { tokens, cost, durationMs };
    const t = window.setTimeout(() => setPulse(null), 250);
    return () => window.clearTimeout(t);
  }, [tokens, cost, durationMs]);

  const secs = Math.floor(durationMs / 1000);
  const clock = `${pad2(Math.floor(secs / 60))}:${pad2(secs % 60)}`;
  const costText = complete ? cost.toFixed(2) : cost.toFixed(4);

  return (
    <footer className="telemetry" aria-label="Run telemetry">
      <div className={`telemetry-stat${pulse === "tokens" ? " is-updated" : ""}`}>
        <span className="telemetry-label">Tokens</span>
        <span className="telemetry-value">
          <Odometer value={String(tokens)} />
        </span>
      </div>
      <div className={`telemetry-stat${pulse === "cost" ? " is-updated" : ""}`}>
        <span className="telemetry-label">Cost</span>
        <span className="telemetry-value">
          $<Odometer value={costText} />
          {complete ? <span className="final-tag">final</span> : null}
        </span>
      </div>
      <div className={`telemetry-stat${pulse === "time" ? " is-updated" : ""}`}>
        <span className="telemetry-label">Elapsed</span>
        <span className="telemetry-value">
          <Odometer value={clock} fadeSeconds />
        </span>
      </div>
    </footer>
  );
}
