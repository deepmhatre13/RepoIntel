import { useEffect, useRef, useState } from "react";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";

type Props = { value: string; fadeSeconds?: boolean };

/** M5 Odometer Roll — only digits that changed animate. */
export function Odometer({ value, fadeSeconds = false }: Props) {
  const reduced = usePrefersReducedMotion();
  const prev = useRef(value);
  const [rolling, setRolling] = useState<string | null>(null);

  useEffect(() => {
    if (fadeSeconds || value === prev.current) return;
    if (reduced) {
      prev.current = value;
      setRolling(null);
      return;
    }
    setRolling(prev.current);
    const t = window.setTimeout(() => {
      prev.current = value;
      setRolling(null);
    }, 350);
    return () => window.clearTimeout(t);
  }, [value, reduced, fadeSeconds]);

  if (fadeSeconds) {
    return (
      <span className="odometer elapsed-tick" aria-label={value} key={value}>
        {value}
      </span>
    );
  }

  const from = rolling ?? value;
  const chars = value.split("");
  const fromChars = from.padStart(chars.length, " ").split("");

  return (
    <span className="odometer" aria-label={value}>
      {chars.map((ch, i) => {
        const oldCh = fromChars[i] ?? " ";
        const changed = rolling != null && oldCh !== ch && /\d/.test(ch);
        return (
          <span className="odometer-digit" key={i}>
            {changed ? (
              <>
                <span className="out" aria-hidden>
                  {oldCh}
                </span>
                <span className="in">{ch}</span>
              </>
            ) : (
              ch
            )}
          </span>
        );
      })}
    </span>
  );
}
