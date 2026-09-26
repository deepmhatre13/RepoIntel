import { useEffect, useState, type ReactNode } from "react";
import type { DraftSection } from "../types/pipeline";
import { CitationChip, CitationScope } from "./CitationChip";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";

type Props = { sections: DraftSection[] };

function BloomText({
  chunks,
  trailing,
  reduced,
}: {
  chunks: string[];
  trailing?: ReactNode;
  reduced: boolean;
}) {
  const [count, setCount] = useState(reduced ? chunks.length : 0);

  useEffect(() => {
    if (reduced) {
      setCount(chunks.length);
      return;
    }
    setCount(0);
    const timers: number[] = [];
    chunks.forEach((_, i) => {
      timers.push(window.setTimeout(() => setCount(i + 1), 40 * i));
    });
    return () => timers.forEach((t) => window.clearTimeout(t));
  }, [chunks, reduced]);

  return (
    <>
      {chunks.slice(0, count).map((chunk, i) => {
        const last = i === chunks.length - 1;
        return (
          <span
            key={`${i}-${chunk}`}
            className="word-bloom"
            data-motion="M10"
            style={{ animationDelay: "0ms" }}
          >
            {chunk}
            {last ? trailing : null}
          </span>
        );
      })}
    </>
  );
}

/** Feature 8 — M10 Word Bloom by chunk; insufficient_evidence is a flat fade. */
export function StreamingDraft({ sections }: Props) {
  const reduced = usePrefersReducedMotion();

  return (
    <CitationScope>
      <article className="draft-panel">
        {sections.map((section) => (
          <section
            key={section.title}
            className={`draft-section${section.insufficientEvidence ? " is-insufficient" : ""}`}
          >
            <h3>{section.title}</h3>
            {section.insufficientEvidence ? (
              <p>
                Insufficient evidence to generate this section.{" "}
                {section.citations.map((c) => (
                  <CitationChip key={`${c.file}-${c.lineStart}`} citation={c} />
                ))}
              </p>
            ) : (
              <p>
                <BloomText
                  chunks={section.chunks}
                  reduced={reduced}
                  trailing={
                    section.complete
                      ? section.citations.map((c) => (
                          <CitationChip key={`${c.file}-${c.lineStart}`} citation={c} />
                        ))
                      : null
                  }
                />
              </p>
            )}
          </section>
        ))}
      </article>
    </CitationScope>
  );
}
