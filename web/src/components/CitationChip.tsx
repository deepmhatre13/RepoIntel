import {
  createContext,
  useCallback,
  useContext,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { Citation } from "../types/pipeline";

const OpenCtx = createContext<{
  openId: string | null;
  setOpenId: (id: string | null) => void;
}>({ openId: null, setOpenId: () => {} });

export function CitationScope({ children }: { children: ReactNode }) {
  const [openId, setOpenId] = useState<string | null>(null);
  return <OpenCtx.Provider value={{ openId, setOpenId }}>{children}</OpenCtx.Provider>;
}

function highlight(code: string) {
  const escaped = code
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return escaped
    .replace(/(#.*)$/gm, '<span class="tok-cmt">$1</span>')
    .replace(/("[^"]*"|'[^']*')/g, '<span class="tok-str">$1</span>')
    .replace(
      /\b(class|def|return|from|import|if|else|True|False|None|dependencies)\b/g,
      '<span class="tok-kw">$1</span>',
    );
}

type Props = {
  citation: Citation;
  onExpand?: (open: boolean) => void;
};

/** Feature 3 — M2 Pin Snap on the adjacent snippet panel. */
export function CitationChip({ citation, onExpand }: Props) {
  const id = useId();
  const { openId, setOpenId } = useContext(OpenCtx);
  const open = openId === id;
  const [bounce, setBounce] = useState(false);
  const [height, setHeight] = useState(0);
  const hoverTimer = useRef<number | null>(null);
  const measure = useRef<HTMLDivElement>(null);
  const alert = Boolean(citation.insufficientEvidence);
  const label = alert
    ? "no evidence"
    : `${citation.file}:${citation.lineStart}–${citation.lineEnd}`;

  useLayoutEffect(() => {
    if (measure.current) setHeight(measure.current.scrollHeight);
  }, [open, citation]);

  const setOpen = useCallback(
    (next: boolean) => {
      setBounce(true);
      window.setTimeout(() => setBounce(false), 150);
      setOpenId(next ? id : null);
      onExpand?.(next);
    },
    [id, onExpand, setOpenId],
  );

  return (
    <div className="citation-wrap">
      <button
        type="button"
        className={`citation-chip${alert ? " is-alert" : ""}${bounce ? " is-pressed" : ""}`}
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        onMouseEnter={() => {
          hoverTimer.current = window.setTimeout(() => {
            if (!open) setOpen(true);
          }, 400);
        }}
        onMouseLeave={() => {
          if (hoverTimer.current) window.clearTimeout(hoverTimer.current);
        }}
      >
        {label}
      </button>
      <div
        className={`snippet-panel${open ? " is-open" : ""}`}
        style={{ ["--snippet-height" as string]: `${height}px` }}
        data-motion="M2"
      >
        <div className="snippet-inner" ref={measure}>
          {alert ? (
            <div>{citation.message ?? "No evidence found."}</div>
          ) : (
            <>
              <div className="snippet-path">
                {citation.file}:{citation.lineStart}–{citation.lineEnd}
              </div>
              <pre dangerouslySetInnerHTML={{ __html: highlight(citation.content) }} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
