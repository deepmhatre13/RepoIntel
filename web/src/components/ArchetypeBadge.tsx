import type { Archetype } from "../types/pipeline";

type Props = { type: Archetype };

/** Feature 11 — deliberately static. Quiet label next to animated diagram. */
export function ArchetypeBadge({ type }: Props) {
  return (
    <span className="archetype-badge" title="Repository archetype">
      {type.replaceAll("_", " ")}
    </span>
  );
}
