export type PipelineNodeId =
  | "planner"
  | "retriever"
  | "synthesis"
  | "reviewer"
  | "finalize"
  | "needs_human_review";

export type ReviewerVerdict = "approve" | "revise" | "needs_human_review";

export type IssueKind = "WORDING" | "COVERAGE_GAP" | "GROUNDING_FAILURE";

export type PipelineEvent = {
  id: string;
  /** Active node after this event. */
  node: PipelineNodeId;
  previousNode?: PipelineNodeId;
  verdict?: ReviewerVerdict;
  iteration?: number;
  route?: "finalize" | "needs_human_review" | "synthesis";
  /** Plain-language description for aria-live. */
  liveDescription: string;
  /** Delay after the previous event before this one fires (scripted playback). */
  delayMs?: number;
};

export type Citation = {
  file: string;
  lineStart: number;
  lineEnd: number;
  content: string;
  insufficientEvidence?: boolean;
  message?: string;
};

export type Checkpoint = {
  id: string;
  timestamp: string;
  node: string;
  nodeId: PipelineNodeId;
  verdict?: ReviewerVerdict;
};

export type RubricItem = {
  id: "groundedness" | "completeness" | "structure" | "citations";
  label: string;
  passed?: boolean;
  reason?: string;
};

export type DraftSection = {
  title: string;
  chunks: string[];
  citations: Citation[];
  insufficientEvidence?: boolean;
  complete?: boolean;
};

export type ReviewIssue = {
  kind: IssueKind;
  description: string;
};

export type Archetype = "CLI_TOOL" | "SERVICE" | "LIBRARY" | "MONOREPO";
