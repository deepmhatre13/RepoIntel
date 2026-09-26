import type {
  Archetype,
  Checkpoint,
  Citation,
  DraftSection,
  PipelineEvent,
  ReviewIssue,
  RubricItem,
} from "../types/pipeline";

/** Marketing 7-step loop — captioned as simulated; same event shape as a live run. */
export const MARKETING_PIPELINE: PipelineEvent[] = [
  {
    id: "m1",
    node: "planner",
    liveDescription: "Planner started the investigation.",
    delayMs: 400,
  },
  {
    id: "m2",
    node: "retriever",
    previousNode: "planner",
    liveDescription: "Retriever began gathering evidence.",
    delayMs: 900,
  },
  {
    id: "m3",
    node: "synthesis",
    previousNode: "retriever",
    liveDescription: "Synthesis started drafting from citations.",
    delayMs: 900,
  },
  {
    id: "m4",
    node: "reviewer",
    previousNode: "synthesis",
    verdict: "revise",
    liveDescription: "Reviewer requested a synthesis revision.",
    delayMs: 1100,
  },
  {
    id: "m5",
    node: "synthesis",
    previousNode: "reviewer",
    iteration: 1,
    liveDescription: "Reflection iteration 1. Synthesis redrafting.",
    delayMs: 900,
  },
  {
    id: "m6",
    node: "reviewer",
    previousNode: "synthesis",
    verdict: "approve",
    liveDescription: "Reviewer approved the draft.",
    delayMs: 1100,
  },
  {
    id: "m7",
    node: "finalize",
    previousNode: "reviewer",
    route: "finalize",
    liveDescription: "Run finalized.",
    delayMs: 900,
  },
];

export const LIVE_PIPELINE: PipelineEvent[] = [
  {
    id: "l1",
    node: "planner",
    liveDescription: "Planner started the investigation.",
    delayMs: 300,
  },
  {
    id: "l2",
    node: "retriever",
    previousNode: "planner",
    liveDescription: "Retriever began gathering evidence.",
    delayMs: 800,
  },
  {
    id: "l3",
    node: "synthesis",
    previousNode: "retriever",
    liveDescription: "Synthesis started drafting from citations.",
    delayMs: 800,
  },
  {
    id: "l4",
    node: "reviewer",
    previousNode: "synthesis",
    verdict: "needs_human_review",
    liveDescription: "Reviewer flagged the draft for a human.",
    delayMs: 1200,
  },
  {
    id: "l5",
    node: "needs_human_review",
    previousNode: "reviewer",
    route: "needs_human_review",
    verdict: "needs_human_review",
    liveDescription: "Run paused for human review.",
    delayMs: 800,
  },
];

export const DEMO_CITATIONS: Record<string, Citation> = {
  pyproject: {
    file: "pyproject.toml",
    lineStart: 12,
    lineEnd: 18,
    content: `dependencies = [
    "fastapi>=0.110.0",
    "pydantic>=2.6.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.13.0",
    "langchain-google-genai>=1.0.0",
]`,
  },
  schema: {
    file: "packages/agentic/state/schema.py",
    lineStart: 63,
    lineEnd: 67,
    content: `class ReviewVerdict(str, Enum):
    APPROVE = "APPROVE"
    REVISE_SYNTHESIS = "REVISE_SYNTHESIS"
    REVISE_PLAN = "REVISE_PLAN"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"`,
  },
  none: {
    file: "unknown",
    lineStart: 0,
    lineEnd: 0,
    content: "",
    insufficientEvidence: true,
    message: "No evidence found for this claim. The indexer did not retrieve a supporting chunk.",
  },
};

export const DEMO_DRAFT: DraftSection[] = [
  {
    title: "Architecture",
    chunks: [
      "RepoIntel ",
      "ingests a repository, ",
      "parses it statically, ",
      "and answers with ",
      "citations ",
      "into the analyzed code. ",
      "The public boundary is ",
      "RepositoryFacade.",
    ],
    citations: [DEMO_CITATIONS.pyproject],
    complete: true,
  },
  {
    title: "Reviewer routing",
    chunks: [
      "The LangGraph ",
      "reflection loop ",
      "routes on ",
      "Reviewer verdicts ",
      "including ",
      "NEEDS_HUMAN_REVIEW.",
    ],
    citations: [DEMO_CITATIONS.schema],
    complete: true,
  },
  {
    title: "Runtime execution traces",
    chunks: [],
    citations: [DEMO_CITATIONS.none],
    insufficientEvidence: true,
    complete: true,
  },
];

export const DEMO_RUBRIC: RubricItem[] = [
  { id: "groundedness", label: "Groundedness", passed: true },
  { id: "completeness", label: "Completeness", passed: false, reason: "Runtime traces section has no supporting chunks." },
  { id: "structure", label: "Structural quality", passed: true },
  { id: "citations", label: "Citation coverage", passed: true },
];

export const DEMO_ISSUES: ReviewIssue[] = [
  {
    kind: "COVERAGE_GAP",
    description: "The runtime-execution section is marked insufficient_evidence and should not be presented as a finding.",
  },
  {
    kind: "WORDING",
    description: "“answers with citations” overstates completeness when one section has no evidence.",
  },
];

export const DEMO_CHECKPOINTS: Checkpoint[] = [
  { id: "c1", timestamp: "00:01", node: "Planner", nodeId: "planner" },
  { id: "c2", timestamp: "00:04", node: "Retriever", nodeId: "retriever" },
  { id: "c3", timestamp: "00:09", node: "Synthesis", nodeId: "synthesis" },
  {
    id: "c4",
    timestamp: "00:14",
    node: "Reviewer",
    nodeId: "reviewer",
    verdict: "needs_human_review",
  },
  {
    id: "c5",
    timestamp: "00:15",
    node: "Human review",
    nodeId: "needs_human_review",
    verdict: "needs_human_review",
  },
];

export type DemoRun = {
  id: string;
  repo: string;
  archetype: Archetype;
  status: "READY" | "RUNNING" | "NEEDS_HUMAN_REVIEW";
  lastVerdict?: "approve" | "revise" | "needs_human_review";
};

export const DEMO_RUNS: DemoRun[] = [
  {
    id: "run_repointel_readme",
    repo: "github.com/you/repointel",
    archetype: "CLI_TOOL",
    status: "NEEDS_HUMAN_REVIEW",
    lastVerdict: "needs_human_review",
  },
  {
    id: "run_fastapi_arch",
    repo: "github.com/encode/starlette",
    archetype: "LIBRARY",
    status: "READY",
    lastVerdict: "approve",
  },
  {
    id: "run_svc_impact",
    repo: "github.com/example/payments-api",
    archetype: "SERVICE",
    status: "READY",
    lastVerdict: "revise",
  },
];
