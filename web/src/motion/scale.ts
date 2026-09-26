/**
 * RepoIntel named motion scale (M1–M10).
 * Every animation in the product must be one of these names.
 * If a new effect does not fit a name, it does not belong.
 */
export const Motion = {
  /** M1 Ink Rise — content settling onto paper. Scroll reveals, section entries. */
  inkRise: {
    name: "Ink Rise",
    id: "M1",
    easing: "cubic-bezier(.19,1,.22,1)",
    durationMs: 800,
    durationRange: [700, 900] as const,
  },
  /** M2 Pin Snap — a pin dropping onto a board. Chip/panel expansion, button press. */
  pinSnap: {
    name: "Pin Snap",
    id: "M2",
    easing: "cubic-bezier(.34,1.56,.64,1)",
    durationMs: 320,
    durationRange: [220, 320] as const,
  },
  /** M3 Signal Pulse — a heartbeat, not a blink. Live status, in-progress. */
  signalPulse: {
    name: "Signal Pulse",
    id: "M3",
    easing: "ease-in-out",
    durationMs: 2000,
    durationRange: [1600, 2200] as const,
  },
  /** M4 Trace Draw — ink following a pen. Pipeline edges, route decisions. */
  traceDraw: {
    name: "Trace Draw",
    id: "M4",
    easing: "linear",
    durationMs: 600,
    durationRange: [400, 900] as const,
  },
  /** M5 Odometer Roll — mechanical counter. Telemetry digits. */
  odometerRoll: {
    name: "Odometer Roll",
    id: "M5",
    easing: "cubic-bezier(.19,1,.22,1)",
    durationMs: 350,
  },
  /** M6 Verdict Morph — a stamp landing. APPROVE / REVISE badge changes. */
  verdictMorph: {
    name: "Verdict Morph",
    id: "M6",
    durationMs: 450,
  },
  /** M7 Grain Drift — atmosphere, not motion. Hero background only; one per page. */
  grainDrift: {
    name: "Grain Drift",
    id: "M7",
    easing: "linear",
    durationMs: 40000,
  },
  /** M8 Page Turn — leaving one page, opening another. */
  pageTurn: {
    name: "Page Turn",
    id: "M8",
    easing: "cubic-bezier(.19,1,.22,1)",
    outMs: 380,
    inMs: 420,
    overlapMs: 80,
  },
  /** M9 Hatch Freeze — a machine pausing mid-motion. Human-review interrupt. */
  hatchFreeze: {
    name: "Hatch Freeze",
    id: "M9",
    durationMs: 500,
  },
  /** M10 Word Bloom — text arriving in reading chunks, never per-character. */
  wordBloom: {
    name: "Word Bloom",
    id: "M10",
    easing: "cubic-bezier(.19,1,.22,1)",
    durationMs: 220,
    staggerMs: 40,
  },
} as const;

export type MotionId = (typeof Motion)[keyof typeof Motion]["id"];
