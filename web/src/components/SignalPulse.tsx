type Props = { active?: boolean; color?: string; title?: string };

/** M3 Signal Pulse — heartbeat, not a blink. */
export function SignalPulse({ active = true, color, title = "in progress" }: Props) {
  if (!active) return null;
  return (
    <span
      className="signal-pulse"
      style={color ? { background: color } : undefined}
      title={title}
      aria-hidden="true"
    />
  );
}
