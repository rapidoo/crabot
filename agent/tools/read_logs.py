"""Read logs tool — lets the agent inspect its own execution traces for self-correction."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)


class ReadLogsTool:
    """Read and filter the agent's own execution traces for self-diagnosis."""

    def __init__(self, trace_file: str = "./logs/agent_trace.jsonl"):
        self._trace_file = Path(trace_file)

    @property
    def name(self) -> str:
        return "read_logs"

    @property
    def description(self) -> str:
        return (
            "Read the agent's own execution logs for self-diagnosis. "
            "Input: last:N (last N entries), errors (entries with errors), "
            "low_scores (entries with scores < 6.5), tool:<name> (filter by tool), "
            "summary (aggregate stats)."
        )

    async def run(self, input: str) -> str:
        """Query the agent's trace logs."""
        cmd = input.strip().lower()

        if not self._trace_file.exists():
            return "No trace logs found."

        traces = self._load_traces()
        if not traces:
            return "Trace log is empty."

        if cmd.startswith("last:"):
            return self._last_n(traces, cmd)
        elif cmd == "errors":
            return self._filter_errors(traces)
        elif cmd == "low_scores":
            return self._filter_low_scores(traces)
        elif cmd.startswith("tool:"):
            return self._filter_by_tool(traces, cmd[5:].strip())
        elif cmd == "summary":
            return self._summary(traces)
        else:
            return self._last_n(traces, "last:5")

    def _load_traces(self) -> list[dict]:
        """Load all trace entries."""
        traces = []
        for line in self._trace_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return traces

    def _last_n(self, traces: list[dict], cmd: str) -> str:
        """Return the last N trace entries with details."""
        try:
            n = int(cmd.split(":")[1])
        except (IndexError, ValueError):
            n = 5
        n = min(n, 20)
        recent = traces[-n:]
        return self._format_entries(recent, f"Last {len(recent)} episodes")

    def _filter_errors(self, traces: list[dict]) -> str:
        """Return entries that contain errors."""
        errors = [t for t in traces if t.get("has_errors")]
        if not errors:
            return "No errors found in recent traces."
        return self._format_entries(errors[-10:], f"{len(errors)} episodes with errors (showing last 10)")

    def _filter_low_scores(self, traces: list[dict]) -> str:
        """Return entries with low scores."""
        low = [t for t in traces if t.get("has_low_scores") or
               any(s < 6.5 for s in t.get("scores", []))]
        if not low:
            return "No low-score episodes found."
        return self._format_entries(low[-10:], f"{len(low)} low-score episodes (showing last 10)")

    def _filter_by_tool(self, traces: list[dict], tool_name: str) -> str:
        """Return entries that used a specific tool."""
        matches = []
        for t in traces:
            for step in t.get("step_details", []):
                if step.get("tool") == tool_name:
                    matches.append(t)
                    break
        if not matches:
            return f"No episodes found using tool '{tool_name}'."
        return self._format_entries(matches[-10:], f"{len(matches)} episodes using '{tool_name}' (showing last 10)")

    def _summary(self, traces: list[dict]) -> str:
        """Aggregate stats for self-diagnosis."""
        total = len(traces)
        all_scores = [s for t in traces for s in t.get("scores", [])]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        error_count = sum(1 for t in traces if t.get("has_errors"))
        low_count = sum(1 for t in traces if t.get("has_low_scores"))

        # Tool usage stats
        tool_counts: dict[str, int] = {}
        tool_errors: dict[str, int] = {}
        for t in traces:
            for step in t.get("step_details", []):
                tool = step.get("tool", "none")
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
                if "error" in step:
                    tool_errors[tool] = tool_errors.get(tool, 0) + 1

        # Timing stats
        timings: dict[str, list[float]] = {}
        for t in traces:
            for phase, dur in t.get("timings", {}).items():
                timings.setdefault(phase, []).append(dur)

        lines = [
            f"=== Agent Self-Diagnosis ({total} episodes) ===",
            f"Avg score: {avg_score:.1f}/10",
            f"Episodes with errors: {error_count}/{total} ({error_count/total*100:.0f}%)" if total else "",
            f"Episodes with low scores: {low_count}/{total}",
            "",
            "Tool usage:",
        ]
        for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            err = tool_errors.get(tool, 0)
            err_str = f" ({err} errors)" if err else ""
            lines.append(f"  {tool}: {count} uses{err_str}")

        if timings:
            lines.append("\nAvg phase timings:")
            for phase, durs in sorted(timings.items()):
                avg_dur = sum(durs) / len(durs)
                lines.append(f"  {phase}: {avg_dur:.1f}s")

        # Recent errors detail
        recent_errors = [t for t in traces[-20:] if t.get("has_errors")]
        if recent_errors:
            lines.append(f"\nRecent errors ({len(recent_errors)} in last 20 episodes):")
            for t in recent_errors[-3:]:
                for step in t.get("step_details", []):
                    if "error" in step:
                        lines.append(f"  [{step['tool']}] {step['error'][:150]}")

        return "\n".join(lines)

    def _format_entries(self, entries: list[dict], header: str) -> str:
        """Format trace entries into readable text."""
        lines = [f"=== {header} ===\n"]
        for t in entries:
            ts = t.get("timestamp", "?")
            goal = t.get("goal", t.get("input", "?"))[:80]
            scores = t.get("scores", [])
            avg = sum(scores) / len(scores) if scores else 0
            elapsed = t.get("elapsed_s", 0)
            path = t.get("path", "?")
            flags = []
            if t.get("has_errors"):
                flags.append("ERRORS")
            if t.get("has_low_scores"):
                flags.append("LOW_SCORES")
            flag_str = f" [{', '.join(flags)}]" if flags else ""

            lines.append(f"[{ts}] {path} | {avg:.1f}/10 | {elapsed}s{flag_str}")
            lines.append(f"  Goal: {goal}")

            # Step details
            for step in t.get("step_details", []):
                score = step.get("score", 0)
                icon = "✓" if score >= 6.5 else "✗"
                tool = step.get("tool", "?")
                lines.append(f"  {icon} Step {step.get('step_id', '?')} ({tool}) [{score:.0f}/10]")
                # Show score breakdown
                breakdown = step.get("scores_breakdown", {})
                if breakdown:
                    parts = [f"{k}:{v:.0f}" for k, v in breakdown.items()]
                    lines.append(f"    Scores: {', '.join(parts)}")
                if "error" in step:
                    lines.append(f"    ERROR: {step['error'][:200]}")
                if step.get("reason"):
                    lines.append(f"    Reason: {step['reason'][:150]}")
            # Timings
            timings = t.get("timings", {})
            if timings:
                parts = [f"{k}:{v:.1f}s" for k, v in timings.items()]
                lines.append(f"  Timings: {', '.join(parts)}")
            lines.append("")
        return "\n".join(lines).strip()


def _factory() -> ReadLogsTool:
    try:
        from agent.config import get_settings
        trace_file = get_settings().logging.trace_file
    except Exception:
        trace_file = "./logs/agent_trace.jsonl"
    return ReadLogsTool(trace_file=trace_file)


register_tool("read_logs", _factory)
