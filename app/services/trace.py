"""Per-message execution record, shared by both agent variants.

Message text is never stored here: only IDs, tool arguments, sizes and counts.
"""

import json
import sys
from dataclasses import asdict, dataclass, field

from agents import AgentsException, RunResult


@dataclass
class ToolCallRecord:
    name: str
    arguments: str
    result_chars: int = 0


@dataclass
class RoutingTrace:
    variant: str
    message_id: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    model_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    decision_attempts: int = 0
    validation_errors: list[str] = field(default_factory=list)
    baseline_evidence_ids: list[str] = field(default_factory=list)
    tool_evidence_ids: list[str] = field(default_factory=list)
    decided_by: str = ""
    final_decision: dict | None = None
    error: str | None = None

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)

    def add_run(self, result: RunResult) -> None:
        """Fold one Runner.run into the trace: usage plus tool calls/results."""
        usage = result.context_wrapper.usage
        self.model_requests += usage.requests
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

        pending: dict[str, ToolCallRecord] = {}
        for item in result.new_items:
            if item.type == "tool_call_item":
                raw = item.raw_item
                record = ToolCallRecord(
                    name=getattr(raw, "name", "?"),
                    arguments=getattr(raw, "arguments", ""),
                )
                self.tool_calls.append(record)
                pending[getattr(raw, "call_id", "")] = record
            elif item.type == "tool_call_output_item":
                call_id = item.raw_item.get("call_id", "") if isinstance(item.raw_item, dict) else ""
                record = pending.get(call_id)
                if record is not None:
                    record.result_chars = len(str(item.output))

    def add_failed_run(self, exc: Exception) -> None:
        """Count the requests a failed run already made, when the SDK exposes them."""
        run_data = getattr(exc, "run_data", None) if isinstance(exc, AgentsException) else None
        if run_data is not None:
            usage = run_data.context_wrapper.usage
            self.model_requests += usage.requests
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens

    def attach_tool_evidence(self, evidence_ids: list[str]) -> None:
        """IDs come from the tool layer, not the model, so they are exactly what was retrieved."""
        self.tool_evidence_ids = list(dict.fromkeys(evidence_ids))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["used_tools"] = self.used_tools
        return data

    def emit(self) -> None:
        print(f"[agent-trace] {json.dumps(self.to_dict())}", file=sys.stderr)
