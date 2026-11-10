"""Per-message execution record, the same shape for every decider.

Message text is never stored here: only IDs, tool arguments, sizes and counts.
"""

import json
import sys
from dataclasses import asdict, dataclass, field


@dataclass
class ToolCallRecord:
    name: str
    arguments: str
    result_chars: int = 0


@dataclass
class Trace:
    decider: str
    message_id: str
    gate_tier: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    model_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    decision_attempts: int = 0
    validation_errors: list[str] = field(default_factory=list)
    baseline_evidence_ids: list[str] = field(default_factory=list)
    tool_evidence_ids: list[str] = field(default_factory=list)
    # Why the decider's own evidence pick was rejected in favour of the retrieved
    # evidence; None when the pick was used or the decider made none.
    evidence_fallback: str | None = None
    decided_by: str = ""
    final_decision: dict | None = None
    error: str | None = None

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["used_tools"] = self.used_tools
        return data

    def emit(self) -> None:
        print(f"[routing-trace] {json.dumps(self.to_dict())}", file=sys.stderr)
