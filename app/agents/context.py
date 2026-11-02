from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_tools import EvidenceTools
    from schemas import EvidenceCandidate, MessageRoutingContext


@dataclass
class NotificationContext:
    routing_context: "MessageRoutingContext"
    evidence_tools: "EvidenceTools"
    retrieved_evidence: list["EvidenceCandidate"] = field(default_factory=list)
