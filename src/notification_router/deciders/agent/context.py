from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from notification_router.agent_tools import EvidenceTools
    from notification_router.schemas import MessageRoutingContext


@dataclass
class NotificationContext:
    routing_context: "MessageRoutingContext"
    evidence_tools: "EvidenceTools"
