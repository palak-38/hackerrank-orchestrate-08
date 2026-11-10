from pydantic import Field

from notification_router.schemas import LLMRoutingDecision


class AgentRoutingDecision(LLMRoutingDecision):
    evidence_message_ids: list[str] = Field(
        ...,
        max_length=3,
        description=(
            "Zero to three relevant historical message IDs "
            "supporting the routing decision."
        ),
    )
