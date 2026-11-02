from code.schemas import LLMRoutingDecision

from pydantic import Field


class AgentRoutingDecision(LLMRoutingDecision):
    evidence_message_ids: list[str] = Field(
        default_factory=list,
        max_length=3,
    )
