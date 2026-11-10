from __future__ import annotations

from notification_router.agent_tools import EvidenceTools
from notification_router.deciders import Verdict
from notification_router.router import LLMRouter
from notification_router.schemas import MessageRoutingContext
from notification_router.trace import Trace


class SingleCallDecider:
    """One structured LLM call per message, with provider failover and the paid-call budget."""

    name = "single"

    def __init__(self, router: LLMRouter | None = None):
        self._router = router or LLMRouter()

    def decide(self, ctx: MessageRoutingContext, tools: EvidenceTools, trace: Trace) -> Verdict:
        decision = self._router.route(ctx)
        trace.decision_attempts += 1
        # instructor's internal re-asks on malformed output are not visible here.
        trace.model_requests += 1
        return Verdict(decision=decision, provider=self._router.provider, model=self._router.model)
