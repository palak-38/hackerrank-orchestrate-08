"""Deciders: the swappable part of routing.

A decider turns one message's routing context into a decision. Everything around
that - context building, the scam check, evidence validation, provenance, caching -
lives in routing.py and is the same whichever decider runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from notification_router.agent_tools import EvidenceTools
    from notification_router.schemas import LLMRoutingDecision, MessageRoutingContext
    from notification_router.trace import Trace

DECIDERS = ("single", "agent")


@dataclass
class Verdict:
    decision: LLMRoutingDecision
    # Who actually answered, read after the call: failover can change it mid-run.
    provider: str
    model: str
    # None: cite the retrieved evidence. A list (possibly empty) is the decider's
    # own pick, which routing checks against everything the decider was shown.
    evidence_ids: list[str] | None = None


class Decider(Protocol):
    name: str

    def decide(self, ctx: MessageRoutingContext, tools: EvidenceTools, trace: Trace) -> Verdict: ...


def get_decider(name: str) -> Decider:
    """Build a decider by name. Imports are deferred so the single decider never
    loads the Agents SDK or asks for a Groq key."""
    if name == "single":
        from notification_router.deciders.single import SingleCallDecider

        return SingleCallDecider()
    if name == "agent":
        from notification_router.deciders.agent.decider import AgentDecider

        return AgentDecider()
    raise ValueError(f"unknown decider: {name!r} (expected one of {', '.join(DECIDERS)})")
