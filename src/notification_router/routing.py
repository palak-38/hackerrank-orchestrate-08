"""Routes one message: the single place a RoutingDecision is made.

    context -> scam check -> decider -> evidence check -> RoutingDecision

Only the decider varies. The scam check, the evidence rules, and provenance are
applied here once, so every decider is held to the same contract.
"""

from __future__ import annotations

import time

from notification_router.agent_tools import EvidenceTools
from notification_router.config import (
    GATED_MUTE_CONFIDENCE,
    PROMPT_VERSION,
    TRACE_VERBOSE,
)
from notification_router.context_builder import ContextBuilder
from notification_router.deciders import Decider
from notification_router.evidence import validate_evidence_selection
from notification_router.schemas import (
    Action,
    IncomingMessage,
    MessageRoutingContext,
    MessageType,
    RoutingDecision,
)
from notification_router.trace import Trace


def _cite(
    ctx: MessageRoutingContext,
    tools: EvidenceTools,
    picks: list[str] | None,
) -> str:
    """Resolve evidence citations for the final routing decision.

    None: the decider did not select evidence; use initial retrieval.
    []: the decider explicitly selected no evidence.
    [IDs]: validate and use exactly the selected IDs.
    """
    if picks is None:
        return ctx.evidence.as_csv_field()

    validate_evidence_selection(ctx, tools, picks)

    return ";".join(picks) if picks else "none"


def route(
    message: IncomingMessage, builder: ContextBuilder, decider: Decider
) -> tuple[RoutingDecision, Trace]:
    trace = Trace(decider=decider.name, message_id=message.message_id)
    started = time.perf_counter()
    try:
        ctx = builder.build(message)
        trace.gate_tier = ctx.gate.tier
        trace.baseline_evidence_ids = ctx.evidence.message_ids()

        if ctx.gate.tier == "certain":
            # The one decision the LLM never sees: muted on the scam check alone.
            decision = RoutingDecision(
                message_id=message.message_id,
                action=Action.mute,
                message_type=MessageType.scam,
                reason=ctx.gate.reasons[0],
                confidence=GATED_MUTE_CONFIDENCE,
                evidence_message_ids="none",
                decided_by="safety_gate",
                prompt_version=PROMPT_VERSION,
                decider=decider.name,
            )
        else:
            # A suspicious message reaches the decider with the scam check's reasons
            # in its prompt, and the decider confirms or rejects them.
            tools = EvidenceTools(builder.retriever, builder.store, message)
            verdict = decider.decide(ctx, tools, trace)
            trace.tool_evidence_ids = list(dict.fromkeys(c.message_id for c in tools.found))
            decision = RoutingDecision(
                message_id=message.message_id,
                action=verdict.decision.action,
                message_type=verdict.decision.message_type,
                reason=verdict.decision.reason,
                confidence=verdict.decision.confidence,
                evidence_message_ids=_cite(ctx, tools, verdict.evidence_ids),
                decided_by="llm",
                prompt_version=PROMPT_VERSION,
                provider=verdict.provider,
                model=verdict.model,
                decider=decider.name,
            )

        trace.decided_by = decision.decided_by
        trace.final_decision = {
            "action": decision.action.value,
            "message_type": decision.message_type.value,
            "confidence": decision.confidence,
            "evidence_message_ids": decision.evidence_message_ids,
        }
        return decision, trace

    except Exception as exc:
        trace.error = f"{type(exc).__name__}: {exc}"[:300]
        exc.routing_trace = trace  # lets evaluation inspect failed runs
        raise
    finally:
        trace.latency_s = round(time.perf_counter() - started, 3)
        if TRACE_VERBOSE:
            trace.emit()
