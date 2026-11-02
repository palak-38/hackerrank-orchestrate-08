"""Steps shared by the single-agent and two-stage routing flows."""

from code.agent_tools import EvidenceTools
from code.context_builder import ContextBuilder
from code.data_store import DataStore
from code.safety_gate import should_hard_mute
from code.schemas import (
    Action,
    IncomingMessage,
    MessageRoutingContext,
    MessageType,
    RoutingDecision,
)

from app.agents.context import NotificationContext
from app.agents.decision_schema import AgentRoutingDecision
from app.core.config import settings
from app.services.trace import RoutingTrace


def prepare(
    message: IncomingMessage,
    store: DataStore,
    builder: ContextBuilder,
    trace: RoutingTrace,
) -> tuple[MessageRoutingContext, NotificationContext | None, RoutingDecision | None]:
    """Build context and apply the safety gate.

    Returns a ready decision when the gate fires; otherwise the agent context.
    """
    context = builder.build(message)
    trace.baseline_evidence_ids = context.evidence.message_ids()

    gated, reason = should_hard_mute(context.features.safety)
    if gated:
        trace.decided_by = "safety_gate"
        return (
            context,
            None,
            RoutingDecision(
                message_id=message.message_id,
                action=Action.mute,
                message_type=MessageType.scam,
                reason=reason,
                confidence=0.95,
                evidence_message_ids="none",
                decided_by="safety_gate",
            ),
        )

    agent_context = NotificationContext(
        routing_context=context,
        evidence_tools=EvidenceTools(builder.retriever, store, message),
    )
    return context, agent_context, None


def validate_evidence_selection(
    message: IncomingMessage,
    context: MessageRoutingContext,
    agent_context: NotificationContext,
    decision: AgentRoutingDecision,
) -> None:
    """Ensure selected evidence was actually available to the agent."""

    candidates = context.evidence.candidates + agent_context.retrieved_evidence

    allowed_ids = {
        candidate.message_id
        for candidate in candidates
        if candidate.message_id != message.message_id
    }

    selected_ids = decision.evidence_message_ids

    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Duplicate evidence IDs selected")

    invalid_ids = [mid for mid in selected_ids if mid not in allowed_ids]

    if invalid_ids:
        raise ValueError(f"Agent selected unavailable evidence IDs: {invalid_ids}")


def finish(
    message: IncomingMessage,
    context: MessageRoutingContext,
    agent_context: NotificationContext,
    decision: AgentRoutingDecision,
    trace: RoutingTrace,
) -> RoutingDecision:
    """Merge baseline and retrieved evidence and build the output row."""
    trace.attach_tool_evidence([c.message_id for c in agent_context.retrieved_evidence])

    validate_evidence_selection(
        message,
        context,
        agent_context,
        decision,
    )

    evidence_ids = (
        ";".join(decision.evidence_message_ids)
        if decision.evidence_message_ids
        else "none"
    )

    trace.decided_by = "llm"
    return RoutingDecision(
        message_id=message.message_id,
        action=decision.action,
        message_type=decision.message_type,
        reason=decision.reason,
        confidence=decision.confidence,
        evidence_message_ids=evidence_ids,
        decided_by="llm",
        provider="groq",
        model=settings.agent_model,
    )


def record_final(trace: RoutingTrace, result: RoutingDecision) -> None:
    trace.final_decision = {
        "action": result.action.value,
        "message_type": result.message_type.value,
        "confidence": result.confidence,
        "evidence_message_ids": result.evidence_message_ids,
    }
