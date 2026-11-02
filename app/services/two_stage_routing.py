"""Two-agent flow (research, then decide), kept temporarily for A/B evaluation.

Decision logic is unchanged from the original: one decision call, no JSON retry.
"""

import time

from code.context_builder import ContextBuilder
from code.data_store import DataStore
from code.router import build_user_prompt
from code.schemas import (
    IncomingMessage,
    LLMRoutingDecision,
    RoutingDecision,
)

from agents import Runner
from pydantic import ValidationError

from app.agents.two_stage_agent import decision_agent, research_agent
from app.core.config import settings
from app.services.routing_common import finish, prepare, record_final
from app.services.trace import RoutingTrace


async def route_message_two_stage(
    message: IncomingMessage,
    store: DataStore,
    builder: ContextBuilder,
) -> tuple[RoutingDecision, RoutingTrace]:
    trace = RoutingTrace(variant="two_stage", message_id=message.message_id)
    started = time.perf_counter()
    try:
        context, agent_context, gated = prepare(message, store, builder, trace)
        if gated is not None:
            record_final(trace, gated)
            return gated, trace

        # Stage 1: research
        research_result = await Runner.run(
            research_agent,
            input=build_user_prompt(context),
            context=agent_context,
            max_turns=settings.agent_max_turns,
        )
        trace.add_run(research_result)
        research_summary = research_result.final_output or ""

        # Stage 2: decision. Include the actual tool results, not just the
        # research agent's interpretation.
        evidence_text = "\n".join(
            (
                f"Message ID: {candidate.message_id}\n"
                f"Message: {candidate.message_text}\n"
                f"Source: {candidate.source}\n"
                f"Match note: {candidate.match_note}"
            )
            for candidate in agent_context.retrieved_evidence[:10]
        )

        decision_input = f"""
        {build_user_prompt(context)}

        === ADDITIONAL RETRIEVED EVIDENCE ===
        {evidence_text or "No additional evidence retrieved."}

        === RESEARCH AGENT FINDINGS ===
        {research_summary}

        Use the supplied context and evidence to make
        the final notification-routing decision.
        """

        trace.decision_attempts += 1
        decision_result = await Runner.run(
            decision_agent,
            input=decision_input,
            max_turns=2,
        )
        trace.add_run(decision_result)

        raw_output = decision_result.final_output
        if not isinstance(raw_output, str):
            raise TypeError(f"Expected JSON text, got {type(raw_output).__name__}")
        try:
            decision = LLMRoutingDecision.model_validate_json(raw_output.strip())
        except ValidationError as exc:
            trace.validation_errors.append(str(exc.errors()[:5]))
            raise ValueError("Decision agent returned invalid routing JSON") from exc

        routed = finish(message, context, agent_context, decision, trace)
        record_final(trace, routed)
        return routed, trace

    except Exception as exc:
        trace.add_failed_run(exc)
        trace.error = f"{type(exc).__name__}: {exc}"[:300]
        exc.routing_trace = trace  # lets evaluation inspect failed runs
        raise
    finally:
        trace.latency_s = round(time.perf_counter() - started, 3)
        if settings.agent_verbose:
            trace.emit()
