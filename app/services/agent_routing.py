import time
from code.context_builder import ContextBuilder
from code.data_store import DataStore
from code.router import build_user_prompt
from code.schemas import (
    IncomingMessage,
    RoutingDecision,
)

from agents import Runner, RunResult
from pydantic import ValidationError

from app.agents.decision_schema import AgentRoutingDecision
from app.agents.notification_agent import notification_agent
from app.core.config import settings
from app.services.routing_common import (
    finish,
    prepare,
    record_final,
    validate_evidence_selection,
)
from app.services.trace import RoutingTrace

# One initial decision plus one corrective retry. Only malformed output is
# retried: provider errors (HTTP 4xx/5xx, ModelBehaviorError, MaxTurnsExceeded)
# propagate, since another attempt would not fix configuration problems.
MAX_DECISION_ATTEMPTS = 2


def _validate(result: RunResult) -> AgentRoutingDecision:
    raw_output = result.final_output

    if not isinstance(raw_output, str):
        raw_output = "" if raw_output is None else str(raw_output)

    return AgentRoutingDecision.model_validate_json(raw_output.strip())


def _summarise_errors(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(map(str, err['loc'])) or 'output'}: {err['msg']}"
        for err in exc.errors()[:5]
    )


async def route_message(
    message: IncomingMessage,
    store: DataStore,
    builder: ContextBuilder,
) -> tuple[RoutingDecision, RoutingTrace]:
    trace = RoutingTrace(variant="single_agent", message_id=message.message_id)
    started = time.perf_counter()
    try:
        context, agent_context, gated = prepare(message, store, builder, trace)
        if gated is not None:
            record_final(trace, gated)
            return gated, trace

        run_input: str | list = build_user_prompt(context)
        decision: AgentRoutingDecision | None = None

        while decision is None:
            trace.decision_attempts += 1
            result = await Runner.run(
                notification_agent,
                input=run_input,
                context=agent_context,
                max_turns=settings.agent_max_turns,
            )
            trace.add_run(result)

            try:
                candidate = _validate(result)

                validate_evidence_selection(
                    message,
                    context,
                    agent_context,
                    candidate,
                )

                decision = candidate

            except (ValidationError, ValueError) as exc:
                errors = (
                    _summarise_errors(exc)
                    if isinstance(exc, ValidationError)
                    else str(exc)
                )

                trace.validation_errors.append(errors)

                if trace.decision_attempts >= MAX_DECISION_ATTEMPTS:
                    raise ValueError(
                        f"Invalid decision after {MAX_DECISION_ATTEMPTS} attempts: {errors}"
                    ) from exc

                run_input = result.to_input_list() + [
                    {
                        "role": "user",
                        "content": (
                            f"Your previous decision was invalid: {errors}. "
                            "Correct the JSON. Select only evidence IDs "
                            "present in the supplied history or tool results."
                        ),
                    }
                ]
        print("\n=== EVIDENCE SELECTION ===")

        print("Initially retrieved:", context.evidence.message_ids())

        print(
            "Additional retrieval:",
            [c.message_id for c in agent_context.retrieved_evidence],
        )

        print("Agent selected:", decision.evidence_message_ids)
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
