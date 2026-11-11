from __future__ import annotations

from agents import AgentsException, Runner, RunResult
from pydantic import ValidationError

from notification_router.agent_tools import EvidenceTools
from notification_router.config import AGENT_MAX_TURNS, AGENT_MODEL
from notification_router.deciders import Verdict
from notification_router.deciders.agent.context import NotificationContext
from notification_router.deciders.agent.decision_schema import AgentRoutingDecision
from notification_router.deciders.agent.notification_agent import build_notification_agent
from notification_router.evidence import (
    InvalidEvidenceSelection,
    validate_evidence_selection,
)
from notification_router.router import build_user_prompt
from notification_router.schemas import MessageRoutingContext
from notification_router.trace import ToolCallRecord, Trace

# One initial decision plus one corrective retry. Only malformed output is
# retried: provider errors (HTTP 4xx/5xx, ModelBehaviorError, MaxTurnsExceeded)
# propagate, since another attempt would not fix configuration problems.
MAX_DECISION_ATTEMPTS = 2

# AGENT_BASE_URL is Groq's endpoint, and there is no failover on this decider.
PROVIDER = "groq"


def _summarise_errors(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(map(str, err['loc'])) or 'output'}: {err['msg']}" for err in exc.errors()[:5]
    )


def _fold_usage(trace: Trace, usage) -> None:
    trace.model_requests += usage.requests
    trace.input_tokens += usage.input_tokens
    trace.output_tokens += usage.output_tokens


def _fold_run(trace: Trace, result: RunResult) -> None:
    """Fold one run into the trace: usage plus tool calls and result sizes."""
    _fold_usage(trace, result.context_wrapper.usage)

    pending: dict[str, ToolCallRecord] = {}
    for item in result.new_items:
        if item.type == "tool_call_item":
            raw = item.raw_item
            record = ToolCallRecord(
                name=getattr(raw, "name", "?"),
                arguments=getattr(raw, "arguments", ""),
            )
            trace.tool_calls.append(record)
            pending[getattr(raw, "call_id", "")] = record
        elif item.type == "tool_call_output_item":
            call_id = item.raw_item.get("call_id", "") if isinstance(item.raw_item, dict) else ""
            record = pending.get(call_id)
            if record is not None:
                record.result_chars = len(str(item.output))


class AgentDecider:
    """OpenAI Agents SDK tool loop: the model may search the user's history before
    deciding, and picks the evidence IDs it cites."""

    name = "agent"

    def __init__(self):
        self._agent = build_notification_agent()

    def decide(self, ctx: MessageRoutingContext, tools: EvidenceTools, trace: Trace) -> Verdict:
        context = NotificationContext(routing_context=ctx, evidence_tools=tools)
        run_input: str | list = build_user_prompt(ctx)

        for attempt in range(1, MAX_DECISION_ATTEMPTS + 1):
            trace.decision_attempts += 1
            try:
                result = Runner.run_sync(
                    self._agent, input=run_input, context=context, max_turns=AGENT_MAX_TURNS
                )
            except AgentsException as exc:
                # Count the requests a failed run already made, when the SDK exposes them.
                run_data = getattr(exc, "run_data", None)
                if run_data is not None:
                    _fold_usage(trace, run_data.context_wrapper.usage)
                raise
            _fold_run(trace, result)

            raw = result.final_output
            raw = "" if raw is None else str(raw)
            try:
                # 1. Validate JSON structure and field types.
                decision = AgentRoutingDecision.model_validate_json(raw.strip())

                # 2. Validate the evidence IDs selected by the agent.
                validate_evidence_selection(
                    ctx,
                    tools,
                    decision.evidence_message_ids,
                )

            except (ValidationError, InvalidEvidenceSelection) as exc:
                if isinstance(exc, ValidationError):
                    errors = _summarise_errors(exc)
                else:
                    errors = str(exc)

                trace.validation_errors.append(errors)

                if attempt == MAX_DECISION_ATTEMPTS:
                    raise ValueError(
                        f"Invalid agent decision after {MAX_DECISION_ATTEMPTS} attempts: {errors}"
                    ) from exc

                # Give the agent its previous response and corrective feedback.
                run_input = result.to_input_list() + [
                    {
                        "role": "user",
                        "content": (
                            f"Your previous decision was invalid: {errors}\n"
                            "Return a complete corrected JSON object.\n"
                            "Select only historical evidence IDs that were "
                            "provided in the context or retrieval tool results.\n"
                            "Do not invent evidence IDs.\n"
                            "If no relevant evidence exists, return []."
                        ),
                    }
                ]

                continue

            return Verdict(
                decision=decision,
                provider=PROVIDER,
                model=AGENT_MODEL,
                evidence_ids=decision.evidence_message_ids,
            )
