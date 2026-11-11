"""Unit tests for the agent's corrective retry logic.

No Groq API calls are made.
The SDK Runner is replaced with controlled fake responses.
"""

import json
from types import SimpleNamespace

import pytest

from notification_router.deciders.agent import decider as agent_module
from notification_router.routing import route
from notification_router.schemas import (
    EvidenceCandidate,
    HistoricalEvidence,
    IncomingMessage,
    MessageRoutingContext,
    UserContext,
)

# --------------------------------------------------
# Test data
# --------------------------------------------------

MESSAGE = IncomingMessage(
    message_id="msg_1",
    user_id="user_1",
    conversation_type="personal",
    sender_user_id="user_2",
    created_at="2026-11-10 10:00",
    message_text="Your electricity bill is due tomorrow.",
)


def make_context():
    """Create context with exactly one available evidence ID."""

    return MessageRoutingContext(
        message=MESSAGE,
        user=UserContext(user_id="user_1"),
        evidence=HistoricalEvidence(
            candidates=[
                EvidenceCandidate(
                    message_id="hist_1",
                    message_text="Previous electricity bill reminder.",
                    source="sql_exact",
                    score=0.9,
                )
            ]
        ),
    )


class FakeBuilder:
    """Minimal context builder for testing routing."""

    def __init__(self, context):
        self.context = context
        self.retriever = None
        self.store = None

    def build(self, message):
        return self.context


# --------------------------------------------------
# Fake SDK results
# --------------------------------------------------


def make_json(evidence_ids):
    """Construct an otherwise valid agent response."""

    return json.dumps(
        {
            "action": "notify",
            "message_type": "payment",
            "reason": "The electricity bill requires attention.",
            "confidence": 0.9,
            "evidence_message_ids": evidence_ids,
        }
    )


class FakeRunResult:
    """Only the RunResult attributes our code uses."""

    def __init__(self, output):
        self.final_output = output

        self.context_wrapper = SimpleNamespace(
            usage=SimpleNamespace(
                requests=1,
                input_tokens=100,
                output_tokens=40,
            )
        )

        self.new_items = []

    def to_input_list(self):
        """Provide the previous response for the correction attempt."""

        return [
            {
                "role": "assistant",
                "content": self.final_output,
            }
        ]


def mock_agent(monkeypatch, outputs):
    """Replace agent creation and SDK execution with fake responses."""

    # Avoid constructing a real Groq client.
    monkeypatch.setattr(
        agent_module,
        "build_notification_agent",
        lambda: object(),
    )

    agent = agent_module.AgentDecider()

    results = iter(FakeRunResult(output) for output in outputs)

    calls = []

    def fake_run_sync(
        _agent,
        *,
        input,
        context,
        max_turns,
    ):
        calls.append(input)
        return next(results)

    monkeypatch.setattr(
        agent_module.Runner,
        "run_sync",
        staticmethod(fake_run_sync),
    )

    return agent, calls


# --------------------------------------------------
# Test 1: Invalid -> Valid
# --------------------------------------------------


def test_agent_corrects_invalid_evidence(monkeypatch):
    agent, calls = mock_agent(
        monkeypatch,
        outputs=[
            make_json(["invented_99"]),
            make_json(["hist_1"]),
        ],
    )

    decision, trace = route(
        MESSAGE,
        FakeBuilder(make_context()),
        agent,
    )

    assert len(calls) == 2

    assert decision.evidence_message_ids == "hist_1"

    assert trace.decision_attempts == 2
    assert trace.model_requests == 2

    assert len(trace.validation_errors) == 1

    # Verify the second call includes corrective feedback.
    assert isinstance(calls[1], list)

    correction = calls[1][-1]["content"]

    assert "Unavailable evidence IDs" in correction
    assert "invented_99" in correction


# --------------------------------------------------
# Test 2: Invalid -> Invalid
# --------------------------------------------------


def test_agent_fails_after_two_invalid_responses(monkeypatch):
    agent, calls = mock_agent(
        monkeypatch,
        outputs=[
            make_json(["invented_99"]),
            make_json(["invented_100"]),
        ],
    )

    with pytest.raises(
        ValueError,
        match="after 2 attempts",
    ) as exc_info:
        route(
            MESSAGE,
            FakeBuilder(make_context()),
            agent,
        )

    assert len(calls) == 2

    # routing.py attaches the trace to raised exceptions.
    trace = exc_info.value.routing_trace

    assert trace.decision_attempts == 2
    assert len(trace.validation_errors) == 2

    assert trace.error is not None


# --------------------------------------------------
# Test 3: Valid first attempt
# --------------------------------------------------


def test_valid_evidence_does_not_trigger_retry(monkeypatch):
    agent, calls = mock_agent(
        monkeypatch,
        outputs=[
            make_json(["hist_1"]),
        ],
    )

    decision, trace = route(
        MESSAGE,
        FakeBuilder(make_context()),
        agent,
    )

    assert len(calls) == 1

    assert decision.evidence_message_ids == "hist_1"

    assert trace.decision_attempts == 1
    assert trace.validation_errors == []
