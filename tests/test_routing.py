"""The routing module's contract, exercised with a scripted decider: no network, no dataset.

These pin down the behaviour the two old routing flows disagreed on - the scam check,
which evidence gets cited, and what provenance is recorded.
"""

from __future__ import annotations

import json

import pytest

from notification_router.agent_tools import EvidenceTools
from notification_router.config import PROMPT_VERSION
from notification_router.deciders import Verdict
from notification_router.evidence import InvalidEvidenceSelection
from notification_router.pipeline import append_cache, load_cache
from notification_router.router import build_user_prompt
from notification_router.routing import route
from notification_router.safety_gate import assess
from notification_router.schemas import (
    Action,
    EvidenceCandidate,
    GateResult,
    HistoricalEvidence,
    IncomingMessage,
    LLMRoutingDecision,
    MessageRoutingContext,
    MessageType,
    RoutingDecision,
    SafetySignals,
    UserContext,
)
from notification_router.trace import Trace

MESSAGE = IncomingMessage(
    message_id="msg_1",
    user_id="user_1",
    conversation_type="personal",
    sender_user_id="user_2",
    created_at="2026-11-10 10:00",
    message_text="hello",
)


def _ctx(
    gate: GateResult | None = None, evidence: list[str] = ("hist_1",)
) -> MessageRoutingContext:
    return MessageRoutingContext(
        message=MESSAGE,
        user=UserContext(user_id="user_1"),
        evidence=HistoricalEvidence(
            candidates=[
                EvidenceCandidate(message_id=m, source="sql_exact", score=0.9) for m in evidence
            ]
        ),
        gate=gate or GateResult(),
    )


class StubBuilder:
    """Stands in for ContextBuilder: hands back a prepared context."""

    def __init__(self, ctx: MessageRoutingContext):
        self._ctx = ctx
        self.retriever = None
        self.store = None

    def build(self, msg: IncomingMessage) -> MessageRoutingContext:
        return self._ctx


class FakeDecider:
    """Returns a scripted decision, records what it was shown, and can simulate a lookup."""

    name = "fake"

    def __init__(
        self,
        action: Action = Action.digest,
        message_type: MessageType = MessageType.personal,
        evidence_ids: list[str] | None = None,
        lookup: list[str] = (),
    ):
        self.action = action
        self.message_type = message_type
        self.evidence_ids = evidence_ids
        self.lookup = lookup
        self.calls = 0
        self.prompts: list[str] = []

    def decide(self, ctx: MessageRoutingContext, tools: EvidenceTools, trace: Trace) -> Verdict:
        self.calls += 1
        self.prompts.append(build_user_prompt(ctx))
        tools.found.extend(
            EvidenceCandidate(message_id=m, source="vector_semantic", score=0.6)
            for m in self.lookup
        )
        decision = LLMRoutingDecision(
            action=self.action, message_type=self.message_type, reason="scripted", confidence=0.8
        )
        return Verdict(
            decision=decision,
            provider="fake-provider",
            model="fake-model",
            evidence_ids=self.evidence_ids,
        )


# --- scam check ---------------------------------------------------------------


def test_assess_tiers():
    assert assess(SafetySignals()).tier == "clear"
    assert assess(SafetySignals(matched_patterns=["prize_claim"])).tier == "suspicious"
    assert assess(SafetySignals(matched_patterns=["otp_solicitation"])).tier == "certain"
    assert assess(SafetySignals(domain_mismatch=True, business_unverified=True)).tier == "certain"
    assert (
        assess(SafetySignals(high_report_count=True, sender_domain_is_new=True)).tier
        == "suspicious"
    )


def test_assess_certain_outranks_suspicious():
    result = assess(SafetySignals(matched_patterns=["prize_claim", "prompt_injection"]))
    assert result.tier == "certain"
    assert result.rules == ["prompt_injection"]


def test_certain_tier_mutes_without_calling_the_decider():
    gate = GateResult(tier="certain", rules=["otp_solicitation"], reasons=["asks for a code"])
    decider = FakeDecider()

    decision, trace = route(MESSAGE, StubBuilder(_ctx(gate)), decider)

    assert decider.calls == 0
    assert (decision.action, decision.message_type) == (Action.mute, MessageType.scam)
    assert decision.decided_by == "safety_gate"
    assert decision.reason == "asks for a code"
    assert decision.evidence_message_ids == "none"
    assert trace.gate_tier == "certain"


def test_suspicious_tier_warns_the_decider_and_it_can_confirm():
    gate = GateResult(tier="suspicious", rules=["prize_claim"], reasons=["claims a prize"])
    decider = FakeDecider(action=Action.mute, message_type=MessageType.scam)

    decision, _ = route(MESSAGE, StubBuilder(_ctx(gate)), decider)

    assert "=== SCAM CHECK ===" in decider.prompts[0]
    assert "claims a prize" in decider.prompts[0]
    assert (decision.action, decision.message_type) == (Action.mute, MessageType.scam)
    assert decision.decided_by == "llm"


def test_suspicious_tier_rejected_by_the_decider_routes_normally():
    gate = GateResult(tier="suspicious", rules=["prize_claim"], reasons=["claims a prize"])
    decider = FakeDecider(action=Action.digest, message_type=MessageType.promotion)

    decision, _ = route(MESSAGE, StubBuilder(_ctx(gate)), decider)

    assert (decision.action, decision.message_type) == (Action.digest, MessageType.promotion)


def test_clear_message_has_no_scam_check_block():
    decider = FakeDecider()
    route(MESSAGE, StubBuilder(_ctx()), decider)
    assert "SCAM CHECK" not in decider.prompts[0]


# --- evidence -----------------------------------------------------------------


def test_no_pick_cites_retrieved_evidence():
    decision, trace = route(MESSAGE, StubBuilder(_ctx()), FakeDecider(evidence_ids=None))
    assert decision.evidence_message_ids == "hist_1"
    assert trace.evidence_fallback is None


def test_pick_from_a_lookup_is_used():
    decider = FakeDecider(evidence_ids=["hist_9"], lookup=["hist_9"])
    decision, trace = route(MESSAGE, StubBuilder(_ctx()), decider)
    assert decision.evidence_message_ids == "hist_9"
    assert trace.tool_evidence_ids == ["hist_9"]


def test_empty_pick_means_none():
    decision, _ = route(MESSAGE, StubBuilder(_ctx()), FakeDecider(evidence_ids=[]))
    assert decision.evidence_message_ids == "none"


# def test_unseen_pick_falls_back_to_retrieved_evidence():
#     decision, trace = route(MESSAGE, StubBuilder(_ctx()), FakeDecider(evidence_ids=["invented_1"]))
#     assert decision.evidence_message_ids == "hist_1"
#     assert "invented_1" in trace.evidence_fallback


# def test_citing_the_message_itself_falls_back():
#     decider = FakeDecider(evidence_ids=["msg_1"], lookup=["msg_1"])
#     decision, trace = route(MESSAGE, StubBuilder(_ctx()), decider)
#     assert decision.evidence_message_ids == "hist_1"
#     assert trace.evidence_fallback is not None


def test_unseen_evidence_is_rejected():
    decider = FakeDecider(evidence_ids=["invented_1"])

    with pytest.raises(
        InvalidEvidenceSelection,
        match="Unavailable evidence IDs",
    ):
        route(MESSAGE, StubBuilder(_ctx()), decider)


def test_cannot_cite_current_message():
    decider = FakeDecider(
        evidence_ids=["msg_1"],
        lookup=["msg_1"],
    )

    with pytest.raises(InvalidEvidenceSelection):
        route(MESSAGE, StubBuilder(_ctx()), decider)


def test_duplicate_evidence_is_rejected():
    decider = FakeDecider(evidence_ids=["hist_1", "hist_1"])

    with pytest.raises(
        InvalidEvidenceSelection,
        match="Duplicate",
    ):
        route(MESSAGE, StubBuilder(_ctx()), decider)


def test_more_than_three_evidence_ids_are_rejected():
    decider = FakeDecider(evidence_ids=["hist_1", "hist_2", "hist_3", "hist_4"])

    with pytest.raises(
        InvalidEvidenceSelection,
        match="maximum allowed",
    ):
        route(
            MESSAGE,
            StubBuilder(_ctx(evidence=["hist_1", "hist_2", "hist_3", "hist_4"])),
            decider,
        )


# --- provenance and cache -----------------------------------------------------


def test_records_who_answered():
    decision, trace = route(MESSAGE, StubBuilder(_ctx()), FakeDecider())
    assert (decision.provider, decision.model, decision.decider) == (
        "fake-provider",
        "fake-model",
        "fake",
    )
    assert trace.decider == "fake"


def test_cache_only_reuses_the_same_decider(tmp_path):
    cache = tmp_path / "routing_results.jsonl"
    single, _ = route(MESSAGE, StubBuilder(_ctx()), FakeDecider())
    single = single.model_copy(update={"decider": "single"})
    agent = single.model_copy(update={"message_id": "msg_2", "decider": "agent"})
    stale = single.model_copy(update={"message_id": "msg_3", "prompt_version": "v0"})
    for decision in (single, agent, stale):
        append_cache(decision, cache)

    assert set(load_cache("single", cache)) == {"msg_1"}
    assert set(load_cache("agent", cache)) == {"msg_2"}


def test_cache_lines_without_a_decider_count_as_single(tmp_path):
    cache = tmp_path / "routing_results.jsonl"
    legacy = RoutingDecision(
        message_id="msg_1",
        action=Action.digest,
        message_type=MessageType.personal,
        reason="r",
        confidence=0.5,
        prompt_version=PROMPT_VERSION,
    ).model_dump(mode="json")
    del legacy["decider"]
    cache.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    assert set(load_cache("single", cache)) == {"msg_1"}
