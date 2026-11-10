from __future__ import annotations

import re

from notification_router.schemas import GateResult, MessageRoutingContext, SafetySignals

# Each pattern must be specific enough that a legitimate sender would not match it.
# The gate forces `mute`, so false positives suppress real messages — precision over recall.
SCAM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "otp_solicitation",
        re.compile(
            r"\b(otp|one[- ]time password|(login|verification|security|access)\s+code|\d\s*digit\s+code)\b",
            re.I,
        ),
    ),
    (
        # "Your OTP leaked / account at risk — verify at <domain>" never comes from a
        # real provider; genuine security alerts do not route users through a link.
        "credential_alert_with_link",
        re.compile(
            r"\b(otp|password|credential|account|profile)\b.{0,60}\b(leak|compromis|breach|misuse|at risk|failed|expire)\w*\b"
            r".{0,90}\b(verify|confirm|update|login|log in|restore|reactivate)\b",
            re.I | re.S,
        ),
    ),
    (
        "kyc_credential_request",
        re.compile(r"\bkyc\b.{0,80}\b(link|verify|confirm|update|complete)\b", re.I | re.S),
    ),
    (
        "card_pin_request",
        re.compile(r"\b(card number|cvv|pin|bank details|account number)\b.{0,80}\b(share|send|confirm|enter|provide)\b", re.I | re.S),
    ),
    (
        "credential_request_reversed",
        re.compile(r"\b(share|send|confirm|enter|provide)\b.{0,80}\b(card number|cvv|pin|bank details|account number)\b", re.I | re.S),
    ),
    (
        "pay_to_restore_service",
        re.compile(r"\b(pay|payment|charge|fee|dues?)\b.{0,90}\b(link|url)\b.{0,90}\b(stop|block|suspend|disconnect|expire|today|immediately|within)\b", re.I | re.S),
    ),
    (
        "account_blocked_threat",
        re.compile(r"\b(account|wallet|sim|number)\b.{0,50}\b(block|suspend|deactivat|clos)\w*\b.{0,90}\b(verify|link|update|confirm)\b", re.I | re.S),
    ),
    (
        "prize_claim",
        re.compile(r"\b(won|winner|lottery|lucky draw|prize)\b.{0,90}\b(claim|link|fee|charge|deposit)\b", re.I | re.S),
    ),
    (
        # Message text trying to command the router itself. Legitimate senders never
        # address the notification system, so this is treated as hostile by construction.
        "prompt_injection",
        re.compile(
            r"(ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+\w*\s*(rules?|instructions?|prompts?)"
            r"|(system\s+note|assistant\s+instruction|routing\s+override|instruction\s+to\s+(the\s+)?(router|assistant|model))"
            r"|\b(set|mark|classify)\b[^.]{0,40}\b(action|confidence|as\s+(notify|urgent|digest))\b"
            r"|\bconfidence\s*=\s*[\d.]+|\baction\s*=\s*(notify|digest|mute)\b)",
            re.I | re.S,
        ),
    ),
]

# OTP alone is the single strongest signal in this dataset, but "your OTP is 1234"
# from a real service is informational, not a solicitation. Require an ask.
OTP_ASK = re.compile(r"\b(enter|share|send|confirm|provide|tell|give|reply with|respond with|forward)\b", re.I)


def _scan_text(text: str) -> list[str]:
    matched: list[str] = []
    for name, pattern in SCAM_PATTERNS:
        if not pattern.search(text):
            continue
        if name == "otp_solicitation" and not OTP_ASK.search(text):
            continue
        matched.append(name)
    return matched


def evaluate(ctx: MessageRoutingContext) -> SafetySignals:
    signals = SafetySignals()
    text = ctx.message.message_text or ""
    if ctx.media_analysis:
        text = f"{text}\n{ctx.media_analysis.transcript}\n{ctx.media_analysis.description}"

    signals.matched_patterns = _scan_text(text)
    signals.scam_text_pattern = bool(signals.matched_patterns)
    signals.heavily_forwarded = ctx.message.forwarded_count >= 5

    if ctx.business:
        p = ctx.business.profile
        official = p.official_domain.strip()
        used = p.domain_used_by_sender.strip()
        signals.business_unverified = not p.verified
        signals.domain_mismatch = bool(official) and bool(used) and official != used
        signals.sender_domain_is_new = 0 < p.domain_used_by_sender_age_days <= 60
        signals.high_report_count = p.user_reports_30d >= 30

    signals.prior_reported_by_user = any(c.was_reported for c in ctx.evidence.candidates)
    return signals


# The scam check sorts every rule into one of two tiers. A rule moves between tiers
# by moving its name between these lists; order sets which reason is reported first.
#
# certain: muted outright, and the message never reaches the LLM. prompt_injection
# must stay here - "asking the LLM to confirm" is exactly what the injection wants.
CERTAIN_RULES = [
    "lookalike_domain",
    "prompt_injection",
    "otp_solicitation",
    "card_pin_request",
    "credential_request_reversed",
]
# suspicious: the LLM sees the message with these reasons attached and confirms or
# rejects the scam call, so a legitimate sender that trips one is not silently lost.
SUSPICIOUS_RULES = [
    "credential_alert_with_link",
    "kyc_credential_request",
    "pay_to_restore_service",
    "account_blocked_threat",
    "prize_claim",
    "reported_new_domain",
]

REASONS = {
    "lookalike_domain": "Unverified sender using a lookalike domain instead of the brand's official domain.",
    "prompt_injection": "Message tries to instruct the notification router itself, which no legitimate sender does.",
    "otp_solicitation": "Message asks the user to share or enter a one-time code, a hallmark of credential theft.",
    "card_pin_request": "Message asks the user to share card, PIN or bank details.",
    "credential_request_reversed": "Message asks the user to share card, PIN or bank details.",
    "credential_alert_with_link": "Message claims an account or credential problem and pushes the user to verify through a link.",
    "kyc_credential_request": "Message requests KYC verification through a link, a common account-takeover scam.",
    "pay_to_restore_service": "Message pressures an urgent payment through a link to avoid service loss.",
    "account_blocked_threat": "Message threatens account suspension unless the user verifies through a link.",
    "prize_claim": "Message claims a prize and asks for a fee or link click.",
    "reported_new_domain": "Heavily reported sender operating from a newly registered domain.",
}


def _fired(signals: SafetySignals) -> set[str]:
    fired = set(signals.matched_patterns)
    if signals.domain_mismatch and signals.business_unverified:
        fired.add("lookalike_domain")
    if signals.high_report_count and signals.sender_domain_is_new:
        fired.add("reported_new_domain")
    return fired


def assess(signals: SafetySignals) -> GateResult:
    """Tier the message: certain (mute, no LLM), suspicious (LLM confirms), or clear."""
    fired = _fired(signals)
    for tier, rules in (("certain", CERTAIN_RULES), ("suspicious", SUSPICIOUS_RULES)):
        hits = [r for r in rules if r in fired]
        if hits:
            reasons = list(dict.fromkeys(REASONS[r] for r in hits))
            return GateResult(tier=tier, rules=hits, reasons=reasons)
    return GateResult()
