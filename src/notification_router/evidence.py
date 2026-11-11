from notification_router.agent_tools import EvidenceTools
from notification_router.config import MAX_EVIDENCE
from notification_router.schemas import MessageRoutingContext


class InvalidEvidenceSelection(ValueError):
    """The decider selected evidence that cannot be cited."""


def validate_evidence_selection(
    ctx: MessageRoutingContext,
    tools: EvidenceTools,
    selected_ids: list[str],
) -> None:
    """Validate an explicit evidence selection.

    Selected IDs must be unique, within the output limit,
    and present in evidence made available to the decider.
    """
    if len(selected_ids) > MAX_EVIDENCE:
        raise InvalidEvidenceSelection(
            f"Selected {len(selected_ids)} evidence IDs; maximum allowed is {MAX_EVIDENCE}."
        )

    if len(selected_ids) != len(set(selected_ids)):
        raise InvalidEvidenceSelection("Duplicate evidence IDs are not allowed.")

    available_ids = {
        candidate.message_id
        for candidate in ctx.evidence.candidates + tools.found
        if candidate.message_id != ctx.message.message_id
    }

    invalid_ids = [message_id for message_id in selected_ids if message_id not in available_ids]

    if invalid_ids:
        raise InvalidEvidenceSelection(
            f"Unavailable evidence IDs: {invalid_ids}. "
            "Only cite historical messages shown in the context "
            "or returned by retrieval tools."
        )
