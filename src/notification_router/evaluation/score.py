"""Scores routed sample_messages.csv against the labels shipped with the dataset.

Usage:
    notification-router evaluate [--decider agent]   # routes the samples first if needed
"""

from __future__ import annotations

import csv
from collections import Counter

from notification_router.config import DATASET_DIR
from notification_router.pipeline import load_cache


def main(decider: str = "single") -> None:
    truth = {r["message_id"]: r for r in csv.DictReader((DATASET_DIR / "sample_messages.csv").open(encoding="utf-8"))}
    predictions = load_cache(decider)
    scored = [(mid, row) for mid, row in truth.items() if mid in predictions]

    if not scored:
        print("No cached predictions for sample_messages. Run:")
        print(f"  notification-router route --table sample_messages --decider {decider}")
        return

    action_hits = 0
    type_hits = 0
    both_hits = 0
    evidence_hits = 0
    evidence_total = 0
    confusion: Counter[tuple[str, str]] = Counter()
    misses: list[str] = []

    for mid, row in scored:
        pred = predictions[mid]
        action_ok = pred.action.value == row["action"]
        type_ok = pred.message_type.value == row["message_type"]
        action_hits += action_ok
        type_hits += type_ok
        both_hits += action_ok and type_ok
        if not action_ok:
            confusion[(row["action"], pred.action.value)] += 1
            misses.append(
                f"  {mid}: expected {row['action']}/{row['message_type']}, "
                f"got {pred.action.value}/{pred.message_type.value} [{pred.decided_by}]\n"
                f"      reason: {pred.reason}"
            )

        expected_ev = {e.strip() for e in row["evidence_message_ids"].split(";") if e.strip() and e.strip() != "none"}
        if expected_ev:
            evidence_total += 1
            got_ev = {e.strip() for e in pred.evidence_message_ids.split(";") if e.strip() != "none"}
            if expected_ev & got_ev:
                evidence_hits += 1

    n = len(scored)
    print(f"scored {n} sample messages\n")
    print(f"action accuracy:       {action_hits}/{n} = {action_hits / n:.1%}")
    print(f"message_type accuracy: {type_hits}/{n} = {type_hits / n:.1%}")
    print(f"both correct:          {both_hits}/{n} = {both_hits / n:.1%}")
    if evidence_total:
        print(f"evidence overlap:      {evidence_hits}/{evidence_total} = {evidence_hits / evidence_total:.1%}")

    confidences = [predictions[mid].confidence for mid, _ in scored]
    print(f"confidence: min {min(confidences):.2f}, mean {sum(confidences) / n:.2f}, max {max(confidences):.2f}")

    if confusion:
        print("\naction confusion (expected -> predicted):")
        for (exp, got), count in confusion.most_common():
            print(f"  {exp} -> {got}: {count}")
    if misses:
        print("\nmisses:")
        print("\n".join(misses))


if __name__ == "__main__":
    main()
