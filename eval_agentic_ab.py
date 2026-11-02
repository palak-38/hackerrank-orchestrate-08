"""A/B comparison of the two app/ agent variants on a fixed set of labelled messages.

Both variants route the same messages from dataset/sample_messages.csv with the same
context builder and model settings. Results go to ab_results/<timestamp>/:
records.jsonl (one line per message per variant) and summary.md.

Usage:
    .venv/bin/python eval_agentic_ab.py
    .venv/bin/python eval_agentic_ab.py --ids sample_msg_001,sample_msg_013
"""

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from code.context_builder import ContextBuilder
from code.data_store import DataStore, connect
from code.media_analysis import load_cached

from openai import RateLimitError

from app.services.agent_routing import route_message
from app.services.trace import RoutingTrace
from app.services.two_stage_routing import route_message_two_stage

# Fixed set: 4 notify / 3 digest / 3 mute, spanning group, business and personal
# chats, text, voice and image, single- and multi-evidence and no-evidence labels.
# Scam samples are excluded: the safety gate decides them before any agent runs.
DEFAULT_IDS = [
    "sample_msg_001", "sample_msg_004", "sample_msg_046", "sample_msg_051",  # notify
    "sample_msg_008", "sample_msg_041", "sample_msg_049",                    # digest
    "sample_msg_013", "sample_msg_014", "sample_msg_045",                    # mute
]

VARIANTS = {
    "two_stage": route_message_two_stage,
    "single_agent": route_message,
}

CALL_DELAY_S = 2.0
RATE_LIMIT_WAIT_S = 30.0
RATE_LIMIT_RETRIES = 3


def _ids(field: str) -> list[str]:
    return [] if field.strip() in ("", "none") else [i for i in field.split(";") if i]


def load_gold() -> dict[str, dict]:
    path = ROOT / "dataset" / "sample_messages.csv"
    return {r["message_id"]: r for r in csv.DictReader(path.open(encoding="utf-8"))}


def tool_use_bucket(trace: RoutingTrace, gold_evidence: list[str]) -> str:
    """Was retrieval needed, and did the agent behave accordingly?

    Needed means the labelled evidence was missing from the baseline context.
    """
    if trace.decided_by != "llm":
        return "n/a"
    gold = set(gold_evidence)
    in_baseline = not gold or bool(gold & set(trace.baseline_evidence_ids))
    if not trace.used_tools:
        return "justified_skip" if in_baseline else "bad_skip"
    found = bool(gold & set(trace.tool_evidence_ids))
    return "useful_retrieval" if found else "wasted_retrieval"


async def run_one(route, message, store, builder) -> tuple[dict | None, RoutingTrace | None, str | None]:
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            decision, trace = await route(message, store, builder)
            return decision.model_dump(mode="json"), trace, None
        except RateLimitError:
            if attempt == RATE_LIMIT_RETRIES:
                return None, None, "RateLimitError: retries exhausted"
            await asyncio.sleep(RATE_LIMIT_WAIT_S)
        except Exception as exc:  # recorded as a failure for this variant
            return None, getattr(exc, "routing_trace", None), f"{type(exc).__name__}: {exc}"[:300]
    return None, None, "unreachable"


def score(record: dict, gold: dict) -> dict:
    pred = record["decision"]
    gold_evidence = _ids(gold["evidence_message_ids"])
    if pred is None:
        return {"action_ok": False, "type_ok": False, "evidence_ok": False}
    pred_evidence = _ids(pred["evidence_message_ids"])
    evidence_ok = (
        not pred_evidence if not gold_evidence else bool(set(gold_evidence) & set(pred_evidence))
    )
    return {
        "action_ok": pred["action"] == gold["action"],
        "type_ok": pred["message_type"] == gold["message_type"],
        "evidence_ok": evidence_ok,
    }


def unsupported_ids(record: dict) -> list[str]:
    """Output evidence IDs that came from neither the baseline context nor a tool result."""
    pred, trace = record["decision"], record["trace"]
    if pred is None or trace is None:
        return []
    known = set(trace["baseline_evidence_ids"]) | set(trace["tool_evidence_ids"])
    return [i for i in _ids(pred["evidence_message_ids"]) if i not in known]


def summarise(records: list[dict]) -> str:
    rows = []
    buckets = ["justified_skip", "bad_skip", "useful_retrieval", "wasted_retrieval", "n/a"]
    for variant in VARIANTS:
        rs = [r for r in records if r["variant"] == variant]
        n = len(rs)
        ok = [r for r in rs if r["decision"] is not None]
        llm = [r for r in ok if r["trace"]["decided_by"] == "llm"]

        def pct(k: int, d: int) -> str:
            return f"{k}/{d} ({100 * k / d:.0f}%)" if d else "-"

        def avg(key: str) -> str:
            return f"{sum(r['trace'][key] for r in llm) / len(llm):.1f}" if llm else "-"

        first_invalid = sum(1 for r in rs if r["trace"] and r["trace"]["validation_errors"])
        bucket_counts = {b: sum(1 for r in rs if r.get("tool_bucket") == b) for b in buckets}
        rows.append({
            "variant": variant,
            "Routing accuracy (action)": pct(sum(r["score"]["action_ok"] for r in rs), n),
            "Type accuracy": pct(sum(r["score"]["type_ok"] for r in rs), n),
            "Action + type": pct(sum(r["score"]["action_ok"] and r["score"]["type_ok"] for r in rs), n),
            "Evidence accuracy": pct(sum(r["score"]["evidence_ok"] for r in rs), n),
            "Failed (no decision)": pct(n - len(ok), n),
            "Invalid JSON (any attempt)": pct(first_invalid, n),
            "Avg model requests": avg("model_requests"),
            "Tool-call rate": pct(sum(1 for r in llm if r["trace"]["tool_calls"]), len(llm)),
            "Avg latency (s)": avg("latency_s"),
            "Avg input tokens": avg("input_tokens"),
            "Avg output tokens": avg("output_tokens"),
            "Unsupported evidence IDs": str(sum(len(unsupported_ids(r)) for r in rs)),
            **{f"Tool use: {b}": str(bucket_counts[b]) for b in buckets},
        })

    metrics = [k for k in rows[0] if k != "variant"]
    lines = ["| Metric | " + " | ".join(r["variant"] for r in rows) + " |",
             "|---|" + "---|" * len(rows)]
    lines += [f"| {m} | " + " | ".join(r[m] for r in rows) + " |" for m in metrics]

    lines.append("\nPer message (action/type, tool bucket):\n")
    lines.append("| message_id | gold | " + " | ".join(VARIANTS) + " |")
    lines.append("|---|---|" + "---|" * len(VARIANTS))
    for mid in dict.fromkeys(r["message_id"] for r in records):
        cells = []
        gold = None
        for variant in VARIANTS:
            r = next(x for x in records if x["message_id"] == mid and x["variant"] == variant)
            gold = f"{r['gold']['action']}/{r['gold']['message_type']}"
            if r["decision"] is None:
                cells.append(f"ERROR: {r['error'][:40]}")
            else:
                d = r["decision"]
                cells.append(f"{d['action']}/{d['message_type']} ({r['tool_bucket']})")
        lines.append(f"| {mid} | {gold} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ids", help="comma-separated sample message IDs (default: fixed set of 10)")
    args = parser.parse_args()
    ids = args.ids.split(",") if args.ids else DEFAULT_IDS

    gold = load_gold()
    missing = [i for i in ids if i not in gold]
    if missing:
        raise SystemExit(f"Unknown sample IDs: {missing}")

    store = DataStore(connect())
    builder = ContextBuilder(store, load_cached())
    messages = {m.message_id: m for m in store.incoming_messages("sample_messages")}

    out_dir = ROOT / "ab_results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as fh:
        for mid in ids:
            for variant, route in VARIANTS.items():
                decision, trace, error = await run_one(route, messages[mid], store, builder)
                record = {
                    "message_id": mid,
                    "variant": variant,
                    "gold": {k: gold[mid][k] for k in ("action", "message_type", "evidence_message_ids")},
                    "decision": decision,
                    "trace": trace.to_dict() if trace else None,
                    "error": error,
                    "tool_bucket": tool_use_bucket(trace, _ids(gold[mid]["evidence_message_ids"])) if trace else "error",
                }
                record["score"] = score(record, gold[mid])
                record["unsupported_evidence_ids"] = unsupported_ids(record)
                records.append(record)
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                status = error or f"{decision['action']}/{decision['message_type']}"
                print(f"{mid} {variant:13s} {status}", flush=True)
                await asyncio.sleep(CALL_DELAY_S)

    summary = summarise(records)
    (out_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    print(f"\nWritten to {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    asyncio.run(main())
