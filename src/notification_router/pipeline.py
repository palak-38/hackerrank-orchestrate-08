from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from notification_router.config import CACHE_DIR, CALL_DELAY_SECONDS, PROMPT_VERSION, ROUTING_CACHE
from notification_router.context_builder import ContextBuilder
from notification_router.data_store import DataStore
from notification_router.deciders import Decider
from notification_router.routing import route
from notification_router.schemas import RoutingDecision


def _safe(text: str) -> str:
    """Windows consoles default to cp1252, and message text carries characters it
    cannot encode (fullwidth punctuation in scam messages, emoji). Printing those
    raised UnicodeEncodeError from inside the error handler itself, hiding the real
    failure, so anything printed is made safe for the active console encoding."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(text).encode(encoding, errors="replace").decode(encoding, errors="replace")


def load_cache(decider: str, path: Path = ROUTING_CACHE) -> dict[str, RoutingDecision]:
    """Only decisions from this decider under the current prompt version are reused;
    anything else is ignored, so switching --decider never replays the other one."""
    if not path.exists():
        return {}
    out: dict[str, RoutingDecision] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("prompt_version") != PROMPT_VERSION:
            continue
        decision = RoutingDecision.model_validate(record)
        if decision.decider != decider:
            continue
        out[decision.message_id] = decision
    return out


def append_cache(decision: RoutingDecision, path: Path = ROUTING_CACHE) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision.model_dump(mode="json"), ensure_ascii=False) + "\n")


def run(
    store: DataStore,
    builder: ContextBuilder,
    decider: Decider,
    table: str = "messages",
    limit: int | None = None,
    cache_path: Path = ROUTING_CACHE,
) -> dict[str, RoutingDecision]:
    """Sequential by design: rate limits are real, and a paused run must be able to
    resume from exactly where it stopped."""
    messages = store.incoming_messages(table)
    if limit:
        messages = messages[:limit]

    done = load_cache(decider.name, cache_path)
    pending = [m for m in messages if m.message_id not in done]
    print(f"{len(messages)} messages, {len(done)} already cached, {len(pending)} to process")

    for i, msg in enumerate(pending, 1):
        try:
            decision, _ = route(msg, builder, decider)
        except Exception as exc:  # noqa: BLE001 - stop, do not fabricate a decision
            print(_safe(f"\nSTOPPED at {msg.message_id} ({i}/{len(pending)}): {type(exc).__name__}: {exc}"))
            print(f"{len(done)} decisions are safe in {cache_path}. Re-run to resume from here.")
            sys.exit(1)

        append_cache(decision, cache_path)
        done[msg.message_id] = decision
        print(_safe(f"[{i}/{len(pending)}] {msg.message_id} -> {decision.action.value}/{decision.message_type.value} ({decision.decided_by})"))
        if i < len(pending) and CALL_DELAY_SECONDS:
            time.sleep(CALL_DELAY_SECONDS)

    return done
