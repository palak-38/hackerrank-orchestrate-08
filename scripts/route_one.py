"""Route one message with a chosen decider and print the decision and its trace.

Usage:
    python scripts/route_one.py --decider agent
    python scripts/route_one.py --decider single --table sample_messages --id sample_msg_013
"""

import argparse

from notification_router.context_builder import ContextBuilder
from notification_router.data_store import DataStore, connect
from notification_router.deciders import DECIDERS, get_decider
from notification_router.media_analysis import load_cached
from notification_router.routing import route


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--decider", default="agent", choices=DECIDERS)
    parser.add_argument("--table", default="messages", choices=["messages", "sample_messages"])
    parser.add_argument("--id", help="message_id to route (default: the first in the table)")
    args = parser.parse_args()

    store = DataStore(connect())
    builder = ContextBuilder(store, load_cached())
    messages = store.incoming_messages(args.table)
    if args.id:
        matches = [m for m in messages if m.message_id == args.id]
        if not matches:
            raise SystemExit(f"{args.id} is not in {args.table}")
        message = matches[0]
    else:
        message = messages[0]

    print("\nIncoming message:")
    print(message.model_dump_json(indent=2))

    decision, trace = route(message, builder, get_decider(args.decider))

    print("\nDecision:")
    print(decision.model_dump_json(indent=2))
    print(
        f"\n{trace.decider}: gate {trace.gate_tier}, {trace.model_requests} model requests, "
        f"{len(trace.tool_calls)} tool calls, {trace.latency_s}s"
    )


if __name__ == "__main__":
    main()
