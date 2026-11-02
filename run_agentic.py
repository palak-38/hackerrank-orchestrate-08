import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from code.context_builder import ContextBuilder
from code.data_store import DataStore, connect
from code.media_analysis import load_cached

from app.services.agent_routing import route_message
from app.services.two_stage_routing import route_message_two_stage


async def main():
    parser = argparse.ArgumentParser(description="Route one message with an agent variant.")
    parser.add_argument("--two-stage", action="store_true", help="use the research+decision variant")
    args = parser.parse_args()

    store = DataStore(connect())

    builder = ContextBuilder(
        store,
        load_cached(),
    )

    messages = store.incoming_messages("messages")

    # Select the first message for our integration test.
    message = messages[0]

    print("\nIncoming message:")
    print(message.model_dump_json(indent=2))

    route = route_message_two_stage if args.two_stage else route_message
    decision, trace = await route(
        message,
        store,
        builder,
    )

    print("\nAgent decision:")
    print(decision.model_dump_json(indent=2))
    print(
        f"\n{trace.variant}: {trace.model_requests} model requests, "
        f"{len(trace.tool_calls)} tool calls, {trace.latency_s}s"
    )


if __name__ == "__main__":
    asyncio.run(main())
