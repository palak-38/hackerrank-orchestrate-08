import argparse
import asyncio

from notification_router.agentic.services.agent_routing import route_message
from notification_router.context_builder import ContextBuilder
from notification_router.data_store import DataStore, connect
from notification_router.media_analysis import load_cached


async def main():
    parser = argparse.ArgumentParser(
        description="Route one message with an agent variant."
    )

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

    decision, trace = await route_message(
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
