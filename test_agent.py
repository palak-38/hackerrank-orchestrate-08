import asyncio
import os
from pathlib import Path

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load .env from the same directory as test_agent.py
ENV_PATH = Path(__file__).resolve().parent / ".env"

load_dotenv(dotenv_path=ENV_PATH)

# Verify that the key loaded
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(f"GROQ_API_KEY not found. Check your .env file at {ENV_PATH}")

client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

model = OpenAIChatCompletionsModel(
    model="openai/gpt-oss-20b",
    openai_client=client,
)

set_tracing_disabled(True)


@function_tool
def get_order_status(order_id: str) -> str:
    """Return the current status of an order."""

    orders = {
        "101": "Shipped",
        "102": "Delayed",
        "103": "Delivered",
    }

    return orders.get(order_id, "Order not found")


agent = Agent(
    name="Test Agent",
    instructions=(
        "You are an order support assistant. "
        "Always use the available tool to verify order status."
    ),
    model=model,
    tools=[get_order_status],
)


async def main():
    result = await Runner.run(
        agent,
        "What is the status of order 102?",
        max_turns=5,
    )

    print("Final answer:", result.final_output)

    print("\nExecution events:")
    for item in result.new_items:
        print(type(item).__name__)


if __name__ == "__main__":
    asyncio.run(main())
