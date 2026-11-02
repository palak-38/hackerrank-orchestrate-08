from agents import ModelSettings, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI

from app.core.config import settings

client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=settings.groq_api_key.get_secret_value(),
)

model = OpenAIChatCompletionsModel(
    model=settings.agent_model,
    openai_client=client,
)

set_tracing_disabled(True)

# Shared by every agent so an A/B comparison differs only in architecture.
# tool_choice is only sent when the agent actually has tools.
DECISION_SETTINGS = ModelSettings(temperature=0)
TOOL_SETTINGS = ModelSettings(temperature=0, tool_choice="auto")

JSON_CONTRACT = """

Return ONLY a valid JSON object.

Required JSON structure:
{
    "action": "notify",
    "message_type": "payment",
    "reason": "A concise explanation.",
    "confidence": 0.85
}

Allowed actions: notify, digest, mute.

Choose message_type only from:
personal, urgent, event, payment,
business_update, promotion, greeting,
forward, spam, scam, unknown.

confidence must be between 0 and 1.

Do not include markdown, explanations outside
the JSON, or additional fields.
"""
