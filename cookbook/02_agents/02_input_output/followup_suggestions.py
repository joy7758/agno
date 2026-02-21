"""
Follow-Up Suggestions
=============================

Get a full free-form response AND structured follow-up suggestions
from a single agent using two calls.

Call 1: The agent responds naturally — full markdown, streaming, no
        truncation, just a normal response.
Call 2: Pass output_schema on the run to get structured suggestions.
        The agent sees its own previous answer via history and generates
        follow-ups based on it.

Key concepts:
- add_history_to_context: the agent remembers its previous response
- output_schema passed per-run: only the second call is structured
- The first call is never constrained by any schema

Example prompts to try:
- "Which national park is the best?"
- "What programming language should I learn first?"
- "How do I start investing?"
"""

from typing import List

from agno.agent import Agent, RunOutput
from agno.models.openai import OpenAIResponses
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured Output Schema (suggestions only — no need to duplicate response)
# ---------------------------------------------------------------------------
class FollowUpSuggestion(BaseModel):
    """A single follow-up suggestion."""

    title: str = Field(..., description="Short action-oriented suggestion (5-10 words)")
    reason: str = Field(
        ..., description="One sentence explaining why this is a good follow-up"
    )


class FollowUpSuggestions(BaseModel):
    """Follow-up suggestions based on the previous response."""

    suggestions: List[FollowUpSuggestion] = Field(
        ...,
        description="3 follow-up suggestions based on the previous response",
        min_length=3,
        max_length=3,
    )


# ---------------------------------------------------------------------------
# Create the Agent
# ---------------------------------------------------------------------------
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="""\
You are a knowledgeable assistant. Answer questions thoroughly.

When asked to suggest follow-ups, generate suggestions that cover
different angles:
- One that digs deeper into the main topic
- One that explores a practical next step
- One that offers an alternative perspective or comparison\
""",
    # History lets the second call see the first response
    add_history_to_context=True,
    num_history_runs=1,
    markdown=True,
)


# ---------------------------------------------------------------------------
# Run the Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Call 1: Free-form response — stream it, no schema, no truncation
    response: RunOutput = agent.run("Which national park is the best?")

    print(f"\n{'=' * 60}")
    print("Response:")
    print(f"{'=' * 60}")
    print(response.content)

    # Call 2: Structured suggestions — agent sees its own answer via history
    suggestion_run: RunOutput = agent.run(
        "Based on your previous response, suggest 3 follow-ups.",
        output_schema=FollowUpSuggestions,
    )
    suggestions: FollowUpSuggestions = suggestion_run.content

    print(f"\n{'=' * 60}")
    print("Follow-Up Suggestions:")
    print(f"{'=' * 60}")
    for i, suggestion in enumerate(suggestions.suggestions, 1):
        print(f"\n  {i}. {suggestion.title}")
        print(f"     {suggestion.reason}")

    print()
