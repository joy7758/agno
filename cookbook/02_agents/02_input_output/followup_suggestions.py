"""
Follow-Up Suggestions
=============================

Get a response AND follow-up suggestions in one call using structured output.

The agent answers a question and generates contextual follow-up suggestions
based on its own response — great for building chatbots with "suggested next
questions" or conversational UIs.

Key concepts:
- output_schema with a Pydantic model that includes both response and suggestions
- The agent sees its own answer while generating suggestions
- Access typed fields via response.content

Example prompts to try:
- "Which national park is the best?"
- "What programming language should I learn first?"
- "How do I start investing?"
"""

from typing import List

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------------------------
class FollowUpSuggestion(BaseModel):
    """A single follow-up suggestion."""

    title: str = Field(..., description="Short action-oriented suggestion (5-10 words)")
    reason: str = Field(
        ..., description="One sentence explaining why this is a good follow-up"
    )


class ResponseWithSuggestions(BaseModel):
    """Agent response with follow-up suggestions."""

    response: str = Field(
        ...,
        description="The main answer to the user's question, detailed and helpful",
    )
    follow_up_suggestions: List[FollowUpSuggestion] = Field(
        ...,
        description="3 follow-up suggestions based on the response",
        min_length=3,
        max_length=3,
    )


# ---------------------------------------------------------------------------
# Agent Instructions
# ---------------------------------------------------------------------------
instructions = """\
You are a knowledgeable assistant that provides helpful answers and suggests
natural follow-up topics.

## Response Guidelines

1. Answer the user's question thoroughly but concisely.
2. Generate exactly 3 follow-up suggestions based on YOUR response.

## Follow-Up Suggestion Rules

- Each suggestion should be a natural next question or action the user might take.
- Suggestions should cover different angles:
  - One that digs deeper into the main topic
  - One that explores a practical next step
  - One that offers an alternative perspective or comparison
- Keep suggestion titles short and action-oriented.\
"""

# ---------------------------------------------------------------------------
# Create the Agent
# ---------------------------------------------------------------------------
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=instructions,
    output_schema=ResponseWithSuggestions,
    markdown=True,
)


# ---------------------------------------------------------------------------
# Run the Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    response = agent.run("Which national park is the best?")

    result: ResponseWithSuggestions = response.content

    print(f"\n{'=' * 60}")
    print("Response:")
    print(f"{'=' * 60}")
    print(result.response)

    print(f"\n{'=' * 60}")
    print("Follow-Up Suggestions:")
    print(f"{'=' * 60}")
    for i, suggestion in enumerate(result.follow_up_suggestions, 1):
        print(f"\n  {i}. {suggestion.title}")
        print(f"     {suggestion.reason}")

    print()
