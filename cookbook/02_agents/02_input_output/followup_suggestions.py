"""
Follow-Up Suggestions
=============================

Get a full response AND follow-up suggestions from a single agent.

Uses parser_model so the main model responds freely (full streaming, no
truncation), then a cheap parser model extracts structured suggestions
from that response.

Key concepts:
- output_schema: defines the structured shape (response text + suggestions)
- parser_model: a second model that reads the free-form response and
  produces the structured output — the main model is never constrained

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
# Structured Output Schema
# ---------------------------------------------------------------------------
class FollowUpSuggestion(BaseModel):
    """A single follow-up suggestion."""

    title: str = Field(..., description="Short action-oriented suggestion (5-10 words)")
    reason: str = Field(
        ..., description="One sentence explaining why this is a good follow-up"
    )


class ResponseWithSuggestions(BaseModel):
    """The main response plus follow-up suggestions."""

    response: str = Field(
        ...,
        description="The full response text from the assistant, copied verbatim",
    )
    follow_up_suggestions: List[FollowUpSuggestion] = Field(
        ...,
        description="3 follow-up suggestions based on the response",
        min_length=3,
        max_length=3,
    )


# ---------------------------------------------------------------------------
# Create the Agent
# ---------------------------------------------------------------------------
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="You are a knowledgeable assistant. Answer questions thoroughly.",
    # The schema the parser model will fill in
    output_schema=ResponseWithSuggestions,
    # A cheap model extracts structured output from the main response
    parser_model=OpenAIResponses(id="gpt-5-mini"),
    parser_model_prompt=(
        "Extract the assistant's response verbatim into the 'response' field. "
        "Then generate 3 follow-up suggestions based on the response. "
        "Each suggestion should cover a different angle: "
        "one that digs deeper, one practical next step, and one alternative perspective."
    ),
    markdown=True,
)


# ---------------------------------------------------------------------------
# Run the Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run: RunOutput = agent.run("Which national park is the best?")

    result: ResponseWithSuggestions = run.content

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
