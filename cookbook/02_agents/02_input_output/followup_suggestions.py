"""
Follow-Up Suggestions
=============================

Get a full response AND follow-up suggestions using a two-step approach.

Step 1: The agent answers the question freely — no schema constraints,
        full streaming, no truncation risk.
Step 2: A lightweight structured call extracts follow-up suggestions
        from the conversation history.

This avoids the main pitfall of putting everything in one output_schema:
the model compressing its answer to fit the JSON budget.

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
# Structured Output Schema (for suggestions only)
# ---------------------------------------------------------------------------
class FollowUpSuggestion(BaseModel):
    """A single follow-up suggestion."""

    title: str = Field(..., description="Short action-oriented suggestion (5-10 words)")
    reason: str = Field(
        ..., description="One sentence explaining why this is a good follow-up"
    )


class FollowUpSuggestions(BaseModel):
    """Follow-up suggestions extracted from a conversation."""

    suggestions: List[FollowUpSuggestion] = Field(
        ...,
        description="3 follow-up suggestions based on the conversation",
        min_length=3,
        max_length=3,
    )


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

# Step 1: Main agent — responds freely, supports streaming, no truncation
main_agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="You are a knowledgeable assistant. Answer questions thoroughly.",
    add_history_to_context=True,
    num_history_runs=1,
    markdown=True,
)

# Step 2: Suggestion agent — reads the conversation and generates suggestions
suggestion_agent = Agent(
    model=OpenAIResponses(id="gpt-5-mini"),
    instructions="""\
Given the conversation above, generate exactly 3 follow-up suggestions.
Each suggestion should be a natural next question the user might ask.

Cover different angles:
- One that digs deeper into the main topic
- One that explores a practical next step
- One that offers an alternative perspective or comparison\
""",
    output_schema=FollowUpSuggestions,
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    query = "Which national park is the best?"

    # Step 1: Get the full response (can also use stream=True here)
    response = main_agent.run(query)
    main_answer = response.content

    print(f"\n{'=' * 60}")
    print("Response:")
    print(f"{'=' * 60}")
    print(main_answer)

    # Step 2: Generate suggestions from the Q&A
    suggestion_response = suggestion_agent.run(
        f"User question: {query}\n\nAssistant response: {main_answer}"
    )
    suggestions: FollowUpSuggestions = suggestion_response.content

    print(f"\n{'=' * 60}")
    print("Follow-Up Suggestions:")
    print(f"{'=' * 60}")
    for i, suggestion in enumerate(suggestions.suggestions, 1):
        print(f"\n  {i}. {suggestion.title}")
        print(f"     {suggestion.reason}")

    print()
