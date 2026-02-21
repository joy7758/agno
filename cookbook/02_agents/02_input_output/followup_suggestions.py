"""
Follow-Up Suggestions (Built-in)
================================

Enable built-in follow-up suggestions on any agent with a single flag.

After the main response, the agent automatically makes a second model call
to generate structured follow-up suggestions and attaches them to RunOutput.

Key concepts:
- follow_up_suggestions=True: enables the feature
- num_follow_up_suggestions: controls how many suggestions (default 3)
- follow_up_model: optional cheaper model for generating suggestions
- run_response.follow_up_suggestions: the structured result

The main response is never constrained — it streams freely as normal text.

Example prompts to try:
- "Which national park is the best?"
- "What programming language should I learn first?"
- "How do I start investing?"
"""

from agno.agent import Agent, RunOutput
from agno.models.openai import OpenAIResponses

# ---------------------------------------------------------------------------
# Create the Agent — just set follow_up_suggestions=True
# ---------------------------------------------------------------------------
agent = Agent(
    model=OpenAIResponses(id="gpt-4o"),
    instructions="You are a knowledgeable assistant. Answer questions thoroughly.",
    # Enable built-in follow-up suggestions
    follow_up_suggestions=True,
    num_follow_up_suggestions=3,
    # Optionally use a cheaper model for suggestions
    # follow_up_model=OpenAIResponses(id="gpt-4o-mini"),
    markdown=True,
)


# ---------------------------------------------------------------------------
# Run the Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run: RunOutput = agent.run("Which national park is the best?")

    # The main response — full free-form text
    print(f"\n{'=' * 60}")
    print("Response:")
    print(f"{'=' * 60}")
    print(run.content)

    # Follow-up suggestions — structured, attached to RunOutput
    print(f"\n{'=' * 60}")
    print("Follow-Up Suggestions:")
    print(f"{'=' * 60}")
    if run.follow_up_suggestions:
        for i, suggestion in enumerate(run.follow_up_suggestions.suggestions, 1):
            print(f"\n  {i}. {suggestion.title}")
            print(f"     {suggestion.reason}")
    else:
        print("  No suggestions generated.")

    print()
