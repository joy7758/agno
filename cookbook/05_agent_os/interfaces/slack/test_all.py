"""
Comprehensive Slack Test
========================

Two Slack apps (Dash and Ace) running on one server.
Each app gets its own token/signing_secret via the constructor.

Dash: streaming + image generation (DALL-E) + web search + reasoning + markdown
Ace:  streaming + web search + reasoning + markdown

Setup:
  1. Set env vars: DASH_SLACK_TOKEN, DASH_SIGNING_SECRET, ACE_SLACK_TOKEN, ACE_SIGNING_SECRET
  2. Start ngrok: ngrok http 7777
  3. Configure Slack Event Subscriptions:
     - Dash app: https://<ngrok>/dash/events
     - Ace app: https://<ngrok>/ace/events
  4. Required scopes: chat:write, app_mentions:read, im:history, assistant:write,
     files:read, files:write, users:read
  5. Required events: app_mention, message.im, assistant_thread_started

Run:
  .venvs/demo/bin/python cookbook/05_agent_os/interfaces/slack/test_all.py
"""

import os

from agno.agent import Agent
from agno.db.sqlite.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.os.app import AgentOS
from agno.os.interfaces.slack import Slack
from agno.team import Team
from agno.tools.dalle import DalleTools
from agno.tools.websearch import WebSearchTools

# ---------------------------------------------------------------------------
# Credentials (explicit per-instance)
# ---------------------------------------------------------------------------

DASH_TOKEN = os.getenv("DASH_SLACK_TOKEN")
DASH_SECRET = os.getenv("DASH_SIGNING_SECRET")
ACE_TOKEN = os.getenv("ACE_SLACK_TOKEN")
ACE_SECRET = os.getenv("ACE_SIGNING_SECRET")

# ---------------------------------------------------------------------------
# Shared DB
# ---------------------------------------------------------------------------

dash_db = SqliteDb(session_table="dash_sessions", db_file="tmp/test_all.db")
ace_db = SqliteDb(session_table="ace_sessions", db_file="tmp/test_all.db")

# ---------------------------------------------------------------------------
# Dash Agent — streaming + DALL-E + web search + reasoning + markdown
# ---------------------------------------------------------------------------

dash_agent = Agent(
    name="Dash",
    model=OpenAIChat(id="gpt-4o"),
    reasoning_model=OpenAIChat(id="o4-mini"),
    tools=[WebSearchTools(), DalleTools()],
    instructions=[
        "You are Dash, a creative assistant with image generation and web search abilities.",
        "When asked to create or generate an image, use DALL-E.",
        "When asked to search, use your web search tools.",
        "Always format responses with markdown: use **bold**, *italics*, bullet points, and code blocks.",
        "Keep responses concise but informative.",
    ],
    db=dash_db,
    add_history_to_context=True,
    num_history_runs=3,
    add_datetime_to_context=True,
    markdown=True,
)

# ---------------------------------------------------------------------------
# Ace Agent — streaming + web search + reasoning + markdown
# ---------------------------------------------------------------------------

ace_agent = Agent(
    name="Ace",
    model=OpenAIChat(id="gpt-4o"),
    reasoning_model=OpenAIChat(id="o4-mini"),
    tools=[WebSearchTools()],
    instructions=[
        "You are Ace, a research assistant.",
        "When asked to search, use your web search tools.",
        "Always format responses with markdown: use **bold**, *italics*, bullet points, headers, and code blocks.",
        "Provide detailed, well-structured answers.",
    ],
    db=ace_db,
    add_history_to_context=True,
    num_history_runs=3,
    add_datetime_to_context=True,
    markdown=True,
)

# ---------------------------------------------------------------------------
# Team (uses Ace app credentials)
# ---------------------------------------------------------------------------

ace_team = Team(
    name="Ace Team",
    mode="coordinate",
    members=[
        Agent(
            name="Researcher",
            model=OpenAIChat(id="gpt-4o"),
            tools=[WebSearchTools()],
            instructions=[
                "You are a web researcher. Search the web to find information."
            ],
        ),
        Agent(
            name="Writer",
            model=OpenAIChat(id="gpt-4o"),
            instructions=[
                "You are a writer. Synthesize information into clear, well-structured responses."
            ],
        ),
    ],
    model=OpenAIChat(id="gpt-4o"),
    instructions=["Coordinate between the researcher and writer to answer questions."],
    db=ace_db,
    add_history_to_context=True,
    markdown=True,
)

# ---------------------------------------------------------------------------
# AgentOS — two apps, explicit credentials
# ---------------------------------------------------------------------------

agent_os = AgentOS(
    agents=[dash_agent, ace_agent],
    teams=[ace_team],
    interfaces=[
        # Dash app — streaming team (Researcher + Writer coordinate)
        Slack(
            team=ace_team,
            prefix="/dash",
            token=DASH_TOKEN,
            signing_secret=DASH_SECRET,
            streaming=True,
            reply_to_mentions_only=True,
            initial_buffer_size=1,
            loading_messages=[
                "Thinking...",
                "Searching the web...",
                "Working on it...",
            ],
            suggested_prompts=[
                {"title": "Help", "message": "What can you help me with?"},
                {"title": "Search", "message": "Search the web for the latest news"},
            ],
        ),
        # Ace app — streaming team (Researcher + Writer coordinate)
        Slack(
            team=ace_team,
            prefix="/ace",
            token=ACE_TOKEN,
            signing_secret=ACE_SECRET,
            streaming=True,
            reply_to_mentions_only=True,
            initial_buffer_size=1,
            loading_messages=[
                "Thinking...",
                "Researching...",
                "Almost there...",
            ],
            suggested_prompts=[
                {"title": "Research", "message": "Research a topic for me"},
                {"title": "Summarize", "message": "Summarize the latest tech news"},
            ],
        ),
    ],
)
app = agent_os.get_app()


if __name__ == "__main__":
    agent_os.serve(app="test_all:app", reload=True, port=7777)
