from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.os.app import AgentOS
from agno.os.interfaces.telegram import Telegram

agent_db = SqliteDb(
    session_table="telegram_sessions", db_file="tmp/telegram_streaming.db"
)

telegram_agent = Agent(
    name="Telegram Streaming Bot",
    model=OpenAIChat(id="gpt-4o-mini"),
    db=agent_db,
    instructions=[
        "You are a helpful assistant on Telegram.",
        "Keep responses concise and friendly.",
        "When in a group, you respond only when mentioned with @.",
    ],
    add_history_to_context=True,
    num_history_runs=3,
    add_datetime_to_context=True,
    markdown=True,
)

agent_os = AgentOS(
    agents=[telegram_agent],
    interfaces=[
        Telegram(
            agent=telegram_agent,
            reply_to_mentions_only=True,
            stream=True,
        )
    ],
)
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="streaming:app", reload=True)
