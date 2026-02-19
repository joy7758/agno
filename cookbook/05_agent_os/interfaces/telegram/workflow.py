from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.os.app import AgentOS
from agno.os.interfaces.telegram import Telegram
from agno.workflow.step import Step
from agno.workflow.steps import Steps
from agno.workflow.workflow import Workflow

agent_db = SqliteDb(
    session_table="telegram_workflow_sessions", db_file="tmp/telegram_workflow.db"
)

drafter = Agent(
    name="Drafter",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="Draft a response to the user's message. Be helpful and informative.",
)

editor = Agent(
    name="Editor",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=[
        "Review and polish the draft for clarity and conciseness.",
        "Keep it short and suitable for a Telegram message.",
    ],
)

draft_step = Step(
    name="draft",
    agent=drafter,
    description="Draft an initial response",
)

edit_step = Step(
    name="edit",
    agent=editor,
    description="Edit and polish the draft",
)

telegram_workflow = Workflow(
    name="Telegram Draft-Edit Workflow",
    description="A two-step workflow that drafts and then edits responses for Telegram",
    steps=[
        Steps(
            name="draft_and_edit",
            description="Draft then edit a response",
            steps=[draft_step, edit_step],
        )
    ],
    db=agent_db,
)

agent_os = AgentOS(
    workflows=[telegram_workflow],
    interfaces=[
        Telegram(
            workflow=telegram_workflow,
            reply_to_mentions_only=True,
        )
    ],
)
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="workflow:app", reload=True)
