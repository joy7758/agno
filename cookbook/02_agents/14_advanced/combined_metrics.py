"""
Combined Metrics
=============================

When an agent uses multiple background features, each model's
calls are tracked under separate detail keys:
- "model" for the agent's own calls
- "memory_model" for memory manager calls
- "culture_model" for culture manager calls
- "eval_model" for evaluation hook calls

This example shows all four detail keys in a single run.
"""

from agno.agent import Agent
from agno.culture.manager import CultureManager
from agno.db.postgres import PostgresDb
from agno.eval.agent_as_judge import AgentAsJudgeEval
from agno.memory.manager import MemoryManager
from agno.models.openai import OpenAIChat
from rich.pretty import pprint

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
db = PostgresDb(db_url="postgresql+psycopg://ai:ai@localhost:5532/ai")

eval_hook = AgentAsJudgeEval(
    name="Quality Check",
    model=OpenAIChat(id="gpt-4o-mini"),
    criteria="Response should be helpful and accurate",
    scoring_strategy="binary",
)

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    memory_manager=MemoryManager(model=OpenAIChat(id="gpt-4o-mini"), db=db),
    culture_manager=CultureManager(model=OpenAIChat(id="gpt-4o-mini"), db=db),
    update_memory_on_run=True,
    update_cultural_knowledge=True,
    post_hooks=[eval_hook],
    db=db,
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_response = agent.run(
        "My name is Alice and I work at Google as a senior engineer."
    )

    print("=" * 50)
    print("RUN METRICS")
    print("=" * 50)
    pprint(run_response.metrics)

    print("=" * 50)
    print("MODEL DETAILS")
    print("=" * 50)
    if run_response.metrics and run_response.metrics.details:
        for model_type, model_metrics_list in run_response.metrics.details.items():
            print(f"\n{model_type}:")
            for model_metric in model_metrics_list:
                pprint(model_metric)
