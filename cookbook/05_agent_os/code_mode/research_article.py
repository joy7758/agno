"""
Code Mode — Research Article Writer
====================================
Side-by-side comparison of Code Mode vs Traditional tool_call mode
on a research task that requires: search -> read each URL -> compile.

This is the ideal pattern for Code Mode because:
  - search_news() must run FIRST to get URLs
  - read_article() must run once PER URL (natural loop)
  - Traditional mode: N+1 model turns (1 search + N reads + final)
  - Code Mode: 1-2 model turns total

Based on: cookbook/90_models/groq/tool_use.py

Run:
  .venvs/demo/bin/python cookbook/05_agent_os/code_mode/research_article.py
"""

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.code_mode import CodeModeTool
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.websearch import WebSearchTools

TASK = (
    "Search for the top 3 recent news articles about AI regulation. "
    "Then read each article URL and extract the key points. "
    "Finally, write a concise summary comparing the different perspectives "
    "on AI regulation across all articles. Present as a markdown report."
)

CODE_MODE_INSTRUCTIONS = (
    "You solve tasks by writing Python code using the run_code tool.\n\n"
    "CRITICAL RULES:\n"
    "1. Write ONE COMPLETE Python program per run_code call. NEVER make multiple small calls.\n"
    "2. Your code should handle the ENTIRE task: search, loop through URLs, read articles, format.\n"
    "3. Call functions DIRECTLY: web_search(query='x'). No prefix.\n"
    "4. json and math are pre-loaded. Do NOT write import statements.\n"
    "5. All functions return JSON strings. Always use json.loads() to parse results.\n"
    "6. Store your final answer in `result` as a formatted markdown string.\n\n"
    "PATTERN:\n"
    "  search_results = json.loads(search_news(query='AI regulation', max_results=3))\n"
    "  for item in search_results:\n"
    "      article = json.loads(read_article(url=item['url']))\n"
    "      # extract key points...\n"
    "  result = '# AI Regulation Report\\n' + sections"
)

MODEL = "claude-sonnet-4-20250514"


def run_code_mode():
    ws = WebSearchTools(enable_search=True, enable_news=True)
    news = Newspaper4kTools()

    agent = Agent(
        name="Code Mode Researcher",
        model=Claude(id=MODEL),
        tools=[CodeModeTool(tools=[ws, news])],
        tool_call_limit=3,
        instructions=CODE_MODE_INSTRUCTIONS,
        markdown=True,
    )

    return agent.run(TASK)


def run_traditional():
    agent = Agent(
        name="Traditional Researcher",
        model=Claude(id=MODEL),
        tools=[
            WebSearchTools(enable_search=True, enable_news=True),
            Newspaper4kTools(),
        ],
        instructions=[
            "You are a senior research journalist.",
            "For a given topic, search for the top news articles.",
            "Then read each URL and extract the article text.",
            "Compile a well-sourced summary comparing perspectives.",
        ],
        markdown=True,
    )

    return agent.run(TASK)


if __name__ == "__main__":
    print("=" * 70)
    print("CODE MODE")
    print("=" * 70)
    code_response = run_code_mode()
    print(f"\n{code_response.content}\n")
    cm = code_response.metrics

    print("\n" + "=" * 70)
    print("TRADITIONAL MODE")
    print("=" * 70)
    trad_response = run_traditional()
    print(f"\n{trad_response.content}\n")
    tm = trad_response.metrics

    cm_input = cm.input_tokens if cm else 0
    tm_input = tm.input_tokens if tm else 0
    cm_output = cm.output_tokens if cm else 0
    tm_output = tm.output_tokens if tm else 0
    cm_total = cm.total_tokens if cm else 0
    tm_total = tm.total_tokens if tm else 0

    print("\n" + "=" * 70)
    print("TOKEN COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {'Code Mode':>15} {'Traditional':>15} {'Savings':>15}")
    print("-" * 70)
    print(
        f"{'Input tokens':<25} {cm_input:>15,} {tm_input:>15,} {tm_input - cm_input:>15,}"
    )
    print(
        f"{'Output tokens':<25} {cm_output:>15,} {tm_output:>15,} {tm_output - cm_output:>15,}"
    )
    print(
        f"{'Total tokens':<25} {cm_total:>15,} {tm_total:>15,} {tm_total - cm_total:>15,}"
    )
    if tm_total > 0:
        pct = (1 - cm_total / tm_total) * 100
        print(f"{'Reduction':<25} {pct:>14.0f}%")
