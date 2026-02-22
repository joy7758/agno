"""
Code Mode — Discovery Mode Demo
================================
When CodeModeTool wraps many tools (> 15 functions), it automatically
switches to discovery mode: the LLM gets two tools instead of one:

  - search_tools(query)  — find relevant functions by keyword
  - run_code(code)       — execute Python using discovered functions

This keeps the model context small even with 20+ tools. The run_code
description only shows a brief catalog (name + one-line description),
not full function stubs. The model searches first, then codes.

Toolkits combined here:
  - YFinanceTools (all=True)   — 9 financial data tools
  - CalculatorTools            — 8 math tools
  - WebSearchTools             — 2 search tools
  - Newspaper4kTools           — 1 article reader
  Total: 20 functions → auto-enables discovery (threshold=15)

Run:
  .venvs/demo/bin/python cookbook/05_agent_os/code_mode/discovery.py
"""

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.calculator import CalculatorTools
from agno.tools.code_mode import CodeModeTool
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.websearch import WebSearchTools
from agno.tools.yfinance import YFinanceTools

TASK = (
    "I want to analyze NVDA stock. First search for recent news about NVIDIA. "
    "Then get the current stock price and key financial ratios. "
    "Calculate the price-to-earnings ratio divided by the earnings growth rate (PEG ratio) "
    "using the calculator tools. "
    "Finally, compile everything into a brief investment summary."
)

INSTRUCTIONS = (
    "You solve tasks by writing Python code using the run_code tool.\n\n"
    "WORKFLOW:\n"
    "1. Use search_tools to discover available functions before coding.\n"
    "2. Write ONE COMPLETE Python program per run_code call.\n"
    "3. Call functions DIRECTLY: get_current_stock_price(symbol='NVDA'). No prefix.\n"
    "4. json and math are pre-imported. Do NOT write import statements.\n"
    "5. All tool functions return JSON strings. Use json.loads() to parse them.\n"
    "6. Store your final answer in `result` as a formatted markdown string.\n\n"
    "IMPORTANT: Search for tools FIRST, then write code that uses them."
)

MODEL = "claude-sonnet-4-20250514"


def main():
    yf = YFinanceTools(
        enable_stock_price=True,
        enable_company_info=True,
        enable_stock_fundamentals=True,
        enable_income_statements=True,
        enable_key_financial_ratios=True,
        enable_analyst_recommendations=True,
        enable_company_news=True,
        enable_technical_indicators=True,
        enable_historical_prices=True,
    )
    calc = CalculatorTools()
    ws = WebSearchTools(enable_search=True, enable_news=True)
    news = Newspaper4kTools()

    cm = CodeModeTool(tools=[yf, calc, ws, news])

    print(f"Discovery enabled: {cm._discovery_enabled}")
    print(f"Total functions: {len(cm._sync_functions)}")
    print(f"Registered tools: {list(cm.functions.keys())}")
    print()

    agent = Agent(
        name="Discovery Mode Analyst",
        model=Claude(id=MODEL),
        tools=[cm],
        instructions=INSTRUCTIONS,
        markdown=True,
    )

    response = agent.run(TASK)
    print(response.content)

    if response.metrics:
        m = response.metrics
        print(
            f"\nTokens — input: {m.input_tokens:,}  output: {m.output_tokens:,}  total: {m.total_tokens:,}"
        )


if __name__ == "__main__":
    main()
