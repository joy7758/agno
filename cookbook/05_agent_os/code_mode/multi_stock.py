"""
Code Mode — Multi-Stock Deep Comparison
========================================
Side-by-side comparison of Code Mode vs Traditional tool_call mode
on a multi-stock analysis that requires fetching data for N stocks.

This pattern benefits Code Mode because:
  - For each of 4 stocks, fetch: price + fundamentals + analyst recs = 12 tool calls
  - Traditional mode: even if batched, 3+ model turns with growing context
  - Code Mode: 1 exec() pass with a for-loop over all stocks

Based on: cookbook/00_quickstart/agent_with_tools.py

Run:
  .venvs/demo/bin/python cookbook/05_agent_os/code_mode/multi_stock.py
"""

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.code_mode import CodeModeTool
from agno.tools.yfinance import YFinanceTools

TASK = (
    "Compare these 4 AI stocks: NVDA, AMD, GOOGL, MSFT. "
    "For each stock, get the current price, stock fundamentals, and analyst recommendations. "
    "Then create a comprehensive comparison table with columns: "
    "Symbol, Price, Market Cap, P/E Ratio, Recommendation, Buy/Hold/Sell counts. "
    "End with a brief analysis of which stock looks strongest and why."
)

CODE_MODE_INSTRUCTIONS = (
    "You solve tasks by writing Python code using the run_code tool.\n\n"
    "CRITICAL RULES:\n"
    "1. Write ONE COMPLETE Python program per run_code call. NEVER make multiple small calls.\n"
    "2. Your code should handle the ENTIRE task: loop over stocks, fetch data, format table.\n"
    "3. Call functions DIRECTLY: get_current_stock_price(symbol='NVDA'). No prefix.\n"
    "4. json and math are pre-loaded. Do NOT write import statements.\n"
    "5. All functions return JSON strings. Always use json.loads() to parse results.\n"
    "6. Store your final answer in `result` as a formatted markdown string.\n\n"
    "PATTERN:\n"
    "  symbols = ['NVDA', 'AMD', 'GOOGL', 'MSFT']\n"
    "  rows = []\n"
    "  for sym in symbols:\n"
    "      price = json.loads(get_current_stock_price(symbol=sym))\n"
    "      fundamentals = json.loads(get_stock_fundamentals(symbol=sym))\n"
    "      recs = json.loads(get_analyst_recommendations(symbol=sym))\n"
    "      rows.append(f'| {sym} | ... |')\n"
    "  result = '| Symbol | Price | ... |\\n' + '\\n'.join(rows)"
)

MODEL = "claude-sonnet-4-20250514"


def run_code_mode():
    yf = YFinanceTools(
        enable_stock_price=True,
        enable_stock_fundamentals=True,
        enable_analyst_recommendations=True,
    )

    agent = Agent(
        name="Code Mode Stock Analyst",
        model=Claude(id=MODEL),
        tools=[CodeModeTool(tools=[yf])],
        tool_call_limit=3,
        instructions=CODE_MODE_INSTRUCTIONS,
        markdown=True,
    )

    return agent.run(TASK)


def run_traditional():
    agent = Agent(
        name="Traditional Stock Analyst",
        model=Claude(id=MODEL),
        tools=[
            YFinanceTools(
                enable_stock_price=True,
                enable_stock_fundamentals=True,
                enable_analyst_recommendations=True,
            )
        ],
        instructions=[
            "You are a financial analyst. Use the available tools to answer questions.",
            "Use tables to display comparison data.",
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
