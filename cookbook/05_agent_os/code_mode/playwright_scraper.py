"""
Code Mode + Playwright MCP — Browser Automation Pipeline
=========================================================
Demonstrates CodeModeTool with Playwright MCP server (26 tools).
Discovery mode auto-triggers, letting the LLM search for relevant
browser actions before writing a single code block that chains
navigate -> snapshot -> click -> extract into one exec() pass.

Without Code Mode, a browser scraping task with 10+ sequential
tool calls would snowball tokens on every model turn. With Code Mode,
the LLM writes one program that drives the entire browser session.

Setup:
  npm install -g @playwright/mcp@latest
  # or: npx @playwright/mcp@latest (auto-installed)

Run:
  .venvs/demo/bin/python cookbook/05_agent_os/code_mode/playwright_scraper.py
"""

import asyncio
import os

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.code_mode import CodeModeTool
from agno.tools.mcp import MCPTools

TASK = (
    "Go to https://news.ycombinator.com and extract the top 5 stories. "
    "For each story, get the title, URL, points, and number of comments. "
    "Format the results as a markdown table with columns: Rank, Title, URL, Points, Comments."
)

CODE_MODE_INSTRUCTIONS = (
    "You solve tasks by writing Python code using the run_code tool.\n\n"
    "WORKFLOW:\n"
    "1. First use search_tools to find available browser functions.\n"
    "2. Then write ONE complete Python program in run_code that performs the entire task.\n\n"
    "CRITICAL RULES:\n"
    "- Write ONE complete program per run_code call. Do NOT make multiple small calls.\n"
    "- Call functions DIRECTLY: browser_navigate(url='...'), NOT browser.navigate().\n"
    "- json is pre-imported. Use json.loads() to parse any JSON string results.\n"
    "- Store your final answer in `result` as a formatted markdown string.\n"
    "- Use browser_snapshot() to read page content (returns accessibility tree).\n"
    "- Parse the snapshot text to extract data — no need for complex selectors."
)


async def run_playwright_code_mode():
    npx_cmd = "npx.cmd" if os.name == "nt" else "npx"

    async with MCPTools(f"{npx_cmd} -y @playwright/mcp@latest") as mcp:
        tool_count = len(mcp.functions)
        print(f"Playwright MCP tools loaded: {tool_count}")

        cm = CodeModeTool(tools=[mcp])
        print(f"Discovery mode: {cm._discovery_enabled}")

        agent = Agent(
            name="Browser Scraper",
            model=Claude(id="claude-sonnet-4-20250514"),
            tools=[cm],
            tool_call_limit=5,
            instructions=CODE_MODE_INSTRUCTIONS,
            markdown=True,
        )

        response = await agent.arun(TASK)
        print(f"\n{response.content}\n")

        m = response.metrics
        if m:
            print(f"Total tokens: {m.total_tokens:,}")
            print(f"Input tokens: {m.input_tokens:,}")
            print(f"Output tokens: {m.output_tokens:,}")


if __name__ == "__main__":
    asyncio.run(run_playwright_code_mode())
