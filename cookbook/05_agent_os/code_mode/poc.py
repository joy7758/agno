"""
Code Mode — Proof of Concept
==============================
Instead of an LLM calling tools one-at-a-time via tool_call protocol, it writes
Python code that calls multiple tools in a single exec() pass.

This PoC proves:
  1. LLM can write valid Python using generated function stubs
  2. exec() with tool wrappers correctly routes to real function calls
  3. Loops, conditionals, and multi-tool chaining work in one pass
  4. Token savings from eliminating intermediate model round-trips

Run:
  .venvs/demo/bin/python cookbook/05_agent_os/code_mode/poc.py
"""

import builtins as _builtins_mod
import io
import json
from contextlib import redirect_stdout
from typing import Any, Callable, Dict, List, Optional

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools import Toolkit
from agno.utils.code_execution import prepare_python_code

# ---------------------------------------------------------------------------
# Mock tools — simulate real API tools with deterministic responses
# ---------------------------------------------------------------------------

MOCK_DB = {
    "products": [
        {
            "id": 1,
            "name": "Laptop Pro",
            "category": "electronics",
            "price": 1299,
            "rating": 4.5,
            "stock": 23,
        },
        {
            "id": 2,
            "name": "Wireless Mouse",
            "category": "electronics",
            "price": 29,
            "rating": 4.2,
            "stock": 150,
        },
        {
            "id": 3,
            "name": "Standing Desk",
            "category": "furniture",
            "price": 599,
            "rating": 4.8,
            "stock": 12,
        },
        {
            "id": 4,
            "name": "Monitor 4K",
            "category": "electronics",
            "price": 449,
            "rating": 4.6,
            "stock": 34,
        },
        {
            "id": 5,
            "name": "Ergonomic Chair",
            "category": "furniture",
            "price": 899,
            "rating": 4.7,
            "stock": 8,
        },
        {
            "id": 6,
            "name": "USB-C Hub",
            "category": "electronics",
            "price": 49,
            "rating": 3.9,
            "stock": 200,
        },
        {
            "id": 7,
            "name": "Desk Lamp",
            "category": "furniture",
            "price": 79,
            "rating": 4.1,
            "stock": 67,
        },
        {
            "id": 8,
            "name": "Mechanical Keyboard",
            "category": "electronics",
            "price": 159,
            "rating": 4.4,
            "stock": 45,
        },
    ],
}


def search_products(query: str, category: Optional[str] = None) -> str:
    """Search products by name keyword and optional category filter.

    Args:
        query: Search keyword to match against product names (case-insensitive)
        category: Optional category filter ("electronics" or "furniture")

    Returns:
        JSON array of objects with keys: id (int), name (str), price (float)
    """
    results = []
    for p in MOCK_DB["products"]:
        if query.lower() in p["name"].lower():
            if category is None or p["category"] == category:
                results.append({"id": p["id"], "name": p["name"], "price": p["price"]})
    return json.dumps(results)


def get_product_details(product_id: int) -> str:
    """Get full details for a product by its ID.

    Args:
        product_id: The unique product ID

    Returns:
        JSON object with keys: id, name, category, price, rating (float), stock (int)
    """
    for p in MOCK_DB["products"]:
        if p["id"] == product_id:
            return json.dumps(p)
    return json.dumps({"error": f"Product {product_id} not found"})


def check_inventory(product_id: int) -> str:
    """Check if a product is in stock and return availability info.

    Args:
        product_id: The unique product ID

    Returns:
        JSON object with keys: product_id (int), stock (int), status ("in_stock" or "out_of_stock")
    """
    for p in MOCK_DB["products"]:
        if p["id"] == product_id:
            status = "in_stock" if p["stock"] > 0 else "out_of_stock"
            return json.dumps(
                {"product_id": p["id"], "stock": p["stock"], "status": status}
            )
    return json.dumps({"error": f"Product {product_id} not found"})


def calculate_discount(price: float, discount_percent: float) -> str:
    """Calculate the discounted price.

    Args:
        price: Original price
        discount_percent: Discount percentage (e.g., 10 for 10%)

    Returns:
        JSON object with keys: original (float), discount_percent (float), final_price (float)
    """
    discounted = price * (1 - discount_percent / 100)
    return json.dumps(
        {
            "original": price,
            "discount_percent": discount_percent,
            "final_price": round(discounted, 2),
        }
    )


# ---------------------------------------------------------------------------
# Stub generator — converts tool functions into Python API documentation
# ---------------------------------------------------------------------------

JSON_TYPE_MAP = {
    "string": "str",
    "number": "float",
    "integer": "int",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


def generate_stubs(tools: Dict[str, Callable]) -> str:
    """Generate Python function stubs with docstrings from callable tools."""
    from inspect import getdoc

    from agno.tools.function import Function

    stubs = []
    for name, func in tools.items():
        fn = Function.from_callable(func, name=name)
        params = fn.parameters.get("properties", {})
        required = set(fn.parameters.get("required", []))

        args = []
        for pname, schema in params.items():
            py_type = JSON_TYPE_MAP.get(schema.get("type", "string"), "Any")
            if pname in required:
                args.append(f"{pname}: {py_type}")
            else:
                args.append(f"{pname}: {py_type} = None")

        sig = ", ".join(args)

        # Use full docstring from the original function (includes Returns section)
        full_doc = getdoc(func) or fn.description or "No description"

        stub = f"def {name}({sig}) -> str:\n"
        stub += f'    """{full_doc}\n    """'
        stubs.append(stub)

    return "\n\n".join(stubs)


# ---------------------------------------------------------------------------
# Code Mode Toolkit — the agent's single tool
# ---------------------------------------------------------------------------

_ALLOWED = {
    "len",
    "min",
    "max",
    "sum",
    "sorted",
    "reversed",
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    "any",
    "all",
    "str",
    "int",
    "float",
    "bool",
    "dict",
    "list",
    "set",
    "tuple",
    "round",
    "abs",
    "print",
    "isinstance",
    "type",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "Exception",
    "True",
    "False",
    "None",
}
SAFE_BUILTINS = {
    k: getattr(_builtins_mod, k) for k in _ALLOWED if hasattr(_builtins_mod, k)
}


class CodeModeTool(Toolkit):
    def __init__(self, tools: List[Callable]):
        self._tool_funcs: Dict[str, Callable] = {f.__name__: f for f in tools}
        self._stubs = generate_stubs(self._tool_funcs)
        super().__init__(name="code_mode", tools=[self.run_code])

    def run_code(self, code: str) -> str:
        """Execute Python code that calls tool functions directly.

        RULES:
        - Call functions DIRECTLY: search_products(query="x"), NOT functions.search_products()
        - json and math are pre-imported. Do NOT write import statements.
        - All tool functions return JSON strings. Use json.loads() to parse them.
        - Store your final answer in a variable called `result` (as a string).
        - You can use: json, math, loops, conditionals, list comprehensions, try/except

        Example:
            data = json.loads(search_products(query="laptop"))
            details = json.loads(get_product_details(product_id=data[0]["id"]))
            result = f"Found: {details['name']} at ${details['price']}"
        """
        try:
            code = prepare_python_code(code)

            namespace: Dict[str, Any] = {
                "__builtins__": SAFE_BUILTINS,
                "json": __import__("json"),
                "math": __import__("math"),
            }

            for name, func in self._tool_funcs.items():
                namespace[name] = func

            stdout_buf = io.StringIO()
            with redirect_stdout(stdout_buf):
                exec(code, namespace)

            result = namespace.get("result")
            stdout = stdout_buf.getvalue()

            output_parts = []
            if result is not None:
                output_parts.append(str(result))
            if stdout:
                output_parts.append(stdout.strip())

            if output_parts:
                return "\n".join(output_parts)

            # If no result variable and no stdout, return last expression value
            # by checking all namespace vars that aren't tools or builtins
            user_vars = {
                k: v
                for k, v in namespace.items()
                if k not in self._tool_funcs
                and k not in ("__builtins__", "json", "math")
                and not k.startswith("_")
            }
            if user_vars:
                return str(list(user_vars.values())[-1])

            return "Code executed successfully but produced no output. Set a `result` variable or use print()."
        except Exception as e:
            return f"Error executing code: {type(e).__name__}: {e}"

    def get_functions(self):
        funcs = super().get_functions()
        for name, func in funcs.items():
            if name == "run_code" and self._stubs not in (func.description or ""):
                base = func.description or ""
                func.description = base + "\n\nAvailable functions:\n\n" + self._stubs
        return funcs

    def get_async_functions(self):
        funcs = super().get_async_functions()
        for name, func in funcs.items():
            if name == "run_code" and self._stubs not in (func.description or ""):
                base = func.description or ""
                func.description = base + "\n\nAvailable functions:\n\n" + self._stubs
        return funcs


# ---------------------------------------------------------------------------
# Traditional agent (for comparison)
# ---------------------------------------------------------------------------


class TraditionalTools(Toolkit):
    def __init__(self):
        super().__init__(
            name="product_tools",
            tools=[
                search_products,
                get_product_details,
                check_inventory,
                calculate_discount,
            ],
        )


# ---------------------------------------------------------------------------
# Main — run both approaches and compare
# ---------------------------------------------------------------------------

TASK = (
    "Search for all electronics products, then for each one check if it's in stock. "
    "For products that are in stock with rating above 4.3, calculate a 15% discount. "
    "Give me a final report with product name, original price, discounted price, and stock count."
)

if __name__ == "__main__":
    print("=" * 70)
    print("CODE MODE (single exec pass)")
    print("=" * 70)

    code_mode_agent = Agent(
        name="Code Mode Agent",
        model=Claude(id="claude-sonnet-4-20250514"),
        tools=[
            CodeModeTool(
                tools=[
                    search_products,
                    get_product_details,
                    check_inventory,
                    calculate_discount,
                ]
            )
        ],
        tool_call_limit=3,
        instructions=(
            "You solve tasks by writing Python code using the run_code tool.\n\n"
            "CRITICAL RULES:\n"
            "1. Write ONE COMPLETE Python program per run_code call. NEVER make multiple small calls.\n"
            "2. Your code should handle the ENTIRE task: search, loop, filter, calculate, format.\n"
            "3. Call functions DIRECTLY: search_products(query='x'). No prefix, no namespace.\n"
            "4. json and math are pre-loaded. Do NOT write import statements.\n"
            "5. All functions return JSON strings. Always use json.loads() to parse results.\n"
            "6. Store your final answer in `result` as a formatted markdown string.\n\n"
            "PATTERN:\n"
            "  data = json.loads(search_products(query='...'))\n"
            "  for item in data:\n"
            "      details = json.loads(get_product_details(product_id=item['id']))\n"
            "      # process...\n"
            "  result = '| Name | Price |\\n' + rows"
        ),
        markdown=True,
        debug_mode=True,
    )

    code_mode_response = code_mode_agent.run(TASK)
    print(f"\n{code_mode_response.content}\n")
    cm_metrics = code_mode_response.metrics
    cm_input = cm_metrics.input_tokens if cm_metrics else 0
    cm_output = cm_metrics.output_tokens if cm_metrics else 0
    cm_total = cm_metrics.total_tokens if cm_metrics else 0

    print("\n" + "=" * 70)
    print("TRADITIONAL MODE (tool calls)")
    print("=" * 70)

    traditional_agent = Agent(
        name="Traditional Agent",
        model=Claude(id="claude-sonnet-4-20250514"),
        tools=[TraditionalTools()],
        instructions="You are a product analyst. Use the available tools to answer questions about products.",
        markdown=True,
        debug_mode=True,
    )

    traditional_response = traditional_agent.run(TASK)
    print(f"\n{traditional_response.content}\n")
    trad_metrics = traditional_response.metrics
    trad_input = trad_metrics.input_tokens if trad_metrics else 0
    trad_output = trad_metrics.output_tokens if trad_metrics else 0
    trad_total = trad_metrics.total_tokens if trad_metrics else 0

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {'Code Mode':>15} {'Traditional':>15} {'Savings':>15}")
    print("-" * 70)
    print(
        f"{'Input tokens':<25} {cm_input:>15,} {trad_input:>15,} {trad_input - cm_input:>15,}"
    )
    print(
        f"{'Output tokens':<25} {cm_output:>15,} {trad_output:>15,} {trad_output - cm_output:>15,}"
    )
    print(
        f"{'Total tokens':<25} {cm_total:>15,} {trad_total:>15,} {trad_total - cm_total:>15,}"
    )
